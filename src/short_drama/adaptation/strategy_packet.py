"""Deterministic, zero-provider B1.2-A packet preparation and measurement."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from short_drama.artifacts import ArtifactRef, FileArtifactStore, canonical_json_bytes, content_hash, strict_json_loads
from short_drama.foundation import FilePointerStore
from short_drama.guidance.models import DirectionStage, UserDirectionSet
from short_drama.guidance.projection import TargetCatalog, TargetRef, TargetScope, project_effective_directions, load_user_direction
from short_drama.story.chunking import estimate_tokens
from short_drama.story.consolidation import (
    CanonicalEventSet, CanonicalRelationshipSet, ConsolidationManifest, StoryConflictSet,
)
from short_drama.story.reconciliation import EntityMap, UnresolvedEntitySet
from short_drama.story.story_analysis_persistence import StoryAnalysisPersistenceService, a6_base_artifact_id

from .strategy import AdaptationStrategyError, SnapshotLocalPlotline
from .target import AdaptationSourceSnapshot, ResolvedAdaptationTarget


B12_PACKET_SCHEMA_VERSION = 1
B12_SEMANTIC_VIEW_SCHEMA_VERSION = 1


def _load(store: FileArtifactStore, ref: ArtifactRef, model: Any) -> Any:
    try:
        payload = store.get_ref(ref).payload
        value = model.from_dict(payload)
    except Exception as exc:
        raise AdaptationStrategyError(f"cannot resolve exact {ref.artifact_type} source leaf") from exc
    if value.to_dict() != payload:
        raise AdaptationStrategyError(f"persisted {ref.artifact_type} is not canonical")
    return value


def derive_snapshot_local_plotlines(a6_manifest_ref: ArtifactRef, structure: Any) -> tuple[SnapshotLocalPlotline, ...]:
    """Derive handles from the ordered immutable A6 structure, never allocating IDs."""
    if not isinstance(a6_manifest_ref, ArtifactRef) or a6_manifest_ref.artifact_type != "story_analysis_manifest":
        raise AdaptationStrategyError("plotline handles require exact A6 story_analysis_manifest ref")
    items = (("main", structure.main_plot), *( ("subplot", item) for item in structure.subplots))
    result = []
    for ordinal, (role, interpretation) in enumerate(items):
        # Include both exact manifest hash and immutable source interpretation
        # hash so source-order handles cannot claim cross-snapshot stability.
        suffix = content_hash({"manifest": a6_manifest_ref.to_dict(), "ordinal": ordinal, "role": role, "interpretation": interpretation.to_dict()})[:16]
        result.append(SnapshotLocalPlotline(f"a6pl_{ordinal:03d}_{suffix}", a6_manifest_ref.content_hash, ordinal, role))
    return tuple(result)


@dataclass(frozen=True, slots=True)
class B12Packet:
    """Whole-story compact semantic input with canonical bytes as its authority.

    The public component accessors decode a fresh defensive copy.  Thus callers
    may inspect a normal JSON shape, but can never mutate bytes previously
    measured or included in semantic identity.
    """
    source_snapshot: AdaptationSourceSnapshot
    resolved_target: ResolvedAdaptationTarget
    plotlines: tuple[SnapshotLocalPlotline, ...]
    _canonical_payload: bytes

    def __post_init__(self) -> None:
        if self.resolved_target.source_snapshot != self.source_snapshot:
            raise AdaptationStrategyError("packet target must bind exactly the packet source snapshot")
        if not isinstance(self._canonical_payload, bytes):
            raise AdaptationStrategyError("packet canonical payload must be bytes")
        payload = strict_json_loads(self._canonical_payload)
        if not isinstance(payload, dict) or canonical_json_bytes(payload) != self._canonical_payload:
            raise AdaptationStrategyError("packet canonical payload is invalid")
        if payload.get("source_snapshot") != self.source_snapshot.to_dict() or payload.get("resolved_target") != self.resolved_target.to_dict():
            raise AdaptationStrategyError("packet canonical payload does not match exact source/target")
        if payload.get("plotlines") != [item.to_dict() for item in self.plotlines]:
            raise AdaptationStrategyError("packet canonical payload does not match derived plotlines")
        if len({item.handle for item in self.plotlines}) != len(self.plotlines) or any(item.a6_manifest_hash != self.source_snapshot.story_analysis_manifest_ref.content_hash for item in self.plotlines):
            raise AdaptationStrategyError("packet plotlines must be unique and bound to the exact A6 snapshot")
        event_index = payload.get("canonical_event_index")
        if not isinstance(event_index, list):
            raise AdaptationStrategyError("packet canonical event index must be an array")
        event_ids = [entry.get("event_id") for entry in event_index if isinstance(entry, dict)]
        if not event_ids or any(not isinstance(x, str) for x in event_ids) or len(set(event_ids)) != len(event_ids):
            raise AdaptationStrategyError("packet canonical event index must contain unique complete IDs")
        if any(not isinstance(entry, dict) or not isinstance(entry.get("narrative_order"), int) for entry in event_index):
            raise AdaptationStrategyError("packet canonical event index entries must include narrative_order")
        if event_ids != [entry["event_id"] for entry in sorted(event_index, key=lambda entry: (entry["narrative_order"], entry["event_id"]))]:
            raise AdaptationStrategyError("packet canonical event index must use deterministic narrative order")
        expected_events = {ref.canonical_id for ref in self.source_snapshot.canonical_references if ref.kind == "event"}
        if set(event_ids) != expected_events:
            raise AdaptationStrategyError("packet canonical event index must exactly cover target source events")
        relationship_index = payload.get("canonical_relationship_index")
        if not isinstance(relationship_index, list) or any(not isinstance(entry, dict) for entry in relationship_index):
            raise AdaptationStrategyError("packet canonical relationship index must be an array")
        relationship_ids = [entry.get("relationship_id") for entry in relationship_index]
        if relationship_ids != sorted(relationship_ids) or len(set(relationship_ids)) != len(relationship_ids):
            raise AdaptationStrategyError("packet canonical relationship index must be uniquely sorted")
        expected_relationships = {ref.canonical_id for ref in self.source_snapshot.canonical_references if ref.kind == "relationship"}
        if set(relationship_ids) != expected_relationships:
            raise AdaptationStrategyError("packet canonical relationship index must exactly cover target source relationships")
        for field, key in (("unresolved_entity_index", "unresolved_id"), ("source_conflict_index", "conflict_id")):
            index = payload.get(field)
            if not isinstance(index, list) or any(not isinstance(entry, dict) for entry in index) or [entry.get(key) for entry in index] != sorted(entry.get(key) for entry in index):
                raise AdaptationStrategyError(f"packet {field} must be deterministically ordered")

    @classmethod
    def build(
        cls, source_snapshot: AdaptationSourceSnapshot, resolved_target: ResolvedAdaptationTarget,
        effective_directions: dict[str, Any], plotlines: tuple[SnapshotLocalPlotline, ...],
        story_bible: dict[str, Any], global_structure: dict[str, Any],
        character_analyses: tuple[dict[str, Any], ...], global_event_analysis: dict[str, Any],
        arc_analysis: dict[str, Any], canonical_event_index: tuple[dict[str, Any], ...],
        canonical_relationship_index: tuple[dict[str, Any], ...],
        unresolved_entity_index: tuple[dict[str, Any], ...], source_conflict_index: tuple[dict[str, Any], ...],
    ) -> "B12Packet":
        payload = {"schema_version": B12_PACKET_SCHEMA_VERSION, "source_snapshot": source_snapshot.to_dict(), "resolved_target": resolved_target.to_dict(), "effective_directions": effective_directions, "plotlines": [x.to_dict() for x in plotlines], "story_bible": story_bible, "global_structure": global_structure, "character_analyses": list(character_analyses), "global_event_analysis": global_event_analysis, "arc_analysis": arc_analysis, "canonical_event_index": list(canonical_event_index), "canonical_relationship_index": list(canonical_relationship_index), "unresolved_entity_index": list(unresolved_entity_index), "source_conflict_index": list(source_conflict_index)}
        return cls(source_snapshot, resolved_target, tuple(plotlines), canonical_json_bytes(payload))

    def to_dict(self) -> dict[str, Any]:
        payload = strict_json_loads(self._canonical_payload)
        assert isinstance(payload, dict)
        return payload

    def _component(self, name: str) -> Any:
        return self.to_dict()[name]

    @property
    def effective_directions(self) -> dict[str, Any]: return self._component("effective_directions")
    @property
    def story_bible(self) -> dict[str, Any]: return self._component("story_bible")
    @property
    def global_structure(self) -> dict[str, Any]: return self._component("global_structure")
    @property
    def character_analyses(self) -> tuple[dict[str, Any], ...]: return tuple(self._component("character_analyses"))
    @property
    def global_event_analysis(self) -> dict[str, Any]: return self._component("global_event_analysis")
    @property
    def arc_analysis(self) -> dict[str, Any]: return self._component("arc_analysis")
    @property
    def canonical_event_index(self) -> tuple[dict[str, Any], ...]: return tuple(self._component("canonical_event_index"))
    @property
    def canonical_relationship_index(self) -> tuple[dict[str, Any], ...]: return tuple(self._component("canonical_relationship_index"))
    @property
    def unresolved_entity_index(self) -> tuple[dict[str, Any], ...]: return tuple(self._component("unresolved_entity_index"))
    @property
    def source_conflict_index(self) -> tuple[dict[str, Any], ...]: return tuple(self._component("source_conflict_index"))

    def canonical_bytes(self) -> bytes: return self._canonical_payload
    def content_hash(self) -> str: return content_hash(strict_json_loads(self._canonical_payload))


@dataclass(frozen=True, slots=True)
class B12SemanticView:
    """The provider-facing, immutable projection of a :class:`B12Packet`.

    The A6/A5 snapshot is Python-owned authority and occurs twice in the full
    packet (once directly and once in ``resolved_target``).  A provider cannot
    choose or alter that authority, so this view deliberately omits both
    copies.  It retains the target's non-lineage instructions and all story
    evidence/indexes.  This is a lossless-for-semantic-planning compaction, not
    a replacement for the authoritative packet.
    """

    packet_hash: str
    _canonical_payload: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.packet_hash, str) or len(self.packet_hash) != 64:
            raise AdaptationStrategyError("semantic view requires a packet hash")
        if not isinstance(self._canonical_payload, bytes):
            raise AdaptationStrategyError("semantic view canonical payload must be bytes")
        payload = strict_json_loads(self._canonical_payload)
        if not isinstance(payload, dict) or canonical_json_bytes(payload) != self._canonical_payload:
            raise AdaptationStrategyError("semantic view canonical payload is invalid")
        if payload.get("schema_version") != B12_SEMANTIC_VIEW_SCHEMA_VERSION:
            raise AdaptationStrategyError("unsupported semantic view schema_version")
        if payload.get("packet_hash") != self.packet_hash:
            raise AdaptationStrategyError("semantic view packet hash does not match")
        target = payload.get("adaptation_target")
        if not isinstance(target, dict) or "source_snapshot" in target:
            raise AdaptationStrategyError("semantic view must not expose Python-owned source snapshot")
        event_index = payload.get("canonical_event_index")
        if not isinstance(event_index, list) or not event_index:
            raise AdaptationStrategyError("semantic view must preserve complete event index")
        event_ids = [item.get("event_id") for item in event_index if isinstance(item, dict)]
        if len(event_ids) != len(event_index) or len(set(event_ids)) != len(event_ids):
            raise AdaptationStrategyError("semantic view event IDs must be unique")
        windows = payload.get("global_event_analysis", {}).get("windows") if isinstance(payload.get("global_event_analysis"), dict) else None
        if not isinstance(windows, list) or not windows:
            raise AdaptationStrategyError("semantic view must preserve all plot windows")

    @classmethod
    def from_packet(cls, packet: B12Packet) -> "B12SemanticView":
        target = packet.resolved_target.to_dict().copy()
        target.pop("source_snapshot")
        payload = {
            "schema_version": B12_SEMANTIC_VIEW_SCHEMA_VERSION,
            "packet_hash": packet.content_hash(),
            "adaptation_target": target,
            "effective_directions": packet.effective_directions,
            "plotlines": [item.to_dict() for item in packet.plotlines],
            "story_bible": packet.story_bible,
            "global_structure": packet.global_structure,
            "character_analyses": list(packet.character_analyses),
            "global_event_analysis": packet.global_event_analysis,
            "arc_analysis": packet.arc_analysis,
            "canonical_event_index": list(packet.canonical_event_index),
            "canonical_relationship_index": list(packet.canonical_relationship_index),
            "unresolved_entity_index": list(packet.unresolved_entity_index),
            "source_conflict_index": list(packet.source_conflict_index),
        }
        return cls(packet.content_hash(), canonical_json_bytes(payload))

    def to_dict(self) -> dict[str, Any]:
        value = strict_json_loads(self._canonical_payload)
        assert isinstance(value, dict)
        return value

    def canonical_bytes(self) -> bytes: return self._canonical_payload
    def content_hash(self) -> str: return content_hash(strict_json_loads(self._canonical_payload))


@dataclass(frozen=True, slots=True)
class B12PacketMeasurement:
    packet_bytes: int
    packet_estimated_tokens: int
    component_sizes: tuple[tuple[str, int, int], ...]
    canonical_character_count: int
    canonical_event_count: int
    plot_window_count: int
    source_snapshot_hash: str
    packet_hash: str
    effective_b1_direction_hash: str
    rendered_prompt_bytes: int | None = None
    rendered_prompt_estimated_tokens: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"packet_bytes": self.packet_bytes, "packet_estimated_tokens": self.packet_estimated_tokens, "component_sizes": [{"component": n, "canonical_bytes": b, "estimated_tokens": t} for n, b, t in self.component_sizes], "canonical_character_count": self.canonical_character_count, "canonical_event_count": self.canonical_event_count, "plot_window_count": self.plot_window_count, "source_snapshot_hash": self.source_snapshot_hash, "packet_hash": self.packet_hash, "effective_b1_direction_hash": self.effective_b1_direction_hash, "rendered_prompt_bytes": self.rendered_prompt_bytes, "rendered_prompt_estimated_tokens": self.rendered_prompt_estimated_tokens}


@dataclass(frozen=True, slots=True)
class B12Preparation:
    packet: B12Packet
    semantic_view: B12SemanticView
    measurement: B12PacketMeasurement
    semantic_input_identity: str

    def validate_strategy(self, strategy: Any) -> None:
        """Mandatory semantic-acceptance seam for a decoded B1.2 output.

        ``AdaptationStrategy.from_dict`` is intentionally a pure structural
        decoder.  Any future B1.2-B acceptance/publication path must invoke
        this exact preparation-bound method before treating it as valid.
        """
        from .strategy import AdaptationStrategy
        if not isinstance(strategy, AdaptationStrategy):
            raise AdaptationStrategyError("semantic acceptance requires AdaptationStrategy")
        if strategy.resolved_target != self.packet.resolved_target:
            raise AdaptationStrategyError("strategy target differs from prepared exact target")
        strategy.validate_source_membership(self.packet.plotlines)
        effective = self.packet.effective_directions.get("effective_set", {})
        expected_refs = effective.get("direction_refs", []) if isinstance(effective, dict) else []
        allowed_direction_refs = {content_hash(item) for item in expected_refs if isinstance(item, dict)}
        for conflict in strategy.explicit_conflicts:
            if any(content_hash(ref.to_dict()) not in allowed_direction_refs for ref in conflict.user_direction_refs):
                raise AdaptationStrategyError("strategy conflict references a direction outside the exact effective B1 set")
        if strategy.ending_disposition.value == "adapted_ending" and not strategy.deliberate_divergences:
            raise AdaptationStrategyError("adapted_ending requires explicit deliberate divergence provenance")


def _direction_payload(store: FileArtifactStore, effective: Any) -> dict[str, Any]:
    return {"effective_set": effective.to_dict(), "directions": [load_user_direction(store, ref).to_dict() for ref in effective.direction_refs]}


def _component_sizes(packet: B12Packet) -> tuple[tuple[str, int, int], ...]:
    components = {
        "source_snapshot": packet.source_snapshot.to_dict(), "resolved_target": packet.resolved_target.to_dict(),
        "effective_directions": packet.effective_directions, "plotlines": [x.to_dict() for x in packet.plotlines],
        "story_bible": packet.story_bible, "global_structure": packet.global_structure,
        "character_analyses": list(packet.character_analyses), "global_event_analysis": packet.global_event_analysis,
        "arc_analysis": packet.arc_analysis, "canonical_event_index": list(packet.canonical_event_index),
        "canonical_relationship_index": list(packet.canonical_relationship_index),
        "unresolved_entity_index": list(packet.unresolved_entity_index),
        "source_conflict_index": list(packet.source_conflict_index),
    }
    return tuple((name, len(canonical_json_bytes(value)), estimate_tokens(canonical_json_bytes(value).decode("utf-8"))) for name, value in components.items())


def prepare_b12_packet(
    store: FileArtifactStore, pointers: FilePointerStore, resolved_target: ResolvedAdaptationTarget,
    direction_set: UserDirectionSet | None = None,
) -> B12Preparation:
    """Verify exact A5/A6 authority, project B1 guidance, then build one packet.

    This function is deliberately read-only: no artifact writes, pointer
    changes, prompts, semantic ceilings, or provider calls occur here.
    """
    if not isinstance(resolved_target, ResolvedAdaptationTarget):
        raise AdaptationStrategyError("resolved_target must be ResolvedAdaptationTarget")
    bound = AdaptationSourceSnapshot.from_store(store, pointers, resolved_target.source_snapshot.story_analysis_manifest_ref)
    if bound != resolved_target.source_snapshot:
        raise AdaptationStrategyError("resolved target source snapshot does not equal exact validated A5/A6 snapshot")
    # The service owns exact leaf identity validation.  Project/document/profile
    # are read from its verified manifest rather than caller parameters.
    from short_drama.story.story_analysis import StoryAnalysisManifest
    manifest = _load(store, bound.story_analysis_manifest_ref, StoryAnalysisManifest)
    base = a6_base_artifact_id(manifest.project_id, manifest.document_id, manifest.semantic_identity.story_analysis_profile_id)
    verified = StoryAnalysisPersistenceService(store, pointers)._load_current_integrity(base, bound.story_analysis_manifest_ref)
    _manifest, chars, events, arcs, structure, bible, _report = verified
    event_set = _load(store, bound.canonical_event_set_ref, CanonicalEventSet)
    ordered_events = tuple(sorted(event_set.events, key=lambda event: (event.narrative_order, event.event_id)))
    event_index = tuple({"event_id": e.event_id, "narrative_order": e.narrative_order, "summary_zh": e.summary_zh, "participants": list(e.participants), "locations": list(e.locations), "temporal_mode": e.temporal_mode} for e in ordered_events)
    expected_events = tuple(e.event_id for e in ordered_events)
    owned = tuple(event_id for window in events.windows for event_id in window.owned_event_refs)
    if len(owned) != len(set(owned)) or set(owned) != set(expected_events):
        raise AdaptationStrategyError("A6 global event windows do not provide exact complete A5 event coverage")
    if tuple(sorted(item.event_ref for item in events.event_importance_overlay)) and not set(item.event_ref for item in events.event_importance_overlay) <= set(expected_events):
        raise AdaptationStrategyError("A6 event importance references unknown A5 event")
    a5_manifest = _load(store, bound.consolidation_manifest_ref, ConsolidationManifest)
    if a5_manifest.entity_map_ref != bound.entity_map_ref:
        raise AdaptationStrategyError("bound A5 manifest disagrees with exact source EntityMap")
    entity_map = _load(store, bound.entity_map_ref, EntityMap)
    relationship_set = _load(store, bound.canonical_relationship_set_ref, CanonicalRelationshipSet)
    unresolved_set = _load(store, entity_map.unresolved_entity_set_ref, UnresolvedEntitySet)
    conflict_set = _load(store, a5_manifest.story_conflict_set_ref, StoryConflictSet)
    relationship_index = tuple(
        {"relationship_id": rel.relationship_id, "source_entity_ref": rel.source_entity_ref,
         "target_entity_ref": rel.target_entity_ref, "direction": rel.direction,
         "relationship_type_zh": rel.relationship_type_zh, "state_count": len(rel.state_history)}
        for rel in sorted(relationship_set.relationships, key=lambda item: item.relationship_id)
    )
    unresolved_index = tuple(
        {"unresolved_id": item.unresolved_id, "entity_kind": item.entity_kind,
         "candidate_refs": list(item.candidate_refs), "possible_candidate_refs": list(item.possible_candidate_refs),
         "first_appearance_candidate_ref": item.first_appearance_candidate_ref}
        for item in sorted(unresolved_set.entities, key=lambda item: item.unresolved_id)
    )
    conflict_index = tuple(
        {"conflict_id": item.conflict_id, "conflict_kind": item.conflict_kind,
         "fact_ids": list(item.fact_ids), "relationship_ids": list(item.relationship_ids), "status": item.status}
        for item in sorted(conflict_set.conflicts, key=lambda item: item.conflict_id)
    )
    catalog = TargetCatalog(manifest.project_id, tuple(TargetRef(TargetScope.CHARACTER, ref.canonical_id) for ref in bound.canonical_references if ref.kind == "character"), bound.canonical_character_registry_ref)
    if direction_set is None:
        direction_set = UserDirectionSet(manifest.project_id, "b12-empty-directions", 1, ())
    if direction_set.project_id != manifest.project_id:
        raise AdaptationStrategyError("G1 UserDirectionSet project does not match exact A6 snapshot")
    effective = project_effective_directions(store, direction_set, catalog, stage=DirectionStage.B1_ADAPTATION)
    packet = B12Packet.build(bound, resolved_target, _direction_payload(store, effective), derive_snapshot_local_plotlines(bound.story_analysis_manifest_ref, structure), bible.to_dict(), structure.to_dict(), tuple(x.to_dict() for x in chars.analyses), events.to_dict(), arcs.to_dict(), event_index, relationship_index, unresolved_index, conflict_index)
    sizes = _component_sizes(packet)
    raw = packet.canonical_bytes()
    measurement = B12PacketMeasurement(len(raw), estimate_tokens(raw.decode("utf-8")), sizes, len(chars.analyses), len(event_index), len(events.windows), content_hash(bound.to_dict()), packet.content_hash(), effective.effective_user_direction_set_hash)
    identity = content_hash({"stage": "B1.2", "a6_manifest_ref": bound.story_analysis_manifest_ref.to_dict(), "a5_manifest_ref": bound.consolidation_manifest_ref.to_dict(), "resolved_target_hash": resolved_target.content_hash(), "effective_b1_direction_hash": effective.effective_user_direction_set_hash, "packet_hash": packet.content_hash()})
    return B12Preparation(packet, B12SemanticView.from_packet(packet), measurement, identity)
