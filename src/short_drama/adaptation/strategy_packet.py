"""Deterministic, zero-provider B1.2-A packet preparation and measurement."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from short_drama.artifacts import ArtifactRef, FileArtifactStore, canonical_json_bytes, content_hash
from short_drama.foundation import FilePointerStore
from short_drama.guidance.models import DirectionStage, UserDirectionSet
from short_drama.guidance.projection import TargetCatalog, TargetRef, TargetScope, project_effective_directions, load_user_direction
from short_drama.story.chunking import estimate_tokens
from short_drama.story.consolidation import CanonicalEventSet
from short_drama.story.story_analysis_persistence import StoryAnalysisPersistenceService, a6_base_artifact_id

from .strategy import AdaptationStrategyError, SnapshotLocalPlotline
from .target import AdaptationSourceSnapshot, ResolvedAdaptationTarget


B12_PACKET_SCHEMA_VERSION = 1


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
    """Whole-story compact semantic input, with no prompt/provider configuration."""
    source_snapshot: AdaptationSourceSnapshot
    resolved_target: ResolvedAdaptationTarget
    effective_directions: dict[str, Any]
    plotlines: tuple[SnapshotLocalPlotline, ...]
    story_bible: dict[str, Any]
    global_structure: dict[str, Any]
    character_analyses: tuple[dict[str, Any], ...]
    global_event_analysis: dict[str, Any]
    arc_analysis: dict[str, Any]
    canonical_event_index: tuple[dict[str, Any], ...]

    def __post_init__(self) -> None:
        if self.resolved_target.source_snapshot != self.source_snapshot:
            raise AdaptationStrategyError("packet target must bind exactly the packet source snapshot")
        event_ids = [entry.get("event_id") for entry in self.canonical_event_index]
        if not event_ids or any(not isinstance(x, str) for x in event_ids) or len(set(event_ids)) != len(event_ids):
            raise AdaptationStrategyError("packet canonical event index must contain unique complete IDs")

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": B12_PACKET_SCHEMA_VERSION, "source_snapshot": self.source_snapshot.to_dict(), "resolved_target": self.resolved_target.to_dict(), "effective_directions": self.effective_directions, "plotlines": [x.to_dict() for x in self.plotlines], "story_bible": self.story_bible, "global_structure": self.global_structure, "character_analyses": list(self.character_analyses), "global_event_analysis": self.global_event_analysis, "arc_analysis": self.arc_analysis, "canonical_event_index": list(self.canonical_event_index)}

    def canonical_bytes(self) -> bytes: return canonical_json_bytes(self.to_dict())
    def content_hash(self) -> str: return content_hash(self.to_dict())


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
    measurement: B12PacketMeasurement
    semantic_input_identity: str


def _direction_payload(store: FileArtifactStore, effective: Any) -> dict[str, Any]:
    return {"effective_set": effective.to_dict(), "directions": [load_user_direction(store, ref).to_dict() for ref in effective.direction_refs]}


def _component_sizes(packet: B12Packet) -> tuple[tuple[str, int, int], ...]:
    components = {
        "source_snapshot": packet.source_snapshot.to_dict(), "resolved_target": packet.resolved_target.to_dict(),
        "effective_directions": packet.effective_directions, "plotlines": [x.to_dict() for x in packet.plotlines],
        "story_bible": packet.story_bible, "global_structure": packet.global_structure,
        "character_analyses": list(packet.character_analyses), "global_event_analysis": packet.global_event_analysis,
        "arc_analysis": packet.arc_analysis, "canonical_event_index": list(packet.canonical_event_index),
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
    event_index = tuple({"event_id": e.event_id, "narrative_order": e.narrative_order, "summary_zh": e.summary_zh, "participants": list(e.participants), "locations": list(e.locations), "temporal_mode": e.temporal_mode} for e in event_set.events)
    expected_events = tuple(e.event_id for e in event_set.events)
    owned = tuple(event_id for window in events.windows for event_id in window.owned_event_refs)
    if len(owned) != len(set(owned)) or set(owned) != set(expected_events):
        raise AdaptationStrategyError("A6 global event windows do not provide exact complete A5 event coverage")
    if tuple(sorted(item.event_ref for item in events.event_importance_overlay)) and not set(item.event_ref for item in events.event_importance_overlay) <= set(expected_events):
        raise AdaptationStrategyError("A6 event importance references unknown A5 event")
    catalog = TargetCatalog(manifest.project_id, tuple(TargetRef(TargetScope.CHARACTER, ref.canonical_id) for ref in bound.canonical_references if ref.kind == "character"), bound.canonical_character_registry_ref)
    if direction_set is None:
        direction_set = UserDirectionSet(manifest.project_id, "b12-empty-directions", 1, ())
    if direction_set.project_id != manifest.project_id:
        raise AdaptationStrategyError("G1 UserDirectionSet project does not match exact A6 snapshot")
    effective = project_effective_directions(store, direction_set, catalog, stage=DirectionStage.B1_ADAPTATION)
    packet = B12Packet(bound, resolved_target, _direction_payload(store, effective), derive_snapshot_local_plotlines(bound.story_analysis_manifest_ref, structure), bible.to_dict(), structure.to_dict(), tuple(x.to_dict() for x in chars.analyses), events.to_dict(), arcs.to_dict(), event_index)
    sizes = _component_sizes(packet)
    raw = packet.canonical_bytes()
    measurement = B12PacketMeasurement(len(raw), estimate_tokens(raw.decode("utf-8")), sizes, len(chars.analyses), len(event_index), len(events.windows), content_hash(bound.to_dict()), packet.content_hash(), effective.effective_user_direction_set_hash)
    identity = content_hash({"stage": "B1.2", "a6_manifest_ref": bound.story_analysis_manifest_ref.to_dict(), "a5_manifest_ref": bound.consolidation_manifest_ref.to_dict(), "resolved_target_hash": resolved_target.content_hash(), "effective_b1_direction_hash": effective.effective_user_direction_set_hash, "packet_hash": packet.content_hash()})
    return B12Preparation(packet, measurement, identity)
