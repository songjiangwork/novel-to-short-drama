"""Deterministic, read-only B1.3-B1 disposition-window preparation.

This is deliberately the seam *before* B1.3 semantic execution.  It resolves
already-persisted authority, derives a complete A5-narrative-ordered event
universe, and creates one immutable packet per A6-owned plot window.  It does
not call a provider, make a disposition, reconcile proposals, or publish.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from short_drama.artifacts import (
    ArtifactError, ArtifactRef, FileArtifactStore, canonical_json_bytes,
    content_hash, strict_json_loads,
)
from short_drama.foundation import FilePointerStore, PointerKind, PointerNotFoundError
from short_drama.story.chunking import estimate_tokens
from short_drama.story.consolidation import CanonicalEvent, CanonicalEventSet
from short_drama.story.story_analysis import (
    ArcAnalysis, GlobalEventAnalysis, PlotWindowAnalysis, StoryAnalysisManifest,
)
from short_drama.story.story_analysis_persistence import (
    StoryAnalysisPersistenceService, a6_base_artifact_id,
)
from short_drama.story.story_analysis_planning import (
    PlotWindowPlan, validate_window_ownership,
)

from .strategy_execution import (
    B12_STRATEGY_ARTIFACT_SCHEMA_VERSION, B12_STRATEGY_ARTIFACT_TYPE,
    B12StrategyArtifact, _B12StrategyPersistenceService,
    b12_strategy_artifact_id, b12_strategy_pointer_id,
)
from .target import AdaptationSourceSnapshot


B13_DISPOSITION_PACKET_SCHEMA_VERSION = 1


class B13PreparationError(ValueError):
    """A B1.3-B1 read-only authority, window, or packet invariant failed."""


def _load(store: FileArtifactStore, ref: ArtifactRef, model: Any) -> Any:
    try:
        payload = store.get_ref(ref).payload
        value = model.from_dict(payload)
    except Exception as exc:
        raise B13PreparationError(
            f"cannot resolve exact {getattr(ref, 'artifact_type', 'unknown')} source leaf"
        ) from exc
    if value.to_dict() != payload:
        raise B13PreparationError(f"persisted {ref.artifact_type} is not canonical")
    return value


@dataclass(frozen=True, slots=True)
class B12StrategySelection:
    """A fully verified B1.2 strategy revision selected for B1.3.

    ``current_pointer_ref`` is present only where this selection came from the
    stage-local CURRENT pointer.  The target strategy revision itself is always
    explicit, including when it was supplied by the caller.
    """

    strategy_ref: ArtifactRef
    artifact: B12StrategyArtifact
    current_pointer_ref: ArtifactRef | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.strategy_ref, ArtifactRef) or self.strategy_ref.artifact_type != B12_STRATEGY_ARTIFACT_TYPE:
            raise B13PreparationError("selected B1.2 strategy_ref must be an adaptation_strategy ArtifactRef")
        if not isinstance(self.artifact, B12StrategyArtifact):
            raise B13PreparationError("selected B1.2 strategy must be a B12StrategyArtifact")
        if self.current_pointer_ref is not None and (
            not isinstance(self.current_pointer_ref, ArtifactRef)
            or self.current_pointer_ref.artifact_type != "current_pointer"
        ):
            raise B13PreparationError("selected B1.2 current_pointer_ref must be a current_pointer ArtifactRef")

    @property
    def strategy_hash(self) -> str:
        return self.artifact.strategy.content_hash()


def resolve_b12_strategy_selection(
    *,
    store: FileArtifactStore,
    pointers: FilePointerStore,
    source_snapshot: AdaptationSourceSnapshot,
    adaptation_target_id: str,
    strategy_ref: ArtifactRef | None = None,
) -> B12StrategySelection:
    """Select and fully validate an immutable B1.2 strategy for this snapshot.

    With ``strategy_ref`` this pins precisely that revision; otherwise it
    resolves B1.2's CURRENT pointer and records the pointer revision used.  In
    both modes it re-verifies A5/A6 CURRENT authority and B1.2's intrinsic
    lineage/provenance before returning.  There is no historical search or
    fallback to a merely available strategy revision.
    """
    if not isinstance(store, FileArtifactStore) or not isinstance(pointers, FilePointerStore):
        raise B13PreparationError("store/pointers must be Foundation file stores")
    if not isinstance(source_snapshot, AdaptationSourceSnapshot):
        raise B13PreparationError("source_snapshot must be an exact AdaptationSourceSnapshot")
    if not isinstance(adaptation_target_id, str) or not adaptation_target_id:
        raise B13PreparationError("adaptation_target_id must be non-empty text")

    # This is intentionally before either explicit-ref or CURRENT selection:
    # a stale/corrupt A5/A6 input must never be masked by a valid old strategy.
    bound = AdaptationSourceSnapshot.from_store(
        store, pointers, source_snapshot.story_analysis_manifest_ref
    )
    if bound != source_snapshot:
        raise B13PreparationError("B1.3 source snapshot is not the exact validated A5/A6 CURRENT")
    manifest = _load(store, bound.story_analysis_manifest_ref, StoryAnalysisManifest)
    expected_id = b12_strategy_artifact_id(manifest.project_id, adaptation_target_id)
    pointer_ref: ArtifactRef | None = None
    if strategy_ref is None:
        pointer_id = b12_strategy_pointer_id(manifest.project_id, adaptation_target_id)
        try:
            pointer_ref = pointers.resolve_current_pointer_ref(pointer_id)
            # Resolve this exact immutable snapshot, rather than asking the
            # mutable head a second time through ``resolve_current``.  A later
            # head check below detects (and rejects) any concurrent advance.
            pointer = pointers._load_pointer_ref(pointer_ref)
        except PointerNotFoundError as exc:
            raise B13PreparationError("B1.2 CURRENT strategy is missing") from exc
        except Exception as exc:
            raise B13PreparationError("B1.2 CURRENT pointer fails integrity verification") from exc
        if (
            pointer.pointer_kind is not PointerKind.CURRENT
            or pointer.target_ref.artifact_type != B12_STRATEGY_ARTIFACT_TYPE
            or pointer.target_ref.artifact_id != expected_id
        ):
            raise B13PreparationError("B1.2 CURRENT pointer has invalid identity")
        selected_ref = pointer.target_ref
    else:
        selected_ref = strategy_ref
        if (
            selected_ref.artifact_type != B12_STRATEGY_ARTIFACT_TYPE
            or selected_ref.artifact_id != expected_id
        ):
            raise B13PreparationError("selected B1.2 strategy revision has invalid identity")
    try:
        envelope = store.get_ref(selected_ref)
    except ArtifactError as exc:
        raise B13PreparationError("selected B1.2 strategy revision fails integrity verification") from exc
    if envelope.schema_version != B12_STRATEGY_ARTIFACT_SCHEMA_VERSION or not isinstance(envelope.payload, dict):
        raise B13PreparationError("selected B1.2 strategy revision has unsupported schema")
    try:
        artifact = B12StrategyArtifact.from_dict(envelope.payload)
        _B12StrategyPersistenceService(store, pointers)._validate_intrinsic_artifact(artifact)
    except Exception as exc:
        raise B13PreparationError("selected B1.2 strategy revision is unverified") from exc
    if artifact.source_snapshot != bound:
        raise B13PreparationError("selected B1.2 strategy has mismatched A5/A6 source lineage")
    if artifact.strategy.resolved_target.adaptation_target_id != adaptation_target_id:
        raise B13PreparationError("selected B1.2 strategy has mismatched adaptation target")
    # Guard the source authority again after all leaf/provenance reads.
    if AdaptationSourceSnapshot.from_store(store, pointers, bound.story_analysis_manifest_ref) != bound:
        raise B13PreparationError("A5/A6 CURRENT changed during B1.2 strategy selection")
    if pointer_ref is not None:
        try:
            if pointers.resolve_current_pointer_ref(pointer_id) != pointer_ref:
                raise B13PreparationError("B1.2 CURRENT pointer changed during strategy selection")
        except PointerNotFoundError as exc:
            raise B13PreparationError("B1.2 CURRENT pointer changed during strategy selection") from exc
    return B12StrategySelection(selected_ref, artifact, pointer_ref)


def _ordered_events(event_set: CanonicalEventSet) -> tuple[CanonicalEvent, ...]:
    events = tuple(event_set.events)
    if not events:
        raise B13PreparationError("A5 canonical event universe must not be empty")
    if len({event.event_id for event in events}) != len(events):
        raise B13PreparationError("A5 canonical event universe contains duplicate event IDs")
    return tuple(sorted(events, key=lambda event: (event.narrative_order, event.event_id)))


def _validate_a6_windows(
    windows: tuple[PlotWindowAnalysis, ...], ordered_events: tuple[CanonicalEvent, ...]
) -> tuple[PlotWindowAnalysis, ...]:
    """Apply A6's ownership contract plus B1.3 canonical ordering checks."""
    if not windows:
        raise B13PreparationError("A6 global event analysis has no windows")
    ordered = tuple(sorted(windows, key=lambda item: item.window_ordinal))
    if windows != ordered:
        raise B13PreparationError("A6 global event windows are not in deterministic ordinal order")
    all_ids = tuple(event.event_id for event in ordered_events)
    position = {event_id: index for index, event_id in enumerate(all_ids)}
    plans = tuple(
        PlotWindowPlan(window.window_id, window.window_ordinal, window.owned_event_refs, window.context_event_refs)
        for window in ordered
    )
    try:
        validate_window_ownership(plans, frozenset(all_ids))
    except Exception as exc:
        raise B13PreparationError("A6 window ownership does not provide exact-one canonical coverage") from exc
    for window in ordered:
        refs = (*window.owned_event_refs, *window.context_event_refs)
        if len(window.context_event_refs) != len(set(window.context_event_refs)):
            raise B13PreparationError(f"A6 window {window.window_id!r} repeats a context event")
        if tuple(sorted(window.owned_event_refs, key=position.__getitem__)) != window.owned_event_refs:
            raise B13PreparationError(f"A6 window {window.window_id!r} owned events are not in narrative order")
        if tuple(sorted(window.context_event_refs, key=position.__getitem__)) != window.context_event_refs:
            raise B13PreparationError(f"A6 window {window.window_id!r} context events are not in narrative order")
        if any(ref not in position for ref in refs):  # defensive; planner already checks it
            raise B13PreparationError(f"A6 window {window.window_id!r} contains an unknown event reference")
    counts = Counter(ref for window in ordered for ref in window.owned_event_refs)
    if tuple(ref for ref in all_ids if counts[ref] != 1):
        raise B13PreparationError("A6 window ownership does not give every event exactly one owner")
    if tuple(ref for window in ordered for ref in window.owned_event_refs) != all_ids:
        raise B13PreparationError("A6 window owned-event sequence does not match A5 narrative order")
    return ordered


def _relevant_arc_context(arcs: ArcAnalysis, event_ids: frozenset[str]) -> dict[str, Any]:
    """Select A6 arc records by exact event-ref intersection, never semantics."""
    return {
        "arcs": [item.to_dict() for item in arcs.arcs if event_ids & set(item.supporting_event_refs + (item.start_event_ref, item.end_event_ref))],
        "turning_points": [item.to_dict() for item in arcs.turning_points if item.event_ref in event_ids],
        "reveals": [item.to_dict() for item in arcs.reveals if event_ids & set(item.reveal_event_refs + item.setup_event_refs)],
        "foreshadow_payoffs": [item.to_dict() for item in arcs.foreshadow_payoffs if event_ids & set(item.setup_event_refs + item.payoff_event_refs)],
    }


def build_b13_disposition_preparation_from_authority(
    *,
    source_snapshot: AdaptationSourceSnapshot,
    strategy_selection: B12StrategySelection,
    event_set: CanonicalEventSet,
    event_analysis: GlobalEventAnalysis,
    arc_analysis: ArcAnalysis,
) -> "B13DispositionPreparation":
    """Materialize B1.3 packets from already verified immutable authority.

    This is deliberately pure: callers own source/CURRENT selection and exact
    artifact loading.  B4B1 uses the same construction to verify a historical
    B4A publication without consulting any live head.
    """
    if source_snapshot != strategy_selection.artifact.source_snapshot:
        raise B13PreparationError("B1.3 packet authority source/strategy mismatch")
    ordered_events = _ordered_events(event_set)
    windows = _validate_a6_windows(event_analysis.windows, ordered_events)
    event_by_id = {event.event_id: event for event in ordered_events}
    importance = {item.event_ref: item for item in event_analysis.event_importance_overlay}
    if not set(importance) <= set(event_by_id):
        raise B13PreparationError("A6 event-importance overlay references an unknown canonical event")

    packets: list[B13DispositionWindowPacket] = []
    measurements: list[B13DispositionPacketMeasurement] = []
    for window in windows:
        covered = frozenset((*window.owned_event_refs, *window.context_event_refs))
        payload = {
            "schema_version": B13_DISPOSITION_PACKET_SCHEMA_VERSION,
            "source_snapshot": source_snapshot.to_dict(),
            "global_adaptation_strategy": {
                "strategy_ref": strategy_selection.strategy_ref.to_dict(),
                "strategy_hash": strategy_selection.strategy_hash,
                "strategy": strategy_selection.artifact.strategy.to_dict(),
            },
            "window": {
                "window_id": window.window_id,
                "window_ordinal": window.window_ordinal,
                "owned_event_refs": list(window.owned_event_refs),
                "context_event_refs": list(window.context_event_refs),
            },
            "owned_source_events": [event_by_id[event_id].to_dict() for event_id in window.owned_event_refs],
            "context_source_events": [event_by_id[event_id].to_dict() for event_id in window.context_event_refs],
            "a6_window_analysis": window.to_dict(),
            "a6_event_importance_context": [importance[event_id].to_dict() for event_id in sorted(covered, key=lambda event_id: (event_by_id[event_id].narrative_order, event_id)) if event_id in importance],
            "a6_arc_context": _relevant_arc_context(arc_analysis, covered),
        }
        raw = canonical_json_bytes(payload)
        packet = B13DispositionWindowPacket(
            source_snapshot, strategy_selection, window.window_id, window.window_ordinal, raw
        )
        packets.append(packet)
        measurements.append(B13DispositionPacketMeasurement(
            window.window_id, len(raw), estimate_tokens(raw.decode("utf-8")), packet.content_hash()
        ))
    identity = content_hash({
        "stage": "B1.3-B1",
        "source_snapshot": source_snapshot.to_dict(),
        "strategy_ref": strategy_selection.strategy_ref.to_dict(),
        "strategy_hash": strategy_selection.strategy_hash,
        "packet_hashes": [packet.content_hash() for packet in packets],
    })
    return B13DispositionPreparation(
        source_snapshot, strategy_selection, tuple(packets), tuple(measurements),
        tuple(event.event_id for event in ordered_events), identity,
    )


@dataclass(frozen=True, slots=True)
class B13DispositionWindowPacket:
    """Canonical, immutable input for exactly one future B1.3 semantic window."""

    source_snapshot: AdaptationSourceSnapshot
    strategy_selection: B12StrategySelection
    window_id: str
    window_ordinal: int
    _canonical_payload: bytes

    def __post_init__(self) -> None:
        if self.source_snapshot != self.strategy_selection.artifact.source_snapshot:
            raise B13PreparationError("B1.3 packet source snapshot must match selected B1.2 strategy")
        if not isinstance(self._canonical_payload, bytes):
            raise B13PreparationError("B1.3 packet canonical payload must be bytes")
        try:
            payload = strict_json_loads(self._canonical_payload)
        except Exception as exc:
            raise B13PreparationError("B1.3 packet canonical payload is invalid") from exc
        if not isinstance(payload, dict) or canonical_json_bytes(payload) != self._canonical_payload:
            raise B13PreparationError("B1.3 packet canonical payload is invalid")
        if payload.get("schema_version") != B13_DISPOSITION_PACKET_SCHEMA_VERSION:
            raise B13PreparationError("unsupported B1.3 disposition packet schema_version")
        if payload.get("source_snapshot") != self.source_snapshot.to_dict():
            raise B13PreparationError("B1.3 packet source snapshot does not match")
        strategy = payload.get("global_adaptation_strategy")
        if not isinstance(strategy, dict) or strategy.get("strategy_ref") != self.strategy_selection.strategy_ref.to_dict() or strategy.get("strategy_hash") != self.strategy_selection.strategy_hash or strategy.get("strategy") != self.strategy_selection.artifact.strategy.to_dict():
            raise B13PreparationError("B1.3 packet global strategy identity does not match")
        window = payload.get("window")
        if not isinstance(window, dict) or window.get("window_id") != self.window_id or window.get("window_ordinal") != self.window_ordinal:
            raise B13PreparationError("B1.3 packet window identity does not match")
        owned = window.get("owned_event_refs")
        context = window.get("context_event_refs")
        if not isinstance(owned, list) or not owned or not isinstance(context, list):
            raise B13PreparationError("B1.3 packet window event references are invalid")
        if set(owned) & set(context) or len(owned) != len(set(owned)) or len(context) != len(set(context)):
            raise B13PreparationError("B1.3 packet context must never confer owned-event disposition")
        known_event_ids = {
            reference.canonical_id for reference in self.source_snapshot.canonical_references
            if reference.kind == "event"
        }
        if not set(owned) <= known_event_ids or not set(context) <= known_event_ids:
            raise B13PreparationError("B1.3 packet window references an unknown canonical event")
        owned_events = payload.get("owned_source_events")
        context_events = payload.get("context_source_events")
        if not isinstance(owned_events, list) or not isinstance(context_events, list):
            raise B13PreparationError("B1.3 packet source events are invalid")
        if [event.get("event_id") for event in owned_events if isinstance(event, dict)] != owned:
            raise B13PreparationError("B1.3 packet owned source events do not preserve exact IDs")
        if [event.get("event_id") for event in context_events if isinstance(event, dict)] != context:
            raise B13PreparationError("B1.3 packet context source events do not preserve exact IDs")
        for events, label in ((owned_events, "owned"), (context_events, "context")):
            positions = [(event.get("narrative_order"), event.get("event_id")) for event in events]
            if any(not isinstance(order, int) or not isinstance(event_id, str) for order, event_id in positions) or positions != sorted(positions):
                raise B13PreparationError(f"B1.3 packet {label} source events are not in narrative order")

    def to_dict(self) -> dict[str, Any]:
        payload = strict_json_loads(self._canonical_payload)
        assert isinstance(payload, dict)
        return payload

    def canonical_bytes(self) -> bytes:
        return self._canonical_payload

    def content_hash(self) -> str:
        return content_hash(strict_json_loads(self._canonical_payload))


@dataclass(frozen=True, slots=True)
class B13DispositionPacketMeasurement:
    window_id: str
    packet_bytes: int
    packet_estimated_tokens: int
    packet_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "window_id": self.window_id,
            "packet_bytes": self.packet_bytes,
            "packet_estimated_tokens": self.packet_estimated_tokens,
            "packet_hash": self.packet_hash,
        }


@dataclass(frozen=True, slots=True)
class B13DispositionPreparation:
    """Complete, read-only B1.3-B1 preparation result for later semantics."""

    source_snapshot: AdaptationSourceSnapshot
    strategy_selection: B12StrategySelection
    packets: tuple[B13DispositionWindowPacket, ...]
    measurements: tuple[B13DispositionPacketMeasurement, ...]
    canonical_event_ids: tuple[str, ...]
    preparation_identity: str

    def __post_init__(self) -> None:
        if self.source_snapshot != self.strategy_selection.artifact.source_snapshot:
            raise B13PreparationError("B1.3 preparation source snapshot must match selected B1.2 strategy")
        packets = tuple(self.packets)
        measurements = tuple(self.measurements)
        if not packets or len(packets) != len(measurements):
            raise B13PreparationError("B1.3 preparation requires one packet and measurement per window")
        if tuple(packet.window_ordinal for packet in packets) != tuple(range(1, len(packets) + 1)):
            raise B13PreparationError("B1.3 packets must be in deterministic window order")
        if tuple(item.window_id for item in measurements) != tuple(packet.window_id for packet in packets):
            raise B13PreparationError("B1.3 packet measurements do not match windows")
        if any(packet.source_snapshot != self.source_snapshot or packet.strategy_selection != self.strategy_selection for packet in packets):
            raise B13PreparationError("B1.3 packets must bind the exact selected source and strategy")
        owned = tuple(
            event_id for packet in packets
            for event_id in packet.to_dict()["window"]["owned_event_refs"]
        )
        if len(owned) != len(set(owned)) or set(owned) != set(self.canonical_event_ids):
            raise B13PreparationError("B1.3 packets do not provide exact-one canonical event coverage")
        object.__setattr__(self, "packets", packets)
        object.__setattr__(self, "measurements", measurements)
        object.__setattr__(self, "canonical_event_ids", tuple(self.canonical_event_ids))


def prepare_b13_disposition_windows(
    *,
    store: FileArtifactStore,
    pointers: FilePointerStore,
    source_snapshot: AdaptationSourceSnapshot,
    adaptation_target_id: str,
    strategy_ref: ArtifactRef | None = None,
) -> B13DispositionPreparation:
    """Create all B1.3 disposition-window packets without provider side effects.

    This has no input-token ceiling by design: it measures the actual canonical
    bytes/tokens for every packet and leaves any later provider budget decision
    to B1.3-B2.
    """
    selection = resolve_b12_strategy_selection(
        store=store, pointers=pointers, source_snapshot=source_snapshot,
        adaptation_target_id=adaptation_target_id, strategy_ref=strategy_ref,
    )
    bound = selection.artifact.source_snapshot
    manifest = _load(store, bound.story_analysis_manifest_ref, StoryAnalysisManifest)
    service = StoryAnalysisPersistenceService(store, pointers)
    base = a6_base_artifact_id(
        manifest.project_id, manifest.document_id, manifest.semantic_identity.story_analysis_profile_id
    )
    try:
        verified_manifest, _characters, event_analysis, arc_analysis, _structure, _bible, _report = service._load_current_integrity(
            base, bound.story_analysis_manifest_ref
        )
    except Exception as exc:
        raise B13PreparationError("cannot verify exact A6 global event authority") from exc
    if verified_manifest != manifest:
        raise B13PreparationError("verified A6 manifest differs from selected source snapshot")
    event_set = _load(store, bound.canonical_event_set_ref, CanonicalEventSet)
    return build_b13_disposition_preparation_from_authority(
        source_snapshot=bound, strategy_selection=selection, event_set=event_set,
        event_analysis=event_analysis, arc_analysis=arc_analysis,
    )


__all__ = [
    "B13_DISPOSITION_PACKET_SCHEMA_VERSION", "B13PreparationError",
    "B12StrategySelection", "resolve_b12_strategy_selection",
    "B13DispositionWindowPacket", "B13DispositionPacketMeasurement",
    "B13DispositionPreparation", "build_b13_disposition_preparation_from_authority",
    "prepare_b13_disposition_windows",
]
