"""Deterministic, read-only validation for a B1.4 story skeleton.

This deliberately validates graph shape and pinned authority, not narrative
plausibility.  In particular, an absent edge in chronology is *not* evidence
of the reverse order.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from short_drama.artifacts import ArtifactRef, FileArtifactStore
from short_drama.foundation import FilePointerStore

from .disposition import B13DispositionPlan
from .disposition_persistence import B13DispositionPersistenceService
from .disposition_publication import verify_b13_disposition_publication
from .skeleton import (
    AdaptedStorySkeleton, BeatPresentationTreatment, GapAnchorKind,
    KnowledgeStatus, KnowledgeSubjectKind, PresentationEntry,
)


class B14SkeletonValidationError(ValueError):
    """The skeleton has a deterministic structural or authority defect."""


@dataclass(frozen=True, slots=True)
class B14ValidationFinding:
    code: str
    subject_id: str | None
    message: str


@dataclass(frozen=True, slots=True)
class PrimaryPresentationIdentity:
    """The B1.5-consumable identity of one primary presentation occurrence."""

    presentation_id: str
    adapted_story_beat_id: str
    presentation_index: int


@dataclass(frozen=True, slots=True)
class B14SkeletonAuthority:
    """Exact B1.2/B1.3 authority already verified by the B1.3 verifier."""

    strategy_ref: ArtifactRef
    disposition_publication_ref: ArtifactRef
    disposition_plan: B13DispositionPlan


def resolve_b14_skeleton_semantic_authority(
    *, strategy_ref: ArtifactRef, disposition_publication_ref: ArtifactRef,
    store: FileArtifactStore, pointers: FilePointerStore,
) -> B14SkeletonAuthority:
    """Resolve explicit B1.4 inputs through the existing B1.3 verifier.

    This pre-skeleton entry point never consults CURRENT.  The B1.3
    persistence boundary remains the single owner of historical publication,
    source, and B1.2 selection verification.
    """
    if not isinstance(strategy_ref, ArtifactRef) or not isinstance(disposition_publication_ref, ArtifactRef):
        raise B14SkeletonValidationError("B1.4 semantic preparation requires exact B1.2/B1.3 ArtifactRefs")
    try:
        service = B13DispositionPersistenceService(store, pointers)
        publication = service._load_and_verify_publication(disposition_publication_ref)
        _historical, selection = service._verify_historical_publication(publication)
        plan = verify_b13_disposition_publication(publication).plan
        if selection.strategy_ref != strategy_ref:
            raise ValueError("explicit B1.2 strategy differs from B1.3 publication selection")
        if plan.source_snapshot != selection.artifact.source_snapshot:
            raise ValueError("B1.3 plan/source snapshot differs from exact B1.2 authority")
        return B14SkeletonAuthority(strategy_ref, disposition_publication_ref, plan)
    except Exception as exc:
        raise B14SkeletonValidationError("pinned B1.2/B1.3 semantic authority cannot be verified") from exc


@dataclass(frozen=True, slots=True)
class B14SkeletonValidationResult:
    findings: tuple[B14ValidationFinding, ...]
    primary_presentation: tuple[PrimaryPresentationIdentity, ...]

    @property
    def valid(self) -> bool:
        return not self.findings

    def require_valid(self) -> "B14SkeletonValidationResult":
        if self.findings:
            raise B14SkeletonValidationError("; ".join(
                f"{item.code}{'' if item.subject_id is None else ':' + item.subject_id}"
                for item in self.findings
            ))
        return self


def resolve_b14_skeleton_authority(
    skeleton: AdaptedStorySkeleton, *, store: FileArtifactStore, pointers: FilePointerStore,
) -> B14SkeletonAuthority:
    """Resolve only the refs pinned by ``skeleton``; never consult CURRENT.

    The B1.3 persistence service remains the single owner of its publication,
    historical-source, and B1.2 selection verification.  This adapter merely
    exposes the already-verified final plan to B1.4.
    """
    if not isinstance(skeleton, AdaptedStorySkeleton):
        raise B14SkeletonValidationError("skeleton must be an AdaptedStorySkeleton")
    return resolve_b14_skeleton_semantic_authority(
        strategy_ref=skeleton.strategy_ref,
        disposition_publication_ref=skeleton.disposition_publication_ref,
        store=store,
        pointers=pointers,
    )


def validate_stored_adapted_story_skeleton(
    skeleton: AdaptedStorySkeleton, *, store: FileArtifactStore, pointers: FilePointerStore,
) -> B14SkeletonValidationResult:
    """Read-only full entry point for a persisted B1.4 candidate."""
    return validate_adapted_story_skeleton(
        skeleton, authority=resolve_b14_skeleton_authority(skeleton, store=store, pointers=pointers),
    )


def validate_adapted_story_skeleton(
    skeleton: AdaptedStorySkeleton, *, authority: B14SkeletonAuthority,
) -> B14SkeletonValidationResult:
    """Validate a candidate against an exact, previously verified B1.3 plan.

    This pure entry point is intentionally useful for semantic-stage candidate
    validation tests.  Production callers should obtain ``authority`` through
    :func:`resolve_b14_skeleton_authority`, rather than constructing an ID set.
    """
    if not isinstance(skeleton, AdaptedStorySkeleton) or not isinstance(authority, B14SkeletonAuthority):
        raise B14SkeletonValidationError("skeleton and verified B14 authority are required")
    findings: list[B14ValidationFinding] = []
    def fail(code: str, message: str, subject: str | None = None) -> None:
        findings.append(B14ValidationFinding(code, subject, message))

    if skeleton.strategy_ref != authority.strategy_ref:
        fail("authority.strategy_ref", "skeleton does not pin the verified B1.2 strategy")
    if skeleton.disposition_publication_ref != authority.disposition_publication_ref:
        fail("authority.disposition_ref", "skeleton does not pin the verified B1.3 publication")
    plan = authority.disposition_plan
    upstream_beats = {item.adapted_story_beat_id: item for item in plan.adapted_story_beats}
    local_beats = {item.adapted_story_beat_id: item for item in skeleton.local_invented_beats}
    if set(upstream_beats) & set(local_beats):
        for beat_id in sorted(set(upstream_beats) & set(local_beats)):
            fail("beat.local_identity_conflict", "local invented beat conflicts with B1.3 identity", beat_id)
    beats = set(upstream_beats) | set(local_beats)
    characters = {item.adapted_character_id for item in plan.adapted_characters}
    canonical_refs = set(plan.source_snapshot.canonical_references)
    for beat_id, beat in local_beats.items():
        if not beat.canonical_anchor_refs or beat.invention_purpose is None or not beat.invention_provenance_zh:
            fail("beat.local_provenance", "local invented beat lacks purpose, anchors, or provenance", beat_id)
        if any(ref not in canonical_refs for ref in beat.canonical_anchor_refs):
            fail("beat.local_anchor", "local invented beat anchor is outside the exact source snapshot", beat_id)

    projection = _primary_projection(skeleton.primary_presentation, beats, fail)
    index = {item.presentation_id: item.presentation_index for item in projection}

    edges: dict[str, set[str]] = {beat_id: set() for beat_id in beats}
    for edge in skeleton.chronology:
        if edge.before_beat_id not in beats or edge.after_beat_id not in beats:
            fail("chronology.endpoint", "chronology endpoint does not resolve to an active beat", f"{edge.before_beat_id}->{edge.after_beat_id}")
            continue
        edges[edge.before_beat_id].add(edge.after_beat_id)
    cyclic = _cyclic_nodes(edges)
    for beat_id in sorted(cyclic):
        fail("chronology.cycle", "chronology must be a DAG", beat_id)

    def precedes(before: str, after: str) -> bool:
        return _reachable(edges, before, after)
    def reverse_or_unknown(before: str, after: str, code: str, label: str) -> None:
        if precedes(after, before):
            fail(code + ".contradiction", f"{label} contradicts explicit chronology", f"{before}->{after}")
        elif not precedes(before, after):
            fail(code + ".unproven", f"{label} requires explicit chronology precedence", f"{before}->{after}")

    outcomes = {item.outcome_id: item for item in skeleton.outcomes}
    for outcome in skeleton.outcomes:
        if outcome.produced_by_beat_id not in beats:
            fail("outcome.producer", "outcome producer beat does not resolve", outcome.outcome_id)
    for dependency in skeleton.dependencies:
        outcome = outcomes.get(dependency.prerequisite_outcome_id)
        if outcome is None or dependency.required_by_beat_id not in beats:
            fail("dependency.reference", "dependency outcome or required beat does not resolve", dependency.dependency_id)
        elif outcome.produced_by_beat_id == dependency.required_by_beat_id:
            fail("dependency.self", "dependency cannot require an outcome produced by the same beat", dependency.dependency_id)
        else:
            reverse_or_unknown(outcome.produced_by_beat_id, dependency.required_by_beat_id, "dependency.chronology", "dependency")
    for link in skeleton.setup_payoffs:
        if link.setup_beat_id not in beats or link.payoff_beat_id not in beats:
            fail("setup_payoff.endpoint", "setup/payoff endpoint does not resolve", link.link_id)
        elif link.setup_beat_id == link.payoff_beat_id:
            fail("setup_payoff.self", "setup and payoff cannot be the same beat", link.link_id)
        else:
            reverse_or_unknown(link.setup_beat_id, link.payoff_beat_id, "setup_payoff.chronology", "setup/payoff")
        if any(ref not in canonical_refs for ref in link.source_provenance_refs):
            fail("setup_payoff.provenance", "setup/payoff provenance is outside exact source snapshot", link.link_id)

    _validate_knowledge(skeleton, beats, characters, index, precedes, fail)
    _validate_overlay_refs(skeleton, beats, characters, canonical_refs, outcomes, index, fail)
    _validate_accounting(skeleton, beats, projection, fail)
    return B14SkeletonValidationResult(tuple(sorted(findings, key=lambda item: (item.code, item.subject_id or "", item.message))), projection)


def _primary_projection(items: Iterable[str | PresentationEntry], beats: set[str], fail):
    result: list[PrimaryPresentationIdentity] = []
    seen: set[str] = set()
    for position, item in enumerate(items):
        identity, beat_id = (item, item) if isinstance(item, str) else (item.presentation_entry_id, item.adapted_story_beat_id)
        if identity in seen:
            fail("presentation.identity", "primary presentation identity is duplicated", identity)
        seen.add(identity)
        if beat_id not in beats:
            fail("presentation.beat", "presentation references an unknown adapted beat", identity)
        result.append(PrimaryPresentationIdentity(identity, beat_id, position))
    return tuple(result)


def _validate_knowledge(skeleton, beats, characters, index, precedes, fail) -> None:
    initial_audience = {}
    for state in skeleton.audience_knowledge_states:
        if state.knowledge_key in initial_audience:
            fail("audience_knowledge.initial", "audience knowledge has duplicate initial state", state.state_id)
            continue
        initial_audience[state.knowledge_key] = state.status
    for reveal in skeleton.reveal_transitions:
        if reveal.triggered_by_presentation_id not in index:
            fail("reveal.presentation", "reveal presentation does not resolve", reveal.reveal_id)
    for key in sorted({item.knowledge_key for item in skeleton.reveal_transitions}):
        if key not in initial_audience:
            fail("reveal.initial", "audience reveal chain has no initial state", key)
            continue
        transitions = sorted((item for item in skeleton.reveal_transitions if item.knowledge_key == key), key=lambda item: index.get(item.triggered_by_presentation_id, -1))
        status = initial_audience[key]
        last = -1
        for item in transitions:
            now = index.get(item.triggered_by_presentation_id)
            if now is None:
                continue
            if now == last or item.from_status != status:
                fail("reveal.chain", "audience reveal chain is contradictory or unordered", item.reveal_id)
            status = item.to_status
            last = now
    initial_characters = {}
    established_states = {}
    for state in skeleton.character_knowledge_states:
        if state.character_id not in characters:
            fail("character_knowledge.character", "knowledge state character does not resolve", state.state_id)
        if state.established_by_beat_id is not None and state.established_by_beat_id not in beats:
            fail("character_knowledge.beat", "knowledge state beat does not resolve", state.state_id)
        if state.established_by_beat_id is None:
            key = (state.character_id, state.knowledge_key)
            if key in initial_characters:
                fail("character_knowledge.initial", "character knowledge has duplicate initial state", state.state_id)
            initial_characters[key] = state.status
        else:
            key = (state.character_id, state.knowledge_key, state.established_by_beat_id)
            if key in established_states:
                fail("character_knowledge.state", "character knowledge has duplicate state at a beat", state.state_id)
            established_states[key] = state
    for key in sorted({(item.character_id, item.knowledge_key) for item in skeleton.character_knowledge_transitions}):
        transitions = [item for item in skeleton.character_knowledge_transitions if (item.character_id, item.knowledge_key) == key]
        if key not in initial_characters:
            fail("character_knowledge.initial", "character transition has no initial state", ":".join(key))
            continue
        if any(item.character_id not in characters or item.triggered_by_beat_id not in beats for item in transitions):
            for item in transitions:
                if item.character_id not in characters or item.triggered_by_beat_id not in beats:
                    fail("character_knowledge.reference", "character transition reference does not resolve", item.transition_id)
            continue
        ordered: list = []
        for item in transitions:
            if any(other is not item and not precedes(other.triggered_by_beat_id, item.triggered_by_beat_id) and not precedes(item.triggered_by_beat_id, other.triggered_by_beat_id) for other in transitions):
                fail("character_knowledge.order", "character knowledge progression is incomparable in chronology", item.transition_id)
            ordered.append(item)
        ordered.sort(key=lambda item: sum(precedes(other.triggered_by_beat_id, item.triggered_by_beat_id) for other in transitions))
        status = initial_characters[key]
        for item in ordered:
            if item.from_status != status:
                fail("character_knowledge.chain", "character knowledge transition contradicts prior state", item.transition_id)
            status = item.to_status
        states = [
            state for state in skeleton.character_knowledge_states
            if (state.character_id, state.knowledge_key) == key and state.established_by_beat_id is not None
        ]
        transitions_by_beat = {item.triggered_by_beat_id: item for item in transitions}
        for state in states:
            transition = transitions_by_beat.get(state.established_by_beat_id)
            if transition is None:
                fail("character_knowledge.state_unsupported", "post-initial knowledge state has no transition at its establishing beat", state.state_id)
            elif transition.to_status != state.status:
                fail("character_knowledge.state_conflict", "knowledge state contradicts transition result at its establishing beat", state.state_id)
    transition_keys = {(item.character_id, item.knowledge_key) for item in skeleton.character_knowledge_transitions}
    for state in skeleton.character_knowledge_states:
        if state.established_by_beat_id is not None and (state.character_id, state.knowledge_key) not in transition_keys:
            fail("character_knowledge.state_unsupported", "post-initial knowledge state has no progression", state.state_id)
    for boundary in skeleton.withholding_boundaries:
        if boundary.must_not_be_confirmed_before_presentation_id not in index:
            fail("withholding.presentation", "withholding boundary presentation does not resolve", boundary.boundary_id)
            continue
        if boundary.subject_kind is KnowledgeSubjectKind.CHARACTER and boundary.subject_id not in characters:
            fail("withholding.character", "withholding boundary character does not resolve", boundary.boundary_id)
        elif boundary.subject_kind is KnowledgeSubjectKind.CHARACTER:
            fail("withholding.character_unverifiable", "character withholding cannot be verified from a presentation-time boundary", boundary.boundary_id)
        if boundary.subject_kind is KnowledgeSubjectKind.AUDIENCE:
            if boundary.knowledge_key not in initial_audience:
                fail("withholding.knowledge", "audience withholding knowledge key has no initial audience state", boundary.boundary_id)
                continue
            if initial_audience[boundary.knowledge_key] is KnowledgeStatus.CONFIRMED:
                fail("withholding.violation", "audience is initially confirmed before withholding boundary", boundary.boundary_id)
            for reveal in skeleton.reveal_transitions:
                if (reveal.knowledge_key == boundary.knowledge_key and reveal.to_status is KnowledgeStatus.CONFIRMED
                        and index.get(reveal.triggered_by_presentation_id, -1) < index[boundary.must_not_be_confirmed_before_presentation_id]):
                    fail("withholding.violation", "audience confirmation occurs before withholding boundary", boundary.boundary_id)


def _validate_overlay_refs(skeleton, beats, characters, canonical_refs, outcomes, presentation_index, fail) -> None:
    dependencies = {item.dependency_id for item in skeleton.dependencies}
    knowledge_keys = ({item.knowledge_key for item in skeleton.character_knowledge_states}
                      | {item.knowledge_key for item in skeleton.audience_knowledge_states}
                      | {item.knowledge_key for item in skeleton.character_knowledge_transitions}
                      | {item.knowledge_key for item in skeleton.reveal_transitions})
    for state in skeleton.world_truth_states:
        if state.established_by_beat_id is not None and state.established_by_beat_id not in beats:
            fail("world_truth.beat", "world truth establishing beat does not resolve", state.state_id)
    for checkpoint in skeleton.arc_checkpoints:
        if checkpoint.trigger_beat_id not in beats:
            fail("arc.beat", "arc checkpoint trigger beat does not resolve", checkpoint.checkpoint_id)
        if checkpoint.kind.value == "character" and checkpoint.subject_id not in characters:
            fail("arc.subject", "character arc checkpoint subject does not resolve", checkpoint.checkpoint_id)
        if checkpoint.kind.value == "relationship" and not any(ref.kind == "relationship" and ref.canonical_id == checkpoint.subject_id for ref in canonical_refs):
            fail("arc.subject", "relationship arc checkpoint subject does not resolve", checkpoint.checkpoint_id)
    for gap in skeleton.gaps:
        for anchor in gap.anchors:
            valid = ((anchor.kind is GapAnchorKind.BEAT and anchor.reference_id in beats)
                     or (anchor.kind is GapAnchorKind.DEPENDENCY and anchor.reference_id in dependencies)
                     or (anchor.kind is GapAnchorKind.PRESENTATION and anchor.reference_id in presentation_index)
                     or (anchor.kind is GapAnchorKind.KNOWLEDGE and anchor.reference_id in knowledge_keys))
            if not valid:
                fail("gap.anchor", "gap anchor does not resolve in its typed namespace", gap.gap_id)


def _validate_accounting(skeleton, beats, projection, fail) -> None:
    accounting = {item.adapted_story_beat_id: item for item in skeleton.beat_accounting}
    for beat_id in sorted(beats - set(accounting)):
        fail("accounting.missing", "active beat has no accounting", beat_id)
    for beat_id in sorted(set(accounting) - beats):
        fail("accounting.unknown", "accounting references an unknown beat", beat_id)
    presented = {item.adapted_story_beat_id for item in projection}
    consequential = _beats_reaching_presented(skeleton, presented)
    for beat_id, item in accounting.items():
        if item.treatment is BeatPresentationTreatment.DIRECTLY_PRESENTED and beat_id not in presented:
            fail("accounting.direct_evidence", "directly presented beat has no presentation evidence", beat_id)
        if item.treatment is BeatPresentationTreatment.BACKGROUND_ONLY and beat_id in presented:
            fail("accounting.background_presented", "background-only beat has direct presentation evidence", beat_id)
        if item.treatment is BeatPresentationTreatment.INDIRECTLY_REVEALED and beat_id not in consequential:
            fail("accounting.indirect_evidence", "indirectly revealed beat lacks consequential structured evidence", beat_id)
        if item.treatment is BeatPresentationTreatment.BACKGROUND_ONLY and beat_id not in consequential:
            fail("accounting.orphan", "background-only beat has no causal, knowledge, or continuity role", beat_id)


def _beats_reaching_presented(skeleton, presented: set[str]) -> set[str]:
    """Return beats with an explicit causal/setup path to presented material.

    A produced outcome is not evidence by itself.  It becomes consequential only
    when a dependency consumes it; setup/payoff links are similarly explicit.
    """
    outcomes = {item.outcome_id: item for item in skeleton.outcomes}
    predecessors: dict[str, set[str]] = {}
    for dependency in skeleton.dependencies:
        outcome = outcomes.get(dependency.prerequisite_outcome_id)
        if outcome is not None and outcome.produced_by_beat_id != dependency.required_by_beat_id:
            predecessors.setdefault(dependency.required_by_beat_id, set()).add(outcome.produced_by_beat_id)
    for link in skeleton.setup_payoffs:
        if link.setup_beat_id != link.payoff_beat_id:
            predecessors.setdefault(link.payoff_beat_id, set()).add(link.setup_beat_id)
    chronology = {beat_id: set() for beat_id in (
        item.adapted_story_beat_id for item in skeleton.beat_accounting
    )}
    for edge in skeleton.chronology:
        chronology.setdefault(edge.before_beat_id, set()).add(edge.after_beat_id)
    transitions = skeleton.character_knowledge_transitions
    for later in transitions:
        for earlier in transitions:
            if (
                earlier is not later
                and (earlier.character_id, earlier.knowledge_key) == (later.character_id, later.knowledge_key)
                and earlier.to_status is later.from_status
                and _reachable(chronology, earlier.triggered_by_beat_id, later.triggered_by_beat_id)
            ):
                predecessors.setdefault(later.triggered_by_beat_id, set()).add(earlier.triggered_by_beat_id)
    reached, pending = set(presented), list(presented)
    while pending:
        beat_id = pending.pop()
        for predecessor in predecessors.get(beat_id, ()):
            if predecessor not in reached:
                reached.add(predecessor)
                pending.append(predecessor)
    return reached


def _reachable(edges: dict[str, set[str]], start: str, target: str) -> bool:
    if start == target:
        return False
    todo, seen = list(edges.get(start, ())), set()
    while todo:
        node = todo.pop()
        if node == target:
            return True
        if node not in seen:
            seen.add(node)
            todo.extend(edges.get(node, ()))
    return False


def _cyclic_nodes(edges: dict[str, set[str]]) -> set[str]:
    cyclic = set()
    for node in edges:
        if any(_reachable(edges, child, node) for child in edges[node]):
            cyclic.add(node)
    return cyclic
