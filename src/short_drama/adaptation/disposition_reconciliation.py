"""B1.3-B3A deterministic cross-window candidate inventory.

This is deliberately a review boundary, not a final graph builder.  Window
proposal local keys remain local: every candidate reference carries its window
identity.  B3B is the first slice permitted to assign final adapted IDs.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable

from short_drama.artifacts import canonical_json_bytes, content_hash, strict_json_loads

from .disposition import CanonicalEventDisposition
from .disposition_execution import B13DispositionExecutionResult, B13WindowAcceptedProposal
from .target import AdaptationSourceSnapshot, CanonicalTargetReference


_KEY = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")


class B13ReconciliationError(ValueError):
    """The exact B1.3-B2B result or a B3 reconciliation input is invalid."""


class B13CandidateKind(str, Enum):
    EVENT_DISPOSITION = "event_disposition"
    ADAPTED_CHARACTER = "adapted_character"
    ANONYMOUS_ROLE = "anonymous_role"
    ADAPTED_STORY_BEAT = "adapted_story_beat"
    ADAPTATION_MAPPING = "adaptation_mapping"
    BEAT_PARTICIPANT = "beat_participant"
    FACTUAL_DIVERGENCE = "factual_divergence"
    RELATIONSHIP_REQUIREMENT = "relationship_requirement"
    STORY_STATE_REQUIREMENT = "story_state_requirement"


class B13ReconciliationFindingKind(str, Enum):
    OWNER_OMIT_CONTEXT_MEMBERSHIP = "owner_omit_context_membership"
    COMPETING_FACTUAL_DIVERGENCES = "competing_factual_divergences"
    COMPETING_RELATIONSHIP_REQUIREMENTS = "competing_relationship_requirements"
    COMPETING_STORY_STATE_REQUIREMENTS = "competing_story_state_requirements"
    INCOMPATIBLE_CHARACTER_EQUIVALENCE = "incompatible_character_equivalence"
    INCOMPATIBLE_BEAT_EQUIVALENCE = "incompatible_beat_equivalence"


class B13ReconciliationDecisionKind(str, Enum):
    KEEP_DISTINCT = "keep_distinct"
    IDENTIFY_EQUIVALENT = "identify_equivalent"
    ACCEPT_CONTRIBUTION = "accept_contribution"
    REJECT_CONTRIBUTION = "reject_contribution"
    MARK_UNRESOLVED = "mark_unresolved"


def _key(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _KEY.fullmatch(value):
        raise B13ReconciliationError(f"{name} must be a safe identifier")
    return value


def _hash(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise B13ReconciliationError(f"{name} must be a SHA-256 hash")
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\0" in value:
        raise B13ReconciliationError(f"{name} must be non-empty text")
    return value


def _refs(values: Iterable[CanonicalTargetReference]) -> tuple[CanonicalTargetReference, ...]:
    refs = tuple(values)
    if any(not isinstance(value, CanonicalTargetReference) for value in refs):
        raise B13ReconciliationError("candidate source references must be canonical references")
    if len(refs) != len(set(refs)):
        raise B13ReconciliationError("candidate source references must not repeat")
    return tuple(sorted(refs, key=lambda value: (value.kind, value.canonical_id)))


@dataclass(frozen=True, slots=True, order=True)
class B13CandidateRef:
    """The only authoritative identity of a B1.3 window-local candidate."""

    window_id: str
    candidate_kind: B13CandidateKind
    local_key: str

    def __post_init__(self) -> None:
        _key(self.window_id, "candidate window_id")
        try:
            kind = self.candidate_kind if isinstance(self.candidate_kind, B13CandidateKind) else B13CandidateKind(self.candidate_kind)
        except (TypeError, ValueError) as exc:
            raise B13ReconciliationError("candidate_kind is invalid") from exc
        object.__setattr__(self, "candidate_kind", kind)
        _key(self.local_key, "candidate local_key")

    def to_dict(self) -> dict[str, str]:
        return {"window_id": self.window_id, "candidate_kind": self.candidate_kind.value, "local_key": self.local_key}


@dataclass(frozen=True, slots=True)
class B13ReconciliationCandidate:
    """One preserved proposal contribution, including its exact local links."""

    ref: B13CandidateRef
    source_refs: tuple[CanonicalTargetReference, ...]
    local_links: tuple[B13CandidateRef, ...]
    _canonical_payload: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.ref, B13CandidateRef):
            raise B13ReconciliationError("candidate requires a scoped candidate reference")
        object.__setattr__(self, "source_refs", _refs(self.source_refs))
        links = tuple(self.local_links)
        if any(not isinstance(item, B13CandidateRef) for item in links) or len(links) != len(set(links)):
            raise B13ReconciliationError("candidate local links must be unique scoped references")
        if any(item.window_id != self.ref.window_id for item in links):
            raise B13ReconciliationError("a window candidate cannot contain a cross-window local link")
        object.__setattr__(self, "local_links", tuple(sorted(links)))
        if not isinstance(self._canonical_payload, bytes):
            raise B13ReconciliationError("candidate payload must be canonical bytes")
        try:
            value = strict_json_loads(self._canonical_payload)
        except Exception as exc:
            raise B13ReconciliationError("candidate payload is invalid") from exc
        if not isinstance(value, dict) or canonical_json_bytes(value) != self._canonical_payload:
            raise B13ReconciliationError("candidate payload must be a canonical object")

    def payload(self) -> dict[str, Any]:
        value = strict_json_loads(self._canonical_payload)
        assert isinstance(value, dict)
        return value

    def content_hash(self) -> str:
        return content_hash(self.payload())

    def to_dict(self) -> dict[str, Any]:
        return {
            "ref": self.ref.to_dict(),
            "source_refs": [item.to_dict() for item in self.source_refs],
            "local_links": [item.to_dict() for item in self.local_links],
            "payload": self.payload(),
        }


@dataclass(frozen=True, slots=True)
class B13InventoryWindow:
    """Pinned accepted-window authority retained by the reconciliation input."""

    window_id: str
    window_ordinal: int
    packet_hash: str
    owned_event_ids: tuple[str, ...]
    context_event_ids: tuple[str, ...]
    request_identity: str
    semantic_request_hash: str
    rendered_prompt_hash: str
    proposal_hash: str
    semantic_profile_identity: dict[str, Any]
    prompt_identity: dict[str, Any]
    output_schema_identity: dict[str, Any]
    invocation_provenance: dict[str, Any]
    rejected_round_provenance: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "window_id": self.window_id, "window_ordinal": self.window_ordinal,
            "packet_hash": self.packet_hash, "owned_event_ids": list(self.owned_event_ids),
            "context_event_ids": list(self.context_event_ids),
            "request_identity": self.request_identity,
            "semantic_request_hash": self.semantic_request_hash,
            "rendered_prompt_hash": self.rendered_prompt_hash,
            "proposal_hash": self.proposal_hash,
            "semantic_profile_identity": self.semantic_profile_identity,
            "prompt_identity": self.prompt_identity,
            "output_schema_identity": self.output_schema_identity,
            "invocation_provenance": self.invocation_provenance,
            "rejected_round_provenance": list(self.rejected_round_provenance),
        }


@dataclass(frozen=True, slots=True)
class B13ReconciliationFinding:
    finding_id: str
    kind: B13ReconciliationFindingKind
    candidate_refs: tuple[B13CandidateRef, ...]
    source_refs: tuple[CanonicalTargetReference, ...]
    detail: str

    def __post_init__(self) -> None:
        _hash(self.finding_id, "finding_id")
        try:
            kind = self.kind if isinstance(self.kind, B13ReconciliationFindingKind) else B13ReconciliationFindingKind(self.kind)
        except (TypeError, ValueError) as exc:
            raise B13ReconciliationError("finding kind is invalid") from exc
        object.__setattr__(self, "kind", kind)
        refs = tuple(self.candidate_refs)
        if len(refs) < 2 or any(not isinstance(item, B13CandidateRef) for item in refs):
            raise B13ReconciliationError("finding needs at least two candidate references")
        object.__setattr__(self, "candidate_refs", tuple(sorted(set(refs))))
        object.__setattr__(self, "source_refs", _refs(self.source_refs))
        _text(self.detail, "finding detail")

    def to_dict(self) -> dict[str, Any]:
        return {"finding_id": self.finding_id, "kind": self.kind.value,
                "candidate_refs": [item.to_dict() for item in self.candidate_refs],
                "source_refs": [item.to_dict() for item in self.source_refs], "detail": self.detail}


@dataclass(frozen=True, slots=True)
class B13ReconciliationInventory:
    """Read-only complete B3A input; it has no final adapted identities."""

    source_snapshot: AdaptationSourceSnapshot
    strategy_ref: dict[str, Any]
    strategy_hash: str
    preparation_identity: str
    windows: tuple[B13InventoryWindow, ...]
    candidates: tuple[B13ReconciliationCandidate, ...]
    findings: tuple[B13ReconciliationFinding, ...]
    inventory_identity: str

    def __post_init__(self) -> None:
        if not isinstance(self.source_snapshot, AdaptationSourceSnapshot):
            raise B13ReconciliationError("inventory requires an exact source snapshot")
        _hash(self.strategy_hash, "strategy_hash")
        _hash(self.preparation_identity, "preparation_identity")
        _hash(self.inventory_identity, "inventory_identity")
        windows = tuple(self.windows)
        candidates = tuple(self.candidates)
        findings = tuple(self.findings)
        if not windows or tuple(item.window_ordinal for item in windows) != tuple(range(1, len(windows) + 1)):
            raise B13ReconciliationError("inventory windows must retain exact deterministic order")
        if len({item.window_id for item in windows}) != len(windows):
            raise B13ReconciliationError("inventory windows must not repeat")
        if any(not isinstance(item, B13ReconciliationCandidate) for item in candidates):
            raise B13ReconciliationError("inventory candidates are invalid")
        if len({item.ref for item in candidates}) != len(candidates):
            raise B13ReconciliationError("inventory candidates must have unique scoped references")
        if any(item.ref.window_id not in {window.window_id for window in windows} for item in candidates):
            raise B13ReconciliationError("inventory candidate references an unknown window")
        if any(not isinstance(item, B13ReconciliationFinding) for item in findings):
            raise B13ReconciliationError("inventory findings are invalid")
        if len({item.finding_id for item in findings}) != len(findings):
            raise B13ReconciliationError("inventory findings must be unique")
        candidate_refs = {item.ref for item in candidates}
        if any(not set(item.local_links) <= candidate_refs for item in candidates):
            raise B13ReconciliationError("inventory candidate has an unknown local link")
        if any(not set(item.candidate_refs) <= candidate_refs for item in findings):
            raise B13ReconciliationError("finding references an unknown candidate")
        expected_identity = content_hash({
            "stage": "B1.3-B3A", "source_snapshot": self.source_snapshot.to_dict(),
            "strategy_ref": self.strategy_ref, "strategy_hash": self.strategy_hash,
            "preparation_identity": self.preparation_identity,
            "windows": [item.to_dict() for item in windows],
        })
        if self.inventory_identity != expected_identity:
            raise B13ReconciliationError("inventory identity does not bind its exact execution result")
        object.__setattr__(self, "windows", tuple(sorted(windows, key=lambda item: item.window_ordinal)))
        object.__setattr__(self, "candidates", tuple(sorted(candidates, key=lambda item: item.ref)))
        object.__setattr__(self, "findings", tuple(sorted(findings, key=lambda item: item.finding_id)))

    def candidate(self, ref: B13CandidateRef) -> B13ReconciliationCandidate:
        for candidate in self.candidates:
            if candidate.ref == ref:
                return candidate
        raise B13ReconciliationError("unknown candidate reference")

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_snapshot": self.source_snapshot.to_dict(), "strategy_ref": self.strategy_ref,
            "strategy_hash": self.strategy_hash, "preparation_identity": self.preparation_identity,
            "windows": [item.to_dict() for item in self.windows],
            "candidates": [item.to_dict() for item in self.candidates],
            "findings": [item.to_dict() for item in self.findings],
            "inventory_identity": self.inventory_identity,
        }


@dataclass(frozen=True, slots=True)
class B13ReconciliationDecision:
    """An explicit B3B input; it never makes Python infer semantic equality."""

    kind: B13ReconciliationDecisionKind
    candidate_refs: tuple[B13CandidateRef, ...] = ()
    finding_ids: tuple[str, ...] = ()
    rationale_zh: str = ""
    provenance_zh: str = ""

    def __post_init__(self) -> None:
        try:
            kind = self.kind if isinstance(self.kind, B13ReconciliationDecisionKind) else B13ReconciliationDecisionKind(self.kind)
        except (TypeError, ValueError) as exc:
            raise B13ReconciliationError("decision kind is invalid") from exc
        object.__setattr__(self, "kind", kind)
        refs = tuple(self.candidate_refs)
        if any(not isinstance(item, B13CandidateRef) for item in refs) or len(refs) != len(set(refs)):
            raise B13ReconciliationError("decision candidate references must be unique")
        object.__setattr__(self, "candidate_refs", tuple(sorted(refs)))
        findings = tuple(self.finding_ids)
        if any(not isinstance(item, str) or len(item) != 64 for item in findings) or len(findings) != len(set(findings)):
            raise B13ReconciliationError("decision finding identities are invalid")
        object.__setattr__(self, "finding_ids", tuple(sorted(findings)))
        _text(self.rationale_zh, "decision rationale_zh")
        _text(self.provenance_zh, "decision provenance_zh")
        if kind in {B13ReconciliationDecisionKind.KEEP_DISTINCT, B13ReconciliationDecisionKind.IDENTIFY_EQUIVALENT}:
            if len(refs) != 2 or findings:
                raise B13ReconciliationError("identity decisions require exactly two candidates and no finding IDs")
        elif kind in {B13ReconciliationDecisionKind.ACCEPT_CONTRIBUTION, B13ReconciliationDecisionKind.REJECT_CONTRIBUTION}:
            if len(refs) != 1 or not findings:
                raise B13ReconciliationError("contribution decisions require one candidate and affected findings")
        elif kind is B13ReconciliationDecisionKind.MARK_UNRESOLVED:
            if refs or not findings:
                raise B13ReconciliationError("mark_unresolved requires affected findings only")

    @property
    def decision_id(self) -> str:
        return content_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind.value, "candidate_refs": [item.to_dict() for item in self.candidate_refs],
                "finding_ids": list(self.finding_ids), "rationale_zh": self.rationale_zh,
                "provenance_zh": self.provenance_zh}


@dataclass(frozen=True, slots=True)
class B13ReconciliationDecisionSet:
    inventory_identity: str
    decisions: tuple[B13ReconciliationDecision, ...]

    def __post_init__(self) -> None:
        _hash(self.inventory_identity, "decision inventory_identity")
        decisions = tuple(self.decisions)
        if any(not isinstance(item, B13ReconciliationDecision) for item in decisions):
            raise B13ReconciliationError("decision set contains an invalid decision")
        if len({item.decision_id for item in decisions}) != len(decisions):
            raise B13ReconciliationError("decision set contains duplicate decisions")
        object.__setattr__(self, "decisions", tuple(sorted(decisions, key=lambda item: item.decision_id)))


@dataclass(frozen=True, slots=True)
class B13ReconciliationReview:
    inventory: B13ReconciliationInventory
    decisions: B13ReconciliationDecisionSet
    findings: tuple[B13ReconciliationFinding, ...]
    unresolved_finding_ids: tuple[str, ...]

    @property
    def is_resolved(self) -> bool:
        return not self.unresolved_finding_ids

    def require_resolved(self) -> None:
        if self.unresolved_finding_ids:
            raise B13ReconciliationError("cross-window reconciliation contains unresolved findings")


def _candidate(window_id: str, kind: B13CandidateKind, local_key: str, payload: dict[str, Any], refs: Iterable[CanonicalTargetReference] = (), links: Iterable[B13CandidateRef] = ()) -> B13ReconciliationCandidate:
    return B13ReconciliationCandidate(B13CandidateRef(window_id, kind, local_key), _refs(refs), tuple(links), canonical_json_bytes(payload))


def _ref(value: dict[str, Any]) -> CanonicalTargetReference:
    return CanonicalTargetReference.from_dict(value)


def _window_from_accepted(accepted: B13WindowAcceptedProposal) -> B13InventoryWindow:
    packet = accepted.proposal.packet
    if packet.window_id != accepted.window_id or packet.window_ordinal != accepted.window_ordinal:
        raise B13ReconciliationError("accepted proposal window identity is stale")
    if accepted.semantic_request_hash != accepted.invocation_provenance.request_hash or accepted.rendered_prompt_hash != accepted.invocation_provenance.rendered_prompt_hash:
        raise B13ReconciliationError("accepted semantic request provenance is mismatched")
    profile = accepted.semantic_profile_identity
    prompt = accepted.prompt_identity
    schema = accepted.output_schema_identity
    if (profile != {"profile_id": accepted.invocation_provenance.semantic_profile_id, "profile_hash": accepted.invocation_provenance.semantic_profile_hash}
            or prompt != {"prompt_id": accepted.invocation_provenance.prompt_id, "prompt_version": accepted.invocation_provenance.prompt_version, "prompt_content_hash": accepted.invocation_provenance.prompt_content_hash}
            or schema != {"schema_id": accepted.invocation_provenance.output_schema_id, "schema_version": accepted.invocation_provenance.output_schema_version, "schema_hash": accepted.invocation_provenance.output_schema_hash}):
        raise B13ReconciliationError("accepted semantic asset provenance is mismatched")
    payload = packet.to_dict()
    window = payload["window"]
    expected_request_identity = content_hash({
        "stage": "B1.3-B2A", "source_snapshot": packet.source_snapshot.to_dict(),
        "strategy_ref": packet.strategy_selection.strategy_ref.to_dict(),
        "strategy_hash": packet.strategy_selection.strategy_hash,
        "window_id": packet.window_id, "window_ordinal": packet.window_ordinal,
        "owned_event_ids": window["owned_event_refs"], "window_packet_hash": packet.content_hash(),
        "prompt": {"prompt_id": prompt["prompt_id"], "prompt_version": prompt["prompt_version"], "prompt_content_hash": prompt["prompt_content_hash"]},
        "output_schema": {"schema_id": schema["schema_id"], "schema_version": schema["schema_version"], "schema_content_hash": schema["schema_hash"]},
        "semantic_profile": {"profile_id": profile["profile_id"], "profile_hash": profile["profile_hash"]},
        "rendered_request_hash": accepted.semantic_request_hash,
    })
    if accepted.request_identity != expected_request_identity:
        raise B13ReconciliationError("accepted semantic request identity is stale or mismatched")
    return B13InventoryWindow(
        accepted.window_id, accepted.window_ordinal, packet.content_hash(),
        tuple(window["owned_event_refs"]), tuple(window["context_event_refs"]),
        accepted.request_identity, accepted.semantic_request_hash, accepted.rendered_prompt_hash,
        accepted.proposal.content_hash(), profile, prompt, schema,
        accepted.invocation_provenance.to_dict(),
        tuple({"semantic_round": item.semantic_round, "provider_attempts": item.provider_attempts,
               "diagnostic": item.diagnostic, "invocation_provenance": item.invocation_provenance.to_dict()}
              for item in accepted.rejected_semantic_rounds),
    )


def _proposal_candidates(accepted: B13WindowAcceptedProposal) -> tuple[B13ReconciliationCandidate, ...]:
    window_id = accepted.window_id
    proposal = accepted.proposal.to_dict()
    candidates: list[B13ReconciliationCandidate] = []
    for item in proposal["event_dispositions"]:
        candidates.append(_candidate(window_id, B13CandidateKind.EVENT_DISPOSITION, item["source_event_id"], item, (_ref({"kind": "event", "canonical_id": item["source_event_id"]}),)))
    for item in proposal["adapted_characters"]:
        candidates.append(_candidate(window_id, B13CandidateKind.ADAPTED_CHARACTER, item["local_key"], item, (_ref(part["source_character_ref"]) for part in item["function_provenance"])))
    for item in proposal["anonymous_roles"]:
        candidates.append(_candidate(window_id, B13CandidateKind.ANONYMOUS_ROLE, item["local_key"], item))
    for beat in proposal["adapted_story_beats"]:
        beat_ref = B13CandidateRef(window_id, B13CandidateKind.ADAPTED_STORY_BEAT, beat["local_key"])
        sources = [_ref(item) for item in (*beat["source_event_refs"], *beat["canonical_anchor_refs"])]
        links: list[B13CandidateRef] = []
        for participant in beat["participants"]:
            participant_kind = B13CandidateKind.ADAPTED_CHARACTER if participant["kind"] == "adapted_character" else B13CandidateKind.ANONYMOUS_ROLE
            target = B13CandidateRef(window_id, participant_kind, participant["local_key"])
            links.append(target)
            participant_payload = {"adapted_story_beat_local_key": beat["local_key"], **participant}
            candidates.append(_candidate(window_id, B13CandidateKind.BEAT_PARTICIPANT,
                                         f"participant-{content_hash(participant_payload)[:24]}", participant_payload,
                                         sources, (beat_ref, target)))
        candidates.append(_candidate(window_id, B13CandidateKind.ADAPTED_STORY_BEAT, beat["local_key"], beat, sources, links))
    for item in proposal["adaptation_mappings"]:
        endpoint = item["adapted_endpoint"]
        target_kind = B13CandidateKind.ADAPTED_STORY_BEAT if endpoint["kind"] == "story_beat" else B13CandidateKind.ADAPTED_CHARACTER
        candidates.append(_candidate(window_id, B13CandidateKind.ADAPTATION_MAPPING,
                                     f"mapping-{content_hash(item)[:24]}", item, (_ref(item["source_ref"]),),
                                     (B13CandidateRef(window_id, target_kind, endpoint["local_key"]),)))
    overlay = proposal["continuity_overlay_candidates"]
    for item in overlay["factual_divergences"]:
        candidates.append(_candidate(window_id, B13CandidateKind.FACTUAL_DIVERGENCE,
                                     f"fact-{item['source_fact_ref']['canonical_id']}", item, (_ref(item["source_fact_ref"]),)))
    for item in overlay["relationship_requirements"]:
        candidates.append(_candidate(window_id, B13CandidateKind.RELATIONSHIP_REQUIREMENT,
                                     f"relationship-{item['source_relationship_ref']['canonical_id']}", item, (_ref(item["source_relationship_ref"]),)))
    for item in overlay["story_state_requirements"]:
        beat_ref = B13CandidateRef(window_id, B13CandidateKind.ADAPTED_STORY_BEAT, item["adapted_story_beat_local_key"])
        candidates.append(_candidate(window_id, B13CandidateKind.STORY_STATE_REQUIREMENT,
                                     f"state-{item['adapted_story_beat_local_key']}", item, (), (beat_ref,)))
    return tuple(candidates)


def _finding(inventory_identity: str, kind: B13ReconciliationFindingKind, candidates: Iterable[B13CandidateRef], sources: Iterable[CanonicalTargetReference], detail: str) -> B13ReconciliationFinding:
    refs = tuple(sorted(set(candidates)))
    source_refs = _refs(sources)
    finding_id = content_hash({"inventory_identity": inventory_identity, "kind": kind.value,
                               "candidate_refs": [item.to_dict() for item in refs],
                               "source_refs": [item.to_dict() for item in source_refs]})
    return B13ReconciliationFinding(finding_id, kind, refs, source_refs, detail)


def _static_findings(inventory_identity: str, candidates: tuple[B13ReconciliationCandidate, ...]) -> tuple[B13ReconciliationFinding, ...]:
    findings: list[B13ReconciliationFinding] = []
    dispositions = [item for item in candidates if item.ref.candidate_kind is B13CandidateKind.EVENT_DISPOSITION]
    beats = [item for item in candidates if item.ref.candidate_kind is B13CandidateKind.ADAPTED_STORY_BEAT]
    for disposition in dispositions:
        if disposition.payload()["disposition"] != CanonicalEventDisposition.OMIT.value:
            continue
        event_ref = disposition.source_refs[0]
        conflicting = [beat for beat in beats if beat.ref.window_id != disposition.ref.window_id and event_ref in beat.source_refs]
        for beat in conflicting:
            findings.append(_finding(inventory_identity, B13ReconciliationFindingKind.OWNER_OMIT_CONTEXT_MEMBERSHIP,
                                     (disposition.ref, beat.ref), (event_ref,),
                                     "owner-only omit contradicts another window's source-backed beat candidate"))
    for kind, finding_kind, detail in (
        (B13CandidateKind.FACTUAL_DIVERGENCE, B13ReconciliationFindingKind.COMPETING_FACTUAL_DIVERGENCES,
         "multiple factual-divergence candidates cite one source fact; Python does not compare their assertions"),
        (B13CandidateKind.RELATIONSHIP_REQUIREMENT, B13ReconciliationFindingKind.COMPETING_RELATIONSHIP_REQUIREMENTS,
         "multiple relationship-requirement candidates cite one source relationship; Python does not compare their requirements"),
    ):
        grouped: dict[CanonicalTargetReference, list[B13ReconciliationCandidate]] = {}
        for candidate in candidates:
            if candidate.ref.candidate_kind is kind:
                grouped.setdefault(candidate.source_refs[0], []).append(candidate)
        for source_ref, group in grouped.items():
            if len(group) > 1:
                findings.append(_finding(inventory_identity, finding_kind, (item.ref for item in group), (source_ref,), detail))
    return tuple(sorted(findings, key=lambda item: item.finding_id))


def build_b13_reconciliation_inventory(result: B13DispositionExecutionResult) -> B13ReconciliationInventory:
    """Inventory every accepted local proposal without provider calls or graph assembly."""
    if not isinstance(result, B13DispositionExecutionResult):
        raise B13ReconciliationError("inventory requires B13DispositionExecutionResult")
    accepted = tuple(result.accepted_windows)
    if not accepted or tuple(item.window_ordinal for item in accepted) != tuple(range(1, len(accepted) + 1)):
        raise B13ReconciliationError("execution result is missing or reorders accepted windows")
    if len({item.window_id for item in accepted}) != len(accepted):
        raise B13ReconciliationError("execution result duplicates an accepted window")
    windows = tuple(_window_from_accepted(item) for item in accepted)
    packets = tuple(item.proposal.packet for item in accepted)
    if any(packet.source_snapshot != result.source_snapshot or packet.strategy_selection != result.strategy_selection for packet in packets):
        raise B13ReconciliationError("execution result has a source or B1.2 strategy mismatch")
    expected_preparation_identity = content_hash({
        "stage": "B1.3-B1", "source_snapshot": result.source_snapshot.to_dict(),
        "strategy_ref": result.strategy_selection.strategy_ref.to_dict(),
        "strategy_hash": result.strategy_selection.strategy_hash,
        "packet_hashes": [packet.content_hash() for packet in packets],
    })
    if result.preparation_identity != expected_preparation_identity:
        raise B13ReconciliationError("execution result preparation identity does not bind accepted window order")
    owned = tuple(event_id for window in windows for event_id in window.owned_event_ids)
    expected_events = tuple(sorted((ref.canonical_id for ref in result.source_snapshot.canonical_references if ref.kind == "event")))
    if len(owned) != len(set(owned)) or set(owned) != set(expected_events):
        raise B13ReconciliationError("accepted windows do not provide complete exact-one event ownership")
    candidates = tuple(candidate for item in accepted for candidate in _proposal_candidates(item))
    if len({item.ref for item in candidates}) != len(candidates):
        raise B13ReconciliationError("accepted proposals duplicate a scoped candidate reference")
    if any(not set(candidate.source_refs) <= set(result.source_snapshot.canonical_references) for candidate in candidates):
        raise B13ReconciliationError("candidate references source outside exact snapshot")
    identity_material = {
        "stage": "B1.3-B3A", "source_snapshot": result.source_snapshot.to_dict(),
        "strategy_ref": result.strategy_selection.strategy_ref.to_dict(),
        "strategy_hash": result.strategy_selection.strategy_hash,
        "preparation_identity": result.preparation_identity,
        "windows": [item.to_dict() for item in windows],
    }
    inventory_identity = content_hash(identity_material)
    findings = _static_findings(inventory_identity, candidates)
    return B13ReconciliationInventory(result.source_snapshot, result.strategy_selection.strategy_ref.to_dict(),
                                      result.strategy_selection.strategy_hash, result.preparation_identity,
                                      windows, candidates, findings, inventory_identity)


def _identity_pair(decision: B13ReconciliationDecision) -> tuple[B13CandidateRef, B13CandidateRef]:
    return decision.candidate_refs[0], decision.candidate_refs[1]


def _equivalence_finding(inventory: B13ReconciliationInventory, decision: B13ReconciliationDecision) -> B13ReconciliationFinding | None:
    left, right = (inventory.candidate(ref) for ref in _identity_pair(decision))
    if left.ref.candidate_kind is B13CandidateKind.ADAPTED_CHARACTER:
        compatible = left.payload()["origin"] == right.payload()["origin"] and left.payload()["function_provenance"] == right.payload()["function_provenance"]
        kind = B13ReconciliationFindingKind.INCOMPATIBLE_CHARACTER_EQUIVALENCE
        detail = "explicit character equivalence has incompatible origin or function provenance"
    elif left.ref.candidate_kind is B13CandidateKind.ADAPTED_STORY_BEAT:
        compatible = (left.payload()["origin"] == right.payload()["origin"]
                      and left.payload()["source_event_refs"] == right.payload()["source_event_refs"]
                      and left.payload()["canonical_anchor_refs"] == right.payload()["canonical_anchor_refs"])
        kind = B13ReconciliationFindingKind.INCOMPATIBLE_BEAT_EQUIVALENCE
        detail = "explicit beat equivalence has incompatible origin or source membership"
    else:
        return None
    if compatible:
        return None
    return _finding(inventory.inventory_identity, kind, (left.ref, right.ref),
                    (*left.source_refs, *right.source_refs), detail)


def _story_state_equivalence_findings(inventory: B13ReconciliationInventory, decision: B13ReconciliationDecision) -> tuple[B13ReconciliationFinding, ...]:
    """State candidates only compete after an explicit beat-identity decision."""
    left, right = _identity_pair(decision)
    if left.candidate_kind is not B13CandidateKind.ADAPTED_STORY_BEAT:
        return ()
    state_candidates = [item for item in inventory.candidates if item.ref.candidate_kind is B13CandidateKind.STORY_STATE_REQUIREMENT]
    left_states = [item for item in state_candidates if left in item.local_links]
    right_states = [item for item in state_candidates if right in item.local_links]
    return tuple(
        _finding(inventory.inventory_identity, B13ReconciliationFindingKind.COMPETING_STORY_STATE_REQUIREMENTS,
                 (first.ref, second.ref), (),
                 "state candidates target explicitly equivalent beats; Python does not compare their assertions")
        for first in left_states for second in right_states
    )


def review_b13_reconciliation(inventory: B13ReconciliationInventory, decisions: B13ReconciliationDecisionSet) -> B13ReconciliationReview:
    """Validate explicit decisions and expose every unresolved conflict to B3B."""
    if not isinstance(inventory, B13ReconciliationInventory) or not isinstance(decisions, B13ReconciliationDecisionSet):
        raise B13ReconciliationError("review requires an inventory and a decision set")
    if decisions.inventory_identity != inventory.inventory_identity:
        raise B13ReconciliationError("decision set references another or stale reconciliation inventory")
    candidates = {item.ref: item for item in inventory.candidates}
    static = {item.finding_id: item for item in inventory.findings}
    identity_decisions: dict[tuple[B13CandidateRef, B13CandidateRef], B13ReconciliationDecisionKind] = {}
    contribution_decisions: dict[B13CandidateRef, B13ReconciliationDecisionKind] = {}
    dynamic: list[B13ReconciliationFinding] = []
    for decision in decisions.decisions:
        if any(ref not in candidates for ref in decision.candidate_refs):
            raise B13ReconciliationError("decision references an unknown candidate")
        if any(finding_id not in static for finding_id in decision.finding_ids):
            raise B13ReconciliationError("decision references an unknown finding")
        if decision.kind in {B13ReconciliationDecisionKind.KEEP_DISTINCT, B13ReconciliationDecisionKind.IDENTIFY_EQUIVALENT}:
            left, right = _identity_pair(decision)
            if candidates[left].ref.candidate_kind != candidates[right].ref.candidate_kind or candidates[left].ref.candidate_kind not in {B13CandidateKind.ADAPTED_CHARACTER, B13CandidateKind.ADAPTED_STORY_BEAT, B13CandidateKind.ANONYMOUS_ROLE}:
                raise B13ReconciliationError("identity decisions require two candidates of one identity-bearing kind")
            pair = tuple(sorted((left, right)))
            prior = identity_decisions.get(pair)
            if prior is not None:
                raise B13ReconciliationError("duplicate or contradictory identity decision")
            identity_decisions[pair] = decision.kind
            if decision.kind is B13ReconciliationDecisionKind.IDENTIFY_EQUIVALENT:
                finding = _equivalence_finding(inventory, decision)
                if finding is not None:
                    dynamic.append(finding)
                dynamic.extend(_story_state_equivalence_findings(inventory, decision))
        elif decision.kind in {B13ReconciliationDecisionKind.ACCEPT_CONTRIBUTION, B13ReconciliationDecisionKind.REJECT_CONTRIBUTION}:
            ref = decision.candidate_refs[0]
            if ref.candidate_kind is B13CandidateKind.EVENT_DISPOSITION and decision.kind is B13ReconciliationDecisionKind.REJECT_CONTRIBUTION:
                raise B13ReconciliationError("owner-only event disposition cannot be rejected")
            prior = contribution_decisions.get(ref)
            if prior is not None:
                raise B13ReconciliationError("duplicate or contradictory contribution decision")
            for finding_id in decision.finding_ids:
                if ref not in static[finding_id].candidate_refs:
                    raise B13ReconciliationError("contribution decision does not affect its claimed finding")
            contribution_decisions[ref] = decision.kind
    all_findings = tuple(sorted((*inventory.findings, *dynamic), key=lambda item: item.finding_id))
    unresolved: list[str] = []
    marked = {finding_id for decision in decisions.decisions if decision.kind is B13ReconciliationDecisionKind.MARK_UNRESOLVED for finding_id in decision.finding_ids}
    for finding in all_findings:
        # Dynamic equivalence findings cannot be claimed by a stale static finding ID.
        if finding.finding_id in marked:
            unresolved.append(finding.finding_id)
            continue
        if not any(contribution_decisions.get(ref) is B13ReconciliationDecisionKind.REJECT_CONTRIBUTION for ref in finding.candidate_refs):
            unresolved.append(finding.finding_id)
    return B13ReconciliationReview(inventory, decisions, all_findings, tuple(sorted(unresolved)))


__all__ = [
    "B13ReconciliationError", "B13CandidateKind", "B13CandidateRef", "B13ReconciliationCandidate",
    "B13InventoryWindow", "B13ReconciliationFindingKind", "B13ReconciliationFinding",
    "B13ReconciliationInventory", "B13ReconciliationDecisionKind", "B13ReconciliationDecision",
    "B13ReconciliationDecisionSet", "B13ReconciliationReview",
    "build_b13_reconciliation_inventory", "review_b13_reconciliation",
]
