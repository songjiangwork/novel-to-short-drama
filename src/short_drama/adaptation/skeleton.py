"""B1.4-A1 adapted-story skeleton domain contracts.

The models in this module deliberately describe only the B1.4-owned narrative
overlay.  They pin, rather than copy, the exact B1.2 strategy and B1.3
disposition publication.  Resolving those pins and checking the resulting
cross-object graph belongs to the B1.4-A2 validator.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable

from short_drama.artifacts import ArtifactRef, content_hash

from .disposition import AdaptedStoryBeat, AdaptedStoryBeatOrigin
from .disposition_publication import B13_DISPOSITION_PUBLICATION_ARTIFACT_TYPE
from .strategy_execution import B12_STRATEGY_ARTIFACT_TYPE
from .target import CanonicalTargetReference


B14_SKELETON_SCHEMA_VERSION = 1
_KEY = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")


class B14SkeletonError(ValueError):
    """A B1.4-A1 contract or local deterministic invariant failed."""


class PresentationSemantics(str, Enum):
    FULL = "full"
    TEASER = "teaser"
    PARTIAL_REVEAL = "partial_reveal"
    DELAYED_REVEAL = "delayed_reveal"
    FLASHBACK = "flashback"
    RECAP = "recap"


class ContinuityStateKind(str, Enum):
    PLOT_CRITICAL_SECRET = "plot_critical_secret"
    KEY_OBJECT = "key_object"
    RELATIONSHIP = "relationship"
    IDENTITY_STATUS = "identity_status"
    LIFE_STATUS = "life_status"
    GOAL = "goal"
    COMMITMENT = "commitment"


class KnowledgeStatus(str, Enum):
    UNKNOWN = "unknown"
    SUSPECTED = "suspected"
    CONFIRMED = "confirmed"


class KnowledgeSubjectKind(str, Enum):
    AUDIENCE = "audience"
    CHARACTER = "character"


class ArcCheckpointKind(str, Enum):
    CHARACTER = "character"
    RELATIONSHIP = "relationship"


class ArcDimension(str, Enum):
    GOAL = "goal"
    BELIEF = "belief"
    TRUST = "trust"
    COMMITMENT = "commitment"
    RELATIONSHIP_STANCE = "relationship_stance"
    SELF_UNDERSTANDING = "self_understanding"


class SkeletonGapKind(str, Enum):
    CAUSAL = "causal"
    KNOWLEDGE = "knowledge"
    MOTIVATION = "motivation"
    RELATIONSHIP = "relationship"
    SETUP_PAYOFF = "setup_payoff"
    IDENTITY_ROLE = "identity_role"
    CONTINUITY_STATE = "continuity_state"
    TIMELINE = "timeline"


class GapDisposition(str, Enum):
    ACCIDENTAL = "accidental"
    INTENTIONAL_AMBIGUITY = "intentional_ambiguity"
    INTENTIONAL_WITHHOLDING = "intentional_withholding"


class GapAnchorKind(str, Enum):
    BEAT = "beat"
    DEPENDENCY = "dependency"
    PRESENTATION = "presentation"
    KNOWLEDGE = "knowledge"


class BeatPresentationTreatment(str, Enum):
    DIRECTLY_PRESENTED = "directly_presented"
    INDIRECTLY_REVEALED = "indirectly_revealed"
    BACKGROUND_ONLY = "background_only"


def _exact(value: Any, fields: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise B14SkeletonError(f"{name} must contain exactly: {', '.join(sorted(fields))}")
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\0" in value:
        raise B14SkeletonError(f"{name} must be non-empty UTF-8 text without NUL")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise B14SkeletonError(f"{name} must be UTF-8 text") from exc
    return value


def _key(value: Any, name: str) -> str:
    value = _text(value, name)
    if not _KEY.fullmatch(value):
        raise B14SkeletonError(f"{name} must be a safe identifier")
    return value


def _enum(value: Any, kind: type[Enum], name: str) -> Enum:
    try:
        return value if isinstance(value, kind) else kind(value)
    except (TypeError, ValueError) as exc:
        raise B14SkeletonError(f"{name} is not a permitted value") from exc


def _items(value: Iterable[Any], kind: type[Any], name: str, key: Any) -> tuple[Any, ...]:
    try:
        items = tuple(value)
    except TypeError as exc:
        raise B14SkeletonError(f"{name} must be an iterable of {kind.__name__}") from exc
    if any(not isinstance(item, kind) for item in items) or len({key(item) for item in items}) != len(items):
        raise B14SkeletonError(f"{name} must contain unique {kind.__name__} values")
    return tuple(sorted(items, key=key))


def _subject(kind: Any, subject_id: Any, name: str) -> tuple[KnowledgeSubjectKind, str | None]:
    kind = _enum(kind, KnowledgeSubjectKind, f"{name} kind")
    if kind is KnowledgeSubjectKind.AUDIENCE:
        if subject_id is not None:
            raise B14SkeletonError(f"{name} audience subject_id must be null")
        return kind, None
    return kind, _key(subject_id, f"{name} character subject_id")


def _knowledge_transition(before: Any, after: Any, name: str) -> tuple[KnowledgeStatus, KnowledgeStatus]:
    before = _enum(before, KnowledgeStatus, f"{name} from_status")
    after = _enum(after, KnowledgeStatus, f"{name} to_status")
    allowed = {
        (KnowledgeStatus.UNKNOWN, KnowledgeStatus.SUSPECTED),
        (KnowledgeStatus.SUSPECTED, KnowledgeStatus.CONFIRMED),
        # Definitive evidence may be received without an invented suspicion beat.
        (KnowledgeStatus.UNKNOWN, KnowledgeStatus.CONFIRMED),
    }
    if (before, after) not in allowed:
        raise B14SkeletonError(
            f"{name} must be unknown→suspected, suspected→confirmed, or unknown→confirmed"
        )
    return before, after


@dataclass(frozen=True, slots=True)
class ChronologyPrecedence:
    """One partial-order edge in story-world chronology, not presentation order."""

    before_beat_id: str
    after_beat_id: str

    def __post_init__(self) -> None:
        _key(self.before_beat_id, "chronology before_beat_id")
        _key(self.after_beat_id, "chronology after_beat_id")
        if self.before_beat_id == self.after_beat_id:
            raise B14SkeletonError("chronology precedence cannot self-reference")

    def to_dict(self) -> dict[str, Any]:
        return {"before_beat_id": self.before_beat_id, "after_beat_id": self.after_beat_id}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ChronologyPrecedence":
        return cls(**_exact(value, {"before_beat_id", "after_beat_id"}, cls.__name__))


@dataclass(frozen=True, slots=True)
class PresentationEntry:
    """A non-compact audience delivery of one adapted narrative occurrence."""

    presentation_entry_id: str
    adapted_story_beat_id: str
    semantics: PresentationSemantics
    description_zh: str

    def __post_init__(self) -> None:
        _key(self.presentation_entry_id, "presentation_entry_id")
        _key(self.adapted_story_beat_id, "presentation adapted_story_beat_id")
        object.__setattr__(self, "semantics", _enum(self.semantics, PresentationSemantics, "presentation semantics"))
        _text(self.description_zh, "presentation description")

    def to_dict(self) -> dict[str, Any]:
        return {"presentation_entry_id": self.presentation_entry_id, "adapted_story_beat_id": self.adapted_story_beat_id, "semantics": self.semantics.value, "description_zh": self.description_zh}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "PresentationEntry":
        return cls(**_exact(value, {"presentation_entry_id", "adapted_story_beat_id", "semantics", "description_zh"}, cls.__name__))


@dataclass(frozen=True, slots=True)
class NarrativeOutcome:
    outcome_id: str
    produced_by_beat_id: str
    description_zh: str

    def __post_init__(self) -> None:
        _key(self.outcome_id, "outcome_id")
        _key(self.produced_by_beat_id, "outcome produced_by_beat_id")
        _text(self.description_zh, "outcome description")

    def to_dict(self) -> dict[str, Any]:
        return {"outcome_id": self.outcome_id, "produced_by_beat_id": self.produced_by_beat_id, "description_zh": self.description_zh}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "NarrativeOutcome":
        return cls(**_exact(value, {"outcome_id", "produced_by_beat_id", "description_zh"}, cls.__name__))


@dataclass(frozen=True, slots=True)
class NarrativeDependency:
    dependency_id: str
    prerequisite_outcome_id: str
    required_by_beat_id: str
    description_zh: str

    def __post_init__(self) -> None:
        _key(self.dependency_id, "dependency_id")
        _key(self.prerequisite_outcome_id, "dependency prerequisite_outcome_id")
        _key(self.required_by_beat_id, "dependency required_by_beat_id")
        _text(self.description_zh, "dependency description")

    def to_dict(self) -> dict[str, Any]:
        return {"dependency_id": self.dependency_id, "prerequisite_outcome_id": self.prerequisite_outcome_id, "required_by_beat_id": self.required_by_beat_id, "description_zh": self.description_zh}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "NarrativeDependency":
        return cls(**_exact(value, {"dependency_id", "prerequisite_outcome_id", "required_by_beat_id", "description_zh"}, cls.__name__))


@dataclass(frozen=True, slots=True)
class WorldTruthState:
    """A plot-critical fact true in the adapted world, independent of knowers."""

    state_id: str
    kind: ContinuityStateKind
    subject_key: str
    assertion_zh: str
    established_by_beat_id: str | None
    provenance_zh: str

    def __post_init__(self) -> None:
        _key(self.state_id, "world truth state_id")
        object.__setattr__(self, "kind", _enum(self.kind, ContinuityStateKind, "world truth kind"))
        _key(self.subject_key, "world truth subject_key")
        _text(self.assertion_zh, "world truth assertion")
        if self.established_by_beat_id is not None:
            _key(self.established_by_beat_id, "world truth established_by_beat_id")
        _text(self.provenance_zh, "world truth provenance")

    def to_dict(self) -> dict[str, Any]:
        return {"state_id": self.state_id, "kind": self.kind.value, "subject_key": self.subject_key, "assertion_zh": self.assertion_zh, "established_by_beat_id": self.established_by_beat_id, "provenance_zh": self.provenance_zh}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "WorldTruthState":
        return cls(**_exact(value, {"state_id", "kind", "subject_key", "assertion_zh", "established_by_beat_id", "provenance_zh"}, cls.__name__))


@dataclass(frozen=True, slots=True)
class CharacterKnowledgeState:
    """A character's plot-critical state; ``None`` marks pre-story initial knowledge."""

    state_id: str
    character_id: str
    knowledge_key: str
    status: KnowledgeStatus
    established_by_beat_id: str | None
    provenance_zh: str

    def __post_init__(self) -> None:
        _key(self.state_id, "character knowledge state_id")
        _key(self.character_id, "character knowledge character_id")
        _key(self.knowledge_key, "character knowledge knowledge_key")
        object.__setattr__(self, "status", _enum(self.status, KnowledgeStatus, "character knowledge status"))
        if self.established_by_beat_id is not None:
            _key(self.established_by_beat_id, "character knowledge established_by_beat_id")
        _text(self.provenance_zh, "character knowledge provenance")

    def to_dict(self) -> dict[str, Any]:
        return {"state_id": self.state_id, "character_id": self.character_id, "knowledge_key": self.knowledge_key, "status": self.status.value, "established_by_beat_id": self.established_by_beat_id, "provenance_zh": self.provenance_zh}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CharacterKnowledgeState":
        return cls(**_exact(value, {"state_id", "character_id", "knowledge_key", "status", "established_by_beat_id", "provenance_zh"}, cls.__name__))


@dataclass(frozen=True, slots=True)
class AudienceKnowledgeState:
    state_id: str
    knowledge_key: str
    status: KnowledgeStatus
    provenance_zh: str

    def __post_init__(self) -> None:
        _key(self.state_id, "audience knowledge state_id")
        _key(self.knowledge_key, "audience knowledge knowledge_key")
        object.__setattr__(self, "status", _enum(self.status, KnowledgeStatus, "audience knowledge status"))
        _text(self.provenance_zh, "audience knowledge provenance")

    def to_dict(self) -> dict[str, Any]:
        return {"state_id": self.state_id, "knowledge_key": self.knowledge_key, "status": self.status.value, "provenance_zh": self.provenance_zh}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "AudienceKnowledgeState":
        return cls(**_exact(value, {"state_id", "knowledge_key", "status", "provenance_zh"}, cls.__name__))


@dataclass(frozen=True, slots=True)
class RevealTransition:
    """An audience knowledge transition, anchored to audience presentation."""

    reveal_id: str
    knowledge_key: str
    from_status: KnowledgeStatus
    to_status: KnowledgeStatus
    triggered_by_presentation_id: str

    def __post_init__(self) -> None:
        _key(self.reveal_id, "reveal_id")
        _key(self.knowledge_key, "reveal knowledge_key")
        before, after = _knowledge_transition(self.from_status, self.to_status, "audience reveal transition")
        object.__setattr__(self, "from_status", before)
        object.__setattr__(self, "to_status", after)
        _key(self.triggered_by_presentation_id, "reveal triggered_by_presentation_id")

    def to_dict(self) -> dict[str, Any]:
        return {"reveal_id": self.reveal_id, "knowledge_key": self.knowledge_key, "from_status": self.from_status.value, "to_status": self.to_status.value, "triggered_by_presentation_id": self.triggered_by_presentation_id}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RevealTransition":
        return cls(**_exact(value, {"reveal_id", "knowledge_key", "from_status", "to_status", "triggered_by_presentation_id"}, cls.__name__))


@dataclass(frozen=True, slots=True)
class CharacterKnowledgeTransition:
    """A character knowledge transition at one story-world beat occurrence."""

    transition_id: str
    character_id: str
    knowledge_key: str
    from_status: KnowledgeStatus
    to_status: KnowledgeStatus
    triggered_by_beat_id: str

    def __post_init__(self) -> None:
        _key(self.transition_id, "character knowledge transition_id")
        _key(self.character_id, "character knowledge transition character_id")
        _key(self.knowledge_key, "character knowledge transition knowledge_key")
        before, after = _knowledge_transition(self.from_status, self.to_status, "character knowledge transition")
        object.__setattr__(self, "from_status", before)
        object.__setattr__(self, "to_status", after)
        _key(self.triggered_by_beat_id, "character knowledge transition triggered_by_beat_id")

    def to_dict(self) -> dict[str, Any]:
        return {"transition_id": self.transition_id, "character_id": self.character_id, "knowledge_key": self.knowledge_key, "from_status": self.from_status.value, "to_status": self.to_status.value, "triggered_by_beat_id": self.triggered_by_beat_id}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CharacterKnowledgeTransition":
        return cls(**_exact(value, {"transition_id", "character_id", "knowledge_key", "from_status", "to_status", "triggered_by_beat_id"}, cls.__name__))


@dataclass(frozen=True, slots=True)
class WithholdingBoundary:
    boundary_id: str
    knowledge_key: str
    subject_kind: KnowledgeSubjectKind
    subject_id: str | None
    must_not_be_confirmed_before_presentation_id: str
    rationale_zh: str

    def __post_init__(self) -> None:
        _key(self.boundary_id, "withholding boundary_id")
        _key(self.knowledge_key, "withholding knowledge_key")
        kind, subject_id = _subject(self.subject_kind, self.subject_id, "withholding subject")
        object.__setattr__(self, "subject_kind", kind)
        object.__setattr__(self, "subject_id", subject_id)
        _key(self.must_not_be_confirmed_before_presentation_id, "withholding presentation id")
        _text(self.rationale_zh, "withholding rationale")

    def to_dict(self) -> dict[str, Any]:
        return {"boundary_id": self.boundary_id, "knowledge_key": self.knowledge_key, "subject_kind": self.subject_kind.value, "subject_id": self.subject_id, "must_not_be_confirmed_before_presentation_id": self.must_not_be_confirmed_before_presentation_id, "rationale_zh": self.rationale_zh}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "WithholdingBoundary":
        return cls(**_exact(value, {"boundary_id", "knowledge_key", "subject_kind", "subject_id", "must_not_be_confirmed_before_presentation_id", "rationale_zh"}, cls.__name__))


@dataclass(frozen=True, slots=True)
class SetupPayoffLink:
    link_id: str
    setup_beat_id: str
    payoff_beat_id: str
    source_provenance_refs: tuple[CanonicalTargetReference, ...]
    provenance_zh: str

    def __post_init__(self) -> None:
        _key(self.link_id, "setup/payoff link_id")
        _key(self.setup_beat_id, "setup beat_id")
        _key(self.payoff_beat_id, "payoff beat_id")
        refs = tuple(self.source_provenance_refs)
        if any(not isinstance(ref, CanonicalTargetReference) for ref in refs) or len(set(refs)) != len(refs):
            raise B14SkeletonError("setup/payoff source_provenance_refs must contain unique canonical references")
        object.__setattr__(self, "source_provenance_refs", tuple(sorted(refs, key=lambda ref: (ref.kind, ref.canonical_id))))
        _text(self.provenance_zh, "setup/payoff provenance")

    def to_dict(self) -> dict[str, Any]:
        return {"link_id": self.link_id, "setup_beat_id": self.setup_beat_id, "payoff_beat_id": self.payoff_beat_id, "source_provenance_refs": [item.to_dict() for item in self.source_provenance_refs], "provenance_zh": self.provenance_zh}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SetupPayoffLink":
        value = _exact(value, {"link_id", "setup_beat_id", "payoff_beat_id", "source_provenance_refs", "provenance_zh"}, cls.__name__)
        if not isinstance(value["source_provenance_refs"], list):
            raise B14SkeletonError("setup/payoff source_provenance_refs must be an array")
        return cls(value["link_id"], value["setup_beat_id"], value["payoff_beat_id"], tuple(CanonicalTargetReference.from_dict(item) for item in value["source_provenance_refs"]), value["provenance_zh"])


@dataclass(frozen=True, slots=True)
class ArcCheckpoint:
    checkpoint_id: str
    kind: ArcCheckpointKind
    subject_id: str
    dimension: ArcDimension
    state_zh: str
    trigger_beat_id: str
    provenance_zh: str

    def __post_init__(self) -> None:
        _key(self.checkpoint_id, "arc checkpoint_id")
        object.__setattr__(self, "kind", _enum(self.kind, ArcCheckpointKind, "arc checkpoint kind"))
        _key(self.subject_id, "arc checkpoint subject_id")
        object.__setattr__(self, "dimension", _enum(self.dimension, ArcDimension, "arc checkpoint dimension"))
        _text(self.state_zh, "arc checkpoint state")
        _key(self.trigger_beat_id, "arc checkpoint trigger_beat_id")
        _text(self.provenance_zh, "arc checkpoint provenance")

    def to_dict(self) -> dict[str, Any]:
        return {"checkpoint_id": self.checkpoint_id, "kind": self.kind.value, "subject_id": self.subject_id, "dimension": self.dimension.value, "state_zh": self.state_zh, "trigger_beat_id": self.trigger_beat_id, "provenance_zh": self.provenance_zh}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ArcCheckpoint":
        return cls(**_exact(value, {"checkpoint_id", "kind", "subject_id", "dimension", "state_zh", "trigger_beat_id", "provenance_zh"}, cls.__name__))


@dataclass(frozen=True, slots=True)
class GapAnchor:
    """A narrow locator for a gap; A2 resolves the referenced object."""

    kind: GapAnchorKind
    reference_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _enum(self.kind, GapAnchorKind, "gap anchor kind"))
        _key(self.reference_id, "gap anchor reference_id")

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind.value, "reference_id": self.reference_id}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "GapAnchor":
        return cls(**_exact(value, {"kind", "reference_id"}, cls.__name__))


@dataclass(frozen=True, slots=True)
class SkeletonGap:
    gap_id: str
    kind: SkeletonGapKind
    disposition: GapDisposition
    description_zh: str
    provenance_zh: str
    anchors: tuple[GapAnchor, ...] = ()

    def __post_init__(self) -> None:
        _key(self.gap_id, "skeleton gap_id")
        object.__setattr__(self, "kind", _enum(self.kind, SkeletonGapKind, "skeleton gap kind"))
        object.__setattr__(self, "disposition", _enum(self.disposition, GapDisposition, "skeleton gap disposition"))
        _text(self.description_zh, "skeleton gap description")
        _text(self.provenance_zh, "skeleton gap provenance")
        object.__setattr__(self, "anchors", _items(self.anchors, GapAnchor, "skeleton gap anchors", lambda item: (item.kind.value, item.reference_id)))

    def to_dict(self) -> dict[str, Any]:
        return {"gap_id": self.gap_id, "kind": self.kind.value, "disposition": self.disposition.value, "description_zh": self.description_zh, "provenance_zh": self.provenance_zh, "anchors": [item.to_dict() for item in self.anchors]}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SkeletonGap":
        value = _exact(value, {"gap_id", "kind", "disposition", "description_zh", "provenance_zh", "anchors"}, cls.__name__)
        if not isinstance(value["anchors"], list):
            raise B14SkeletonError("skeleton gap anchors must be an array")
        return cls(value["gap_id"], value["kind"], value["disposition"], value["description_zh"], value["provenance_zh"], tuple(GapAnchor.from_dict(item) for item in value["anchors"]))


@dataclass(frozen=True, slots=True)
class BeatAccounting:
    adapted_story_beat_id: str
    treatment: BeatPresentationTreatment
    rationale_zh: str

    def __post_init__(self) -> None:
        _key(self.adapted_story_beat_id, "beat accounting adapted_story_beat_id")
        object.__setattr__(self, "treatment", _enum(self.treatment, BeatPresentationTreatment, "beat accounting treatment"))
        _text(self.rationale_zh, "beat accounting rationale")

    def to_dict(self) -> dict[str, Any]:
        return {"adapted_story_beat_id": self.adapted_story_beat_id, "treatment": self.treatment.value, "rationale_zh": self.rationale_zh}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "BeatAccounting":
        return cls(**_exact(value, {"adapted_story_beat_id", "treatment", "rationale_zh"}, cls.__name__))


@dataclass(frozen=True, slots=True)
class AdaptedStorySkeleton:
    """Versioned B1.4 overlay pinned to exact immutable B1.2/B1.3 authority."""

    strategy_ref: ArtifactRef
    disposition_publication_ref: ArtifactRef
    chronology: tuple[ChronologyPrecedence, ...] = ()
    primary_presentation: tuple[str | PresentationEntry, ...] = ()
    outcomes: tuple[NarrativeOutcome, ...] = ()
    dependencies: tuple[NarrativeDependency, ...] = ()
    world_truth_states: tuple[WorldTruthState, ...] = ()
    character_knowledge_states: tuple[CharacterKnowledgeState, ...] = ()
    character_knowledge_transitions: tuple[CharacterKnowledgeTransition, ...] = ()
    audience_knowledge_states: tuple[AudienceKnowledgeState, ...] = ()
    reveal_transitions: tuple[RevealTransition, ...] = ()
    withholding_boundaries: tuple[WithholdingBoundary, ...] = ()
    setup_payoffs: tuple[SetupPayoffLink, ...] = ()
    arc_checkpoints: tuple[ArcCheckpoint, ...] = ()
    gaps: tuple[SkeletonGap, ...] = ()
    beat_accounting: tuple[BeatAccounting, ...] = ()
    local_invented_beats: tuple[AdaptedStoryBeat, ...] = ()
    schema_version: int = B14_SKELETON_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or not isinstance(self.schema_version, int) or self.schema_version != B14_SKELETON_SCHEMA_VERSION:
            raise B14SkeletonError("unsupported B1.4 skeleton schema_version")
        if not isinstance(self.strategy_ref, ArtifactRef) or self.strategy_ref.artifact_type != B12_STRATEGY_ARTIFACT_TYPE:
            raise B14SkeletonError("strategy_ref must be an exact B1.2 adaptation_strategy ArtifactRef")
        if not isinstance(self.disposition_publication_ref, ArtifactRef) or self.disposition_publication_ref.artifact_type != B13_DISPOSITION_PUBLICATION_ARTIFACT_TYPE:
            raise B14SkeletonError("disposition_publication_ref must be an exact B1.3 publication ArtifactRef")
        object.__setattr__(self, "chronology", _items(self.chronology, ChronologyPrecedence, "chronology", lambda item: (item.before_beat_id, item.after_beat_id)))
        presentation: list[str | PresentationEntry] = []
        presentation_ids: set[str] = set()
        for item in self.primary_presentation:
            if isinstance(item, str):
                presentation_id = _key(item, "compact presentation beat id")
            elif isinstance(item, PresentationEntry):
                presentation_id = item.presentation_entry_id
            else:
                raise B14SkeletonError("primary_presentation must contain compact beat IDs or PresentationEntry values")
            if presentation_id in presentation_ids:
                raise B14SkeletonError("primary_presentation must not contain duplicate presentation identities")
            presentation_ids.add(presentation_id)
            presentation.append(item)
        object.__setattr__(self, "primary_presentation", tuple(presentation))
        for field_name, kind, key in (
            ("outcomes", NarrativeOutcome, lambda item: item.outcome_id),
            ("dependencies", NarrativeDependency, lambda item: item.dependency_id),
            ("world_truth_states", WorldTruthState, lambda item: item.state_id),
            ("character_knowledge_states", CharacterKnowledgeState, lambda item: item.state_id),
            ("character_knowledge_transitions", CharacterKnowledgeTransition, lambda item: item.transition_id),
            ("audience_knowledge_states", AudienceKnowledgeState, lambda item: item.state_id),
            ("reveal_transitions", RevealTransition, lambda item: item.reveal_id),
            ("withholding_boundaries", WithholdingBoundary, lambda item: item.boundary_id),
            ("setup_payoffs", SetupPayoffLink, lambda item: item.link_id),
            ("arc_checkpoints", ArcCheckpoint, lambda item: item.checkpoint_id),
            ("gaps", SkeletonGap, lambda item: item.gap_id),
            ("beat_accounting", BeatAccounting, lambda item: item.adapted_story_beat_id),
        ):
            object.__setattr__(self, field_name, _items(getattr(self, field_name), kind, field_name, key))
        knowledge_state_ids = (
            [item.state_id for item in self.world_truth_states]
            + [item.state_id for item in self.character_knowledge_states]
            + [item.state_id for item in self.audience_knowledge_states]
        )
        if len(set(knowledge_state_ids)) != len(knowledge_state_ids):
            raise B14SkeletonError("world, character, and audience state_ids must not overlap")
        invented = _items(self.local_invented_beats, AdaptedStoryBeat, "local_invented_beats", lambda item: item.adapted_story_beat_id)
        if any(item.origin is not AdaptedStoryBeatOrigin.ADAPTATION_INVENTED for item in invented):
            raise B14SkeletonError("local_invented_beats must contain only adaptation_invented AdaptedStoryBeat values")
        object.__setattr__(self, "local_invented_beats", invented)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "strategy_ref": self.strategy_ref.to_dict(),
            "disposition_publication_ref": self.disposition_publication_ref.to_dict(),
            "chronology": [item.to_dict() for item in self.chronology],
            "primary_presentation": [item if isinstance(item, str) else item.to_dict() for item in self.primary_presentation],
            "outcomes": [item.to_dict() for item in self.outcomes],
            "dependencies": [item.to_dict() for item in self.dependencies],
            "world_truth_states": [item.to_dict() for item in self.world_truth_states],
            "character_knowledge_states": [item.to_dict() for item in self.character_knowledge_states],
            "character_knowledge_transitions": [item.to_dict() for item in self.character_knowledge_transitions],
            "audience_knowledge_states": [item.to_dict() for item in self.audience_knowledge_states],
            "reveal_transitions": [item.to_dict() for item in self.reveal_transitions],
            "withholding_boundaries": [item.to_dict() for item in self.withholding_boundaries],
            "setup_payoffs": [item.to_dict() for item in self.setup_payoffs],
            "arc_checkpoints": [item.to_dict() for item in self.arc_checkpoints],
            "gaps": [item.to_dict() for item in self.gaps],
            "beat_accounting": [item.to_dict() for item in self.beat_accounting],
            "local_invented_beats": [item.to_dict() for item in self.local_invented_beats],
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "AdaptedStorySkeleton":
        fields = {"schema_version", "strategy_ref", "disposition_publication_ref", "chronology", "primary_presentation", "outcomes", "dependencies", "world_truth_states", "character_knowledge_states", "character_knowledge_transitions", "audience_knowledge_states", "reveal_transitions", "withholding_boundaries", "setup_payoffs", "arc_checkpoints", "gaps", "beat_accounting", "local_invented_beats"}
        value = _exact(value, fields, cls.__name__)
        list_fields = fields - {"schema_version", "strategy_ref", "disposition_publication_ref"}
        if any(not isinstance(value[name], list) for name in list_fields):
            raise B14SkeletonError("B1.4 skeleton collections must be arrays")
        presentation = tuple(item if isinstance(item, str) else PresentationEntry.from_dict(item) for item in value["primary_presentation"])
        return cls(
            ArtifactRef.from_dict(value["strategy_ref"]), ArtifactRef.from_dict(value["disposition_publication_ref"]),
            tuple(ChronologyPrecedence.from_dict(item) for item in value["chronology"]), presentation,
            tuple(NarrativeOutcome.from_dict(item) for item in value["outcomes"]),
            tuple(NarrativeDependency.from_dict(item) for item in value["dependencies"]),
            tuple(WorldTruthState.from_dict(item) for item in value["world_truth_states"]),
            tuple(CharacterKnowledgeState.from_dict(item) for item in value["character_knowledge_states"]),
            tuple(CharacterKnowledgeTransition.from_dict(item) for item in value["character_knowledge_transitions"]),
            tuple(AudienceKnowledgeState.from_dict(item) for item in value["audience_knowledge_states"]),
            tuple(RevealTransition.from_dict(item) for item in value["reveal_transitions"]),
            tuple(WithholdingBoundary.from_dict(item) for item in value["withholding_boundaries"]),
            tuple(SetupPayoffLink.from_dict(item) for item in value["setup_payoffs"]),
            tuple(ArcCheckpoint.from_dict(item) for item in value["arc_checkpoints"]),
            tuple(SkeletonGap.from_dict(item) for item in value["gaps"]),
            tuple(BeatAccounting.from_dict(item) for item in value["beat_accounting"]),
            tuple(AdaptedStoryBeat.from_dict(item) for item in value["local_invented_beats"]), value["schema_version"],
        )

    def content_hash(self) -> str:
        return content_hash(self.to_dict())
