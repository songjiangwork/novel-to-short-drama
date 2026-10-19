"""Frozen B1.3-A adapted-identity, disposition, and mapping contracts.

This module is deliberately provider-free.  It records Qwen-owned semantic
decisions in strict, deterministic data structures; Python owns identities,
ordering, source-universe checks, and graph invariants.  It does not create a
chronology, presentation entries, episodes, or a second Story Bible.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable

from short_drama.artifacts import content_hash

from .target import AdaptationSourceSnapshot, CanonicalTargetReference


B13_DISPOSITION_SCHEMA_VERSION = 1
_KEY = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")


class B13DispositionError(ValueError):
    """A B1.3-A contract, identity, or deterministic graph invariant failed."""


class CanonicalEventDisposition(str, Enum):
    KEEP = "keep"
    COMPRESS = "compress"
    MERGE_MEMBER = "merge_member"
    OMIT = "omit"


class AdaptedCharacterOrigin(str, Enum):
    SOURCE_SINGLE = "source_single"
    SOURCE_COMPOSITE = "source_composite"
    ADAPTATION_INVENTED = "adaptation_invented"


class AdaptedStoryBeatOrigin(str, Enum):
    SOURCE_DIRECT = "source_direct"
    SOURCE_CONDENSED = "source_condensed"
    SOURCE_COMPOSITE = "source_composite"
    ADAPTATION_INVENTED = "adaptation_invented"


class CanonCompatibility(str, Enum):
    COMPATIBLE = "compatible"
    DELIBERATE_DIVERGENCE = "deliberate_divergence"
    UNCERTAIN = "uncertain"


class InventedBeatPurpose(str, Enum):
    BRIDGE = "bridge"
    EXTERNALIZATION = "externalization"
    CONTINUITY_REPAIR = "continuity_repair"
    SETUP = "setup"
    PAYOFF = "payoff"
    EPISODE_HOOK = "episode_hook"
    DRAMATIC_ESCALATION = "dramatic_escalation"
    DELIBERATE_DIVERGENCE = "deliberate_divergence"
    OTHER = "other"


class MappingContributionRole(str, Enum):
    PRIMARY = "primary"
    SUPPORTING = "supporting"
    CONTEXT = "context"


class AdaptedEndpointKind(str, Enum):
    STORY_BEAT = "story_beat"
    CHARACTER = "character"


class ParticipantKind(str, Enum):
    ADAPTED_CHARACTER = "adapted_character"
    ANONYMOUS_ROLE = "anonymous_role"


class AnonymousRoleKind(str, Enum):
    ATTENDANT = "attendant"
    CROWD_MEMBER = "crowd_member"
    GUARD = "guard"
    MESSENGER = "messenger"
    OFFICIAL = "official"
    PASSERBY = "passerby"
    OTHER_FUNCTIONAL = "other_functional"


def _exact(value: Any, fields: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise B13DispositionError(f"{name} must contain exactly: {', '.join(sorted(fields))}")
    return value


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\0" in value:
        raise B13DispositionError(f"{field} must be non-empty UTF-8 text without NUL")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise B13DispositionError(f"{field} must be UTF-8 text") from exc
    return value


def _key(value: Any, field: str) -> str:
    value = _text(value, field)
    if not _KEY.fullmatch(value):
        raise B13DispositionError(f"{field} must be a safe identifier")
    return value


def _enum(value: Any, kind: type[Enum], field: str) -> Enum:
    try:
        return value if isinstance(value, kind) else kind(value)
    except (TypeError, ValueError) as exc:
        raise B13DispositionError(f"{field} is not a permitted value") from exc


def _refs(value: Iterable[CanonicalTargetReference], field: str, *, kind: str | None = None, minimum: int = 0) -> tuple[CanonicalTargetReference, ...]:
    refs = tuple(value)
    if len(refs) < minimum or any(not isinstance(ref, CanonicalTargetReference) for ref in refs):
        raise B13DispositionError(f"{field} must contain canonical source references")
    if kind is not None and any(ref.kind != kind for ref in refs):
        raise B13DispositionError(f"{field} must contain only {kind} references")
    if len(set(refs)) != len(refs):
        raise B13DispositionError(f"{field} must not contain duplicate references")
    return tuple(sorted(refs, key=lambda ref: (ref.kind, ref.canonical_id)))


def _derived_id(prefix: str, material: dict[str, Any]) -> str:
    return f"{prefix}_{content_hash(material)[:24]}"


@dataclass(frozen=True, slots=True)
class EventDisposition:
    """The one B1.3 disposition assigned to one canonical event."""

    source_event_ref: CanonicalTargetReference
    disposition: CanonicalEventDisposition
    rationale_zh: str

    def __post_init__(self) -> None:
        if not isinstance(self.source_event_ref, CanonicalTargetReference) or self.source_event_ref.kind != "event":
            raise B13DispositionError("source_event_ref must be a canonical event reference")
        object.__setattr__(self, "disposition", _enum(self.disposition, CanonicalEventDisposition, "event disposition"))
        _text(self.rationale_zh, "event disposition rationale")

    def to_dict(self) -> dict[str, Any]:
        return {"source_event_ref": self.source_event_ref.to_dict(), "disposition": self.disposition.value, "rationale_zh": self.rationale_zh}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "EventDisposition":
        value = _exact(value, {"source_event_ref", "disposition", "rationale_zh"}, cls.__name__)
        return cls(CanonicalTargetReference.from_dict(value["source_event_ref"]), value["disposition"], value["rationale_zh"])


@dataclass(frozen=True, slots=True)
class CharacterFunctionProvenance:
    """A source-character narrative function transferred to an adapted role."""

    source_character_ref: CanonicalTargetReference
    function_zh: str
    provenance_zh: str

    def __post_init__(self) -> None:
        if not isinstance(self.source_character_ref, CanonicalTargetReference) or self.source_character_ref.kind != "character":
            raise B13DispositionError("source_character_ref must be a canonical character reference")
        _text(self.function_zh, "character function")
        _text(self.provenance_zh, "character function provenance")

    def to_dict(self) -> dict[str, Any]:
        return {"source_character_ref": self.source_character_ref.to_dict(), "function_zh": self.function_zh, "provenance_zh": self.provenance_zh}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CharacterFunctionProvenance":
        value = _exact(value, {"source_character_ref", "function_zh", "provenance_zh"}, cls.__name__)
        return cls(CanonicalTargetReference.from_dict(value["source_character_ref"]), value["function_zh"], value["provenance_zh"])


@dataclass(frozen=True, slots=True)
class AdaptedCharacter:
    """An adaptation-owned character identity, never an A4 entity mutation."""

    identity_key: str
    display_name_zh: str
    origin: AdaptedCharacterOrigin
    function_provenance: tuple[CharacterFunctionProvenance, ...] = ()
    invention_rationale_zh: str | None = None
    adapted_character_id: str = field(init=False)

    def __post_init__(self) -> None:
        _key(self.identity_key, "character identity_key")
        _text(self.display_name_zh, "adapted character display_name_zh")
        object.__setattr__(self, "origin", _enum(self.origin, AdaptedCharacterOrigin, "adapted character origin"))
        provenance = tuple(self.function_provenance)
        if any(not isinstance(item, CharacterFunctionProvenance) for item in provenance):
            raise B13DispositionError("function_provenance must contain CharacterFunctionProvenance values")
        if len({item.source_character_ref for item in provenance}) != len(provenance):
            raise B13DispositionError("function_provenance must have one entry per source character")
        provenance = tuple(sorted(provenance, key=lambda item: item.source_character_ref.canonical_id))
        object.__setattr__(self, "function_provenance", provenance)
        if self.origin is AdaptedCharacterOrigin.SOURCE_SINGLE and len(provenance) != 1:
            raise B13DispositionError("source_single character requires exactly one source function provenance")
        if self.origin is AdaptedCharacterOrigin.SOURCE_COMPOSITE and len(provenance) < 2:
            raise B13DispositionError("source_composite character requires at least two source function provenances")
        if self.origin is AdaptedCharacterOrigin.ADAPTATION_INVENTED:
            if provenance:
                raise B13DispositionError("adaptation_invented character must not claim source character provenance")
            _text(self.invention_rationale_zh, "invented character rationale")
        elif self.invention_rationale_zh is not None:
            raise B13DispositionError("invention_rationale_zh is only permitted for adaptation_invented characters")
        material = {"identity_key": self.identity_key, "origin": self.origin.value, "source_character_refs": [item.source_character_ref.to_dict() for item in provenance]}
        object.__setattr__(self, "adapted_character_id", _derived_id("adapted_char", material))

    def to_dict(self) -> dict[str, Any]:
        return {"adapted_character_id": self.adapted_character_id, "identity_key": self.identity_key, "display_name_zh": self.display_name_zh, "origin": self.origin.value, "function_provenance": [item.to_dict() for item in self.function_provenance], "invention_rationale_zh": self.invention_rationale_zh}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "AdaptedCharacter":
        value = _exact(value, {"adapted_character_id", "identity_key", "display_name_zh", "origin", "function_provenance", "invention_rationale_zh"}, cls.__name__)
        if not isinstance(value["function_provenance"], list):
            raise B13DispositionError("function_provenance must be an array")
        result = cls(value["identity_key"], value["display_name_zh"], value["origin"], tuple(CharacterFunctionProvenance.from_dict(item) for item in value["function_provenance"]), value["invention_rationale_zh"])
        if value["adapted_character_id"] != result.adapted_character_id:
            raise B13DispositionError("adapted_character_id does not match deterministic identity")
        return result


@dataclass(frozen=True, slots=True)
class AnonymousRole:
    """A controlled, single-beat functional participant without continuity."""

    role_key: str
    role_kind: AnonymousRoleKind
    function_zh: str
    anonymous_role_id: str = field(init=False)

    def __post_init__(self) -> None:
        _key(self.role_key, "anonymous role_key")
        object.__setattr__(self, "role_kind", _enum(self.role_kind, AnonymousRoleKind, "anonymous role kind"))
        _text(self.function_zh, "anonymous role function")
        object.__setattr__(self, "anonymous_role_id", _derived_id("anonymous_role", {"role_key": self.role_key, "role_kind": self.role_kind.value, "function_zh": self.function_zh}))

    def to_dict(self) -> dict[str, Any]:
        return {"anonymous_role_id": self.anonymous_role_id, "role_key": self.role_key, "role_kind": self.role_kind.value, "function_zh": self.function_zh}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "AnonymousRole":
        value = _exact(value, {"anonymous_role_id", "role_key", "role_kind", "function_zh"}, cls.__name__)
        result = cls(value["role_key"], value["role_kind"], value["function_zh"])
        if value["anonymous_role_id"] != result.anonymous_role_id:
            raise B13DispositionError("anonymous_role_id does not match deterministic identity")
        return result


@dataclass(frozen=True, slots=True)
class BeatParticipant:
    kind: ParticipantKind
    participant_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _enum(self.kind, ParticipantKind, "participant kind"))
        _key(self.participant_id, "participant_id")

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind.value, "participant_id": self.participant_id}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "BeatParticipant":
        return cls(**_exact(value, {"kind", "participant_id"}, cls.__name__))


@dataclass(frozen=True, slots=True)
class AdaptedStoryBeat:
    """An adaptation-owned beat; origin and compatibility remain orthogonal."""

    identity_key: str
    summary_zh: str
    origin: AdaptedStoryBeatOrigin
    canon_compatibility: CanonCompatibility
    source_event_refs: tuple[CanonicalTargetReference, ...] = ()
    participants: tuple[BeatParticipant, ...] = ()
    canonical_anchor_refs: tuple[CanonicalTargetReference, ...] = ()
    invention_purpose: InventedBeatPurpose | None = None
    invention_rationale_zh: str | None = None
    invention_provenance_zh: str | None = None
    adapted_story_beat_id: str = field(init=False)

    def __post_init__(self) -> None:
        _key(self.identity_key, "story beat identity_key")
        _text(self.summary_zh, "adapted story beat summary")
        object.__setattr__(self, "origin", _enum(self.origin, AdaptedStoryBeatOrigin, "story beat origin"))
        object.__setattr__(self, "canon_compatibility", _enum(self.canon_compatibility, CanonCompatibility, "canon compatibility"))
        source_refs = _refs(self.source_event_refs, "source_event_refs", kind="event")
        anchors = _refs(self.canonical_anchor_refs, "canonical_anchor_refs")
        participants = tuple(self.participants)
        if any(not isinstance(item, BeatParticipant) for item in participants) or len(set(participants)) != len(participants):
            raise B13DispositionError("participants must be unique BeatParticipant values")
        object.__setattr__(self, "source_event_refs", source_refs)
        object.__setattr__(self, "canonical_anchor_refs", anchors)
        object.__setattr__(self, "participants", tuple(sorted(participants, key=lambda item: (item.kind.value, item.participant_id))))
        if self.origin in {AdaptedStoryBeatOrigin.SOURCE_DIRECT, AdaptedStoryBeatOrigin.SOURCE_CONDENSED} and len(source_refs) != 1:
            raise B13DispositionError("source_direct/source_condensed beat requires exactly one source event")
        if self.origin is AdaptedStoryBeatOrigin.SOURCE_COMPOSITE and len(source_refs) < 2:
            raise B13DispositionError("source_composite beat requires at least two source events")
        if self.origin is AdaptedStoryBeatOrigin.ADAPTATION_INVENTED:
            if source_refs:
                raise B13DispositionError("adaptation_invented beat must not claim source event membership")
            purpose = _enum(self.invention_purpose, InventedBeatPurpose, "invented beat purpose")
            object.__setattr__(self, "invention_purpose", purpose)
            _text(self.invention_rationale_zh, "invented beat rationale")
            _text(self.invention_provenance_zh, "invented beat provenance")
            if not anchors:
                raise B13DispositionError("adaptation_invented beat requires canonical anchors")
            if purpose is InventedBeatPurpose.OTHER:
                _text(self.invention_rationale_zh, "other invented beat rationale")
        elif any(value is not None for value in (self.invention_purpose, self.invention_rationale_zh, self.invention_provenance_zh)):
            raise B13DispositionError("invention fields are only permitted for adaptation_invented beats")
        material = {"identity_key": self.identity_key, "origin": self.origin.value, "source_event_refs": [ref.to_dict() for ref in source_refs], "canonical_anchor_refs": [ref.to_dict() for ref in anchors]}
        object.__setattr__(self, "adapted_story_beat_id", _derived_id("adapted_beat", material))

    def to_dict(self) -> dict[str, Any]:
        return {"adapted_story_beat_id": self.adapted_story_beat_id, "identity_key": self.identity_key, "summary_zh": self.summary_zh, "origin": self.origin.value, "canon_compatibility": self.canon_compatibility.value, "source_event_refs": [item.to_dict() for item in self.source_event_refs], "participants": [item.to_dict() for item in self.participants], "canonical_anchor_refs": [item.to_dict() for item in self.canonical_anchor_refs], "invention_purpose": None if self.invention_purpose is None else self.invention_purpose.value, "invention_rationale_zh": self.invention_rationale_zh, "invention_provenance_zh": self.invention_provenance_zh}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "AdaptedStoryBeat":
        fields = {"adapted_story_beat_id", "identity_key", "summary_zh", "origin", "canon_compatibility", "source_event_refs", "participants", "canonical_anchor_refs", "invention_purpose", "invention_rationale_zh", "invention_provenance_zh"}
        value = _exact(value, fields, cls.__name__)
        if not all(isinstance(value[name], list) for name in ("source_event_refs", "participants", "canonical_anchor_refs")):
            raise B13DispositionError("story beat reference collections must be arrays")
        result = cls(value["identity_key"], value["summary_zh"], value["origin"], value["canon_compatibility"], tuple(CanonicalTargetReference.from_dict(item) for item in value["source_event_refs"]), tuple(BeatParticipant.from_dict(item) for item in value["participants"]), tuple(CanonicalTargetReference.from_dict(item) for item in value["canonical_anchor_refs"]), value["invention_purpose"], value["invention_rationale_zh"], value["invention_provenance_zh"])
        if value["adapted_story_beat_id"] != result.adapted_story_beat_id:
            raise B13DispositionError("adapted_story_beat_id does not match deterministic identity")
        return result


@dataclass(frozen=True, slots=True)
class AdaptedEndpoint:
    kind: AdaptedEndpointKind
    adapted_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _enum(self.kind, AdaptedEndpointKind, "adapted endpoint kind"))
        _key(self.adapted_id, "adapted endpoint id")

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind.value, "adapted_id": self.adapted_id}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "AdaptedEndpoint":
        return cls(**_exact(value, {"kind", "adapted_id"}, cls.__name__))


@dataclass(frozen=True, slots=True)
class AdaptationMapping:
    """One active/inactive edge in the first-class source-to-adapted graph."""

    source_ref: CanonicalTargetReference
    adapted_endpoint: AdaptedEndpoint
    contribution_role: MappingContributionRole
    retained_semantics_zh: str
    transformed_semantics_zh: str
    provenance_zh: str
    active: bool = True
    mapping_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.source_ref, CanonicalTargetReference):
            raise B13DispositionError("mapping source_ref must be a canonical source reference")
        if not isinstance(self.adapted_endpoint, AdaptedEndpoint):
            raise B13DispositionError("mapping adapted_endpoint must be an AdaptedEndpoint")
        object.__setattr__(self, "contribution_role", _enum(self.contribution_role, MappingContributionRole, "mapping contribution role"))
        for field_name in ("retained_semantics_zh", "transformed_semantics_zh", "provenance_zh"):
            _text(getattr(self, field_name), field_name)
        if not isinstance(self.active, bool):
            raise B13DispositionError("mapping active must be boolean")
        material = {"source_ref": self.source_ref.to_dict(), "adapted_endpoint": self.adapted_endpoint.to_dict(), "contribution_role": self.contribution_role.value, "retained_semantics_zh": self.retained_semantics_zh, "transformed_semantics_zh": self.transformed_semantics_zh, "provenance_zh": self.provenance_zh, "active": self.active}
        object.__setattr__(self, "mapping_id", _derived_id("adaptation_map", material))

    def to_dict(self) -> dict[str, Any]:
        return {"mapping_id": self.mapping_id, "source_ref": self.source_ref.to_dict(), "adapted_endpoint": self.adapted_endpoint.to_dict(), "contribution_role": self.contribution_role.value, "retained_semantics_zh": self.retained_semantics_zh, "transformed_semantics_zh": self.transformed_semantics_zh, "provenance_zh": self.provenance_zh, "active": self.active}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "AdaptationMapping":
        fields = {"mapping_id", "source_ref", "adapted_endpoint", "contribution_role", "retained_semantics_zh", "transformed_semantics_zh", "provenance_zh", "active"}
        value = _exact(value, fields, cls.__name__)
        result = cls(CanonicalTargetReference.from_dict(value["source_ref"]), AdaptedEndpoint.from_dict(value["adapted_endpoint"]), value["contribution_role"], value["retained_semantics_zh"], value["transformed_semantics_zh"], value["provenance_zh"], value["active"])
        if value["mapping_id"] != result.mapping_id:
            raise B13DispositionError("mapping_id does not match deterministic identity")
        return result


@dataclass(frozen=True, slots=True)
class AdaptedFactDivergence:
    source_fact_ref: CanonicalTargetReference
    adapted_assertion_zh: str
    provenance_zh: str
    canon_compatibility: CanonCompatibility = CanonCompatibility.DELIBERATE_DIVERGENCE

    def __post_init__(self) -> None:
        if not isinstance(self.source_fact_ref, CanonicalTargetReference) or self.source_fact_ref.kind != "fact":
            raise B13DispositionError("source_fact_ref must be a canonical fact reference")
        _text(self.adapted_assertion_zh, "adapted factual assertion")
        _text(self.provenance_zh, "adapted factual divergence provenance")
        compatibility = _enum(self.canon_compatibility, CanonCompatibility, "adapted factual divergence canon compatibility")
        if compatibility is not CanonCompatibility.DELIBERATE_DIVERGENCE:
            raise B13DispositionError("adapted factual divergence must be an explicit deliberate_divergence")
        object.__setattr__(self, "canon_compatibility", compatibility)

    def to_dict(self) -> dict[str, Any]: return {"source_fact_ref": self.source_fact_ref.to_dict(), "adapted_assertion_zh": self.adapted_assertion_zh, "provenance_zh": self.provenance_zh, "canon_compatibility": self.canon_compatibility.value}
    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "AdaptedFactDivergence":
        value = _exact(value, {"source_fact_ref", "adapted_assertion_zh", "provenance_zh", "canon_compatibility"}, cls.__name__)
        return cls(CanonicalTargetReference.from_dict(value["source_fact_ref"]), value["adapted_assertion_zh"], value["provenance_zh"], value["canon_compatibility"])


@dataclass(frozen=True, slots=True)
class RelationshipArcRequirement:
    source_relationship_ref: CanonicalTargetReference
    requirement_zh: str
    provenance_zh: str

    def __post_init__(self) -> None:
        if not isinstance(self.source_relationship_ref, CanonicalTargetReference) or self.source_relationship_ref.kind != "relationship":
            raise B13DispositionError("source_relationship_ref must be a canonical relationship reference")
        _text(self.requirement_zh, "relationship requirement")
        _text(self.provenance_zh, "relationship requirement provenance")

    def to_dict(self) -> dict[str, Any]: return {"source_relationship_ref": self.source_relationship_ref.to_dict(), "requirement_zh": self.requirement_zh, "provenance_zh": self.provenance_zh}
    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RelationshipArcRequirement":
        value = _exact(value, {"source_relationship_ref", "requirement_zh", "provenance_zh"}, cls.__name__)
        return cls(CanonicalTargetReference.from_dict(value["source_relationship_ref"]), value["requirement_zh"], value["provenance_zh"])


@dataclass(frozen=True, slots=True)
class ContinuityStoryState:
    adapted_story_beat_id: str
    assertion_zh: str
    provenance_zh: str

    def __post_init__(self) -> None:
        _key(self.adapted_story_beat_id, "continuity story beat id")
        _text(self.assertion_zh, "continuity story-state assertion")
        _text(self.provenance_zh, "continuity story-state provenance")

    def to_dict(self) -> dict[str, Any]: return {"adapted_story_beat_id": self.adapted_story_beat_id, "assertion_zh": self.assertion_zh, "provenance_zh": self.provenance_zh}
    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ContinuityStoryState":
        return cls(**_exact(value, {"adapted_story_beat_id", "assertion_zh", "provenance_zh"}, cls.__name__))


@dataclass(frozen=True, slots=True)
class AdaptedContinuityOverlay:
    """Sparse adaptation-owned deltas, never a copied canonical Story Bible."""

    factual_divergences: tuple[AdaptedFactDivergence, ...] = ()
    relationship_requirements: tuple[RelationshipArcRequirement, ...] = ()
    story_state_requirements: tuple[ContinuityStoryState, ...] = ()

    def __post_init__(self) -> None:
        facts, relationships, states = tuple(self.factual_divergences), tuple(self.relationship_requirements), tuple(self.story_state_requirements)
        if any(not isinstance(item, AdaptedFactDivergence) for item in facts) or len({item.source_fact_ref for item in facts}) != len(facts):
            raise B13DispositionError("factual_divergences must have unique typed source facts")
        if any(not isinstance(item, RelationshipArcRequirement) for item in relationships) or len({item.source_relationship_ref for item in relationships}) != len(relationships):
            raise B13DispositionError("relationship_requirements must have unique typed source relationships")
        if any(not isinstance(item, ContinuityStoryState) for item in states) or len({item.adapted_story_beat_id for item in states}) != len(states):
            raise B13DispositionError("story_state_requirements must have unique typed adapted beats")
        object.__setattr__(self, "factual_divergences", tuple(sorted(facts, key=lambda item: item.source_fact_ref.canonical_id)))
        object.__setattr__(self, "relationship_requirements", tuple(sorted(relationships, key=lambda item: item.source_relationship_ref.canonical_id)))
        object.__setattr__(self, "story_state_requirements", tuple(sorted(states, key=lambda item: item.adapted_story_beat_id)))

    def to_dict(self) -> dict[str, Any]:
        return {"factual_divergences": [item.to_dict() for item in self.factual_divergences], "relationship_requirements": [item.to_dict() for item in self.relationship_requirements], "story_state_requirements": [item.to_dict() for item in self.story_state_requirements]}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "AdaptedContinuityOverlay":
        value = _exact(value, {"factual_divergences", "relationship_requirements", "story_state_requirements"}, cls.__name__)
        if not all(isinstance(value[name], list) for name in value):
            raise B13DispositionError("continuity overlay collections must be arrays")
        return cls(tuple(AdaptedFactDivergence.from_dict(item) for item in value["factual_divergences"]), tuple(RelationshipArcRequirement.from_dict(item) for item in value["relationship_requirements"]), tuple(ContinuityStoryState.from_dict(item) for item in value["story_state_requirements"]))


@dataclass(frozen=True, slots=True)
class B13DispositionPlan:
    """Complete event disposition and adapted-identity graph over one A5/A6 snapshot."""

    source_snapshot: AdaptationSourceSnapshot
    event_dispositions: tuple[EventDisposition, ...]
    adapted_characters: tuple[AdaptedCharacter, ...]
    anonymous_roles: tuple[AnonymousRole, ...]
    adapted_story_beats: tuple[AdaptedStoryBeat, ...]
    adaptation_mappings: tuple[AdaptationMapping, ...]
    continuity_overlay: AdaptedContinuityOverlay = field(default_factory=AdaptedContinuityOverlay)
    schema_version: int = B13_DISPOSITION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != B13_DISPOSITION_SCHEMA_VERSION:
            raise B13DispositionError("unsupported B1.3 disposition schema_version")
        if not isinstance(self.source_snapshot, AdaptationSourceSnapshot):
            raise B13DispositionError("source_snapshot must be an exact AdaptationSourceSnapshot")
        for field_name, expected, key in (("event_dispositions", EventDisposition, lambda item: item.source_event_ref.canonical_id), ("adapted_characters", AdaptedCharacter, lambda item: item.adapted_character_id), ("anonymous_roles", AnonymousRole, lambda item: item.anonymous_role_id), ("adapted_story_beats", AdaptedStoryBeat, lambda item: item.adapted_story_beat_id), ("adaptation_mappings", AdaptationMapping, lambda item: item.mapping_id)):
            values = tuple(getattr(self, field_name))
            if any(not isinstance(item, expected) for item in values) or len({key(item) for item in values}) != len(values):
                raise B13DispositionError(f"{field_name} must contain unique {expected.__name__} values")
            object.__setattr__(self, field_name, tuple(sorted(values, key=key)))
        if not isinstance(self.continuity_overlay, AdaptedContinuityOverlay):
            raise B13DispositionError("continuity_overlay must be an AdaptedContinuityOverlay")
        self.validate()

    def validate(self) -> None:
        snapshot_refs = set(self.source_snapshot.canonical_references)
        event_refs = {ref for ref in snapshot_refs if ref.kind == "event"}
        dispositions = {item.source_event_ref: item for item in self.event_dispositions}
        if set(dispositions) != event_refs:
            raise B13DispositionError("event dispositions must cover exactly every canonical event in the source snapshot")
        characters = {item.adapted_character_id: item for item in self.adapted_characters}
        roles = {item.anonymous_role_id: item for item in self.anonymous_roles}
        beats = {item.adapted_story_beat_id: item for item in self.adapted_story_beats}
        anonymous_uses: dict[str, int] = {role_id: 0 for role_id in roles}
        for character in characters.values():
            if any(item.source_character_ref not in snapshot_refs for item in character.function_provenance):
                raise B13DispositionError("adapted character provenance references source outside exact snapshot")
            if character.origin is AdaptedCharacterOrigin.SOURCE_COMPOSITE and character.adapted_character_id.startswith("char_"):
                raise B13DispositionError("source_composite adapted character must not use an A4 canonical identity")
        for beat in beats.values():
            if any(ref not in snapshot_refs for ref in (*beat.source_event_refs, *beat.canonical_anchor_refs)):
                raise B13DispositionError("adapted story beat references source outside exact snapshot")
            for participant in beat.participants:
                if participant.kind is ParticipantKind.ADAPTED_CHARACTER:
                    if participant.participant_id not in characters:
                        raise B13DispositionError("material beat participant does not resolve to an AdaptedCharacter")
                else:
                    if participant.participant_id not in roles:
                        raise B13DispositionError("material beat participant does not resolve to a valid AnonymousRole")
                    anonymous_uses[participant.participant_id] += 1
        if any(count != 1 for count in anonymous_uses.values()):
            raise B13DispositionError("AnonymousRole must resolve exactly one material beat participant")
        active_event_mappings: dict[CanonicalTargetReference, list[AdaptationMapping]] = {ref: [] for ref in event_refs}
        for mapping in self.adaptation_mappings:
            if mapping.source_ref not in snapshot_refs:
                raise B13DispositionError("mapping source_ref is absent from exact source snapshot")
            endpoint = mapping.adapted_endpoint
            if endpoint.kind is AdaptedEndpointKind.STORY_BEAT:
                beat = beats.get(endpoint.adapted_id)
                if beat is None:
                    raise B13DispositionError("mapping adapted story-beat endpoint does not resolve")
                if mapping.source_ref.kind == "event" and mapping.source_ref not in beat.source_event_refs:
                    raise B13DispositionError("event mapping must be declared by its adapted story beat source membership")
                if mapping.source_ref.kind != "event" and mapping.source_ref not in beat.canonical_anchor_refs:
                    raise B13DispositionError("non-event beat mapping must be declared by a canonical anchor")
            else:
                character = characters.get(endpoint.adapted_id)
                if character is None:
                    raise B13DispositionError("mapping adapted character endpoint does not resolve")
                if mapping.source_ref.kind != "character" or mapping.source_ref not in {item.source_character_ref for item in character.function_provenance}:
                    raise B13DispositionError("character mapping must be backed by character function provenance")
            if mapping.active and mapping.source_ref.kind == "event":
                active_event_mappings[mapping.source_ref].append(mapping)
        for event_ref, disposition in dispositions.items():
            active = active_event_mappings[event_ref]
            if disposition.disposition is CanonicalEventDisposition.OMIT:
                if active:
                    raise B13DispositionError("omit event forbids active mappings")
                continue
            if not active:
                raise B13DispositionError("keep/compress/merge_member event requires an active mapping")
            if disposition.disposition is CanonicalEventDisposition.MERGE_MEMBER:
                if not any(beats[mapping.adapted_endpoint.adapted_id].origin is AdaptedStoryBeatOrigin.SOURCE_COMPOSITE and len(beats[mapping.adapted_endpoint.adapted_id].source_event_refs) >= 2 for mapping in active):
                    raise B13DispositionError("merge_member event requires an active mapping to a multi-source source_composite beat")
        overlay = self.continuity_overlay
        if any(item.source_fact_ref not in snapshot_refs for item in overlay.factual_divergences):
            raise B13DispositionError("continuity factual divergence references source outside exact snapshot")
        if any(item.source_relationship_ref not in snapshot_refs for item in overlay.relationship_requirements):
            raise B13DispositionError("continuity relationship requirement references source outside exact snapshot")
        if any(item.adapted_story_beat_id not in beats for item in overlay.story_state_requirements):
            raise B13DispositionError("continuity story-state requirement references an unknown adapted beat")

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": self.schema_version, "source_snapshot": self.source_snapshot.to_dict(), "event_dispositions": [item.to_dict() for item in self.event_dispositions], "adapted_characters": [item.to_dict() for item in self.adapted_characters], "anonymous_roles": [item.to_dict() for item in self.anonymous_roles], "adapted_story_beats": [item.to_dict() for item in self.adapted_story_beats], "adaptation_mappings": [item.to_dict() for item in self.adaptation_mappings], "continuity_overlay": self.continuity_overlay.to_dict()}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "B13DispositionPlan":
        fields = {"schema_version", "source_snapshot", "event_dispositions", "adapted_characters", "anonymous_roles", "adapted_story_beats", "adaptation_mappings", "continuity_overlay"}
        value = _exact(value, fields, cls.__name__)
        if not all(isinstance(value[name], list) for name in ("event_dispositions", "adapted_characters", "anonymous_roles", "adapted_story_beats", "adaptation_mappings")):
            raise B13DispositionError("B1.3 plan collections must be arrays")
        return cls(AdaptationSourceSnapshot.from_dict(value["source_snapshot"]), tuple(EventDisposition.from_dict(item) for item in value["event_dispositions"]), tuple(AdaptedCharacter.from_dict(item) for item in value["adapted_characters"]), tuple(AnonymousRole.from_dict(item) for item in value["anonymous_roles"]), tuple(AdaptedStoryBeat.from_dict(item) for item in value["adapted_story_beats"]), tuple(AdaptationMapping.from_dict(item) for item in value["adaptation_mappings"]), AdaptedContinuityOverlay.from_dict(value["continuity_overlay"]), value["schema_version"])

    def content_hash(self) -> str:
        return content_hash(self.to_dict())
