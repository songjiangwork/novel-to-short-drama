"""Frozen B1.1 Adaptation Target contracts and deterministic normalization.

This module is intentionally provider-free.  It represents a target, resolves
the three declared input layers, and rejects structural or hard numeric
contradictions before semantic planning begins.  B1.5, not this module,
resolves a final episode count.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Mapping

from short_drama.artifacts import ArtifactRef, FileArtifactStore, content_hash
from short_drama.foundation import FilePointerStore


ADAPTATION_TARGET_SCHEMA_VERSION = 1
_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
_REF_PATTERNS = {
    "character": re.compile(r"^char_[0-9]{4,}$"),
    "location": re.compile(r"^loc_[0-9]{4,}$"),
    "fact": re.compile(r"^fact_[0-9]{6,}$"),
    "event": re.compile(r"^evt_[0-9]{6,}$"),
    "relationship": re.compile(r"^rel_[0-9]{6,}$"),
    "arc": re.compile(r"^arc_[0-9]{6,}$"),
    "turning_point": re.compile(r"^turn_[0-9]{6,}$"),
    "reveal": re.compile(r"^reveal_[0-9]{6,}$"),
    "payoff": re.compile(r"^payoff_[0-9]{6,}$"),
}


class AdaptationTargetError(ValueError):
    """A B1.1 target contract or deterministic feasibility invariant failed."""


class Provenance(str, Enum):
    USER = "user"
    PROFILE_DEFAULT = "profile_default"
    DERIVED = "derived"
    MODEL_PROPOSED = "model_proposed"


class FidelityProfile(str, Enum):
    HIGH = "high"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"


class ConstraintKind(str, Enum):
    FIXED = "fixed"
    TARGET = "target"
    RANGE = "range"
    DERIVED = "derived"


class ConstraintStrength(str, Enum):
    HARD = "hard"
    SOFT = "soft"


def _exact(value: Any, fields: set[str], model: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise AdaptationTargetError(f"{model} must contain exactly: {', '.join(sorted(fields))}")
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\0" in value:
        raise AdaptationTargetError(f"{name} must be non-empty UTF-8 text without NUL")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise AdaptationTargetError(f"{name} must contain valid UTF-8 text") from exc
    return value


def _id(value: Any, name: str) -> str:
    value = _text(value, name)
    if not _ID.fullmatch(value):
        raise AdaptationTargetError(f"{name} must be a safe identifier")
    return value


def _enum(value: Any, enum: type[Enum], name: str) -> Enum:
    try:
        return value if isinstance(value, enum) else enum(value)
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(item.value for item in enum)
        raise AdaptationTargetError(f"{name} must be one of: {allowed}") from exc


def _number(value: Any, name: str, *, integer: bool = False) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise AdaptationTargetError(f"{name} must be a finite number > 0")
    if integer and (not isinstance(value, int) or value < 1):
        raise AdaptationTargetError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True, slots=True)
class TargetValue:
    """A value plus the precise origin of this effective field."""

    value: Any
    provenance: Provenance

    def __post_init__(self) -> None:
        object.__setattr__(self, "provenance", _enum(self.provenance, Provenance, "provenance"))


@dataclass(frozen=True, slots=True)
class CanonicalTargetReference:
    kind: str
    canonical_id: str

    def __post_init__(self) -> None:
        if self.kind not in _REF_PATTERNS:
            raise AdaptationTargetError(f"unknown canonical target kind: {self.kind!r}")
        if not isinstance(self.canonical_id, str) or not _REF_PATTERNS[self.kind].fullmatch(self.canonical_id):
            raise AdaptationTargetError(f"canonical_id is not a valid {self.kind} stable ID")

    def to_dict(self) -> dict[str, str]:
        return {"kind": self.kind, "canonical_id": self.canonical_id}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CanonicalTargetReference":
        value = _exact(value, {"kind", "canonical_id"}, cls.__name__)
        return cls(value["kind"], value["canonical_id"])


@dataclass(frozen=True, slots=True)
class AdaptationSourceSnapshot:
    """The exact validated A5/A6 input identity consumed by a B1.1 target.

    This is a read-only binding to existing Foundation ``ArtifactRef`` values,
    not a new artifact or publication mechanism.  The contained ID universe is
    the sole authority for canonical target refs in the resolved target.
    """

    story_analysis_manifest_ref: ArtifactRef
    consolidation_manifest_ref: ArtifactRef
    entity_map_ref: ArtifactRef
    canonical_character_registry_ref: ArtifactRef
    canonical_location_registry_ref: ArtifactRef
    canonical_fact_set_ref: ArtifactRef
    canonical_event_set_ref: ArtifactRef
    canonical_relationship_set_ref: ArtifactRef
    arc_analysis_ref: ArtifactRef
    canonical_references: tuple[CanonicalTargetReference, ...]

    def __post_init__(self) -> None:
        expected = {
            "story_analysis_manifest_ref": "story_analysis_manifest",
            "consolidation_manifest_ref": "consolidation_manifest",
            "entity_map_ref": "entity_map",
            "canonical_character_registry_ref": "canonical_character_registry",
            "canonical_location_registry_ref": "canonical_location_registry",
            "canonical_fact_set_ref": "canonical_fact_set",
            "canonical_event_set_ref": "canonical_event_set",
            "canonical_relationship_set_ref": "canonical_relationship_set",
            "arc_analysis_ref": "arc_analysis",
        }
        for field, artifact_type in expected.items():
            ref = getattr(self, field)
            if not isinstance(ref, ArtifactRef) or ref.artifact_type != artifact_type:
                raise AdaptationTargetError(f"{field} must be an exact {artifact_type} ArtifactRef")
        refs = tuple(self.canonical_references)
        if any(not isinstance(ref, CanonicalTargetReference) for ref in refs):
            raise AdaptationTargetError("canonical_references must contain CanonicalTargetReference values")
        if len({(ref.kind, ref.canonical_id) for ref in refs}) != len(refs):
            raise AdaptationTargetError("canonical_references must not contain duplicates")
        object.__setattr__(self, "canonical_references", refs)

    def contains(self, reference: CanonicalTargetReference) -> bool:
        return reference in self.canonical_references

    def to_dict(self) -> dict[str, Any]:
        return {
            "story_analysis_manifest_ref": self.story_analysis_manifest_ref.to_dict(),
            "consolidation_manifest_ref": self.consolidation_manifest_ref.to_dict(),
            "entity_map_ref": self.entity_map_ref.to_dict(),
            "canonical_character_registry_ref": self.canonical_character_registry_ref.to_dict(),
            "canonical_location_registry_ref": self.canonical_location_registry_ref.to_dict(),
            "canonical_fact_set_ref": self.canonical_fact_set_ref.to_dict(),
            "canonical_event_set_ref": self.canonical_event_set_ref.to_dict(),
            "canonical_relationship_set_ref": self.canonical_relationship_set_ref.to_dict(),
            "arc_analysis_ref": self.arc_analysis_ref.to_dict(),
            "canonical_references": [ref.to_dict() for ref in self.canonical_references],
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "AdaptationSourceSnapshot":
        fields = {
            "story_analysis_manifest_ref", "consolidation_manifest_ref", "entity_map_ref",
            "canonical_character_registry_ref", "canonical_location_registry_ref",
            "canonical_fact_set_ref", "canonical_event_set_ref", "canonical_relationship_set_ref",
            "arc_analysis_ref", "canonical_references",
        }
        value = _exact(value, fields, cls.__name__)
        if not isinstance(value["canonical_references"], list):
            raise AdaptationTargetError("canonical_references must be an array")
        try:
            return cls(
                **{field: ArtifactRef.from_dict(value[field]) for field in fields if field.endswith("_ref")},
                canonical_references=tuple(CanonicalTargetReference.from_dict(item) for item in value["canonical_references"]),
            )
        except (TypeError, ValueError) as exc:
            raise AdaptationTargetError(f"invalid AdaptationSourceSnapshot: {exc}") from exc

    @classmethod
    def from_store(
        cls, store: FileArtifactStore, pointers: FilePointerStore,
        story_analysis_manifest_ref: ArtifactRef,
    ) -> "AdaptationSourceSnapshot":
        """Read and verify the exact A5/A6 member artifacts without writing.

        ``ArtifactRef`` hash verification is delegated to the existing
        FileArtifactStore.  Existing A6/A5 CURRENT verification seams establish
        the validated authority before this adapter reads the pinned A4
        registries.  It performs no writes or history/latest scan.
        """
        if not isinstance(store, FileArtifactStore) or not isinstance(pointers, FilePointerStore):
            raise AdaptationTargetError("store/pointers must be Foundation file stores")
        from short_drama.story.consolidation import (
            CanonicalEventSet, CanonicalFactSet, CanonicalRelationshipSet, ConsolidationManifest,
        )
        from short_drama.story.consolidation_persistence import ConsolidationPersistenceService
        from short_drama.story.reconciliation import (
            CanonicalCharacterRegistry, CanonicalLocationRegistry, EntityMap,
        )
        from short_drama.story.story_analysis import ArcAnalysis, StoryAnalysisManifest
        from short_drama.story.story_analysis_persistence import (
            StoryAnalysisPersistenceService, a6_base_artifact_id, a6_pointer_id,
        )

        def load(ref: ArtifactRef, artifact_type: str, model: Any) -> Any:
            if not isinstance(ref, ArtifactRef) or ref.artifact_type != artifact_type:
                raise AdaptationTargetError(f"expected exact {artifact_type} ArtifactRef")
            try:
                envelope = store.get_ref(ref)
                if not isinstance(envelope.schema_version, int) or not isinstance(envelope.payload, dict):
                    raise AdaptationTargetError(f"invalid persisted {artifact_type}")
                typed = model.from_dict(envelope.payload)
                if typed.to_dict() != envelope.payload:
                    raise AdaptationTargetError(f"persisted {artifact_type} is not canonical")
                return typed
            except AdaptationTargetError:
                raise
            except Exception as exc:  # Existing store verifies ref/hash before this point.
                raise AdaptationTargetError(f"cannot resolve exact {artifact_type} snapshot input") from exc

        untrusted_a6 = load(story_analysis_manifest_ref, "story_analysis_manifest", StoryAnalysisManifest)
        a6_base = a6_base_artifact_id(
            untrusted_a6.project_id, untrusted_a6.document_id,
            untrusted_a6.semantic_identity.story_analysis_profile_id,
        )
        a6_service = StoryAnalysisPersistenceService(store, pointers)
        pointer_id = a6_pointer_id(
            untrusted_a6.project_id, untrusted_a6.document_id,
            untrusted_a6.semantic_identity.story_analysis_profile_id,
        )
        try:
            _pointer_ref, current_a6_ref = a6_service._current_a6_pointer(pointer_id)
            if current_a6_ref != story_analysis_manifest_ref:
                raise AdaptationTargetError("bound A6 manifest is not the exact validated A6 CURRENT")
            a6, _chars, _events, arcs, _structure, _bible, _report_ref = a6_service._load_current_integrity(
                a6_base, story_analysis_manifest_ref
            )
        except AdaptationTargetError:
            raise
        except Exception as exc:
            raise AdaptationTargetError("cannot verify exact validated A6 snapshot") from exc
        if a6.consolidation_manifest_ref != a6.upstream_identity.consolidation_manifest_ref:
            raise AdaptationTargetError("A6 manifest has a mismatched A5 upstream binding")
        a5 = load(a6.consolidation_manifest_ref, "consolidation_manifest", ConsolidationManifest)
        try:
            validated_a5 = ConsolidationPersistenceService(store, pointers).require_current_validated(
                project_id=a5.project_id,
                document_id=a5.document_id,
                consolidation_profile_id=a5.semantic_identity.consolidation_profile_id,
            )
            if validated_a5.manifest_ref != a6.consolidation_manifest_ref:
                raise AdaptationTargetError("A6 binds a non-current or unvalidated A5 manifest")
        except AdaptationTargetError:
            raise
        except Exception as exc:
            raise AdaptationTargetError("cannot verify exact validated A5 snapshot") from exc
        if (a5.project_id, a5.document_id) != (a6.project_id, a6.document_id):
            raise AdaptationTargetError("A5/A6 snapshot project or document binding differs")
        leaf_refs = (
            a5.canonical_fact_set_ref, a5.canonical_event_set_ref,
            a5.canonical_relationship_set_ref,
        )
        if any(ref.revision != a6.consolidation_manifest_ref.revision for ref in leaf_refs):
            raise AdaptationTargetError("A5 canonical leaves cross revisions within the bound snapshot")
        entity_map = load(a5.entity_map_ref, "entity_map", EntityMap)
        characters = load(
            entity_map.canonical_character_registry_ref,
            "canonical_character_registry", CanonicalCharacterRegistry,
        )
        locations = load(
            entity_map.canonical_location_registry_ref,
            "canonical_location_registry", CanonicalLocationRegistry,
        )
        facts = load(a5.canonical_fact_set_ref, "canonical_fact_set", CanonicalFactSet)
        events = load(a5.canonical_event_set_ref, "canonical_event_set", CanonicalEventSet)
        relationships = load(a5.canonical_relationship_set_ref, "canonical_relationship_set", CanonicalRelationshipSet)
        arcs = load(a6.arc_analysis_ref, "arc_analysis", ArcAnalysis)
        refs = list(_canonical_references_from_a4(
            entity_map, characters.entities, locations.entities,
        ))
        refs.extend(CanonicalTargetReference("fact", item.fact_id) for item in facts.facts)
        refs.extend(CanonicalTargetReference("event", item.event_id) for item in events.events)
        refs.extend(CanonicalTargetReference("relationship", item.relationship_id) for item in relationships.relationships)
        refs.extend(CanonicalTargetReference("arc", item.arc_id) for item in arcs.arcs)
        refs.extend(CanonicalTargetReference("turning_point", item.turning_point_id) for item in arcs.turning_points)
        refs.extend(CanonicalTargetReference("reveal", item.reveal_id) for item in arcs.reveals)
        refs.extend(CanonicalTargetReference("payoff", item.payoff_id) for item in arcs.foreshadow_payoffs)
        return cls(
            story_analysis_manifest_ref=story_analysis_manifest_ref,
            consolidation_manifest_ref=a6.consolidation_manifest_ref,
            entity_map_ref=a5.entity_map_ref,
            canonical_character_registry_ref=entity_map.canonical_character_registry_ref,
            canonical_location_registry_ref=entity_map.canonical_location_registry_ref,
            canonical_fact_set_ref=a5.canonical_fact_set_ref,
            canonical_event_set_ref=a5.canonical_event_set_ref,
            canonical_relationship_set_ref=a5.canonical_relationship_set_ref,
            arc_analysis_ref=a6.arc_analysis_ref,
            canonical_references=tuple(refs),
        )


def _canonical_references_from_a4(
    entity_map: Any, canonical_characters: Iterable[Any], canonical_locations: Iterable[Any],
) -> tuple[CanonicalTargetReference, ...]:
    """Use A4 registries, not candidate-to-canonical map entries, as the universe."""
    characters = tuple(canonical_characters)
    locations = tuple(canonical_locations)
    character_ids = {item.canonical_id for item in characters}
    location_ids = {item.canonical_id for item in locations}
    if len(character_ids) != len(characters) or len(location_ids) != len(locations):
        raise AdaptationTargetError("pinned A4 canonical registries contain duplicate canonical IDs")
    mapped_ids = {item.canonical_id for item in entity_map.entries if item.canonical_id is not None}
    if not mapped_ids <= character_ids | location_ids:
        raise AdaptationTargetError("A4 EntityMap canonical mappings disagree with its pinned registries")
    return (
        *(CanonicalTargetReference("character", item.canonical_id) for item in characters),
        *(CanonicalTargetReference("location", item.canonical_id) for item in locations),
    )


@dataclass(frozen=True, slots=True)
class UnresolvedTargetReference:
    """A natural-language target that was not safely canonicalized."""

    text: str

    def __post_init__(self) -> None:
        _text(self.text, "unresolved reference text")

    def to_dict(self) -> dict[str, str]:
        return {"text": self.text}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "UnresolvedTargetReference":
        value = _exact(value, {"text"}, cls.__name__)
        return cls(value["text"])


TargetReference = CanonicalTargetReference | UnresolvedTargetReference


def _reference_from_dict(value: Any) -> TargetReference:
    if not isinstance(value, dict):
        raise AdaptationTargetError("target reference must be an object")
    if set(value) == {"kind", "canonical_id"}:
        return CanonicalTargetReference.from_dict(value)
    if set(value) == {"text"}:
        return UnresolvedTargetReference.from_dict(value)
    raise AdaptationTargetError("target reference must be canonical or explicitly unresolved")


@dataclass(frozen=True, slots=True)
class TargetElement:
    """A required or forbidden target element with a resolved-or-unresolved ref."""

    reference: TargetReference
    instruction: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.reference, (CanonicalTargetReference, UnresolvedTargetReference)):
            raise AdaptationTargetError("TargetElement.reference must be a target reference")
        if self.instruction is not None:
            _text(self.instruction, "TargetElement.instruction")

    def to_dict(self) -> dict[str, Any]:
        return {
            "reference": self.reference.to_dict(),
            "instruction": self.instruction,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TargetElement":
        value = _exact(value, {"reference", "instruction"}, cls.__name__)
        return cls(_reference_from_dict(value["reference"]), value["instruction"])


@dataclass(frozen=True, slots=True)
class NumericConstraint:
    """One duration/count constraint; kind and strength are intentionally orthogonal."""

    kind: ConstraintKind
    strength: ConstraintStrength
    value: int | float | None = None
    minimum: int | float | None = None
    maximum: int | float | None = None
    derivation: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _enum(self.kind, ConstraintKind, "constraint.kind"))
        object.__setattr__(self, "strength", _enum(self.strength, ConstraintStrength, "constraint.strength"))
        if self.kind in (ConstraintKind.FIXED, ConstraintKind.TARGET):
            _number(self.value, "constraint.value")
            if self.minimum is not None or self.maximum is not None or self.derivation is not None:
                raise AdaptationTargetError("fixed/target constraint may only have value")
        elif self.kind is ConstraintKind.RANGE:
            if self.value is not None or self.derivation is not None:
                raise AdaptationTargetError("range constraint may only have minimum/maximum")
            if self.minimum is None and self.maximum is None:
                raise AdaptationTargetError("range constraint requires minimum or maximum")
            if self.minimum is not None:
                _number(self.minimum, "constraint.minimum")
            if self.maximum is not None:
                _number(self.maximum, "constraint.maximum")
            if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
                raise AdaptationTargetError("constraint.minimum cannot exceed constraint.maximum")
        else:
            if self.value is not None or self.minimum is not None or self.maximum is not None:
                raise AdaptationTargetError("derived constraint may only have derivation")
            _text(self.derivation, "constraint.derivation")

    def validate_for(self, field: str) -> None:
        if field in {"total_duration", "per_episode_duration", "episode_count"}:
            for value in (self.value, self.minimum, self.maximum):
                if value is not None:
                    _number(value, f"{field} constraint", integer=True)

    def hard_interval(self, *, lower_default: int) -> tuple[int | float, int | float] | None:
        """Return the feasibility interval for hard fixed/range constraints only.

        A target remains an optimization target even when it is hard, so it
        does not silently become a fixed value.  Derived values cannot be
        evaluated at B1.1 without the later inputs named by their derivation.
        """
        if self.strength is not ConstraintStrength.HARD:
            return None
        if self.kind is ConstraintKind.FIXED:
            return (self.value, self.value)
        if self.kind is ConstraintKind.RANGE:
            return (
                lower_default if self.minimum is None else self.minimum,
                math.inf if self.maximum is None else self.maximum,
            )
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value, "strength": self.strength.value,
            "value": self.value, "minimum": self.minimum, "maximum": self.maximum,
            "derivation": self.derivation,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "NumericConstraint":
        value = _exact(value, {"kind", "strength", "value", "minimum", "maximum", "derivation"}, cls.__name__)
        return cls(**value)


def _constraints(value: Any, field: str) -> tuple[NumericConstraint, ...]:
    if not isinstance(value, (tuple, list)):
        raise AdaptationTargetError(f"{field} must be an array of NumericConstraint")
    constraints = tuple(value)
    if not constraints:
        raise AdaptationTargetError(f"{field} must not be empty when specified")
    if any(not isinstance(item, NumericConstraint) for item in constraints):
        raise AdaptationTargetError(f"{field} must contain NumericConstraint values")
    for item in constraints:
        item.validate_for(field)
    # Exact duplicate conditions add no meaning and make a target unnecessarily ambiguous.
    if len({content_hash(item.to_dict()) for item in constraints}) != len(constraints):
        raise AdaptationTargetError(f"{field} must not contain duplicate constraints")
    return constraints


_VALUE_FIELDS = (
    "narrative_format", "presentation_orientation", "adaptation_genre", "fidelity",
    "pacing", "compression", "primary_focus", "audience", "required_elements",
    "forbidden_elements", "working_language", "total_duration_constraints",
    "per_episode_duration_constraints", "episode_count_constraints",
)


def _validate_value(field: str, value: Any) -> Any:
    if field in {"narrative_format", "presentation_orientation", "adaptation_genre", "pacing", "compression", "audience"}:
        return _text(value, field)
    if field == "fidelity":
        return _enum(value, FidelityProfile, field)
    if field == "working_language":
        if _text(value, field) != "zh-CN":
            raise AdaptationTargetError("working_language must be zh-CN; it is not spoken-dialogue language")
        return value
    if field == "primary_focus":
        if not isinstance(value, (tuple, list)):
            raise AdaptationTargetError("primary_focus must be an array")
        refs = tuple(value)
        if not refs or any(not isinstance(ref, (CanonicalTargetReference, UnresolvedTargetReference)) for ref in refs):
            raise AdaptationTargetError("primary_focus must contain target references")
        if len({content_hash(ref.to_dict()) for ref in refs}) != len(refs):
            raise AdaptationTargetError("primary_focus must not contain duplicate references")
        return refs
    if field in {"required_elements", "forbidden_elements"}:
        if not isinstance(value, (tuple, list)):
            raise AdaptationTargetError(f"{field} must be an array")
        items = tuple(value)
        if any(not isinstance(item, TargetElement) for item in items):
            raise AdaptationTargetError(f"{field} must contain TargetElement values")
        if len({content_hash(item.to_dict()) for item in items}) != len(items):
            raise AdaptationTargetError(f"{field} must not contain duplicate elements")
        return items
    if field.endswith("_constraints"):
        return _constraints(value, field.removesuffix("_constraints"))
    raise AssertionError(f"unhandled field: {field}")


def _serialize_value(field: str, value: Any) -> Any:
    if field == "fidelity":
        return value.value
    if field == "primary_focus":
        return [ref.to_dict() for ref in value]
    if field in {"required_elements", "forbidden_elements"}:
        return [item.to_dict() for item in value]
    if field.endswith("_constraints"):
        return [item.to_dict() for item in value]
    return value


def _deserialize_value(field: str, value: Any) -> Any:
    if field == "fidelity":
        return FidelityProfile(value)
    if field == "primary_focus":
        if not isinstance(value, list):
            raise AdaptationTargetError("primary_focus must be an array")
        return tuple(_reference_from_dict(item) for item in value)
    if field in {"required_elements", "forbidden_elements"}:
        if not isinstance(value, list):
            raise AdaptationTargetError(f"{field} must be an array")
        return tuple(TargetElement.from_dict(item) for item in value)
    if field.endswith("_constraints"):
        if not isinstance(value, list):
            raise AdaptationTargetError(f"{field} must be an array")
        return tuple(NumericConstraint.from_dict(item) for item in value)
    return value


class _TargetLayer:
    """Shared strict layer serializer; subclasses define their identity field."""

    _identity_field: str

    def _validate_fields(self) -> None:
        for field in _VALUE_FIELDS:
            item = getattr(self, field)
            if item is None:
                continue
            if not isinstance(item, TargetValue):
                raise AdaptationTargetError(f"{field} must be TargetValue or null")
            normalized = _validate_value(field, item.value)
            if normalized is not item.value:
                object.__setattr__(self, field, TargetValue(normalized, item.provenance))

    def _fields_to_dict(self) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for field in _VALUE_FIELDS:
            item = getattr(self, field)
            output[field] = None if item is None else {
                "value": _serialize_value(field, item.value), "provenance": item.provenance.value,
            }
        return output

    @classmethod
    def _fields_from_dict(cls, value: Mapping[str, Any]) -> dict[str, TargetValue | None]:
        parsed: dict[str, TargetValue | None] = {}
        for field in _VALUE_FIELDS:
            raw = value[field]
            if raw is None:
                parsed[field] = None
                continue
            raw = _exact(raw, {"value", "provenance"}, f"{cls.__name__}.{field}")
            parsed[field] = TargetValue(_deserialize_value(field, raw["value"]), raw["provenance"])
        return parsed


@dataclass(frozen=True, slots=True)
class AdaptationTarget(_TargetLayer):
    """Raw B1.1 intent.  It has no adaptation decisions beyond its stated fields."""

    adaptation_target_id: str
    narrative_format: TargetValue | None = None
    presentation_orientation: TargetValue | None = None
    adaptation_genre: TargetValue | None = None
    fidelity: TargetValue | None = None
    pacing: TargetValue | None = None
    compression: TargetValue | None = None
    primary_focus: TargetValue | None = None
    audience: TargetValue | None = None
    required_elements: TargetValue | None = None
    forbidden_elements: TargetValue | None = None
    working_language: TargetValue | None = None
    total_duration_constraints: TargetValue | None = None
    per_episode_duration_constraints: TargetValue | None = None
    episode_count_constraints: TargetValue | None = None

    def __post_init__(self) -> None:
        _id(self.adaptation_target_id, "adaptation_target_id")
        self._validate_fields()
        for field in _VALUE_FIELDS:
            item = getattr(self, field)
            if item is not None and item.provenance is not Provenance.USER:
                raise AdaptationTargetError("raw intent fields must have user provenance")

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": ADAPTATION_TARGET_SCHEMA_VERSION, "adaptation_target_id": self.adaptation_target_id, **self._fields_to_dict()}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "AdaptationTarget":
        value = _exact(value, {"schema_version", "adaptation_target_id", *_VALUE_FIELDS}, cls.__name__)
        if value["schema_version"] != ADAPTATION_TARGET_SCHEMA_VERSION:
            raise AdaptationTargetError("unsupported AdaptationTarget schema_version")
        return cls(value["adaptation_target_id"], **cls._fields_from_dict(value))


@dataclass(frozen=True, slots=True)
class AdaptationTargetDefaults(_TargetLayer):
    profile_id: str
    narrative_format: TargetValue | None = None
    presentation_orientation: TargetValue | None = None
    adaptation_genre: TargetValue | None = None
    fidelity: TargetValue | None = None
    pacing: TargetValue | None = None
    compression: TargetValue | None = None
    primary_focus: TargetValue | None = None
    audience: TargetValue | None = None
    required_elements: TargetValue | None = None
    forbidden_elements: TargetValue | None = None
    working_language: TargetValue | None = None
    total_duration_constraints: TargetValue | None = None
    per_episode_duration_constraints: TargetValue | None = None
    episode_count_constraints: TargetValue | None = None

    def __post_init__(self) -> None:
        _id(self.profile_id, "profile_id")
        self._validate_fields()
        for field in _VALUE_FIELDS:
            item = getattr(self, field)
            if item is not None and item.provenance is not Provenance.PROFILE_DEFAULT:
                raise AdaptationTargetError("profile default fields must have profile_default provenance")

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": ADAPTATION_TARGET_SCHEMA_VERSION, "profile_id": self.profile_id, **self._fields_to_dict()}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "AdaptationTargetDefaults":
        value = _exact(value, {"schema_version", "profile_id", *_VALUE_FIELDS}, cls.__name__)
        if value["schema_version"] != ADAPTATION_TARGET_SCHEMA_VERSION:
            raise AdaptationTargetError("unsupported AdaptationTargetDefaults schema_version")
        return cls(value["profile_id"], **cls._fields_from_dict(value))


@dataclass(frozen=True, slots=True)
class AdaptationTargetOverrides(_TargetLayer):
    override_id: str
    narrative_format: TargetValue | None = None
    presentation_orientation: TargetValue | None = None
    adaptation_genre: TargetValue | None = None
    fidelity: TargetValue | None = None
    pacing: TargetValue | None = None
    compression: TargetValue | None = None
    primary_focus: TargetValue | None = None
    audience: TargetValue | None = None
    required_elements: TargetValue | None = None
    forbidden_elements: TargetValue | None = None
    working_language: TargetValue | None = None
    total_duration_constraints: TargetValue | None = None
    per_episode_duration_constraints: TargetValue | None = None
    episode_count_constraints: TargetValue | None = None

    def __post_init__(self) -> None:
        _id(self.override_id, "override_id")
        self._validate_fields()
        for field in _VALUE_FIELDS:
            item = getattr(self, field)
            if item is not None and item.provenance is not Provenance.USER:
                raise AdaptationTargetError("explicit overrides require explicit user acceptance/provenance")

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": ADAPTATION_TARGET_SCHEMA_VERSION, "override_id": self.override_id, **self._fields_to_dict()}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "AdaptationTargetOverrides":
        value = _exact(value, {"schema_version", "override_id", *_VALUE_FIELDS}, cls.__name__)
        if value["schema_version"] != ADAPTATION_TARGET_SCHEMA_VERSION:
            raise AdaptationTargetError("unsupported AdaptationTargetOverrides schema_version")
        return cls(value["override_id"], **cls._fields_from_dict(value))


@dataclass(frozen=True, slots=True)
class ResolvedAdaptationTarget(_TargetLayer):
    """The one target shape consumed by later B1 stages; no defaults are re-inferred."""

    adaptation_target_id: str
    source_snapshot: AdaptationSourceSnapshot
    narrative_format: TargetValue | None = None
    presentation_orientation: TargetValue | None = None
    adaptation_genre: TargetValue | None = None
    fidelity: TargetValue | None = None
    pacing: TargetValue | None = None
    compression: TargetValue | None = None
    primary_focus: TargetValue | None = None
    audience: TargetValue | None = None
    required_elements: TargetValue | None = None
    forbidden_elements: TargetValue | None = None
    working_language: TargetValue | None = None
    total_duration_constraints: TargetValue | None = None
    per_episode_duration_constraints: TargetValue | None = None
    episode_count_constraints: TargetValue | None = None

    def __post_init__(self) -> None:
        _id(self.adaptation_target_id, "adaptation_target_id")
        if not isinstance(self.source_snapshot, AdaptationSourceSnapshot):
            raise AdaptationTargetError("source_snapshot must be an exact AdaptationSourceSnapshot")
        self._validate_fields()
        _validate_snapshot_membership(self)
        _validate_element_conflicts(self)
        _validate_hard_numeric_feasibility(self)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": ADAPTATION_TARGET_SCHEMA_VERSION,
            "adaptation_target_id": self.adaptation_target_id,
            "source_snapshot": self.source_snapshot.to_dict(),
            **self._fields_to_dict(),
        }

    def content_hash(self) -> str:
        return content_hash(self.to_dict())

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ResolvedAdaptationTarget":
        value = _exact(value, {"schema_version", "adaptation_target_id", "source_snapshot", *_VALUE_FIELDS}, cls.__name__)
        if value["schema_version"] != ADAPTATION_TARGET_SCHEMA_VERSION:
            raise AdaptationTargetError("unsupported ResolvedAdaptationTarget schema_version")
        return cls(value["adaptation_target_id"], AdaptationSourceSnapshot.from_dict(value["source_snapshot"]), **cls._fields_from_dict(value))


def _canonical_element_ids(value: TargetValue | None) -> set[tuple[str, str]]:
    if value is None:
        return set()
    return {
        (item.reference.kind, item.reference.canonical_id)
        for item in value.value
        if isinstance(item.reference, CanonicalTargetReference)
    }


def _validate_element_conflicts(target: ResolvedAdaptationTarget) -> None:
    overlap = _canonical_element_ids(target.required_elements) & _canonical_element_ids(target.forbidden_elements)
    if overlap:
        refs = ", ".join(f"{kind}:{identifier}" for kind, identifier in sorted(overlap))
        raise AdaptationTargetError(f"required_elements and forbidden_elements contradict: {refs}")


def _validate_snapshot_membership(target: ResolvedAdaptationTarget) -> None:
    for field in ("primary_focus", "required_elements", "forbidden_elements"):
        item = getattr(target, field)
        if item is None:
            continue
        values = item.value
        refs = values if field == "primary_focus" else tuple(element.reference for element in values)
        for ref in refs:
            if isinstance(ref, CanonicalTargetReference) and not target.source_snapshot.contains(ref):
                raise AdaptationTargetError(
                    f"{field} references {ref.kind}:{ref.canonical_id} outside the exact bound A5/A6 snapshot"
                )


def _combined_hard_interval(
    constraints: Iterable[NumericConstraint], *, lower_default: int, label: str,
) -> tuple[int | float, int | float]:
    lower, upper = lower_default, math.inf
    for constraint in constraints:
        interval = constraint.hard_interval(lower_default=lower_default)
        if interval is None:
            continue
        lower, upper = max(lower, interval[0]), min(upper, interval[1])
    if lower > upper:
        raise AdaptationTargetError(f"hard {label} constraints have no feasible value")
    return lower, upper


def _constraint_values(target: ResolvedAdaptationTarget, field: str) -> tuple[NumericConstraint, ...]:
    item = getattr(target, field)
    return () if item is None else item.value


def _validate_hard_numeric_feasibility(target: ResolvedAdaptationTarget) -> None:
    total = _combined_hard_interval(_constraint_values(target, "total_duration_constraints"), lower_default=1, label="total duration")
    per_episode = _combined_hard_interval(_constraint_values(target, "per_episode_duration_constraints"), lower_default=1, label="per-episode duration")
    episode_count = _combined_hard_interval(_constraint_values(target, "episode_count_constraints"), lower_default=1, label="episode count")
    _validate_independent_duration_count(total, per_episode, episode_count)


def _validate_independent_duration_count(
    total: tuple[int | float, int | float], per_episode: tuple[int | float, int | float],
    episode_count: tuple[int | float, int | float],
) -> None:
    """Fail closed only when independent integer episode durations cannot fit.

    For a permitted count ``n``, each episode may independently use any
    integer duration in the per-episode interval.  Its attainable totals are
    therefore the complete integer interval ``[n * d_min, n * d_max]``.  The
    derived count bounds below establish numeric feasibility without choosing
    the final count or allocating episode boundaries, both of which remain
    B1.5 responsibilities.
    """
    count_low, count_high = int(episode_count[0]), episode_count[1]
    total_low = int(total[0])
    total_high = math.inf if total[1] == math.inf else int(total[1])
    per_low = int(per_episode[0])
    per_high = math.inf if per_episode[1] == math.inf else int(per_episode[1])

    feasible_count_low = count_low
    if per_high != math.inf:
        feasible_count_low = max(feasible_count_low, _ceil_div(total_low, per_high))
    feasible_count_high = count_high
    if total_high != math.inf:
        feasible_count_high = min(feasible_count_high, total_high // per_low)
    if feasible_count_low > feasible_count_high:
        raise AdaptationTargetError("hard total/per-episode/count duration constraints are contradictory")


def _ceil_div(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


def resolve_adaptation_target(
    raw_intent: AdaptationTarget,
    profile_defaults: AdaptationTargetDefaults | None = None,
    explicit_overrides: AdaptationTargetOverrides | None = None,
    *,
    source_snapshot: AdaptationSourceSnapshot,
) -> ResolvedAdaptationTarget:
    """Resolve exactly ``raw > profile default`` and ``override > raw`` per field.

    The result is wholly typed and validates deterministic hard feasibility.
    No provider, source-truth mutation, episode partitioning, or approval
    operation is performed here.
    """
    if not isinstance(raw_intent, AdaptationTarget):
        raise AdaptationTargetError("raw_intent must be AdaptationTarget")
    if profile_defaults is not None and not isinstance(profile_defaults, AdaptationTargetDefaults):
        raise AdaptationTargetError("profile_defaults must be AdaptationTargetDefaults or null")
    if explicit_overrides is not None and not isinstance(explicit_overrides, AdaptationTargetOverrides):
        raise AdaptationTargetError("explicit_overrides must be AdaptationTargetOverrides or null")
    resolved: dict[str, TargetValue | None] = {}
    for field in _VALUE_FIELDS:
        # This assignment order is the contract, rather than input order or any
        # implicit 'last text wins' behavior.
        value = None if profile_defaults is None else getattr(profile_defaults, field)
        if getattr(raw_intent, field) is not None:
            value = getattr(raw_intent, field)
        if explicit_overrides is not None and getattr(explicit_overrides, field) is not None:
            value = getattr(explicit_overrides, field)
        resolved[field] = value
    if not isinstance(source_snapshot, AdaptationSourceSnapshot):
        raise AdaptationTargetError("source_snapshot must be an exact AdaptationSourceSnapshot")
    return ResolvedAdaptationTarget(raw_intent.adaptation_target_id, source_snapshot, **resolved)


@dataclass(frozen=True, slots=True)
class StableIdIndex:
    """Deterministic exact/alias stable-ID resolution for B1.1 input parsing.

    Ambiguous or unknown text deliberately returns ``UnresolvedTargetReference``;
    no fuzzy match or model guess is permitted.
    """

    references: tuple[CanonicalTargetReference, ...]
    aliases: tuple[tuple[str, CanonicalTargetReference], ...] = ()

    def __post_init__(self) -> None:
        refs = tuple(self.references)
        aliases = tuple(self.aliases)
        if any(not isinstance(ref, CanonicalTargetReference) for ref in refs):
            raise AdaptationTargetError("StableIdIndex.references must contain canonical references")
        if len({(ref.kind, ref.canonical_id) for ref in refs}) != len(refs):
            raise AdaptationTargetError("StableIdIndex.references must not contain duplicates")
        seen: dict[str, CanonicalTargetReference] = {}
        valid = {(ref.kind, ref.canonical_id) for ref in refs}
        for alias, ref in aliases:
            key = _normalize_lookup(alias)
            if not isinstance(ref, CanonicalTargetReference) or (ref.kind, ref.canonical_id) not in valid:
                raise AdaptationTargetError("StableIdIndex aliases must point to listed canonical references")
            if key in seen and seen[key] != ref:
                raise AdaptationTargetError("StableIdIndex aliases cannot ambiguously map to multiple IDs")
            seen[key] = ref
        object.__setattr__(self, "references", refs)
        object.__setattr__(self, "aliases", aliases)

    def resolve(self, text: str) -> TargetReference:
        text = _text(text, "reference text")
        direct = [ref for ref in self.references if ref.canonical_id == text]
        if len(direct) == 1:
            return direct[0]
        lookup = _normalize_lookup(text)
        aliases = [ref for alias, ref in self.aliases if _normalize_lookup(alias) == lookup]
        return aliases[0] if len(aliases) == 1 else UnresolvedTargetReference(text)


def _normalize_lookup(value: str) -> str:
    return " ".join(_text(value, "alias").casefold().split())
