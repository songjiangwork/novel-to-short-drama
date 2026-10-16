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

from short_drama.artifacts import content_hash


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
        if field == "episode_count":
            for value in (self.value, self.minimum, self.maximum):
                if value is not None:
                    _number(value, "episode_count constraint", integer=True)

    def hard_interval(self, *, lower_default: float) -> tuple[float, float] | None:
        """Return the feasibility interval for hard fixed/range constraints only.

        A target remains an optimization target even when it is hard, so it
        does not silently become a fixed value.  Derived values cannot be
        evaluated at B1.1 without the later inputs named by their derivation.
        """
        if self.strength is not ConstraintStrength.HARD:
            return None
        if self.kind is ConstraintKind.FIXED:
            return (float(self.value), float(self.value))
        if self.kind is ConstraintKind.RANGE:
            return (
                lower_default if self.minimum is None else float(self.minimum),
                math.inf if self.maximum is None else float(self.maximum),
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
            if item is not None and item.provenance not in {Provenance.USER, Provenance.MODEL_PROPOSED}:
                raise AdaptationTargetError("explicit overrides must retain user or model_proposed provenance")

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
        _validate_element_conflicts(self)
        _validate_hard_numeric_feasibility(self)

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": ADAPTATION_TARGET_SCHEMA_VERSION, "adaptation_target_id": self.adaptation_target_id, **self._fields_to_dict()}

    def content_hash(self) -> str:
        return content_hash(self.to_dict())

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ResolvedAdaptationTarget":
        value = _exact(value, {"schema_version", "adaptation_target_id", *_VALUE_FIELDS}, cls.__name__)
        if value["schema_version"] != ADAPTATION_TARGET_SCHEMA_VERSION:
            raise AdaptationTargetError("unsupported ResolvedAdaptationTarget schema_version")
        return cls(value["adaptation_target_id"], **cls._fields_from_dict(value))


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


def _combined_hard_interval(constraints: Iterable[NumericConstraint], *, lower_default: float, label: str) -> tuple[float, float]:
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
    total = _combined_hard_interval(_constraint_values(target, "total_duration_constraints"), lower_default=0.0, label="total duration")
    per_episode = _combined_hard_interval(_constraint_values(target, "per_episode_duration_constraints"), lower_default=0.0, label="per-episode duration")
    episode_count = _combined_hard_interval(_constraint_values(target, "episode_count_constraints"), lower_default=1.0, label="episode count")
    # A positive duration is fundamental even if it has no explicit range.
    per_lower = max(per_episode[0], 0.0)
    product_lower, product_upper = per_lower * episode_count[0], per_episode[1] * episode_count[1]
    if product_upper < total[0] or product_lower > total[1]:
        raise AdaptationTargetError(
            "hard total/per-episode/count duration constraints are contradictory"
        )


def resolve_adaptation_target(
    raw_intent: AdaptationTarget,
    profile_defaults: AdaptationTargetDefaults | None = None,
    explicit_overrides: AdaptationTargetOverrides | None = None,
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
    return ResolvedAdaptationTarget(raw_intent.adaptation_target_id, **resolved)


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
