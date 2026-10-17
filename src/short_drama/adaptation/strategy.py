"""Frozen B1.2-A adaptation-strategy contracts.

This module contains no semantic generator and no event dispositions.  It is a
small, strict output contract for the future B1.2 provider pass.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable

from short_drama.artifacts import ArtifactRef, content_hash
from .target import AdaptationSourceSnapshot, CanonicalTargetReference, ResolvedAdaptationTarget


ADAPTATION_STRATEGY_SCHEMA_VERSION = 1


class AdaptationStrategyError(ValueError):
    """A B1.2 strategy is structurally invalid or is outside its source universe."""


class PlotlineDisposition(str, Enum):
    PRESERVE = "preserve"
    PRESERVE_CORE = "preserve_core"
    COMPRESS = "compress"
    MERGE_INTO = "merge_into"
    OMIT = "omit"


class CharacterDisposition(str, Enum):
    RETAIN = "retain"
    REDUCE = "reduce"
    OMIT = "omit"
    COMPOSITE = "composite"
    RETAIN_FUNCTION = "retain_function"


class RelationshipDisposition(str, Enum):
    PRESERVE = "preserve"
    PRESERVE_CORE = "preserve_core"
    COMPRESS = "compress"
    SIMPLIFY = "simplify"
    OMIT = "omit"


class PreservationPriority(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class RevealDisposition(str, Enum):
    PRESERVE = "preserve"
    ADVANCE = "advance"
    DELAY = "delay"
    MERGE = "merge"
    OMIT = "omit"
    REPLACE_MECHANISM = "replace_mechanism"


class EndingDisposition(str, Enum):
    SOURCE_ENDING_RETAINED = "source_ending_retained"
    SOURCE_ENDING_COMPRESSED = "source_ending_compressed"
    SOURCE_ENDING_REFRAMED = "source_ending_reframed"
    ADAPTED_ENDING = "adapted_ending"


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\0" in value:
        raise AdaptationStrategyError(f"{field} must be non-empty text without NUL")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise AdaptationStrategyError(f"{field} must be UTF-8 text") from exc
    return value


def _enum(value: Any, kind: type[Enum], field: str) -> Enum:
    try:
        return value if isinstance(value, kind) else kind(value)
    except (TypeError, ValueError) as exc:
        raise AdaptationStrategyError(f"{field} is not a permitted value") from exc


def _exact(value: Any, fields: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise AdaptationStrategyError(f"{name} must contain exactly: {', '.join(sorted(fields))}")
    return value


def _refs(value: Iterable[CanonicalTargetReference], field: str) -> tuple[CanonicalTargetReference, ...]:
    refs = tuple(value)
    if not refs or any(not isinstance(item, CanonicalTargetReference) for item in refs):
        raise AdaptationStrategyError(f"{field} must contain canonical source references")
    if len(set(refs)) != len(refs):
        raise AdaptationStrategyError(f"{field} must not contain duplicate source references")
    return tuple(sorted(refs, key=lambda item: (item.kind, item.canonical_id)))


@dataclass(frozen=True, slots=True)
class SnapshotLocalPlotline:
    """A B1.2-local handle for one exact immutable A6 main/subplot entry.

    It is deliberately not an A-layer canonical entity.  ``a6_manifest_ref``
    binds source-order-derived handles to their exact snapshot.
    """

    handle: str
    a6_manifest_hash: str
    ordinal: int
    role: str

    def __post_init__(self) -> None:
        _text(self.handle, "plotline handle")
        if not self.handle.startswith("a6pl_"):
            raise AdaptationStrategyError("plotline handle must be snapshot-local a6pl_ handle")
        if not isinstance(self.a6_manifest_hash, str) or len(self.a6_manifest_hash) != 64:
            raise AdaptationStrategyError("a6_manifest_hash must be a SHA-256 hash")
        if isinstance(self.ordinal, bool) or not isinstance(self.ordinal, int) or self.ordinal < 0:
            raise AdaptationStrategyError("plotline ordinal must be >= 0")
        if self.role not in {"main", "subplot"}:
            raise AdaptationStrategyError("plotline role must be main or subplot")

    def to_dict(self) -> dict[str, Any]:
        return {"handle": self.handle, "a6_manifest_hash": self.a6_manifest_hash, "ordinal": self.ordinal, "role": self.role}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SnapshotLocalPlotline":
        return cls(**_exact(value, {"handle", "a6_manifest_hash", "ordinal", "role"}, cls.__name__))


@dataclass(frozen=True, slots=True)
class PlotlineStrategy:
    plotline_handle: str
    disposition: PlotlineDisposition
    rationale_zh: str
    merge_into_handle: str | None = None

    def __post_init__(self) -> None:
        _text(self.plotline_handle, "plotline_handle")
        object.__setattr__(self, "disposition", _enum(self.disposition, PlotlineDisposition, "plotline disposition"))
        _text(self.rationale_zh, "plotline rationale")
        if self.disposition is PlotlineDisposition.MERGE_INTO:
            _text(self.merge_into_handle, "merge_into_handle")
        elif self.merge_into_handle is not None:
            raise AdaptationStrategyError("merge_into_handle is only permitted for merge_into")

    def to_dict(self) -> dict[str, Any]:
        return {"plotline_handle": self.plotline_handle, "disposition": self.disposition.value, "rationale_zh": self.rationale_zh, "merge_into_handle": self.merge_into_handle}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "PlotlineStrategy":
        return cls(**_exact(value, {"plotline_handle", "disposition", "rationale_zh", "merge_into_handle"}, cls.__name__))


@dataclass(frozen=True, slots=True)
class SourceStrategy:
    source_ref: CanonicalTargetReference
    disposition: Enum
    rationale_zh: str

    def __post_init__(self) -> None:
        if not isinstance(self.source_ref, CanonicalTargetReference):
            raise AdaptationStrategyError("source_ref must be a canonical source reference")
        _text(self.rationale_zh, "strategy rationale")

    def to_dict(self) -> dict[str, Any]:
        return {"source_ref": self.source_ref.to_dict(), "disposition": self.disposition.value, "rationale_zh": self.rationale_zh}


@dataclass(frozen=True, slots=True)
class CharacterStrategy(SourceStrategy):
    disposition: CharacterDisposition

    def __post_init__(self) -> None:
        super(CharacterStrategy, self).__post_init__()
        if self.source_ref.kind != "character":
            raise AdaptationStrategyError("CharacterStrategy requires a character reference")
        object.__setattr__(self, "disposition", _enum(self.disposition, CharacterDisposition, "character disposition"))

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CharacterStrategy":
        value = _exact(value, {"source_ref", "disposition", "rationale_zh"}, cls.__name__)
        return cls(CanonicalTargetReference.from_dict(value["source_ref"]), value["disposition"], value["rationale_zh"])


@dataclass(frozen=True, slots=True)
class RelationshipStrategy(SourceStrategy):
    disposition: RelationshipDisposition

    def __post_init__(self) -> None:
        super(RelationshipStrategy, self).__post_init__()
        if self.source_ref.kind != "relationship":
            raise AdaptationStrategyError("RelationshipStrategy requires a relationship reference")
        object.__setattr__(self, "disposition", _enum(self.disposition, RelationshipDisposition, "relationship disposition"))

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RelationshipStrategy":
        value = _exact(value, {"source_ref", "disposition", "rationale_zh"}, cls.__name__)
        return cls(CanonicalTargetReference.from_dict(value["source_ref"]), value["disposition"], value["rationale_zh"])


@dataclass(frozen=True, slots=True)
class RevealStrategy(SourceStrategy):
    disposition: RevealDisposition

    def __post_init__(self) -> None:
        super(RevealStrategy, self).__post_init__()
        if self.source_ref.kind != "reveal":
            raise AdaptationStrategyError("RevealStrategy requires a reveal reference")
        object.__setattr__(self, "disposition", _enum(self.disposition, RevealDisposition, "reveal disposition"))

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RevealStrategy":
        value = _exact(value, {"source_ref", "disposition", "rationale_zh"}, cls.__name__)
        return cls(CanonicalTargetReference.from_dict(value["source_ref"]), value["disposition"], value["rationale_zh"])


@dataclass(frozen=True, slots=True)
class PriorityEntry:
    source_ref: CanonicalTargetReference
    priority: PreservationPriority

    def __post_init__(self) -> None:
        if not isinstance(self.source_ref, CanonicalTargetReference):
            raise AdaptationStrategyError("priority source_ref must be canonical")
        object.__setattr__(self, "priority", _enum(self.priority, PreservationPriority, "preservation priority"))

    def to_dict(self) -> dict[str, Any]: return {"source_ref": self.source_ref.to_dict(), "priority": self.priority.value}
    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "PriorityEntry":
        value = _exact(value, {"source_ref", "priority"}, cls.__name__)
        return cls(CanonicalTargetReference.from_dict(value["source_ref"]), value["priority"])


@dataclass(frozen=True, slots=True)
class DeliberateDivergence:
    description_zh: str
    source_refs: tuple[CanonicalTargetReference, ...]
    provenance: str

    def __post_init__(self) -> None:
        _text(self.description_zh, "divergence description")
        object.__setattr__(self, "source_refs", _refs(self.source_refs, "divergence source_refs"))
        _text(self.provenance, "divergence provenance")

    def to_dict(self) -> dict[str, Any]: return {"description_zh": self.description_zh, "source_refs": [x.to_dict() for x in self.source_refs], "provenance": self.provenance}
    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "DeliberateDivergence":
        value = _exact(value, {"description_zh", "source_refs", "provenance"}, cls.__name__)
        return cls(value["description_zh"], tuple(CanonicalTargetReference.from_dict(x) for x in value["source_refs"]), value["provenance"])


@dataclass(frozen=True, slots=True)
class StrategyConflict:
    description_zh: str
    source_refs: tuple[CanonicalTargetReference, ...]
    user_direction_refs: tuple[ArtifactRef, ...] = ()

    def __post_init__(self) -> None:
        _text(self.description_zh, "conflict description")
        object.__setattr__(self, "source_refs", _refs(self.source_refs, "conflict source_refs"))
        refs = tuple(self.user_direction_refs)
        if any(not isinstance(x, ArtifactRef) or x.artifact_type != "user_direction" for x in refs) or len(set(refs)) != len(refs):
            raise AdaptationStrategyError("user_direction_refs must be unique exact user_direction ArtifactRefs")
        object.__setattr__(self, "user_direction_refs", refs)

    def to_dict(self) -> dict[str, Any]: return {"description_zh": self.description_zh, "source_refs": [x.to_dict() for x in self.source_refs], "user_direction_refs": [x.to_dict() for x in self.user_direction_refs]}
    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "StrategyConflict":
        value = _exact(value, {"description_zh", "source_refs", "user_direction_refs"}, cls.__name__)
        return cls(value["description_zh"], tuple(CanonicalTargetReference.from_dict(x) for x in value["source_refs"]), tuple(ArtifactRef.from_dict(x) for x in value["user_direction_refs"]))


@dataclass(frozen=True, slots=True)
class AdaptationStrategy:
    """Immutable B1.2 semantic output; event-level disposition remains B1.3."""
    resolved_target: ResolvedAdaptationTarget
    adaptation_thesis_zh: str
    dramatic_promise_zh: str
    primary_dramatic_question_zh: str
    plotline_strategies: tuple[PlotlineStrategy, ...]
    character_strategies: tuple[CharacterStrategy, ...]
    relationship_strategies: tuple[RelationshipStrategy, ...]
    reveal_strategies: tuple[RevealStrategy, ...]
    series_climax_target_zh: str
    series_climax_source_refs: tuple[CanonicalTargetReference, ...]
    ending_disposition: EndingDisposition
    ending_strategy_zh: str
    preservation_priorities: tuple[PriorityEntry, ...]
    deliberate_divergences: tuple[DeliberateDivergence, ...] = ()
    explicit_conflicts: tuple[StrategyConflict, ...] = ()
    schema_version: int = ADAPTATION_STRATEGY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ADAPTATION_STRATEGY_SCHEMA_VERSION: raise AdaptationStrategyError("unsupported strategy schema_version")
        if not isinstance(self.resolved_target, ResolvedAdaptationTarget): raise AdaptationStrategyError("resolved_target must be exact ResolvedAdaptationTarget")
        for name in ("adaptation_thesis_zh", "dramatic_promise_zh", "primary_dramatic_question_zh", "series_climax_target_zh", "ending_strategy_zh"): _text(getattr(self, name), name)
        object.__setattr__(self, "ending_disposition", _enum(self.ending_disposition, EndingDisposition, "ending disposition"))
        object.__setattr__(self, "series_climax_source_refs", _refs(self.series_climax_source_refs, "series climax source refs"))
        object.__setattr__(self, "plotline_strategies", tuple(sorted(self.plotline_strategies, key=lambda x: x.plotline_handle)))
        object.__setattr__(self, "character_strategies", tuple(sorted(self.character_strategies, key=lambda x: (x.source_ref.kind, x.source_ref.canonical_id))))
        object.__setattr__(self, "relationship_strategies", tuple(sorted(self.relationship_strategies, key=lambda x: (x.source_ref.kind, x.source_ref.canonical_id))))
        object.__setattr__(self, "reveal_strategies", tuple(sorted(self.reveal_strategies, key=lambda x: (x.source_ref.kind, x.source_ref.canonical_id))))
        object.__setattr__(self, "preservation_priorities", tuple(sorted(self.preservation_priorities, key=lambda x: (x.source_ref.kind, x.source_ref.canonical_id))))
        object.__setattr__(self, "deliberate_divergences", tuple(sorted(self.deliberate_divergences, key=lambda x: content_hash(x.to_dict()))))
        object.__setattr__(self, "explicit_conflicts", tuple(sorted(self.explicit_conflicts, key=lambda x: content_hash(x.to_dict()))))
        self._unique("plotline_strategies", self.plotline_strategies, lambda x: x.plotline_handle, PlotlineStrategy)
        self._unique("character_strategies", self.character_strategies, lambda x: x.source_ref, CharacterStrategy)
        self._unique("relationship_strategies", self.relationship_strategies, lambda x: x.source_ref, RelationshipStrategy)
        self._unique("reveal_strategies", self.reveal_strategies, lambda x: x.source_ref, RevealStrategy)
        self._unique("preservation_priorities", self.preservation_priorities, lambda x: x.source_ref, PriorityEntry)
        if any(not isinstance(x, DeliberateDivergence) for x in self.deliberate_divergences) or any(not isinstance(x, StrategyConflict) for x in self.explicit_conflicts): raise AdaptationStrategyError("invalid divergence/conflict records")
        self.validate_source_membership()

    @staticmethod
    def _unique(name: str, values: Iterable[Any], key: Any, expected: type) -> None:
        values = tuple(values)
        if any(not isinstance(x, expected) for x in values) or len({key(x) for x in values}) != len(values): raise AdaptationStrategyError(f"{name} must have unique typed members")

    def validate_source_membership(self, plotlines: Iterable[SnapshotLocalPlotline] | None = None) -> None:
        snapshot = self.resolved_target.source_snapshot
        refs = list(self.series_climax_source_refs)
        for group in (self.character_strategies, self.relationship_strategies, self.reveal_strategies, self.preservation_priorities): refs.extend(item.source_ref for item in group)
        for item in (*self.deliberate_divergences, *self.explicit_conflicts): refs.extend(item.source_refs)
        if any(not snapshot.contains(ref) for ref in refs): raise AdaptationStrategyError("strategy references source ID absent from exact target snapshot")
        if plotlines is not None:
            plotlines = tuple(plotlines)
            if any(not isinstance(x, SnapshotLocalPlotline) or x.a6_manifest_hash != snapshot.story_analysis_manifest_ref.content_hash for x in plotlines):
                raise AdaptationStrategyError("plotline handles must be bound to the exact target A6 snapshot")
            if len({x.handle for x in plotlines}) != len(plotlines):
                raise AdaptationStrategyError("snapshot-local plotline handles must be unique")
            allowed = {x.handle for x in plotlines}
            if {x.plotline_handle for x in self.plotline_strategies} != allowed: raise AdaptationStrategyError("strategy must cover exactly the snapshot-local plotline universe")
            by_handle = {x.plotline_handle: x for x in self.plotline_strategies}
            for item in self.plotline_strategies:
                if item.disposition is PlotlineDisposition.MERGE_INTO:
                    if item.merge_into_handle not in allowed or item.merge_into_handle == item.plotline_handle: raise AdaptationStrategyError("merge_into must name another existing snapshot-local plotline")
                    if by_handle[item.merge_into_handle].disposition in {PlotlineDisposition.MERGE_INTO, PlotlineDisposition.OMIT}:
                        raise AdaptationStrategyError("merge_into target must be an eligible retained plotline")

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": self.schema_version, "resolved_target": self.resolved_target.to_dict(), "adaptation_thesis_zh": self.adaptation_thesis_zh, "dramatic_promise_zh": self.dramatic_promise_zh, "primary_dramatic_question_zh": self.primary_dramatic_question_zh, "plotline_strategies": [x.to_dict() for x in self.plotline_strategies], "character_strategies": [x.to_dict() for x in self.character_strategies], "relationship_strategies": [x.to_dict() for x in self.relationship_strategies], "reveal_strategies": [x.to_dict() for x in self.reveal_strategies], "series_climax_target_zh": self.series_climax_target_zh, "series_climax_source_refs": [x.to_dict() for x in self.series_climax_source_refs], "ending_disposition": self.ending_disposition.value, "ending_strategy_zh": self.ending_strategy_zh, "preservation_priorities": [x.to_dict() for x in self.preservation_priorities], "deliberate_divergences": [x.to_dict() for x in self.deliberate_divergences], "explicit_conflicts": [x.to_dict() for x in self.explicit_conflicts]}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "AdaptationStrategy":
        fields = {"schema_version", "resolved_target", "adaptation_thesis_zh", "dramatic_promise_zh", "primary_dramatic_question_zh", "plotline_strategies", "character_strategies", "relationship_strategies", "reveal_strategies", "series_climax_target_zh", "series_climax_source_refs", "ending_disposition", "ending_strategy_zh", "preservation_priorities", "deliberate_divergences", "explicit_conflicts"}
        value = _exact(value, fields, cls.__name__)
        return cls(value["resolved_target"] if isinstance(value["resolved_target"], ResolvedAdaptationTarget) else ResolvedAdaptationTarget.from_dict(value["resolved_target"]), value["adaptation_thesis_zh"], value["dramatic_promise_zh"], value["primary_dramatic_question_zh"], tuple(PlotlineStrategy.from_dict(x) for x in value["plotline_strategies"]), tuple(CharacterStrategy.from_dict(x) for x in value["character_strategies"]), tuple(RelationshipStrategy.from_dict(x) for x in value["relationship_strategies"]), tuple(RevealStrategy.from_dict(x) for x in value["reveal_strategies"]), value["series_climax_target_zh"], tuple(CanonicalTargetReference.from_dict(x) for x in value["series_climax_source_refs"]), value["ending_disposition"], value["ending_strategy_zh"], tuple(PriorityEntry.from_dict(x) for x in value["preservation_priorities"]), tuple(DeliberateDivergence.from_dict(x) for x in value["deliberate_divergences"]), tuple(StrategyConflict.from_dict(x) for x in value["explicit_conflicts"]), value["schema_version"])

    def content_hash(self) -> str: return content_hash(self.to_dict())
