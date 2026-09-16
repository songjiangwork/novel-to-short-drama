"""v1.2 A3A — chunk-extraction candidate domain contracts + profile.

This module is the static/domain-contract layer for A3 (chunk extraction). It
defines the typed, deterministic models that the later A3B-A3E slices depend
on WITHOUT changing their public contract:

    StoryExtractionProfile
    EvidenceRef
    CharacterCandidate / LocationCandidate / FactCandidate
    EventCandidate / RelationshipCandidate / UnresolvedMentionCandidate
    CandidatePayload
    CandidateExtraction (typed persisted shape)

A3A is deliberately a pure domain/static-contract slice. It does NOT:

  * call an LLM or build a semantic request (A3D);
  * validate source/ownership/excerpt semantics (A3B);
  * validate cross-reference graph closure, ID gaps/duplicates, or canonical
    ordering (A3B);
  * persist artifacts, write CURRENT pointers, or reuse (A3C);
  * provide a stage CLI or a real-Qwen smoke (A3D/A3E).

It reuses the existing authorities rather than inventing a second one:

  * Foundation ``ArtifactRef`` for ``source_document_ref`` / ``source_chunk_ref``;
  * A-I3 ``LLMInvocationProvenance`` for ``generation_provenance``;
  * the canonical ``content_hash`` for the extraction profile hash.

Style follows the rest of the story package: frozen dataclasses, explicit
``to_dict()`` / ``from_dict()``, exact-key / fail-closed parsing, and
deterministic serialization shape. No Pydantic, no database/ORM.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from short_drama.artifacts import ArtifactRef, content_hash
from short_drama.io import load_yaml
from short_drama.llm import LLMConfigError, LLMInvocationProvenance

from .errors import ExtractionModelError

# ---------------------------------------------------------------------------
# Schema / artifact identity constants
# ---------------------------------------------------------------------------

STORY_EXTRACTION_PROFILE_SCHEMA_VERSION = 1
# Frozen A-I4 v1 ceiling: first generation + at most one semantic-regeneration
# round. Combined with A-I3's bounded technical attempts (max 3 per
# generate_structured call) this keeps total provider attempts bounded
# (2 * 3 = 6). Enforced at the static-contract layer, not by retry logic.
STORY_EXTRACTION_MAX_GENERATION_ROUNDS_V1 = 2
CANDIDATE_EXTRACTION_SCHEMA_VERSION = 1
CANDIDATE_EXTRACTION_ARTIFACT_TYPE = "candidate_extraction"

# ---------------------------------------------------------------------------
# Frozen value domains (kept in lockstep with the parent A-I4 contract)
# ---------------------------------------------------------------------------

EVIDENCE_ROLES = frozenset({"primary", "supporting"})
EVIDENCE_STRENGTHS = frozenset({"explicit", "implied", "uncertain"})
FACT_TYPES = frozenset(
    {
        "identity",
        "appearance",
        "possession",
        "knowledge",
        "relationship_state",
        "location_state",
        "world_fact",
        "continuity_relevant",
        "other",
    }
)
TEMPORAL_MODES = frozenset(
    {"normal", "flashback", "flashforward", "dream", "memory", "unknown"}
)
RELATIONSHIP_DIRECTIONS = frozenset({"directed", "symmetric", "unknown"})
MENTION_KINDS = frozenset({"person", "location", "other", "unknown"})

# Chunk-local candidate ID namespaces. A3A enforces only the structural shape;
# per-category uniqueness, continuous numbering, and gap detection are A3B.
_CANDIDATE_ID_PATTERNS = {
    "character": re.compile(r"^cand_char_[0-9]{3,}$"),
    "location": re.compile(r"^cand_loc_[0-9]{3,}$"),
    "fact": re.compile(r"^cand_fact_[0-9]{3,}$"),
    "event": re.compile(r"^cand_evt_[0-9]{3,}$"),
    "relationship": re.compile(r"^cand_rel_[0-9]{3,}$"),
    "unresolved_mention": re.compile(r"^cand_unres_[0-9]{3,}$"),
}

_STORAGE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

# Exact field set of the A-I3 LLMInvocationProvenance (reused, not redefined).
_LLM_PROVENANCE_KEYS = frozenset(
    {
        "provider_family",
        "model",
        "semantic_profile_id",
        "semantic_profile_hash",
        "prompt_id",
        "prompt_version",
        "prompt_content_hash",
        "rendered_prompt_hash",
        "output_schema_id",
        "output_schema_version",
        "output_schema_hash",
        "request_hash",
        "provider_response_id",
        "finish_reason",
        "usage",
    }
)

# Exact top-level field set of the persisted CandidateExtraction shape.
_CANDIDATE_EXTRACTION_KEYS = frozenset(
    {
        "schema_version",
        "project_id",
        "document_id",
        "chunk_profile_id",
        "chunk_id",
        "source_document_ref",
        "source_chunk_ref",
        "extraction_profile_id",
        "extraction_profile_hash",
        "generation_provenance",
        "candidates",
    }
)


# ---------------------------------------------------------------------------
# Structural validation helpers
# ---------------------------------------------------------------------------


def _require_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ExtractionModelError(
            f"{field_name} must be a non-empty string without NUL"
        )
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ExtractionModelError(
            f"{field_name} must contain valid UTF-8 text"
        ) from exc
    return value


def _require_optional_text(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    return _require_text(value, field_name)


def _require_storage_id(value: Any, field_name: str) -> str:
    _require_text(value, field_name)
    if _STORAGE_ID_RE.fullmatch(value) is None:
        raise ExtractionModelError(
            f"{field_name} must be a safe lowercase storage identifier"
        )
    return value


def _require_hash(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ExtractionModelError(
            f"{field_name} must be a lowercase 64-char SHA-256 hex digest"
        )
    return value


def _require_positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ExtractionModelError(f"{field_name} must be an integer >= 1")
    return value


def _require_exact_keys(value: Any, keys: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise ExtractionModelError(
            f"{name} must contain exactly: {', '.join(sorted(keys))}"
        )
    return value


def _require_enum(value: Any, allowed: frozenset[str], field_name: str) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise ExtractionModelError(
            f"{field_name} must be one of: {', '.join(sorted(allowed))}"
        )
    return value


def _require_candidate_id(value: Any, category: str, field_name: str) -> str:
    _require_text(value, field_name)
    pattern = _CANDIDATE_ID_PATTERNS[category]
    if pattern.fullmatch(value) is None:
        raise ExtractionModelError(
            f"{field_name} does not match the {category} namespace "
            f"({pattern.pattern!r}): {value!r}"
        )
    return value


def _to_string_tuple(
    value: Any,
    field_name: str,
    *,
    min_items: int = 0,
    unique: bool = True,
) -> tuple[str, ...]:
    """Normalize a string collection to a validated tuple (fail closed)."""

    if isinstance(value, str):
        raise ExtractionModelError(f"{field_name} must be a list of strings")
    try:
        items = tuple(value)
    except TypeError as exc:
        raise ExtractionModelError(f"{field_name} must be a list of strings") from exc
    if len(items) < min_items:
        raise ExtractionModelError(
            f"{field_name} must contain at least {min_items} item(s)"
        )
    for item in items:
        _require_text(item, field_name)
    if unique and len(items) != len(set(items)):
        raise ExtractionModelError(f"{field_name} must not contain duplicates")
    return items


def _to_evidence_tuple(
    value: Any, field_name: str
) -> tuple["EvidenceRef", ...]:
    """Normalize an EvidenceRef collection to a validated non-empty tuple."""

    if isinstance(value, EvidenceRef):
        raise ExtractionModelError(f"{field_name} must be a list of EvidenceRef")
    try:
        items = tuple(value)
    except TypeError as exc:
        raise ExtractionModelError(f"{field_name} must be a list of EvidenceRef") from exc
    if not items:
        raise ExtractionModelError(
            f"{field_name} must contain at least one EvidenceRef"
        )
    for item in items:
        if not isinstance(item, EvidenceRef):
            raise ExtractionModelError(
                f"{field_name} must contain EvidenceRef values"
            )
    return items


def _to_candidate_tuple(
    value: Any, field_name: str, expected_type: type
) -> tuple[Any, ...]:
    """Normalize a candidate collection to a validated tuple (may be empty)."""

    if isinstance(value, expected_type):
        raise ExtractionModelError(
            f"CandidatePayload.{field_name} must be a list of candidates"
        )
    try:
        items = tuple(value)
    except TypeError as exc:
        raise ExtractionModelError(
            f"CandidatePayload.{field_name} must be a list of candidates"
        ) from exc
    for item in items:
        if not isinstance(item, expected_type):
            raise ExtractionModelError(
                f"CandidatePayload.{field_name} must contain "
                f"{expected_type.__name__} values"
            )
    return items


# ---------------------------------------------------------------------------
# EvidenceRef
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EvidenceRef:
    """Paragraph-granularity source evidence reference.

    A3A validates only structure and field type/domain. Whether the paragraph
    actually exists, belongs to the chunk's ownership/context, the excerpt is
    an exact source substring, and the candidate has ownership-primary
    evidence are all A3B semantic checks, intentionally NOT done here.
    """

    paragraph_id: str
    role: str
    strength: str
    excerpt: str | None

    def __post_init__(self) -> None:
        _require_text(self.paragraph_id, "EvidenceRef.paragraph_id")
        _require_enum(self.role, EVIDENCE_ROLES, "EvidenceRef.role")
        _require_enum(self.strength, EVIDENCE_STRENGTHS, "EvidenceRef.strength")
        _require_optional_text(self.excerpt, "EvidenceRef.excerpt")

    def to_dict(self) -> dict[str, Any]:
        return {
            "paragraph_id": self.paragraph_id,
            "role": self.role,
            "strength": self.strength,
            "excerpt": self.excerpt,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "EvidenceRef":
        _require_exact_keys(
            value, {"paragraph_id", "role", "strength", "excerpt"}, "EvidenceRef"
        )
        return cls(
            paragraph_id=value["paragraph_id"],
            role=value["role"],
            strength=value["strength"],
            excerpt=value["excerpt"],
        )


# ---------------------------------------------------------------------------
# Six local candidate models
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CharacterCandidate:
    candidate_id: str
    display_name_original: str
    aliases_original: tuple[str, ...]
    descriptors_zh: tuple[str, ...]
    summary_zh: str
    evidence_strength: str
    evidence: tuple[EvidenceRef, ...]

    def __post_init__(self) -> None:
        _require_candidate_id(
            self.candidate_id, "character", "CharacterCandidate.candidate_id"
        )
        _require_text(
            self.display_name_original,
            "CharacterCandidate.display_name_original",
        )
        object.__setattr__(
            self,
            "aliases_original",
            _to_string_tuple(
                self.aliases_original, "CharacterCandidate.aliases_original"
            ),
        )
        object.__setattr__(
            self,
            "descriptors_zh",
            _to_string_tuple(
                self.descriptors_zh, "CharacterCandidate.descriptors_zh"
            ),
        )
        _require_text(self.summary_zh, "CharacterCandidate.summary_zh")
        _require_enum(
            self.evidence_strength,
            EVIDENCE_STRENGTHS,
            "CharacterCandidate.evidence_strength",
        )
        object.__setattr__(
            self,
            "evidence",
            _to_evidence_tuple(self.evidence, "CharacterCandidate.evidence"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "display_name_original": self.display_name_original,
            "aliases_original": list(self.aliases_original),
            "descriptors_zh": list(self.descriptors_zh),
            "summary_zh": self.summary_zh,
            "evidence_strength": self.evidence_strength,
            "evidence": [evidence.to_dict() for evidence in self.evidence],
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CharacterCandidate":
        _require_exact_keys(
            value,
            {
                "candidate_id",
                "display_name_original",
                "aliases_original",
                "descriptors_zh",
                "summary_zh",
                "evidence_strength",
                "evidence",
            },
            "CharacterCandidate",
        )
        for list_field in ("aliases_original", "descriptors_zh", "evidence"):
            if not isinstance(value[list_field], list):
                raise ExtractionModelError(
                    f"CharacterCandidate.{list_field} must be a list"
                )
        return cls(
            candidate_id=value["candidate_id"],
            display_name_original=value["display_name_original"],
            aliases_original=tuple(value["aliases_original"]),
            descriptors_zh=tuple(value["descriptors_zh"]),
            summary_zh=value["summary_zh"],
            evidence_strength=value["evidence_strength"],
            evidence=tuple(
                EvidenceRef.from_dict(item) for item in value["evidence"]
            ),
        )


@dataclass(frozen=True, slots=True)
class LocationCandidate:
    candidate_id: str
    display_name_original: str
    aliases_original: tuple[str, ...]
    descriptors_zh: tuple[str, ...]
    summary_zh: str
    evidence_strength: str
    evidence: tuple[EvidenceRef, ...]

    def __post_init__(self) -> None:
        _require_candidate_id(
            self.candidate_id, "location", "LocationCandidate.candidate_id"
        )
        _require_text(
            self.display_name_original,
            "LocationCandidate.display_name_original",
        )
        object.__setattr__(
            self,
            "aliases_original",
            _to_string_tuple(
                self.aliases_original, "LocationCandidate.aliases_original"
            ),
        )
        object.__setattr__(
            self,
            "descriptors_zh",
            _to_string_tuple(
                self.descriptors_zh, "LocationCandidate.descriptors_zh"
            ),
        )
        _require_text(self.summary_zh, "LocationCandidate.summary_zh")
        _require_enum(
            self.evidence_strength,
            EVIDENCE_STRENGTHS,
            "LocationCandidate.evidence_strength",
        )
        object.__setattr__(
            self,
            "evidence",
            _to_evidence_tuple(self.evidence, "LocationCandidate.evidence"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "display_name_original": self.display_name_original,
            "aliases_original": list(self.aliases_original),
            "descriptors_zh": list(self.descriptors_zh),
            "summary_zh": self.summary_zh,
            "evidence_strength": self.evidence_strength,
            "evidence": [evidence.to_dict() for evidence in self.evidence],
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "LocationCandidate":
        _require_exact_keys(
            value,
            {
                "candidate_id",
                "display_name_original",
                "aliases_original",
                "descriptors_zh",
                "summary_zh",
                "evidence_strength",
                "evidence",
            },
            "LocationCandidate",
        )
        for list_field in ("aliases_original", "descriptors_zh", "evidence"):
            if not isinstance(value[list_field], list):
                raise ExtractionModelError(
                    f"LocationCandidate.{list_field} must be a list"
                )
        return cls(
            candidate_id=value["candidate_id"],
            display_name_original=value["display_name_original"],
            aliases_original=tuple(value["aliases_original"]),
            descriptors_zh=tuple(value["descriptors_zh"]),
            summary_zh=value["summary_zh"],
            evidence_strength=value["evidence_strength"],
            evidence=tuple(
                EvidenceRef.from_dict(item) for item in value["evidence"]
            ),
        )


@dataclass(frozen=True, slots=True)
class FactCandidate:
    candidate_id: str
    fact_type: str
    statement_zh: str
    subject_refs: tuple[str, ...]
    object_refs: tuple[str, ...]
    evidence_strength: str
    evidence: tuple[EvidenceRef, ...]

    def __post_init__(self) -> None:
        _require_candidate_id(
            self.candidate_id, "fact", "FactCandidate.candidate_id"
        )
        _require_enum(self.fact_type, FACT_TYPES, "FactCandidate.fact_type")
        _require_text(self.statement_zh, "FactCandidate.statement_zh")
        object.__setattr__(
            self,
            "subject_refs",
            _to_string_tuple(self.subject_refs, "FactCandidate.subject_refs"),
        )
        object.__setattr__(
            self,
            "object_refs",
            _to_string_tuple(self.object_refs, "FactCandidate.object_refs"),
        )
        _require_enum(
            self.evidence_strength,
            EVIDENCE_STRENGTHS,
            "FactCandidate.evidence_strength",
        )
        object.__setattr__(
            self,
            "evidence",
            _to_evidence_tuple(self.evidence, "FactCandidate.evidence"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "fact_type": self.fact_type,
            "statement_zh": self.statement_zh,
            "subject_refs": list(self.subject_refs),
            "object_refs": list(self.object_refs),
            "evidence_strength": self.evidence_strength,
            "evidence": [evidence.to_dict() for evidence in self.evidence],
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "FactCandidate":
        _require_exact_keys(
            value,
            {
                "candidate_id",
                "fact_type",
                "statement_zh",
                "subject_refs",
                "object_refs",
                "evidence_strength",
                "evidence",
            },
            "FactCandidate",
        )
        for list_field in ("subject_refs", "object_refs", "evidence"):
            if not isinstance(value[list_field], list):
                raise ExtractionModelError(
                    f"FactCandidate.{list_field} must be a list"
                )
        return cls(
            candidate_id=value["candidate_id"],
            fact_type=value["fact_type"],
            statement_zh=value["statement_zh"],
            subject_refs=tuple(value["subject_refs"]),
            object_refs=tuple(value["object_refs"]),
            evidence_strength=value["evidence_strength"],
            evidence=tuple(
                EvidenceRef.from_dict(item) for item in value["evidence"]
            ),
        )


@dataclass(frozen=True, slots=True)
class EventCandidate:
    candidate_id: str
    summary_zh: str
    participant_refs: tuple[str, ...]
    location_refs: tuple[str, ...]
    temporal_mode: str
    evidence_strength: str
    evidence: tuple[EvidenceRef, ...]

    def __post_init__(self) -> None:
        _require_candidate_id(
            self.candidate_id, "event", "EventCandidate.candidate_id"
        )
        _require_text(self.summary_zh, "EventCandidate.summary_zh")
        object.__setattr__(
            self,
            "participant_refs",
            _to_string_tuple(
                self.participant_refs, "EventCandidate.participant_refs"
            ),
        )
        object.__setattr__(
            self,
            "location_refs",
            _to_string_tuple(
                self.location_refs, "EventCandidate.location_refs"
            ),
        )
        _require_enum(
            self.temporal_mode, TEMPORAL_MODES, "EventCandidate.temporal_mode"
        )
        _require_enum(
            self.evidence_strength,
            EVIDENCE_STRENGTHS,
            "EventCandidate.evidence_strength",
        )
        object.__setattr__(
            self,
            "evidence",
            _to_evidence_tuple(self.evidence, "EventCandidate.evidence"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "summary_zh": self.summary_zh,
            "participant_refs": list(self.participant_refs),
            "location_refs": list(self.location_refs),
            "temporal_mode": self.temporal_mode,
            "evidence_strength": self.evidence_strength,
            "evidence": [evidence.to_dict() for evidence in self.evidence],
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "EventCandidate":
        _require_exact_keys(
            value,
            {
                "candidate_id",
                "summary_zh",
                "participant_refs",
                "location_refs",
                "temporal_mode",
                "evidence_strength",
                "evidence",
            },
            "EventCandidate",
        )
        for list_field in ("participant_refs", "location_refs", "evidence"):
            if not isinstance(value[list_field], list):
                raise ExtractionModelError(
                    f"EventCandidate.{list_field} must be a list"
                )
        return cls(
            candidate_id=value["candidate_id"],
            summary_zh=value["summary_zh"],
            participant_refs=tuple(value["participant_refs"]),
            location_refs=tuple(value["location_refs"]),
            temporal_mode=value["temporal_mode"],
            evidence_strength=value["evidence_strength"],
            evidence=tuple(
                EvidenceRef.from_dict(item) for item in value["evidence"]
            ),
        )


@dataclass(frozen=True, slots=True)
class RelationshipCandidate:
    candidate_id: str
    source_ref: str
    target_ref: str
    relationship_type_zh: str
    state_zh: str | None
    direction: str
    evidence_strength: str
    evidence: tuple[EvidenceRef, ...]

    def __post_init__(self) -> None:
        _require_candidate_id(
            self.candidate_id,
            "relationship",
            "RelationshipCandidate.candidate_id",
        )
        _require_text(self.source_ref, "RelationshipCandidate.source_ref")
        _require_text(self.target_ref, "RelationshipCandidate.target_ref")
        _require_text(
            self.relationship_type_zh,
            "RelationshipCandidate.relationship_type_zh",
        )
        _require_optional_text(self.state_zh, "RelationshipCandidate.state_zh")
        _require_enum(
            self.direction,
            RELATIONSHIP_DIRECTIONS,
            "RelationshipCandidate.direction",
        )
        _require_enum(
            self.evidence_strength,
            EVIDENCE_STRENGTHS,
            "RelationshipCandidate.evidence_strength",
        )
        object.__setattr__(
            self,
            "evidence",
            _to_evidence_tuple(
                self.evidence, "RelationshipCandidate.evidence"
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "source_ref": self.source_ref,
            "target_ref": self.target_ref,
            "relationship_type_zh": self.relationship_type_zh,
            "state_zh": self.state_zh,
            "direction": self.direction,
            "evidence_strength": self.evidence_strength,
            "evidence": [evidence.to_dict() for evidence in self.evidence],
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RelationshipCandidate":
        _require_exact_keys(
            value,
            {
                "candidate_id",
                "source_ref",
                "target_ref",
                "relationship_type_zh",
                "state_zh",
                "direction",
                "evidence_strength",
                "evidence",
            },
            "RelationshipCandidate",
        )
        if not isinstance(value["evidence"], list):
            raise ExtractionModelError(
                "RelationshipCandidate.evidence must be a list"
            )
        return cls(
            candidate_id=value["candidate_id"],
            source_ref=value["source_ref"],
            target_ref=value["target_ref"],
            relationship_type_zh=value["relationship_type_zh"],
            state_zh=value["state_zh"],
            direction=value["direction"],
            evidence_strength=value["evidence_strength"],
            evidence=tuple(
                EvidenceRef.from_dict(item) for item in value["evidence"]
            ),
        )


@dataclass(frozen=True, slots=True)
class UnresolvedMentionCandidate:
    candidate_id: str
    mention_original: str
    mention_kind: str
    reason_zh: str
    possible_candidate_refs: tuple[str, ...]
    evidence_strength: str
    evidence: tuple[EvidenceRef, ...]

    def __post_init__(self) -> None:
        _require_candidate_id(
            self.candidate_id,
            "unresolved_mention",
            "UnresolvedMentionCandidate.candidate_id",
        )
        _require_text(
            self.mention_original,
            "UnresolvedMentionCandidate.mention_original",
        )
        _require_enum(
            self.mention_kind, MENTION_KINDS, "UnresolvedMentionCandidate.mention_kind"
        )
        _require_text(self.reason_zh, "UnresolvedMentionCandidate.reason_zh")
        object.__setattr__(
            self,
            "possible_candidate_refs",
            _to_string_tuple(
                self.possible_candidate_refs,
                "UnresolvedMentionCandidate.possible_candidate_refs",
            ),
        )
        # Uncertainty is a successful result, not an error: an unresolved
        # mention must always be recorded as structurally uncertain.
        if self.evidence_strength != "uncertain":
            raise ExtractionModelError(
                "UnresolvedMentionCandidate.evidence_strength must be "
                "'uncertain'"
            )
        object.__setattr__(
            self,
            "evidence",
            _to_evidence_tuple(
                self.evidence, "UnresolvedMentionCandidate.evidence"
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "mention_original": self.mention_original,
            "mention_kind": self.mention_kind,
            "reason_zh": self.reason_zh,
            "possible_candidate_refs": list(self.possible_candidate_refs),
            "evidence_strength": self.evidence_strength,
            "evidence": [evidence.to_dict() for evidence in self.evidence],
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "UnresolvedMentionCandidate":
        _require_exact_keys(
            value,
            {
                "candidate_id",
                "mention_original",
                "mention_kind",
                "reason_zh",
                "possible_candidate_refs",
                "evidence_strength",
                "evidence",
            },
            "UnresolvedMentionCandidate",
        )
        for list_field in ("possible_candidate_refs", "evidence"):
            if not isinstance(value[list_field], list):
                raise ExtractionModelError(
                    f"UnresolvedMentionCandidate.{list_field} must be a list"
                )
        return cls(
            candidate_id=value["candidate_id"],
            mention_original=value["mention_original"],
            mention_kind=value["mention_kind"],
            reason_zh=value["reason_zh"],
            possible_candidate_refs=tuple(value["possible_candidate_refs"]),
            evidence_strength=value["evidence_strength"],
            evidence=tuple(
                EvidenceRef.from_dict(item) for item in value["evidence"]
            ),
        )


# ---------------------------------------------------------------------------
# CandidatePayload (provider structured-output shape)
# ---------------------------------------------------------------------------

# Deterministic category ordering for serialization / deserialization.
_CANDIDATE_PAYLOAD_CATEGORIES = (
    ("characters", CharacterCandidate),
    ("locations", LocationCandidate),
    ("facts", FactCandidate),
    ("events", EventCandidate),
    ("relationships", RelationshipCandidate),
    ("unresolved_mentions", UnresolvedMentionCandidate),
)


@dataclass(frozen=True, slots=True)
class CandidatePayload:
    """The exact LLM structured-output shape: six candidate arrays.

    All six categories always exist; an empty category is ``[]`` (a present
    but empty array), never a missing key. A3A performs no cross-candidate
    graph closure, duplicate/gap detection, or canonical ordering; those are
    A3B. Within a category the input order is preserved verbatim so the
    serializer is deterministic for a given object.
    """

    characters: tuple[CharacterCandidate, ...] = ()
    locations: tuple[LocationCandidate, ...] = ()
    facts: tuple[FactCandidate, ...] = ()
    events: tuple[EventCandidate, ...] = ()
    relationships: tuple[RelationshipCandidate, ...] = ()
    unresolved_mentions: tuple[UnresolvedMentionCandidate, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "characters",
            _to_candidate_tuple(self.characters, "characters", CharacterCandidate),
        )
        object.__setattr__(
            self,
            "locations",
            _to_candidate_tuple(self.locations, "locations", LocationCandidate),
        )
        object.__setattr__(
            self,
            "facts",
            _to_candidate_tuple(self.facts, "facts", FactCandidate),
        )
        object.__setattr__(
            self,
            "events",
            _to_candidate_tuple(self.events, "events", EventCandidate),
        )
        object.__setattr__(
            self,
            "relationships",
            _to_candidate_tuple(
                self.relationships, "relationships", RelationshipCandidate
            ),
        )
        object.__setattr__(
            self,
            "unresolved_mentions",
            _to_candidate_tuple(
                self.unresolved_mentions,
                "unresolved_mentions",
                UnresolvedMentionCandidate,
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "characters": [c.to_dict() for c in self.characters],
            "locations": [c.to_dict() for c in self.locations],
            "facts": [c.to_dict() for c in self.facts],
            "events": [c.to_dict() for c in self.events],
            "relationships": [c.to_dict() for c in self.relationships],
            "unresolved_mentions": [c.to_dict() for c in self.unresolved_mentions],
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CandidatePayload":
        _require_exact_keys(
            value,
            {name for name, _ in _CANDIDATE_PAYLOAD_CATEGORIES},
            "CandidatePayload",
        )
        for name, _ in _CANDIDATE_PAYLOAD_CATEGORIES:
            if not isinstance(value[name], list):
                raise ExtractionModelError(
                    f"CandidatePayload.{name} must be a list"
                )
        return cls(
            characters=tuple(
                CharacterCandidate.from_dict(item) for item in value["characters"]
            ),
            locations=tuple(
                LocationCandidate.from_dict(item) for item in value["locations"]
            ),
            facts=tuple(FactCandidate.from_dict(item) for item in value["facts"]),
            events=tuple(EventCandidate.from_dict(item) for item in value["events"]),
            relationships=tuple(
                RelationshipCandidate.from_dict(item)
                for item in value["relationships"]
            ),
            unresolved_mentions=tuple(
                UnresolvedMentionCandidate.from_dict(item)
                for item in value["unresolved_mentions"]
            ),
        )


# ---------------------------------------------------------------------------
# StoryExtractionProfile
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StoryExtractionProfile:
    """Model-independent orchestration profile for A3 chunk extraction.

    This is deliberately separate from the A-I3 runtime/semantic LLM profile:
    it carries no endpoint, credential, timeout, concrete model, temperature,
    or reasoning field. Its canonical hash is the ``extraction_profile_hash``
    that participates in CandidateExtraction reuse identity.
    """

    schema_version: int
    profile_id: str
    working_language: str
    prompt_id: str
    prompt_version: int
    output_schema_id: str
    output_schema_version: int
    max_generation_rounds: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
            or self.schema_version != STORY_EXTRACTION_PROFILE_SCHEMA_VERSION
        ):
            raise ExtractionModelError(
                "StoryExtractionProfile.schema_version must be "
                f"{STORY_EXTRACTION_PROFILE_SCHEMA_VERSION}"
            )
        _require_storage_id(
            self.profile_id, "StoryExtractionProfile.profile_id"
        )
        _require_text(
            self.working_language, "StoryExtractionProfile.working_language"
        )
        _require_storage_id(self.prompt_id, "StoryExtractionProfile.prompt_id")
        _require_positive_int(
            self.prompt_version, "StoryExtractionProfile.prompt_version"
        )
        _require_storage_id(
            self.output_schema_id, "StoryExtractionProfile.output_schema_id"
        )
        _require_positive_int(
            self.output_schema_version,
            "StoryExtractionProfile.output_schema_version",
        )
        # Frozen A-I4 v1 hard ceiling: schema_version 1 requires exactly 2
        # semantic generation rounds (see STORY_EXTRACTION_MAX_GENERATION_ROUNDS_V1).
        _require_positive_int(
            self.max_generation_rounds,
            "StoryExtractionProfile.max_generation_rounds",
        )
        if (
            self.schema_version == STORY_EXTRACTION_PROFILE_SCHEMA_VERSION
            and self.max_generation_rounds != STORY_EXTRACTION_MAX_GENERATION_ROUNDS_V1
        ):
            raise ExtractionModelError(
                "StoryExtractionProfile.max_generation_rounds must be "
                f"{STORY_EXTRACTION_MAX_GENERATION_ROUNDS_V1} for schema_version "
                f"{STORY_EXTRACTION_PROFILE_SCHEMA_VERSION} (frozen A-I4 v1 ceiling)"
            )

    @property
    def profile_hash(self) -> str:
        """Canonical identity of the profile (reuses the shared ``content_hash``)."""

        return content_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "profile_id": self.profile_id,
            "working_language": self.working_language,
            "prompt_id": self.prompt_id,
            "prompt_version": self.prompt_version,
            "output_schema_id": self.output_schema_id,
            "output_schema_version": self.output_schema_version,
            "max_generation_rounds": self.max_generation_rounds,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "StoryExtractionProfile":
        _require_exact_keys(
            value,
            {
                "schema_version",
                "profile_id",
                "working_language",
                "prompt_id",
                "prompt_version",
                "output_schema_id",
                "output_schema_version",
                "max_generation_rounds",
            },
            "StoryExtractionProfile",
        )
        return cls(**value)


def load_story_extraction_profile(path: str | Path) -> StoryExtractionProfile:
    """Load and validate a tracked StoryExtractionProfile from a YAML file."""

    path = Path(path).expanduser()
    if not path.is_file():
        raise ExtractionModelError(f"story extraction profile not found: {path}")
    try:
        data = load_yaml(path)
    except Exception as exc:  # noqa: BLE001 - report any load failure
        raise ExtractionModelError(
            f"failed to load story extraction profile: {exc}"
        ) from exc
    if not isinstance(data, dict):
        raise ExtractionModelError(
            "story extraction profile must contain an object"
        )
    try:
        return StoryExtractionProfile.from_dict(data)
    except ExtractionModelError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ExtractionModelError(f"invalid story extraction profile: {exc}") from exc


# ---------------------------------------------------------------------------
# CandidateExtraction (typed persisted shape)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CandidateExtraction:
    """The typed/static persisted shape of a chunk CandidateExtraction.

    A3A establishes only the deterministic serialization/deserialization
    contract. It does NOT assign an artifact revision, wrap an immutable
    envelope, write the artifact store, set a CURRENT pointer, or reuse.
    Those are A3C (``#13``).
    """

    schema_version: int
    project_id: str
    document_id: str
    chunk_profile_id: str
    chunk_id: str
    source_document_ref: ArtifactRef
    source_chunk_ref: ArtifactRef
    extraction_profile_id: str
    extraction_profile_hash: str
    generation_provenance: LLMInvocationProvenance
    candidates: CandidatePayload

    def __post_init__(self) -> None:
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
            or self.schema_version != CANDIDATE_EXTRACTION_SCHEMA_VERSION
        ):
            raise ExtractionModelError(
                "CandidateExtraction.schema_version must be "
                f"{CANDIDATE_EXTRACTION_SCHEMA_VERSION}"
            )
        _require_text(self.project_id, "CandidateExtraction.project_id")
        _require_text(self.document_id, "CandidateExtraction.document_id")
        _require_text(
            self.chunk_profile_id, "CandidateExtraction.chunk_profile_id"
        )
        _require_text(self.chunk_id, "CandidateExtraction.chunk_id")
        if not isinstance(self.source_document_ref, ArtifactRef):
            raise ExtractionModelError(
                "CandidateExtraction.source_document_ref must be an ArtifactRef"
            )
        if not isinstance(self.source_chunk_ref, ArtifactRef):
            raise ExtractionModelError(
                "CandidateExtraction.source_chunk_ref must be an ArtifactRef"
            )
        _require_storage_id(
            self.extraction_profile_id,
            "CandidateExtraction.extraction_profile_id",
        )
        _require_hash(
            self.extraction_profile_hash,
            "CandidateExtraction.extraction_profile_hash",
        )
        if not isinstance(self.generation_provenance, LLMInvocationProvenance):
            raise ExtractionModelError(
                "CandidateExtraction.generation_provenance must be an "
                "LLMInvocationProvenance"
            )
        if not isinstance(self.candidates, CandidatePayload):
            raise ExtractionModelError(
                "CandidateExtraction.candidates must be a CandidatePayload"
            )

    @property
    def artifact_type(self) -> str:
        return CANDIDATE_EXTRACTION_ARTIFACT_TYPE

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "project_id": self.project_id,
            "document_id": self.document_id,
            "chunk_profile_id": self.chunk_profile_id,
            "chunk_id": self.chunk_id,
            "source_document_ref": self.source_document_ref.to_dict(),
            "source_chunk_ref": self.source_chunk_ref.to_dict(),
            "extraction_profile_id": self.extraction_profile_id,
            "extraction_profile_hash": self.extraction_profile_hash,
            "generation_provenance": self.generation_provenance.to_dict(),
            "candidates": self.candidates.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CandidateExtraction":
        _require_exact_keys(value, set(_CANDIDATE_EXTRACTION_KEYS), "CandidateExtraction")
        try:
            source_document_ref = ArtifactRef.from_dict(
                value["source_document_ref"]
            )
        except Exception as exc:  # noqa: BLE001
            raise ExtractionModelError(f"invalid source_document_ref: {exc}") from exc
        try:
            source_chunk_ref = ArtifactRef.from_dict(value["source_chunk_ref"])
        except Exception as exc:  # noqa: BLE001
            raise ExtractionModelError(f"invalid source_chunk_ref: {exc}") from exc
        prov = value["generation_provenance"]
        if not isinstance(prov, dict) or set(prov) != _LLM_PROVENANCE_KEYS:
            raise ExtractionModelError(
                "CandidateExtraction.generation_provenance must contain exactly "
                "the A-I3 LLMInvocationProvenance fields"
            )
        try:
            generation_provenance = LLMInvocationProvenance(**prov)
        except (TypeError, LLMConfigError) as exc:
            raise ExtractionModelError(f"invalid generation_provenance: {exc}") from exc
        candidates = CandidatePayload.from_dict(value["candidates"])
        return cls(
            schema_version=value["schema_version"],
            project_id=value["project_id"],
            document_id=value["document_id"],
            chunk_profile_id=value["chunk_profile_id"],
            chunk_id=value["chunk_id"],
            source_document_ref=source_document_ref,
            source_chunk_ref=source_chunk_ref,
            extraction_profile_id=value["extraction_profile_id"],
            extraction_profile_hash=value["extraction_profile_hash"],
            generation_provenance=generation_provenance,
            candidates=candidates,
        )
