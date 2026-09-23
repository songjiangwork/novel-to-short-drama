"""v1.2 A5A: fact / event / relationship consolidation domain contracts.

This module defines the *domain and static contract* for the A5 consolidation
stage: the versioned consolidation profile (A2-style pinned asset ids, no
runtime/credentials, and a versioned blocking-policy slot with no algorithm
parameters); the chunk-local candidate refs and indexed candidate sets; the
persisted fact / event / relationship semantic decision sets; the canonical
fact / event / relationship / story-conflict sets; the A5 semantic identity
(backend-neutral, like A4); and the ``ConsolidationManifest`` root aggregate.

A5A is deliberately static: it pins the contract but does NOT implement
candidate collection, blocking, the LLM semantic pass, canonical-id allocation,
conflict detection, or persistence / CURRENT reuse. Those are A5B-A5G.

Design notes (parity with A3 / A4A):

* Reuse the existing authorities rather than redefining them:
  :class:`EvidenceRef`, :class:`ArtifactRef`,
  :class:`LLMInvocationProvenance` (A-I3), :class:`A3InputIdentity`,
  ``content_hash``, and the canonical-JSON hash.
* All domain objects are frozen, hashable, and fail-closed
  (``to_dict`` / ``from_dict`` with exact-key parsing).
* The persisted decision set distinguishes fact / event / relationship pair
  decisions with domain-specific closed decision enums. The provider
  (LLM) contract uses pair-local evidence selectors (L0 / R0 / ...) rather
  than letting the model freely generate arbitrary persisted
  :class:`EvidenceRef` objects; the exact evidence refs are resolved and
  carried on the persisted decision.
* A5 semantic decisions carry the A-I3 ``LLMInvocationProvenance`` for the
  llm path; deterministic / manual decisions carry null prompt identity and
  null provenance.
* The A5 semantic identity is backend-neutral: it pins the consolidation
  profile, the per-domain semantic LLM profile ids/hashes, the exact prompt /
  output-schema asset ids, and the deterministic semantic request hashes; it
  does NOT pin a provider family, model name, endpoint, or timeout.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from short_drama.artifacts import ArtifactRef, content_hash
from short_drama.llm import LLMInvocationProvenance
from short_drama.story.errors import ConsolidationModelError
from short_drama.story.extraction import (
    EVIDENCE_STRENGTHS,
    FACT_TYPES,
    RELATIONSHIP_DIRECTIONS,
    TEMPORAL_MODES,
)
from short_drama.story.reconciliation import A3InputIdentity

# ---------------------------------------------------------------------------
# Error type
# ---------------------------------------------------------------------------
# (ConsolidationModelError is defined in errors.py.)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

FACT_DECISIONS: frozenset[str] = frozenset(
    {"same_fact", "compatible_fact", "state_change", "conflict", "unrelated", "uncertain"}
)
EVENT_DECISIONS: frozenset[str] = frozenset({"same_event", "different_event", "uncertain"})
RELATIONSHIP_DECISIONS: frozenset[str] = frozenset(
    {"same_relationship", "different_relationship", "uncertain"}
)
CONSOLIDATION_METHODS: frozenset[str] = frozenset(
    {"deterministic", "llm", "manual"}
)
STATE_TRANSITION_KINDS: frozenset[str] = frozenset(
    {"state_change", "reassertion", "clarification", "correction"}
)
STORY_CONFLICT_KINDS: frozenset[str] = frozenset(
    {"fact_conflict", "relationship_state_conflict", "other"}
)
STORY_CONFLICT_STATUSES: frozenset[str] = frozenset({"unresolved"})

# A5 canonical-id namespaces (reserved prefixes, zero-padded numeric suffix).
FACT_ID_PATTERN = r"fact_[0-9]{6,}"
EVENT_ID_PATTERN = r"evt_[0-9]{6,}"
RELATIONSHIP_ID_PATTERN = r"rel_[0-9]{6,}"
STATE_TRANSITION_ID_PATTERN = r"trans_[0-9]{6,}"
STORY_CONFLICT_ID_PATTERN = r"conf_[0-9]{6,}"

# Candidate local-id prefixes (chunk-local, zero-padded numeric suffix).
_FACT_LOCAL_ID = re.compile(r"cand_fact_[0-9]{3,}")
_EVENT_LOCAL_ID = re.compile(r"cand_evt_[0-9]{3,}")
_REL_LOCAL_ID = re.compile(r"cand_rel_[0-9]{3,}")
_CHUNK_ID = re.compile(r"CH[0-9]{3,}_C[0-9]{3,}")

# Tracked contract schema versions (all A5A contracts are version 1).
CONTRACT_SCHEMA_VERSION = 1
CONTRACT_PROFILE_SCHEMA_VERSION = 1

# A consolidated A5 candidate-ref pattern (chunk-local, domain-prefixed).
CONSOLIDATION_CANDIDATE_REF_PATTERN = (
    r"^CH[0-9]{3,}_C[0-9]{3,}:cand_(?:fact|evt|rel)_[0-9]{3,}$"
)

# A5 pair-local evidence selector (L0 / R0 / L1 / ...).
A5_EVIDENCE_SELECTOR_PATTERN = r"^LR[0-9]+$"

# Candidate ref namespaces keyed by local-id prefix.
_NAMESPACE_BY_LOCAL_PREFIX = {
    "cand_fact": ("fact", _FACT_LOCAL_ID),
    "cand_evt": ("event", _EVENT_LOCAL_ID),
    "cand_rel": ("relationship", _REL_LOCAL_ID),
}

_FACT_ID_RE = re.compile(FACT_ID_PATTERN)
_EVENT_ID_RE = re.compile(EVENT_ID_PATTERN)
_REL_ID_RE = re.compile(RELATIONSHIP_ID_PATTERN)
_TRANSITION_ID_RE = re.compile(STATE_TRANSITION_ID_PATTERN)
_CONFLICT_ID_RE = re.compile(STORY_CONFLICT_ID_PATTERN)
_HASH64_RE = re.compile(r"^[0-9a-f]{64}$")
# Bound entity ids from A2/A4 (entity map): char / loc / obj, zero-padded.
_BOUND_ENTITY_RE = re.compile(r"(?:char|loc|obj)_[0-9]{4,}")


# ---------------------------------------------------------------------------
# Field validation helpers (fail closed)
# ---------------------------------------------------------------------------


def _require_text(value: object, name: str, *, min_length: int = 1) -> str:
    if not isinstance(value, str) or len(value) < min_length:
        raise ConsolidationModelError(f"{name} must be a non-empty string")
    return value


def _require_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or not _HASH64_RE.fullmatch(value):
        raise ConsolidationModelError(f"{name} must be a 64-char lowercase hex sha256")
    return value


def _require_positive_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ConsolidationModelError(f"{name} must be a positive integer")
    return value


def _require_non_negative_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ConsolidationModelError(f"{name} must be a non-negative integer")
    return value


def _require_bool(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise ConsolidationModelError(f"{name} must be a boolean")
    return value


def _require_enum(value: object, allowed: frozenset[str], name: str) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise ConsolidationModelError(
            f"{name} must be one of {sorted(allowed)}, got {value!r}"
        )
    return value


def _require_exact_keys(value: dict[str, Any], allowed: set[str], name: str) -> None:
    keys = set(value)
    extra = keys - allowed
    if extra:
        raise ConsolidationModelError(f"{name} has unexpected keys: {sorted(extra)}")
    missing = allowed - keys
    if missing:
        raise ConsolidationModelError(f"{name} is missing keys: {sorted(missing)}")


def _require_text_tuple(
    value: object, name: str, *, allow_empty: bool = True
) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ConsolidationModelError(f"{name} must be a list of strings")
    items: list[str] = []
    for item in value:
        _require_text(item, f"{name}[]")
        items.append(item)
    if not allow_empty and not items:
        raise ConsolidationModelError(f"{name} must be non-empty")
    return tuple(items)


def _require_text_or_null_tuple(
    value: object, name: str
) -> tuple[str | None, ...]:
    if not isinstance(value, (list, tuple)):
        raise ConsolidationModelError(f"{name} must be a list")
    items: list[str | None] = []
    for item in value:
        if item is None:
            items.append(None)
        else:
            _require_text(item, f"{name}[]")
            items.append(item)
    return tuple(items)


def _to_evidence_tuple(value: object, name: str) -> tuple[Any, ...]:
    """Normalize an EvidenceRef collection to a validated tuple (fail closed).

    Accepts EvidenceRef objects (direct construction) or dicts (from_dict) and
    returns a tuple of EvidenceRef objects.
    """
    if not isinstance(value, (list, tuple)):
        raise ConsolidationModelError(f"{name} must be a list")
    from short_drama.story.extraction import EvidenceRef

    items: list[Any] = []
    for item in value:
        if isinstance(item, EvidenceRef):
            items.append(item)
        elif isinstance(item, dict):
            items.append(EvidenceRef.from_dict(item))
        else:
            raise ConsolidationModelError(
                f"{name} must contain EvidenceRef values or dicts"
            )
    return tuple(items)


def _require_bound_entity(value: object, name: str) -> str:
    if not isinstance(value, str) or not _BOUND_ENTITY_RE.fullmatch(value):
        raise ConsolidationModelError(
            f"{name} must reference a bound entity id (char_/loc_/obj_ + zero-padded digits)"
        )
    return value


def _require_id_pattern(value: object, pattern: re.Pattern[str], name: str) -> str:
    if not isinstance(value, str) or not pattern.fullmatch(value):
        raise ConsolidationModelError(f"{name} must match {pattern.pattern}")
    return value


def _require_hash_tuple(
    value: object, name: str, *, allow_duplicates: bool = False
) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ConsolidationModelError(f"{name} must be a list of sha256 strings")
    items: list[str] = []
    seen: set[str] = set()
    for item in value:
        _require_sha256(item, f"{name}[]")
        if not allow_duplicates and item in seen:
            raise ConsolidationModelError(f"{name} must be unique")
        seen.add(item)
        items.append(item)
    return tuple(items)


def _to_llm_provenance(value: object, name: str) -> LLMInvocationProvenance | None:
    """Normalize the generation provenance (accepts an object or a dict)."""
    if value is None:
        return None
    if isinstance(value, LLMInvocationProvenance):
        return value
    if not isinstance(value, dict):
        raise ConsolidationModelError(f"{name} must be null or an object")
    return LLMInvocationProvenance.from_dict(value)


def _validate_candidate_identity(
    *,
    global_candidate_ref: object,
    chunk_id: object,
    local_candidate_id: object,
    candidate_extraction_ref: object,
    evidence_refs: object,
    evidence_strength: object,
    name: str,
) -> tuple[Any, ...]:
    """Validate the shared fields of every indexed candidate."""
    _require_text(global_candidate_ref, f"{name}.global_candidate_ref")
    _require_text(chunk_id, f"{name}.chunk_id")
    _require_text(local_candidate_id, f"{name}.local_candidate_id")
    # The composite ref must be exactly chunk_id:local_candidate_id.
    expected = f"{chunk_id}:{local_candidate_id}"
    if global_candidate_ref != expected:
        raise ConsolidationModelError(
            f"{name}.global_candidate_ref must be 'chunk_id:local_candidate_id'"
        )
    _require_text(chunk_id, f"{name}.source_order_key")
    return _to_evidence_tuple(evidence_refs, f"{name}.evidence_refs")


# ---------------------------------------------------------------------------
# ConsolidationCandidateRef
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ConsolidationCandidateRef:
    """A chunk-local consolidation candidate ref and its parsed parts.

    The persisted canonical form is ``CH###_C###:cand_<domain>_###`` (or longer),
    e.g. ``CH003_C005:cand_fact_012``. The domain (``fact`` / ``event`` /
    ``relationship``) is derived from the local candidate id prefix. This is a
    thin validated wrapper over the string so that the Python side can enforce
    the namespace and consistency without re-parsing ad hoc.
    """

    chunk_id: str
    local_candidate_id: str
    global_candidate_ref: str
    namespace: str  # "fact" | "event" | "relationship"

    def __post_init__(self) -> None:
        _require_text(self.chunk_id, "chunk_id")
        _require_text(self.local_candidate_id, "local_candidate_id")
        _require_text(self.global_candidate_ref, "global_candidate_ref")
        if self.namespace not in ("fact", "event", "relationship"):
            raise ConsolidationModelError(
                f"namespace must be fact/event/relationship, got {self.namespace!r}"
            )
        expected = f"{self.chunk_id}:{self.local_candidate_id}"
        if self.global_candidate_ref != expected:
            raise ConsolidationModelError(
                f"global_candidate_ref {self.global_candidate_ref!r} is inconsistent "
                f"with chunk_id/local_candidate_id ({expected!r})"
            )
        prefix, _ = _namespace_for_local_id(self.local_candidate_id)
        if prefix != self.namespace:
            raise ConsolidationModelError(
                f"namespace {self.namespace!r} does not match local_candidate_id "
                f"{self.local_candidate_id!r} (expected {prefix!r})"
            )

    def to_string(self) -> str:
        return self.global_candidate_ref

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.global_candidate_ref

    @classmethod
    def parse(cls, value: object) -> "ConsolidationCandidateRef":
        """Parse the persisted canonical form (fail closed on any deviation)."""
        if not isinstance(value, str) or "\x00" in value:
            raise ConsolidationModelError(
                "ConsolidationCandidateRef must be a non-empty string without NUL"
            )
        if value.count(":") != 1:
            raise ConsolidationModelError(
                f"candidate ref must be 'chunk_id:local_candidate_id', got {value!r}"
            )
        chunk_id, local_candidate_id = value.split(":", 1)
        if not _CHUNK_ID.fullmatch(chunk_id):
            raise ConsolidationModelError(f"invalid chunk_id in candidate ref {value!r}")
        namespace, _ = _namespace_for_local_id(local_candidate_id)
        return cls(
            chunk_id=chunk_id,
            local_candidate_id=local_candidate_id,
            global_candidate_ref=value,
            namespace=namespace,
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "chunk_id": self.chunk_id,
            "local_candidate_id": self.local_candidate_id,
            "global_candidate_ref": self.global_candidate_ref,
            "namespace": self.namespace,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ConsolidationCandidateRef":
        _require_exact_keys(
            value,
            {"chunk_id", "local_candidate_id", "global_candidate_ref", "namespace"},
            "ConsolidationCandidateRef",
        )
        return cls(
            chunk_id=value["chunk_id"],
            local_candidate_id=value["local_candidate_id"],
            global_candidate_ref=value["global_candidate_ref"],
            namespace=value["namespace"],
        )


def _namespace_for_local_id(local_candidate_id: str) -> tuple[str, re.Pattern[str]]:
    for prefix, (namespace, pattern) in _NAMESPACE_BY_LOCAL_PREFIX.items():
        if local_candidate_id.startswith(prefix):
            if not pattern.fullmatch(local_candidate_id):
                raise ConsolidationModelError(
                    f"invalid local_candidate_id {local_candidate_id!r}"
                )
            return namespace, pattern
    raise ConsolidationModelError(
        f"local_candidate_id {local_candidate_id!r} is not a recognized "
        "consolidation candidate id"
    )


def _require_consolidation_candidate_ref(
    value: object, name: str, *, namespace: str | None = None
) -> ConsolidationCandidateRef:
    ref = ConsolidationCandidateRef.parse(value)
    if namespace is not None and ref.namespace != namespace:
        raise ConsolidationModelError(
            f"{name} must reference a {namespace} candidate, got {ref.namespace!r}"
        )
    return ref


def _require_pair_canonical_order(
    left: object,
    right: object,
    *,
    left_name: str,
    right_name: str,
) -> None:
    left_text = _require_text(left, left_name)
    right_text = _require_text(right, right_name)
    if not left_text < right_text:
        raise ConsolidationModelError(
            f"{left_name} must be strictly less than {right_name} (canonical order)"
        )


# ---------------------------------------------------------------------------
# ConsolidationProfile
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ConsolidationSemanticPass:
    """One A5 semantic pass pin (fact / event / relationship).

    Pins the semantic LLM profile id, the prompt id / version, and the output
    schema id / version that produce the domain selector payload.
    """

    semantic_profile_id: str
    prompt_id: str
    prompt_version: int
    output_schema_id: str
    output_schema_version: int

    def __post_init__(self) -> None:
        _require_text(self.semantic_profile_id, "semantic_profile_id")
        _require_text(self.prompt_id, "prompt_id")
        _require_positive_int(self.prompt_version, "prompt_version")
        _require_text(self.output_schema_id, "output_schema_id")
        _require_positive_int(self.output_schema_version, "output_schema_version")

    def to_dict(self) -> dict[str, Any]:
        return {
            "semantic_profile_id": self.semantic_profile_id,
            "prompt_id": self.prompt_id,
            "prompt_version": self.prompt_version,
            "output_schema_id": self.output_schema_id,
            "output_schema_version": self.output_schema_version,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ConsolidationSemanticPass":
        _require_exact_keys(
            value,
            {
                "semantic_profile_id",
                "prompt_id",
                "prompt_version",
                "output_schema_id",
                "output_schema_version",
            },
            "ConsolidationSemanticPass",
        )
        return cls(**value)


@dataclass(frozen=True, slots=True)
class ConsolidationProfile:
    """Versioned A5 consolidation orchestration profile (static, A2-style).

    Pins the working language, the versioned blocking-policy slot, the
    candidate-source chunk ids, the max generation rounds, and one
    ``ConsolidationSemanticPass`` pin per domain (fact / event / relationship).
    It deliberately carries NO runtime fields (endpoint / provider /
    credentials / timeouts / temperature) and NO blocking algorithm parameters
    (A5B binds the blocking-policy slot to the audited production blocking
    identity).
    """

    schema_version: int
    profile_id: str
    working_language: str
    blocking_policy_id: str
    max_generation_rounds: int
    fact: ConsolidationSemanticPass
    event: ConsolidationSemanticPass
    relationship: ConsolidationSemanticPass

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ConsolidationModelError(
                f"unsupported schema_version: {self.schema_version!r}"
            )
        _require_text(self.profile_id, "profile_id")
        _require_text(self.working_language, "working_language")
        _require_text(self.blocking_policy_id, "blocking_policy_id")
        _require_positive_int(self.max_generation_rounds, "max_generation_rounds")

    def content_hash(self) -> str:
        return content_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "profile_id": self.profile_id,
            "working_language": self.working_language,
            "blocking_policy_id": self.blocking_policy_id,
            "max_generation_rounds": self.max_generation_rounds,
            "fact": self.fact.to_dict(),
            "event": self.event.to_dict(),
            "relationship": self.relationship.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ConsolidationProfile":
        _require_exact_keys(
            value,
            {
                "schema_version",
                "profile_id",
                "working_language",
                "blocking_policy_id",
                "max_generation_rounds",
                "fact",
                "event",
                "relationship",
            },
            "ConsolidationProfile",
        )
        return cls(
            schema_version=value["schema_version"],
            profile_id=value["profile_id"],
            working_language=value["working_language"],
            blocking_policy_id=value["blocking_policy_id"],
            max_generation_rounds=value["max_generation_rounds"],
            fact=ConsolidationSemanticPass.from_dict(value["fact"]),
            event=ConsolidationSemanticPass.from_dict(value["event"]),
            relationship=ConsolidationSemanticPass.from_dict(value["relationship"]),
        )


def load_consolidation_profile(path: Path) -> ConsolidationProfile:
    """Load a tracked consolidation profile from YAML (fail closed)."""
    try:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConsolidationModelError(f"consolidation profile not found: {path}") from exc
    if not isinstance(raw, dict):
        raise ConsolidationModelError("consolidation profile must be a mapping")
    return ConsolidationProfile.from_dict(raw)


# ---------------------------------------------------------------------------
# Indexed candidates
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class IndexedFactCandidate:
    """A chunk-local fact candidate, indexed for A5 consolidation."""

    global_candidate_ref: str
    chunk_id: str
    local_candidate_id: str
    source_order_key: str
    fact_type: str
    statement_zh: str
    subject_refs: tuple[str, ...]
    object_refs: tuple[str | None, ...]
    evidence_strength: str
    evidence_refs: tuple[Any, ...]
    candidate_extraction_ref: ArtifactRef

    def __post_init__(self) -> None:
        _require_text(self.global_candidate_ref, "global_candidate_ref")
        _require_text(self.chunk_id, "chunk_id")
        _require_text(self.local_candidate_id, "local_candidate_id")
        expected = f"{self.chunk_id}:{self.local_candidate_id}"
        if self.global_candidate_ref != expected:
            raise ConsolidationModelError(
                f"global_candidate_ref {self.global_candidate_ref!r} is inconsistent "
                f"with chunk_id/local_candidate_id ({expected!r})"
            )
        _require_text(self.source_order_key, "source_order_key")
        _require_enum(self.fact_type, FACT_TYPES, "fact_type")
        _require_text(self.statement_zh, "statement_zh")
        subjects = _require_text_tuple(self.subject_refs, "subject_refs")
        objects = _require_text_or_null_tuple(self.object_refs, "object_refs")
        _require_enum(self.evidence_strength, EVIDENCE_STRENGTHS, "evidence_strength")
        evidence = _to_evidence_tuple(self.evidence_refs, "evidence_refs")
        if not evidence:
            raise ConsolidationModelError("evidence_refs must be non-empty")
        if not isinstance(self.candidate_extraction_ref, ArtifactRef):
            raise ConsolidationModelError("candidate_extraction_ref must be an ArtifactRef")
        object.__setattr__(self, "subject_refs", subjects)
        object.__setattr__(self, "object_refs", objects)
        object.__setattr__(self, "evidence_refs", evidence)

    def to_dict(self) -> dict[str, Any]:
        return {
            "global_candidate_ref": self.global_candidate_ref,
            "chunk_id": self.chunk_id,
            "local_candidate_id": self.local_candidate_id,
            "source_order_key": self.source_order_key,
            "fact_type": self.fact_type,
            "statement_zh": self.statement_zh,
            "subject_refs": list(self.subject_refs),
            "object_refs": list(self.object_refs),
            "evidence_strength": self.evidence_strength,
            "evidence_refs": [e.to_dict() for e in self.evidence_refs],
            "candidate_extraction_ref": self.candidate_extraction_ref.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "IndexedFactCandidate":
        _require_exact_keys(
            value,
            {
                "global_candidate_ref",
                "chunk_id",
                "local_candidate_id",
                "source_order_key",
                "fact_type",
                "statement_zh",
                "subject_refs",
                "object_refs",
                "evidence_strength",
                "evidence_refs",
                "candidate_extraction_ref",
            },
            "IndexedFactCandidate",
        )
        return cls(
            global_candidate_ref=value["global_candidate_ref"],
            chunk_id=value["chunk_id"],
            local_candidate_id=value["local_candidate_id"],
            source_order_key=value["source_order_key"],
            fact_type=value["fact_type"],
            statement_zh=value["statement_zh"],
            subject_refs=tuple(value["subject_refs"]),
            object_refs=tuple(value["object_refs"]),
            evidence_strength=value["evidence_strength"],
            evidence_refs=tuple(value["evidence_refs"]),
            candidate_extraction_ref=ArtifactRef.from_dict(
                value["candidate_extraction_ref"]
            ),
        )


@dataclass(frozen=True, slots=True)
class IndexedEventCandidate:
    """A chunk-local event candidate, indexed for A5 consolidation."""

    global_candidate_ref: str
    chunk_id: str
    local_candidate_id: str
    source_order_key: str
    summary_zh: str
    participants: tuple[str, ...]
    locations: tuple[str, ...]
    temporal_mode: str
    evidence_strength: str
    evidence_refs: tuple[Any, ...]
    candidate_extraction_ref: ArtifactRef

    def __post_init__(self) -> None:
        _require_text(self.global_candidate_ref, "global_candidate_ref")
        _require_text(self.chunk_id, "chunk_id")
        _require_text(self.local_candidate_id, "local_candidate_id")
        expected = f"{self.chunk_id}:{self.local_candidate_id}"
        if self.global_candidate_ref != expected:
            raise ConsolidationModelError(
                f"global_candidate_ref {self.global_candidate_ref!r} is inconsistent "
                f"with chunk_id/local_candidate_id ({expected!r})"
            )
        _require_text(self.source_order_key, "source_order_key")
        _require_text(self.summary_zh, "summary_zh")
        participants = _require_text_tuple(self.participants, "participants")
        if not participants:
            raise ConsolidationModelError("participants must be non-empty")
        locations = _require_text_tuple(self.locations, "locations")
        _require_enum(self.temporal_mode, TEMPORAL_MODES, "temporal_mode")
        _require_enum(self.evidence_strength, EVIDENCE_STRENGTHS, "evidence_strength")
        evidence = _to_evidence_tuple(self.evidence_refs, "evidence_refs")
        if not evidence:
            raise ConsolidationModelError("evidence_refs must be non-empty")
        if not isinstance(self.candidate_extraction_ref, ArtifactRef):
            raise ConsolidationModelError("candidate_extraction_ref must be an ArtifactRef")
        object.__setattr__(self, "participants", participants)
        object.__setattr__(self, "locations", locations)
        object.__setattr__(self, "evidence_refs", evidence)

    def to_dict(self) -> dict[str, Any]:
        return {
            "global_candidate_ref": self.global_candidate_ref,
            "chunk_id": self.chunk_id,
            "local_candidate_id": self.local_candidate_id,
            "source_order_key": self.source_order_key,
            "summary_zh": self.summary_zh,
            "participants": list(self.participants),
            "locations": list(self.locations),
            "temporal_mode": self.temporal_mode,
            "evidence_strength": self.evidence_strength,
            "evidence_refs": [e.to_dict() for e in self.evidence_refs],
            "candidate_extraction_ref": self.candidate_extraction_ref.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "IndexedEventCandidate":
        _require_exact_keys(
            value,
            {
                "global_candidate_ref",
                "chunk_id",
                "local_candidate_id",
                "source_order_key",
                "summary_zh",
                "participants",
                "locations",
                "temporal_mode",
                "evidence_strength",
                "evidence_refs",
                "candidate_extraction_ref",
            },
            "IndexedEventCandidate",
        )
        return cls(
            global_candidate_ref=value["global_candidate_ref"],
            chunk_id=value["chunk_id"],
            local_candidate_id=value["local_candidate_id"],
            source_order_key=value["source_order_key"],
            summary_zh=value["summary_zh"],
            participants=tuple(value["participants"]),
            locations=tuple(value["locations"]),
            temporal_mode=value["temporal_mode"],
            evidence_strength=value["evidence_strength"],
            evidence_refs=tuple(value["evidence_refs"]),
            candidate_extraction_ref=ArtifactRef.from_dict(
                value["candidate_extraction_ref"]
            ),
        )


@dataclass(frozen=True, slots=True)
class IndexedRelationshipCandidate:
    """A chunk-local relationship candidate, indexed for A5 consolidation."""

    global_candidate_ref: str
    chunk_id: str
    local_candidate_id: str
    source_order_key: str
    source_entity_ref: str
    target_entity_ref: str
    relationship_type_zh: str
    state_zh: str | None
    direction: str
    evidence_strength: str
    evidence_refs: tuple[Any, ...]
    candidate_extraction_ref: ArtifactRef

    def __post_init__(self) -> None:
        _require_text(self.global_candidate_ref, "global_candidate_ref")
        _require_text(self.chunk_id, "chunk_id")
        _require_text(self.local_candidate_id, "local_candidate_id")
        expected = f"{self.chunk_id}:{self.local_candidate_id}"
        if self.global_candidate_ref != expected:
            raise ConsolidationModelError(
                f"global_candidate_ref {self.global_candidate_ref!r} is inconsistent "
                f"with chunk_id/local_candidate_id ({expected!r})"
            )
        _require_text(self.source_order_key, "source_order_key")
        _require_bound_entity(self.source_entity_ref, "source_entity_ref")
        _require_bound_entity(self.target_entity_ref, "target_entity_ref")
        _require_text(self.relationship_type_zh, "relationship_type_zh")
        if self.state_zh is not None:
            _require_text(self.state_zh, "state_zh")
        _require_enum(
            self.direction,
            RELATIONSHIP_DIRECTIONS,
            "direction",
        )
        _require_enum(self.evidence_strength, EVIDENCE_STRENGTHS, "evidence_strength")
        evidence = _to_evidence_tuple(self.evidence_refs, "evidence_refs")
        if not evidence:
            raise ConsolidationModelError("evidence_refs must be non-empty")
        if not isinstance(self.candidate_extraction_ref, ArtifactRef):
            raise ConsolidationModelError("candidate_extraction_ref must be an ArtifactRef")
        object.__setattr__(self, "evidence_refs", evidence)

    def to_dict(self) -> dict[str, Any]:
        return {
            "global_candidate_ref": self.global_candidate_ref,
            "chunk_id": self.chunk_id,
            "local_candidate_id": self.local_candidate_id,
            "source_order_key": self.source_order_key,
            "source_entity_ref": self.source_entity_ref,
            "target_entity_ref": self.target_entity_ref,
            "relationship_type_zh": self.relationship_type_zh,
            "state_zh": self.state_zh,
            "direction": self.direction,
            "evidence_strength": self.evidence_strength,
            "evidence_refs": [e.to_dict() for e in self.evidence_refs],
            "candidate_extraction_ref": self.candidate_extraction_ref.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "IndexedRelationshipCandidate":
        _require_exact_keys(
            value,
            {
                "global_candidate_ref",
                "chunk_id",
                "local_candidate_id",
                "source_order_key",
                "source_entity_ref",
                "target_entity_ref",
                "relationship_type_zh",
                "state_zh",
                "direction",
                "evidence_strength",
                "evidence_refs",
                "candidate_extraction_ref",
            },
            "IndexedRelationshipCandidate",
        )
        return cls(
            global_candidate_ref=value["global_candidate_ref"],
            chunk_id=value["chunk_id"],
            local_candidate_id=value["local_candidate_id"],
            source_order_key=value["source_order_key"],
            source_entity_ref=value["source_entity_ref"],
            target_entity_ref=value["target_entity_ref"],
            relationship_type_zh=value["relationship_type_zh"],
            state_zh=value["state_zh"],
            direction=value["direction"],
            evidence_strength=value["evidence_strength"],
            evidence_refs=tuple(value["evidence_refs"]),
            candidate_extraction_ref=ArtifactRef.from_dict(
                value["candidate_extraction_ref"]
            ),
        )


@dataclass(frozen=True, slots=True)
class ConsolidationCandidateIndex:
    """The A5 chunk-local candidate index (facts / events / relationships)."""

    schema_version: int
    facts: tuple[IndexedFactCandidate, ...]
    events: tuple[IndexedEventCandidate, ...]
    relationships: tuple[IndexedRelationshipCandidate, ...]

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ConsolidationModelError(
                f"unsupported schema_version: {self.schema_version!r}"
            )
        for cand in self.facts:
            if not isinstance(cand, IndexedFactCandidate):
                raise ConsolidationModelError("facts must be IndexedFactCandidate")
        for cand in self.events:
            if not isinstance(cand, IndexedEventCandidate):
                raise ConsolidationModelError("events must be IndexedEventCandidate")
        for cand in self.relationships:
            if not isinstance(cand, IndexedRelationshipCandidate):
                raise ConsolidationModelError(
                    "relationships must be IndexedRelationshipCandidate"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "facts": [c.to_dict() for c in self.facts],
            "events": [c.to_dict() for c in self.events],
            "relationships": [c.to_dict() for c in self.relationships],
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ConsolidationCandidateIndex":
        _require_exact_keys(
            value, {"schema_version", "facts", "events", "relationships"},
            "ConsolidationCandidateIndex",
        )
        return cls(
            schema_version=value["schema_version"],
            facts=tuple(IndexedFactCandidate.from_dict(c) for c in value["facts"]),
            events=tuple(IndexedEventCandidate.from_dict(c) for c in value["events"]),
            relationships=tuple(
                IndexedRelationshipCandidate.from_dict(c)
                for c in value["relationships"]
            ),
        )


# ---------------------------------------------------------------------------
# Persisted semantic decisions
# ---------------------------------------------------------------------------

_SEMANTIC_DECISION_KEYS = {
    "decision_id",
    "left_candidate_ref",
    "right_candidate_ref",
    "decision",
    "method",
    "reason_zh",
    "evidence_refs",
    "prompt_id",
    "prompt_version",
    "generation_provenance",
}
_LLM_PROVENANCE_KEYS = {
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


def _validate_semantic_decision_fields(
    *,
    decision_id: object,
    left_candidate_ref: object,
    right_candidate_ref: object,
    decision: object,
    method: object,
    reason_zh: object,
    evidence_refs: object,
    prompt_id: object,
    prompt_version: object,
    generation_provenance: object,
    allowed_decisions: frozenset[str],
    ref_namespace: str,
    domain: str,
) -> tuple[Any, ...]:
    """Validate a persisted semantic decision's fields (fail closed).

    Returns the normalized evidence_refs tuple so the caller can reassign it on
    the frozen instance. Enforces:
      * the pair is in canonical order (left < right);
      * the decision is in the domain's closed set;
      * the method is deterministic / llm / manual;
      * llm decisions carry the exact prompt identity + provenance;
      * deterministic / manual decisions carry null prompt identity + null
        provenance.
    """
    _require_text(decision_id, "decision_id")
    _require_consolidation_candidate_ref(
        left_candidate_ref, "left_candidate_ref", namespace=ref_namespace
    )
    _require_consolidation_candidate_ref(
        right_candidate_ref, "right_candidate_ref", namespace=ref_namespace
    )
    _require_pair_canonical_order(
        left_candidate_ref, right_candidate_ref,
        left_name="left_candidate_ref", right_name="right_candidate_ref",
    )
    _require_enum(decision, allowed_decisions, "decision")
    _require_enum(method, CONSOLIDATION_METHODS, "method")
    _require_text(reason_zh, "reason_zh")
    evidence = _to_evidence_tuple(evidence_refs, "evidence_refs")
    provenance = _to_llm_provenance(generation_provenance, "generation_provenance")
    if method == "llm":
        if not isinstance(prompt_id, str) or not prompt_id:
            raise ConsolidationModelError(f"llm {domain} decision requires prompt_id")
        if not isinstance(prompt_version, int) or isinstance(prompt_version, bool) or prompt_version < 1:
            raise ConsolidationModelError(
                f"llm {domain} decision requires a positive prompt_version"
            )
        if provenance is None:
            raise ConsolidationModelError(
                f"llm {domain} decision requires generation_provenance"
            )
    else:
        if prompt_id is not None:
            raise ConsolidationModelError(
                f"{domain} {method} decision must have null prompt_id"
            )
        if prompt_version is not None:
            raise ConsolidationModelError(
                f"{domain} {method} decision must have null prompt_version"
            )
        if provenance is not None:
            raise ConsolidationModelError(
                f"{domain} {method} decision must have null generation_provenance"
            )
    return evidence


def _semantic_decision_to_dict(
    *,
    decision_id: str,
    left_candidate_ref: str,
    right_candidate_ref: str,
    decision: str,
    method: str,
    reason_zh: str,
    evidence_refs: tuple[Any, ...],
    prompt_id: str | None,
    prompt_version: int | None,
    generation_provenance: LLMInvocationProvenance | None,
) -> dict[str, Any]:
    return {
        "decision_id": decision_id,
        "left_candidate_ref": left_candidate_ref,
        "right_candidate_ref": right_candidate_ref,
        "decision": decision,
        "method": method,
        "reason_zh": reason_zh,
        "evidence_refs": [e.to_dict() for e in evidence_refs],
        "prompt_id": prompt_id,
        "prompt_version": prompt_version,
        "generation_provenance": (
            generation_provenance.to_dict() if generation_provenance is not None else None
        ),
    }


@dataclass(frozen=True, slots=True)
class FactSemanticDecision:
    """A persisted fact-pair semantic decision."""

    decision_id: str
    left_candidate_ref: str
    right_candidate_ref: str
    decision: str
    method: str
    reason_zh: str
    evidence_refs: tuple[Any, ...]
    prompt_id: str | None
    prompt_version: int | None
    generation_provenance: LLMInvocationProvenance | None

    def __post_init__(self) -> None:
        evidence = _validate_semantic_decision_fields(
            decision_id=self.decision_id,
            left_candidate_ref=self.left_candidate_ref,
            right_candidate_ref=self.right_candidate_ref,
            decision=self.decision,
            method=self.method,
            reason_zh=self.reason_zh,
            evidence_refs=self.evidence_refs,
            prompt_id=self.prompt_id,
            prompt_version=self.prompt_version,
            generation_provenance=self.generation_provenance,
            allowed_decisions=FACT_DECISIONS,
            ref_namespace="fact",
            domain="fact",
        )
        object.__setattr__(self, "evidence_refs", evidence)

    def to_dict(self) -> dict[str, Any]:
        return _semantic_decision_to_dict(
            decision_id=self.decision_id,
            left_candidate_ref=self.left_candidate_ref,
            right_candidate_ref=self.right_candidate_ref,
            decision=self.decision,
            method=self.method,
            reason_zh=self.reason_zh,
            evidence_refs=self.evidence_refs,
            prompt_id=self.prompt_id,
            prompt_version=self.prompt_version,
            generation_provenance=self.generation_provenance,
        )

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "FactSemanticDecision":
        return _semantic_decision_from_dict(cls, value, "FactSemanticDecision")


@dataclass(frozen=True, slots=True)
class EventSemanticDecision:
    """A persisted event-pair semantic decision."""

    decision_id: str
    left_candidate_ref: str
    right_candidate_ref: str
    decision: str
    method: str
    reason_zh: str
    evidence_refs: tuple[Any, ...]
    prompt_id: str | None
    prompt_version: int | None
    generation_provenance: LLMInvocationProvenance | None

    def __post_init__(self) -> None:
        evidence = _validate_semantic_decision_fields(
            decision_id=self.decision_id,
            left_candidate_ref=self.left_candidate_ref,
            right_candidate_ref=self.right_candidate_ref,
            decision=self.decision,
            method=self.method,
            reason_zh=self.reason_zh,
            evidence_refs=self.evidence_refs,
            prompt_id=self.prompt_id,
            prompt_version=self.prompt_version,
            generation_provenance=self.generation_provenance,
            allowed_decisions=EVENT_DECISIONS,
            ref_namespace="event",
            domain="event",
        )
        object.__setattr__(self, "evidence_refs", evidence)

    def to_dict(self) -> dict[str, Any]:
        return _semantic_decision_to_dict(
            decision_id=self.decision_id,
            left_candidate_ref=self.left_candidate_ref,
            right_candidate_ref=self.right_candidate_ref,
            decision=self.decision,
            method=self.method,
            reason_zh=self.reason_zh,
            evidence_refs=self.evidence_refs,
            prompt_id=self.prompt_id,
            prompt_version=self.prompt_version,
            generation_provenance=self.generation_provenance,
        )

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "EventSemanticDecision":
        return _semantic_decision_from_dict(cls, value, "EventSemanticDecision")


@dataclass(frozen=True, slots=True)
class RelationshipSemanticDecision:
    """A persisted relationship-pair semantic decision."""

    decision_id: str
    left_candidate_ref: str
    right_candidate_ref: str
    decision: str
    method: str
    reason_zh: str
    evidence_refs: tuple[Any, ...]
    prompt_id: str | None
    prompt_version: int | None
    generation_provenance: LLMInvocationProvenance | None

    def __post_init__(self) -> None:
        evidence = _validate_semantic_decision_fields(
            decision_id=self.decision_id,
            left_candidate_ref=self.left_candidate_ref,
            right_candidate_ref=self.right_candidate_ref,
            decision=self.decision,
            method=self.method,
            reason_zh=self.reason_zh,
            evidence_refs=self.evidence_refs,
            prompt_id=self.prompt_id,
            prompt_version=self.prompt_version,
            generation_provenance=self.generation_provenance,
            allowed_decisions=RELATIONSHIP_DECISIONS,
            ref_namespace="relationship",
            domain="relationship",
        )
        object.__setattr__(self, "evidence_refs", evidence)

    def to_dict(self) -> dict[str, Any]:
        return _semantic_decision_to_dict(
            decision_id=self.decision_id,
            left_candidate_ref=self.left_candidate_ref,
            right_candidate_ref=self.right_candidate_ref,
            decision=self.decision,
            method=self.method,
            reason_zh=self.reason_zh,
            evidence_refs=self.evidence_refs,
            prompt_id=self.prompt_id,
            prompt_version=self.prompt_version,
            generation_provenance=self.generation_provenance,
        )

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RelationshipSemanticDecision":
        return _semantic_decision_from_dict(cls, value, "RelationshipSemanticDecision")


def _semantic_decision_from_dict(
    cls: type, value: dict[str, Any], name: str
) -> Any:
    _require_exact_keys(value, _SEMANTIC_DECISION_KEYS, name)
    if not isinstance(value["evidence_refs"], list):
        raise ConsolidationModelError(f"{name}.evidence_refs must be a list")
    prov = value["generation_provenance"]
    if prov is None:
        generation_provenance = None
    else:
        if not isinstance(prov, dict) or set(prov) != _LLM_PROVENANCE_KEYS:
            raise ConsolidationModelError(
                f"{name}.generation_provenance must contain exactly the A-I3 "
                "LLMInvocationProvenance fields"
            )
        try:
            generation_provenance = LLMInvocationProvenance(**prov)
        except (TypeError, ValueError) as exc:
            raise ConsolidationModelError(f"invalid generation_provenance: {exc}") from exc
    return cls(
        decision_id=value["decision_id"],
        left_candidate_ref=value["left_candidate_ref"],
        right_candidate_ref=value["right_candidate_ref"],
        decision=value["decision"],
        method=value["method"],
        reason_zh=value["reason_zh"],
        evidence_refs=tuple(value["evidence_refs"]),
        prompt_id=value["prompt_id"],
        prompt_version=value["prompt_version"],
        generation_provenance=generation_provenance,
    )


@dataclass(frozen=True, slots=True)
class ConsolidationDecisionSet:
    """The A5 semantic decision set (fact / event / relationship decisions)."""

    schema_version: int
    fact_decisions: tuple[FactSemanticDecision, ...]
    event_decisions: tuple[EventSemanticDecision, ...]
    relationship_decisions: tuple[RelationshipSemanticDecision, ...]

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ConsolidationModelError(
                f"unsupported schema_version: {self.schema_version!r}"
            )
        for d in self.fact_decisions:
            if not isinstance(d, FactSemanticDecision):
                raise ConsolidationModelError(
                    "fact_decisions must be FactSemanticDecision"
                )
        for d in self.event_decisions:
            if not isinstance(d, EventSemanticDecision):
                raise ConsolidationModelError(
                    "event_decisions must be EventSemanticDecision"
                )
        for d in self.relationship_decisions:
            if not isinstance(d, RelationshipSemanticDecision):
                raise ConsolidationModelError(
                    "relationship_decisions must be RelationshipSemanticDecision"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "fact_decisions": [d.to_dict() for d in self.fact_decisions],
            "event_decisions": [d.to_dict() for d in self.event_decisions],
            "relationship_decisions": [
                d.to_dict() for d in self.relationship_decisions
            ],
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ConsolidationDecisionSet":
        _require_exact_keys(
            value,
            {"schema_version", "fact_decisions", "event_decisions", "relationship_decisions"},
            "ConsolidationDecisionSet",
        )
        return cls(
            schema_version=value["schema_version"],
            fact_decisions=tuple(
                FactSemanticDecision.from_dict(d) for d in value["fact_decisions"]
            ),
            event_decisions=tuple(
                EventSemanticDecision.from_dict(d) for d in value["event_decisions"]
            ),
            relationship_decisions=tuple(
                RelationshipSemanticDecision.from_dict(d)
                for d in value["relationship_decisions"]
            ),
        )


# ---------------------------------------------------------------------------
# Provider selector payloads
# ---------------------------------------------------------------------------


def _validate_selector_item_fields(
    *,
    left_candidate_ref: object,
    right_candidate_ref: object,
    decision: object,
    reason_zh: object,
    evidence_selectors: object,
    allowed_decisions: frozenset[str],
    ref_namespace: str,
) -> tuple[str, ...]:
    _require_consolidation_candidate_ref(
        left_candidate_ref, "left_candidate_ref", namespace=ref_namespace
    )
    _require_consolidation_candidate_ref(
        right_candidate_ref, "right_candidate_ref", namespace=ref_namespace
    )
    _require_pair_canonical_order(
        left_candidate_ref, right_candidate_ref,
        left_name="left_candidate_ref", right_name="right_candidate_ref",
    )
    _require_enum(decision, allowed_decisions, "decision")
    _require_text(reason_zh, "reason_zh")
    selectors = _require_text_tuple(
        evidence_selectors, "evidence_selectors", allow_empty=True
    )
    return selectors


def _selector_item_to_dict(
    *,
    left_candidate_ref: str,
    right_candidate_ref: str,
    decision: str,
    reason_zh: str,
    evidence_selectors: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "left_candidate_ref": left_candidate_ref,
        "right_candidate_ref": right_candidate_ref,
        "decision": decision,
        "reason_zh": reason_zh,
        "evidence_selectors": list(evidence_selectors),
    }


def _selector_item_from_dict(
    cls: type, value: dict[str, Any], name: str
) -> Any:
    _require_exact_keys(
        value,
        {
            "left_candidate_ref",
            "right_candidate_ref",
            "decision",
            "reason_zh",
            "evidence_selectors",
        },
        name,
    )
    return cls(
        left_candidate_ref=value["left_candidate_ref"],
        right_candidate_ref=value["right_candidate_ref"],
        decision=value["decision"],
        reason_zh=value["reason_zh"],
        evidence_selectors=tuple(value["evidence_selectors"]),
    )


@dataclass(frozen=True, slots=True)
class FactSelectorDecisionItem:
    """One fact-pair selector decision (provider output)."""

    left_candidate_ref: str
    right_candidate_ref: str
    decision: str
    reason_zh: str
    evidence_selectors: tuple[str, ...]

    def __post_init__(self) -> None:
        selectors = _validate_selector_item_fields(
            left_candidate_ref=self.left_candidate_ref,
            right_candidate_ref=self.right_candidate_ref,
            decision=self.decision,
            reason_zh=self.reason_zh,
            evidence_selectors=self.evidence_selectors,
            allowed_decisions=FACT_DECISIONS,
            ref_namespace="fact",
        )
        object.__setattr__(self, "evidence_selectors", selectors)

    def to_dict(self) -> dict[str, Any]:
        return _selector_item_to_dict(
            left_candidate_ref=self.left_candidate_ref,
            right_candidate_ref=self.right_candidate_ref,
            decision=self.decision,
            reason_zh=self.reason_zh,
            evidence_selectors=self.evidence_selectors,
        )

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "FactSelectorDecisionItem":
        return _selector_item_from_dict(cls, value, "FactSelectorDecisionItem")


@dataclass(frozen=True, slots=True)
class EventSelectorDecisionItem:
    """One event-pair selector decision (provider output)."""

    left_candidate_ref: str
    right_candidate_ref: str
    decision: str
    reason_zh: str
    evidence_selectors: tuple[str, ...]

    def __post_init__(self) -> None:
        selectors = _validate_selector_item_fields(
            left_candidate_ref=self.left_candidate_ref,
            right_candidate_ref=self.right_candidate_ref,
            decision=self.decision,
            reason_zh=self.reason_zh,
            evidence_selectors=self.evidence_selectors,
            allowed_decisions=EVENT_DECISIONS,
            ref_namespace="event",
        )
        object.__setattr__(self, "evidence_selectors", selectors)

    def to_dict(self) -> dict[str, Any]:
        return _selector_item_to_dict(
            left_candidate_ref=self.left_candidate_ref,
            right_candidate_ref=self.right_candidate_ref,
            decision=self.decision,
            reason_zh=self.reason_zh,
            evidence_selectors=self.evidence_selectors,
        )

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "EventSelectorDecisionItem":
        return _selector_item_from_dict(cls, value, "EventSelectorDecisionItem")


@dataclass(frozen=True, slots=True)
class RelationshipSelectorDecisionItem:
    """One relationship-pair selector decision (provider output)."""

    left_candidate_ref: str
    right_candidate_ref: str
    decision: str
    reason_zh: str
    evidence_selectors: tuple[str, ...]

    def __post_init__(self) -> None:
        selectors = _validate_selector_item_fields(
            left_candidate_ref=self.left_candidate_ref,
            right_candidate_ref=self.right_candidate_ref,
            decision=self.decision,
            reason_zh=self.reason_zh,
            evidence_selectors=self.evidence_selectors,
            allowed_decisions=RELATIONSHIP_DECISIONS,
            ref_namespace="relationship",
        )
        object.__setattr__(self, "evidence_selectors", selectors)

    def to_dict(self) -> dict[str, Any]:
        return _selector_item_to_dict(
            left_candidate_ref=self.left_candidate_ref,
            right_candidate_ref=self.right_candidate_ref,
            decision=self.decision,
            reason_zh=self.reason_zh,
            evidence_selectors=self.evidence_selectors,
        )

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RelationshipSelectorDecisionItem":
        return _selector_item_from_dict(cls, value, "RelationshipSelectorDecisionItem")


@dataclass(frozen=True, slots=True)
class FactSelectorDecisionPayload:
    """Provider output: the fact selector decision payload."""

    decisions: tuple[FactSelectorDecisionItem, ...]

    def __post_init__(self) -> None:
        for d in self.decisions:
            if not isinstance(d, FactSelectorDecisionItem):
                raise ConsolidationModelError(
                    "decisions must be FactSelectorDecisionItem"
                )

    def to_dict(self) -> dict[str, Any]:
        return {"decisions": [d.to_dict() for d in self.decisions]}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "FactSelectorDecisionPayload":
        _require_exact_keys(value, {"decisions"}, "FactSelectorDecisionPayload")
        return cls(
            decisions=tuple(
                FactSelectorDecisionItem.from_dict(d) for d in value["decisions"]
            )
        )


@dataclass(frozen=True, slots=True)
class EventSelectorDecisionPayload:
    """Provider output: the event selector decision payload."""

    decisions: tuple[EventSelectorDecisionItem, ...]

    def __post_init__(self) -> None:
        for d in self.decisions:
            if not isinstance(d, EventSelectorDecisionItem):
                raise ConsolidationModelError(
                    "decisions must be EventSelectorDecisionItem"
                )

    def to_dict(self) -> dict[str, Any]:
        return {"decisions": [d.to_dict() for d in self.decisions]}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "EventSelectorDecisionPayload":
        _require_exact_keys(value, {"decisions"}, "EventSelectorDecisionPayload")
        return cls(
            decisions=tuple(
                EventSelectorDecisionItem.from_dict(d) for d in value["decisions"]
            )
        )


@dataclass(frozen=True, slots=True)
class RelationshipSelectorDecisionPayload:
    """Provider output: the relationship selector decision payload."""

    decisions: tuple[RelationshipSelectorDecisionItem, ...]

    def __post_init__(self) -> None:
        for d in self.decisions:
            if not isinstance(d, RelationshipSelectorDecisionItem):
                raise ConsolidationModelError(
                    "decisions must be RelationshipSelectorDecisionItem"
                )

    def to_dict(self) -> dict[str, Any]:
        return {"decisions": [d.to_dict() for d in self.decisions]}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RelationshipSelectorDecisionPayload":
        _require_exact_keys(
            value, {"decisions"}, "RelationshipSelectorDecisionPayload"
        )
        return cls(
            decisions=tuple(
                RelationshipSelectorDecisionItem.from_dict(d)
                for d in value["decisions"]
            )
        )


# ---------------------------------------------------------------------------
# Canonical facts + state transitions
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CanonicalFact:
    """A canonical (consolidated) fact."""

    fact_id: str
    fact_type: str
    statement_zh: str
    subject_refs: tuple[str, ...]
    object_refs: tuple[str | None, ...]
    candidate_fact_refs: tuple[str, ...]
    evidence_refs: tuple[Any, ...]
    first_source_order: str
    continuity_relevant: bool

    def __post_init__(self) -> None:
        _require_id_pattern(self.fact_id, _FACT_ID_RE, "fact_id")
        _require_enum(self.fact_type, FACT_TYPES, "fact_type")
        _require_text(self.statement_zh, "statement_zh")
        subjects = _require_text_tuple(self.subject_refs, "subject_refs")
        objects = _require_text_or_null_tuple(self.object_refs, "object_refs")
        candidate_refs = _require_text_tuple(
            self.candidate_fact_refs, "candidate_fact_refs", allow_empty=False
        )
        for ref in candidate_refs:
            ConsolidationCandidateRef.parse(ref)
        evidence = _to_evidence_tuple(self.evidence_refs, "evidence_refs")
        _require_text(self.first_source_order, "first_source_order")
        _require_bool(self.continuity_relevant, "continuity_relevant")
        object.__setattr__(self, "subject_refs", subjects)
        object.__setattr__(self, "object_refs", objects)
        object.__setattr__(self, "candidate_fact_refs", candidate_refs)
        object.__setattr__(self, "evidence_refs", evidence)

    def to_dict(self) -> dict[str, Any]:
        return {
            "fact_id": self.fact_id,
            "fact_type": self.fact_type,
            "statement_zh": self.statement_zh,
            "subject_refs": list(self.subject_refs),
            "object_refs": list(self.object_refs),
            "candidate_fact_refs": list(self.candidate_fact_refs),
            "evidence_refs": [e.to_dict() for e in self.evidence_refs],
            "first_source_order": self.first_source_order,
            "continuity_relevant": self.continuity_relevant,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CanonicalFact":
        _require_exact_keys(
            value,
            {
                "fact_id",
                "fact_type",
                "statement_zh",
                "subject_refs",
                "object_refs",
                "candidate_fact_refs",
                "evidence_refs",
                "first_source_order",
                "continuity_relevant",
            },
            "CanonicalFact",
        )
        return cls(
            fact_id=value["fact_id"],
            fact_type=value["fact_type"],
            statement_zh=value["statement_zh"],
            subject_refs=tuple(value["subject_refs"]),
            object_refs=tuple(value["object_refs"]),
            candidate_fact_refs=tuple(value["candidate_fact_refs"]),
            evidence_refs=tuple(value["evidence_refs"]),
            first_source_order=value["first_source_order"],
            continuity_relevant=value["continuity_relevant"],
        )


@dataclass(frozen=True, slots=True)
class StateTransition:
    """A canonical state transition between two continuity-relevant facts."""

    transition_id: str
    from_fact_id: str
    to_fact_id: str
    subject_refs: tuple[str, ...]
    transition_kind: str
    source_decision_ref: str
    evidence_refs: tuple[Any, ...]
    narrative_order: int

    def __post_init__(self) -> None:
        _require_id_pattern(self.transition_id, _TRANSITION_ID_RE, "transition_id")
        _require_id_pattern(self.from_fact_id, _FACT_ID_RE, "from_fact_id")
        _require_id_pattern(self.to_fact_id, _FACT_ID_RE, "to_fact_id")
        subjects = _require_text_tuple(self.subject_refs, "subject_refs")
        _require_enum(self.transition_kind, STATE_TRANSITION_KINDS, "transition_kind")
        _require_text(self.source_decision_ref, "source_decision_ref")
        evidence = _to_evidence_tuple(self.evidence_refs, "evidence_refs")
        _require_positive_int(self.narrative_order, "narrative_order")
        object.__setattr__(self, "subject_refs", subjects)
        object.__setattr__(self, "evidence_refs", evidence)

    def to_dict(self) -> dict[str, Any]:
        return {
            "transition_id": self.transition_id,
            "from_fact_id": self.from_fact_id,
            "to_fact_id": self.to_fact_id,
            "subject_refs": list(self.subject_refs),
            "transition_kind": self.transition_kind,
            "source_decision_ref": self.source_decision_ref,
            "evidence_refs": [e.to_dict() for e in self.evidence_refs],
            "narrative_order": self.narrative_order,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "StateTransition":
        _require_exact_keys(
            value,
            {
                "transition_id",
                "from_fact_id",
                "to_fact_id",
                "subject_refs",
                "transition_kind",
                "source_decision_ref",
                "evidence_refs",
                "narrative_order",
            },
            "StateTransition",
        )
        return cls(
            transition_id=value["transition_id"],
            from_fact_id=value["from_fact_id"],
            to_fact_id=value["to_fact_id"],
            subject_refs=tuple(value["subject_refs"]),
            transition_kind=value["transition_kind"],
            source_decision_ref=value["source_decision_ref"],
            evidence_refs=tuple(value["evidence_refs"]),
            narrative_order=value["narrative_order"],
        )


@dataclass(frozen=True, slots=True)
class CanonicalFactSet:
    """The canonical fact set (with state transitions)."""

    schema_version: int
    facts: tuple[CanonicalFact, ...]
    state_transitions: tuple[StateTransition, ...]

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ConsolidationModelError(
                f"unsupported schema_version: {self.schema_version!r}"
            )
        for f in self.facts:
            if not isinstance(f, CanonicalFact):
                raise ConsolidationModelError("facts must be CanonicalFact")
        for t in self.state_transitions:
            if not isinstance(t, StateTransition):
                raise ConsolidationModelError(
                    "state_transitions must be StateTransition"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "facts": [f.to_dict() for f in self.facts],
            "state_transitions": [t.to_dict() for t in self.state_transitions],
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CanonicalFactSet":
        _require_exact_keys(
            value, {"schema_version", "facts", "state_transitions"},
            "CanonicalFactSet",
        )
        return cls(
            schema_version=value["schema_version"],
            facts=tuple(CanonicalFact.from_dict(f) for f in value["facts"]),
            state_transitions=tuple(
                StateTransition.from_dict(t) for t in value["state_transitions"]
            ),
        )


# ---------------------------------------------------------------------------
# Canonical events
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CanonicalEvent:
    """A canonical (consolidated) event."""

    event_id: str
    narrative_order: int
    summary_zh: str
    participants: tuple[str, ...]
    locations: tuple[str, ...]
    temporal_mode: str
    candidate_event_refs: tuple[str, ...]
    evidence_refs: tuple[Any, ...]
    first_source_order: str

    def __post_init__(self) -> None:
        _require_id_pattern(self.event_id, _EVENT_ID_RE, "event_id")
        _require_positive_int(self.narrative_order, "narrative_order")
        _require_text(self.summary_zh, "summary_zh")
        participants = _require_text_tuple(self.participants, "participants")
        if not participants:
            raise ConsolidationModelError("participants must be non-empty")
        locations = _require_text_tuple(self.locations, "locations")
        _require_enum(self.temporal_mode, TEMPORAL_MODES, "temporal_mode")
        candidate_refs = _require_text_tuple(
            self.candidate_event_refs, "candidate_event_refs", allow_empty=False
        )
        for ref in candidate_refs:
            ConsolidationCandidateRef.parse(ref)
        evidence = _to_evidence_tuple(self.evidence_refs, "evidence_refs")
        _require_text(self.first_source_order, "first_source_order")
        object.__setattr__(self, "participants", participants)
        object.__setattr__(self, "locations", locations)
        object.__setattr__(self, "candidate_event_refs", candidate_refs)
        object.__setattr__(self, "evidence_refs", evidence)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "narrative_order": self.narrative_order,
            "summary_zh": self.summary_zh,
            "participants": list(self.participants),
            "locations": list(self.locations),
            "temporal_mode": self.temporal_mode,
            "candidate_event_refs": list(self.candidate_event_refs),
            "evidence_refs": [e.to_dict() for e in self.evidence_refs],
            "first_source_order": self.first_source_order,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CanonicalEvent":
        _require_exact_keys(
            value,
            {
                "event_id",
                "narrative_order",
                "summary_zh",
                "participants",
                "locations",
                "temporal_mode",
                "candidate_event_refs",
                "evidence_refs",
                "first_source_order",
            },
            "CanonicalEvent",
        )
        return cls(
            event_id=value["event_id"],
            narrative_order=value["narrative_order"],
            summary_zh=value["summary_zh"],
            participants=tuple(value["participants"]),
            locations=tuple(value["locations"]),
            temporal_mode=value["temporal_mode"],
            candidate_event_refs=tuple(value["candidate_event_refs"]),
            evidence_refs=tuple(value["evidence_refs"]),
            first_source_order=value["first_source_order"],
        )


@dataclass(frozen=True, slots=True)
class CanonicalEventSet:
    """The canonical event set."""

    schema_version: int
    events: tuple[CanonicalEvent, ...]

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ConsolidationModelError(
                f"unsupported schema_version: {self.schema_version!r}"
            )
        for e in self.events:
            if not isinstance(e, CanonicalEvent):
                raise ConsolidationModelError("events must be CanonicalEvent")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "events": [e.to_dict() for e in self.events],
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CanonicalEventSet":
        _require_exact_keys(value, {"schema_version", "events"}, "CanonicalEventSet")
        return cls(
            schema_version=value["schema_version"],
            events=tuple(CanonicalEvent.from_dict(e) for e in value["events"]),
        )


# ---------------------------------------------------------------------------
# Canonical relationships + state history
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RelationshipState:
    """One point in a relationship's ordered state history."""

    state_zh: str
    candidate_relationship_refs: tuple[str, ...]
    evidence_refs: tuple[Any, ...]
    narrative_order: int

    def __post_init__(self) -> None:
        _require_text(self.state_zh, "state_zh")
        candidate_refs = _require_text_tuple(
            self.candidate_relationship_refs,
            "candidate_relationship_refs",
            allow_empty=False,
        )
        for ref in candidate_refs:
            ConsolidationCandidateRef.parse(ref)
        evidence = _to_evidence_tuple(self.evidence_refs, "evidence_refs")
        _require_positive_int(self.narrative_order, "narrative_order")
        object.__setattr__(self, "candidate_relationship_refs", candidate_refs)
        object.__setattr__(self, "evidence_refs", evidence)

    def to_dict(self) -> dict[str, Any]:
        return {
            "state_zh": self.state_zh,
            "candidate_relationship_refs": list(self.candidate_relationship_refs),
            "evidence_refs": [e.to_dict() for e in self.evidence_refs],
            "narrative_order": self.narrative_order,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RelationshipState":
        _require_exact_keys(
            value,
            {
                "state_zh",
                "candidate_relationship_refs",
                "evidence_refs",
                "narrative_order",
            },
            "RelationshipState",
        )
        return cls(
            state_zh=value["state_zh"],
            candidate_relationship_refs=tuple(value["candidate_relationship_refs"]),
            evidence_refs=tuple(value["evidence_refs"]),
            narrative_order=value["narrative_order"],
        )


@dataclass(frozen=True, slots=True)
class CanonicalRelationship:
    """A canonical (consolidated) relationship with ordered state history."""

    relationship_id: str
    source_entity_ref: str
    target_entity_ref: str
    direction: str
    relationship_type_zh: str
    candidate_relationship_refs: tuple[str, ...]
    state_history: tuple[RelationshipState, ...]
    first_source_order: str

    def __post_init__(self) -> None:
        _require_id_pattern(
            self.relationship_id, _REL_ID_RE, "relationship_id"
        )
        _require_bound_entity(self.source_entity_ref, "source_entity_ref")
        _require_bound_entity(self.target_entity_ref, "target_entity_ref")
        _require_enum(
            self.direction,
            RELATIONSHIP_DIRECTIONS,
            "direction",
        )
        _require_text(self.relationship_type_zh, "relationship_type_zh")
        candidate_refs = _require_text_tuple(
            self.candidate_relationship_refs,
            "candidate_relationship_refs",
            allow_empty=False,
        )
        for ref in candidate_refs:
            ConsolidationCandidateRef.parse(ref)
        for state in self.state_history:
            if not isinstance(state, RelationshipState):
                raise ConsolidationModelError(
                    "state_history must be RelationshipState"
                )
        _require_text(self.first_source_order, "first_source_order")
        object.__setattr__(self, "candidate_relationship_refs", candidate_refs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "relationship_id": self.relationship_id,
            "source_entity_ref": self.source_entity_ref,
            "target_entity_ref": self.target_entity_ref,
            "direction": self.direction,
            "relationship_type_zh": self.relationship_type_zh,
            "candidate_relationship_refs": list(self.candidate_relationship_refs),
            "state_history": [s.to_dict() for s in self.state_history],
            "first_source_order": self.first_source_order,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CanonicalRelationship":
        _require_exact_keys(
            value,
            {
                "relationship_id",
                "source_entity_ref",
                "target_entity_ref",
                "direction",
                "relationship_type_zh",
                "candidate_relationship_refs",
                "state_history",
                "first_source_order",
            },
            "CanonicalRelationship",
        )
        return cls(
            relationship_id=value["relationship_id"],
            source_entity_ref=value["source_entity_ref"],
            target_entity_ref=value["target_entity_ref"],
            direction=value["direction"],
            relationship_type_zh=value["relationship_type_zh"],
            candidate_relationship_refs=tuple(value["candidate_relationship_refs"]),
            state_history=tuple(
                RelationshipState.from_dict(s) for s in value["state_history"]
            ),
            first_source_order=value["first_source_order"],
        )


@dataclass(frozen=True, slots=True)
class CanonicalRelationshipSet:
    """The canonical relationship set."""

    schema_version: int
    relationships: tuple[CanonicalRelationship, ...]

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ConsolidationModelError(
                f"unsupported schema_version: {self.schema_version!r}"
            )
        for r in self.relationships:
            if not isinstance(r, CanonicalRelationship):
                raise ConsolidationModelError(
                    "relationships must be CanonicalRelationship"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "relationships": [r.to_dict() for r in self.relationships],
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CanonicalRelationshipSet":
        _require_exact_keys(
            value, {"schema_version", "relationships"}, "CanonicalRelationshipSet"
        )
        return cls(
            schema_version=value["schema_version"],
            relationships=tuple(
                CanonicalRelationship.from_dict(r) for r in value["relationships"]
            ),
        )


# ---------------------------------------------------------------------------
# Story conflicts
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StoryConflict:
    """An unresolved story-level conflict (facts / relationship states)."""

    conflict_id: str
    conflict_kind: str
    fact_ids: tuple[str, ...]
    relationship_ids: tuple[str, ...]
    candidate_refs: tuple[str, ...]
    decision_refs: tuple[str, ...]
    evidence_refs: tuple[Any, ...]
    status: str

    def __post_init__(self) -> None:
        _require_id_pattern(self.conflict_id, _CONFLICT_ID_RE, "conflict_id")
        _require_enum(self.conflict_kind, STORY_CONFLICT_KINDS, "conflict_kind")
        fact_ids = _require_text_tuple(self.fact_ids, "fact_ids")
        for fid in fact_ids:
            _require_id_pattern(fid, _FACT_ID_RE, "fact_ids[]")
        relationship_ids = _require_text_tuple(self.relationship_ids, "relationship_ids")
        for rid in relationship_ids:
            _require_id_pattern(rid, _REL_ID_RE, "relationship_ids[]")
        candidate_refs = _require_text_tuple(
            self.candidate_refs, "candidate_refs", allow_empty=False
        )
        for ref in candidate_refs:
            ConsolidationCandidateRef.parse(ref)
        decision_refs = _require_text_tuple(self.decision_refs, "decision_refs")
        evidence = _to_evidence_tuple(self.evidence_refs, "evidence_refs")
        _require_enum(self.status, STORY_CONFLICT_STATUSES, "status")
        object.__setattr__(self, "fact_ids", fact_ids)
        object.__setattr__(self, "relationship_ids", relationship_ids)
        object.__setattr__(self, "candidate_refs", candidate_refs)
        object.__setattr__(self, "decision_refs", decision_refs)
        object.__setattr__(self, "evidence_refs", evidence)

    def to_dict(self) -> dict[str, Any]:
        return {
            "conflict_id": self.conflict_id,
            "conflict_kind": self.conflict_kind,
            "fact_ids": list(self.fact_ids),
            "relationship_ids": list(self.relationship_ids),
            "candidate_refs": list(self.candidate_refs),
            "decision_refs": list(self.decision_refs),
            "evidence_refs": [e.to_dict() for e in self.evidence_refs],
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "StoryConflict":
        _require_exact_keys(
            value,
            {
                "conflict_id",
                "conflict_kind",
                "fact_ids",
                "relationship_ids",
                "candidate_refs",
                "decision_refs",
                "evidence_refs",
                "status",
            },
            "StoryConflict",
        )
        return cls(
            conflict_id=value["conflict_id"],
            conflict_kind=value["conflict_kind"],
            fact_ids=tuple(value["fact_ids"]),
            relationship_ids=tuple(value["relationship_ids"]),
            candidate_refs=tuple(value["candidate_refs"]),
            decision_refs=tuple(value["decision_refs"]),
            evidence_refs=tuple(value["evidence_refs"]),
            status=value["status"],
        )


@dataclass(frozen=True, slots=True)
class StoryConflictSet:
    """The story conflict set (A5 records only unresolved conflicts)."""

    schema_version: int
    conflicts: tuple[StoryConflict, ...]

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ConsolidationModelError(
                f"unsupported schema_version: {self.schema_version!r}"
            )
        for c in self.conflicts:
            if not isinstance(c, StoryConflict):
                raise ConsolidationModelError("conflicts must be StoryConflict")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "conflicts": [c.to_dict() for c in self.conflicts],
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "StoryConflictSet":
        _require_exact_keys(
            value, {"schema_version", "conflicts"}, "StoryConflictSet"
        )
        return cls(
            schema_version=value["schema_version"],
            conflicts=tuple(StoryConflict.from_dict(c) for c in value["conflicts"]),
        )


# ---------------------------------------------------------------------------
# Semantic identity / upstream identity / coverage / manifest
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PromptAssetIdentity:
    """Exact identity of a tracked prompt asset."""

    prompt_id: str
    prompt_version: int
    prompt_content_hash: str

    def __post_init__(self) -> None:
        _require_text(self.prompt_id, "prompt_id")
        _require_positive_int(self.prompt_version, "prompt_version")
        _require_sha256(self.prompt_content_hash, "prompt_content_hash")

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_id": self.prompt_id,
            "prompt_version": self.prompt_version,
            "prompt_content_hash": self.prompt_content_hash,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "PromptAssetIdentity":
        _require_exact_keys(
            value,
            {"prompt_id", "prompt_version", "prompt_content_hash"},
            "PromptAssetIdentity",
        )
        return cls(**value)


@dataclass(frozen=True, slots=True)
class OutputSchemaAssetIdentity:
    """Exact identity of a tracked output-schema asset."""

    schema_id: str
    schema_version: int
    schema_hash: str

    def __post_init__(self) -> None:
        _require_text(self.schema_id, "schema_id")
        _require_positive_int(self.schema_version, "schema_version")
        _require_sha256(self.schema_hash, "schema_hash")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_id": self.schema_id,
            "schema_version": self.schema_version,
            "schema_hash": self.schema_hash,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "OutputSchemaAssetIdentity":
        _require_exact_keys(
            value, {"schema_id", "schema_version", "schema_hash"},
            "OutputSchemaAssetIdentity",
        )
        return cls(**value)


@dataclass(frozen=True, slots=True)
class A5SemanticIdentity:
    """Backend-neutral A5 semantic identity (like A4).

    Pins the consolidation profile id/hash, the per-domain semantic LLM profile
    ids/hashes, the exact prompt / output-schema asset identities, the
    deterministic consolidation plan hash, and the set of semantic request
    hashes. It does NOT pin a provider family, model name, endpoint, or timeout.
    """

    consolidation_profile_id: str
    consolidation_profile_hash: str
    fact_semantic_profile_id: str
    fact_semantic_profile_hash: str
    event_semantic_profile_id: str
    event_semantic_profile_hash: str
    relationship_semantic_profile_id: str
    relationship_semantic_profile_hash: str
    prompt_identities: tuple[PromptAssetIdentity, ...]
    output_schema_identities: tuple[OutputSchemaAssetIdentity, ...]
    plan_hash: str
    semantic_request_hashes: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_text(self.consolidation_profile_id, "consolidation_profile_id")
        _require_sha256(self.consolidation_profile_hash, "consolidation_profile_hash")
        for field in (
            "fact_semantic_profile_id",
            "event_semantic_profile_id",
            "relationship_semantic_profile_id",
        ):
            _require_text(getattr(self, field), field)
        for field in (
            "fact_semantic_profile_hash",
            "event_semantic_profile_hash",
            "relationship_semantic_profile_hash",
        ):
            _require_sha256(getattr(self, field), field)
        for item in self.prompt_identities:
            if not isinstance(item, PromptAssetIdentity):
                raise ConsolidationModelError(
                    "prompt_identities must be PromptAssetIdentity"
                )
        for item in self.output_schema_identities:
            if not isinstance(item, OutputSchemaAssetIdentity):
                raise ConsolidationModelError(
                    "output_schema_identities must be OutputSchemaAssetIdentity"
                )
        _require_sha256(self.plan_hash, "plan_hash")
        hashes = _require_hash_tuple(
            self.semantic_request_hashes, "semantic_request_hashes"
        )
        object.__setattr__(self, "semantic_request_hashes", hashes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "consolidation_profile_id": self.consolidation_profile_id,
            "consolidation_profile_hash": self.consolidation_profile_hash,
            "fact_semantic_profile_id": self.fact_semantic_profile_id,
            "fact_semantic_profile_hash": self.fact_semantic_profile_hash,
            "event_semantic_profile_id": self.event_semantic_profile_id,
            "event_semantic_profile_hash": self.event_semantic_profile_hash,
            "relationship_semantic_profile_id": self.relationship_semantic_profile_id,
            "relationship_semantic_profile_hash": self.relationship_semantic_profile_hash,
            "prompt_identities": [p.to_dict() for p in self.prompt_identities],
            "output_schema_identities": [
                s.to_dict() for s in self.output_schema_identities
            ],
            "plan_hash": self.plan_hash,
            "semantic_request_hashes": list(self.semantic_request_hashes),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "A5SemanticIdentity":
        _require_exact_keys(
            value,
            {
                "consolidation_profile_id",
                "consolidation_profile_hash",
                "fact_semantic_profile_id",
                "fact_semantic_profile_hash",
                "event_semantic_profile_id",
                "event_semantic_profile_hash",
                "relationship_semantic_profile_id",
                "relationship_semantic_profile_hash",
                "prompt_identities",
                "output_schema_identities",
                "plan_hash",
                "semantic_request_hashes",
            },
            "A5SemanticIdentity",
        )
        return cls(
            consolidation_profile_id=value["consolidation_profile_id"],
            consolidation_profile_hash=value["consolidation_profile_hash"],
            fact_semantic_profile_id=value["fact_semantic_profile_id"],
            fact_semantic_profile_hash=value["fact_semantic_profile_hash"],
            event_semantic_profile_id=value["event_semantic_profile_id"],
            event_semantic_profile_hash=value["event_semantic_profile_hash"],
            relationship_semantic_profile_id=value["relationship_semantic_profile_id"],
            relationship_semantic_profile_hash=value["relationship_semantic_profile_hash"],
            prompt_identities=tuple(
                PromptAssetIdentity.from_dict(p) for p in value["prompt_identities"]
            ),
            output_schema_identities=tuple(
                OutputSchemaAssetIdentity.from_dict(s)
                for s in value["output_schema_identities"]
            ),
            plan_hash=value["plan_hash"],
            semantic_request_hashes=tuple(value["semantic_request_hashes"]),
        )


@dataclass(frozen=True, slots=True)
class A5UpstreamIdentity:
    """The exact upstream identity A5 consumed.

    A5 consumes the exact A4 EntityMap (pinned in the manifest via
    ``entity_map_ref``); together with the A3 input identity, this expresses
    the exact A3 -> A4 -> A5 chain.
    """

    a3_input: A3InputIdentity

    def __post_init__(self) -> None:
        if not isinstance(self.a3_input, A3InputIdentity):
            raise ConsolidationModelError("a3_input must be an A3InputIdentity")

    def to_dict(self) -> dict[str, Any]:
        return {"a3_input": self.a3_input.to_dict()}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "A5UpstreamIdentity":
        _require_exact_keys(value, {"a3_input"}, "A5UpstreamIdentity")
        return cls(a3_input=A3InputIdentity.from_dict(value["a3_input"]))


@dataclass(frozen=True, slots=True)
class ConsolidationCoverageSummary:
    """A5 coverage summary (non-negative counts)."""

    fact_candidate_count: int
    event_candidate_count: int
    relationship_candidate_count: int
    canonical_fact_count: int
    canonical_event_count: int
    canonical_relationship_count: int
    uncertain_decision_count: int
    story_conflict_count: int

    def __post_init__(self) -> None:
        for field in (
            "fact_candidate_count",
            "event_candidate_count",
            "relationship_candidate_count",
            "canonical_fact_count",
            "canonical_event_count",
            "canonical_relationship_count",
            "uncertain_decision_count",
            "story_conflict_count",
        ):
            _require_non_negative_int(getattr(self, field), field)

    def to_dict(self) -> dict[str, Any]:
        return {
            "fact_candidate_count": self.fact_candidate_count,
            "event_candidate_count": self.event_candidate_count,
            "relationship_candidate_count": self.relationship_candidate_count,
            "canonical_fact_count": self.canonical_fact_count,
            "canonical_event_count": self.canonical_event_count,
            "canonical_relationship_count": self.canonical_relationship_count,
            "uncertain_decision_count": self.uncertain_decision_count,
            "story_conflict_count": self.story_conflict_count,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ConsolidationCoverageSummary":
        _require_exact_keys(
            value,
            {
                "fact_candidate_count",
                "event_candidate_count",
                "relationship_candidate_count",
                "canonical_fact_count",
                "canonical_event_count",
                "canonical_relationship_count",
                "uncertain_decision_count",
                "story_conflict_count",
            },
            "ConsolidationCoverageSummary",
        )
        return cls(**value)


@dataclass(frozen=True, slots=True)
class ConsolidationManifest:
    """The A5 root aggregate (manifest).

    Pins the exact A4 EntityMap and the six A5 outputs (candidate index,
    decision set, canonical fact / event / relationship sets, story conflict
    set), the A5 semantic identity, the upstream (A3) identity, and the
    coverage summary.
    """

    schema_version: int
    project_id: str
    document_id: str
    entity_map_ref: ArtifactRef
    consolidation_candidate_index_ref: ArtifactRef
    consolidation_decision_set_ref: ArtifactRef
    canonical_fact_set_ref: ArtifactRef
    canonical_event_set_ref: ArtifactRef
    canonical_relationship_set_ref: ArtifactRef
    story_conflict_set_ref: ArtifactRef
    semantic_identity: A5SemanticIdentity
    upstream_identity: A5UpstreamIdentity
    coverage_summary: ConsolidationCoverageSummary

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ConsolidationModelError(
                f"unsupported schema_version: {self.schema_version!r}"
            )
        _require_text(self.project_id, "project_id")
        _require_text(self.document_id, "document_id")
        for field in (
            "entity_map_ref",
            "consolidation_candidate_index_ref",
            "consolidation_decision_set_ref",
            "canonical_fact_set_ref",
            "canonical_event_set_ref",
            "canonical_relationship_set_ref",
            "story_conflict_set_ref",
        ):
            if not isinstance(getattr(self, field), ArtifactRef):
                raise ConsolidationModelError(f"{field} must be an ArtifactRef")
        if not isinstance(self.semantic_identity, A5SemanticIdentity):
            raise ConsolidationModelError(
                "semantic_identity must be an A5SemanticIdentity"
            )
        if not isinstance(self.upstream_identity, A5UpstreamIdentity):
            raise ConsolidationModelError(
                "upstream_identity must be an A5UpstreamIdentity"
            )
        if not isinstance(self.coverage_summary, ConsolidationCoverageSummary):
            raise ConsolidationModelError(
                "coverage_summary must be a ConsolidationCoverageSummary"
            )

    def content_hash(self) -> str:
        return content_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "project_id": self.project_id,
            "document_id": self.document_id,
            "entity_map_ref": self.entity_map_ref.to_dict(),
            "consolidation_candidate_index_ref": self.consolidation_candidate_index_ref.to_dict(),
            "consolidation_decision_set_ref": self.consolidation_decision_set_ref.to_dict(),
            "canonical_fact_set_ref": self.canonical_fact_set_ref.to_dict(),
            "canonical_event_set_ref": self.canonical_event_set_ref.to_dict(),
            "canonical_relationship_set_ref": self.canonical_relationship_set_ref.to_dict(),
            "story_conflict_set_ref": self.story_conflict_set_ref.to_dict(),
            "semantic_identity": self.semantic_identity.to_dict(),
            "upstream_identity": self.upstream_identity.to_dict(),
            "coverage_summary": self.coverage_summary.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ConsolidationManifest":
        _require_exact_keys(
            value,
            {
                "schema_version",
                "project_id",
                "document_id",
                "entity_map_ref",
                "consolidation_candidate_index_ref",
                "consolidation_decision_set_ref",
                "canonical_fact_set_ref",
                "canonical_event_set_ref",
                "canonical_relationship_set_ref",
                "story_conflict_set_ref",
                "semantic_identity",
                "upstream_identity",
                "coverage_summary",
            },
            "ConsolidationManifest",
        )
        return cls(
            schema_version=value["schema_version"],
            project_id=value["project_id"],
            document_id=value["document_id"],
            entity_map_ref=ArtifactRef.from_dict(value["entity_map_ref"]),
            consolidation_candidate_index_ref=ArtifactRef.from_dict(
                value["consolidation_candidate_index_ref"]
            ),
            consolidation_decision_set_ref=ArtifactRef.from_dict(
                value["consolidation_decision_set_ref"]
            ),
            canonical_fact_set_ref=ArtifactRef.from_dict(
                value["canonical_fact_set_ref"]
            ),
            canonical_event_set_ref=ArtifactRef.from_dict(
                value["canonical_event_set_ref"]
            ),
            canonical_relationship_set_ref=ArtifactRef.from_dict(
                value["canonical_relationship_set_ref"]
            ),
            story_conflict_set_ref=ArtifactRef.from_dict(
                value["story_conflict_set_ref"]
            ),
            semantic_identity=A5SemanticIdentity.from_dict(
                value["semantic_identity"]
            ),
            upstream_identity=A5UpstreamIdentity.from_dict(
                value["upstream_identity"]
            ),
            coverage_summary=ConsolidationCoverageSummary.from_dict(
                value["coverage_summary"]
            ),
        )
