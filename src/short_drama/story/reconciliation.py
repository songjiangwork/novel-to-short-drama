"""v1.2 A4A — entity reconciliation domain contracts + tracked assets.

This module is the static/domain-contract layer for A4 (entity reconciliation).
It defines the typed, deterministic models and the ``EntityReconciliationProfile``
that the later A4B-A4E slices depend on WITHOUT changing their public contract:

    GlobalCandidateRef
    CandidateEntityIndexEntry / CandidateEntityIndex
    ReconciliationDecisionItem / ReconciliationDecisionPayload
    ReconciliationDecision / ReconciliationDecisionSet
    CanonicalEntity / CanonicalCharacterRegistry / CanonicalLocationRegistry
    UnresolvedEntity / UnresolvedEntitySet
    A3InputIdentity / EntityMapEntry / EntityMap
    EntityReconciliationProfile

A4A is deliberately a pure domain/static-contract slice. It does NOT:

  * collect candidates or compute source ordering (A4B);
  * implement name normalization / safe-name classification / must-not-merge /
    blocking / pair planning / auto-same (A4B);
  * invoke an LLM or build a semantic request (A4C);
  * build the identity graph, detect conflicts, allocate canonical IDs, or group
    unresolved ambiguity (A4D);
  * persist artifacts, write CURRENT pointers, or reuse (A4D);
  * provide a stage CLI or a real-novel smoke (A4E).

It reuses the existing authorities rather than inventing a second one:

  * Foundation ``ArtifactRef`` for every cross-artifact ref (``candidate_
    extraction_ref``, the EntityMap pins, and ``A3InputIdentity``);
  * A3A ``EvidenceRef`` for ``evidence_refs``;
  * A-I3 ``LLMInvocationProvenance`` for the persisted decision's
    ``generation_provenance``;
  * the canonical ``content_hash`` for the reconciliation profile hash.

Frozen coverage / merge boundary (kept in lockstep with the A4 contract):

  * coverage universe = character + location + A3 unresolved;
  * identity merge graph = character + location only;
  * A3 ``cand_unres_*`` are explicit unresolved passthrough, never guessed into a
    canonical identity.

Style follows the rest of the story package: frozen dataclasses, explicit
``to_dict()`` / ``from_dict()``, exact-key / fail-closed parsing, and
deterministic serialization shape. No Pydantic, no database/ORM, no ArtifactStore
access.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from short_drama.artifacts import ArtifactRef, content_hash
from short_drama.io import load_yaml
from short_drama.llm import LLMConfigError, LLMInvocationProvenance

from .errors import ReconciliationModelError
from .extraction import EvidenceRef

# ---------------------------------------------------------------------------
# Schema / artifact identity constants
# ---------------------------------------------------------------------------

RECONCILIATION_PROFILE_SCHEMA_VERSION = 1
# Frozen A-I5 v1 ceiling: first semantic generation + at most one semantic
# regeneration. Combined with A-I3's bounded technical attempts this keeps the
# total provider attempts bounded. Enforced at the static-contract layer.
RECONCILIATION_MAX_GENERATION_ROUNDS_V1 = 2

CANDIDATE_ENTITY_INDEX_SCHEMA_VERSION = 1
RECONCILIATION_DECISION_SET_SCHEMA_VERSION = 1
CANONICAL_ENTITY_REGISTRY_SCHEMA_VERSION = 1
UNRESOLVED_ENTITY_SET_SCHEMA_VERSION = 1
ENTITY_MAP_SCHEMA_VERSION = 2

# ---------------------------------------------------------------------------
# Frozen value domains (kept in lockstep with the parent A-I5 contract)
# ---------------------------------------------------------------------------

# CandidateEntityIndex.candidate_kind. The coverage universe is character +
# location + A3 unresolved. The four ``unresolved_*`` kinds are the A3
# ``UnresolvedMentionCandidate.mention_kind`` values (person / location /
# unknown / other) lifted into the A4 namespace; ``character`` / ``location``
# are the merge-graph member kinds.
CANDIDATE_KINDS = frozenset(
    {
        "character",
        "location",
        "unresolved_person",
        "unresolved_location",
        "unresolved_unknown",
        "unresolved_other",
    }
)

# Reconciliation identity decision (section 15): a fixed, closed classification.
RECONCILIATION_DECISIONS = frozenset(
    {"same_entity", "different_entity", "uncertain"}
)

# Decision provenance method (section 18). ``manual`` is a reserved
# schema/domain extension point; the A4 v1 pipeline implements only
# ``deterministic`` and ``llm`` (no manual-review workflow/UI).
RECONCILIATION_METHODS = frozenset({"deterministic", "llm", "manual"})

# EntityMapEntry.status (section 26).
ENTITY_MAP_STATUSES = frozenset({"resolved", "unresolved"})

# CanonicalEntity.entity_type (section 24). Only merge-graph members become
# canonical entities.
CANONICAL_ENTITY_TYPES = frozenset({"character", "location"})

# UnresolvedEntity.entity_kind (section 25). Sources:
#   * A4 uncertain ambiguity groups  -> character / location;
#   * A3 unresolved mention passthrough -> person / location / other / unknown.
# The union is the closed domain; ``location`` is shared by both sources.
UNRESOLVED_ENTITY_KINDS = frozenset(
    {"character", "location", "person", "other", "unknown"}
)

# ---------------------------------------------------------------------------
# Reference / identifier patterns (kept in lockstep with the frozen upstream
# authorities; A4A enforces only the structural shape, not ordering/coverage)
# ---------------------------------------------------------------------------

# A2 chunk-id authority (chunking._CHUNK_ID_RE). A4A references chunks, it does
# not re-plan them; the pattern stays in lockstep with A2.
_CHUNK_ID_RE = re.compile(r"^CH[0-9]{3,}_C[0-9]{3,}$")

# A4 coverage-universe local-candidate namespaces (a strict subset of A3A's
# _CANDIDATE_ID_PATTERNS: only character / location / unresolved participate in
# A4). A4A reuses the A3A numeric-suffix rule verbatim.
_A4_LOCAL_CANDIDATE_PATTERNS = {
    "cand_char_": re.compile(r"^cand_char_[0-9]{3,}$"),
    "cand_loc_": re.compile(r"^cand_loc_[0-9]{3,}$"),
    "cand_unres_": re.compile(r"^cand_unres_[0-9]{3,}$"),
}
_A4_LOCAL_CANDIDATE_NAMESPACES = {
    "cand_char_": "character",
    "cand_loc_": "location",
    "cand_unres_": "unresolved",
}

# Persisted canonical form of a cross-chunk candidate reference:
#   <chunk_id>:<local_candidate_id>
# e.g. CH003_C002:cand_char_001
GLOBAL_CANDIDATE_REF_PATTERN = (
    r"^CH[0-9]{3,}_C[0-9]{3,}:cand_(?:char|loc|unres)_[0-9]{3,}$"
)
_GLOBAL_CANDIDATE_REF_RE = re.compile(GLOBAL_CANDIDATE_REF_PATTERN)

# Persisted canonical form of an A4 identity merge-graph candidate reference.
# The A4 identity merge graph contains ONLY character + location candidates:
# A3 ``cand_unres_*`` are explicit unresolved passthrough and never participate
# in identity merging. This is a strict subset of the coverage-universe global
# ref (all three namespaces) and is what a reconciliation pair (left/right) may
# reference. (This is a structural domain tightening only; it does not plan or
# resolve pairs -- that is A4B/A4C.)
MERGE_GRAPH_CANDIDATE_REF_PATTERN = (
    r"^CH[0-9]{3,}_C[0-9]{3,}:cand_(?:char|loc)_[0-9]{3,}$"
)
_MERGE_GRAPH_CANDIDATE_REF_RE = re.compile(MERGE_GRAPH_CANDIDATE_REF_PATTERN)

# Canonical entity id namespaces (section 23): char_0001 / loc_0001 / unres_0001.
_CANONICAL_ID_PATTERNS = {
    "character": re.compile(r"^char_[0-9]{4,}$"),
    "location": re.compile(r"^loc_[0-9]{4,}$"),
}
_ANY_CANONICAL_ID_RE = re.compile(r"^(?:char|loc)_[0-9]{4,}$")
_UNRESOLVED_ID_RE = re.compile(r"^unres_[0-9]{4,}$")

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


# ---------------------------------------------------------------------------
# Structural validation helpers (fail closed; mirror the A3A conventions)
# ---------------------------------------------------------------------------


def _require_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ReconciliationModelError(
            f"{field_name} must be a non-empty string without NUL"
        )
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ReconciliationModelError(
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
        raise ReconciliationModelError(
            f"{field_name} must be a safe lowercase storage identifier"
        )
    return value


def _require_hash(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ReconciliationModelError(
            f"{field_name} must be a lowercase 64-char SHA-256 hex digest"
        )
    return value


def _require_positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ReconciliationModelError(f"{field_name} must be an integer >= 1")
    return value


def _require_exact_keys(value: Any, keys: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise ReconciliationModelError(
            f"{name} must contain exactly: {', '.join(sorted(keys))}"
        )
    return value


def _require_enum(value: Any, allowed: frozenset[str], field_name: str) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise ReconciliationModelError(
            f"{field_name} must be one of: {', '.join(sorted(allowed))}"
        )
    return value


def _require_artifact_ref(value: Any, field_name: str) -> ArtifactRef:
    if not isinstance(value, ArtifactRef):
        raise ReconciliationModelError(
            f"{field_name} must be an ArtifactRef"
        )
    return value


def _require_global_candidate_ref(value: Any, field_name: str) -> str:
    """Validate a cross-chunk candidate reference and return its canonical form.

    A4A enforces only the structural shape of the persisted reference (a valid
    ``<chunk_id>:<local_candidate_id>`` in a coverage-universe namespace). It
    does NOT check that the referenced candidate actually exists (that is an
    A4B coverage finding, ``A4_CANDIDATE_REF_NOT_FOUND``).
    """
    ref = GlobalCandidateRef.parse(value)
    return ref.to_string()


def _require_merge_graph_candidate_ref(value: Any, field_name: str) -> str:
    """Validate an identity merge-graph candidate reference and return its
    canonical form.

    A reconciliation pair member must be a character or location candidate
    reference. A3 ``cand_unres_*`` are explicit unresolved passthrough and never
    participate in the A4 identity merge graph, so they are rejected here even
    though they are valid coverage-universe (global) candidate references.
    """
    if not isinstance(value, str) or _MERGE_GRAPH_CANDIDATE_REF_RE.fullmatch(value) is None:
        raise ReconciliationModelError(
            f"{field_name} must be a character/location merge-graph candidate "
            f"reference ({MERGE_GRAPH_CANDIDATE_REF_PATTERN!r}): {value!r}"
        )
    # The merge-graph pattern is a strict subset of the global candidate ref,
    # so the global parse/round-trip canonicalization always succeeds here.
    return GlobalCandidateRef.parse(value).to_string()


def _require_canonical_id(value: Any, entity_type: str, field_name: str) -> str:
    _require_text(value, field_name)
    pattern = _CANONICAL_ID_PATTERNS.get(entity_type)
    if pattern is None or pattern.fullmatch(value) is None:
        raise ReconciliationModelError(
            f"{field_name} does not match the {entity_type} canonical-id "
            f"namespace: {value!r}"
        )
    return value


def _to_string_tuple(
    value: Any,
    field_name: str,
    *,
    min_items: int = 0,
    unique: bool = True,
    global_ref: bool = False,
) -> tuple[str, ...]:
    """Normalize a string collection to a validated tuple (fail closed).

    ``global_ref=True`` additionally requires every item to be a valid A4
    cross-chunk candidate reference (persisted canonical form).
    """
    if isinstance(value, str):
        raise ReconciliationModelError(f"{field_name} must be a list of strings")
    try:
        items = tuple(value)
    except TypeError as exc:
        raise ReconciliationModelError(f"{field_name} must be a list of strings") from exc
    if len(items) < min_items:
        raise ReconciliationModelError(
            f"{field_name} must contain at least {min_items} item(s)"
        )
    for item in items:
        if global_ref:
            _require_global_candidate_ref(item, field_name)
        else:
            _require_text(item, field_name)
    if unique and len(items) != len(set(items)):
        raise ReconciliationModelError(f"{field_name} must not contain duplicates")
    return items


def _to_evidence_tuple(
    value: Any, field_name: str, *, min_items: int = 0
) -> tuple[EvidenceRef, ...]:
    """Normalize an EvidenceRef collection to a validated tuple (fail closed)."""
    if isinstance(value, EvidenceRef):
        raise ReconciliationModelError(f"{field_name} must be a list of EvidenceRef")
    try:
        items = tuple(value)
    except TypeError as exc:
        raise ReconciliationModelError(f"{field_name} must be a list of EvidenceRef") from exc
    if len(items) < min_items:
        raise ReconciliationModelError(
            f"{field_name} must contain at least {min_items} item(s)"
        )
    for item in items:
        if not isinstance(item, EvidenceRef):
            raise ReconciliationModelError(
                f"{field_name} must contain EvidenceRef values"
            )
    return items


def _to_artifact_ref_tuple(
    value: Any, field_name: str, *, min_items: int = 0
) -> tuple[ArtifactRef, ...]:
    """Normalize an ArtifactRef collection to a validated tuple (fail closed)."""
    if isinstance(value, ArtifactRef):
        raise ReconciliationModelError(
            f"{field_name} must be a list of ArtifactRef"
        )
    try:
        items = tuple(value)
    except TypeError as exc:
        raise ReconciliationModelError(
            f"{field_name} must be a list of ArtifactRef"
        ) from exc
    if len(items) < min_items:
        raise ReconciliationModelError(
            f"{field_name} must contain at least {min_items} item(s)"
        )
    for item in items:
        _require_artifact_ref(item, field_name)
    return items


def _require_pair_canonical_order(
    left: str, right: str, *, left_name: str, right_name: str
) -> None:
    """Enforce left != right and the canonical (sorted) pair ordering.

    The persisted/provider pair is stored in canonical order: the smaller
    reference string is ``left`` and the larger is ``right``. This is a
    structural invariant only; it does not decide whether the pair should merge
    (A4B/A4C).
    """
    if left == right:
        raise ReconciliationModelError(
            f"a reconciliation pair must reference two distinct candidates "
            f"({left_name} == {right_name}: {left!r})"
        )
    if left > right:
        raise ReconciliationModelError(
            f"reconciliation pair must be stored in canonical order "
            f"({left_name} must sort <= {right_name}): {left!r} > {right!r}"
        )


# ---------------------------------------------------------------------------
# GlobalCandidateRef — typed helper for the persisted canonical reference form
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GlobalCandidateRef:
    """A cross-chunk candidate reference in persisted canonical form.

    The persisted form is ``<chunk_id>:<local_candidate_id>`` where ``chunk_id``
    follows the A2 chunk-id authority and ``local_candidate_id`` is an A3A
    coverage-universe id (``cand_char_*`` / ``cand_loc_*`` / ``cand_unres_*``).

    This is a deterministic parse/validate/round-trip helper. It does NOT touch
    the ArtifactStore and does NOT check that the referenced candidate exists;
    it only enforces the structural shape of the reference.
    """

    chunk_id: str
    local_candidate_id: str

    def __post_init__(self) -> None:
        _require_text(self.chunk_id, "GlobalCandidateRef.chunk_id")
        if _CHUNK_ID_RE.fullmatch(self.chunk_id) is None:
            raise ReconciliationModelError(
                f"GlobalCandidateRef.chunk_id must match the A2 chunk-id "
                f"authority ({_CHUNK_ID_RE.pattern!r}): {self.chunk_id!r}"
            )
        _require_text(
            self.local_candidate_id, "GlobalCandidateRef.local_candidate_id"
        )
        if self.namespace is None:
            raise ReconciliationModelError(
                "GlobalCandidateRef.local_candidate_id must be a known A4 "
                f"coverage-universe namespace "
                f"({', '.join(sorted(_A4_LOCAL_CANDIDATE_NAMESPACES))}): "
                f"{self.local_candidate_id!r}"
            )

    @property
    def namespace(self) -> str | None:
        """The coverage-universe namespace, or None for an unknown namespace."""
        for prefix, name in _A4_LOCAL_CANDIDATE_NAMESPACES.items():
            if self.local_candidate_id.startswith(prefix):
                # A prefix match is necessary but not sufficient; confirm the
                # full id (including its numeric suffix) matches the pattern.
                if _A4_LOCAL_CANDIDATE_PATTERNS[prefix].fullmatch(
                    self.local_candidate_id
                ):
                    return name
        return None

    def to_string(self) -> str:
        """Return the persisted canonical form ``<chunk_id>:<local_candidate_id>``."""
        return f"{self.chunk_id}:{self.local_candidate_id}"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.to_string()

    @classmethod
    def parse(cls, value: Any) -> "GlobalCandidateRef":
        """Parse the persisted canonical form (fail closed on any deviation)."""
        if not isinstance(value, str) or "\x00" in value:
            raise ReconciliationModelError(
                "GlobalCandidateRef must be a non-empty string without NUL"
            )
        # A structural pre-check keeps the error message precise for the common
        # case; the constructor performs the authoritative validation.
        if _GLOBAL_CANDIDATE_REF_RE.fullmatch(value) is None:
            raise ReconciliationModelError(
                f"invalid global candidate reference "
                f"({GLOBAL_CANDIDATE_REF_PATTERN!r}): {value!r}"
            )
        chunk_id, local_candidate_id = value.split(":", 1)
        return cls(chunk_id=chunk_id, local_candidate_id=local_candidate_id)

    @classmethod
    def from_string(cls, value: Any) -> "GlobalCandidateRef":
        return cls.parse(value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "local_candidate_id": self.local_candidate_id,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "GlobalCandidateRef":
        _require_exact_keys(
            value,
            {"chunk_id", "local_candidate_id"},
            "GlobalCandidateRef",
        )
        return cls(chunk_id=value["chunk_id"], local_candidate_id=value["local_candidate_id"])


# ---------------------------------------------------------------------------
# CandidateEntityIndex contracts
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CandidateEntityIndexEntry:
    """One indexed candidate in the (global) A4 coverage universe.

    A4A establishes only the deterministic serialization/shape contract. It does
    NOT collect candidates, compute ``source_order_key`` ordering, normalize
    names, or build blocks (A4B). ``candidate_ref`` and every entry in
    ``possible_candidate_refs`` is a cross-chunk (global) candidate reference.
    """

    candidate_ref: str
    candidate_kind: str
    candidate_extraction_ref: ArtifactRef
    source_order_key: str
    display_name_original: str
    aliases_original: tuple[str, ...]
    descriptors_zh: tuple[str, ...]
    evidence_refs: tuple[EvidenceRef, ...]
    possible_candidate_refs: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "candidate_ref",
            _require_global_candidate_ref(
                self.candidate_ref, "CandidateEntityIndexEntry.candidate_ref"
            ),
        )
        _require_enum(self.candidate_kind, CANDIDATE_KINDS,
                      "CandidateEntityIndexEntry.candidate_kind")
        _require_artifact_ref(
            self.candidate_extraction_ref,
            "CandidateEntityIndexEntry.candidate_extraction_ref",
        )
        _require_text(self.source_order_key, "CandidateEntityIndexEntry.source_order_key")
        _require_text(
            self.display_name_original,
            "CandidateEntityIndexEntry.display_name_original",
        )
        object.__setattr__(
            self,
            "aliases_original",
            _to_string_tuple(self.aliases_original,
                             "CandidateEntityIndexEntry.aliases_original"),
        )
        object.__setattr__(
            self,
            "descriptors_zh",
            _to_string_tuple(self.descriptors_zh,
                             "CandidateEntityIndexEntry.descriptors_zh"),
        )
        object.__setattr__(
            self,
            "evidence_refs",
            _to_evidence_tuple(self.evidence_refs,
                               "CandidateEntityIndexEntry.evidence_refs"),
        )
        object.__setattr__(
            self,
            "possible_candidate_refs",
            _to_string_tuple(
                self.possible_candidate_refs,
                "CandidateEntityIndexEntry.possible_candidate_refs",
                global_ref=True,
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_ref": self.candidate_ref,
            "candidate_kind": self.candidate_kind,
            "candidate_extraction_ref": self.candidate_extraction_ref.to_dict(),
            "source_order_key": self.source_order_key,
            "display_name_original": self.display_name_original,
            "aliases_original": list(self.aliases_original),
            "descriptors_zh": list(self.descriptors_zh),
            "evidence_refs": [e.to_dict() for e in self.evidence_refs],
            "possible_candidate_refs": list(self.possible_candidate_refs),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CandidateEntityIndexEntry":
        _require_exact_keys(
            value,
            {
                "candidate_ref",
                "candidate_kind",
                "candidate_extraction_ref",
                "source_order_key",
                "display_name_original",
                "aliases_original",
                "descriptors_zh",
                "evidence_refs",
                "possible_candidate_refs",
            },
            "CandidateEntityIndexEntry",
        )
        for list_field in ("aliases_original", "descriptors_zh", "evidence_refs",
                           "possible_candidate_refs"):
            if not isinstance(value[list_field], list):
                raise ReconciliationModelError(
                    f"CandidateEntityIndexEntry.{list_field} must be a list"
                )
        try:
            extraction_ref = ArtifactRef.from_dict(value["candidate_extraction_ref"])
        except Exception as exc:  # noqa: BLE001
            raise ReconciliationModelError(
                f"invalid candidate_extraction_ref: {exc}"
            ) from exc
        return cls(
            candidate_ref=value["candidate_ref"],
            candidate_kind=value["candidate_kind"],
            candidate_extraction_ref=extraction_ref,
            source_order_key=value["source_order_key"],
            display_name_original=value["display_name_original"],
            aliases_original=tuple(value["aliases_original"]),
            descriptors_zh=tuple(value["descriptors_zh"]),
            evidence_refs=tuple(
                EvidenceRef.from_dict(item) for item in value["evidence_refs"]
            ),
            possible_candidate_refs=tuple(value["possible_candidate_refs"]),
        )


@dataclass(frozen=True, slots=True)
class CandidateEntityIndex:
    """The (global) A4 coverage-universe candidate index.

    A4A establishes only the shape. A4B fills ``entries`` deterministically
    from the validated A3 CURRENT set and computes each entry's
    ``source_order_key`` from the exact source-order authority.
    """

    schema_version: int
    entries: tuple[CandidateEntityIndexEntry, ...]

    def __post_init__(self) -> None:
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
            or self.schema_version != CANDIDATE_ENTITY_INDEX_SCHEMA_VERSION
        ):
            raise ReconciliationModelError(
                "CandidateEntityIndex.schema_version must be "
                f"{CANDIDATE_ENTITY_INDEX_SCHEMA_VERSION}"
            )
        if isinstance(self.entries, CandidateEntityIndexEntry):
            raise ReconciliationModelError(
                "CandidateEntityIndex.entries must be a list of entries"
            )
        try:
            items = tuple(self.entries)
        except TypeError as exc:
            raise ReconciliationModelError(
                "CandidateEntityIndex.entries must be a list of entries"
            ) from exc
        for item in items:
            if not isinstance(item, CandidateEntityIndexEntry):
                raise ReconciliationModelError(
                    "CandidateEntityIndex.entries must contain "
                    "CandidateEntityIndexEntry values"
                )
        object.__setattr__(self, "entries", items)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "entries": [entry.to_dict() for entry in self.entries],
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CandidateEntityIndex":
        _require_exact_keys(
            value, {"schema_version", "entries"}, "CandidateEntityIndex"
        )
        if not isinstance(value["entries"], list):
            raise ReconciliationModelError(
                "CandidateEntityIndex.entries must be a list"
            )
        return cls(
            schema_version=value["schema_version"],
            entries=tuple(
                CandidateEntityIndexEntry.from_dict(item)
                for item in value["entries"]
            ),
        )


# ---------------------------------------------------------------------------
# Reconciliation decision contracts
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ReconciliationDecisionItem:
    """One identity decision as returned by the provider (section 17).

    Python (A4C) later enriches each item into a persisted
    :class:`ReconciliationDecision` (adding ``decision_id``, ``method``,
    ``reason_code``, prompt identity, and generation provenance). A4A validates
    only the structural shape: canonical pair ordering, the closed decision
    enum, and the exact key set. The pair members (``left_candidate_ref`` /
    ``right_candidate_ref``) are identity merge-graph references -- character or
    location candidates only -- because A3 ``cand_unres_*`` never participate in
    the A4 identity merge graph.
    """

    left_candidate_ref: str
    right_candidate_ref: str
    decision: str
    reason_zh: str
    evidence_refs: tuple[EvidenceRef, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "left_candidate_ref",
            _require_merge_graph_candidate_ref(
                self.left_candidate_ref,
                "ReconciliationDecisionItem.left_candidate_ref",
            ),
        )
        object.__setattr__(
            self,
            "right_candidate_ref",
            _require_merge_graph_candidate_ref(
                self.right_candidate_ref,
                "ReconciliationDecisionItem.right_candidate_ref",
            ),
        )
        _require_pair_canonical_order(
            self.left_candidate_ref,
            self.right_candidate_ref,
            left_name="left_candidate_ref",
            right_name="right_candidate_ref",
        )
        _require_enum(self.decision, RECONCILIATION_DECISIONS,
                      "ReconciliationDecisionItem.decision")
        _require_text(self.reason_zh, "ReconciliationDecisionItem.reason_zh")
        object.__setattr__(
            self,
            "evidence_refs",
            _to_evidence_tuple(self.evidence_refs,
                               "ReconciliationDecisionItem.evidence_refs"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "left_candidate_ref": self.left_candidate_ref,
            "right_candidate_ref": self.right_candidate_ref,
            "decision": self.decision,
            "reason_zh": self.reason_zh,
            "evidence_refs": [e.to_dict() for e in self.evidence_refs],
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ReconciliationDecisionItem":
        _require_exact_keys(
            value,
            {
                "left_candidate_ref",
                "right_candidate_ref",
                "decision",
                "reason_zh",
                "evidence_refs",
            },
            "ReconciliationDecisionItem",
        )
        if not isinstance(value["evidence_refs"], list):
            raise ReconciliationModelError(
                "ReconciliationDecisionItem.evidence_refs must be a list"
            )
        return cls(
            left_candidate_ref=value["left_candidate_ref"],
            right_candidate_ref=value["right_candidate_ref"],
            decision=value["decision"],
            reason_zh=value["reason_zh"],
            evidence_refs=tuple(
                EvidenceRef.from_dict(item) for item in value["evidence_refs"]
            ),
        )


@dataclass(frozen=True, slots=True)
class ReconciliationDecisionPayload:
    """The provider structured-output shape: ``ReconciliationDecisionPayload``.

    Section 17. The provider returns ``{"decisions": [...]}`` where every
    requested pair appears exactly once. A4A does NOT check requested-pair
    coverage/uniqueness against an ambiguity plan (that is A4C validation); it
    validates only that each element is a well-formed decision item.
    """

    decisions: tuple[ReconciliationDecisionItem, ...]

    def __post_init__(self) -> None:
        if isinstance(self.decisions, ReconciliationDecisionItem):
            raise ReconciliationModelError(
                "ReconciliationDecisionPayload.decisions must be a list"
            )
        try:
            items = tuple(self.decisions)
        except TypeError as exc:
            raise ReconciliationModelError(
                "ReconciliationDecisionPayload.decisions must be a list"
            ) from exc
        for item in items:
            if not isinstance(item, ReconciliationDecisionItem):
                raise ReconciliationModelError(
                    "ReconciliationDecisionPayload.decisions must contain "
                    "ReconciliationDecisionItem values"
                )
        object.__setattr__(self, "decisions", items)

    def to_dict(self) -> dict[str, Any]:
        return {
            "decisions": [decision.to_dict() for decision in self.decisions],
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ReconciliationDecisionPayload":
        _require_exact_keys(value, {"decisions"}, "ReconciliationDecisionPayload")
        if not isinstance(value["decisions"], list):
            raise ReconciliationModelError(
                "ReconciliationDecisionPayload.decisions must be a list"
            )
        return cls(
            decisions=tuple(
                ReconciliationDecisionItem.from_dict(item)
                for item in value["decisions"]
            ),
        )


@dataclass(frozen=True, slots=True)
class ReconciliationDecision:
    """A persisted identity decision (section 18).

    Shape-level invariants (enforced here, A4A):
      * left != right and stored in canonical (sorted) order;
      * ``decision`` and ``method`` are closed enums;
      * method consistency: ``deterministic`` / ``manual`` carry null prompt
        identity and null generation provenance; ``llm`` carries exact prompt
        identity and an ``LLMInvocationProvenance``.

    A4A does NOT decide whether the pair should merge, belongs to a block, or is
    proven identical (A4B/A4C/A4D).
    """

    decision_id: str
    left_candidate_ref: str
    right_candidate_ref: str
    decision: str
    method: str
    reason_code: str
    reason_zh: str
    evidence_refs: tuple[EvidenceRef, ...]
    prompt_id: str | None
    prompt_version: int | None
    generation_provenance: LLMInvocationProvenance | None

    def __post_init__(self) -> None:
        _require_text(self.decision_id, "ReconciliationDecision.decision_id")
        object.__setattr__(
            self,
            "left_candidate_ref",
            _require_merge_graph_candidate_ref(
                self.left_candidate_ref,
                "ReconciliationDecision.left_candidate_ref",
            ),
        )
        object.__setattr__(
            self,
            "right_candidate_ref",
            _require_merge_graph_candidate_ref(
                self.right_candidate_ref,
                "ReconciliationDecision.right_candidate_ref",
            ),
        )
        _require_pair_canonical_order(
            self.left_candidate_ref,
            self.right_candidate_ref,
            left_name="left_candidate_ref",
            right_name="right_candidate_ref",
        )
        _require_enum(self.decision, RECONCILIATION_DECISIONS,
                      "ReconciliationDecision.decision")
        _require_enum(self.method, RECONCILIATION_METHODS,
                      "ReconciliationDecision.method")
        _require_text(self.reason_code, "ReconciliationDecision.reason_code")
        _require_text(self.reason_zh, "ReconciliationDecision.reason_zh")
        object.__setattr__(
            self,
            "evidence_refs",
            _to_evidence_tuple(self.evidence_refs,
                               "ReconciliationDecision.evidence_refs"),
        )
        self._check_method_consistency()

    def _check_method_consistency(self) -> None:
        if self.method == "llm":
            if self.prompt_id is None:
                raise ReconciliationModelError(
                    "ReconciliationDecision.prompt_id is required when "
                    "method == 'llm'"
                )
            if self.prompt_version is None:
                raise ReconciliationModelError(
                    "ReconciliationDecision.prompt_version is required when "
                    "method == 'llm'"
                )
            if not isinstance(self.generation_provenance, LLMInvocationProvenance):
                raise ReconciliationModelError(
                    "ReconciliationDecision.generation_provenance is required "
                    "when method == 'llm'"
                )
            _require_storage_id(self.prompt_id,
                                "ReconciliationDecision.prompt_id")
            _require_positive_int(self.prompt_version,
                                  "ReconciliationDecision.prompt_version")
        else:
            # deterministic and manual carry no LLM invocation and no prompt
            # identity.
            if self.prompt_id is not None or self.prompt_version is not None:
                raise ReconciliationModelError(
                    "ReconciliationDecision.prompt_id/prompt_version must be "
                    "null when method != 'llm'"
                )
            if self.generation_provenance is not None:
                raise ReconciliationModelError(
                    "ReconciliationDecision.generation_provenance must be null "
                    "when method != 'llm'"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "left_candidate_ref": self.left_candidate_ref,
            "right_candidate_ref": self.right_candidate_ref,
            "decision": self.decision,
            "method": self.method,
            "reason_code": self.reason_code,
            "reason_zh": self.reason_zh,
            "evidence_refs": [e.to_dict() for e in self.evidence_refs],
            "prompt_id": self.prompt_id,
            "prompt_version": self.prompt_version,
            "generation_provenance": (
                self.generation_provenance.to_dict()
                if self.generation_provenance is not None
                else None
            ),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ReconciliationDecision":
        _require_exact_keys(
            value,
            {
                "decision_id",
                "left_candidate_ref",
                "right_candidate_ref",
                "decision",
                "method",
                "reason_code",
                "reason_zh",
                "evidence_refs",
                "prompt_id",
                "prompt_version",
                "generation_provenance",
            },
            "ReconciliationDecision",
        )
        if not isinstance(value["evidence_refs"], list):
            raise ReconciliationModelError(
                "ReconciliationDecision.evidence_refs must be a list"
            )
        prov = value["generation_provenance"]
        if prov is None:
            generation_provenance = None
        else:
            if not isinstance(prov, dict) or set(prov) != _LLM_PROVENANCE_KEYS:
                raise ReconciliationModelError(
                    "ReconciliationDecision.generation_provenance must contain "
                    "exactly the A-I3 LLMInvocationProvenance fields"
                )
            try:
                generation_provenance = LLMInvocationProvenance(**prov)
            except (TypeError, LLMConfigError) as exc:
                raise ReconciliationModelError(
                    f"invalid generation_provenance: {exc}"
                ) from exc
        return cls(
            decision_id=value["decision_id"],
            left_candidate_ref=value["left_candidate_ref"],
            right_candidate_ref=value["right_candidate_ref"],
            decision=value["decision"],
            method=value["method"],
            reason_code=value["reason_code"],
            reason_zh=value["reason_zh"],
            evidence_refs=tuple(
                EvidenceRef.from_dict(item) for item in value["evidence_refs"]
            ),
            prompt_id=value["prompt_id"],
            prompt_version=value["prompt_version"],
            generation_provenance=generation_provenance,
        )


@dataclass(frozen=True, slots=True)
class ReconciliationDecisionSet:
    """The set of persisted reconciliation decisions.

    A4A establishes only the shape. A4B/A4D enforce requested-pair coverage,
    exact pair uniqueness, and that the LLM does not override deterministic
    constraints (findings ``A4_DECISION_PAIR_*`` / ``A4_DETERMINISTIC_CONSTRAINT_OVERRIDE``).
    """

    schema_version: int
    decisions: tuple[ReconciliationDecision, ...]

    def __post_init__(self) -> None:
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
            or self.schema_version != RECONCILIATION_DECISION_SET_SCHEMA_VERSION
        ):
            raise ReconciliationModelError(
                "ReconciliationDecisionSet.schema_version must be "
                f"{RECONCILIATION_DECISION_SET_SCHEMA_VERSION}"
            )
        if isinstance(self.decisions, ReconciliationDecision):
            raise ReconciliationModelError(
                "ReconciliationDecisionSet.decisions must be a list"
            )
        try:
            items = tuple(self.decisions)
        except TypeError as exc:
            raise ReconciliationModelError(
                "ReconciliationDecisionSet.decisions must be a list"
            ) from exc
        for item in items:
            if not isinstance(item, ReconciliationDecision):
                raise ReconciliationModelError(
                    "ReconciliationDecisionSet.decisions must contain "
                    "ReconciliationDecision values"
                )
        object.__setattr__(self, "decisions", items)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "decisions": [decision.to_dict() for decision in self.decisions],
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ReconciliationDecisionSet":
        _require_exact_keys(
            value, {"schema_version", "decisions"}, "ReconciliationDecisionSet"
        )
        if not isinstance(value["decisions"], list):
            raise ReconciliationModelError(
                "ReconciliationDecisionSet.decisions must be a list"
            )
        return cls(
            schema_version=value["schema_version"],
            decisions=tuple(
                ReconciliationDecision.from_dict(item) for item in value["decisions"]
            ),
        )


# ---------------------------------------------------------------------------
# Canonical entity contracts
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CanonicalEntity:
    """A resolved, unambiguous canonical entity (section 24).

    A4A validates only the id format / namespace and the exact key set. It does
    NOT allocate ``canonical_id``, build connected components, or merge aliases
    (A4D). ``candidate_refs`` are cross-chunk (global) candidate references.
    """

    canonical_id: str
    entity_type: str
    display_name_original: str
    aliases_original: tuple[str, ...]
    candidate_refs: tuple[str, ...]
    first_appearance_candidate_ref: str

    def __post_init__(self) -> None:
        _require_enum(self.entity_type, CANONICAL_ENTITY_TYPES,
                      "CanonicalEntity.entity_type")
        _require_canonical_id(
            self.canonical_id, self.entity_type,
            "CanonicalEntity.canonical_id",
        )
        _require_text(
            self.display_name_original, "CanonicalEntity.display_name_original"
        )
        object.__setattr__(
            self,
            "aliases_original",
            _to_string_tuple(self.aliases_original,
                             "CanonicalEntity.aliases_original"),
        )
        object.__setattr__(
            self,
            "candidate_refs",
            _to_string_tuple(
                self.candidate_refs,
                "CanonicalEntity.candidate_refs",
                min_items=1,
                global_ref=True,
            ),
        )
        object.__setattr__(
            self,
            "first_appearance_candidate_ref",
            _require_global_candidate_ref(
                self.first_appearance_candidate_ref,
                "CanonicalEntity.first_appearance_candidate_ref",
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "canonical_id": self.canonical_id,
            "entity_type": self.entity_type,
            "display_name_original": self.display_name_original,
            "aliases_original": list(self.aliases_original),
            "candidate_refs": list(self.candidate_refs),
            "first_appearance_candidate_ref": self.first_appearance_candidate_ref,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CanonicalEntity":
        _require_exact_keys(
            value,
            {
                "canonical_id",
                "entity_type",
                "display_name_original",
                "aliases_original",
                "candidate_refs",
                "first_appearance_candidate_ref",
            },
            "CanonicalEntity",
        )
        for list_field in ("aliases_original", "candidate_refs"):
            if not isinstance(value[list_field], list):
                raise ReconciliationModelError(
                    f"CanonicalEntity.{list_field} must be a list"
                )
        return cls(
            canonical_id=value["canonical_id"],
            entity_type=value["entity_type"],
            display_name_original=value["display_name_original"],
            aliases_original=tuple(value["aliases_original"]),
            candidate_refs=tuple(value["candidate_refs"]),
            first_appearance_candidate_ref=value["first_appearance_candidate_ref"],
        )


def _registry_to_dict(
    schema_version: int, entity_type: str, entities: tuple[CanonicalEntity, ...]
) -> dict[str, Any]:
    return {
        "schema_version": schema_version,
        "entity_type": entity_type,
        "entities": [entity.to_dict() for entity in entities],
    }


def _registry_from_dict(
    value: Any, entity_type: str, cls: type
) -> Any:
    _require_exact_keys(
        value, {"schema_version", "entity_type", "entities"}, cls.__name__
    )
    if value["entity_type"] != entity_type:
        raise ReconciliationModelError(
            f"{cls.__name__}.entity_type must be {entity_type!r} "
            f"(got {value['entity_type']!r})"
        )
    if not isinstance(value["entities"], list):
        raise ReconciliationModelError(
            f"{cls.__name__}.entities must be a list"
        )
    entities = tuple(CanonicalEntity.from_dict(item) for item in value["entities"])
    # Enforce every entity matches the registry's entity_type (the single shared
    # registry schema distinguishes the two registries by entity_type).
    for entity in entities:
        if entity.entity_type != entity_type:
            raise ReconciliationModelError(
                f"{cls.__name__}.entities must all have "
                f"entity_type == {entity_type!r}"
            )
    return cls(schema_version=value["schema_version"], entities=entities)


@dataclass(frozen=True, slots=True)
class CanonicalCharacterRegistry:
    """The registry of resolved canonical characters (entity_type = character)."""

    schema_version: int
    entities: tuple[CanonicalEntity, ...]

    def __post_init__(self) -> None:
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
            or self.schema_version != CANONICAL_ENTITY_REGISTRY_SCHEMA_VERSION
        ):
            raise ReconciliationModelError(
                "CanonicalCharacterRegistry.schema_version must be "
                f"{CANONICAL_ENTITY_REGISTRY_SCHEMA_VERSION}"
            )
        if isinstance(self.entities, CanonicalEntity):
            raise ReconciliationModelError(
                "CanonicalCharacterRegistry.entities must be a list"
            )
        try:
            items = tuple(self.entities)
        except TypeError as exc:
            raise ReconciliationModelError(
                "CanonicalCharacterRegistry.entities must be a list"
            ) from exc
        for item in items:
            if not isinstance(item, CanonicalEntity):
                raise ReconciliationModelError(
                    "CanonicalCharacterRegistry.entities must contain "
                    "CanonicalEntity values"
                )
            if item.entity_type != "character":
                raise ReconciliationModelError(
                    "CanonicalCharacterRegistry.entities must all have "
                    "entity_type == 'character'"
                )
        object.__setattr__(self, "entities", items)

    def to_dict(self) -> dict[str, Any]:
        return _registry_to_dict(self.schema_version, "character", self.entities)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CanonicalCharacterRegistry":
        return _registry_from_dict(value, "character", cls)


@dataclass(frozen=True, slots=True)
class CanonicalLocationRegistry:
    """The registry of resolved canonical locations (entity_type = location)."""

    schema_version: int
    entities: tuple[CanonicalEntity, ...]

    def __post_init__(self) -> None:
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
            or self.schema_version != CANONICAL_ENTITY_REGISTRY_SCHEMA_VERSION
        ):
            raise ReconciliationModelError(
                "CanonicalLocationRegistry.schema_version must be "
                f"{CANONICAL_ENTITY_REGISTRY_SCHEMA_VERSION}"
            )
        if isinstance(self.entities, CanonicalEntity):
            raise ReconciliationModelError(
                "CanonicalLocationRegistry.entities must be a list"
            )
        try:
            items = tuple(self.entities)
        except TypeError as exc:
            raise ReconciliationModelError(
                "CanonicalLocationRegistry.entities must be a list"
            ) from exc
        for item in items:
            if not isinstance(item, CanonicalEntity):
                raise ReconciliationModelError(
                    "CanonicalLocationRegistry.entities must contain "
                    "CanonicalEntity values"
                )
            if item.entity_type != "location":
                raise ReconciliationModelError(
                    "CanonicalLocationRegistry.entities must all have "
                    "entity_type == 'location'"
                )
        object.__setattr__(self, "entities", items)

    def to_dict(self) -> dict[str, Any]:
        return _registry_to_dict(self.schema_version, "location", self.entities)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CanonicalLocationRegistry":
        return _registry_from_dict(value, "location", cls)


# ---------------------------------------------------------------------------
# Unresolved entity contracts
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class UnresolvedEntity:
    """An explicit unresolved identity group / A3 unresolved passthrough (section 25).

    Sources: A3 ``cand_unres_*`` passthrough and A4 uncertain ambiguity groups.
    A4A validates only the id format / domain and exact keys. It does NOT
    compute unresolved groups or assign ``unresolved_id`` (A4D).
    ``candidate_refs`` / ``possible_candidate_refs`` are global references.
    """

    unresolved_id: str
    entity_kind: str
    candidate_refs: tuple[str, ...]
    decision_refs: tuple[str, ...]
    possible_candidate_refs: tuple[str, ...]
    first_appearance_candidate_ref: str

    def __post_init__(self) -> None:
        _require_text(self.unresolved_id, "UnresolvedEntity.unresolved_id")
        if _UNRESOLVED_ID_RE.fullmatch(self.unresolved_id) is None:
            raise ReconciliationModelError(
                f"UnresolvedEntity.unresolved_id must match "
                f"{_UNRESOLVED_ID_RE.pattern!r}: {self.unresolved_id!r}"
            )
        _require_enum(self.entity_kind, UNRESOLVED_ENTITY_KINDS,
                      "UnresolvedEntity.entity_kind")
        object.__setattr__(
            self,
            "candidate_refs",
            _to_string_tuple(
                self.candidate_refs,
                "UnresolvedEntity.candidate_refs",
                min_items=1,
                global_ref=True,
            ),
        )
        object.__setattr__(
            self,
            "decision_refs",
            _to_string_tuple(self.decision_refs,
                             "UnresolvedEntity.decision_refs"),
        )
        object.__setattr__(
            self,
            "possible_candidate_refs",
            _to_string_tuple(
                self.possible_candidate_refs,
                "UnresolvedEntity.possible_candidate_refs",
                global_ref=True,
            ),
        )
        object.__setattr__(
            self,
            "first_appearance_candidate_ref",
            _require_global_candidate_ref(
                self.first_appearance_candidate_ref,
                "UnresolvedEntity.first_appearance_candidate_ref",
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "unresolved_id": self.unresolved_id,
            "entity_kind": self.entity_kind,
            "candidate_refs": list(self.candidate_refs),
            "decision_refs": list(self.decision_refs),
            "possible_candidate_refs": list(self.possible_candidate_refs),
            "first_appearance_candidate_ref": self.first_appearance_candidate_ref,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "UnresolvedEntity":
        _require_exact_keys(
            value,
            {
                "unresolved_id",
                "entity_kind",
                "candidate_refs",
                "decision_refs",
                "possible_candidate_refs",
                "first_appearance_candidate_ref",
            },
            "UnresolvedEntity",
        )
        for list_field in ("candidate_refs", "decision_refs",
                           "possible_candidate_refs"):
            if not isinstance(value[list_field], list):
                raise ReconciliationModelError(
                    f"UnresolvedEntity.{list_field} must be a list"
                )
        return cls(
            unresolved_id=value["unresolved_id"],
            entity_kind=value["entity_kind"],
            candidate_refs=tuple(value["candidate_refs"]),
            decision_refs=tuple(value["decision_refs"]),
            possible_candidate_refs=tuple(value["possible_candidate_refs"]),
            first_appearance_candidate_ref=value["first_appearance_candidate_ref"],
        )


@dataclass(frozen=True, slots=True)
class UnresolvedEntitySet:
    """The set of explicit unresolved entities."""

    schema_version: int
    entities: tuple[UnresolvedEntity, ...]

    def __post_init__(self) -> None:
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
            or self.schema_version != UNRESOLVED_ENTITY_SET_SCHEMA_VERSION
        ):
            raise ReconciliationModelError(
                "UnresolvedEntitySet.schema_version must be "
                f"{UNRESOLVED_ENTITY_SET_SCHEMA_VERSION}"
            )
        if isinstance(self.entities, UnresolvedEntity):
            raise ReconciliationModelError(
                "UnresolvedEntitySet.entities must be a list"
            )
        try:
            items = tuple(self.entities)
        except TypeError as exc:
            raise ReconciliationModelError(
                "UnresolvedEntitySet.entities must be a list"
            ) from exc
        for item in items:
            if not isinstance(item, UnresolvedEntity):
                raise ReconciliationModelError(
                    "UnresolvedEntitySet.entities must contain "
                    "UnresolvedEntity values"
                )
        object.__setattr__(self, "entities", items)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "entities": [entity.to_dict() for entity in self.entities],
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "UnresolvedEntitySet":
        _require_exact_keys(
            value, {"schema_version", "entities"}, "UnresolvedEntitySet"
        )
        if not isinstance(value["entities"], list):
            raise ReconciliationModelError(
                "UnresolvedEntitySet.entities must be a list"
            )
        return cls(
            schema_version=value["schema_version"],
            entities=tuple(
                UnresolvedEntity.from_dict(item) for item in value["entities"]
            ),
        )


# ---------------------------------------------------------------------------
# EntityMap contracts
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class A3InputIdentity:
    """The exact upstream A3 identity material the EntityMap pins (section 28).

    This is the precise A3 input set A4 consumed: the exact SourceDocument and
    ChunkManifest, the exact ordered CandidateExtraction refs (the A3 CURRENT
    set), and the A3 extraction-profile identity. A4A does NOT resolve, reuse,
    or validate these against the store (A4D); it only pins their refs/identity.
    """

    source_document_ref: ArtifactRef
    chunk_manifest_ref: ArtifactRef
    candidate_extraction_refs: tuple[ArtifactRef, ...]
    extraction_profile_id: str
    extraction_profile_hash: str

    def __post_init__(self) -> None:
        _require_artifact_ref(
            self.source_document_ref, "A3InputIdentity.source_document_ref"
        )
        _require_artifact_ref(
            self.chunk_manifest_ref, "A3InputIdentity.chunk_manifest_ref"
        )
        object.__setattr__(
            self,
            "candidate_extraction_refs",
            _to_artifact_ref_tuple(
                self.candidate_extraction_refs,
                "A3InputIdentity.candidate_extraction_refs",
                min_items=1,
            ),
        )
        _require_storage_id(
            self.extraction_profile_id, "A3InputIdentity.extraction_profile_id"
        )
        _require_hash(
            self.extraction_profile_hash, "A3InputIdentity.extraction_profile_hash"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_document_ref": self.source_document_ref.to_dict(),
            "chunk_manifest_ref": self.chunk_manifest_ref.to_dict(),
            "candidate_extraction_refs": [
                ref.to_dict() for ref in self.candidate_extraction_refs
            ],
            "extraction_profile_id": self.extraction_profile_id,
            "extraction_profile_hash": self.extraction_profile_hash,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "A3InputIdentity":
        _require_exact_keys(
            value,
            {
                "source_document_ref",
                "chunk_manifest_ref",
                "candidate_extraction_refs",
                "extraction_profile_id",
                "extraction_profile_hash",
            },
            "A3InputIdentity",
        )
        if not isinstance(value["candidate_extraction_refs"], list):
            raise ReconciliationModelError(
                "A3InputIdentity.candidate_extraction_refs must be a list"
            )
        try:
            source_document_ref = ArtifactRef.from_dict(value["source_document_ref"])
            chunk_manifest_ref = ArtifactRef.from_dict(value["chunk_manifest_ref"])
            candidate_extraction_refs = tuple(
                ArtifactRef.from_dict(item)
                for item in value["candidate_extraction_refs"]
            )
        except Exception as exc:  # noqa: BLE001
            raise ReconciliationModelError(f"invalid A3 input artifact ref: {exc}") from exc
        return cls(
            source_document_ref=source_document_ref,
            chunk_manifest_ref=chunk_manifest_ref,
            candidate_extraction_refs=candidate_extraction_refs,
            extraction_profile_id=value["extraction_profile_id"],
            extraction_profile_hash=value["extraction_profile_hash"],
        )


@dataclass(frozen=True, slots=True)
class EntityMapEntry:
    """One coverage-universe candidate's final resolution (section 26).

    Strict, fail-closed exclusivity invariant:
      * ``resolved``   -> ``canonical_id != null`` AND ``unresolved_id == null``
      * ``unresolved`` -> ``canonical_id == null`` AND ``unresolved_id != null``

    Both-set and neither-set are rejected. A4A validates the exclusivity and the
    id formats; it does NOT audit coverage/unaccounted/duplicate (A4D).
    """

    candidate_ref: str
    status: str
    canonical_id: str | None
    unresolved_id: str | None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "candidate_ref",
            _require_global_candidate_ref(
                self.candidate_ref, "EntityMapEntry.candidate_ref"
            ),
        )
        _require_enum(self.status, ENTITY_MAP_STATUSES, "EntityMapEntry.status")
        _require_optional_text(self.canonical_id, "EntityMapEntry.canonical_id")
        _require_optional_text(self.unresolved_id, "EntityMapEntry.unresolved_id")
        self._check_exclusivity()

    def _check_exclusivity(self) -> None:
        if self.status == "resolved":
            if self.canonical_id is None:
                raise ReconciliationModelError(
                    "EntityMapEntry.canonical_id is required when "
                    "status == 'resolved'"
                )
            if not _ANY_CANONICAL_ID_RE.fullmatch(self.canonical_id):
                raise ReconciliationModelError(
                    f"EntityMapEntry.canonical_id must be a canonical "
                    f"char_/loc_ id: {self.canonical_id!r}"
                )
            if self.unresolved_id is not None:
                raise ReconciliationModelError(
                    "EntityMapEntry.unresolved_id must be null when "
                    "status == 'resolved'"
                )
        else:  # unresolved
            if self.unresolved_id is None:
                raise ReconciliationModelError(
                    "EntityMapEntry.unresolved_id is required when "
                    "status == 'unresolved'"
                )
            if not _UNRESOLVED_ID_RE.fullmatch(self.unresolved_id):
                raise ReconciliationModelError(
                    f"EntityMapEntry.unresolved_id must be an unres_ id: "
                    f"{self.unresolved_id!r}"
                )
            if self.canonical_id is not None:
                raise ReconciliationModelError(
                    "EntityMapEntry.canonical_id must be null when "
                    "status == 'unresolved'"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_ref": self.candidate_ref,
            "status": self.status,
            "canonical_id": self.canonical_id,
            "unresolved_id": self.unresolved_id,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "EntityMapEntry":
        _require_exact_keys(
            value,
            {"candidate_ref", "status", "canonical_id", "unresolved_id"},
            "EntityMapEntry",
        )
        return cls(
            candidate_ref=value["candidate_ref"],
            status=value["status"],
            canonical_id=value["canonical_id"],
            unresolved_id=value["unresolved_id"],
        )


@dataclass(frozen=True, slots=True)
class A4SemanticIdentity:
    """The exact backend-neutral A4 semantic-generation identity (EntityMap v2).

    This is the precise A4 semantic-identity material pinned by the final
    aggregate authority so that CURRENT reuse can be reliably decided *before*
    any provider call. It is deliberately backend-neutral: it carries NO
    endpoint, provider family, concrete model, provider response id, credential,
    timeout, wall clock, or llama.cpp slot. Changing only the backend routing
    (Qwen -> Gemma) therefore does NOT invalidate A4 reuse.

    Backend provenance still lives in each ``ReconciliationDecision``
    ``generation_provenance`` for audit, but it never participates in this
    reuse identity.

    All hash fields are lowercase 64-char SHA-256. ``semantic_request_hashes``
    is in canonical block order, unique, and may be empty (the zero-semantic-
    pair case).
    """

    reconciliation_profile_id: str
    reconciliation_profile_hash: str
    semantic_profile_id: str
    semantic_profile_hash: str
    prompt_id: str
    prompt_version: int
    prompt_content_hash: str
    output_schema_id: str
    output_schema_version: int
    output_schema_hash: str
    plan_hash: str
    semantic_request_hashes: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_storage_id(
            self.reconciliation_profile_id,
            "A4SemanticIdentity.reconciliation_profile_id",
        )
        _require_hash(
            self.reconciliation_profile_hash,
            "A4SemanticIdentity.reconciliation_profile_hash",
        )
        _require_storage_id(
            self.semantic_profile_id, "A4SemanticIdentity.semantic_profile_id"
        )
        _require_hash(
            self.semantic_profile_hash, "A4SemanticIdentity.semantic_profile_hash"
        )
        _require_storage_id(self.prompt_id, "A4SemanticIdentity.prompt_id")
        _require_positive_int(
            self.prompt_version, "A4SemanticIdentity.prompt_version"
        )
        _require_hash(
            self.prompt_content_hash, "A4SemanticIdentity.prompt_content_hash"
        )
        _require_storage_id(
            self.output_schema_id, "A4SemanticIdentity.output_schema_id"
        )
        _require_positive_int(
            self.output_schema_version, "A4SemanticIdentity.output_schema_version"
        )
        _require_hash(
            self.output_schema_hash, "A4SemanticIdentity.output_schema_hash"
        )
        _require_hash(self.plan_hash, "A4SemanticIdentity.plan_hash")
        object.__setattr__(
            self,
            "semantic_request_hashes",
            _to_string_tuple(
                self.semantic_request_hashes,
                "A4SemanticIdentity.semantic_request_hashes",
                unique=True,
            ),
        )
        for h in self.semantic_request_hashes:
            _require_hash(h, "A4SemanticIdentity.semantic_request_hashes item")

    def to_dict(self) -> dict[str, Any]:
        return {
            "reconciliation_profile_id": self.reconciliation_profile_id,
            "reconciliation_profile_hash": self.reconciliation_profile_hash,
            "semantic_profile_id": self.semantic_profile_id,
            "semantic_profile_hash": self.semantic_profile_hash,
            "prompt_id": self.prompt_id,
            "prompt_version": self.prompt_version,
            "prompt_content_hash": self.prompt_content_hash,
            "output_schema_id": self.output_schema_id,
            "output_schema_version": self.output_schema_version,
            "output_schema_hash": self.output_schema_hash,
            "plan_hash": self.plan_hash,
            "semantic_request_hashes": list(self.semantic_request_hashes),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "A4SemanticIdentity":
        _require_exact_keys(
            value,
            {
                "reconciliation_profile_id",
                "reconciliation_profile_hash",
                "semantic_profile_id",
                "semantic_profile_hash",
                "prompt_id",
                "prompt_version",
                "prompt_content_hash",
                "output_schema_id",
                "output_schema_version",
                "output_schema_hash",
                "plan_hash",
                "semantic_request_hashes",
            },
            "A4SemanticIdentity",
        )
        if not isinstance(value["semantic_request_hashes"], list):
            raise ReconciliationModelError(
                "A4SemanticIdentity.semantic_request_hashes must be a list"
            )
        return cls(
            reconciliation_profile_id=value["reconciliation_profile_id"],
            reconciliation_profile_hash=value["reconciliation_profile_hash"],
            semantic_profile_id=value["semantic_profile_id"],
            semantic_profile_hash=value["semantic_profile_hash"],
            prompt_id=value["prompt_id"],
            prompt_version=value["prompt_version"],
            prompt_content_hash=value["prompt_content_hash"],
            output_schema_id=value["output_schema_id"],
            output_schema_version=value["output_schema_version"],
            output_schema_hash=value["output_schema_hash"],
            plan_hash=value["plan_hash"],
            semantic_request_hashes=tuple(value["semantic_request_hashes"]),
        )


@dataclass(frozen=True, slots=True)
class EntityMap:
    """A4's final aggregate authority (EntityMap v2).

    Pins the exact refs to every A4 output (CandidateEntityIndex,
    ReconciliationDecisionSet, both canonical registries, UnresolvedEntitySet)
    plus the exact upstream A3 identity material (``a3_input``) and the exact
    backend-neutral A4 semantic-generation identity (``semantic_identity``).
    A4A establishes only the shape: it does NOT persist the EntityMap, create a
    CURRENT pointer, or audit coverage (A4D).
    """

    schema_version: int
    entries: tuple[EntityMapEntry, ...]
    candidate_entity_index_ref: ArtifactRef
    reconciliation_decision_set_ref: ArtifactRef
    canonical_character_registry_ref: ArtifactRef
    canonical_location_registry_ref: ArtifactRef
    unresolved_entity_set_ref: ArtifactRef
    a3_input: A3InputIdentity
    semantic_identity: A4SemanticIdentity

    def __post_init__(self) -> None:
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
            or self.schema_version != ENTITY_MAP_SCHEMA_VERSION
        ):
            raise ReconciliationModelError(
                "EntityMap.schema_version must be "
                f"{ENTITY_MAP_SCHEMA_VERSION}"
            )
        if isinstance(self.entries, EntityMapEntry):
            raise ReconciliationModelError(
                "EntityMap.entries must be a list of entries"
            )
        try:
            items = tuple(self.entries)
        except TypeError as exc:
            raise ReconciliationModelError(
                "EntityMap.entries must be a list of entries"
            ) from exc
        for item in items:
            if not isinstance(item, EntityMapEntry):
                raise ReconciliationModelError(
                    "EntityMap.entries must contain EntityMapEntry values"
                )
        object.__setattr__(self, "entries", items)
        _require_artifact_ref(
            self.candidate_entity_index_ref,
            "EntityMap.candidate_entity_index_ref",
        )
        _require_artifact_ref(
            self.reconciliation_decision_set_ref,
            "EntityMap.reconciliation_decision_set_ref",
        )
        _require_artifact_ref(
            self.canonical_character_registry_ref,
            "EntityMap.canonical_character_registry_ref",
        )
        _require_artifact_ref(
            self.canonical_location_registry_ref,
            "EntityMap.canonical_location_registry_ref",
        )
        _require_artifact_ref(
            self.unresolved_entity_set_ref,
            "EntityMap.unresolved_entity_set_ref",
        )
        if not isinstance(self.a3_input, A3InputIdentity):
            raise ReconciliationModelError(
                "EntityMap.a3_input must be an A3InputIdentity"
            )
        if not isinstance(self.semantic_identity, A4SemanticIdentity):
            raise ReconciliationModelError(
                "EntityMap.semantic_identity must be an A4SemanticIdentity"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "entries": [entry.to_dict() for entry in self.entries],
            "candidate_entity_index_ref": self.candidate_entity_index_ref.to_dict(),
            "reconciliation_decision_set_ref": (
                self.reconciliation_decision_set_ref.to_dict()
            ),
            "canonical_character_registry_ref": (
                self.canonical_character_registry_ref.to_dict()
            ),
            "canonical_location_registry_ref": (
                self.canonical_location_registry_ref.to_dict()
            ),
            "unresolved_entity_set_ref": self.unresolved_entity_set_ref.to_dict(),
            "a3_input": self.a3_input.to_dict(),
            "semantic_identity": self.semantic_identity.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "EntityMap":
        _require_exact_keys(
            value,
            {
                "schema_version",
                "entries",
                "candidate_entity_index_ref",
                "reconciliation_decision_set_ref",
                "canonical_character_registry_ref",
                "canonical_location_registry_ref",
                "unresolved_entity_set_ref",
                "a3_input",
                "semantic_identity",
            },
            "EntityMap",
        )
        if not isinstance(value["entries"], list):
            raise ReconciliationModelError("EntityMap.entries must be a list")
        try:
            a3_input = A3InputIdentity.from_dict(value["a3_input"])
            semantic_identity = A4SemanticIdentity.from_dict(value["semantic_identity"])
            refs = {
                name: ArtifactRef.from_dict(value[name])
                for name in (
                    "candidate_entity_index_ref",
                    "reconciliation_decision_set_ref",
                    "canonical_character_registry_ref",
                    "canonical_location_registry_ref",
                    "unresolved_entity_set_ref",
                )
            }
        except Exception as exc:  # noqa: BLE001
            raise ReconciliationModelError(f"invalid EntityMap artifact ref: {exc}") from exc
        return cls(
            schema_version=value["schema_version"],
            entries=tuple(
                EntityMapEntry.from_dict(item) for item in value["entries"]
            ),
            candidate_entity_index_ref=refs["candidate_entity_index_ref"],
            reconciliation_decision_set_ref=refs["reconciliation_decision_set_ref"],
            canonical_character_registry_ref=refs["canonical_character_registry_ref"],
            canonical_location_registry_ref=refs["canonical_location_registry_ref"],
            unresolved_entity_set_ref=refs["unresolved_entity_set_ref"],
            a3_input=a3_input,
            semantic_identity=semantic_identity,
        )


# ---------------------------------------------------------------------------
# EntityReconciliationProfile
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EntityReconciliationProfile:
    """Model-independent orchestration profile for A4 entity reconciliation.

    Deliberately separate from the A-I3 runtime/semantic LLM profile: it carries
    no endpoint, credential, timeout, concrete model, temperature, or reasoning
    field, and no backend routing identity. Its canonical hash (over ``to_dict()``
    via the shared ``content_hash``) is the ``reconciliation_profile_hash`` that
    participates in A4 reuse identity (section 9 / 31).
    """

    schema_version: int
    profile_id: str
    working_language: str
    name_normalization_policy_id: str
    blocking_policy_id: str
    canonicalization_policy_id: str
    prompt_id: str
    prompt_version: int
    output_schema_id: str
    output_schema_version: int
    max_generation_rounds: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
            or self.schema_version != RECONCILIATION_PROFILE_SCHEMA_VERSION
        ):
            raise ReconciliationModelError(
                "EntityReconciliationProfile.schema_version must be "
                f"{RECONCILIATION_PROFILE_SCHEMA_VERSION}"
            )
        _require_storage_id(self.profile_id, "EntityReconciliationProfile.profile_id")
        _require_text(
            self.working_language, "EntityReconciliationProfile.working_language"
        )
        _require_storage_id(
            self.name_normalization_policy_id,
            "EntityReconciliationProfile.name_normalization_policy_id",
        )
        _require_storage_id(
            self.blocking_policy_id, "EntityReconciliationProfile.blocking_policy_id"
        )
        _require_storage_id(
            self.canonicalization_policy_id,
            "EntityReconciliationProfile.canonicalization_policy_id",
        )
        _require_storage_id(self.prompt_id, "EntityReconciliationProfile.prompt_id")
        _require_positive_int(
            self.prompt_version, "EntityReconciliationProfile.prompt_version"
        )
        _require_storage_id(
            self.output_schema_id, "EntityReconciliationProfile.output_schema_id"
        )
        _require_positive_int(
            self.output_schema_version,
            "EntityReconciliationProfile.output_schema_version",
        )
        _require_positive_int(
            self.max_generation_rounds,
            "EntityReconciliationProfile.max_generation_rounds",
        )
        if (
            self.schema_version == RECONCILIATION_PROFILE_SCHEMA_VERSION
            and self.max_generation_rounds != RECONCILIATION_MAX_GENERATION_ROUNDS_V1
        ):
            raise ReconciliationModelError(
                "EntityReconciliationProfile.max_generation_rounds must be "
                f"{RECONCILIATION_MAX_GENERATION_ROUNDS_V1} for schema_version "
                f"{RECONCILIATION_PROFILE_SCHEMA_VERSION} (frozen A-I5 v1 ceiling)"
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
            "name_normalization_policy_id": self.name_normalization_policy_id,
            "blocking_policy_id": self.blocking_policy_id,
            "canonicalization_policy_id": self.canonicalization_policy_id,
            "prompt_id": self.prompt_id,
            "prompt_version": self.prompt_version,
            "output_schema_id": self.output_schema_id,
            "output_schema_version": self.output_schema_version,
            "max_generation_rounds": self.max_generation_rounds,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "EntityReconciliationProfile":
        _require_exact_keys(
            value,
            {
                "schema_version",
                "profile_id",
                "working_language",
                "name_normalization_policy_id",
                "blocking_policy_id",
                "canonicalization_policy_id",
                "prompt_id",
                "prompt_version",
                "output_schema_id",
                "output_schema_version",
                "max_generation_rounds",
            },
            "EntityReconciliationProfile",
        )
        return cls(**value)


def load_entity_reconciliation_profile(
    path: str | Path,
) -> EntityReconciliationProfile:
    """Load and validate a tracked EntityReconciliationProfile from a YAML file."""
    path = Path(path).expanduser()
    if not path.is_file():
        raise ReconciliationModelError(
            f"entity reconciliation profile not found: {path}"
        )
    try:
        data = load_yaml(path)
    except Exception as exc:  # noqa: BLE001 - report any load failure
        raise ReconciliationModelError(
            f"failed to load entity reconciliation profile: {exc}"
        ) from exc
    if not isinstance(data, dict):
        raise ReconciliationModelError(
            "entity reconciliation profile must contain an object"
        )
    try:
        return EntityReconciliationProfile.from_dict(data)
    except ReconciliationModelError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ReconciliationModelError(
            f"invalid entity reconciliation profile: {exc}"
        ) from exc
