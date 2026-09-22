"""v1.2 A4B — deterministic indexing and blocking.

This module implements the deterministic reconciliation planning slice:

    validated typed snapshot
        ↓
    snapshot coherence validation (fail closed)
        ↓
    CandidateEntityIndex construction (global refs, source_order_key)
        ↓
    name normalization v1
        ↓
    identity-key / blocking-token extraction
        ↓
    strong/weak identity-key classification
        ↓
    deterministic must-not-merge v1 (empty production set)
        ↓
    indexed candidate blocking (no whole-document N²)
        ↓
    pair state assignment (auto_same / must_not_merge / needs_semantic_decision)
        ↓
    deterministic ambiguity-plan hash

A4B does NOT:
  * invoke an LLM or build a semantic request (A4C);
  * build the final same/different/uncertain identity graph (A4D);
  * persist A4 artifacts, write CURRENT pointers, or reuse (A4D);
  * provide a stage CLI or a real-novel smoke (A4E).

Style follows the story package: frozen dataclasses, explicit ``to_dict()``,
fail-closed validation, deterministic serialization.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

from short_drama.artifacts import ArtifactRef, content_hash
from short_drama.story.extraction import (
    CandidateExtraction,
    CharacterCandidate,
    EvidenceRef,
    LocationCandidate,
    UnresolvedMentionCandidate,
)
from short_drama.story.extraction_persistence import (
    candidate_extraction_artifact_id,
)
from short_drama.story.persistence import (
    source_chunk_artifact_id,
    source_document_artifact_id,
)
from short_drama.story.reconciliation import (
    CANDIDATE_ENTITY_INDEX_SCHEMA_VERSION,
    CandidateEntityIndex,
    CandidateEntityIndexEntry,
    GlobalCandidateRef,
    ReconciliationDecision,
)
from short_drama.story.source import SourceDocument
from short_drama.story.chunking import ChunkManifest, SourceChunk

from .errors import ReconciliationPlanningError

# ---------------------------------------------------------------------------
# Policy identifiers (frozen)
# ---------------------------------------------------------------------------

NAME_NORMALIZATION_POLICY_ID = "a4-name-normalization-v1"
BLOCKING_POLICY_ID = "a4-blocking-v2"
CANONICALIZATION_POLICY_ID = "a4-canonicalization-v1"

# ---------------------------------------------------------------------------
# a4-blocking-v2 minimal function-word exclusion (frozen)
# ---------------------------------------------------------------------------
# This set belongs ONLY to identity-token blocking overlap (the
# IDENTITY_TOKEN_OVERLAP signal and the shared_tokens recorded in a pair plan).
# It is NOT a generic stopword list and must not be applied to
# extract_identity_keys(), extract_blocking_tokens() (raw extraction),
# is_strong_identity_key(), exact identity-key authority, or persisted
# names/aliases. See docs/v1.2-A4B-blocking-v2-refinement-plan.md.
_BLOCKING_TOKEN_EXCLUSIONS_V2: frozenset[str] = frozenset({"the", "of"})

# ---------------------------------------------------------------------------
# Category ordinals for source_order_key
# ---------------------------------------------------------------------------

_CATEGORY_ORDINAL = {
    "character": 1,
    "location": 2,
    "unresolved": 3,
}

# ---------------------------------------------------------------------------
# Weak/generic keys v1 (frozen exact-key set)
# ---------------------------------------------------------------------------

_WEAK_GENERIC_KEYS: frozenset[str] = frozenset(
    {
        # English roles/titles
        "mr",
        "mrs",
        "ms",
        "miss",
        "sir",
        "madam",
        "captain",
        "doctor",
        "teacher",
        "professor",
        "student",
        "mother",
        "father",
        "mom",
        "mum",
        "dad",
        "boy",
        "girl",
        "man",
        "woman",
        "the teacher",
        "the doctor",
        "the captain",
        # zh-CN common role/title mentions
        "老师",
        "医生",
        "教授",
        "学生",
        "同学",
        "先生",
        "女士",
        "小姐",
        "队长",
        "父亲",
        "母亲",
        "爸爸",
        "妈妈",
        "男孩",
        "女孩",
        "男人",
        "女人",
        "班主任",
    }
)

# ---------------------------------------------------------------------------
# CJK Unified Ideograph range (U+4E00–U+9FFF)
# ---------------------------------------------------------------------------

_CJK_UNIFIED_RE = re.compile(r"[\u4e00-\u9fff]")
# Alphanumeric run for token extraction: Unicode alphanumeric code points
_WHITESPACE_RE = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# Name normalization v1
# ---------------------------------------------------------------------------


def normalize_name(text: str) -> str:
    """Apply ``a4-name-normalization-v1``.

    Exact algorithm:
      1. Unicode NFKC
      2. Unicode casefold
      3. strip leading/trailing whitespace
      4. collapse Unicode whitespace runs to one ASCII space

    This function is idempotent: ``normalize_name(normalize_name(s)) ==
    normalize_name(s)``.
    """
    result = unicodedata.normalize("NFKC", text)
    result = result.casefold()
    result = result.strip()
    # Collapse all Unicode whitespace runs to a single ASCII space
    result = _WHITESPACE_RE.sub(" ", result)
    return result


# ---------------------------------------------------------------------------
# Identity keys
# ---------------------------------------------------------------------------


def extract_identity_keys(
    display_name_original: str,
    aliases_original: tuple[str, ...],
) -> tuple[str, ...]:
    """Extract normalized identity keys from a character/location candidate.

    Returns deduplicated, lexically sorted tuple of non-empty normalized keys.
    """
    keys: set[str] = set()
    normalized_display = normalize_name(display_name_original)
    if normalized_display:
        keys.add(normalized_display)
    for alias in aliases_original:
        normalized_alias = normalize_name(alias)
        if normalized_alias:
            keys.add(normalized_alias)
    return tuple(sorted(keys))


# ---------------------------------------------------------------------------
# Blocking tokens
# ---------------------------------------------------------------------------


def extract_blocking_tokens(identity_keys: tuple[str, ...]) -> tuple[str, ...]:
    """Extract deterministic blocking tokens from normalized identity keys.

    Rules:
      * split on non-alphanumeric boundaries (true Unicode alphanumeric
        via ``str.isalnum()``);
      * drop empty tokens;
      * token must be at least 2 Unicode code points;
      * pure numeric tokens are excluded;
      * unique, lexically sorted.
    """
    tokens: set[str] = set()
    for key in identity_keys:
        # key is already normalized (NFKC + casefold + strip + collapse)
        # Scan for Unicode alphanumeric runs
        run: list[str] = []
        for ch in key:
            if ch.isalnum():
                run.append(ch)
            else:
                if run:
                    token = "".join(run)
                    run = []
                    if len(token) >= 2 and not token.isdigit():
                        tokens.add(token)
        if run:
            token = "".join(run)
            if len(token) >= 2 and not token.isdigit():
                tokens.add(token)
    return tuple(sorted(tokens))


def effective_blocking_tokens(
    raw_blocking_tokens: tuple[str, ...],
) -> tuple[str, ...]:
    """Apply the a4-blocking-v2 minimal function-word exclusion.

    Pipeline: raw identity tokens → remove the frozen set → effective tokens.

    This filter belongs ONLY to identity-token blocking overlap (the
    IDENTITY_TOKEN_OVERLAP signal and the shared_tokens recorded in a pair
    plan). It must NOT change ``normalize_name()``, ``extract_identity_keys()``,
    the raw token-extraction semantics used by strong-key classification,
    ``is_strong_identity_key()``, exact identity-key authority, or persisted
    names/aliases. The narrow seam is intentionally applied at the blocking
    boundary (see ``_plan_from_candidate_index``), not inside
    ``extract_blocking_tokens`` or ``is_strong_identity_key``.

    Deterministic and idempotent: the result is the sorted, unique subset of
    the input that excludes ``_BLOCKING_TOKEN_EXCLUSIONS_V2``.
    """
    return tuple(
        sorted(
            {
                token
                for token in raw_blocking_tokens
                if token not in _BLOCKING_TOKEN_EXCLUSIONS_V2
            }
        )
    )


# ---------------------------------------------------------------------------
# Strong/weak identity key classification
# ---------------------------------------------------------------------------


def _count_cjk_unified(text: str) -> int:
    """Count CJK Unified Ideograph characters in text."""
    return len(_CJK_UNIFIED_RE.findall(text))


def is_strong_identity_key(key: str) -> bool:
    """Classify a normalized identity key as strong (v1 conservative).

    Strong requires ALL of:
      1. non-empty
      2. contains at least one Unicode alphabetic character
      3. is not a frozen weak/generic key
      4. AND (at least 2 alphanumeric tokens OR at least 3 CJK Unified
         Ideographs)
    """
    if not key:
        return False
    # Must contain at least one Unicode alphabetic character
    if not any(ch.isalpha() for ch in key):
        return False
    # Must not be a weak/generic key
    if key in _WEAK_GENERIC_KEYS:
        return False
    # Must satisfy structural requirement:
    #   >= 2 alphanumeric tokens OR >= 3 CJK Unified Ideographs
    tokens = extract_blocking_tokens((key,))
    cjk_count = _count_cjk_unified(key)
    if len(tokens) >= 2:
        return True
    if cjk_count >= 3:
        return True
    return False


# ---------------------------------------------------------------------------
# Deterministic must-not-merge v1
# ---------------------------------------------------------------------------


def derive_must_not_merge_constraints(
    candidate_index: CandidateEntityIndex,
) -> frozenset[tuple[str, str]]:
    """Derive the production must-not-merge constraint set (v1: empty).

    The v1 policy does NOT infer hard-distinctness from:
      * EventCandidate participant co-occurrence
      * RelationshipCandidate source/target
      * different descriptions / occupations / appearance / names
      * source proximity

    A3 local distinct refs != proven global distinct identities.

    Returns an empty immutable set. The concept is preserved for future
    versioned rules; tests may inject synthetic constraints into the lower-level
    planner to prove precedence.
    """
    return frozenset()


# ---------------------------------------------------------------------------
# Source order key
# ---------------------------------------------------------------------------


def compute_source_order_key(
    *,
    chunk_ordinal: int,
    paragraph_ordinal: int,
    category: str,
    candidate_suffix: int,
    global_ref: str,
) -> str:
    """Compute the exact deterministic source_order_key.

    Format:
        {chunk_ordinal:06d}:{paragraph_ordinal:09d}:{category_ordinal:02d}:
        {candidate_suffix:09d}:{candidate_ref}
    """
    category_ordinal = _CATEGORY_ORDINAL[category]
    return (
        f"{chunk_ordinal:06d}"
        f":{paragraph_ordinal:09d}"
        f":{category_ordinal:02d}"
        f":{candidate_suffix:09d}"
        f":{global_ref}"
    )


def _local_candidate_suffix(local_id: str) -> int:
    """Extract the numeric suffix from a local candidate id (e.g. cand_char_002 → 2)."""
    return int(local_id.rsplit("_", 1)[1])


# ---------------------------------------------------------------------------
# Frozen source_order_key parser / validator (A4D replanning + source-order gate)
# ---------------------------------------------------------------------------

# Exact frozen A4B source_order_key shape:
#   {chunk_ordinal:06d}:{paragraph_ordinal:09d}:{category_ordinal:02d}:
#   {candidate_suffix:09d}:{candidate_ref}
# where candidate_ref == CH<digits>_C<digits>:cand_(char|loc|unres)_<digits>
_SOURCE_ORDER_KEY_RE = re.compile(
    r"^(?P<chunk_ordinal>[0-9]{6}):"
    r"(?P<paragraph_ordinal>[0-9]{9}):"
    r"(?P<category_ordinal>[0-9]{2}):"
    r"(?P<candidate_suffix>[0-9]{9}):"
    r"(?P<candidate_ref>CH[0-9]{3,}_C[0-9]{3,}:cand_(?:char|loc|unres)_[0-9]{3,})$"
)


def _category_ordinal_for_kind(candidate_kind: str) -> int:
    """Map a candidate_kind to its frozen source_order_key category ordinal."""
    if candidate_kind == "character":
        return 1
    if candidate_kind == "location":
        return 2
    if candidate_kind.startswith("unresolved_"):
        return 3
    raise ReconciliationPlanningError(f"unknown candidate_kind: {candidate_kind!r}")


@dataclass(frozen=True, slots=True)
class ParsedSourceOrderKey:
    """The parsed fields of a frozen A4B ``source_order_key``."""

    chunk_ordinal: int
    paragraph_ordinal: int
    category_ordinal: int
    candidate_suffix: int
    candidate_ref: str


def parse_source_order_key(key: str) -> ParsedSourceOrderKey:
    """Parse a frozen A4B ``source_order_key`` into its deterministic fields.

    Fails closed unless the key exactly matches the frozen shape. This is the
    single authority for recovering ``chunk_ordinal`` (adjacency blocking) from
    a persisted :class:`CandidateEntityIndex`.
    """
    match = _SOURCE_ORDER_KEY_RE.match(key)
    if match is None:
        raise ReconciliationPlanningError(
            f"source_order_key {key!r} does not match the frozen A4B format"
        )
    return ParsedSourceOrderKey(
        chunk_ordinal=int(match["chunk_ordinal"]),
        paragraph_ordinal=int(match["paragraph_ordinal"]),
        category_ordinal=int(match["category_ordinal"]),
        candidate_suffix=int(match["candidate_suffix"]),
        candidate_ref=match["candidate_ref"],
    )


def validate_candidate_index_source_order(
    candidate_index: CandidateEntityIndex,
) -> dict[str, int]:
    """Strictly validate a ``CandidateEntityIndex`` source-order material.

    Requires, simultaneously (fail closed on any violation):
      * every ``source_order_key`` matches the exact frozen shape;
      * the embedded candidate_ref equals the entry's candidate_ref;
      * the category ordinal matches the entry's candidate_kind;
      * the candidate numeric suffix matches the ref's local candidate id;
      * candidate refs are unique;
      * entries are strictly ascending by source_order_key.

    Returns the ``candidate_ref -> chunk_ordinal`` mapping recovered for
    adjacency blocking.
    """
    chunk_ordinal_map: dict[str, int] = {}
    seen_refs: set[str] = set()
    ordered_keys: list[str] = []
    for entry in candidate_index.entries:
        ref = entry.candidate_ref
        if ref in seen_refs:
            raise ReconciliationPlanningError(
                f"duplicate candidate_ref in candidate index: {ref!r}"
            )
        seen_refs.add(ref)
        parsed = parse_source_order_key(entry.source_order_key)
        if parsed.candidate_ref != ref:
            raise ReconciliationPlanningError(
                f"source_order_key embedded candidate_ref "
                f"{parsed.candidate_ref!r} != entry candidate_ref {ref!r}"
            )
        expected_ordinal = _category_ordinal_for_kind(entry.candidate_kind)
        if parsed.category_ordinal != expected_ordinal:
            raise ReconciliationPlanningError(
                f"source_order_key category ordinal {parsed.category_ordinal!r} "
                f"does not match candidate_kind {entry.candidate_kind!r} for "
                f"ref {ref!r}"
            )
        local_id = ref.split(":", 1)[1]
        if parsed.candidate_suffix != _local_candidate_suffix(local_id):
            raise ReconciliationPlanningError(
                f"source_order_key candidate suffix {parsed.candidate_suffix!r} "
                f"does not match ref {ref!r}"
            )
        chunk_ordinal_map[ref] = parsed.chunk_ordinal
        ordered_keys.append(entry.source_order_key)
    for i in range(1, len(ordered_keys)):
        if ordered_keys[i] <= ordered_keys[i - 1]:
            raise ReconciliationPlanningError(
                f"candidate index entries are not strictly ascending by "
                f"source_order_key at position {i}"
            )
    return chunk_ordinal_map


def _earliest_evidence_paragraph_ordinal(
    evidence: tuple[EvidenceRef, ...],
    paragraph_index: dict[str, int],
) -> int:
    """Find the earliest evidence paragraph ordinal (1-based position in
    SourceDocument.paragraphs).

    Fail closed if no evidence paragraph resolves.
    """
    min_ordinal: int | None = None
    for ev in evidence:
        ordinal = paragraph_index.get(ev.paragraph_id)
        if ordinal is None:
            raise ReconciliationPlanningError(
                f"evidence paragraph {ev.paragraph_id!r} not found in "
                "SourceDocument.paragraphs"
            )
        if min_ordinal is None or ordinal < min_ordinal:
            min_ordinal = ordinal
    if min_ordinal is None:
        raise ReconciliationPlanningError(
            "candidate has no evidence; cannot compute paragraph ordinal"
        )
    return min_ordinal


# ---------------------------------------------------------------------------
# Input snapshot
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ReconciliationInputSnapshot:
    """The validated typed snapshot A4B receives.

    The caller must have already exact-resolved and validated this snapshot.
    A4B performs deterministic lineage/coherence validation on it.
    """

    source_document: SourceDocument
    source_document_ref: ArtifactRef
    chunk_manifest: ChunkManifest
    source_chunks: tuple[SourceChunk, ...]
    source_chunk_refs: tuple[ArtifactRef, ...]
    candidate_extractions: tuple[CandidateExtraction, ...]
    candidate_extraction_refs: tuple[ArtifactRef, ...]
    a3_validation_report_refs: tuple[ArtifactRef, ...]


# ---------------------------------------------------------------------------
# Pair plan (in-memory only, no persisted schema)
# ---------------------------------------------------------------------------


# Pair states
PAIR_STATE_AUTO_SAME = "auto_same"
PAIR_STATE_MUST_NOT_MERGE = "must_not_merge"
PAIR_STATE_NEEDS_SEMANTIC_DECISION = "needs_semantic_decision"

# Pair signals
SIGNAL_EXACT_IDENTITY_KEY = "exact_identity_key"
SIGNAL_IDENTITY_TOKEN_OVERLAP = "identity_token_overlap"
SIGNAL_ADJACENT_CHUNK = "adjacent_chunk"
SIGNAL_HARD_MUST_NOT_MERGE = "hard_must_not_merge"


@dataclass(frozen=True, slots=True)
class ReconciliationPairPlan:
    """One explicit blocked pair in the ambiguity plan.

    In-memory only; not a persisted artifact.
    """

    left_candidate_ref: str
    right_candidate_ref: str
    state: str
    signals: tuple[str, ...]
    shared_identity_keys: tuple[str, ...]
    shared_tokens: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.left_candidate_ref >= self.right_candidate_ref:
            raise ReconciliationPlanningError(
                f"pair must be in canonical order (left < right): "
                f"{self.left_candidate_ref!r} vs {self.right_candidate_ref!r}"
            )
        if self.state not in (
            PAIR_STATE_AUTO_SAME,
            PAIR_STATE_MUST_NOT_MERGE,
            PAIR_STATE_NEEDS_SEMANTIC_DECISION,
        ):
            raise ReconciliationPlanningError(
                f"invalid pair state: {self.state!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "left_candidate_ref": self.left_candidate_ref,
            "right_candidate_ref": self.right_candidate_ref,
            "state": self.state,
            "signals": list(self.signals),
            "shared_identity_keys": list(self.shared_identity_keys),
            "shared_tokens": list(self.shared_tokens),
        }


# ---------------------------------------------------------------------------
# Planning result
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ReconciliationPlanningResult:
    """The complete deterministic planning output.

    In-memory only; A4B does not persist.
    """

    candidate_index: CandidateEntityIndex
    pair_plans: tuple[ReconciliationPairPlan, ...]
    decisions: tuple[ReconciliationDecision, ...]
    normalization_policy_id: str
    blocking_policy_id: str
    canonicalization_policy_id: str
    plan_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_index": self.candidate_index.to_dict(),
            "pair_plans": [p.to_dict() for p in self.pair_plans],
            "decisions": [d.to_dict() for d in self.decisions],
            "normalization_policy_id": self.normalization_policy_id,
            "blocking_policy_id": self.blocking_policy_id,
            "canonicalization_policy_id": self.canonicalization_policy_id,
            "plan_hash": self.plan_hash,
        }


# ---------------------------------------------------------------------------
# Deterministic decision ID
# ---------------------------------------------------------------------------


def compute_deterministic_decision_id(
    left_ref: str,
    right_ref: str,
    decision: str,
    method: str,
    reason_code: str,
    reason_zh: str,
) -> str:
    """Compute the deterministic decision_id from canonical decision material.

    This is the single authority for A4B deterministic decision ids, reused by
    A4D CURRENT verification to require ``persisted decision_id ==
    deterministically recomputed decision_id``.
    """
    material = {
        "left_candidate_ref": left_ref,
        "right_candidate_ref": right_ref,
        "decision": decision,
        "method": method,
        "reason_code": reason_code,
        "reason_zh": reason_zh,
    }
    return f"dec_{content_hash(material)[:20]}"


# ---------------------------------------------------------------------------
# Snapshot coherence validation
# ---------------------------------------------------------------------------


def _validate_snapshot(snapshot: ReconciliationInputSnapshot) -> None:
    """Fail-closed snapshot coherence validation.

    Verifies exact positional alignment and identity consistency using the
    production artifact-id authorities. Does NOT re-run A3 semantic validation.
    """
    doc = snapshot.source_document
    manifest = snapshot.chunk_manifest
    chunks = snapshot.source_chunks
    chunk_refs = snapshot.source_chunk_refs
    extractions = snapshot.candidate_extractions
    extraction_refs = snapshot.candidate_extraction_refs
    report_refs = snapshot.a3_validation_report_refs
    profile_id = manifest.profile.profile_id

    # Source project/document identity
    if manifest.project_id != doc.project_id:
        raise ReconciliationPlanningError(
            f"manifest project_id {manifest.project_id!r} != "
            f"source_document project_id {doc.project_id!r}"
        )
    if manifest.document_id != doc.document_id:
        raise ReconciliationPlanningError(
            f"manifest document_id {manifest.document_id!r} != "
            f"source_document document_id {doc.document_id!r}"
        )

    # Manifest pins exact source_document_ref
    if manifest.source_document_ref != snapshot.source_document_ref:
        raise ReconciliationPlanningError(
            "manifest.source_document_ref != snapshot.source_document_ref"
        )

    # Manifest chunk count and refs
    if manifest.chunk_count != len(manifest.chunk_refs):
        raise ReconciliationPlanningError(
            f"manifest.chunk_count {manifest.chunk_count} != "
            f"len(manifest.chunk_refs) {len(manifest.chunk_refs)}"
        )

    # Supplied chunk refs must match manifest chunk_refs exactly (order + content)
    if tuple(chunk_refs) != tuple(manifest.chunk_refs):
        raise ReconciliationPlanningError(
            "source_chunk_refs do not match manifest.chunk_refs exactly"
        )

    # Counts must align
    if len(chunks) != len(manifest.chunk_refs):
        raise ReconciliationPlanningError(
            f"len(source_chunks) {len(chunks)} != "
            f"len(manifest.chunk_refs) {len(manifest.chunk_refs)}"
        )
    if len(extractions) != len(manifest.chunk_refs):
        raise ReconciliationPlanningError(
            f"len(candidate_extractions) {len(extractions)} != "
            f"len(manifest.chunk_refs) {len(manifest.chunk_refs)}"
        )
    if len(extraction_refs) != len(extractions):
        raise ReconciliationPlanningError(
            f"len(candidate_extraction_refs) {len(extraction_refs)} != "
            f"len(candidate_extractions) {len(extractions)}"
        )
    if len(report_refs) != len(extractions):
        raise ReconciliationPlanningError(
            f"len(a3_validation_report_refs) {len(report_refs)} != "
            f"len(candidate_extractions) {len(extractions)}"
        )

    # Each SourceChunk: positional identity via production artifact-id authority
    for i, (chunk, chunk_ref) in enumerate(zip(chunks, chunk_refs)):
        if chunk_ref.artifact_type != "source_chunk":
            raise ReconciliationPlanningError(
                f"source_chunk_refs[{i}].artifact_type "
                f"{chunk_ref.artifact_type!r} != 'source_chunk'"
            )
        expected_chunk_artifact_id = source_chunk_artifact_id(
            manifest.project_id,
            manifest.document_id,
            profile_id,
            chunk.chunk_id,
        )
        if chunk_ref.artifact_id != expected_chunk_artifact_id:
            raise ReconciliationPlanningError(
                f"source_chunk_refs[{i}].artifact_id "
                f"{chunk_ref.artifact_id!r} != expected "
                f"{expected_chunk_artifact_id!r} "
                f"(chunk_id={chunk.chunk_id!r})"
            )
        if chunk.project_id != manifest.project_id:
            raise ReconciliationPlanningError(
                f"source_chunks[{i}].project_id {chunk.project_id!r} != "
                f"manifest.project_id {manifest.project_id!r}"
            )
        if chunk.document_id != manifest.document_id:
            raise ReconciliationPlanningError(
                f"source_chunks[{i}].document_id {chunk.document_id!r} != "
                f"manifest.document_id {manifest.document_id!r}"
            )
        if chunk.source_document_ref != snapshot.source_document_ref:
            raise ReconciliationPlanningError(
                f"source_chunks[{i}].source_document_ref does not match "
                "snapshot.source_document_ref"
            )

    # Each CandidateExtraction: positional identity + profile consistency
    extraction_profile_id: str | None = None
    extraction_profile_hash: str | None = None
    for i, (ext, ext_ref) in enumerate(zip(extractions, extraction_refs)):
        if ext.project_id != doc.project_id:
            raise ReconciliationPlanningError(
                f"candidate_extractions[{i}].project_id {ext.project_id!r} "
                f"!= source_document project_id {doc.project_id!r}"
            )
        if ext.document_id != doc.document_id:
            raise ReconciliationPlanningError(
                f"candidate_extractions[{i}].document_id {ext.document_id!r} "
                f"!= source_document document_id {doc.document_id!r}"
            )
        # chunk_profile_id must match the manifest's chunk-planning profile
        if ext.chunk_profile_id != profile_id:
            raise ReconciliationPlanningError(
                f"candidate_extractions[{i}].chunk_profile_id "
                f"{ext.chunk_profile_id!r} != manifest.profile.profile_id "
                f"{profile_id!r}"
            )
        # Positional chunk_id alignment
        if ext.chunk_id != chunks[i].chunk_id:
            raise ReconciliationPlanningError(
                f"candidate_extractions[{i}].chunk_id {ext.chunk_id!r} != "
                f"source_chunks[{i}].chunk_id {chunks[i].chunk_id!r}"
            )
        # Source refs
        if ext.source_document_ref != snapshot.source_document_ref:
            raise ReconciliationPlanningError(
                f"candidate_extractions[{i}].source_document_ref does not "
                "match snapshot.source_document_ref"
            )
        if ext.source_chunk_ref != chunk_refs[i]:
            raise ReconciliationPlanningError(
                f"candidate_extractions[{i}].source_chunk_ref does not "
                f"match source_chunk_refs[{i}]"
            )
        # CandidateExtraction ArtifactRef identity via production authority
        if ext_ref.artifact_type != "candidate_extraction":
            raise ReconciliationPlanningError(
                f"candidate_extraction_refs[{i}].artifact_type "
                f"{ext_ref.artifact_type!r} != 'candidate_extraction'"
            )
        expected_ext_artifact_id = candidate_extraction_artifact_id(
            ext.project_id,
            ext.document_id,
            ext.chunk_profile_id,
            ext.chunk_id,
            ext.extraction_profile_id,
        )
        if ext_ref.artifact_id != expected_ext_artifact_id:
            raise ReconciliationPlanningError(
                f"candidate_extraction_refs[{i}].artifact_id "
                f"{ext_ref.artifact_id!r} != expected "
                f"{expected_ext_artifact_id!r}"
            )
        # Profile consistency across all extractions
        if extraction_profile_id is None:
            extraction_profile_id = ext.extraction_profile_id
            extraction_profile_hash = ext.extraction_profile_hash
        elif ext.extraction_profile_id != extraction_profile_id:
            raise ReconciliationPlanningError(
                f"candidate_extractions[{i}].extraction_profile_id "
                f"{ext.extraction_profile_id!r} != first extraction's "
                f"{extraction_profile_id!r}"
            )
        elif ext.extraction_profile_hash != extraction_profile_hash:
            raise ReconciliationPlanningError(
                f"candidate_extractions[{i}].extraction_profile_hash != "
                f"first extraction's hash"
            )


# ---------------------------------------------------------------------------
# CandidateEntityIndex construction
# ---------------------------------------------------------------------------


def _build_candidate_index(
    snapshot: ReconciliationInputSnapshot,
    paragraph_position: dict[str, int],
) -> CandidateEntityIndex:
    """Build the CandidateEntityIndex from the validated snapshot.

    Coverage universe: CharacterCandidate + LocationCandidate +
    UnresolvedMentionCandidate (NOT fact/event/relationship).
    """
    entries: list[CandidateEntityIndexEntry] = []

    for chunk_ordinal_0, (ext, ext_ref) in enumerate(
        zip(snapshot.candidate_extractions, snapshot.candidate_extraction_refs)
    ):
        chunk_ordinal = chunk_ordinal_0 + 1  # 1-based
        chunk_id = ext.chunk_id

        # Characters
        for char in ext.candidates.characters:
            global_ref = f"{chunk_id}:{char.candidate_id}"
            paragraph_ordinal = _earliest_evidence_paragraph_ordinal(
                char.evidence, paragraph_position
            )
            suffix = _local_candidate_suffix(char.candidate_id)
            order_key = compute_source_order_key(
                chunk_ordinal=chunk_ordinal,
                paragraph_ordinal=paragraph_ordinal,
                category="character",
                candidate_suffix=suffix,
                global_ref=global_ref,
            )
            entries.append(
                CandidateEntityIndexEntry(
                    candidate_ref=global_ref,
                    candidate_kind="character",
                    candidate_extraction_ref=ext_ref,
                    source_order_key=order_key,
                    display_name_original=char.display_name_original,
                    aliases_original=char.aliases_original,
                    descriptors_zh=char.descriptors_zh,
                    evidence_refs=char.evidence,
                    possible_candidate_refs=(),
                )
            )

        # Locations
        for loc in ext.candidates.locations:
            global_ref = f"{chunk_id}:{loc.candidate_id}"
            paragraph_ordinal = _earliest_evidence_paragraph_ordinal(
                loc.evidence, paragraph_position
            )
            suffix = _local_candidate_suffix(loc.candidate_id)
            order_key = compute_source_order_key(
                chunk_ordinal=chunk_ordinal,
                paragraph_ordinal=paragraph_ordinal,
                category="location",
                candidate_suffix=suffix,
                global_ref=global_ref,
            )
            entries.append(
                CandidateEntityIndexEntry(
                    candidate_ref=global_ref,
                    candidate_kind="location",
                    candidate_extraction_ref=ext_ref,
                    source_order_key=order_key,
                    display_name_original=loc.display_name_original,
                    aliases_original=loc.aliases_original,
                    descriptors_zh=loc.descriptors_zh,
                    evidence_refs=loc.evidence,
                    possible_candidate_refs=(),
                )
            )

        # Unresolved mentions
        for unres in ext.candidates.unresolved_mentions:
            global_ref = f"{chunk_id}:{unres.candidate_id}"
            paragraph_ordinal = _earliest_evidence_paragraph_ordinal(
                unres.evidence, paragraph_position
            )
            suffix = _local_candidate_suffix(unres.candidate_id)
            order_key = compute_source_order_key(
                chunk_ordinal=chunk_ordinal,
                paragraph_ordinal=paragraph_ordinal,
                category="unresolved",
                candidate_suffix=suffix,
                global_ref=global_ref,
            )
            # Map mention_kind to candidate_kind
            kind_map = {
                "person": "unresolved_person",
                "location": "unresolved_location",
                "unknown": "unresolved_unknown",
                "other": "unresolved_other",
            }
            candidate_kind = kind_map[unres.mention_kind]
            # Globalize possible_candidate_refs (local → global, same chunk)
            possible_refs = tuple(
                f"{chunk_id}:{ref}" for ref in unres.possible_candidate_refs
            )
            entries.append(
                CandidateEntityIndexEntry(
                    candidate_ref=global_ref,
                    candidate_kind=candidate_kind,
                    candidate_extraction_ref=ext_ref,
                    source_order_key=order_key,
                    display_name_original=unres.mention_original,
                    aliases_original=(),
                    descriptors_zh=(),
                    evidence_refs=unres.evidence,
                    possible_candidate_refs=possible_refs,
                )
            )

    # Sort by source_order_key
    entries.sort(key=lambda e: e.source_order_key)

    return CandidateEntityIndex(
        schema_version=CANDIDATE_ENTITY_INDEX_SCHEMA_VERSION,
        entries=tuple(entries),
    )


# ---------------------------------------------------------------------------
# Coverage audit
# ---------------------------------------------------------------------------


def _audit_coverage(
    candidate_index: CandidateEntityIndex,
    pair_plans: tuple[ReconciliationPairPlan, ...],
) -> None:
    """Fail-closed coverage audit.

    Verifies:
      * no duplicate candidate_ref
      * pair plans only reference char/loc entries
      * every explicit pair unique, left < right, same entity type
    """
    # No duplicate candidate refs
    refs = [entry.candidate_ref for entry in candidate_index.entries]
    if len(refs) != len(set(refs)):
        raise ReconciliationPlanningError(
            "candidate index contains duplicate candidate_ref"
        )

    # Build ref → entry lookup
    ref_to_entry = {e.candidate_ref: e for e in candidate_index.entries}

    # Pair audit
    seen_pairs: set[tuple[str, str]] = set()
    for plan in pair_plans:
        left_entry = ref_to_entry.get(plan.left_candidate_ref)
        right_entry = ref_to_entry.get(plan.right_candidate_ref)
        if left_entry is None:
            raise ReconciliationPlanningError(
                f"pair left_candidate_ref {plan.left_candidate_ref!r} not in "
                "candidate index"
            )
        if right_entry is None:
            raise ReconciliationPlanningError(
                f"pair right_candidate_ref {plan.right_candidate_ref!r} not "
                "in candidate index"
            )
        # Must be char/loc only
        if left_entry.candidate_kind not in ("character", "location"):
            raise ReconciliationPlanningError(
                f"pair left ref {plan.left_candidate_ref!r} has kind "
                f"{left_entry.candidate_kind!r}, not character/location"
            )
        if right_entry.candidate_kind not in ("character", "location"):
            raise ReconciliationPlanningError(
                f"pair right ref {plan.right_candidate_ref!r} has kind "
                f"{right_entry.candidate_kind!r}, not character/location"
            )
        # Same entity type
        if left_entry.candidate_kind != right_entry.candidate_kind:
            raise ReconciliationPlanningError(
                f"pair ({plan.left_candidate_ref}, {plan.right_candidate_ref}) "
                f"has mismatched entity types: {left_entry.candidate_kind!r} "
                f"vs {right_entry.candidate_kind!r}"
            )
        # Uniqueness
        pair_key = (plan.left_candidate_ref, plan.right_candidate_ref)
        if pair_key in seen_pairs:
            raise ReconciliationPlanningError(
                f"duplicate pair: {pair_key}"
            )
        seen_pairs.add(pair_key)

    # Verify unresolved possible_candidate_refs resolve to concrete char/loc
    for entry in candidate_index.entries:
        if entry.candidate_kind.startswith("unresolved_"):
            chunk_id = entry.candidate_ref.split(":", 1)[0]
            for possible_ref in entry.possible_candidate_refs:
                target = ref_to_entry.get(possible_ref)
                if target is None:
                    raise ReconciliationPlanningError(
                        f"unresolved candidate {entry.candidate_ref!r} has "
                        f"possible_candidate_ref {possible_ref!r} that does "
                        "not resolve to an indexed candidate"
                    )
                if target.candidate_kind not in ("character", "location"):
                    raise ReconciliationPlanningError(
                        f"unresolved candidate {entry.candidate_ref!r} has "
                        f"possible_candidate_ref {possible_ref!r} that does "
                        "not resolve to a character/location"
                    )
                # Must be in same chunk
                target_chunk = possible_ref.split(":", 1)[0]
                if target_chunk != chunk_id:
                    raise ReconciliationPlanningError(
                        f"unresolved candidate {entry.candidate_ref!r} has "
                        f"possible_candidate_ref {possible_ref!r} in a "
                        "different chunk"
                    )


# ---------------------------------------------------------------------------
# Indexed blocking pair generation (no whole-document N²)
# ---------------------------------------------------------------------------


def _generate_blocked_pairs(
    candidate_index: CandidateEntityIndex,
    identity_keys_map: dict[str, tuple[str, ...]],
    tokens_map: dict[str, tuple[str, ...]],
    chunk_ordinal_map: dict[str, int],
    must_not_merge: frozenset[tuple[str, str]],
) -> dict[tuple[str, str], dict[str, Any]]:
    """Generate blocked pairs using indexed/bucketed generation.

    Returns a dict mapping (left_ref, right_ref) → signal info.
    """
    # Only character and location entries participate in blocking
    merge_entries = [
        e for e in candidate_index.entries
        if e.candidate_kind in ("character", "location")
    ]

    pairs: dict[tuple[str, str], dict[str, Any]] = {}

    def _add_pair(left: str, right: str, signal: str, shared_keys: tuple[str, ...] = (), shared_tokens: tuple[str, ...] = ()) -> None:
        if left == right:
            return
        if left > right:
            left, right = right, left
        key = (left, right)
        if key not in pairs:
            pairs[key] = {
                "signals": set(),
                "shared_identity_keys": set(),
                "shared_tokens": set(),
            }
        pairs[key]["signals"].add(signal)
        pairs[key]["shared_identity_keys"].update(shared_keys)
        pairs[key]["shared_tokens"].update(shared_tokens)

    # --- Exact identity key buckets ---
    # (entity_type, normalized_identity_key) → ordered candidate refs
    exact_key_buckets: dict[tuple[str, str], list[str]] = {}
    for entry in merge_entries:
        entity_type = entry.candidate_kind
        keys = identity_keys_map.get(entry.candidate_ref, ())
        for k in keys:
            bucket_key = (entity_type, k)
            if bucket_key not in exact_key_buckets:
                exact_key_buckets[bucket_key] = []
            exact_key_buckets[bucket_key].append(entry.candidate_ref)

    for (entity_type, key), refs in exact_key_buckets.items():
        if len(refs) < 2:
            continue
        for i in range(len(refs)):
            for j in range(i + 1, len(refs)):
                _add_pair(
                    refs[i], refs[j],
                    SIGNAL_EXACT_IDENTITY_KEY,
                    shared_keys=(key,),
                )

    # --- Token buckets ---
    # (entity_type, token) → ordered candidate refs
    token_buckets: dict[tuple[str, str], list[str]] = {}
    for entry in merge_entries:
        entity_type = entry.candidate_kind
        tokens = tokens_map.get(entry.candidate_ref, ())
        for tok in tokens:
            bucket_key = (entity_type, tok)
            if bucket_key not in token_buckets:
                token_buckets[bucket_key] = []
            if entry.candidate_ref not in token_buckets[bucket_key]:
                token_buckets[bucket_key].append(entry.candidate_ref)

    for (entity_type, token), refs in token_buckets.items():
        if len(refs) < 2:
            continue
        for i in range(len(refs)):
            for j in range(i + 1, len(refs)):
                _add_pair(
                    refs[i], refs[j],
                    SIGNAL_IDENTITY_TOKEN_OVERLAP,
                    shared_tokens=(token,),
                )

    # --- Same-chunk adjacency (a4-blocking-v2 generative recall safety net) ---
    # (entity_type, chunk_ordinal) → ordered candidate refs
    chunk_buckets: dict[tuple[str, int], list[str]] = {}
    for entry in merge_entries:
        entity_type = entry.candidate_kind
        ordinal = chunk_ordinal_map.get(entry.candidate_ref)
        if ordinal is None:
            continue
        bucket_key = (entity_type, ordinal)
        if bucket_key not in chunk_buckets:
            chunk_buckets[bucket_key] = []
        chunk_buckets[bucket_key].append(entry.candidate_ref)

    # Same chunk: generate combinations inside each bucket (still generative).
    # Two same-type candidates in the same chunk get a pair on source proximity
    # alone, even with no exact identity key and no effective token overlap.
    # This is a deliberate conservative recall safety net; v2 does NOT adopt
    # Policy C (which would delete these pairs).
    for (entity_type, ordinal), refs in chunk_buckets.items():
        if len(refs) < 2:
            continue
        for i in range(len(refs)):
            for j in range(i + 1, len(refs)):
                _add_pair(
                    refs[i], refs[j],
                    SIGNAL_ADJACENT_CHUNK,
                )

    # --- Hard must-not-merge constraints (already canonicalized at boundary) ---
    for (left, right) in must_not_merge:
        _add_pair(left, right, SIGNAL_HARD_MUST_NOT_MERGE)

    # --- Cross-chunk adjacency is supplemental, NOT generative (a4-blocking-v2) ---
    # Two candidates in chunks with ordinal distance exactly 1 do NOT create a
    # pair on adjacency alone (cross-chunk adjacency-only → implicit
    # not_compared). But if a pair already exists because of an exact identity
    # key, an effective identity-token overlap, a same-chunk adjacency
    # (distance 0), or a hard constraint, we additionally attach the
    # adjacent_chunk deterministic signal. We iterate only over already-existing
    # pairs (bounded by the explicit plan) — never a whole-document N² scan and
    # never a cross-chunk candidate-bucket Cartesian product.
    for (left, right) in pairs:
        left_ordinal = chunk_ordinal_map.get(left)
        right_ordinal = chunk_ordinal_map.get(right)
        if left_ordinal is None or right_ordinal is None:
            continue
        if abs(left_ordinal - right_ordinal) == 1:
            pairs[(left, right)]["signals"].add(SIGNAL_ADJACENT_CHUNK)

    return pairs


# ---------------------------------------------------------------------------
# Pair state assignment
# ---------------------------------------------------------------------------


def _assign_pair_state(
    pair_info: dict[str, Any],
    left_ref: str,
    right_ref: str,
    identity_keys_map: dict[str, tuple[str, ...]],
    must_not_merge: frozenset[tuple[str, str]],
) -> str:
    """Assign the pair state using frozen precedence:
    must_not_merge > auto_same > needs_semantic_decision
    """
    signals = pair_info["signals"]

    # 1. must_not_merge
    if (left_ref, right_ref) in must_not_merge:
        return PAIR_STATE_MUST_NOT_MERGE

    # 2. auto_same: shared exact STRONG identity key
    if SIGNAL_EXACT_IDENTITY_KEY in signals:
        shared_keys = pair_info["shared_identity_keys"]
        for key in shared_keys:
            if is_strong_identity_key(key):
                return PAIR_STATE_AUTO_SAME

    # 3. needs_semantic_decision
    return PAIR_STATE_NEEDS_SEMANTIC_DECISION


# ---------------------------------------------------------------------------
# Plan hash
# ---------------------------------------------------------------------------


def _compute_plan_hash(
    candidate_index: CandidateEntityIndex,
    pair_plans: tuple[ReconciliationPairPlan, ...],
    normalization_policy_id: str,
    blocking_policy_id: str,
    canonicalization_policy_id: str,
) -> str:
    """Compute the deterministic plan_hash."""
    # Pair plans in canonical order (left, right)
    sorted_plans = sorted(
        pair_plans,
        key=lambda p: (p.left_candidate_ref, p.right_candidate_ref),
    )
    material = {
        "normalization_policy_id": normalization_policy_id,
        "blocking_policy_id": blocking_policy_id,
        "canonicalization_policy_id": canonicalization_policy_id,
        "candidate_index": candidate_index.to_dict(),
        "pair_plans": [p.to_dict() for p in sorted_plans],
    }
    return content_hash(material)


# ---------------------------------------------------------------------------
# must-not-merge canonicalization + shared index -> planning pipeline
# ---------------------------------------------------------------------------


def _canonicalize_must_not_merge(
    must_not_merge: frozenset[tuple[str, str]],
) -> frozenset[tuple[str, str]]:
    """Canonicalize an explicit hard-constraint set (fail closed on bad pairs)."""
    hard_constraints: frozenset[tuple[str, str]] = frozenset()
    for pair in must_not_merge:
        if len(pair) != 2:
            raise ReconciliationPlanningError(
                f"must_not_merge pair must have exactly 2 elements: {pair!r}"
            )
        a, b = pair
        if a == b:
            raise ReconciliationPlanningError(
                f"must_not_merge pair must reference distinct candidates: {pair!r}"
            )
        hard_constraints = hard_constraints | frozenset({(min(a, b), max(a, b))})
    return hard_constraints


def _plan_from_candidate_index(
    candidate_index: CandidateEntityIndex,
    chunk_ordinal_map: dict[str, int],
    hard_constraints: frozenset[tuple[str, str]],
) -> ReconciliationPlanningResult:
    """Run the shared A4B index -> planning pipeline (identity keys, blocking,
    pair states, deterministic decisions, coverage audit, plan hash).

    This is the single source of planning logic reused by both
    :func:`plan_reconciliation` (fresh A4B) and :func:`plan_candidate_index_v1`
    (A4D replanning from a persisted index).
    """
    # Identity keys + blocking tokens
    identity_keys_map: dict[str, tuple[str, ...]] = {}
    tokens_map: dict[str, tuple[str, ...]] = {}
    for entry in candidate_index.entries:
        if entry.candidate_kind in ("character", "location"):
            keys = extract_identity_keys(
                entry.display_name_original, entry.aliases_original
            )
            identity_keys_map[entry.candidate_ref] = keys
            # a4-blocking-v2: raw identity tokens → blocking-only function-word
            # filter → effective blocking tokens. Only this (filtered) token set
            # feeds the IDENTITY_TOKEN_OVERLAP buckets and the recorded
            # shared_tokens; strong-key classification still uses the raw
            # extract_blocking_tokens() semantics.
            tokens_map[entry.candidate_ref] = effective_blocking_tokens(
                extract_blocking_tokens(keys)
            )
        else:
            identity_keys_map[entry.candidate_ref] = ()
            tokens_map[entry.candidate_ref] = ()

    # Blocked pairs (indexed, not N^2)
    pair_info = _generate_blocked_pairs(
        candidate_index,
        identity_keys_map,
        tokens_map,
        chunk_ordinal_map,
        hard_constraints,
    )

    # Pair states + pair plans
    pair_plans: list[ReconciliationPairPlan] = []
    for (left, right), info in pair_info.items():
        state = _assign_pair_state(
            info, left, right, identity_keys_map, hard_constraints
        )
        signals = tuple(sorted(info["signals"]))
        shared_keys = tuple(sorted(info["shared_identity_keys"]))
        shared_tokens = tuple(sorted(info["shared_tokens"]))
        pair_plans.append(
            ReconciliationPairPlan(
                left_candidate_ref=left,
                right_candidate_ref=right,
                state=state,
                signals=signals,
                shared_identity_keys=shared_keys,
                shared_tokens=shared_tokens,
            )
        )
    pair_plans.sort(key=lambda p: (p.left_candidate_ref, p.right_candidate_ref))
    pair_plans_tuple = tuple(pair_plans)

    # Deterministic decisions for auto_same / must_not_merge
    decisions: list[ReconciliationDecision] = []
    for plan in pair_plans_tuple:
        if plan.state == PAIR_STATE_AUTO_SAME:
            decision_id = compute_deterministic_decision_id(
                plan.left_candidate_ref,
                plan.right_candidate_ref,
                "same_entity",
                "deterministic",
                "same_strong_exact_identity_key",
                "确定性自动合并：共享强身份键",
            )
            decisions.append(
                ReconciliationDecision(
                    decision_id=decision_id,
                    left_candidate_ref=plan.left_candidate_ref,
                    right_candidate_ref=plan.right_candidate_ref,
                    decision="same_entity",
                    method="deterministic",
                    reason_code="same_strong_exact_identity_key",
                    reason_zh="确定性自动合并：共享强身份键",
                    evidence_refs=(),
                    prompt_id=None,
                    prompt_version=None,
                    generation_provenance=None,
                )
            )
        elif plan.state == PAIR_STATE_MUST_NOT_MERGE:
            decision_id = compute_deterministic_decision_id(
                plan.left_candidate_ref,
                plan.right_candidate_ref,
                "different_entity",
                "deterministic",
                "hard_must_not_merge",
                "确定性硬约束：不可合并",
            )
            decisions.append(
                ReconciliationDecision(
                    decision_id=decision_id,
                    left_candidate_ref=plan.left_candidate_ref,
                    right_candidate_ref=plan.right_candidate_ref,
                    decision="different_entity",
                    method="deterministic",
                    reason_code="hard_must_not_merge",
                    reason_zh="确定性硬约束：不可合并",
                    evidence_refs=(),
                    prompt_id=None,
                    prompt_version=None,
                    generation_provenance=None,
                )
            )

    # Coverage audit (fail closed)
    _audit_coverage(candidate_index, pair_plans_tuple)

    # Plan hash
    plan_hash = _compute_plan_hash(
        candidate_index,
        pair_plans_tuple,
        NAME_NORMALIZATION_POLICY_ID,
        BLOCKING_POLICY_ID,
        CANONICALIZATION_POLICY_ID,
    )

    return ReconciliationPlanningResult(
        candidate_index=candidate_index,
        pair_plans=pair_plans_tuple,
        decisions=tuple(decisions),
        normalization_policy_id=NAME_NORMALIZATION_POLICY_ID,
        blocking_policy_id=BLOCKING_POLICY_ID,
        canonicalization_policy_id=CANONICALIZATION_POLICY_ID,
        plan_hash=plan_hash,
    )


def plan_candidate_index_v1(
    candidate_index: CandidateEntityIndex,
    *,
    must_not_merge: frozenset[tuple[str, str]] | None = None,
) -> ReconciliationPlanningResult:
    """Rebuild A4B v1 planning from a persisted :class:`CandidateEntityIndex`.

    This is the single-source planning authority reused by A4D CURRENT
    verification / publication revalidation so the replanned ``plan_hash`` and
    pair plans are byte-identical to a fresh A4B run over the same index.

    The candidate index must carry the frozen ``source_order_key`` shape so
    ``chunk_ordinal`` can be recovered deterministically for adjacency blocking
    (enforced by :func:`validate_candidate_index_source_order`). Production v1
    derives an empty ``must_not_merge`` set unless one is explicitly supplied.
    """
    chunk_ordinal_map = validate_candidate_index_source_order(candidate_index)
    if must_not_merge is not None:
        hard_constraints = _canonicalize_must_not_merge(must_not_merge)
    else:
        hard_constraints = derive_must_not_merge_constraints(candidate_index)
    return _plan_from_candidate_index(
        candidate_index, chunk_ordinal_map, hard_constraints
    )


# ---------------------------------------------------------------------------
# Main planning entry point
# ---------------------------------------------------------------------------


def plan_reconciliation(
    snapshot: ReconciliationInputSnapshot,
    *,
    must_not_merge: frozenset[tuple[str, str]] | None = None,
) -> ReconciliationPlanningResult:
    """Execute the full A4B deterministic planning pipeline.

    Parameters:
        snapshot: validated typed input snapshot
        must_not_merge: optional explicit hard constraint set (for testing /
            future versioned rules). Production v1 callers pass None (→ empty).

    Returns:
        ReconciliationPlanningResult with candidate index, pair plans,
        decisions, and deterministic plan_hash.
    """
    # Step 1: Validate snapshot coherence (fail closed)
    _validate_snapshot(snapshot)

    # Step 2: Build paragraph position index (1-based ordinal)
    paragraph_position: dict[str, int] = {
        para.paragraph_id: idx + 1
        for idx, para in enumerate(snapshot.source_document.paragraphs)
    }

    # Step 3: Build candidate index
    candidate_index = _build_candidate_index(snapshot, paragraph_position)

    # Delegate the index -> planning pipeline to the single-source authority so
    # fresh A4B runs and A4D CURRENT verification / publication revalidation
    # share byte-identical pair plans, deterministic decisions, and plan_hash.
    return plan_candidate_index_v1(candidate_index, must_not_merge=must_not_merge)
