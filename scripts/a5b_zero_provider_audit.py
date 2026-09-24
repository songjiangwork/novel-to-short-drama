"""v1.2 A5B — zero-provider input binding / indexing / corpus-shape audit.

This is the Phase A A5B delivery slice (issue #51). It resolves the exact
current-eligible A4 CURRENT EntityMap from a *pre-existing* run tree, loads the
exact pinned A3 input, binds every A3 local entity ref to its A4-bound A5
entity id, builds the source-ordered ``ConsolidationCandidateIndex``, and
reports a **corpus-shape audit** designed to feed the later A5B blocking-v1
refinement.

It is **zero-provider** (no LLM / provider call) and **read-only** (no artifact,
pointer, CURRENT, validation-report, or source-run write). It enumerates the
Alice pair universe purely for diagnostics (counts, source-distance buckets,
overlap signals, exact-duplicate groups, bounded representative samples) -- it
does NOT authorize production O(N^2) pair generation and does NOT implement
blocking, semantic decisions, canonical ids, A5 persistence/CURRENT, or a CLI
(those are later A5B-A5H slices).

The audit first prints the exact corpus snapshot identity (EntityMap
artifact-type/id/revision/content-hash, the exact SourceDocument / ChunkManifest
refs, and every ordered CandidateExtraction ref) so the later blocking
refinement can bind to a precise corpus snapshot.

Usage:
    python scripts/a5b_zero_provider_audit.py \
        [--runs-root PATH] [--project ID] [--document ID] [--profile ID]

Exit code 0 on a clean audit, 2 on any structural / binding / integrity failure.
"""

from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from short_drama.artifacts import ArtifactRef, FileArtifactStore
from short_drama.foundation import FilePointerStore
from short_drama.paths import REPO_ROOT
from short_drama.story import (
    ConsolidationCurrentMissingError,
    ConsolidationPlanningResult,
    StoryIntegrityError,
    build_consolidation_planning,
    load_consolidation_profile,
)

DEFAULT_RUNS_ROOT = REPO_ROOT / "runs" / "a4e_real_novel"
DEFAULT_PROJECT = "a3e-real-novel"
DEFAULT_DOCUMENT = "src_001"
DEFAULT_PROFILE = "entity-reconciliation-v2"
DEFAULT_CONSOLIDATION_PROFILE = REPO_ROOT / "profiles" / "consolidation_v1.yaml"

# Frozen Alice v1.2-A5B blocking-v1 acceptance gates (post-audit refinement).
# ``relationship`` is a DIRECTION-AWARE expectation: the actual explicit count
# must equal the independently-computed direction-aware endpoint bucket count
# AND be <= the unordered Phase-A ceiling (845). The rest are upper bounds;
# auto_same is an EXACT zero expectation (the Alice corpus produces no
# deterministic merges).
ALICE_GATE_FACT_MAX = 5300
ALICE_GATE_EVENT_MAX = 4300
ALICE_GATE_RELATIONSHIP_MAX = 845
ALICE_GATE_TOTAL_MAX = 10450
ALICE_GATE_AUTO_SAME_EXACT = 0

# Bound-reference field kinds (mirrors the A3 validation authority).
_PERSON_FIELDS = ("participants", "source_entity_ref", "target_entity_ref")
_LOCATION_FIELDS = ("locations",)

# Diagnostic overlap-signal label sets (frozen order, audit-only).
_FACT_SIGNALS = (
    "same_fact_type",
    "exact_normalized_statement",
    "subject_overlap",
    "object_overlap",
    "bound_entity_overlap",
    "evidence_paragraph_overlap",
)
_EVENT_SIGNALS = (
    "exact_normalized_summary",
    "participant_overlap",
    "location_overlap",
    "bound_entity_overlap",
    "evidence_paragraph_overlap",
    "temporal_mode_equal",
)
_REL_SIGNALS = (
    "exact_or_symmetric_endpoint_group",
    "direction_equal",
    "exact_normalized_relationship_type",
    "relationship_type_token_overlap",
    "evidence_paragraph_overlap",
)

# Bounded sample budget (deterministic; never random).
_MAX_SAMPLES_PER_BUCKET = 10
# Distribution tail threshold (exact integer buckets up to this, then a "+N" tail).
_DIST_TAIL_THRESHOLD = 10
# Endpoint-group / relationship-type frequency display cap.
_FREQ_TOP = 10


# ---------------------------------------------------------------------------
# Stores
# ---------------------------------------------------------------------------


def _stores(runs_root: str | Path, project_id: str):
    root = Path(runs_root).expanduser() / project_id / "story"
    artifact_store = FileArtifactStore(root / "artifacts")
    pointer_store = FilePointerStore(root / "pointers", artifact_store)
    return artifact_store, pointer_store


# ---------------------------------------------------------------------------
# Diagnostic normalization / tokenization (audit-only, NOT production policy)
# ---------------------------------------------------------------------------

_WS_RE = re.compile(r"\s+")
# Deterministic minimal diagnostic tokenizer: ASCII alnum runs + individual
# CJK ideographs (relationship/statement text in this corpus is zh).
_TOKEN_RE = re.compile(r"[0-9a-z]+|[\u4e00-\u9fff]")


def normalize_diagnostic_text(text: str) -> str:
    """Diagnostic-only exact-text normalization (NOT production blocking policy).

    Unicode NFKC -> casefold -> strip -> collapse all Unicode whitespace to a
    single ASCII space. No fuzzy threshold, edit distance, stopword list,
    semantic similarity, embedding, or LLM is applied anywhere in this audit.
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.casefold()
    text = text.strip()
    return _WS_RE.sub(" ", text)


def diagnostic_tokens(text: str) -> frozenset[str]:
    """Diagnostic-only deterministic minimal tokenizer (NOT production policy).

    Tokenizes the normalized text into ASCII alphanumeric runs and individual
    CJK ideographs, lowercased. Used solely for the relationship-type
    ``relationship_type_token_overlap`` diagnostic signal.
    """
    return frozenset(_TOKEN_RE.findall(normalize_diagnostic_text(text)))


# ---------------------------------------------------------------------------
# Small pure helpers
# ---------------------------------------------------------------------------


def _namespace(bound_id: str) -> str:
    for prefix in ("char_", "loc_", "unres_"):
        if bound_id.startswith(prefix):
            return prefix[:-1]
    return "other"


def _artifact_ref_str(ref: ArtifactRef) -> str:
    """Full ArtifactRef identity string (type/id/revision/hash)."""
    return (
        f"{ref.artifact_type} | {ref.artifact_id} | r{ref.revision} | "
        f"{ref.content_hash}"
    )


def _naive_pair_count(n: int) -> int:
    """Exact n-choose-2 pair count (n * (n - 1) // 2)."""
    return n * (n - 1) // 2


def _decision_set_size(decision_set) -> int:
    """Total number of A5A-domain deterministic decisions in the set.

    The Phase B deterministic decisions are A5A-domain ``Fact`` / ``Event`` /
    ``Relationship`` semantic decisions grouped in the A5A ``ConsolidationDecisionSet``
    (BLOCK 4); there is no separate ``decisions`` attribute.
    """
    return (
        len(decision_set.fact_decisions)
        + len(decision_set.event_decisions)
        + len(decision_set.relationship_decisions)
    )


def _relationship_direction_aware_expected(index) -> int:
    """Independently-computed DIRECTION-AWARE relationship endpoint bucket count.

    Uses the frozen production ``_endpoint_identity_key`` (section 9.1) directly --
    NOT the production pair planner's output -- so the acceptance gate is not a
    tautology. This is the exact relationship explicit-pair count the direction-aware
    blocking-v1 generator is expected to produce.
    """
    from short_drama.story.consolidation_planning import _endpoint_identity_key

    buckets: dict = defaultdict(int)
    for rel in index.relationships:
        key = _endpoint_identity_key(
            rel.source_entity_ref, rel.target_entity_ref, rel.direction
        )
        buckets[key] += 1
    return sum(_naive_pair_count(count) for count in buckets.values())


def _relationship_unordered_ceiling(index) -> int:
    """The unordered Phase-A relationship endpoint-group ceiling (n-choose-2 over the
    unordered frozenset of the two bound endpoints, direction-agnostic).

    This is the historical Phase-A relationship blocking count (845 for Alice) and is
    the upper bound the direction-aware count must satisfy.
    """
    buckets: dict = defaultdict(int)
    for rel in index.relationships:
        key = frozenset((rel.source_entity_ref, rel.target_entity_ref))
        buckets[key] += 1
    return sum(_naive_pair_count(count) for count in buckets.values())


def _source_distance_label(ordinal_a: int, ordinal_b: int) -> str:
    """Chunk-distance bucket from authoritative ChunkManifest chunk ordinals."""
    d = abs(ordinal_a - ordinal_b)
    if d == 0:
        return "same_chunk"
    if d == 1:
        return "distance_1"
    if d == 2:
        return "distance_2"
    return "distance_3_plus"


def _chunk_ordinal_by_chunk_id(snapshot) -> dict[str, int]:
    """chunk_id -> 1-based ordinal, from the authoritative ChunkManifest order.

    The source chunks are loaded in ``ChunkManifest.chunk_refs`` order, so the
    position in that order is the authoritative chunk ordinal (never inferred
    from lexical chunk-id ordering).
    """
    return {chunk.chunk_id: index + 1 for index, chunk in enumerate(snapshot.source_chunks)}


def _evidence_paragraphs(candidate) -> frozenset[str]:
    return frozenset(ref.paragraph_id for ref in candidate.evidence_refs)


def _fact_bound_refs(f) -> tuple[str, ...]:
    return f.subject_refs + f.object_refs


def _event_bound_refs(e) -> tuple[str, ...]:
    return e.participants + e.locations


def _relationship_bound_refs(r) -> tuple[str, ...]:
    return (r.source_entity_ref, r.target_entity_ref)


def _candidate_bound_refs(candidate) -> tuple[str, ...]:
    if hasattr(candidate, "subject_refs"):
        return _fact_bound_refs(candidate)
    if hasattr(candidate, "participants"):
        return _event_bound_refs(candidate)
    return _relationship_bound_refs(candidate)


def _has_unresolved_ref(candidate) -> bool:
    return any(ref.startswith("unres_") for ref in _candidate_bound_refs(candidate))


def _distribution_rows(values: list[int], tail_threshold: int = _DIST_TAIL_THRESHOLD):
    """Deterministic count-distribution rows: exact integer buckets 0..min(max,
    threshold) (including zero-count buckets) plus a '<threshold>+' tail."""
    if not values:
        return []
    counter = Counter(values)
    max_value = max(counter)
    top = min(max_value, tail_threshold)
    rows = [(value, counter[value]) for value in range(0, top + 1)]
    if max_value > tail_threshold:
        tail = sum(count for value, count in counter.items() if value > tail_threshold)
        rows.append((f"{tail_threshold}+", tail))
    return rows


def _format_distribution(counter: Counter) -> list[str]:
    rows = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
    return [f"    {key}: {count}" for key, count in rows] or ["    (none)"]


# ---------------------------------------------------------------------------
# Per-domain overlap signals (audit-only)
# ---------------------------------------------------------------------------


def _fact_signals(a, b) -> frozenset[str]:
    signals: set[str] = set()
    if a.fact_type == b.fact_type:
        signals.add("same_fact_type")
    if normalize_diagnostic_text(a.statement_zh) == normalize_diagnostic_text(b.statement_zh):
        signals.add("exact_normalized_statement")
    if set(a.subject_refs) & set(b.subject_refs):
        signals.add("subject_overlap")
    if set(a.object_refs) & set(b.object_refs):
        signals.add("object_overlap")
    if (set(a.subject_refs) | set(a.object_refs)) & (set(b.subject_refs) | set(b.object_refs)):
        signals.add("bound_entity_overlap")
    if _evidence_paragraphs(a) & _evidence_paragraphs(b):
        signals.add("evidence_paragraph_overlap")
    return frozenset(signals)


def _event_signals(a, b) -> frozenset[str]:
    signals: set[str] = set()
    if normalize_diagnostic_text(a.summary_zh) == normalize_diagnostic_text(b.summary_zh):
        signals.add("exact_normalized_summary")
    if set(a.participants) & set(b.participants):
        signals.add("participant_overlap")
    if set(a.locations) & set(b.locations):
        signals.add("location_overlap")
    if (set(a.participants) | set(a.locations)) & (set(b.participants) | set(b.locations)):
        signals.add("bound_entity_overlap")
    if _evidence_paragraphs(a) & _evidence_paragraphs(b):
        signals.add("evidence_paragraph_overlap")
    if a.temporal_mode == b.temporal_mode:
        signals.add("temporal_mode_equal")
    return frozenset(signals)


def _relationship_signals(a, b) -> frozenset[str]:
    signals: set[str] = set()
    group_a = frozenset((a.source_entity_ref, a.target_entity_ref))
    group_b = frozenset((b.source_entity_ref, b.target_entity_ref))
    if group_a == group_b:
        signals.add("exact_or_symmetric_endpoint_group")
    if a.direction == b.direction:
        signals.add("direction_equal")
    if normalize_diagnostic_text(a.relationship_type_zh) == normalize_diagnostic_text(
        b.relationship_type_zh
    ):
        signals.add("exact_normalized_relationship_type")
    if diagnostic_tokens(a.relationship_type_zh) & diagnostic_tokens(b.relationship_type_zh):
        signals.add("relationship_type_token_overlap")
    if _evidence_paragraphs(a) & _evidence_paragraphs(b):
        signals.add("evidence_paragraph_overlap")
    return frozenset(signals)


# ---------------------------------------------------------------------------
# Pair enumeration / aggregation (audit-only; deterministic)
# ---------------------------------------------------------------------------


@dataclass
class _PairAnalysis:
    total_pairs: int
    no_signal: int
    signal_counts: Counter
    combo_counts: Counter  # frozenset[str] -> count
    combo_pairs: dict  # frozenset[str] -> list[(left, right)] in source order
    distance_counts: Counter  # distance label -> count


def _pair_analysis(candidates, *, signal_fn, chunk_ordinal: dict[str, int]) -> _PairAnalysis:
    """Enumerate the exact n-choose-2 pair universe (in source order) and
    aggregate the diagnostic signals / source-distance buckets (audit-only)."""
    n = len(candidates)
    no_signal = 0
    signal_counts: Counter = Counter()
    combo_counts: Counter = Counter()
    combo_pairs: dict = {}
    distance_counts: Counter = Counter()
    for i in range(n):
        for j in range(i + 1, n):
            a, b = candidates[i], candidates[j]
            distance_counts[
                _source_distance_label(chunk_ordinal[a.chunk_id], chunk_ordinal[b.chunk_id])
            ] += 1
            signals = signal_fn(a, b)
            if not signals:
                no_signal += 1
                continue
            for sig in signals:
                signal_counts[sig] += 1
            combo = frozenset(signals)
            combo_counts[combo] += 1
            combo_pairs.setdefault(combo, []).append((a, b))
    return _PairAnalysis(
        total_pairs=_naive_pair_count(n),
        no_signal=no_signal,
        signal_counts=signal_counts,
        combo_counts=combo_counts,
        combo_pairs=combo_pairs,
        distance_counts=distance_counts,
    )


def _combo_sort_key(item):
    """Deterministic combo order: count desc, then signal tuple lexical."""
    combo, count = item
    return (-count, tuple(sorted(combo)))


def _duplicate_diagnostics(candidates, key_fn, chunk_ordinal: dict[str, int]) -> dict[str, int]:
    """Conservative exact-duplicate group diagnostics (audit-only)."""
    groups: dict = {}
    for cand in candidates:
        groups.setdefault(key_fn(cand), []).append(cand)
    dup_groups = [members for members in groups.values() if len(members) >= 2]
    same_chunk = 0
    cross_chunk = 0
    for members in dup_groups:
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                if chunk_ordinal[members[i].chunk_id] == chunk_ordinal[members[j].chunk_id]:
                    same_chunk += 1
                else:
                    cross_chunk += 1
    return {
        "duplicate_group_count": len(dup_groups),
        "candidate_count_in_duplicate_groups": sum(len(m) for m in dup_groups),
        "same_chunk_duplicate_pair_count": same_chunk,
        "cross_chunk_duplicate_pair_count": cross_chunk,
    }


def _fact_duplicate_key(f):
    return (
        f.fact_type,
        normalize_diagnostic_text(f.statement_zh),
        tuple(sorted(f.subject_refs)),
        tuple(sorted(f.object_refs)),
    )


def _event_duplicate_key(e):
    return (
        normalize_diagnostic_text(e.summary_zh),
        tuple(sorted(e.participants)),
        tuple(sorted(e.locations)),
        e.temporal_mode,
    )


def _relationship_duplicate_key(r):
    return (
        tuple(sorted((r.source_entity_ref, r.target_entity_ref))),
        r.direction,
        normalize_diagnostic_text(r.relationship_type_zh),
        normalize_diagnostic_text(r.state_zh) if r.state_zh is not None else None,
    )


# ---------------------------------------------------------------------------
# Per-field namespace distribution (kept from the pre-audit audit)
# ---------------------------------------------------------------------------


def _field_namespaces(result: ConsolidationPlanningResult) -> dict[str, Counter]:
    index = result.index
    counts: dict[str, Counter] = {
        "subject_refs": Counter(),
        "object_refs": Counter(),
        "participants": Counter(),
        "locations": Counter(),
        "source_entity_ref": Counter(),
        "target_entity_ref": Counter(),
    }
    for fact in index.facts:
        for ref in fact.subject_refs:
            counts["subject_refs"][_namespace(ref)] += 1
        for ref in fact.object_refs:
            counts["object_refs"][_namespace(ref)] += 1
    for event in index.events:
        for ref in event.participants:
            counts["participants"][_namespace(ref)] += 1
        for ref in event.locations:
            counts["locations"][_namespace(ref)] += 1
    for rel in index.relationships:
        counts["source_entity_ref"][_namespace(rel.source_entity_ref)] += 1
        counts["target_entity_ref"][_namespace(rel.target_entity_ref)] += 1
    return counts


def _source_order_check(candidates) -> tuple[bool, str]:
    if not candidates:
        return True, "<empty>"
    keys = [c.source_order_key for c in candidates]
    if keys != sorted(keys) or len(set(keys)) != len(keys):
        return False, keys[0]
    return True, keys[0]


# ---------------------------------------------------------------------------
# Representative samples (bounded, deterministic)
# ---------------------------------------------------------------------------


def _sample_pairs(combo_pairs: dict, max_samples: int = _MAX_SAMPLES_PER_BUCKET) -> list:
    """All combo pairs, sorted by (left_ref, right_ref), truncated to budget."""
    pairs = list(combo_pairs)
    pairs.sort(key=lambda pair: (pair[0].global_candidate_ref, pair[1].global_candidate_ref))
    return pairs[:max_samples]


def _fact_sample_line(a, b, distance: str, signals: frozenset[str]) -> str:
    return (
        f"      {a.global_candidate_ref} <-> {b.global_candidate_ref} "
        f"[dist={distance}] type={a.fact_type}/{b.fact_type} "
        f"subj={list(a.subject_refs)}/{list(b.subject_refs)} "
        f"obj={list(a.object_refs)}/{list(b.object_refs)} "
        f"stmt_a={normalize_diagnostic_text(a.statement_zh)[:60]!r} "
        f"stmt_b={normalize_diagnostic_text(b.statement_zh)[:60]!r} "
        f"signals={sorted(signals)}"
    )


def _event_sample_line(a, b, distance: str, signals: frozenset[str]) -> str:
    return (
        f"      {a.global_candidate_ref} <-> {b.global_candidate_ref} "
        f"[dist={distance}] mode={a.temporal_mode}/{b.temporal_mode} "
        f"parts={list(a.participants)}/{list(b.participants)} "
        f"locs={list(a.locations)}/{list(b.locations)} "
        f"sum_a={normalize_diagnostic_text(a.summary_zh)[:60]!r} "
        f"sum_b={normalize_diagnostic_text(b.summary_zh)[:60]!r} "
        f"signals={sorted(signals)}"
    )


def _relationship_sample_line(a, b, distance: str, signals: frozenset[str]) -> str:
    return (
        f"      {a.global_candidate_ref} <-> {b.global_candidate_ref} "
        f"[dist={distance}] dir={a.direction}/{b.direction} "
        f"src={a.source_entity_ref}/{b.source_entity_ref} "
        f"tgt={a.target_entity_ref}/{b.target_entity_ref} "
        f"type_a={normalize_diagnostic_text(a.relationship_type_zh)[:40]!r} "
        f"type_b={normalize_diagnostic_text(b.relationship_type_zh)[:40]!r} "
        f"state_a={(a.state_zh or '')[:30]!r} state_b={(b.state_zh or '')[:30]!r} "
        f"signals={sorted(signals)}"
    )


def _distance_for(chunk_ordinal: dict[str, int], a, b) -> str:
    return _source_distance_label(chunk_ordinal[a.chunk_id], chunk_ordinal[b.chunk_id])


# ---------------------------------------------------------------------------
# Audit checks (read-only gate)
# ---------------------------------------------------------------------------


def _audit_checks(
    result: ConsolidationPlanningResult,
    *,
    alice_gates: bool = False,
    direction_aware_expected: int | None = None,
    unordered_ceiling: int | None = None,
) -> list[tuple[str, bool]]:
    index = result.index
    coverage = result.coverage
    counts = _field_namespaces(result)
    checks: list[tuple[str, bool]] = []

    checks.append(
        (
            "index counts match coverage (facts/events/relationships)",
            coverage.fact_candidate_count == len(index.facts)
            and coverage.event_candidate_count == len(index.events)
            and coverage.relationship_candidate_count == len(index.relationships),
        )
    )
    checks.append(
        (
            "Phase A canonical/decision/conflict counts are zero",
            coverage.canonical_fact_count == 0
            and coverage.canonical_event_count == 0
            and coverage.canonical_relationship_count == 0
            and coverage.uncertain_decision_count == 0
            and coverage.story_conflict_count == 0,
        )
    )
    for label, cands in (
        ("facts source-ordered and unique", index.facts),
        ("events source-ordered and unique", index.events),
        ("relationships source-ordered and unique", index.relationships),
    ):
        ok, _ = _source_order_check(cands)
        checks.append((label, ok))

    person_ok = all(
        counts[field][_ns] == 0 for field in _PERSON_FIELDS for _ns in ("loc",)
    )
    checks.append(("person fields bind no loc_*", person_ok))
    location_ok = all(
        counts[field][_ns] == 0 for field in _LOCATION_FIELDS for _ns in ("char",)
    )
    checks.append(("location fields bind no char_*", location_ok))
    other_ok = all(
        counts[field][_ns] == 0 for field in counts for _ns in ("other",)
    )
    checks.append(("every bound id is in the char_/loc_/unres_ namespace", other_ok))

    # Phase B deterministic-planning invariants (always checked).
    fact_auto, _ = _pair_plan_split(result.fact_pair_plans)
    event_auto, _ = _pair_plan_split(result.event_pair_plans)
    rel_auto, _ = _pair_plan_split(result.relationship_pair_plans)
    auto_total = fact_auto + event_auto + rel_auto
    n_fact = len(result.fact_pair_plans)
    n_event = len(result.event_pair_plans)
    n_rel = len(result.relationship_pair_plans)
    total = n_fact + n_event + n_rel
    decision_total = _decision_set_size(result.deterministic_decision_set)
    checks.append(
        (
            "deterministic decision set size == auto_same pair count",
            decision_total == auto_total,
        )
    )
    checks.append(
        (
            "plan hash is a 64-hex canonical content hash",
            isinstance(result.plan_hash, str)
            and len(result.plan_hash) == 64
            and all(c in "0123456789abcdef" for c in result.plan_hash),
        )
    )

    # Alice-specific acceptance gates (only for the Alice corpus, blocking-v1).
    if not alice_gates:
        return checks
    checks.append(
        (
            f"auto_same pair count == {ALICE_GATE_AUTO_SAME_EXACT} (got {auto_total})",
            auto_total == ALICE_GATE_AUTO_SAME_EXACT,
        )
    )
    checks.append(
        (
            f"fact pair plans <= {ALICE_GATE_FACT_MAX} (got {n_fact})",
            n_fact <= ALICE_GATE_FACT_MAX,
        )
    )
    checks.append(
        (
            f"event pair plans <= {ALICE_GATE_EVENT_MAX} (got {n_event})",
            n_event <= ALICE_GATE_EVENT_MAX,
        )
    )
    # Relationship acceptance is direction-aware (section 9.1): the actual
    # explicit count must equal the independently-computed direction-aware
    # endpoint bucket count AND be <= the unordered Phase-A ceiling (845).
    if direction_aware_expected is None:
        direction_aware_expected = _relationship_direction_aware_expected(index)
    if unordered_ceiling is None:
        unordered_ceiling = _relationship_unordered_ceiling(index)
    checks.append(
        (
            f"relationship pair plans == direction_aware_expected "
            f"{direction_aware_expected} and <= unordered ceiling "
            f"{ALICE_GATE_RELATIONSHIP_MAX} (got {n_rel})",
            n_rel == direction_aware_expected
            and n_rel <= ALICE_GATE_RELATIONSHIP_MAX,
        )
    )
    checks.append(
        (
            f"total pair plans <= {ALICE_GATE_TOTAL_MAX} (got {total})",
            total <= ALICE_GATE_TOTAL_MAX,
        )
    )
    return checks


# ---------------------------------------------------------------------------
# Report sections
# ---------------------------------------------------------------------------


def _print_input_identity(result: ConsolidationPlanningResult) -> None:
    snapshot = result.snapshot
    em_ref = snapshot.entity_map_ref
    print("=== EXACT INPUT IDENTITY ===")
    print("EntityMap:")
    print(f"  artifact_type:  {em_ref.artifact_type}")
    print(f"  artifact_id:    {em_ref.artifact_id}")
    print(f"  revision:       {em_ref.revision}")
    print(f"  content_hash:   {em_ref.content_hash}")
    entries = snapshot.entity_map.entries
    resolved = sum(1 for e in entries if e.status == "resolved")
    unresolved = sum(1 for e in entries if e.status == "unresolved")
    print(f"  entries:        {len(entries)} (resolved={resolved} / unresolved={unresolved})")
    print(f"A4 validation report ref:  {_artifact_ref_str(snapshot.a4_validation_report_ref)}")
    print(f"A4 CURRENT pointer ref:    {snapshot.a4_current_pointer_ref.artifact_id}")
    print(f"extraction profile:        {snapshot.a3_input.extraction_profile_id}:"
          f"{snapshot.a3_input.extraction_profile_hash}")
    print()
    print("source_document_ref:")
    print(f"  {_artifact_ref_str(snapshot.a3_input.source_document_ref)}")
    print("chunk_manifest_ref:")
    print(f"  {_artifact_ref_str(snapshot.a3_input.chunk_manifest_ref)}")
    print(f"CandidateExtraction refs ({len(snapshot.candidate_extraction_refs)}):")
    for position, ref in enumerate(snapshot.candidate_extraction_refs, start=1):
        print(f"  [{position:2}] {_artifact_ref_str(ref)}")
    print()


def _print_candidate_universe(result: ConsolidationPlanningResult) -> None:
    index = result.index
    total = len(index.facts) + len(index.events) + len(index.relationships)
    print("=== CANDIDATE UNIVERSE ===")
    print(f"fact_candidate_count:         {len(index.facts)}")
    print(f"event_candidate_count:        {len(index.events)}")
    print(f"relationship_candidate_count: {len(index.relationships)}")
    print(f"total_candidate_count:        {total}")
    print()


def _print_bound_refs(result: ConsolidationPlanningResult) -> None:
    index = result.index
    per_field = _field_namespaces(result)

    def _total(*namespaces: str) -> int:
        return sum(
            per_field[field][ns]
            for field in per_field
            for ns in namespaces
        )

    char_total = _total("char")
    loc_total = _total("loc")
    unres_total = _total("unres")
    all_total = char_total + loc_total + unres_total + _total("other")

    facts_unres = sum(1 for f in index.facts if _has_unresolved_ref(f))
    events_unres = sum(1 for e in index.events if _has_unresolved_ref(e))
    rels_unres = sum(1 for r in index.relationships if _has_unresolved_ref(r))
    any_unres = facts_unres + events_unres + rels_unres

    print("=== BOUND REFS ===")
    print("bound_ref_occurrence_count:              "
          f"{all_total}")
    print("resolved_char_ref_occurrence_count:      "
          f"{char_total}")
    print("resolved_loc_ref_occurrence_count:       "
          f"{loc_total}")
    print("unresolved_ref_occurrence_count:         "
          f"{unres_total}")
    print()
    print("candidates_with_any_unresolved_ref:      "
          f"{any_unres}")
    print("facts_with_unresolved_ref:               "
          f"{facts_unres}")
    print("events_with_unresolved_ref:              "
          f"{events_unres}")
    print("relationships_with_unresolved_ref:       "
          f"{rels_unres}")
    print()
    print("per-field namespace distribution (occurrence counts):")
    for field, counter in per_field.items():
        dist = " ".join(f"{ns}={n}" for ns, n in sorted(counter.items())) or "(none)"
        print(f"  {field:18} {dist}")
    print()


def _print_facts(result: ConsolidationPlanningResult) -> None:
    index = result.index
    print("=== FACTS ===")
    print("facts_by_type:")
    print("\n".join(_format_distribution(Counter(f.fact_type for f in index.facts))))
    print("fact_subject_count_distribution:")
    print("\n".join(
        f"    {value}: {count}"
        for value, count in _distribution_rows([len(f.subject_refs) for f in index.facts])
    ))
    print("fact_object_count_distribution:")
    print("\n".join(
        f"    {value}: {count}"
        for value, count in _distribution_rows([len(f.object_refs) for f in index.facts])
    ))
    print()


def _print_events(result: ConsolidationPlanningResult) -> None:
    index = result.index
    print("=== EVENTS ===")
    print("events_by_temporal_mode:")
    print("\n".join(_format_distribution(Counter(e.temporal_mode for e in index.events))))
    print("event_participant_count_distribution:")
    print("\n".join(
        f"    {value}: {count}"
        for value, count in _distribution_rows([len(e.participants) for e in index.events])
    ))
    print("event_location_count_distribution:")
    print("\n".join(
        f"    {value}: {count}"
        for value, count in _distribution_rows([len(e.locations) for e in index.events])
    ))
    print()


def _print_relationships(result: ConsolidationPlanningResult) -> None:
    index = result.index
    print("=== RELATIONSHIPS ===")
    print("relationships_by_direction:")
    print("\n".join(_format_distribution(Counter(r.direction for r in index.relationships))))

    endpoint_groups = Counter(
        frozenset((r.source_entity_ref, r.target_entity_ref)) for r in index.relationships
    )
    unique_groups = len(endpoint_groups)
    top = sorted(endpoint_groups.items(), key=lambda kv: (-kv[1], tuple(sorted(kv[0]))))[
        :_FREQ_TOP
    ]
    print(f"relationship_endpoint_groups (unique={unique_groups}, top {_FREQ_TOP}):")
    for group, count in top:
        print(f"    {{{', '.join(sorted(group))}}}: {count}")

    type_freq = Counter(normalize_diagnostic_text(r.relationship_type_zh) for r in index.relationships)
    unique_types = len(type_freq)
    top_types = sorted(type_freq.items(), key=lambda kv: (-kv[1], kv[0]))[:_FREQ_TOP]
    print(f"normalized_relationship_type_frequency (unique={unique_types}, top {_FREQ_TOP}):")
    for norm_type, count in top_types:
        print(f"    {norm_type!r}: {count}")
    print()


def _print_naive_pair_universe(result: ConsolidationPlanningResult) -> None:
    index = result.index
    n_fact = _naive_pair_count(len(index.facts))
    n_event = _naive_pair_count(len(index.events))
    n_rel = _naive_pair_count(len(index.relationships))
    print("=== NAIVE PAIR UNIVERSE ===")
    print(f"naive_fact_pair_count:         {n_fact}")
    print(f"naive_event_pair_count:        {n_event}")
    print(f"naive_relationship_pair_count: {n_rel}")
    print(f"naive_pair_count_total:        {n_fact + n_event + n_rel}")
    print()


def _pair_plan_split(plans) -> tuple[int, int]:
    """Return ``(auto_same_count, needs_semantic_decision_count)`` for a plan list."""
    auto = sum(1 for p in plans if p.state == "auto_same")
    needs = sum(1 for p in plans if p.state == "needs_semantic_decision")
    return auto, needs


def _signal_combination_lines(plans, *, top: int = 25) -> list[str]:
    """Deterministic atomic-signal-combination distribution for a plan list.

    Each pair plan carries the COMPLETE set of frozen atomic signals (a
    lexical-sorted, unique tuple). This reports how many pairs share each
    distinct signal combination (bounded to ``top`` for a stable report).
    """
    combos: dict = Counter(tuple(p.signals) for p in plans)
    lines = [
        f"      distinct signal combinations: {len(combos)}  (pairs: {len(plans)})"
    ]
    for combo, count in sorted(combos.items(), key=lambda item: (-item[1], item[0]))[:top]:
        lines.append(f"      [{', '.join(combo)}]: {count}")
    if len(combos) > top:
        lines.append(f"      ... ({len(combos) - top} more combinations)")
    return lines


def _print_pair_planning(
    result: ConsolidationPlanningResult,
    *,
    direction_aware_expected: int,
    unordered_ceiling: int,
) -> None:
    """Report the production blocking-v1 pair plans (Phase B, zero-provider).

    This is the deterministic, no-N^2 pair generation output -- the exact
    material that would feed the semantic stream (#52/#53). It is NOT the
    diagnostic corpus-shape enumeration above.
    """
    index = result.index
    n_fact = len(index.facts)
    n_event = len(index.events)
    n_rel = len(index.relationships)
    fact_auto, fact_needs = _pair_plan_split(result.fact_pair_plans)
    event_auto, event_needs = _pair_plan_split(result.event_pair_plans)
    rel_auto, rel_needs = _pair_plan_split(result.relationship_pair_plans)
    auto_total = fact_auto + event_auto + rel_auto
    semantic_total = fact_needs + event_needs + rel_needs
    fact_explicit = len(result.fact_pair_plans)
    event_explicit = len(result.event_pair_plans)
    rel_explicit = len(result.relationship_pair_plans)
    explicit_total = fact_explicit + event_explicit + rel_explicit
    naive_total = (
        _naive_pair_count(n_fact) + _naive_pair_count(n_event) + _naive_pair_count(n_rel)
    )

    print("=== A5B ALICE PRODUCTION PLANNING ===")
    print(f"blocking_policy_id:            {result.blocking_policy_id}")
    print(f"text_normalization_policy_id:  {result.text_normalization_policy_id}")
    print(f"exact_safe_policy_id:          {result.exact_safe_policy_id}")
    print(f"planning_policy_id:            {result.planning_policy_id}")
    print()
    print("Fact:")
    print(f"  candidate:   {n_fact}")
    print(f"  naive:       {_naive_pair_count(n_fact)}")
    print(f"  explicit:    {fact_explicit}")
    print(f"  auto_same:   {fact_auto}")
    print(f"  semantic:    {fact_needs}")
    print("Event:")
    print(f"  candidate:   {n_event}")
    print(f"  naive:       {_naive_pair_count(n_event)}")
    print(f"  explicit:    {event_explicit}")
    print(f"  auto_same:   {event_auto}")
    print(f"  semantic:    {event_needs}")
    print("Relationship:")
    print(f"  candidate:   {n_rel}")
    print(f"  naive:       {_naive_pair_count(n_rel)}")
    print(f"  unordered_phase_a_ceiling:  {unordered_ceiling}")
    print(f"  direction_aware_expected:   {direction_aware_expected}")
    print(f"  explicit:    {rel_explicit}")
    print(f"  auto_same:   {rel_auto}")
    print(f"  semantic:    {rel_needs}")
    print("Totals:")
    print(f"  naive:       {naive_total}")
    print(f"  explicit:    {explicit_total}")
    print(f"  auto_same:   {auto_total}")
    print(f"  semantic:    {semantic_total}")
    print(f"plan_hash:     {result.plan_hash}")
    print()
    print("Production atomic signal combinations:")
    print("  Fact:")
    for line in _signal_combination_lines(result.fact_pair_plans):
        print(line)
    print("  Event:")
    for line in _signal_combination_lines(result.event_pair_plans):
        print(line)
    print("  Relationship:")
    for line in _signal_combination_lines(result.relationship_pair_plans):
        print(line)
    print()


def _print_source_distance(result: ConsolidationPlanningResult) -> None:
    index = result.index
    ordinal = _chunk_ordinal_by_chunk_id(result.snapshot)
    analyses = {
        "FACT": _pair_analysis(index.facts, signal_fn=_fact_signals, chunk_ordinal=ordinal),
        "EVENT": _pair_analysis(index.events, signal_fn=_event_signals, chunk_ordinal=ordinal),
        "RELATIONSHIP": _pair_analysis(
            index.relationships, signal_fn=_relationship_signals, chunk_ordinal=ordinal
        ),
    }
    print("=== SOURCE DISTANCE ===")
    for domain, analysis in analyses.items():
        d = analysis.distance_counts
        print(
            f"{domain:13} same_chunk={d['same_chunk']} "
            f"distance_1={d['distance_1']} distance_2={d['distance_2']} "
            f"distance_3_plus={d['distance_3_plus']}"
        )
    print()
    return analyses


def _print_signal_section(
    domain: str,
    analysis: _PairAnalysis,
    signal_labels: tuple[str, ...],
) -> None:
    print(f"{domain} SIGNAL COUNTS")
    for label in signal_labels:
        print(f"  {label:34} {analysis.signal_counts.get(label, 0)}")
    print(f"{domain} no_signal: {analysis.no_signal}")
    print(f"{domain} SIGNAL COMBINATIONS")
    if not analysis.combo_counts:
        print("  (none)")
    for combo, count in sorted(analysis.combo_counts.items(), key=_combo_sort_key):
        print(f"  {count:6} {' | '.join(sorted(combo))}")
    print()


def _print_samples(
    domain: str,
    analysis: _PairAnalysis,
    chunk_ordinal: dict[str, int],
    sample_line_fn,
) -> None:
    print(f"{domain}:")
    if not analysis.combo_counts:
        print("  (no signal-bearing pairs)")
    for combo, count in sorted(analysis.combo_counts.items(), key=_combo_sort_key):
        print(f"  [count={count}] {' | '.join(sorted(combo))}")
        for a, b in _sample_pairs(analysis.combo_pairs[combo]):
            print(
                sample_line_fn(
                    a, b, _distance_for(chunk_ordinal, a, b), frozenset(combo)
                )
            )
    print()


def _print_duplicate_diagnostics(
    domain: str, diagnostics: dict[str, int]
) -> None:
    print(f"{domain}:")
    for label in (
        "duplicate_group_count",
        "candidate_count_in_duplicate_groups",
        "same_chunk_duplicate_pair_count",
        "cross_chunk_duplicate_pair_count",
    ):
        print(f"  {label:36} {diagnostics[label]}")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_audit(
    store: FileArtifactStore,
    pointers: FilePointerStore,
    *,
    project_id: str,
    document_id: str,
    reconciliation_profile_id: str,
    consolidation_profile_path: Path = DEFAULT_CONSOLIDATION_PROFILE,
    alice_gates: bool = False,
) -> int:
    consolidation_profile = load_consolidation_profile(consolidation_profile_path)
    result = build_consolidation_planning(
        store,
        pointers,
        project_id=project_id,
        document_id=document_id,
        reconciliation_profile_id=reconciliation_profile_id,
        consolidation_profile=consolidation_profile,
    )
    index = result.index
    ordinal = _chunk_ordinal_by_chunk_id(result.snapshot)
    # Direction-aware relationship acceptance (section 9.1): the independently
    # computed direction-aware endpoint bucket count and the unordered Phase-A
    # ceiling (845 for Alice). These are NOT read from the production planner's
    # own output, so the relationship gate is not a tautology.
    rel_direction_aware = _relationship_direction_aware_expected(index)
    rel_unordered_ceiling = _relationship_unordered_ceiling(index)

    print("=== A5B ZERO-PROVIDER ALICE AUDIT ===")
    print(f"project:              {project_id}")
    print(f"document:             {document_id}")
    print(f"reconciliation:       {reconciliation_profile_id}")
    print()

    # 1. Exact input identity
    _print_input_identity(result)
    # 2. Candidate universe
    _print_candidate_universe(result)
    # 3. Bound refs
    _print_bound_refs(result)
    # 4. Fact distributions
    _print_facts(result)
    # 5. Event distributions
    _print_events(result)
    # 6. Relationship distributions
    _print_relationships(result)
    # 7. Naive pair universe
    _print_naive_pair_universe(result)
    # 7b. Production blocking-v1 pair plans (Phase B, zero-provider)
    _print_pair_planning(
        result,
        direction_aware_expected=rel_direction_aware,
        unordered_ceiling=rel_unordered_ceiling,
    )
    # 8. Source distance (also returns the pair analyses reused below)
    analyses = _print_source_distance(result)

    # 9-11. Diagnostic signals per domain
    print("=== OVERLAP SIGNALS ===")
    _print_signal_section("FACT", analyses["FACT"], _FACT_SIGNALS)
    _print_signal_section("EVENT", analyses["EVENT"], _EVENT_SIGNALS)
    _print_signal_section("RELATIONSHIP", analyses["RELATIONSHIP"], _REL_SIGNALS)

    # 13. Exact duplicate diagnostics
    print("=== EXACT DUPLICATE DIAGNOSTICS ===")
    _print_duplicate_diagnostics(
        "FACT",
        _duplicate_diagnostics(index.facts, _fact_duplicate_key, ordinal),
    )
    _print_duplicate_diagnostics(
        "EVENT",
        _duplicate_diagnostics(index.events, _event_duplicate_key, ordinal),
    )
    _print_duplicate_diagnostics(
        "RELATIONSHIP",
        _duplicate_diagnostics(index.relationships, _relationship_duplicate_key, ordinal),
    )
    print()

    # 14. Representative samples
    print("=== REPRESENTATIVE SAMPLES ===")
    _print_samples("FACT", analyses["FACT"], ordinal, _fact_sample_line)
    _print_samples("EVENT", analyses["EVENT"], ordinal, _event_sample_line)
    _print_samples(
        "RELATIONSHIP", analyses["RELATIONSHIP"], ordinal, _relationship_sample_line
    )

    # Read-only audit checks (gate)
    print("=== AUDIT CHECKS ===")
    checks = _audit_checks(
        result,
        alice_gates=alice_gates,
        direction_aware_expected=rel_direction_aware,
        unordered_ceiling=rel_unordered_ceiling,
    )
    all_pass = True
    for label, passed in checks:
        all_pass = all_pass and passed
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}")
    print()

    if not all_pass:
        print("A5B AUDIT RESULT: FAIL", file=sys.stderr)
        return 2
    print("A5B AUDIT RESULT: PASS (zero-provider, read-only)")
    print("Phase A + Phase B complete: input binding + indexing + source order +")
    print("coverage + corpus-shape audit + deterministic blocking-v1 pair planning.")
    print("Still zero-provider: no LLM call, no canonical id, no A5 persistence/CURRENT.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="A5B zero-provider input binding / indexing / corpus-shape audit"
    )
    parser.add_argument("--runs-root", default=str(DEFAULT_RUNS_ROOT))
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--document", default=DEFAULT_DOCUMENT)
    parser.add_argument("--profile", default=DEFAULT_PROFILE)
    parser.add_argument(
        "--no-alice-gates",
        action="store_true",
        help="skip the Alice-specific blocking-v1 acceptance gates "
        "(structural + deterministic checks still apply)",
    )
    args = parser.parse_args(argv)

    try:
        store, pointers = _stores(args.runs_root, args.project)
        return run_audit(
            store,
            pointers,
            project_id=args.project,
            document_id=args.document,
            reconciliation_profile_id=args.profile,
            alice_gates=not args.no_alice_gates,
        )
    except (ConsolidationCurrentMissingError, StoryIntegrityError) as exc:
        print(f"A5B AUDIT RESULT: FAIL ({type(exc).__name__}: {exc})", file=sys.stderr)
        return 2
    except Exception as exc:  # pragma: no cover - defensive (unexpected)
        print(f"A5B AUDIT RESULT: ERROR ({type(exc).__name__}: {exc})", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
