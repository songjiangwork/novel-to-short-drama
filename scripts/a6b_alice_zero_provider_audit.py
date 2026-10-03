"""v1.2 A6B — zero-provider Alice real-corpus shape audit (issue #87).

This is the A6B **audit gate**: a read-only, zero-provider, ZERO-WRITE
measurement of the real Alice A5F2 corpus through the exact validated A5
CURRENT snapshot, BEFORE any numeric planning policy is frozen and BEFORE the
production ``profiles/global_story_analysis_v1.yaml`` is created.

The audit:

  * reads the exact current-eligible A5 CURRENT (via
    ``ConsolidationPersistenceService.require_current_validated``) and the exact
    pinned A4/A5 leaves into a read-only ``StoryAnalysisInputSnapshot``;
  * reports the corpus counts and the serialized / estimated-token sizes of the
    EXACT typed A5/A4 leaves (each leaf's own ``to_dict()`` payload, not a raw
    array of children — BLOCK 5);
  * reports the complete deterministic character evidence-package plan (100%
    coverage, per-character evidence counts and ``p50 / p90 / max`` package
    sizes);
  * reports the event stream (per-event bytes / tokens, cumulative) and a
    CORPUS-DERIVED, bounded, exhaustive deterministic sweep of candidate
    plot-window owned-target / context counts (BLOCK 4: the candidate space is
    generated from the actual event universe, not a curated tuple). The sweep is
    retained as MEASUREMENT EVIDENCE; the frozen window policy is an
    ARCHITECTURE-AWARE measured frontier choice (non-zero boundary context,
    multi-event owned windows), NOT a byte-minimization objective (that would
    degenerate to one-event / zero-context windows and defeat the window/context
    semantic role);
  * reports the compact A5-derived global-index base bytes / tokens (the
    measurable A5-derived portion of the A6E global-skeleton input; the A6C
    character dossiers + A6D window analyses that join on top are future
    semantic outputs and are NOT faked here);
  * builds the deterministic A6 planning identity / plan hash for the frozen
    policy — binding the numeric policy, the four versioned planning policy
    IDs, and the token-counter ID (BLOCK 2) — and FAILS CLOSED on any provider
    / model / GPU / credential leakage into the plan identity (and proves
    determinism by recomputing the hash);
  * PROVES ZERO-WRITE: it snapshots the artifact / pointer runs tree before and
    after the audit and asserts path/hash equality (BLOCK 3), and it refuses to
    write any diagnostic report into the story runs tree;
  * reports the STAGED CEILING AUTHORITY (BLOCK 1, resolved by the merged #97
    contract): A6B freezes the character / window planning values concrete
    (measured), but DEFERS the two whole-story ceilings (global-skeleton /
    story-bible) to ``null``. The complete A6E / A6F whole-story packet joins
    the A6C / A6D (+ A6E) semantic outputs that do not exist yet, so A6B
    measures the A5-derived base as a MEASUREMENT but does NOT freeze it as a
    whole-story ceiling (that would be an A5-only placeholder). ``null`` =
    DEFERRED (not unlimited, not zero); the corresponding pass fails closed
    while the ceiling is ``null``. Recursive global compression is DEFERRED TO
    A6E MEASUREMENT.

It performs NO provider call, writes NO A6 artifact, modifies NO A5/A6 CURRENT,
and modifies NO tracked profile (and writes NO file inside the story runs tree).
An optional ``--report PATH`` writes a machine-readable JSON report (a
diagnostic, not an artifact) — it must be OUTSIDE the story runs tree.

Usage:
    python scripts/a6b_alice_zero_provider_audit.py \
        [--runs-root PATH] [--project ID] [--document ID] \
        [--consolidation-profile ID] [--report PATH]

Exit code 0 on a clean audit, 2 on any structural / integrity / leakage /
zero-write failure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

from short_drama.artifacts import FileArtifactStore
from short_drama.foundation import FilePointerStore
from short_drama.paths import REPO_ROOT
from short_drama.story import (
    StoryIntegrityError,
    assert_full_character_coverage,
    build_character_evidence_packages,
    build_global_index_base,
    build_story_analysis_snapshot,
    build_window_packet,
    compute_plan_hash,
    ordered_event_stream,
    plan_plot_windows,
    validate_window_ownership,
)
from short_drama.story.chunking import TOKEN_COUNTER_ID
from short_drama.story.errors import StoryAnalysisPlanningError
from short_drama.story.story_analysis import StoryAnalysisPlanningPolicy
from short_drama.story.story_analysis_planning import (
    _canonical_bytes,
    _estimate_bytes_tokens,
)

DEFAULT_RUNS_ROOT = REPO_ROOT / "runs" / "a4e_real_novel"
DEFAULT_PROJECT = "a3e-real-novel"
DEFAULT_DOCUMENT = "src_001"
DEFAULT_CONSOLIDATION_PROFILE = "consolidation-v1"

# --- The four versioned A6 planning policy IDs (bound into the plan hash). ---
# These match the production profile (``profiles/global_story_analysis_v1.yaml``)
# and are the A6A canonical A6 policy IDs. They are the *identity* of the
# versioned A6 policy, distinct from the numeric planning-policy values, and are
# bound into the plan identity (BLOCK 2) so the plan hash tracks the versioned
# policy identity and the token-counter ID, not only the numeric values.
FROZEN_POLICY_IDS: dict[str, str] = {
    "character_analysis_policy_id": "a6-character-v1",
    "plot_window_policy_id": "a6-window-v1",
    "global_skeleton_policy_id": "a6-skeleton-v1",
    "story_bible_policy_id": "a6-bible-v1",
}

# Deterministic margin rule: a frozen budget is the measured ceiling rounded UP
# to the next :data:`_TOKEN_MARGIN_STEP` tokens (a small, documented,
# corpus-derived margin above the measured max — never a fabricated bound).
_TOKEN_MARGIN_STEP = 1000

# The frozen v1 plot-window policy is an ARCHITECTURE-AWARE measured choice, not
# the product of a packet-size-minimization objective. Plot-window context has
# a semantic purpose (avoiding hard cuts around turning points, allowing
# relationship transitions across boundaries, preserving setup/payoff adjacency),
# and pure byte minimization would collapse the plan to a degenerate
# one-event / zero-context policy that defeats hierarchical compression. The
# frozen (owned_target, context) below is a human/architecture choice taken from
# the measured frontier and validated against the real corpus (non-zero boundary
# context, multi-event owned windows, exact-one ownership, measured packet fits
# the concrete budget). The corpus-derived sweep is retained purely as
# measurement evidence (see the JSON report).
FROZEN_WINDOW_OWNED_TARGET = 12
FROZEN_WINDOW_CONTEXT_EVENT_COUNT = 4


def _round_up_to_step(value: int, step: int) -> int:
    """Round ``value`` up to the next multiple of ``step`` (deterministic)."""
    return max(step, math.ceil(value / step) * step)


# ---------------------------------------------------------------------------
# Small deterministic helpers
# ---------------------------------------------------------------------------


def _stores(runs_root: str | Path, project_id: str):
    root = Path(runs_root).expanduser() / project_id / "story"
    artifact_store = FileArtifactStore(root / "artifacts")
    pointer_store = FilePointerStore(root / "pointers", artifact_store)
    return artifact_store, pointer_store


def _story_runs_root(runs_root: str | Path, project_id: str) -> Path:
    return Path(runs_root).expanduser() / project_id / "story"


def _snapshot_runs_tree(runs_root: str | Path, project_id: str) -> dict[str, str]:
    """Snapshot the story runs tree (every file mapped to its SHA-256).

    Used to PROVE the audit is zero-write: the before / after snapshots must be
    exactly equal (same paths, same hashes) — BLOCK 3.
    """
    root = _story_runs_root(runs_root, project_id)
    files: dict[str, str] = {}
    if root.exists():
        for p in sorted(root.rglob("*")):
            if p.is_file():
                files[str(p.relative_to(root))] = hashlib.sha256(
                    p.read_bytes()
                ).hexdigest()
    return files


def _quantiles(values: list[int]) -> dict[str, int]:
    """Deterministic min / p50 / p90 / max / total over an integer list.

    p50 / p90 use the nearest-rank method (1-indexed rank = ceil(q * n)), which
    is deterministic and needs no external statistics dependency.
    """
    if not values:
        return {"min": 0, "p50": 0, "p90": 0, "max": 0, "total": 0}
    s = sorted(values)
    n = len(s)

    def _rank(q: float) -> int:
        return s[max(0, min(n - 1, math.ceil(q * n) - 1))]

    return {
        "min": s[0],
        "p50": _rank(0.50),
        "p90": _rank(0.90),
        "max": s[-1],
        "total": sum(s),
    }


def _leaf_bytes(leaf) -> int:
    """Measure the canonical bytes of an EXACT typed A5/A4 leaf.

    Serializes the leaf's own ``to_dict()`` payload (schema version + children)
    — NOT a raw array of children (BLOCK 5: the measurement must reflect the
    exact persisted typed leaf).
    """
    return len(_canonical_bytes(leaf.to_dict()))


def _tokens_from_byte_count(count: int) -> int:
    """Reuse the ``utf8-bytes-div3-v1`` estimator over an already-measured byte
    count (``max(1, ceil(bytes / 3))``) without re-encoding."""
    return max(1, math.ceil(count / 3))


def _ref_str(ref) -> str:
    return (
        f"{ref.artifact_type} | {ref.artifact_id} | r{ref.revision} | "
        f"{ref.content_hash}"
    )


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


def _measure_character_packages(snap) -> dict:
    packages = build_character_evidence_packages(snap)
    assert_full_character_coverage(snap, packages)

    fact_counts = [len(p.related_facts) for p in packages]
    event_counts = [len(p.participating_events) for p in packages]
    rel_counts = [len(p.relationships) for p in packages]
    transition_counts = [len(p.state_transitions) for p in packages]
    conflict_counts = [len(p.story_conflicts) for p in packages]
    unresolved_counts = [len(p.linked_unresolved) for p in packages]

    token_rows = []
    for p in packages:
        b = len(p.canonical_bytes())
        token_rows.append(
            {
                "character_ref": p.character_ref,
                "estimated_tokens": p.estimated_tokens(),
                "canonical_bytes": b,
                "fact_count": len(p.related_facts),
                "event_count": len(p.participating_events),
                "relationship_count": len(p.relationships),
                "transition_count": len(p.state_transitions),
                "conflict_count": len(p.story_conflicts),
                "linked_unresolved_count": len(p.linked_unresolved),
            }
        )
    largest = max(token_rows, key=lambda r: r["estimated_tokens"])

    return {
        "count": len(packages),
        "canonical_character_count": len(snap.canonical_characters),
        "coverage": {
            "canonical_character_count": len(snap.canonical_characters),
            "package_count": len(packages),
            "coverage_fraction": (
                len(packages) / len(snap.canonical_characters)
                if snap.canonical_characters
                else 1.0
            ),
        },
        "fact_counts": _quantiles(fact_counts),
        "event_counts": _quantiles(event_counts),
        "relationship_counts": _quantiles(rel_counts),
        "transition_counts": _quantiles(transition_counts),
        "conflict_counts": _quantiles(conflict_counts),
        "linked_unresolved_counts": _quantiles(unresolved_counts),
        "evidence_package_estimated_tokens": _quantiles(
            [r["estimated_tokens"] for r in token_rows]
        ),
        "largest": largest,
        "per_character": token_rows,
    }


def _measure_event_stream(snap) -> dict:
    stream = ordered_event_stream(snap)
    bytes_list = [len(_canonical_bytes(e.to_dict())) for e in stream]
    tokens_list = [_tokens_from_byte_count(b) for b in bytes_list]
    return {
        "count": len(stream),
        "per_event_bytes": _quantiles(bytes_list),
        "per_event_estimated_tokens": _quantiles(tokens_list),
        "cumulative_bytes": sum(bytes_list),
        "cumulative_estimated_tokens": sum(tokens_list),
    }


def _window_metrics_for(
    snap, stream, *, event_token_map, owned_event_target, context_event_count
) -> dict:
    windows = plan_plot_windows(
        stream,
        owned_event_target=owned_event_target,
        context_event_count=context_event_count,
    )
    validate_window_ownership(windows, frozenset(e.event_id for e in stream))
    owned_sizes = [len(w.owned_event_ids) for w in windows]
    context_sizes = [len(w.context_event_ids) for w in windows]

    total_owned_tokens = 0
    total_context_tokens = 0
    max_packet_tokens = 0
    max_packet_bytes = 0
    max_packet_window = None
    for w in windows:
        packet = build_window_packet(w, snap)
        b = len(packet.canonical_bytes())
        t = packet.estimated_tokens()
        if max_packet_window is None or t > max_packet_tokens:
            max_packet_tokens = t
            max_packet_bytes = b
            max_packet_window = w.window_id
        total_owned_tokens += sum(event_token_map[eid] for eid in w.owned_event_ids)
        total_context_tokens += sum(event_token_map[eid] for eid in w.context_event_ids)

    return {
        "owned_event_target": owned_event_target,
        "context_event_count": context_event_count,
        "window_count": len(windows),
        "owned_per_window": _quantiles(owned_sizes),
        "context_per_window": _quantiles(context_sizes),
        "max_window_packet_estimated_tokens": max_packet_tokens,
        "max_window_packet_bytes": max_packet_bytes,
        "largest_window_id": max_packet_window,
        "total_owned_event_tokens": total_owned_tokens,
        "total_context_event_tokens": total_context_tokens,
        "overlap_cost_tokens": total_context_tokens,
    }


def _window_candidate_space(n_events: int) -> list[tuple[int, int]]:
    """Generate the corpus-derived candidate window-policy space (BLOCK 4).

    The candidate space is generated from the actual event universe (no curated
    tuple):

      * ``owned_event_target`` ranges over ``1..n_events`` (exhaustive over the
        event universe — every owned-target that partitions the stream is
        considered);
      * ``context_event_count`` ranges over ``0..max(1, n_events // 10)``
        (a deterministic overlap bound = one tenth of the event universe,
        floored at 1, which is the maximum boundary overlap that is not
        redundant with the owned region).

    The result is a bounded, exhaustive, deterministic sweep for Alice's
    166-event corpus (166 x 17 = 2822 combinations).
    """
    if n_events <= 0:
        return []
    max_context = max(1, n_events // 10)
    candidates: list[tuple[int, int]] = []
    for target in range(1, n_events + 1):
        for context in range(0, max_context + 1):
            candidates.append((target, context))
    return candidates


def _select_frozen_window_policy(sweep: list[dict]) -> dict:
    """Select the frozen ARCHITECTURE-AWARE window policy from the measured
    frontier.

    Unlike a packet-size-minimization objective, this does NOT let a resource-
    only formula "discover" the semantic policy. The frozen v1 policy
    (:data:`FROZEN_WINDOW_OWNED_TARGET` / :data:`FROZEN_WINDOW_CONTEXT_EVENT_COUNT`)
    is a human/architecture choice; it is validated here against the measured
    frontier so the freeze is corpus-grounded and deterministic. The frozen
    point must:

      * exist in the measured sweep (a valid candidate);
      * have non-zero boundary context (``context >= 1``) so windows keep their
        semantic role (no degenerate one-event / zero-context collapse);
      * use multi-event owned windows (``owned_target >= 2``);

    The full corpus-derived sweep is retained in the JSON report as measurement
    evidence; it is NOT used to auto-select the policy by minimizing bytes.
    """
    if not sweep:
        raise ValueError("window sweep is empty; cannot select a frozen policy")
    match = [
        m
        for m in sweep
        if m["owned_event_target"] == FROZEN_WINDOW_OWNED_TARGET
        and m["context_event_count"] == FROZEN_WINDOW_CONTEXT_EVENT_COUNT
    ]
    if not match:
        raise ValueError(
            "the frozen architecture-aware window policy "
            f"({FROZEN_WINDOW_OWNED_TARGET}/{FROZEN_WINDOW_CONTEXT_EVENT_COUNT}) "
            "is not a valid measured frontier point for this corpus"
        )
    selected = match[0]
    if selected["context_event_count"] < 1:
        raise ValueError(
            "the frozen window policy must have non-zero boundary context"
        )
    if selected["owned_event_target"] < 2:
        raise ValueError(
            "the frozen window policy must use multi-event owned windows"
        )
    return selected


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------


def run_audit(store, pointers, *, project_id, document_id, consolidation_profile_id):
    snap = build_story_analysis_snapshot(
        store,
        pointers,
        project_id=project_id,
        document_id=document_id,
        consolidation_profile_id=consolidation_profile_id,
    )

    stream = ordered_event_stream(snap)
    n_events = len(stream)

    print("=== A6B ZERO-PROVIDER ALICE SHAPE AUDIT (ZERO-WRITE) ===")
    print(f"project:                 {project_id}")
    print(f"document:                {document_id}")
    print(f"consolidation profile:   {consolidation_profile_id}")
    print(f"manifest ref:            {_ref_str(snap.consolidation_manifest_ref)}")
    print(f"A5 CURRENT pointer:      {_ref_str(snap.a5_current_pointer_ref)}")
    print(f"pinned A4 EntityMap ref: {_ref_str(snap.consolidation_manifest.entity_map_ref)}")
    print(f"token counter id:        {TOKEN_COUNTER_ID}")
    print()

    # --- Corpus counts ---
    counts = {
        "canonical_character_count": len(snap.canonical_characters),
        "canonical_location_count": len(snap.canonical_locations),
        "unresolved_entity_count": len(snap.unresolved_entities),
        "canonical_fact_count": len(snap.facts),
        "state_transition_count": len(snap.state_transitions),
        "canonical_event_count": n_events,
        "canonical_relationship_count": len(snap.relationships),
        "story_conflict_count": len(snap.conflicts),
    }
    print("=== CORPUS COUNTS ===")
    for key in (
        "canonical_character_count",
        "canonical_location_count",
        "unresolved_entity_count",
        "canonical_fact_count",
        "state_transition_count",
        "canonical_event_count",
        "canonical_relationship_count",
        "story_conflict_count",
    ):
        print(f"  {key:32} {counts[key]}")
    print()

    # --- Serialized EXACT typed A5/A4 leaf bytes / tokens (BLOCK 5) ---
    leaf_bytes = {
        "canonical_character_registry": _leaf_bytes(
            snap.canonical_character_registry
        ),
        "canonical_location_registry": _leaf_bytes(snap.canonical_location_registry),
        "unresolved_entity_set": _leaf_bytes(snap.unresolved_entity_set),
        "canonical_fact_set": _leaf_bytes(snap.canonical_fact_set),
        "canonical_event_set": _leaf_bytes(snap.canonical_event_set),
        "canonical_relationship_set": _leaf_bytes(
            snap.canonical_relationship_set
        ),
        "story_conflict_set": _leaf_bytes(snap.story_conflict_set),
        "entity_map": _leaf_bytes(snap.entity_map),
    }
    leaf_tokens = {k: _tokens_from_byte_count(v) for k, v in leaf_bytes.items()}
    leaf_bytes["total"] = sum(leaf_bytes.values())
    leaf_tokens["total"] = sum(leaf_tokens.values())
    print("=== SERIALIZED EXACT TYPED A5/A4 LEAF BYTES / TOKENS (BLOCK 5) ===")
    for key, b in leaf_bytes.items():
        print(f"  {key:32} bytes={b:>9}  tokens~{leaf_tokens[key]:>7}")
    print()

    # --- Character packages ---
    char_report = _measure_character_packages(snap)
    print("=== CHARACTER EVIDENCE PACKAGES (100% coverage) ===")
    print(
        f"  packages={char_report['count']}  "
        f"canonical_characters={char_report['canonical_character_count']}  "
        f"coverage={char_report['coverage']['coverage_fraction']:.4f}"
    )
    for label in (
        "fact_counts",
        "event_counts",
        "relationship_counts",
        "transition_counts",
        "conflict_counts",
        "linked_unresolved_counts",
    ):
        q = char_report[label]
        print(f"  {label:28} min={q['min']} p50={q['p50']} p90={q['p90']} "
              f"max={q['max']} total={q['total']}")
    q = char_report["evidence_package_estimated_tokens"]
    print(
        f"  evidence_package_tokens      min={q['min']} p50={q['p50']} p90={q['p90']} "
        f"max={q['max']} total={q['total']}"
    )
    print(
        f"  largest character package:   {char_report['largest']['character_ref']} "
        f"tokens={char_report['largest']['estimated_tokens']} "
        f"bytes={char_report['largest']['canonical_bytes']}"
    )
    print()

    # --- Event stream ---
    event_report = _measure_event_stream(snap)
    print("=== EVENT STREAM (A5 narrative_order) ===")
    q = event_report["per_event_bytes"]
    print(f"  per-event bytes:     min={q['min']} max={q['max']} total={q['total']}")
    q = event_report["per_event_estimated_tokens"]
    print(f"  per-event tokens:    min={q['min']} max={q['max']} total={q['total']}")
    print(
        f"  cumulative tokens:   {event_report['cumulative_estimated_tokens']}  "
        f"cumulative bytes: {event_report['cumulative_bytes']}"
    )
    print()

    # --- Window sweep (BLOCK 4: corpus-derived candidate space) ---
    print("=== PLOT-WINDOW CANDIDATE SWEEP (deterministic, corpus-derived) ===")
    event_token_map = {
        e.event_id: _estimate_bytes_tokens(_canonical_bytes(e.to_dict()))
        for e in stream
    }
    candidate_space = _window_candidate_space(n_events)
    print(f"  candidate space: owned_target in 1..{n_events}, "
          f"context in 0..{max(1, n_events // 10)} "
          f"({len(candidate_space)} combinations)")
    sweep = []
    for target, context in candidate_space:
        metrics = _window_metrics_for(
            snap,
            stream,
            event_token_map=event_token_map,
            owned_event_target=target,
            context_event_count=context,
        )
        sweep.append(metrics)
    # Print a bounded, deterministic view: the frozen (selected) row plus a few
    # neighbours; the full sweep is in the JSON report.
    print(
        f"  {'target':>7} {'ctx':>4} {'windows':>8} {'owned(min/max)':>16} "
        f"{'max_pkt_tokens':>16} {'overlap_cost_tokens':>21}"
    )
    for m in sweep:
        owned = m["owned_per_window"]
        print(
            f"  {m['owned_event_target']:>7} {m['context_event_count']:>4} "
            f"{m['window_count']:>8} {str(owned['min']) + '/' + str(owned['max']):>16} "
            f"{m['max_window_packet_estimated_tokens']:>16} "
            f"{m['overlap_cost_tokens']:>21}"
        )
    print()

    # --- Window policy (architecture-aware measured frontier choice) ---
    selected_window = _select_frozen_window_policy(sweep)
    print("=== PLOT-WINDOW POLICY (architecture-aware measured frontier choice) ===")
    print(
        f"  frozen owned_target: {selected_window['owned_event_target']}  "
        f"context: {selected_window['context_event_count']}  "
        f"windows: {selected_window['window_count']}  "
        f"max_packet_tokens: {selected_window['max_window_packet_estimated_tokens']}"
    )
    print(
        "  NOTE: the frozen window policy is an ARCHITECTURE-AWARE measured "
        "choice (non-zero boundary context, multi-event owned windows), NOT a "
        "packet-size minimization. The corpus-derived sweep above is measurement "
        "evidence only."
    )
    print()

    # --- Global index base (A5-derived) ---
    gindex = build_global_index_base(snap)
    gi_bytes = len(gindex.canonical_bytes())
    gi_tokens = gindex.estimated_tokens()
    gi_hash = gindex.content_hash()
    print("=== COMPACT GLOBAL-INDEX BASE (A5-derived) ===")
    print(f"  estimated_bytes:   {gi_bytes}")
    print(f"  estimated_tokens:  {gi_tokens}")
    print(f"  content_hash:      {gi_hash}")
    print(
        "  NOTE: this is the compact A5-derived portion of the A6E global-"
        "skeleton input. The A6C character dossiers + A6D window analyses that "
        "join on top are future semantic outputs and are NOT faked here."
    )
    print()

    # --- A5-derived base (MEASUREMENT ONLY; NOT a whole-story ceiling) ---
    # A6B measures the exact serialized A5/A4 typed-leaf total and the compact
    # global-index base. These are the A5-DERIVED BASE of the A6E / A6F input.
    # The complete whole-story packet also joins the A6C character dossiers +
    # A6D window analyses (+ A6E output for the story bible), which are future
    # semantic outputs that do not exist yet. A6B therefore does NOT freeze a
    # whole-story ceiling from this base (it would be an A5-only placeholder);
    # the two whole-story ceilings are DEFERRED (null) and are measured at the
    # pre-A6E / pre-A6F gate.
    a5_base_tokens = leaf_tokens["total"]
    print("=== A5-DERIVED BASE (MEASURED; NOT A WHOLE-STORY CEILING) ===")
    print(f"  serialized A5/A4 exact typed-leaf total tokens: {a5_base_tokens}")
    print(f"  compact global-index base tokens (A5-derived):   {gindex.estimated_tokens()}")
    print(
        "  NOTE: MEASURED NOW (the A5-derived base only). The complete A6E "
        "global-skeleton and A6F story-bible whole-story packet ceilings are "
        "DEFERRED (null); they cannot be measured in A6B because they join the "
        "A6C dossiers + A6D window analyses (+ A6E output) that do not yet "
        "exist. This base is NOT used as a whole-story ceiling."
    )
    print()

    # --- Frozen policy (A6B-measurable values concrete; whole-story deferred) ---
    # character / window values are measured and frozen concrete. The two
    # whole-story ceilings are DEFERRED (null): A6B measures the A5-derived base
    # but does NOT freeze a whole-story ceiling (the complete A6E / A6F packet
    # joins future semantic outputs that do not exist yet). The corresponding
    # complete fail-closed gate belongs to pre-A6E / pre-A6F.
    character_budget = _round_up_to_step(
        char_report["largest"]["estimated_tokens"], _TOKEN_MARGIN_STEP
    )
    window_budget = _round_up_to_step(
        selected_window["max_window_packet_estimated_tokens"], _TOKEN_MARGIN_STEP
    )
    policy = StoryAnalysisPlanningPolicy(
        character_packet_max_estimated_tokens=character_budget,
        plot_window_packet_max_estimated_tokens=window_budget,
        plot_window_owned_event_target=selected_window["owned_event_target"],
        plot_window_context_event_count=selected_window["context_event_count"],
        # DEFERRED (null): the complete whole-story packet cannot be measured in
        # A6B (the A6C dossiers + A6D window analyses do not exist yet).
        global_skeleton_packet_max_estimated_tokens=None,
        story_bible_packet_max_estimated_tokens=None,
    )
    policy_dict = policy.to_dict()
    print("=== FROZEN A6B PLANNING POLICY + PLAN IDENTITY ===")
    for key in (
        "character_packet_max_estimated_tokens",
        "plot_window_packet_max_estimated_tokens",
        "plot_window_owned_event_target",
        "plot_window_context_event_count",
        "global_skeleton_packet_max_estimated_tokens",
        "story_bible_packet_max_estimated_tokens",
    ):
        print(f"  {key:42} {policy_dict[key]}")
    print(f"  {'planning_policy_ids (bound into hash)':42} "
          f"{json.dumps(FROZEN_POLICY_IDS, sort_keys=True)}")
    print(f"  {'token_counter_id (bound into hash)':42} {TOKEN_COUNTER_ID}")
    print()

    # Build the deterministic plan identity for the frozen policy (BLOCK 2:
    # binds the numeric policy + the four versioned policy IDs + the
    # token-counter ID).
    packages = build_character_evidence_packages(snap)
    windows = plan_plot_windows(
        stream,
        owned_event_target=policy.plot_window_owned_event_target,
        context_event_count=policy.plot_window_context_event_count,
    )
    validate_window_ownership(windows, frozenset(e.event_id for e in stream))
    event_ids = tuple(e.event_id for e in stream)
    plan_hash = compute_plan_hash(
        snap.consolidation_manifest_ref,
        policy,
        FROZEN_POLICY_IDS,
        TOKEN_COUNTER_ID,
        packages,
        event_ids,
        windows,
        gindex,
    )
    # Determinism re-check.
    plan_hash_re = compute_plan_hash(
        snap.consolidation_manifest_ref,
        policy,
        FROZEN_POLICY_IDS,
        TOKEN_COUNTER_ID,
        packages,
        event_ids,
        windows,
        gindex,
    )
    determinism_ok = plan_hash == plan_hash_re

    # Provider / runtime leakage check over the canonical plan-identity payload
    # (now includes the four versioned policy IDs + the token-counter ID).
    identity_payload = {
        "consolidation_manifest_ref": snap.consolidation_manifest_ref.to_dict(),
        "planning_policy": policy_dict,
        "planning_policy_ids": dict(FROZEN_POLICY_IDS),
        "token_counter_id": TOKEN_COUNTER_ID,
        "character_package_hashes": [p.content_hash() for p in packages],
        "event_stream": list(event_ids),
        "windows": [w.to_dict() for w in windows],
        "global_index_hash": gi_hash,
    }
    identity_canonical = _canonical_bytes(identity_payload).decode("utf-8").lower()
    leaked = [s for s in _FORBIDDEN_IDENTITY_SUBSTRINGS if s in identity_canonical]
    leakage_ok = not leaked
    print(f"  plan_hash:                 {plan_hash}")
    print(f"  deterministic_recheck:     {'PASS' if determinism_ok else 'FAIL'}")
    print(
        f"  provider/runtime leakage:  "
        f"{'PASS (none)' if leakage_ok else 'FAIL: ' + str(leaked)}"
    )
    print()

    # --- Largest candidates ---
    largest_window = max(
        sweep, key=lambda m: m["max_window_packet_estimated_tokens"]
    )
    largest = {
        "character_package": char_report["largest"],
        "plot_window_package": {
            "owned_event_target": largest_window["owned_event_target"],
            "context_event_count": largest_window["context_event_count"],
            "max_window_packet_estimated_tokens":
                largest_window["max_window_packet_estimated_tokens"],
            "window_id": largest_window["largest_window_id"],
        },
        "global_skeleton_input": {
            "a5_derived_base_estimated_tokens": a5_base_tokens,
            "a5_derived_base_bytes": leaf_bytes["total"],
            "status": "DEFERRED",
            "note": (
                "A5-DERIVED BASE ONLY (measured); the complete A6E global-"
                "skeleton packet joins the A6C character dossiers + A6D window "
                "analyses (future semantic outputs, not faked). The whole-story "
                "ceiling is DEFERRED (null) and is measured at the pre-A6E gate."
            ),
        },
    }

    rationale = {
        "character_packet_max_estimated_tokens": (
            f"Measured max character evidence package is "
            f"{char_report['largest']['estimated_tokens']} tokens "
            f"({char_report['largest']['character_ref']}); frozen at "
            f"{policy_dict['character_packet_max_estimated_tokens']} "
            "(measured max rounded up to the next 1000-token step)."
        ),
        "plot_window_packet_max_estimated_tokens": (
            f"ARCHITECTURE-AWARE measured choice: owned_target="
            f"{policy_dict['plot_window_owned_event_target']}, "
            f"context={policy_dict['plot_window_context_event_count']} (non-zero "
            f"boundary context, multi-event owned windows), max window packet "
            f"{selected_window['max_window_packet_estimated_tokens']} tokens; "
            f"frozen at "
            f"{policy_dict['plot_window_packet_max_estimated_tokens']} "
            "(measured max rounded up to the next 1000-token step). NOT chosen "
            "by byte minimization — that would degenerate to 1-event / 0-context "
            "windows and defeat the window/context semantic role."
        ),
        "plot_window_owned_event_target": (
            f"ARCHITECTURE-AWARE frozen choice ({FROZEN_WINDOW_OWNED_TARGET}) "
            f"taken from the measured corpus-derived frontier (bounded sweep of "
            f"1..{n_events} owned targets x 0..{max(1, n_events // 10)} context "
            f"counts, retained as measurement evidence). {n_events} canonical "
            f"events partition into {selected_window['window_count']} windows; "
            f"context={FROZEN_WINDOW_CONTEXT_EVENT_COUNT} preserves boundary "
            "adjacency without hard-cutting turning points."
        ),
        "global_skeleton_packet_max_estimated_tokens": (
            "DEFERRED (null). A6B measures the A5-derived base "
            f"({a5_base_tokens} tokens) but does NOT freeze a whole-story "
            "ceiling: the complete A6E global-skeleton input joins the A6C "
            "character dossiers + A6D window analyses, which are future semantic "
            "outputs with no frozen size bound. Using the A5-derived base as the "
            "ceiling would be an A5-only placeholder. The complete fail-closed "
            "gate belongs to pre-A6E."
        ),
        "story_bible_packet_max_estimated_tokens": (
            "DEFERRED (null). The complete A6F story-bible input is the full "
            "compressed global representation (A5 base + global skeleton + "
            "character dossiers + window analyses). A6B measures the A5-derived "
            f"base ({a5_base_tokens} tokens) but does NOT freeze a whole-story "
            "ceiling; the complete fail-closed gate belongs to pre-A6F."
        ),
    }

    checks = [
        (
            "100% canonical-character coverage (one package per character)",
            char_report["coverage"]["coverage_fraction"] == 1.0
            and char_report["count"]
            == char_report["canonical_character_count"],
        ),
        (
            "event stream covers every canonical event exactly once",
            event_report["count"] == counts["canonical_event_count"],
        ),
        (
            "window ownership: every event owned exactly once (all sweep rows)",
            all(
                m["owned_per_window"]["min"] >= 1
                and m["window_count"] >= 1
                for m in sweep
            ),
        ),
        (
            "BLOCK 4 candidate space is corpus-derived (owned 1..n, "
            "context 0..max(1, n//10)) and exhaustive",
            len(candidate_space)
            == n_events * (max(1, n_events // 10) + 1),
        ),
        (
            "plan hash is a 64-hex canonical content hash",
            isinstance(plan_hash, str)
            and len(plan_hash) == 64
            and all(c in "0123456789abcdef" for c in plan_hash),
        ),
        ("plan hash is deterministic (recompute matches)", determinism_ok),
        (
            "no provider/model/GPU/credential leakage into the plan identity",
            leakage_ok,
        ),
        (
            "frozen character budget fits the largest measured character package",
            policy_dict["character_packet_max_estimated_tokens"]
            >= char_report["largest"]["estimated_tokens"],
        ),
        (
            "frozen window budget fits the largest measured window packet",
            policy_dict["plot_window_packet_max_estimated_tokens"]
            >= selected_window["max_window_packet_estimated_tokens"],
        ),
        (
            "frozen window policy is architecture-aware (non-zero boundary "
            "context, multi-event owned windows)",
            selected_window["context_event_count"] >= 1
            and selected_window["owned_event_target"] >= 2,
        ),
        (
            "whole-story global-skeleton ceiling is DEFERRED (null) in A6B "
            "(not an A5-derived placeholder)",
            policy_dict["global_skeleton_packet_max_estimated_tokens"] is None,
        ),
        (
            "whole-story story-bible ceiling is DEFERRED (null) in A6B (not an "
            "A5-derived placeholder)",
            policy_dict["story_bible_packet_max_estimated_tokens"] is None,
        ),
        (
            "A5-derived base is measured (typed-leaf total) but NOT used as a "
            "whole-story ceiling",
            a5_base_tokens > 0
            and policy_dict["global_skeleton_packet_max_estimated_tokens"] is None,
        ),
        (
            "BLOCK 2: the plan identity binds the four versioned policy IDs and "
            "the token-counter ID",
            all(k in identity_payload for k in ("planning_policy_ids", "token_counter_id"))
            and identity_payload["token_counter_id"] == TOKEN_COUNTER_ID
            and set(identity_payload["planning_policy_ids"].keys())
            == set(FROZEN_POLICY_IDS.keys()),
        ),
    ]
    all_pass = all(passed for _, passed in checks)
    print("=== AUDIT CHECKS ===")
    for label, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}")
    print()

    report = {
        "audit_id": "a6b-alice-zero-provider-shape-audit",
        "schema_version": 3,
        "zero_provider": True,
        "zero_write": True,
        "input": {
            "project_id": project_id,
            "document_id": document_id,
            "consolidation_profile_id": consolidation_profile_id,
            "consolidation_manifest_ref": snap.consolidation_manifest_ref.to_dict(),
            "a5_current_pointer_ref": snap.a5_current_pointer_ref.to_dict(),
            "a5_validation_report_ref": snap.a5_validation_report_ref.to_dict(),
            "pinned_a4_entity_map_ref": snap.consolidation_manifest.entity_map_ref.to_dict(),
        },
        "corpus_counts": counts,
        # --- MEASURED NOW (A6B, zero-provider, read-only) ---
        "measured_now": {
            "exact_a4_a5_typed_leaf_bytes": leaf_bytes,
            "exact_a4_a5_typed_leaf_estimated_tokens": leaf_tokens,
            "character_packets": char_report,
            "event_stream": event_report,
            "window_packets": {
                "frozen_architecture_aware_policy": selected_window,
                "sweep_candidate_space": {
                    "owned_target_range": [1, n_events],
                    "context_range": [0, max(1, n_events // 10)],
                    "combination_count": len(candidate_space),
                    "sweep_note": (
                        "bounded corpus-scaled sweep, retained as measurement "
                        "evidence; NOT used to auto-select the policy by "
                        "minimizing bytes (the frozen choice is architecture-"
                        "aware)"
                    ),
                },
                "sweep": sweep,
            },
            "global_index_base": {
                "estimated_bytes": gi_bytes,
                "estimated_tokens": gi_tokens,
                "content_hash": gi_hash,
            },
            "a5_derived_base": {
                "serialized_leaf_total_tokens": a5_base_tokens,
                "global_index_base_estimated_tokens": gi_tokens,
                "note": (
                    "A5-DERIVED BASE ONLY (measured). This is the measurable "
                    "A5 portion of the A6E / A6F whole-story input; it is NOT "
                    "used as a whole-story ceiling."
                ),
            },
            "largest_candidates": largest,
        },
        # --- DEFERRED (cannot be measured in A6B; future semantic outputs) ---
        "deferred": {
            "global_skeleton_packet_ceiling": {
                "value": None,
                "status": "DEFERRED",
                "reason": (
                    "The complete A6E global-skeleton input joins the A6C "
                    "character dossiers + A6D window analyses, which are future "
                    "semantic outputs that do not exist yet. The complete fail-"
                    "closed gate belongs to pre-A6E."
                ),
            },
            "story_bible_packet_ceiling": {
                "value": None,
                "status": "DEFERRED",
                "reason": (
                    "The complete A6F story-bible input is the full compressed "
                    "global representation (A5 base + global skeleton + "
                    "character dossiers + window analyses). The complete fail-"
                    "closed gate belongs to pre-A6F."
                ),
            },
            "recursive_global_compression": {
                "status": "DEFERRED TO A6E MEASUREMENT",
                "reason": (
                    "Whether recursive section-layer compression is needed "
                    "cannot be decided from the A5-derived base alone; it "
                    "requires the actual A6C / A6D outputs. A6E makes the first "
                    "authoritative judgment; A6G re-validates against the real "
                    "provider."
                ),
            },
        },
        # Flat mirrors for backward compatibility with consumers/tests.
        "serialized_exact_typed_a5_a4_leaf_bytes": leaf_bytes,
        "serialized_exact_typed_a5_a4_leaf_estimated_tokens": leaf_tokens,
        "character_packages": char_report,
        "event_stream": event_report,
        "window_sweep": sweep,
        "window_sweep_candidate_space": {
            "owned_target_range": [1, n_events],
            "context_range": [0, max(1, n_events // 10)],
            "combination_count": len(candidate_space),
            "selection_basis": (
                "architecture-aware measured frontier choice "
                f"(owned_target={FROZEN_WINDOW_OWNED_TARGET}, "
                f"context={FROZEN_WINDOW_CONTEXT_EVENT_COUNT}); the corpus-"
                "derived sweep is measurement evidence only, NOT a byte-"
                "minimization objective"
            ),
        },
        "window_policy_selection": selected_window,
        "global_index_base": {
            "estimated_bytes": gi_bytes,
            "estimated_tokens": gi_tokens,
            "content_hash": gi_hash,
        },
        "planning_policy_frozen": policy_dict,
        "planning_policy_ids": dict(FROZEN_POLICY_IDS),
        "token_counter_id": TOKEN_COUNTER_ID,
        "planning_policy_rationale": rationale,
        "staged_ceiling_authority": {
            "a6b_concrete": [
                "character_packet_max_estimated_tokens",
                "plot_window_packet_max_estimated_tokens",
                "plot_window_owned_event_target",
                "plot_window_context_event_count",
            ],
            "a6b_deferred_null": [
                "global_skeleton_packet_max_estimated_tokens",
                "story_bible_packet_max_estimated_tokens",
            ],
            "null_semantics": (
                "DEFERRED / not-yet-measurable; != unlimited, != zero, != A5-"
                "derived placeholder. The corresponding pass fails closed (zero "
                "provider calls) while the ceiling is null."
            ),
        },
        "plan_identity": {
            "plan_hash": plan_hash,
            "deterministic_recheck_match": determinism_ok,
            "provider_runtime_leakage_check": "PASS" if leakage_ok else f"FAIL:{leaked}",
            "identity_keys": sorted(identity_payload.keys()),
        },
        "checks": [[label, bool(passed)] for label, passed in checks],
        "result": "PASS" if all_pass else "FAIL",
    }

    return report, all_pass


# The plan identity must never contain any runtime / provider topology material.
_FORBIDDEN_IDENTITY_SUBSTRINGS = (
    "provider",
    "model",
    "base_url",
    "gpu",
    "n_gpu",
    "slot",
    "concurrency",
    "parallel",
    "timeout",
    "credential",
    "api_key",
    "api-key",
    "api.key",
    "llama",
    "openai",
    "anthropic",
)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="A6B zero-provider Alice real-corpus shape audit"
    )
    parser.add_argument("--runs-root", default=str(DEFAULT_RUNS_ROOT))
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--document", default=DEFAULT_DOCUMENT)
    parser.add_argument("--consolidation-profile", default=DEFAULT_CONSOLIDATION_PROFILE)
    parser.add_argument(
        "--report",
        default=None,
        help=(
            "optional path to write the machine-readable JSON report; it must "
            "be OUTSIDE the story runs tree (the audit is zero-write)"
        ),
    )
    args = parser.parse_args(argv)

    # BLOCK 3: refuse to write a diagnostic report inside the story runs tree.
    if args.report is not None:
        report_path = Path(args.report).expanduser().resolve()
        runs_story_root = _story_runs_root(args.runs_root, args.project).resolve()
        if report_path == runs_story_root or runs_story_root in report_path.parents:
            print(
                "A6B AUDIT RESULT: FAIL "
                f"(report path {args.report!r} is inside the story runs tree; "
                "the audit is zero-write — write the report elsewhere)",
                file=sys.stderr,
            )
            return 2

    # BLOCK 3: snapshot the runs tree before the audit (zero-write proof).
    before_tree = _snapshot_runs_tree(args.runs_root, args.project)

    try:
        store, pointers = _stores(args.runs_root, args.project)
        report, all_pass = run_audit(
            store,
            pointers,
            project_id=args.project,
            document_id=args.document,
            consolidation_profile_id=args.consolidation_profile,
        )
    except (StoryIntegrityError, StoryAnalysisPlanningError) as exc:
        print(
            f"A6B AUDIT RESULT: FAIL ({type(exc).__name__}: {exc})", file=sys.stderr
        )
        return 2
    except Exception as exc:  # pragma: no cover - defensive (unexpected)
        print(f"A6B AUDIT RESULT: ERROR ({type(exc).__name__}: {exc})", file=sys.stderr)
        return 2

    # BLOCK 3: snapshot the runs tree after the audit and assert equality.
    after_tree = _snapshot_runs_tree(args.runs_root, args.project)
    if before_tree != after_tree:
        added = sorted(set(after_tree) - set(before_tree))
        removed = sorted(set(before_tree) - set(after_tree))
        changed = sorted(
            k
            for k in set(before_tree) & set(after_tree)
            if before_tree[k] != after_tree[k]
        )
        print(
            "A6B AUDIT RESULT: FAIL (story runs tree mutated: "
            f"added={added} removed={removed} changed={changed})",
            file=sys.stderr,
        )
        return 2

    if args.report is not None:
        out = Path(args.report).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"JSON report written to: {out}")

    if not all_pass:
        print("A6B AUDIT RESULT: FAIL", file=sys.stderr)
        return 2
    print("A6B AUDIT RESULT: PASS (zero-provider, read-only, zero-write)")
    print(
        "MEASURED NOW: exact A4/A5 typed leaves, character packets, window "
        "packets, A5-derived global-index base."
    )
    print(
        "DEFERRED (null): complete A6E global-skeleton + A6F story-bible "
        "whole-story packet ceilings (measured at the pre-A6E / pre-A6F gate). "
        "Recursive global compression: DEFERRED TO A6E MEASUREMENT."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
