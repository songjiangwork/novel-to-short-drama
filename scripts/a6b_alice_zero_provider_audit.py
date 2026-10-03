"""v1.2 A6B — zero-provider Alice real-corpus shape audit (issue #87).

This is the A6B **audit gate**: a read-only, zero-provider measurement of the
real Alice A5F2 corpus through the exact validated A5 CURRENT snapshot, BEFORE
any numeric planning policy is frozen and BEFORE the production
``profiles/global_story_analysis_v1.yaml`` is created.

The audit:

  * reads the exact current-eligible A5 CURRENT (via
    ``ConsolidationPersistenceService.require_current_validated``) and the exact
    pinned A4/A5 leaves into a read-only ``StoryAnalysisInputSnapshot``;
  * reports the corpus counts and the serialized / estimated-token sizes of the
    A5/A4 leaves;
  * reports the complete deterministic character evidence-package plan (100%
    coverage, per-character evidence counts and ``p50 / p90 / max`` package
    sizes);
  * reports the event stream (per-event bytes / tokens, cumulative) and a
    bounded, corpus-derived deterministic sweep of candidate plot-window
    owned-target / context counts (window distributions + overlap costs);
  * reports the compact A5-derived global-index base bytes / tokens (the
    measurable A5-derived portion of the A6E global-skeleton input; the A6C
    character dossiers + A6D window analyses that join on top are future
    semantic outputs and are NOT faked here);
  * builds the deterministic A6 planning identity / plan hash for the frozen
    policy and FAILS CLOSED on any provider / model / GPU / credential leakage
    into the plan identity (and proves determinism by recomputing the hash);
  * reports the largest candidate packages (character / plot-window / global
    skeleton input) that the frozen numeric policy must fit.

It performs NO provider call, writes NO A6 artifact, modifies NO A5/A6 CURRENT,
and modifies NO tracked profile. An optional ``--report PATH`` writes a
machine-readable JSON report (a diagnostic, not an artifact).

Usage:
    python scripts/a6b_alice_zero_provider_audit.py \
        [--runs-root PATH] [--project ID] [--document ID] \
        [--consolidation-profile ID] [--report PATH]

Exit code 0 on a clean audit, 2 on any structural / integrity / leakage failure.
"""

from __future__ import annotations

import argparse
import json
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

# --- Frozen A6B numeric planning policy (measured from the Alice corpus). ---
# character_packet_max_estimated_tokens: measured max character evidence package
#   (the protagonist package, 69550 tokens) rounded up with a small margin.
# plot_window_packet_max_estimated_tokens: measured max window packet at the
#   frozen owned-target / context (57360 tokens at 12/4) rounded up.
# plot_window_owned_event_target / context_event_count: corpus-derived choice
#   balancing window count (14 windows), packet size, and boundary overlap.
# global_skeleton / story_bible budgets: the measurable A5-derived global-index
#   base (21940 tokens) plus a documented compact semantic-layer bound for the
#   A6C character dossiers + A6D window analyses (NOT a fake of the future
#   semantic output size, and NOT set from the 262K theoretical context). See
#   planning_policy_rationale in the report.
# Documented compact semantic-layer bounds (per-item, used only to size the
#   two whole-story budgets):
_DOSSIER_COMPACT_BOUND = 2000        # per character dossier (compressed A6C output)
_WINDOW_ANALYSIS_COMPACT_BOUND = 2000  # per plot-window analysis (A6D output)
_GLOBAL_STRUCTURE_COMPACT_BOUND = 32000  # compact GlobalStructure (A6E output)
FROZEN_POLICY = StoryAnalysisPlanningPolicy(
    character_packet_max_estimated_tokens=72000,
    plot_window_packet_max_estimated_tokens=58000,
    plot_window_owned_event_target=12,
    plot_window_context_event_count=4,
    global_skeleton_packet_max_estimated_tokens=160000,
    story_bible_packet_max_estimated_tokens=192000,
)

# Deterministic bounded candidate sweep (corpus-derived, not hardware-guessed).
# owned targets are round numbers up to the full event universe; context counts
# are a deterministic set. The frozen choice (12, 4) is always included.
_CONTEXT_CANDIDATES = (0, 2, 4, 8, 12)
_OWNED_TARGET_CANDIDATES = (4, 6, 8, 10, 12, 14, 16, 20, 24, 32, 48, 64, 96)

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
# Small deterministic helpers
# ---------------------------------------------------------------------------


def _stores(runs_root: str | Path, project_id: str):
    root = Path(runs_root).expanduser() / project_id / "story"
    artifact_store = FileArtifactStore(root / "artifacts")
    pointer_store = FilePointerStore(root / "pointers", artifact_store)
    return artifact_store, pointer_store


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
        import math

        return s[max(0, min(n - 1, math.ceil(q * n) - 1))]

    return {
        "min": s[0],
        "p50": _rank(0.50),
        "p90": _rank(0.90),
        "max": s[-1],
        "total": sum(s),
    }


def _leaf_bytes(items) -> int:
    return len(_canonical_bytes([it.to_dict() for it in items]))


def _tokens_from_byte_count(count: int) -> int:
    """Reuse the ``utf8-bytes-div3-v1`` estimator over an already-measured byte
    count (``max(1, ceil(bytes / 3))``) without re-encoding."""
    import math

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

    # Candidate owned-target set, bounded to the corpus event universe.
    owned_candidates = sorted(
        {t for t in _OWNED_TARGET_CANDIDATES if 1 <= t <= n_events} | {n_events, 1}
    )

    print("=== A6B ZERO-PROVIDER ALICE SHAPE AUDIT ===")
    print(f"project:                 {project_id}")
    print(f"document:                {document_id}")
    print(f"consolidation profile:   {consolidation_profile_id}")
    print(f"manifest ref:            {_ref_str(snap.consolidation_manifest_ref)}")
    print(f"A5 CURRENT pointer:      {_ref_str(snap.a5_current_pointer_ref)}")
    print(f"pinned A4 EntityMap ref: {_ref_str(snap.consolidation_manifest.entity_map_ref)}")
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

    # --- Serialized A5/A4 leaf bytes ---
    leaf_bytes = {
        "canonical_fact_set": _leaf_bytes(snap.facts),
        "state_transitions": _leaf_bytes(snap.state_transitions),
        "canonical_event_set": _leaf_bytes(snap.events),
        "canonical_relationship_set": _leaf_bytes(snap.relationships),
        "story_conflict_set": _leaf_bytes(snap.conflicts),
        "canonical_character_registry": _leaf_bytes(snap.canonical_characters),
        "canonical_location_registry": _leaf_bytes(snap.canonical_locations),
        "unresolved_entity_set": _leaf_bytes(snap.unresolved_entities),
    }
    leaf_tokens = {k: _tokens_from_byte_count(v) for k, v in leaf_bytes.items()}
    leaf_bytes["total"] = sum(leaf_bytes.values())
    leaf_tokens["total"] = sum(leaf_tokens.values())
    print("=== SERIALIZED A5/A4 LEAF BYTES / TOKENS ===")
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

    # --- Window sweep ---
    print("=== PLOT-WINDOW CANDIDATE SWEEP (deterministic, corpus-derived) ===")
    event_token_map = {
        e.event_id: _estimate_bytes_tokens(_canonical_bytes(e.to_dict()))
        for e in stream
    }
    sweep = []
    for target in owned_candidates:
        for ctx in _CONTEXT_CANDIDATES:
            metrics = _window_metrics_for(
                snap,
                stream,
                event_token_map=event_token_map,
                owned_event_target=target,
                context_event_count=ctx,
            )
            sweep.append(metrics)
    # Print a bounded, deterministic view: for each target, the frozen context and
    # a couple of neighbours; the full sweep is in the JSON report.
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
        "  NOTE: this is the measurable A5-derived portion of the A6E global-"
        "skeleton input. The A6C character dossiers + A6D window analyses that "
        "join on top are future semantic outputs and are NOT faked here."
    )
    print()

    # --- Frozen policy + plan identity ---
    print("=== FROZEN A6B PLANNING POLICY + PLAN IDENTITY ===")
    policy = FROZEN_POLICY
    policy_dict = policy.to_dict()
    for key in (
        "character_packet_max_estimated_tokens",
        "plot_window_packet_max_estimated_tokens",
        "plot_window_owned_event_target",
        "plot_window_context_event_count",
        "global_skeleton_packet_max_estimated_tokens",
        "story_bible_packet_max_estimated_tokens",
    ):
        print(f"  {key:42} {policy_dict[key]}")
    print()

    # Build the deterministic plan identity for the frozen policy.
    packages = build_character_evidence_packages(snap)
    windows = plan_plot_windows(
        stream,
        owned_event_target=policy.plot_window_owned_event_target,
        context_event_count=policy.plot_window_context_event_count,
    )
    validate_window_ownership(windows, frozenset(e.event_id for e in stream))
    event_ids = tuple(e.event_id for e in stream)
    plan_hash = compute_plan_hash(
        snap.consolidation_manifest_ref, policy, packages, event_ids, windows, gindex
    )
    # Determinism re-check.
    plan_hash_re = compute_plan_hash(
        snap.consolidation_manifest_ref, policy, packages, event_ids, windows, gindex
    )
    determinism_ok = plan_hash == plan_hash_re

    # Provider / runtime leakage check over the canonical plan-identity payload.
    identity_payload = {
        "consolidation_manifest_ref": snap.consolidation_manifest_ref.to_dict(),
        "planning_policy": policy_dict,
        "character_package_hashes": [p.content_hash() for p in packages],
        "event_stream": list(event_ids),
        "windows": [w.to_dict() for w in windows],
        "global_index_hash": gi_hash,
    }
    identity_canonical = _canonical_bytes(identity_payload).decode("utf-8").lower()
    leaked = [
        s
        for s in _FORBIDDEN_IDENTITY_SUBSTRINGS
        if s in identity_canonical
    ]
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
            "a5_derived_base_estimated_tokens": gi_tokens,
            "a5_derived_base_bytes": gi_bytes,
            "note": (
                "A5-derived base only; A6C dossiers + A6D window analyses are "
                "future semantic outputs joined on top (not faked)."
            ),
        },
    }

    rationale = {
        "character_packet_max_estimated_tokens": (
            f"Measured max character evidence package is "
            f"{char_report['largest']['estimated_tokens']} tokens "
            f"({char_report['largest']['character_ref']}); frozen at "
            f"{policy_dict['character_packet_max_estimated_tokens']} "
            "(small margin above the measured max)."
        ),
        "plot_window_packet_max_estimated_tokens": (
            f"Measured max window packet at the frozen owned-target "
            f"{policy_dict['plot_window_owned_event_target']} / context "
            f"{policy_dict['plot_window_context_event_count']} is "
            f"{_max_window_tokens_at(snap, stream, policy)} tokens; frozen at "
            f"{policy_dict['plot_window_packet_max_estimated_tokens']}."
        ),
        "plot_window_owned_event_target": (
            f"Corpus-derived: {n_events} canonical events partition into "
            f"{len(windows)} windows (owned target "
            f"{policy_dict['plot_window_owned_event_target']}, context "
            f"{policy_dict['plot_window_context_event_count']}); keeps packets "
            "under budget while preserving boundary overlap."
        ),
        "global_skeleton_packet_max_estimated_tokens": (
            f"A5-derived global-index base is {gi_tokens} tokens (measured). "
            "The A6E global-skeleton input also joins the compact A6C character "
            f"dossiers ({counts['canonical_character_count']} x "
            f"{_DOSSIER_COMPACT_BOUND}) and A6D window analyses ({len(windows)} x "
            f"{_WINDOW_ANALYSIS_COMPACT_BOUND}); the budget of "
            f"{policy_dict['global_skeleton_packet_max_estimated_tokens']} = "
            f"base + compact semantic layer bound + margin, well under the 262K "
            "context (NOT set from the 262K theoretical context, and NOT a fake "
            "of the future semantic output size)."
        ),
        "story_bible_packet_max_estimated_tokens": (
            "The A6F story-bible synthesis input is the complete compressed "
            "global representation (global skeleton + character dossiers + "
            f"window analyses + compact GlobalStructure ({_GLOBAL_STRUCTURE_COMPACT_BOUND})); "
            "the budget of "
            f"{policy_dict['story_bible_packet_max_estimated_tokens']} is the "
            "largest whole-story packet, sized from the A5-derived base plus the "
            "compact semantic layers, well under the 262K context."
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
            >= _max_window_tokens_at(snap, stream, policy),
        ),
        (
            "frozen global-skeleton budget fits the A5-derived base",
            policy_dict["global_skeleton_packet_max_estimated_tokens"] >= gi_tokens,
        ),
        (
            "frozen story-bible budget is the largest whole-story packet",
            policy_dict["story_bible_packet_max_estimated_tokens"]
            >= policy_dict["global_skeleton_packet_max_estimated_tokens"],
        ),
    ]
    all_pass = all(passed for _, passed in checks)
    print("=== AUDIT CHECKS ===")
    for label, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}")
    print()

    report = {
        "audit_id": "a6b-alice-zero-provider-shape-audit",
        "schema_version": 1,
        "zero_provider": True,
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
        "serialized_a5_a4_leaf_bytes": leaf_bytes,
        "serialized_a5_a4_leaf_estimated_tokens": leaf_tokens,
        "character_packages": char_report,
        "event_stream": event_report,
        "window_sweep": sweep,
        "global_index_base": {
            "estimated_bytes": gi_bytes,
            "estimated_tokens": gi_tokens,
            "content_hash": gi_hash,
        },
        "largest_candidates": largest,
        "planning_policy_frozen": policy_dict,
        "planning_policy_rationale": rationale,
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


def _max_window_tokens_at(snap, stream, policy) -> int:
    windows = plan_plot_windows(
        stream,
        owned_event_target=policy.plot_window_owned_event_target,
        context_event_count=policy.plot_window_context_event_count,
    )
    return max(
        build_window_packet(w, snap).estimated_tokens() for w in windows
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
        help="optional path to write the machine-readable JSON report",
    )
    args = parser.parse_args(argv)

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

    if args.report is not None:
        out = Path(args.report).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"JSON report written to: {out}")

    if not all_pass:
        print("A6B AUDIT RESULT: FAIL", file=sys.stderr)
        return 2
    print("A6B AUDIT RESULT: PASS (zero-provider, read-only)")
    print("Measurements collected; numeric planning policy is corpus-derived.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
