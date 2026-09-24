"""v1.2 A5C-A -- deterministic block-packing + request-shape audit (zero provider).

Covers A5C-A BLOCK 9 / 10 (the audit side) implemented in
``short_drama.story.consolidation_semantic`` +
``scripts/a5c_fact_request_shape_audit.py``:

  * the three deterministic block-packing candidates (P1 6 pairs / 12
    candidates, P2 12 / 24, P3 24 / 48) -- audit ONLY, NOT frozen -- each
    reporting a full deterministic distribution (min / median / p95 / max) for
    block pair count, unique candidate count, evidence items, pair-context bytes
    and rendered-prompt bytes, plus total pair / request-hash counts and a
    LARGEST-BLOCK diagnostic (block id, pair count, candidate count, bytes) to
    expose one dominant outlier;
  * the Alice zero-provider request-shape smoke test (exact acceptance gates,
    read-only, deterministic): the Alice corpus yields exactly 2113 fact
    semantic pairs, 0 auto_same, and the two-limit packing gives P1 = 353,
    P2 = 177, P3 = 89 blocks;
  * the script is zero-provider (no transport) and read-only (run tree
    fingerprint unchanged) and deterministic.

All fixtures are self-contained synthetic run trees; the Alice smoke test is
skipped when the local (gitignored) run tree is absent. No provider is called
and nothing is persisted anywhere in this file.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest

from short_drama.artifacts import FileArtifactStore
from short_drama.foundation import FilePointerStore
from short_drama.story import (
    A5C_PACKING_CANDIDATES,
    DEFAULT_PROMPT_BASE_DIR,
    FactSemanticPackingPolicy,
    build_consolidation_planning,
    build_fact_packing_audit,
    build_fact_semantic_preparation,
    load_consolidation_profile,
    load_fact_semantic_profile,
)
from short_drama.llm import PromptRegistry
from test_story_a5c_fact_preparation import _PROMPTS, _SEM_PROFILE, _planning, _tree
from test_story_a5b_planning import _consolidation_profile

REPO_ROOT = Path(__file__).resolve().parents[1]
_AUDIT_PATH = REPO_ROOT / "scripts" / "a5c_fact_request_shape_audit.py"
_spec = importlib.util.spec_from_file_location("a5c_fact_request_shape_audit", _AUDIT_PATH)
audit = importlib.util.module_from_spec(_spec)
sys.modules["a5c_fact_request_shape_audit"] = audit
_spec.loader.exec_module(audit)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _fingerprint_dir(root: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not root.exists():
        return out
    for path in sorted(root.rglob("*")):
        if path.is_file():
            out[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


def _semantic_pair_count(planning) -> int:
    return sum(
        1 for p in planning.fact_pair_plans if p.state == "needs_semantic_decision"
    )


# ---------------------------------------------------------------------------
# BLOCK 9 / 10 -- deterministic packing candidates (synthetic corpus)
# ---------------------------------------------------------------------------


def test_packing_candidates_synthetic(tmp_path):
    tree = _tree(tmp_path)
    planning = _planning(tree)
    profile = _consolidation_profile()
    rows = build_fact_packing_audit(planning, profile, _SEM_PROFILE, prompts=_PROMPTS)
    # Exactly the three documented candidates, in order (two independent limits).
    assert [row["packing_name"] for row in rows] == ["P1", "P2", "P3"]
    assert [(row["max_pairs_per_block"], row["max_candidates_per_block"]) for row in rows] == [
        (6, 12), (12, 24), (24, 48),
    ]
    total = _semantic_pair_count(planning)
    assert total >= 1
    for row in rows:
        # Cross-check every figure against a direct preparation with the same
        # explicit policy (the source of truth for the two-limit packing).
        policy = FactSemanticPackingPolicy(
            row["packing_name"], row["max_pairs_per_block"], row["max_candidates_per_block"]
        )
        prep = build_fact_semantic_preparation(
            planning, profile, _SEM_PROFILE, prompts=_PROMPTS, packing_policy=policy
        )
        pair_counts = [b.pair_count for b in prep.blocks]
        cand_counts = [len(b.candidate_refs) for b in prep.blocks]
        # No pair is lost or duplicated; one request per block.
        assert row["total_pairs_in_blocks"] == total == sum(pair_counts)
        assert row["block_count"] == len(prep.blocks)
        assert row["request_hash_count"] == len(prep.semantic_request_hashes)
        assert row["unique_request_hash_count"] == row["request_hash_count"]
        # Every block respects BOTH independent limits.
        assert all(pc <= row["max_pairs_per_block"] for pc in pair_counts)
        assert all(cc <= row["max_candidates_per_block"] for cc in cand_counts)
        # Distributions are consistent with the block pair / candidate counts.
        pairs = row["pairs_per_block"]
        cands = row["unique_candidates_per_block"]
        assert pairs["min"] == min(pair_counts) and pairs["max"] == max(pair_counts)
        assert cands["min"] == min(cand_counts) and cands["max"] == max(cand_counts)
        for dist in (pairs, cands, row["evidence_items_per_block"],
                     row["pair_contexts_json_bytes"], row["rendered_prompt_bytes"]):
            assert dist["min"] <= dist["median"] <= dist["p95"] <= dist["max"]
        # Largest-block diagnostic (BLOCK 10) is present and non-empty.
        largest = row["largest_block"]
        assert largest is not None
        assert largest["block_id"].startswith("a5fblk_")
        assert len(largest["block_id"]) == 27
        assert largest["pair_count"] >= 1
        assert largest["unique_candidate_count"] >= 1
        assert largest["rendered_prompt_bytes"] >= 0


def test_packing_candidates_deterministic(tmp_path):
    tree = _tree(tmp_path)
    planning = _planning(tree)
    profile = _consolidation_profile()
    rows1 = build_fact_packing_audit(planning, profile, _SEM_PROFILE, prompts=_PROMPTS)
    rows2 = build_fact_packing_audit(planning, profile, _SEM_PROFILE, prompts=_PROMPTS)
    assert rows1 == rows2


def test_packing_matches_direct_preparation(tmp_path):
    # The packing audit's P2 row matches a direct P2 preparation.
    tree = _tree(tmp_path)
    planning = _planning(tree)
    profile = _consolidation_profile()
    rows = build_fact_packing_audit(planning, profile, _SEM_PROFILE, prompts=_PROMPTS)
    p2 = next(r for r in rows if r["packing_name"] == "P2")
    p2_policy = A5C_PACKING_CANDIDATES[1]
    assert (p2["packing_name"], p2_policy.name) == ("P2", "P2")
    prep = build_fact_semantic_preparation(
        planning, profile, _SEM_PROFILE, prompts=_PROMPTS, packing_policy=p2_policy
    )
    assert p2["block_count"] == len(prep.blocks)
    assert p2["request_hash_count"] == len(prep.semantic_request_hashes)
    assert p2["total_pairs_in_blocks"] == _semantic_pair_count(planning)


# ---------------------------------------------------------------------------
# Zero-provider / read-only (synthetic)
# ---------------------------------------------------------------------------


def test_packing_is_zero_provider_and_read_only(tmp_path):
    tree = _tree(tmp_path)
    profile = _consolidation_profile()
    planning = _planning(tree)
    before = _fingerprint_dir(tree.store.root.parent)
    rows = build_fact_packing_audit(planning, profile, _SEM_PROFILE, prompts=_PROMPTS)
    after = _fingerprint_dir(tree.store.root.parent)
    assert before == after
    # Request hashes exist for every candidate.
    for row in rows:
        assert row["request_hash_count"] > 0
    _assert_module_is_provider_neutral()


def _assert_module_is_provider_neutral():
    import inspect

    from short_drama.story import consolidation_semantic

    src = inspect.getsource(consolidation_semantic)
    assert "adapter" not in src
    assert "transport" not in src


# ---------------------------------------------------------------------------
# Alice zero-provider request-shape smoke test (read-only, deterministic, gates)
# ---------------------------------------------------------------------------


_ALICE_RUNS = Path("runs") / "a4e_real_novel" / "a3e-real-novel"
_ALICE_PROJECT = "a3e-real-novel"
_ALICE_DOCUMENT = "src_001"
_ALICE_PROFILE = "entity-reconciliation-v2"

# Frozen Alice v1.2-A5C-A acceptance gates.
ALICE_FACT_CANDIDATES = 158
ALICE_FACT_SEMANTIC_PAIRS = 2113
ALICE_AUTO_SAME_EXACT = 0
# Two-limit packing block counts on the Alice corpus (verified, deterministic).
# The unique-candidate limit equals 2x the pair limit, so it can only close a
# block early (never late); on Alice the counts equal the pair-limit ceiling.
ALICE_P1_BLOCKS = 353
ALICE_P2_BLOCKS = 177
ALICE_P3_BLOCKS = 89


def _alice_tree_present() -> bool:
    root = Path(__file__).resolve().parents[1] / _ALICE_RUNS / "story"
    return (root / "artifacts").is_dir() and (root / "pointers").is_dir()


def _alice_stores():
    root = Path(__file__).resolve().parents[1] / _ALICE_RUNS / "story"
    store = FileArtifactStore(root / "artifacts")
    pointers = FilePointerStore(root / "pointers", store)
    return store, pointers, root


@pytest.mark.skipif(not _alice_tree_present(), reason="Alice run tree not present (local, gitignored)")
def test_alice_zero_provider_request_shape_smoke():
    store, pointers, root = _alice_stores()
    profile = load_consolidation_profile(audit.DEFAULT_CONSOLIDATION_PROFILE)
    sem_profile = load_fact_semantic_profile()
    prompts = PromptRegistry(DEFAULT_PROMPT_BASE_DIR)
    planning = build_consolidation_planning(
        store, pointers,
        project_id=_ALICE_PROJECT, document_id=_ALICE_DOCUMENT,
        reconciliation_profile_id=_ALICE_PROFILE, consolidation_profile=profile,
    )

    before = _fingerprint_dir(root)
    rows = build_fact_packing_audit(planning, profile, sem_profile, prompts=prompts)
    p2_policy = A5C_PACKING_CANDIDATES[1]
    prep = build_fact_semantic_preparation(
        planning, profile, sem_profile, prompts=prompts, packing_policy=p2_policy
    )
    prep_again = build_fact_semantic_preparation(
        planning, profile, sem_profile, prompts=prompts, packing_policy=p2_policy
    )
    after = _fingerprint_dir(root)

    # Read-only.
    assert before == after
    # Deterministic.
    assert [b.block_id for b in prep.blocks] == [b.block_id for b in prep_again.blocks]
    assert prep.semantic_request_hashes == prep_again.semantic_request_hashes

    # Exact gates.
    assert len(planning.index.facts) == ALICE_FACT_CANDIDATES
    assert prep.total_fact_pair_count == ALICE_FACT_SEMANTIC_PAIRS
    assert prep.auto_same_pair_count == ALICE_AUTO_SAME_EXACT
    assert prep.semantic_pair_count == ALICE_FACT_SEMANTIC_PAIRS

    by_name = {row["packing_name"]: row for row in rows}
    assert by_name["P1"]["block_count"] == ALICE_P1_BLOCKS == 353
    assert by_name["P2"]["block_count"] == ALICE_P2_BLOCKS == 177
    assert by_name["P3"]["block_count"] == ALICE_P3_BLOCKS == 89
    for row in rows:
        assert row["total_pairs_in_blocks"] == ALICE_FACT_SEMANTIC_PAIRS
        assert row["request_hash_count"] == row["block_count"]
        assert row["unique_request_hash_count"] == row["block_count"]
        assert row["largest_block"] is not None
        assert row["largest_block"]["block_id"].startswith("a5fblk_")


@pytest.mark.skipif(not _alice_tree_present(), reason="Alice run tree not present (local, gitignored)")
def test_alice_script_passes(capsys):
    rc = audit.main([])
    out = capsys.readouterr().out
    assert rc == 0
    assert "A5C AUDIT RESULT: PASS (zero-provider, read-only)" in out
    # Semantic stream figures.
    assert "semantic_fact_pairs:" in out
    assert "2113" in out
    # All three packing candidates are reported with the two-limit profile.
    assert "P1 (max_pairs_per_block=6 / max_candidates_per_block=12):" in out
    assert "P2 (max_pairs_per_block=12 / max_candidates_per_block=24):" in out
    assert "P3 (max_pairs_per_block=24 / max_candidates_per_block=48):" in out
    # The real request shape section is present.
    assert "structured_request_count:" in out
    assert "request_hash:" in out
    # Every check passes.
    assert "[FAIL]" not in out
