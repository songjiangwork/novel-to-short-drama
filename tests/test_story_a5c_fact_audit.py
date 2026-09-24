"""v1.2 A5C-A -- deterministic block-packing + request-shape audit (zero provider).

Covers A5C-A BLOCK 3 / 5 (the audit side) implemented in
``short_drama.story.consolidation_semantic`` + ``scripts/a5c_fact_request_shape_audit.py``:

  * the three deterministic block-packing candidates (P1 6/12, P2 12/24,
    P3 24/48) -- audit ONLY, NOT frozen -- each with a consistent block count /
    min-avg-max block size / max block bytes / largest block id / per-request
    hashes / total request count;
  * the Alice zero-provider request-shape smoke test (exact acceptance gates,
    read-only, deterministic): the Alice corpus yields exactly 2113 fact
    semantic pairs, 0 auto_same, and the packing counts match the closed-form
    ceiling for each candidate;
  * the script is zero-provider (no transport) and read-only (run tree
    fingerprint unchanged) and deterministic (a second preparation is
    byte-identical).

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
    DEFAULT_PROMPT_BASE_DIR,
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


def _expected_block_count(total: int, max_block_size: int) -> int:
    if total == 0:
        return 0
    return -(-total // max_block_size)  # ceil division


# ---------------------------------------------------------------------------
# BLOCK 3 / BLOCK 5 -- deterministic packing candidates (synthetic corpus)
# ---------------------------------------------------------------------------


def test_packing_candidates_synthetic(tmp_path):
    tree = _tree(tmp_path)
    planning = _planning(tree)
    profile = _consolidation_profile()
    audit_rows = build_fact_packing_audit(planning, profile, _SEM_PROFILE, prompts=_PROMPTS)
    # Exactly the three documented candidates, in order.
    assert [row["candidate"] for row in audit_rows] == ["P1", "P2", "P3"]
    assert [(row["min_block_size"], row["requested_max_block_size"]) for row in audit_rows] == [
        (6, 12), (12, 24), (24, 48),
    ]
    total = _semantic_pair_count(planning)
    assert total >= 1
    for row in audit_rows:
        max_size = row["requested_max_block_size"]
        expected_blocks = _expected_block_count(total, max_size)
        assert row["total_fact_semantic_pairs"] == total
        assert row["block_count"] == expected_blocks
        assert row["total_request_count"] == expected_blocks
        assert row["max_block_size_actual"] == min(max_size, total)
        assert 1 <= row["min_block_size_actual"] <= max_size
        assert row["min_block_size_actual"] == min(max_size, total - (expected_blocks - 1) * max_size)
        assert row["max_block_bytes"] >= 0
        assert row["largest_block_id"].startswith("a5fblk_")
        assert len(row["largest_block_id"]) == 27
        # Per-request hashes: one per block, all 64-hex, unique.
        hashes = row["request_hashes"]
        assert len(hashes) == expected_blocks
        assert all(len(h) == 64 and h == h.lower() for h in hashes)
        assert len(set(hashes)) == len(hashes)


def test_packing_candidates_deterministic(tmp_path):
    tree = _tree(tmp_path)
    planning = _planning(tree)
    profile = _consolidation_profile()
    rows1 = build_fact_packing_audit(planning, profile, _SEM_PROFILE, prompts=_PROMPTS)
    rows2 = build_fact_packing_audit(planning, profile, _SEM_PROFILE, prompts=_PROMPTS)
    assert rows1 == rows2


def test_packing_matches_direct_preparation(tmp_path):
    # The packing audit's request hashes equal a direct preparation's hashes.
    tree = _tree(tmp_path)
    planning = _planning(tree)
    profile = _consolidation_profile()
    rows = build_fact_packing_audit(planning, profile, _SEM_PROFILE, prompts=_PROMPTS)
    p2 = next(r for r in rows if r["candidate"] == "P2")
    prep = build_fact_semantic_preparation(
        planning, profile, _SEM_PROFILE, prompts=_PROMPTS, max_block_size=24
    )
    assert p2["request_hashes"] == tuple(prep.semantic_request_hashes)
    assert p2["block_count"] == len(prep.blocks)


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
    # Provider-neutral request material for every candidate.
    for row in rows:
        assert row["request_hashes"]
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
# Closed-form block counts for the three candidates (ceil division).
ALICE_P1_BLOCKS = _expected_block_count(ALICE_FACT_SEMANTIC_PAIRS, 12)
ALICE_P2_BLOCKS = _expected_block_count(ALICE_FACT_SEMANTIC_PAIRS, 24)
ALICE_P3_BLOCKS = _expected_block_count(ALICE_FACT_SEMANTIC_PAIRS, 48)


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
    prep = build_fact_semantic_preparation(planning, profile, sem_profile, prompts=prompts)
    rows = build_fact_packing_audit(planning, profile, sem_profile, prompts=prompts)
    prep_again = build_fact_semantic_preparation(
        planning, profile, sem_profile, prompts=prompts
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

    by_name = {row["candidate"]: row for row in rows}
    assert by_name["P1"]["block_count"] == ALICE_P1_BLOCKS == 177
    assert by_name["P2"]["block_count"] == ALICE_P2_BLOCKS == 89
    assert by_name["P3"]["block_count"] == ALICE_P3_BLOCKS == 45
    for row in rows:
        assert row["total_fact_semantic_pairs"] == ALICE_FACT_SEMANTIC_PAIRS
        assert row["total_request_count"] == row["block_count"]
        assert row["largest_block_id"].startswith("a5fblk_")


@pytest.mark.skipif(not _alice_tree_present(), reason="Alice run tree not present (local, gitignored)")
def test_alice_script_passes(capsys):
    rc = audit.main([])
    out = capsys.readouterr().out
    assert rc == 0
    assert "A5C AUDIT RESULT: PASS (zero-provider, read-only)" in out
    assert "semantic_fact_pairs:       2113" in out
    assert "auto_same_fact_pairs:      0" in out
    # All three packing candidates are reported.
    assert "P1 (min 6 / requested max 12):" in out
    assert "P2 (min 12 / requested max 24):" in out
    assert "P3 (min 24 / requested max 48):" in out
    # Every check passes.
    assert "[FAIL]" not in out
    # The real request shape section is present.
    assert "structured_request_count:" in out
    assert "request_hash:" in out
