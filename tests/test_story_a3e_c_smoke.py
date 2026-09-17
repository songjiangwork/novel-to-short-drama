"""v1.2 A3E-C — offline tests for the real-novel smoke control logic.

These tests exercise the gate / selection / guard logic of
``scripts/a3e_real_novel_smoke.py`` with stubs and fakes. No live Qwen,
no provider, no A1/A2 ingest.

Covered:
  1. wrong served model blocks before provider generation
  2. unknown served model blocks before provider generation
  3. exact tracked model proceeds
  4. three-chunk selection is consecutive manifest order
  5. rerun gate fails if any additional provider call occurs
  6. invalidation gate fails if modified semantic identity incorrectly
     reuses old CURRENT
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# Import the smoke module by file path (it lives in scripts/, not a package).
SMOKE_PATH = REPO_ROOT / "scripts" / "a3e_real_novel_smoke.py"
spec = importlib.util.spec_from_file_location("a3e_real_novel_smoke", SMOKE_PATH)
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


# ---------------------------------------------------------------------------
# 1–3. Acceptance model guard
# ---------------------------------------------------------------------------


class TestAcceptanceModelGuard:
    def test_wrong_served_model_blocks(self) -> None:
        """Wrong served model → BLOCKED reason (not None)."""
        reason = smoke.acceptance_model_guard(
            tracked_model="ggml-org/Qwen3.8-27B-GGUF:Q4_K_M",
            server_model="some-other-model",
        )
        assert reason is not None
        assert "BLOCKED_BY_RUNTIME_ENVIRONMENT" in reason
        assert "some-other-model" in reason

    def test_unknown_served_model_blocks(self) -> None:
        """Unknown / unreachable server model → BLOCKED reason."""
        reason = smoke.acceptance_model_guard(
            tracked_model="ggml-org/Qwen3.8-27B-GGUF:Q4_K_M",
            server_model=None,
        )
        assert reason is not None
        assert "BLOCKED_BY_RUNTIME_ENVIRONMENT" in reason
        assert "unknown" in reason

    def test_exact_tracked_model_proceeds(self) -> None:
        """Exact model match → None (proceed)."""
        reason = smoke.acceptance_model_guard(
            tracked_model="ggml-org/Qwen3.8-27B-GGUF:Q4_K_M",
            server_model="ggml-org/Qwen3.8-27B-GGUF:Q4_K_M",
        )
        assert reason is None


# ---------------------------------------------------------------------------
# 4. Chunk selection is consecutive manifest order
# ---------------------------------------------------------------------------


class TestSelectConsecutiveChunks:
    def _fake_ref(self, i: int) -> object:
        """Create a lightweight fake ArtifactRef-like object."""
        return _FakeRef(f"chunk_{i}")

    def test_selects_first_three_consecutive(self) -> None:
        refs = tuple(self._fake_ref(i) for i in range(7))
        selected = smoke.select_consecutive_chunks(refs, 3)
        assert len(selected) == 3
        assert selected[0] is refs[0]
        assert selected[1] is refs[1]
        assert selected[2] is refs[2]

    def test_selection_is_consecutive_not_scattered(self) -> None:
        """Selected chunks must be consecutive in manifest order."""
        refs = tuple(self._fake_ref(i) for i in range(5))
        selected = smoke.select_consecutive_chunks(refs, 3)
        # Verify the selected refs are refs[0], refs[1], refs[2] — not
        # refs[0], refs[2], refs[4] or any other non-consecutive subset.
        for i, ref in enumerate(selected):
            assert ref is refs[i], (
                f"selected[{i}] is not refs[{i}]; "
                "selection must be consecutive manifest order"
            )

    def test_fewer_than_needed_raises(self) -> None:
        refs = tuple(self._fake_ref(i) for i in range(2))
        with pytest.raises(ValueError, match="need at least 3"):
            smoke.select_consecutive_chunks(refs, 3)

    def test_exact_count_succeeds(self) -> None:
        refs = tuple(self._fake_ref(i) for i in range(3))
        selected = smoke.select_consecutive_chunks(refs, 3)
        assert len(selected) == 3


class _FakeRef:
    """Minimal stand-in for ``ArtifactRef`` (identity-based)."""

    def __init__(self, artifact_id: str) -> None:
        self.artifact_id = artifact_id
        self.revision = 1
        self.artifact_type = "source_chunk"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _FakeRef):
            return NotImplemented
        return self.artifact_id == other.artifact_id

    def __hash__(self) -> int:
        return hash(self.artifact_id)

    def __repr__(self) -> str:
        return f"_FakeRef({self.artifact_id!r})"


# ---------------------------------------------------------------------------
# 5. Rerun gate fails if any additional provider call occurs
# ---------------------------------------------------------------------------


class TestRerunGate:
    def _make_result(self, reused: bool, extraction_id: str) -> dict:
        return {
            "reused": reused,
            "candidate_extraction_ref": _FakeRef(extraction_id),
            "validation_report_ref": _FakeRef(f"report_{extraction_id}"),
            "validation_result": "PASS",
            "candidate_counts": {},
            "total_candidates": 0,
        }

    def test_rerun_pass_when_all_reused_and_same_refs(self) -> None:
        first = [self._make_result(False, "ext_0"), self._make_result(False, "ext_1")]
        rerun = [self._make_result(True, "ext_0"), self._make_result(True, "ext_1")]
        failures = smoke.check_rerun_gate(
            first, rerun, 2, 2, 3, 3
        )
        assert failures == []

    def test_rerun_fails_when_additional_semantic_calls(self) -> None:
        first = [self._make_result(False, "ext_0")]
        rerun = [self._make_result(True, "ext_0")]
        failures = smoke.check_rerun_gate(
            first, rerun, 2, 3, 3, 3
        )
        assert len(failures) == 1
        assert "additional semantic generation calls" in failures[0]

    def test_rerun_fails_when_additional_provider_attempts(self) -> None:
        first = [self._make_result(False, "ext_0")]
        rerun = [self._make_result(True, "ext_0")]
        failures = smoke.check_rerun_gate(
            first, rerun, 2, 2, 3, 4
        )
        assert len(failures) == 1
        assert "additional provider attempts" in failures[0]

    def test_rerun_fails_when_not_reused(self) -> None:
        first = [self._make_result(False, "ext_0")]
        rerun = [self._make_result(False, "ext_0")]
        failures = smoke.check_rerun_gate(
            first, rerun, 2, 2, 3, 3
        )
        assert any("reused" in f for f in failures)

    def test_rerun_fails_when_ref_changed(self) -> None:
        first = [self._make_result(False, "ext_0")]
        rerun = [self._make_result(True, "ext_1")]
        failures = smoke.check_rerun_gate(
            first, rerun, 2, 2, 3, 3
        )
        assert any("candidate_extraction_ref" in f for f in failures)


# ---------------------------------------------------------------------------
# 6. Invalidation gate fails if modified identity reuses old CURRENT
# ---------------------------------------------------------------------------


class TestInvalidationGate:
    def _make_result(self, reused: bool, extraction_id: str) -> dict:
        return {
            "reused": reused,
            "candidate_extraction_ref": _FakeRef(extraction_id),
            "validation_report_ref": _FakeRef(f"report_{extraction_id}"),
            "validation_result": "PASS",
            "candidate_counts": {},
            "total_candidates": 0,
        }

    def test_pass_when_modified_profile_generates_fresh(self) -> None:
        tracked = self._make_result(False, "ext_tracked")
        modified = self._make_result(False, "ext_modified")
        failures = smoke.check_invalidation_gate(
            tracked, modified, 3, 4, 4, 5
        )
        assert failures == []

    def test_fails_when_modified_profile_reuses_old_current(self) -> None:
        tracked = self._make_result(False, "ext_tracked")
        # Modified profile incorrectly reuses the old CURRENT.
        modified = self._make_result(True, "ext_tracked")
        failures = smoke.check_invalidation_gate(
            tracked, modified, 3, 3, 4, 4
        )
        assert any("incorrectly reused" in f for f in failures)
        assert any("same CandidateExtraction ref" in f for f in failures)
        assert any("no semantic generation calls" in f for f in failures)
        assert any("no provider attempts" in f for f in failures)

    def test_fails_when_no_generation_calls(self) -> None:
        tracked = self._make_result(False, "ext_tracked")
        modified = self._make_result(False, "ext_modified")
        # No generation calls or attempts (should not happen if reuse is denied).
        failures = smoke.check_invalidation_gate(
            tracked, modified, 3, 3, 4, 4
        )
        assert any("no semantic generation calls" in f for f in failures)
        assert any("no provider attempts" in f for f in failures)
