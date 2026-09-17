"""v1.2 A3E-C — offline tests for the real-novel smoke control logic.

These tests exercise the gate / selection logic of
``scripts/a3e_real_novel_smoke.py`` with stubs and fakes. No live Qwen,
no provider, no A1/A2 ingest.

Covered:
  1. three-chunk selection is consecutive manifest order
  2. rerun gate fails if any additional provider call occurs
  3. invalidation gate fails if modified semantic identity incorrectly
     reuses old CURRENT
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from short_drama.foundation import ValidationResult

REPO_ROOT = Path(__file__).resolve().parents[1]

# Import the smoke module by file path (it lives in scripts/, not a package).
SMOKE_PATH = REPO_ROOT / "scripts" / "a3e_real_novel_smoke.py"
spec = importlib.util.spec_from_file_location("a3e_real_novel_smoke", SMOKE_PATH)
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


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
# 1. Chunk selection is consecutive manifest order
# ---------------------------------------------------------------------------


class TestSelectConsecutiveChunks:
    def _fake_ref(self, i: int) -> _FakeRef:
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

    def test_more_than_needed_selects_prefix(self) -> None:
        refs = tuple(self._fake_ref(i) for i in range(10))
        selected = smoke.select_consecutive_chunks(refs, 3)
        assert selected == (refs[0], refs[1], refs[2])


# ---------------------------------------------------------------------------
# 2. Rerun gate fails if any additional provider call occurs
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
        first = [
            self._make_result(False, "ext_0"),
            self._make_result(False, "ext_1"),
        ]
        rerun = [
            self._make_result(True, "ext_0"),
            self._make_result(True, "ext_1"),
        ]
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

    def test_rerun_fails_multiple_chunks_partial_reuse(self) -> None:
        first = [
            self._make_result(False, "ext_0"),
            self._make_result(False, "ext_1"),
        ]
        rerun = [
            self._make_result(True, "ext_0"),
            self._make_result(False, "ext_1"),  # not reused
        ]
        failures = smoke.check_rerun_gate(
            first, rerun, 2, 2, 3, 3
        )
        assert any("chunk[1]" in f for f in failures)


# ---------------------------------------------------------------------------
# 3. Invalidation gate fails if modified identity reuses old CURRENT
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

    def test_passes_with_different_ref_and_generation(self) -> None:
        tracked = self._make_result(False, "ext_tracked")
        modified = self._make_result(False, "ext_new")
        failures = smoke.check_invalidation_gate(
            tracked, modified, 3, 4, 4, 5
        )
        assert failures == []


# ---------------------------------------------------------------------------
# 4. Validation gate
# ---------------------------------------------------------------------------


class TestValidationGate:
    def test_pass_when_all_pass(self) -> None:
        results = [
            {"validation_result": ValidationResult.PASS},
            {"validation_result": ValidationResult.PASS},
        ]
        failures = smoke.check_validation_gate(
            ["c1", "c2"], results
        )
        assert failures == []

    def test_fails_when_one_fails(self) -> None:
        results = [
            {"validation_result": ValidationResult.PASS},
            {"validation_result": ValidationResult.FAIL},
        ]
        failures = smoke.check_validation_gate(
            ["c1", "c2"], results
        )
        assert len(failures) == 1
        assert "c2" in failures[0]
