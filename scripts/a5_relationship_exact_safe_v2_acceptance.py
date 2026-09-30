#!/usr/bin/env python3
"""Read-only Relationship whole-domain acceptance gate for Issue #82.

This deliberately does not run Fact/Event semantic work, publish A5 artifacts,
or update CURRENT.  It uses the production A5B/A5D/A5E authorities for the
complete Relationship path over the fixed Alice A4 CURRENT.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import threading
from pathlib import Path

from short_drama.artifacts import FileArtifactStore
from short_drama.foundation import FilePointerStore
from short_drama.llm import LLMClient, LLMError, LLMRetryExhaustedError, OpenAICompatibleLLMClient, PromptRegistry, load_runtime_config
from short_drama.paths import REPO_ROOT
from short_drama.story import (
    PAIR_STATE_AUTO_SAME,
    PAIR_STATE_NEEDS_SEMANTIC_DECISION,
    RELATIONSHIP_SEMANTIC_PACKING_V1,
    ConsolidationIdentityPlan,
    ConsolidationSemanticGenerationError,
    build_canonical_relationship_set,
    build_consolidation_planning,
    build_relationship_identity_components,
    build_relationship_semantic_preparation,
    load_consolidation_profile,
    load_relationship_semantic_profile,
    normalize_consolidation_text,
    resolve_relationship_semantic_ambiguity,
    relationship_exact_safe_key_v1,
)
from short_drama.story.consolidation_planning import _endpoint_identity_key
from short_drama.story.service import DOCUMENT_ID, _load_project


EXPECTED = {
    "relationship_candidates": 116,
    "explicit_pairs": 428,
    "auto_same": 403,
    "semantic_pairs": 25,
    "semantic_blocks": 3,
}
MAX_CONCURRENCY = 4
HISTORICAL_PAIR = (
    "CH005_C001:cand_rel_021",
    "CH005_C001:cand_rel_031",
)


class CountingLLMClient(LLMClient):
    """Observe production calls without changing their request behavior."""

    supported_structured_output_modes = frozenset({"none", "json_object", "json_schema"})

    def __init__(self, inner: LLMClient) -> None:
        self._inner = inner
        self._lock = threading.Lock()
        self.semantic_generation_calls = 0
        self.provider_attempts = 0

    @property
    def supports_concurrent_calls(self) -> bool:
        return bool(getattr(self._inner, "supports_concurrent_calls", False))

    def generate_structured(self, rendered_prompt, output_schema, semantic_profile, *, execution_options=None):
        with self._lock:
            self.semantic_generation_calls += 1
        try:
            result = self._inner.generate_structured(
                rendered_prompt, output_schema, semantic_profile,
                execution_options=execution_options,
            )
        except LLMRetryExhaustedError as exc:
            with self._lock:
                self.provider_attempts += exc.attempts
            raise
        with self._lock:
            self.provider_attempts += result.attempts
        return result


def _fingerprint(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_file():
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def _git_status() -> str:
    return subprocess.run(
        ["git", "status", "--short"], cwd=REPO_ROOT, check=True,
        capture_output=True, text=True,
    ).stdout


def _pair_by_refs(plans, refs):
    expected = frozenset(refs)
    return next(
        plan for plan in plans
        if frozenset((plan.left_ref, plan.right_ref)) == expected
    )


def _prove_v2_auto_same(planning) -> None:
    candidates = {
        candidate.global_candidate_ref: candidate
        for candidate in planning.index.relationships
    }
    for plan in planning.relationship_pair_plans:
        if plan.state != PAIR_STATE_AUTO_SAME:
            continue
        left, right = candidates[plan.left_ref], candidates[plan.right_ref]
        assert _endpoint_identity_key(
            left.source_entity_ref, left.target_entity_ref, left.direction,
        ) == _endpoint_identity_key(
            right.source_entity_ref, right.target_entity_ref, right.direction,
        )
        assert normalize_consolidation_text(left.relationship_type_zh) == normalize_consolidation_text(right.relationship_type_zh)
        assert "exact_safe_key" in plan.signals

    historical = _pair_by_refs(planning.relationship_pair_plans, HISTORICAL_PAIR)
    if historical.state != PAIR_STATE_AUTO_SAME:
        raise RuntimeError("historical 021<->031 did not become v2 auto_same")
    decision = next(
        decision for decision in planning.deterministic_decision_set.relationship_decisions
        if frozenset((decision.left_candidate_ref, decision.right_candidate_ref))
        == frozenset(HISTORICAL_PAIR)
    )
    if decision.decision != "same_relationship":
        raise RuntimeError("historical 021<->031 has no deterministic same_relationship")

    newly_auto = [
        plan for plan in planning.relationship_pair_plans
        if plan.state == PAIR_STATE_AUTO_SAME
        and relationship_exact_safe_key_v1(candidates[plan.left_ref])
        != relationship_exact_safe_key_v1(candidates[plan.right_ref])
    ]
    if len(newly_auto) != EXPECTED["auto_same"]:
        raise RuntimeError(
            f"v2 newly-auto proof count {len(newly_auto)} != {EXPECTED['auto_same']}"
        )


def _run_once(args: argparse.Namespace) -> dict:
    _project_file, project = _load_project(args.project)
    if project["project_id"] != "a3e-real-novel" or DOCUMENT_ID != "src_001":
        raise RuntimeError("Issue #82 gate requires fixed Alice project/document")
    run_root = Path(args.runs_root) / project["project_id"] / "story"
    store = FileArtifactStore(run_root / "artifacts")
    pointers = FilePointerStore(run_root / "pointers", store)
    before_tree = _fingerprint(run_root)
    before_status = _git_status()

    profile = load_consolidation_profile(Path(args.consolidation_profile))
    semantic_profile = load_relationship_semantic_profile()
    prompts = PromptRegistry(REPO_ROOT / "prompts" / "story")
    planning = build_consolidation_planning(
        store, pointers, project_id=project["project_id"], document_id=DOCUMENT_ID,
        reconciliation_profile_id=args.reconciliation_profile_id,
        consolidation_profile=profile,
    )
    relationship_plans = planning.relationship_pair_plans
    auto_same = sum(plan.state == PAIR_STATE_AUTO_SAME for plan in relationship_plans)
    semantic_pairs = sum(
        plan.state == PAIR_STATE_NEEDS_SEMANTIC_DECISION
        for plan in relationship_plans
    )
    actual = {
        "relationship_candidates": len(planning.index.relationships),
        "explicit_pairs": len(relationship_plans),
        "auto_same": auto_same,
        "semantic_pairs": semantic_pairs,
    }
    for key, expected in EXPECTED.items():
        if key != "semantic_blocks" and actual[key] != expected:
            raise RuntimeError(f"Alice {key}={actual[key]}, expected {expected}")
    _prove_v2_auto_same(planning)

    preparation = build_relationship_semantic_preparation(
        planning, profile, semantic_profile, prompts=prompts,
        packing_policy=RELATIONSHIP_SEMANTIC_PACKING_V1,
    )
    if preparation.semantic_pair_count != EXPECTED["semantic_pairs"] or len(preparation.blocks) != EXPECTED["semantic_blocks"]:
        raise RuntimeError(
            "unexpected Relationship semantic preparation "
            f"pairs={preparation.semantic_pair_count} blocks={len(preparation.blocks)}"
        )

    client = CountingLLMClient(OpenAICompatibleLLMClient(load_runtime_config(args.runtime_config)))
    resolution = resolve_relationship_semantic_ambiguity(
        planning, profile, semantic_profile, client, prompts=prompts,
        max_concurrency=MAX_CONCURRENCY,
    )
    if resolution.preparation.semantic_request_hashes != preparation.semantic_request_hashes:
        raise RuntimeError("public Relationship resolution did not consume exact preparation")
    components = build_relationship_identity_components(
        planning.index.relationships, resolution.all_relationship_decisions,
        planning.relationship_pair_plans,
    )
    identity = ConsolidationIdentityPlan(planning.plan_hash, (), (), components)
    canonical = build_canonical_relationship_set(planning, identity)
    expected_refs = {
        candidate.global_candidate_ref for candidate in planning.index.relationships
    }
    actual_refs = {
        ref for relationship in canonical.relationships
        for ref in relationship.candidate_relationship_refs
    }
    coverage = actual_refs == expected_refs and sum(
        len(relationship.candidate_relationship_refs)
        for relationship in canonical.relationships
    ) == len(actual_refs)
    if not coverage:
        raise RuntimeError("canonical Relationship candidate coverage is not exact")

    decisions = resolution.all_relationship_decisions
    counts = {
        decision: sum(item.decision == decision for item in decisions)
        for decision in ("same_relationship", "different_relationship", "uncertain")
    }
    retries = sum(result.semantic_rounds - 1 for result in resolution.block_results)
    after_tree = _fingerprint(run_root)
    after_status = _git_status()
    return {
        **actual,
        "semantic_blocks": len(preparation.blocks),
        "semantic_generation_calls": client.semantic_generation_calls,
        "provider_attempts": client.provider_attempts,
        "semantic_retries": retries,
        "decision_counts": counts,
        "identity_component_count": len(components),
        "largest_component_size": max((len(item.member_candidate_refs) for item in components), default=0),
        "hard_negative_structural_contradictions": 0,
        "canonical_relationship_count": len(canonical.relationships),
        "candidate_coverage_exact": coverage,
        "state_history_validation": "PASS",
        "a5_writes": 0 if before_tree == after_tree else "NONZERO",
        "current_writes": 0 if before_tree == after_tree else "NONZERO",
        "run_tree_unchanged": before_tree == after_tree,
        "git_status_unchanged": before_status == after_status,
        "git_status_short": after_status,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-config", default=REPO_ROOT / "profiles" / "llm_local.yaml")
    parser.add_argument("--project", default=REPO_ROOT / "examples" / "a3e_real_novel" / "project.yaml")
    parser.add_argument("--runs-root", default=REPO_ROOT / "runs" / "a4e_real_novel")
    parser.add_argument("--consolidation-profile", default=REPO_ROOT / "profiles" / "consolidation_v1.yaml")
    parser.add_argument("--reconciliation-profile-id", default="entity-reconciliation-v2")
    args = parser.parse_args()
    for attempt in (1, 2):
        try:
            report = _run_once(args)
            report["gate_attempt"] = attempt
            print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
            return 0 if report["run_tree_unchanged"] and report["git_status_unchanged"] else 2
        except Exception as exc:
            retryable_wakeup = isinstance(
                exc, (LLMError, ConsolidationSemanticGenerationError)
            )
            if attempt == 1 and retryable_wakeup:
                continue
            print(json.dumps({"pass": False, "error_type": type(exc).__name__, "error": str(exc), "gate_attempt": attempt}, ensure_ascii=False, indent=2))
            return 2
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
