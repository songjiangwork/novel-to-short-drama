"""v1.2 A5C-A — zero-provider fact request-shape audit.

This is the A5C-A delivery slice (issue #52, BLOCK 1-6). It resolves the exact
current-eligible A4 CURRENT from a *pre-existing* run tree, builds the A5B
``ConsolidationPlanningResult``, and builds the A5C **fact semantic
preparation** + a **deterministic block-packing audit** for three packing
candidates -- with NO provider call and NO persistence anywhere.

Specifically it:

* selects the A5C fact semantic stream (the ``needs_semantic_decision`` fact
  pairs; ``auto_same`` fact pairs are already resolved deterministically by
  A5B and are never sent to the provider);
* deterministically packs the fact stream into blocks and evaluates the three
  audit-only packing candidates (P1 6/12, P2 12/24, P3 24/48);
* renders **real** ``StructuredGenerationRequest`` objects through the existing
  ``PromptRegistry`` / ``OutputSchema`` / ``SemanticLLMProfile`` infrastructure
  (the tracked ``a5.fact-consolidation`` prompt v1, the
  ``consolidation-fact-selector-payload`` schema v1, and the
  ``consolidation-llm-v1`` semantic profile);
* verifies the exact consolidation profile + prompt + schema + semantic profile
  identity (fail closed on any mismatch);
* proves it is zero-provider (no transport is touched; the requests are
  provider-neutral) and read-only (the run tree fingerprint is unchanged) and
  deterministic (a second preparation is byte-identical).

The three packing candidates are AUDIT ONLY: this script does NOT freeze the
production default block size and does NOT implement A5C-B (no response
handling, validation, canonical fact set, or CURRENT publication).

Usage:
    python scripts/a5c_fact_request_shape_audit.py \
        [--runs-root PATH] [--project ID] [--document ID] [--profile ID]

Exit code 0 on a clean audit, 2 on any structural / binding / integrity failure.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

from short_drama.artifacts import FileArtifactStore
from short_drama.foundation import FilePointerStore
from short_drama.llm import PromptRegistry
from short_drama.paths import REPO_ROOT
from short_drama.story import (
    A5C_DEFAULT_FACT_BLOCK_SIZE,
    ConsolidationCurrentMissingError,
    ConsolidationPlanningResult,
    DEFAULT_PROMPT_BASE_DIR,
    FactSemanticPreparation,
    StoryIntegrityError,
    build_consolidation_planning,
    build_fact_packing_audit,
    build_fact_semantic_identity,
    build_fact_semantic_preparation,
    load_consolidation_profile,
    load_fact_semantic_profile,
)

DEFAULT_RUNS_ROOT = REPO_ROOT / "runs" / "a4e_real_novel"
DEFAULT_PROJECT = "a3e-real-novel"
DEFAULT_DOCUMENT = "src_001"
DEFAULT_PROFILE = "entity-reconciliation-v2"
DEFAULT_CONSOLIDATION_PROFILE = REPO_ROOT / "profiles" / "consolidation_v1.yaml"

# Provider-neutrality: these keys must NEVER appear in the canonical request
# material (endpoint / transport / credential / routing identity).
_FORBIDDEN_REQUEST_KEYS = frozenset(
    {
        "endpoint",
        "url",
        "base_url",
        "hostname",
        "host",
        "api_key",
        "api-key",
        "credential",
        "token",
        "authorization",
        "timeout",
    }
)


# ---------------------------------------------------------------------------
# Stores + fingerprint
# ---------------------------------------------------------------------------


def _stores(runs_root: str | Path, project_id: str):
    root = Path(runs_root).expanduser() / project_id / "story"
    artifact_store = FileArtifactStore(root / "artifacts")
    pointer_store = FilePointerStore(root / "pointers", artifact_store)
    return artifact_store, pointer_store, root


def _fingerprint_dir(root: Path) -> dict[str, str]:
    """path-relative -> sha256(content) for every file under ``root``."""
    out: dict[str, str] = {}
    if not root.exists():
        return out
    for path in sorted(root.rglob("*")):
        if path.is_file():
            out[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


def _request_material_keys(request) -> set[str]:
    """All keys present anywhere in the canonical semantic request material."""
    keys: set[str] = set()

    def _walk(value) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                keys.add(str(key))
                _walk(child)
        elif isinstance(value, list):
            for child in value:
                _walk(child)

    _walk(request.semantic_request_material())
    return keys


# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------


def _print_identity(project_id: str, document_id: str, reconciliation_profile_id: str,
                    prep: FactSemanticPreparation) -> None:
    identity = build_fact_semantic_identity(prep)
    print("=== A5C-A FACT REQUEST-SHAPE AUDIT (zero-provider, read-only) ===")
    print(f"project:              {project_id}")
    print(f"document:             {document_id}")
    print(f"reconciliation:       {reconciliation_profile_id}")
    print()
    print("planning identity:")
    print(f"  planning_policy_id:              {identity['planning_policy_id']}")
    print(f"  blocking_policy_id:              {identity['blocking_policy_id']}")
    print(f"  text_normalization_policy_id:    {identity['text_normalization_policy_id']}")
    print(f"  exact_safe_policy_id:            {identity['exact_safe_policy_id']}")
    print(f"  plan_hash:                       {identity['plan_hash']}")
    print()
    print("verified consolidation identity (fail closed on mismatch):")
    print(f"  profile_id:                      {identity['profile_id']}")
    print(f"  profile_hash:                    {identity['profile_hash']}")
    print(f"  working_language:                {identity['working_language']}")
    print(f"  semantic_profile_id:             {identity['semantic_profile_id']}")
    print(f"  semantic_profile_hash:           {identity['semantic_profile_hash']}")
    print(f"  prompt_id:                       {identity['prompt_id']}")
    print(f"  prompt_version:                  {identity['prompt_version']}")
    print(f"  prompt_content_hash:             {identity['prompt_content_hash']}")
    print(f"  output_schema_id:                {identity['output_schema_id']}")
    print(f"  output_schema_version:           {identity['output_schema_version']}")
    print(f"  output_schema_hash:              {identity['output_schema_hash']}")
    print()


def _print_candidate_universe(prep: FactSemanticPreparation) -> None:
    index = prep.planning_result.index
    print("=== FACT CANDIDATE UNIVERSE ===")
    print(f"fact_candidate_count:      {len(index.facts)}")
    print(f"fact_pair_plan_count:      {prep.total_fact_pair_count}")
    print(f"auto_same_fact_pairs:      {prep.auto_same_pair_count}")
    print(f"semantic_fact_pairs:       {prep.semantic_pair_count} (needs_semantic_decision)")
    print()


def _print_packing(audit: tuple[dict, ...]) -> None:
    print("=== DETERMINISTIC PACKING CANDIDATES (audit-only, NOT frozen) ===")
    for row in audit:
        print(f"{row['candidate']} (min {row['min_block_size']} / requested max "
              f"{row['requested_max_block_size']}):")
        print(f"  total_fact_semantic_pairs: {row['total_fact_semantic_pairs']}")
        print(f"  block_count:               {row['block_count']}")
        print(f"  min_block_size:            {row['min_block_size_actual']}")
        print(f"  avg_block_size:            {row['avg_block_size']:.4f}")
        print(f"  max_block_size:            {row['max_block_size_actual']}")
        print(f"  max_block_bytes:           {row['max_block_bytes']}")
        print(f"  largest_block_id:          {row['largest_block_id']}")
        print(f"  total_request_count:       {row['total_request_count']}")
        hashes = row["request_hashes"]
        print(f"  request_hashes ({len(hashes)}):")
        for ordinal, request_hash in enumerate(hashes[:5]):
            print(f"    [{ordinal:3}] {request_hash}")
        if len(hashes) > 5:
            print(f"    ... (+{len(hashes) - 5} more)")
        print()


def _print_request_shape(prep: FactSemanticPreparation) -> None:
    print("=== REQUEST SHAPE (real StructuredGenerationRequest, default block size "
          f"{prep.max_block_size}) ===")
    requests = prep.structured_requests
    print(f"structured_request_count:  {len(requests)}")
    if not requests:
        print("  (no fact semantic pairs -> no requests)")
        print()
        return
    print(f"request_schema_version:    {requests[0].schema_version}")
    for ordinal, (block, request) in enumerate(zip(prep.blocks, requests)):
        if ordinal >= 2:
            print(f"... (+{len(requests) - 2} more requests)")
            break
        user_text = request.rendered_prompt.user_text
        system_text = request.rendered_prompt.system_text
        print(f"request[{ordinal}]:")
        print(f"  block_id:                      {block.block_id}")
        print(f"  pair_count:                    {block.pair_count}")
        print(f"  payload_bytes:                 {block.payload_bytes}")
        print(f"  messages:                      {[m['role'] for m in request.messages]}")
        print(f"  system_text_len:               {len(system_text)}")
        print(f"  user_text_len:                 {len(user_text)}")
        print(f"  user_contains_block_id:        {block.block_id in user_text}")
        print(f"  request_hash:                  {request.request_hash}")
        print(f"  semantic_profile_hash:         {request.semantic_profile.semantic_profile_hash}")
        print(f"  output_schema_hash:            {request.output_schema.schema_hash}")
        print(f"  rendered_prompt_hash:          {request.rendered_prompt.rendered_prompt_hash}")
    print()


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def _packing_row_consistent(row: dict, total_pairs: int) -> tuple[bool, str]:
    requested_max = row["requested_max_block_size"]
    block_count = row["block_count"]
    if row["total_fact_semantic_pairs"] != total_pairs:
        return False, f"{row['candidate']} total pairs mismatch"
    if row["total_request_count"] != block_count:
        return False, f"{row['candidate']} request count != block count"
    if row["min_block_size_actual"] < 1 or row["max_block_size_actual"] > requested_max:
        return False, f"{row['candidate']} block size out of [1, {requested_max}]"
    if row["block_count"] == 0:
        return total_pairs == 0, f"{row['candidate']} empty corpus mismatch"
    # The first block_count-1 blocks are exactly requested_max; the last is the
    # remainder (or requested_max if it divides evenly). So the total must equal
    # (block_count-1)*requested_max + last, with last in [1, requested_max].
    expected_full = (block_count - 1) * requested_max
    last = total_pairs - expected_full
    if not 1 <= last <= requested_max:
        return False, f"{row['candidate']} block sizes do not sum to total pairs"
    if row["max_block_size_actual"] != requested_max:
        return False, f"{row['candidate']} max block size != requested max"
    if row["min_block_size_actual"] != min(requested_max, last):
        return False, f"{row['candidate']} min block size mismatch"
    return True, "ok"


def _run_checks(
    prep: FactSemanticPreparation,
    prep_again: FactSemanticPreparation,
    packing_audit: tuple[dict, ...],
    before: dict,
    after: dict,
) -> list[tuple[str, bool]]:
    checks: list[tuple[str, bool]] = []

    # Read-only: the run tree fingerprint is unchanged.
    checks.append(("read-only (run tree fingerprint unchanged)", before == after))

    # Deterministic: a second preparation is byte-identical (block ids + hashes).
    checks.append(
        (
            "deterministic (block ids + request hashes stable)",
            (tuple(b.block_id for b in prep.blocks)
             == tuple(b.block_id for b in prep_again.blocks)
             and prep.semantic_request_hashes == prep_again.semantic_request_hashes),
        )
    )

    # Real requests: every request is a real StructuredGenerationRequest.
    from short_drama.llm.models import StructuredGenerationRequest

    checks.append(
        (
            "real requests (StructuredGenerationRequest objects)",
            all(isinstance(r, StructuredGenerationRequest) for r in prep.structured_requests),
        )
    )

    # Zero provider: the canonical request material carries no endpoint /
    # transport / credential / routing identity.
    neutral = all(
        not (_FORBIDDEN_REQUEST_KEYS & _request_material_keys(r))
        for r in prep.structured_requests
    )
    checks.append(("zero-provider (request material is provider-neutral)", neutral))

    # Block id shape: every block id is a5fblk_ + 20 hex chars (27 total).
    block_id_ok = all(
        block.block_id.startswith("a5fblk_") and len(block.block_id) == 7 + 20
        for block in prep.blocks
    )
    checks.append(("block id shape (a5fblk_ + 20 hex)", block_id_ok))

    # Pair-contexts JSON is deterministic canonical JSON (re-canonicalizes).
    from short_drama.artifacts.canonical import canonical_json_bytes

    canonical_ok = all(
        canonical_json_bytes([dict(pc) for pc in block.pair_contexts]).decode("utf-8")
        == block.pair_contexts_json
        for block in prep.blocks
    )
    checks.append(("pair_contexts_json is deterministic canonical JSON", canonical_ok))

    # The rendered user prompt carries the block id (variable 1) and the
    # pair contexts (variable 2). render_prompt already enforces that these are
    # EXACTLY the two frozen required variables.
    user_ok = all(
        request.rendered_prompt.user_text.startswith(f"Fact consolidation block: {block.block_id}")
        and block.pair_contexts_json in request.rendered_prompt.user_text
        for block, request in zip(prep.blocks, prep.structured_requests)
    )
    checks.append(("rendered user prompt carries block_id + pair contexts", user_ok))

    # Semantic-stream boundary: auto_same pairs are excluded from the requests.
    total_in_blocks = sum(block.pair_count for block in prep.blocks)
    checks.append(
        (
            "auto_same pairs excluded (blocks carry only semantic pairs)",
            total_in_blocks == prep.semantic_pair_count,
        )
    )

    # Packing candidates are internally consistent.
    for row in packing_audit:
        ok, _ = _packing_row_consistent(row, prep.semantic_pair_count)
        checks.append((f"packing {row['candidate']} consistent", ok))

    # Pair-local selectors: each endpoint packet's evidence selectors are exactly
    # L0..L{n-1} (left) / R0..R{n-1} (right), and no block-wide pool exists.
    selector_ok = True
    for block in prep.blocks:
        for pc in block.pair_contexts:
            for side, prefix in (("left", "L"), ("right", "R")):
                selectors = [ev["selector"] for ev in pc[side]["evidence"]]
                expected = [f"{prefix}{idx}" for idx in range(len(pc[side]["evidence"]))]
                if selectors != expected:
                    selector_ok = False
    checks.append(("pair-local evidence selectors (L0.. / R0.., no block pool)", selector_ok))

    return checks


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_audit(
    store: FileArtifactStore,
    pointers: FilePointerStore,
    root: Path,
    *,
    project_id: str,
    document_id: str,
    reconciliation_profile_id: str,
    consolidation_profile_path: Path = DEFAULT_CONSOLIDATION_PROFILE,
    default_block_size: int = A5C_DEFAULT_FACT_BLOCK_SIZE,
) -> int:
    consolidation_profile = load_consolidation_profile(consolidation_profile_path)
    semantic_profile = load_fact_semantic_profile()
    prompts = PromptRegistry(DEFAULT_PROMPT_BASE_DIR)
    result: ConsolidationPlanningResult = build_consolidation_planning(
        store,
        pointers,
        project_id=project_id,
        document_id=document_id,
        reconciliation_profile_id=reconciliation_profile_id,
        consolidation_profile=consolidation_profile,
    )

    before = _fingerprint_dir(root)

    prep = build_fact_semantic_preparation(
        result,
        consolidation_profile,
        semantic_profile,
        prompts=prompts,
        max_block_size=default_block_size,
    )
    packing_audit = build_fact_packing_audit(
        result,
        consolidation_profile,
        semantic_profile,
        prompts=prompts,
    )
    # Determinism: build a second preparation and compare.
    prep_again = build_fact_semantic_preparation(
        result,
        consolidation_profile,
        semantic_profile,
        prompts=prompts,
        max_block_size=default_block_size,
    )

    after = _fingerprint_dir(root)

    _print_identity(project_id, document_id, reconciliation_profile_id, prep)
    _print_candidate_universe(prep)
    _print_packing(packing_audit)
    _print_request_shape(prep)

    print("=== AUDIT CHECKS ===")
    checks = _run_checks(prep, prep_again, packing_audit, before, after)
    all_pass = True
    for label, passed in checks:
        all_pass = all_pass and passed
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}")
    print()

    if not all_pass:
        print("A5C AUDIT RESULT: FAIL", file=sys.stderr)
        return 2
    print("A5C AUDIT RESULT: PASS (zero-provider, read-only)")
    print("A5C-A complete: fact semantic preparation + deterministic packing audit +")
    print("real StructuredGenerationRequest rendering. Still zero-provider and")
    print("read-only: no LLM call, no canonical fact set, no A5C-B persistence/CURRENT.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="A5C-A zero-provider fact request-shape audit"
    )
    parser.add_argument("--runs-root", default=str(DEFAULT_RUNS_ROOT))
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--document", default=DEFAULT_DOCUMENT)
    parser.add_argument("--profile", default=DEFAULT_PROFILE)
    parser.add_argument(
        "--default-block-size",
        type=int,
        default=A5C_DEFAULT_FACT_BLOCK_SIZE,
        help="requested maximum block size for the request-shape section",
    )
    args = parser.parse_args(argv)

    try:
        store, pointers, root = _stores(args.runs_root, args.project)
        return run_audit(
            store,
            pointers,
            root,
            project_id=args.project,
            document_id=args.document,
            reconciliation_profile_id=args.profile,
            default_block_size=args.default_block_size,
        )
    except (ConsolidationCurrentMissingError, StoryIntegrityError) as exc:
        print(f"A5C AUDIT RESULT: FAIL ({type(exc).__name__}: {exc})", file=sys.stderr)
        return 2
    except Exception as exc:  # pragma: no cover - defensive (unexpected)
        print(f"A5C AUDIT RESULT: ERROR ({type(exc).__name__}: {exc})", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
