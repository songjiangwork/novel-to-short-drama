"""v1.2 A5D-A — zero-provider event + relationship request-shape audit.

This is the A5D-A delivery slice (issue #53). It resolves the exact
current-eligible A4 CURRENT from a *pre-existing* run tree, builds the A5B
``ConsolidationPlanningResult``, validates the A5D **event** and **relationship**
semantic streams fail closed, builds the A5D event + relationship semantic
preparations under an EXPLICIT packing policy, and runs a **deterministic
block-packing audit** over three packing candidates per domain -- with NO
provider call and NO persistence anywhere.

Specifically, for BOTH the event and relationship domains it:

* selects and validates the semantic stream (EXACTLY the
  ``needs_semantic_decision`` pairs; ``auto_same`` pairs are deterministic and
  excluded) -- domain namespace, ``left_ref < right_ref``, no duplicate pair,
  canonical ``(left_ref, right_ref)`` order;
* deterministically packs the stream under TWO independent limits (at most
  ``max_pairs_per_block`` pairs AND at most ``max_candidates_per_block`` unique
  candidates) and evaluates the three audit-only packing candidates
  (P1 6/12, P2 12/24, P3 24/48);
* reports the deterministic distribution statistics for each candidate
  (pairs / unique candidates / evidence items / pair-context bytes / rendered
  prompt bytes, each min/median/p95/max), the deterministic largest-block
  diagnostics, and the request-hash counts;
* renders **real** ``StructuredGenerationRequest`` objects through the existing
  ``PromptRegistry`` / ``OutputSchema`` / ``SemanticLLMProfile`` infrastructure
  (the tracked ``a5.event-consolidation`` / ``a5.relationship-consolidation``
  prompts v1, the ``consolidation-{event,relationship}-selector-payload``
  schemas v1, and the shared ``consolidation-llm-v1`` semantic profile);
* verifies the exact consolidation profile + prompt + schema + semantic profile
  identity (fail closed on any mismatch, including ``max_generation_rounds == 2``);
* proves it is zero-provider (no transport is touched; the requests are
  provider-neutral) and read-only (the run tree fingerprint is unchanged) and
  deterministic (a second preparation is byte-identical);
* verifies the exact Alice real-novel candidate / pair counts (event 167
  candidates / 2941 pairs / 0 auto_same; relationship 116 / 428 / 0).

The packing candidates are AUDIT ONLY: ``event-semantic-packing-v1`` and
``relationship-semantic-packing-v1`` are NOT YET FROZEN, so this script requires
an explicit policy, does NOT freeze a production default, and does NOT implement
A5D-B (no response handling, payload validation, canonical event / relationship
sets, or CURRENT publication).

Usage:
    python scripts/a5d_event_relationship_request_shape_audit.py \
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
    A5D_EVENT_PACKING_CANDIDATES,
    A5D_RELATIONSHIP_PACKING_CANDIDATES,
    ConsolidationCurrentMissingError,
    ConsolidationPlanningResult,
    DEFAULT_PROMPT_BASE_DIR,
    EventSemanticPreparation,
    RelationshipSemanticPreparation,
    StoryIntegrityError,
    build_consolidation_planning,
    build_event_packing_audit,
    build_event_semantic_identity,
    build_event_semantic_preparation,
    build_relationship_packing_audit,
    build_relationship_semantic_identity,
    build_relationship_semantic_preparation,
    load_consolidation_profile,
    load_event_semantic_profile,
    load_relationship_semantic_profile,
)

DEFAULT_RUNS_ROOT = REPO_ROOT / "runs" / "a4e_real_novel"
DEFAULT_PROJECT = "a3e-real-novel"
DEFAULT_DOCUMENT = "src_001"
DEFAULT_PROFILE = "entity-reconciliation-v2"
DEFAULT_CONSOLIDATION_PROFILE = REPO_ROOT / "profiles" / "consolidation_v1.yaml"

#: The policy used for the sample request-shape section (audit-only, P2).
EVENT_REQUEST_SHAPE_POLICY = A5D_EVENT_PACKING_CANDIDATES[1]
RELATIONSHIP_REQUEST_SHAPE_POLICY = A5D_RELATIONSHIP_PACKING_CANDIDATES[1]

#: The exact Alice real-novel candidate / pair counts the audit must reproduce
#: (from the frozen A4 current-eligible tree + A5B planning).
ALICE_EVENT_CANDIDATE_COUNT = 167
ALICE_EVENT_TOTAL_PAIR_COUNT = 2941
ALICE_EVENT_AUTO_SAME_COUNT = 0
ALICE_RELATIONSHIP_CANDIDATE_COUNT = 116
ALICE_RELATIONSHIP_TOTAL_PAIR_COUNT = 428
ALICE_RELATIONSHIP_AUTO_SAME_COUNT = 0

#: The rendered user-prompt prefix per domain (the block_id variable).
EVENT_USER_PROMPT_PREFIX = "Event consolidation block: "
RELATIONSHIP_USER_PROMPT_PREFIX = "Relationship consolidation block: "

#: Provider-neutrality: these keys must NEVER appear in the canonical request
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


def _fmt_distribution(dist: dict[str, float]) -> str:
    return (
        f"min={int(dist['min'])} median={int(dist['median'])} "
        f"p95={int(dist['p95'])} max={int(dist['max'])}"
    )


def _fmt_pair(pair) -> str:
    if pair is None:
        return "-"
    left, right = pair
    return f"{left} <-> {right}"


def _print_identity(project_id, document_id, reconciliation_profile_id, prep) -> None:
    identity = (
        build_event_semantic_identity(prep)
        if isinstance(prep, EventSemanticPreparation)
        else build_relationship_semantic_identity(prep)
    )
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


def _print_candidate_universe(prep, domain: str) -> None:
    index = prep.planning_result.index
    if isinstance(prep, EventSemanticPreparation):
        candidate_count = len(index.events)
        total_pairs = prep.total_event_pair_count
        semantic_pairs = prep.semantic_pair_count
        auto_same = prep.auto_same_pair_count
    else:
        candidate_count = len(index.relationships)
        total_pairs = prep.total_relationship_pair_count
        semantic_pairs = prep.semantic_pair_count
        auto_same = prep.auto_same_pair_count
    print(f"=== {domain.upper()} CANDIDATE UNIVERSE ===")
    print(f"{domain}_candidate_count:      {candidate_count}")
    print(f"{domain}_pair_plan_count:      {total_pairs}")
    print(f"auto_same_{domain}_pairs:      {auto_same}")
    print(f"semantic_{domain}_pairs:       {semantic_pairs} (needs_semantic_decision)")
    print()


def _print_packing(audit: tuple[dict, ...]) -> None:
    print("=== DETERMINISTIC PACKING CANDIDATES (audit-only, NOT frozen) ===")
    for row in audit:
        print(
            f"{row['packing_name']} "
            f"(max_pairs_per_block={row['max_pairs_per_block']} / "
            f"max_candidates_per_block={row['max_candidates_per_block']}):"
        )
        print(f"  block_count:               {row['block_count']}")
        print(f"  total_pairs_in_blocks:     {row['total_pairs_in_blocks']}")
        print(f"  pairs/block:               {_fmt_distribution(row['pairs_per_block'])}")
        print(f"  candidates/block:          {_fmt_distribution(row['unique_candidates_per_block'])}")
        print(f"  evidence/block:            {_fmt_distribution(row['evidence_items_per_block'])}")
        print(f"  pair_contexts_bytes:       {_fmt_distribution(row['pair_contexts_json_bytes'])}")
        print(f"  rendered_prompt_bytes:     {_fmt_distribution(row['rendered_prompt_bytes'])}")
        print(f"  request_hash_count:        {row['request_hash_count']}")
        print(f"  unique_request_hash_count: {row['unique_request_hash_count']}")
        largest = row["largest_block"]
        if largest is None:
            print("  largest_block:             (none)")
        else:
            print("  largest_block:")
            print(f"    block_id:                {largest['block_id']}")
            print(f"    block_ordinal:           {largest['block_ordinal']}")
            print(f"    pair_count:              {largest['pair_count']}")
            print(f"    unique_candidate_count:  {largest['unique_candidate_count']}")
            print(f"    evidence_item_count:     {largest['evidence_item_count']}")
            print(f"    pair_contexts_json_bytes: {largest['pair_contexts_json_bytes']}")
            print(f"    rendered_prompt_bytes:   {largest['rendered_prompt_bytes']}")
            print(f"    first_pair:              {_fmt_pair(largest['first_pair'])}")
            print(f"    last_pair:               {_fmt_pair(largest['last_pair'])}")
        print()


def _print_request_shape(prep, domain: str, prefix: str) -> None:
    policy = prep.packing_policy
    print(
        f"=== REQUEST SHAPE (real StructuredGenerationRequest, policy "
        f"{policy.name} {policy.max_pairs_per_block}/{policy.max_candidates_per_block}) ==="
    )
    requests = prep.structured_requests
    print(f"structured_request_count:  {len(requests)}")
    if not requests:
        print(f"  (no {domain} semantic pairs -> no requests)")
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
        print(f"  block_ordinal:                 {block.block_ordinal}")
        print(f"  pair_count:                    {block.pair_count}")
        print(f"  unique_candidate_count:        {len(block.candidate_refs)}")
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
    name = row["packing_name"]
    max_pairs = row["max_pairs_per_block"]
    max_candidates = row["max_candidates_per_block"]
    block_count = row["block_count"]
    if total_pairs == 0:
        return block_count == 0, f"{name} empty corpus mismatch"
    if row["total_pairs_in_blocks"] != total_pairs:
        return False, f"{name} total pairs in blocks != semantic pair count (pair loss/dup)"
    if row["block_count"] == 0:
        return False, f"{name} produced no blocks for {total_pairs} pairs"
    # Both independent limits must hold for every block (min/median/p95/max).
    for metric, limit, label in (
        ("pairs_per_block", max_pairs, "pairs/block"),
        ("unique_candidates_per_block", max_candidates, "candidates/block"),
    ):
        if row[metric]["max"] > limit:
            return False, f"{name} {label} exceeds {label} limit {limit}"
    # Request hashes: one per block, all unique.
    if row["request_hash_count"] != block_count:
        return False, f"{name} request_hash_count != block_count"
    if row["unique_request_hash_count"] != block_count:
        return False, f"{name} unique_request_hash_count != block_count (duplicate request)"
    return True, "ok"


def _domain_checks(
    prep,
    prep_again,
    packing_audit: tuple[dict, ...],
    *,
    domain: str,
    block_prefix: str,
    user_prompt_prefix: str,
    before: dict,
    after: dict,
    candidate_count: int | None = None,
    total_pairs: int | None = None,
    auto_same: int | None = None,
) -> list[tuple[str, bool]]:
    from short_drama.artifacts.canonical import canonical_json_bytes
    from short_drama.llm.models import StructuredGenerationRequest

    checks: list[tuple[str, bool]] = []

    # Read-only: the run tree fingerprint is unchanged.
    checks.append((f"{domain}: read-only (run tree fingerprint unchanged)", before == after))

    # Deterministic: a second preparation is byte-identical (block ids + hashes).
    checks.append(
        (
            f"{domain}: deterministic (block ids + request hashes stable)",
            (
                tuple(b.block_id for b in prep.blocks)
                == tuple(b.block_id for b in prep_again.blocks)
                and prep.semantic_request_hashes == prep_again.semantic_request_hashes
            ),
        )
    )

    # Real requests: every request is a real StructuredGenerationRequest.
    checks.append(
        (
            f"{domain}: real requests (StructuredGenerationRequest objects)",
            all(isinstance(r, StructuredGenerationRequest) for r in prep.structured_requests),
        )
    )

    # Zero provider: the canonical request material carries no endpoint /
    # transport / credential / routing identity.
    neutral = all(
        not (_FORBIDDEN_REQUEST_KEYS & _request_material_keys(r))
        for r in prep.structured_requests
    )
    checks.append(
        (f"{domain}: zero-provider (request material is provider-neutral)", neutral)
    )

    # Block id shape: every block id is <prefix> + 20 hex chars.
    hex_len = len(block_prefix) + 20
    block_id_ok = all(
        block.block_id.startswith(block_prefix) and len(block.block_id) == hex_len
        for block in prep.blocks
    )
    checks.append(
        (f"{domain}: block id shape ({block_prefix} + 20 hex)", block_id_ok)
    )

    # Pair-contexts JSON is deterministic canonical JSON (re-canonicalizes).
    canonical_ok = all(
        canonical_json_bytes([dict(pc) for pc in block.pair_contexts]).decode("utf-8")
        == block.pair_contexts_json
        for block in prep.blocks
    )
    checks.append(
        (f"{domain}: pair_contexts_json is deterministic canonical JSON", canonical_ok)
    )

    # The rendered user prompt carries the block id (variable 1) and the
    # pair contexts (variable 2). render_prompt already enforces that these are
    # EXACTLY the two frozen required variables.
    user_ok = all(
        request.rendered_prompt.user_text.startswith(f"{user_prompt_prefix}{block.block_id}")
        and block.pair_contexts_json in request.rendered_prompt.user_text
        for block, request in zip(prep.blocks, prep.structured_requests)
    )
    checks.append(
        (f"{domain}: rendered user prompt carries block_id + pair contexts", user_ok)
    )

    # Semantic-stream boundary: auto_same pairs are excluded from the requests.
    total_in_blocks = sum(block.pair_count for block in prep.blocks)
    checks.append(
        (
            f"{domain}: auto_same pairs excluded (blocks carry only semantic pairs)",
            total_in_blocks == prep.semantic_pair_count,
        )
    )

    # Pair-local selectors: each endpoint packet's evidence selectors are exactly
    # L0..L{n-1} (left) / R0..R{n-1} (right), in the exact indexed A5B evidence
    # order, and no block-wide pool exists.
    selector_ok = True
    for block in prep.blocks:
        for pc in block.pair_contexts:
            for side, prefix in (("left", "L"), ("right", "R")):
                selectors = [ev["selector"] for ev in pc[side]["evidence"]]
                expected = [f"{prefix}{idx}" for idx in range(len(pc[side]["evidence"]))]
                if selectors != expected:
                    selector_ok = False
    checks.append(
        (
            f"{domain}: pair-local evidence selectors (L0.. / R0.., no block pool)",
            selector_ok,
        )
    )

    # Packing candidates are internally consistent (both limits + no pair loss +
    # unique request hashes).
    for row in packing_audit:
        ok, _ = _packing_row_consistent(row, prep.semantic_pair_count)
        checks.append((f"{domain}: packing {row['packing_name']} consistent", ok))

    # Exact Alice counts (only when provided).
    if candidate_count is not None:
        checks.append(
            (
                f"{domain}: exact Alice candidate count",
                candidate_count == (
                    len(prep.planning_result.index.events)
                    if domain == "event"
                    else len(prep.planning_result.index.relationships)
                ),
            )
        )
    if total_pairs is not None:
        checks.append(
            (
                f"{domain}: exact Alice total pair count",
                total_pairs == (
                    prep.total_event_pair_count
                    if domain == "event"
                    else prep.total_relationship_pair_count
                ),
            )
        )
    if auto_same is not None:
        checks.append(
            (
                f"{domain}: exact Alice auto_same count",
                auto_same == prep.auto_same_pair_count,
            )
        )

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
) -> int:
    consolidation_profile = load_consolidation_profile(consolidation_profile_path)
    event_profile = load_event_semantic_profile()
    relationship_profile = load_relationship_semantic_profile()
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

    # The request-shape section uses explicit audit policies (P2), NOT
    # production defaults (event/relationship-semantic-packing are not yet
    # frozen).
    event_prep = build_event_semantic_preparation(
        result,
        consolidation_profile,
        event_profile,
        prompts=prompts,
        packing_policy=EVENT_REQUEST_SHAPE_POLICY,
    )
    event_audit = build_event_packing_audit(
        result, consolidation_profile, event_profile, prompts=prompts
    )
    event_prep_again = build_event_semantic_preparation(
        result,
        consolidation_profile,
        event_profile,
        prompts=prompts,
        packing_policy=EVENT_REQUEST_SHAPE_POLICY,
    )

    relationship_prep = build_relationship_semantic_preparation(
        result,
        consolidation_profile,
        relationship_profile,
        prompts=prompts,
        packing_policy=RELATIONSHIP_REQUEST_SHAPE_POLICY,
    )
    relationship_audit = build_relationship_packing_audit(
        result, consolidation_profile, relationship_profile, prompts=prompts
    )
    relationship_prep_again = build_relationship_semantic_preparation(
        result,
        consolidation_profile,
        relationship_profile,
        prompts=prompts,
        packing_policy=RELATIONSHIP_REQUEST_SHAPE_POLICY,
    )

    after = _fingerprint_dir(root)

    print("=== A5D EVENT + RELATIONSHIP ZERO-PROVIDER REQUEST-SHAPE AUDIT ===")
    print(f"project:              {project_id}")
    print(f"document:             {document_id}")
    print(f"reconciliation:       {reconciliation_profile_id}")
    print()

    for label, prep, audit, shape_policy in (
        ("EVENT", event_prep, event_audit, EVENT_REQUEST_SHAPE_POLICY),
        ("RELATIONSHIP", relationship_prep, relationship_audit, RELATIONSHIP_REQUEST_SHAPE_POLICY),
    ):
        domain = "event" if label == "EVENT" else "relationship"
        print(f"==================== {label} ====================")
        _print_identity(project_id, document_id, reconciliation_profile_id, prep)
        _print_candidate_universe(prep, domain)
        _print_packing(audit)
        _print_request_shape(
            prep,
            domain,
            EVENT_USER_PROMPT_PREFIX if domain == "event" else RELATIONSHIP_USER_PROMPT_PREFIX,
        )

    print("=== AUDIT CHECKS ===")
    checks = _domain_checks(
        event_prep,
        event_prep_again,
        event_audit,
        domain="event",
        block_prefix="a5eblk_",
        user_prompt_prefix=EVENT_USER_PROMPT_PREFIX,
        before=before,
        after=after,
        candidate_count=ALICE_EVENT_CANDIDATE_COUNT,
        total_pairs=ALICE_EVENT_TOTAL_PAIR_COUNT,
        auto_same=ALICE_EVENT_AUTO_SAME_COUNT,
    )
    checks += _domain_checks(
        relationship_prep,
        relationship_prep_again,
        relationship_audit,
        domain="relationship",
        block_prefix="a5rblk_",
        user_prompt_prefix=RELATIONSHIP_USER_PROMPT_PREFIX,
        before=before,
        after=after,
        candidate_count=ALICE_RELATIONSHIP_CANDIDATE_COUNT,
        total_pairs=ALICE_RELATIONSHIP_TOTAL_PAIR_COUNT,
        auto_same=ALICE_RELATIONSHIP_AUTO_SAME_COUNT,
    )
    all_pass = True
    for label_check, passed in checks:
        all_pass = all_pass and passed
        print(f"  [{'PASS' if passed else 'FAIL'}] {label_check}")
    print()

    if not all_pass:
        print("A5D AUDIT RESULT: FAIL", file=sys.stderr)
        return 2
    print("A5D AUDIT RESULT: PASS (zero-provider, read-only)")
    print("A5D-A complete: event + relationship semantic preparation + deterministic")
    print("packing audit + real StructuredGenerationRequest rendering. Still")
    print("zero-provider and read-only: no LLM call, no A5D-B response handling, no CURRENT.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="A5D-A zero-provider event + relationship request-shape audit"
    )
    parser.add_argument("--runs-root", default=str(DEFAULT_RUNS_ROOT))
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--document", default=DEFAULT_DOCUMENT)
    parser.add_argument("--profile", default=DEFAULT_PROFILE)
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
        )
    except (ConsolidationCurrentMissingError, StoryIntegrityError) as exc:
        print(f"A5D AUDIT RESULT: FAIL ({type(exc).__name__}: {exc})", file=sys.stderr)
        return 2
    except Exception as exc:  # pragma: no cover - defensive (unexpected)
        print(f"A5D AUDIT RESULT: ERROR ({type(exc).__name__}: {exc})", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
