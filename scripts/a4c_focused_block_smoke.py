"""A4C focused real-model block smoke (Issue #41).

Rebuilds A4B planning + A4C preparation from the real A3 corpus (ZERO provider
calls), locates one semantic block, and runs a focused, NON-publishing
real-model semantic smoke for that single block using the tracked reconciliation
profile v2 (prompt v3) + the real production request construction + the strict
A4C pair-local evidence-selector validator.

The provider cites evidence only by pair-local selectors (L0/L1/... for the
decision's own left endpoint, R0/R1/... for its right endpoint). A4C validates
each selector against that exact pair and resolves it to the exact endpoint
EvidenceRef. This smoke drives ONLY the located block and NEVER publishes an A4
CURRENT or persists anything.

It does NOT:
  * publish an A4 CURRENT
  * persist artifacts
  * run the full corpus (only the located block)
  * modify any production code / validator / schema / blocking / retry policy

Usage:
    python scripts/a4c_focused_block_smoke.py \\
        [--block-id a4blk_f1f2d8217f337e3b5927] \\
        [--runtime-config profiles/llm_local.yaml] \\
        [--reconciliation-profile profiles/entity_reconciliation_v2.yaml] \\
        [--llm-profile profiles/entity_reconciliation_llm_v1.yaml]

Exit code 0 on PASS, 2 on FAIL.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from short_drama.llm import (
    LLMError,
    OpenAICompatibleLLMClient,
    load_runtime_config,
    load_semantic_profile,
)
from short_drama.story import (
    ReconciliationInputSnapshot,
    ReconciliationModelError,
    ReconciliationSelectorDecisionPayload,
    load_story_extraction_profile,
    plan_reconciliation,
    resolve_current_a3_reconciliation_inputs,
)
from short_drama.story.reconciliation import (
    load_entity_reconciliation_profile,
)
from short_drama.story.reconciliation_semantic import (
    _block_endpoint_evidence,
    _convert_to_decision,
    _validate_selector_block_payload,
    _verify_provenance,
    prepare_semantic_resolution,
)
from short_drama.story.service import DOCUMENT_ID, _load_project, _load_profile, _stores

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_BLOCK_ID = "a4blk_f1f2d8217f337e3b5927"
DEFAULT_RUNTIME_CONFIG = "profiles/llm_local.yaml"
DEFAULT_RECONCILIATION_PROFILE = "profiles/entity_reconciliation_v2.yaml"
DEFAULT_LLM_PROFILE = "profiles/entity_reconciliation_llm_v1.yaml"


# ---------------------------------------------------------------------------
# Zero-provider rebuild of A4B planning + A4C preparation from the real A3 corpus
# ---------------------------------------------------------------------------


def build_preparation():
    """Resolve the real A3 CURRENT set and build A4B + A4C prep (no provider)."""
    project_path = REPO_ROOT / "examples" / "a3e_real_novel" / "project.yaml"
    _, project = _load_project(project_path)
    project_id = project["project_id"]
    chunk_profile = _load_profile("profiles/story_analysis_v1.yaml")
    extraction_profile = load_story_extraction_profile(
        "profiles/story_extraction_v1.yaml"
    )
    reconciliation_profile = load_entity_reconciliation_profile(
        DEFAULT_RECONCILIATION_PROFILE
    )
    semantic_profile = load_semantic_profile(DEFAULT_LLM_PROFILE)
    store, pointers = _stores("runs/a4e_real_novel", project_id)

    current_a3 = resolve_current_a3_reconciliation_inputs(
        store,
        pointers,
        project_id=project_id,
        document_id=DOCUMENT_ID,
        chunk_profile=chunk_profile,
        extraction_profile=extraction_profile,
    )
    story = current_a3.story_snapshot
    snapshot = ReconciliationInputSnapshot(
        source_document=story.source_document,
        source_document_ref=story.source_document_ref,
        chunk_manifest=story.chunk_manifest,
        source_chunks=story.source_chunks,
        source_chunk_refs=story.source_chunk_refs,
        candidate_extractions=current_a3.candidate_extractions,
        candidate_extraction_refs=current_a3.candidate_extraction_refs,
        a3_validation_report_refs=current_a3.candidate_validation_report_refs,
    )
    planning_result = plan_reconciliation(snapshot)
    preparation = prepare_semantic_resolution(
        planning_result,
        reconciliation_profile,
        semantic_profile,
    )
    return reconciliation_profile, semantic_profile, planning_result, preparation


# ---------------------------------------------------------------------------
# Independent per-pair selector audit (pair-scoped, Python-owned)
# ---------------------------------------------------------------------------


def analyze_selectors(block, payload, endpoint_evidence) -> dict:
    """Independent per-pair audit of the provider's evidence selectors.

    Confirms (against the exact pair's own endpoint evidence):
      * refs_exact        -- each decision's left/right refs match the pair plan;
      * all_selectors_ok  -- every selector is L<index>/R<index>, in range for
                             the pair's own endpoint, and not a duplicate.
    A selector can never reach a third candidate: L -> left endpoint, R -> right
    endpoint only.
    """
    from short_drama.story.reconciliation_semantic import _EVIDENCE_SELECTOR_RE

    refs_exact = True
    all_selectors_ok = True
    detail: list[str] = []
    for i, item in enumerate(payload.decisions):
        expected_left = block.pair_plans[i].left_candidate_ref
        expected_right = block.pair_plans[i].right_candidate_ref
        if item.left_candidate_ref != expected_left or item.right_candidate_ref != expected_right:
            refs_exact = False
        left_ev, right_ev = endpoint_evidence[i]
        seen: set[str] = set()
        for sel in item.evidence_selectors:
            ok = _EVIDENCE_SELECTOR_RE.fullmatch(sel) is not None
            if ok:
                idx = int(sel[1:])
                ev = left_ev if sel[0] == "L" else right_ev
                if idx >= len(ev):
                    ok = False
                if sel in seen:
                    ok = False
                seen.add(sel)
            if not ok:
                all_selectors_ok = False
                detail.append(f"pair {i}: {sel!r} invalid")
    return {
        "refs_exact": refs_exact,
        "all_selectors_ok": all_selectors_ok,
        "detail": detail,
        "decision_count": len(payload.decisions),
    }


# ---------------------------------------------------------------------------
# Focused single-block smoke (real request + strict validator, no publish)
# ---------------------------------------------------------------------------


def run_focused_block(
    reconciliation_profile,
    semantic_profile,
    planning_result,
    preparation,
    block_index: int,
    client: OpenAICompatibleLLMClient,
    debug: bool = False,
) -> dict:
    """Drive ONLY the located block through the real per-block A4C loop."""
    block = preparation.blocks[block_index]
    request = preparation.structured_requests[block_index]
    rendered_prompt = request.rendered_prompt
    output_schema = request.output_schema
    endpoint_evidence = _block_endpoint_evidence(planning_result, list(block.pair_plans))
    max_rounds = reconciliation_profile.max_generation_rounds

    generation_calls = 0
    rounds = 0
    decisions = []
    provenance = None
    last_failure = ""
    final_payload = None
    resolved_evidence: list = []
    is_valid = False
    round_summaries = []

    for round_number in range(1, max_rounds + 1):
        rounds = round_number
        generation_calls += 1
        result = client.generate_structured(rendered_prompt, output_schema, semantic_profile)
        _verify_provenance(result.provenance, request, rendered_prompt, output_schema, semantic_profile)
        try:
            payload = ReconciliationSelectorDecisionPayload.from_dict(result.parsed_json)
        except ReconciliationModelError:
            last_failure = "typed payload load failed"
            round_summaries.append((round_number, "typed-load-failed", []))
            continue
        final_payload = payload
        is_valid, failure_detail, resolved_evidence = _validate_selector_block_payload(
            payload, block.pair_plans, endpoint_evidence
        )
        round_summaries.append(
            (
                round_number,
                "valid" if is_valid else failure_detail,
                [
                    {
                        "decision": item.decision,
                        "left": item.left_candidate_ref,
                        "right": item.right_candidate_ref,
                        "evidence_selectors": list(item.evidence_selectors),
                        "resolved_evidence": [
                            {
                                "paragraph_id": ev.paragraph_id,
                                "role": ev.role,
                                "strength": ev.strength,
                                "excerpt": ev.excerpt,
                            }
                            for ev in (
                                resolved_evidence[i]
                                if is_valid and i < len(resolved_evidence)
                                else ()
                            )
                        ],
                    }
                    for i, item in enumerate(payload.decisions)
                ],
            )
        )
        if not is_valid:
            last_failure = failure_detail
            resolved_evidence = []
            continue
        for item, resolved in zip(payload.decisions, resolved_evidence):
            decisions.append(
                _convert_to_decision(
                    left_ref=item.left_candidate_ref,
                    right_ref=item.right_candidate_ref,
                    decision=item.decision,
                    reason_zh=item.reason_zh,
                    evidence_refs=resolved,
                    request_hash=request.request_hash,
                    prompt_id=rendered_prompt.prompt_id,
                    prompt_version=rendered_prompt.prompt_version,
                    provenance=result.provenance,
                )
            )
        provenance = result.provenance
        break

    audit = analyze_selectors(block, final_payload, endpoint_evidence) if final_payload is not None else {
        "refs_exact": None, "all_selectors_ok": None, "detail": [], "decision_count": 0
    }

    return {
        "block_id": block.block_id,
        "pair_count": len(block.pair_plans),
        "candidate_count": len(block.candidate_refs),
        "generation_calls": generation_calls,
        "rounds": rounds,
        "is_valid": is_valid,
        "last_failure": last_failure,
        "decision_count": len(decisions),
        "prompt_id": reconciliation_profile.prompt_id,
        "prompt_version": reconciliation_profile.prompt_version,
        "request_hash": request.request_hash,
        "refs_exact": audit["refs_exact"],
        "all_selectors_ok": audit["all_selectors_ok"],
        "selector_detail": audit["detail"],
        "audit_decision_count": audit["decision_count"],
        "generation_provenance": provenance,
        "round_summaries": round_summaries,
        "debug": debug,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="A4C focused real-model block smoke (Issue #41)")
    parser.add_argument("--block-id", default=DEFAULT_BLOCK_ID)
    parser.add_argument("--runtime-config", default=DEFAULT_RUNTIME_CONFIG)
    parser.add_argument("--debug", action="store_true", help="Print per-round selector / resolved evidence")
    args = parser.parse_args()

    # 1. Zero-provider rebuild.
    reconciliation_profile, semantic_profile, planning_result, preparation = build_preparation()

    semantic_pairs = sum(
        1 for p in planning_result.pair_plans if p.state == "needs_semantic_decision"
    )
    print("=== A4C focused block smoke (Issue #41) ===")
    print(f"total semantic pairs : {semantic_pairs}")
    print(f"total blocks         : {len(preparation.blocks)}")
    print(f"prompt_id/version    : {preparation.prompt_id} v{preparation.prompt_version}")
    print(f"prompt_content_hash  : {preparation.prompt_content_hash}")
    print(f"reconciliation_profile_hash: {reconciliation_profile.profile_hash}")

    matches = [i for i, b in enumerate(preparation.blocks) if b.block_id == args.block_id]
    if not matches:
        print(f"\nFAIL: block id not found: {args.block_id}")
        sys.exit(2)
    block_index = matches[0]
    print(f"located block index  : {block_index}")

    # 2. Focused single-block real-model smoke (non-publishing).
    runtime_config = load_runtime_config(args.runtime_config)
    client = OpenAICompatibleLLMClient(runtime_config)
    print(f"runtime: {runtime_config.base_url} model={runtime_config.request_model}")

    try:
        res = run_focused_block(
            reconciliation_profile,
            semantic_profile,
            planning_result,
            preparation,
            block_index,
            client,
            debug=args.debug,
        )
    except LLMError as exc:
        print(f"\nFAIL: LLM transport error: {exc}")
        sys.exit(2)

    # 3. Report.
    print("\n--- Focused block result ---")
    print(f"block_id              : {res['block_id']}")
    print(f"pair_count            : {res['pair_count']}")
    print(f"candidate_count       : {res['candidate_count']}")
    print(f"prompt_id/version     : {res['prompt_id']} v{res['prompt_version']}")
    print(f"semantic generation calls: {res['generation_calls']}")
    print(f"semantic rounds        : {res['rounds']}")
    print(f"decision_count         : {res['decision_count']}")
    print(f"refs_exact             : {res['refs_exact']}")
    print(f"all_selectors_ok       : {res['all_selectors_ok']}")
    if res["selector_detail"]:
        print(f"selector_detail        : {res['selector_detail']}")
    print(
        f"generation_provenance : {res['generation_provenance'].provider_family}"
        f"/{res['generation_provenance'].model}"
        if res["generation_provenance"]
        else "generation_provenance : (none)"
    )
    print(f"request_hash           : {res['request_hash']}")

    if args.debug and res.get("round_summaries"):
        print("\n--- Per-round summaries ---")
        for round_number, status, items in res["round_summaries"]:
            print(f"  round {round_number}: {status}")
            for it in items:
                print(f"    {it['decision']:>17}  {it['left']}  <->  {it['right']}")
                print(f"        selectors: {it['evidence_selectors']}")
                for ev in it["resolved_evidence"]:
                    excerpt = ev["excerpt"]
                    if excerpt is None:
                        excerpt = "<null excerpt>"
                    elif len(excerpt) > 40:
                        excerpt = excerpt[:40] + "..."
                    print(
                        f"        evidence: pid={ev['paragraph_id']} "
                        f"role={ev['role']} strength={ev['strength']} excerpt={excerpt!r}"
                    )

    ok = (
        res["is_valid"]
        and res["refs_exact"] is True
        and res["all_selectors_ok"] is True
        and res["decision_count"] == res["pair_count"]
        and res["generation_calls"] >= 1
    )
    print(f"\n{'PASS' if ok else 'FAIL'}: "
          f"block={res['block_id']} calls={res['generation_calls']} "
          f"rounds={res['rounds']} decisions={res['decision_count']} "
          f"is_valid={res['is_valid']} refs_exact={res['refs_exact']} "
          f"all_selectors_ok={res['all_selectors_ok']}")
    if not ok and res.get("last_failure"):
        print(f"last_failure: {res['last_failure']}")
    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()
