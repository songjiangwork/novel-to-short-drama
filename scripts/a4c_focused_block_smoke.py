"""A4C focused real-model block smoke (Issue #39).

Rebuilds A4B planning + A4C preparation from the real A3 corpus (ZERO provider
calls), locates one semantic block, and runs a focused, NON-publishing
real-model semantic smoke for that single block using the tracked
reconciliation profile (prompt v2) + the real production request construction +
the strict A4C endpoint-evidence validator.

This mirrors the per-block loop of ``resolve_semantic_ambiguity`` (real request,
real provenance verification, real typed load, real strict validator) but drives
ONLY the located block and NEVER publishes an A4 CURRENT or persists anything.

It does NOT:
  * publish an A4 CURRENT
  * persist artifacts
  * run the full corpus (only the located block)
  * modify any production code / validator / schema / blocking / retry policy

Usage:
    python scripts/a4c_focused_block_smoke.py \\
        [--block-id a4blk_f1f2d8217f337e3b5927] \\
        [--runtime-config profiles/llm_local.yaml] \\
        [--reconciliation-profile profiles/entity_reconciliation_v1.yaml] \\
        [--llm-profile profiles/entity_reconciliation_llm_v1.yaml]

Exit code 0 on PASS, 2 on FAIL.
"""

from __future__ import annotations

import argparse
import json
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
    ReconciliationDecisionPayload,
    load_story_extraction_profile,
    plan_reconciliation,
    resolve_current_a3_reconciliation_inputs,
)
from short_drama.story.reconciliation import (
    load_entity_reconciliation_profile,
)
from short_drama.story.reconciliation_semantic import (
    _convert_to_decision,
    _validate_block_payload,
    _verify_provenance,
    prepare_semantic_resolution,
)
from short_drama.story.service import DOCUMENT_ID, _load_project, _load_profile, _stores

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_BLOCK_ID = "a4blk_f1f2d8217f337e3b5927"
DEFAULT_RUNTIME_CONFIG = "profiles/llm_local.yaml"
DEFAULT_RECONCILIATION_PROFILE = "profiles/entity_reconciliation_v1.yaml"
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
# Focused single-block smoke (real request + strict validator, no publish)
# ---------------------------------------------------------------------------


def analyze_evidence(block, payload) -> dict:
    """Independent per-pair endpoint-membership audit of the payload evidence."""
    candidate_evidence = {
        packet["candidate_ref"]: {
            (ev["paragraph_id"], ev["role"], ev["strength"], ev["excerpt"])
            for ev in packet["evidence_refs"]
        }
        for packet in json.loads(block.candidate_packets_json)
    }
    all_block_evidence = set()
    for ref in candidate_evidence:
        all_block_evidence |= candidate_evidence[ref]

    refs_exact = True
    third_candidate = False
    out_of_block = False
    for i, item in enumerate(payload.decisions):
        expected_left, expected_right = block.pair_plans[i].left_candidate_ref, block.pair_plans[i].right_candidate_ref
        if item.left_candidate_ref != expected_left or item.right_candidate_ref != expected_right:
            refs_exact = False
        valid = candidate_evidence[item.left_candidate_ref] | candidate_evidence[item.right_candidate_ref]
        for ev in item.evidence_refs:
            ev_tuple = (ev.paragraph_id, ev.role, ev.strength, ev.excerpt)
            if ev_tuple not in valid:
                if ev_tuple in all_block_evidence:
                    third_candidate = True
                else:
                    out_of_block = True
    return {
        "refs_exact": refs_exact,
        "third_candidate": third_candidate,
        "out_of_block": out_of_block,
        "decision_count": len(payload.decisions),
    }


def run_focused_block(
    reconciliation_profile,
    semantic_profile,
    preparation,
    block_index: int,
    client: OpenAICompatibleLLMClient,
) -> dict:
    """Drive ONLY the located block through the real per-block A4C loop."""
    block = preparation.blocks[block_index]
    request = preparation.structured_requests[block_index]
    rendered_prompt = request.rendered_prompt
    output_schema = request.output_schema
    candidate_packets = json.loads(block.candidate_packets_json)
    max_rounds = reconciliation_profile.max_generation_rounds

    generation_calls = 0
    rounds = 0
    decisions = []
    provenance = None
    last_failure = ""
    final_payload = None
    is_valid = False
    round_summaries = []

    for round_number in range(1, max_rounds + 1):
        rounds = round_number
        generation_calls += 1
        result = client.generate_structured(rendered_prompt, output_schema, semantic_profile)
        _verify_provenance(result.provenance, request, rendered_prompt, output_schema, semantic_profile)
        try:
            payload = ReconciliationDecisionPayload.from_dict(result.parsed_json)
        except ReconciliationModelError:
            last_failure = "typed payload load failed"
            round_summaries.append((round_number, "typed-load-failed", []))
            continue
        final_payload = payload
        is_valid, failure_detail = _validate_block_payload(
            payload, block.pair_plans, candidate_packets
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
                        "evidence": [
                            {
                                "paragraph_id": ev.paragraph_id,
                                "role": ev.role,
                                "strength": ev.strength,
                                "excerpt": ev.excerpt,
                            }
                            for ev in item.evidence_refs
                        ],
                    }
                    for item in payload.decisions
                ],
            )
        )
        if not is_valid:
            last_failure = failure_detail
            continue
        for item in payload.decisions:
            decisions.append(
                _convert_to_decision(
                    item,
                    request_hash=request.request_hash,
                    prompt_id=rendered_prompt.prompt_id,
                    prompt_version=rendered_prompt.prompt_version,
                    provenance=result.provenance,
                )
            )
        provenance = result.provenance
        break

    audit = analyze_evidence(block, final_payload) if final_payload is not None else {
        "refs_exact": None, "third_candidate": None, "out_of_block": None, "decision_count": 0
    }

    return {
        "block_id": block.block_id,
        "pair_count": len(block.pair_plans),
        "candidate_count": len(candidate_packets),
        "generation_calls": generation_calls,
        "rounds": rounds,
        "is_valid": is_valid,
        "last_failure": last_failure,
        "decision_count": len(decisions),
        "prompt_id": reconciliation_profile.prompt_id,
        "prompt_version": reconciliation_profile.prompt_version,
        "request_hash": request.request_hash,
        "refs_exact": audit["refs_exact"],
        "third_candidate": audit["third_candidate"],
        "out_of_block": audit["out_of_block"],
        "endpoint_audit_decision_count": audit["decision_count"],
        "generation_provenance": provenance,
        "round_summaries": round_summaries,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="A4C focused real-model block smoke (Issue #39)")
    parser.add_argument("--block-id", default=DEFAULT_BLOCK_ID)
    parser.add_argument("--runtime-config", default=DEFAULT_RUNTIME_CONFIG)
    parser.add_argument("--debug", action="store_true", help="Print per-round decision evidence")
    args = parser.parse_args()

    # 1. Zero-provider rebuild.
    reconciliation_profile, semantic_profile, planning_result, preparation = build_preparation()

    semantic_pairs = sum(
        1 for p in planning_result.pair_plans if p.state == "needs_semantic_decision"
    )
    print("=== A4C focused block smoke (Issue #39) ===")
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
            preparation,
            block_index,
            client,
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
    print(f"third_candidate_evidence: {res['third_candidate']}")
    print(f"out_of_block_evidence  : {res['out_of_block']}")
    print(f"final strict semantic validation: {'PASS' if res['is_valid'] else 'FAIL'}")
    if not res["is_valid"]:
        print(f"last_failure           : {res['last_failure']}")

    if args.debug:
        print("\n--- per-round decision evidence (debug) ---")
        for round_number, status, items in res["round_summaries"]:
            print(f"round {round_number}: {status}")
            for idx, item in enumerate(items):
                print(
                    f"  pair {idx}: {item['decision']}  "
                    f"{item['left']} <-> {item['right']}"
                )
                for ev in item["evidence"]:
                    excerpt = ev["excerpt"]
                    excerpt = excerpt if excerpt is None else excerpt[:70]
                    print(
                        f"     - {ev['paragraph_id']} / {ev['role']} / "
                        f"{ev['strength']} / {excerpt!r}"
                    )
    if res["generation_provenance"] is not None:
        prov = res["generation_provenance"]
        print(f"provider_family/model  : {prov.provider_family}/{prov.model}")
        print(f"request_hash           : {res['request_hash'][:40]}...")

    # Acceptance: block PASS, no third-candidate evidence, no out-of-block evidence,
    # every pair preserves exact refs.
    failures = []
    if not res["is_valid"]:
        failures.append("block did not pass strict semantic validation")
    if res["third_candidate"]:
        failures.append("third-candidate evidence occurred")
    if res["out_of_block"]:
        failures.append("out-of-block evidence occurred")
    if res["refs_exact"] is not True:
        failures.append("a pair did not preserve exact left/right refs")

    print("\n" + ("SMOKE PASS" if not failures else "SMOKE FAIL"))
    for f in failures:
        print(f"  - {f}")
    sys.exit(0 if not failures else 2)


if __name__ == "__main__":
    main()
