#!/usr/bin/env python3
"""Non-publishing real-provider acceptance gate for Issue #80 Fact prompt v2."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from short_drama.artifacts import FileArtifactStore, canonical_json_bytes
from short_drama.foundation import FilePointerStore
from short_drama.llm import GenerationExecutionOptions, OpenAICompatibleLLMClient, PromptRegistry, build_structured_request, load_runtime_config
from short_drama.paths import REPO_ROOT
from short_drama.story import FACT_SEMANTIC_PACKING_V2, build_consolidation_planning, build_fact_semantic_preparation, load_consolidation_profile, load_fact_semantic_profile
from short_drama.story.consolidation import FactSelectorDecisionPayload
from short_drama.story.consolidation_finalization import _FACT_HARD_NEGATIVES
from short_drama.story.consolidation_semantic import _attempt_fact_semantic_block, _execute_two_stage_semantic_blocks, _validate_fact_selector_block_payload, _verify_fact_provenance, load_fact_output_schema
from short_drama.story.extraction import EvidenceRef
from short_drama.story.service import DOCUMENT_ID, _load_project

TARGETS = frozenset({"CH005_C001:cand_fact_029", "CH005_C001:cand_fact_031", "CH005_C001:cand_fact_037"})
_ALLOWED_COHERENT_NONIDENTICAL = frozenset({"compatible_fact", "uncertain"})
_ALICE_MAX_CONCURRENCY = 4


def _fingerprint(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")) if root.exists() else ():
        if path.is_file():
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def _contradictions(decisions):
    """Apply the frozen Fact same-component hard-negative invariant."""
    parent = {}

    def find(ref):
        parent.setdefault(ref, ref)
        if parent[ref] != ref:
            parent[ref] = find(parent[ref])
        return parent[ref]

    def union(left, right):
        left, right = find(left), find(right)
        if left != right:
            parent[right] = left

    for decision in decisions:
        if decision.decision == "same_fact":
            union(decision.left_candidate_ref, decision.right_candidate_ref)
    return ([decision for decision in decisions if decision.decision in _FACT_HARD_NEGATIVES and find(decision.left_candidate_ref) == find(decision.right_candidate_ref)], find)


def _synthetic_endpoint(ref, statement, side, paragraph_id):
    """Build the exact production Fact endpoint-packet/evidence shape."""
    selector = "L0" if side == "left" else "R0"
    evidence = EvidenceRef(paragraph_id=paragraph_id, role="primary", strength="explicit", excerpt=statement)
    return ({
        "candidate_ref": ref,
        "chunk_id": "CH999_C001",
        "local_candidate_id": ref.rsplit(":", 1)[1],
        "fact_type": "action",
        "statement_zh": statement,
        "subject_refs": ["char_000001"],
        "object_refs": [],
        "evidence_strength": "explicit",
        "source_order_key": "000999:000001:fact:000001:" + ref,
        "evidence": [{"selector": selector, "paragraph_id": evidence.paragraph_id, "role": evidence.role, "strength": evidence.strength, "excerpt": evidence.excerpt}],
    }, evidence)


def _synthetic(client, profile):
    """Run the material-addition triangle through v2/parser/selector checks."""
    refs = ("CH999_C001:cand_fact_001", "CH999_C001:cand_fact_002", "CH999_C001:cand_fact_003")
    pairs = ((refs[0], refs[1]), (refs[0], refs[2]), (refs[1], refs[2]))
    statements = {refs[0]: "张三已经离开村庄。", refs[1]: "张三已经离开村庄，并带走了密信。", refs[2]: "张三已经离开村庄，并留下了暗号。"}
    contexts, endpoint_evidence = [], []
    for index, (left, right) in enumerate(pairs, start=1):
        left_packet, left_evidence = _synthetic_endpoint(left, statements[left], "left", f"CH999_P{index:03d}L")
        right_packet, right_evidence = _synthetic_endpoint(right, statements[right], "right", f"CH999_P{index:03d}R")
        contexts.append({"left_candidate_ref": left, "right_candidate_ref": right, "signals": ["same_fact_type", "subject_overlap"], "left": left_packet, "right": right_packet})
        endpoint_evidence.append(((left_evidence,), (right_evidence,)))
    prompt = PromptRegistry(REPO_ROOT / "prompts" / "story").load("a5.fact-consolidation", version=2)
    request = build_structured_request(
        rendered_prompt=prompt.render({"block_id": "issue80_synthetic", "pair_contexts_json": canonical_json_bytes(contexts).decode()}),
        output_schema=load_fact_output_schema(), semantic_profile=profile,
    )
    for round_number in (1, 2):
        if round_number == 1:
            result = client.generate_structured(request.rendered_prompt, request.output_schema, profile)
        else:
            result = client.generate_structured(request.rendered_prompt, request.output_schema, profile, execution_options=GenerationExecutionOptions(prompt_context_reuse="disabled"))
        _verify_fact_provenance(result.provenance, request, profile)
        try:
            payload = FactSelectorDecisionPayload.from_dict(result.parsed_json)
        except Exception:
            continue
        valid, _detail, _resolved = _validate_fact_selector_block_payload(payload, list(pairs), endpoint_evidence)
        if not valid:
            continue
        decisions = list(payload.decisions)
        contradictions, _find = _contradictions(decisions)
        violations = [decision for decision in decisions if decision.decision not in _ALLOWED_COHERENT_NONIDENTICAL]
        return {"rounds": round_number, "request_hash": request.request_hash, "decisions": decisions, "semantic_violations": violations, "contradictions": contradictions}
    raise RuntimeError("synthetic payload failed production parser/selector validation")


def _alice(client, args, profile, semantic_profile):
    """Run exact Alice blocks through the production two-stage executor."""
    _project_file, project = _load_project(args.project)
    run_root = Path(args.runs_root) / project["project_id"] / "story"
    store = FileArtifactStore(run_root / "artifacts")
    pointers = FilePointerStore(run_root / "pointers", store)
    planning = build_consolidation_planning(store, pointers, project_id=project["project_id"], document_id=DOCUMENT_ID, reconciliation_profile_id=args.reconciliation_profile_id, consolidation_profile=profile)
    preparation = build_fact_semantic_preparation(planning, profile, semantic_profile, prompts=PromptRegistry(REPO_ROOT / "prompts" / "story"), packing_policy=FACT_SEMANTIC_PACKING_V2)
    selected = [(block, request) for block, request in zip(preparation.blocks, preparation.structured_requests) if any({left, right} <= TARGETS for left, right in block.pair_refs)]
    if not selected:
        raise RuntimeError("no exact production block covers the Alice target pairs")
    blocks, requests = zip(*selected)
    results = _execute_two_stage_semantic_blocks(
        blocks, requests,
        lambda block, request, semantic_round: _attempt_fact_semantic_block(planning, semantic_profile, client, block, request, semantic_round=semantic_round),
        llm_client=client, max_concurrency=_ALICE_MAX_CONCURRENCY,
    )
    decisions, records = [], []
    for block, request, result in zip(blocks, requests, results):
        decisions.extend(result.decisions)
        for decision in result.decisions:
            if {decision.left_candidate_ref, decision.right_candidate_ref} <= TARGETS:
                records.append({"left": decision.left_candidate_ref, "right": decision.right_candidate_ref, "decision": decision.decision, "reason_zh": decision.reason_zh, "block_ordinal": block.block_ordinal, "block_id": block.block_id, "request_hash": request.request_hash, "semantic_rounds": result.semantic_rounds})
    contradictions, find = _contradictions(decisions)
    return {"target_decisions": records, "semantic_violations": [record for record in records if record["decision"] not in _ALLOWED_COHERENT_NONIDENTICAL], "same_fact_components": {ref: find(ref) for ref in TARGETS}, "contradiction_count": len(contradictions), "production_two_stage_executor": True}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-config", default=REPO_ROOT / "profiles" / "llm_local.yaml")
    parser.add_argument("--project", default=REPO_ROOT / "examples" / "a3e_real_novel" / "project.yaml")
    parser.add_argument("--runs-root", default=REPO_ROOT / "runs" / "a4e_real_novel")
    parser.add_argument("--reconciliation-profile-id", default="entity-reconciliation-v2")
    args = parser.parse_args()
    profile = load_consolidation_profile(REPO_ROOT / "profiles" / "consolidation_v1.yaml")
    semantic_profile = load_fact_semantic_profile()
    client = OpenAICompatibleLLMClient(load_runtime_config(args.runtime_config))
    root = Path(args.runs_root) / "a3e-real-novel" / "story"
    before = _fingerprint(root)
    synthetic = _synthetic(client, semantic_profile)
    alice = _alice(client, args, profile, semantic_profile)
    after = _fingerprint(root)
    report = {"prompt": "a5.fact-consolidation v2", "synthetic": {**synthetic, "decisions": [d.to_dict() for d in synthetic["decisions"]], "semantic_violations": [d.to_dict() for d in synthetic["semantic_violations"]], "contradictions": [d.to_dict() for d in synthetic["contradictions"]], "round_2_cache_isolation": True}, "alice": alice, "read_only": before == after}
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if (not synthetic["semantic_violations"] and not synthetic["contradictions"] and not alice["semantic_violations"] and alice["contradiction_count"] == 0 and before == after) else 2


if __name__ == "__main__":
    raise SystemExit(main())
