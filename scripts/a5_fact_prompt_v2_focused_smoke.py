#!/usr/bin/env python3
"""Non-publishing real-provider gate for Issue #80 Fact prompt v2.

Runs the v2 prompt through production request construction, provenance and
selector validation.  It writes neither A5 artifacts nor CURRENT pointers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from short_drama.artifacts import FileArtifactStore, canonical_json_bytes
from short_drama.foundation import FilePointerStore
from short_drama.llm import (
    OpenAICompatibleLLMClient, PromptRegistry, build_structured_request,
    load_runtime_config,
)
from short_drama.paths import REPO_ROOT
from short_drama.story import (
    FACT_SEMANTIC_PACKING_V2, build_consolidation_planning,
    build_fact_semantic_preparation, load_consolidation_profile,
    load_fact_semantic_profile,
)
from short_drama.story.consolidation import FactSelectorDecisionPayload
from short_drama.story.consolidation_finalization import _FACT_HARD_NEGATIVES
from short_drama.story.consolidation_semantic import (
    _attempt_fact_semantic_block, _validate_fact_selector_block_payload,
    _verify_fact_provenance,
)
from short_drama.story.extraction import EvidenceRef
from short_drama.story.service import DOCUMENT_ID, _load_project

TARGETS = frozenset({
    "CH005_C001:cand_fact_029", "CH005_C001:cand_fact_031", "CH005_C001:cand_fact_037",
})


def _fingerprint(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")) if root.exists() else ():
        if path.is_file():
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def _contradictions(decisions):
    parent = {}
    def find(x):
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[b] = a
    for d in decisions:
        if d.decision == "same_fact":
            union(d.left_candidate_ref, d.right_candidate_ref)
    return [d for d in decisions if d.decision in _FACT_HARD_NEGATIVES
            and find(d.left_candidate_ref) == find(d.right_candidate_ref)], parent, find


def _synthetic(client, profile):
    """One production v2 request for A/base, B/addition, C/addition triangle."""
    refs = ("CH999_C001:cand_fact_001", "CH999_C001:cand_fact_002", "CH999_C001:cand_fact_003")
    pairs = ((refs[0], refs[1]), (refs[0], refs[2]), (refs[1], refs[2]))
    statements = {
        refs[0]: "张三已经离开村庄。",
        refs[1]: "张三已经离开村庄，并带走了密信。",
        refs[2]: "张三已经离开村庄，并留下了暗号。",
    }
    contexts = [{"left_candidate_ref": l, "right_candidate_ref": r,
                 "left_fact": {"candidate_ref": l, "statement_zh": statements[l], "evidence": []},
                 "right_fact": {"candidate_ref": r, "statement_zh": statements[r], "evidence": []},
                 "signals": ["same_fact_type"]} for l, r in pairs]
    prompt = PromptRegistry(REPO_ROOT / "prompts" / "story").load("a5.fact-consolidation", version=2)
    from short_drama.story.consolidation_semantic import load_fact_output_schema
    request = build_structured_request(
        rendered_prompt=prompt.render({"block_id": "issue80_synthetic", "pair_contexts_json": canonical_json_bytes(contexts).decode()}),
        output_schema=load_fact_output_schema(), semantic_profile=profile,
    )
    for round_number in (1, 2):
        result = client.generate_structured(request.rendered_prompt, request.output_schema, profile)
        _verify_fact_provenance(result.provenance, request, profile)
        try:
            payload = FactSelectorDecisionPayload.from_dict(result.parsed_json)
        except Exception:
            continue
        valid, _detail, evidence = _validate_fact_selector_block_payload(payload, list(pairs), [((), ())] * 3)
        if valid:
            decisions = list(payload.decisions)
            bad, _parent, _find = _contradictions(decisions)
            return {"rounds": round_number, "request_hash": request.request_hash,
                    "decisions": decisions, "contradictions": bad}
    raise RuntimeError("synthetic payload failed production parser/selector validation")


def _alice(client, args, profile, semantic_profile):
    project_file, project = _load_project(args.project)
    run_root = Path(args.runs_root) / project["project_id"] / "story"
    store = FileArtifactStore(run_root / "artifacts")
    pointers = FilePointerStore(run_root / "pointers", store)
    planning = build_consolidation_planning(store, pointers, project_id=project["project_id"],
        document_id=DOCUMENT_ID, reconciliation_profile_id=args.reconciliation_profile_id,
        consolidation_profile=profile)
    prep = build_fact_semantic_preparation(planning, profile, semantic_profile,
        prompts=PromptRegistry(REPO_ROOT / "prompts" / "story"), packing_policy=FACT_SEMANTIC_PACKING_V2)
    selected = [(block, request) for block, request in zip(prep.blocks, prep.structured_requests)
                if any({left, right} <= TARGETS for left, right in block.pair_refs)]
    if not selected:
        raise RuntimeError("no exact production block covers the Alice target pairs")
    decisions, records = [], []
    for block, request in selected:
        result = _attempt_fact_semantic_block(planning, semantic_profile, client, block, request, semantic_round=1)
        if result.__class__.__name__ == "_RetryableSemanticInvalid":
            result = _attempt_fact_semantic_block(planning, semantic_profile, client, block, request, semantic_round=2)
        if result.__class__.__name__ == "_RetryableSemanticInvalid":
            raise RuntimeError(f"semantic-invalid block {block.block_id}: {result.last_failure_details}")
        decisions.extend(result.decisions)
        for d in result.decisions:
            if {d.left_candidate_ref, d.right_candidate_ref} <= TARGETS:
                records.append({"left": d.left_candidate_ref, "right": d.right_candidate_ref,
                    "decision": d.decision, "reason_zh": d.reason_zh, "block_ordinal": block.block_ordinal,
                    "block_id": block.block_id, "request_hash": request.request_hash,
                    "semantic_rounds": result.semantic_rounds})
    bad, parent, find = _contradictions(decisions)
    components = {ref: find(ref) for ref in TARGETS}
    return {"target_decisions": records, "same_fact_components": components,
            "contradiction_count": len(bad)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-config", default=REPO_ROOT / "profiles" / "llm_local.yaml")
    parser.add_argument("--project", default=REPO_ROOT / "examples" / "a3e_real_novel" / "project.yaml")
    parser.add_argument("--runs-root", default=REPO_ROOT / "runs")
    parser.add_argument("--reconciliation-profile-id", default="entity-reconciliation-v2")
    args = parser.parse_args()
    profile = load_consolidation_profile(REPO_ROOT / "profiles" / "consolidation_v1.yaml")
    semantic_profile = load_fact_semantic_profile()
    client = OpenAICompatibleLLMClient(load_runtime_config(args.runtime_config))
    before = _fingerprint(Path(args.runs_root) / "a3e-real-novel" / "story")
    synthetic = _synthetic(client, semantic_profile)
    alice = _alice(client, args, profile, semantic_profile)
    after = _fingerprint(Path(args.runs_root) / "a3e-real-novel" / "story")
    report = {"prompt": "a5.fact-consolidation v2", "synthetic": {
        **synthetic, "decisions": [d.to_dict() for d in synthetic["decisions"]],
        "contradictions": [d.to_dict() for d in synthetic["contradictions"]]},
        "alice": alice, "read_only": before == after}
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if not synthetic["contradictions"] and alice["contradiction_count"] == 0 and before == after else 2


if __name__ == "__main__":
    raise SystemExit(main())
