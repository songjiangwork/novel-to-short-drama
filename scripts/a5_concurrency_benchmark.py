#!/usr/bin/env python3
"""Manual, read-only A5 Event-block concurrency benchmark.

This intentionally bypasses A5 finalization and persistence.  It is a manual
high-cost provider checkpoint; it does not start or probe a provider.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import threading
import time
from pathlib import Path

from short_drama.artifacts import FileArtifactStore, canonical_json_bytes
from short_drama.foundation import FilePointerStore
from short_drama.llm import (
    OpenAICompatibleLLMClient,
    PromptRegistry,
    load_runtime_config,
    load_semantic_profile,
)
from short_drama.paths import REPO_ROOT
from short_drama.story.consolidation import load_consolidation_profile
from short_drama.story.consolidation_planning import build_consolidation_planning
from short_drama.story.consolidation_semantic import (
    EVENT_SEMANTIC_PACKING_V1,
    _attempt_event_semantic_block,
    _execute_two_stage_semantic_blocks,
    build_event_semantic_preparation,
    validate_a5_max_concurrency,
)
from short_drama.story.service import DOCUMENT_ID, _load_project


def _tree_fingerprint(root: Path) -> str:
    """Hash the relevant run tree without changing it (missing tree is stable)."""
    digest = hashlib.sha256()
    if not root.exists():
        digest.update(b"missing")
        return digest.hexdigest()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def _prompt_bytes(request) -> int:
    return len(canonical_json_bytes(request.rendered_prompt.to_dict()))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project")
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--reconciliation-profile-id", default="entity-reconciliation-v2")
    parser.add_argument("--consolidation-profile", default=REPO_ROOT / "profiles" / "consolidation_v1.yaml")
    parser.add_argument("--runtime-config", default=REPO_ROOT / "profiles" / "llm_local.yaml")
    parser.add_argument("--llm-profile", default=REPO_ROOT / "profiles" / "consolidation_llm_v1.yaml")
    parser.add_argument("--max-concurrency", type=int, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    max_concurrency = validate_a5_max_concurrency(args.max_concurrency)
    _project_file, project = _load_project(args.project)
    project_id = project["project_id"]
    runs_root = Path(args.runs_root)
    relevant_tree = runs_root / project_id
    before = _tree_fingerprint(relevant_tree)
    story_root = relevant_tree / "story"
    artifact_root = story_root / "artifacts"
    pointer_root = story_root / "pointers"
    # The normal store constructors create absent roots.  A benchmark must be
    # read-only even on a malformed invocation, so require the pre-existing
    # A1--A4 run tree before constructing their read APIs.
    if not artifact_root.is_dir() or not pointer_root.is_dir():
        raise RuntimeError("existing A1--A4 artifact and pointer roots are required")
    store = FileArtifactStore(artifact_root)
    pointers = FilePointerStore(pointer_root, store)
    planning = build_consolidation_planning(
        store, pointers, project_id=project_id, document_id=DOCUMENT_ID,
        reconciliation_profile_id=args.reconciliation_profile_id,
        consolidation_profile=load_consolidation_profile(Path(args.consolidation_profile)),
    )
    profile = load_consolidation_profile(Path(args.consolidation_profile))
    semantic_profile = load_semantic_profile(args.llm_profile)
    preparation = build_event_semantic_preparation(
        planning, profile, semantic_profile,
        packing_policy=EVENT_SEMANTIC_PACKING_V1,
        prompts=PromptRegistry(REPO_ROOT / "prompts" / "story"),
    )
    selected = sorted(
        zip(preparation.blocks, preparation.structured_requests),
        key=lambda item: (-_prompt_bytes(item[1]), item[0].block_id),
    )[:8]
    client = OpenAICompatibleLLMClient(load_runtime_config(args.runtime_config))
    lock = threading.Lock()
    in_flight = 0
    max_observed_in_flight = 0
    latencies: dict[str, float] = {}
    failures: list[str] = []

    def execute_one_round(block, request, semantic_round):
        nonlocal in_flight, max_observed_in_flight
        started = time.monotonic()
        with lock:
            in_flight += 1
            max_observed_in_flight = max(max_observed_in_flight, in_flight)
        try:
            return _attempt_event_semantic_block(
                planning, semantic_profile, client, block, request,
                semantic_round=semantic_round,
            )
        except Exception as exc:
            with lock:
                failures.append(f"{block.block_ordinal}:{type(exc).__name__}: {exc}")
            raise
        finally:
            with lock:
                in_flight -= 1
                latencies[block.block_id] = time.monotonic() - started

    started = time.monotonic()
    results = ()
    try:
        results = _execute_two_stage_semantic_blocks(
            tuple(block for block, _request in selected),
            tuple(request for _block, request in selected),
            execute_one_round, llm_client=client, max_concurrency=max_concurrency,
        )
    except Exception as exc:
        # Report after the read-only fingerprint assertion, then preserve the
        # provider/domain failure for shell automation.
        with lock:
            if not failures:
                failures.append(f"{type(exc).__name__}: {exc}")
        failed = True
    else:
        failed = False
    elapsed = time.monotonic() - started
    after = _tree_fingerprint(relevant_tree)
    usage: dict[str, int] = {}
    for result in results:
        for key, value in (result.generation_provenance.usage or {}).items():
            usage[key] = usage.get(key, 0) + value
    report = {
        "selected_blocks": [
            {"block_id": block.block_id, "block_ordinal": block.block_ordinal,
             "prompt_bytes": _prompt_bytes(request)}
            for block, request in selected
        ],
        "max_concurrency": max_concurrency,
        "wall_clock_seconds": elapsed,
        "requests_completed": len(results),
        "semantic_rounds": sum(result.semantic_rounds for result in results),
        # A-I3 exposes attempts on StructuredGenerationResult, while semantic
        # block results intentionally retain only provenance.  Keep this null
        # rather than inventing a count from unrelated provider metadata.
        "provider_attempts": None,
        "usage": usage or None,
        "aggregate_tokens_per_second": (
            usage.get("total_tokens", 0) / elapsed if usage.get("total_tokens") and elapsed else None
        ),
        "per_request_latency_seconds": latencies,
        "observed_max_in_flight": max_observed_in_flight,
        "failures": failures,
        "run_tree_fingerprint_before": before,
        "run_tree_fingerprint_after": after,
        "read_only_verified": before == after,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if before != after:
        raise RuntimeError("benchmark changed the production run tree")
    if failed:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
