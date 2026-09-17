"""v1.2 A3E-C — real-novel local-Qwen acceptance smoke.

This is the FINAL A3E delivery slice. It exercises the complete production
authority against a real published novel (Alice's Adventures in Wonderland)
with 3 consecutive chunks selected from the production A2 chunk profile:

    A1 SourceDocument (isolated runs root)
      -> A2 ChunkManifest (production chunk profile)
      -> A3D single-chunk extraction x3 (exact tracked profiles)
      -> A3B semantic validation (PASS required)
      -> CandidateExtraction published
      -> exact rerun (reused=True, 0 additional provider calls)
      -> semantic identity invalidation probe

The acceptance validates pipeline capability + output contract, NOT backend
identity. Any OpenAI-compatible backend that satisfies the A-I3 v2 contract
(original Qwen, uncensored Qwen, Gemma, future compatible backend) passes.

Usage:
    python scripts/a3e_real_novel_smoke.py \\
        --runtime-config profiles/llm_local.yaml \\
        --profile profiles/story_extraction_llm_v1.yaml \\
        --extraction-profile profiles/story_extraction_v1.yaml \\
        --chunk-profile profiles/story_analysis_v1.yaml

Exit code 0 on success, 2 on failure.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

from short_drama.artifacts import ArtifactRef, FileArtifactStore
from short_drama.foundation import (
    FilePointerStore,
    ValidationResult,
    load_validation_report,
)
from short_drama.llm import (
    LLMClient,
    LLMError,
    OpenAICompatibleLLMClient,
    PromptRegistry,
    load_runtime_config,
    load_semantic_profile,
)
from short_drama.paths import REPO_ROOT
from short_drama.story import (
    ChunkExtractionService,
    load_candidate_extraction,
    load_chunk_manifest,
    load_source_chunk,
    load_source_document,
    load_story_extraction_profile,
    plan_chunks_project,
    ingest_source_project,
)
from short_drama.story.persistence import chunk_pointer_id, source_pointer_id
from short_drama.story.service import DOCUMENT_ID, _current_pointer, _stores

PROJECT_PATH = REPO_ROOT / "examples" / "a3e_real_novel" / "project.yaml"
PROJECT_ID = "a3e-real-novel"
DEFAULT_RUNTIME_CONFIG = REPO_ROOT / "profiles" / "llm_local.yaml"
DEFAULT_SEMANTIC_PROFILE = REPO_ROOT / "profiles" / "story_extraction_llm_v1.yaml"
DEFAULT_EXTRACTION_PROFILE = REPO_ROOT / "profiles" / "story_extraction_v1.yaml"
DEFAULT_CHUNK_PROFILE = REPO_ROOT / "profiles" / "story_analysis_v1.yaml"
PROMPTS_DIR = REPO_ROOT / "prompts" / "story"
SMOKE_CHUNK_COUNT = 3


# ---------------------------------------------------------------------------
# Counting LLM client
# ---------------------------------------------------------------------------


class CountingLLMClient(LLMClient):
    """Tiny counting wrapper around the provider-neutral ``LLMClient``.

    Tracks two distinct counters:
      * ``semantic_generation_calls`` — A3D ``generate_structured`` calls;
      * ``provider_attempts`` — actual provider/technical attempts.
    """

    supported_structured_output_modes = frozenset(
        {"none", "json_object", "json_schema"}
    )

    def __init__(self, inner: LLMClient) -> None:
        self._inner = inner
        self.semantic_generation_calls = 0
        self.provider_attempts = 0

    def generate_structured(
        self, rendered_prompt, output_schema, semantic_profile
    ):
        self.semantic_generation_calls += 1
        result = self._inner.generate_structured(
            rendered_prompt, output_schema, semantic_profile
        )
        self.provider_attempts += result.attempts
        return result


# ---------------------------------------------------------------------------
# Chunk selection
# ---------------------------------------------------------------------------


def select_consecutive_chunks(
    chunk_refs: tuple[ArtifactRef, ...], count: int = SMOKE_CHUNK_COUNT
) -> tuple[ArtifactRef, ...]:
    """Select ``count`` consecutive chunks from the start of the manifest.

    Raises ``ValueError`` if the manifest has fewer than ``count`` chunks.
    """
    if len(chunk_refs) < count:
        raise ValueError(
            f"manifest has {len(chunk_refs)} chunks; need at least {count}"
        )
    return tuple(chunk_refs[:count])


# ---------------------------------------------------------------------------
# Per-chunk result record
# ---------------------------------------------------------------------------


def _chunk_result_record(
    publication, store: FileArtifactStore
) -> dict[str, Any]:
    """Extract a concise result record for one chunk."""
    loaded = load_candidate_extraction(
        store, publication.candidate_extraction_ref
    )
    report = load_validation_report(
        store, publication.validation_report_ref
    )
    candidates = loaded.candidates
    counts = {
        "characters": len(candidates.characters),
        "locations": len(candidates.locations),
        "facts": len(candidates.facts),
        "events": len(candidates.events),
        "relationships": len(candidates.relationships),
        "unresolved_mentions": len(candidates.unresolved_mentions),
    }
    return {
        "reused": publication.reused,
        "candidate_extraction_ref": publication.candidate_extraction_ref,
        "validation_report_ref": publication.validation_report_ref,
        "validation_result": report.summary.result,
        "candidate_counts": counts,
        "total_candidates": sum(counts.values()),
    }


# ---------------------------------------------------------------------------
# Gate checks (testable offline)
# ---------------------------------------------------------------------------


def check_validation_gate(
    chunk_ids: list[str], results: list[dict[str, Any]]
) -> list[str]:
    """Return a list of gate failures for A3 validation (empty = all pass)."""
    failures: list[str] = []
    for chunk_id, result in zip(chunk_ids, results):
        if result["validation_result"] is not ValidationResult.PASS:
            failures.append(
                f"{chunk_id}: validation result is "
                f"{result['validation_result']} (expected PASS)"
            )
    return failures


def check_rerun_gate(
    first_results: list[dict[str, Any]],
    rerun_results: list[dict[str, Any]],
    sem_calls_before_rerun: int,
    sem_calls_after_rerun: int,
    prov_attempts_before_rerun: int,
    prov_attempts_after_rerun: int,
) -> list[str]:
    """Return a list of gate failures for the exact-rerun reuse proof."""
    failures: list[str] = []
    for i, (first, rerun) in enumerate(zip(first_results, rerun_results)):
        if rerun["reused"] is not True:
            failures.append(
                f"chunk[{i}]: rerun reused={rerun['reused']} (expected True)"
            )
        if rerun["candidate_extraction_ref"] != first[
            "candidate_extraction_ref"
        ]:
            failures.append(
                f"chunk[{i}]: candidate_extraction_ref changed on rerun"
            )
        if rerun["validation_report_ref"] != first["validation_report_ref"]:
            failures.append(
                f"chunk[{i}]: validation_report_ref changed on rerun"
            )
    if sem_calls_after_rerun != sem_calls_before_rerun:
        failures.append(
            f"additional semantic generation calls on rerun: "
            f"{sem_calls_after_rerun - sem_calls_before_rerun}"
        )
    if prov_attempts_after_rerun != prov_attempts_before_rerun:
        failures.append(
            f"additional provider attempts on rerun: "
            f"{prov_attempts_after_rerun - prov_attempts_before_rerun}"
        )
    return failures


def check_invalidation_gate(
    tracked_result: dict[str, Any],
    modified_result: dict[str, Any],
    sem_calls_before: int,
    sem_calls_after: int,
    prov_attempts_before: int,
    prov_attempts_after: int,
) -> list[str]:
    """Return a list of gate failures for the semantic-identity invalidation
    probe.

    The modified profile must NOT reuse the old CURRENT, must trigger real
    generation, and must produce a different CandidateExtraction ref.
    """
    failures: list[str] = []
    if modified_result["reused"] is True:
        failures.append(
            "modified profile incorrectly reused old CURRENT "
            "(reuse must be denied for a different semantic identity)"
        )
    if modified_result["candidate_extraction_ref"] == tracked_result[
        "candidate_extraction_ref"
    ]:
        failures.append(
            "modified profile returned the same CandidateExtraction ref as "
            "the tracked-profile run"
        )
    if sem_calls_after - sem_calls_before < 1:
        failures.append(
            f"no semantic generation calls for modified profile "
            f"(delta={sem_calls_after - sem_calls_before})"
        )
    if prov_attempts_after - prov_attempts_before < 1:
        failures.append(
            f"no provider attempts for modified profile "
            f"(delta={prov_attempts_after - prov_attempts_before})"
        )
    return failures


# ---------------------------------------------------------------------------
# Main smoke
# ---------------------------------------------------------------------------


def run_smoke(
    *,
    runtime_config_path: str | Path,
    semantic_profile_path: str | Path,
    extraction_profile_path: str | Path,
    chunk_profile_path: str | Path = DEFAULT_CHUNK_PROFILE,
    project_path: str | Path = PROJECT_PATH,
) -> int:
    """Run the A3E-C real-novel acceptance smoke. Returns 0 on success, 2 on
    failure."""
    # 1. Load profiles.
    runtime_config = load_runtime_config(runtime_config_path)
    semantic_profile = load_semantic_profile(semantic_profile_path)
    extraction_profile = load_story_extraction_profile(extraction_profile_path)

    print("== A3E-C real-novel acceptance smoke ==")
    print(f"project:                  {project_path}")
    print(f"runtime endpoint:         {runtime_config.base_url}")
    print(f"request_model:            {runtime_config.request_model}")
    print(f"provider_family:          {runtime_config.provider_family}")
    print(f"semantic profile hash:    {semantic_profile.semantic_profile_hash}")
    print()

    # 2. Isolated runs root + A1/A2 preparation.
    with tempfile.TemporaryDirectory(prefix="a3e_smoke_") as workdir:
        runs_root = Path(workdir) / "runs"
        runs_root.mkdir()

        # A1: ingest the real novel source.
        ingest_source_project(project_path, runs_root=runs_root)

        # A2: plan chunks with the production chunk profile.
        plan_chunks_project(
            project_path, runs_root=runs_root, profile_path=chunk_profile_path
        )

        # Resolve the current source + manifest.
        store, pointers = _stores(runs_root, PROJECT_ID)
        source_ptr = source_pointer_id(PROJECT_ID, DOCUMENT_ID)
        _sp, source_ref = _current_pointer(pointers, source_ptr)
        if source_ref is None:
            print("ERROR: SourceDocument is not current", file=sys.stderr)
            return 2
        manifest_ptr = chunk_pointer_id(
            PROJECT_ID, DOCUMENT_ID, _load_chunk_profile_id(chunk_profile_path)
        )
        _mp, manifest_ref = _current_pointer(pointers, manifest_ptr)
        if manifest_ref is None:
            print("ERROR: ChunkManifest is not current", file=sys.stderr)
            return 2
        manifest = load_chunk_manifest(store, manifest_ref)
        source_document = load_source_document(store, source_ref)

        total_chunks = len(manifest.chunk_refs)
        chunk_profile_id = _load_chunk_profile_id(chunk_profile_path)

        print(f"source identity:          {source_ref.artifact_id}:"
              f"{source_ref.revision}")
        print(f"chunk profile:            {chunk_profile_id}")
        print(f"total chunks:             {total_chunks}")

        # 3. Select 3 consecutive chunks.
        try:
            selected_refs = select_consecutive_chunks(
                tuple(manifest.chunk_refs), SMOKE_CHUNK_COUNT
            )
        except ValueError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2

        chunk_ids: list[str] = []
        for ref in selected_refs:
            chunk = load_source_chunk(store, ref)
            chunk_ids.append(chunk.chunk_id)

        print(f"selected chunks:          {chunk_ids}")
        print()

        # 4. Set up the A3D service + counting client.
        service = ChunkExtractionService(
            store, pointers, PromptRegistry(PROMPTS_DIR)
        )
        client = CountingLLMClient(OpenAICompatibleLLMClient(runtime_config))

        def extract_one(
            chunk_ref: ArtifactRef,
            sem_profile: Any = semantic_profile,
        ) -> Any:
            return service.extract_chunk(
                source_document_ref=source_ref,
                source_chunk_ref=chunk_ref,
                chunk_profile_id=chunk_profile_id,
                extraction_profile=extraction_profile,
                semantic_profile=sem_profile,
                llm_client=client,
            )

        # 5. First real run (3 chunks).
        first_results: list[dict[str, Any]] = []
        for chunk_id, chunk_ref in zip(chunk_ids, selected_refs):
            sem_before = client.semantic_generation_calls
            prov_before = client.provider_attempts
            publication = extract_one(chunk_ref)
            sem_after = client.semantic_generation_calls
            prov_after = client.provider_attempts
            record = _chunk_result_record(publication, store)
            record["chunk_id"] = chunk_id
            record["sem_calls"] = sem_after - sem_before
            record["prov_attempts"] = prov_after - prov_before
            first_results.append(record)
            print(
                f"  {chunk_id}: reused={record['reused']} "
                f"sem_calls={record['sem_calls']} "
                f"prov_attempts={record['prov_attempts']} "
                f"candidates={record['total_candidates']} "
                f"validation={record['validation_result']}"
            )
        print()

        total_sem_first = client.semantic_generation_calls
        total_prov_first = client.provider_attempts
        print(f"first-run total: sem_calls={total_sem_first} "
              f"prov_attempts={total_prov_first}")

        # 6. Validation gate.
        validation_failures = check_validation_gate(chunk_ids, first_results)
        if validation_failures:
            for f in validation_failures:
                print(f"VALIDATION FAIL: {f}", file=sys.stderr)
            return 2
        print("validation gate: PASS")

        # 7. Exact rerun (reuse proof).
        sem_before_rerun = client.semantic_generation_calls
        prov_before_rerun = client.provider_attempts
        rerun_results: list[dict[str, Any]] = []
        for chunk_id, chunk_ref in zip(chunk_ids, selected_refs):
            publication = extract_one(chunk_ref)
            record = _chunk_result_record(publication, store)
            record["chunk_id"] = chunk_id
            rerun_results.append(record)
        sem_after_rerun = client.semantic_generation_calls
        prov_after_rerun = client.provider_attempts

        rerun_failures = check_rerun_gate(
            first_results,
            rerun_results,
            sem_before_rerun,
            sem_after_rerun,
            prov_before_rerun,
            prov_after_rerun,
        )
        if rerun_failures:
            for f in rerun_failures:
                print(f"RERUN GATE FAIL: {f}", file=sys.stderr)
            return 2
        print(
            f"rerun gate: PASS "
            f"(reused={sum(1 for r in rerun_results if r['reused'])}/"
            f"{len(rerun_results)}, "
            f"delta calls/attempts = "
            f"{sem_after_rerun - sem_before_rerun} / "
            f"{prov_after_rerun - prov_before_rerun})"
        )

        # 8. Semantic identity invalidation probe.
        #    Change temperature (a true semantic contract property).
        modified_profile = dataclasses.replace(
            semantic_profile, temperature=0.5
        )
        print()
        print(f"invalidation field:       temperature "
              f"{semantic_profile.temperature} -> "
              f"{modified_profile.temperature}")
        print(f"old semantic_profile_hash:  "
              f"{semantic_profile.semantic_profile_hash}")
        print(f"new semantic_profile_hash:  "
              f"{modified_profile.semantic_profile_hash}")

        inv_chunk_id = chunk_ids[0]
        inv_chunk_ref = selected_refs[0]

        sem_before_inv = client.semantic_generation_calls
        prov_before_inv = client.provider_attempts
        inv_publication = extract_one(inv_chunk_ref, modified_profile)
        sem_after_inv = client.semantic_generation_calls
        prov_after_inv = client.provider_attempts
        inv_record = _chunk_result_record(inv_publication, store)

        print(f"invalidation chunk:       {inv_chunk_id}")
        print(f"  reused:                 {inv_record['reused']}")
        print(f"  sem_calls delta:        "
              f"{sem_after_inv - sem_before_inv}")
        print(f"  prov_attempts delta:    "
              f"{prov_after_inv - prov_before_inv}")
        print(f"  new extraction ref:     "
              f"{inv_record['candidate_extraction_ref'].artifact_id}:"
              f"{inv_record['candidate_extraction_ref'].revision}")
        print(f"  validation:             "
              f"{inv_record['validation_result']}")

        inv_failures = check_invalidation_gate(
            first_results[0],
            inv_record,
            sem_before_inv,
            sem_after_inv,
            prov_before_inv,
            prov_after_inv,
        )
        if inv_failures:
            for f in inv_failures:
                print(f"INVALIDATION GATE FAIL: {f}", file=sys.stderr)
            return 2
        print("invalidation gate: PASS")

        # 9. Semantic plausibility summary.
        print()
        print("== semantic plausibility summary ==")
        for record in first_results:
            print(
                f"  {record['chunk_id']}: "
                f"candidates={record['total_candidates']} "
                f"({json.dumps(record['candidate_counts'], ensure_ascii=False)}) "
                f"validation={record['validation_result']}"
            )

        print()
        print("A3E REAL-NOVEL SMOKE OK")
        return 0


def _load_chunk_profile_id(path: str | Path) -> str:
    """Read the profile_id from a chunk profile YAML file."""
    from short_drama.io import load_yaml

    data = load_yaml(path)
    return data["profile_id"]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="A3E-C real-novel local-Qwen acceptance smoke"
    )
    parser.add_argument(
        "--runtime-config",
        default=str(DEFAULT_RUNTIME_CONFIG),
        help="runtime transport config YAML",
    )
    parser.add_argument(
        "--profile",
        default=str(DEFAULT_SEMANTIC_PROFILE),
        help="semantic LLM profile YAML",
    )
    parser.add_argument(
        "--extraction-profile",
        default=str(DEFAULT_EXTRACTION_PROFILE),
        help="story extraction profile YAML",
    )
    parser.add_argument(
        "--chunk-profile",
        default=str(DEFAULT_CHUNK_PROFILE),
        help="chunk planning profile YAML",
    )
    args = parser.parse_args()
    try:
        return run_smoke(
            runtime_config_path=args.runtime_config,
            semantic_profile_path=args.profile,
            extraction_profile_path=args.extraction_profile,
            chunk_profile_path=args.chunk_profile,
        )
    except LLMError as exc:
        print(f"SMOKE FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001
        msg = f"SMOKE FAILED: {type(exc).__name__}: {exc}"
        # For semantic generation failures, include the final findings.
        findings = getattr(exc, "final_findings", None)
        if findings:
            msg += f"\nfinal findings ({len(findings)}):"
            for f in findings[:10]:
                msg += f"\n  {f.code}: {f.message}"
            if len(findings) > 10:
                msg += f"\n  ... and {len(findings) - 10} more"
        print(msg, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
