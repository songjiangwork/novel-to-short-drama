"""v1.2 A3D — controlled real local-Qwen SINGLE-chunk semantic smoke.

This is a developer smoke tool (NOT the #15 ``extract-chunks`` batch CLI). It
exercises the complete ACTUAL A3D single-chunk path against a real local
OpenAI-compatible (llama.cpp / Qwen) server:

    persist exact SourceDocument
      -> persist exact SourceChunk
      -> build real A3 prompt (a3.chunk-extraction v1)
      -> build real CandidatePayload OutputSchema
      -> A3C pre-generation reuse check
      -> real OpenAICompatibleLLMClient
      -> local Qwen structured generation
      -> CandidatePayload typed load
      -> A3B semantic validation
      -> A3C CandidateExtraction persistence
      -> ValidationReport
      -> CURRENT
      -> exact rerun (reused=True, 0 additional provider calls, same ref)

It uses a tiny deterministic single-chunk story fixture. A tiny counting
wrapper around the provider-neutral ``LLMClient`` proves the exact rerun makes
no second provider call. It never saves raw provider envelopes or full raw
assistant text as canonical artifacts (only the typed A3 artifacts are
persisted). It prints concise evidence and never prints secrets or credential
values.

The tracked semantic model is pinned in the semantic profile. If the exact
server model genuinely differs, the mismatch is REPORTED (and an explicit
``--model`` override may be supplied); the tracked semantic identity is not
silently changed.

Usage:
    python scripts/a3_chunk_smoke.py \
        --runtime-config profiles/llm_local.yaml \
        --profile profiles/story_llm_qwen_v1.yaml \
        --extraction-profile profiles/story_extraction_v1.yaml \
        [--model <exact-server-model-name>]

Exit code 0 on success, 2 on failure.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import tempfile
import urllib.request
from pathlib import Path

from short_drama.artifacts import FileArtifactStore
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
    LANGUAGE_DETECTOR_ID,
    ChunkExtractionService,
    ParagraphSpan,
    SourceChapter,
    SourceChunk,
    SourceDocument,
    SourceInfo,
    SourceParagraph,
    estimate_paragraphs,
    load_candidate_extraction,
    load_story_extraction_profile,
    persist_source_chunk,
    persist_source_document,
)
from short_drama.story.source import NormalizationInfo

PROJECT = "classroom"
DOCUMENT = "a3_smoke"
CHUNK_PROFILE_ID = "story-analysis-v1"
CHUNK_ID = "CH001_C001"
CHAPTER_ID = "CH001"
PROMPTS_DIR = REPO_ROOT / "prompts" / "story"

# Tiny deterministic single-chunk story fixture (working language zh-CN).
# OWNERSHIP span = P0003..P0006 (the action); LEFT/RIGHT context on both sides.
FIXTURE_PARAGRAPHS = {
    "CH001_P0001": "窗外的雨越下越大，夜色渐渐沉了下来。",
    "CH001_P0002": "老槐树的叶子被雨打得沙沙作响。",
    "CH001_P0003": "林晚推开老屋的门走进屋内，手里握着一把油纸伞。",
    "CH001_P0004": "她轻声唤道：“阿爹，我回来了。”",
    "CH001_P0005": "坐在老屋炕头的老李抬起头，浑浊的眼睛里泛起泪花。",
    "CH001_P0006": "他伸出粗糙的手，紧紧握住了女儿的手。",
    "CH001_P0007": "院子里的狗听见动静，摇着尾巴跑了出来。",
    "CH001_P0008": "远处的钟声敲响了，已是深夜。",
}
OWNERSHIP_START = "CH001_P0003"
OWNERSHIP_END = "CH001_P0006"


def build_fixture_document() -> SourceDocument:
    chapter = SourceChapter(
        CHAPTER_ID,
        None,
        "synthetic",
        tuple(
            SourceParagraph(pid, FIXTURE_PARAGRAPHS[pid], None)
            for pid in sorted(FIXTURE_PARAGRAPHS)
        ),
    )
    return SourceDocument(
        schema_version=1,
        project_id=PROJECT,
        document_id=DOCUMENT,
        source=SourceInfo(
            "txt",
            "fixture/a3_smoke.txt",
            "f" * 64,
            sum(len(t.encode("utf-8")) for t in FIXTURE_PARAGRAPHS.values()),
            "zh-CN",
            "zh",
            LANGUAGE_DETECTOR_ID,
        ),
        normalization=NormalizationInfo(
            "utf-8", "LF", "short_drama_source_ingestion_v1", "1"
        ),
        chapters=(chapter,),
    )


def build_fixture_chunk(source_document_ref) -> SourceChunk:
    pids = tuple(sorted(FIXTURE_PARAGRAPHS))
    index = {pid: i for i, pid in enumerate(pids)}
    context_paragraphs = [SourceParagraph(pid, FIXTURE_PARAGRAPHS[pid], None) for pid in pids]
    ownership_paragraphs = [
        SourceParagraph(pid, FIXTURE_PARAGRAPHS[pid], None)
        for pid in pids[index[OWNERSHIP_START] : index[OWNERSHIP_END] + 1]
    ]
    return SourceChunk(
        schema_version=1,
        chunk_id=CHUNK_ID,
        project_id=PROJECT,
        document_id=DOCUMENT,
        chapter_id=CHAPTER_ID,
        source_document_ref=source_document_ref,
        context_span=ParagraphSpan(pids[0], pids[-1]),
        ownership_span=ParagraphSpan(OWNERSHIP_START, OWNERSHIP_END),
        paragraph_ids=pids,
        token_count_method="utf8-bytes-div3-v1",
        context_token_count=estimate_paragraphs(context_paragraphs),
        ownership_token_count=estimate_paragraphs(ownership_paragraphs),
    )


def query_server_model(base_url: str, credential_env: str | None) -> str | None:
    """Best-effort query of the served model name (never a smoke gate)."""
    url = base_url.rstrip("/") + "/models"
    request = urllib.request.Request(url)
    if credential_env is not None:
        import os

        value = os.environ.get(credential_env)
        if value:
            request.add_header("Authorization", f"Bearer {value}")
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            data = json.loads(response.read().decode("utf-8"))
    except Exception:  # noqa: BLE001 - a model query failure is not fatal
        return None
    try:
        models = data.get("data") or data.get("models") or []
        first = models[0]
        name = first.get("id") or first.get("model") or first.get("name")
        return name if isinstance(name, str) else None
    except Exception:  # noqa: BLE001
        return None


class CountingLLMClient(LLMClient):
    """Tiny counting wrapper around the real provider-neutral LLMClient.

    Exists solely in the smoke to prove no second provider call occurred on the
    exact rerun. It adds no caching and no retry logic.
    """

    supported_structured_output_modes = frozenset({"none", "json_object", "json_schema"})

    def __init__(self, inner: LLMClient) -> None:
        self._inner = inner
        self.call_count = 0

    def generate_structured(self, rendered_prompt, output_schema, semantic_profile):
        self.call_count += 1
        return self._inner.generate_structured(
            rendered_prompt, output_schema, semantic_profile
        )


def run_smoke(
    *,
    runtime_config_path: str,
    profile_path: str,
    extraction_profile_path: str,
    model_override: str | None,
) -> int:
    runtime_config = load_runtime_config(runtime_config_path)
    semantic_profile = load_semantic_profile(profile_path)
    extraction_profile = load_story_extraction_profile(extraction_profile_path)

    tracked_model = semantic_profile.model
    server_model = query_server_model(
        runtime_config.base_url, runtime_config.credential_environment_name
    )
    effective_model = model_override if model_override is not None else tracked_model

    print("== A3D single-chunk real local-Qwen smoke ==")
    print(f"base_url:                 {runtime_config.base_url}")
    print(f"tracked semantic model:   {tracked_model}")
    print(f"served model (queried):   {server_model or 'unknown'}")
    print(f"effective model (used):   {effective_model}")
    if server_model is not None and server_model != tracked_model:
        print(
            "NOTE: tracked semantic model differs from the served model; "
            "reported, not silently changed."
        )
    if effective_model != tracked_model:
        print(f"model override applied:   {tracked_model} -> {effective_model}")
    if server_model is not None and effective_model != server_model:
        print(
            "WARNING: effective model differs from the served model; the "
            "provider request may be rejected."
        )

    # Apply the (explicit or default) model to the semantic profile for this
    # run only. The tracked profile file on disk is never modified.
    if effective_model != tracked_model:
        semantic_profile = dataclasses.replace(semantic_profile, model=effective_model)

    with tempfile.TemporaryDirectory(prefix="a3d_smoke_") as workdir:
        store = FileArtifactStore(Path(workdir) / "artifacts")
        pointers = FilePointerStore(Path(workdir) / "pointers", store)

        document = build_fixture_document()
        document_ref = persist_source_document(store, document, revision=1)
        chunk = build_fixture_chunk(document_ref)
        chunk_ref = persist_source_chunk(
            store, chunk, profile_id=CHUNK_PROFILE_ID, revision=1
        )

        service = ChunkExtractionService(store, pointers, PromptRegistry(PROMPTS_DIR))
        client = CountingLLMClient(OpenAICompatibleLLMClient(runtime_config))

        def extract() -> object:
            return service.extract_chunk(
                source_document_ref=document_ref,
                source_chunk_ref=chunk_ref,
                chunk_profile_id=CHUNK_PROFILE_ID,
                extraction_profile=extraction_profile,
                semantic_profile=semantic_profile,
                llm_client=client,
            )

        calls_before = client.call_count
        first = extract()
        calls_after_first = client.call_count
        second = extract()
        calls_after_second = client.call_count

        loaded = load_candidate_extraction(store, first.candidate_extraction_ref)
        report = load_validation_report(store, first.validation_report_ref)
        candidates = loaded.candidates
        counts = {
            "characters": len(candidates.characters),
            "locations": len(candidates.locations),
            "facts": len(candidates.facts),
            "events": len(candidates.events),
            "relationships": len(candidates.relationships),
            "unresolved_mentions": len(candidates.unresolved_mentions),
        }
        blocking_codes = {
            finding.code for finding in report.findings
        }

        print()
        print(f"model:                      {effective_model}")
        print(f"request_hash:               {loaded.generation_provenance.request_hash}")
        print(f"semantic rounds (run 1):    {calls_after_first - calls_before}")
        print(f"provider calls before:      {calls_before}")
        print(f"provider calls after run 1: {calls_after_first}")
        print(f"provider calls after rerun: {calls_after_second}")
        print(f"additional calls on rerun:  {calls_after_second - calls_after_first}")
        print(
            "CandidateExtraction ref:    "
            f"{first.candidate_extraction_ref.artifact_id}:"
            f"{first.candidate_extraction_ref.revision}"
        )
        print(
            "ValidationReport ref:       "
            f"{first.validation_report_ref.artifact_id}:"
            f"{first.validation_report_ref.revision}"
        )
        print(f"reused (run 1):             {first.reused}")
        print(f"reused (rerun):             {second.reused}")
        print(f"same extraction ref rerun:  {second.candidate_extraction_ref == first.candidate_extraction_ref}")
        print(
            "candidates by category:     "
            + json.dumps(counts, ensure_ascii=False)
        )
        print(f"A3 validation result:       {report.summary.result}")
        # A PASS report has zero blocking findings, which is exactly the
        # ownership-primary-evidence + local cross-ref closure guarantee.
        print(
            "ownership-primary evidence: "
            f"{'PASS' if report.summary.result is ValidationResult.PASS and not blocking_codes else 'FAIL'}"
        )
        print(
            "local cross-ref validation: "
            f"{'PASS' if report.summary.result is ValidationResult.PASS and not blocking_codes else 'FAIL'}"
        )
        if blocking_codes:
            print(f"blocking findings:          {sorted(blocking_codes)}")
        print()

        # Explicit success criteria (not `assert`, stripped under `python -O`).
        ok = (
            report.summary.result is ValidationResult.PASS
            and not blocking_codes
            and first.reused is False
            and second.reused is True
            and calls_after_first - calls_before >= 1
            and calls_after_second - calls_after_first == 0
            and second.candidate_extraction_ref == first.candidate_extraction_ref
        )
        if not ok:
            print("SMOKE FAILED: success criteria not met", file=sys.stderr)
            return 2
        print("SMOKE OK")
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="A3D single-chunk real local-Qwen semantic smoke"
    )
    parser.add_argument("--runtime-config", required=True, help="runtime transport config YAML")
    parser.add_argument("--profile", required=True, help="semantic LLM profile YAML")
    parser.add_argument("--extraction-profile", required=True, help="story extraction profile YAML")
    parser.add_argument(
        "--model",
        default=None,
        help="override the profile's model with the exact server model name",
    )
    args = parser.parse_args()
    try:
        return run_smoke(
            runtime_config_path=args.runtime_config,
            profile_path=args.profile,
            extraction_profile_path=args.extraction_profile,
            model_override=args.model,
        )
    except LLMError as exc:
        print(f"SMOKE FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 - report any other smoke failure
        print(f"SMOKE FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
