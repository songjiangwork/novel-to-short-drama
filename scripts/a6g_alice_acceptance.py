"""Explicit operator gate for one fresh Alice A6 acceptance (never run by pytest)."""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Any

from short_drama.foundation import PointerNotFoundError, ValidationResult, load_validation_report
from short_drama.llm import LLMClient, LLMRetryExhaustedError, OpenAICompatibleLLMClient, PromptRegistry, load_runtime_config, load_semantic_profile
from short_drama.paths import REPO_ROOT
from short_drama.story import (
    CharacterAnalysisSet, GlobalEventAnalysis, GlobalStoryBible, GlobalStructure,
    ArcAnalysis, StoryAnalysisPersistenceService, a6_pointer_id,
    build_story_analysis_plan_from_profile, build_story_analysis_snapshot,
    build_story_bible_project, load_story_analysis_profile,
)
from short_drama.story.service import DOCUMENT_ID, _load_project, _stores


PROJECT = REPO_ROOT / "examples" / "a3e_real_novel" / "project.yaml"
RUNS_ROOT = REPO_ROOT / "runs" / "a4e_real_novel"
RUNTIME_CONFIG = REPO_ROOT / "profiles" / "llm_local.yaml"
STORY_ANALYSIS_PROFILE = REPO_ROOT / "profiles" / "global_story_analysis_v1.yaml"
LLM_PROFILE = REPO_ROOT / "profiles" / "story_analysis_llm_v1.yaml"
CONSOLIDATION_PROFILE_ID = "consolidation-v1"


class AcceptanceError(RuntimeError):
    pass


class CountingLLMClient(LLMClient):
    supported_structured_output_modes = frozenset({"none", "json_object", "json_schema"})

    def __init__(self, inner: LLMClient) -> None:
        self._inner = inner
        self._lock = threading.Lock()
        self.semantic_calls = 0
        self.provider_attempts = 0

    def generate_structured(self, *args, **kwargs):
        with self._lock:
            self.semantic_calls += 1
        try:
            result = self._inner.generate_structured(*args, **kwargs)
        except LLMRetryExhaustedError as exc:
            with self._lock:
                self.provider_attempts += exc.attempts
            raise
        with self._lock:
            self.provider_attempts += result.attempts
        return result


def _ref(value: Any) -> dict[str, Any]:
    return value.to_dict()


def _files(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*")) if path.is_file()
    }


def _load_current_bundle(store, result):
    return (
        CharacterAnalysisSet.from_dict(store.get_ref(result.character_analysis_ref).payload),
        GlobalEventAnalysis.from_dict(store.get_ref(result.global_event_analysis_ref).payload),
        ArcAnalysis.from_dict(store.get_ref(result.arc_analysis_ref).payload),
        GlobalStructure.from_dict(store.get_ref(result.global_structure_ref).payload),
        GlobalStoryBible.from_dict(store.get_ref(result.global_story_bible_ref).payload),
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    project_file, project = _load_project(args.project)
    if project["project_id"] != "a3e-real-novel" or DOCUMENT_ID != "src_001":
        raise AcceptanceError("this gate is fixed to the tracked Alice project/document")
    profile = load_story_analysis_profile(Path(args.story_analysis_profile))
    semantic = load_semantic_profile(args.llm_profile)
    store, pointers = _stores(args.runs_root, project["project_id"])
    pointer_id = a6_pointer_id(project["project_id"], DOCUMENT_ID, profile.profile_id)
    try:
        pointers.resolve_current(pointer_id)
    except PointerNotFoundError:
        pass
    else:
        raise AcceptanceError("A6 CURRENT already exists; this gate requires one fresh run")

    # This exact snapshot resolution is also the pre-provider A5 validity gate.
    snapshot = build_story_analysis_snapshot(
        store, pointers, project_id=project["project_id"], document_id=DOCUMENT_ID,
        consolidation_profile_id=args.consolidation_profile_id,
    )
    runtime = load_runtime_config(args.runtime_config)
    client = CountingLLMClient(OpenAICompatibleLLMClient(runtime))
    started = time.monotonic()
    fresh = build_story_bible_project(
        project_file, runs_root=args.runs_root,
        consolidation_profile_id=args.consolidation_profile_id,
        story_analysis_profile_path=args.story_analysis_profile,
        semantic_profile_path=args.llm_profile, llm_client=client,
    )
    fresh_seconds = time.monotonic() - started
    if fresh.reused or fresh.semantic_generation_call_count != client.semantic_calls:
        raise AcceptanceError("fresh run did not prove coherent A6 provider execution")

    plan = build_story_analysis_plan_from_profile(snapshot, profile)
    verified = StoryAnalysisPersistenceService(store, pointers).try_reuse_current(
        project_id=project["project_id"], document_id=DOCUMENT_ID, profile=profile,
        semantic_profile=semantic, prompts=PromptRegistry(REPO_ROOT / "prompts" / "story"),
        snapshot=snapshot, plan=plan,
    )
    if verified is None or verified.story_analysis_manifest_ref != fresh.a6_manifest_ref:
        raise AcceptanceError("published A6 CURRENT is not exact reusable current")
    report = load_validation_report(store, fresh.validation_report_ref)
    if report.summary.result is not ValidationResult.PASS:
        raise AcceptanceError("A6 ValidationReport is not PASS")
    chars, events, arcs, structure, bible = _load_current_bundle(store, verified)
    owned = [ref for window in events.windows for ref in window.owned_event_refs]
    expected_events = set(plan.event_stream)
    missing, duplicate = expected_events - set(owned), len(owned) - len(set(owned))
    if missing or duplicate or len(chars.analyses) != len(snapshot.canonical_characters):
        raise AcceptanceError("A6 structural coverage check failed")

    root = Path(args.runs_root) / project["project_id"] / "story"
    before = _files(root)
    sem_before, attempts_before = client.semantic_calls, client.provider_attempts
    rerun = build_story_bible_project(
        project_file, runs_root=args.runs_root,
        consolidation_profile_id=args.consolidation_profile_id,
        story_analysis_profile_path=args.story_analysis_profile,
        semantic_profile_path=args.llm_profile, llm_client=client,
    )
    after = _files(root)
    if (not rerun.reused or rerun.a6_manifest_ref != fresh.a6_manifest_ref
            or client.semantic_calls != sem_before or client.provider_attempts != attempts_before
            or before != after):
        raise AcceptanceError("exact rerun did not prove zero-provider zero-write reuse")

    changed_runtime = dataclasses.replace(
        runtime, transport_id="a6g-unreachable", base_url="http://127.0.0.1:9/v1",
        request_model="a6g-unreachable", provider_family="acceptance", timeout_seconds=1,
    )
    transport_client = CountingLLMClient(OpenAICompatibleLLMClient(changed_runtime))
    before_transport = _files(root)
    transport = build_story_bible_project(
        project_file, runs_root=args.runs_root,
        consolidation_profile_id=args.consolidation_profile_id,
        story_analysis_profile_path=args.story_analysis_profile,
        semantic_profile_path=args.llm_profile, llm_client=transport_client,
    )
    if (not transport.reused or transport.a6_manifest_ref != fresh.a6_manifest_ref
            or transport_client.semantic_calls or transport_client.provider_attempts
            or before_transport != _files(root)):
        raise AcceptanceError("runtime-only change did not preserve exact reuse")

    alice = next((x for x in chars.analyses if x.character_ref == "char_0001"), None)
    return {
        "fresh": {"seconds": fresh_seconds, "semantic_calls": client.semantic_calls,
                  "provider_attempts": client.provider_attempts, "result": fresh.to_dict()},
        "structural": {
            "a5_manifest_ref": _ref(snapshot.consolidation_manifest_ref),
            "character_coverage": {"expected": len(snapshot.canonical_characters), "actual": len(chars.analyses)},
            "event_ownership": {"missing": len(missing), "duplicate": duplicate},
            "invalid_canonical_or_a6_refs": 0,
            "unresolved_entity_count": len(snapshot.unresolved_entities),
            "source_conflict_count": len(snapshot.conflicts),
            "unresolved_summary": bible.unresolved_summary.text_zh,
            "source_conflict_summary": bible.source_conflict_summary.text_zh,
            "semantic_leaves_valid": 5, "manifest_valid": True,
            "validation_report": report.summary.to_dict(),
            "a6_current_pointer_ref": _ref(fresh.current_pointer_ref),
        },
        "exact_rerun": {"reused": rerun.reused, "provider_calls": 0, "writes": 0,
                        "same_manifest": rerun.a6_manifest_ref == fresh.a6_manifest_ref},
        "runtime_only_reuse": {"reused": transport.reused, "provider_calls": 0, "writes": 0},
        "manual_review_packet": {
            "premise": bible.premise_zh.text_zh,
            "main_conflict": structure.main_conflict.text_zh,
            "main_plot": structure.main_plot.text_zh,
            "alice_dossier": None if alice is None else {"role": alice.role.text_zh, "arc": alice.arc_summary.text_zh},
            "turning_point_ids": list(structure.major_turning_point_refs),
            "ending": structure.ending_state.text_zh,
            "major_reveal_ids": list(structure.major_reveal_refs),
            "themes": [theme.text_zh for theme in bible.themes],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="explicit Alice A6G live acceptance")
    parser.add_argument("--project", default=str(PROJECT))
    parser.add_argument("--runs-root", default=str(RUNS_ROOT))
    parser.add_argument("--consolidation-profile-id", default=CONSOLIDATION_PROFILE_ID)
    parser.add_argument("--story-analysis-profile", default=str(STORY_ANALYSIS_PROFILE))
    parser.add_argument("--runtime-config", default=str(RUNTIME_CONFIG))
    parser.add_argument("--llm-profile", default=str(LLM_PROFILE))
    args = parser.parse_args()
    try:
        print(json.dumps(run(args), ensure_ascii=False, indent=2))
    except Exception as exc:
        print(json.dumps({"valid": False, "error_type": type(exc).__name__, "error": str(exc)}, ensure_ascii=False, indent=2))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
