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


class NoProviderLLMClient(LLMClient):
    """A verification-only sentinel: a semantic call is a gate failure.

    Carrying the runtime object makes the changed-runtime check explicit while
    keeping the verification path physically unable to reach a provider.
    """

    supported_structured_output_modes = frozenset({"none", "json_object", "json_schema"})

    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime
        self.calls = 0

    def generate_structured(self, *args, **kwargs):
        self.calls += 1
        raise AcceptanceError("verification-only reuse attempted a Provider call")


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


def _load_verified_current(args: argparse.Namespace):
    """Read one exact A5/A6 CURRENT pair and prove its whole-A6 validity.

    ``try_reuse_current`` is the existing current-only verification authority:
    it loads the manifest and all leaves, validates their exact identity and
    PASS report, runs ``validate_whole_a6``, and re-checks the captured A5
    CURRENT head.  It does no historical scan and no writes.
    """
    project_file, project = _load_project(args.project)
    if project["project_id"] != "a3e-real-novel" or DOCUMENT_ID != "src_001":
        raise AcceptanceError("this gate is fixed to the tracked Alice project/document")
    profile = load_story_analysis_profile(Path(args.story_analysis_profile))
    semantic = load_semantic_profile(args.llm_profile)
    store, pointers = _stores(args.runs_root, project["project_id"])
    snapshot = build_story_analysis_snapshot(
        store, pointers, project_id=project["project_id"], document_id=DOCUMENT_ID,
        consolidation_profile_id=args.consolidation_profile_id,
    )
    plan = build_story_analysis_plan_from_profile(snapshot, profile)
    verified = StoryAnalysisPersistenceService(store, pointers).try_reuse_current(
        project_id=project["project_id"], document_id=DOCUMENT_ID, profile=profile,
        semantic_profile=semantic, prompts=PromptRegistry(REPO_ROOT / "prompts" / "story"),
        snapshot=snapshot, plan=plan,
    )
    if verified is None:
        raise AcceptanceError("no exact validated A6 CURRENT for this A5/profile/semantic identity")
    report = load_validation_report(store, verified.validation_report_ref)
    if report.summary.result is not ValidationResult.PASS:
        raise AcceptanceError("A6 ValidationReport is not PASS")
    chars, events, arcs, structure, bible = _load_current_bundle(store, verified)
    owned = [ref for window in events.windows for ref in window.owned_event_refs]
    expected_events = set(plan.event_stream)
    missing = expected_events - set(owned)
    duplicate = len(owned) - len(set(owned))
    if missing or duplicate or len(chars.analyses) != len(snapshot.canonical_characters):
        raise AcceptanceError("A6 structural coverage check failed")
    return project_file, project, profile, snapshot, plan, verified, report, chars, events, arcs, structure, bible


def _alice_dossier(snapshot, chars) -> dict[str, Any]:
    """Find Alice from the exact canonical registry, never a positional ID."""
    def normalized(value: str) -> str:
        return " ".join(value.casefold().split())

    matches = []
    for entity in snapshot.canonical_characters:
        names = (entity.display_name_original, *entity.aliases_original)
        if any(normalized(name) == "alice" for name in names):
            matches.append(entity)
    if len(matches) != 1:
        return {
            "status": "PENDING_CANONICAL_IDENTITY_CONFIRMATION",
            "matching_canonical_character_count": len(matches),
        }
    entity = matches[0]
    analysis = next((item for item in chars.analyses if item.character_ref == entity.canonical_id), None)
    if analysis is None:
        raise AcceptanceError("validated character coverage lacks canonical Alice analysis")
    return {
        "status": "CANONICAL_IDENTITY_VERIFIED",
        "canonical_character_ref": entity.canonical_id,
        "canonical_names": [entity.display_name_original, *entity.aliases_original],
        "role": analysis.role.text_zh,
        "arc": analysis.arc_summary.text_zh,
    }


def _structural_report(snapshot, plan, verified, report, chars, events, arcs, structure, bible) -> dict[str, Any]:
    owned = [ref for window in events.windows for ref in window.owned_event_refs]
    expected_events = set(plan.event_stream)
    missing = expected_events - set(owned)
    duplicate = len(owned) - len(set(owned))
    return {
        "a5_manifest_ref": _ref(snapshot.consolidation_manifest_ref),
        "character_coverage": {
            "expected": len(snapshot.canonical_characters),
            "actual": len(chars.analyses),
        },
        "event_ownership": {"missing": len(missing), "duplicate": duplicate},
        "reference_validation": {
            "status": "PASS",
            "authority": "StoryAnalysisPersistenceService.try_reuse_current / validate_whole_a6",
            "note": "whole-A6 validation passed; no separate invalid-reference count is claimed",
        },
        "unresolved_entity_count": len(snapshot.unresolved_entities),
        "source_conflict_count": len(snapshot.conflicts),
        "unresolved_summary": bible.unresolved_summary.text_zh,
        "source_conflict_summary": bible.source_conflict_summary.text_zh,
        "semantic_leaves": {"typed_loaded_and_validated": 5},
        "manifest_valid": True,
        "validation_report": report.summary.to_dict(),
        "a6_current_pointer_ref": _ref(verified.current_pointer_ref),
    }


def _manual_review_packet(bible, structure, alice_dossier: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": "PENDING_MANUAL_REVIEW",
        "premise": bible.premise_zh.text_zh,
        "main_conflict": structure.main_conflict.text_zh,
        "main_plot": structure.main_plot.text_zh,
        "alice_dossier": alice_dossier,
        "turning_point_ids": list(structure.major_turning_point_refs),
        "ending": structure.ending_state.text_zh,
        "major_reveal_ids": list(structure.major_reveal_refs),
        "themes": [theme.text_zh for theme in bible.themes],
    }


def _verify_reuse_only(args: argparse.Namespace) -> dict[str, Any]:
    """Read-only exact and runtime-only reuse verification; no Provider path."""
    (project_file, project, _profile, snapshot, plan, verified, report,
     chars, events, arcs, structure, bible) = _load_verified_current(args)
    root = Path(args.runs_root) / project["project_id"] / "story"
    runtime = load_runtime_config(args.runtime_config)
    sentinel = NoProviderLLMClient(runtime)
    before = _files(root)
    rerun = build_story_bible_project(
        project_file, runs_root=args.runs_root,
        consolidation_profile_id=args.consolidation_profile_id,
        story_analysis_profile_path=args.story_analysis_profile,
        semantic_profile_path=args.llm_profile, llm_client=sentinel,
    )
    after = _files(root)
    if (not rerun.reused or rerun.a6_manifest_ref != verified.story_analysis_manifest_ref
            or sentinel.calls or before != after):
        raise AcceptanceError("exact verification-only rerun did not prove zero-call zero-write reuse")

    changed_runtime = dataclasses.replace(
        runtime, transport_id="a6g-unreachable", base_url="http://127.0.0.1:9/v1",
        request_model="a6g-unreachable", provider_family="acceptance", timeout_seconds=1,
    )
    runtime_sentinel = NoProviderLLMClient(changed_runtime)
    before_transport = _files(root)
    transport = build_story_bible_project(
        project_file, runs_root=args.runs_root,
        consolidation_profile_id=args.consolidation_profile_id,
        story_analysis_profile_path=args.story_analysis_profile,
        semantic_profile_path=args.llm_profile, llm_client=runtime_sentinel,
    )
    if (not transport.reused or transport.a6_manifest_ref != verified.story_analysis_manifest_ref
            or runtime_sentinel.calls or before_transport != _files(root)):
        raise AcceptanceError("runtime-only verification did not preserve exact zero-call reuse")
    return {
        "mode": "verification-only",
        "fresh": {"status": "NOT_RUN"},
        "structural": _structural_report(snapshot, plan, verified, report, chars, events, arcs, structure, bible),
        "exact_rerun": {"reused": rerun.reused, "provider_calls": sentinel.calls, "writes": 0,
                        "same_manifest": rerun.a6_manifest_ref == verified.story_analysis_manifest_ref},
        "runtime_only_reuse": {"reused": transport.reused, "provider_calls": runtime_sentinel.calls,
                                "writes": 0},
        "manual_review_packet": _manual_review_packet(bible, structure, _alice_dossier(snapshot, chars)),
    }


def _run_fresh(args: argparse.Namespace) -> dict[str, Any]:
    project_file, project = _load_project(args.project)
    if project["project_id"] != "a3e-real-novel" or DOCUMENT_ID != "src_001":
        raise AcceptanceError("this gate is fixed to the tracked Alice project/document")
    profile = load_story_analysis_profile(Path(args.story_analysis_profile))
    store, pointers = _stores(args.runs_root, project["project_id"])
    pointer_id = a6_pointer_id(project["project_id"], DOCUMENT_ID, profile.profile_id)
    try:
        pointers.resolve_current(pointer_id)
    except PointerNotFoundError:
        pass
    else:
        raise AcceptanceError("A6 CURRENT already exists; this gate requires one fresh run")

    # Resolve exact validated A5 before constructing a Provider client.
    build_story_analysis_snapshot(
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

    (_verified_project_file, _verified_project, _verified_profile, snapshot, plan,
     verified, report, chars, events, arcs, structure, bible) = _load_verified_current(args)
    if verified.story_analysis_manifest_ref != fresh.a6_manifest_ref:
        raise AcceptanceError("published A6 CURRENT is not exact reusable current")
    verification = _verify_reuse_only(args)
    return {
        "mode": "fresh",
        "fresh": {"seconds": fresh_seconds, "semantic_calls": client.semantic_calls,
                  "provider_attempts": client.provider_attempts, "result": fresh.to_dict()},
        "structural": verification["structural"],
        "exact_rerun": verification["exact_rerun"],
        "runtime_only_reuse": verification["runtime_only_reuse"],
        "manual_review_packet": verification["manual_review_packet"],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    return _verify_reuse_only(args) if args.verification_only else _run_fresh(args)


def main() -> int:
    parser = argparse.ArgumentParser(description="explicit Alice A6G live acceptance")
    parser.add_argument("--project", default=str(PROJECT))
    parser.add_argument("--runs-root", default=str(RUNS_ROOT))
    parser.add_argument("--consolidation-profile-id", default=CONSOLIDATION_PROFILE_ID)
    parser.add_argument("--story-analysis-profile", default=str(STORY_ANALYSIS_PROFILE))
    parser.add_argument("--runtime-config", default=str(RUNTIME_CONFIG))
    parser.add_argument("--llm-profile", default=str(LLM_PROFILE))
    parser.add_argument("--verification-only", action="store_true")
    args = parser.parse_args()
    try:
        print(json.dumps(run(args), ensure_ascii=False, indent=2))
    except Exception as exc:
        print(json.dumps({"valid": False, "error_type": type(exc).__name__, "error": str(exc)}, ensure_ascii=False, indent=2))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
