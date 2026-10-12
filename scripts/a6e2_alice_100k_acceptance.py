#!/usr/bin/env python3
"""Read-only, checkpointable real-provider Alice A6C -> A6D -> A6E acceptance.

Versioned 100k-ceiling successor to the operator-side ``a6e2_alice_resume.py``
(Issue #102). It runs the exact same acceptance as the original but is pinned
to the REVISDED frozen global-skeleton ceiling (100000) and its resulting
StoryAnalysisProfile / A6B-plan / A6C / A6D identity transition:

* the tracked profile ``profiles/global_story_analysis_v1.yaml`` now freezes
  ``global_skeleton_packet_max_estimated_tokens = 100000`` (was 74000);
* the new identity is deterministic and pinned below (a mismatched profile /
  plan / request identity fails closed);
* all 61 upstream A6C/A6D analyses are regenerated fresh under the new
  identity (the old 74000 checkpoints are NOT reused);
* diagnostics are written OUTSIDE the story runs tree; no A5/A6 persistence or
  CURRENT mutation; the two-round semantic-generation budget is preserved; the
  full global packet is validated before the A6E provider call.

The original operator script is left untouched. Run from the repository root:

  PYTHONPATH=src .venv/bin/python -u scripts/a6e2_alice_100k_acceptance.py --stage all
For a short first run:
  PYTHONPATH=src .venv/bin/python -u scripts/a6e2_alice_100k_acceptance.py --stage all --max-new 3
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
import traceback
import urllib.request
from pathlib import Path

from short_drama.artifacts import FileArtifactStore
from short_drama.foundation import FilePointerStore
from short_drama.llm import OpenAICompatibleLLMClient, PromptRegistry, load_runtime_config
from short_drama.paths import REPO_ROOT
from short_drama.story import (
    CharacterAnalysis, PlotWindowAnalysis, DEFAULT_PROMPT_BASE_DIR,
    build_story_analysis_snapshot, build_story_analysis_plan_from_profile,
    load_story_analysis_profile, load_character_semantic_profile,
    load_window_semantic_profile,
)
from short_drama.story.story_analysis_semantic import (
    build_character_semantic_preparation, _execute_character_request,
    validate_character_evidence, validate_character_coverage,
)
from short_drama.story.story_analysis_window_semantic import (
    build_plot_window_semantic_preparation, _execute_window_request,
    validate_plot_window_evidence, validate_plot_window_coverage,
)
from short_drama.story.story_analysis_global_skeleton_semantic import (
    build_global_skeleton_semantic_preparation,
    load_global_skeleton_semantic_profile,
    resolve_global_skeleton,
)

ALICE_A5_HASH = "02973505f6a499e075d003120ad83305a42f315e5fe1b313a786e8182581903a"
# Revised 100000-ceiling identity (deterministic; see
# docs/v1.2-A6E-1-global-skeleton-measurement-report.md section 11).
FROZEN_PROFILE_HASH = (
    "0fa7bb93fcc7c4e2e09c6075efc528bdb835b6dde398c64cb7bf5495ed9df4a7"
)
FROZEN_PLAN_HASH = (
    "a427e1cfdde2faa9be77fad55594ae4247ca5440d45a1dbf3073fb0e6e6cf34b"
)
FROZEN_CEILING = 100000
DEFAULT_WORK = Path("/tmp/a6e2_alice_uncensored_100k_resume")
STORY_ROOT = REPO_ROOT / "runs" / "a4e_real_novel" / "a3e-real-novel" / "story"
POINTER_FILE = STORY_ROOT / "pointers" / "heads" / "a3e-real-novel.a5.src_001.consolidation-v1.json"


def log(msg: str) -> None:
    print(time.strftime("%Y-%m-%d %H:%M:%S"), msg, flush=True)


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    filename = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                         prefix=".incomplete-", suffix=".json", delete=False) as fp:
            filename = fp.name
            json.dump(payload, fp, ensure_ascii=False, indent=2, sort_keys=True)
            fp.write("\n")
            fp.flush()
            os.fsync(fp.fileno())
        os.replace(filename, path)
    finally:
        if filename and os.path.exists(filename):
            os.unlink(filename)


def read_json(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return data


def current_server_model() -> str:
    url = "http://127.0.0.1:8080/v1/models"
    with urllib.request.urlopen(url, timeout=10) as response:
        obj = json.load(response)
    models = [x.get("id", "") for x in obj.get("data", [])]
    if len(models) != 1 or not models[0]:
        raise RuntimeError(f"Expected exactly one live model at {url}, got {models}")
    return models[0]


def require_memory(min_gib: float) -> None:
    info = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        parts = line.split()
        if parts[0].rstrip(":") in ("MemTotal", "MemAvailable", "SwapFree"):
            info[parts[0].rstrip(":")] = int(parts[1]) / 1024**2
    available = info.get("MemAvailable", 0)
    log(f"WSL memory total={info.get('MemTotal', 0):.1f} GiB; "
        f"available={available:.1f} GiB; swap_free={info.get('SwapFree', 0):.1f} GiB")
    if available < min_gib:
        raise RuntimeError(f"Only {available:.1f} GiB available; "
                           f"safety minimum {min_gib:.1f} GiB. Stop before provider call.")


def checked_checkpoint(path: Path, *, plan_hash: str, profile_hash: str,
                       manifest_hash: str, identity_hash: str, model: str, validate):
    data = read_json(path)
    for field, expected in (("plan_hash", plan_hash),
                            ("story_analysis_profile_hash", profile_hash),
                            ("a5_manifest_hash", manifest_hash),
                            ("request_identity_hash", identity_hash)):
        if data.get(field) != expected:
            raise RuntimeError(f"Stale/mismatched checkpoint {path}: {field} "
                               f"is {data.get(field)!r}, expected {expected!r}")
    cached_model = data.get("provider_model")
    if cached_model and cached_model != model:
        raise RuntimeError(f"Checkpoint {path} came from {cached_model!r}; "
                           f"current live model is {model!r}. Refusing mixed-model acceptance.")
    analysis = validate(data["analysis"])
    return analysis, int(data.get("rounds", 0))


def checkpoint_record(analysis, *, plan_hash: str, profile_hash: str, manifest_hash: str,
                      identity_hash: str, model: str, rounds: int) -> dict:
    return {
        "a5_manifest_hash": manifest_hash,
        "story_analysis_profile_hash": profile_hash,
        "plan_hash": plan_hash,
        "request_identity_hash": identity_hash,
        "provider_model": model,
        "rounds": rounds,
        "analysis": analysis.to_dict(),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=("all", "character", "window", "global"), default="all")
    ap.add_argument("--max-new", type=int, default=0,
                    help="Stop after this many NEW provider items across stages; 0 means unlimited")
    ap.add_argument("--work-dir", type=Path, default=DEFAULT_WORK)
    ap.add_argument("--min-available-gib", type=float, default=6.0,
                    help="Fail closed before a provider call if WSL MemAvailable is lower")
    args = ap.parse_args()
    if args.max_new < 0 or args.min_available_gib < 0:
        ap.error("--max-new and --min-available-gib cannot be negative")

    work = args.work_dir.expanduser()
    work.mkdir(parents=True, exist_ok=True)
    report_path = work / "acceptance_report.json"
    report = {"verdict": "STARTED", "stage": args.stage, "new_provider_items": 0,
              "completed_characters": 0, "completed_windows": 0,
              "global_executed": False}
    pointer_before = POINTER_FILE.read_bytes()
    report["a5_current_pointer_sha256_before"] = hashlib.sha256(pointer_before).hexdigest()

    try:
        log("A6E-2 100k-ceil checkpointable Alice acceptance; diagnostic output only under /tmp")
        actual_model = current_server_model()
        report["live_provider_model"] = actual_model
        log(f"ACTUAL loaded llama.cpp model: {actual_model}")
        if "uncensored" not in actual_model.lower():
            raise RuntimeError("Model ID does not identify Uncensored. "
                               "Check actual /v1/models before proceeding.")
        require_memory(args.min_available_gib)

        artifact_store = FileArtifactStore(STORY_ROOT / "artifacts")
        pointers = FilePointerStore(STORY_ROOT / "pointers", artifact_store)
        snapshot = build_story_analysis_snapshot(
            artifact_store, pointers,
            project_id="a3e-real-novel", document_id="src_001",
            consolidation_profile_id="consolidation-v1")
        profile = load_story_analysis_profile(REPO_ROOT / "profiles" / "global_story_analysis_v1.yaml")
        plan = build_story_analysis_plan_from_profile(snapshot, profile)
        manifest_hash = snapshot.consolidation_manifest_ref.content_hash
        profile_hash = profile.content_hash()
        if manifest_hash != ALICE_A5_HASH:
            raise RuntimeError(f"Unexpected A5 manifest hash: {manifest_hash}")
        if profile_hash != FROZEN_PROFILE_HASH:
            raise RuntimeError(f"Unexpected StoryAnalysisProfile hash: {profile_hash}")
        if plan.plan_hash != FROZEN_PLAN_HASH:
            raise RuntimeError(f"Unexpected A6B plan hash: {plan.plan_hash}")
        ceiling = profile.planning_policy.global_skeleton_packet_max_estimated_tokens
        if ceiling != FROZEN_CEILING:
            raise RuntimeError(f"Global ceiling is not exactly the frozen {FROZEN_CEILING} (got {ceiling})")
        report.update({"a5_manifest_hash": manifest_hash, "a6_plan_hash": plan.plan_hash,
                       "story_analysis_profile_hash": profile_hash,
                       "ceiling": FROZEN_CEILING})
        prompts = PromptRegistry(DEFAULT_PROMPT_BASE_DIR)
        sem = load_character_semantic_profile()
        client = OpenAICompatibleLLMClient(
            load_runtime_config(REPO_ROOT / "profiles" / "llm_local.yaml"))
        cprep = build_character_semantic_preparation(plan, profile, sem, prompts=prompts)
        wsem = load_window_semantic_profile()
        wprep = build_plot_window_semantic_preparation(snapshot, plan, profile, wsem, prompts=prompts)
        log(f"Pinned A5: {manifest_hash}; profile: {profile_hash}; plan: {plan.plan_hash}; "
            f"ceiling={FROZEN_CEILING}; characters={len(cprep.character_refs)}, "
            f"windows={len(wprep.window_ids)}")
        report["a6c_request_identities"] = list(cprep.character_request_identity_hashes)
        report["a6d_request_identities"] = list(wprep.window_request_identity_hashes)

        char_dir, window_dir = work / "characters", work / "windows"
        char_dir.mkdir(exist_ok=True)
        window_dir.mkdir(exist_ok=True)
        characters = []
        for idx, (package, request) in enumerate(zip(plan.character_packages, cprep.character_requests)):
            char_ref = cprep.character_refs[idx]
            cid = cprep.character_request_identity_hashes[idx]
            path = char_dir / f"{char_ref}.json"

            def valid_char(data, p=package):
                obj = CharacterAnalysis.from_dict(data)
                err = validate_character_evidence(obj, p)
                if err:
                    raise RuntimeError(f"A6C cached evidence invalid: {err}")
                return obj

            if path.exists():
                analysis, rounds = checked_checkpoint(path, plan_hash=plan.plan_hash,
                    profile_hash=profile_hash, manifest_hash=manifest_hash,
                    identity_hash=cid, model=actual_model, validate=valid_char)
                log(f"[A6C {idx+1}/{len(cprep.character_refs)}] cache verified: {char_ref}, rounds={rounds}")
            elif args.stage in ("all", "character"):
                if args.max_new and report["new_provider_items"] >= args.max_new:
                    report["verdict"] = "PAUSED"
                    log("PAUSED at --max-new limit; rerun same command to resume")
                    return 0
                require_memory(args.min_available_gib)
                log(f"[A6C {idx+1}/{len(cprep.character_refs)}] real provider START {char_ref}")
                start = time.monotonic()
                analysis, rounds = _execute_character_request(package, request, sem, client)
                log(f"[A6C {idx+1}/{len(cprep.character_refs)}] PASS {char_ref}; "
                    f"rounds={rounds}; elapsed={time.monotonic()-start:.1f}s")
                write_json_atomic(path, checkpoint_record(analysis, plan_hash=plan.plan_hash,
                    profile_hash=profile_hash, manifest_hash=manifest_hash,
                    identity_hash=cid, model=actual_model, rounds=rounds))
                report["new_provider_items"] += 1
            else:
                raise RuntimeError(f"Missing validated A6C checkpoint {path}; run --stage character first")
            characters.append(analysis)
            report["completed_characters"] = len(characters)
        validate_character_coverage(characters, cprep.character_refs)
        log(f"A6C COMPLETE: {len(characters)} of {len(cprep.character_refs)}")
        if args.stage == "character":
            report["verdict"] = "A6C_PASS"
            return 0

        windows = []
        for idx, (packet, request) in enumerate(zip(wprep.windows, wprep.window_requests)):
            window_id = wprep.window_ids[idx]
            wid = wprep.window_request_identity_hashes[idx]
            path = window_dir / f"{window_id}.json"

            def valid_window(data, p=packet):
                obj = PlotWindowAnalysis.from_dict(data)
                err = validate_plot_window_evidence(obj, p)
                if err:
                    raise RuntimeError(f"A6D cached evidence invalid: {err}")
                return obj

            if path.exists():
                analysis, rounds = checked_checkpoint(path, plan_hash=plan.plan_hash,
                    profile_hash=profile_hash, manifest_hash=manifest_hash,
                    identity_hash=wid, model=actual_model, validate=valid_window)
                log(f"[A6D {idx+1}/{len(wprep.window_ids)}] cache verified: {window_id}, rounds={rounds}")
            elif args.stage in ("all", "window"):
                if args.max_new and report["new_provider_items"] >= args.max_new:
                    report["verdict"] = "PAUSED"
                    log("PAUSED at --max-new limit; rerun same command to resume")
                    return 0
                require_memory(args.min_available_gib)
                log(f"[A6D {idx+1}/{len(wprep.window_ids)}] real provider START {window_id}")
                start = time.monotonic()
                analysis, rounds = _execute_window_request(packet, request, wsem, client)
                log(f"[A6D {idx+1}/{len(wprep.window_ids)}] PASS {window_id}; "
                    f"rounds={rounds}; elapsed={time.monotonic()-start:.1f}s")
                write_json_atomic(path, checkpoint_record(analysis, plan_hash=plan.plan_hash,
                    profile_hash=profile_hash, manifest_hash=manifest_hash,
                    identity_hash=wid, model=actual_model, rounds=rounds))
                report["new_provider_items"] += 1
            else:
                raise RuntimeError(f"Missing validated A6D checkpoint {path}; run --stage window first")
            windows.append(analysis)
            report["completed_windows"] = len(windows)
        validate_plot_window_coverage(windows, plan.windows)
        log(f"A6D COMPLETE: {len(windows)} of {len(wprep.window_ids)}")
        if args.stage == "window":
            report["verdict"] = "A6D_PASS"
            return 0

        gsem = load_global_skeleton_semantic_profile()
        # Validate the full global packet (ceiling + coverage + upstream binding)
        # BEFORE any A6E provider call.
        prepared = build_global_skeleton_semantic_preparation(
            snapshot, plan, profile, gsem, characters, windows, prompts=prompts,
            character_request_identity_hashes=cprep.character_request_identity_hashes,
            window_request_identity_hashes=wprep.window_request_identity_hashes,
        )
        report["a6e_packet_estimated_tokens"] = prepared.packet_token_estimate
        report["a6e_rendered_estimated_tokens"] = prepared.rendered_request_token_estimate
        report["a6e_request_identity_hash"] = prepared.request_identity_hash
        log(f"[A6E] packet={prepared.packet_token_estimate}/{FROZEN_CEILING} estimated tokens; "
            f"rendered={prepared.rendered_request_token_estimate}; identity verified")
        if args.max_new and report["new_provider_items"] >= args.max_new:
            report["verdict"] = "PAUSED"
            log("PAUSED before A6E due to --max-new limit; rerun to resume")
            return 0
        # A6E is a single whole-story provider request and is not persisted until fully valid.
        require_memory(args.min_available_gib)
        log("[A6E] REAL provider START (one whole-story request, max 2 semantic rounds)")
        start = time.monotonic()
        result = resolve_global_skeleton(
            snapshot, plan, profile, gsem, characters, windows, client,
            prompts=prompts,
            character_request_identity_hashes=cprep.character_request_identity_hashes,
            window_request_identity_hashes=wprep.window_request_identity_hashes,
        )
        outcome = {
            "a5_manifest_hash": manifest_hash, "plan_hash": plan.plan_hash,
            "story_analysis_profile_hash": profile_hash,
            "request_identity_hash": result.preparation.request_identity_hash,
            "provider_model": actual_model, "rounds": result.rounds_consumed,
            "global_event_analysis": result.global_event_analysis.to_dict(),
            "arc_analysis": result.arc_analysis.to_dict(),
            "global_structure": result.global_structure.to_dict(),
        }
        write_json_atomic(work / "global_skeleton_result.json", outcome)
        report.update({"global_executed": True, "global_rounds": result.rounds_consumed,
                       "global_event_importance": len(result.global_event_analysis.event_importance_overlay),
                       "arcs": len(result.arc_analysis.arcs),
                       "turning_points": len(result.arc_analysis.turning_points),
                       "reveals": len(result.arc_analysis.reveals),
                       "foreshadow_payoffs": len(result.arc_analysis.foreshadow_payoffs),
                       "global_sections": len(result.global_structure.global_sections),
                       "verdict": "PASS"})
        log(f"[A6E] PASS in {time.monotonic()-start:.1f}s; rounds={result.rounds_consumed}")
        return 0
    except Exception as exc:
        report["verdict"] = "FAIL"
        report["error"] = f"{type(exc).__name__}: {exc}"
        log(f"FAIL CLOSED: {report['error']}")
        traceback.print_exc()
        return 2
    finally:
        report["a5_current_pointer_sha256_after"] = hashlib.sha256(POINTER_FILE.read_bytes()).hexdigest()
        report["a5_current_pointer_unchanged"] = (POINTER_FILE.read_bytes() == pointer_before)
        write_json_atomic(report_path, report)
        log(f"A5 CURRENT unchanged: {report['a5_current_pointer_unchanged']}; report: {report_path}")


if __name__ == "__main__":
    sys.exit(main())
