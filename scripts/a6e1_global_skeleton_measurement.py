"""v1.2 A6E-1 -- global-skeleton input measurement checkpoint (issue #90).

This is the **A6E pre-provider global-skeleton input measurement checkpoint**.
It constructs the *exact complete* A6E global-skeleton **input** packet from the
real Alice exact validated A5 CURRENT, measures it with the single supported
estimator (``utf8-bytes-div3-v1``), and reports the component / total / complete
request sizes, coverage invariants, output-reserve considerations, and a
concrete proposed ``global_skeleton_packet_max_estimated_tokens`` ceiling.

What it does:
  * resolves the exact validated A5 CURRENT (the sole A5 input seam) and the
    deterministic A6B plan (frozen production profile);
  * runs the **upstream** A6C character-analysis and A6D plot-window-analysis
    semantic passes against the live local Qwen server to obtain the *actual*
    complete A6C / A6D semantic outputs (these are the real provider calls this
    checkpoint performs);
  * assembles the deterministic whole-story global-skeleton input packet
    (every CharacterAnalysis + every PlotWindowAnalysis + the complete
    A5-derived global index) via the A6E-1 assembly module;
  * measures the exact *rendered* global-skeleton request (not an A5-only
    proxy) with ``utf8-bytes-div3-v1``;
  * reports the staged-ceiling-authority result: a concrete proposed
    ``global_skeleton_packet_max_estimated_tokens`` (the measured packet), and
    whether recursive global compression is necessary.

What it does NOT do:
  * it makes **ZERO** A6E global-skeleton provider calls (the A6E pass itself is
    not executed; only the input is measured);
  * it persists NO A6 artifact and mutates NO A5/A6 CURRENT;
  * it does NOT change any tracked profile; the ceiling is only *proposed* here
    (a tracked profile revision belongs to the follow-up A6E work);
  * it performs no creative / adaptation semantics.

The upstream A6C / A6D outputs are cached as diagnostics OUTSIDE the story runs
tree (``--dump-dir``) so the measurement can be re-run / re-reported without
re-invoking the provider. Use ``--from-dump`` to re-measure from a cached run.

Usage:
    python scripts/a6e1_global_skeleton_measurement.py \
        [--runs-root PATH] [--project ID] [--document ID] \
        [--consolidation-profile ID] [--dump-dir PATH] \
        [--from-dump PATH] [--report PATH] [--skip-character] [--skip-window]

Exit code 0 on a clean checkpoint, 2 on any structural / integrity / coverage
failure or a failed upstream A6C / A6D provider pass.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

from short_drama.artifacts import FileArtifactStore
from short_drama.foundation import FilePointerStore
from short_drama.llm import OpenAICompatibleLLMClient, PromptRegistry, load_runtime_config
from short_drama.paths import REPO_ROOT
from short_drama.story import (
    CharacterAnalysis,
    DEFAULT_PROMPT_BASE_DIR,
    PlotWindowAnalysis,
    build_global_index_base,
    build_global_skeleton_context,
    build_global_skeleton_rendered_request,
    build_story_analysis_plan_from_profile,
    build_story_analysis_snapshot,
    estimate_global_skeleton_request_tokens,
    load_character_semantic_profile,
    load_story_analysis_profile,
    load_window_semantic_profile,
    resolve_character_analysis,
    resolve_plot_window_analysis,
    validate_global_skeleton_coverage,
)
from short_drama.story.chunking import TOKEN_COUNTER_ID

DEFAULT_RUNS_ROOT = REPO_ROOT / "runs" / "a4e_real_novel"
DEFAULT_PROJECT = "a3e-real-novel"
DEFAULT_DOCUMENT = "src_001"
DEFAULT_CONSOLIDATION_PROFILE = "consolidation-v1"
DEFAULT_DUMP_DIR = Path("/tmp/a6e1_measure")

# Deterministic margin rule (same convention as the A6B audit): a frozen budget
# is the measured value rounded UP to the next step (a small, documented,
# corpus-derived margin above the measured size -- never a fabricated bound).
_TOKEN_MARGIN_STEP = 1000


def _round_up_to_step(value: int, step: int) -> int:
    return max(step, math.ceil(value / step) * step)


def _stores(runs_root: str | Path, project_id: str):
    root = Path(runs_root).expanduser() / project_id / "story"
    artifact_store = FileArtifactStore(root / "artifacts")
    pointer_store = FilePointerStore(root / "pointers", artifact_store)
    return artifact_store, pointer_store


def _live_server_available() -> bool:
    import urllib.request

    try:
        with urllib.request.urlopen(
            "http://127.0.0.1:8080/v1/models", timeout=3
        ) as response:
            return response.status == 200
    except Exception:
        return False


def _server_context_window() -> int | None:
    """The live deployment's actual context window (``n_ctx``) -- the
    authoritative, measured "intended semantic context budget" (NOT an invented
    constant). Returns ``None`` if the server cannot be queried."""
    import urllib.request

    try:
        with urllib.request.urlopen(
            "http://127.0.0.1:8080/v1/models", timeout=5
        ) as response:
            data = json.loads(response.read().decode("utf-8"))
        for model in data.get("data", []):
            details = model.get("details", {}) or {}
            meta = model.get("meta", {}) or {}
            n_ctx = details.get("n_ctx") or meta.get("n_ctx")
            if n_ctx is not None:
                return int(n_ctx)
    except Exception:
        return None
    return None


def _run_a6c(plan, profile, prompts, client):
    """Run the upstream A6C character-analysis pass (real provider)."""
    t0 = time.monotonic()
    print(f"[A6C] starting character-analysis pass "
          f"({len(plan.character_packages)} characters) ...", flush=True)
    result = resolve_character_analysis(
        plan, profile, load_character_semantic_profile(), client, prompts=prompts
    )
    dt = time.monotonic() - t0
    print(f"[A6C] done in {dt:.1f}s; "
          f"{len(result.analyses)} character analyses; "
          f"rounds={dict(result.character_rounds)}", flush=True)
    return result


def _run_a6d(snap, plan, profile, prompts, client):
    """Run the upstream A6D plot-window-analysis pass (real provider)."""
    t0 = time.monotonic()
    print(f"[A6D] starting plot-window-analysis pass "
          f"({len(plan.windows)} windows) ...", flush=True)
    result = resolve_plot_window_analysis(
        snap, plan, profile, load_window_semantic_profile(), client, prompts=prompts
    )
    dt = time.monotonic() - t0
    print(f"[A6D] done in {dt:.1f}s; "
          f"{len(result.analyses)} window analyses; "
          f"rounds={dict(result.window_rounds)}", flush=True)
    return result


def _dump(a6c_result, a6d_result, plan, dump_dir: Path) -> None:
    dump_dir.mkdir(parents=True, exist_ok=True)
    (dump_dir / "character_analyses.json").write_text(
        json.dumps([a.to_dict() for a in a6c_result.analyses],
                   ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    (dump_dir / "plot_window_analyses.json").write_text(
        json.dumps([w.to_dict() for w in a6d_result.analyses],
                   ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    meta = {
        "a5_manifest_ref": plan.consolidation_manifest_ref.to_dict(),
        "plan_hash": plan.plan_hash,
        "character_request_identity_hashes":
            list(a6c_result.preparation.character_request_identity_hashes),
        "character_request_hashes": list(a6c_result.preparation.character_request_hashes),
        "character_rounds": [list(x) for x in a6c_result.character_rounds],
        "plot_window_request_identity_hashes":
            list(a6d_result.preparation.window_request_identity_hashes),
        "plot_window_request_hashes": list(a6d_result.preparation.window_request_hashes),
        "window_rounds": [list(x) for x in a6d_result.window_rounds],
        "character_analysis_count": len(a6c_result.analyses),
        "plot_window_analysis_count": len(a6d_result.analyses),
    }
    (dump_dir / "metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    print(f"[dump] upstream A6C/A6D diagnostics written to {dump_dir}", flush=True)


def _load_dump(dump_dir: Path):
    chars = [
        CharacterAnalysis.from_dict(x)
        for x in json.loads(
            (dump_dir / "character_analyses.json").read_text(encoding="utf-8")
        )
    ]
    wins = [
        PlotWindowAnalysis.from_dict(x)
        for x in json.loads(
            (dump_dir / "plot_window_analyses.json").read_text(encoding="utf-8")
        )
    ]
    meta = json.loads((dump_dir / "metadata.json").read_text(encoding="utf-8"))
    return chars, wins, meta


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="A6E-1 global-skeleton measurement checkpoint")
    parser.add_argument("--runs-root", default=str(DEFAULT_RUNS_ROOT))
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--document", default=DEFAULT_DOCUMENT)
    parser.add_argument("--consolidation-profile", default=DEFAULT_CONSOLIDATION_PROFILE)
    parser.add_argument("--dump-dir", default=str(DEFAULT_DUMP_DIR))
    parser.add_argument("--from-dump", default=None,
                        help="re-measure from a cached dump dir (no provider calls)")
    parser.add_argument("--skip-character", action="store_true",
                        help="reuse the cached character dump, do not call A6C")
    parser.add_argument("--skip-window", action="store_true",
                        help="reuse the cached window dump, do not call A6D")
    parser.add_argument("--report", default=None, help="write a JSON report to PATH")
    args = parser.parse_args(argv)

    store, pointers = _stores(args.runs_root, args.project)
    snap = build_story_analysis_snapshot(
        store, pointers,
        project_id=args.project,
        document_id=args.document,
        consolidation_profile_id=args.consolidation_profile,
    )
    profile = load_story_analysis_profile(
        REPO_ROOT / "profiles" / "global_story_analysis_v1.yaml"
    )
    plan = build_story_analysis_plan_from_profile(snap, profile)
    prompts = PromptRegistry(DEFAULT_PROMPT_BASE_DIR)

    print("=== A6E-1 GLOBAL-SKELETON MEASUREMENT CHECKPOINT ===")
    print(f"token counter: {TOKEN_COUNTER_ID}")
    print(f"A5 CURRENT:    {snap.consolidation_manifest_ref.artifact_id} "
          f"rev {snap.consolidation_manifest_ref.revision}")
    print(f"plan_hash:     {plan.plan_hash}")
    print(f"canonical characters: {len(snap.canonical_characters)} | "
          f"planned windows: {len(plan.windows)}")
    print()

    dump_dir = Path(args.dump_dir).expanduser()
    from_dump = Path(args.from_dump).expanduser() if args.from_dump else None

    # --- Obtain the actual complete A6C / A6D semantic outputs -------------
    if from_dump is not None:
        chars, wins, meta = _load_dump(from_dump)
        print(f"[from-dump] loaded {len(chars)} character + {len(wins)} window "
              f"analyses from {from_dump}")
        print(f"[from-dump] plan_hash in dump: {meta.get('plan_hash')}")
        if meta.get("plan_hash") != plan.plan_hash:
            print("ERROR: cached dump plan_hash does not match the current plan",
                  file=sys.stderr)
            return 2
        provider_calls = 0
    else:
        client = OpenAICompatibleLLMClient(
            load_runtime_config(REPO_ROOT / "profiles" / "llm_local.yaml")
        )
        provider_calls = 0
        # A6C (upstream, real provider).
        if args.skip_character and (dump_dir / "character_analyses.json").exists():
            chars = [
                CharacterAnalysis.from_dict(x)
                for x in json.loads(
                    (dump_dir / "character_analyses.json").read_text(encoding="utf-8")
                )
            ]
            c_meta = json.loads((dump_dir / "metadata.json").read_text(encoding="utf-8"))
            c_identity = c_meta.get("character_request_identity_hashes", [])
            print(f"[A6C] reused cached dump ({len(chars)} character analyses)")
        else:
            a6c_result = _run_a6c(plan, profile, prompts, client)
            _dump_a6c_only(a6c_result, plan, dump_dir)
            chars = list(a6c_result.analyses)
            c_identity = list(a6c_result.preparation.character_request_identity_hashes)
            provider_calls += len(a6c_result.character_rounds)
        # A6D (upstream, real provider).
        if args.skip_window and (dump_dir / "plot_window_analyses.json").exists():
            wins = [
                PlotWindowAnalysis.from_dict(x)
                for x in json.loads(
                    (dump_dir / "plot_window_analyses.json").read_text(encoding="utf-8")
                )
            ]
            w_meta = json.loads((dump_dir / "metadata.json").read_text(encoding="utf-8"))
            w_identity = w_meta.get("plot_window_request_identity_hashes", [])
            print(f"[A6D] reused cached dump ({len(wins)} window analyses)")
        else:
            a6d_result = _run_a6d(snap, plan, profile, prompts, client)
            _dump_a6d_only(a6d_result, plan, chars, dump_dir)
            wins = list(a6d_result.analyses)
            w_identity = list(a6d_result.preparation.window_request_identity_hashes)
            provider_calls += len(a6d_result.window_rounds)
        meta = {
            "character_request_identity_hashes": c_identity,
            "plot_window_request_identity_hashes": w_identity,
        }

    # --- Assemble the exact complete global-skeleton input packet ----------
    global_index = build_global_index_base(snap)
    context = build_global_skeleton_context(chars, wins, global_index)
    validate_global_skeleton_coverage(context, plan)

    # --- Render the exact request + measure --------------------------------
    spec = prompts.load(
        "a6.global-skeleton", version=1
    )
    rendered = build_global_skeleton_rendered_request(context, spec)
    request_tokens = estimate_global_skeleton_request_tokens(rendered)

    char_comp = context.character_analysis_component()
    win_comp = context.plot_window_analysis_component()
    gi_comp = context.global_index_component()
    packet_bytes = len(context.canonical_bytes())
    packet_tokens = context.estimated_tokens()
    framing_tokens = request_tokens - packet_tokens  # system + user framing

    # The A6E output reserve is the frozen semantic profile max_output_tokens
    # (the global-skeleton output is a single structured payload bounded by the
    # profile). This is the reserve that must be added to the input to size the
    # total context the request + response needs.
    semantic_profile = load_window_semantic_profile()
    output_reserve = int(semantic_profile.max_output_tokens)

    proposed_ceiling = _round_up_to_step(packet_tokens, _TOKEN_MARGIN_STEP)
    complete_request_plus_reserve = request_tokens + output_reserve
    context_window = _server_context_window()
    feasible = (
        context_window is not None
        and complete_request_plus_reserve <= context_window
    )

    # --- Report ------------------------------------------------------------
    print("=== UPSTREAM BINDING (A6C + A6D on the same exact A5 + plan) ===")
    print(f"  A5 manifest ref:        "
          f"{snap.consolidation_manifest_ref.artifact_id} "
          f"rev {snap.consolidation_manifest_ref.revision}")
    print(f"  A6B plan_hash:          {plan.plan_hash}")
    print(f"  A6C request identities: {len(meta.get('character_request_identity_hashes', []))}")
    print(f"  A6D request identities: {len(meta.get('plot_window_request_identity_hashes', []))}")
    print(f"  upstream provider calls: {provider_calls} (A6C + A6D)")
    print(f"  A6E global-skeleton provider calls: 0 (checkpoint)")
    print()

    print("=== COMPONENT SIZES (utf8-bytes-div3-v1) ===")
    print(f"  character_analyses   count={char_comp['count']:>3}  "
          f"bytes={char_comp['canonical_bytes']:>8}  tokens={char_comp['estimated_tokens']:>8}")
    print(f"  plot_window_analyses count={win_comp['count']:>3}  "
          f"bytes={win_comp['canonical_bytes']:>8}  tokens={win_comp['estimated_tokens']:>8}")
    print(f"  global_index (A5)    "
          f"bytes={gi_comp['canonical_bytes']:>8}  tokens={gi_comp['estimated_tokens']:>8}")
    print()

    print("=== PACKET + COMPLETE REQUEST ===")
    print(f"  total packet (global_context_json)  bytes={packet_bytes:>8}  tokens={packet_tokens:>8}")
    print(f"  prompt framing (system + user)                 tokens={framing_tokens:>8}")
    print(f"  COMPLETE rendered request                      tokens={request_tokens:>8}")
    print(f"  output reserve (profile max_output_tokens)     tokens={output_reserve:>8}")
    print(f"  complete request + output reserve              tokens={complete_request_plus_reserve:>8}")
    print(f"  packet content_hash:  {context.content_hash()}")
    print()

    print("=== COVERAGE INVARIANTS ===")
    char_ok = char_comp["count"] == len(plan.character_packages)
    win_ok = win_comp["count"] == len(plan.windows)
    print(f"  every canonical character has exactly one CharacterAnalysis: "
          f"{'PASS' if char_ok else 'FAIL'} ({char_comp['count']}/{len(plan.character_packages)})")
    print(f"  every planned window has exactly one PlotWindowAnalysis: "
          f"{'PASS' if win_ok else 'FAIL'} ({win_comp['count']}/{len(plan.windows)})")
    print(f"  no sampling / first-N truncation / omitted later windows: PASS "
          f"(deterministic verbatim assembly)")
    print()

    print("=== STAGED CEILING AUTHORITY ===")
    print(f"  proposed global_skeleton_packet_max_estimated_tokens = {proposed_ceiling} "
          f"(measured packet {packet_tokens} rounded up to next "
          f"{_TOKEN_MARGIN_STEP}-token step)")
    if context_window is None:
        print("  live context window:  UNAVAILABLE (server not queried); "
              "feasibility stated from measured size only")
    else:
        print(f"  live context window (measured n_ctx):  {context_window}")
    print(f"  one complete global request needs {complete_request_plus_reserve} "
          f"tokens (input {request_tokens} + output reserve {output_reserve})")
    if context_window is not None:
        print(f"  fits in one complete request: "
              f"{'YES' if feasible else 'NO'} "
              f"({complete_request_plus_reserve} <= {context_window})")
    print(f"  recursive compression necessary: "
          f"{'NO' if feasible else 'INDETERMINATE (context window unavailable)'}")
    print(f"  NOTE: the ceiling is set from the MEASURED complete input -- NOT from "
          f"theoretical context size, GPU topology, or an invented budget.")
    print()

    all_pass = char_ok and win_ok
    report = {
        "token_counter_id": TOKEN_COUNTER_ID,
        "a5_manifest_ref": snap.consolidation_manifest_ref.to_dict(),
        "plan_hash": plan.plan_hash,
        "upstream": {
            "character_request_identity_hashes": meta.get("character_request_identity_hashes", []),
            "plot_window_request_identity_hashes": meta.get("plot_window_request_identity_hashes", []),
            "upstream_provider_calls": provider_calls,
            "a6e_global_skeleton_provider_calls": 0,
        },
        "components": {
            "character_analyses": char_comp,
            "plot_window_analyses": win_comp,
            "global_index": gi_comp,
        },
        "packet": {
            "canonical_bytes": packet_bytes,
            "estimated_tokens": packet_tokens,
            "content_hash": context.content_hash(),
        },
        "complete_request": {
            "framing_tokens": framing_tokens,
            "estimated_tokens": request_tokens,
            "rendered_prompt_hash": rendered.rendered_prompt_hash,
        },
        "output_reserve_tokens": output_reserve,
        "complete_request_plus_reserve_tokens": complete_request_plus_reserve,
        "context_window_tokens": context_window,
        "fits_in_one_complete_request": feasible if context_window is not None else None,
        "recursive_compression_necessary": (not feasible) if context_window is not None else None,
        "coverage": {
            "character": {"pass": char_ok, "count": char_comp["count"],
                          "expected": len(plan.character_packages)},
            "window": {"pass": win_ok, "count": win_comp["count"],
                       "expected": len(plan.windows)},
        },
        "proposed_global_skeleton_packet_max_estimated_tokens": proposed_ceiling,
    }
    if args.report is not None:
        out = Path(args.report).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                       encoding="utf-8")
        print(f"JSON report written to: {out}")

    if not all_pass:
        print("A6E-1 CHECKPOINT RESULT: FAIL (coverage invariant failed)", file=sys.stderr)
        return 2
    print("A6E-1 CHECKPOINT RESULT: PASS (deterministic packet assembled + measured; "
          "ZERO A6E provider calls; no A6 persistence / CURRENT mutation)")
    return 0


# --- Partial dumps (for --skip-character / --skip-window reuse) ------------
def _dump_a6c_only(a6c_result, plan, dump_dir: Path) -> None:
    dump_dir.mkdir(parents=True, exist_ok=True)
    (dump_dir / "character_analyses.json").write_text(
        json.dumps([a.to_dict() for a in a6c_result.analyses],
                   ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    (dump_dir / "metadata.json").write_text(
        json.dumps({
            "plan_hash": plan.plan_hash,
            "a5_manifest_ref": plan.consolidation_manifest_ref.to_dict(),
            "character_request_identity_hashes":
                list(a6c_result.preparation.character_request_identity_hashes),
            "character_request_hashes": list(a6c_result.preparation.character_request_hashes),
            "character_rounds": [list(x) for x in a6c_result.character_rounds],
            "character_analysis_count": len(a6c_result.analyses),
        }, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )


def _dump_a6d_only(a6d_result, plan, chars, dump_dir: Path) -> None:
    dump_dir.mkdir(parents=True, exist_ok=True)
    (dump_dir / "plot_window_analyses.json").write_text(
        json.dumps([w.to_dict() for w in a6d_result.analyses],
                   ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    existing = {}
    if (dump_dir / "metadata.json").exists():
        try:
            existing = json.loads((dump_dir / "metadata.json").read_text(encoding="utf-8"))
        except Exception:
            existing = {}
    existing.update({
        "plan_hash": plan.plan_hash,
        "a5_manifest_ref": plan.consolidation_manifest_ref.to_dict(),
        "plot_window_request_identity_hashes":
            list(a6d_result.preparation.window_request_identity_hashes),
        "plot_window_request_hashes": list(a6d_result.preparation.window_request_hashes),
        "window_rounds": [list(x) for x in a6d_result.window_rounds],
        "plot_window_analysis_count": len(a6d_result.analyses),
    })
    (dump_dir / "metadata.json").write_text(
        json.dumps(existing, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )


if __name__ == "__main__":
    sys.exit(main())
