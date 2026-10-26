#!/usr/bin/env python3
"""Fresh Alice authority and runtime preflight followed by read-only resume planning."""
from __future__ import annotations

import argparse
import json
import os
import stat
import sys
from pathlib import Path

import b13_alice_runner_preparation as runner
import b13_alice_token_budget_preflight as pf
from short_drama.adaptation.disposition_resume import plan_b13_resume
from short_drama.llm import RuntimeConfig


def prepare_b13_alice_resume(*, store, pointers, source_snapshot, runtime_config, client,
                             input_ceiling_tokens, checkpoint_root, model_context_tokens=None,
                             strategy_ref=None, adaptation_target_id=pf.ADAPTATION_TARGET_ID):
    """Freshly prepare the session and plan in this invocation; never execute."""
    session = runner.prepare_b13_alice_execution_session(
        store=store, pointers=pointers, source_snapshot=source_snapshot,
        runtime_config=runtime_config, client=client, input_ceiling_tokens=input_ceiling_tokens,
        model_context_tokens=model_context_tokens, strategy_ref=strategy_ref,
        adaptation_target_id=adaptation_target_id)
    return plan_b13_resume(store=store, pointers=pointers, session=session, checkpoint_root=checkpoint_root)


_MAX_REPORT_BYTES = 1_000_000


def _report(plan):
    source = plan.source_snapshot
    return {
        "schema_version": 1, "stage": "B1.3 Alice Resume Preflight", "verdict": plan.verdict,
        "read_only": True, "provider_generation_calls": 0,
        "preparation_identity": plan.preparation_identity,
        "strategy_ref": plan.strategy_ref.to_dict(),
        "source": {
            "a5_consolidation_manifest_ref": source.consolidation_manifest_ref.to_dict(),
            "a6_story_analysis_manifest_ref": source.story_analysis_manifest_ref.to_dict(),
        },
        "windows_total": plan.windows_total, "verified_reuse_count": plan.verified_reuse_count,
        "needs_generation_count": plan.needs_generation_count,
        "windows": [{"window_ordinal": window.window_ordinal, "window_id": window.window_id,
                     "request_identity": window.request_identity, "decision": window.decision.value}
                    for window in plan.windows],
    }


def _report_path(path, protected_roots):
    """Validate lexical containment; the writer separately checks every directory."""
    path = Path(path)
    if not path.is_absolute() or ".." in path.parts or not path.is_relative_to(Path("/tmp")):
        raise ValueError("report path must be under /tmp")
    # A direct child of /tmp has no user-controlled parent component that can
    # be moved between validation and creation. Do not follow nested paths.
    if path.parent != Path("/tmp"):
        raise ValueError("report parent must be /tmp")
    if any(path.is_relative_to(Path(root).resolve()) for root in protected_roots):
        raise ValueError("report path must be outside the story and checkpoint roots")
    return path


def _write_report(path, raw):
    """Create a direct /tmp child exclusively without following a symlink.

    The /tmp mount and privileged users are trusted, as is the filesystem.
    Direct children avoid a renameable user-controlled destination parent.
    """
    _report_path(path, ())
    if not all(hasattr(os, name) for name in ("O_NOFOLLOW", "O_DIRECTORY", "O_CLOEXEC")):
        raise OSError("safe report creation is unavailable")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    fd = os.open("/tmp", flags)
    try:
        tmp = os.fstat(fd)
        if (tmp.st_uid != 0 or not stat.S_ISDIR(tmp.st_mode)
                or (tmp.st_mode & 0o022 and not tmp.st_mode & stat.S_ISVTX)):
            raise OSError("unsafe /tmp directory")
        target = os.open(path.name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                         0o600, dir_fd=fd)
        try:
            remaining = memoryview(raw)
            while remaining:
                written = os.write(target, remaining)
                if written <= 0:
                    raise OSError("short report write")
                remaining = remaining[written:]
        finally:
            os.close(target)
    finally:
        os.close(fd)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--story-root", type=Path, default=pf.DEFAULT_STORY_ROOT)
    parser.add_argument("--base-url", default=pf.DEFAULT_BASE_URL)
    parser.add_argument("--request-model", default=pf.DEFAULT_REQUEST_MODEL)
    parser.add_argument("--provider-family", default=pf.DEFAULT_PROVIDER_FAMILY)
    parser.add_argument("--transport-id", default=pf.DEFAULT_TRANSPORT_ID)
    parser.add_argument("--input-ceiling-tokens", type=int, required=True)
    parser.add_argument("--model-context-tokens", type=int)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--max-response-bytes", type=int, default=4_000_000)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    if args.report is not None:
        try:
            _report_path(args.report, (args.story_root, args.checkpoint_root))
        except (OSError, ValueError):
            print("report path must be under /tmp and outside the story and checkpoint roots", file=sys.stderr)
            return 1
    try:
        store, pointers, source = pf.load_alice_authority(args.story_root)
        runtime = RuntimeConfig(schema_version=2, transport_id=args.transport_id, base_url=args.base_url,
                                request_model=args.request_model, provider_family=args.provider_family,
                                credential_environment_name=None, timeout_seconds=args.timeout)
        client = pf.LlamaClient(base_url=pf._server_root(args.base_url), timeout_seconds=args.timeout,
                                max_response_bytes=args.max_response_bytes)
        plan = prepare_b13_alice_resume(
            store=store, pointers=pointers, source_snapshot=source, runtime_config=runtime, client=client,
            input_ceiling_tokens=args.input_ceiling_tokens, model_context_tokens=args.model_context_tokens,
            checkpoint_root=args.checkpoint_root)
        payload = _report(plan)
        raw = (json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii")
        if len(raw) > _MAX_REPORT_BYTES:
            raise ValueError("report exceeds diagnostic size limit")
        if args.report is not None:
            path = _report_path(args.report, (args.story_root, args.checkpoint_root, store.root, pointers.root))
            _write_report(path, raw)
        sys.stdout.write(raw.decode("ascii"))
        return 0
    except Exception:  # No raw prompts, checkpoint contents, or credentials in diagnostics.
        print("resume preflight failed", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
