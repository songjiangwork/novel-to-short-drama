#!/usr/bin/env python3
"""Fresh Alice authority and runtime preflight followed by read-only resume planning."""
from __future__ import annotations

import argparse
import json
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


def _report(plan):
    return {"schema_version": 1, "stage": "B1.3 Alice Resume Preflight", "verdict": plan.verdict,
            "read_only": True, "provider_generation_calls": 0, "windows_total": plan.windows_total,
            "verified_reuse_count": plan.verified_reuse_count,
            "needs_generation_count": plan.needs_generation_count}


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
        report = args.report.resolve()
        if not report.is_relative_to(Path('/tmp')):
            print("report path must be under /tmp", file=sys.stderr)
            return 1
        if any(report.is_relative_to(root.resolve()) for root in (args.story_root, args.checkpoint_root)):
            print("report path must be outside the story and checkpoint roots", file=sys.stderr)
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
        if args.report is not None:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(payload))
        return 0
    except Exception:  # No raw prompts, checkpoint contents, or credentials in diagnostics.
        print("resume preflight failed", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
