#!/usr/bin/env python3
"""B1.3 Alice zero-provider execution-session preparation (dry-run only).

Connects the read-only PR #150 tokenizer/budget preflight to the existing B1.3
production APIs and builds one fail-closed, in-memory
:class:`B13AliceExecutionSession` holding the *exact typed inputs* the B1.3
window executor expects -- without executing any generation.  The session is
created in the same process as the fresh preflight and only after a fully
trusted PASS: the exact A5/A6 + pinned B1.2 authority is revalidated, the model
and server identity are trusted, the verified per-slot ``n_ctx`` is present, an
explicit operator input ceiling is accepted, and every window has both
estimator and tokenizer PASS with its request identity matching the measured
preflight.

This is a preparation boundary, **not** live authorization.  An in-memory
session is not permanent: a future live runner must revalidate authority,
runtime capacity, and request identity immediately before generation.  The CLI
is dry-run only and never exposes ``--execute``/``--live``/``--publish``.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
import b13_alice_token_budget_preflight as pf  # noqa: E402

from short_drama.adaptation import (  # noqa: E402
    AdaptationSourceSnapshot, B13DispositionPreparation, B13WindowSemanticBudget,
    B13WindowSemanticPreparation, build_b13_window_semantic_preparation,
    prepare_b13_disposition_windows,
)
from short_drama.artifacts import ArtifactRef  # noqa: E402
from short_drama.llm import RuntimeConfig  # noqa: E402


class B13AliceSessionError(ValueError):
    """A zero-provider execution-session preparation precondition failed."""


@dataclass(frozen=True, slots=True)
class B13AliceWindowEvidence:
    """Bounded per-window preflight evidence bound into the session."""

    window_id: str
    window_ordinal: int
    request_identity: str
    estimator_budget: str
    tokenizer_budget: str


@dataclass(frozen=True, slots=True)
class B13AliceExecutionSession:
    """Exact, in-memory inputs for ``execute_b13_disposition_windows``.

    One preparation containing every window, one semantic preparation and one
    explicit budget per window, an identical RuntimeConfig throughout, plus the
    minimal metadata that binds the exact trusted preflight evidence.  A future
    runner must revalidate before generation; this is not permanent authorization.
    """

    preparation: B13DispositionPreparation
    semantic_preparations: tuple[B13WindowSemanticPreparation, ...]
    budgets: tuple[B13WindowSemanticBudget, ...]
    runtime_config: RuntimeConfig
    preflight_verdict: str
    request_model: str
    effective_n_ctx: int
    input_ceiling_tokens: int
    strategy_ref: ArtifactRef
    window_evidence: tuple[B13AliceWindowEvidence, ...]


def prepare_b13_alice_execution_session(
    *, store, pointers, source_snapshot: AdaptationSourceSnapshot,
    runtime_config: RuntimeConfig,
    client, input_ceiling_tokens: int, model_context_tokens: int | None = None,
    strategy_ref: ArtifactRef | None = None,
    adaptation_target_id: str = pf.ADAPTATION_TARGET_ID,
) -> B13AliceExecutionSession:
    """Build a fail-closed zero-provider session from a fresh trusted preflight.

    ``strategy_ref`` defaults to the pinned Alice B1.2 strategy (explicit, never
    the CURRENT fallback).  Every gate below must hold or a
    :class:`B13AliceSessionError` is raised with an explicit reason; the
    function never returns a success-shaped object on failure.
    """
    resolved_ref = strategy_ref if strategy_ref is not None else pf.ALICE_STRATEGY_REF
    report = pf.run_preflight(
        store=store, pointers=pointers, source_snapshot=source_snapshot,
        runtime_config=runtime_config, client=client,
        input_ceiling_tokens=input_ceiling_tokens,
        model_context_tokens=model_context_tokens,
        strategy_ref=resolved_ref, adaptation_target_id=adaptation_target_id)
    if report["overall_verdict"] != "PASS":
        raise B13AliceSessionError(
            f"preflight verdict is {report['overall_verdict']!r}; only a full PASS may seed a session")
    cap, policy = report["capacity"], report["policy"]
    if not (cap["server_ok"] and cap["model_identity_established"] and cap["model_matched"]
            and cap["slots_trusted"] and policy["capacity_trusted"]):
        raise B13AliceSessionError(
            f"runtime identity/capacity not trusted (server_ok={cap['server_ok']} "
            f"model_matched={cap['model_matched']} slots_trusted={cap['slots_trusted']} "
            f"capacity_trusted={policy['capacity_trusted']})")
    effective = policy["model_context_tokens"]
    if effective is None or cap["discovered_effective_n_ctx"] is None:
        raise B13AliceSessionError("verified per-slot context is unavailable")
    if input_ceiling_tokens > effective:
        raise B13AliceSessionError(
            f"operator input ceiling {input_ceiling_tokens} exceeds verified n_ctx {effective}")
    rows = report["windows"]
    if not rows or any(row["estimator_budget"] != "PASS" or row["tokenizer_budget"] != "PASS"
                       for row in rows):
        raise B13AliceSessionError("a window lacks estimator+tokenizer PASS evidence")

    # Rebuild the complete preparation and per-window semantic requests through
    # the production APIs; a second independent rebuild revalidates authority.
    preparation = prepare_b13_disposition_windows(
        store=store, pointers=pointers, source_snapshot=source_snapshot,
        adaptation_target_id=adaptation_target_id, strategy_ref=resolved_ref)
    if preparation.source_snapshot != source_snapshot:
        raise B13AliceSessionError("exact A5/A6 source authority changed during preparation")
    if len(preparation.packets) != len(rows):
        raise B13AliceSessionError("window count differs from the measured preflight")
    semantic_preparations: list[B13WindowSemanticPreparation] = []
    evidence: list[B13AliceWindowEvidence] = []
    for packet, row in zip(preparation.packets, rows):
        if packet.window_id != row["window_id"] or packet.window_ordinal != row["window_ordinal"]:
            raise B13AliceSessionError(
                f"window identity {packet.window_id!r}/{packet.window_ordinal} "
                f"differs from preflight {row['window_id']!r}/{row['window_ordinal']}")
        semantic = build_b13_window_semantic_preparation(packet, runtime_config=runtime_config)
        if semantic.request_identity != row["request_identity"]:
            raise B13AliceSessionError(
                f"window {packet.window_id!r} request identity differs from the measured preflight")
        semantic_preparations.append(semantic)
        evidence.append(B13AliceWindowEvidence(
            packet.window_id, packet.window_ordinal, semantic.request_identity,
            row["estimator_budget"], row["tokenizer_budget"]))
    budget = B13WindowSemanticBudget(
        runtime_model_id=runtime_config.request_model,
        model_context_tokens=effective, input_ceiling_tokens=input_ceiling_tokens)
    budgets = tuple(budget for _ in preparation.packets)
    return B13AliceExecutionSession(
        preparation=preparation, semantic_preparations=tuple(semantic_preparations),
        budgets=budgets, runtime_config=runtime_config, preflight_verdict="PASS",
        request_model=runtime_config.request_model, effective_n_ctx=effective,
        input_ceiling_tokens=input_ceiling_tokens, strategy_ref=resolved_ref,
        window_evidence=tuple(evidence))


def _report_payload(session: B13AliceExecutionSession) -> dict:
    """Bounded, secret-safe diagnostic payload (never prompts/bodies/credentials)."""
    source = session.preparation.source_snapshot
    return {
        "schema_version": 1, "stage": "B1.3-B2B execution-session preparation",
        "read_only": True, "dry_run": True,
        "source": {"a5_consolidation_manifest_ref": source.consolidation_manifest_ref.to_dict(),
                    "a6_story_analysis_manifest_ref": source.story_analysis_manifest_ref.to_dict()},
        "strategy_ref": session.strategy_ref.to_dict(),
        "preflight_verdict": session.preflight_verdict, "request_model": session.request_model,
        "effective_n_ctx": session.effective_n_ctx, "input_ceiling_tokens": session.input_ceiling_tokens,
        "windows_total": len(session.preparation.packets),
        "canonical_event_count": len(session.preparation.canonical_event_ids),
        "windows": [{"window_id": e.window_id, "window_ordinal": e.window_ordinal,
                     "request_identity": e.request_identity, "estimator_budget": e.estimator_budget,
                     "tokenizer_budget": e.tokenizer_budget} for e in session.window_evidence],
        "session_readiness": "READY", "provider_generation_calls": 0,
    }


def _print_summary(session: B13AliceExecutionSession) -> None:
    source = session.preparation.source_snapshot
    a5 = source.consolidation_manifest_ref
    a6 = source.story_analysis_manifest_ref
    print("B1.3 Alice execution-session preparation (dry-run; zero-provider)")
    print(f"  source a5={a5.artifact_id} r{a5.revision} "
          f"a6={a6.artifact_id} r{a6.revision}")
    print(f"  strategy={session.strategy_ref.artifact_id} r{session.strategy_ref.revision}")
    print(f"  preflight={session.preflight_verdict} model={session.request_model} "
          f"verified_per_slot_n_ctx={session.effective_n_ctx} "
          f"input_ceiling={session.input_ceiling_tokens}")
    print(f"  windows={len(session.preparation.packets)} "
          f"canonical_events={len(session.preparation.canonical_event_ids)}")
    for item in session.window_evidence:
        print(f"    {item.window_id} ord={item.window_ordinal} "
              f"req={item.request_identity[:16]} est={item.estimator_budget} "
              f"tok={item.tokenizer_budget}")
    print(f"  session readiness=READY provider_generation_calls=0")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="B1.3 Alice zero-provider execution-session preparation (dry-run only)")
    parser.add_argument("--story-root", type=Path, default=pf.DEFAULT_STORY_ROOT)
    parser.add_argument("--base-url", default=pf.DEFAULT_BASE_URL)
    parser.add_argument("--request-model", default=pf.DEFAULT_REQUEST_MODEL)
    parser.add_argument("--provider-family", default=pf.DEFAULT_PROVIDER_FAMILY)
    parser.add_argument("--transport-id", default=pf.DEFAULT_TRANSPORT_ID)
    parser.add_argument("--input-ceiling-tokens", type=int, required=True,
                        help="operator-supplied input ceiling (never auto-defaulted)")
    parser.add_argument("--model-context-tokens", type=int, default=None,
                        help="explicit operator n_ctx override (may only lower verified capacity)")
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--max-response-bytes", type=int, default=4_000_000)
    parser.add_argument("--report", type=Path, default=None,
                        help="optional bounded diagnostic report path (must be under /tmp)")
    args = parser.parse_args(argv)
    if args.report is not None and not Path(args.report).resolve().is_relative_to(Path("/tmp")):
        print("report path must be under /tmp", file=sys.stderr)
        return 1
    try:
        store, pointers, source_snapshot = pf.load_alice_authority(args.story_root)
        runtime_config = RuntimeConfig(
            schema_version=2, transport_id=args.transport_id, base_url=args.base_url,
            request_model=args.request_model, provider_family=args.provider_family,
            credential_environment_name=None, timeout_seconds=args.timeout)
        client = pf.LlamaClient(base_url=pf._server_root(args.base_url),
                                timeout_seconds=args.timeout,
                                max_response_bytes=args.max_response_bytes)
        session = prepare_b13_alice_execution_session(
            store=store, pointers=pointers, source_snapshot=source_snapshot,
            runtime_config=runtime_config, client=client,
            input_ceiling_tokens=args.input_ceiling_tokens,
            model_context_tokens=args.model_context_tokens)
    except Exception as exc:  # noqa: BLE001 - bounded diagnostic (never success-shaped)
        print(f"execution-session preparation failed: {exc}", file=sys.stderr)
        return 1
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(_report_payload(session), ensure_ascii=False, indent=2), encoding="utf-8")
    _print_summary(session)
    return 0


if __name__ == "__main__":
    sys.exit(main())
