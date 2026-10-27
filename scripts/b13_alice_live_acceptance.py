#!/usr/bin/env python3
"""Inspect Alice B1.3 readiness or explicitly execute one missing window.

Use the same roots on every invocation. Their existing parents must be trusted,
private, stable directories on a filesystem with reliable locking and fsync.
A failed live invocation may leave a claim and must not be retried automatically.
"""
from __future__ import annotations

import argparse
import json
import os
import stat
import sys
from pathlib import Path

import b13_alice_bounded_runner as bounded
import b13_alice_runner_preparation as preparation
import b13_alice_token_budget_preflight as pf
from short_drama.adaptation.disposition_checkpoint import encode_checkpoint
from short_drama.adaptation.disposition_execution import (
    B13WindowSemanticGenerationPolicy, B13WindowAuthorityError,
    B13WindowProvenanceError, B13WindowGenerationExhaustedError,
)
from short_drama.adaptation.disposition_semantic import B13WindowSchemaError
from short_drama.adaptation.disposition_checkpoint_storage import B13CheckpointStorageError
from short_drama.adaptation.disposition_runner import B13RunnerError
from short_drama.llm.errors import (
    LLMTimeoutError, LLMTransportError, LLMHTTPError, LLMResponseError,
    LLMStructuredOutputError, LLMRetryExhaustedError,
)
from short_drama.adaptation.disposition_execution_claim import ClaimState, inspect_claim
from short_drama.adaptation.disposition_resume import B13ResumeDecision, plan_b13_resume
from short_drama.adaptation.disposition_runner import _roots
from short_drama.llm import DEFAULT_MAX_ATTEMPTS, OpenAICompatibleLLMClient, RuntimeConfig


def _failure_code(exc, phase):
    """Classify known boundaries without exposing exception data or guessing counts.

    Retry exhaustion describes its final technical failure, not earlier attempts.
    Only unwrap the two wrappers whose causes are set by these execution paths.
    """
    while type(exc) in (B13RunnerError, LLMRetryExhaustedError) and exc.__cause__ is not None:
        exc = exc.__cause__
    if isinstance(exc, LLMTimeoutError):
        return 'PROVIDER_TIMEOUT'
    if isinstance(exc, LLMTransportError):
        return 'PROVIDER_TRANSPORT_FAILURE'
    if isinstance(exc, LLMHTTPError):
        return 'PROVIDER_HTTP_FAILURE'
    if isinstance(exc, (LLMResponseError, LLMStructuredOutputError, B13WindowSchemaError)):
        return 'STRUCTURED_OUTPUT_FAILURE'
    if isinstance(exc, B13WindowGenerationExhaustedError):
        return 'SEMANTIC_GENERATION_EXHAUSTED'
    if isinstance(exc, B13WindowProvenanceError):
        return 'PROVENANCE_FAILURE'
    if isinstance(exc, B13WindowAuthorityError) or phase == 'authority':
        return 'AUTHORITY_FAILURE'
    if isinstance(exc, B13CheckpointStorageError):
        return 'CHECKPOINT_STORAGE_FAILURE'
    return 'UNKNOWN_FAILURE'


def _load_authority_read_only(story_root):
    # Store constructors create missing directories; require their full layout first.
    for path in (story_root / "artifacts", story_root / "pointers",
                 story_root / "pointers" / "heads", story_root / "pointers" / "locks"):
        if path.is_symlink() or not path.is_dir():
            raise ValueError("Foundation store layout must already exist")
    return pf.load_alice_authority(story_root)


def _validate_roots(store, pointers, execution_root, checkpoint_root):
    paths = _roots(store, pointers, execution_root, checkpoint_root)
    for path in paths:
        parent = os.stat(path.parent)
        if (not stat.S_ISDIR(parent.st_mode) or parent.st_uid != os.geteuid()
                or parent.st_mode & 0o022):
            raise ValueError('root parents must be owned by the operator and not group/world writable')
    return paths


def _inspect(store, pointers, session, execution_root, checkpoint_root):
    plan = plan_b13_resume(store=store, pointers=pointers, session=session, checkpoint_root=checkpoint_root)
    missing = [w for w in plan.windows if w.decision is B13ResumeDecision.NEEDS_GENERATION]
    claims = [inspect_claim(execution_root, session, w.window_id).state for w in missing]
    if any(state is not ClaimState.ABSENT for state in claims):
        raise ValueError('unresolved or invalid claim')
    return plan, missing


def _timeout(value):
    try:
        seconds = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError('timeout must be a number in (0, 3600] seconds') from exc
    # Match RuntimeConfig's timeout bound, including rejection of NaN and infinity.
    if not 0 < seconds <= 3600:
        raise argparse.ArgumentTypeError('timeout must be in (0, 3600] seconds')
    return seconds


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--story-root', type=Path, default=pf.DEFAULT_STORY_ROOT)
    parser.add_argument('--base-url', default=pf.DEFAULT_BASE_URL)
    parser.add_argument('--input-ceiling-tokens', type=int, required=True)
    parser.add_argument('--model-context-tokens', type=int)
    parser.add_argument('--execution-root', type=Path, required=True)
    parser.add_argument('--checkpoint-root', type=Path, required=True)
    parser.add_argument('--preflight-timeout', type=_timeout, default=120.0)
    parser.add_argument('--generation-timeout', type=_timeout,
                        help='required for live execution; timeout per provider attempt in seconds')
    parser.add_argument('--max-response-bytes', type=int, default=4_000_000)
    parser.add_argument('--execute-live', action='store_true', help='authorize generation of at most one window')
    args = parser.parse_args(argv)
    if args.execute_live and args.generation_timeout is None:
        parser.error('--generation-timeout is required with --execute-live')
    summary = {'status': 'FAILED', 'window_id': None, 'ordinal': None, 'request_identity': None,
               'reused_count': None, 'generated_count': None, 'semantic_rounds': None,
               'observed_provider_attempts': 0, 'checkpoint_identity': None, 'exit_status': 1}
    live_started = False
    phase = 'authority'
    try:
        phase = 'runtime_configuration'
        runtime = RuntimeConfig(schema_version=2, transport_id=pf.DEFAULT_TRANSPORT_ID,
            base_url=args.base_url, request_model=pf.DEFAULT_REQUEST_MODEL,
            provider_family=pf.DEFAULT_PROVIDER_FAMILY, credential_environment_name=None,
            timeout_seconds=(args.generation_timeout if args.generation_timeout is not None
                             else args.preflight_timeout))
        phase = 'authority'
        store, pointers, source = _load_authority_read_only(args.story_root)
        phase = 'roots'
        execution_root, checkpoint_root = _validate_roots(store, pointers, args.execution_root, args.checkpoint_root)
        preflight = pf.LlamaClient(base_url=pf._server_root(args.base_url), timeout_seconds=args.preflight_timeout,
                                   max_response_bytes=args.max_response_bytes)
        common = dict(store=store, pointers=pointers, source_snapshot=source, runtime_config=runtime,
                      input_ceiling_tokens=args.input_ceiling_tokens,
                      model_context_tokens=args.model_context_tokens)
        phase = 'preflight'
        session = preparation.prepare_b13_alice_execution_session(client=preflight, **common)
        phase = 'checkpoint_or_claim_inspection'
        plan, missing = _inspect(store, pointers, session, execution_root, checkpoint_root)
        selected = missing[0] if missing else None
        if selected:
            summary.update(window_id=selected.window_id, ordinal=selected.window_ordinal,
                           request_identity=selected.request_identity)
        summary.update(status='READY' if missing else 'COMPLETE', reused_count=plan.verified_reuse_count,
                       generated_count=0, semantic_rounds=0, exit_status=0)
        if args.execute_live:
            rounds = B13WindowSemanticGenerationPolicy().max_rounds
            print(json.dumps({'max_new_windows': 1, 'max_semantic_rounds': rounds,
                              'max_attempts_per_round': DEFAULT_MAX_ATTEMPTS,
                              'max_provider_attempts': rounds * DEFAULT_MAX_ATTEMPTS,
                              'preflight_timeout_seconds': args.preflight_timeout,
                              'generation_timeout_seconds': args.generation_timeout}), flush=True)
            phase = 'live_execution_or_checkpoint_verification'
            live_started = True
            result = bounded.run_b13_alice_bounded(
                preflight_client=preflight, llm_client=OpenAICompatibleLLMClient(runtime),
                execution_root=execution_root, checkpoint_root=checkpoint_root, max_new_windows=1, **common)
            # Revalidate checkpoint evidence for reporting after the guarded runner returns.
            final = plan_b13_resume(store=store, pointers=pointers, session=session, checkpoint_root=checkpoint_root)
            newly = [w for w in final.windows if w.accepted_proposal is not None
                     and any(old.window_id == w.window_id for old in missing)]
            if result.generated_windows and len(newly) != 1:
                raise ValueError('cannot identify generated checkpoint')
            if newly:
                w = newly[0]
                identity = json.loads(encode_checkpoint(w.accepted_proposal, session))['checkpoint_identity']
                summary.update(window_id=w.window_id, ordinal=w.window_ordinal,
                               request_identity=w.request_identity, checkpoint_identity=identity)
            summary.update(status=result.status.value, reused_count=result.reused_windows,
                           generated_count=result.generated_windows, semantic_rounds=result.new_semantic_calls,
                           observed_provider_attempts=result.new_provider_attempts, exit_status=0)
    except Exception as exc:  # Never print exception text: it may contain provider or source content.
        summary.update(status='FAILED', exit_status=1, failure_phase=phase,
                       error_code=_failure_code(exc, phase))
        if live_started:
            summary.update(generated_count=None, semantic_rounds=None, observed_provider_attempts=None)
    print(json.dumps(summary, sort_keys=True), flush=True)
    return summary['exit_status']


if __name__ == '__main__':
    sys.exit(main())
