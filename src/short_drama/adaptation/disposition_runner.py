"""Bounded checkpoint-on-accept execution; session freshness is caller-owned.

Operators must use the same explicit roots on every invocation. Locks require a
cooperating local filesystem and trusted, stable root parents. Claims are never
reclaimed. This module does not publish a disposition or update CURRENT.
"""
from __future__ import annotations

import os
import stat
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from short_drama.artifacts import FileArtifactStore
from short_drama.foundation import FilePointerStore
from short_drama.llm import LLMClient

from .disposition_checkpoint_storage import (
    confirm_feedback_checkpoint_directory, load_checkpoint, save_checkpoint,
    save_feedback_checkpoint_v2,
)
from .disposition_execution import (
    B13DispositionExecutionResult, B13WindowSemanticGenerationPolicy,
    _execute_one_b13_window, _verify_client_runtime,
)
from .disposition_execution_claim import ClaimState, create_claim, execution_guard, inspect_claim
from .disposition_resume import B13ResumeDecision, plan_b13_resume, inspect_b13_resume_versions


class B13RunnerStatus(str, Enum):
    COMPLETE = "COMPLETE"
    BOUNDED_PROGRESS = "BOUNDED_PROGRESS"
    FAILED = "FAILED"


class B13RunnerError(RuntimeError):
    """Fail-closed outcome; a Provider call may have been interrupted."""

    status = B13RunnerStatus.FAILED


@dataclass(frozen=True, slots=True)
class B13RunnerResult:
    status: B13RunnerStatus
    execution_result: B13DispositionExecutionResult | None = field(repr=False)
    reused_windows: int
    generated_windows: int
    new_semantic_calls: int
    new_provider_attempts: int
    historical_semantic_rounds: int
    remaining_windows: int


def _roots(store, pointers, execution_root, checkpoint_root):
    if (not isinstance(store, FileArtifactStore) or not isinstance(pointers, FilePointerStore)
            or pointers.artifact_store is not store):
        raise B13RunnerError("matching artifact and pointer stores are required")
    paths = []
    for value in (execution_root, checkpoint_root):
        if not isinstance(value, (str, os.PathLike)) or not os.fspath(value):
            raise B13RunnerError("both roots must be explicitly configured")
        raw = Path(value).expanduser()
        if not raw.is_absolute() or raw.name in ('', '.', '..'):
            raise B13RunnerError("roots must be absolute paths with a final component")
        parent = raw.parent.resolve(strict=True)
        # Checkpoint storage requires a non-symlink final parent.
        if parent != raw.parent:
            raise B13RunnerError("root parents must be canonical and stable")
        path = parent / raw.name
        try:
            st = os.lstat(path)
        except FileNotFoundError:
            pass
        else:
            if not stat.S_ISDIR(st.st_mode) or st.st_uid != os.geteuid() or st.st_mode & 0o077:
                raise B13RunnerError("roots must be private owned directories, never symlinks")
        paths.append(path)
    protected = [Path(item.root).resolve(strict=True) for item in (store, pointers)]
    for i, path in enumerate(paths):
        for other in protected + paths[:i]:
            if path.is_relative_to(other) or other.is_relative_to(path):
                raise B13RunnerError("execution, checkpoint and protected roots must be disjoint")
            if path.exists() and other.exists() and os.path.samefile(path, other):
                raise B13RunnerError("root aliases are forbidden")
    return tuple(paths)


def _verify_feedback_client_binding(*, session, llm_client, tokenizer):
    """Reject replaced transports, methods, routing or tokenizer clients."""
    from b13_alice_token_budget_preflight import LlamaClient, _server_root, _urllib_request
    from short_drama.llm import OpenAICompatibleLLMClient
    from short_drama.llm.openai_compatible import UrllibTransport
    from short_drama.llm.retry import real_sleeper, validate_max_attempts

    if (type(tokenizer) is not LlamaClient or tokenizer.transport is not _urllib_request
            or set(vars(tokenizer)) != {"base_url", "transport", "timeout_seconds", "max_response_bytes", "requested_paths"}
            or type(tokenizer.requested_paths) is not list
            or any(type(path) is not str for path in tokenizer.requested_paths)
            or type(llm_client) is not OpenAICompatibleLLMClient
            or set(vars(llm_client)) != {"_runtime", "_transport", "_sleeper", "_max_attempts"}
            or type(llm_client._transport) is not UrllibTransport
            or vars(llm_client._transport) != {}
            or llm_client._sleeper is not real_sleeper
            or getattr(llm_client._transport.post, "__func__", None) is not UrllibTransport.post
            or getattr(llm_client.generate_structured, "__func__", None) is not OpenAICompatibleLLMClient.generate_structured
            or getattr(tokenizer.apply_template, "__func__", None) is not LlamaClient.apply_template
            or getattr(tokenizer.tokenize, "__func__", None) is not LlamaClient.tokenize
            or llm_client.runtime_config != session.runtime_config
            or _server_root(session.runtime_config.base_url) != tokenizer.base_url.rstrip("/")):
        raise B13RunnerError("feedback requires matched live generation/tokenizer runtime")
    try:
        validate_max_attempts(llm_client._max_attempts)
    except Exception:
        raise B13RunnerError("feedback requires a bounded technical retry policy") from None


def _build_feedback_refresh(*, store, pointers, session, llm_client, tokenizer):
    """Bind the actual clients and construct fresh preflight internally."""
    from b13_alice_runner_preparation import prepare_b13_alice_execution_session

    _verify_feedback_client_binding(session=session, llm_client=llm_client, tokenizer=tokenizer)
    def refresh():
        _verify_feedback_client_binding(session=session, llm_client=llm_client, tokenizer=tokenizer)
        return prepare_b13_alice_execution_session(
            store=store, pointers=pointers, source_snapshot=session.preparation.source_snapshot,
            runtime_config=session.runtime_config, client=tokenizer,
            input_ceiling_tokens=session.input_ceiling_tokens,
            model_context_tokens=session.effective_n_ctx, strategy_ref=session.strategy_ref,
            adaptation_target_id=session.preparation.strategy_selection.artifact.strategy.resolved_target.adaptation_target_id)
    return refresh


def _run_b13_resumable_impl(*, store, pointers, session, llm_client, execution_root,
                      checkpoint_root, max_new_windows: int, enable_feedback: bool = False,
                      feedback_tokenizer=None, refresh_session=None,
                      selected_window_ids: tuple[str, ...] | None = None) -> B13RunnerResult:
    """Run at most the explicit number of missing windows under one guard.

    A caller-supplied session does not independently establish runtime freshness.
    Exceptions carry no success-shaped accounting: in-flight attempts are unknown.
    """
    try:
        execution_root, checkpoint_root = _roots(store, pointers, execution_root, checkpoint_root)
        if not isinstance(llm_client, LLMClient):
            raise B13RunnerError("an injected LLMClient is required")
        _verify_client_runtime(llm_client, session.runtime_config)
        if type(enable_feedback) is not bool:
            raise B13RunnerError("feedback mode must be an explicit boolean")
        if enable_feedback:
            if refresh_session is not None:
                raise B13RunnerError("feedback preflight cannot be caller supplied")
            refresh_session = _build_feedback_refresh(store=store, pointers=pointers, session=session,
                                                      llm_client=llm_client, tokenizer=feedback_tokenizer)
            feedback_attempt_limit = getattr(llm_client, "_max_attempts", None)
            if (not isinstance(selected_window_ids, tuple) or not selected_window_ids
                    or any(type(item) is not str or not item for item in selected_window_ids)
                    or len(set(selected_window_ids)) != len(selected_window_ids)):
                raise B13RunnerError("feedback execution requires unique explicit window IDs")
        elif any(value is not None for value in (feedback_tokenizer, refresh_session, selected_window_ids)):
            raise B13RunnerError("feedback-only inputs cannot be supplied in the v1 runner")
        planning = inspect_b13_resume_versions if enable_feedback else plan_b13_resume
        plan = planning(store=store, pointers=pointers, session=session, checkpoint_root=checkpoint_root)
        if enable_feedback and not set(selected_window_ids) <= {w.window_id for w in plan.windows}:
            raise B13RunnerError("selected feedback window is not in the trusted preparation")
        if (isinstance(max_new_windows, bool) or not isinstance(max_new_windows, int)
                or not 0 <= max_new_windows <= plan.windows_total):
            raise B13RunnerError("max_new_windows must be between zero and the complete window count")

        def any_feedback_evidence(resume_plan):
            return any(w.accepted_proposal is not None
                       and w.accepted_proposal.feedback_evidence is not None
                       for w in resume_plan.windows)

        def finish(initial_reused, generated):
            final = planning(store=store, pointers=pointers, session=session, checkpoint_root=checkpoint_root)
            if enable_feedback and any_feedback_evidence(final):
                # Treat a visible post-link target from a failed prior fsync as
                # indeterminate until a guarded fsync durably confirms it.
                confirm_feedback_checkpoint_directory(checkpoint_root)
                final = planning(store=store, pointers=pointers, session=session,
                                 checkpoint_root=checkpoint_root)
            expected = {w.window_id: w.accepted_proposal for w in plan.windows if w.accepted_proposal is not None}
            expected.update({p.window_id: p for p in generated})
            if any(w.accepted_proposal != expected.get(w.window_id) for w in final.windows):
                raise B13RunnerError("checkpoint evidence changed during execution")
            remaining = final.needs_generation_count
            complete = None
            if not remaining:
                prep = session.preparation
                complete = B13DispositionExecutionResult(prep.source_snapshot, prep.strategy_selection,
                    prep.preparation_identity, tuple(w.accepted_proposal for w in final.windows))
            return B13RunnerResult(
                B13RunnerStatus.COMPLETE if complete else B13RunnerStatus.BOUNDED_PROGRESS,
                complete, initial_reused, len(generated), sum(p.semantic_rounds for p in generated),
                sum(p.provider_attempts for p in generated),
                sum(w.accepted_proposal.semantic_rounds for w in final.windows if w.accepted_proposal), remaining)

        if not plan.needs_generation_count:
            if enable_feedback and any_feedback_evidence(plan):
                with execution_guard(execution_root, store=store, pointers=pointers) as guard:
                    _roots(store, pointers, execution_root, checkpoint_root)
                    confirm_feedback_checkpoint_directory(checkpoint_root)
                    guard.verify()
                    return finish(plan.verified_reuse_count, ())
            return finish(plan.verified_reuse_count, ())
        with execution_guard(execution_root, store=store, pointers=pointers) as guard:
            def check_runtime():
                # The runner constructs this guarded preflight itself.
                # Equality to the pinned session is necessary, not proof of
                # cryptographically authenticated serving-model identity.
                guard.verify()
                if refresh_session() != session:
                    raise B13RunnerError("fresh guarded runtime/tokenizer preflight changed")
                if getattr(llm_client, "_max_attempts", None) != feedback_attempt_limit:
                    raise B13RunnerError("feedback retry limit changed")
                _verify_client_runtime(llm_client, session.runtime_config)
                guard.verify()

            def check_feedback_guard():
                guard.verify()
                _verify_feedback_client_binding(session=session, llm_client=llm_client, tokenizer=feedback_tokenizer)
                if getattr(llm_client, "_max_attempts", None) != feedback_attempt_limit:
                    raise B13RunnerError("feedback retry limit changed")
                guard.verify()

            _roots(store, pointers, execution_root, checkpoint_root)
            if enable_feedback:
                check_runtime()
            plan = planning(store=store, pointers=pointers, session=session, checkpoint_root=checkpoint_root)
            if enable_feedback and any_feedback_evidence(plan):
                confirm_feedback_checkpoint_directory(checkpoint_root)
                plan = planning(store=store, pointers=pointers, session=session,
                                checkpoint_root=checkpoint_root)
            missing = [w for w in plan.windows if w.decision is B13ResumeDecision.NEEDS_GENERATION
                       and (not enable_feedback or w.window_id in selected_window_ids)]
            for window in missing:
                if inspect_claim(execution_root, session, window.window_id).state is not ClaimState.ABSENT:
                    raise B13RunnerError("a missing checkpoint has an unresolved or invalid claim")
            # A complete rebuild checks authority, every request and both budget rounds.
            checked = planning(store=store, pointers=pointers, session=session, checkpoint_root=checkpoint_root)
            if checked != plan:
                raise B13RunnerError("resume plan changed during claim inspection")
            guard.verify()
            _verify_client_runtime(llm_client, session.runtime_config)
            generated = []
            for window in missing[:max_new_windows]:
                guard.verify()
                _roots(store, pointers, execution_root, checkpoint_root)
                if enable_feedback:
                    check_runtime()
                create_claim(guard=guard, store=store, pointers=pointers, session=session,
                             checkpoint_root=checkpoint_root, window_id=window.window_id)
                i = window.window_ordinal - 1
                execution_kwargs = dict(store=store, pointers=pointers,
                    preparation=session.preparation, semantic=session.semantic_preparations[i],
                    budget=session.budgets[i], runtime_config=session.runtime_config,
                    llm_client=llm_client, policy=B13WindowSemanticGenerationPolicy())
                if enable_feedback:
                    execution_kwargs.update(_feedback_session=session,
                        _feedback_tokenizer=feedback_tokenizer, _feedback_runtime_check=check_runtime,
                        _feedback_guard_check=check_feedback_guard)
                accepted = _execute_one_b13_window(**execution_kwargs)
                guard.verify()
                _roots(store, pointers, execution_root, checkpoint_root)
                # Rebuild before publishing and verify the saved exact bytes before advancing.
                planning(store=store, pointers=pointers, session=session, checkpoint_root=checkpoint_root)
                if accepted.feedback_evidence is not None:
                    if not enable_feedback:
                        raise B13RunnerError("v2 feedback cannot be written by the legacy runner")
                    save_feedback_checkpoint_v2(checkpoint_root, accepted, session)
                else:
                    save_checkpoint(checkpoint_root, accepted, session)
                if enable_feedback:
                    verified = planning(store=store, pointers=pointers, session=session,
                                        checkpoint_root=checkpoint_root)
                    persisted = next(w.accepted_proposal for w in verified.windows
                                     if w.window_id == window.window_id)
                else:
                    persisted = load_checkpoint(checkpoint_root, session, window.window_id)
                if persisted != accepted:
                    raise B13RunnerError("saved checkpoint did not verify exactly")
                generated.append(accepted)
            guard.verify()
            return finish(plan.verified_reuse_count, generated)
    except B13RunnerError:
        if not enable_feedback:
            raise
    except Exception as exc:
        if not enable_feedback:
            raise B13RunnerError("bounded execution failed closed; a claim may remain unresolved") from exc
    # Do not expose Provider or candidate data even through a direct call to
    # this internal feedback seam. Raise outside the handlers to drop context.
    raise B13RunnerError("feedback execution failed closed; inspect the private execution state")


def run_b13_resumable(*, store, pointers, session, llm_client, execution_root,
                      checkpoint_root, max_new_windows: int) -> B13RunnerResult:
    """Public, legacy v1-only bounded runner; no feedback opt-in surface."""
    return _run_b13_resumable_impl(
        store=store, pointers=pointers, session=session, llm_client=llm_client,
        execution_root=execution_root, checkpoint_root=checkpoint_root,
        max_new_windows=max_new_windows)


def _run_b13_feedback_resumable(*, store, pointers, session, llm_client, execution_root,
                                checkpoint_root, max_new_windows: int,
                                feedback_tokenizer,
                                selected_window_ids: tuple[str, ...]) -> B13RunnerResult:
    """Internal feedback executor with independently checked client binding.

    The supported Alice entry remains responsible for initial preflight and
    live authorization; the internal runner rechecks its actual clients and
    constructs each guarded refresh itself. No underlying exception
    text or cause, including Provider/schema diagnostics, may cross this
    feedback execution boundary.
    """
    failed = False
    try:
        return _run_b13_resumable_impl(
            store=store, pointers=pointers, session=session, llm_client=llm_client,
            execution_root=execution_root, checkpoint_root=checkpoint_root,
            max_new_windows=max_new_windows, enable_feedback=True,
            feedback_tokenizer=feedback_tokenizer,
            selected_window_ids=selected_window_ids)
    except Exception:
        failed = True

    # Intentionally outside the except handler: no __cause__ or __context__
    # retaining schema instances, rejected candidates or Provider messages.
    if failed:
        raise B13RunnerError(
            "feedback execution failed closed; inspect the private execution state")
    raise AssertionError("unreachable feedback failure state")
