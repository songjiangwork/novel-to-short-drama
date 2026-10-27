"""Bounded checkpoint-on-accept execution; session freshness is caller-owned.

Operators must use the same explicit roots on every invocation. Locks require a
cooperating local filesystem and trusted, stable root parents. Claims are never
reclaimed. This module does not publish a disposition or update CURRENT.
"""
from __future__ import annotations

import os
import stat
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from short_drama.artifacts import FileArtifactStore
from short_drama.foundation import FilePointerStore
from short_drama.llm import LLMClient

from .disposition_checkpoint_storage import load_checkpoint, save_checkpoint
from .disposition_execution import (
    B13DispositionExecutionResult, B13WindowSemanticGenerationPolicy,
    _execute_one_b13_window, _verify_client_runtime,
)
from .disposition_execution_claim import ClaimState, create_claim, execution_guard, inspect_claim
from .disposition_resume import B13ResumeDecision, plan_b13_resume


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
    execution_result: B13DispositionExecutionResult | None
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


def run_b13_resumable(*, store, pointers, session, llm_client, execution_root,
                      checkpoint_root, max_new_windows: int) -> B13RunnerResult:
    """Run at most the explicit number of missing windows under one guard.

    A caller-supplied session does not independently establish runtime freshness.
    Exceptions carry no success-shaped accounting: in-flight attempts are unknown.
    """
    try:
        execution_root, checkpoint_root = _roots(store, pointers, execution_root, checkpoint_root)
        if not isinstance(llm_client, LLMClient):
            raise B13RunnerError("an injected LLMClient is required")
        _verify_client_runtime(llm_client, session.runtime_config)
        plan = plan_b13_resume(store=store, pointers=pointers, session=session, checkpoint_root=checkpoint_root)
        if (isinstance(max_new_windows, bool) or not isinstance(max_new_windows, int)
                or not 0 <= max_new_windows <= plan.windows_total):
            raise B13RunnerError("max_new_windows must be between zero and the complete window count")

        def finish(initial_reused, generated):
            final = plan_b13_resume(store=store, pointers=pointers, session=session, checkpoint_root=checkpoint_root)
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
            return finish(plan.verified_reuse_count, ())
        with execution_guard(execution_root, store=store, pointers=pointers) as guard:
            _roots(store, pointers, execution_root, checkpoint_root)
            plan = plan_b13_resume(store=store, pointers=pointers, session=session, checkpoint_root=checkpoint_root)
            missing = [w for w in plan.windows if w.decision is B13ResumeDecision.NEEDS_GENERATION]
            for window in missing:
                if inspect_claim(execution_root, session, window.window_id).state is not ClaimState.ABSENT:
                    raise B13RunnerError("a missing checkpoint has an unresolved or invalid claim")
            # A complete rebuild checks authority, every request and both budget rounds.
            checked = plan_b13_resume(store=store, pointers=pointers, session=session, checkpoint_root=checkpoint_root)
            if checked != plan:
                raise B13RunnerError("resume plan changed during claim inspection")
            guard.verify()
            _verify_client_runtime(llm_client, session.runtime_config)
            generated = []
            for window in missing[:max_new_windows]:
                guard.verify()
                _roots(store, pointers, execution_root, checkpoint_root)
                create_claim(guard=guard, store=store, pointers=pointers, session=session,
                             checkpoint_root=checkpoint_root, window_id=window.window_id)
                i = window.window_ordinal - 1
                accepted = _execute_one_b13_window(store=store, pointers=pointers,
                    preparation=session.preparation, semantic=session.semantic_preparations[i],
                    budget=session.budgets[i], runtime_config=session.runtime_config,
                    llm_client=llm_client, policy=B13WindowSemanticGenerationPolicy())
                guard.verify()
                _roots(store, pointers, execution_root, checkpoint_root)
                # Rebuild before publishing and verify the saved exact bytes before advancing.
                plan_b13_resume(store=store, pointers=pointers, session=session, checkpoint_root=checkpoint_root)
                save_checkpoint(checkpoint_root, accepted, session)
                if load_checkpoint(checkpoint_root, session, window.window_id) != accepted:
                    raise B13RunnerError("saved checkpoint did not verify exactly")
                generated.append(accepted)
            guard.verify()
            return finish(plan.verified_reuse_count, generated)
    except B13RunnerError:
        raise
    except Exception as exc:
        raise B13RunnerError("bounded execution failed closed; a claim may remain unresolved") from exc
