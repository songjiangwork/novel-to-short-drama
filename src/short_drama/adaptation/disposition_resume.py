"""Read-only B1.3 resume planning; never authorization to execute.

A caller-supplied session cannot prove that its runtime preflight is fresh. The
Alice entry point constructs it in the same invocation. A future runner must
revalidate authority, runtime and each reuse at its execution boundary.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from .disposition_checkpoint import _trusted_session
from .disposition_checkpoint_storage import load_checkpoint
from .disposition_execution import B13WindowAcceptedProposal, _verify_preflight
from .target import AdaptationSourceSnapshot
from short_drama.artifacts import ArtifactRef


class B13ResumeError(RuntimeError):
    """Resume planning failed closed; no part of a plan is usable."""


class B13ResumeDecision(str, Enum):
    VERIFIED_REUSE = "VERIFIED_REUSE"
    NEEDS_GENERATION = "NEEDS_GENERATION"


@dataclass(frozen=True, slots=True)
class B13ResumeWindow:
    window_id: str
    window_ordinal: int
    request_identity: str
    decision: B13ResumeDecision
    accepted_proposal: B13WindowAcceptedProposal | None = None

    def __post_init__(self):
        if not isinstance(self.decision, B13ResumeDecision):
            raise B13ResumeError("invalid resume decision")
        if self.decision is B13ResumeDecision.VERIFIED_REUSE:
            p = self.accepted_proposal
            if (not isinstance(p, B13WindowAcceptedProposal) or p.window_id != self.window_id
                    or p.window_ordinal != self.window_ordinal or p.request_identity != self.request_identity):
                raise B13ResumeError("reuse proposal identity mismatch")
        elif self.accepted_proposal is not None:
            raise B13ResumeError("missing checkpoint cannot contain a proposal")


@dataclass(frozen=True, slots=True)
class B13ResumePlan:
    source_snapshot: AdaptationSourceSnapshot
    strategy_ref: ArtifactRef
    preparation_identity: str
    windows: tuple[B13ResumeWindow, ...]
    schema_version: int = 1
    verdict: str = "READY"

    def __post_init__(self):
        if (self.schema_version != 1 or self.verdict != "READY" or not self.windows
                or any(not isinstance(w, B13ResumeWindow) for w in self.windows)
                or tuple(w.window_ordinal for w in self.windows) != tuple(range(1, len(self.windows) + 1))
                or len({w.window_id for w in self.windows}) != len(self.windows)):
            raise B13ResumeError("resume plan has invalid window order or identity")

    @property
    def windows_total(self) -> int:
        return len(self.windows)

    @property
    def verified_reuse_count(self) -> int:
        return sum(w.decision is B13ResumeDecision.VERIFIED_REUSE for w in self.windows)

    @property
    def needs_generation_count(self) -> int:
        return sum(w.decision is B13ResumeDecision.NEEDS_GENERATION for w in self.windows)


def _verify(store, pointers, session):
    preparation, semantics, runtime = _trusted_session(session)
    _verify_preflight(store=store, pointers=pointers, preparation=preparation,
                      semantic_preparations=semantics, budgets=tuple(session.budgets), runtime=runtime)
    return preparation, semantics


def plan_b13_resume(*, store, pointers, session, checkpoint_root: Path | str) -> B13ResumePlan:
    """Build a complete plan from declared evidence, not standalone authorization.

    Fresh runtime discovery cannot be established by this lower-level API.
    The pre- and post-read checks are verification boundaries, not atomic
    snapshots and cannot detect unobservable ABA changes.
    """
    try:
        preparation, semantics = _verify(store, pointers, session)
        session_evidence = (session.runtime_config, tuple(session.budgets), session.preflight_verdict,
                            session.request_model, session.effective_n_ctx, session.input_ceiling_tokens,
                            session.strategy_ref, tuple(session.window_evidence))
        windows = []
        for packet, semantic in zip(preparation.packets, semantics):
            accepted = load_checkpoint(checkpoint_root, session, packet.window_id)
            decision = (B13ResumeDecision.NEEDS_GENERATION if accepted is None
                        else B13ResumeDecision.VERIFIED_REUSE)
            windows.append(B13ResumeWindow(packet.window_id, packet.window_ordinal,
                                            semantic.request_identity, decision, accepted))
        final_preparation, final_semantics = _verify(store, pointers, session)
        final_evidence = (session.runtime_config, tuple(session.budgets), session.preflight_verdict,
                          session.request_model, session.effective_n_ctx, session.input_ceiling_tokens,
                          session.strategy_ref, tuple(session.window_evidence))
        if (final_preparation != preparation or final_semantics != semantics
                or final_evidence != session_evidence):
            raise B13ResumeError("preparation or session evidence changed during checkpoint loading")
        return B13ResumePlan(preparation.source_snapshot, preparation.strategy_selection.strategy_ref,
                             preparation.preparation_identity, tuple(windows))
    except B13ResumeError:
        raise
    except Exception as exc:
        raise B13ResumeError("resume verification failed") from exc
