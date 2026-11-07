"""Alice orchestration API; no live-generation command line interface."""
import sys
from pathlib import Path

_scripts = str(Path(__file__).resolve().parent)
if _scripts not in sys.path:
    sys.path.insert(0, _scripts)

from b13_alice_runner_preparation import prepare_b13_alice_execution_session
from b13_alice_token_budget_preflight import LlamaClient, _server_root, _urllib_request
from short_drama.adaptation.disposition_runner import (
    B13RunnerError, _run_b13_feedback_resumable, run_b13_resumable,
)
from short_drama.llm import OpenAICompatibleLLMClient
from short_drama.llm.openai_compatible import UrllibTransport


def run_b13_alice_bounded(*, store, pointers, source_snapshot, runtime_config,
                          preflight_client, llm_client, input_ceiling_tokens,
                          execution_root, checkpoint_root, max_new_windows,
                          model_context_tokens=None, strategy_ref=None,
                          enable_feedback=False, selected_window_ids=None):
    """Fresh Alice-only runtime binding; default remains v1-only.

    Production feedback accepts exactly the supported local endpoint types.
    Caller-supplied callbacks or substitute tokenizer clients are not supported.
    """
    if type(enable_feedback) is not bool:
        raise B13RunnerError("feedback mode must be explicitly boolean")
    if enable_feedback:
        if (type(preflight_client) is not LlamaClient
                or preflight_client.transport is not _urllib_request
                or type(llm_client) is not OpenAICompatibleLLMClient
                or type(llm_client._transport) is not UrllibTransport
                or llm_client.runtime_config != runtime_config
                or _server_root(runtime_config.base_url) != preflight_client.base_url.rstrip("/")):
            raise B13RunnerError("production feedback requires a matched live generation/tokenizer runtime")
        if (not isinstance(selected_window_ids, tuple) or not selected_window_ids
                or any(type(item) is not str or not item for item in selected_window_ids)
                or len(set(selected_window_ids)) != len(selected_window_ids)):
            raise B13RunnerError("production feedback requires explicit unique window IDs")
    elif selected_window_ids is not None:
        raise B13RunnerError("window selection requires the guarded production feedback entry")

    def refresh_session():
        # The same server-backed /apply-template, /tokenize and /slots
        # preflight runs under the runner's original execution guard.
        return prepare_b13_alice_execution_session(
            store=store, pointers=pointers, source_snapshot=source_snapshot,
            runtime_config=runtime_config, client=preflight_client,
            input_ceiling_tokens=input_ceiling_tokens, model_context_tokens=model_context_tokens,
            strategy_ref=strategy_ref)

    if enable_feedback:
        failed = False
        try:
            session = refresh_session()
            return _run_b13_feedback_resumable(
                store=store, pointers=pointers, session=session, llm_client=llm_client,
                execution_root=execution_root, checkpoint_root=checkpoint_root,
                max_new_windows=max_new_windows, feedback_tokenizer=preflight_client,
                selected_window_ids=selected_window_ids)
        except Exception:
            failed = True
        # A failed read-only model/tokenizer discovery may also include remote
        # response body text. Do not let any original exception graph escape.
        if failed:
            raise B13RunnerError("Alice feedback preflight or execution failed closed")
    session = refresh_session()
    return run_b13_resumable(
        store=store, pointers=pointers, session=session, llm_client=llm_client,
        execution_root=execution_root, checkpoint_root=checkpoint_root,
        max_new_windows=max_new_windows)
