"""Alice orchestration API; no live-generation command line interface."""
import sys
from pathlib import Path

_scripts = str(Path(__file__).resolve().parent)
if _scripts not in sys.path:
    sys.path.insert(0, _scripts)

from b13_alice_runner_preparation import prepare_b13_alice_execution_session
from short_drama.adaptation.disposition_runner import run_b13_resumable


def run_b13_alice_bounded(*, store, pointers, source_snapshot, runtime_config,
                          preflight_client, llm_client, input_ceiling_tokens,
                          execution_root, checkpoint_root, max_new_windows,
                          model_context_tokens=None, strategy_ref=None,
                          enable_feedback=False, selected_window_ids=None):
    """Discover runtime and prepare a fresh session within this invocation."""
    def refresh_session():
        # The same server-backed /apply-template, /tokenize and /slots
        # preflight runs under the runner's original execution guard.
        return prepare_b13_alice_execution_session(
            store=store, pointers=pointers, source_snapshot=source_snapshot,
            runtime_config=runtime_config, client=preflight_client,
            input_ceiling_tokens=input_ceiling_tokens, model_context_tokens=model_context_tokens,
            strategy_ref=strategy_ref)

    session = refresh_session()
    return run_b13_resumable(store=store, pointers=pointers, session=session,
        llm_client=llm_client, execution_root=execution_root, checkpoint_root=checkpoint_root,
        max_new_windows=max_new_windows, enable_feedback=enable_feedback,
        feedback_tokenizer=preflight_client if enable_feedback else None,
        refresh_session=refresh_session if enable_feedback else None,
        selected_window_ids=selected_window_ids)
