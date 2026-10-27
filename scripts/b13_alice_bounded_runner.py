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
                          model_context_tokens=None, strategy_ref=None):
    """Discover runtime and prepare a fresh session within this invocation."""
    session = prepare_b13_alice_execution_session(
        store=store, pointers=pointers, source_snapshot=source_snapshot,
        runtime_config=runtime_config, client=preflight_client,
        input_ceiling_tokens=input_ceiling_tokens, model_context_tokens=model_context_tokens,
        strategy_ref=strategy_ref)
    return run_b13_resumable(store=store, pointers=pointers, session=session,
        llm_client=llm_client, execution_root=execution_root, checkpoint_root=checkpoint_root,
        max_new_windows=max_new_windows)
