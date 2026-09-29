"""Offline proofs for Issue #72's bounded A5 execution policy."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import pytest

from short_drama.llm import LLMClient, LLMError, StructuredGenerationResult, build_structured_request
from short_drama.story import (
    ConsolidationProvenanceError,
    ConsolidationSemanticError,
    ConsolidationSemanticGenerationError,
    FACT_SEMANTIC_PACKING_V2,
    build_fact_semantic_preparation,
    resolve_fact_semantic_ambiguity,
)
from short_drama.story.consolidation_semantic import (
    _RetryableSemanticInvalid,
    _execute_prepared_blocks,
    _execute_two_stage_semantic_blocks,
    validate_a5_max_concurrency,
)


@dataclass(frozen=True)
class _Block:
    block_ordinal: int


class _ConcurrentClient(LLMClient):
    supports_concurrent_calls = True

    def generate_structured(self, rendered_prompt, output_schema, semantic_profile):
        raise AssertionError("executor tests do not call the provider boundary")


class _SerialOnlyClient(_ConcurrentClient):
    supports_concurrent_calls = False


def _blocks(count: int):
    return tuple(_Block(index) for index in range(count))


@pytest.mark.parametrize("value", [0, -1, True, False, "2", 1.5, None])
def test_max_concurrency_rejects_non_positive_or_non_integer_values(value):
    with pytest.raises(ConsolidationSemanticError, match="integer >= 1"):
        validate_a5_max_concurrency(value)


@pytest.mark.parametrize("value", [1, 2, 4, 8])
def test_max_concurrency_accepts_positive_integers(value):
    assert validate_a5_max_concurrency(value) == value


@pytest.mark.parametrize("limit", [2, 4])
def test_bounded_executor_never_exceeds_rolling_window_and_reassembles_order(limit):
    blocks = _blocks(12)
    lock = threading.Lock()
    in_flight = 0
    maximum = 0

    def execute(block, _request):
        nonlocal in_flight, maximum
        with lock:
            in_flight += 1
            maximum = max(maximum, in_flight)
        # Reverse delays force non-canonical completion timing.
        time.sleep(0.003 * (len(blocks) - block.block_ordinal))
        with lock:
            in_flight -= 1
        return block.block_ordinal

    result = _execute_prepared_blocks(
        blocks, tuple(object() for _ in blocks), execute,
        llm_client=_ConcurrentClient(), max_concurrency=limit,
    )

    assert maximum <= limit
    assert result == tuple(range(len(blocks)))


def test_serial_executor_calls_in_exact_preparation_order():
    calls: list[int] = []
    blocks = _blocks(5)
    result = _execute_prepared_blocks(
        blocks, tuple(object() for _ in blocks),
        lambda block, _request: calls.append(block.block_ordinal) or block.block_ordinal,
        llm_client=_SerialOnlyClient(), max_concurrency=1,
    )
    assert calls == [0, 1, 2, 3, 4]
    assert result == (0, 1, 2, 3, 4)


def test_unsupported_client_fails_before_any_block_is_started():
    calls: list[int] = []
    blocks = _blocks(4)
    with pytest.raises(ConsolidationSemanticError, match="explicitly supports"):
        _execute_prepared_blocks(
            blocks, tuple(object() for _ in blocks),
            lambda block, _request: calls.append(block.block_ordinal),
            llm_client=_SerialOnlyClient(), max_concurrency=2,
        )
    assert calls == []


def test_failure_stops_new_scheduling_and_propagates_lowest_original_exception():
    blocks = _blocks(8)
    started: list[int] = []
    lock = threading.Lock()
    lower = RuntimeError("lowest ordinal failure")
    higher = ValueError("higher ordinal failure")

    def execute(block, _request):
        with lock:
            started.append(block.block_ordinal)
        if block.block_ordinal == 0:
            # Let ordinal 1 fail first; the scheduler must still wait for this
            # started block and choose its lower ordinal original exception.
            time.sleep(0.04)
            raise lower
        if block.block_ordinal == 1:
            raise higher
        raise AssertionError("no later block may be scheduled after observation")

    with pytest.raises(RuntimeError) as caught:
        _execute_prepared_blocks(
            blocks, tuple(object() for _ in blocks), execute,
            llm_client=_ConcurrentClient(), max_concurrency=2,
        )

    assert caught.value is lower
    assert set(started) == {0, 1}


def test_failure_winner_uses_real_block_ordinal_not_input_sequence_index():
    blocks = (_Block(50), _Block(7))
    ordinal_50 = RuntimeError("ordinal 50 fails first")
    ordinal_7 = ValueError("ordinal 7 fails later but wins")
    ordinal_7_started = threading.Event()

    def execute(block, _request):
        if block.block_ordinal == 50:
            assert ordinal_7_started.wait(timeout=1)
            raise ordinal_50
        ordinal_7_started.set()
        time.sleep(0.03)
        raise ordinal_7

    with pytest.raises(ValueError) as caught:
        _execute_prepared_blocks(
            blocks, (object(), object()), execute,
            llm_client=_ConcurrentClient(), max_concurrency=2,
        )

    assert caught.value is ordinal_7


def _retryable(block: _Block) -> _RetryableSemanticInvalid:
    return _RetryableSemanticInvalid(
        block_id=f"block-{block.block_ordinal}",
        request_hash=f"request-{block.block_ordinal}",
        last_failure_details="semantic-invalid",
        expected_pairs=((f"left-{block.block_ordinal}", f"right-{block.block_ordinal}"),),
    )


def test_two_stage_all_first_round_valid_is_bounded_and_has_no_phase_two_calls():
    blocks = _blocks(8)
    lock = threading.Lock()
    in_flight = 0
    maximum = 0
    calls: list[tuple[int, int]] = []

    def execute(block, _request, semantic_round):
        nonlocal in_flight, maximum
        assert semantic_round == 1
        with lock:
            calls.append((block.block_ordinal, semantic_round))
            in_flight += 1
            maximum = max(maximum, in_flight)
        time.sleep(0.002)
        with lock:
            in_flight -= 1
        return (block.block_ordinal, semantic_round)

    result = _execute_two_stage_semantic_blocks(
        blocks, tuple(object() for _ in blocks), execute,
        llm_client=_ConcurrentClient(), max_concurrency=3,
    )

    assert maximum <= 3
    assert sorted(calls) == [(index, 1) for index in range(8)]
    assert result == tuple((index, 1) for index in range(8))


def test_two_stage_waits_for_all_first_round_calls_to_settle_before_retry():
    blocks = _blocks(3)
    phase_one_finished: set[int] = set()
    lock = threading.Lock()
    slow_block_started = threading.Event()
    release_slow_block = threading.Event()
    retry_observed_after: list[set[int]] = []

    def execute(block, _request, semantic_round):
        if semantic_round == 1:
            if block.block_ordinal == 1:
                slow_block_started.set()
                assert release_slow_block.wait(timeout=1)
            with lock:
                phase_one_finished.add(block.block_ordinal)
            return _retryable(block) if block.block_ordinal == 0 else block.block_ordinal
        with lock:
            retry_observed_after.append(set(phase_one_finished))
        return block.block_ordinal

    runner_errors: list[BaseException] = []

    def run() -> None:
        try:
            _execute_two_stage_semantic_blocks(
                blocks, tuple(object() for _ in blocks), execute,
                llm_client=_ConcurrentClient(), max_concurrency=3,
            )
        except BaseException as exc:  # pragma: no cover - assertion below
            runner_errors.append(exc)

    runner = threading.Thread(target=run)
    runner.start()
    assert slow_block_started.wait(timeout=1)
    # A retry cannot have begun while a started Phase-1 call is still blocked.
    assert retry_observed_after == []
    release_slow_block.set()
    runner.join(timeout=1)
    assert not runner.is_alive()
    assert runner_errors == []
    assert retry_observed_after == [{0, 1, 2}]


def test_two_stage_retry_success_restores_preparation_order_and_round_count():
    blocks = (_Block(30), _Block(10), _Block(20))
    requests = tuple(object() for _ in blocks)
    calls: list[tuple[int, int, int]] = []

    def execute(block, request, semantic_round):
        calls.append((block.block_ordinal, semantic_round, id(request)))
        if semantic_round == 1 and block.block_ordinal == 10:
            return _retryable(block)
        return (block.block_ordinal, semantic_round, id(request))

    result = _execute_two_stage_semantic_blocks(
        blocks, requests, execute, llm_client=_ConcurrentClient(), max_concurrency=3,
    )

    assert [item[:2] for item in calls] == [(30, 1), (10, 1), (20, 1), (10, 2)]
    assert result == (
        (30, 1, id(requests[0])),
        (10, 2, id(requests[1])),
        (20, 1, id(requests[2])),
    )


def test_two_stage_terminal_retry_raises_existing_semantic_generation_error():
    block = _Block(4)

    with pytest.raises(ConsolidationSemanticGenerationError) as caught:
        _execute_two_stage_semantic_blocks(
            (block,), (object(),),
            lambda current, _request, _round: _retryable(current),
            llm_client=_SerialOnlyClient(), max_concurrency=1,
        )

    assert caught.value.block_id == "block-4"
    assert caught.value.rounds_attempted == 2


def test_two_stage_multiple_retries_are_serial_and_ascending_block_ordinal():
    blocks = (_Block(9), _Block(2), _Block(5))
    retry_calls: list[int] = []
    retry_in_flight = 0
    maximum_retry_in_flight = 0

    def execute(block, _request, semantic_round):
        nonlocal retry_in_flight, maximum_retry_in_flight
        if semantic_round == 1:
            return _retryable(block)
        retry_in_flight += 1
        maximum_retry_in_flight = max(maximum_retry_in_flight, retry_in_flight)
        retry_calls.append(block.block_ordinal)
        retry_in_flight -= 1
        return (block.block_ordinal, semantic_round)

    result = _execute_two_stage_semantic_blocks(
        blocks, tuple(object() for _ in blocks), execute,
        llm_client=_ConcurrentClient(), max_concurrency=3,
    )

    assert retry_calls == [2, 5, 9]
    assert maximum_retry_in_flight == 1
    assert result == ((9, 2), (2, 2), (5, 2))


def test_two_stage_lowest_retry_terminal_failure_stops_higher_retry_calls():
    blocks = (_Block(8), _Block(3), _Block(5))
    retry_calls: list[int] = []

    def execute(block, _request, semantic_round):
        if semantic_round == 1:
            return _retryable(block)
        retry_calls.append(block.block_ordinal)
        return _retryable(block) if block.block_ordinal == 3 else block.block_ordinal

    with pytest.raises(ConsolidationSemanticGenerationError) as caught:
        _execute_two_stage_semantic_blocks(
            blocks, tuple(object() for _ in blocks), execute,
            llm_client=_ConcurrentClient(), max_concurrency=3,
        )

    assert caught.value.block_id == "block-3"
    assert retry_calls == [3]


@pytest.mark.parametrize("error", [LLMError("technical"), ConsolidationProvenanceError("bad provenance")])
def test_two_stage_fatal_first_round_errors_do_not_become_deferred_retries(error):
    calls: list[tuple[int, int]] = []

    def execute(block, _request, semantic_round):
        calls.append((block.block_ordinal, semantic_round))
        if block.block_ordinal == 0:
            raise error
        return _retryable(block)

    with pytest.raises(type(error)) as caught:
        _execute_two_stage_semantic_blocks(
            _blocks(2), (object(), object()), execute,
            llm_client=_SerialOnlyClient(), max_concurrency=1,
        )

    assert caught.value is error
    assert calls == [(0, 1)]


def test_fact_resolution_is_identical_after_out_of_order_concurrent_completion(tmp_path):
    """The real fact callback remains canonical despite runtime completion order."""
    from test_story_a5c_fact_preparation import _planning, _tree
    from test_story_a5c_fact_resolution import (
        FakeLLMClient,
        _PROFILE,
        _PROMPTS,
        _SEM_PROFILE,
        _make_provenance,
        _payload_for_block,
        _single_facts_specs,
    )

    planning = _planning(_tree(tmp_path, specs=_single_facts_specs(40)))
    preparation = build_fact_semantic_preparation(
        planning, _PROFILE, _SEM_PROFILE, prompts=_PROMPTS,
        packing_policy=FACT_SEMANTIC_PACKING_V2,
    )
    payloads = {
        request.request_hash: _payload_for_block(block)
        for block, request in zip(preparation.blocks, preparation.structured_requests)
    }
    delays = {
        request.request_hash: 0.003 * (len(preparation.blocks) - block.block_ordinal)
        for block, request in zip(preparation.blocks, preparation.structured_requests)
    }

    class DelayedClient(LLMClient):
        supports_concurrent_calls = True

        def generate_structured(self, rendered_prompt, output_schema, semantic_profile):
            request = build_structured_request(
                rendered_prompt=rendered_prompt, output_schema=output_schema,
                semantic_profile=semantic_profile,
            )
            time.sleep(delays[request.request_hash])
            return StructuredGenerationResult(
                parsed_json=payloads[request.request_hash],
                provenance=_make_provenance(request), attempts=1,
            )

    serial = resolve_fact_semantic_ambiguity(
        planning, _PROFILE, _SEM_PROFILE, FakeLLMClient(list(payloads.values())),
        prompts=_PROMPTS, max_concurrency=1,
    )
    concurrent = resolve_fact_semantic_ambiguity(
        planning, _PROFILE, _SEM_PROFILE, DelayedClient(),
        prompts=_PROMPTS, max_concurrency=4,
    )

    assert concurrent.semantic_decisions == serial.semantic_decisions
    assert concurrent.all_fact_decisions == serial.all_fact_decisions
    assert concurrent.block_results == serial.block_results
    assert tuple(result.block_id for result in concurrent.block_results) == tuple(
        block.block_id for block in preparation.blocks
    )
    assert tuple(request.request_hash for request in concurrent.preparation.structured_requests) == tuple(
        request.request_hash for request in preparation.structured_requests
    )
