"""Offline proofs for Issue #72's bounded A5 execution policy."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import pytest

from short_drama.llm import LLMClient, StructuredGenerationResult, build_structured_request
from short_drama.story import (
    ConsolidationSemanticError,
    FACT_SEMANTIC_PACKING_V1,
    build_fact_semantic_preparation,
    resolve_fact_semantic_ambiguity,
)
from short_drama.story.consolidation_semantic import (
    _execute_prepared_blocks,
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
        packing_policy=FACT_SEMANTIC_PACKING_V1,
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
