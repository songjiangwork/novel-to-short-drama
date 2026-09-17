from __future__ import annotations

import json
import socket as _socket
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from short_drama.llm import (
    LLMConfigError,
    LLMHTTPError,
    LLMResponseError,
    LLMRetryExhaustedError,
    LLMStructuredOutputError,
    LLMTimeoutError,
    LLMTransportError,
    LLMPromptError,
    OpenAICompatibleLLMClient,
    OutputSchema,
    PromptSpec,
    ReasoningSettings,
    RenderedPrompt,
    RuntimeConfig,
    SemanticLLMProfile,
    StructuredGenerationRequest,
    TransportResponse,
    UrllibTransport,
    build_structured_request,
    parse_and_validate_response,
    render_prompt,
    validate_max_attempts,
)


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


def make_envelope_bytes(
    content,
    *,
    response_id: str = "chatcmpl-1",
    finish_reason: str = "stop",
    usage: dict | None = None,
    raw: bytes | None = None,
) -> bytes:
    if raw is not None:
        return raw
    envelope = {
        "id": response_id,
        "object": "chat.completion",
        "created": 1234,
        "model": "qwen",
        "choices": [
            {
                "index": 0,
                "finish_reason": finish_reason,
                "message": {"role": "assistant", "content": content},
            }
        ],
    }
    if usage is not None:
        envelope["usage"] = usage
    return json.dumps(envelope).encode("utf-8")


def ok_response(content, **kwargs) -> TransportResponse:
    return TransportResponse(200, make_envelope_bytes(content, **kwargs))


@dataclass
class RecordedCall:
    url: str
    headers: dict
    body: dict
    timeout_seconds: float


class FakeTransport:
    def __init__(self) -> None:
        self._behaviors: list = []
        self.calls: list[RecordedCall] = []

    def queue(self, *behaviors) -> None:
        self._behaviors.extend(behaviors)

    def post(self, *, url, headers, body, timeout_seconds) -> TransportResponse:
        self.calls.append(
            RecordedCall(url, dict(headers), json.loads(body), timeout_seconds)
        )
        if not self._behaviors:
            raise AssertionError("FakeTransport: no behavior queued")
        behavior = self._behaviors.pop(0)
        if isinstance(behavior, Exception):
            raise behavior
        return behavior


class RecordingSleeper:
    def __init__(self) -> None:
        self.delays: list[float] = []

    def __call__(self, delay: float) -> None:
        self.delays.append(delay)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def make_profile(**overrides) -> SemanticLLMProfile:
    # A semantic profile carries NO backend identity (no provider_family and
    # no model); the backend is supplied by the RuntimeConfig instead.
    values = dict(
        schema_version=1,
        profile_id="story-extraction-llm-v1",
        temperature=0.0,
        max_output_tokens=256,
        structured_output_mode="json_schema",
        reasoning=ReasoningSettings(False),
    )
    values.update(overrides)
    return SemanticLLMProfile(**values)


def make_schema(**overrides) -> OutputSchema:
    values = dict(
        schema_id="candidate-extraction",
        schema_version=1,
        schema={
            "type": "object",
            "additionalProperties": False,
            "required": ["a"],
            "properties": {"a": {"type": "string"}},
        },
    )
    values.update(overrides)
    return OutputSchema.create(**values)


def make_rendered(
    *,
    prompt_id: str = "synthetic-echo",
    version: int = 1,
    system: str = "You are a strict extractor.",
    user: str = "Echo the value: {{text}}",
    text: str = "hello",
) -> RenderedPrompt:
    spec = PromptSpec.create(
        prompt_id=prompt_id,
        version=version,
        system_template=system,
        user_template=user,
        required_variables=["text"],
    )
    return render_prompt(spec, {"text": text})


def make_client(
    transport,
    *,
    sleeper=None,
    **runtime_overrides,
) -> OpenAICompatibleLLMClient:
    values = dict(
        schema_version=1,
        transport_id="llm-local",
        base_url="http://127.0.0.1:8080/v1",
        request_model="qwen",
        provider_family="qwen",
        credential_environment_name=None,
        timeout_seconds=30,
    )
    values.update(runtime_overrides)
    runtime = RuntimeConfig(**values)
    if sleeper is None:
        sleeper = RecordingSleeper()
    return OpenAICompatibleLLMClient(runtime, transport=transport, sleeper=sleeper)


def request_for(
    *,
    profile=None,
    schema=None,
    rendered=None,
) -> StructuredGenerationRequest:
    return build_structured_request(
        rendered_prompt=rendered or make_rendered(),
        output_schema=schema or make_schema(),
        semantic_profile=profile or make_profile(),
    )


# ---------------------------------------------------------------------------
# Request mapping
# ---------------------------------------------------------------------------


def test_request_mapping_json_schema():
    transport = FakeTransport()
    transport.queue(ok_response(json.dumps({"a": "hello"})))
    client = make_client(transport)
    client.generate_structured(make_rendered(), make_schema(), make_profile())
    call = transport.calls[0]
    assert call.url == "http://127.0.0.1:8080/v1/chat/completions"
    assert call.headers["Content-Type"] == "application/json"
    assert "Authorization" not in call.headers
    assert call.body["model"] == "qwen"
    assert call.body["temperature"] == 0.0
    assert call.body["max_tokens"] == 256
    # Reasoning is disabled in the default profile -> an EXPLICIT "none" must
    # be sent so the server startup default never decides behavior.
    assert call.body["reasoning_effort"] == "none"
    assert call.body["messages"] == [
        {"role": "system", "content": "You are a strict extractor."},
        {"role": "user", "content": "Echo the value: hello"},
    ]
    rf = call.body["response_format"]
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["name"] == "candidate-extraction"
    assert rf["json_schema"]["strict"] is True
    assert rf["json_schema"]["schema"]["required"] == ["a"]


def test_request_mapping_json_object():
    transport = FakeTransport()
    transport.queue(ok_response(json.dumps({"a": "hello"})))
    client = make_client(transport)
    client.generate_structured(
        make_rendered(), make_schema(), make_profile(structured_output_mode="json_object")
    )
    assert transport.calls[0].body["response_format"] == {"type": "json_object"}


def test_request_mapping_none_omits_response_format():
    transport = FakeTransport()
    transport.queue(ok_response(json.dumps({"a": "hello"})))
    client = make_client(transport)
    client.generate_structured(
        make_rendered(), make_schema(), make_profile(structured_output_mode="none")
    )
    assert "response_format" not in transport.calls[0].body


def test_request_mapping_omits_empty_system():
    transport = FakeTransport()
    transport.queue(ok_response(json.dumps({"a": "hello"})))
    client = make_client(transport)
    client.generate_structured(
        make_rendered(system=""), make_schema(), make_profile()
    )
    assert transport.calls[0].body["messages"] == [
        {"role": "user", "content": "Echo the value: hello"}
    ]


def test_request_mapping_sends_auth_header_when_credential_set(monkeypatch):
    monkeypatch.setenv("LLM_KEY", "sekrit")
    transport = FakeTransport()
    transport.queue(ok_response(json.dumps({"a": "hello"})))
    client = make_client(transport, credential_environment_name="LLM_KEY")
    client.generate_structured(make_rendered(), make_schema(), make_profile())
    assert transport.calls[0].headers["Authorization"] == "Bearer sekrit"


# ---------------------------------------------------------------------------
# Finding 3: reasoning settings map to a concrete request field.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "settings,expected",
    [
        (ReasoningSettings(False), "none"),
        (ReasoningSettings(True, "low"), "low"),
        (ReasoningSettings(True, "medium"), "medium"),
        (ReasoningSettings(True, "high"), "high"),
    ],
)
def test_request_mapping_reasoning_effort_explicit(settings, expected):
    transport = FakeTransport()
    transport.queue(ok_response(json.dumps({"a": "hello"})))
    client = make_client(transport)
    client.generate_structured(
        make_rendered(), make_schema(), make_profile(reasoning=settings)
    )
    assert transport.calls[0].body["reasoning_effort"] == expected


def test_build_request_body_reflects_reasoning():
    # build_request_body is a read-only inspection hook (no transport used).
    client = make_client(FakeTransport())
    body = client.build_request_body(
        make_rendered(), make_schema(), make_profile(reasoning=ReasoningSettings(True, "low"))
    )
    assert body["reasoning_effort"] == "low"
    assert body["model"] == "qwen"


def test_request_body_and_provenance_use_runtime_config_model():
    """The provider request body's model and the provenance backend identity
    come from the RuntimeConfig (request_model / provider_family), NOT from the
    semantic profile."""

    transport = FakeTransport()
    transport.queue(ok_response(json.dumps({"a": "hello"})))
    client = make_client(
        transport, request_model="ggml-org/Gemma-27B:Q8_0", provider_family="gemma"
    )
    result = client.generate_structured(make_rendered(), make_schema(), make_profile())
    # The provider request body uses the runtime-config model.
    assert transport.calls[0].body["model"] == "ggml-org/Gemma-27B:Q8_0"
    # The provenance records the backend family + model from the runtime config.
    assert result.provenance.provider_family == "gemma"
    assert result.provenance.model == "ggml-org/Gemma-27B:Q8_0"


# ---------------------------------------------------------------------------
# Finding 4: a provider length truncation is invalid structured output.
# ---------------------------------------------------------------------------


def test_length_truncation_rejected_even_if_json_and_schema_valid():
    # Structurally valid JSON that ALSO passes the local schema, BUT the
    # provider reported finish_reason=length -> must be rejected as invalid
    # structured output (never silently accepted as a valid extraction).
    response = ok_response(json.dumps({"a": "hello"}), finish_reason="length")
    with pytest.raises(LLMStructuredOutputError, match="truncated"):
        parse_and_validate_response(response, make_schema())


def test_stop_finish_reason_accepted_by_parser():
    response = ok_response(json.dumps({"a": "hello"}), finish_reason="stop")
    parsed, meta = parse_and_validate_response(response, make_schema())
    assert parsed == {"a": "hello"}
    assert meta.finish_reason == "stop"


def test_length_truncation_never_accepted_end_to_end():
    # End-to-end: a length truncation is a retryable structured-output failure
    # (per the contract). It is retried and then surfaces as retry-exhausted,
    # so a truncated (even schema-valid) extraction is never returned as valid.
    transport = FakeTransport()
    for _ in range(3):
        transport.queue(
            ok_response(json.dumps({"a": "hello"}), finish_reason="length")
        )
    client = make_client(transport)
    with pytest.raises(LLMRetryExhaustedError):
        client.generate_structured(make_rendered(), make_schema(), make_profile())
    assert len(transport.calls) == 3


# ---------------------------------------------------------------------------
# Provider capability
# ---------------------------------------------------------------------------


def test_capability_supports_json_schema():
    assert "json_schema" in OpenAICompatibleLLMClient.supported_structured_output_modes


def test_capability_unsupported_mode_fails_before_request():
    class _NoSchemaClient(OpenAICompatibleLLMClient):
        supported_structured_output_modes = frozenset({"none"})

    transport = FakeTransport()
    client = _NoSchemaClient(
        RuntimeConfig(1, "t", "http://127.0.0.1:8080/v1", "qwen", "qwen", None, 30),
        transport=transport,
        sleeper=lambda d: None,
    )
    with pytest.raises(LLMConfigError, match="does not support"):
        client.generate_structured(
            make_rendered(), make_schema(), make_profile(structured_output_mode="json_schema")
        )
    assert transport.calls == []  # no request was sent


# ---------------------------------------------------------------------------
# Response parsing / structured output
# ---------------------------------------------------------------------------


def test_valid_provider_response():
    transport = FakeTransport()
    transport.queue(
        ok_response(
            json.dumps({"a": "hello"}),
            response_id="chatcmpl-xyz",
            finish_reason="stop",
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )
    )
    client = make_client(transport)
    result = client.generate_structured(make_rendered(), make_schema(), make_profile())
    assert result.attempts == 1
    assert result.parsed_json == {"a": "hello"}


def test_malformed_envelope_is_retryable_and_exhausts():
    transport = FakeTransport()
    bad = TransportResponse(200, b'{"error": "boom"}')
    transport.queue(bad, bad, bad)
    client = make_client(transport)
    with pytest.raises(LLMRetryExhaustedError) as exc:
        client.generate_structured(make_rendered(), make_schema(), make_profile())
    assert exc.value.attempts == 3
    assert len(transport.calls) == 3
    assert LLMResponseError.retryable is True


def test_empty_content_is_retryable_and_exhausts():
    transport = FakeTransport()
    transport.queue(ok_response(""), ok_response(""), ok_response("   "))
    client = make_client(transport)
    with pytest.raises(LLMRetryExhaustedError):
        client.generate_structured(make_rendered(), make_schema(), make_profile())
    assert len(transport.calls) == 3


def test_invalid_json_is_retryable_and_exhausts():
    transport = FakeTransport()
    transport.queue(
        ok_response("not-json"), ok_response("{"), ok_response('{"a": }')
    )
    client = make_client(transport)
    with pytest.raises(LLMRetryExhaustedError):
        client.generate_structured(make_rendered(), make_schema(), make_profile())
    assert len(transport.calls) == 3


def test_schema_violation_is_retryable_and_exhausts():
    transport = FakeTransport()
    bad = ok_response(json.dumps({"b": 1}))  # missing required 'a'
    transport.queue(bad, bad, bad)
    client = make_client(transport)
    with pytest.raises(LLMRetryExhaustedError):
        client.generate_structured(make_rendered(), make_schema(), make_profile())
    assert len(transport.calls) == 3


def test_structured_output_error_is_retryable():
    assert LLMStructuredOutputError.retryable is True


# ---------------------------------------------------------------------------
# HTTP retryability
# ---------------------------------------------------------------------------


def test_http_429_then_success():
    transport = FakeTransport()
    transport.queue(
        TransportResponse(429, b'{"error": "rate limited"}'),
        ok_response(json.dumps({"a": "hello"})),
    )
    client = make_client(transport)
    result = client.generate_structured(make_rendered(), make_schema(), make_profile())
    assert result.attempts == 2
    assert len(transport.calls) == 2


def test_http_500_then_success():
    transport = FakeTransport()
    transport.queue(
        TransportResponse(500, b'{"error": "oops"}'),
        ok_response(json.dumps({"a": "hello"})),
    )
    client = make_client(transport)
    result = client.generate_structured(make_rendered(), make_schema(), make_profile())
    assert result.attempts == 2


def test_http_503_is_retryable():
    transport = FakeTransport()
    transport.queue(
        TransportResponse(503, b"unavailable"),
        ok_response(json.dumps({"a": "hello"})),
    )
    client = make_client(transport)
    result = client.generate_structured(make_rendered(), make_schema(), make_profile())
    assert result.attempts == 2


def test_http_400_is_not_retryable():
    transport = FakeTransport()
    transport.queue(
        TransportResponse(400, b'{"error": "bad request"}'),
        TransportResponse(400, b'{"error": "bad request"}'),
    )
    client = make_client(transport)
    with pytest.raises(LLMHTTPError) as exc:
        client.generate_structured(make_rendered(), make_schema(), make_profile())
    assert exc.value.status == 400
    assert exc.value.retryable is False
    assert len(transport.calls) == 1  # no retry


def test_http_404_is_not_retryable():
    transport = FakeTransport()
    transport.queue(TransportResponse(404, b"not found"))
    client = make_client(transport)
    with pytest.raises(LLMHTTPError) as exc:
        client.generate_structured(make_rendered(), make_schema(), make_profile())
    assert exc.value.status == 404
    assert exc.value.retryable is False
    assert len(transport.calls) == 1


def test_http_401_is_not_retryable():
    transport = FakeTransport()
    transport.queue(TransportResponse(401, b'{"error": "unauthorized"}'))
    client = make_client(transport)
    with pytest.raises(LLMHTTPError) as exc:
        client.generate_structured(make_rendered(), make_schema(), make_profile())
    assert exc.value.status == 401
    assert exc.value.retryable is False
    assert len(transport.calls) == 1


def test_timeout_is_retryable_then_success():
    transport = FakeTransport()
    transport.queue(
        LLMTimeoutError("timed out"),
        ok_response(json.dumps({"a": "hello"})),
    )
    client = make_client(transport)
    result = client.generate_structured(make_rendered(), make_schema(), make_profile())
    assert result.attempts == 2


def test_connection_failure_is_retryable_then_success():
    transport = FakeTransport()
    transport.queue(
        LLMTransportError("connection refused"),
        ok_response(json.dumps({"a": "hello"})),
    )
    client = make_client(transport)
    result = client.generate_structured(make_rendered(), make_schema(), make_profile())
    assert result.attempts == 2


def test_3_attempt_exhaustion():
    transport = FakeTransport()
    transport.queue(
        TransportResponse(500, b"e"),
        TransportResponse(500, b"e"),
        TransportResponse(500, b"e"),
    )
    client = make_client(transport)
    with pytest.raises(LLMRetryExhaustedError) as exc:
        client.generate_structured(make_rendered(), make_schema(), make_profile())
    assert exc.value.attempts == 3
    assert len(transport.calls) == 3


def test_retry_uses_identical_semantic_request():
    transport = FakeTransport()
    transport.queue(
        TransportResponse(429, b"rate limited"),
        ok_response(json.dumps({"a": "hello"})),
    )
    client = make_client(transport)
    client.generate_structured(make_rendered(), make_schema(), make_profile())
    assert len(transport.calls) == 2
    first, second = transport.calls
    assert first.url == second.url
    assert first.body == second.body
    assert first.timeout_seconds == second.timeout_seconds


def test_no_real_sleep_in_tests():
    started = time.monotonic()
    sleeper = RecordingSleeper()
    transport = FakeTransport()
    transport.queue(
        TransportResponse(500, b"e"),
        TransportResponse(500, b"e"),
        ok_response(json.dumps({"a": "hello"})),
    )
    client = make_client(transport, sleeper=sleeper)
    result = client.generate_structured(make_rendered(), make_schema(), make_profile())
    elapsed = time.monotonic() - started
    assert result.attempts == 3
    # Two retries -> two injected backoff sleeps (recorded, not real).
    assert len(sleeper.delays) == 2
    assert all(delay > 0 for delay in sleeper.delays)
    # The test must not actually sleep for the backoff.
    assert elapsed < 1.0


def test_max_attempts_three_is_the_ceiling_and_works():
    # 3 is the maximum allowed budget and still functions (exhausts after 3).
    sleeper = RecordingSleeper()
    transport = FakeTransport()
    for _ in range(3):
        transport.queue(TransportResponse(500, b"e"))
    client = OpenAICompatibleLLMClient(
        RuntimeConfig(1, "t", "http://127.0.0.1:8080/v1", "qwen", "qwen", None, 30),
        transport=transport,
        sleeper=sleeper,
        max_attempts=3,
    )
    with pytest.raises(LLMRetryExhaustedError) as exc:
        client.generate_structured(make_rendered(), make_schema(), make_profile())
    assert exc.value.attempts == 3
    assert len(transport.calls) == 3


def test_max_attempts_above_three_rejected():
    # The contract caps the retry budget at 1..3; 4 must be rejected before any
    # request is sent.
    with pytest.raises(LLMConfigError):
        OpenAICompatibleLLMClient(
            RuntimeConfig(1, "t", "http://127.0.0.1:8080/v1", "qwen", "qwen", None, 30),
            transport=FakeTransport(),
            sleeper=lambda d: None,
            max_attempts=4,
        )


def test_invalid_max_attempts_rejected():
    with pytest.raises(LLMConfigError):
        OpenAICompatibleLLMClient(
            RuntimeConfig(1, "t", "http://127.0.0.1:8080/v1", "qwen", "qwen", None, 30),
            transport=FakeTransport(),
            sleeper=lambda d: None,
            max_attempts=0,
        )


def test_validate_max_attempts_rejects_out_of_range():
    assert validate_max_attempts(1) == 1
    assert validate_max_attempts(3) == 3
    for bad in (0, 4, 10, -1, "3", True):
        with pytest.raises(LLMConfigError):
            validate_max_attempts(bad)


# ---------------------------------------------------------------------------
# OutputSchema error taxonomy: invalid schema input is a non-retryable config
# error, never a raw canonical-serialization / low-level parser exception.
# ---------------------------------------------------------------------------


def test_output_schema_create_rejects_nan_as_config_error():
    # A non-canonical numeric value (NaN) is rejected by the canonical JSON
    # authority; it must surface as a non-retryable LLMConfigError, not a raw
    # CanonicalSerializationError / ArtifactError / ValueError.
    schema = {"type": "object", "properties": {"a": {"const": float("nan")}}}
    with pytest.raises(LLMConfigError) as exc:
        OutputSchema.create(schema_id="s", schema_version=1, schema=schema)
    assert exc.value.retryable is False


def test_output_schema_create_rejects_inf_as_config_error():
    schema = {"type": "object", "properties": {"a": {"const": float("inf")}}}
    with pytest.raises(LLMConfigError):
        OutputSchema.create(schema_id="s", schema_version=1, schema=schema)


def test_output_schema_create_rejects_nested_nan_as_config_error():
    schema = {"type": "object", "properties": {"a": {"enum": [1, float("nan")]}}}
    with pytest.raises(LLMConfigError):
        OutputSchema.create(schema_id="s", schema_version=1, schema=schema)


def test_output_schema_from_dict_rejects_nan_as_config_error():
    value = {
        "schema_id": "s",
        "schema_version": 1,
        "schema_hash": "0" * 64,
        "schema": {"type": "object", "properties": {"a": {"const": float("nan")}}},
    }
    with pytest.raises(LLMConfigError) as exc:
        OutputSchema.from_dict(value)
    assert exc.value.retryable is False


def test_output_schema_create_accepts_canonical_schema():
    # A canonical, valid schema still constructs fine (regression guard).
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["a"],
        "properties": {"a": {"type": "string"}},
    }
    spec = OutputSchema.create(schema_id="s", schema_version=1, schema=schema)
    assert len(spec.schema_hash) == 64
    assert spec.schema == schema


# ---------------------------------------------------------------------------
# Request fingerprint
# ---------------------------------------------------------------------------


def test_fingerprint_stable():
    a = request_for().fingerprint
    b = request_for().fingerprint
    assert a == b
    assert a.request_hash == b.request_hash
    assert len(a.request_hash) == 64


def test_fingerprint_prompt_change_alters_it():
    base = request_for().fingerprint
    alt = request_for(rendered=make_rendered(user="Different: {{text}}")).fingerprint
    assert base.request_hash != alt.request_hash
    assert base.prompt_content_hash != alt.prompt_content_hash
    assert base.rendered_prompt_hash != alt.rendered_prompt_hash


def test_fingerprint_prompt_variable_change_alters_it():
    base = request_for().fingerprint
    alt = request_for(rendered=make_rendered(text="other")).fingerprint
    assert base.request_hash != alt.request_hash
    assert base.rendered_prompt_hash != alt.rendered_prompt_hash


def test_fingerprint_schema_change_alters_it():
    base = request_for().fingerprint
    alt = request_for(
        schema=make_schema(schema={"type": "object", "required": ["a", "c"]})
    ).fingerprint
    assert base.output_schema_hash != alt.output_schema_hash
    assert base.request_hash != alt.request_hash


def test_backend_model_change_does_not_alter_request_hash():
    """Changing the backend model does NOT change the A-I3 request hash.

    The concrete backend model is runtime identity (RuntimeConfig.request_model),
    not part of the semantic request. Switching Qwen -> Gemma keeps the
    request_hash / semantic_profile_hash identical; only the recorded backend
    identity (provider_family / model) in the provenance differs.
    """

    qwen_transport = FakeTransport()
    qwen_transport.queue(ok_response(json.dumps({"a": "hello"})))
    qwen_result = make_client(qwen_transport).generate_structured(
        make_rendered(), make_schema(), make_profile()
    )

    gemma_transport = FakeTransport()
    gemma_transport.queue(ok_response(json.dumps({"a": "hello"})))
    gemma_result = make_client(
        gemma_transport, request_model="gemma-27b", provider_family="gemma"
    ).generate_structured(make_rendered(), make_schema(), make_profile())

    # The semantic request identity is backend-independent.
    assert qwen_result.provenance.request_hash == gemma_result.provenance.request_hash
    assert (
        qwen_result.provenance.semantic_profile_hash
        == gemma_result.provenance.semantic_profile_hash
    )
    # Only the recorded backend identity differs.
    assert qwen_result.provenance.provider_family == "qwen"
    assert qwen_result.provenance.model == "qwen"
    assert gemma_result.provenance.provider_family == "gemma"
    assert gemma_result.provenance.model == "gemma-27b"


def test_fingerprint_profile_change_alters_it():
    base = request_for().fingerprint
    alt = request_for(profile=make_profile(temperature=0.7)).fingerprint
    assert base.semantic_profile_hash != alt.semantic_profile_hash
    assert base.request_hash != alt.request_hash


def test_fingerprint_endpoint_and_credential_do_not_alter_it():
    """Runtime transport details never enter the semantic request fingerprint."""

    base = request_for()
    # A request built from the identical semantic inputs has the identical
    # fingerprint, regardless of which runtime endpoint/credential is used.
    alt = request_for()
    assert base.fingerprint == alt.fingerprint
    # The fingerprint object carries no endpoint/timeout/credential fields.
    fp_dict = base.fingerprint.to_dict()
    assert "base_url" not in fp_dict
    assert "credential" not in json.dumps(fp_dict).lower()


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def test_provenance_population():
    transport = FakeTransport()
    transport.queue(
        ok_response(
            json.dumps({"a": "hello"}),
            response_id="chatcmpl-abc",
            finish_reason="stop",
            usage={"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
        )
    )
    client = make_client(transport)
    result = client.generate_structured(make_rendered(), make_schema(), make_profile())
    prov = result.provenance
    assert prov.provider_family == "qwen"
    assert prov.model == "qwen"
    assert prov.semantic_profile_id == "story-extraction-llm-v1"
    assert prov.prompt_id == "synthetic-echo"
    assert prov.prompt_version == 1
    assert prov.output_schema_id == "candidate-extraction"
    assert prov.output_schema_version == 1
    assert prov.provider_response_id == "chatcmpl-abc"
    assert prov.finish_reason == "stop"
    assert prov.usage == {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}
    assert prov.request_hash == request_for().request_hash


def test_provenance_optional_fields_null_when_absent():
    transport = FakeTransport()
    # Envelope without id, finish_reason, or usage.
    transport.queue(
        ok_response(
            json.dumps({"a": "hello"}),
            raw=json.dumps(
                {"choices": [{"index": 0, "message": {"role": "assistant", "content": '{"a": "hello"}'}}]}
            ).encode(),
        )
    )
    client = make_client(transport)
    result = client.generate_structured(make_rendered(), make_schema(), make_profile())
    prov = result.provenance
    assert prov.provider_response_id is None
    assert prov.finish_reason is None
    assert prov.usage is None


def test_provenance_never_contains_runtime_details(monkeypatch):
    # A real credential value is set so auth-header resolution succeeds; the
    # provenance must still never contain the endpoint host/port or the secret.
    monkeypatch.setenv("SECRET", "super-secret-token")
    transport = FakeTransport()
    transport.queue(ok_response(json.dumps({"a": "hello"})))
    client = make_client(
        transport, base_url="http://10.1.2.3:9999/v1", credential_environment_name="SECRET"
    )
    result = client.generate_structured(make_rendered(), make_schema(), make_profile())
    prov_text = json.dumps(result.provenance.to_dict())
    assert "10.1.2.3" not in prov_text
    assert "super-secret-token" not in prov_text
    assert "9999" not in prov_text


# ---------------------------------------------------------------------------
# Secret safety
# ---------------------------------------------------------------------------


def test_error_messages_do_not_leak_credential(monkeypatch):
    monkeypatch.setenv("LLM_KEY", "super-secret-value")
    transport = FakeTransport()
    transport.queue(
        TransportResponse(401, b'{"error": "unauthorized"}'),
        TransportResponse(401, b'{"error": "unauthorized"}'),
    )
    client = make_client(transport, credential_environment_name="LLM_KEY")
    with pytest.raises(LLMHTTPError) as exc:
        client.generate_structured(make_rendered(), make_schema(), make_profile())
    message = str(exc.value)
    assert "super-secret-value" not in message
    assert "Bearer" not in message
    # The credential is still sent as the auth header on the wire.
    assert transport.calls[0].headers["Authorization"] == "Bearer super-secret-value"


def test_retry_exhausted_message_is_secret_safe(monkeypatch):
    monkeypatch.setenv("LLM_KEY", "super-secret-value")
    transport = FakeTransport()
    transport.queue(
        LLMTimeoutError("timed out"),
        LLMTimeoutError("timed out"),
        LLMTimeoutError("timed out"),
    )
    client = make_client(transport, credential_environment_name="LLM_KEY")
    with pytest.raises(LLMRetryExhaustedError) as exc:
        client.generate_structured(make_rendered(), make_schema(), make_profile())
    assert "super-secret-value" not in str(exc.value)


# ---------------------------------------------------------------------------
# Real urllib transport (deterministic local HTTP server)
# ---------------------------------------------------------------------------


class _RecordHandler(BaseHTTPRequestHandler):
    received = None
    response_status = 200
    response_body = b"{}"

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        _RecordHandler.received = {
            "path": self.path,
            "headers": {k: v for k, v in self.headers.items()},
            "body": body,
        }
        self.send_response(_RecordHandler.response_status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(_RecordHandler.response_body)

    def log_message(self, *args):  # noqa: A003
        pass


def _start_server(status: int, body: bytes):
    server = ThreadingHTTPServer(("127.0.0.1", 0), _RecordHandler)
    _RecordHandler.received = None
    _RecordHandler.response_status = status
    _RecordHandler.response_body = body
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, server.server_address[1]


def test_real_urllib_request_mapping_and_success():
    body = make_envelope_bytes(json.dumps({"a": "hello"}))
    server, port = _start_server(200, body)
    try:
        client = make_client(
            UrllibTransport(), base_url=f"http://127.0.0.1:{port}/v1"
        )
        result = client.generate_structured(make_rendered(), make_schema(), make_profile())
        assert result.parsed_json == {"a": "hello"}
        received = _RecordHandler.received
        assert received["path"] == "/v1/chat/completions"
        assert received["headers"]["Content-Type"] == "application/json"
        sent = json.loads(received["body"])
        assert sent["model"] == "qwen"
        assert sent["response_format"]["type"] == "json_schema"
    finally:
        server.shutdown()


def test_real_urllib_http_500_is_retryable():
    server, port = _start_server(500, b'{"error": "boom"}')
    try:
        client = make_client(UrllibTransport(), base_url=f"http://127.0.0.1:{port}/v1")
        with pytest.raises(LLMRetryExhaustedError) as exc:
            client.generate_structured(make_rendered(), make_schema(), make_profile())
        assert exc.value.attempts == 3
    finally:
        server.shutdown()


def test_real_urllib_http_400_is_not_retryable():
    server, port = _start_server(400, b'{"error": "nope"}')
    try:
        client = make_client(UrllibTransport(), base_url=f"http://127.0.0.1:{port}/v1")
        with pytest.raises(LLMHTTPError) as exc:
            client.generate_structured(make_rendered(), make_schema(), make_profile())
        assert exc.value.status == 400
        assert exc.value.retryable is False
    finally:
        server.shutdown()


def _closed_port() -> int:
    sock = _socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def test_real_urllib_connection_failure_classified():
    port = _closed_port()
    transport = UrllibTransport()
    with pytest.raises(LLMTransportError):
        transport.post(
            url=f"http://127.0.0.1:{port}/chat/completions",
            headers={"Content-Type": "application/json"},
            body=b"{}",
            timeout_seconds=5,
        )


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1 /v1/chat/completions",  # space in the hostname
        "http://127.0.0.1\x00/v1/chat/completions",  # NUL in the hostname
        "http://127.0.0.1\n/v1/chat/completions",  # newline in the hostname
    ],
)
def test_urllib_transport_invalid_url_translated_to_config_error(url):
    # Defense in depth: even if a malformed URL reaches the transport seam
    # (bypassing config validation), the stdlib http.client.InvalidURL must
    # never escape as a raw exception. It is translated to a non-retryable,
    # secret-safe LLMConfigError. This is deterministic: InvalidURL is raised
    # during request-line construction, before any I/O.
    transport = UrllibTransport()
    with pytest.raises(LLMConfigError) as exc:
        transport.post(
            url=url,
            headers={"Content-Type": "application/json"},
            body=b"{}",
            timeout_seconds=5,
        )
    error = exc.value
    assert error.retryable is False
    # Secret-safe: the raw malformed URL content is never echoed, and the
    # stdlib cause (whose message embeds the URL) is dropped.
    assert url not in str(error)
    assert error.__cause__ is None
    assert error.__suppress_context__ is True


class _SlowHandler(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802
        time.sleep(1.0)
        try:
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"a": "hello"}')
        except Exception:  # noqa: BLE001 - client already gave up
            pass

    def log_message(self, *args):  # noqa: A003
        pass


def test_real_urllib_timeout_classified():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _SlowHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        transport = UrllibTransport()
        started = time.monotonic()
        with pytest.raises(LLMTimeoutError):
            transport.post(
                url=f"http://127.0.0.1:{server.server_address[1]}/chat/completions",
                headers={"Content-Type": "application/json"},
                body=b"{}",
                timeout_seconds=0.2,
            )
        elapsed = time.monotonic() - started
        assert elapsed < 1.0
    finally:
        server.shutdown()
