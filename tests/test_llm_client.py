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
    render_prompt,
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
    values = dict(
        schema_version=1,
        profile_id="story-llm-qwen-v1",
        provider_family="qwen",
        model="qwen",
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
        base_url="http://127.0.0.1:8080",
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
    assert call.url == "http://127.0.0.1:8080/chat/completions"
    assert call.headers["Content-Type"] == "application/json"
    assert "Authorization" not in call.headers
    assert call.body["model"] == "qwen"
    assert call.body["temperature"] == 0.0
    assert call.body["max_tokens"] == 256
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
# Provider capability
# ---------------------------------------------------------------------------


def test_capability_supports_json_schema():
    assert "json_schema" in OpenAICompatibleLLMClient.supported_structured_output_modes


def test_capability_unsupported_mode_fails_before_request():
    class _NoSchemaClient(OpenAICompatibleLLMClient):
        supported_structured_output_modes = frozenset({"none"})

    transport = FakeTransport()
    client = _NoSchemaClient(
        RuntimeConfig(1, "t", "http://127.0.0.1:8080", None, 30),
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


def test_max_attempts_is_bounded():
    sleeper = RecordingSleeper()
    transport = FakeTransport()
    for _ in range(4):
        transport.queue(TransportResponse(500, b"e"))
    client = OpenAICompatibleLLMClient(
        RuntimeConfig(1, "t", "http://127.0.0.1:8080", None, 30),
        transport=transport,
        sleeper=sleeper,
        max_attempts=4,
    )
    with pytest.raises(LLMRetryExhaustedError) as exc:
        client.generate_structured(make_rendered(), make_schema(), make_profile())
    assert exc.value.attempts == 4
    assert len(transport.calls) == 4


def test_invalid_max_attempts_rejected():
    with pytest.raises(LLMConfigError):
        OpenAICompatibleLLMClient(
            RuntimeConfig(1, "t", "http://127.0.0.1:8080", None, 30),
            transport=FakeTransport(),
            sleeper=lambda d: None,
            max_attempts=0,
        )


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


def test_fingerprint_model_change_alters_it():
    base = request_for().fingerprint
    alt = request_for(profile=make_profile(model="other-model")).fingerprint
    assert base.model != alt.model
    assert base.request_hash != alt.request_hash


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
    assert prov.semantic_profile_id == "story-llm-qwen-v1"
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


def test_provenance_never_contains_runtime_details():
    transport = FakeTransport()
    transport.queue(ok_response(json.dumps({"a": "hello"})))
    client = make_client(
        transport, base_url="http://10.1.2.3:9999", credential_environment_name="SECRET"
    )
    result = client.generate_structured(make_rendered(), make_schema(), make_profile())
    prov_text = json.dumps(result.provenance.to_dict())
    assert "10.1.2.3" not in prov_text
    assert "SECRET" not in prov_text
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
            UrllibTransport(), base_url=f"http://127.0.0.1:{port}"
        )
        result = client.generate_structured(make_rendered(), make_schema(), make_profile())
        assert result.parsed_json == {"a": "hello"}
        received = _RecordHandler.received
        assert received["path"] == "/chat/completions"
        assert received["headers"]["Content-Type"] == "application/json"
        sent = json.loads(received["body"])
        assert sent["model"] == "qwen"
        assert sent["response_format"]["type"] == "json_schema"
    finally:
        server.shutdown()


def test_real_urllib_http_500_is_retryable():
    server, port = _start_server(500, b'{"error": "boom"}')
    try:
        client = make_client(UrllibTransport(), base_url=f"http://127.0.0.1:{port}")
        with pytest.raises(LLMRetryExhaustedError) as exc:
            client.generate_structured(make_rendered(), make_schema(), make_profile())
        assert exc.value.attempts == 3
    finally:
        server.shutdown()


def test_real_urllib_http_400_is_not_retryable():
    server, port = _start_server(400, b'{"error": "nope"}')
    try:
        client = make_client(UrllibTransport(), base_url=f"http://127.0.0.1:{port}")
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
