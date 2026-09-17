from __future__ import annotations

import pytest
import yaml

from short_drama.llm import (
    LLMConfigError,
    ReasoningSettings,
    RuntimeConfig,
    SemanticLLMProfile,
    load_runtime_config,
    load_semantic_profile,
    resolve_auth_header,
)

# Backend runtime identity carried by the runtime config (NOT the semantic
# profile). These values are arbitrary for config-loading tests.
REQ_MODEL = "qwen3-27b"
PROV_FAMILY = "qwen"


def _write_runtime_config(path, **overrides) -> None:
    values = {
        "schema_version": 2,
        "transport_id": "llm-local",
        "base_url": "http://127.0.0.1:8080/v1",
        "request_model": REQ_MODEL,
        "provider_family": PROV_FAMILY,
        "credential_environment_name": None,
        "timeout_seconds": 30,
    }
    values.update(overrides)
    path.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")


def _write_profile(path, **overrides) -> None:
    # A semantic profile carries NO backend identity (no provider_family and no
    # model); it is result-affecting generation semantics only.
    values = {
        "schema_version": 2,
        "profile_id": "story-extraction-llm-v1",
        "temperature": 0.0,
        "max_output_tokens": 4096,
        "structured_output_mode": "json_schema",
        "reasoning": {"enabled": False, "effort": None},
    }
    values.update(overrides)
    path.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")


def _runtime(**overrides) -> RuntimeConfig:
    values = dict(
        schema_version=2,
        transport_id="t",
        base_url="http://127.0.0.1:8080/v1",
        request_model=REQ_MODEL,
        provider_family=PROV_FAMILY,
        credential_environment_name=None,
        timeout_seconds=30,
    )
    values.update(overrides)
    return RuntimeConfig(**values)


def _profile(**overrides) -> SemanticLLMProfile:
    values = dict(
        schema_version=2,
        profile_id="p",
        temperature=0.0,
        max_output_tokens=512,
        structured_output_mode="json_schema",
        reasoning=ReasoningSettings(False),
    )
    values.update(overrides)
    return SemanticLLMProfile(**values)


# ---------------------------------------------------------------------------
# Runtime transport config
# ---------------------------------------------------------------------------


def test_runtime_config_loads_valid(tmp_path):
    path = tmp_path / "llm.yaml"
    _write_runtime_config(path)
    config = load_runtime_config(path)
    assert config.transport_id == "llm-local"
    assert config.base_url == "http://127.0.0.1:8080/v1"
    assert config.request_model == REQ_MODEL
    assert config.provider_family == PROV_FAMILY
    assert config.credential_environment_name is None
    assert config.timeout_seconds == 30.0


def test_runtime_config_missing_file(tmp_path):
    with pytest.raises(LLMConfigError):
        load_runtime_config(tmp_path / "nope.yaml")


def test_runtime_config_rejects_old_v1_form(tmp_path):
    # The A-I3 backend/runtime split is a v2 schema change: a v1 config (no
    # request_model / provider_family, old schema_version) must fail closed.
    path = tmp_path / "llm.yaml"
    _write_runtime_config(path, schema_version=1)
    with pytest.raises(LLMConfigError):
        load_runtime_config(path)


def test_runtime_config_rejects_invalid_base_url(tmp_path):
    for bad in ("ftp://x", "127.0.0.1:8080", "http://", "http://user:pass@127.0.0.1"):
        path = tmp_path / "llm.yaml"
        _write_runtime_config(path, base_url=bad)
        with pytest.raises(LLMConfigError):
            load_runtime_config(path)


def test_runtime_config_rejects_credentials_in_url(tmp_path):
    path = tmp_path / "llm.yaml"
    _write_runtime_config(path, base_url="http://secret:pw@127.0.0.1:8080")
    with pytest.raises(LLMConfigError):
        load_runtime_config(path)


def test_runtime_config_rejects_semantic_fields(tmp_path):
    # The exact-key contract rejects any unknown field, including result-
    # affecting generation semantics that belong to the semantic profile.
    for field in ("temperature", "prompt_version", "max_output_tokens", "reasoning"):
        path = tmp_path / "llm.yaml"
        values = {
            "schema_version": 2,
            "transport_id": "llm-local",
            "base_url": "http://127.0.0.1:8080/v1",
            "request_model": REQ_MODEL,
            "provider_family": PROV_FAMILY,
            "credential_environment_name": None,
            "timeout_seconds": 30,
            field: "x",
        }
        path.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")
        with pytest.raises(LLMConfigError, match="exactly"):
            load_runtime_config(path)


def test_runtime_config_rejects_bad_timeout(tmp_path):
    for bad in (0, -1, 3601, "30"):
        path = tmp_path / "llm.yaml"
        _write_runtime_config(path, timeout_seconds=bad)
        with pytest.raises(LLMConfigError):
            load_runtime_config(path)


def test_runtime_config_rejects_bad_env_name(tmp_path):
    path = tmp_path / "llm.yaml"
    _write_runtime_config(path, credential_environment_name="not a name!")
    with pytest.raises(LLMConfigError):
        load_runtime_config(path)


def test_runtime_config_rejects_empty_request_model(tmp_path):
    path = tmp_path / "llm.yaml"
    _write_runtime_config(path, request_model="")
    with pytest.raises(LLMConfigError, match="request_model"):
        load_runtime_config(path)


def test_runtime_config_rejects_empty_provider_family(tmp_path):
    path = tmp_path / "llm.yaml"
    _write_runtime_config(path, provider_family="")
    with pytest.raises(LLMConfigError, match="provider_family"):
        load_runtime_config(path)


@pytest.mark.parametrize(
    "bad_url",
    [
        "http://127.0.0.1:8080/v1/",  # trailing slash
        "https://user:pass@127.0.0.1:8080/v1",  # embedded credentials
        "127.0.0.1:8080/v1",  # no scheme
        "ftp://127.0.0.1:8080/v1",  # wrong scheme
        "http://127.0.0.1:8080/v1?token=secret",  # query string
        "http://127.0.0.1:8080/v1#frag",  # fragment
        "http://127.0.0.1:99999/v1",  # port out of range
        "not a url",  # garbage
    ],
)
def test_runtime_config_rejects_bad_base_url(tmp_path, bad_url):
    path = tmp_path / "llm.yaml"
    _write_runtime_config(path, base_url=bad_url)
    with pytest.raises(LLMConfigError):
        load_runtime_config(path)


def test_runtime_config_accepts_canonical_local_base_url(tmp_path):
    path = tmp_path / "llm.yaml"
    _write_runtime_config(path, base_url="http://127.0.0.1:8080/v1")
    config = load_runtime_config(path)
    assert config.base_url == "http://127.0.0.1:8080/v1"


def test_runtime_config_accepts_well_formed_ipv6_base_url(tmp_path):
    # A well-formed bracketed IPv6 address with /v1 is a valid base URL.
    path = tmp_path / "llm.yaml"
    _write_runtime_config(path, base_url="http://[::1]:8080/v1")
    config = load_runtime_config(path)
    assert config.base_url == "http://[::1]:8080/v1"


@pytest.mark.parametrize(
    "bad_url",
    [
        "http://[::1/v1",  # unclosed bracket
        "http://[gg:gg:gg]/v1",  # non-hex IPv6 group
        "http://[1:2:3:4:5:6:7:8:9]/v1",  # too many IPv6 groups
        "http://[::1]junk:80/v1",  # trailing junk after a bracketed IPv6
    ],
)
def test_runtime_config_rejects_malformed_ipv6_as_config_error(tmp_path, bad_url):
    # Malformed bracketed IPv6 must surface as a secret-safe, non-retryable
    # LLMConfigError, never a raw parser ValueError.
    path = tmp_path / "llm.yaml"
    _write_runtime_config(path, base_url=bad_url)
    with pytest.raises(LLMConfigError):
        load_runtime_config(path)


@pytest.mark.parametrize(
    "bad_url",
    [
        "http://127.0.0.1:99999/v1",  # port out of range
        "http://127.0.0.1:0/v1",  # port 0 is not usable
        "http://127.0.0.1:abc/v1",  # non-numeric port
        "http://[::1]:99999/v1",  # out-of-range port on an IPv6 host
    ],
)
def test_runtime_config_rejects_invalid_port_as_config_error(tmp_path, bad_url):
    path = tmp_path / "llm.yaml"
    _write_runtime_config(path, base_url=bad_url)
    with pytest.raises(LLMConfigError):
        load_runtime_config(path)


@pytest.mark.parametrize(
    "bad_url",
    [
        "http://127.0.0.1 /v1",  # space in the hostname
        "http://127.0.0.1:8080 /v1",  # space after the port
        "http://127.0.0.1\x00/v1",  # NUL in the hostname
        "http://127.0.0.1\n/v1",  # newline in the hostname
        "http://127.0.0.1\t/v1",  # tab in the hostname
    ],
)
def test_runtime_config_rejects_whitespace_and_control_chars(tmp_path, bad_url):
    # Whitespace and ASCII control characters are rejected by Python's HTTP
    # stack (http.client.InvalidURL). urlsplit does NOT catch them (e.g.
    # "http://127.0.0.1 /v1" parses with hostname "127.0.0.1 "), so they must
    # fail closed at config time as a secret-safe, non-retryable
    # LLMConfigError, never escape to the transport as a raw exception.
    path = tmp_path / "llm.yaml"
    _write_runtime_config(path, base_url=bad_url)
    with pytest.raises(LLMConfigError) as exc:
        load_runtime_config(path)
    assert exc.value.retryable is False
    # Secret-safe: the fixed message never echoes the raw URL content.
    assert str(exc.value) == (
        "base_url must not contain whitespace or control characters"
    )


def test_runtime_config_base_url_error_is_non_retryable():
    # LLMConfigError is a non-retryable, secret-safe configuration failure.
    assert LLMConfigError.retryable is False
    for bad_url in (
        "http://[::1/v1",
        "http://127.0.0.1:99999/v1",
        "http://127.0.0.1 /v1",  # whitespace/control char
    ):
        with pytest.raises(LLMConfigError):
            _runtime(base_url=bad_url)


# ---------------------------------------------------------------------------
# Credential isolation
# ---------------------------------------------------------------------------


def test_credential_resolution_none_when_null():
    config = _runtime(credential_environment_name=None)
    assert resolve_auth_header(config) is None


def test_credential_resolution_from_env(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "hunter2")
    config = _runtime(credential_environment_name="LLM_API_KEY")
    assert resolve_auth_header(config) == "Bearer hunter2"


def test_credential_missing_env_fails_closed(monkeypatch):
    # A configured credential variable that is missing must fail closed before
    # any request (never silently send unauthenticated traffic).
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    config = _runtime(credential_environment_name="LLM_API_KEY")
    with pytest.raises(LLMConfigError, match="LLM_API_KEY"):
        resolve_auth_header(config)


def test_credential_empty_env_fails_closed(monkeypatch):
    # An EMPTY credential variable is also treated as missing -> fail closed.
    monkeypatch.setenv("LLM_API_KEY", "")
    config = _runtime(credential_environment_name="LLM_API_KEY")
    with pytest.raises(LLMConfigError, match="LLM_API_KEY"):
        resolve_auth_header(config)


def test_credential_value_not_in_config_dict():
    config = _runtime(credential_environment_name="SOME_SECRET")
    as_text = str(config.to_dict())
    assert "SOME_SECRET" in as_text  # the NAME is present
    # but no resolved value can ever appear (there is no value to leak)
    assert "Bearer" not in as_text


def test_runtime_config_roundtrip():
    config = _runtime(credential_environment_name="SOME_SECRET")
    assert RuntimeConfig.from_dict(config.to_dict()) == config


# ---------------------------------------------------------------------------
# Semantic profile
# ---------------------------------------------------------------------------


def test_semantic_profile_loads_valid(tmp_path):
    path = tmp_path / "profile.yaml"
    _write_profile(path)
    profile = load_semantic_profile(path)
    assert profile.profile_id == "story-extraction-llm-v1"
    assert not hasattr(profile, "model")
    assert not hasattr(profile, "provider_family")
    assert profile.structured_output_mode == "json_schema"
    assert isinstance(profile.reasoning, ReasoningSettings)


def test_semantic_profile_hash_stable():
    a = _profile().semantic_profile_hash
    b = _profile().semantic_profile_hash
    assert a == b
    assert len(a) == 64


def test_semantic_profile_hash_changes_on_semantic_field(tmp_path):
    base = tmp_path / "base.yaml"
    _write_profile(base)
    baseline = load_semantic_profile(base)
    for field, value in (
        ("temperature", 0.7),
        ("max_output_tokens", 1024),
        ("structured_output_mode", "json_object"),
    ):
        alt = tmp_path / f"alt-{field}.yaml"
        _write_profile(alt, **{field: value})
        assert load_semantic_profile(alt).semantic_profile_hash != (
            baseline.semantic_profile_hash
        ), field


def test_semantic_profile_hash_independent_of_backend():
    """The backend model / provider family is NOT part of the semantic profile.

    Switching the backend (Qwen -> Gemma) is a runtime-config change; it must
    not change the semantic profile hash or the request fingerprint hash.
    """

    # The semantic profile carries no backend identity at all, so two requests
    # built from the same profile are semantically identical regardless of the
    # backend in effect.
    profile = _profile()
    from short_drama.llm import build_structured_request, OutputSchema, RenderedPrompt
    from short_drama.llm.models import compute_rendered_prompt_hash

    rendered = RenderedPrompt(
        prompt_id="p",
        prompt_version=1,
        prompt_content_hash="b" * 64,
        variables_hash="2" * 64,
        system_text="s",
        user_text="u",
        rendered_prompt_hash=compute_rendered_prompt_hash(
            prompt_id="p",
            prompt_version=1,
            prompt_content_hash="b" * 64,
            variables_hash="2" * 64,
            system_text="s",
            user_text="u",
        ),
    )
    schema = OutputSchema.create(
        schema_id="s", schema_version=1, schema={"type": "object"}
    )
    req_a = build_structured_request(
        rendered_prompt=rendered, output_schema=schema, semantic_profile=profile
    )
    # A different backend (Qwen -> Gemma) only changes the runtime config.
    _ = _runtime(request_model="qwen3-27b", provider_family="qwen")
    _ = _runtime(request_model="gemma-27b", provider_family="gemma")
    req_b = build_structured_request(
        rendered_prompt=rendered, output_schema=schema, semantic_profile=profile
    )
    assert req_a.request_hash == req_b.request_hash
    assert req_a.fingerprint == req_b.fingerprint


def test_semantic_profile_hash_changes_on_reasoning():
    # Reasoning is result-affecting semantic identity: flipping it (or its
    # effort) must change the semantic profile hash.
    base = _profile()
    enabled = _profile(reasoning=ReasoningSettings(True, "low"))
    enabled_high = _profile(reasoning=ReasoningSettings(True, "high"))
    assert base.semantic_profile_hash != enabled.semantic_profile_hash
    assert enabled.semantic_profile_hash != enabled_high.semantic_profile_hash


def test_reasoning_settings_rejects_incoherent_combos():
    # enabled=True without an effort, and enabled=False with an effort, are
    # both incoherent and rejected at the ReasoningSettings layer.
    for make_bad in (lambda: ReasoningSettings(True), lambda: ReasoningSettings(False, "low")):
        with pytest.raises(LLMConfigError):
            make_bad()


def test_semantic_profile_requires_reasoning_settings_type():
    # A raw dict is not accepted where a ReasoningSettings instance is required.
    with pytest.raises(LLMConfigError):
        _profile(reasoning={"enabled": True, "effort": "low"})


def test_semantic_profile_rejects_bad_mode(tmp_path):
    path = tmp_path / "p.yaml"
    _write_profile(path, structured_output_mode="bogus")
    with pytest.raises(LLMConfigError):
        load_semantic_profile(path)


def test_semantic_profile_rejects_bad_temperature(tmp_path):
    path = tmp_path / "p.yaml"
    _write_profile(path, temperature=5.0)
    with pytest.raises(LLMConfigError):
        load_semantic_profile(path)


def test_semantic_profile_rejects_backend_fields(tmp_path):
    # A legacy profile that still carries backend identity (provider_family /
    # model) is rejected: those are runtime fields, not semantic fields.
    path = tmp_path / "p.yaml"
    values = {
        "schema_version": 2,
        "profile_id": "p",
        "provider_family": "qwen",
        "model": "qwen",
        "temperature": 0.0,
        "max_output_tokens": 128,
        "structured_output_mode": "json_schema",
        "reasoning": {"enabled": False, "effort": None},
    }
    path.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")
    with pytest.raises(LLMConfigError, match="exactly"):
        load_semantic_profile(path)


def test_semantic_profile_missing_field(tmp_path):
    path = tmp_path / "p.yaml"
    values = {
        "schema_version": 2,
        "profile_id": "p",
        "temperature": 0.0,
        "max_output_tokens": 128,
        "structured_output_mode": "none",
    }
    path.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")
    with pytest.raises(LLMConfigError):
        load_semantic_profile(path)


def test_semantic_profile_rejects_old_v1_form(tmp_path):
    # The A-I3 backend/runtime split is a v2 schema change: a v1 semantic
    # profile (old schema_version, still carrying backend identity) must fail
    # closed -- there is no backward-compat v1 loader.
    path = tmp_path / "p.yaml"
    values = {
        "schema_version": 1,
        "profile_id": "p",
        "provider_family": "qwen",
        "model": "qwen",
        "temperature": 0.0,
        "max_output_tokens": 128,
        "structured_output_mode": "json_schema",
        "reasoning": {"enabled": False, "effort": None},
    }
    path.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")
    with pytest.raises(LLMConfigError):
        load_semantic_profile(path)


def test_endpoint_change_does_not_affect_semantic_hash():
    """The semantic profile hash is independent of runtime transport details.

    Endpoint/timeout/credential live in RuntimeConfig, not the profile, so they
    cannot invalidate downstream semantic identity.
    """

    profile_a = _profile()
    # A different runtime endpoint/timeout/credential does not appear in the
    # profile at all.
    _ = _runtime(
        transport_id="other",
        base_url="http://10.0.0.9:9999/v1",
        credential_environment_name="SOME_SECRET",
        timeout_seconds=1,
    )
    profile_b = _profile()
    assert profile_a.semantic_profile_hash == profile_b.semantic_profile_hash
