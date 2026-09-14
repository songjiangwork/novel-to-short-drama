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


def _write_runtime_config(path, **overrides) -> None:
    values = {
        "schema_version": 1,
        "transport_id": "llm-local",
        "base_url": "http://127.0.0.1:8080/v1",
        "credential_environment_name": None,
        "timeout_seconds": 30,
    }
    values.update(overrides)
    path.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")


def _write_profile(path, **overrides) -> None:
    values = {
        "schema_version": 1,
        "profile_id": "story-llm-qwen-v1",
        "provider_family": "qwen",
        "model": "qwen",
        "temperature": 0.0,
        "max_output_tokens": 4096,
        "structured_output_mode": "json_schema",
        "reasoning": {"enabled": False, "effort": None},
    }
    values.update(overrides)
    path.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# Runtime transport config
# ---------------------------------------------------------------------------


def test_runtime_config_loads_valid(tmp_path):
    path = tmp_path / "llm.yaml"
    _write_runtime_config(path)
    config = load_runtime_config(path)
    assert config.transport_id == "llm-local"
    assert config.base_url == "http://127.0.0.1:8080/v1"
    assert config.credential_environment_name is None
    assert config.timeout_seconds == 30.0


def test_runtime_config_missing_file(tmp_path):
    with pytest.raises(LLMConfigError):
        load_runtime_config(tmp_path / "nope.yaml")


def test_runtime_config_rejects_wrong_schema_version(tmp_path):
    path = tmp_path / "llm.yaml"
    _write_runtime_config(path, schema_version=2)
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
    # The exact-key contract rejects any unknown field, including semantic
    # generation decisions that belong to the semantic profile.
    for field in ("temperature", "model", "prompt_version", "max_output_tokens"):
        path = tmp_path / "llm.yaml"
        values = {
            "schema_version": 1,
            "transport_id": "llm-local",
            "base_url": "http://127.0.0.1:8080",
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


# ---------------------------------------------------------------------------
# Credential isolation
# ---------------------------------------------------------------------------


def test_credential_resolution_none_when_null():
    config = RuntimeConfig(
        schema_version=1,
        transport_id="t",
        base_url="http://127.0.0.1:8080/v1",
        credential_environment_name=None,
        timeout_seconds=30,
    )
    assert resolve_auth_header(config) is None


def test_credential_resolution_from_env(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "hunter2")
    config = RuntimeConfig(
        schema_version=1,
        transport_id="t",
        base_url="http://127.0.0.1:8080/v1",
        credential_environment_name="LLM_API_KEY",
        timeout_seconds=30,
    )
    assert resolve_auth_header(config) == "Bearer hunter2"


def test_credential_missing_env_fails_closed(monkeypatch):
    # A configured credential variable that is missing must fail closed before
    # any request (never silently send unauthenticated traffic).
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    config = RuntimeConfig(
        schema_version=1,
        transport_id="t",
        base_url="http://127.0.0.1:8080/v1",
        credential_environment_name="LLM_API_KEY",
        timeout_seconds=30,
    )
    with pytest.raises(LLMConfigError, match="LLM_API_KEY"):
        resolve_auth_header(config)


def test_credential_empty_env_fails_closed(monkeypatch):
    # An EMPTY credential variable is also treated as missing -> fail closed.
    monkeypatch.setenv("LLM_API_KEY", "")
    config = RuntimeConfig(
        schema_version=1,
        transport_id="t",
        base_url="http://127.0.0.1:8080/v1",
        credential_environment_name="LLM_API_KEY",
        timeout_seconds=30,
    )
    with pytest.raises(LLMConfigError, match="LLM_API_KEY"):
        resolve_auth_header(config)


def test_credential_value_not_in_config_dict():
    config = RuntimeConfig(
        schema_version=1,
        transport_id="t",
        base_url="http://127.0.0.1:8080/v1",
        credential_environment_name="SOME_SECRET",
        timeout_seconds=30,
    )
    as_text = str(config.to_dict())
    assert "SOME_SECRET" in as_text  # the NAME is present
    # but no resolved value can ever appear (there is no value to leak)
    assert "Bearer" not in as_text


# ---------------------------------------------------------------------------
# Semantic profile
# ---------------------------------------------------------------------------


def test_semantic_profile_loads_valid(tmp_path):
    path = tmp_path / "profile.yaml"
    _write_profile(path)
    profile = load_semantic_profile(path)
    assert profile.profile_id == "story-llm-qwen-v1"
    assert profile.model == "qwen"
    assert profile.structured_output_mode == "json_schema"
    assert isinstance(profile.reasoning, ReasoningSettings)


def test_semantic_profile_hash_stable():
    kwargs = dict(
        schema_version=1,
        profile_id="p",
        provider_family="qwen",
        model="qwen",
        temperature=0.0,
        max_output_tokens=512,
        structured_output_mode="json_schema",
        reasoning=ReasoningSettings(False),
    )
    a = SemanticLLMProfile(**kwargs).semantic_profile_hash
    b = SemanticLLMProfile(**kwargs).semantic_profile_hash
    assert a == b
    assert len(a) == 64


def test_semantic_profile_hash_changes_on_semantic_field(tmp_path):
    base = tmp_path / "base.yaml"
    _write_profile(base)
    baseline = load_semantic_profile(base)
    for field, value in (
        ("temperature", 0.7),
        ("max_output_tokens", 1024),
        ("model", "other-model"),
        ("provider_family", "gpt"),
        ("structured_output_mode", "json_object"),
    ):
        alt = tmp_path / f"alt-{field}.yaml"
        _write_profile(alt, **{field: value})
        assert load_semantic_profile(alt).semantic_profile_hash != (
            baseline.semantic_profile_hash
        ), field


def test_semantic_profile_hash_changes_on_reasoning():
    # Reasoning is result-affecting semantic identity: flipping it (or its
    # effort) must change the semantic profile hash.
    base = SemanticLLMProfile(
        schema_version=1,
        profile_id="p",
        provider_family="qwen",
        model="m",
        temperature=0.0,
        max_output_tokens=512,
        structured_output_mode="json_schema",
        reasoning=ReasoningSettings(False),
    )
    enabled = SemanticLLMProfile(
        schema_version=1,
        profile_id="p",
        provider_family="qwen",
        model="m",
        temperature=0.0,
        max_output_tokens=512,
        structured_output_mode="json_schema",
        reasoning=ReasoningSettings(True, "low"),
    )
    enabled_high = SemanticLLMProfile(
        schema_version=1,
        profile_id="p",
        provider_family="qwen",
        model="m",
        temperature=0.0,
        max_output_tokens=512,
        structured_output_mode="json_schema",
        reasoning=ReasoningSettings(True, "high"),
    )
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
        SemanticLLMProfile(
            schema_version=1,
            profile_id="p",
            provider_family="qwen",
            model="m",
            temperature=0.0,
            max_output_tokens=512,
            structured_output_mode="json_schema",
            reasoning={"enabled": True, "effort": "low"},
        )


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


def test_semantic_profile_missing_field(tmp_path):
    path = tmp_path / "p.yaml"
    values = {
        "schema_version": 1,
        "profile_id": "p",
        "provider_family": "qwen",
        "model": "qwen",
        "temperature": 0.0,
        "max_output_tokens": 128,
        "structured_output_mode": "none",
    }
    path.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")
    with pytest.raises(LLMConfigError):
        load_semantic_profile(path)


def test_endpoint_change_does_not_affect_semantic_hash():
    """The semantic profile hash is independent of runtime transport details.

    Endpoint/timeout/credential live in RuntimeConfig, not the profile, so they
    cannot invalidate downstream semantic identity.
    """

    profile_a = SemanticLLMProfile(
        schema_version=1,
        profile_id="p",
        provider_family="qwen",
        model="qwen",
        temperature=0.0,
        max_output_tokens=128,
        structured_output_mode="json_schema",
        reasoning=ReasoningSettings(False),
    )
    # A different runtime endpoint/timeout/credential does not appear in the profile at all.
    _ = RuntimeConfig(
        schema_version=1,
        transport_id="other",
        base_url="http://10.0.0.9:9999/v1",
        credential_environment_name="SOME_SECRET",
        timeout_seconds=1,
    )
    profile_b = SemanticLLMProfile(
        schema_version=1,
        profile_id="p",
        provider_family="qwen",
        model="qwen",
        temperature=0.0,
        max_output_tokens=128,
        structured_output_mode="json_schema",
        reasoning=ReasoningSettings(False),
    )
    assert profile_a.semantic_profile_hash == profile_b.semantic_profile_hash
