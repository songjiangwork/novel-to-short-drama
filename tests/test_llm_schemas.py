from __future__ import annotations

import pytest
from jsonschema import Draft202012Validator

from short_drama.io import load_json, load_yaml
from short_drama.paths import PROFILES_DIR, REPO_ROOT, SCHEMAS_DIR


def _schema(name: str) -> dict:
    return load_json(SCHEMAS_DIR / name)


def _validate(data: dict, schema: dict) -> list[str]:
    validator = Draft202012Validator(schema)
    return [
        f"{'/'.join(str(part) for part in error.absolute_path) or '<root>'}: "
        f"{error.message}"
        for error in sorted(
            validator.iter_errors(data),
            key=lambda e: list(e.absolute_path),
        )
    ]


def test_schemas_are_valid_json_schemas():
    for name in (
        "llm-semantic-profile.schema.json",
        "prompt-spec.schema.json",
        "llm-runtime-config.schema.json",
    ):
        schema = _schema(name)
        Draft202012Validator.check_schema(schema)


def test_example_runtime_config_matches_schema():
    data = load_yaml(PROFILES_DIR / "llm_local.example.yaml")
    errors = _validate(data, _schema("llm-runtime-config.schema.json"))
    assert errors == []


def test_example_semantic_profile_matches_schema():
    data = load_yaml(PROFILES_DIR / "story_extraction_llm_v1.yaml")
    errors = _validate(data, _schema("llm-semantic-profile.schema.json"))
    assert errors == []


def test_runtime_config_rejects_semantic_field_via_schema():
    data = load_yaml(PROFILES_DIR / "llm_local.example.yaml")
    data["temperature"] = 0.0
    errors = _validate(data, _schema("llm-runtime-config.schema.json"))
    assert errors


def _runtime_config_with_base_url(base_url: str) -> dict:
    return {
        "schema_version": 1,
        "transport_id": "llm-local",
        "base_url": base_url,
        "request_model": "qwen",
        "provider_family": "qwen",
        "credential_environment_name": None,
        "timeout_seconds": 30,
    }


@pytest.mark.parametrize(
    "bad_base_url",
    [
        "http://127.0.0.1:8080/foo",  # arbitrary non-canonical path
        "http://127.0.0.1:8080/v1/",  # trailing slash
        "http://127.0.0.1:8080",  # missing /v1 path
        "https://user:pass@127.0.0.1:8080/v1",  # embedded credentials
    ],
)
def test_runtime_config_schema_rejects_non_v1_base_url(bad_base_url):
    # The schema no longer advertises arbitrary paths: a base_url that is not
    # exactly a /v1 prefix (and carries no embedded credentials) is rejected.
    errors = _validate(
        _runtime_config_with_base_url(bad_base_url),
        _schema("llm-runtime-config.schema.json"),
    )
    assert errors


def test_runtime_config_schema_accepts_v1_base_url():
    assert (
        _validate(
            _runtime_config_with_base_url("http://127.0.0.1:8080/v1"),
            _schema("llm-runtime-config.schema.json"),
        )
        == []
    )


def test_semantic_profile_rejects_unknown_field_via_schema():
    data = load_yaml(PROFILES_DIR / "story_extraction_llm_v1.yaml")
    data["base_url"] = "http://127.0.0.1:8080"
    errors = _validate(data, _schema("llm-semantic-profile.schema.json"))
    assert errors


def test_prompt_spec_schema_accepts_valid_metadata(tmp_path):
    metadata = {
        "schema_version": 1,
        "prompt_id": "a3.chunk-extraction",
        "version": 1,
        "required_variables": ["chunk_text", "left_context"],
        "content_hash": "0" * 64,
    }
    assert _validate(metadata, _schema("prompt-spec.schema.json")) == []


def test_prompt_spec_schema_rejects_missing_content_hash(tmp_path):
    # The pinned content hash is now a REQUIRED field of the prompt metadata.
    metadata = {
        "schema_version": 1,
        "prompt_id": "a3.chunk-extraction",
        "version": 1,
        "required_variables": ["chunk_text"],
    }
    assert _validate(metadata, _schema("prompt-spec.schema.json"))


def test_prompt_spec_schema_rejects_malformed_content_hash(tmp_path):
    metadata = {
        "schema_version": 1,
        "prompt_id": "a3.chunk-extraction",
        "version": 1,
        "required_variables": ["chunk_text"],
        "content_hash": "zzz",  # not a 64-char lowercase hex digest
    }
    assert _validate(metadata, _schema("prompt-spec.schema.json"))


def test_prompt_spec_schema_rejects_bad_metadata(tmp_path):
    bad = {
        "schema_version": 1,
        "prompt_id": "Bad_ID",
        "version": 0,
        "required_variables": ["ok", "ok"],
        "content_hash": "0" * 64,
    }
    errors = _validate(bad, _schema("prompt-spec.schema.json"))
    assert errors


def _profile_with_reasoning(reasoning: dict) -> dict:
    return {
        "schema_version": 1,
        "profile_id": "story-extraction-llm-v1",
        "temperature": 0.0,
        "max_output_tokens": 4096,
        "structured_output_mode": "json_schema",
        "reasoning": reasoning,
    }


def test_semantic_profile_schema_reasoning_disabled_requires_null_effort():
    schema = _schema("llm-semantic-profile.schema.json")
    # enabled=false + effort=null is valid.
    assert _validate(_profile_with_reasoning({"enabled": False, "effort": None}), schema) == []
    # enabled=false + a real effort is INVALID (incoherent).
    assert _validate(_profile_with_reasoning({"enabled": False, "effort": "low"}), schema)


def test_semantic_profile_schema_reasoning_enabled_requires_supported_effort():
    schema = _schema("llm-semantic-profile.schema.json")
    # enabled=true + each supported effort is valid.
    for effort in ("low", "medium", "high"):
        assert _validate(_profile_with_reasoning({"enabled": True, "effort": effort}), schema) == []
    # enabled=true + effort=null is INVALID (no concrete value).
    assert _validate(_profile_with_reasoning({"enabled": True, "effort": None}), schema)
    # enabled=true + an unsupported effort is INVALID.
    assert _validate(_profile_with_reasoning({"enabled": True, "effort": "bogus"}), schema)
