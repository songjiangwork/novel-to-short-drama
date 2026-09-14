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
    data = load_yaml(PROFILES_DIR / "story_llm_qwen_v1.yaml")
    errors = _validate(data, _schema("llm-semantic-profile.schema.json"))
    assert errors == []


def test_runtime_config_rejects_semantic_field_via_schema():
    data = load_yaml(PROFILES_DIR / "llm_local.example.yaml")
    data["temperature"] = 0.0
    errors = _validate(data, _schema("llm-runtime-config.schema.json"))
    assert errors


def test_semantic_profile_rejects_unknown_field_via_schema():
    data = load_yaml(PROFILES_DIR / "story_llm_qwen_v1.yaml")
    data["base_url"] = "http://127.0.0.1:8080"
    errors = _validate(data, _schema("llm-semantic-profile.schema.json"))
    assert errors


def test_prompt_spec_schema_accepts_valid_metadata(tmp_path):
    metadata = {
        "schema_version": 1,
        "prompt_id": "a3.chunk-extraction",
        "version": 1,
        "required_variables": ["chunk_text", "left_context"],
    }
    assert _validate(metadata, _schema("prompt-spec.schema.json")) == []


def test_prompt_spec_schema_rejects_bad_metadata(tmp_path):
    bad = {
        "schema_version": 1,
        "prompt_id": "Bad_ID",
        "version": 0,
        "required_variables": ["ok", "ok"],
    }
    errors = _validate(bad, _schema("prompt-spec.schema.json"))
    assert errors
