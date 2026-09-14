from __future__ import annotations

import pytest
import yaml

from short_drama.llm import (
    LLMPromptError,
    PromptRegistry,
    PromptSpec,
    extract_placeholders,
    render_prompt,
)


def _write_prompt(
    base: str,
    prompt_id: str,
    version: int,
    *,
    system: str = "You are a strict extractor.",
    user: str = "Extract from: {{chunk_text}}",
    required_variables=None,
    yaml_overrides=None,
) -> None:
    version_dir = base / prompt_id / f"v{version}"
    version_dir.mkdir(parents=True, exist_ok=True)
    if required_variables is None:
        required_variables = sorted(
            extract_placeholders(system) | extract_placeholders(user)
        )
    metadata = {
        "schema_version": 1,
        "prompt_id": prompt_id,
        "version": version,
        "required_variables": list(required_variables),
    }
    if yaml_overrides:
        metadata.update(yaml_overrides)
    (version_dir / "prompt.yaml").write_text(
        yaml.safe_dump(metadata, sort_keys=False), encoding="utf-8"
    )
    (version_dir / "system.txt").write_text(system, encoding="utf-8")
    (version_dir / "user.txt").write_text(user, encoding="utf-8")


def test_prompt_spec_loads(tmp_path):
    _write_prompt(tmp_path, "synthetic-echo", 1)
    spec = PromptRegistry(tmp_path).load("synthetic-echo", version=1)
    assert spec.prompt_id == "synthetic-echo"
    assert spec.version == 1
    assert spec.required_variables == ("chunk_text",)
    assert len(spec.content_hash) == 64


def test_prompt_content_hash_stable(tmp_path):
    registry = PromptRegistry(tmp_path)
    _write_prompt(tmp_path, "synthetic-echo", 1)
    a = registry.load("synthetic-echo", version=1).content_hash
    b = registry.load("synthetic-echo", version=1).content_hash
    assert a == b


def test_prompt_content_hash_changes_on_template_change(tmp_path):
    _write_prompt(tmp_path, "p", 1, user="Extract: {{x}}")
    base = PromptRegistry(tmp_path).load("p", version=1).content_hash
    _write_prompt(tmp_path, "p", 1, user="Extract differently: {{x}}")
    changed = PromptRegistry(tmp_path).load("p", version=1).content_hash
    assert changed != base


def test_prompt_content_hash_changes_on_required_vars(tmp_path):
    _write_prompt(tmp_path, "p", 1, user="A: {{a}}", required_variables=("a",))
    base = PromptRegistry(tmp_path).load("p", version=1).content_hash
    _write_prompt(tmp_path, "p", 1, user="A: {{a}} B: {{b}}", required_variables=("a", "b"))
    changed = PromptRegistry(tmp_path).load("p", version=1).content_hash
    assert changed != base


def test_prompt_content_hash_covers_system_template(tmp_path):
    _write_prompt(tmp_path, "p", 1, system="Alpha", user="X: {{x}}")
    a = PromptRegistry(tmp_path).load("p", version=1).content_hash
    _write_prompt(tmp_path, "p", 1, system="Beta", user="X: {{x}}")
    b = PromptRegistry(tmp_path).load("p", version=1).content_hash
    assert a != b


def test_prompt_version_not_found(tmp_path):
    _write_prompt(tmp_path, "p", 1)
    with pytest.raises(LLMPromptError):
        PromptRegistry(tmp_path).load("p", version=2)


def test_prompt_missing_file(tmp_path):
    version_dir = tmp_path / "p" / "v1"
    version_dir.mkdir(parents=True)
    (version_dir / "prompt.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "prompt_id": "p",
                "version": 1,
                "required_variables": [],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(LLMPromptError, match="not found"):
        PromptRegistry(tmp_path).load("p", version=1)


def test_prompt_prompt_id_mismatch_fails(tmp_path):
    _write_prompt(tmp_path, "p", 1, yaml_overrides={"prompt_id": "other"})
    with pytest.raises(LLMPromptError, match="prompt_id"):
        PromptRegistry(tmp_path).load("p", version=1)


def test_prompt_duplicate_required_variable_fails(tmp_path):
    _write_prompt(
        tmp_path,
        "p",
        1,
        user="X: {{x}}",
        required_variables=("x", "x"),
    )
    with pytest.raises(LLMPromptError, match="duplicate"):
        PromptRegistry(tmp_path).load("p", version=1)


def test_prompt_required_variables_must_match_placeholders(tmp_path):
    _write_prompt(tmp_path, "p", 1, user="X: {{x}}", required_variables=("y",))
    with pytest.raises(LLMPromptError, match="placeholders"):
        PromptRegistry(tmp_path).load("p", version=1)


def test_prompt_unsupported_placeholder_fails(tmp_path):
    for bad_user in ("X: {{ bad }}", "X: {{}}", "X: {{unclosed", "X: {{1bad}}"):
        base = tmp_path / "bad"
        _write_prompt(base, "p", 1, user=bad_user, required_variables=("bad",))
        with pytest.raises(LLMPromptError):
            PromptRegistry(base).load("p", version=1)


def test_render_success(tmp_path):
    _write_prompt(tmp_path, "p", 1)
    spec = PromptRegistry(tmp_path).load("p", version=1)
    rendered = render_prompt(spec, {"chunk_text": "some text"})
    assert rendered.user_text == "Extract from: some text"
    assert rendered.system_text == "You are a strict extractor."
    assert rendered.prompt_id == "p"
    assert rendered.prompt_version == 1
    assert rendered.prompt_content_hash == spec.content_hash
    assert len(rendered.rendered_prompt_hash) == 64
    assert len(rendered.variables_hash) == 64


def test_render_missing_variable_fails(tmp_path):
    _write_prompt(tmp_path, "p", 1)
    spec = PromptRegistry(tmp_path).load("p", version=1)
    with pytest.raises(LLMPromptError, match="missing"):
        render_prompt(spec, {})


def test_render_unexpected_variable_fails(tmp_path):
    _write_prompt(tmp_path, "p", 1)
    spec = PromptRegistry(tmp_path).load("p", version=1)
    with pytest.raises(LLMPromptError, match="unexpected"):
        render_prompt(spec, {"chunk_text": "x", "extra": "y"})


def test_render_rejects_non_string_variable(tmp_path):
    _write_prompt(tmp_path, "p", 1)
    spec = PromptRegistry(tmp_path).load("p", version=1)
    with pytest.raises(LLMPromptError, match="string"):
        render_prompt(spec, {"chunk_text": 42})


def test_render_deterministic(tmp_path):
    _write_prompt(tmp_path, "p", 1)
    spec = PromptRegistry(tmp_path).load("p", version=1)
    a = render_prompt(spec, {"chunk_text": "abc"})
    b = render_prompt(spec, {"chunk_text": "abc"})
    assert a == b
    assert a.rendered_prompt_hash == b.rendered_prompt_hash


def test_render_hash_changes_on_variable_change(tmp_path):
    _write_prompt(tmp_path, "p", 1)
    spec = PromptRegistry(tmp_path).load("p", version=1)
    a = render_prompt(spec, {"chunk_text": "abc"})
    b = render_prompt(spec, {"chunk_text": "xyz"})
    assert a.rendered_prompt_hash != b.rendered_prompt_hash
    assert a.variables_hash != b.variables_hash


def test_render_preserves_single_braces(tmp_path):
    _write_prompt(
        tmp_path,
        "p",
        1,
        user='Return JSON: {"a": 1} for {{name}}',
        required_variables=("name",),
    )
    spec = PromptRegistry(tmp_path).load("p", version=1)
    rendered = render_prompt(spec, {"name": "bob"})
    assert rendered.user_text == 'Return JSON: {"a": 1} for bob'


def test_render_empty_system_allowed(tmp_path):
    _write_prompt(tmp_path, "p", 1, system="", user="X: {{x}}")
    spec = PromptRegistry(tmp_path).load("p", version=1)
    rendered = render_prompt(spec, {"x": "y"})
    assert rendered.system_text == ""


def test_render_no_variables_prompt(tmp_path):
    _write_prompt(tmp_path, "p", 1, user="Just say hi", required_variables=())
    spec = PromptRegistry(tmp_path).load("p", version=1)
    rendered = render_prompt(spec, {})
    assert rendered.user_text == "Just say hi"


def test_prompt_spec_render_method_matches_documented_usage(tmp_path):
    # A-I4 usage: prompt = registry.load(...); rendered = prompt.render(...)
    _write_prompt(tmp_path, "p", 1)
    spec = PromptRegistry(tmp_path).load("p", version=1)
    via_method = spec.render({"chunk_text": "abc"})
    via_function = render_prompt(spec, {"chunk_text": "abc"})
    assert via_method == via_function
    assert via_method.user_text == "Extract from: abc"
