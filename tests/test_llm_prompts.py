from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from short_drama.llm import (
    LLMPromptError,
    PromptRegistry,
    PromptSpec,
    compute_prompt_content_hash,
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
    # Pin the authoritative content hash for this version. It is computed over
    # the semantic prompt material (not the metadata file) and must match what
    # the registry recomputes, or the load fails closed.
    content_hash = compute_prompt_content_hash(
        prompt_id=prompt_id,
        version=version,
        system_template=system,
        user_template=user,
        required_variables=required_variables,
    )
    metadata = {
        "schema_version": 1,
        "prompt_id": prompt_id,
        "version": version,
        "required_variables": list(required_variables),
        "content_hash": content_hash,
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


# ---------------------------------------------------------------------------
# Finding 1: prompt identity (prompt_id + version + content_hash) is immutable.
# ---------------------------------------------------------------------------


def test_load_rejects_system_template_mutated_same_version(tmp_path):
    _write_prompt(tmp_path, "p", 1)
    version_dir = tmp_path / "p" / "v1"
    # Mutate the semantic material WITHOUT re-versioning: the pinned content
    # hash no longer matches the recomputed hash -> fail closed.
    (version_dir / "system.txt").write_text(
        "You are a DIFFERENT extractor.", encoding="utf-8"
    )
    with pytest.raises(LLMPromptError, match="content_hash"):
        PromptRegistry(tmp_path).load("p", version=1)


def test_load_rejects_required_variables_mutated_same_version(tmp_path):
    _write_prompt(
        tmp_path, "p", 1, user="Extract from: {{chunk_text}}", required_variables=("chunk_text",)
    )
    # Rewrite the pinned metadata with a DIFFERENT required_variables but the
    # OLD content_hash -> the recomputed hash differs -> fail closed.
    version_dir = tmp_path / "p" / "v1"
    old_yaml = yaml.safe_load((version_dir / "prompt.yaml").read_text(encoding="utf-8"))
    old_yaml["required_variables"] = ["something_else"]
    (version_dir / "prompt.yaml").write_text(
        yaml.safe_dump(old_yaml, sort_keys=False), encoding="utf-8"
    )
    with pytest.raises(LLMPromptError, match="content_hash"):
        PromptRegistry(tmp_path).load("p", version=1)


def test_load_accepts_reversioned_prompt_with_new_hash(tmp_path):
    _write_prompt(tmp_path, "p", 1)
    # A NEW version with different content and its own correct hash is fine.
    _write_prompt(
        tmp_path,
        "p",
        2,
        system="You are a v2 extractor.",
        user="Extract from: {{chunk_text}}",
    )
    spec = PromptRegistry(tmp_path).load("p", version=2)
    assert spec.version == 2
    assert spec.content_hash != PromptRegistry(tmp_path).load("p", version=1).content_hash


def test_load_rejects_wrong_pinned_content_hash(tmp_path):
    _write_prompt(tmp_path, "p", 1)
    version_dir = tmp_path / "p" / "v1"
    bad = {
        "schema_version": 1,
        "prompt_id": "p",
        "version": 1,
        "required_variables": ["chunk_text"],
        "content_hash": "0" * 64,  # wrong pinned hash
    }
    (version_dir / "prompt.yaml").write_text(
        yaml.safe_dump(bad, sort_keys=False), encoding="utf-8"
    )
    with pytest.raises(LLMPromptError, match="content_hash"):
        PromptRegistry(tmp_path).load("p", version=1)


def test_load_rejects_malformed_content_hash(tmp_path):
    _write_prompt(tmp_path, "p", 1)
    version_dir = tmp_path / "p" / "v1"
    bad = {
        "schema_version": 1,
        "prompt_id": "p",
        "version": 1,
        "required_variables": ["chunk_text"],
        "content_hash": "not-a-hash",
    }
    (version_dir / "prompt.yaml").write_text(
        yaml.safe_dump(bad, sort_keys=False), encoding="utf-8"
    )
    with pytest.raises(LLMPromptError, match="SHA-256"):
        PromptRegistry(tmp_path).load("p", version=1)


def test_load_rejects_prompt_id_before_path_construction(tmp_path):
    # An unsafe prompt_id must be rejected BEFORE it is used to build a path.
    with pytest.raises(LLMPromptError):
        PromptRegistry(tmp_path).load("../evil", version=1)
    with pytest.raises(LLMPromptError):
        PromptRegistry(tmp_path).load("UPPER", version=1)


def test_prompt_spec_hash_matches_pinned_hash(tmp_path):
    _write_prompt(tmp_path, "p", 1)
    spec = PromptRegistry(tmp_path).load("p", version=1)
    # The spec's content_hash equals the registry-recomputed hash and is
    # deterministic.
    assert spec.content_hash == compute_prompt_content_hash(
        prompt_id="p",
        version=1,
        system_template="You are a strict extractor.",
        user_template="Extract from: {{chunk_text}}",
        required_variables=["chunk_text"],
    )


# ---------------------------------------------------------------------------
# Tracked a3.chunk-extraction v1 (immutable) + v2 (evidence excerpt contract)
# ---------------------------------------------------------------------------


PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts" / "story"


def test_a3_chunk_extraction_v1_immutable_and_loadable():
    """v1 content is untouched and loads with its pinned hash."""
    reg = PromptRegistry(PROMPTS_DIR)
    spec = reg.load("a3.chunk-extraction", version=1)
    assert spec.version == 1
    assert spec.content_hash == "118469c47ea401ee35ef9164089f918494d884158df73b02f493aa260ad40a09"


def test_a3_chunk_extraction_v2_loads_with_correct_hash():
    """v2 loads and its content_hash matches the pinned value."""
    reg = PromptRegistry(PROMPTS_DIR)
    spec = reg.load("a3.chunk-extraction", version=2)
    assert spec.version == 2
    assert spec.content_hash == "614e6bbb866b497b550cee1feb0dd9d955ec37ad97b209a2b7bd50a19ee0df06"
    assert spec.required_variables == (
        "chunk_id",
        "left_context_json",
        "ownership_json",
        "right_context_json",
    )


def test_a3_chunk_extraction_v2_contains_exact_substring_contract():
    """v2 system template contains the key evidence-fidelity requirements."""
    reg = PromptRegistry(PROMPTS_DIR)
    spec = reg.load("a3.chunk-extraction", version=2)
    sys_text = spec.system_template
    # Exact contiguous substring rule
    assert "exact contiguous substring" in sys_text
    assert "text_original" in sys_text
    # Same paragraph_id requirement
    assert "SAME" in sys_text and "paragraph_id" in sys_text
    # No paraphrase
    assert "paraphrase" in sys_text
    # Null fallback
    assert 'set "excerpt" to null' in sys_text
    # Never substitute
    assert "Never substitute" in sys_text


def test_a3_chunk_extraction_v2_prefers_short_verbatim():
    """v2 instructs the model to prefer short verbatim excerpts."""
    reg = PromptRegistry(PROMPTS_DIR)
    spec = reg.load("a3.chunk-extraction", version=2)
    assert "single contiguous span" in spec.system_template
    assert "Do not copy an entire paragraph" in spec.system_template


def test_a3_chunk_extraction_v2_contains_self_check():
    """v2 includes the self-check instruction."""
    reg = PromptRegistry(PROMPTS_DIR)
    spec = reg.load("a3.chunk-extraction", version=2)
    assert "SELF-CHECK" in spec.system_template
    assert "character-for-character" in spec.system_template
    assert "set the excerpt to null instead" in spec.system_template


def test_a3_chunk_extraction_v2_preserves_v1_rules():
    """v2 retains all core v1 rules (chunk-local, zh-CN, local refs, etc.)."""
    reg = PromptRegistry(PROMPTS_DIR)
    spec = reg.load("a3.chunk-extraction", version=2)
    sys_text = spec.system_template
    assert "CHUNK-LOCAL ONLY" in sys_text
    assert "zh-CN" in sys_text
    assert "LOCAL REFERENCES" in sys_text
    assert "UNRESOLVED IS VALID" in sys_text
    assert "NO EXTRA DECISIONS" in sys_text
    assert "OUTPUT FORMAT" in sys_text
