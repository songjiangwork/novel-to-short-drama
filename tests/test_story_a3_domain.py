"""A3A domain/static-contract tests.

Covers the frozen A3A contract:
  * tracked StoryExtractionProfile load / round-trip / hash / schema;
  * the six candidate models + EvidenceRef round-trips;
  * candidate ID namespace shape, enum, and exact-key / fail-closed parsing;
  * CandidatePayload full / empty six-category round-trips;
  * CandidateExtraction typed persisted shape round-trip with a real A-I3
    LLMInvocationProvenance;
  * JSON Schema parity + provider-schema strict-mode discipline;
  * PromptRegistry load / pinned hash / strict render variables.

Deliberately does NOT require: real Qwen, an HTTP fake transport, an artifact
store, a CURRENT pointer, or a real-novel fixture (those are later slices).
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from short_drama.artifacts import ArtifactRef, content_hash
from short_drama.io import load_json, load_yaml
from short_drama.llm import (
    LLMPromptError,
    LLMInvocationProvenance,
    OutputSchema,
    PromptRegistry,
    compute_prompt_content_hash,
    render_prompt,
)
from short_drama.paths import PROFILES_DIR, REPO_ROOT, SCHEMAS_DIR
from short_drama.story import (
    CandidateExtraction,
    CandidatePayload,
    CharacterCandidate,
    EventCandidate,
    EvidenceRef,
    ExtractionModelError,
    FactCandidate,
    LocationCandidate,
    RelationshipCandidate,
    STORY_EXTRACTION_MAX_GENERATION_ROUNDS_V1,
    StoryExtractionProfile,
    UnresolvedMentionCandidate,
    load_story_extraction_profile,
)

PROMPTS_STORY_DIR = REPO_ROOT / "prompts" / "story"
A3_PROMPT_ID = "a3.chunk-extraction"
A3_PROMPT_VERSION = 1
PROMPT_CONTENT_HASH = (
    "118469c47ea401ee35ef9164089f918494d884158df73b02f493aa260ad40a09"
)


# ---------------------------------------------------------------------------
# Fixtures / builders
# ---------------------------------------------------------------------------


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


def make_evidence(
    *,
    paragraph_id: str = "CH003_P0017",
    role: str = "primary",
    strength: str = "explicit",
    excerpt: str | None = None,
) -> EvidenceRef:
    return EvidenceRef(
        paragraph_id=paragraph_id, role=role, strength=strength, excerpt=excerpt
    )


def make_character(**overrides) -> CharacterCandidate:
    values = {
        "candidate_id": "cand_char_001",
        "display_name_original": "林晚",
        "aliases_original": ("阿晚",),
        "descriptors_zh": ("学生",),
        "summary_zh": "本章中的女主角。",
        "evidence_strength": "explicit",
        "evidence": (make_evidence(),),
    }
    values.update(overrides)
    return CharacterCandidate(**values)


def make_location(**overrides) -> LocationCandidate:
    values = {
        "candidate_id": "cand_loc_001",
        "display_name_original": "教室",
        "aliases_original": (),
        "descriptors_zh": ("教室",),
        "summary_zh": "故事发生的教室。",
        "evidence_strength": "explicit",
        "evidence": (make_evidence(paragraph_id="CH003_P0018"),),
    }
    values.update(overrides)
    return LocationCandidate(**values)


def make_fact(**overrides) -> FactCandidate:
    values = {
        "candidate_id": "cand_fact_001",
        "fact_type": "identity",
        "statement_zh": "林晚是一名学生。",
        "subject_refs": ("cand_char_001",),
        "object_refs": (),
        "evidence_strength": "explicit",
        "evidence": (make_evidence(),),
    }
    values.update(overrides)
    return FactCandidate(**values)


def make_event(**overrides) -> EventCandidate:
    values = {
        "candidate_id": "cand_evt_001",
        "summary_zh": "林晚走进教室。",
        "participant_refs": ("cand_char_001",),
        "location_refs": ("cand_loc_001",),
        "temporal_mode": "normal",
        "evidence_strength": "explicit",
        "evidence": (make_evidence(),),
    }
    values.update(overrides)
    return EventCandidate(**values)


def make_relationship(**overrides) -> RelationshipCandidate:
    values = {
        "candidate_id": "cand_rel_001",
        "source_ref": "cand_char_001",
        "target_ref": "cand_unres_001",
        "relationship_type_zh": "师生",
        "state_zh": None,
        "direction": "directed",
        "evidence_strength": "implied",
        "evidence": (make_evidence(),),
    }
    values.update(overrides)
    return RelationshipCandidate(**values)


def make_unresolved(**overrides) -> UnresolvedMentionCandidate:
    values = {
        "candidate_id": "cand_unres_001",
        "mention_original": "他",
        "mention_kind": "person",
        "reason_zh": "无法确定该代词指代的具体人物。",
        "possible_candidate_refs": ("cand_char_001",),
        "evidence_strength": "uncertain",
        "evidence": (make_evidence(strength="uncertain"),),
    }
    values.update(overrides)
    return UnresolvedMentionCandidate(**values)


CANDIDATE_BUILDERS = (
    make_character,
    make_location,
    make_fact,
    make_event,
    make_relationship,
    make_unresolved,
)


def full_payload() -> CandidatePayload:
    return CandidatePayload(
        characters=(make_character(),),
        locations=(make_location(),),
        facts=(make_fact(),),
        events=(make_event(),),
        relationships=(make_relationship(),),
        unresolved_mentions=(make_unresolved(),),
    )


def make_provenance(**overrides) -> LLMInvocationProvenance:
    values = {
        "provider_family": "qwen",
        "model": "qwen",
        "semantic_profile_id": "story-extraction-llm-v1",
        "semantic_profile_hash": "a" * 64,
        "prompt_id": A3_PROMPT_ID,
        "prompt_version": 1,
        "prompt_content_hash": "b" * 64,
        "rendered_prompt_hash": "c" * 64,
        "output_schema_id": "a3-candidate-payload",
        "output_schema_version": 1,
        "output_schema_hash": "d" * 64,
        "request_hash": "e" * 64,
        "provider_response_id": None,
        "finish_reason": None,
        "usage": None,
    }
    values.update(overrides)
    return LLMInvocationProvenance(**values)


def make_extraction(**overrides) -> CandidateExtraction:
    values = {
        "schema_version": 1,
        "project_id": "classroom",
        "document_id": "src_001",
        "chunk_profile_id": "story-analysis-v1",
        "chunk_id": "CH003_C002",
        "source_document_ref": ArtifactRef(
            artifact_type="source_document",
            artifact_id="classroom.src_001",
            revision=1,
            content_hash="f" * 64,
        ),
        "source_chunk_ref": ArtifactRef(
            artifact_type="source_chunk",
            artifact_id="classroom.src_001.story-analysis-v1.ch003_c002",
            revision=1,
            content_hash="0" * 64,
        ),
        "extraction_profile_id": "story-extraction-v1",
        "extraction_profile_hash": "1" * 64,
        "generation_provenance": make_provenance(),
        "candidates": full_payload(),
    }
    values.update(overrides)
    return CandidateExtraction(**values)


# ---------------------------------------------------------------------------
# 1. StoryExtractionProfile
# ---------------------------------------------------------------------------


def test_tracked_story_extraction_profile_loads_and_round_trips():
    profile = load_story_extraction_profile(
        PROFILES_DIR / "story_extraction_v1.yaml"
    )
    assert profile.schema_version == 1
    assert profile.profile_id == "story-extraction-v1"
    assert profile.working_language == "zh-CN"
    assert profile.prompt_id == "a3.chunk-extraction"
    assert profile.prompt_version == 2
    assert profile.output_schema_id == "a3-candidate-payload"
    assert profile.output_schema_version == 1
    assert profile.max_generation_rounds == 2
    assert StoryExtractionProfile.from_dict(profile.to_dict()) == profile


def test_profile_hash_uses_canonical_content_hash():
    profile = load_story_extraction_profile(
        PROFILES_DIR / "story_extraction_v1.yaml"
    )
    assert profile.profile_hash == content_hash(profile.to_dict())
    assert len(profile.profile_hash) == 64
    # hash is stable across reloads of the same tracked file
    again = load_story_extraction_profile(PROFILES_DIR / "story_extraction_v1.yaml")
    assert again.profile_hash == profile.profile_hash


def test_story_extraction_profile_schema_accepts_tracked_profile():
    profile = load_story_extraction_profile(
        PROFILES_DIR / "story_extraction_v1.yaml"
    )
    assert _validate(profile.to_dict(), _schema("story-extraction-profile.schema.json")) == []


def test_story_extraction_profile_schema_rejects_invalid():
    base = load_story_extraction_profile(
        PROFILES_DIR / "story_extraction_v1.yaml"
    ).to_dict()
    bad = dict(base)
    bad["schema_version"] = 2
    assert _validate(bad, _schema("story-extraction-profile.schema.json"))
    extra = dict(base)
    extra["base_url"] = "http://127.0.0.1:8080/v1"  # runtime field must be absent
    assert _validate(extra, _schema("story-extraction-profile.schema.json"))


def test_story_extraction_profile_rejects_runtime_and_bad_shape():
    base = load_story_extraction_profile(
        PROFILES_DIR / "story_extraction_v1.yaml"
    ).to_dict()
    extra = dict(base)
    extra["model"] = "qwen"
    with pytest.raises(ExtractionModelError):
        StoryExtractionProfile.from_dict(extra)
    missing = {k: v for k, v in base.items() if k != "max_generation_rounds"}
    with pytest.raises(ExtractionModelError):
        StoryExtractionProfile.from_dict(missing)
    bad_version = dict(base)
    bad_version["schema_version"] = 2
    with pytest.raises(ExtractionModelError):
        StoryExtractionProfile.from_dict(bad_version)


# Frozen A-I4 v1 ceiling: schema_version 1 requires exactly 2 semantic rounds.
def test_max_generation_rounds_v1_ceiling_accepted():
    base = load_story_extraction_profile(
        PROFILES_DIR / "story_extraction_v1.yaml"
    ).to_dict()
    assert STORY_EXTRACTION_MAX_GENERATION_ROUNDS_V1 == 2
    assert base["max_generation_rounds"] == 2
    # tracked value 2 passes both the Python model and the schema
    assert StoryExtractionProfile.from_dict(base) == StoryExtractionProfile.from_dict(
        base
    )
    assert _validate(base, _schema("story-extraction-profile.schema.json")) == []


@pytest.mark.parametrize("rounds", [1, 3, 100])
def test_max_generation_rounds_v1_ceiling_rejected(rounds):
    base = load_story_extraction_profile(
        PROFILES_DIR / "story_extraction_v1.yaml"
    ).to_dict()
    bad = dict(base)
    bad["max_generation_rounds"] = rounds
    # Python authoritative validation rejects any value other than 2
    with pytest.raises(ExtractionModelError):
        StoryExtractionProfile.from_dict(bad)
    # the schema enforces the same exact constraint (const: 2)
    assert _validate(bad, _schema("story-extraction-profile.schema.json"))


# ---------------------------------------------------------------------------
# 7. EvidenceRef
# ---------------------------------------------------------------------------


def test_evidence_ref_round_trip():
    for excerpt in (None, "林晚走进教室。"):
        for role in ("primary", "supporting"):
            for strength in ("explicit", "implied", "uncertain"):
                ref = make_evidence(role=role, strength=strength, excerpt=excerpt)
                assert EvidenceRef.from_dict(ref.to_dict()) == ref


def test_evidence_ref_rejects_bad_shape():
    with pytest.raises(ExtractionModelError):
        EvidenceRef(paragraph_id="", role="primary", strength="explicit", excerpt=None)
    with pytest.raises(ExtractionModelError):
        EvidenceRef(paragraph_id="CH003_P0017", role="bogus", strength="explicit", excerpt=None)
    with pytest.raises(ExtractionModelError):
        EvidenceRef(paragraph_id="CH003_P0017", role="primary", strength="bogus", excerpt=None)
    with pytest.raises(ExtractionModelError):
        EvidenceRef.from_dict({"paragraph_id": "CH003_P0017", "role": "primary"})


# ---------------------------------------------------------------------------
# 3. Each candidate model round-trips
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("builder", CANDIDATE_BUILDERS)
def test_each_candidate_type_round_trips(builder):
    obj = builder()
    assert type(obj).from_dict(obj.to_dict()) == obj


@pytest.mark.parametrize("builder", CANDIDATE_BUILDERS)
def test_candidate_model_accepts_list_inputs(builder):
    # from_dict feeds lists; to_dict emits lists; construction accepts lists.
    obj = builder()
    as_lists = {
        key: (list(value) if isinstance(value, tuple) else value)
        for key, value in obj.to_dict().items()
    }
    assert type(obj).from_dict(as_lists) == obj


# ---------------------------------------------------------------------------
# 4. Wrong candidate namespace rejected
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("builder", "bad_id"),
    [
        (make_character, "cand_loc_001"),
        (make_location, "cand_char_001"),
        (make_fact, "cand_evt_001"),
        (make_event, "cand_char_001"),
        (make_relationship, "cand_loc_001"),
        (make_unresolved, "cand_fact_001"),
        (make_character, "char_0001"),  # canonical ID shape is not allowed
        (make_character, "cand_char_01"),  # too short a numeric suffix
    ],
)
def test_wrong_candidate_namespace_rejected(builder, bad_id):
    with pytest.raises(ExtractionModelError):
        builder(candidate_id=bad_id)


# ---------------------------------------------------------------------------
# 5. Invalid enum rejected
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("builder", "field", "bad_value"),
    [
        (make_character, "evidence_strength", "bogus"),
        (make_fact, "fact_type", "bogus"),
        (make_event, "temporal_mode", "bogus"),
        (make_relationship, "direction", "bogus"),
        (make_unresolved, "mention_kind", "bogus"),
    ],
)
def test_invalid_enum_rejected(builder, field, bad_value):
    with pytest.raises(ExtractionModelError):
        builder(**{field: bad_value})


# ---------------------------------------------------------------------------
# 6. Missing and extra keys rejected
# ---------------------------------------------------------------------------


def test_missing_and_extra_keys_rejected():
    base = make_character().to_dict()
    missing = {key: value for key, value in base.items() if key != "summary_zh"}
    with pytest.raises(ExtractionModelError):
        CharacterCandidate.from_dict(missing)
    extra = dict(base)
    extra["surprise"] = "x"
    with pytest.raises(ExtractionModelError):
        CharacterCandidate.from_dict(extra)
    not_a_list = dict(base)
    not_a_list["evidence"] = "not-a-list"
    with pytest.raises(ExtractionModelError):
        CharacterCandidate.from_dict(not_a_list)


def test_duplicate_string_items_rejected():
    with pytest.raises(ExtractionModelError):
        make_character(aliases_original=("a", "a"))
    with pytest.raises(ExtractionModelError):
        make_fact(subject_refs=("cand_char_001", "cand_char_001"))


def test_empty_evidence_rejected():
    with pytest.raises(ExtractionModelError):
        make_character(evidence=())


# ---------------------------------------------------------------------------
# 8 / 9. CandidatePayload
# ---------------------------------------------------------------------------


def test_candidate_payload_full_six_category_round_trip():
    payload = full_payload()
    assert CandidatePayload.from_dict(payload.to_dict()) == payload
    as_dict = payload.to_dict()
    assert set(as_dict) == {
        "characters",
        "locations",
        "facts",
        "events",
        "relationships",
        "unresolved_mentions",
    }
    assert all(isinstance(v, list) and len(v) == 1 for v in as_dict.values())


def test_empty_category_arrays_accepted():
    payload = CandidatePayload()
    assert CandidatePayload.from_dict(payload.to_dict()) == payload
    assert payload.to_dict() == {
        "characters": [],
        "locations": [],
        "facts": [],
        "events": [],
        "relationships": [],
        "unresolved_mentions": [],
    }
    assert _validate(payload.to_dict(), _schema("candidate-payload.schema.json")) == []


# ---------------------------------------------------------------------------
# 10. Unresolved strength must be uncertain
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("strength", ["explicit", "implied"])
def test_unresolved_non_uncertain_strength_rejected(strength):
    with pytest.raises(ExtractionModelError):
        make_unresolved(evidence_strength=strength)


# ---------------------------------------------------------------------------
# 11. CandidateExtraction round-trip with a real A-I3 provenance
# ---------------------------------------------------------------------------


def test_candidate_extraction_round_trip_with_real_provenance():
    extraction = make_extraction()
    assert isinstance(extraction.generation_provenance, LLMInvocationProvenance)
    assert CandidateExtraction.from_dict(extraction.to_dict()) == extraction
    as_dict = extraction.to_dict()
    assert set(as_dict) == {
        "schema_version",
        "project_id",
        "document_id",
        "chunk_profile_id",
        "chunk_id",
        "source_document_ref",
        "source_chunk_ref",
        "extraction_profile_id",
        "extraction_profile_hash",
        "generation_provenance",
        "candidates",
    }
    assert extraction.artifact_type == "candidate_extraction"


def test_candidate_extraction_rejects_bad_shape():
    base = make_extraction().to_dict()
    # wrong provenance field set (dropping a real A-I3 field)
    prov = dict(base["generation_provenance"])
    del prov["request_hash"]
    bad = copy.deepcopy(base)
    bad["generation_provenance"] = prov
    with pytest.raises(ExtractionModelError):
        CandidateExtraction.from_dict(bad)
    # non-ArtifactRef source ref
    bad = copy.deepcopy(base)
    bad["source_document_ref"] = {"artifact_type": "x"}
    with pytest.raises(ExtractionModelError):
        CandidateExtraction.from_dict(bad)


def test_candidate_extraction_schema_accepts_round_tripped_extraction():
    extraction = make_extraction()
    assert _validate(
        extraction.to_dict(), _schema("candidate-extraction.schema.json")
    ) == []


# ---------------------------------------------------------------------------
# 12 / 13. Provider schema validation + Python/schema parity
# ---------------------------------------------------------------------------


def test_candidate_payload_schema_is_valid_output_schema():
    schema = _schema("candidate-payload.schema.json")
    out = OutputSchema.create(
        schema_id="a3-candidate-payload", schema_version=1, schema=schema
    )
    assert out.schema_id == "a3-candidate-payload"
    assert len(out.schema_hash) == 64


def test_representative_full_payload_passes_provider_schema():
    payload = full_payload()
    assert _validate(payload.to_dict(), _schema("candidate-payload.schema.json")) == []


def _set_fact_type(payload: dict, value: str) -> None:
    payload["facts"][0]["fact_type"] = value


def _set_character_namespace(payload: dict, value: str) -> None:
    payload["characters"][0]["candidate_id"] = value


def _drop_category(payload: dict, key: str) -> None:
    del payload[key]


def _empty_evidence(payload: dict) -> None:
    payload["facts"][0]["evidence"] = []


def _add_extra_category(payload: dict) -> None:
    payload["characters_extra"] = []


@pytest.mark.parametrize(
    "mutator",
    [
        lambda p: _set_fact_type(p, "bogus"),
        lambda p: _set_character_namespace(p, "cand_loc_001"),
        lambda p: _drop_category(p, "facts"),
        lambda p: _empty_evidence(p),
        lambda p: _add_extra_category(p),
    ],
)
def test_invalid_payload_fails_both_model_and_schema(mutator):
    bad = copy.deepcopy(full_payload().to_dict())
    mutator(bad)
    with pytest.raises(ExtractionModelError):
        CandidatePayload.from_dict(bad)
    assert _validate(bad, _schema("candidate-payload.schema.json"))


def test_valid_payload_accepted_by_both_model_and_schema():
    payload = full_payload()
    obj = CandidatePayload.from_dict(payload.to_dict())
    assert obj == payload
    assert _validate(payload.to_dict(), _schema("candidate-payload.schema.json")) == []


# ---------------------------------------------------------------------------
# 14. Provider schema strict-mode discipline
# ---------------------------------------------------------------------------


def _iter_object_nodes(schema):
    stack = [schema]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            if node.get("type") == "object" and isinstance(node.get("properties"), dict):
                yield node
            for value in node.values():
                if isinstance(value, (dict, list)):
                    stack.append(value)
        elif isinstance(node, list):
            stack.extend(node)


def _collect_keywords(node, found: set[str]) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            found.add(key)
            _collect_keywords(value, found)
    elif isinstance(node, list):
        for item in node:
            _collect_keywords(item, found)


def test_provider_schema_strict_mode_discipline():
    schema = _schema("candidate-payload.schema.json")
    object_nodes = list(_iter_object_nodes(schema))
    # root + six candidate objects + six evidence-ref objects
    assert object_nodes
    assert len(object_nodes) == 1 + 6 + 6
    for node in object_nodes:
        assert node.get("additionalProperties") is False, node
        for prop in node["properties"]:
            assert prop in node.get("required", []), (
                f"provider property {prop!r} is not explicitly required"
            )
    # No constructs that require provider-side repair or are disallowed in
    # strict structured-output mode. In particular the llama.cpp
    # JSON-schema-to-grammar path does not support `uniqueItems` (an
    # unsupported feature may be silently skipped rather than enforced), so it
    # must NOT appear in the provider-facing schema.
    found: set[str] = set()
    _collect_keywords(schema, found)
    forbidden = {
        "anyOf",
        "oneOf",
        "allOf",
        "not",
        "if",
        "then",
        "else",
        "$ref",
        "$defs",
        "$schema",
        "$id",
        "$comment",
        "title",
        "description",
        "definitions",
        "patternProperties",
        "uniqueItems",
    }
    assert not (found & forbidden), f"provider schema uses strict-incompatible keys: {found & forbidden}"


def test_provider_schema_defers_uniqueness_to_authoritative_validation():
    """Provider constraint < Python/local authoritative validation.

    The provider-facing schema intentionally omits `uniqueItems` because the
    local llama.cpp provider cannot enforce it. Uniqueness of string-list fields
    is still enforced authoritatively by the Python domain models (fail closed
    on duplicates), so dropping the keyword from the provider schema does not
    weaken the contract. The persisted artifact schema, which is local-only and
    never sent to the provider, may keep the stronger `uniqueItems` validation.
    """
    # provider schema carries no uniqueItems anywhere
    found: set[str] = set()
    _collect_keywords(_schema("candidate-payload.schema.json"), found)
    assert "uniqueItems" not in found
    # the persisted schema (local-only) retains the stronger uniqueness check
    found_persisted: set[str] = set()
    _collect_keywords(_schema("candidate-extraction.schema.json"), found_persisted)
    assert "uniqueItems" in found_persisted
    # the Python model remains authoritative: duplicates still fail closed
    with pytest.raises(ExtractionModelError):
        make_character(aliases_original=("a", "a"))
    # and a duplicate string list is rejected by the persisted schema too
    dup = full_payload().to_dict()
    dup["characters"][0]["aliases_original"] = ["a", "a"]
    assert _validate(dup, _schema("candidate-extraction.schema.json"))
    # while the provider schema (which cannot enforce uniqueness) accepts it
    assert _validate(dup, _schema("candidate-payload.schema.json")) == []


# ---------------------------------------------------------------------------
# 15 / 16 / 17. Prompt Registry
# ---------------------------------------------------------------------------


def test_prompt_registry_loads_real_a3_prompt():
    registry = PromptRegistry(PROMPTS_STORY_DIR)
    spec = registry.load(A3_PROMPT_ID, version=A3_PROMPT_VERSION)
    assert spec.prompt_id == A3_PROMPT_ID
    assert spec.version == A3_PROMPT_VERSION
    assert spec.required_variables == (
        "chunk_id",
        "left_context_json",
        "ownership_json",
        "right_context_json",
    )


def test_prompt_content_hash_pinned_matches():
    registry = PromptRegistry(PROMPTS_STORY_DIR)
    spec = registry.load(A3_PROMPT_ID, version=A3_PROMPT_VERSION)
    metadata = load_yaml(
        PROMPTS_STORY_DIR / A3_PROMPT_ID / "v1" / "prompt.yaml"
    )
    assert metadata["content_hash"] == PROMPT_CONTENT_HASH
    assert spec.content_hash == PROMPT_CONTENT_HASH
    expected = compute_prompt_content_hash(
        prompt_id=spec.prompt_id,
        version=spec.version,
        system_template=spec.system_template,
        user_template=spec.user_template,
        required_variables=spec.required_variables,
    )
    assert spec.content_hash == expected


def test_prompt_render_rejects_missing_and_unexpected_variables():
    registry = PromptRegistry(PROMPTS_STORY_DIR)
    spec = registry.load(A3_PROMPT_ID, version=A3_PROMPT_VERSION)
    full = {
        "chunk_id": "CH003_C002",
        "left_context_json": "[]",
        "ownership_json": '[]',
        "right_context_json": "[]",
    }
    # valid render works
    rendered = render_prompt(spec, full)
    assert len(rendered.rendered_prompt_hash) == 64
    # missing a required variable
    missing = {k: v for k, v in full.items() if k != "ownership_json"}
    with pytest.raises(LLMPromptError):
        render_prompt(spec, missing)
    # unexpected extra variable
    extra = dict(full)
    extra["bonus"] = "x"
    with pytest.raises(LLMPromptError):
        render_prompt(spec, extra)


def test_prompt_templates_carry_chunk_local_instructions():
    registry = PromptRegistry(PROMPTS_STORY_DIR)
    spec = registry.load(A3_PROMPT_ID, version=A3_PROMPT_VERSION)
    combined = spec.system_template + "\n" + spec.user_template
    # The prompt text must pin the key semantic disciplines.
    assert "CHUNK-LOCAL ONLY" in combined
    assert "WORKING LANGUAGE" in combined
    assert "OWNERSHIP SPAN" in combined
    assert "UNRESOLVED IS VALID" in combined
    assert "NO EXTRA DECISIONS" in combined
    assert "JSON object" in combined


def test_prompt_is_source_language_neutral():
    registry = PromptRegistry(PROMPTS_STORY_DIR)
    spec = registry.load(A3_PROMPT_ID, version=A3_PROMPT_VERSION)
    system = spec.system_template
    # The prompt must NOT constrain the source language (the old wording said
    # "a Chinese novel"). It states the source may be any language, keeps the
    # working language as zh-CN for summaries, and keeps source text original.
    assert "Chinese novel" not in system
    assert "any language" in system
    assert "WORKING LANGUAGE" in system
    assert "zh-CN" in system
    assert "original source form and original language" in system


def test_prompt_covers_local_reference_contract():
    registry = PromptRegistry(PROMPTS_STORY_DIR)
    spec = registry.load(A3_PROMPT_ID, version=A3_PROMPT_VERSION)
    system = spec.system_template
    # The local candidate-reference contract must be stated concisely.
    assert "LOCAL REFERENCES" in system
    for prefix in ("cand_char_", "cand_loc_", "cand_fact_", "cand_evt_", "cand_rel_", "cand_unres_"):
        assert prefix in system
    for ref_field in (
        "subject_refs",
        "object_refs",
        "participant_refs",
        "location_refs",
        "source_ref",
        "target_ref",
        "possible_candidate_refs",
    ):
        assert ref_field in system
    assert "same output payload" in system
    assert "Do not invent canonical" in system
    # unresolved identity remains valid and must not be forced
    assert "Do not force a guess" in system
