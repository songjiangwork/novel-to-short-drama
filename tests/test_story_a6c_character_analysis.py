"""A6C tests: character analysis semantic pass (fake provider).

Covers the A6C character analysis semantic pass implemented in
``short_drama.story.story_analysis_semantic`` (Issue #88):

* deterministic character request rendering -- one provider request per canonical
  planned character, whose ``character_context_json`` is the canonical JSON of the
  character's exact A6B evidence package;
* a backend-neutral *stable request identity* binding the exact upstream A5
  identity, the A6 plan identity, the ``StoryAnalysisProfile`` /
  ``SemanticLLMProfile`` id/hash, the prompt / output-schema asset identities, the
  character evidence packet hash, and the actual rendered request hash -- with NO
  runtime / provider fields;
* exact supporting-ref validation (a ref that exists somewhere in the global A5
  snapshot is not sufficient; it must belong to the character's own evidence
  package);
* bounded semantic regeneration (max 2 rounds); a schema-invalid output and every
  other technical LLM failure remains A-I3-owned (propagated, never a semantic
  retry);
* a complete in-memory ``CharacterAnalysisSet`` with 100% canonical-character
  coverage, no persistence, and no A5/A4 CURRENT mutation.

All provider invocations use deterministic fake ``LLMClient`` subclasses that
perform the same local authoritative JSON Schema validation as the real A-I3
provider boundary; no real provider is called. A real synthetic A5 CURRENT is
produced by reusing the A5F2 publication helpers (the exact persistence
bindings, not mocks).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from short_drama.llm import (
    GenerationExecutionOptions,
    LLMClient,
    LLMInvocationProvenance,
    PromptRegistry,
    StructuredGenerationResult,
    build_structured_request,
)
from short_drama.llm.errors import (
    LLMError,
    LLMStructuredOutputError,
    LLMTransportError,
)
from short_drama.llm.openai_compatible import validate_against_output_schema
from test_story_a5f2_current_publication import (
    CONSOLIDATION_PROFILE_ID,
    DOCUMENT,
    PROJECT,
    _build_tree_with_a4,
    _planning,
    _publish_a5,
)

from short_drama.story import (
    DEFAULT_PROMPT_BASE_DIR,
    CharacterAnalysis,
    CharacterEvidencePackage,
    StoryAnalysisProvenanceError,
    StoryAnalysisSemanticError,
    StoryAnalysisSemanticGenerationError,
    a5_pointer_id,
    build_story_analysis_plan_from_profile,
    build_story_analysis_snapshot,
    character_request_identity_hash,
    load_character_output_schema,
    load_character_semantic_profile,
    load_story_analysis_profile,
    resolve_character_analysis,
    build_character_semantic_preparation,
    validate_character_coverage,
    validate_character_evidence,
    A6C_CHARACTER_OUTPUT_SCHEMA_ID,
    A6C_CHARACTER_PROMPT_ID,
    A6C_SEMANTIC_PROFILE_ID,
)
from short_drama.foundation import PointerKind
from short_drama.paths import REPO_ROOT


# ---------------------------------------------------------------------------
# Fixture: a real synthetic A5 CURRENT (2 chars / 1 fact / 1 event / 1 rel).
# ---------------------------------------------------------------------------


def _profile():
    return load_story_analysis_profile(
        REPO_ROOT / "profiles" / "global_story_analysis_v1.yaml"
    )


def _semantic_profile():
    return load_character_semantic_profile()


def _prompts() -> PromptRegistry:
    return PromptRegistry(DEFAULT_PROMPT_BASE_DIR)


def _snapshot(tmp_path, *, suffix: str = "001"):
    tree, _ext, _a3, _idx = _build_tree_with_a4(tmp_path, suffix=suffix)
    planning = _planning(tree)
    pub = _publish_a5(tree, planning)
    snap = build_story_analysis_snapshot(
        tree.store,
        tree.pointers,
        project_id=PROJECT,
        document_id=DOCUMENT,
        consolidation_profile_id=CONSOLIDATION_PROFILE_ID,
    )
    return tree, planning, pub, snap


def _plan(snap, profile=None):
    return build_story_analysis_plan_from_profile(snap, profile or _profile())


def _prep(snap, *, profile=None, semantic_profile=None):
    return build_character_semantic_preparation(
        _plan(snap, profile),
        profile or _profile(),
        semantic_profile or _semantic_profile(),
        prompts=_prompts(),
    )


def _resolve(plan, client, *, profile=None, semantic_profile=None):
    return resolve_character_analysis(
        plan,
        profile or _profile(),
        semantic_profile or _semantic_profile(),
        client,
        prompts=_prompts(),
    )


# ---------------------------------------------------------------------------
# Fake provider (local authoritative schema validation, like A-I3).
# ---------------------------------------------------------------------------


def _make_provenance(request, *, provider_family="qwen", model="qwen3-27b", **overrides):
    """Build an ``LLMInvocationProvenance`` matching ``request`` exactly."""
    rendered = request.rendered_prompt
    schema = request.output_schema
    base = dict(
        provider_family=provider_family,
        model=model,
        semantic_profile_id=request.semantic_profile.profile_id,
        semantic_profile_hash=request.semantic_profile.semantic_profile_hash,
        prompt_id=rendered.prompt_id,
        prompt_version=rendered.prompt_version,
        prompt_content_hash=rendered.prompt_content_hash,
        rendered_prompt_hash=rendered.rendered_prompt_hash,
        output_schema_id=schema.schema_id,
        output_schema_version=schema.schema_version,
        output_schema_hash=schema.schema_hash,
        request_hash=request.request_hash,
        provider_response_id="resp",
        finish_reason="stop",
        usage=None,
    )
    base.update(overrides)
    return LLMInvocationProvenance(**base)


class FakeLLMClient(LLMClient):
    """Deterministic fake provider for A6C offline tests.

    Scripted responses (popped in order):
      * a ``dict`` -> local authoritative schema validation; a valid payload
        yields a successful result with provenance matching the request;
      * a ``(dict, LLMInvocationProvenance)`` tuple -> schema validation then a
        successful result with the EXACT supplied provenance;
      * an ``Exception`` -> raised from ``generate_structured``.

    Records ``call_count`` and the exact request identity per call so tests can
    assert the number of provider calls and the stable per-round request hash.
    """

    supported_structured_output_modes = frozenset({"none", "json_object", "json_schema"})

    def __init__(self, responses, *, provider_family="qwen", request_model="qwen3-27b"):
        self.responses = list(responses)
        self.call_count = 0
        self.request_hashes: list[str] = []
        self.request_identities: list[tuple[object, ...]] = []
        self.execution_options: list[GenerationExecutionOptions | None] = []
        self.provider_family = provider_family
        self.request_model = request_model

    def generate_structured(
        self, rendered_prompt, output_schema, semantic_profile, *, execution_options=None
    ):
        self.call_count += 1
        self.execution_options.append(execution_options)
        request = build_structured_request(
            rendered_prompt=rendered_prompt,
            output_schema=output_schema,
            semantic_profile=semantic_profile,
        )
        self.request_hashes.append(request.request_hash)
        self.request_identities.append(
            (
                request.request_hash,
                request.rendered_prompt.rendered_prompt_hash,
                request.output_schema.schema_id,
                request.output_schema.schema_version,
                request.output_schema.schema_hash,
                request.semantic_profile.profile_id,
                request.semantic_profile.semantic_profile_hash,
            )
        )
        if not self.responses:
            raise AssertionError("unexpected extra generate_structured call")
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        if isinstance(response, tuple):
            parsed, provenance = response
        else:
            parsed = response
            provenance = _make_provenance(
                request,
                provider_family=self.provider_family,
                model=self.request_model,
            )
        # Local authoritative JSON Schema validation (the A-I3 trust boundary).
        # A schema-invalid payload raises LLMStructuredOutputError (retryable,
        # A-I3-owned) BEFORE a result is returned -- exactly like the real
        # provider boundary.
        validate_against_output_schema(parsed, output_schema)
        return StructuredGenerationResult(
            parsed_json=parsed, provenance=provenance, attempts=1
        )


# ---------------------------------------------------------------------------
# Payload builders (schema-valid by construction).
# ---------------------------------------------------------------------------


def _interp(
    text_zh="分析",
    evidence_mode="inferred",
    supporting_fact_refs=(),
    supporting_event_refs=(),
    supporting_relationship_refs=(),
    supporting_conflict_refs=(),
):
    return {
        "text_zh": text_zh,
        "evidence_mode": evidence_mode,
        "supporting_fact_refs": list(supporting_fact_refs),
        "supporting_event_refs": list(supporting_event_refs),
        "supporting_relationship_refs": list(supporting_relationship_refs),
        "supporting_conflict_refs": list(supporting_conflict_refs),
    }


def _analysis_payload(package: CharacterEvidencePackage, **overrides):
    """A schema-valid, evidence-universe-valid CharacterAnalysis payload for
    ``package`` (references only that character's own evidence refs)."""
    payload = {
        "character_ref": package.character_ref,
        "role": _interp("主角角色", "explicit"),
        "goals": [_interp("追求目标", "inferred")],
        "motivations": [_interp("内在动机", "inferred")],
        "traits": [_interp("性格特质", "inferred")],
        "key_event_refs": [e.event_id for e in package.participating_events],
        "key_fact_refs": [f.fact_id for f in package.related_facts],
        "important_relationship_refs": [r.relationship_id for r in package.relationships],
        "arc_summary": _interp("成长弧线", "explicit"),
        "unresolved_or_conflicting_points": [],
    }
    payload.update(overrides)
    return payload


# ---------------------------------------------------------------------------
# A. Deterministic same-input same-request
# ---------------------------------------------------------------------------


def test_a_deterministic_same_input_same_request(tmp_path: Path) -> None:
    _tree, _planning, _pub, snap = _snapshot(tmp_path)
    plan_a = _plan(snap)
    plan_b = _plan(snap)  # rebuild the identical plan
    prep_a = build_character_semantic_preparation(
        plan_a, _profile(), _semantic_profile(), prompts=_prompts()
    )
    prep_b = build_character_semantic_preparation(
        plan_b, _profile(), _semantic_profile(), prompts=_prompts()
    )
    # Same input -> byte-identical requests and stable request identities.
    assert prep_a.character_request_hashes == prep_b.character_request_hashes
    assert (
        prep_a.character_request_identity_hashes
        == prep_b.character_request_identity_hashes
    )
    assert prep_a.plan.plan_hash == prep_b.plan.plan_hash
    # The rendered request content is stable across rebuilds too.
    for ra, rb in zip(prep_a.character_requests, prep_b.character_requests):
        assert ra.rendered_prompt.user_text == rb.rendered_prompt.user_text
        assert ra.rendered_prompt.system_text == rb.rendered_prompt.system_text
        assert ra.rendered_prompt.rendered_prompt_hash == rb.rendered_prompt.rendered_prompt_hash


# ---------------------------------------------------------------------------
# B. Different input identity changes request hash
# ---------------------------------------------------------------------------


def test_b_different_input_identity_changes_request_hash(tmp_path: Path) -> None:
    # Two disjoint A5 CURRENTs (different candidate identities) -> different
    # evidence packets -> different rendered request hashes and identities,
    # even though the canonical character refs are positional (char_0001/02).
    snap1 = _snapshot(tmp_path / "a", suffix="001")[3]
    snap2 = _snapshot(tmp_path / "b", suffix="002")[3]
    prep1 = build_character_semantic_preparation(
        _plan(snap1), _profile(), _semantic_profile(), prompts=_prompts()
    )
    prep2 = build_character_semantic_preparation(
        _plan(snap2), _profile(), _semantic_profile(), prompts=_prompts()
    )
    assert prep1.character_refs == prep2.character_refs  # same positional refs
    assert (
        prep1.character_request_identity_hashes
        != prep2.character_request_identity_hashes
    )
    assert prep1.character_request_hashes != prep2.character_request_hashes
    assert prep1.plan.consolidation_manifest_ref != prep2.plan.consolidation_manifest_ref


# ---------------------------------------------------------------------------
# C. Stable identity contains no runtime/provider fields
# ---------------------------------------------------------------------------


def test_c_stable_identity_has_no_runtime_or_provider_fields(tmp_path: Path) -> None:
    _tree, _planning, _pub, snap = _snapshot(tmp_path)
    plan = _plan(snap)
    prep = build_character_semantic_preparation(
        plan, _profile(), _semantic_profile(), prompts=_prompts()
    )
    # Recompute each stable request identity from ONLY the semantic fields and
    # verify it matches exactly (proving no hidden runtime/provider input).
    for package, request, ident in zip(
        plan.character_packages, prep.character_requests, prep.character_request_identity_hashes
    ):
        recomputed = character_request_identity_hash(
            consolidation_manifest_ref=plan.consolidation_manifest_ref,
            plan_hash=plan.plan_hash,
            profile=prep.profile,
            semantic_profile=prep.semantic_profile,
            prompt_identity=prep.prompt_identity,
            output_schema_identity=prep.output_schema_identity,
            character_ref=package.character_ref,
            packet_hash=package.content_hash(),
            request_hash=request.request_hash,
        )
        assert recomputed == ident
    # The identity is a 64-char sha256 hex string, deterministic and stable.
    for ident in prep.character_request_identity_hashes:
        assert len(ident) == 64
        assert all(c in "0123456789abcdef" for c in ident)


# ---------------------------------------------------------------------------
# D. Invalid provenance or request hash (fail closed, no retry)
# ---------------------------------------------------------------------------


class ProvenanceMismatchClient(FakeLLMClient):
    """A client whose successful result carries a provenance with one field
    mismatched. If A6C (incorrectly) retried on a provenance mismatch, the call
    count would exceed 1."""

    def __init__(self, valid_payload, *, field):
        super().__init__([None])  # responses are never consumed
        self._payload = valid_payload
        self._field = field

    def generate_structured(
        self, rendered_prompt, output_schema, semantic_profile, *, execution_options=None
    ):
        self.call_count += 1
        self.execution_options.append(execution_options)
        request = build_structured_request(
            rendered_prompt=rendered_prompt,
            output_schema=output_schema,
            semantic_profile=semantic_profile,
        )
        self.request_hashes.append(request.request_hash)
        provenance = dataclasses.replace(
            _make_provenance(request), **{self._field: "0" * 64}
        )
        validate_against_output_schema(self._payload, output_schema)
        return StructuredGenerationResult(
            parsed_json=self._payload, provenance=provenance, attempts=1
        )


@pytest.mark.parametrize(
    "field",
    [
        "semantic_profile_hash",
        "prompt_content_hash",
        "rendered_prompt_hash",
        "output_schema_hash",
        "request_hash",
    ],
)
def test_d_provenance_or_request_hash_mismatch_fails_closed_no_retry(
    tmp_path: Path, field: str
) -> None:
    _tree, _planning, _pub, snap = _snapshot(tmp_path)
    plan = _plan(snap)
    first_package = plan.character_packages[0]
    client = ProvenanceMismatchClient(_analysis_payload(first_package), field=field)
    with pytest.raises(StoryAnalysisProvenanceError, match="provenance field mismatch"):
        _resolve(plan, client)
    # FAIL CLOSED: exactly one provider call, no semantic retry.
    assert client.call_count == 1
    assert client.execution_options == [None]


# ---------------------------------------------------------------------------
# E. Character packet budget overflow (no truncation)
# ---------------------------------------------------------------------------


def test_e_packet_budget_overflow_no_truncation(tmp_path: Path) -> None:
    _tree, _planning, _pub, snap = _snapshot(tmp_path)
    # Build the plan with the production (large) budget, then validate against a
    # tiny budget: A6C must fail closed before any provider call, never
    # truncate the evidence packet.
    plan = _plan(snap)
    tiny_policy = dataclasses.replace(
        _profile().planning_policy, character_packet_max_estimated_tokens=1
    )
    tiny_profile = dataclasses.replace(_profile(), planning_policy=tiny_policy)
    assert plan.character_packages[0].estimated_tokens() > 1
    with pytest.raises(StoryAnalysisSemanticError, match="refusing to truncate"):
        build_character_semantic_preparation(
            plan, tiny_profile, _semantic_profile(), prompts=_prompts()
        )


# ---------------------------------------------------------------------------
# F. Correct complete CharacterAnalysisSet with valid refs
# ---------------------------------------------------------------------------


def test_f_complete_character_analysis_set_with_valid_refs(tmp_path: Path) -> None:
    _tree, _planning, _pub, snap = _snapshot(tmp_path)
    plan = _plan(snap)
    payloads = [_analysis_payload(p) for p in plan.character_packages]
    client = FakeLLMClient(payloads)
    result = _resolve(plan, client)
    # One provider call per canonical character (valid on round 1).
    assert client.call_count == len(plan.character_packages)
    # Complete set: every planned character, exactly once, in canonical order.
    planned_refs = [p.character_ref for p in plan.character_packages]
    assert [a.character_ref for a in result.analyses] == planned_refs
    assert len(result.character_analysis_set.analyses) == len(plan.character_packages)
    assert [a.character_ref for a in result.character_analysis_set.analyses] == planned_refs
    assert all(rounds == 1 for _ref, rounds in result.character_rounds)
    # The typed analyses carry the model's text, preserve evidence modes, and
    # reference only that character's own evidence universe.
    for analysis, package in zip(result.analyses, plan.character_packages):
        assert analysis.role.text_zh == "主角角色"
        assert analysis.role.evidence_mode == "explicit"
        assert set(analysis.key_fact_refs) <= {
            f.fact_id for f in package.related_facts
        }
        assert validate_character_evidence(analysis, package) is None
        assert analysis in result.character_analysis_set.analyses


# ---------------------------------------------------------------------------
# G. Invalid character_ref (wrong character)
# ---------------------------------------------------------------------------


def test_g_wrong_character_ref_rejected(tmp_path: Path) -> None:
    _tree, _planning, _pub, snap = _snapshot(tmp_path)
    plan = _plan(snap)
    char1, char2 = plan.character_packages
    # A schema-valid payload whose character_ref does NOT match the requested
    # character -> semantic invalid (wrong character_ref), both rounds.
    wrong = _analysis_payload(char1, character_ref=char2.character_ref)
    client = FakeLLMClient([wrong, wrong])
    with pytest.raises(StoryAnalysisSemanticGenerationError) as exc_info:
        _resolve(plan, client)
    err = exc_info.value
    assert err.character_ref == char1.character_ref
    assert err.rounds_attempted == 2
    assert "wrong character_ref" in " ".join(err.last_failure_details)
    # Both rounds consumed for the first character; the second was never reached.
    assert client.call_count == 2
    assert client.execution_options == [None, GenerationExecutionOptions(prompt_context_reuse="disabled")]


# ---------------------------------------------------------------------------
# H. Foreign-character refs (exists in global snapshot, not this char's package)
# ---------------------------------------------------------------------------


def test_h_foreign_ref_rejected(tmp_path: Path) -> None:
    _tree, _planning, _pub, snap = _snapshot(tmp_path)
    plan = _plan(snap)
    char1, char2 = plan.character_packages
    # char_0002 has no facts; char_0001's fact exists in the global A5 snapshot
    # but NOT in char_0002's evidence package -> a foreign ref.
    foreign_fact = char1.related_facts[0].fact_id
    assert foreign_fact not in {f.fact_id for f in char2.related_facts}
    foreign_payload = _analysis_payload(char2)
    foreign_payload["role"] = _interp(
        "角色", "explicit", supporting_fact_refs=(foreign_fact,)
    )
    # char_0001 valid on round 1; char_0002 foreign on both rounds -> exhausted.
    client = FakeLLMClient([_analysis_payload(char1), foreign_payload, foreign_payload])
    with pytest.raises(StoryAnalysisSemanticGenerationError) as exc_info:
        _resolve(plan, client)
    err = exc_info.value
    assert err.character_ref == char2.character_ref
    assert err.rounds_attempted == 2
    assert any(foreign_fact in detail for detail in err.last_failure_details)
    assert client.call_count == 3


# ---------------------------------------------------------------------------
# I / J. Coverage invariants (missing / duplicate / extra)
# ---------------------------------------------------------------------------


def _analysis(char_ref: str) -> CharacterAnalysis:
    return CharacterAnalysis.from_dict(
        {
            "character_ref": char_ref,
            "role": _interp("角色"),
            "goals": [],
            "motivations": [],
            "traits": [],
            "key_event_refs": [],
            "key_fact_refs": [],
            "important_relationship_refs": [],
            "arc_summary": _interp("弧线"),
            "unresolved_or_conflicting_points": [],
        }
    )


def test_i_missing_character_response_rejected() -> None:
    with pytest.raises(StoryAnalysisSemanticError, match="missing character analysis"):
        validate_character_coverage(
            [_analysis("char_0001")], ["char_0001", "char_0002"]
        )


def test_j_duplicate_character_response_rejected() -> None:
    with pytest.raises(StoryAnalysisSemanticError, match="duplicate character analysis"):
        validate_character_coverage(
            [_analysis("char_0001"), _analysis("char_0001")], ["char_0001"]
        )


def test_j_extra_character_response_rejected() -> None:
    with pytest.raises(StoryAnalysisSemanticError, match="extra/unknown character analysis"):
        validate_character_coverage(
            [_analysis("char_0001"), _analysis("char_0002"), _analysis("char_0003")],
            ["char_0001", "char_0002"],
        )


# ---------------------------------------------------------------------------
# K. Malformed provider JSON/schema (technical, A-I3-owned, propagated)
# ---------------------------------------------------------------------------


def test_k_malformed_schema_is_technical_error(tmp_path: Path) -> None:
    _tree, _planning, _pub, snap = _snapshot(tmp_path)
    plan = _plan(snap)
    char1, char2 = plan.character_packages
    bad = _analysis_payload(char1)
    bad["role"] = _interp("角色", "invalid_mode")  # violates the schema enum
    client = FakeLLMClient([bad, _analysis_payload(char2)])
    # A schema-invalid output is a technical LLM failure (the local authoritative
    # JSON Schema validation rejects it) and is PROPAGATED, never a semantic
    # retry.
    with pytest.raises(LLMStructuredOutputError) as exc_info:
        _resolve(plan, client)
    assert client.call_count == 1
    # The error is a technical LLM failure, not an A6C semantic error.
    assert isinstance(exc_info.value, LLMError)
    assert not isinstance(exc_info.value, StoryAnalysisSemanticError)


# ---------------------------------------------------------------------------
# Store-state helpers (persistence / CURRENT-mutation checks).
# ---------------------------------------------------------------------------


def _a5_current_ref(tree):
    return tree.pointers.resolve_current(
        a5_pointer_id(PROJECT, DOCUMENT, CONSOLIDATION_PROFILE_ID)
    ).target_ref


def _store_file_count(tree) -> int:
    return len(list(tree.store.root.rglob("*.json")))


# ---------------------------------------------------------------------------
# L. Semantic regeneration success on round 2
# ---------------------------------------------------------------------------


def test_l_regeneration_success_on_round_2(tmp_path: Path) -> None:
    _tree, _planning, _pub, snap = _snapshot(tmp_path)
    plan = _plan(snap)
    char1, char2 = plan.character_packages
    # Round 1 for char_0001: schema-valid but a foreign fact ref (domain-invalid);
    # round 2: valid. char_0002: valid on round 1.
    bad = _analysis_payload(char1)
    bad["role"] = _interp("角色", "explicit", supporting_fact_refs=("fact_000999",))
    client = FakeLLMClient(
        [bad, _analysis_payload(char1), _analysis_payload(char2)]
    )
    result = _resolve(plan, client)
    assert client.call_count == 3
    assert dict(result.character_rounds) == {
        char1.character_ref: 2,
        char2.character_ref: 1,
    }
    # Round 2 disables prompt-context reuse (A5 semantic-retry pattern).
    assert client.execution_options == [
        None,
        GenerationExecutionOptions(prompt_context_reuse="disabled"),
        None,
    ]
    # Complete set: both characters present, in canonical order.
    assert [a.character_ref for a in result.analyses] == [
        char1.character_ref,
        char2.character_ref,
    ]
    assert client.request_hashes[0] == client.request_hashes[1]  # same request both rounds


# ---------------------------------------------------------------------------
# M. Semantic regeneration exhausted at round 2 (fail closed)
# ---------------------------------------------------------------------------


def test_m_regeneration_exhausted_at_round_2(tmp_path: Path) -> None:
    _tree, _planning, _pub, snap = _snapshot(tmp_path)
    plan = _plan(snap)
    char1, char2 = plan.character_packages
    bad1 = _analysis_payload(char1)
    bad1["role"] = _interp("角色", "explicit", supporting_fact_refs=("fact_000999",))
    bad2 = _analysis_payload(char1)
    bad2["key_fact_refs"] = ["fact_000888"]  # a different foreign ref on round 2
    client = FakeLLMClient([bad1, bad2])
    with pytest.raises(StoryAnalysisSemanticGenerationError) as exc_info:
        _resolve(plan, client)
    err = exc_info.value
    assert err.character_ref == char1.character_ref
    assert err.rounds_attempted == 2
    assert err.request_hash == client.request_hashes[0]
    assert len(err.last_failure_details) == 2
    assert client.call_count == 2


# ---------------------------------------------------------------------------
# N. Technical retry remains A-I3-owned (LLMError propagated, no semantic retry)
# ---------------------------------------------------------------------------


def test_n_technical_retry_remains_ai3_owned(tmp_path: Path) -> None:
    _tree, _planning, _pub, snap = _snapshot(tmp_path)
    plan = _plan(snap)
    client = FakeLLMClient(
        [LLMTransportError("connection reset (technical retry budget exhausted)")]
    )
    with pytest.raises(LLMError) as exc_info:
        _resolve(plan, client)
    # A technical LLMError is propagated immediately; it is NOT routed into a
    # semantic retry and is NOT an A6C semantic error.
    assert client.call_count == 1
    assert not isinstance(exc_info.value, StoryAnalysisSemanticError)
    assert client.execution_options == [None]


# ---------------------------------------------------------------------------
# O. No partial persistence after regeneration failure (in-memory only)
# ---------------------------------------------------------------------------


def test_o_no_partial_persistence_after_regeneration_failure(tmp_path: Path) -> None:
    tree, _planning, _pub, snap = _snapshot(tmp_path)
    plan = _plan(snap)
    before_a5 = _a5_current_ref(tree)
    before_files = _store_file_count(tree)
    char1, char2 = plan.character_packages
    bad = _analysis_payload(char1)
    bad["role"] = _interp("角色", "explicit", supporting_fact_refs=("fact_000999",))
    client = FakeLLMClient([bad, bad])
    with pytest.raises(StoryAnalysisSemanticGenerationError):
        _resolve(plan, client)
    # In-memory only: no A6 artifact is persisted and the A5 CURRENT is unchanged.
    assert _a5_current_ref(tree) == before_a5
    assert _store_file_count(tree) == before_files


# ---------------------------------------------------------------------------
# P. No silent truncation (complete rendered request includes the full packet)
# ---------------------------------------------------------------------------


def test_p_no_silent_truncation(tmp_path: Path) -> None:
    _tree, _planning, _pub, snap = _snapshot(tmp_path)
    plan = _plan(snap)
    prep = build_character_semantic_preparation(
        plan, _profile(), _semantic_profile(), prompts=_prompts()
    )
    for package, request in zip(plan.character_packages, prep.character_requests):
        full_packet = package.canonical_bytes().decode("utf-8")
        # The complete rendered request contains the ENTIRE evidence packet
        # (no truncation), wrapped in the prompt framing.
        assert full_packet in request.rendered_prompt.user_text
        assert request.rendered_prompt.user_text.strip() != full_packet
    # The complete rendered request is strictly larger than the packet alone
    # (prompt framing is measured separately, never counted against the budget).
    assert all(
        rendered > packet
        for rendered, packet in zip(
            prep.rendered_request_token_estimates, prep.packet_token_estimates
        )
    )


# ---------------------------------------------------------------------------
# Q. No A6 persistence / CURRENT mutation
# ---------------------------------------------------------------------------


def test_q_no_a6_current_mutation(tmp_path: Path) -> None:
    tree, _planning, _pub, snap = _snapshot(tmp_path)
    plan = _plan(snap)
    before_a5 = _a5_current_ref(tree)
    before_files = _store_file_count(tree)
    client = FakeLLMClient([_analysis_payload(p) for p in plan.character_packages])
    result = _resolve(plan, client)
    # A6C is in-memory only: the A5 CURRENT is unchanged, no A6 artifact is
    # persisted, and the result exists only in memory.
    assert _a5_current_ref(tree) == before_a5
    assert _store_file_count(tree) == before_files
    assert len(result.character_analysis_set.analyses) == len(plan.character_packages)


# ---------------------------------------------------------------------------
# R. Request hash stable across backend/provider fields
# ---------------------------------------------------------------------------


def test_r_request_hash_stable_across_backend_fields(tmp_path: Path) -> None:
    _tree, _planning, _pub, snap = _snapshot(tmp_path)
    plan = _plan(snap)
    payloads_a = [_analysis_payload(p) for p in plan.character_packages]
    payloads_b = [_analysis_payload(p) for p in plan.character_packages]
    result_a = _resolve(
        plan,
        FakeLLMClient(payloads_a, provider_family="qwen", request_model="qwen3-27b"),
    )
    result_b = _resolve(
        plan,
        FakeLLMClient(payloads_b, provider_family="llama", request_model="llama-405b"),
    )
    # The raw request hashes and the stable request identities are
    # backend-neutral: identical across provider family / model changes.
    assert (
        result_a.preparation.character_request_hashes
        == result_b.preparation.character_request_hashes
    )
    assert (
        result_a.preparation.character_request_identity_hashes
        == result_b.preparation.character_request_identity_hashes
    )
    assert result_a.preparation.character_request_identity_hashes  # non-empty


# ---------------------------------------------------------------------------
# Robustness: frozen A6C identity is enforced (fail closed).
# ---------------------------------------------------------------------------


def test_profile_identity_mismatch_fails_closed(tmp_path: Path) -> None:
    _tree, _planning, _pub, snap = _snapshot(tmp_path)
    plan = _plan(snap)
    base = _profile()
    # A wrong character-analysis prompt id -> the frozen A6C identity is broken.
    bad_prompt = dataclasses.replace(
        base.character_analysis, prompt_id="a6.wrong-prompt"
    )
    bad_profile = dataclasses.replace(base, character_analysis=bad_prompt)
    with pytest.raises(StoryAnalysisSemanticError, match="prompt must be"):
        build_character_semantic_preparation(
            plan, bad_profile, _semantic_profile(), prompts=_prompts()
        )


def test_wrong_semantic_profile_id_fails_closed(tmp_path: Path) -> None:
    _tree, _planning, _pub, snap = _snapshot(tmp_path)
    plan = _plan(snap)
    # A semantic profile with the WRONG id must be rejected by the identity
    # verification (fail closed).
    bad_semantic = dataclasses.replace(
        _semantic_profile(), profile_id="story-analysis-llm-v2"
    )
    with pytest.raises(StoryAnalysisSemanticError, match="semantic profile"):
        build_character_semantic_preparation(
            plan, _profile(), bad_semantic, prompts=_prompts()
        )


def test_request_identity_changes_when_semantic_field_changes(tmp_path: Path) -> None:
    _tree, _planning, _pub, snap = _snapshot(tmp_path)
    plan = _plan(snap)
    base_prep = build_character_semantic_preparation(
        plan, _profile(), _semantic_profile(), prompts=_prompts()
    )
    # A different semantic profile (different content hash) -> different stable
    # request identities (the identity binds the semantic profile id/hash).
    other_semantic = dataclasses.replace(
        _semantic_profile(), temperature=0.3
    )
    other_prep = build_character_semantic_preparation(
        plan, _profile(), other_semantic, prompts=_prompts()
    )
    assert base_prep.character_request_identity_hashes != other_prep.character_request_identity_hashes
    # The raw request hashes also differ (the request binds the semantic profile).
    assert base_prep.character_request_hashes != other_prep.character_request_hashes
    # But the character refs and packet hashes are unchanged (input-driven).
    assert base_prep.character_refs == other_prep.character_refs


# ---------------------------------------------------------------------------
# Real-provider smoke (bounded, gated on a live OpenAI-compatible server).
# ---------------------------------------------------------------------------


def _live_server_available() -> bool:
    """Whether the local OpenAI-compatible server (profiles/llm_local.yaml) is
    reachable. The real-provider smoke is gated on this so the default suite
    stays offline."""
    import urllib.request

    try:
        with urllib.request.urlopen(
            "http://127.0.0.1:8080/v1/models", timeout=3
        ) as response:
            return response.status == 200
    except Exception:
        return False


@pytest.mark.skipif(
    not _live_server_available(),
    reason="live OpenAI-compatible server not available",
)
def test_real_provider_smoke(tmp_path: Path) -> None:
    """One bounded real-provider smoke: exact request rendering + real JSON/
    schema output + typed parsing + provenance verification + exact
    supporting-ref validation, end to end against a live local Qwen server.

    A schema-valid-but-semantically-invalid output (refs outside the character's
    own evidence universe) is a legitimate A6C semantic outcome: A6C fails
    closed with ``StoryAnalysisSemanticGenerationError`` after two rounds. That
    still proves the real provider path worked (no technical LLM failure).
    """
    from short_drama.llm import (
        OpenAICompatibleLLMClient,
        load_runtime_config,
    )

    _tree, _planning, _pub, snap = _snapshot(tmp_path)
    plan = _plan(snap)
    profile = _profile()
    semantic_profile = load_character_semantic_profile()
    prompts = _prompts()

    runtime_config = load_runtime_config(REPO_ROOT / "profiles" / "llm_local.yaml")
    client = OpenAICompatibleLLMClient(runtime_config)

    try:
        result = resolve_character_analysis(
            plan, profile, semantic_profile, client, prompts=prompts
        )
    except StoryAnalysisSemanticGenerationError as exc:
        # Semantic rejection (model refs outside the universe): the real
        # provider path worked end to end; A6C failed closed correctly.
        print(
            f"\n[smoke] semantic rejection: character={exc.character_ref} "
            f"rounds={exc.rounds_attempted} details={exc.last_failure_details}"
        )
        return

    # Success: complete in-memory set, every ref in the character's own evidence
    # universe, and no persistence / CURRENT mutation.
    assert len(result.character_analysis_set.analyses) == len(plan.character_packages)
    assert [a.character_ref for a in result.analyses] == [
        p.character_ref for p in plan.character_packages
    ]
    for analysis, package in zip(result.analyses, plan.character_packages):
        assert validate_character_evidence(analysis, package) is None
    print(
        f"\n[smoke] SUCCESS: {len(plan.character_packages)} characters, "
        f"rounds={dict(result.character_rounds)}"
    )
