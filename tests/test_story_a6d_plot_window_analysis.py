"""A6D tests: plot-window analysis semantic pass (fake provider).

Covers the A6D plot-window analysis semantic pass implemented in
``short_drama.story.story_analysis_window_semantic`` (Issue #89):

* deterministic window request rendering -- one provider request per canonical
  planned window, whose ``window_context_json`` is the canonical JSON of the
  window's exact A6B evidence packet;
* a backend-neutral *stable request identity* binding the exact upstream A5
  identity, the A6 plan identity, the ``StoryAnalysisProfile`` /
  ``SemanticLLMProfile`` id/hash, the prompt / output-schema asset identities,
  the window identity/ordinal, the exact owned/context event membership, the
  window packet content hash, and the actual rendered request hash -- with NO
  runtime / provider fields;
* exact local-evidence-ref validation (a ref that exists somewhere in the global
  A5 snapshot is not sufficient; it must belong to the window's own evidence
  universe = the union of the window's owned + context events and the packet's
  relevant fact / relationship / conflict collections);
* Python-owned ordinal / owned / context membership (copied from the
  authoritative A6B plan, never from the provider);
* bounded semantic regeneration (max 2 rounds); a schema-invalid output and every
  other technical LLM failure remains A-I3-owned (propagated, never a semantic
  retry); a provenance / request-hash mismatch fails closed;
* a complete in-memory ordered ``PlotWindowAnalysis`` set with exact window
  coverage, no persistence, and no A5/A4 CURRENT mutation.

All provider invocations use deterministic fake ``LLMClient`` subclasses that
perform the same local authoritative JSON Schema validation as the real A-I3
provider boundary; no real provider is called (except the final bounded
real-provider smoke, gated on a live local Qwen server).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from types import SimpleNamespace

import pytest

from short_drama.artifacts import ArtifactRef
from short_drama.artifacts.canonical import content_hash
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
    CanonicalEvent,
    CanonicalFact,
    CanonicalRelationship,
    StoryConflict,
    EvidenceBackedInterpretation,
    PlotWindowAnalysis,
    StoryAnalysisInputSnapshot,
    StoryAnalysisProvenanceError,
    StoryAnalysisSemanticError,
    StoryAnalysisWindowSemanticGenerationError,
    StoryIntegrityError,
    a5_pointer_id,
    build_story_analysis_plan,
    build_story_analysis_snapshot,
    build_window_packets,
    load_story_analysis_profile,
    planning_policy_ids_from_profile,
    # A6D
    A6D_MAX_GENERATION_ROUNDS,
    A6D_SEMANTIC_PROFILE_ID,
    A6D_WINDOW_OUTPUT_SCHEMA_ID,
    A6D_WINDOW_OUTPUT_SCHEMA_VERSION,
    A6D_WINDOW_PROMPT_ID,
    A6D_WINDOW_PROMPT_VERSION,
    build_plot_window_semantic_preparation,
    load_window_output_schema,
    load_window_semantic_assets,
    load_window_semantic_profile,
    resolve_plot_window_analysis,
    validate_plot_window_coverage,
    validate_plot_window_evidence,
    window_request_identity_hash,
)
import short_drama.paths as paths
import short_drama.story.story_analysis_window_semantic as a6d
from short_drama.story.extraction import EvidenceRef
from short_drama.foundation import PointerKind
from short_drama.paths import REPO_ROOT


# ---------------------------------------------------------------------------
# Deterministic in-memory A5 snapshot (multi-window, real context overlap).
# ---------------------------------------------------------------------------

#: A small deterministic "meaningful" plot: 20 consecutive canonical events
#: forming a coherent beat sequence. Window target 12 / context 4 (the frozen
#: A6B planning policy) yields 2 windows with a real context-overlap boundary.
_PLOT_BEATS = [
    "主角收到一封来自已故母亲的遗书。",
    "主角决定回到故乡的旧宅。",
    "主角在旧宅发现一只上锁的木箱。",
    "主角与多年未见的堂兄重逢。",
    "堂兄警告主角不要打开木箱。",
    "主角深夜独自打开了木箱。",
    "木箱里是一叠发黄的信和一张地图。",
    "主角读到母亲隐瞒多年的秘密。",
    "主角与堂兄爆发激烈争吵。",
    "主角决定追查地图上的地点。",
    "主角离开旧宅前往码头。",
    "主角在码头遇见一名神秘的船夫。",
    "船夫向主角提出一个危险的交易。",
    "主角犹豫是否接受交易。",
    "堂兄暗中跟踪主角来到码头。",
    "堂兄和船夫之间的旧怨浮出水面。",
    "主角发现船夫与母亲的秘密有关。",
    "主角被迫登上小船离开港口。",
    "风暴中主角找到了地图上的孤岛。",
    "主角在孤岛揭开了家族最终的真相。",
]


def _cand(domain: str, idx: int) -> str:
    """A valid consolidation candidate ref (``CH###_C###:cand_<domain>_###``)."""
    return f"CH001_C001:cand_{domain}_{idx:03d}"


def _char_for_event(i: int) -> str:
    """Assign events to 4 characters so the two windows have distinct (but
    overlapping) evidence universes:

    * window 1 (owned evt 1..12, context evt 13..16) binds chars 1..3;
    * window 2 (owned evt 13..20, context evt 9..12) binds chars 2..4.
    """
    if i <= 8:
        return "char_0001"
    if i <= 12:
        return "char_0002"
    if i <= 16:
        return "char_0003"
    return "char_0004"


def _beat(i: int) -> str:
    return _PLOT_BEATS[i - 1] if i <= len(_PLOT_BEATS) else f"事件{i}：剧情继续推进。"


def _event(i: int) -> CanonicalEvent:
    return CanonicalEvent(
        event_id=f"evt_{i:06d}",
        narrative_order=i,
        participants=(_char_for_event(i),),
        locations=(),
        summary_zh=_beat(i),
        candidate_event_refs=(_cand("evt", i),),
        temporal_mode="normal",
        evidence_refs=(
            EvidenceRef(
                paragraph_id=f"para_{i:05d}",
                role="primary",
                strength="explicit",
                excerpt=_beat(i),
            ),
        ),
        first_source_order=str(i),
    )


def _fact(i: int, subject: str) -> CanonicalFact:
    return CanonicalFact(
        fact_id=f"fact_{i:06d}",
        fact_type="identity",
        statement_zh=f"关于{subject}的第{i}条关键事实。",
        subject_refs=(subject,),
        object_refs=(),
        candidate_fact_refs=(_cand("fact", i),),
        evidence_refs=(
            EvidenceRef(
                paragraph_id=f"para_{100 + i:05d}",
                role="primary",
                strength="explicit",
                excerpt=f"事实{i}的原文摘录。",
            ),
        ),
        first_source_order=str(i),
        continuity_relevant=True,
    )


def _relationship(i: int, source: str, target: str) -> CanonicalRelationship:
    return CanonicalRelationship(
        relationship_id=f"rel_{i:06d}",
        source_entity_ref=source,
        target_entity_ref=target,
        direction="directed",
        relationship_type_zh="亲属与利益关系",
        candidate_relationship_refs=(_cand("rel", i),),
        state_history=(),
        first_source_order=str(i),
    )


def _conflict(i: int, fact_ids) -> StoryConflict:
    return StoryConflict(
        conflict_id=f"conf_{i:06d}",
        conflict_kind="fact_conflict",
        fact_ids=tuple(fact_ids),
        relationship_ids=(),
        candidate_refs=(_cand("fact", 100 + i),),
        decision_refs=(),
        evidence_refs=(
            EvidenceRef(
                paragraph_id=f"para_{200 + i:05d}",
                role="primary",
                strength="explicit",
                excerpt=f"冲突{i}的原文摘录。",
            ),
        ),
        status="unresolved",
    )


def _build_snapshot(*, num_events: int = 20) -> StoryAnalysisInputSnapshot:
    """A deterministic in-memory A5 snapshot (no store / CURRENT involved).

    ``num_events`` controls the window count under the frozen planning policy
    (target 12 / context 4): 1..12 -> 1 window, 13..24 -> 2 windows, 25..36 -> 3
    windows. The evidence facts / relationships / conflicts are attached to
    characters so the two windows have distinct local evidence universes (a
    fact / relationship / conflict that exists in the global snapshot but not in
    a window's own universe is a *foreign* ref for that window).
    """
    events = [_event(i) for i in range(1, num_events + 1)]
    facts = [
        _fact(1, "char_0001"),  # window 1 only (char_0001)
        _fact(2, "char_0004"),  # window 2 only (char_0004)
        _fact(3, "char_0003"),  # both (char_0003)
    ]
    relationships = [
        _relationship(1, "char_0001", "char_0002"),  # both
        _relationship(2, "char_0003", "char_0004"),  # both
        _relationship(3, "char_0004", "char_0004"),  # window 2 only (char_0004)
    ]
    conflicts = [
        _conflict(1, ("fact_000001",)),  # window 1 only
        _conflict(2, ("fact_000002",)),  # window 2 only
    ]
    return StoryAnalysisInputSnapshot(
        consolidation_manifest=None,
        consolidation_manifest_ref=ArtifactRef(
            artifact_type="consolidation_manifest",
            artifact_id="am",
            revision=1,
            content_hash="a" * 64,
        ),
        a5_validation_report_ref=ArtifactRef(
            artifact_type="a5_validation_report",
            artifact_id="vr",
            revision=1,
            content_hash="b" * 64,
        ),
        a5_current_pointer_ref=ArtifactRef(
            artifact_type="a5_current_pointer",
            artifact_id="cp",
            revision=1,
            content_hash="c" * 64,
        ),
        entity_map=None,
        canonical_character_registry=SimpleNamespace(entities=()),
        canonical_location_registry=SimpleNamespace(entities=()),
        unresolved_entity_set=SimpleNamespace(entities=()),
        canonical_fact_set=SimpleNamespace(facts=facts, state_transitions=()),
        canonical_event_set=SimpleNamespace(events=events),
        canonical_relationship_set=SimpleNamespace(relationships=relationships),
        story_conflict_set=SimpleNamespace(conflicts=conflicts),
    )


def _profile():
    return load_story_analysis_profile(
        REPO_ROOT / "profiles" / "global_story_analysis_v1.yaml"
    )


def _semantic_profile():
    return load_window_semantic_profile()


def _prompts() -> PromptRegistry:
    return PromptRegistry(DEFAULT_PROMPT_BASE_DIR)


def _plan(snapshot: StoryAnalysisInputSnapshot):
    return build_story_analysis_plan(
        snapshot,
        _profile().planning_policy,
        planning_policy_ids=planning_policy_ids_from_profile(_profile()),
    )


def _resolve(snapshot, plan, client, *, profile=None, semantic_profile=None, prompts=None):
    return resolve_plot_window_analysis(
        snapshot,
        plan,
        profile or _profile(),
        semantic_profile or _semantic_profile(),
        client,
        prompts=prompts or _prompts(),
    )


def _prep(snapshot, plan, *, profile=None, semantic_profile=None, prompts=None):
    return build_plot_window_semantic_preparation(
        snapshot,
        plan,
        profile or _profile(),
        semantic_profile or _semantic_profile(),
        prompts=prompts or _prompts(),
    )


# ---------------------------------------------------------------------------
# Deterministic fake provider (mirrors the A6C offline pattern).
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
    """Deterministic fake provider for A6D offline tests.

    Scripted responses (popped in order):
      * a ``dict`` -> local authoritative schema validation; a valid payload
        yields a successful result with provenance matching the request;
      * a ``(dict, LLMInvocationProvenance)`` tuple -> schema validation then a
        successful result with the EXACT supplied provenance;
      * an ``Exception`` -> raised from ``generate_structured``.

    Records ``call_count`` and the exact request identity per call.
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
        validate_against_output_schema(parsed, output_schema)
        return StructuredGenerationResult(
            parsed_json=parsed, provenance=provenance, attempts=1
        )


# ---------------------------------------------------------------------------
# Payload builders (schema-valid by construction).
# ---------------------------------------------------------------------------


def _interp(
    text_zh="窗口内剧情解读。",
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


def _window_payload(packet, **overrides):
    """A schema-valid ``PlotWindowAnalysis`` provider payload for ``packet``.

    By construction the default payload has empty supporting refs (valid, and
    inside the window's evidence universe); tests override specific fields to
    inject foreign refs / wrong window ids / invalid evidence modes.
    """
    payload = {
        "window_id": packet.window_id,
        "interpretation": _interp(),
        "candidate_turning_points": [],
        "candidate_reveals": [],
        "arc_continuation_markers": [],
    }
    payload.update(overrides)
    return payload


def _window_analysis(window) -> PlotWindowAnalysis:
    """A typed ``PlotWindowAnalysis`` matching a planned window (Python-owned
    fields copied from the plan) -- used by the coverage invariant tests."""
    return PlotWindowAnalysis(
        window_id=window.window_id,
        window_ordinal=window.window_ordinal,
        owned_event_refs=window.owned_event_ids,
        context_event_refs=window.context_event_ids,
        interpretation=EvidenceBackedInterpretation(
            text_zh="窗口分析", evidence_mode="inferred"
        ),
    )


# ---------------------------------------------------------------------------
# Asset-loading error boundary: missing/invalid schema raises the INTENDED
# exception (StoryIntegrityError), not a NameError from an unbound name.
# ---------------------------------------------------------------------------


def test_schema_load_missing_file_raises_integrity_error(monkeypatch: pytest.MonkeyPatch) -> None:
    bad = REPO_ROOT / "no-such-a6d-schema.json"
    monkeypatch.setattr(a6d, "A6D_WINDOW_OUTPUT_SCHEMA_PATH", bad)
    with pytest.raises(StoryIntegrityError):
        load_window_output_schema()


def test_schema_load_non_object_json_raises_integrity_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bad = tmp_path / "a6d-schema-nonobject.json"
    bad.write_text("[1, 2, 3]", encoding="utf-8")
    monkeypatch.setattr(a6d, "A6D_WINDOW_OUTPUT_SCHEMA_PATH", bad)
    with pytest.raises(StoryIntegrityError):
        load_window_output_schema()


def test_schema_load_invalid_json_raises_integrity_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bad = tmp_path / "a6d-schema-bad.json"
    bad.write_text("{not valid json", encoding="utf-8")
    monkeypatch.setattr(a6d, "A6D_WINDOW_OUTPUT_SCHEMA_PATH", bad)
    with pytest.raises(StoryIntegrityError):
        load_window_output_schema()


def test_schema_load_valid_object_succeeds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    good = tmp_path / "a6d-schema-good.json"
    real = (paths.SCHEMAS_DIR / "a6-plot-window-analysis-output.schema.json").read_text(
        encoding="utf-8"
    )
    good.write_text(real, encoding="utf-8")
    monkeypatch.setattr(a6d, "A6D_WINDOW_OUTPUT_SCHEMA_PATH", good)
    schema = load_window_output_schema()
    assert schema.schema_id == A6D_WINDOW_OUTPUT_SCHEMA_ID


def test_semantic_profile_and_schema_asset_pair_loads() -> None:
    semantic_profile, output_schema = load_window_semantic_assets()
    assert semantic_profile.profile_id == A6D_SEMANTIC_PROFILE_ID
    assert output_schema.schema_id == A6D_WINDOW_OUTPUT_SCHEMA_ID
    assert output_schema.schema_version == A6D_WINDOW_OUTPUT_SCHEMA_VERSION


def test_frozen_a6d_identity_constants() -> None:
    # The frozen A6D prompt / output-schema / semantic-profile identity.
    assert A6D_WINDOW_PROMPT_ID == "a6.plot-window-analysis"
    assert A6D_WINDOW_PROMPT_VERSION == 1
    assert A6D_WINDOW_OUTPUT_SCHEMA_ID == "a6-plot-window-analysis-output"
    assert A6D_WINDOW_OUTPUT_SCHEMA_VERSION == 1
    assert A6D_SEMANTIC_PROFILE_ID == "story-analysis-llm-v1"
    assert A6D_MAX_GENERATION_ROUNDS == 2
    # The loaded assets carry the exact frozen identities.
    assert load_window_semantic_profile().profile_id == A6D_SEMANTIC_PROFILE_ID
    schema = load_window_output_schema()
    assert schema.schema_id == A6D_WINDOW_OUTPUT_SCHEMA_ID
    assert schema.schema_version == A6D_WINDOW_OUTPUT_SCHEMA_VERSION
    # The prompt is loadable at the exact frozen id/version.
    prompt_spec = _prompts().load(
        A6D_WINDOW_PROMPT_ID, version=A6D_WINDOW_PROMPT_VERSION
    )
    assert prompt_spec.prompt_id == A6D_WINDOW_PROMPT_ID
    assert prompt_spec.version == A6D_WINDOW_PROMPT_VERSION


# ---------------------------------------------------------------------------
# Robustness: snapshot / plan exact A5 manifest identity consistency.
# ---------------------------------------------------------------------------


def test_mismatched_snapshot_plan_manifest_ref_fails_closed() -> None:
    """A snapshot and plan that name different exact A5 ConsolidationManifests
    must fail closed before packet construction / provider execution, even when
    their canonical event IDs are otherwise compatible.
    """
    snap = _build_snapshot()
    plan = _plan(snap)
    # Sanity: a consistent snapshot/plan pair prepares fine.
    assert _prep(snap, plan).window_ids
    # Otherwise-compatible event IDs, but a different exact A5 upstream ref.
    base = plan.consolidation_manifest_ref
    mismatched_ref = ArtifactRef(
        artifact_type=base.artifact_type,
        artifact_id=base.artifact_id,
        revision=base.revision,
        content_hash="f" * 64,
    )
    assert mismatched_ref != base
    mismatched_snapshot = dataclasses.replace(
        snap, consolidation_manifest_ref=mismatched_ref
    )
    # The preparation path fails closed before packet construction.
    with pytest.raises(StoryAnalysisSemanticError, match="consolidation manifest"):
        _prep(mismatched_snapshot, plan)
    # The full path (preparation + execution) fails closed with zero provider
    # calls: the mismatch is detected before any window is sent.
    client = FakeLLMClient([])
    with pytest.raises(StoryAnalysisSemanticError, match="consolidation manifest"):
        _resolve(mismatched_snapshot, plan, client)
    assert client.call_count == 0


# ---------------------------------------------------------------------------
# A. Deterministic same-input same-request
# ---------------------------------------------------------------------------


def test_a_deterministic_same_input_same_request() -> None:
    snap1 = _build_snapshot()
    snap2 = _build_snapshot()
    plan_a = _plan(snap1)
    plan_b = _plan(snap2)  # rebuild the identical plan
    prep_a = _prep(snap1, plan_a)
    prep_b = _prep(snap2, plan_b)
    assert len(prep_a.window_requests) == 2
    # Same input -> byte-identical requests and stable request identities.
    assert prep_a.window_request_hashes == prep_b.window_request_hashes
    assert prep_a.window_request_identity_hashes == prep_b.window_request_identity_hashes
    assert prep_a.plan.plan_hash == prep_b.plan.plan_hash
    for ra, rb in zip(prep_a.window_requests, prep_b.window_requests):
        assert ra.rendered_prompt.user_text == rb.rendered_prompt.user_text
        assert ra.rendered_prompt.system_text == rb.rendered_prompt.system_text
        assert (
            ra.rendered_prompt.rendered_prompt_hash
            == rb.rendered_prompt.rendered_prompt_hash
        )


# ---------------------------------------------------------------------------
# B. Different input identity changes request hash
# ---------------------------------------------------------------------------


def test_b_different_input_identity_changes_request_hash() -> None:
    # Two snapshots with different canonical event content -> different
    # window packets -> different rendered request hashes and identities.
    snap1 = _build_snapshot(num_events=20)
    snap2 = _build_snapshot(num_events=24)  # same window count, different content
    plan1 = _plan(snap1)
    plan2 = _plan(snap2)
    prep1 = _prep(snap1, plan1)
    prep2 = _prep(snap2, plan2)
    # Same positional window ids, but different packet content.
    assert prep1.window_ids == prep2.window_ids
    assert (
        prep1.window_request_identity_hashes != prep2.window_request_identity_hashes
    )
    assert prep1.window_request_hashes != prep2.window_request_hashes
    assert plan1.plan_hash != plan2.plan_hash


# ---------------------------------------------------------------------------
# C. Stable identity contains no runtime/provider fields
# ---------------------------------------------------------------------------


def test_c_stable_identity_has_no_runtime_or_provider_fields() -> None:
    snap = _build_snapshot()
    plan = _plan(snap)
    prep = _prep(snap, plan)
    # Recompute each stable request identity from ONLY the semantic fields and
    # verify it matches exactly (proving no hidden runtime/provider input).
    for packet, request, ident in zip(
        prep.windows, prep.window_requests, prep.window_request_identity_hashes
    ):
        recomputed = window_request_identity_hash(
            consolidation_manifest_ref=plan.consolidation_manifest_ref,
            plan_hash=plan.plan_hash,
            profile=prep.profile,
            semantic_profile=prep.semantic_profile,
            prompt_identity=prep.prompt_identity,
            output_schema_identity=prep.output_schema_identity,
            window_id=packet.window_id,
            window_ordinal=packet.window_ordinal,
            owned_event_ids=packet.owned_event_ids,
            context_event_ids=packet.context_event_ids,
            packet_hash=content_hash(packet.to_dict()),
            request_hash=request.request_hash,
        )
        assert recomputed == ident
    for ident in prep.window_request_identity_hashes:
        assert len(ident) == 64
        assert all(c in "0123456789abcdef" for c in ident)


def test_r_request_hash_stable_across_backend_fields() -> None:
    snap = _build_snapshot()
    plan = _plan(snap)
    packets = build_window_packets(plan.windows, snap)
    payloads_a = [_window_payload(p) for p in packets]
    payloads_b = [_window_payload(p) for p in packets]
    result_a = _resolve(
        snap,
        plan,
        FakeLLMClient(payloads_a, provider_family="qwen", request_model="qwen3-27b"),
    )
    result_b = _resolve(
        snap,
        plan,
        FakeLLMClient(payloads_b, provider_family="llama", request_model="llama-405b"),
    )
    assert (
        result_a.preparation.window_request_hashes
        == result_b.preparation.window_request_hashes
    )
    assert (
        result_a.preparation.window_request_identity_hashes
        == result_b.preparation.window_request_identity_hashes
    )
    assert result_a.preparation.window_request_identity_hashes  # non-empty


# ---------------------------------------------------------------------------
# D. Invalid provenance or request hash (fail closed, no retry)
# ---------------------------------------------------------------------------


class ProvenanceMismatchClient(FakeLLMClient):
    """A client whose successful result carries a provenance with one field
    mismatched. If A6D (incorrectly) retried on a provenance mismatch, the call
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
def test_d_provenance_or_request_hash_mismatch_fails_closed_no_retry(field: str) -> None:
    snap = _build_snapshot()
    plan = _plan(snap)
    packets = build_window_packets(plan.windows, snap)
    client = ProvenanceMismatchClient(_window_payload(packets[0]), field=field)
    with pytest.raises(StoryAnalysisProvenanceError, match="provenance field mismatch"):
        _resolve(snap, plan, client)
    # FAIL CLOSED: exactly one provider call, no semantic retry.
    assert client.call_count == 1
    assert client.execution_options == [None]


# ---------------------------------------------------------------------------
# E. Window packet budget overflow (no truncation)
# ---------------------------------------------------------------------------


def test_e_packet_budget_overflow_no_truncation() -> None:
    snap = _build_snapshot()
    plan = _plan(snap)
    tiny_policy = dataclasses.replace(
        _profile().planning_policy, plot_window_packet_max_estimated_tokens=1
    )
    tiny_profile = dataclasses.replace(_profile(), planning_policy=tiny_policy)
    packets = build_window_packets(plan.windows, snap)
    assert packets[0].estimated_tokens() > 1
    with pytest.raises(StoryAnalysisSemanticError, match="refusing to truncate"):
        _prep(snap, plan, profile=tiny_profile)


# ---------------------------------------------------------------------------
# F. Correct complete PlotWindowAnalysis set with valid refs
# ---------------------------------------------------------------------------


def test_f_complete_window_analysis_set_with_valid_refs() -> None:
    snap = _build_snapshot()
    plan = _plan(snap)
    packets = build_window_packets(plan.windows, snap)
    payloads = [_window_payload(p) for p in packets]
    client = FakeLLMClient(payloads)
    result = _resolve(snap, plan, client)
    # One provider call per canonical planned window (valid on round 1).
    assert client.call_count == len(plan.windows)
    # Complete set: every planned window, exactly once, in canonical order.
    planned_ids = [w.window_id for w in plan.windows]
    assert [a.window_id for a in result.analyses] == planned_ids
    assert len(result.analyses) == len(plan.windows)
    assert all(rounds == 1 for _wid, rounds in result.window_rounds)
    # Typed analyses preserve evidence modes and reference only the window's own
    # local evidence universe.
    for analysis, packet in zip(result.analyses, packets):
        assert analysis.interpretation.evidence_mode == "inferred"
        assert validate_plot_window_evidence(analysis, packet) is None
        assert isinstance(analysis, PlotWindowAnalysis)


# ---------------------------------------------------------------------------
# Owned/context boundary correctness (test category 4)
# ---------------------------------------------------------------------------


def test_ownership_context_boundary() -> None:
    snap = _build_snapshot()
    plan = _plan(snap)
    w1, w2 = plan.windows
    # Owned and context are disjoint within a window.
    assert set(w1.owned_event_ids).isdisjoint(w1.context_event_ids)
    assert set(w2.owned_event_ids).isdisjoint(w2.context_event_ids)
    # The two windows share a real context-overlap boundary: window 1's context
    # events are window 2's owned events (and vice-versa on the other side).
    assert set(w1.context_event_ids) <= set(w2.owned_event_ids)
    assert set(w2.context_event_ids) <= set(w1.owned_event_ids)
    # Complete canonical-event coverage: union of all owned = all events.
    all_owned = set(w1.owned_event_ids) | set(w2.owned_event_ids)
    assert all_owned == {e.event_id for e in snap.events}
    # The typed analyses carry exactly the plan's Python-owned membership.
    packets = build_window_packets(plan.windows, snap)
    payloads = [_window_payload(p) for p in packets]
    result = _resolve(snap, plan, FakeLLMClient(payloads))
    a1, a2 = result.analyses
    assert a1.owned_event_refs == w1.owned_event_ids
    assert a1.context_event_refs == w1.context_event_ids
    assert a2.owned_event_refs == w2.owned_event_ids
    assert a2.context_event_refs == w2.context_event_ids


# ---------------------------------------------------------------------------
# G. Wrong window_id (semantic invalid, exhausted)
# ---------------------------------------------------------------------------


def test_g_wrong_window_id_rejected() -> None:
    snap = _build_snapshot()
    plan = _plan(snap)
    w1, w2 = plan.windows
    wrong = _window_payload(w1, window_id=w2.window_id)
    client = FakeLLMClient([wrong, wrong])
    with pytest.raises(StoryAnalysisWindowSemanticGenerationError) as exc_info:
        _resolve(snap, plan, client)
    err = exc_info.value
    assert err.window_id == w1.window_id
    assert err.rounds_attempted == 2
    assert "wrong window_id" in " ".join(err.last_failure_details)
    # Both rounds consumed for the first window; the second was never reached.
    assert client.call_count == 2
    assert client.execution_options == [
        None,
        GenerationExecutionOptions(prompt_context_reuse="disabled"),
    ]


# ---------------------------------------------------------------------------
# H. Foreign supporting refs (exists in global snapshot, not this window)
# ---------------------------------------------------------------------------


def test_h_foreign_ref_rejected() -> None:
    snap = _build_snapshot()
    plan = _plan(snap)
    packets = build_window_packets(plan.windows, snap)
    w1_packet, _w2_packet = packets
    # Pick refs that exist in the global A5 snapshot but are NOT in window 1's
    # own local evidence universe (owned + context events and relevant
    # facts/relationships/conflicts): a fact, an event, a relationship, and a
    # conflict each in window 2 only.
    universe_events = set(w1_packet.owned_event_ids) | set(w1_packet.context_event_ids)
    universe_facts = {f.fact_id for f in w1_packet.relevant_facts}
    universe_rels = {r.relationship_id for r in w1_packet.relevant_relationships}
    universe_confs = {c.conflict_id for c in w1_packet.relevant_conflicts}

    # window-2-only refs (verified against window 1's universe):
    foreign_fact = "fact_000002"
    foreign_event = "evt_000020"
    foreign_rel = "rel_000003"
    foreign_conf = "conf_000002"
    assert foreign_fact not in universe_facts
    assert foreign_event not in universe_events
    assert foreign_rel not in universe_rels
    assert foreign_conf not in universe_confs

    cases = {
        "fact": dict(supporting_fact_refs=(foreign_fact,)),
        "evt": dict(supporting_event_refs=(foreign_event,)),
        "rel": dict(supporting_relationship_refs=(foreign_rel,)),
        "conf": dict(supporting_conflict_refs=(foreign_conf,)),
    }
    for label, interp_overrides in cases.items():
        payload = _window_payload(
            w1_packet,
            interpretation=_interp(
                "窗口解读", "explicit", **interp_overrides
            ),
        )
        # window 1 foreign on both rounds -> exhausted (window 2 never reached).
        client = FakeLLMClient([payload, payload])
        with pytest.raises(StoryAnalysisWindowSemanticGenerationError) as exc_info:
            _resolve(snap, plan, client)
        err = exc_info.value
        assert err.window_id == w1_packet.window_id
        assert err.rounds_attempted == 2
        assert client.call_count == 2
        # The bounded failure detail names the offending namespace ref.
        assert label in " ".join(err.last_failure_details)


# ---------------------------------------------------------------------------
# Legitimate context-event supporting refs (test category 8)
# ---------------------------------------------------------------------------


def test_context_event_ref_is_legitimate() -> None:
    snap = _build_snapshot()
    plan = _plan(snap)
    packets = build_window_packets(plan.windows, snap)
    w1_packet, _w2_packet = packets
    # A window 1 context event is owned by window 2 -- it is in window 1's
    # local evidence universe (context events are legitimate supporting refs).
    ctx_event = w1_packet.context_event_ids[0]
    assert ctx_event in set(w1_packet.context_event_ids)
    assert ctx_event in set(_w2_packet.owned_event_ids)
    # A context-event supporting ref (explicit) is valid, not a foreign ref.
    payload = _window_payload(
        w1_packet,
        interpretation=_interp("窗口解读", "explicit", supporting_event_refs=(ctx_event,)),
    )
    result = _resolve(snap, plan, FakeLLMClient([payload, _window_payload(_w2_packet)]))
    a1 = result.analyses[0]
    assert ctx_event in a1.interpretation.supporting_event_refs
    assert a1.interpretation.evidence_mode == "explicit"
    assert validate_plot_window_evidence(a1, w1_packet) is None


# ---------------------------------------------------------------------------
# Python-owned ordinal / ownership integrity (test category 9)
# ---------------------------------------------------------------------------


def test_python_owned_fields_match_plan() -> None:
    snap = _build_snapshot()
    plan = _plan(snap)
    packets = build_window_packets(plan.windows, snap)
    # Provider payloads never include window_ordinal / owned / context refs (the
    # output schema has no such fields); A6D copies them from the authoritative
    # A6B plan.
    payloads = [_window_payload(p) for p in packets]
    result = _resolve(snap, plan, FakeLLMClient(payloads))
    for analysis, window in zip(result.analyses, plan.windows):
        assert analysis.window_ordinal == window.window_ordinal
        assert analysis.owned_event_refs == window.owned_event_ids
        assert analysis.context_event_refs == window.context_event_ids
    # The provider payload cannot influence these Python-owned fields: an
    # analysis's ordinal is exactly the plan's ordinal, not a provider value.
    assert [a.window_ordinal for a in result.analyses] == [
        w.window_ordinal for w in plan.windows
    ]


# ---------------------------------------------------------------------------
# I / J / K. Coverage invariants (missing / duplicate / extra)
# ---------------------------------------------------------------------------


def test_i_missing_window_result_rejected() -> None:
    snap = _build_snapshot()
    plan = _plan(snap)
    w1, w2 = plan.windows
    with pytest.raises(
        StoryAnalysisSemanticError,
        match="window result count 1 does not equal planned window count 2",
    ):
        validate_plot_window_coverage([_window_analysis(w1)], [w1, w2])


def test_j_duplicate_window_result_rejected() -> None:
    snap = _build_snapshot()
    plan = _plan(snap)
    w1, w2 = plan.windows
    # Two analyses both for window 1: position 1 does not match window 2.
    with pytest.raises(StoryAnalysisSemanticError, match="window id mismatch"):
        validate_plot_window_coverage(
            [_window_analysis(w1), _window_analysis(w1)], [w1, w2]
        )


def test_j_extra_window_result_rejected() -> None:
    snap = _build_snapshot()
    plan = _plan(snap)
    w1, w2 = plan.windows
    with pytest.raises(
        StoryAnalysisSemanticError,
        match="window result count 3 does not equal planned window count 2",
    ):
        validate_plot_window_coverage(
            [_window_analysis(w1), _window_analysis(w2), _window_analysis(w2)],
            [w1, w2],
        )


# ---------------------------------------------------------------------------
# Stable canonical output order (test category 10)
# ---------------------------------------------------------------------------


def test_stable_canonical_output_order() -> None:
    # 3 windows under the frozen planning policy (25 events).
    snap = _build_snapshot(num_events=25)
    plan = _plan(snap)
    assert len(plan.windows) == 3
    packets = build_window_packets(plan.windows, snap)
    payloads = [_window_payload(p) for p in packets]
    result = _resolve(snap, plan, FakeLLMClient(payloads))
    # The complete result set is in the exact A6B canonical planned order.
    assert [a.window_id for a in result.analyses] == [
        w.window_id for w in plan.windows
    ]
    assert [a.window_ordinal for a in result.analyses] == [
        w.window_ordinal for w in plan.windows
    ]
    assert [a.window_ordinal for a in result.analyses] == [1, 2, 3]
    # window_rounds is aligned to the same canonical order.
    assert [wid for wid, _r in result.window_rounds] == [
        w.window_id for w in plan.windows
    ]


# ---------------------------------------------------------------------------
# Valid explicit / inferred interpretations (test category 11)
# ---------------------------------------------------------------------------


def test_valid_explicit_and_inferred_interpretations() -> None:
    snap = _build_snapshot()
    plan = _plan(snap)
    packets = build_window_packets(plan.windows, snap)
    w1_packet, w2_packet = packets
    # explicit: a window 1 owned event + a window 1 local fact (both in-universe).
    owned_event = w1_packet.owned_event_ids[0]
    local_fact = w1_packet.relevant_facts[0].fact_id
    payload1 = _window_payload(
        w1_packet,
        interpretation=_interp(
            "明确引用窗口内事件与事实。",
            "explicit",
            supporting_event_refs=(owned_event,),
            supporting_fact_refs=(local_fact,),
        ),
        candidate_reveals=[
            _interp("推断的揭示。", "inferred"),
        ],
        arc_continuation_markers=[
            _interp("弧线延续。", "inferred"),
        ],
    )
    payload2 = _window_payload(
        w2_packet,
        interpretation=_interp("推断窗口走向。", "inferred"),
    )
    result = _resolve(snap, plan, FakeLLMClient([payload1, payload2]))
    a1 = result.analyses[0]
    assert a1.interpretation.evidence_mode == "explicit"
    assert owned_event in a1.interpretation.supporting_event_refs
    assert local_fact in a1.interpretation.supporting_fact_refs
    assert len(a1.candidate_reveals) == 1
    assert a1.candidate_reveals[0].evidence_mode == "inferred"
    assert len(a1.arc_continuation_markers) == 1
    assert a1.arc_continuation_markers[0].evidence_mode == "inferred"
    assert validate_plot_window_evidence(a1, w1_packet) is None
    assert result.analyses[1].interpretation.evidence_mode == "inferred"


# ---------------------------------------------------------------------------
# Empty candidate arrays (test category 12)
# ---------------------------------------------------------------------------


def test_empty_candidate_arrays() -> None:
    snap = _build_snapshot()
    plan = _plan(snap)
    packets = build_window_packets(plan.windows, snap)
    result = _resolve(snap, plan, FakeLLMClient([_window_payload(p) for p in packets]))
    for analysis in result.analyses:
        assert analysis.candidate_turning_points == ()
        assert analysis.candidate_reveals == ()
        assert analysis.arc_continuation_markers == ()


# ---------------------------------------------------------------------------
# Malformed provider JSON/schema (technical, A-I3-owned, propagated)
# ---------------------------------------------------------------------------


def test_k_malformed_schema_is_technical_error() -> None:
    snap = _build_snapshot()
    plan = _plan(snap)
    packets = build_window_packets(plan.windows, snap)
    w1_packet, w2_packet = packets
    bad = _window_payload(
        w1_packet,
        interpretation=_interp("角色", "invalid_mode"),  # violates the schema enum
    )
    client = FakeLLMClient([bad, _window_payload(w2_packet)])
    with pytest.raises(LLMStructuredOutputError) as exc_info:
        _resolve(snap, plan, client)
    assert client.call_count == 1
    # A technical LLM failure, not an A6D semantic error.
    assert isinstance(exc_info.value, LLMError)
    assert not isinstance(exc_info.value, StoryAnalysisSemanticError)


# ---------------------------------------------------------------------------
# L. Semantic regeneration success on round 2
# ---------------------------------------------------------------------------


def test_l_regeneration_success_on_round_2() -> None:
    snap = _build_snapshot()
    plan = _plan(snap)
    packets = build_window_packets(plan.windows, snap)
    w1_packet, w2_packet = packets
    # Round 1 for window 1: schema-valid but a foreign fact ref (semantic
    # invalid); round 2: valid. Window 2: valid on round 1.
    bad = _window_payload(
        w1_packet,
        interpretation=_interp("窗口解读", "explicit", supporting_fact_refs=("fact_000999",)),
    )
    client = FakeLLMClient([bad, _window_payload(w1_packet), _window_payload(w2_packet)])
    result = _resolve(snap, plan, client)
    assert client.call_count == 3
    assert dict(result.window_rounds) == {
        w1_packet.window_id: 2,
        w2_packet.window_id: 1,
    }
    # Round 2 disables prompt-context reuse (A5 semantic-retry pattern).
    assert client.execution_options == [
        None,
        GenerationExecutionOptions(prompt_context_reuse="disabled"),
        None,
    ]
    assert [a.window_id for a in result.analyses] == [
        w1_packet.window_id,
        w2_packet.window_id,
    ]
    assert client.request_hashes[0] == client.request_hashes[1]  # same request both rounds


# ---------------------------------------------------------------------------
# M. Semantic regeneration exhausted at round 2 (fail closed)
# ---------------------------------------------------------------------------


def test_m_regeneration_exhausted_at_round_2() -> None:
    snap = _build_snapshot()
    plan = _plan(snap)
    packets = build_window_packets(plan.windows, snap)
    w1_packet, _w2_packet = packets
    bad1 = _window_payload(
        w1_packet,
        interpretation=_interp("窗口解读", "explicit", supporting_fact_refs=("fact_000999",)),
    )
    bad2 = _window_payload(
        w1_packet,
        interpretation=_interp("窗口解读", "explicit", supporting_event_refs=("evt_000099",)),
    )
    client = FakeLLMClient([bad1, bad2])
    with pytest.raises(StoryAnalysisWindowSemanticGenerationError) as exc_info:
        _resolve(snap, plan, client)
    err = exc_info.value
    assert err.window_id == w1_packet.window_id
    assert err.rounds_attempted == 2
    assert err.request_hash == client.request_hashes[0]
    assert len(err.last_failure_details) == 2
    assert client.call_count == 2


# ---------------------------------------------------------------------------
# N. Technical LLM failure (propagated, A-I3-owned, no semantic retry)
# ---------------------------------------------------------------------------


def test_n_technical_retry_remains_ai3_owned() -> None:
    snap = _build_snapshot()
    plan = _plan(snap)
    client = FakeLLMClient(
        [LLMTransportError("connection reset (technical retry budget exhausted)")]
    )
    with pytest.raises(LLMError) as exc_info:
        _resolve(snap, plan, client)
    assert client.call_count == 1
    assert not isinstance(exc_info.value, StoryAnalysisSemanticError)
    assert client.execution_options == [None]


# ---------------------------------------------------------------------------
# No truncation or sampling (test category 18)
# ---------------------------------------------------------------------------


def test_p_no_silent_truncation() -> None:
    snap = _build_snapshot()
    plan = _plan(snap)
    prep = _prep(snap, plan)
    for packet, request in zip(prep.windows, prep.window_requests):
        full_packet = packet.canonical_bytes().decode("utf-8")
        # The complete rendered request contains the ENTIRE evidence packet
        # (no truncation), wrapped in the prompt framing.
        assert full_packet in request.rendered_prompt.user_text
        assert request.rendered_prompt.user_text.strip() != full_packet
    # The complete rendered request is strictly larger than the packet alone.
    assert all(
        rendered > packet
        for rendered, packet in zip(
            prep.rendered_request_token_estimates, prep.packet_token_estimates
        )
    )


# ---------------------------------------------------------------------------
# No partial authoritative result (test category 19)
# ---------------------------------------------------------------------------


def test_o_no_partial_result_on_later_window_failure() -> None:
    snap = _build_snapshot()
    plan = _plan(snap)
    packets = build_window_packets(plan.windows, snap)
    w1_packet, w2_packet = packets
    # window 1 succeeds (round 1); window 2 exhausts (both rounds foreign) ->
    # A6D must raise rather than return a partial (window 1-only) result.
    client = FakeLLMClient(
        [
            _window_payload(w1_packet),  # window 1 valid
            _window_payload(
                w2_packet,
                interpretation=_interp(
                    "窗口解读", "explicit", supporting_event_refs=("evt_000099",)
                ),
            ),
            _window_payload(
                w2_packet,
                interpretation=_interp(
                    "窗口解读", "explicit", supporting_event_refs=("evt_000098",)
                ),
            ),
        ]
    )
    with pytest.raises(StoryAnalysisWindowSemanticGenerationError) as exc_info:
        _resolve(snap, plan, client)
    err = exc_info.value
    # The failure is attributed to window 2 (the later window), not window 1.
    assert err.window_id == w2_packet.window_id
    # window 1 (1 round) + window 2 (2 rounds) = 3 provider calls.
    assert client.call_count == 3


# ---------------------------------------------------------------------------
# No persistence / CURRENT mutation (test category 20)
# ---------------------------------------------------------------------------


def _real_snapshot(tmp_path, *, suffix: str = "001"):
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
    return tree, snap


def _a5_current_ref(tree):
    return tree.pointers.resolve_current(
        a5_pointer_id(PROJECT, DOCUMENT, CONSOLIDATION_PROFILE_ID)
    ).target_ref


def _store_file_count(tree) -> int:
    return len(list(tree.store.root.rglob("*.json")))


def test_q_no_a6_current_mutation(tmp_path: Path) -> None:
    tree, snap = _real_snapshot(tmp_path)
    plan = _plan(snap)
    before_a5 = _a5_current_ref(tree)
    before_files = _store_file_count(tree)
    packets = build_window_packets(plan.windows, snap)
    client = FakeLLMClient([_window_payload(p) for p in packets])
    result = _resolve(snap, plan, client)
    # A6D is in-memory only: the A5 CURRENT is unchanged and no A6 artifact is
    # persisted; the result exists only in memory.
    assert _a5_current_ref(tree) == before_a5
    assert _store_file_count(tree) == before_files
    assert len(result.analyses) == len(plan.windows)


# ---------------------------------------------------------------------------
# Robustness: frozen A6D identity is enforced (fail closed).
# ---------------------------------------------------------------------------


def test_profile_identity_mismatch_fails_closed() -> None:
    snap = _build_snapshot()
    plan = _plan(snap)
    base = _profile()
    bad_pw = dataclasses.replace(base.plot_window_analysis, prompt_id="a6.wrong-prompt")
    bad_profile = dataclasses.replace(base, plot_window_analysis=bad_pw)
    with pytest.raises(StoryAnalysisSemanticError, match="prompt must be"):
        _prep(snap, plan, profile=bad_profile)


def test_wrong_semantic_profile_id_fails_closed() -> None:
    snap = _build_snapshot()
    plan = _plan(snap)
    bad_semantic = dataclasses.replace(
        _semantic_profile(), profile_id="story-analysis-llm-v2"
    )
    with pytest.raises(StoryAnalysisSemanticError, match="semantic profile"):
        _prep(snap, plan, semantic_profile=bad_semantic)


def test_request_identity_changes_when_semantic_field_changes() -> None:
    snap = _build_snapshot()
    plan = _plan(snap)
    base_prep = _prep(snap, plan)
    other_semantic = dataclasses.replace(_semantic_profile(), temperature=0.3)
    other_prep = _prep(snap, plan, semantic_profile=other_semantic)
    assert (
        base_prep.window_request_identity_hashes
        != other_prep.window_request_identity_hashes
    )
    assert base_prep.window_request_hashes != other_prep.window_request_hashes
    assert base_prep.window_ids == other_prep.window_ids


# ---------------------------------------------------------------------------
# Real-provider smoke (bounded, gated on a live OpenAI-compatible server).
# ---------------------------------------------------------------------------


def _live_server_available() -> bool:
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
def test_real_provider_smoke() -> None:
    """One bounded real-provider smoke: exact request rendering + real JSON/
    schema output + typed parsing + provenance verification + exact local
    evidence-ref validation, end to end against a live local Qwen server.

    Uses a deterministic small in-memory sample (20 canonical events -> 2
    windows with a real context-overlap boundary).

    PASS contract (strict): the smoke passes ONLY if
      * the real provider returns valid structured output (no technical LLM
        failure from request rendering / JSON parsing / schema validation);
      * typed ``PlotWindowAnalysis`` parsing succeeds;
      * every supporting ref passes exact local-evidence-universe validation; and
      * the complete set is in canonical planned order with correct
        Python-owned fields.

    Semantic generation exhaustion (a schema-valid but semantically-invalid
    result on both bounded rounds) and any other failure must FAIL the smoke --
    it is never a silent pass.
    """
    from short_drama.llm import (
        OpenAICompatibleLLMClient,
        load_runtime_config,
    )

    snap = _build_snapshot()  # 20 events -> 2 windows with a context overlap
    plan = _plan(snap)
    assert len(plan.windows) >= 2
    profile = _profile()
    semantic_profile = load_window_semantic_profile()
    prompts = _prompts()

    runtime_config = load_runtime_config(REPO_ROOT / "profiles" / "llm_local.yaml")
    client = OpenAICompatibleLLMClient(runtime_config)

    # PASS requires a full success: a complete, valid in-memory window set.
    result = resolve_plot_window_analysis(
        snap, plan, profile, semantic_profile, client, prompts=prompts
    )

    packets = build_window_packets(plan.windows, snap)
    planned_ids = [w.window_id for w in plan.windows]
    assert [a.window_id for a in result.analyses] == planned_ids
    assert len(result.analyses) == len(planned_ids)
    for analysis, window, packet in zip(result.analyses, plan.windows, packets):
        assert isinstance(analysis, PlotWindowAnalysis)
        # Correct Python-owned fields (copied from the authoritative plan).
        assert analysis.window_ordinal == window.window_ordinal
        assert analysis.owned_event_refs == window.owned_event_ids
        assert analysis.context_event_refs == window.context_event_ids
        # Exact local evidence-ref authorization.
        assert validate_plot_window_evidence(analysis, packet) is None
    print(
        f"\n[smoke] PASS: {len(planned_ids)} windows, "
        f"rounds={dict(result.window_rounds)}"
    )
