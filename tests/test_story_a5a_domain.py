"""A5A domain/static-contract tests.

Covers the frozen A5A contract (fact / event / relationship consolidation
domain contracts + tracked assets):

  * tracked ``ConsolidationProfile`` load / round-trip / hash / no runtime
    fields / no blocking algorithm parameters / schema parity;
  * ``ConsolidationCandidateRef`` parse / round-trip / namespace + fail-closed;
  * ``IndexedFact/Event/RelationshipCandidate`` round-trips, enums,
    global-ref/namespace consistency, exact-key parsing;
  * ``ConsolidationCandidateIndex`` round-trip / empty / schema parity;
  * persisted ``Fact/Event/RelationshipSemanticDecision`` (canonical pair
    ordering, closed decision enum, method/provenance consistency);
  * ``ConsolidationDecisionSet`` round-trip / schema parity;
  * provider ``Fact/Event/RelationshipSelectorDecisionPayload`` (pair-local
    evidence selectors only; the provider never reproduces EvidenceRef fields);
  * ``CanonicalFact`` / ``StateTransition`` / ``CanonicalFactSet`` (id
    namespace, state-transition kinds);
  * ``CanonicalEvent`` / ``CanonicalEventSet`` (narrative_order, temporal mode);
  * ``CanonicalRelationship`` / ``RelationshipState`` / set (state history);
  * ``StoryConflict`` / ``StoryConflictSet`` (kinds, unresolved status);
  * ``A5SemanticIdentity`` (backend-neutral, exact asset ids, request hashes);
  * ``A5UpstreamIdentity`` / ``ConsolidationCoverageSummary`` /
    ``ConsolidationManifest`` (root aggregate);
  * JSON Schema parity for the eight persisted A5 schemas + the three provider
    selector-payload schemas (strict-mode discipline).

Deliberately does NOT implement or require: candidate collection, blocking, an
LLM call, canonical-id allocation, conflict detection, persistence/CURRENT
reuse, or a CLI (those are A5B-A5G).
"""

from __future__ import annotations

import json

import pytest
from jsonschema import Draft202012Validator

from short_drama.artifacts import ArtifactRef, content_hash
from short_drama.io import load_json
from short_drama.llm import (
    LLMPromptError,
    LLMInvocationProvenance,
    PromptRegistry,
    render_prompt,
    load_semantic_profile,
)
from short_drama.paths import PROFILES_DIR, REPO_ROOT, SCHEMAS_DIR
from short_drama.story import (
    A5SemanticIdentity,
    A5_EVIDENCE_SELECTOR_PATTERN,
    A5UpstreamIdentity,
    CanonicalEvent,
    CanonicalEventSet,
    CanonicalFact,
    CanonicalFactSet,
    CanonicalRelationship,
    CanonicalRelationshipSet,
    ConsolidationCandidateIndex,
    ConsolidationCandidateRef,
    ConsolidationCoverageSummary,
    ConsolidationDecisionSet,
    ConsolidationManifest,
    ConsolidationModelError,
    ConsolidationProfile,
    EVENT_DECISIONS,
    EVENT_ID_PATTERN,
    FACT_DECISIONS,
    FACT_ID_PATTERN,
    RELATIONSHIP_DECISIONS,
    RELATIONSHIP_ID_PATTERN,
    STATE_TRANSITION_ID_PATTERN,
    STATE_TRANSITION_KINDS,
    STORY_CONFLICT_ID_PATTERN,
    STORY_CONFLICT_KINDS,
    STORY_CONFLICT_STATUSES,
    EventSelectorDecisionItem,
    EventSelectorDecisionPayload,
    EventSemanticDecision,
    FactSelectorDecisionItem,
    FactSelectorDecisionPayload,
    FactSemanticDecision,
    IndexedEventCandidate,
    IndexedFactCandidate,
    IndexedRelationshipCandidate,
    OutputSchemaAssetIdentity,
    PromptAssetIdentity,
    RelationshipSelectorDecisionItem,
    RelationshipSelectorDecisionPayload,
    RelationshipSemanticDecision,
    RelationshipState,
    StateTransition,
    StoryConflict,
    StoryConflictSet,
    EvidenceRef,
    A3InputIdentity,
    load_consolidation_profile,
)

PROMPTS_STORY_DIR = REPO_ROOT / "prompts" / "story"
CONSOLIDATION_PROFILE_PATH = PROFILES_DIR / "consolidation_v1.yaml"
A5_LLM_PROFILE_PATH = PROFILES_DIR / "consolidation_llm_v1.yaml"

A5_FACT_PROMPT_ID = "a5.fact-consolidation"
A5_EVENT_PROMPT_ID = "a5.event-consolidation"
A5_RELATIONSHIP_PROMPT_ID = "a5.relationship-consolidation"
PROMPT_VERSION = 1
FACT_PROMPT_CONTENT_HASH = (
    "bb505266f990b018287ef841c9a99c1778c0160b9b9758fcc5b6ccc7cd256b7f"
)
EVENT_PROMPT_CONTENT_HASH = (
    "79a25a0ceb9a87ca49a78f01fc46d5cc0b78e52161a614fcfba50182169fc2c7"
)
RELATIONSHIP_PROMPT_CONTENT_HASH = (
    "8ecc19b32ceaa7535e24bad8224c9016fdd5423e0aae0b022d8e8d8c9c175dae"
)

H = "a" * 64
H2 = "b" * 64

# Canonical A5 pair refs (left < right) per domain.
FACT_LEFT = "CH003_C005:cand_fact_012"
FACT_RIGHT = "CH007_C009:cand_fact_013"
EVENT_LEFT = "CH003_C005:cand_evt_012"
EVENT_RIGHT = "CH007_C009:cand_evt_013"
REL_LEFT = "CH003_C005:cand_rel_012"
REL_RIGHT = "CH007_C009:cand_rel_013"


# ---------------------------------------------------------------------------
# Helpers / builders
# ---------------------------------------------------------------------------


def _schema(name: str) -> dict:
    return load_json(SCHEMAS_DIR / name)


def _validate(data: dict, schema: dict) -> list[str]:
    validator = Draft202012Validator(schema)
    return [
        f"{'/'.join(str(part) for part in error.absolute_path) or '<root>'}: "
        f"{error.message}"
        for error in sorted(
            validator.iter_errors(data), key=lambda e: list(e.absolute_path)
        )
    ]


def make_artifact_ref(artifact_type: str = "candidate_extraction", **overrides) -> ArtifactRef:
    values = {
        "artifact_type": artifact_type,
        "artifact_id": "cand-extraction-0001",
        "revision": 1,
        "content_hash": H,
    }
    values.update(overrides)
    return ArtifactRef(**values)


def make_evidence(**overrides) -> EvidenceRef:
    values = {
        "paragraph_id": "CH001_P0017",
        "role": "primary",
        "strength": "explicit",
        "excerpt": "林晚走进了教室。",
    }
    values.update(overrides)
    return EvidenceRef(**values)


def make_provenance(**overrides) -> LLMInvocationProvenance:
    values = {
        "provider_family": "qwen",
        "model": "qwen-max",
        "semantic_profile_id": "consolidation-llm-v1",
        "semantic_profile_hash": H,
        "prompt_id": A5_FACT_PROMPT_ID,
        "prompt_version": PROMPT_VERSION,
        "prompt_content_hash": FACT_PROMPT_CONTENT_HASH,
        "rendered_prompt_hash": H2,
        "output_schema_id": "consolidation-fact-selector-payload",
        "output_schema_version": 1,
        "output_schema_hash": H,
        "request_hash": H,
        "provider_response_id": "resp_0001",
        "finish_reason": "stop",
        "usage": {"input_tokens": 10, "output_tokens": 20},
    }
    values.update(overrides)
    return LLMInvocationProvenance(**values)


def make_fact_candidate(**overrides) -> IndexedFactCandidate:
    values = {
        "global_candidate_ref": FACT_LEFT,
        "chunk_id": "CH003_C005",
        "local_candidate_id": "cand_fact_012",
        "source_order_key": "CH003_C005:P0017",
        "fact_type": "identity",
        "statement_zh": "林晚是一名学生。",
        "subject_refs": ("char_0001",),
        "object_refs": (),
        "evidence_strength": "explicit",
        "evidence_refs": (make_evidence(),),
        "candidate_extraction_ref": make_artifact_ref(),
    }
    values.update(overrides)
    return IndexedFactCandidate(**values)


def make_event_candidate(**overrides) -> IndexedEventCandidate:
    values = {
        "global_candidate_ref": EVENT_LEFT,
        "chunk_id": "CH003_C005",
        "local_candidate_id": "cand_evt_012",
        "source_order_key": "CH003_C005:P0017",
        "summary_zh": "林晚走进了教室。",
        "participants": ("char_0001",),
        "locations": ("loc_0001",),
        "temporal_mode": "normal",
        "evidence_strength": "explicit",
        "evidence_refs": (make_evidence(),),
        "candidate_extraction_ref": make_artifact_ref(),
    }
    values.update(overrides)
    return IndexedEventCandidate(**values)


def make_relationship_candidate(**overrides) -> IndexedRelationshipCandidate:
    values = {
        "global_candidate_ref": REL_LEFT,
        "chunk_id": "CH003_C005",
        "local_candidate_id": "cand_rel_012",
        "source_order_key": "CH003_C005:P0017",
        "source_entity_ref": "char_0001",
        "target_entity_ref": "char_0002",
        "relationship_type_zh": "朋友",
        "state_zh": None,
        "direction": "directed",
        "evidence_strength": "explicit",
        "evidence_refs": (make_evidence(),),
        "candidate_extraction_ref": make_artifact_ref(),
    }
    values.update(overrides)
    return IndexedRelationshipCandidate(**values)


def make_fact_decision(**overrides) -> FactSemanticDecision:
    values = {
        "decision_id": "factd_000001",
        "left_candidate_ref": FACT_LEFT,
        "right_candidate_ref": FACT_RIGHT,
        "decision": "same_fact",
        "method": "llm",
        "reason_zh": "两条事实表述同一身份。",
        "evidence_refs": (make_evidence(),),
        "prompt_id": A5_FACT_PROMPT_ID,
        "prompt_version": PROMPT_VERSION,
        "generation_provenance": make_provenance(),
    }
    values.update(overrides)
    return FactSemanticDecision(**values)


def make_event_decision(**overrides) -> EventSemanticDecision:
    values = {
        "decision_id": "evtd_000001",
        "left_candidate_ref": EVENT_LEFT,
        "right_candidate_ref": EVENT_RIGHT,
        "decision": "same_event",
        "method": "deterministic",
        "reason_zh": "同一事件。",
        "evidence_refs": (),
        "prompt_id": None,
        "prompt_version": None,
        "generation_provenance": None,
    }
    values.update(overrides)
    return EventSemanticDecision(**values)


def make_relationship_decision(**overrides) -> RelationshipSemanticDecision:
    values = {
        "decision_id": "reld_000001",
        "left_candidate_ref": REL_LEFT,
        "right_candidate_ref": REL_RIGHT,
        "decision": "uncertain",
        "method": "manual",
        "reason_zh": "证据不足。",
        "evidence_refs": (),
        "prompt_id": None,
        "prompt_version": None,
        "generation_provenance": None,
    }
    values.update(overrides)
    return RelationshipSemanticDecision(**values)


def make_canonical_fact(**overrides) -> CanonicalFact:
    values = {
        "fact_id": "fact_000001",
        "fact_type": "identity",
        "statement_zh": "林晚是一名学生。",
        "subject_refs": ("char_0001",),
        "object_refs": (),
        "candidate_fact_refs": (FACT_LEFT,),
        "evidence_refs": (make_evidence(),),
        "first_source_order": "CH003_C005:P0017",
        "continuity_relevant": True,
    }
    values.update(overrides)
    return CanonicalFact(**values)


def make_state_transition(**overrides) -> StateTransition:
    values = {
        "transition_id": "trans_000001",
        "from_fact_id": "fact_000001",
        "to_fact_id": "fact_000002",
        "subject_refs": ("char_0001",),
        "transition_kind": "state_change",
        "source_decision_ref": "factd_000001",
        "evidence_refs": (make_evidence(),),
        "narrative_order": 1,
    }
    values.update(overrides)
    return StateTransition(**values)


def make_canonical_event(**overrides) -> CanonicalEvent:
    values = {
        "event_id": "evt_000001",
        "narrative_order": 1,
        "summary_zh": "林晚走进了教室。",
        "participants": ("char_0001",),
        "locations": ("loc_0001",),
        "temporal_mode": "normal",
        "candidate_event_refs": (EVENT_LEFT,),
        "evidence_refs": (make_evidence(),),
        "first_source_order": "CH003_C005:P0017",
    }
    values.update(overrides)
    return CanonicalEvent(**values)


def make_relationship_state(**overrides) -> RelationshipState:
    values = {
        "state_zh": "好友",
        "candidate_relationship_refs": (REL_LEFT,),
        "evidence_refs": (make_evidence(),),
        "narrative_order": 1,
    }
    values.update(overrides)
    return RelationshipState(**values)


def make_canonical_relationship(**overrides) -> CanonicalRelationship:
    values = {
        "relationship_id": "rel_000001",
        "source_entity_ref": "char_0001",
        "target_entity_ref": "char_0002",
        "direction": "directed",
        "relationship_type_zh": "朋友",
        "candidate_relationship_refs": (REL_LEFT,),
        "state_history": (make_relationship_state(),),
        "first_source_order": "CH003_C005:P0017",
    }
    values.update(overrides)
    return CanonicalRelationship(**values)


def make_story_conflict(**overrides) -> StoryConflict:
    values = {
        "conflict_id": "conf_000001",
        "conflict_kind": "fact_conflict",
        "fact_ids": ("fact_000001", "fact_000002"),
        "relationship_ids": (),
        "candidate_refs": (FACT_LEFT, FACT_RIGHT),
        "decision_refs": ("factd_000001",),
        "evidence_refs": (make_evidence(),),
        "status": "unresolved",
    }
    values.update(overrides)
    return StoryConflict(**values)


def make_a3_input(**overrides) -> A3InputIdentity:
    values = {
        "source_document_ref": make_artifact_ref("source_document"),
        "chunk_manifest_ref": make_artifact_ref("chunk_manifest"),
        "candidate_extraction_refs": (make_artifact_ref("candidate_extraction"),),
        "extraction_profile_id": "story-extraction-v1",
        "extraction_profile_hash": H,
    }
    values.update(overrides)
    return A3InputIdentity(**values)


def make_semantic_identity(**overrides) -> A5SemanticIdentity:
    values = {
        "consolidation_profile_id": "consolidation-v1",
        "consolidation_profile_hash": H,
        "fact_semantic_profile_id": "consolidation-llm-v1",
        "fact_semantic_profile_hash": H,
        "event_semantic_profile_id": "consolidation-llm-v1",
        "event_semantic_profile_hash": H,
        "relationship_semantic_profile_id": "consolidation-llm-v1",
        "relationship_semantic_profile_hash": H,
        "prompt_identities": (
            PromptAssetIdentity(
                prompt_id=A5_FACT_PROMPT_ID,
                prompt_version=PROMPT_VERSION,
                prompt_content_hash=FACT_PROMPT_CONTENT_HASH,
            ),
        ),
        "output_schema_identities": (
            OutputSchemaAssetIdentity(
                schema_id="consolidation-fact-selector-payload",
                schema_version=1,
                schema_hash=H,
            ),
        ),
        "plan_hash": H,
        "semantic_request_hashes": (H, H2),
    }
    values.update(overrides)
    return A5SemanticIdentity(**values)


def make_manifest(**overrides) -> ConsolidationManifest:
    values = {
        "schema_version": 1,
        "project_id": "proj_0001",
        "document_id": "doc_0001",
        "entity_map_ref": make_artifact_ref("entity_map"),
        "consolidation_candidate_index_ref": make_artifact_ref(
            "consolidation_candidate_index"
        ),
        "consolidation_decision_set_ref": make_artifact_ref(
            "consolidation_decision_set"
        ),
        "canonical_fact_set_ref": make_artifact_ref("canonical_fact_set"),
        "canonical_event_set_ref": make_artifact_ref("canonical_event_set"),
        "canonical_relationship_set_ref": make_artifact_ref(
            "canonical_relationship_set"
        ),
        "story_conflict_set_ref": make_artifact_ref("story_conflict_set"),
        "semantic_identity": make_semantic_identity(),
        "upstream_identity": A5UpstreamIdentity(a3_input=make_a3_input()),
        "coverage_summary": ConsolidationCoverageSummary(
            fact_candidate_count=5,
            event_candidate_count=3,
            relationship_candidate_count=2,
            canonical_fact_count=4,
            canonical_event_count=2,
            canonical_relationship_count=2,
            uncertain_decision_count=0,
            story_conflict_count=0,
        ),
    }
    values.update(overrides)
    return ConsolidationManifest(**values)


# ---------------------------------------------------------------------------
# ConsolidationCandidateRef
# ---------------------------------------------------------------------------


class TestConsolidationCandidateRef:
    def test_parse_round_trip(self) -> None:
        ref = ConsolidationCandidateRef.parse(FACT_LEFT)
        assert ref.global_candidate_ref == FACT_LEFT
        assert ref.chunk_id == "CH003_C005"
        assert ref.local_candidate_id == "cand_fact_012"
        assert ref.namespace == "fact"

    def test_from_dict_round_trip(self) -> None:
        ref = ConsolidationCandidateRef.parse(EVENT_LEFT)
        assert ConsolidationCandidateRef.from_dict(ref.to_dict()) == ref

    def test_namespace_derived_from_local_id(self) -> None:
        assert (
            ConsolidationCandidateRef.parse("CH003_C005:cand_fact_012").namespace
            == "fact"
        )
        assert ConsolidationCandidateRef.parse(EVENT_LEFT).namespace == "event"
        assert ConsolidationCandidateRef.parse(REL_LEFT).namespace == "relationship"

    def test_to_string(self) -> None:
        assert ConsolidationCandidateRef.parse(FACT_LEFT).to_string() == FACT_LEFT

    @pytest.mark.parametrize(
        "bad",
        [
            "",
            "CH003_C005:cand_char_012",
            "CH003_C005",
            "CH003_C005:cand_fact_12",  # local id too short
            "ch003_c005:cand_fact_012",
            "CH003_C005:cand_fact_012:extra",
            "CH3_C5:cand_fact_012",
        ],
    )
    def test_parse_rejects_invalid(self, bad: object) -> None:
        with pytest.raises(ConsolidationModelError):
            ConsolidationCandidateRef.parse(bad)

    def test_constructor_rejects_namespace_mismatch(self) -> None:
        with pytest.raises(ConsolidationModelError):
            ConsolidationCandidateRef(
                chunk_id="CH003_C005",
                local_candidate_id="cand_fact_012",
                global_candidate_ref=FACT_LEFT,
                namespace="event",
            )

    def test_constructor_rejects_ref_mismatch(self) -> None:
        with pytest.raises(ConsolidationModelError):
            ConsolidationCandidateRef(
                chunk_id="CH999_C999",
                local_candidate_id="cand_fact_012",
                global_candidate_ref=FACT_LEFT,
                namespace="fact",
            )


# ---------------------------------------------------------------------------
# ConsolidationProfile
# ---------------------------------------------------------------------------


class TestConsolidationProfile:
    def test_tracked_profile_loads(self) -> None:
        profile = load_consolidation_profile(CONSOLIDATION_PROFILE_PATH)
        assert profile.profile_id == "consolidation-v1"
        assert profile.working_language == "zh"
        assert profile.blocking_policy_id == "consolidation-blocking-v1"
        assert profile.fact.prompt_id == A5_FACT_PROMPT_ID
        assert profile.event.prompt_id == A5_EVENT_PROMPT_ID
        assert profile.relationship.prompt_id == A5_RELATIONSHIP_PROMPT_ID
        assert profile.fact.prompt_version == PROMPT_VERSION
        assert profile.max_generation_rounds == 2

    def test_profile_hash_is_stable(self) -> None:
        profile = load_consolidation_profile(CONSOLIDATION_PROFILE_PATH)
        assert profile.content_hash() == content_hash(profile.to_dict())
        assert len(profile.content_hash()) == 64

    def test_profile_round_trip(self) -> None:
        profile = load_consolidation_profile(CONSOLIDATION_PROFILE_PATH)
        assert ConsolidationProfile.from_dict(profile.to_dict()) == profile

    def test_profile_carries_no_runtime_fields(self) -> None:
        profile = load_consolidation_profile(CONSOLIDATION_PROFILE_PATH)
        flat = json.dumps(profile.to_dict(), ensure_ascii=False)
        for banned in (
            "endpoint",
            "timeout",
            "provider_family",
            "request_model",
            "temperature",
            "credential",
            "max_output_tokens",
        ):
            assert banned not in flat

    def test_profile_pins_no_blocking_algorithm_parameters(self) -> None:
        profile = load_consolidation_profile(CONSOLIDATION_PROFILE_PATH)
        data = profile.to_dict()
        for banned in (
            "fuzzy_similarity_threshold",
            "stopwords",
            "stopword_list",
            "source_distance_cutoff",
            "token_overlap_threshold",
            "semantic_pair_threshold",
        ):
            assert banned not in data

    def test_missing_file_fails_closed(self, tmp_path) -> None:
        with pytest.raises(ConsolidationModelError):
            load_consolidation_profile(tmp_path / "nope.yaml")

    def test_tracked_profile_passes_profile_schema(self) -> None:
        profile = load_consolidation_profile(CONSOLIDATION_PROFILE_PATH)
        assert _validate(profile.to_dict(), _schema("consolidation-profile.schema.json")) == []

    def test_profile_schema_rejects_extra_field(self) -> None:
        data = load_consolidation_profile(CONSOLIDATION_PROFILE_PATH).to_dict()
        data["endpoint"] = "http://localhost:8080"
        assert _validate(data, _schema("consolidation-profile.schema.json")) != []

    def test_profile_schema_rejects_missing_field(self) -> None:
        data = load_consolidation_profile(CONSOLIDATION_PROFILE_PATH).to_dict()
        del data["blocking_policy_id"]
        assert _validate(data, _schema("consolidation-profile.schema.json")) != []

    def test_profile_schema_rejects_wrong_schema_version(self) -> None:
        data = load_consolidation_profile(CONSOLIDATION_PROFILE_PATH).to_dict()
        data["schema_version"] = 2
        assert _validate(data, _schema("consolidation-profile.schema.json")) != []

    def test_profile_rejects_unknown_key(self) -> None:
        data = load_consolidation_profile(CONSOLIDATION_PROFILE_PATH).to_dict()
        data["extra"] = 1
        with pytest.raises(ConsolidationModelError):
            ConsolidationProfile.from_dict(data)

    def test_profile_rejects_non_positive_version(self) -> None:
        data = load_consolidation_profile(CONSOLIDATION_PROFILE_PATH).to_dict()
        data["fact"]["prompt_version"] = 0
        with pytest.raises(ConsolidationModelError):
            ConsolidationProfile.from_dict(data)


class TestTrackedConsolidationBlockingSlot:
    """The tracked A5 profile pins a versioned blocking-policy slot (no
    algorithm parameters).

    A5B binds this slot to the audited production blocking identity. A5A only
    guarantees the slot exists and is a safe storage id, and that the profile
    does NOT pre-decide any blocking algorithm parameters.
    """

    def test_blocking_policy_slot_is_a_storage_id(self) -> None:
        profile = load_consolidation_profile(CONSOLIDATION_PROFILE_PATH)
        assert profile.blocking_policy_id == "consolidation-blocking-v1"

    def test_semantic_passes_pin_distinct_prompts_and_schemas(self) -> None:
        profile = load_consolidation_profile(CONSOLIDATION_PROFILE_PATH)
        assert profile.fact.prompt_id != profile.event.prompt_id
        assert profile.event.prompt_id != profile.relationship.prompt_id
        assert profile.fact.output_schema_id == "consolidation-fact-selector-payload"
        assert profile.event.output_schema_id == "consolidation-event-selector-payload"
        assert (
            profile.relationship.output_schema_id
            == "consolidation-relationship-selector-payload"
        )


# ---------------------------------------------------------------------------
# Indexed candidates
# ---------------------------------------------------------------------------


class TestIndexedFactCandidate:
    def test_round_trip(self) -> None:
        cand = make_fact_candidate()
        assert IndexedFactCandidate.from_dict(cand.to_dict()) == cand

    def test_global_ref_chunk_mismatch_rejected(self) -> None:
        with pytest.raises(ConsolidationModelError):
            make_fact_candidate(chunk_id="CH999_C999")

    def test_bad_fact_type_rejected(self) -> None:
        with pytest.raises(ConsolidationModelError):
            make_fact_candidate(fact_type="nonsense")

    def test_bad_evidence_strength_rejected(self) -> None:
        with pytest.raises(ConsolidationModelError):
            make_fact_candidate(evidence_strength="nonsense")

    def test_requires_min_one_evidence(self) -> None:
        with pytest.raises(ConsolidationModelError):
            make_fact_candidate(evidence_refs=())

    def test_extra_key_rejected(self) -> None:
        data = make_fact_candidate().to_dict()
        data["extra"] = 1
        with pytest.raises(ConsolidationModelError):
            IndexedFactCandidate.from_dict(data)


class TestIndexedEventCandidate:
    def test_round_trip(self) -> None:
        cand = make_event_candidate()
        assert IndexedEventCandidate.from_dict(cand.to_dict()) == cand

    def test_bad_temporal_mode_rejected(self) -> None:
        with pytest.raises(ConsolidationModelError):
            make_event_candidate(temporal_mode="nonsense")


class TestIndexedRelationshipCandidate:
    def test_round_trip(self) -> None:
        cand = make_relationship_candidate()
        assert IndexedRelationshipCandidate.from_dict(cand.to_dict()) == cand

    def test_null_state_zh_allowed(self) -> None:
        assert make_relationship_candidate(state_zh=None).state_zh is None

    def test_non_empty_state_zh_allowed(self) -> None:
        assert make_relationship_candidate(state_zh="好友").state_zh == "好友"

    def test_bad_direction_rejected(self) -> None:
        with pytest.raises(ConsolidationModelError):
            make_relationship_candidate(direction="nonsense")


class TestConsolidationCandidateIndex:
    def test_round_trip(self) -> None:
        index = ConsolidationCandidateIndex(
            schema_version=1,
            facts=(make_fact_candidate(),),
            events=(make_event_candidate(),),
            relationships=(make_relationship_candidate(),),
        )
        assert ConsolidationCandidateIndex.from_dict(index.to_dict()) == index

    def test_empty_ok(self) -> None:
        index = ConsolidationCandidateIndex(schema_version=1, facts=(), events=(), relationships=())
        assert index.to_dict() == {"schema_version": 1, "facts": [], "events": [], "relationships": []}

    def test_wrong_schema_version_rejected(self) -> None:
        with pytest.raises(ConsolidationModelError):
            ConsolidationCandidateIndex(schema_version=2, facts=(), events=(), relationships=())

    def test_schema_parity(self) -> None:
        index = ConsolidationCandidateIndex(
            schema_version=1,
            facts=(make_fact_candidate(),),
            events=(make_event_candidate(),),
            relationships=(make_relationship_candidate(),),
        )
        assert _validate(index.to_dict(), _schema("consolidation-candidate-index.schema.json")) == []


# ---------------------------------------------------------------------------
# Persisted semantic decisions
# ---------------------------------------------------------------------------


class TestFactSemanticDecision:
    def test_llm_round_trip(self) -> None:
        decision = make_fact_decision()
        assert FactSemanticDecision.from_dict(decision.to_dict()) == decision

    def test_canonical_order_enforced(self) -> None:
        with pytest.raises(ConsolidationModelError):
            make_fact_decision(
                left_candidate_ref=FACT_RIGHT, right_candidate_ref=FACT_LEFT
            )

    def test_same_ref_rejected(self) -> None:
        with pytest.raises(ConsolidationModelError):
            make_fact_decision(right_candidate_ref=FACT_LEFT)

    def test_closed_decision_enum(self) -> None:
        assert FACT_DECISIONS == {
            "same_fact",
            "compatible_fact",
            "state_change",
            "conflict",
            "unrelated",
            "uncertain",
        }

    def test_bad_decision_rejected(self) -> None:
        with pytest.raises(ConsolidationModelError):
            make_fact_decision(decision="merge")

    def test_llm_requires_provenance(self) -> None:
        with pytest.raises(ConsolidationModelError):
            make_fact_decision(method="llm", generation_provenance=None)

    def test_deterministic_requires_null_prompt_identity_and_provenance(self) -> None:
        decision = make_fact_decision(
            method="deterministic",
            prompt_id=None,
            prompt_version=None,
            generation_provenance=None,
        )
        assert decision.generation_provenance is None
        assert decision.prompt_id is None
        assert decision.prompt_version is None

    def test_deterministic_with_provenance_rejected(self) -> None:
        with pytest.raises(ConsolidationModelError):
            make_fact_decision(
                method="deterministic",
                prompt_id=None,
                prompt_version=None,
                generation_provenance=make_provenance(),
            )

    def test_deterministic_with_prompt_id_rejected(self) -> None:
        with pytest.raises(ConsolidationModelError):
            make_fact_decision(
                method="deterministic",
                prompt_id=A5_FACT_PROMPT_ID,
                prompt_version=PROMPT_VERSION,
                generation_provenance=None,
            )

    def test_fact_namespace_enforced(self) -> None:
        # An event candidate ref is not a valid fact decision endpoint.
        with pytest.raises(ConsolidationModelError):
            make_fact_decision(left_candidate_ref=EVENT_LEFT)


class TestEventSemanticDecision:
    def test_round_trip(self) -> None:
        decision = make_event_decision()
        assert EventSemanticDecision.from_dict(decision.to_dict()) == decision

    def test_closed_decision_enum(self) -> None:
        assert EVENT_DECISIONS == {"same_event", "different_event", "uncertain"}

    def test_bad_decision_rejected(self) -> None:
        with pytest.raises(ConsolidationModelError):
            make_event_decision(decision="same_fact")  # fact decision in event domain


class TestRelationshipSemanticDecision:
    def test_round_trip(self) -> None:
        decision = make_relationship_decision()
        assert RelationshipSemanticDecision.from_dict(decision.to_dict()) == decision

    def test_closed_decision_enum(self) -> None:
        assert RELATIONSHIP_DECISIONS == {
            "same_relationship",
            "different_relationship",
            "uncertain",
        }

    def test_bad_decision_rejected(self) -> None:
        with pytest.raises(ConsolidationModelError):
            make_relationship_decision(decision="same_event")


class TestConsolidationDecisionSet:
    def test_round_trip(self) -> None:
        ds = ConsolidationDecisionSet(
            schema_version=1,
            fact_decisions=(make_fact_decision(),),
            event_decisions=(make_event_decision(),),
            relationship_decisions=(make_relationship_decision(),),
        )
        assert ConsolidationDecisionSet.from_dict(ds.to_dict()) == ds

    def test_wrong_schema_version_rejected(self) -> None:
        with pytest.raises(ConsolidationModelError):
            ConsolidationDecisionSet(
                schema_version=2,
                fact_decisions=(),
                event_decisions=(),
                relationship_decisions=(),
            )

    def test_schema_parity(self) -> None:
        ds = ConsolidationDecisionSet(
            schema_version=1,
            fact_decisions=(
                make_fact_decision(),
                make_fact_decision(
                    decision_id="factd_000002",
                    method="deterministic",
                    reason_zh="同一。",
                    prompt_id=None,
                    prompt_version=None,
                    generation_provenance=None,
                ),
            ),
            event_decisions=(make_event_decision(),),
            relationship_decisions=(make_relationship_decision(),),
        )
        assert _validate(ds.to_dict(), _schema("consolidation-decision-set.schema.json")) == []


# ---------------------------------------------------------------------------
# Provider selector payloads (pair-local evidence selectors)
# ---------------------------------------------------------------------------


def _validate_raw_decision_set(
    fact: list[dict], event: list[dict], relationship: list[dict]
) -> list[str]:
    return _validate(
        {
            "schema_version": 1,
            "fact_decisions": fact,
            "event_decisions": event,
            "relationship_decisions": relationship,
        },
        _schema("consolidation-decision-set.schema.json"),
    )


class TestFactSelectorDecisionItem:
    def test_round_trip(self) -> None:
        item = FactSelectorDecisionItem(
            left_candidate_ref=FACT_LEFT,
            right_candidate_ref=FACT_RIGHT,
            decision="same_fact",
            reason_zh="同一事实。",
            evidence_selectors=("L0", "R0"),
        )
        assert FactSelectorDecisionItem.from_dict(item.to_dict()) == item

    def test_canonical_order_enforced(self) -> None:
        with pytest.raises(ConsolidationModelError):
            FactSelectorDecisionItem(
                left_candidate_ref=FACT_RIGHT,
                right_candidate_ref=FACT_LEFT,
                decision="same_fact",
                reason_zh="x",
                evidence_selectors=(),
            )

    def test_closed_decision_enum(self) -> None:
        with pytest.raises(ConsolidationModelError):
            FactSelectorDecisionItem(
                left_candidate_ref=FACT_LEFT,
                right_candidate_ref=FACT_RIGHT,
                decision="merge",
                reason_zh="x",
                evidence_selectors=(),
            )

    def test_payload_round_trip(self) -> None:
        payload = FactSelectorDecisionPayload(
            decisions=(
                FactSelectorDecisionItem(
                    left_candidate_ref=FACT_LEFT,
                    right_candidate_ref=FACT_RIGHT,
                    decision="same_fact",
                    reason_zh="x",
                    evidence_selectors=("L0",),
                ),
            )
        )
        assert FactSelectorDecisionPayload.from_dict(payload.to_dict()) == payload


class TestEventSelectorDecisionItem:
    def test_round_trip(self) -> None:
        payload = EventSelectorDecisionPayload(
            decisions=(
                EventSelectorDecisionItem(
                    left_candidate_ref=EVENT_LEFT,
                    right_candidate_ref=EVENT_RIGHT,
                    decision="same_event",
                    reason_zh="x",
                    evidence_selectors=(),
                ),
            )
        )
        assert EventSelectorDecisionPayload.from_dict(payload.to_dict()) == payload


class TestRelationshipSelectorDecisionItem:
    def test_round_trip(self) -> None:
        payload = RelationshipSelectorDecisionPayload(
            decisions=(
                RelationshipSelectorDecisionItem(
                    left_candidate_ref=REL_LEFT,
                    right_candidate_ref=REL_RIGHT,
                    decision="same_relationship",
                    reason_zh="x",
                    evidence_selectors=("L0",),
                ),
            )
        )
        assert (
            RelationshipSelectorDecisionPayload.from_dict(payload.to_dict()) == payload
        )

    def test_domain_ref_namespace_enforced(self) -> None:
        with pytest.raises(ConsolidationModelError):
            RelationshipSelectorDecisionItem(
                left_candidate_ref=FACT_LEFT,
                right_candidate_ref=REL_RIGHT,
                decision="same_relationship",
                reason_zh="x",
                evidence_selectors=(),
            )


class TestProviderSelectorSchemas:
    """The three provider selector-payload schemas are strict-mode clean and
    enforce the domain decision enum + domain candidate-ref pattern. The
    provider NEVER reproduces EvidenceRef fields (only pair-local selectors)."""

    _SCHEMAS = {
        "fact": "consolidation-fact-selector-payload.schema.json",
        "event": "consolidation-event-selector-payload.schema.json",
        "relationship": "consolidation-relationship-selector-payload.schema.json",
    }

    _FORBIDDEN_PROVIDER_KEYS = {
        "$schema",
        "$id",
        "$ref",
        "$defs",
        "definitions",
        "anyOf",
        "oneOf",
        "allOf",
        "not",
        "uniqueItems",
        "title",
        "description",
        "$comment",
    }

    def _collect_keys(self, node: object, found: set[str]) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                found.add(key)
                self._collect_keys(value, found)
        elif isinstance(node, list):
            for item in node:
                self._collect_keys(item, found)

    @pytest.mark.parametrize("domain", ["fact", "event", "relationship"])
    def test_provider_schema_is_strict_mode_clean(self, domain: str) -> None:
        schema = _schema(self._SCHEMAS[domain])
        found: set[str] = set()
        self._collect_keys(schema, found)
        assert found & self._FORBIDDEN_PROVIDER_KEYS == set()

    def test_fact_schema_accepts_valid_payload(self) -> None:
        payload = FactSelectorDecisionPayload(
            decisions=(
                FactSelectorDecisionItem(
                    left_candidate_ref=FACT_LEFT,
                    right_candidate_ref=FACT_RIGHT,
                    decision="same_fact",
                    reason_zh="x",
                    evidence_selectors=("L0", "R0"),
                ),
            )
        )
        assert _validate(payload.to_dict(), _schema(self._SCHEMAS["fact"])) == []

    def test_event_schema_accepts_valid_payload(self) -> None:
        payload = EventSelectorDecisionPayload(
            decisions=(
                EventSelectorDecisionItem(
                    left_candidate_ref=EVENT_LEFT,
                    right_candidate_ref=EVENT_RIGHT,
                    decision="different_event",
                    reason_zh="x",
                    evidence_selectors=(),
                ),
            )
        )
        assert _validate(payload.to_dict(), _schema(self._SCHEMAS["event"])) == []

    def test_relationship_schema_accepts_valid_payload(self) -> None:
        payload = RelationshipSelectorDecisionPayload(
            decisions=(
                RelationshipSelectorDecisionItem(
                    left_candidate_ref=REL_LEFT,
                    right_candidate_ref=REL_RIGHT,
                    decision="uncertain",
                    reason_zh="x",
                    evidence_selectors=("L1",),
                ),
            )
        )
        assert _validate(payload.to_dict(), _schema(self._SCHEMAS["relationship"])) == []

    def test_fact_schema_rejects_bad_decision(self) -> None:
        payload = {
            "decisions": [
                {
                    "left_candidate_ref": FACT_LEFT,
                    "right_candidate_ref": FACT_RIGHT,
                    "decision": "merge",
                    "reason_zh": "x",
                    "evidence_selectors": [],
                }
            ]
        }
        assert _validate(payload, _schema(self._SCHEMAS["fact"])) != []

    def test_fact_schema_rejects_event_candidate_ref(self) -> None:
        payload = {
            "decisions": [
                {
                    "left_candidate_ref": EVENT_LEFT,
                    "right_candidate_ref": FACT_RIGHT,
                    "decision": "same_fact",
                    "reason_zh": "x",
                    "evidence_selectors": [],
                }
            ]
        }
        assert _validate(payload, _schema(self._SCHEMAS["fact"])) != []

    def test_fact_schema_rejects_extra_key(self) -> None:
        payload = FactSelectorDecisionPayload(
            decisions=(
                FactSelectorDecisionItem(
                    left_candidate_ref=FACT_LEFT,
                    right_candidate_ref=FACT_RIGHT,
                    decision="same_fact",
                    reason_zh="x",
                    evidence_selectors=(),
                ),
            )
        ).to_dict()
        payload["extra"] = 1
        assert _validate(payload, _schema(self._SCHEMAS["fact"])) != []

    def test_provider_schema_rejects_evidence_ref_fields(self) -> None:
        # The provider must NOT emit EvidenceRef fields; only selectors.
        item = {
            "left_candidate_ref": FACT_LEFT,
            "right_candidate_ref": FACT_RIGHT,
            "decision": "same_fact",
            "reason_zh": "x",
            "evidence_selectors": ["L0"],
            "evidence_refs": [{"paragraph_id": "p", "role": "primary",
                               "strength": "explicit", "excerpt": "x"}],
        }
        assert _validate({"decisions": [item]}, _schema(self._SCHEMAS["fact"])) != []


# ---------------------------------------------------------------------------
# Persisted decision schema <-> Python parity
# ---------------------------------------------------------------------------


class TestDecisionSchemaParity:
    def test_accepts_llm_fact_decision(self) -> None:
        ds = ConsolidationDecisionSet(
            schema_version=1, fact_decisions=(make_fact_decision(),),
            event_decisions=(), relationship_decisions=(),
        )
        assert _validate(ds.to_dict(), _schema("consolidation-decision-set.schema.json")) == []

    def test_accepts_deterministic_event_decision(self) -> None:
        ds = ConsolidationDecisionSet(
            schema_version=1, fact_decisions=(),
            event_decisions=(make_event_decision(),), relationship_decisions=(),
        )
        assert _validate(ds.to_dict(), _schema("consolidation-decision-set.schema.json")) == []

    def test_rejects_llm_with_null_provenance(self) -> None:
        base = make_fact_decision().to_dict()
        base["generation_provenance"] = None
        assert _validate_raw_decision_set([base], [], []) != []

    def test_rejects_llm_with_null_prompt_id(self) -> None:
        base = make_fact_decision().to_dict()
        base["prompt_id"] = None
        assert _validate_raw_decision_set([base], [], []) != []

    def test_rejects_deterministic_with_llm_provenance(self) -> None:
        base = make_fact_decision().to_dict()
        base["method"] = "deterministic"
        assert _validate_raw_decision_set([base], [], []) != []

    def test_rejects_bad_decision_enum(self) -> None:
        base = make_fact_decision().to_dict()
        base["decision"] = "merge"
        assert _validate_raw_decision_set([base], [], []) != []


# ---------------------------------------------------------------------------
# Canonical facts + state transitions
# ---------------------------------------------------------------------------


class TestCanonicalFactSet:
    def test_fact_round_trip(self) -> None:
        assert CanonicalFact.from_dict(make_canonical_fact().to_dict()) == make_canonical_fact()

    def test_transition_round_trip(self) -> None:
        assert (
            StateTransition.from_dict(make_state_transition().to_dict())
            == make_state_transition()
        )

    def test_set_round_trip(self) -> None:
        cs = CanonicalFactSet(
            schema_version=1,
            facts=(make_canonical_fact(),),
            state_transitions=(make_state_transition(),),
        )
        assert CanonicalFactSet.from_dict(cs.to_dict()) == cs

    def test_bad_fact_id_rejected(self) -> None:
        with pytest.raises(ConsolidationModelError):
            make_canonical_fact(fact_id="fact_1")

    def test_bad_transition_id_rejected(self) -> None:
        with pytest.raises(ConsolidationModelError):
            make_state_transition(transition_id="trans_1")

    def test_bad_transition_kind_rejected(self) -> None:
        with pytest.raises(ConsolidationModelError):
            make_state_transition(transition_kind="nonsense")

    def test_transition_from_to_must_be_fact_ids(self) -> None:
        with pytest.raises(ConsolidationModelError):
            make_state_transition(from_fact_id="evt_000001")

    def test_requires_min_one_candidate_ref(self) -> None:
        with pytest.raises(ConsolidationModelError):
            make_canonical_fact(candidate_fact_refs=())

    def test_schema_parity(self) -> None:
        cs = CanonicalFactSet(
            schema_version=1,
            facts=(make_canonical_fact(),),
            state_transitions=(make_state_transition(),),
        )
        assert _validate(cs.to_dict(), _schema("canonical-fact-set.schema.json")) == []


# ---------------------------------------------------------------------------
# Canonical events
# ---------------------------------------------------------------------------


class TestCanonicalEventSet:
    def test_round_trip(self) -> None:
        es = CanonicalEventSet(schema_version=1, events=(make_canonical_event(),))
        assert CanonicalEventSet.from_dict(es.to_dict()) == es

    def test_bad_event_id_rejected(self) -> None:
        with pytest.raises(ConsolidationModelError):
            make_canonical_event(event_id="evt_1")

    def test_bad_temporal_mode_rejected(self) -> None:
        with pytest.raises(ConsolidationModelError):
            make_canonical_event(temporal_mode="nonsense")

    def test_narrative_order_must_be_positive(self) -> None:
        with pytest.raises(ConsolidationModelError):
            make_canonical_event(narrative_order=0)

    def test_schema_parity(self) -> None:
        es = CanonicalEventSet(schema_version=1, events=(make_canonical_event(),))
        assert _validate(es.to_dict(), _schema("canonical-event-set.schema.json")) == []


# ---------------------------------------------------------------------------
# Canonical relationships + state history
# ---------------------------------------------------------------------------


class TestCanonicalRelationshipSet:
    def test_round_trip(self) -> None:
        rs = CanonicalRelationshipSet(
            schema_version=1, relationships=(make_canonical_relationship(),)
        )
        assert CanonicalRelationshipSet.from_dict(rs.to_dict()) == rs

    def test_bad_relationship_id_rejected(self) -> None:
        with pytest.raises(ConsolidationModelError):
            make_canonical_relationship(relationship_id="rel_1")

    def test_unbound_source_ref_rejected(self) -> None:
        with pytest.raises(ConsolidationModelError):
            make_canonical_relationship(source_entity_ref="char_1")

    def test_multi_state_history_ok(self) -> None:
        rel = make_canonical_relationship(
            state_history=(
                make_relationship_state(state_zh="好友", narrative_order=1),
                make_relationship_state(state_zh="决裂", narrative_order=2),
            )
        )
        assert len(rel.state_history) == 2

    def test_schema_parity(self) -> None:
        rs = CanonicalRelationshipSet(
            schema_version=1, relationships=(make_canonical_relationship(),)
        )
        assert (
            _validate(rs.to_dict(), _schema("canonical-relationship-set.schema.json"))
            == []
        )


# ---------------------------------------------------------------------------
# Story conflicts
# ---------------------------------------------------------------------------


class TestStoryConflictSet:
    def test_round_trip(self) -> None:
        cs = StoryConflictSet(schema_version=1, conflicts=(make_story_conflict(),))
        assert StoryConflictSet.from_dict(cs.to_dict()) == cs

    def test_bad_conflict_id_rejected(self) -> None:
        with pytest.raises(ConsolidationModelError):
            make_story_conflict(conflict_id="conf_1")

    def test_bad_kind_rejected(self) -> None:
        with pytest.raises(ConsolidationModelError):
            make_story_conflict(conflict_kind="nonsense")

    def test_status_must_be_unresolved(self) -> None:
        assert STORY_CONFLICT_STATUSES == {"unresolved"}
        with pytest.raises(ConsolidationModelError):
            make_story_conflict(status="resolved")

    def test_bad_fact_id_rejected(self) -> None:
        with pytest.raises(ConsolidationModelError):
            make_story_conflict(fact_ids=("fact_1",))

    def test_requires_min_one_candidate_ref(self) -> None:
        with pytest.raises(ConsolidationModelError):
            make_story_conflict(candidate_refs=())

    def test_all_kinds_accepted(self) -> None:
        assert STORY_CONFLICT_KINDS == {
            "fact_conflict",
            "relationship_state_conflict",
            "other",
        }

    def test_schema_parity(self) -> None:
        cs = StoryConflictSet(schema_version=1, conflicts=(make_story_conflict(),))
        assert _validate(cs.to_dict(), _schema("story-conflict-set.schema.json")) == []


# ---------------------------------------------------------------------------
# A5SemanticIdentity (backend-neutral)
# ---------------------------------------------------------------------------


class TestA5SemanticIdentity:
    def test_round_trip(self) -> None:
        ident = make_semantic_identity()
        assert A5SemanticIdentity.from_dict(ident.to_dict()) == ident

    def test_empty_request_hashes_allowed(self) -> None:
        ident = make_semantic_identity(semantic_request_hashes=())
        assert ident.semantic_request_hashes == ()

    def test_is_backend_neutral(self) -> None:
        ident = make_semantic_identity()
        data = ident.to_dict()
        for banned in ("provider_family", "model", "request_model", "endpoint", "timeout"):
            assert banned not in data

    @pytest.mark.parametrize(
        "field",
        [
            "consolidation_profile_hash",
            "fact_semantic_profile_hash",
            "event_semantic_profile_hash",
            "relationship_semantic_profile_hash",
            "plan_hash",
        ],
    )
    def test_bad_hash_rejected(self, field: str) -> None:
        with pytest.raises(ConsolidationModelError):
            make_semantic_identity(**{field: "not-a-hash"})

    def test_non_hash_request_hash_rejected(self) -> None:
        with pytest.raises(ConsolidationModelError):
            make_semantic_identity(semantic_request_hashes=("nothex",))

    def test_duplicate_request_hashes_rejected(self) -> None:
        with pytest.raises(ConsolidationModelError):
            make_semantic_identity(semantic_request_hashes=(H, H))

    def test_bad_prompt_asset_identity_rejected(self) -> None:
        with pytest.raises(ConsolidationModelError):
            PromptAssetIdentity(prompt_id="x", prompt_version=1, prompt_content_hash="bad")


# ---------------------------------------------------------------------------
# Upstream identity / coverage / manifest
# ---------------------------------------------------------------------------


class TestA5UpstreamIdentity:
    def test_round_trip(self) -> None:
        up = A5UpstreamIdentity(a3_input=make_a3_input())
        assert A5UpstreamIdentity.from_dict(up.to_dict()) == up

    def test_requires_a3_input(self) -> None:
        with pytest.raises(ConsolidationModelError):
            A5UpstreamIdentity(a3_input="not-an-identity")  # type: ignore[arg-type]


class TestConsolidationCoverageSummary:
    def test_round_trip(self) -> None:
        cov = ConsolidationCoverageSummary(
            fact_candidate_count=5,
            event_candidate_count=3,
            relationship_candidate_count=2,
            canonical_fact_count=4,
            canonical_event_count=2,
            canonical_relationship_count=2,
            uncertain_decision_count=0,
            story_conflict_count=0,
        )
        assert ConsolidationCoverageSummary.from_dict(cov.to_dict()) == cov

    def test_negative_count_rejected(self) -> None:
        with pytest.raises(ConsolidationModelError):
            ConsolidationCoverageSummary(
                fact_candidate_count=-1,
                event_candidate_count=0,
                relationship_candidate_count=0,
                canonical_fact_count=0,
                canonical_event_count=0,
                canonical_relationship_count=0,
                uncertain_decision_count=0,
                story_conflict_count=0,
            )


class TestConsolidationManifest:
    def test_round_trip(self) -> None:
        manifest = make_manifest()
        assert ConsolidationManifest.from_dict(manifest.to_dict()) == manifest

    def test_content_hash_stable(self) -> None:
        manifest = make_manifest()
        assert manifest.content_hash() == content_hash(manifest.to_dict())
        assert len(manifest.content_hash()) == 64

    def test_wrong_schema_version_rejected(self) -> None:
        with pytest.raises(ConsolidationModelError):
            make_manifest(schema_version=2)

    def test_schema_parity(self) -> None:
        manifest = make_manifest()
        assert _validate(manifest.to_dict(), _schema("consolidation-manifest.schema.json")) == []

    def test_schema_rejects_bad_semantic_request_hash(self) -> None:
        manifest = make_manifest()
        data = manifest.to_dict()
        data["semantic_identity"]["semantic_request_hashes"] = ["not-a-hash"]
        assert _validate(data, _schema("consolidation-manifest.schema.json")) != []


# ---------------------------------------------------------------------------
# JSON Schema parity (all eight persisted A5 schemas)
# ---------------------------------------------------------------------------


class TestSchemaParity:
    def test_profile(self) -> None:
        profile = load_consolidation_profile(CONSOLIDATION_PROFILE_PATH)
        assert _validate(profile.to_dict(), _schema("consolidation-profile.schema.json")) == []

    def test_candidate_index(self) -> None:
        index = ConsolidationCandidateIndex(
            schema_version=1,
            facts=(make_fact_candidate(),),
            events=(make_event_candidate(),),
            relationships=(make_relationship_candidate(),),
        )
        assert _validate(index.to_dict(), _schema("consolidation-candidate-index.schema.json")) == []

    def test_decision_set(self) -> None:
        ds = ConsolidationDecisionSet(
            schema_version=1,
            fact_decisions=(make_fact_decision(),),
            event_decisions=(make_event_decision(),),
            relationship_decisions=(make_relationship_decision(),),
        )
        assert _validate(ds.to_dict(), _schema("consolidation-decision-set.schema.json")) == []

    def test_canonical_fact_set(self) -> None:
        cs = CanonicalFactSet(
            schema_version=1,
            facts=(make_canonical_fact(),),
            state_transitions=(make_state_transition(),),
        )
        assert _validate(cs.to_dict(), _schema("canonical-fact-set.schema.json")) == []

    def test_canonical_event_set(self) -> None:
        es = CanonicalEventSet(schema_version=1, events=(make_canonical_event(),))
        assert _validate(es.to_dict(), _schema("canonical-event-set.schema.json")) == []

    def test_canonical_relationship_set(self) -> None:
        rs = CanonicalRelationshipSet(
            schema_version=1, relationships=(make_canonical_relationship(),)
        )
        assert _validate(rs.to_dict(), _schema("canonical-relationship-set.schema.json")) == []

    def test_story_conflict_set(self) -> None:
        cs = StoryConflictSet(schema_version=1, conflicts=(make_story_conflict(),))
        assert _validate(cs.to_dict(), _schema("story-conflict-set.schema.json")) == []

    def test_manifest(self) -> None:
        manifest = make_manifest()
        assert _validate(manifest.to_dict(), _schema("consolidation-manifest.schema.json")) == []


# ---------------------------------------------------------------------------
# Tracked A5 assets (semantic LLM profile + 3 prompts)
# ---------------------------------------------------------------------------


class TestTrackedA5Assets:
    def test_semantic_profile_loads_and_is_schema_valid(self) -> None:
        profile = load_semantic_profile(A5_LLM_PROFILE_PATH)
        assert profile.profile_id == "consolidation-llm-v1"
        schema = _schema("llm-semantic-profile.schema.json")
        assert _validate(profile.to_dict(), schema) == []

    def test_semantic_profile_is_backend_independent(self) -> None:
        profile = load_semantic_profile(A5_LLM_PROFILE_PATH)
        data = profile.to_dict()
        for banned in ("provider_family", "model", "request_model", "endpoint", "timeout"):
            assert banned not in data

    def test_semantic_profile_is_deterministic(self) -> None:
        profile = load_semantic_profile(A5_LLM_PROFILE_PATH)
        assert profile.temperature == 0.0
        assert profile.structured_output_mode == "json_schema"
        assert profile.reasoning.enabled is False

    @pytest.mark.parametrize(
        "prompt_id, content_hash",
        [
            (A5_FACT_PROMPT_ID, FACT_PROMPT_CONTENT_HASH),
            (A5_EVENT_PROMPT_ID, EVENT_PROMPT_CONTENT_HASH),
            (A5_RELATIONSHIP_PROMPT_ID, RELATIONSHIP_PROMPT_CONTENT_HASH),
        ],
    )
    def test_prompt_loads_with_pinned_hash(self, prompt_id: str, content_hash: str) -> None:
        registry = PromptRegistry(PROMPTS_STORY_DIR)
        spec = registry.load(prompt_id, version=PROMPT_VERSION)
        assert spec.content_hash == content_hash
        assert tuple(spec.required_variables) == ("block_id", "pair_contexts_json")

    def test_fact_prompt_renders_with_exact_variables(self) -> None:
        registry = PromptRegistry(PROMPTS_STORY_DIR)
        spec = registry.load(A5_FACT_PROMPT_ID, version=PROMPT_VERSION)
        rendered = render_prompt(
            spec,
            {"block_id": "block_0001", "pair_contexts_json": "[]"},
        )
        assert rendered.prompt_id == A5_FACT_PROMPT_ID
        assert len(rendered.rendered_prompt_hash) == 64

    def test_prompt_rejects_extra_variable(self) -> None:
        registry = PromptRegistry(PROMPTS_STORY_DIR)
        spec = registry.load(A5_FACT_PROMPT_ID, version=PROMPT_VERSION)
        with pytest.raises(LLMPromptError):
            render_prompt(
                spec,
                {
                    "block_id": "block_0001",
                    "pair_contexts_json": "[]",
                    "unexpected": "x",
                },
            )

    def test_prompt_rejects_missing_variable(self) -> None:
        registry = PromptRegistry(PROMPTS_STORY_DIR)
        spec = registry.load(A5_FACT_PROMPT_ID, version=PROMPT_VERSION)
        with pytest.raises(LLMPromptError):
            render_prompt(spec, {"block_id": "block_0001"})

    def test_tracked_profile_pins_tracked_prompts(self) -> None:
        profile = load_consolidation_profile(CONSOLIDATION_PROFILE_PATH)
        registry = PromptRegistry(PROMPTS_STORY_DIR)
        expected_hashes = {
            A5_FACT_PROMPT_ID: FACT_PROMPT_CONTENT_HASH,
            A5_EVENT_PROMPT_ID: EVENT_PROMPT_CONTENT_HASH,
            A5_RELATIONSHIP_PROMPT_ID: RELATIONSHIP_PROMPT_CONTENT_HASH,
        }
        for name in ("fact", "event", "relationship"):
            psg = getattr(profile, name)
            spec = registry.load(psg.prompt_id, version=psg.prompt_version)
            assert spec.content_hash == expected_hashes[psg.prompt_id]
            # The tracked profile pins the shared A5 semantic LLM profile id.
            assert psg.semantic_profile_id == "consolidation-llm-v1"


class TestPromptPairLocalSelectorContract:
    """The tracked A5 prompts pin the pair-local evidence-selector contract
    (inherited from A4 v3): the provider cites evidence only by L0/R0/...
    selectors and never reproduces EvidenceRef fields."""

    @staticmethod
    def _system_text(prompt_id: str) -> str:
        registry = PromptRegistry(PROMPTS_STORY_DIR)
        spec = registry.load(prompt_id, version=PROMPT_VERSION)
        return " ".join(spec.system_template.split())

    def test_fact_prompt_documents_pair_local_selectors(self) -> None:
        text = self._system_text(A5_FACT_PROMPT_ID)
        assert "labeled L0, L1, L2" in text
        assert "labeled R0, R1, R2" in text
        assert "the ONLY way to cite evidence" in text

    def test_fact_prompt_forbids_reproducing_evidence_fields(self) -> None:
        text = self._system_text(A5_FACT_PROMPT_ID)
        assert "Do not copy or reproduce the underlying reference fields" in text

    def test_event_prompt_documents_pair_local_selectors(self) -> None:
        text = self._system_text(A5_EVENT_PROMPT_ID)
        assert "labeled L0, L1, L2" in text
        assert "the ONLY way to cite evidence" in text

    def test_relationship_prompt_documents_pair_local_selectors(self) -> None:
        text = self._system_text(A5_RELATIONSHIP_PROMPT_ID)
        assert "labeled L0, L1, L2" in text
        assert "the ONLY way to cite evidence" in text


# ---------------------------------------------------------------------------
# A5A contract / parity repair (issue #50 independent review)
# ---------------------------------------------------------------------------


class TestA5AContractParityRepair:
    """Pin that the Python domain models and the persisted JSON Schemas agree
    on the frozen A5A contract after the review repairs:

    * bound-entity namespaces ``char_*`` / ``loc_*`` / ``unres_*`` (``obj_*``
      and arbitrary strings rejected);
    * exactly three ``StateTransition`` kinds;
    * domain-specific candidate-ref namespaces on indexed + canonical models;
    * non-nullable fact object refs;
    * participantless events remain representable;
    * the pair-local evidence-selector shape ``^[LR][0-9]+$``.
    """

    # -- helpers ----------------------------------------------------------

    @staticmethod
    def _fact_set_payload(**overrides: object) -> dict:
        return CanonicalFactSet(
            schema_version=1,
            facts=(make_canonical_fact(**overrides),),
            state_transitions=(),
        ).to_dict()

    @staticmethod
    def _event_set_payload(**overrides: object) -> dict:
        return CanonicalEventSet(
            schema_version=1, events=(make_canonical_event(**overrides),)
        ).to_dict()

    # -- Finding 1: bound entity namespace --------------------------------

    @pytest.mark.parametrize("ref", ["char_0001", "loc_0001", "unres_0001"])
    def test_bound_entity_namespaces_accepted_python(self, ref: str) -> None:
        fact = make_canonical_fact(subject_refs=(ref,), object_refs=(ref,))
        assert fact.subject_refs == (ref,) and fact.object_refs == (ref,)

    @pytest.mark.parametrize(
        "bad", ["obj_0001", "someone", "char_1", "CHAR_0001", "unres_1"]
    )
    def test_bound_entity_bad_rejected_python(self, bad: str) -> None:
        with pytest.raises(ConsolidationModelError):
            make_canonical_fact(subject_refs=(bad,))

    @pytest.mark.parametrize("ref", ["char_0001", "loc_0001", "unres_0001"])
    def test_bound_entity_namespaces_accepted_schema(self, ref: str) -> None:
        payload = self._fact_set_payload()
        payload["facts"][0]["subject_refs"] = [ref]
        assert _validate(payload, _schema("canonical-fact-set.schema.json")) == []

    @pytest.mark.parametrize("bad", ["obj_0001", "someone", "char_1"])
    def test_bound_entity_bad_rejected_schema(self, bad: str) -> None:
        payload = self._fact_set_payload()
        payload["facts"][0]["subject_refs"] = [bad]
        assert _validate(payload, _schema("canonical-fact-set.schema.json")) != []

    # -- Finding 2: StateTransition kinds ---------------------------------

    def test_state_transition_kinds_constant(self) -> None:
        assert STATE_TRANSITION_KINDS == {
            "state_change",
            "relationship_state_change",
            "other",
        }

    @pytest.mark.parametrize(
        "kind", ["state_change", "relationship_state_change", "other"]
    )
    def test_state_transition_kinds_accepted_python(self, kind: str) -> None:
        assert make_state_transition(transition_kind=kind).transition_kind == kind

    @pytest.mark.parametrize(
        "bad", ["reassertion", "clarification", "correction", "nonsense"]
    )
    def test_state_transition_kinds_rejected_python(self, bad: str) -> None:
        with pytest.raises(ConsolidationModelError):
            make_state_transition(transition_kind=bad)

    def test_state_transition_kind_python_schema_identical(self) -> None:
        schema = _schema("canonical-fact-set.schema.json")
        enum = schema["$defs"]["state_transition"]["properties"]["transition_kind"]["enum"]
        assert set(enum) == set(STATE_TRANSITION_KINDS)

    @pytest.mark.parametrize(
        "kind", ["state_change", "relationship_state_change", "other"]
    )
    def test_state_transition_kinds_accepted_schema(self, kind: str) -> None:
        payload = CanonicalFactSet(
            schema_version=1,
            facts=(),
            state_transitions=(make_state_transition(transition_kind=kind),),
        ).to_dict()
        assert _validate(payload, _schema("canonical-fact-set.schema.json")) == []

    # -- Finding 3: indexed candidate namespace ---------------------------

    def test_indexed_fact_rejects_event_and_rel_refs(self) -> None:
        with pytest.raises(ConsolidationModelError):
            make_fact_candidate(global_candidate_ref=EVENT_LEFT)
        with pytest.raises(ConsolidationModelError):
            make_fact_candidate(global_candidate_ref=REL_LEFT)

    def test_indexed_event_rejects_fact_and_rel_refs(self) -> None:
        with pytest.raises(ConsolidationModelError):
            make_event_candidate(global_candidate_ref=FACT_LEFT)
        with pytest.raises(ConsolidationModelError):
            make_event_candidate(global_candidate_ref=REL_LEFT)

    def test_indexed_rel_rejects_fact_and_event_refs(self) -> None:
        with pytest.raises(ConsolidationModelError):
            make_relationship_candidate(global_candidate_ref=FACT_LEFT)
        with pytest.raises(ConsolidationModelError):
            make_relationship_candidate(global_candidate_ref=EVENT_LEFT)

    def test_indexed_parts_must_match_ref_contract(self) -> None:
        # chunk_id that does not match the parsed ref part fails closed.
        with pytest.raises(ConsolidationModelError):
            make_fact_candidate(chunk_id="not-a-chunk-id")
        # local_candidate_id that does not match the parsed ref part.
        with pytest.raises(ConsolidationModelError):
            make_fact_candidate(local_candidate_id="cand_fact_999")

    def test_indexed_global_ref_must_satisfy_contract(self) -> None:
        # A global ref whose local id is malformed fails closed at parse.
        with pytest.raises(ConsolidationModelError):
            make_fact_candidate(
                global_candidate_ref="CH003_C005:cand_fact_12",
                local_candidate_id="cand_fact_12",
            )

    # -- Finding 4: canonical member refs are domain-scoped ---------------

    def test_canonical_fact_rejects_event_and_rel_refs(self) -> None:
        with pytest.raises(ConsolidationModelError):
            make_canonical_fact(candidate_fact_refs=(EVENT_LEFT,))
        with pytest.raises(ConsolidationModelError):
            make_canonical_fact(candidate_fact_refs=(REL_LEFT,))

    def test_canonical_event_rejects_fact_and_rel_refs(self) -> None:
        with pytest.raises(ConsolidationModelError):
            make_canonical_event(candidate_event_refs=(FACT_LEFT,))
        with pytest.raises(ConsolidationModelError):
            make_canonical_event(candidate_event_refs=(REL_LEFT,))

    def test_relationship_state_rejects_fact_and_event_refs(self) -> None:
        with pytest.raises(ConsolidationModelError):
            make_relationship_state(candidate_relationship_refs=(FACT_LEFT,))
        with pytest.raises(ConsolidationModelError):
            make_relationship_state(candidate_relationship_refs=(EVENT_LEFT,))

    def test_canonical_relationship_rejects_fact_and_event_refs(self) -> None:
        with pytest.raises(ConsolidationModelError):
            make_canonical_relationship(candidate_relationship_refs=(FACT_LEFT,))
        with pytest.raises(ConsolidationModelError):
            make_canonical_relationship(candidate_relationship_refs=(EVENT_LEFT,))

    def test_story_conflict_candidate_refs_remain_cross_domain(self) -> None:
        conflict = make_story_conflict(
            candidate_refs=(FACT_LEFT, EVENT_LEFT, REL_LEFT)
        )
        assert conflict.candidate_refs == (FACT_LEFT, EVENT_LEFT, REL_LEFT)
        # And still fail closed on an arbitrary string.
        with pytest.raises(ConsolidationModelError):
            make_story_conflict(candidate_refs=("not-a-ref",))

    # -- Finding 5: bound refs fail closed across the A5A contracts -------

    @pytest.mark.parametrize("field", ["subject_refs", "object_refs"])
    def test_indexed_fact_bound_refs_fail_closed(self, field: str) -> None:
        with pytest.raises(ConsolidationModelError):
            make_fact_candidate(**{field: ("arbitrary-string",)})

    @pytest.mark.parametrize("field", ["participants", "locations"])
    def test_indexed_event_bound_refs_fail_closed(self, field: str) -> None:
        with pytest.raises(ConsolidationModelError):
            make_event_candidate(**{field: ("arbitrary-string",)})

    @pytest.mark.parametrize("field", ["source_entity_ref", "target_entity_ref"])
    def test_indexed_rel_bound_refs_fail_closed(self, field: str) -> None:
        with pytest.raises(ConsolidationModelError):
            make_relationship_candidate(**{field: "arbitrary-string"})

    @pytest.mark.parametrize("field", ["subject_refs", "object_refs"])
    def test_canonical_fact_bound_refs_fail_closed(self, field: str) -> None:
        with pytest.raises(ConsolidationModelError):
            make_canonical_fact(**{field: ("arbitrary-string",)})

    def test_state_transition_subject_refs_fail_closed(self) -> None:
        with pytest.raises(ConsolidationModelError):
            make_state_transition(subject_refs=("arbitrary-string",))

    @pytest.mark.parametrize("field", ["participants", "locations"])
    def test_canonical_event_bound_refs_fail_closed(self, field: str) -> None:
        with pytest.raises(ConsolidationModelError):
            make_canonical_event(**{field: ("arbitrary-string",)})

    @pytest.mark.parametrize("field", ["source_entity_ref", "target_entity_ref"])
    def test_canonical_rel_bound_refs_fail_closed(self, field: str) -> None:
        with pytest.raises(ConsolidationModelError):
            make_canonical_relationship(**{field: "arbitrary-string"})

    # -- Finding 6: fact object refs are non-nullable ---------------------

    def test_indexed_fact_object_refs_reject_null(self) -> None:
        with pytest.raises(ConsolidationModelError):
            make_fact_candidate(object_refs=(None,))  # type: ignore[arg-type]
        with pytest.raises(ConsolidationModelError):
            make_fact_candidate(object_refs=("char_0001", None))  # type: ignore[list-item]

    def test_canonical_fact_object_refs_reject_null(self) -> None:
        with pytest.raises(ConsolidationModelError):
            make_canonical_fact(object_refs=(None,))  # type: ignore[arg-type]
        with pytest.raises(ConsolidationModelError):
            make_canonical_fact(object_refs=("char_0001", None))  # type: ignore[list-item]

    def test_empty_object_refs_still_legal(self) -> None:
        assert make_fact_candidate(object_refs=()).object_refs == ()
        assert make_canonical_fact(object_refs=()).object_refs == ()

    def test_fact_object_refs_null_rejected_schema(self) -> None:
        schema = _schema("canonical-fact-set.schema.json")
        payload = self._fact_set_payload()
        payload["facts"][0]["object_refs"] = [None]
        assert _validate(payload, schema) != []
        payload2 = self._fact_set_payload()
        payload2["facts"][0]["object_refs"] = ["char_0001", None]
        assert _validate(payload2, schema) != []

    # -- Finding 7: participantless events remain representable -----------

    def test_indexed_event_no_participants_round_trip(self) -> None:
        cand = make_event_candidate(participants=(), locations=())
        assert cand.participants == ()
        assert IndexedEventCandidate.from_dict(cand.to_dict()) == cand

    def test_canonical_event_no_participants_round_trip(self) -> None:
        evt = make_canonical_event(participants=(), locations=())
        assert evt.participants == ()
        assert CanonicalEvent.from_dict(evt.to_dict()) == evt

    def test_participantless_event_schema_parity(self) -> None:
        assert _validate(
            self._event_set_payload(participants=(), locations=()),
            _schema("canonical-event-set.schema.json"),
        ) == []
        index = ConsolidationCandidateIndex(
            schema_version=1,
            facts=(),
            events=(make_event_candidate(participants=(), locations=()),),
            relationships=(),
        )
        assert _validate(
            index.to_dict(), _schema("consolidation-candidate-index.schema.json")
        ) == []

    # -- Finding 8: pair-local evidence-selector shape --------------------

    def test_selector_pattern_constant(self) -> None:
        import re

        assert A5_EVIDENCE_SELECTOR_PATTERN == r"^[LR][0-9]+$"
        for good in ("L0", "R0", "L1", "R12"):
            assert re.fullmatch(A5_EVIDENCE_SELECTOR_PATTERN, good) is not None
        for bad in ("LR0", "L", "R", "0", "X0", "l0", "L0R"):
            assert re.fullmatch(A5_EVIDENCE_SELECTOR_PATTERN, bad) is None


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------


class TestExports:
    def test_all_a5a_names_importable_from_package(self) -> None:
        import short_drama.story as story

        for name in (
            "ConsolidationCandidateRef",
            "ConsolidationProfile",
            "ConsolidationCandidateIndex",
            "IndexedFactCandidate",
            "IndexedEventCandidate",
            "IndexedRelationshipCandidate",
            "FactSemanticDecision",
            "EventSemanticDecision",
            "RelationshipSemanticDecision",
            "ConsolidationDecisionSet",
            "FactSelectorDecisionPayload",
            "EventSelectorDecisionPayload",
            "RelationshipSelectorDecisionPayload",
            "CanonicalFact",
            "StateTransition",
            "CanonicalFactSet",
            "CanonicalEvent",
            "CanonicalEventSet",
            "CanonicalRelationship",
            "RelationshipState",
            "CanonicalRelationshipSet",
            "StoryConflict",
            "StoryConflictSet",
            "A5SemanticIdentity",
            "A5UpstreamIdentity",
            "ConsolidationCoverageSummary",
            "ConsolidationManifest",
            "ConsolidationModelError",
            "load_consolidation_profile",
            "FACT_DECISIONS",
            "EVENT_DECISIONS",
            "RELATIONSHIP_DECISIONS",
        ):
            assert hasattr(story, name), name

    def test_id_pattern_constants(self) -> None:
        import re

        assert re.fullmatch(FACT_ID_PATTERN, "fact_000001") is not None
        assert re.fullmatch(EVENT_ID_PATTERN, "evt_000001") is not None
        assert re.fullmatch(RELATIONSHIP_ID_PATTERN, "rel_000001") is not None
        assert re.fullmatch(STATE_TRANSITION_ID_PATTERN, "trans_000001") is not None
        assert re.fullmatch(STORY_CONFLICT_ID_PATTERN, "conf_000001") is not None
        assert re.fullmatch(FACT_ID_PATTERN, "fact_1") is None
