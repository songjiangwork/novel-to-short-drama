"""A4A domain/static-contract tests.

Covers the frozen A4A contract (entity reconciliation domain contracts +
tracked assets):

  * tracked ``EntityReconciliationProfile`` load / round-trip / hash /
    max_generation_rounds ceiling;
  * ``GlobalCandidateRef`` parse / round-trip / namespace + fail-closed;
  * ``CandidateEntityIndex`` / entry round-trips, enums, exact-key parsing;
  * provider ``ReconciliationDecisionPayload`` / item (canonical pair ordering,
    closed decision enum);
  * persisted ``ReconciliationDecision`` method/provenance consistency;
  * ``CanonicalEntity`` / character+location registries (id namespace,
    entity_type const);
  * ``UnresolvedEntity`` / set (entity_kind domain, unres_ id);
  * ``A3InputIdentity`` / ``EntityMapEntry`` resolved-unresolved exclusivity /
    ``EntityMap`` aggregate;
  * JSON Schema parity for the six persisted schemas + provider strict-mode
    discipline;
  * tracked A4 semantic LLM profile (loads, schema-valid, backend-independent);
  * PromptRegistry load / pinned hash / exact required variables / render.

Deliberately does NOT implement or require: candidate collection, name
normalization / blocking / pair planning, an LLM call, identity-graph
construction, persistence/CURRENT reuse, or a CLI (those are A4B-A4E).
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
    load_semantic_profile,
)
from short_drama.paths import PROFILES_DIR, REPO_ROOT, SCHEMAS_DIR
from short_drama.story import (
    A3InputIdentity,
    A4SemanticIdentity,
    BLOCKING_POLICY_ID,
    CANONICALIZATION_POLICY_ID,
    NAME_NORMALIZATION_POLICY_ID,
    CandidateEntityIndex,
    CandidateEntityIndexEntry,
    CanonicalCharacterRegistry,
    CanonicalEntity,
    CanonicalLocationRegistry,
    ENTITY_MAP_SCHEMA_VERSION,
    EntityMap,
    EntityMapEntry,
    EntityReconciliationProfile,
    GLOBAL_CANDIDATE_REF_PATTERN,
    GlobalCandidateRef,
    MERGE_GRAPH_CANDIDATE_REF_PATTERN,
    RECONCILIATION_MAX_GENERATION_ROUNDS_V1,
    ReconciliationDecision,
    ReconciliationDecisionItem,
    ReconciliationDecisionPayload,
    ReconciliationDecisionSet,
    ReconciliationModelError,
    UnresolvedEntity,
    UnresolvedEntitySet,
    EvidenceRef,
    load_entity_reconciliation_profile,
)

PROMPTS_STORY_DIR = REPO_ROOT / "prompts" / "story"
A4_PROMPT_ID = "a4.entity-reconciliation"
A4_PROMPT_VERSION = 2
PROMPT_CONTENT_HASH = (
    "0e54ffd284e984a5b8d2932351744a4a9032c06248fa25a96225d743263ba7d9"
)

RECON_PROFILE_PATH = PROFILES_DIR / "entity_reconciliation_v1.yaml"
RECON_PROFILE_V2_PATH = PROFILES_DIR / "entity_reconciliation_v2.yaml"
A4_LLM_PROFILE_PATH = PROFILES_DIR / "entity_reconciliation_llm_v1.yaml"

H = "a" * 64
H2 = "b" * 64


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
            validator.iter_errors(data), key=lambda e: list(e.absolute_path)
        )
    ]


def make_artifact_ref(**overrides) -> ArtifactRef:
    values = {
        "artifact_type": "candidate_extraction",
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
        "semantic_profile_id": "entity-reconciliation-llm-v1",
        "semantic_profile_hash": H,
        "prompt_id": A4_PROMPT_ID,
        "prompt_version": A4_PROMPT_VERSION,
        "prompt_content_hash": PROMPT_CONTENT_HASH,
        "rendered_prompt_hash": H2,
        "output_schema_id": "a4-reconciliation-decision-payload",
        "output_schema_version": 1,
        "output_schema_hash": H,
        "request_hash": H,
        "provider_response_id": "resp_0001",
        "finish_reason": "stop",
        "usage": {"input_tokens": 10, "output_tokens": 20},
    }
    values.update(overrides)
    return LLMInvocationProvenance(**values)


# Canonical pair: left (CH001_C001) < right (CH001_C002).
LEFT_REF = "CH001_C001:cand_char_001"
RIGHT_REF = "CH001_C002:cand_char_002"


def make_index_entry(**overrides) -> CandidateEntityIndexEntry:
    values = {
        "candidate_ref": LEFT_REF,
        "candidate_kind": "character",
        "candidate_extraction_ref": make_artifact_ref(),
        "source_order_key": "CH001_C001:P0017",
        "display_name_original": "林晚",
        "aliases_original": ("阿晚",),
        "descriptors_zh": ("学生",),
        "evidence_refs": (make_evidence(),),
        "possible_candidate_refs": (LEFT_REF, RIGHT_REF),
    }
    values.update(overrides)
    return CandidateEntityIndexEntry(**values)


def make_decision_item(**overrides) -> ReconciliationDecisionItem:
    values = {
        "left_candidate_ref": LEFT_REF,
        "right_candidate_ref": RIGHT_REF,
        "decision": "uncertain",
        "reason_zh": "两名候选在名字与角色上相似但证据不足以判定同一。",
        "evidence_refs": (make_evidence(),),
    }
    values.update(overrides)
    return ReconciliationDecisionItem(**values)


def make_decision(**overrides) -> ReconciliationDecision:
    values = {
        "decision_id": "dec_0001",
        "left_candidate_ref": LEFT_REF,
        "right_candidate_ref": RIGHT_REF,
        "decision": "uncertain",
        "method": "llm",
        "reason_code": "uncertain_insufficient_evidence",
        "reason_zh": "证据不足以判定同一或不同。",
        "evidence_refs": (make_evidence(),),
        "prompt_id": A4_PROMPT_ID,
        "prompt_version": A4_PROMPT_VERSION,
        "generation_provenance": make_provenance(),
    }
    values.update(overrides)
    return ReconciliationDecision(**values)


def make_character_entity(**overrides) -> CanonicalEntity:
    values = {
        "canonical_id": "char_0001",
        "entity_type": "character",
        "display_name_original": "林晚",
        "aliases_original": ("阿晚",),
        "candidate_refs": (LEFT_REF, RIGHT_REF),
        "first_appearance_candidate_ref": LEFT_REF,
    }
    values.update(overrides)
    return CanonicalEntity(**values)


def make_location_entity(**overrides) -> CanonicalEntity:
    values = {
        "canonical_id": "loc_0001",
        "entity_type": "location",
        "display_name_original": "教室",
        "aliases_original": (),
        "candidate_refs": ("CH001_C001:cand_loc_001",),
        "first_appearance_candidate_ref": "CH001_C001:cand_loc_001",
    }
    values.update(overrides)
    return CanonicalEntity(**values)


def make_unresolved(**overrides) -> UnresolvedEntity:
    values = {
        "unresolved_id": "unres_0001",
        "entity_kind": "character",
        "candidate_refs": (LEFT_REF, RIGHT_REF),
        "decision_refs": ("dec_0001",),
        "possible_candidate_refs": (LEFT_REF, RIGHT_REF),
        "first_appearance_candidate_ref": LEFT_REF,
    }
    values.update(overrides)
    return UnresolvedEntity(**values)


def make_a3_input(**overrides) -> A3InputIdentity:
    values = {
        "source_document_ref": make_artifact_ref(
            artifact_type="source_document", artifact_id="source-0001"
        ),
        "chunk_manifest_ref": make_artifact_ref(
            artifact_type="chunk_manifest", artifact_id="chunk-manifest-0001"
        ),
        "candidate_extraction_refs": (
            make_artifact_ref(artifact_id="cand-extraction-0001"),
            make_artifact_ref(artifact_id="cand-extraction-0002"),
        ),
        "extraction_profile_id": "story-extraction-v1",
        "extraction_profile_hash": H,
    }
    values.update(overrides)
    return A3InputIdentity(**values)


def make_semantic_identity(**overrides) -> A4SemanticIdentity:
    values = {
        "reconciliation_profile_id": "entity-reconciliation-v1",
        "reconciliation_profile_hash": H,
        "semantic_profile_id": "entity-reconciliation-llm-v1",
        "semantic_profile_hash": H2,
        "prompt_id": A4_PROMPT_ID,
        "prompt_version": A4_PROMPT_VERSION,
        "prompt_content_hash": PROMPT_CONTENT_HASH,
        "output_schema_id": "a4-reconciliation-decision-payload",
        "output_schema_version": 1,
        "output_schema_hash": H,
        "plan_hash": H2,
        "semantic_request_hashes": (),
    }
    values.update(overrides)
    return A4SemanticIdentity(**values)


def make_entry(**overrides) -> EntityMapEntry:
    values = {
        "candidate_ref": LEFT_REF,
        "status": "resolved",
        "canonical_id": "char_0001",
        "unresolved_id": None,
    }
    values.update(overrides)
    return EntityMapEntry(**values)


def make_entity_map(**overrides) -> EntityMap:
    values = {
        "schema_version": ENTITY_MAP_SCHEMA_VERSION,
        "entries": (
            make_entry(),
            make_entry(
                candidate_ref=RIGHT_REF,
                status="unresolved",
                canonical_id=None,
                unresolved_id="unres_0001",
            ),
        ),
        "candidate_entity_index_ref": make_artifact_ref(
            artifact_type="candidate_entity_index", artifact_id="cei-0001"
        ),
        "reconciliation_decision_set_ref": make_artifact_ref(
            artifact_type="reconciliation_decision_set", artifact_id="rds-0001"
        ),
        "canonical_character_registry_ref": make_artifact_ref(
            artifact_type="canonical_character_registry", artifact_id="ccr-0001"
        ),
        "canonical_location_registry_ref": make_artifact_ref(
            artifact_type="canonical_location_registry", artifact_id="clr-0001"
        ),
        "unresolved_entity_set_ref": make_artifact_ref(
            artifact_type="unresolved_entity_set", artifact_id="ues-0001"
        ),
        "a3_input": make_a3_input(),
        "semantic_identity": make_semantic_identity(),
    }
    values.update(overrides)
    return EntityMap(**values)


# ---------------------------------------------------------------------------
# GlobalCandidateRef
# ---------------------------------------------------------------------------


class TestGlobalCandidateRef:
    def test_parse_round_trip(self) -> None:
        ref = GlobalCandidateRef.parse(LEFT_REF)
        assert ref.chunk_id == "CH001_C001"
        assert ref.local_candidate_id == "cand_char_001"
        assert ref.to_string() == LEFT_REF
        assert ref == GlobalCandidateRef.from_string(LEFT_REF)

    def test_parse_from_dict_round_trip(self) -> None:
        ref = GlobalCandidateRef.parse(LEFT_REF)
        assert GlobalCandidateRef.from_dict(ref.to_dict()) == ref

    def test_namespace(self) -> None:
        assert GlobalCandidateRef.parse("CH001_C001:cand_char_001").namespace == "character"
        assert GlobalCandidateRef.parse("CH001_C001:cand_loc_001").namespace == "location"
        assert (
            GlobalCandidateRef.parse("CH001_C001:cand_unres_001").namespace
            == "unresolved"
        )

    @pytest.mark.parametrize(
        "bad",
        [
            "CH01_C01:cand_char_001",  # chunk too short
            "CH001_C001:cand_event_001",  # not a coverage namespace
            "CH001_C001:cand_char_1",  # suffix too short
            "CH001_C001",  # no local id
            "cand_char_001",  # no chunk id
            "CH001_C001:cand_char_001:extra",  # trailing
            "",
            123,
            None,
        ],
    )
    def test_parse_rejects_invalid(self, bad: object) -> None:
        with pytest.raises(ReconciliationModelError):
            GlobalCandidateRef.parse(bad)

    def test_constructor_rejects_unknown_namespace(self) -> None:
        with pytest.raises(ReconciliationModelError):
            GlobalCandidateRef(chunk_id="CH001_C001", local_candidate_id="cand_event_001")

    def test_pattern_constant(self) -> None:
        # The documented pattern must actually accept a valid ref and reject a bad one.
        import re

        rx = re.compile(GLOBAL_CANDIDATE_REF_PATTERN)
        assert rx.fullmatch(LEFT_REF) is not None
        assert rx.fullmatch("CH001_C001:cand_event_001") is None


# ---------------------------------------------------------------------------
# Tracked EntityReconciliationProfile
# ---------------------------------------------------------------------------


class TestReconciliationProfile:
    def test_tracked_profile_loads(self) -> None:
        profile = load_entity_reconciliation_profile(RECON_PROFILE_PATH)
        assert profile.profile_id == "entity-reconciliation-v1"
        assert profile.prompt_id == A4_PROMPT_ID
        assert profile.prompt_version == 2
        assert profile.max_generation_rounds == RECONCILIATION_MAX_GENERATION_ROUNDS_V1

    def test_profile_hash_is_stable(self) -> None:
        profile = load_entity_reconciliation_profile(RECON_PROFILE_PATH)
        assert profile.profile_hash == content_hash(profile.to_dict())
        assert len(profile.profile_hash) == 64

    def test_profile_round_trip(self) -> None:
        profile = load_entity_reconciliation_profile(RECON_PROFILE_PATH)
        assert EntityReconciliationProfile.from_dict(profile.to_dict()) == profile

    def test_profile_carries_no_runtime_fields(self) -> None:
        profile = load_entity_reconciliation_profile(RECON_PROFILE_PATH)
        for banned in (
            "endpoint",
            "timeout",
            "provider_family",
            "request_model",
            "temperature",
            "model",
            "credential",
            "backend",
        ):
            assert banned not in profile.to_dict()

    def test_max_generation_rounds_ceiling_enforced(self) -> None:
        profile = load_entity_reconciliation_profile(RECON_PROFILE_PATH)
        data = profile.to_dict()
        data["max_generation_rounds"] = 3
        with pytest.raises(ReconciliationModelError):
            EntityReconciliationProfile.from_dict(data)

    def test_missing_file_fails_closed(self, tmp_path) -> None:
        with pytest.raises(ReconciliationModelError):
            load_entity_reconciliation_profile(tmp_path / "nope.yaml")

    def test_tracked_profile_passes_profile_schema(self) -> None:
        profile = load_entity_reconciliation_profile(RECON_PROFILE_PATH)
        assert (
            _validate(
                profile.to_dict(),
                _schema("entity-reconciliation-profile.schema.json"),
            )
            == []
        )

    def test_profile_schema_rejects_extra_field(self) -> None:
        data = load_entity_reconciliation_profile(RECON_PROFILE_PATH).to_dict()
        data["endpoint"] = "http://localhost:8080"
        assert (
            _validate(
                data, _schema("entity-reconciliation-profile.schema.json")
            )
            != []
        )

    def test_profile_schema_rejects_missing_field(self) -> None:
        data = load_entity_reconciliation_profile(RECON_PROFILE_PATH).to_dict()
        del data["blocking_policy_id"]
        assert (
            _validate(
                data, _schema("entity-reconciliation-profile.schema.json")
            )
            != []
        )

    def test_profile_schema_rejects_wrong_schema_version(self) -> None:
        data = load_entity_reconciliation_profile(RECON_PROFILE_PATH).to_dict()
        data["schema_version"] = 2
        assert (
            _validate(
                data, _schema("entity-reconciliation-profile.schema.json")
            )
            != []
        )

    def test_profile_schema_rejects_wrong_max_generation_rounds(self) -> None:
        data = load_entity_reconciliation_profile(RECON_PROFILE_PATH).to_dict()
        data["max_generation_rounds"] = 3
        assert (
            _validate(
                data, _schema("entity-reconciliation-profile.schema.json")
            )
            != []
        )

    def test_profile_schema_rejects_non_positive_version(self) -> None:
        data = load_entity_reconciliation_profile(RECON_PROFILE_PATH).to_dict()
        data["prompt_version"] = 0
        assert (
            _validate(
                data, _schema("entity-reconciliation-profile.schema.json")
            )
            != []
        )


class TestActiveReconciliationProfileBlockingIdentity:
    """The active tracked A4 profile (v2) must pin the production planner's
    blocking identity.

    Guards against a semantic-authority mismatch between the tracked
    ``EntityReconciliationProfile`` and
    ``reconciliation_planning.BLOCKING_POLICY_ID`` (the profile hash participates
    in A4 semantic identity / reuse, so a stale blocking identity is a real
    inconsistency, not a cosmetic one).
    """

    def test_active_v2_profile_pins_production_blocking_identity(self) -> None:
        profile = load_entity_reconciliation_profile(RECON_PROFILE_V2_PATH)
        assert profile.profile_id == "entity-reconciliation-v2"
        # The active tracked profile pins the production planner identity ...
        assert profile.blocking_policy_id == BLOCKING_POLICY_ID
        # ... and that identity is a4-blocking-v2 (not the historical v1).
        assert profile.blocking_policy_id == "a4-blocking-v2"
        assert BLOCKING_POLICY_ID == "a4-blocking-v2"

    def test_active_v2_profile_keeps_name_and_canonicalization_identity(self) -> None:
        profile = load_entity_reconciliation_profile(RECON_PROFILE_V2_PATH)
        assert (
            profile.name_normalization_policy_id
            == NAME_NORMALIZATION_POLICY_ID
            == "a4-name-normalization-v1"
        )
        assert (
            profile.canonicalization_policy_id
            == CANONICALIZATION_POLICY_ID
            == "a4-canonicalization-v1"
        )


# ---------------------------------------------------------------------------
# CandidateEntityIndex
# ---------------------------------------------------------------------------


class TestCandidateEntityIndex:
    def test_entry_round_trip(self) -> None:
        entry = make_index_entry()
        assert CandidateEntityIndexEntry.from_dict(entry.to_dict()) == entry

    def test_index_round_trip(self) -> None:
        index = CandidateEntityIndex(
            schema_version=1,
            entries=(
                make_index_entry(),
                make_index_entry(
                    candidate_ref=RIGHT_REF,
                    candidate_kind="location",
                    display_name_original="教室",
                    aliases_original=(),
                ),
            ),
        )
        assert CandidateEntityIndex.from_dict(index.to_dict()) == index

    def test_index_empty_ok(self) -> None:
        index = CandidateEntityIndex(schema_version=1, entries=())
        assert index.to_dict() == {"schema_version": 1, "entries": []}

    def test_bad_candidate_kind_rejected(self) -> None:
        with pytest.raises(ReconciliationModelError):
            make_index_entry(candidate_kind="event")

    def test_bad_candidate_ref_rejected(self) -> None:
        with pytest.raises(ReconciliationModelError):
            make_index_entry(candidate_ref="CH001_C001:cand_event_001")

    def test_bad_possible_ref_rejected(self) -> None:
        with pytest.raises(ReconciliationModelError):
            make_index_entry(possible_candidate_refs=("CH001_C001:cand_event_001",))

    def test_wrong_schema_version_rejected(self) -> None:
        with pytest.raises(ReconciliationModelError):
            CandidateEntityIndex(schema_version=2, entries=())

    def test_extra_key_rejected(self) -> None:
        data = make_index_entry().to_dict()
        data["extra"] = 1
        with pytest.raises(ReconciliationModelError):
            CandidateEntityIndexEntry.from_dict(data)


# ---------------------------------------------------------------------------
# Reconciliation decision (provider + persisted)
# ---------------------------------------------------------------------------


class TestReconciliationDecision:
    def test_item_round_trip(self) -> None:
        item = make_decision_item()
        assert ReconciliationDecisionItem.from_dict(item.to_dict()) == item

    def test_payload_round_trip(self) -> None:
        payload = ReconciliationDecisionPayload(decisions=(make_decision_item(),))
        assert ReconciliationDecisionPayload.from_dict(payload.to_dict()) == payload

    def test_item_canonical_order_enforced(self) -> None:
        with pytest.raises(ReconciliationModelError):
            ReconciliationDecisionItem(
                left_candidate_ref=RIGHT_REF,
                right_candidate_ref=LEFT_REF,
                decision="uncertain",
                reason_zh="x",
                evidence_refs=(),
            )

    def test_item_rejects_same_ref(self) -> None:
        with pytest.raises(ReconciliationModelError):
            make_decision_item(right_candidate_ref=LEFT_REF)

    def test_item_bad_decision_rejected(self) -> None:
        with pytest.raises(ReconciliationModelError):
            make_decision_item(decision="merge")

    def test_llm_decision_round_trip(self) -> None:
        decision = make_decision()
        assert ReconciliationDecision.from_dict(decision.to_dict()) == decision

    def test_llm_requires_prompt_and_provenance(self) -> None:
        with pytest.raises(ReconciliationModelError):
            make_decision(generation_provenance=None)
        with pytest.raises(ReconciliationModelError):
            make_decision(prompt_id=None)

    def test_deterministic_requires_null_prompt(self) -> None:
        decision = make_decision(
            method="deterministic",
            reason_code="same_exact_alias",
            prompt_id=None,
            prompt_version=None,
            generation_provenance=None,
        )
        assert ReconciliationDecision.from_dict(decision.to_dict()) == decision

    def test_deterministic_with_provenance_rejected(self) -> None:
        with pytest.raises(ReconciliationModelError):
            make_decision(
                method="deterministic",
                prompt_id=A4_PROMPT_ID,
                prompt_version=A4_PROMPT_VERSION,
                generation_provenance=make_provenance(),
            )

    def test_set_round_trip(self) -> None:
        decision_set = ReconciliationDecisionSet(
            schema_version=1, decisions=(make_decision(),)
        )
        assert (
            ReconciliationDecisionSet.from_dict(decision_set.to_dict())
            == decision_set
        )

    def test_bad_method_rejected(self) -> None:
        with pytest.raises(ReconciliationModelError):
            make_decision(method="guess")


# ---------------------------------------------------------------------------
# Canonical entities + registries
# ---------------------------------------------------------------------------


class TestCanonicalEntities:
    def test_character_registry_round_trip(self) -> None:
        registry = CanonicalCharacterRegistry(
            schema_version=1, entities=(make_character_entity(),)
        )
        assert CanonicalCharacterRegistry.from_dict(registry.to_dict()) == registry

    def test_location_registry_round_trip(self) -> None:
        registry = CanonicalLocationRegistry(
            schema_version=1, entities=(make_location_entity(),)
        )
        assert CanonicalLocationRegistry.from_dict(registry.to_dict()) == registry

    def test_entity_id_namespace_must_match_type(self) -> None:
        # A location id in a character entity is rejected.
        with pytest.raises(ReconciliationModelError):
            CanonicalEntity(
                canonical_id="loc_0001",
                entity_type="character",
                display_name_original="林晚",
                aliases_original=(),
                candidate_refs=(LEFT_REF,),
                first_appearance_candidate_ref=LEFT_REF,
            )

    def test_character_registry_rejects_location_entity(self) -> None:
        with pytest.raises(ReconciliationModelError):
            CanonicalCharacterRegistry(schema_version=1, entities=(make_location_entity(),))

    def test_location_registry_entity_type_const(self) -> None:
        registry = CanonicalLocationRegistry(schema_version=1, entities=(make_location_entity(),))
        assert registry.to_dict()["entity_type"] == "location"
        data = registry.to_dict()
        data["entity_type"] = "character"
        with pytest.raises(ReconciliationModelError):
            CanonicalLocationRegistry.from_dict(data)

    def test_empty_candidate_refs_rejected(self) -> None:
        with pytest.raises(ReconciliationModelError):
            make_character_entity(candidate_refs=())


# ---------------------------------------------------------------------------
# Unresolved entities
# ---------------------------------------------------------------------------


class TestUnresolvedEntities:
    def test_round_trip(self) -> None:
        entity = make_unresolved()
        assert UnresolvedEntity.from_dict(entity.to_dict()) == entity

    def test_set_round_trip(self) -> None:
        entity_set = UnresolvedEntitySet(schema_version=1, entities=(make_unresolved(),))
        assert UnresolvedEntitySet.from_dict(entity_set.to_dict()) == entity_set

    def test_bad_kind_rejected(self) -> None:
        with pytest.raises(ReconciliationModelError):
            make_unresolved(entity_kind="place")

    def test_all_allowed_kinds_accepted(self) -> None:
        for kind in ("character", "location", "person", "other", "unknown"):
            make_unresolved(entity_kind=kind)

    def test_bad_unresolved_id_rejected(self) -> None:
        with pytest.raises(ReconciliationModelError):
            make_unresolved(unresolved_id="unres_1")


# ---------------------------------------------------------------------------
# EntityMap
# ---------------------------------------------------------------------------


class TestEntityMap:
    def test_round_trip(self) -> None:
        entity_map = make_entity_map()
        assert EntityMap.from_dict(entity_map.to_dict()) == entity_map

    def test_a3_input_round_trip(self) -> None:
        a3 = make_a3_input()
        assert A3InputIdentity.from_dict(a3.to_dict()) == a3

    def test_resolved_entry_requires_canonical(self) -> None:
        with pytest.raises(ReconciliationModelError):
            make_entry(canonical_id=None)

    def test_unresolved_entry_requires_unresolved_id(self) -> None:
        with pytest.raises(ReconciliationModelError):
            make_entry(status="unresolved", canonical_id=None, unresolved_id=None)

    def test_both_set_rejected(self) -> None:
        with pytest.raises(ReconciliationModelError):
            make_entry(canonical_id="char_0001", unresolved_id="unres_0001")

    def test_neither_set_rejected(self) -> None:
        with pytest.raises(ReconciliationModelError):
            make_entry(status="resolved", canonical_id=None, unresolved_id=None)

    def test_a3_input_requires_min_one_extraction(self) -> None:
        with pytest.raises(ReconciliationModelError):
            make_a3_input(candidate_extraction_refs=())

    def test_a3_input_bad_hash_rejected(self) -> None:
        with pytest.raises(ReconciliationModelError):
            make_a3_input(extraction_profile_hash="not-a-hash")


# ---------------------------------------------------------------------------
# JSON Schema parity (six persisted schemas)
# ---------------------------------------------------------------------------


class TestSchemaParity:
    def test_candidate_entity_index(self) -> None:
        index = CandidateEntityIndex(
            schema_version=1, entries=(make_index_entry(),)
        )
        assert _validate(index.to_dict(), _schema("candidate-entity-index.schema.json")) == []

    def test_reconciliation_decision_set(self) -> None:
        decision_set = ReconciliationDecisionSet(
            schema_version=1,
            decisions=(
                make_decision(),
                make_decision(
                    decision_id="dec_0002",
                    method="deterministic",
                    reason_code="same_exact_alias",
                    prompt_id=None,
                    prompt_version=None,
                    generation_provenance=None,
                ),
            ),
        )
        assert (
            _validate(
                decision_set.to_dict(),
                _schema("reconciliation-decision-set.schema.json"),
            )
            == []
        )

    def test_character_registry(self) -> None:
        registry = CanonicalCharacterRegistry(
            schema_version=1, entities=(make_character_entity(),)
        )
        assert (
            _validate(
                registry.to_dict(),
                _schema("canonical-character-registry.schema.json"),
            )
            == []
        )

    def test_location_registry(self) -> None:
        registry = CanonicalLocationRegistry(
            schema_version=1, entities=(make_location_entity(),)
        )
        assert (
            _validate(
                registry.to_dict(),
                _schema("canonical-location-registry.schema.json"),
            )
            == []
        )

    def test_unresolved_set(self) -> None:
        entity_set = UnresolvedEntitySet(schema_version=1, entities=(make_unresolved(),))
        assert _validate(entity_set.to_dict(), _schema("unresolved-entity-set.schema.json")) == []

    def test_entity_map(self) -> None:
        entity_map = make_entity_map()
        assert _validate(entity_map.to_dict(), _schema("entity-map.schema.json")) == []

    def test_entity_map_rejects_both_set(self) -> None:
        # The Python model rejects the both-set entry at construction, so no
        # valid EntityMap can carry it; the schema additionally rejects it.
        with pytest.raises(ReconciliationModelError):
            make_entry(canonical_id="char_0001", unresolved_id="unres_0001")


# ---------------------------------------------------------------------------
# A4SemanticIdentity (EntityMap v2)
# ---------------------------------------------------------------------------


class TestA4SemanticIdentity:
    def test_round_trip(self) -> None:
        ident = make_semantic_identity(semantic_request_hashes=(H, H2))
        assert A4SemanticIdentity.from_dict(ident.to_dict()) == ident

    def test_empty_request_hashes_allowed(self) -> None:
        ident = make_semantic_identity(semantic_request_hashes=())
        assert ident.semantic_request_hashes == ()

    @pytest.mark.parametrize("field", [
        "reconciliation_profile_hash",
        "semantic_profile_hash",
        "prompt_content_hash",
        "output_schema_hash",
        "plan_hash",
    ])
    def test_bad_hash_rejected(self, field: str) -> None:
        with pytest.raises(ReconciliationModelError):
            make_semantic_identity(**{field: "not-a-hash"})

    def test_non_hash_request_hash_rejected(self) -> None:
        with pytest.raises(ReconciliationModelError):
            make_semantic_identity(semantic_request_hashes=("nothex",))

    def test_duplicate_request_hashes_rejected(self) -> None:
        with pytest.raises(ReconciliationModelError):
            make_semantic_identity(semantic_request_hashes=(H, H))

    def test_schema_parity(self) -> None:
        # Validate the full EntityMap (which embeds semantic_identity) against
        # the v2 schema; a valid semantic_identity must be accepted.
        entity_map = make_entity_map(
            semantic_identity=make_semantic_identity(semantic_request_hashes=(H, H2))
        )
        assert _validate(entity_map.to_dict(), _schema("entity-map.schema.json")) == []

    def test_schema_rejects_bad_request_hash(self) -> None:
        # A semantic_request_hashes item that is not a SHA-256 must be rejected
        # by the schema (build the dict manually to bypass the Python model).
        entity_map = make_entity_map(
            semantic_identity=make_semantic_identity(semantic_request_hashes=(H,))
        )
        data = entity_map.to_dict()
        data["semantic_identity"]["semantic_request_hashes"] = ["not-a-hash"]
        assert _validate(data, _schema("entity-map.schema.json")) != []

    def test_entity_map_requires_semantic_identity(self) -> None:
        with pytest.raises(ReconciliationModelError):
            make_entity_map(semantic_identity=None)


# ---------------------------------------------------------------------------
# Persisted ReconciliationDecision schema <-> Python parity
# ---------------------------------------------------------------------------


def _validate_raw_decision_set(decisions: list[dict]) -> list[str]:
    # Validate a raw persisted decision-set dict against the schema (bypassing
    # the Python model) to prove the schema enforces the same method/provenance
    # consistency as the Python model.
    return _validate(
        {"schema_version": 1, "decisions": decisions},
        _schema("reconciliation-decision-set.schema.json"),
    )


class TestDecisionSchemaParity:
    def test_accepts_llm_decision(self) -> None:
        decision_set = ReconciliationDecisionSet(
            schema_version=1, decisions=(make_decision(),)
        )
        assert (
            _validate(
                decision_set.to_dict(),
                _schema("reconciliation-decision-set.schema.json"),
            )
            == []
        )

    def test_accepts_deterministic_decision(self) -> None:
        decision = make_decision(
            method="deterministic",
            reason_code="same_exact_alias",
            prompt_id=None,
            prompt_version=None,
            generation_provenance=None,
        )
        decision_set = ReconciliationDecisionSet(schema_version=1, decisions=(decision,))
        assert (
            _validate(
                decision_set.to_dict(),
                _schema("reconciliation-decision-set.schema.json"),
            )
            == []
        )

    def test_accepts_manual_decision(self) -> None:
        decision = make_decision(
            method="manual",
            reason_code="manual_review",
            prompt_id=None,
            prompt_version=None,
            generation_provenance=None,
        )
        decision_set = ReconciliationDecisionSet(schema_version=1, decisions=(decision,))
        assert (
            _validate(
                decision_set.to_dict(),
                _schema("reconciliation-decision-set.schema.json"),
            )
            == []
        )

    def test_rejects_llm_with_null_provenance(self) -> None:
        base = make_decision().to_dict()
        base["generation_provenance"] = None
        assert _validate_raw_decision_set([base]) != []

    def test_rejects_llm_with_null_prompt_id(self) -> None:
        base = make_decision().to_dict()
        base["prompt_id"] = None
        assert _validate_raw_decision_set([base]) != []

    def test_rejects_llm_with_null_prompt_version(self) -> None:
        base = make_decision().to_dict()
        base["prompt_version"] = None
        assert _validate_raw_decision_set([base]) != []

    def test_rejects_deterministic_with_llm_provenance(self) -> None:
        base = make_decision().to_dict()
        base["method"] = "deterministic"
        # method is deterministic but prompt identity + provenance are present.
        assert _validate_raw_decision_set([base]) != []

    def test_rejects_manual_with_llm_provenance(self) -> None:
        base = make_decision().to_dict()
        base["method"] = "manual"
        assert _validate_raw_decision_set([base]) != []


# ---------------------------------------------------------------------------
# Merge-graph pair-ref hardening (char/loc only; cand_unres_* excluded)
# ---------------------------------------------------------------------------

UNRES_PAIR_REF = "CH001_C001:cand_unres_001"


class TestMergeGraphPairRefs:
    def test_merge_graph_pattern_accepts_char_and_loc(self) -> None:
        import re

        rx = re.compile(MERGE_GRAPH_CANDIDATE_REF_PATTERN)
        assert rx.fullmatch("CH001_C001:cand_char_001") is not None
        assert rx.fullmatch("CH001_C001:cand_loc_001") is not None
        assert rx.fullmatch(UNRES_PAIR_REF) is None

    def test_global_ref_still_accepts_unres(self) -> None:
        # The coverage-universe global ref (and GlobalCandidateRef) still accept
        # cand_unres_*; only the merge-graph pair refs exclude it.
        assert GlobalCandidateRef.parse(UNRES_PAIR_REF).namespace == "unresolved"

    def test_provider_item_rejects_unres_pair_ref(self) -> None:
        with pytest.raises(ReconciliationModelError):
            make_decision_item(left_candidate_ref=UNRES_PAIR_REF)

    def test_provider_item_rejects_unres_right_ref(self) -> None:
        with pytest.raises(ReconciliationModelError):
            make_decision_item(right_candidate_ref=UNRES_PAIR_REF)

    def test_persisted_decision_rejects_unres_pair_ref(self) -> None:
        with pytest.raises(ReconciliationModelError):
            make_decision(left_candidate_ref=UNRES_PAIR_REF)

    def test_provider_schema_rejects_unres_pair_ref(self) -> None:
        payload = {
            "decisions": [
                {
                    "left_candidate_ref": UNRES_PAIR_REF,
                    "right_candidate_ref": RIGHT_REF,
                    "decision": "uncertain",
                    "reason_zh": "x",
                    "evidence_refs": [],
                }
            ]
        }
        assert (
            _validate(
                payload, _schema("reconciliation-decision-payload.schema.json")
            )
            != []
        )

    def test_persisted_schema_rejects_unres_pair_ref(self) -> None:
        base = make_decision().to_dict()
        base["left_candidate_ref"] = UNRES_PAIR_REF
        assert _validate_raw_decision_set([base]) != []


# ---------------------------------------------------------------------------
# Provider schema strict-mode discipline
# ---------------------------------------------------------------------------

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


def _collect_keys(node: object, found: set[str]) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            found.add(key)
            _collect_keys(value, found)
    elif isinstance(node, list):
        for item in node:
            _collect_keys(item, found)


class TestProviderSchema:
    def test_provider_schema_is_strict_mode_clean(self) -> None:
        schema = _schema("reconciliation-decision-payload.schema.json")
        found: set[str] = set()
        _collect_keys(schema, found)
        assert found & _FORBIDDEN_PROVIDER_KEYS == set()

    def test_provider_schema_accepts_valid_payload(self) -> None:
        payload = ReconciliationDecisionPayload(decisions=(make_decision_item(),))
        assert (
            _validate(
                payload.to_dict(),
                _schema("reconciliation-decision-payload.schema.json"),
            )
            == []
        )

    def test_provider_schema_rejects_bad_decision(self) -> None:
        payload = {
            "decisions": [
                {
                    "left_candidate_ref": LEFT_REF,
                    "right_candidate_ref": RIGHT_REF,
                    "decision": "merge",
                    "reason_zh": "x",
                    "evidence_refs": [],
                }
            ]
        }
        assert (
            _validate(
                payload, _schema("reconciliation-decision-payload.schema.json")
            )
            != []
        )

    def test_provider_schema_rejects_extra_key(self) -> None:
        payload = ReconciliationDecisionPayload(decisions=())
        data = payload.to_dict()
        data["extra"] = 1
        assert (
            _validate(
                data, _schema("reconciliation-decision-payload.schema.json")
            )
            != []
        )


# ---------------------------------------------------------------------------
# Tracked A4 semantic LLM profile + prompt
# ---------------------------------------------------------------------------


class TestTrackedA4Assets:
    def test_semantic_profile_loads_and_is_schema_valid(self) -> None:
        profile = load_semantic_profile(A4_LLM_PROFILE_PATH)
        assert profile.profile_id == "entity-reconciliation-llm-v1"
        schema = _schema("llm-semantic-profile.schema.json")
        assert _validate(profile.to_dict(), schema) == []

    def test_semantic_profile_is_backend_independent(self) -> None:
        profile = load_semantic_profile(A4_LLM_PROFILE_PATH)
        data = profile.to_dict()
        # No backend routing identity may appear in the semantic profile.
        for banned in ("provider_family", "model", "request_model", "endpoint", "timeout"):
            assert banned not in data

    def test_prompt_loads_with_pinned_hash(self) -> None:
        registry = PromptRegistry(PROMPTS_STORY_DIR)
        spec = registry.load(A4_PROMPT_ID, version=A4_PROMPT_VERSION)
        assert spec.content_hash == PROMPT_CONTENT_HASH
        assert tuple(spec.required_variables) == (
            "block_id",
            "candidate_packets_json",
            "requested_pairs_json",
        )

    def test_prompt_renders_with_exact_variables(self) -> None:
        from short_drama.llm import render_prompt

        registry = PromptRegistry(PROMPTS_STORY_DIR)
        spec = registry.load(A4_PROMPT_ID, version=A4_PROMPT_VERSION)
        rendered = render_prompt(
            spec,
            {
                "block_id": "block_0001",
                "candidate_packets_json": "[]",
                "requested_pairs_json": "[]",
            },
        )
        assert rendered.prompt_id == A4_PROMPT_ID
        assert len(rendered.rendered_prompt_hash) == 64

    def test_prompt_rejects_extra_variable(self) -> None:
        from short_drama.llm import render_prompt

        registry = PromptRegistry(PROMPTS_STORY_DIR)
        spec = registry.load(A4_PROMPT_ID, version=A4_PROMPT_VERSION)
        with pytest.raises(LLMPromptError):
            render_prompt(
                spec,
                {
                    "block_id": "block_0001",
                    "candidate_packets_json": "[]",
                    "requested_pairs_json": "[]",
                    "unexpected": "x",
                },
            )

    def test_prompt_rejects_missing_variable(self) -> None:
        from short_drama.llm import render_prompt

        registry = PromptRegistry(PROMPTS_STORY_DIR)
        spec = registry.load(A4_PROMPT_ID, version=A4_PROMPT_VERSION)
        with pytest.raises(LLMPromptError):
            render_prompt(
                spec,
                {
                    "block_id": "block_0001",
                    "candidate_packets_json": "[]",
                },
            )


class TestPromptV2Contract:
    """Issue #39: prompt v2 aligns the provider evidence contract with the
    strict A4C endpoint-only validator.

    These are text-level contract assertions on the tracked prompt v2 (and the
    tracked reconciliation profile that pins it). The strict validator itself
    is covered in ``test_story_a4c_semantic.py``.
    """

    @staticmethod
    def _system_text() -> str:
        registry = PromptRegistry(PROMPTS_STORY_DIR)
        spec = registry.load(A4_PROMPT_ID, version=2)
        assert spec.version == 2
        # Collapse wrapping whitespace so phrase assertions are stable.
        return " ".join(spec.system_template.split())

    @staticmethod
    def _user_text() -> str:
        registry = PromptRegistry(PROMPTS_STORY_DIR)
        spec = registry.load(A4_PROMPT_ID, version=2)
        return " ".join(spec.user_template.split())

    def test_v1_preserved_unchanged(self) -> None:
        """v1 is preserved and still carries its original pinned hash."""
        registry = PromptRegistry(PROMPTS_STORY_DIR)
        v1 = registry.load(A4_PROMPT_ID, version=1)
        assert (
            v1.content_hash
            == "e9bf883330e449f7b19553df4c21b3ff38c24e5f766a93ca1ce483bcf885bc1d"
        )

    def test_v2_pinned_hash_deterministic(self) -> None:
        registry = PromptRegistry(PROMPTS_STORY_DIR)
        spec = registry.load(A4_PROMPT_ID, version=2)
        assert spec.content_hash == PROMPT_CONTENT_HASH
        assert tuple(spec.required_variables) == (
            "block_id",
            "candidate_packets_json",
            "requested_pairs_json",
        )

    def test_requires_endpoint_only_evidence(self) -> None:
        text = self._system_text()
        assert "Evidence is PAIR-ENDPOINT-SCOPED" in text
        assert "taken from those two endpoint packets" in text

    def test_forbids_evidence_from_other_block_candidates(self) -> None:
        text = self._system_text()
        assert (
            "No other candidate packet in this block may contribute an "
            "evidence_ref to this decision"
            in text
        )

    def test_locates_both_endpoint_packets(self) -> None:
        text = self._system_text()
        assert "exactly equals this decision's `left_candidate_ref`" in text
        assert "exactly equals this decision's `right_candidate_ref`" in text

    def test_requires_exact_four_field_copy(self) -> None:
        text = self._system_text()
        assert "Copy every selected evidence_ref's four fields EXACTLY" in text
        for field in ("`paragraph_id`", "`role`", "`strength`", "`excerpt`"):
            assert field in text
        assert "Never paraphrase, normalize, shorten, expand, repair, or substitute" in text

    def test_requires_self_verify_before_citing(self) -> None:
        text = self._system_text()
        assert "match EXACTLY one existing evidence item" in text
        assert "If you cannot verify an exact match to an endpoint item, do not cite it" in text

    def test_documents_null_excerpt(self) -> None:
        text = self._system_text()
        assert "`excerpt` may be a string OR null" in text
        assert "`\"excerpt\": null`, copy `null` exactly" in text

    def test_empty_evidence_still_allowed(self) -> None:
        text = self._system_text()
        assert "an empty `evidence_refs` list" in text

    def test_preserves_pair_count_order_and_refs(self) -> None:
        text = self._system_text()
        assert "Emit exactly one decision per requested pair" in text
        assert (
            "preserving the given left and right candidate references exactly"
            in text
        )

    def test_user_scope_is_pair_endpoint_scoped(self) -> None:
        text = self._user_text()
        assert (
            "A decision may ONLY cite evidence from the two endpoint packets"
            in text
        )

    def test_tracked_profile_pins_v2(self) -> None:
        profile = load_entity_reconciliation_profile(RECON_PROFILE_PATH)
        assert profile.prompt_id == A4_PROMPT_ID
        assert profile.prompt_version == 2
        # The pinned prompt must load deterministically at the pinned version
        # and match the tracked content hash.
        registry = PromptRegistry(PROMPTS_STORY_DIR)
        spec = registry.load(profile.prompt_id, version=profile.prompt_version)
        assert spec.content_hash == PROMPT_CONTENT_HASH
        # Profile hash is deterministic and reflects the pinned prompt version.
        assert profile.profile_hash == content_hash(profile.to_dict())
        assert len(profile.profile_hash) == 64


class TestExports:
    def test_all_a4a_names_importable_from_package(self) -> None:
        import short_drama.story as story

        for name in (
            "GlobalCandidateRef",
            "CandidateEntityIndex",
            "ReconciliationDecision",
            "ReconciliationDecisionSet",
            "CanonicalCharacterRegistry",
            "CanonicalLocationRegistry",
            "UnresolvedEntitySet",
            "EntityMap",
            "EntityReconciliationProfile",
            "ReconciliationModelError",
            "load_entity_reconciliation_profile",
        ):
            assert hasattr(story, name), name
