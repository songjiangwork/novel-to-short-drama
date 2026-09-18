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
A4_PROMPT_VERSION = 1
PROMPT_CONTENT_HASH = (
    "e9bf883330e449f7b19553df4c21b3ff38c24e5f766a93ca1ce483bcf885bc1d"
)

RECON_PROFILE_PATH = PROFILES_DIR / "entity_reconciliation_v1.yaml"
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
        "prompt_version": 1,
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
        "prompt_version": 1,
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
        assert profile.prompt_version == 1
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
                prompt_version=1,
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
