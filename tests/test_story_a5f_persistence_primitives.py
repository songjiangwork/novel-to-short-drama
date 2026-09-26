"""v1.2 A5F1 — consolidation persistence primitives tests.

Covers the A5F1 primitives layer (``short_drama.story.consolidation_persistence``):

  * deterministic A5 artifact identity (base + six leaf outputs + manifest +
    A5 validation report + the frozen CURRENT-pointer id);
  * one shared run revision across the eight A5 artifact slots, with
    orphan-aware next-revision allocation;
  * immutable A5 persist / load round-trips for all seven artifacts;
  * fail-closed loaders: wrong artifact_type, wrong logical artifact id,
    wrong schema_version, corrupt / content-hash-mismatching artifact,
    non-canonical payload (from_dict -> to_dict differs), tracked
    JSON-Schema violation, and missing artifact;
  * the private manifest reference verifier (root identity, leaf logical id /
    type / shared revision, entity_map type, and leaf resolvability);
  * the exact deterministic A5 PASS ValidationReport and its exact-matching
    verification (match / missing / mismatch / non-PASS).

This slice deliberately performs NO provider calls, NO CURRENT publication, NO
pointer write, and NO reuse. It asserts that nothing A5F1 does writes the
CURRENT pointer and that a freshly re-opened store reproduces identical
artifacts.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from short_drama.artifacts import (
    ArtifactNotFoundError,
    ArtifactRef,
    FileArtifactStore,
    canonical_json_bytes,
)
from short_drama.foundation import (
    VALIDATION_REPORT_ARTIFACT_TYPE,
    LineageRef,
    ValidationFinding,
    ValidationReport,
    ValidationSeverity,
    ValidationResult,
    persist_validation_report,
)
from short_drama.io import load_json
from short_drama.paths import SCHEMAS_DIR
from short_drama.story import (
    A5SemanticIdentity,
    A5UpstreamIdentity,
    CanonicalEvent,
    CanonicalEventSet,
    CanonicalFact,
    CanonicalFactSet,
    CanonicalRelationship,
    CanonicalRelationshipSet,
    ConsolidationCandidateIndex,
    ConsolidationCoverageSummary,
    ConsolidationDecisionSet,
    ConsolidationManifest,
    EvidenceRef,
    EventSemanticDecision,
    FactSemanticDecision,
    IndexedEventCandidate,
    IndexedFactCandidate,
    IndexedRelationshipCandidate,
    OutputSchemaAssetIdentity,
    PromptAssetIdentity,
    RelationshipSemanticDecision,
    RelationshipState,
    StoryConflict,
    StoryConflictSet,
)
from short_drama.story import consolidation_persistence as cp
from short_drama.story.reconciliation import A3InputIdentity
from short_drama.story.consolidation_persistence import (
    CANONICAL_FACT_SET_ARTIFACT_TYPE,
    CONSOLIDATION_CANDIDATE_INDEX_ARTIFACT_TYPE,
    CONSOLIDATION_MANIFEST_ARTIFACT_TYPE,
    a5_base_artifact_id,
    a5_pointer_id,
    a5_validation_artifact_id,
    build_a5_validation_report,
    canonical_event_set_artifact_id,
    canonical_fact_set_artifact_id,
    canonical_relationship_set_artifact_id,
    consolidation_candidate_index_artifact_id,
    consolidation_decision_set_artifact_id,
    consolidation_manifest_artifact_id,
    load_canonical_event_set,
    load_canonical_fact_set,
    load_canonical_relationship_set,
    load_consolidation_candidate_index,
    load_consolidation_decision_set,
    load_consolidation_manifest,
    load_story_conflict_set,
    next_a5_revision,
    persist_canonical_event_set,
    persist_canonical_fact_set,
    persist_canonical_relationship_set,
    persist_consolidation_candidate_index,
    persist_consolidation_decision_set,
    persist_consolidation_manifest,
    persist_story_conflict_set,
    story_conflict_set_artifact_id,
)
from short_drama.story.errors import StoryIntegrityError
from short_drama.story.reconciliation_persistence import ENTITY_MAP_ARTIFACT_TYPE


PROJECT_ID = "proj_0001"
DOCUMENT_ID = "doc_0001"
PROFILE_ID = "consolidation-v1"
H = "a" * 64
H2 = "b" * 64
BASE = a5_base_artifact_id(PROJECT_ID, DOCUMENT_ID, PROFILE_ID)

FACTOR_LEFT = "CH003_C005:cand_fact_012"
FACTOR_RIGHT = "CH007_C009:cand_fact_013"
EVENT_LEFT = "CH003_C005:cand_evt_012"
EVENT_RIGHT = "CH007_C009:cand_evt_013"
REL_LEFT = "CH003_C005:cand_rel_012"
REL_RIGHT = "CH007_C009:cand_rel_013"


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def make_artifact_ref(
    artifact_type: str = "candidate_extraction",
    artifact_id: str = "cand-extraction-0001",
    revision: int = 1,
    content_hash: str = H,
) -> ArtifactRef:
    return ArtifactRef(
        artifact_type=artifact_type,
        artifact_id=artifact_id,
        revision=revision,
        content_hash=content_hash,
    )


def make_evidence() -> EvidenceRef:
    return EvidenceRef(
        paragraph_id="CH001_P0017",
        role="primary",
        strength="explicit",
        excerpt="林晚走进了教室。",
    )


def make_fact_candidate() -> IndexedFactCandidate:
    return IndexedFactCandidate(
        global_candidate_ref=FACTOR_LEFT,
        chunk_id="CH003_C005",
        local_candidate_id="cand_fact_012",
        source_order_key="CH003_C005:P0017",
        fact_type="identity",
        statement_zh="林晚是一名学生。",
        subject_refs=("char_0001",),
        object_refs=(),
        evidence_strength="explicit",
        evidence_refs=(make_evidence(),),
        candidate_extraction_ref=make_artifact_ref(),
    )


def make_event_candidate() -> IndexedEventCandidate:
    return IndexedEventCandidate(
        global_candidate_ref=EVENT_LEFT,
        chunk_id="CH003_C005",
        local_candidate_id="cand_evt_012",
        source_order_key="CH003_C005:P0017",
        summary_zh="林晚走进了教室。",
        participants=("char_0001",),
        locations=("loc_0001",),
        temporal_mode="normal",
        evidence_strength="explicit",
        evidence_refs=(make_evidence(),),
        candidate_extraction_ref=make_artifact_ref(),
    )


def make_relationship_candidate() -> IndexedRelationshipCandidate:
    return IndexedRelationshipCandidate(
        global_candidate_ref=REL_LEFT,
        chunk_id="CH003_C005",
        local_candidate_id="cand_rel_012",
        source_order_key="CH003_C005:P0017",
        source_entity_ref="char_0001",
        target_entity_ref="char_0002",
        relationship_type_zh="朋友",
        state_zh=None,
        direction="directed",
        evidence_strength="explicit",
        evidence_refs=(make_evidence(),),
        candidate_extraction_ref=make_artifact_ref(),
    )


def make_candidate_index() -> ConsolidationCandidateIndex:
    return ConsolidationCandidateIndex(
        schema_version=1,
        facts=(make_fact_candidate(),),
        events=(make_event_candidate(),),
        relationships=(make_relationship_candidate(),),
    )


def make_fact_decision() -> FactSemanticDecision:
    return FactSemanticDecision(
        decision_id="factd_000001",
        left_candidate_ref=FACTOR_LEFT,
        right_candidate_ref=FACTOR_RIGHT,
        decision="same_fact",
        method="deterministic",
        reason_zh="两条事实表述同一身份。",
        evidence_refs=(),
        prompt_id=None,
        prompt_version=None,
        generation_provenance=None,
    )


def make_event_decision() -> EventSemanticDecision:
    return EventSemanticDecision(
        decision_id="evtd_000001",
        left_candidate_ref=EVENT_LEFT,
        right_candidate_ref=EVENT_RIGHT,
        decision="same_event",
        method="deterministic",
        reason_zh="同一事件。",
        evidence_refs=(),
        prompt_id=None,
        prompt_version=None,
        generation_provenance=None,
    )


def make_relationship_decision() -> RelationshipSemanticDecision:
    return RelationshipSemanticDecision(
        decision_id="reld_000001",
        left_candidate_ref=REL_LEFT,
        right_candidate_ref=REL_RIGHT,
        decision="uncertain",
        method="manual",
        reason_zh="证据不足。",
        evidence_refs=(),
        prompt_id=None,
        prompt_version=None,
        generation_provenance=None,
    )


def make_decision_set() -> ConsolidationDecisionSet:
    return ConsolidationDecisionSet(
        schema_version=1,
        fact_decisions=(make_fact_decision(),),
        event_decisions=(make_event_decision(),),
        relationship_decisions=(make_relationship_decision(),),
    )


def make_canonical_fact_set() -> CanonicalFactSet:
    fact = CanonicalFact(
        fact_id="fact_000001",
        fact_type="identity",
        statement_zh="林晚是一名学生。",
        subject_refs=("char_0001",),
        object_refs=(),
        candidate_fact_refs=(FACTOR_LEFT,),
        evidence_refs=(make_evidence(),),
        first_source_order="CH003_C005:P0017",
        continuity_relevant=True,
    )
    return CanonicalFactSet(
        schema_version=1,
        facts=(fact,),
        state_transitions=(),
    )


def make_canonical_event_set() -> CanonicalEventSet:
    event = CanonicalEvent(
        event_id="evt_000001",
        narrative_order=1,
        summary_zh="林晚走进了教室。",
        participants=("char_0001",),
        locations=("loc_0001",),
        temporal_mode="normal",
        candidate_event_refs=(EVENT_LEFT,),
        evidence_refs=(make_evidence(),),
        first_source_order="CH003_C005:P0017",
    )
    return CanonicalEventSet(schema_version=1, events=(event,))


def make_canonical_relationship_set() -> CanonicalRelationshipSet:
    rel_state = _make_relationship_state()
    rel = CanonicalRelationship(
        relationship_id="rel_000001",
        source_entity_ref="char_0001",
        target_entity_ref="char_0002",
        direction="directed",
        relationship_type_zh="朋友",
        candidate_relationship_refs=(REL_LEFT,),
        state_history=(rel_state,),
        first_source_order="CH003_C005:P0017",
    )
    return CanonicalRelationshipSet(schema_version=1, relationships=(rel,))


def _make_relationship_state():
    return RelationshipState(
        state_zh="好友",
        candidate_relationship_refs=(REL_LEFT,),
        evidence_refs=(make_evidence(),),
        narrative_order=1,
    )


def make_story_conflict_set() -> StoryConflictSet:
    conflict = StoryConflict(
        conflict_id="conf_000001",
        conflict_kind="fact_conflict",
        fact_ids=("fact_000001", "fact_000002"),
        relationship_ids=(),
        candidate_refs=(FACTOR_LEFT,),
        decision_refs=("factd_000001",),
        evidence_refs=(make_evidence(),),
        status="unresolved",
    )
    return StoryConflictSet(schema_version=1, conflicts=(conflict,))


def make_a3_input() -> A3InputIdentity:
    return A3InputIdentity(
        source_document_ref=make_artifact_ref(
            "source_document", "src-doc-0001"
        ),
        chunk_manifest_ref=make_artifact_ref("chunk_manifest", "chunk-manifest-0001"),
        candidate_extraction_refs=(
            make_artifact_ref("candidate_extraction", "cand-ext-0001"),
            make_artifact_ref("candidate_extraction", "cand-ext-0002"),
        ),
        extraction_profile_id="story-extraction-v1",
        extraction_profile_hash=H,
    )


def make_semantic_identity() -> A5SemanticIdentity:
    return A5SemanticIdentity(
        consolidation_profile_id=PROFILE_ID,
        consolidation_profile_hash=H,
        fact_semantic_profile_id="consolidation-llm-v1",
        fact_semantic_profile_hash=H,
        event_semantic_profile_id="consolidation-llm-v1",
        event_semantic_profile_hash=H,
        relationship_semantic_profile_id="consolidation-llm-v1",
        relationship_semantic_profile_hash=H,
        prompt_identities=(
            PromptAssetIdentity(
                prompt_id="consolidation-fact-prompt",
                prompt_version=1,
                prompt_content_hash=H,
            ),
        ),
        output_schema_identities=(
            OutputSchemaAssetIdentity(
                schema_id="consolidation-fact-selector-payload",
                schema_version=1,
                schema_hash=H,
            ),
        ),
        plan_hash=H,
        semantic_request_hashes=(H, H2),
    )


def make_manifest(leaf_refs: dict, entity_map_ref: ArtifactRef) -> ConsolidationManifest:
    return ConsolidationManifest(
        schema_version=1,
        project_id=PROJECT_ID,
        document_id=DOCUMENT_ID,
        entity_map_ref=entity_map_ref,
        consolidation_candidate_index_ref=leaf_refs["candidate_index"],
        consolidation_decision_set_ref=leaf_refs["decision_set"],
        canonical_fact_set_ref=leaf_refs["fact_set"],
        canonical_event_set_ref=leaf_refs["event_set"],
        canonical_relationship_set_ref=leaf_refs["relationship_set"],
        story_conflict_set_ref=leaf_refs["conflict_set"],
        semantic_identity=make_semantic_identity(),
        upstream_identity=A5UpstreamIdentity(a3_input=make_a3_input()),
        coverage_summary=ConsolidationCoverageSummary(
            fact_candidate_count=1,
            event_candidate_count=1,
            relationship_candidate_count=1,
            canonical_fact_count=1,
            canonical_event_count=1,
            canonical_relationship_count=1,
            uncertain_decision_count=1,
            story_conflict_count=1,
        ),
    )


def make_all(
    store: FileArtifactStore, *, revision: int = 1, entity_map_ref: ArtifactRef | None = None
):
    """Persist all six A5 leaves + manifest at one shared revision.

    Returns (manifest, manifest_ref, leaf_refs, entity_map_ref).
    """
    if entity_map_ref is None:
        entity_map_ref = make_artifact_ref(
            ENTITY_MAP_ARTIFACT_TYPE,
            "proj_0001.doc_0001.a4.reconciliation-v1",
            revision=1,
            content_hash=H,
        )

    index = make_candidate_index()
    decision_set = make_decision_set()
    fact_set = make_canonical_fact_set()
    event_set = make_canonical_event_set()
    relationship_set = make_canonical_relationship_set()
    conflict_set = make_story_conflict_set()

    leaf_refs = {
        "candidate_index": persist_consolidation_candidate_index(
            store, index,
            artifact_id=consolidation_candidate_index_artifact_id(BASE),
            revision=revision,
        ),
        "decision_set": persist_consolidation_decision_set(
            store, decision_set,
            artifact_id=consolidation_decision_set_artifact_id(BASE),
            revision=revision,
        ),
        "fact_set": persist_canonical_fact_set(
            store, fact_set,
            artifact_id=canonical_fact_set_artifact_id(BASE),
            revision=revision,
        ),
        "event_set": persist_canonical_event_set(
            store, event_set,
            artifact_id=canonical_event_set_artifact_id(BASE),
            revision=revision,
        ),
        "relationship_set": persist_canonical_relationship_set(
            store, relationship_set,
            artifact_id=canonical_relationship_set_artifact_id(BASE),
            revision=revision,
        ),
        "conflict_set": persist_story_conflict_set(
            store, conflict_set,
            artifact_id=story_conflict_set_artifact_id(BASE),
            revision=revision,
        ),
    }

    manifest = make_manifest(leaf_refs, entity_map_ref)
    manifest_ref = persist_consolidation_manifest(
        store, manifest,
        artifact_id=consolidation_manifest_artifact_id(BASE),
        revision=revision,
    )
    return manifest, manifest_ref, leaf_refs, entity_map_ref


def _corrupt_file(store: FileArtifactStore, ref: ArtifactRef) -> None:
    """Overwrite an on-disk envelope with a tampered payload (hash mismatch)."""
    path = store._path(ref.artifact_type, ref.artifact_id, ref.revision)
    value = json.loads(path.read_bytes())
    value["payload"]["schema_version"] = 999
    path.write_bytes(canonical_json_bytes(value))


# ---------------------------------------------------------------------------
# Deterministic identity helpers
# ---------------------------------------------------------------------------


class TestIdentity:
    def test_a5_base_artifact_id(self):
        base = a5_base_artifact_id(PROJECT_ID, DOCUMENT_ID, PROFILE_ID)
        assert base == "proj_0001.doc_0001.a5.consolidation-v1"
        # deterministic + idempotent
        assert base == a5_base_artifact_id(PROJECT_ID, DOCUMENT_ID, PROFILE_ID)

    def test_a5_base_artifact_id_isolate(self):
        assert (
            a5_base_artifact_id("p", "d", "prof-a")
            != a5_base_artifact_id("p", "d", "prof-b")
        )
        assert (
            a5_base_artifact_id("p", "d", "prof")
            != a5_base_artifact_id("q", "d", "prof")
        )
        assert (
            a5_base_artifact_id("p", "d", "prof")
            != a5_base_artifact_id("p", "e", "prof")
        )

    def test_leaf_artifact_ids(self):
        assert (
            consolidation_candidate_index_artifact_id(BASE)
            == f"{BASE}.candidate-index"
        )
        assert consolidation_decision_set_artifact_id(BASE) == f"{BASE}.decisions"
        assert canonical_fact_set_artifact_id(BASE) == f"{BASE}.facts"
        assert canonical_event_set_artifact_id(BASE) == f"{BASE}.events"
        assert (
            canonical_relationship_set_artifact_id(BASE) == f"{BASE}.relationships"
        )
        assert story_conflict_set_artifact_id(BASE) == f"{BASE}.conflicts"
        assert consolidation_manifest_artifact_id(BASE) == f"{BASE}.manifest"
        assert a5_validation_artifact_id(BASE) == f"{BASE}.manifest.a5-validation"

    def test_a5_pointer_id(self):
        assert (
            a5_pointer_id(PROJECT_ID, DOCUMENT_ID, PROFILE_ID)
            == "proj_0001.a5.doc_0001.consolidation-v1"
        )
        # distinct from the base (different field order)
        assert a5_pointer_id(PROJECT_ID, DOCUMENT_ID, PROFILE_ID) != BASE


# ---------------------------------------------------------------------------
# Shared revision allocation
# ---------------------------------------------------------------------------


class TestNextRevision:
    def test_empty_store_starts_at_1(self, store):
        assert next_a5_revision(store, base=BASE, current_manifest_ref=None) == 1

    def test_starts_after_current_manifest(self, store):
        _, manifest_ref, _, _ = make_all(store, revision=3)
        assert manifest_ref.revision == 3
        assert next_a5_revision(store, base=BASE, current_manifest_ref=manifest_ref) == 4

    def test_skips_occupied_leaf_slot(self, store):
        # Occupy the run at revision 4 via a single leaf slot.
        persist_consolidation_candidate_index(
            store, make_candidate_index(),
            artifact_id=consolidation_candidate_index_artifact_id(BASE),
            revision=4,
        )
        current = make_artifact_ref(
            CONSOLIDATION_MANIFEST_ARTIFACT_TYPE,
            consolidation_manifest_artifact_id(BASE),
            revision=3,
            content_hash=H,
        )
        assert next_a5_revision(store, base=BASE, current_manifest_ref=current) == 5

    def test_skips_occupied_manifest_slot(self, store):
        # Occupy the manifest slot at revision 1.
        manifest = make_manifest(
            {
                "candidate_index": make_artifact_ref(),
                "decision_set": make_artifact_ref(),
                "fact_set": make_artifact_ref(),
                "event_set": make_artifact_ref(),
                "relationship_set": make_artifact_ref(),
                "conflict_set": make_artifact_ref(),
            },
            make_artifact_ref(),
        )
        persist_consolidation_manifest(
            store, manifest,
            artifact_id=consolidation_manifest_artifact_id(BASE),
            revision=1,
        )
        assert next_a5_revision(store, base=BASE, current_manifest_ref=None) == 2

    def test_skips_occupied_validation_slot(self, store):
        # An orphan ValidationReport at revision 1 must be skipped.
        persist_validation_report(
            store,
            ValidationReport(validated_refs=(), findings=()),
            artifact_id=a5_validation_artifact_id(BASE),
            revision=1,
        )
        assert next_a5_revision(store, base=BASE, current_manifest_ref=None) == 2

    def test_skips_multiple_occupied(self, store):
        # Occupy revisions 1 and 2 (different slots), expect 3.
        persist_canonical_fact_set(
            store, make_canonical_fact_set(),
            artifact_id=canonical_fact_set_artifact_id(BASE),
            revision=1,
        )
        persist_story_conflict_set(
            store, make_story_conflict_set(),
            artifact_id=story_conflict_set_artifact_id(BASE),
            revision=2,
        )
        assert next_a5_revision(store, base=BASE, current_manifest_ref=None) == 3


# ---------------------------------------------------------------------------
# Per-artifact persist / load round-trip
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_candidate_index_roundtrip(self, store):
        ref = persist_consolidation_candidate_index(
            store, make_candidate_index(),
            artifact_id=consolidation_candidate_index_artifact_id(BASE),
            revision=1,
        )
        assert ref.artifact_type == CONSOLIDATION_CANDIDATE_INDEX_ARTIFACT_TYPE
        assert ref.artifact_id == consolidation_candidate_index_artifact_id(BASE)
        assert ref.revision == 1
        loaded = load_consolidation_candidate_index(
            store, ref,
            expected_artifact_id=consolidation_candidate_index_artifact_id(BASE),
        )
        assert loaded == make_candidate_index()

    def test_decision_set_roundtrip(self, store):
        ref = persist_consolidation_decision_set(
            store, make_decision_set(),
            artifact_id=consolidation_decision_set_artifact_id(BASE),
            revision=2,
        )
        loaded = load_consolidation_decision_set(
            store, ref,
            expected_artifact_id=consolidation_decision_set_artifact_id(BASE),
        )
        assert loaded == make_decision_set()

    def test_canonical_fact_set_roundtrip(self, store):
        ref = persist_canonical_fact_set(
            store, make_canonical_fact_set(),
            artifact_id=canonical_fact_set_artifact_id(BASE),
            revision=1,
        )
        loaded = load_canonical_fact_set(
            store, ref,
            expected_artifact_id=canonical_fact_set_artifact_id(BASE),
        )
        assert loaded == make_canonical_fact_set()

    def test_canonical_event_set_roundtrip(self, store):
        ref = persist_canonical_event_set(
            store, make_canonical_event_set(),
            artifact_id=canonical_event_set_artifact_id(BASE),
            revision=1,
        )
        loaded = load_canonical_event_set(
            store, ref,
            expected_artifact_id=canonical_event_set_artifact_id(BASE),
        )
        assert loaded == make_canonical_event_set()

    def test_canonical_relationship_set_roundtrip(self, store):
        ref = persist_canonical_relationship_set(
            store, make_canonical_relationship_set(),
            artifact_id=canonical_relationship_set_artifact_id(BASE),
            revision=1,
        )
        loaded = load_canonical_relationship_set(
            store, ref,
            expected_artifact_id=canonical_relationship_set_artifact_id(BASE),
        )
        assert loaded == make_canonical_relationship_set()

    def test_story_conflict_set_roundtrip(self, store):
        ref = persist_story_conflict_set(
            store, make_story_conflict_set(),
            artifact_id=story_conflict_set_artifact_id(BASE),
            revision=1,
        )
        loaded = load_story_conflict_set(
            store, ref,
            expected_artifact_id=story_conflict_set_artifact_id(BASE),
        )
        assert loaded == make_story_conflict_set()

    def test_manifest_roundtrip(self, store):
        _, manifest_ref, _, _ = make_all(store, revision=1)
        loaded = load_consolidation_manifest(
            store, manifest_ref,
            expected_artifact_id=consolidation_manifest_artifact_id(BASE),
        )
        # Round-trips to an equal manifest.
        manifest, _, _, entity_map_ref = make_all(store, revision=1)
        assert loaded == manifest
        assert loaded.entity_map_ref == entity_map_ref

    def test_content_hash_is_stable(self, store):
        ref = persist_canonical_fact_set(
            store, make_canonical_fact_set(),
            artifact_id=canonical_fact_set_artifact_id(BASE),
            revision=1,
        )
        envelope = store.get_ref(ref)
        assert envelope.content_hash == ref.content_hash
        # A fresh envelope built from the same payload matches.
        from short_drama.artifacts import ImmutableArtifactEnvelope

        rebuilt = ImmutableArtifactEnvelope.create(
            artifact_type=CANONICAL_FACT_SET_ARTIFACT_TYPE,
            artifact_id=ref.artifact_id,
            revision=1,
            schema_version=1,
            payload=make_canonical_fact_set().to_dict(),
        )
        assert rebuilt.content_hash == ref.content_hash


# ---------------------------------------------------------------------------
# Fail-closed loaders
# ---------------------------------------------------------------------------


class TestFailClosedLoaders:
    def test_wrong_artifact_type(self, store):
        ref = persist_consolidation_candidate_index(
            store, make_candidate_index(),
            artifact_id=consolidation_candidate_index_artifact_id(BASE),
            revision=1,
        )
        # Load the candidate-index ref as a fact set -> artifact_type mismatch.
        with pytest.raises(StoryIntegrityError, match="wrong artifact_type"):
            load_canonical_fact_set(
                store, ref,
                expected_artifact_id=canonical_fact_set_artifact_id(BASE),
            )

    def test_wrong_expected_artifact_id(self, store):
        ref = persist_consolidation_candidate_index(
            store, make_candidate_index(),
            artifact_id=consolidation_candidate_index_artifact_id(BASE),
            revision=1,
        )
        with pytest.raises(StoryIntegrityError, match="does not match its logical identity"):
            load_consolidation_candidate_index(
                store, ref,
                expected_artifact_id=f"{BASE}.something-else",
            )

    def test_wrong_schema_version(self, store):
        from short_drama.artifacts import ImmutableArtifactEnvelope

        ref = persist_consolidation_candidate_index(
            store, make_candidate_index(),
            artifact_id=consolidation_candidate_index_artifact_id(BASE),
            revision=1,
        )
        # Re-persist the same payload at a DIFFERENT revision with schema_version 2.
        bad = ImmutableArtifactEnvelope.create(
            artifact_type=CONSOLIDATION_CANDIDATE_INDEX_ARTIFACT_TYPE,
            artifact_id=consolidation_candidate_index_artifact_id(BASE),
            revision=2,
            schema_version=2,
            payload=make_candidate_index().to_dict(),
        )
        bad_ref = store.put(bad)
        assert bad_ref.artifact_id == consolidation_candidate_index_artifact_id(BASE)
        with pytest.raises(StoryIntegrityError, match="unsupported .* schema_version"):
            load_consolidation_candidate_index(
                store, bad_ref,
                expected_artifact_id=consolidation_candidate_index_artifact_id(BASE),
            )

    def test_corrupt_content_hash_mismatch(self, store):
        ref = persist_consolidation_candidate_index(
            store, make_candidate_index(),
            artifact_id=consolidation_candidate_index_artifact_id(BASE),
            revision=1,
        )
        _corrupt_file(store, ref)
        with pytest.raises(StoryIntegrityError, match="failed to resolve"):
            load_consolidation_candidate_index(
                store, ref,
                expected_artifact_id=consolidation_candidate_index_artifact_id(BASE),
            )

    def test_payload_not_canonical(self, store):
        ref = persist_consolidation_candidate_index(
            store, make_candidate_index(),
            artifact_id=consolidation_candidate_index_artifact_id(BASE),
            revision=1,
        )

        def _non_canonical_loader(payload):
            mutated = json.loads(json.dumps(payload))
            mutated["schema_version"] = 99  # to_dict differs from the persisted payload
            return _FrozenToplevel(mutated)

        with pytest.raises(StoryIntegrityError, match="not in canonical semantic form"):
            cp._load_typed(
                store,
                ref,
                artifact_type=CONSOLIDATION_CANDIDATE_INDEX_ARTIFACT_TYPE,
                schema_version=1,
                expected_artifact_id=consolidation_candidate_index_artifact_id(BASE),
                schema_filename="consolidation-candidate-index.schema.json",
                label="ConsolidationCandidateIndex",
                loader=_non_canonical_loader,
            )

    def test_tracked_schema_violation(self, store):
        # A payload that the model accepts shape for but violates the tracked
        # enum must be rejected by the schema gate.
        payload = make_candidate_index().to_dict()
        payload["facts"][0]["fact_type"] = "bogus"
        with pytest.raises(StoryIntegrityError, match="fails consolidation-candidate-index.schema.json"):
            cp._validate_schema(payload, "consolidation-candidate-index.schema.json", "x")

    def test_schema_gate_is_the_tracked_schema(self, store):
        # Sanity: the schema file is tracked and the valid payload passes it.
        schema = load_json(SCHEMAS_DIR / "consolidation-candidate-index.schema.json")
        assert schema["$schema"] is not None
        cp._validate_schema(
            make_candidate_index().to_dict(),
            "consolidation-candidate-index.schema.json",
            "x",
        )

    def test_missing_artifact(self, store):
        ref = make_artifact_ref(
            CONSOLIDATION_CANDIDATE_INDEX_ARTIFACT_TYPE,
            consolidation_candidate_index_artifact_id(BASE),
            revision=1,
            content_hash=H,
        )
        with pytest.raises(StoryIntegrityError, match="failed to resolve"):
            load_consolidation_candidate_index(
                store, ref,
                expected_artifact_id=consolidation_candidate_index_artifact_id(BASE),
            )


class _FrozenToplevel:
    def __init__(self, data):
        self._data = data

    def to_dict(self):
        return self._data


# ---------------------------------------------------------------------------
# Manifest reference verification
# ---------------------------------------------------------------------------


class TestManifestVerification:
    def test_verify_ok(self, store):
        manifest, manifest_ref, _, _ = make_all(store, revision=1)
        cp._verify_consolidation_manifest(
            store, base=BASE, manifest=manifest, manifest_ref=manifest_ref
        )

    def test_wrong_root_type(self, store):
        manifest, manifest_ref, _, _ = make_all(store, revision=1)
        bad_ref = ArtifactRef(
            artifact_type=CONSOLIDATION_MANIFEST_ARTIFACT_TYPE + "-x",
            artifact_id=manifest_ref.artifact_id,
            revision=manifest_ref.revision,
            content_hash=manifest_ref.content_hash,
        )
        with pytest.raises(StoryIntegrityError, match="wrong artifact_type"):
            cp._verify_consolidation_manifest(
                store, base=BASE, manifest=manifest, manifest_ref=bad_ref
            )

    def test_wrong_root_id(self, store):
        manifest, manifest_ref, _, _ = make_all(store, revision=1)
        bad_ref = ArtifactRef(
            artifact_type=CONSOLIDATION_MANIFEST_ARTIFACT_TYPE,
            artifact_id="not-the-manifest",
            revision=manifest_ref.revision,
            content_hash=manifest_ref.content_hash,
        )
        with pytest.raises(StoryIntegrityError, match="does not match its logical identity"):
            cp._verify_consolidation_manifest(
                store, base=BASE, manifest=manifest, manifest_ref=bad_ref
            )

    def test_entity_map_wrong_type(self, store):
        manifest, manifest_ref, _, _ = make_all(store, revision=1)
        bad = make_artifact_ref("entity_map-bogus", "some-entity-map")
        new_manifest = dataclasses.replace(manifest, entity_map_ref=bad)
        with pytest.raises(StoryIntegrityError, match="must be an entity_map ref"):
            cp._verify_consolidation_manifest(
                store, base=BASE, manifest=new_manifest, manifest_ref=manifest_ref
            )

    def test_leaf_wrong_id(self, store):
        manifest, manifest_ref, _, _ = make_all(store, revision=1)
        bad_leaf = ArtifactRef(
            artifact_type=CONSOLIDATION_CANDIDATE_INDEX_ARTIFACT_TYPE,
            artifact_id=f"{BASE}.candidate-index-bogus",
            revision=manifest_ref.revision,
            content_hash=H,
        )
        new_manifest = dataclasses.replace(
            manifest, consolidation_candidate_index_ref=bad_leaf
        )
        with pytest.raises(StoryIntegrityError, match="does not match its logical identity"):
            cp._verify_consolidation_manifest(
                store, base=BASE, manifest=new_manifest, manifest_ref=manifest_ref
            )

    def test_leaf_revision_mismatch(self, store):
        manifest, manifest_ref, _, _ = make_all(store, revision=1)
        # A leaf ref at a different revision than the manifest run.
        bad_leaf = ArtifactRef(
            artifact_type=CANONICAL_FACT_SET_ARTIFACT_TYPE,
            artifact_id=canonical_fact_set_artifact_id(BASE),
            revision=manifest_ref.revision + 1,
            content_hash=H,
        )
        new_manifest = dataclasses.replace(manifest, canonical_fact_set_ref=bad_leaf)
        with pytest.raises(StoryIntegrityError, match="does not share the manifest run revision"):
            cp._verify_consolidation_manifest(
                store, base=BASE, manifest=new_manifest, manifest_ref=manifest_ref
            )

    def test_leaf_missing(self, store):
        manifest, manifest_ref, _, _ = make_all(store, revision=1)
        # Point the fact-set leaf at an unpersisted revision -> unresolvable.
        bad_leaf = ArtifactRef(
            artifact_type=CANONICAL_FACT_SET_ARTIFACT_TYPE,
            artifact_id=canonical_fact_set_artifact_id(BASE),
            revision=999,
            content_hash=H,
        )
        new_manifest = dataclasses.replace(manifest, canonical_fact_set_ref=bad_leaf)
        with pytest.raises(StoryIntegrityError):
            cp._verify_consolidation_manifest(
                store, base=BASE, manifest=new_manifest, manifest_ref=manifest_ref
            )

    def test_leaf_corrupt(self, store):
        manifest, manifest_ref, leaf_refs, _ = make_all(store, revision=1)
        _corrupt_file(store, leaf_refs["fact_set"])
        with pytest.raises(StoryIntegrityError):
            cp._verify_consolidation_manifest(
                store, base=BASE, manifest=manifest, manifest_ref=manifest_ref
            )


# ---------------------------------------------------------------------------
# A5 ValidationReport
# ---------------------------------------------------------------------------


class TestValidationReport:
    def _manifest(self, store):
        manifest, manifest_ref, _, _ = make_all(store, revision=1)
        return manifest, manifest_ref

    def test_build_lineage_order(self, store):
        manifest, manifest_ref = self._manifest(store)
        report = build_a5_validation_report(manifest, manifest_ref)
        assert report.findings == ()
        roles = [r.role for r in report.validated_refs]
        expected_roles = {
            "source_document",
            "chunk_manifest",
            "candidate_extraction_0001",
            "candidate_extraction_0002",
            "entity_map",
            "consolidation_candidate_index",
            "consolidation_decision_set",
            "canonical_fact_set",
            "canonical_event_set",
            "canonical_relationship_set",
            "story_conflict_set",
            "consolidation_manifest",
        }
        # Exact content, and a fully deterministic order (ValidationReport
        # normalizes to its canonical sort order).
        assert set(roles) == expected_roles
        assert roles == sorted(roles)
        # The manifest is pinned under its exact ref.
        manifest_role = next(r for r in report.validated_refs if r.role == "consolidation_manifest")
        assert manifest_role.artifact_ref == manifest_ref

    def test_build_deterministic(self, store):
        manifest, manifest_ref = self._manifest(store)
        r1 = build_a5_validation_report(manifest, manifest_ref)
        r2 = build_a5_validation_report(manifest, manifest_ref)
        assert r1 == r2
        # A different manifest produces a different report.
        manifest_b, manifest_ref_b = make_all(store, revision=2)[0:2]
        assert build_a5_validation_report(manifest_b, manifest_ref_b) != r1

    def test_require_matching_report(self, store):
        manifest, manifest_ref = self._manifest(store)
        expected = build_a5_validation_report(manifest, manifest_ref)
        persist_validation_report(
            store, expected,
            artifact_id=a5_validation_artifact_id(BASE),
            revision=manifest_ref.revision,
        )
        ref = cp._require_a5_validation_report(
            store,
            artifact_id=a5_validation_artifact_id(BASE),
            revision=manifest_ref.revision,
            expected_report=expected,
        )
        assert ref.artifact_type == VALIDATION_REPORT_ARTIFACT_TYPE

    def test_require_missing_report(self, store):
        manifest, manifest_ref = self._manifest(store)
        expected = build_a5_validation_report(manifest, manifest_ref)
        with pytest.raises(StoryIntegrityError, match="missing or invalid"):
            cp._require_a5_validation_report(
                store,
                artifact_id=a5_validation_artifact_id(BASE),
                revision=manifest_ref.revision,
                expected_report=expected,
            )

    def test_require_mismatched_report(self, store):
        manifest, manifest_ref = self._manifest(store)
        expected = build_a5_validation_report(manifest, manifest_ref)
        different = build_a5_validation_report(manifest, manifest_ref)
        # Force a mismatch by persisting a report with an extra lineage role.
        different = ValidationReport(
            validated_refs=different.validated_refs + (
                LineageRef("synthetic_extra", make_artifact_ref()),
            ),
            findings=(),
        )
        persist_validation_report(
            store, different,
            artifact_id=a5_validation_artifact_id(BASE),
            revision=manifest_ref.revision,
        )
        with pytest.raises(StoryIntegrityError, match="does not match the exact expected"):
            cp._require_a5_validation_report(
                store,
                artifact_id=a5_validation_artifact_id(BASE),
                revision=manifest_ref.revision,
                expected_report=expected,
            )

    def test_require_non_pass_report(self, store):
        # Pass a non-PASS expected report and persist exactly it: the report
        # matches, so the non-PASS gate must fire.
        manifest, manifest_ref = self._manifest(store)
        base_report = build_a5_validation_report(manifest, manifest_ref)
        non_pass = ValidationReport(
            validated_refs=base_report.validated_refs,
            findings=(
                ValidationFinding(
                    finding_id="f_0001",
                    code="A5_FAIL",
                    severity=ValidationSeverity.BLOCKING,
                    owner_stage="A5",
                    repair_route="manual",
                    message="blocked",
                ),
            ),
        )
        assert non_pass.summary.result is ValidationResult.FAIL
        persist_validation_report(
            store, non_pass,
            artifact_id=a5_validation_artifact_id(BASE),
            revision=manifest_ref.revision,
        )
        with pytest.raises(StoryIntegrityError, match="non-PASS A5 ValidationReport"):
            cp._require_a5_validation_report(
                store,
                artifact_id=a5_validation_artifact_id(BASE),
                revision=manifest_ref.revision,
                expected_report=non_pass,
            )


# ---------------------------------------------------------------------------
# No side effects / reproducibility
# ---------------------------------------------------------------------------


class TestNoSideEffects:
    def test_no_pointer_written(self, store):
        make_all(store, revision=1)
        persist_validation_report(
            store,
            ValidationReport(validated_refs=(), findings=()),
            artifact_id=a5_validation_artifact_id(BASE),
            revision=1,
        )
        # A5F1 must NOT write the CURRENT pointer.
        pointer_id = a5_pointer_id(PROJECT_ID, DOCUMENT_ID, PROFILE_ID)
        pointer_path = store._path("a5-current", pointer_id, 1)
        assert not pointer_path.exists()
        # Nothing lives under the pointer identity for any plausible pointer type.
        for artifact_type in ("a5-current", "a5_current", "current"):
            try:
                store.get(artifact_type, pointer_id, 1)
            except ArtifactNotFoundError:
                continue
            raise AssertionError(
                f"unexpected pointer artifact under {artifact_type!r}"
            )

    def test_reopen_store_reproduces(self, tmp_path):
        root = tmp_path / "store"
        store_a = FileArtifactStore(root)
        manifest, manifest_ref, leaf_refs, _ = make_all(store_a, revision=1)
        # A brand-new store over the same root sees identical artifacts.
        store_b = FileArtifactStore(root)
        loaded = load_consolidation_manifest(
            store_b, manifest_ref,
            expected_artifact_id=consolidation_manifest_artifact_id(BASE),
        )
        assert loaded == manifest
        assert load_consolidation_candidate_index(
            store_b, leaf_refs["candidate_index"],
            expected_artifact_id=consolidation_candidate_index_artifact_id(BASE),
        ) == make_candidate_index()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path):
    return FileArtifactStore(tmp_path / "store")
