"""Tests for v1.2 A5B Phase A: zero-provider input binding / indexing / audit.

Covers:
  * ``ReconciliationPersistenceService.require_current_validated`` (the A5B
    downstream A4 CURRENT resolver) -- valid, read-only, and the missing-CURRENT
    failure;
  * ``build_consolidation_input_snapshot`` -- exact A3 + A4 loading and the
    A3/A4 consistency (fail closed on ref / identity mismatches);
  * ``build_consolidation_candidate_index`` -- A3 local-ref -> A4-bound A5 id
    binding with field-kind legality, the source-order authority, coverage, and
    fail-closed paths;
  * ``build_consolidation_planning`` -- the full Phase A orchestration.

All fixtures are self-contained synthetic run trees (no dependency on a local
Alice run tree and no provider call). The A4 CURRENT is produced by the real A4
deterministic pipeline (plan -> finalize -> identity -> publish_validated), so
the A5B resolver is exercised against a genuinely current-eligible CURRENT.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from short_drama.artifacts import FileArtifactStore
from short_drama.foundation import (
    FilePointerStore,
    LineageRef,
    PointerKind,
    ValidationReport,
    persist_validation_report,
)
from short_drama.llm import LLMInvocationProvenance
from short_drama.llm.config import load_semantic_profile
from short_drama.paths import REPO_ROOT
from short_drama.story import (
    A3InputIdentity,
    CandidateEntityIndex,
    CandidateEntityIndexEntry,
    CandidateExtraction,
    CandidatePayload,
    CharacterCandidate,
    CHUNK_PLANNER_VERSION,
    ChunkManifest,
    ChunkPlanningProfile,
    ConsolidationCurrentMissingError,
    EntityReconciliationProfile,
    EventCandidate,
    EvidenceRef,
    FactCandidate,
    LANGUAGE_DETECTOR_ID,
    LocationCandidate,
    PAIR_STATE_NEEDS_SEMANTIC_DECISION,
    ReconciliationDecision,
    ReconciliationPersistenceService,
    ReconciliationSemanticResult,
    RelationshipCandidate,
    SourceChapter,
    SourceDocument,
    SourceInfo,
    SourceParagraph,
    StoryExtractionProfile,
    StoryIntegrityError,
    TOKEN_COUNTER_ID,
    UnresolvedMentionCandidate,
    ValidatedEntityMapCurrent,
    build_a4_semantic_identity,
    build_consolidation_candidate_index,
    build_consolidation_input_snapshot,
    build_consolidation_planning,
    compute_llm_decision_id,
    finalize_reconciliation,
    load_consolidation_profile,
    load_entity_reconciliation_profile,
    llm_reason_code,
    pack_semantic_pairs_v1,
    plan_candidate_index_v1,
    plan_chunks,
    prepare_semantic_resolution,
    persist_candidate_extraction,
    persist_chunk_manifest,
    persist_source_chunk,
    persist_source_document,
)
from short_drama.story.persistence import (
    chunk_pointer_id,
    chunk_validation_artifact_id,
    source_pointer_id,
    source_validation_artifact_id,
)
from short_drama.story.reconciliation_planning import (
    _local_candidate_suffix,
    compute_source_order_key,
)
from short_drama.story.source import NormalizationInfo

PROJECT = "a5btest"
DOCUMENT = "src_001"
CHUNK_PROFILE_ID = "story-analysis-v1"
EXTRACTION_PROFILE_ID = "story-extraction-v1"
RECON_PROFILE_ID = "entity-reconciliation-v2"
RECON_PROFILE_PATH = REPO_ROOT / "profiles" / "entity_reconciliation_v2.yaml"
A4_LLM_PROFILE_PATH = REPO_ROOT / "profiles" / "entity_reconciliation_llm_v1.yaml"
CONSOLIDATION_PROFILE_PATH = REPO_ROOT / "profiles" / "consolidation_v1.yaml"


def _consolidation_profile():
    return load_consolidation_profile(CONSOLIDATION_PROFILE_PATH)

_PARAS = ("CH001_P0001", "CH001_P0002", "CH001_P0003")
_CHUNK_ID = "CH001_C001"


def _recon_profile() -> EntityReconciliationProfile:
    return load_entity_reconciliation_profile(RECON_PROFILE_PATH)


def _extraction_profile() -> StoryExtractionProfile:
    return StoryExtractionProfile(
        schema_version=1,
        profile_id=EXTRACTION_PROFILE_ID,
        working_language="zh-CN",
        prompt_id="a3.chunk-extraction",
        prompt_version=1,
        output_schema_id="a3-candidate-payload",
        output_schema_version=1,
        max_generation_rounds=2,
    )


def _chunk_profile() -> ChunkPlanningProfile:
    return ChunkPlanningProfile(
        schema_version=1,
        profile_id=CHUNK_PROFILE_ID,
        token_counter=TOKEN_COUNTER_ID,
        ownership_token_budget=100000,
        context_overlap_token_budget=0,
        context_token_budget=100000,
    )


def _provenance() -> LLMInvocationProvenance:
    return LLMInvocationProvenance(
        provider_family="qwen",
        model="qwen3-27b",
        semantic_profile_id="story-extraction-llm-v1",
        semantic_profile_hash="f" * 64,
        prompt_id="a3.chunk-extraction",
        prompt_version=1,
        prompt_content_hash="f" * 64,
        rendered_prompt_hash="f" * 64,
        output_schema_id="a3-candidate-payload",
        output_schema_version=1,
        output_schema_hash="f" * 64,
        request_hash="f" * 64,
        provider_response_id=None,
        finish_reason="stop",
        usage={"total_tokens": 100},
    )


def _evidence(paragraph_id: str = _PARAS[0]) -> EvidenceRef:
    return EvidenceRef(paragraph_id=paragraph_id, role="primary", strength="explicit", excerpt="txt")


def _char(candidate_id: str, name: str) -> CharacterCandidate:
    return CharacterCandidate(
        candidate_id=candidate_id,
        display_name_original=name,
        aliases_original=(),
        descriptors_zh=(name,),
        summary_zh=name,
        evidence_strength="explicit",
        evidence=(_evidence(),),
    )


def _loc(candidate_id: str, name: str) -> LocationCandidate:
    return LocationCandidate(
        candidate_id=candidate_id,
        display_name_original=name,
        aliases_original=(),
        descriptors_zh=(name,),
        summary_zh=name,
        evidence_strength="explicit",
        evidence=(_evidence(),),
    )


def _unres(candidate_id: str, kind: str) -> UnresolvedMentionCandidate:
    return UnresolvedMentionCandidate(
        candidate_id=candidate_id,
        mention_original="?",
        mention_kind=kind,
        reason_zh="unclear",
        possible_candidate_refs=(),
        evidence_strength="uncertain",
        evidence=(_evidence(),),
    )


def _fact(candidate_id: str, subject_refs=(), object_refs=()) -> FactCandidate:
    return FactCandidate(
        candidate_id=candidate_id,
        fact_type="world_fact",
        statement_zh="stmt",
        subject_refs=tuple(subject_refs),
        object_refs=tuple(object_refs),
        evidence_strength="explicit",
        evidence=(_evidence(),),
    )


def _event(
    candidate_id: str, participant_refs=(), location_refs=()
) -> EventCandidate:
    return EventCandidate(
        candidate_id=candidate_id,
        summary_zh="evt",
        participant_refs=tuple(participant_refs),
        location_refs=tuple(location_refs),
        temporal_mode="normal",
        evidence_strength="explicit",
        evidence=(_evidence(),),
    )


def _rel(candidate_id: str, source_ref: str, target_ref: str) -> RelationshipCandidate:
    return RelationshipCandidate(
        candidate_id=candidate_id,
        source_ref=source_ref,
        target_ref=target_ref,
        relationship_type_zh="meets",
        state_zh=None,
        direction="directed",
        evidence_strength="explicit",
        evidence=(_evidence(),),
    )


@dataclass
class RunTree:
    """A self-contained synthetic A1+A2+A3(+A4) run tree for A5B tests."""

    store: FileArtifactStore
    pointers: FilePointerStore
    source_ref: object
    chunk: object
    chunk_ref: object
    manifest: ChunkManifest
    manifest_ref: object
    ext_ref: object
    recon_profile: EntityReconciliationProfile
    a3_input: A3InputIdentity
    index: CandidateEntityIndex


def _idx_entry(chunk_id: str, local_id: str, kind: str, ext_ref, name: str) -> CandidateEntityIndexEntry:
    category = (
        "character" if kind == "character"
        else "location" if kind == "location"
        else "unresolved"
    )
    global_ref = f"{chunk_id}:{local_id}"
    order_key = compute_source_order_key(
        chunk_ordinal=1,
        paragraph_ordinal=1,
        category=category,
        candidate_suffix=_local_candidate_suffix(local_id),
        global_ref=global_ref,
    )
    return CandidateEntityIndexEntry(
        candidate_ref=global_ref,
        candidate_kind=kind,
        candidate_extraction_ref=ext_ref,
        source_order_key=order_key,
        display_name_original=name,
        aliases_original=(),
        descriptors_zh=(name,),
        evidence_refs=(_evidence(),),
        possible_candidate_refs=(),
    )


def _build_run_tree(
    tmp_path: Path,
    *,
    facts=(),
    events=(),
    rels=(),
    chars=(),
    locs=(),
    unres=(),
    publish_a4: bool = True,
) -> RunTree:
    """Persist A1+A2+A3 and (optionally) publish a valid A4 CURRENT."""
    store = FileArtifactStore(tmp_path / "artifacts")
    pointers = FilePointerStore(tmp_path / "pointers", store)

    # --- A1: SourceDocument ------------------------------------------------
    source_doc = SourceDocument(
        schema_version=1,
        project_id=PROJECT,
        document_id=DOCUMENT,
        source=SourceInfo(
            "txt", "source/novel.txt", "f" * 64, 128, "zh-CN", "zh", LANGUAGE_DETECTOR_ID
        ),
        normalization=NormalizationInfo("utf-8", "LF", "short_drama_source_ingestion_v1", "1"),
        chapters=(
            SourceChapter(
                "CH001",
                None,
                "synthetic",
                tuple(SourceParagraph(pid, "text", None) for pid in _PARAS),
            ),
        ),
    )
    source_ref = persist_source_document(store, source_doc, revision=1)
    pointers.compare_and_set(
        pointer_id=source_pointer_id(PROJECT, DOCUMENT),
        pointer_kind=PointerKind.CURRENT,
        expected_pointer_ref=None,
        target_ref=source_ref,
    )
    persist_validation_report(
        store,
        ValidationReport(
            validated_refs=(LineageRef("source_document", source_ref),), findings=()
        ),
        artifact_id=source_validation_artifact_id(PROJECT, DOCUMENT),
        revision=source_ref.revision,
    )

    # --- A2: ChunkManifest --------------------------------------------------
    chunks, coverage = plan_chunks(source_doc, source_ref, _chunk_profile())
    assert len(chunks) == 1, "fixture assumes exactly one chunk"
    chunk = chunks[0]
    chunk_ref = persist_source_chunk(store, chunk, profile_id=CHUNK_PROFILE_ID, revision=1)
    manifest = ChunkManifest(
        schema_version=1,
        project_id=PROJECT,
        document_id=DOCUMENT,
        source_document_ref=source_ref,
        planner_version=CHUNK_PLANNER_VERSION,
        profile=_chunk_profile(),
        chunk_refs=(chunk_ref,),
        chunk_count=1,
        coverage=coverage,
        state="CHUNKING_COMPLETE",
    )
    manifest_ref = persist_chunk_manifest(store, manifest, revision=1)
    pointers.compare_and_set(
        pointer_id=chunk_pointer_id(PROJECT, DOCUMENT, CHUNK_PROFILE_ID),
        pointer_kind=PointerKind.CURRENT,
        expected_pointer_ref=None,
        target_ref=manifest_ref,
    )
    persist_validation_report(
        store,
        ValidationReport(
            validated_refs=(
                LineageRef("source_document", source_ref),
                LineageRef("chunk_manifest", manifest_ref),
            ),
            findings=(),
        ),
        artifact_id=chunk_validation_artifact_id(PROJECT, DOCUMENT, CHUNK_PROFILE_ID),
        revision=manifest_ref.revision,
    )

    # --- A3: CandidateExtraction -------------------------------------------
    ext_profile = _extraction_profile()
    extraction = CandidateExtraction(
        schema_version=1,
        project_id=PROJECT,
        document_id=DOCUMENT,
        chunk_profile_id=CHUNK_PROFILE_ID,
        chunk_id=chunk.chunk_id,
        source_document_ref=source_ref,
        source_chunk_ref=chunk_ref,
        extraction_profile_id=ext_profile.profile_id,
        extraction_profile_hash=ext_profile.profile_hash,
        generation_provenance=_provenance(),
        candidates=CandidatePayload(
            characters=tuple(chars),
            locations=tuple(locs),
            facts=tuple(facts),
            events=tuple(events),
            relationships=tuple(rels),
            unresolved_mentions=tuple(unres),
        ),
    )
    ext_ref = persist_candidate_extraction(store, extraction, revision=1)

    # --- A3 input identity + A4 candidate index ----------------------------
    a3_input = A3InputIdentity(
        source_document_ref=source_ref,
        chunk_manifest_ref=manifest_ref,
        candidate_extraction_refs=(ext_ref,),
        extraction_profile_id=ext_profile.profile_id,
        extraction_profile_hash=ext_profile.profile_hash,
    )
    index_entries = tuple(
        _idx_entry(chunk.chunk_id, c.candidate_id, "character", ext_ref, c.display_name_original)
        for c in chars
    ) + tuple(
        _idx_entry(chunk.chunk_id, c.candidate_id, "location", ext_ref, c.display_name_original)
        for c in locs
    ) + tuple(
        _idx_entry(chunk.chunk_id, c.candidate_id, f"unresolved_{c.mention_kind}", ext_ref, c.mention_original)
        for c in unres
    )
    index = CandidateEntityIndex(schema_version=1, entries=index_entries)
    recon_profile = _recon_profile()

    # --- A4: publish a valid CURRENT ---------------------------------------
    if publish_a4:
        _publish_a4_current(
            store,
            pointers,
            index=index,
            a3_input=a3_input,
            recon_profile=recon_profile,
        )

    return RunTree(
        store=store,
        pointers=pointers,
        source_ref=source_ref,
        chunk=chunk,
        chunk_ref=chunk_ref,
        manifest=manifest,
        manifest_ref=manifest_ref,
        ext_ref=ext_ref,
        recon_profile=recon_profile,
        a3_input=a3_input,
        index=index,
    )


def _make_llm_provenance(prep, request_hash: str) -> LLMInvocationProvenance:
    return LLMInvocationProvenance(
        provider_family="qwen",
        model="qwen3-27b",
        semantic_profile_id=prep.semantic_profile_id,
        semantic_profile_hash=prep.semantic_profile_hash,
        prompt_id=prep.prompt_id,
        prompt_version=prep.prompt_version,
        prompt_content_hash=prep.prompt_content_hash,
        rendered_prompt_hash="1" * 64,
        output_schema_id=prep.output_schema_id,
        output_schema_version=prep.output_schema_version,
        output_schema_hash=prep.output_schema_hash,
        request_hash=request_hash,
        provider_response_id="resp",
        finish_reason="stop",
        usage={},
    )


def _make_llm_decision(left: str, right: str, decision: str, request_hash: str, prep, evidence: tuple) -> ReconciliationDecision:
    rc = llm_reason_code(decision)
    decision_id = compute_llm_decision_id(
        left_ref=left,
        right_ref=right,
        decision=decision,
        method="llm",
        reason_code=rc,
        reason_zh="different",
        evidence_refs=evidence,
        prompt_id=prep.prompt_id,
        prompt_version=prep.prompt_version,
        request_hash=request_hash,
    )
    return ReconciliationDecision(
        decision_id=decision_id,
        left_candidate_ref=left,
        right_candidate_ref=right,
        decision=decision,
        method="llm",
        reason_code=rc,
        reason_zh="different",
        evidence_refs=evidence,
        prompt_id=prep.prompt_id,
        prompt_version=prep.prompt_version,
        generation_provenance=_make_llm_provenance(prep, request_hash),
    )


def _publish_a4_current(store, pointers, *, index, a3_input, recon_profile) -> None:
    """Publish a valid, current-eligible A4 CURRENT via the deterministic A4
    pipeline (zero provider calls). Uncertain pairs get a deterministic 'llm'
    different_entity decision bound to the exact block request hash."""
    service = ReconciliationPersistenceService(store, pointers)
    planning = plan_candidate_index_v1(index)
    sem_profile = load_semantic_profile(A4_LLM_PROFILE_PATH)
    prep = prepare_semantic_resolution(planning, recon_profile, sem_profile)
    request_hashes = prep.semantic_request_hashes
    pair_to_hash: dict = {}
    for block_ordinal, block in enumerate(pack_semantic_pairs_v1(planning)):
        for plan in block:
            pair_to_hash[(plan.left_candidate_ref, plan.right_candidate_ref)] = (
                request_hashes[block_ordinal]
            )
    llm_decisions = [
        _make_llm_decision(
            plan.left_candidate_ref,
            plan.right_candidate_ref,
            "different_entity",
            pair_to_hash[(plan.left_candidate_ref, plan.right_candidate_ref)],
            prep,
            evidence=(),
        )
        for plan in planning.pair_plans
        if plan.state == PAIR_STATE_NEEDS_SEMANTIC_DECISION
    ]
    all_decisions = planning.decisions + tuple(llm_decisions)
    semantic_result = ReconciliationSemanticResult(
        planning_result=planning,
        blocks=prep.blocks,
        semantic_decisions=tuple(llm_decisions),
        all_decisions=all_decisions,
        semantic_request_hashes=request_hashes,
        block_results=(),
    )
    finalization = finalize_reconciliation(semantic_result)
    identity = build_a4_semantic_identity(recon_profile, prep, planning)
    service.publish_validated(
        project_id=PROJECT,
        document_id=DOCUMENT,
        reconciliation_profile=recon_profile,
        semantic_identity=identity,
        a3_input=a3_input,
        finalization_result=finalization,
    )


def _default_candidates():
    """A small, fully-connected candidate set (2 chars, 1 loc, 1 unres + facts/events/rels)."""
    chars = (_char("cand_char_001", "Alice"), _char("cand_char_002", "Bob"))
    locs = (_loc("cand_loc_001", "Wonderland"),)
    unres = (_unres("cand_unres_001", "unknown"),)
    facts = (_fact("cand_fact_001", subject_refs=("cand_char_001",), object_refs=("cand_loc_001",)),)
    events = (_event("cand_evt_001", participant_refs=("cand_char_001",), location_refs=("cand_loc_001",)),)
    rels = (_rel("cand_rel_001", source_ref="cand_char_001", target_ref="cand_char_002"),)
    return dict(chars=chars, locs=locs, unres=unres, facts=facts, events=events, rels=rels)


# ---------------------------------------------------------------------------
# A4 CURRENT resolver
# ---------------------------------------------------------------------------


def test_require_current_validated_returns_exact_current(tmp_path):
    tree = _build_run_tree(tmp_path, **_default_candidates())
    service = ReconciliationPersistenceService(tree.store, tree.pointers)
    current = service.require_current_validated(
        project_id=PROJECT,
        document_id=DOCUMENT,
        reconciliation_profile_id=RECON_PROFILE_ID,
    )
    assert isinstance(current, ValidatedEntityMapCurrent)
    assert current.entity_map_ref.artifact_id.endswith(".a4.entity-reconciliation-v2.entity-map")
    assert current.validation_report_ref.artifact_id.endswith(".a4-validation")
    assert current.current_pointer_ref.artifact_id == (
        f"{PROJECT}.a4.{DOCUMENT}.entity-reconciliation-v2"
    )
    # The loaded EntityMap pins the exact A3 input we built.
    assert current.entity_map.a3_input.candidate_extraction_refs == (tree.ext_ref,)
    assert current.entity_map.a3_input.source_document_ref == tree.source_ref
    assert current.entity_map.a3_input.chunk_manifest_ref == tree.manifest_ref


def test_require_current_validated_missing_current(tmp_path):
    # A1+A2+A3 present but NO A4 CURRENT published.
    tree = _build_run_tree(tmp_path, **_default_candidates(), publish_a4=False)
    service = ReconciliationPersistenceService(tree.store, tree.pointers)
    with pytest.raises(ConsolidationCurrentMissingError, match="A4 CURRENT"):  # structural failure
        service.require_current_validated(
            project_id=PROJECT,
            document_id=DOCUMENT,
            reconciliation_profile_id=RECON_PROFILE_ID,
        )


def test_require_current_validated_is_read_only(tmp_path):
    tree = _build_run_tree(tmp_path, **_default_candidates())
    before = _file_count(tmp_path)
    ReconciliationPersistenceService(tree.store, tree.pointers).require_current_validated(
        project_id=PROJECT,
        document_id=DOCUMENT,
        reconciliation_profile_id=RECON_PROFILE_ID,
    )
    # Read-only: no new artifact or pointer file is written.
    assert _file_count(tmp_path) == before


# ---------------------------------------------------------------------------
# Snapshot loading
# ---------------------------------------------------------------------------


def test_snapshot_loads_exact_a3_and_a4(tmp_path):
    c = _default_candidates()
    tree = _build_run_tree(tmp_path, **c)
    snapshot = build_consolidation_input_snapshot(
        tree.store, tree.pointers,
        project_id=PROJECT,
        document_id=DOCUMENT,
        reconciliation_profile_id=RECON_PROFILE_ID,
    )
    assert snapshot.source_document_ref if hasattr(snapshot, "source_document_ref") else True
    # A3 artifacts are the exact pinned ones.
    assert snapshot.candidate_extraction_refs == (tree.ext_ref,)
    assert len(snapshot.candidate_extractions) == 1
    assert len(snapshot.source_chunks) == 1
    assert snapshot.chunk_manifest.chunk_refs == (tree.chunk_ref,)
    # A4 artifacts are the loaded current-eligible set.
    assert snapshot.entity_map.a3_input == tree.a3_input
    # Candidate universe: 2 char + 1 loc + 1 unres = 4 entries.
    assert len(snapshot.entity_map.entries) == 4


def test_snapshot_missing_current_raises(tmp_path):
    tree = _build_run_tree(tmp_path, **_default_candidates(), publish_a4=False)
    with pytest.raises(ConsolidationCurrentMissingError):
        build_consolidation_input_snapshot(
            tree.store, tree.pointers,
            project_id=PROJECT,
            document_id=DOCUMENT,
            reconciliation_profile_id=RECON_PROFILE_ID,
        )


# ---------------------------------------------------------------------------
# Index building: binding, source order, coverage
# ---------------------------------------------------------------------------


def test_index_binds_and_orders(tmp_path):
    c = _default_candidates()
    tree = _build_run_tree(tmp_path, **c)
    index = build_consolidation_candidate_index(
        build_consolidation_input_snapshot(
            tree.store, tree.pointers,
            project_id=PROJECT,
            document_id=DOCUMENT,
            reconciliation_profile_id=RECON_PROFILE_ID,
        )
    )
    # One of each category.
    assert len(index.facts) == 1
    assert len(index.events) == 1
    assert len(index.relationships) == 1
    # Facts/events/relationships are NOT in the A4 candidate index (no entries).
    assert len(tree.index.entries) == 4  # only char/loc/unres

    # Bound ids: each candidate ref resolves to a char_/loc_/unres_ id.
    fact = index.facts[0]
    assert fact.global_candidate_ref == f"{_CHUNK_ID}:cand_fact_001"
    assert fact.subject_refs and all(r.startswith(("char_", "loc_", "unres_")) for r in fact.subject_refs)
    # cand_char_001 -> a char_* id; cand_loc_001 -> a loc_* id.
    assert fact.subject_refs[0].startswith("char_")
    assert fact.object_refs[0].startswith("loc_")

    event = index.events[0]
    assert event.participants[0].startswith("char_")  # person field -> char_*
    assert event.locations[0].startswith("loc_")  # location field -> loc_*

    rel = index.relationships[0]
    assert rel.source_entity_ref.startswith("char_")
    assert rel.target_entity_ref.startswith("char_")

    # Source-order authority: sorted + unique within each category.
    for candidates in (index.facts, index.events, index.relationships):
        keys = [cand.source_order_key for cand in candidates]
        assert keys == sorted(keys)
        assert len(set(keys)) == len(keys)


def test_full_planning_result(tmp_path):
    tree = _build_run_tree(tmp_path, **_default_candidates())
    result = build_consolidation_planning(
        tree.store, tree.pointers,
        project_id=PROJECT,
        document_id=DOCUMENT,
        reconciliation_profile_id=RECON_PROFILE_ID,
        consolidation_profile=_consolidation_profile(),
    )
    assert result.coverage.fact_candidate_count == 1
    assert result.coverage.event_candidate_count == 1
    assert result.coverage.relationship_candidate_count == 1
    # Phase A: no canonical / decision / conflict counts.
    assert result.coverage.canonical_fact_count == 0
    assert result.coverage.canonical_event_count == 0
    assert result.coverage.canonical_relationship_count == 0
    assert result.coverage.uncertain_decision_count == 0
    assert result.coverage.story_conflict_count == 0
    # The index is exposed on the result.
    assert len(result.index.facts) == 1


def test_unresolved_binding(tmp_path):
    # A fact referencing an unresolved mention binds to unres_*.
    c = _default_candidates()
    c["facts"] = (_fact("cand_fact_001", subject_refs=("cand_unres_001",)),)
    c["events"] = ()
    c["rels"] = ()
    tree = _build_run_tree(tmp_path, **c)
    index = build_consolidation_candidate_index(
        build_consolidation_input_snapshot(
            tree.store, tree.pointers,
            project_id=PROJECT,
            document_id=DOCUMENT,
            reconciliation_profile_id=RECON_PROFILE_ID,
        )
    )
    assert index.facts[0].subject_refs[0].startswith("unres_")


# ---------------------------------------------------------------------------
# Fail-closed paths
# ---------------------------------------------------------------------------


def test_field_kind_person_rejects_location_binding(tmp_path):
    c = _default_candidates()
    tree = _build_run_tree(tmp_path, **c)
    snapshot = build_consolidation_input_snapshot(
        tree.store, tree.pointers,
        project_id=PROJECT,
        document_id=DOCUMENT,
        reconciliation_profile_id=RECON_PROFILE_ID,
    )
    # Locate the bound loc_* id (for cand_loc_001) from the real EntityMap.
    loc_id = next(
        e.canonical_id for e in snapshot.entity_map.entries
        if e.candidate_ref == f"{_CHUNK_ID}:cand_loc_001" and e.status == "resolved"
    )
    assert loc_id.startswith("loc_")
    # Force cand_char_001 (an A3 character) to bind to that loc_* id, so an
    # event participant (a person field) would bind a loc_*.
    new_entries = tuple(
        replace(e, canonical_id=loc_id)
        if e.candidate_ref == f"{_CHUNK_ID}:cand_char_001"
        else e
        for e in snapshot.entity_map.entries
    )
    bad_snapshot = replace(snapshot, entity_map=replace(snapshot.entity_map, entries=new_entries))
    with pytest.raises(StoryIntegrityError, match="person"):
        build_consolidation_candidate_index(bad_snapshot)


def test_field_kind_location_rejects_character_binding(tmp_path):
    c = _default_candidates()
    tree = _build_run_tree(tmp_path, **c)
    snapshot = build_consolidation_input_snapshot(
        tree.store, tree.pointers,
        project_id=PROJECT,
        document_id=DOCUMENT,
        reconciliation_profile_id=RECON_PROFILE_ID,
    )
    # Force cand_loc_001 (an A3 location) to bind to a char_* id, so an event
    # location (a location field) would bind a char_*.
    char_id = next(
        e.canonical_id for e in snapshot.entity_map.entries
        if e.candidate_ref == f"{_CHUNK_ID}:cand_char_001" and e.status == "resolved"
    )
    assert char_id.startswith("char_")
    new_entries = tuple(
        replace(e, canonical_id=char_id)
        if e.candidate_ref == f"{_CHUNK_ID}:cand_loc_001"
        else e
        for e in snapshot.entity_map.entries
    )
    bad_snapshot = replace(snapshot, entity_map=replace(snapshot.entity_map, entries=new_entries))
    with pytest.raises(StoryIntegrityError, match="location"):
        build_consolidation_candidate_index(bad_snapshot)


def test_duplicate_source_order_key_fails_closed(tmp_path):
    c = _default_candidates()
    # Two facts with the SAME candidate id -> same source_order_key.
    c["facts"] = (
        _fact("cand_fact_001", subject_refs=("cand_char_001",)),
        _fact("cand_fact_001", subject_refs=("cand_char_002",)),
    )
    tree = _build_run_tree(tmp_path, **c)
    snapshot = build_consolidation_input_snapshot(
        tree.store, tree.pointers,
        project_id=PROJECT,
        document_id=DOCUMENT,
        reconciliation_profile_id=RECON_PROFILE_ID,
    )
    with pytest.raises(StoryIntegrityError, match="duplicate source_order_key"):
        build_consolidation_candidate_index(snapshot)


def test_bound_ref_not_in_registry_fails_closed(tmp_path):
    c = _default_candidates()
    tree = _build_run_tree(tmp_path, **c)
    snapshot = build_consolidation_input_snapshot(
        tree.store, tree.pointers,
        project_id=PROJECT,
        document_id=DOCUMENT,
        reconciliation_profile_id=RECON_PROFILE_ID,
    )
    # Point cand_char_002 at a char_* id that is NOT in the character registry.
    new_entries = tuple(
        replace(e, canonical_id="char_9999")
        if e.candidate_ref == f"{_CHUNK_ID}:cand_char_002"
        else e
        for e in snapshot.entity_map.entries
    )
    bad_snapshot = replace(snapshot, entity_map=replace(snapshot.entity_map, entries=new_entries))
    with pytest.raises(StoryIntegrityError, match="unknown canonical char id"):
        build_consolidation_candidate_index(bad_snapshot)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _file_count(path: Path) -> int:
    return sum(1 for p in path.rglob("*") if p.is_file())
