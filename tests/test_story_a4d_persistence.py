"""A4D reconciliation persistence / reuse / publication tests.

Covers the frozen A4D persistence contract:

  * deterministic A4 artifact identity (base + six outputs + A4 validation
    report + CURRENT pointer);
  * immutable persistence + fail-closed typed loaders;
  * the exact deterministic A4 ValidationReport (findings=(), upstream A3 refs +
    all six A4 outputs) and PASS gating;
  * current-only A4 reuse (CURRENT fully verified BEFORE the requested identity
    is compared; corrupt / wrong-logical-target / missing-report / non-PASS
    CURRENT always fails closed and never silently supersedes);
  * identity invalidation (A3 input or A4 semantic identity change) supersedes
    with a new revision under the same logical artifact ID;
  * backend-neutral reuse (the A4 semantic identity carries no provider/model
    metadata, so a backend switch does not invalidate reuse);
  * orphan revision skipping (a failed publication's partial artifacts are never
    overwritten);
  * zero-persistence on a non-publishable finalization (blocking findings).

Deliberately does NOT require: a running LLM, a stage CLI, a real-novel fixture,
or any A4E behavior.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from short_drama.artifacts import ArtifactRef, FileArtifactStore
from short_drama.llm import LLMInvocationProvenance
from short_drama.foundation import (
    FilePointerStore,
    PointerKind,
    PointerNotFoundError,
    ValidationFinding,
    ValidationReport,
    ValidationSeverity,
    ValidationResult,
    load_validation_report,
    validation_report_envelope,
)
from short_drama.llm.config import load_semantic_profile
from short_drama.story import (
    A3InputIdentity,
    A4SemanticIdentity,
    CANDIDATE_ENTITY_INDEX_ARTIFACT_TYPE,
    CANDIDATE_ENTITY_INDEX_SCHEMA_VERSION,
    CANONICAL_CHARACTER_REGISTRY_ARTIFACT_TYPE,
    CANONICAL_ENTITY_REGISTRY_SCHEMA_VERSION,
    CANONICAL_LOCATION_REGISTRY_ARTIFACT_TYPE,
    ENTITY_MAP_ARTIFACT_TYPE,
    RECONCILIATION_DECISION_SET_ARTIFACT_TYPE,
    RECONCILIATION_DECISION_SET_SCHEMA_VERSION,
    UNRESOLVED_ENTITY_SET_ARTIFACT_TYPE,
    PAIR_STATE_NEEDS_SEMANTIC_DECISION,
    CandidateEntityIndex,
    CandidateEntityIndexEntry,
    CanonicalCharacterRegistry,
    EntityMap,
    EvidenceRef,
    ReconciliationDecision,
    ReconciliationDecisionSet,
    ReconciliationFinalizationError,
    ReconciliationPairPlan,
    ReconciliationPlanningResult,
    ReconciliationPersistenceService,
    ReconciliationSemanticResult,
    compute_llm_decision_id,
    llm_reason_code,
    pack_semantic_pairs_v1,
    StoryIntegrityError,
    StoryPersistenceError,
    a4_base_artifact_id,
    a4_pointer_id,
    a4_validation_artifact_id,
    build_a4_semantic_identity,
    build_a4_validation_report,
    build_identity_graph,
    candidate_entity_index_artifact_id,
    canonical_character_registry_artifact_id,
    canonical_location_registry_artifact_id,
    entity_map_artifact_id,
    finalize_reconciliation,
    load_candidate_entity_index,
    load_canonical_character_registry,
    load_canonical_location_registry,
    load_entity_map,
    load_entity_reconciliation_profile,
    load_reconciliation_decision_set,
    load_unresolved_entity_set,
    next_a4_revision,
    persist_entity_map,
    plan_candidate_index_v1,
    prepare_semantic_resolution,
    reconciliation_decision_set_artifact_id,
    unresolved_entity_set_artifact_id,
)
from short_drama.story.reconciliation_persistence import _verify_finalization_bundle

PROFILES = Path(__file__).resolve().parents[1] / "profiles"
RECON_PROFILE_PATH = PROFILES / "entity_reconciliation_v1.yaml"
A4_LLM_PROFILE_PATH = PROFILES / "entity_reconciliation_llm_v1.yaml"

PROJECT = "proj"
DOCUMENT = "doc"
H = "a" * 64
H2 = "b" * 64

# Clean scenario: L and R are the same character ("John Smith"), LOC1 and LOC2
# are the same location ("Central Plaza"). Both pairs are auto_same (shared
# strong identity key) so the plan is fully deterministic (zero LLM).
L = "CH001_C001:cand_char_001"
R = "CH001_C002:cand_char_002"
LOC1 = "CH001_C001:cand_loc_001"
LOC2 = "CH001_C002:cand_loc_002"
# A different scenario (a different plan -> different plan_hash) for identity
# invalidation tests.
L2 = "CH001_C001:cand_char_010"
R2 = "CH001_C002:cand_char_011"

# Per-chunk extraction refs (candidates in a chunk share its extraction ref).
EXT_A = ArtifactRef(
    artifact_type="candidate_extraction", artifact_id="ce-a", revision=1, content_hash=H
)
EXT_B = ArtifactRef(
    artifact_type="candidate_extraction", artifact_id="ce-b", revision=1, content_hash=H
)


def make_ref(artifact_type: str, artifact_id: str, revision: int = 1) -> ArtifactRef:
    return ArtifactRef(
        artifact_type=artifact_type, artifact_id=artifact_id, revision=revision, content_hash=H
    )


def frozen_key(
    chunk_ordinal: int, para_ordinal: int, category_ordinal: int, suffix: int, ref: str
) -> str:
    return (
        f"{chunk_ordinal:06d}:{para_ordinal:09d}:{category_ordinal:02d}"
        f":{suffix:09d}:{ref}"
    )


def idx_entry(
    candidate_ref: str,
    kind: str = "character",
    display: str = "John Smith",
    *,
    ext_ref: ArtifactRef | None = None,
    source_key: str | None = None,
    evidence: tuple[EvidenceRef, ...] = (),
) -> CandidateEntityIndexEntry:
    if source_key is None:
        suffix = int(candidate_ref.rsplit("_", 1)[1])
        chunk = int(candidate_ref.split("_C")[1].split(":")[0])
        category = 1 if kind == "character" else 2
        source_key = frozen_key(chunk, 1, category, suffix, candidate_ref)
    return CandidateEntityIndexEntry(
        candidate_ref=candidate_ref,
        candidate_kind=kind,
        candidate_extraction_ref=ext_ref if ext_ref is not None else EXT_A,
        source_order_key=source_key,
        display_name_original=display,
        aliases_original=(),
        descriptors_zh=(),
        evidence_refs=evidence,
        possible_candidate_refs=(),
    )


def clean_index() -> CandidateEntityIndex:
    return CandidateEntityIndex(
        schema_version=CANDIDATE_ENTITY_INDEX_SCHEMA_VERSION,
        entries=(
            idx_entry(L, display="John Smith", ext_ref=EXT_A),
            idx_entry(LOC1, kind="location", display="Central Plaza", ext_ref=EXT_A),
            idx_entry(R, display="John Smith", ext_ref=EXT_B),
            idx_entry(LOC2, kind="location", display="Central Plaza", ext_ref=EXT_B),
        ),
    )


def different_index() -> CandidateEntityIndex:
    """A different candidate index (same chunks) -> a different plan_hash."""
    return CandidateEntityIndex(
        schema_version=CANDIDATE_ENTITY_INDEX_SCHEMA_VERSION,
        entries=(
            idx_entry(L2, display="Alice Brown", ext_ref=EXT_A),
            idx_entry(R2, display="Alice Brown", ext_ref=EXT_B),
        ),
    )


def make_planning(index, pair_plans, decisions, plan_hash: str = H2) -> ReconciliationPlanningResult:
    return ReconciliationPlanningResult(
        candidate_index=index,
        pair_plans=tuple(pair_plans),
        decisions=tuple(decisions),
        normalization_policy_id="norm",
        blocking_policy_id="block",
        canonicalization_policy_id="canon",
        plan_hash=plan_hash,
    )


def make_semantic(planning, decisions) -> ReconciliationSemanticResult:
    return ReconciliationSemanticResult(
        planning_result=planning,
        blocks=(),
        semantic_decisions=(),
        all_decisions=tuple(decisions),
        semantic_request_hashes=(),
        block_results=(),
    )


def make_scenario(index, a3_input):
    """Build a self-consistent (planning, finalization, identity) from an index.

    The planning is the REAL A4B plan (``plan_candidate_index_v1``), so the
    identity's ``plan_hash`` equals the replanned plan_hash (the shared verifier
    requires this). The identity is built from the zero-provider preparation.
    """
    planning = plan_candidate_index_v1(index)
    finalization = finalize_reconciliation(make_semantic(planning, planning.decisions))
    sem_profile = load_semantic_profile(A4_LLM_PROFILE_PATH)
    prep = prepare_semantic_resolution(planning, _profile(), sem_profile)
    identity = build_a4_semantic_identity(_profile(), prep, planning)
    return planning, finalization, identity


# ---------------------------------------------------------------------------
# Semantic (LLM) scenario helpers -- for the block-binding / evidence tests
# ---------------------------------------------------------------------------

# Weak (single-token) names so each disjoint pair is needs_semantic_decision.
WEAK_NAMES = ("Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta", "Theta")


def make_evidence(i: int) -> tuple[EvidenceRef, ...]:
    return (
        EvidenceRef(
            paragraph_id=f"P{i:03d}",
            role="primary",
            strength="explicit",
            excerpt=f"evidence for candidate {i}",
        ),
    )


def semantic_index(n_pairs: int = 8) -> CandidateEntityIndex:
    """Build an index of ``n_pairs`` disjoint character pairs, each sharing a
    unique WEAK identity key (single token) so the A4B plan marks every pair
    ``needs_semantic_decision``. Chunks are spaced by 100 so no adjacent-chunk
    (or same-chunk) pairs are generated -- exactly ``n_pairs`` pairs result.
    """
    entries = []
    for i in range(1, n_pairs * 2 + 1):
        ref = f"CH001_C001:cand_char_{i:03d}"
        name = WEAK_NAMES[(i - 1) // 2]
        chunk = i * 100  # spaced: no two candidates are in the same/adjacent chunk
        entries.append(
            idx_entry(
                ref,
                display=name,
                source_key=frozen_key(chunk, 1, 1, i, ref),
                ext_ref=EXT_A,
                evidence=make_evidence(i),
            )
        )
    return CandidateEntityIndex(
        schema_version=CANDIDATE_ENTITY_INDEX_SCHEMA_VERSION,
        entries=tuple(entries),
    )


def make_provenance(prep, request_hash: str):
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
        usage=None,
    )


def make_llm_decision(
    left: str,
    right: str,
    decision: str,
    request_hash: str,
    prep,
    *,
    evidence: tuple[EvidenceRef, ...] = (),
    reason_code: str | None = None,
    reason_zh: str = "语义判定",
) -> ReconciliationDecision:
    rc = reason_code if reason_code is not None else llm_reason_code(decision)
    decision_id = compute_llm_decision_id(
        left_ref=left,
        right_ref=right,
        decision=decision,
        method="llm",
        reason_code=rc,
        reason_zh=reason_zh,
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
        reason_zh=reason_zh,
        evidence_refs=evidence,
        prompt_id=prep.prompt_id,
        prompt_version=prep.prompt_version,
        generation_provenance=make_provenance(prep, request_hash),
    )


def make_semantic_scenario(
    n_pairs: int = 8,
    *,
    decision: str = "same_entity",
    use_evidence: bool = True,
) -> tuple:
    """Build a self-consistent semantic scenario with ``n_pairs`` LLM decisions.

    Returns (planning, finalization, identity, request_hashes, pair_to_hash, prep,
    index). Every needs_semantic_decision pair gets a valid LLM decision whose
    provenance request_hash is the request hash of the exact block that contains
    the pair (via the shared packing authority).
    """
    index = semantic_index(n_pairs)
    planning = plan_candidate_index_v1(index)
    sem_profile = load_semantic_profile(A4_LLM_PROFILE_PATH)
    prep = prepare_semantic_resolution(planning, _profile(), sem_profile)
    request_hashes = prep.semantic_request_hashes
    blocks = pack_semantic_pairs_v1(planning)
    pair_to_hash: dict[tuple[str, str], str] = {}
    for block_ordinal, block in enumerate(blocks):
        for plan in block:
            pair_to_hash[(plan.left_candidate_ref, plan.right_candidate_ref)] = (
                request_hashes[block_ordinal]
            )
    evidence_by_ref = {e.candidate_ref: e.evidence_refs for e in index.entries}
    llm_decisions = []
    for plan in planning.pair_plans:
        if plan.state != PAIR_STATE_NEEDS_SEMANTIC_DECISION:
            continue
        left, right = plan.left_candidate_ref, plan.right_candidate_ref
        ev = evidence_by_ref[left] if use_evidence else ()
        llm_decisions.append(
            make_llm_decision(
                left, right, decision, pair_to_hash[(left, right)], prep, evidence=ev
            )
        )
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
    identity = build_a4_semantic_identity(_profile(), prep, planning)
    return planning, finalization, identity, request_hashes, pair_to_hash, prep, index


def clean_a3_input() -> A3InputIdentity:
    return A3InputIdentity(
        source_document_ref=make_ref("source_document", "doc-1"),
        chunk_manifest_ref=make_ref("chunk_manifest", "cm-1"),
        candidate_extraction_refs=(EXT_A, EXT_B),
        extraction_profile_id="a3-v1",
        extraction_profile_hash=H,
    )


def conflict_finalization():
    """A finalization result that carries a blocking finding (conflict).

    L same R, R same M, L different M -> a different_entity decision contradicts
    the L-R-M same component. The finalizer's own validation flags the conflict
    (``has_blocking_findings``), so ``publish_validated`` raises before the
    shared verifier runs.
    """
    M = "CH001_C004:cand_char_004"
    index = CandidateEntityIndex(
        schema_version=CANDIDATE_ENTITY_INDEX_SCHEMA_VERSION,
        entries=(
            idx_entry(L),
            idx_entry(R, source_key=frozen_key(2, 1, 1, 2, R)),
            idx_entry(M, source_key=frozen_key(4, 1, 1, 4, M)),
        ),
    )
    decisions = (
        ReconciliationDecision(
            decision_id="dec_0001", left_candidate_ref=L, right_candidate_ref=R,
            decision="same_entity", method="deterministic", reason_code="x", reason_zh="x",
            evidence_refs=(), prompt_id=None, prompt_version=None, generation_provenance=None,
        ),
        ReconciliationDecision(
            decision_id="dec_0002", left_candidate_ref=R, right_candidate_ref=M,
            decision="same_entity", method="deterministic", reason_code="x", reason_zh="x",
            evidence_refs=(), prompt_id=None, prompt_version=None, generation_provenance=None,
        ),
        ReconciliationDecision(
            decision_id="dec_0003", left_candidate_ref=L, right_candidate_ref=M,
            decision="different_entity", method="deterministic", reason_code="x", reason_zh="x",
            evidence_refs=(), prompt_id=None, prompt_version=None, generation_provenance=None,
        ),
    )
    planning = make_planning(index, (), decisions)
    return finalize_reconciliation(make_semantic(planning, decisions))


@dataclass
class Harness:
    store: FileArtifactStore
    pointers: FilePointerStore
    service: ReconciliationPersistenceService
    profile: object
    a3_input: A3InputIdentity
    semantic_identity: A4SemanticIdentity
    finalization: object


def make_harness(tmp_path: Path) -> Harness:
    store = FileArtifactStore(tmp_path / "artifacts")
    pointers = FilePointerStore(tmp_path / "pointers", store)
    a3_input = clean_a3_input()
    _planning, finalization, identity = make_scenario(clean_index(), a3_input)
    return Harness(
        store=store,
        pointers=pointers,
        service=ReconciliationPersistenceService(store, pointers),
        profile=_profile(),
        a3_input=a3_input,
        semantic_identity=identity,
        finalization=finalization,
    )


def publish(h: Harness, *, a3_input=None, semantic_identity=None, finalization=None):
    return h.service.publish_validated(
        project_id=PROJECT,
        document_id=DOCUMENT,
        reconciliation_profile=h.profile,
        semantic_identity=semantic_identity if semantic_identity is not None else h.semantic_identity,
        a3_input=a3_input if a3_input is not None else h.a3_input,
        finalization_result=finalization if finalization is not None else h.finalization,
    )


def reuse(h: Harness, *, a3_input=None, semantic_identity=None):
    return h.service.try_reuse_current(
        project_id=PROJECT,
        document_id=DOCUMENT,
        reconciliation_profile_id=h.profile.profile_id,
        a3_input=a3_input if a3_input is not None else h.a3_input,
        semantic_identity=semantic_identity if semantic_identity is not None else h.semantic_identity,
    )


def base_id() -> str:
    return a4_base_artifact_id(PROJECT, DOCUMENT, _profile().profile_id)


def pointer_id() -> str:
    return a4_pointer_id(PROJECT, DOCUMENT, _profile().profile_id)


def entity_map_id() -> str:
    return entity_map_artifact_id(base_id())


def store_path(store: FileArtifactStore, artifact_type: str, artifact_id: str, revision: int) -> Path:
    return store.root / artifact_type / artifact_id / f"r{revision:08d}.json"


def _current_target_ref(h: Harness) -> ArtifactRef | None:
    try:
        return h.pointers.resolve_current(pointer_id()).target_ref
    except PointerNotFoundError:
        return None


def _corrupt_artifact_file(h: Harness, artifact_type: str, artifact_id: str, revision: int) -> None:
    path = store_path(h.store, artifact_type, artifact_id, revision)
    path.write_bytes(b"this is not a valid artifact envelope")


def _profile():
    return load_entity_reconciliation_profile(RECON_PROFILE_PATH)


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


def test_artifact_identity_matches_frozen_contract():
    profile_id = _profile().profile_id
    assert base_id() == f"{PROJECT}.{DOCUMENT}.a4.{profile_id}"
    assert candidate_entity_index_artifact_id(base_id()) == f"{base_id()}.candidate-index"
    assert reconciliation_decision_set_artifact_id(base_id()) == f"{base_id()}.decisions"
    assert canonical_character_registry_artifact_id(base_id()) == f"{base_id()}.characters"
    assert canonical_location_registry_artifact_id(base_id()) == f"{base_id()}.locations"
    assert unresolved_entity_set_artifact_id(base_id()) == f"{base_id()}.unresolved"
    assert entity_map_id() == f"{base_id()}.entity-map"
    assert a4_validation_artifact_id(base_id()) == f"{base_id()}.entity-map.a4-validation"
    assert pointer_id() == f"{PROJECT}.a4.{DOCUMENT}.{profile_id}"


def test_validation_report_lineage(tmp_path):
    h = make_harness(tmp_path)
    pub = publish(h)
    report = load_validation_report(h.store, pub.validation_report_ref)
    roles = [ref.role for ref in report.validated_refs]
    assert "source_document" in roles
    assert "chunk_manifest" in roles
    assert "candidate_extraction_0001" in roles
    assert "candidate_extraction_0002" in roles
    for role in (
        "candidate_entity_index",
        "reconciliation_decision_set",
        "canonical_character_registry",
        "canonical_location_registry",
        "unresolved_entity_set",
        "entity_map",
    ):
        assert role in roles
    assert report.summary.result is ValidationResult.PASS
    assert report.findings == ()


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def test_first_valid_publish(tmp_path):
    h = make_harness(tmp_path)
    pub = publish(h)
    assert pub.reused is False
    assert pub.entity_map_ref.revision == 1
    assert pub.validation_report_ref.revision == 1
    assert pub.entity_map_ref.artifact_id == entity_map_id()
    # All six outputs share one run revision.
    loaded_map = load_entity_map(h.store, pub.entity_map_ref, expected_artifact_id=entity_map_id())
    for ref in (
        loaded_map.candidate_entity_index_ref,
        loaded_map.reconciliation_decision_set_ref,
        loaded_map.canonical_character_registry_ref,
        loaded_map.canonical_location_registry_ref,
        loaded_map.unresolved_entity_set_ref,
    ):
        assert ref.revision == 1
    # Pointer created at CURRENT.
    pointer = h.pointers.resolve_current(pointer_id())
    assert pointer.pointer_kind is PointerKind.CURRENT
    assert pointer.target_ref == pub.entity_map_ref


def test_typed_loader_round_trip(tmp_path):
    h = make_harness(tmp_path)
    pub = publish(h)
    loaded = load_entity_map(h.store, pub.entity_map_ref, expected_artifact_id=entity_map_id())
    assert loaded.semantic_identity == h.semantic_identity
    assert loaded.a3_input == h.a3_input
    assert loaded.schema_version == 2


def test_typed_loader_outputs_round_trip(tmp_path):
    h = make_harness(tmp_path)
    pub = publish(h)
    loaded_map = load_entity_map(h.store, pub.entity_map_ref, expected_artifact_id=entity_map_id())
    b = base_id()
    index = load_candidate_entity_index(
        h.store, loaded_map.candidate_entity_index_ref,
        expected_artifact_id=candidate_entity_index_artifact_id(b),
    )
    assert {e.candidate_ref for e in index.entries} == {L, R, LOC1, LOC2}
    decisions = load_reconciliation_decision_set(
        h.store, loaded_map.reconciliation_decision_set_ref,
        expected_artifact_id=reconciliation_decision_set_artifact_id(b),
    )
    assert len(decisions.decisions) == 2
    assert all(d.method == "deterministic" for d in decisions.decisions)
    unresolved = load_unresolved_entity_set(
        h.store, loaded_map.unresolved_entity_set_ref,
        expected_artifact_id=unresolved_entity_set_artifact_id(b),
    )
    # No unresolved passthrough and no uncertain groups in the clean scenario.
    assert unresolved.entities == ()


def test_bad_artifact_type_fails(tmp_path):
    h = make_harness(tmp_path)
    pub = publish(h)
    wrong_type = replace(pub.entity_map_ref, artifact_type="not_an_entity_map")
    with pytest.raises(StoryIntegrityError):
        load_entity_map(h.store, wrong_type, expected_artifact_id=entity_map_id())


def test_artifact_id_mismatch_fails(tmp_path):
    h = make_harness(tmp_path)
    pub = publish(h)
    with pytest.raises(StoryIntegrityError, match="logical identity"):
        load_entity_map(h.store, pub.entity_map_ref, expected_artifact_id="some.other.id")


def test_entity_map_v1_payload_rejected(tmp_path):
    h = make_harness(tmp_path)
    pub = publish(h)
    payload = json.loads(store_path(h.store, ENTITY_MAP_ARTIFACT_TYPE, entity_map_id(), 1).read_text())
    # Downgrade to a v1-style payload (drop semantic_identity, bump envelope is
    # impossible; instead corrupt the payload so the schema gate fails).
    env = h.store.get(ENTITY_MAP_ARTIFACT_TYPE, entity_map_id(), 1)
    bad_payload = dict(env.payload)
    del bad_payload["semantic_identity"]
    bad_payload["schema_version"] = 1
    from short_drama.artifacts import ImmutableArtifactEnvelope

    # A new revision with a v1 payload must be rejected by the typed loader.
    bad = ImmutableArtifactEnvelope.create(
        artifact_type=ENTITY_MAP_ARTIFACT_TYPE,
        artifact_id=entity_map_id(),
        revision=9,
        schema_version=1,
        payload=bad_payload,
    )
    ref = h.store.put(bad)
    with pytest.raises(StoryIntegrityError):
        load_entity_map(h.store, ref, expected_artifact_id=entity_map_id())


# ---------------------------------------------------------------------------
# Reuse
# ---------------------------------------------------------------------------


def test_exact_reuse_same_identity(tmp_path):
    h = make_harness(tmp_path)
    first = publish(h)
    second = reuse(h)
    assert second is not None
    assert second.reused is True
    assert second.entity_map_ref == first.entity_map_ref
    assert second.validation_report_ref == first.validation_report_ref
    assert second.current_pointer_ref == first.current_pointer_ref


def test_republish_same_identity_reuses_not_new_revision(tmp_path):
    h = make_harness(tmp_path)
    first = publish(h)
    second = publish(h)  # post-provider race: same identity, valid current
    assert second.reused is True
    assert second.entity_map_ref == first.entity_map_ref
    # No second revision was allocated.
    with pytest.raises(Exception):
        h.store.get(ENTITY_MAP_ARTIFACT_TYPE, entity_map_id(), 2)


def test_no_current_is_a_miss(tmp_path):
    h = make_harness(tmp_path)
    assert reuse(h) is None


def test_a3_input_change_invalidates(tmp_path):
    h = make_harness(tmp_path)
    first = publish(h)
    new_a3 = replace(
        h.a3_input,
        candidate_extraction_refs=h.a3_input.candidate_extraction_refs
        + (make_ref("candidate_extraction", "ce-99"),),
    )
    second = publish(h, a3_input=new_a3)
    assert second.reused is False
    assert second.entity_map_ref.revision == 2
    assert second.entity_map_ref.artifact_id == first.entity_map_ref.artifact_id
    # pre-generation reuse with the new a3 is a normal miss
    assert reuse(h, a3_input=new_a3) is not None
    assert reuse(h, a3_input=new_a3).reused is True


def test_semantic_identity_change_invalidates(tmp_path):
    h = make_harness(tmp_path)
    first = publish(h)
    # A different candidate index -> a different plan_hash -> a different A4
    # identity (same a3_input). Publishing it must supersede, not reuse.
    _p, finalization, identity = make_scenario(different_index(), h.a3_input)
    second = publish(h, finalization=finalization, semantic_identity=identity)
    assert second.reused is False
    assert second.entity_map_ref.revision == 2
    # The old identity is no longer current; the new one is.
    assert _current_target_ref(h) == second.entity_map_ref
    assert reuse(h, semantic_identity=identity).reused is True
    # The old identity is a normal miss (different from current).
    assert reuse(h) is None


def test_backend_switch_does_not_change_reuse_identity(tmp_path):
    h = make_harness(tmp_path)
    # The A4 semantic identity carries no provider/model metadata.
    payload = h.semantic_identity.to_dict()
    assert "provider" not in payload
    assert "model" not in payload
    # So two runs with the same backend-neutral identity reuse regardless of
    # (untracked) backend routing.
    first = publish(h)
    second = publish(h)
    assert second.reused is True
    assert second.entity_map_ref == first.entity_map_ref


def test_stale_identity_supersedes_same_artifact_id(tmp_path):
    h = make_harness(tmp_path)
    first = publish(h)
    _p, finalization, identity = make_scenario(different_index(), h.a3_input)
    second = publish(h, finalization=finalization, semantic_identity=identity)
    assert first.entity_map_ref.revision == 1
    assert second.entity_map_ref.revision == 2
    assert first.entity_map_ref.artifact_id == second.entity_map_ref.artifact_id
    # First revision remains exactly resolvable (history preserved).
    assert load_entity_map(h.store, first.entity_map_ref, expected_artifact_id=entity_map_id())


def test_no_historical_auto_resurrection(tmp_path):
    h = make_harness(tmp_path)
    first = publish(h)
    _p, finalization, identity = make_scenario(different_index(), h.a3_input)
    publish(h, finalization=finalization, semantic_identity=identity)  # rev 2
    # Re-request the OLD identity: it publishes a NEW revision (rev 3), it does not
    # resurrect historical revision 1.
    third = publish(h)
    assert third.reused is False
    assert third.entity_map_ref.revision == 3
    assert third.entity_map_ref != first.entity_map_ref


# ---------------------------------------------------------------------------
# Verification / fail-closed
# ---------------------------------------------------------------------------


def test_missing_report_prevents_reuse_fails_closed(tmp_path):
    h = make_harness(tmp_path)
    # Publish, then delete the validation report so the current is corrupt.
    pub = publish(h)
    report_path = store_path(
        h.store, "validation_report", a4_validation_artifact_id(base_id()), pub.entity_map_ref.revision
    )
    report_path.unlink()
    with pytest.raises(StoryIntegrityError, match="ValidationReport"):
        reuse(h)


def test_fail_report_cannot_back_current(tmp_path):
    h = make_harness(tmp_path)
    pub = publish(h)
    # Overwrite the backing A4 ValidationReport on disk with a non-PASS report of
    # the same lineage. Re-verification must fail closed (the report no longer
    # matches the exact expected PASS report / is non-PASS).
    map_ref = pub.entity_map_ref
    loaded = load_entity_map(h.store, map_ref, expected_artifact_id=entity_map_id())
    base_lineage = build_a4_validation_report(loaded, map_ref).validated_refs
    bad_report = ValidationReport(
        validated_refs=base_lineage,
        findings=(
            ValidationFinding(
                finding_id="a4_fail",
                code="A4_FAIL",
                severity=ValidationSeverity.BLOCKING,
                owner_stage="A4",
                repair_route="rerun_a4",
                message="fail",
            ),
        ),
    )
    assert bad_report.summary.result is ValidationResult.FAIL
    env = validation_report_envelope(
        bad_report, artifact_id=a4_validation_artifact_id(base_id()), revision=1
    )
    path = store_path(
        h.store, "validation_report", a4_validation_artifact_id(base_id()), 1
    )
    path.write_bytes(env.canonical_bytes())
    with pytest.raises(StoryIntegrityError, match="ValidationReport"):
        reuse(h)


def test_corrupt_current_reuse_fails_closed(tmp_path):
    h = make_harness(tmp_path)
    pub = publish(h)
    # Corrupt a pinned output artifact. The CURRENT pointer still resolves the
    # entity map (its target is intact), but full A4 verification must fail closed.
    _corrupt_artifact_file(
        h, CANDIDATE_ENTITY_INDEX_ARTIFACT_TYPE,
        candidate_entity_index_artifact_id(base_id()), pub.entity_map_ref.revision,
    )
    with pytest.raises(StoryIntegrityError):
        reuse(h)


def test_corrupt_current_publish_fails_closed_zero_persistence(tmp_path):
    h = make_harness(tmp_path)
    pub = publish(h)
    _corrupt_artifact_file(
        h, CANDIDATE_ENTITY_INDEX_ARTIFACT_TYPE,
        candidate_entity_index_artifact_id(base_id()), pub.entity_map_ref.revision,
    )
    # A DIFFERENT identity publish must fail closed (corrupt current is fully
    # verified first) and never write a new revision.
    with pytest.raises(StoryIntegrityError):
        publish(h, semantic_identity=replace(h.semantic_identity, plan_hash="c" * 64))
    # No new revision was written; pointer is unchanged (still points at rev 1).
    with pytest.raises(Exception):
        h.store.get(ENTITY_MAP_ARTIFACT_TYPE, entity_map_id(), 2)
    assert _current_target_ref(h) == pub.entity_map_ref


def test_wrong_logical_target_current_fails_closed(tmp_path):
    h = make_harness(tmp_path)
    pub = publish(h)
    # Persist a VALID entity map under a DIFFERENT logical id, then point the A4
    # CURRENT pointer at it. The target resolves (transitive integrity passes),
    # but it is not the expected logical EntityMap, so verification fails closed.
    from short_drama.artifacts import ImmutableArtifactEnvelope

    other_id = "other.logical.id"
    payload = h.store.get(ENTITY_MAP_ARTIFACT_TYPE, entity_map_id(), 1).payload
    other_env = ImmutableArtifactEnvelope.create(
        artifact_type=ENTITY_MAP_ARTIFACT_TYPE,
        artifact_id=other_id,
        revision=1,
        schema_version=2,
        payload=payload,
    )
    other_ref = h.store.put(other_env)
    h.pointers.compare_and_set(
        pointer_id=pointer_id(),
        pointer_kind=PointerKind.CURRENT,
        expected_pointer_ref=pub.current_pointer_ref,
        target_ref=other_ref,
    )
    with pytest.raises(StoryIntegrityError, match="different logical EntityMap"):
        reuse(h)


def test_output_revision_mismatch_fails_closed(tmp_path):
    h = make_harness(tmp_path)
    pub = publish(h)
    # Publish the SAME map content at a different run revision and point CURRENT
    # there: the pinned outputs (rev 1) no longer share a run revision with the
    # entity map (rev 5) -> incoherent run revision, fail closed.
    from short_drama.artifacts import ImmutableArtifactEnvelope

    map_ref = pub.entity_map_ref
    loaded = load_entity_map(h.store, map_ref, expected_artifact_id=entity_map_id())
    env = ImmutableArtifactEnvelope.create(
        artifact_type=ENTITY_MAP_ARTIFACT_TYPE,
        artifact_id=entity_map_id(),
        revision=5,
        schema_version=2,
        payload=loaded.to_dict(),
    )
    ref = h.store.put(env)
    h.pointers.compare_and_set(
        pointer_id=pointer_id(),
        pointer_kind=PointerKind.CURRENT,
        expected_pointer_ref=pub.current_pointer_ref,
        target_ref=ref,
    )
    with pytest.raises(StoryIntegrityError, match="run revision"):
        reuse(h)


# ---------------------------------------------------------------------------
# Orphan skipping + zero persistence
# ---------------------------------------------------------------------------


def test_orphan_revision_skip(tmp_path):
    h = make_harness(tmp_path)
    # Simulate a failed publication that left a partial revision-1 artifact
    # (an entity map that is never pointed to). The next publish must skip it.
    from short_drama.story.reconciliation import (
        RECONCILIATION_DECISION_SET_SCHEMA_VERSION,
    )

    # Build a valid entity map payload to occupy revision 1.
    dummy_map = EntityMap(
        schema_version=2,
        entries=(),
        candidate_entity_index_ref=make_ref(CANDIDATE_ENTITY_INDEX_ARTIFACT_TYPE, candidate_entity_index_artifact_id(base_id()), 1),
        reconciliation_decision_set_ref=make_ref(RECONCILIATION_DECISION_SET_ARTIFACT_TYPE, reconciliation_decision_set_artifact_id(base_id()), 1),
        canonical_character_registry_ref=make_ref(CANONICAL_CHARACTER_REGISTRY_ARTIFACT_TYPE, canonical_character_registry_artifact_id(base_id()), 1),
        canonical_location_registry_ref=make_ref(CANONICAL_LOCATION_REGISTRY_ARTIFACT_TYPE, canonical_location_registry_artifact_id(base_id()), 1),
        unresolved_entity_set_ref=make_ref(UNRESOLVED_ENTITY_SET_ARTIFACT_TYPE, unresolved_entity_set_artifact_id(base_id()), 1),
        a3_input=h.a3_input,
        semantic_identity=h.semantic_identity,
    )
    persist_entity_map(h.store, dummy_map, artifact_id=entity_map_id(), revision=1)
    # No pointer: a fresh publish must skip the orphaned revision 1.
    revision = next_a4_revision(h.store, base=base_id(), current_entity_map_ref=None)
    assert revision == 2
    pub = publish(h)
    assert pub.entity_map_ref.revision == 2


def test_blocking_findings_prevent_publish_no_partial_write(tmp_path):
    h = make_harness(tmp_path)
    bad = conflict_finalization()
    assert bad.has_blocking_findings is True
    with pytest.raises(ReconciliationFinalizationError):
        publish(h, finalization=bad)
    # No artifacts were written at all.
    with pytest.raises(Exception):
        h.store.get(ENTITY_MAP_ARTIFACT_TYPE, entity_map_id(), 1)
    assert _current_target_ref(h) is None


def test_blocking_findings_do_not_replace_valid_current(tmp_path):
    h = make_harness(tmp_path)
    first = publish(h)
    bad = conflict_finalization()
    with pytest.raises(ReconciliationFinalizationError):
        publish(h, finalization=bad)
    assert _current_target_ref(h) == first.entity_map_ref


# ---------------------------------------------------------------------------
# Semantic identity parity (build_a4_semantic_identity == A4C prepare)
# ---------------------------------------------------------------------------


def test_semantic_identity_equals_prepare_request_hashes(tmp_path):
    profile = _profile()
    sem_profile = load_semantic_profile(A4_LLM_PROFILE_PATH)
    # A needs_llm pair produces a real request hash.
    M = "CH001_C004:cand_char_004"
    index = CandidateEntityIndex(
        schema_version=CANDIDATE_ENTITY_INDEX_SCHEMA_VERSION,
        entries=(idx_entry(L), idx_entry(R, source_key="CH001_C002:P0002")),
    )
    planning = make_planning(
        index,
        (
            ReconciliationPairPlan(
                left_candidate_ref=L, right_candidate_ref=R,
                state="needs_semantic_decision", signals=(), shared_identity_keys=(), shared_tokens=(),
            ),
        ),
        (),
    )
    prep = prepare_semantic_resolution(planning, profile, sem_profile)
    identity = build_a4_semantic_identity(profile, prep, planning)
    assert identity.semantic_request_hashes == prep.semantic_request_hashes
    assert identity.semantic_request_hashes == tuple(
        req.request_hash for req in prep.structured_requests
    )
    assert len(identity.semantic_request_hashes) == 1


# ---------------------------------------------------------------------------
# Shared verifier gates (Blocker 1 / 2 / 3 regression coverage)


def _verifier_kwargs(h):
    return dict(
        index=h.finalization.candidate_index,
        decision_set=h.finalization.decision_set,
        char_registry=h.finalization.canonical_character_registry,
        loc_registry=h.finalization.canonical_location_registry,
        unresolved_set=h.finalization.unresolved_entity_set,
        entity_map_entries=h.finalization.entity_map_entries,
        a3_input=h.a3_input,
        semantic_identity=h.semantic_identity,
        profile=h.profile,
    )


def test_verifier_passes_clean(tmp_path):
    h = make_harness(tmp_path)
    _verify_finalization_bundle(**_verifier_kwargs(h))  # no error


def test_verifier_source_order_gate(tmp_path):
    h = make_harness(tmp_path)
    kw = _verifier_kwargs(h)
    kw["index"] = CandidateEntityIndex(
        schema_version=CANDIDATE_ENTITY_INDEX_SCHEMA_VERSION,
        entries=(idx_entry(L, source_key="BAD:KEY:FORMAT"),),
    )
    with pytest.raises(StoryIntegrityError):
        _verify_finalization_bundle(**kw)


def test_verifier_duplicate_decision_id(tmp_path):
    h = make_harness(tmp_path)
    kw = _verifier_kwargs(h)
    first = h.finalization.decision_set.decisions[0]
    kw["decision_set"] = ReconciliationDecisionSet(
        schema_version=RECONCILIATION_DECISION_SET_SCHEMA_VERSION,
        decisions=(first, first),
    )
    with pytest.raises(StoryIntegrityError):
        _verify_finalization_bundle(**kw)


def test_verifier_plan_hash_parity(tmp_path):
    h = make_harness(tmp_path)
    # A plan_hash that does not match the replanned plan fails closed.
    kw = _verifier_kwargs(h)
    kw["semantic_identity"] = replace(h.semantic_identity, plan_hash="0" * 64)
    with pytest.raises(StoryIntegrityError):
        _verify_finalization_bundle(**kw)


def test_verifier_output_parity(tmp_path):
    h = make_harness(tmp_path)
    kw = _verifier_kwargs(h)
    # Drop every canonical character: it no longer equals the graph-derived output.
    kw["char_registry"] = CanonicalCharacterRegistry(
        schema_version=CANONICAL_ENTITY_REGISTRY_SCHEMA_VERSION,
        entities=(),
    )
    with pytest.raises(StoryIntegrityError):
        _verify_finalization_bundle(**kw)


def test_verifier_decision_mismatch(tmp_path):
    h = make_harness(tmp_path)
    kw = _verifier_kwargs(h)
    decisions = h.finalization.decision_set.decisions
    flipped = tuple(
        replace(d, decision="different_entity") if i == 0 else d
        for i, d in enumerate(decisions)
    )
    kw["decision_set"] = ReconciliationDecisionSet(
        schema_version=RECONCILIATION_DECISION_SET_SCHEMA_VERSION,
        decisions=flipped,
    )
    with pytest.raises(StoryIntegrityError):
        _verify_finalization_bundle(**kw)


def test_verifier_semantic_request_hash_mismatch(tmp_path):
    h = make_harness(tmp_path)
    # An extra request hash not backed by an LLM decision fails closed.
    kw = _verifier_kwargs(h)
    kw["semantic_identity"] = replace(
        h.semantic_identity, semantic_request_hashes=("f" * 64,)
    )
    with pytest.raises(StoryIntegrityError):
        _verify_finalization_bundle(**kw)


def test_verifier_a3_binding(tmp_path):
    h = make_harness(tmp_path)
    # An a3_input that lacks the index's extraction refs fails closed.
    kw = _verifier_kwargs(h)
    kw["a3_input"] = replace(h.a3_input, candidate_extraction_refs=(EXT_A,))
    with pytest.raises(StoryIntegrityError):
        _verify_finalization_bundle(**kw)


def test_verifier_profile_binding(tmp_path):
    h = make_harness(tmp_path)
    # A different reconciliation profile fails closed.
    kw = _verifier_kwargs(h)
    kw["profile"] = replace(h.profile, profile_id="other-profile")
    with pytest.raises(StoryIntegrityError):
        _verify_finalization_bundle(**kw)


# ---------------------------------------------------------------------------
# Reconciliation-profile binding (full prompt/schema identity)
# ---------------------------------------------------------------------------


def _semantic_bundle(
    planning, fin, ident, *, decision_set=None, index=None, a3_input=None, profile=None
):
    """Build ``_verify_finalization_bundle`` kwargs from a semantic scenario."""
    return dict(
        index=index if index is not None else fin.decision_set.candidate_index,
        decision_set=decision_set if decision_set is not None else fin.decision_set,
        char_registry=fin.canonical_character_registry,
        loc_registry=fin.canonical_location_registry,
        unresolved_set=fin.unresolved_entity_set,
        entity_map_entries=fin.entity_map_entries,
        a3_input=a3_input if a3_input is not None else clean_a3_input(),
        semantic_identity=ident,
        profile=profile if profile is not None else _profile(),
    )


def test_profile_binding_correct_full_profile_passes(tmp_path):
    planning, fin, ident, rh, pair_to_hash, prep, index = make_semantic_scenario(7)
    _verify_finalization_bundle(**_semantic_bundle(planning, fin, ident, index=index))


def test_profile_binding_prompt_id_mismatch(tmp_path):
    h = make_harness(tmp_path)
    kw = _verifier_kwargs(h)
    kw["semantic_identity"] = replace(h.semantic_identity, prompt_id="other-prompt")
    with pytest.raises(StoryIntegrityError):
        _verify_finalization_bundle(**kw)


def test_profile_binding_prompt_version_mismatch(tmp_path):
    h = make_harness(tmp_path)
    kw = _verifier_kwargs(h)
    kw["semantic_identity"] = replace(
        h.semantic_identity, prompt_version=h.profile.prompt_version + 1
    )
    with pytest.raises(StoryIntegrityError):
        _verify_finalization_bundle(**kw)


def test_profile_binding_output_schema_id_mismatch(tmp_path):
    h = make_harness(tmp_path)
    kw = _verifier_kwargs(h)
    kw["semantic_identity"] = replace(h.semantic_identity, output_schema_id="other-schema")
    with pytest.raises(StoryIntegrityError):
        _verify_finalization_bundle(**kw)


def test_profile_binding_output_schema_version_mismatch(tmp_path):
    h = make_harness(tmp_path)
    kw = _verifier_kwargs(h)
    kw["semantic_identity"] = replace(
        h.semantic_identity, output_schema_version=h.profile.output_schema_version + 1
    )
    with pytest.raises(StoryIntegrityError):
        _verify_finalization_bundle(**kw)


# ---------------------------------------------------------------------------
# Semantic block request-hash binding + LLM persisted invariants
# ---------------------------------------------------------------------------


def _with_llm_decision(decision_set, match_key, new_decision):
    """Return a decision set with the LLM decision for ``match_key`` replaced."""
    decisions = tuple(
        new_decision
        if (
            d.method == "llm"
            and (d.left_candidate_ref, d.right_candidate_ref) == match_key
        )
        else d
        for d in decision_set.decisions
    )
    return replace(decision_set, decisions=decisions)


def _rebuild_llm_decision(
    d: ReconciliationDecision, *, evidence=None, reason_code=None, request_hash=None
):
    """Rebuild an LLM decision with optional field overrides, recomputing the
    decision id from the (possibly overridden) fields so the decision-id
    authority stays satisfied."""
    ev = evidence if evidence is not None else d.evidence_refs
    rc = reason_code if reason_code is not None else llm_reason_code(d.decision)
    rh = request_hash if request_hash is not None else d.generation_provenance.request_hash
    decision_id = compute_llm_decision_id(
        left_ref=d.left_candidate_ref,
        right_ref=d.right_candidate_ref,
        decision=d.decision,
        method="llm",
        reason_code=rc,
        reason_zh=d.reason_zh,
        evidence_refs=ev,
        prompt_id=d.prompt_id,
        prompt_version=d.prompt_version,
        request_hash=rh,
    )
    provenance = d.generation_provenance
    if request_hash is not None and request_hash != d.generation_provenance.request_hash:
        provenance = make_provenance_from_provenance(provenance, request_hash)
    return replace(
        d,
        decision_id=decision_id,
        evidence_refs=ev,
        reason_code=rc,
        generation_provenance=provenance,
    )


def make_provenance_from_provenance(provenance, request_hash: str):
    return replace(provenance, request_hash=request_hash)


def test_semantic_block_mapping_correct_passes(tmp_path):
    """7 disjoint pairs -> 2 blocks; the correct pair->block-hash mapping
    (via the shared A4C packing authority) passes."""
    planning, fin, ident, rh, pair_to_hash, prep, index = make_semantic_scenario(7)
    assert len(rh) == 2  # 7 pairs -> 2 blocks (6 + 1)
    _verify_finalization_bundle(**_semantic_bundle(planning, fin, ident, index=index))


def test_semantic_block_cross_block_wrong_hash_fails(tmp_path):
    """8 pairs -> block1 (6), block2 (2). Corrupt block2's first pair to h1
    (block1's hash) while block2's second pair keeps h2 -- so the first-
    occurrence sequence is still (h1, h2) but the exact block binding fails."""
    planning, fin, ident, rh, pair_to_hash, prep, index = make_semantic_scenario(8)
    h1, h2 = rh
    blocks = pack_semantic_pairs_v1(planning)
    block2_pairs = [
        (pl.left_candidate_ref, pl.right_candidate_ref) for pl in blocks[1]
    ]
    assert len(block2_pairs) >= 2
    corrupt_key = block2_pairs[0]
    corrupt_dec = next(
        d
        for d in fin.decision_set.decisions
        if d.method == "llm"
        and (d.left_candidate_ref, d.right_candidate_ref) == corrupt_key
    )
    new_dec = _rebuild_llm_decision(corrupt_dec, request_hash=h1)
    decision_set = _with_llm_decision(fin.decision_set, corrupt_key, new_dec)
    with pytest.raises(StoryIntegrityError):
        _verify_finalization_bundle(
            **_semantic_bundle(planning, fin, ident, index=index, decision_set=decision_set)
        )


def test_semantic_block_count_mismatch_fails(tmp_path):
    """The deterministic block count (2) must equal the identity's request-hash
    count. A 1-hash identity fails closed."""
    planning, fin, ident, rh, pair_to_hash, prep, index = make_semantic_scenario(7)
    bad_ident = replace(ident, semantic_request_hashes=(rh[0],))
    with pytest.raises(StoryIntegrityError):
        _verify_finalization_bundle(
            index=index,
            decision_set=fin.decision_set,
            char_registry=fin.canonical_character_registry,
            loc_registry=fin.canonical_location_registry,
            unresolved_set=fin.unresolved_entity_set,
            entity_map_entries=fin.entity_map_entries,
            a3_input=clean_a3_input(),
            semantic_identity=bad_ident,
            profile=_profile(),
        )


def test_llm_wrong_reason_code_fails(tmp_path):
    """An LLM decision whose reason_code does not equal the deterministic A4C
    authority for its decision fails closed (even with a valid decision id)."""
    planning, fin, ident, rh, pair_to_hash, prep, index = make_semantic_scenario(7)
    dec = next(d for d in fin.decision_set.decisions if d.method == "llm")
    assert dec.decision == "same_entity"
    new_dec = replace(dec, reason_code="llm_different_entity")  # decision id unchanged
    decision_set = _with_llm_decision(
        fin.decision_set,
        (dec.left_candidate_ref, dec.right_candidate_ref),
        new_dec,
    )
    with pytest.raises(StoryIntegrityError):
        _verify_finalization_bundle(
            **_semantic_bundle(planning, fin, ident, index=index, decision_set=decision_set)
        )


def test_llm_evidence_foreign_ref_fails(tmp_path):
    """An evidence ref that is not exact evidence of either endpoint fails
    closed."""
    planning, fin, ident, rh, pair_to_hash, prep, index = make_semantic_scenario(7)
    dec = next(d for d in fin.decision_set.decisions if d.method == "llm")
    foreign = make_evidence(999)
    new_dec = _rebuild_llm_decision(dec, evidence=foreign)
    decision_set = _with_llm_decision(
        fin.decision_set,
        (dec.left_candidate_ref, dec.right_candidate_ref),
        new_dec,
    )
    with pytest.raises(StoryIntegrityError):
        _verify_finalization_bundle(
            **_semantic_bundle(planning, fin, ident, index=index, decision_set=decision_set)
        )


def test_llm_evidence_modified_excerpt_fails(tmp_path):
    """An evidence ref whose excerpt was modified (no longer exact-equal to the
    endpoint's evidence) fails closed."""
    planning, fin, ident, rh, pair_to_hash, prep, index = make_semantic_scenario(7)
    dec = next(d for d in fin.decision_set.decisions if d.method == "llm")
    original = dec.evidence_refs[0]
    modified = replace(original, excerpt="modified " + original.excerpt)
    new_dec = _rebuild_llm_decision(dec, evidence=(modified,))
    decision_set = _with_llm_decision(
        fin.decision_set,
        (dec.left_candidate_ref, dec.right_candidate_ref),
        new_dec,
    )
    with pytest.raises(StoryIntegrityError):
        _verify_finalization_bundle(
            **_semantic_bundle(planning, fin, ident, index=index, decision_set=decision_set)
        )


def test_llm_evidence_duplicate_fails(tmp_path):
    """A duplicated evidence ref within one decision fails closed."""
    planning, fin, ident, rh, pair_to_hash, prep, index = make_semantic_scenario(7)
    dec = next(d for d in fin.decision_set.decisions if d.method == "llm")
    ev = dec.evidence_refs[0]
    new_dec = _rebuild_llm_decision(dec, evidence=(ev, ev))
    decision_set = _with_llm_decision(
        fin.decision_set,
        (dec.left_candidate_ref, dec.right_candidate_ref),
        new_dec,
    )
    with pytest.raises(StoryIntegrityError):
        _verify_finalization_bundle(
            **_semantic_bundle(planning, fin, ident, index=index, decision_set=decision_set)
        )


def test_llm_evidence_empty_passes(tmp_path):
    """Empty evidence is allowed: a semantic scenario where every LLM decision
    carries no evidence passes the verifier."""
    planning, fin, ident, rh, pair_to_hash, prep, index = make_semantic_scenario(
        7, use_evidence=False
    )
    assert all(d.evidence_refs == () for d in fin.decision_set.decisions if d.method == "llm")
    _verify_finalization_bundle(**_semantic_bundle(planning, fin, ident, index=index))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
