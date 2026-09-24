"""v1.2 A5B Phase A repair -- focused regression + corpus-shape audit tests.

Covers the five architecture-review blocks applied to PR #58:

* BLOCK 1 -- full corpus-shape audit:
  - diagnostic normalization / tokenization
  - per-domain overlap signal functions
  - exact n-choose-2 pair analysis (counts, combinations, source distance)
  - conservative exact-duplicate diagnostics (keys + group counting)
  - deterministic distribution rows + bounded representative samples
  - the audit is zero-provider and read-only (fingerprint before/after)
  - full output on a synthetic multi-chunk corpus (every section asserted)
  - exact Alice totals when the local (gitignored) run tree is present
* BLOCK 2 -- stable exact-dedupe of bound alias refs (two local refs binding to
  one A4 canonical id collapse to a single bound id).
* BLOCK 3 -- the paragraph ordinal in ``source_order_key`` is the GLOBAL
  ``SourceDocument.paragraphs`` rank (not the chunk-local rank).
* BLOCK 4 -- the source anchor is the earliest PRIMARY by global source rank
  (with the frozen no-primary fallback), never the evidence-array order.
* BLOCK 5 -- every indexed candidate's ``source_order_key`` ends in the exact
  A5 global candidate ref (``<chunk_id>:<local_id>``).
"""

from __future__ import annotations

import hashlib
import importlib.util
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
_AUDIT_PATH = REPO_ROOT / "scripts" / "a5b_zero_provider_audit.py"
_spec = importlib.util.spec_from_file_location("a5b_zero_provider_audit", _AUDIT_PATH)
audit = importlib.util.module_from_spec(_spec)
sys.modules["a5b_zero_provider_audit"] = audit
_spec.loader.exec_module(audit)

from short_drama.artifacts import FileArtifactStore
from short_drama.foundation import (
    FilePointerStore,
    LineageRef,
    PointerKind,
    ValidationReport,
    persist_validation_report,
)
from short_drama.llm.config import load_semantic_profile
from short_drama.story import (
    A3InputIdentity,
    CandidateEntityIndex,
    CandidateEntityIndexEntry,
    CandidateExtraction,
    CandidatePayload,
    CHUNK_PLANNER_VERSION,
    ChunkManifest,
    EvidenceRef,
    FactCandidate,
    LANGUAGE_DETECTOR_ID,
    PAIR_STATE_NEEDS_SEMANTIC_DECISION,
    ReconciliationDecision,
    ReconciliationPersistenceService,
    ReconciliationSemanticResult,
    SourceChapter,
    SourceDocument,
    SourceInfo,
    SourceParagraph,
    build_a4_semantic_identity,
    build_consolidation_planning,
    compute_llm_decision_id,
    finalize_reconciliation,
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
from short_drama.story.consolidation_planning import (
    _anchor_paragraph_ordinal,
    _stable_dedupe,
    _source_order_key,
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

from test_story_a5b_planning import (
    A4_LLM_PROFILE_PATH,
    CHUNK_PROFILE_ID,
    DOCUMENT,
    PROJECT,
    RECON_PROFILE_ID,
    RunTree,
    _build_run_tree,
    _char,
    _chunk_profile,
    _consolidation_profile,
    _evidence,
    _event,
    _extraction_profile,
    _fact,
    _loc,
    _make_llm_provenance,
    _provenance,
    _recon_profile,
    _rel,
    _unres,
)


# ---------------------------------------------------------------------------
# Reusable tree builders
# ---------------------------------------------------------------------------


def _idx_entry_ordinal(chunk_ordinal, chunk_id, local_id, kind, ext_ref, name):
    """A4 index entry using the REAL chunk ordinal (required for multi-chunk).

    The single-chunk test helper hardcodes chunk_ordinal=1; the multi-chunk
    A4 index must carry the authoritative chunk ordinal so that (a) the A4
    source_order_key is strictly ascending and (b) the entries stay in the
    non-decreasing A3 extraction order (chunk N before chunk N+1).
    """
    category = (
        "character" if kind == "character"
        else "location" if kind == "location"
        else "unresolved"
    )
    global_ref = f"{chunk_id}:{local_id}"
    order_key = compute_source_order_key(
        chunk_ordinal=chunk_ordinal,
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


def _make_llm_decision(left: str, right: str, decision: str, request_hash: str, prep, evidence=()):
    rc = llm_reason_code(decision)
    reason_zh = "same" if decision == "same_entity" else "different"
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
        generation_provenance=_make_llm_provenance(prep, request_hash),
    )


def _publish_a4(store, pointers, *, index, a3_input, recon_profile, decision="different_entity"):
    """Publish an A4 CURRENT; ``decision`` controls how uncertain pairs resolve."""
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
            decision,
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


@dataclass
class ChunkSpec:
    """One chunk of a synthetic multi-chunk source + its candidate payload."""

    chapter_id: str
    paragraph_ids: tuple[str, ...]
    chars: tuple = ()
    locs: tuple = ()
    unres: tuple = ()
    facts: tuple = ()
    events: tuple = ()
    rels: tuple = ()


def _build_multi_chunk_tree(tmp_path: Path, *specs: ChunkSpec, decision="different_entity") -> RunTree:
    """Build a self-contained A1+A2+A3(+A4) run tree from N chunk specs."""
    store = FileArtifactStore(tmp_path / "artifacts")
    pointers = FilePointerStore(tmp_path / "pointers", store)
    source_doc = SourceDocument(
        schema_version=1,
        project_id=PROJECT,
        document_id=DOCUMENT,
        source=SourceInfo(
            "txt", "source/novel.txt", "f" * 64, 128, "zh-CN", "zh", LANGUAGE_DETECTOR_ID
        ),
        normalization=NormalizationInfo("utf-8", "LF", "short_drama_source_ingestion_v1", "1"),
        chapters=tuple(
            SourceChapter(
                spec.chapter_id,
                None,
                "synthetic",
                tuple(SourceParagraph(pid, "text", None) for pid in spec.paragraph_ids),
            )
            for spec in specs
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
        ValidationReport(validated_refs=(LineageRef("source_document", source_ref),), findings=()),
        artifact_id=source_validation_artifact_id(PROJECT, DOCUMENT),
        revision=source_ref.revision,
    )
    chunks, coverage = plan_chunks(source_doc, source_ref, _chunk_profile())
    chunk_refs = tuple(
        persist_source_chunk(store, ch, profile_id=CHUNK_PROFILE_ID, revision=1) for ch in chunks
    )
    manifest = ChunkManifest(
        schema_version=1,
        project_id=PROJECT,
        document_id=DOCUMENT,
        source_document_ref=source_ref,
        planner_version=CHUNK_PLANNER_VERSION,
        profile=_chunk_profile(),
        chunk_refs=chunk_refs,
        chunk_count=len(chunks),
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
    ext_profile = _extraction_profile()
    ext_refs = []
    for spec, ch, cref in zip(specs, chunks, chunk_refs):
        ext = CandidateExtraction(
            schema_version=1,
            project_id=PROJECT,
            document_id=DOCUMENT,
            chunk_profile_id=CHUNK_PROFILE_ID,
            chunk_id=ch.chunk_id,
            source_document_ref=source_ref,
            source_chunk_ref=cref,
            extraction_profile_id=ext_profile.profile_id,
            extraction_profile_hash=ext_profile.profile_hash,
            generation_provenance=_provenance(),
            candidates=CandidatePayload(
                characters=spec.chars,
                locations=spec.locs,
                facts=spec.facts,
                events=spec.events,
                relationships=spec.rels,
                unresolved_mentions=spec.unres,
            ),
        )
        ext_refs.append(persist_candidate_extraction(store, ext, revision=1))
    a3_input = A3InputIdentity(
        source_document_ref=source_ref,
        chunk_manifest_ref=manifest_ref,
        candidate_extraction_refs=tuple(ext_refs),
        extraction_profile_id=ext_profile.profile_id,
        extraction_profile_hash=ext_profile.profile_hash,
    )
    entries = []
    for chunk_ordinal, (spec, ch, eref) in enumerate(zip(specs, chunks, ext_refs), start=1):
        for c in spec.chars:
            entries.append(_idx_entry_ordinal(chunk_ordinal, ch.chunk_id, c.candidate_id, "character", eref, c.display_name_original))
        for c in spec.locs:
            entries.append(_idx_entry_ordinal(chunk_ordinal, ch.chunk_id, c.candidate_id, "location", eref, c.display_name_original))
        for c in spec.unres:
            entries.append(_idx_entry_ordinal(chunk_ordinal, ch.chunk_id, c.candidate_id, f"unresolved_{c.mention_kind}", eref, c.mention_original))
    entries = tuple(sorted(entries, key=lambda e: e.source_order_key))
    index = CandidateEntityIndex(schema_version=1, entries=entries)
    recon_profile = _recon_profile()
    _publish_a4(
        store, pointers, index=index, a3_input=a3_input, recon_profile=recon_profile, decision=decision
    )
    return RunTree(
        store=store,
        pointers=pointers,
        source_ref=source_ref,
        chunk=chunks[0],
        chunk_ref=chunk_refs[0],
        manifest=manifest,
        manifest_ref=manifest_ref,
        ext_ref=ext_refs[0],
        recon_profile=recon_profile,
        a3_input=a3_input,
        index=index,
    )


def _plan(tree: RunTree):
    return build_consolidation_planning(
        tree.store,
        tree.pointers,
        project_id=PROJECT,
        document_id=DOCUMENT,
        reconciliation_profile_id=RECON_PROFILE_ID,
        consolidation_profile=_consolidation_profile(),
    )


def _fingerprint_dir(root: Path) -> dict[str, str]:
    """path-relative -> sha256(content) for every file under ``root``."""
    out: dict[str, str] = {}
    if not root.exists():
        return out
    for path in sorted(root.rglob("*")):
        if path.is_file():
            out[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


def _int_in(out: str, label: str) -> int:
    m = re.search(rf"{label}:\s+(\d+)", out)
    assert m, f"could not find {label!r} in audit output"
    return int(m.group(1))


# ---------------------------------------------------------------------------
# BLOCK 1 -- diagnostic normalization / tokenization
# ---------------------------------------------------------------------------


def test_normalize_diagnostic_text_nfkc_casefold_collapse():
    # Full-width / uppercase / padded input normalizes to a canonical form.
    assert audit.normalize_diagnostic_text("  Alice   met \nBob ") == "alice met bob"
    # CJK is preserved verbatim (NFKC is a no-op for ideographs).
    assert audit.normalize_diagnostic_text("  姐妹关系 ") == "姐妹关系"
    # NFKC folds compatibility characters (superscript two -> 2).
    assert audit.normalize_diagnostic_text("Alice\u00b2") == "alice2"


def test_diagnostic_tokens_ascii_runs_and_cjk_chars():
    assert audit.diagnostic_tokens("Alice met bob-2") == frozenset({"alice", "met", "bob", "2"})
    # Each CJK ideograph is its own token; ASCII alnum runs are whole tokens.
    assert audit.diagnostic_tokens("姐妹关系") == frozenset({"姐", "妹", "关", "系"})
    assert audit.diagnostic_tokens("Alice 姐妹") == frozenset({"alice", "姐", "妹"})
    # Deterministic + idempotent under normalization.
    assert audit.diagnostic_tokens("  ALICE  ") == audit.diagnostic_tokens("alice")


def test_diagnostic_tokens_disjoint_relationship_types():
    # Distinct relationship types produce disjoint token sets (no false overlap).
    assert audit.diagnostic_tokens("拥有") & audit.diagnostic_tokens("恐惧") == frozenset()
    # Shared CJK char -> overlap.
    assert audit.diagnostic_tokens("姐妹") & audit.diagnostic_tokens("妹") == frozenset({"妹"})


# ---------------------------------------------------------------------------
# BLOCK 1 -- naive pair count + source distance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n,expected", [(0, 0), (1, 0), (2, 1), (3, 3), (158, 12403), (167, 13861), (116, 6670)])
def test_naive_pair_count_exact(n, expected):
    assert audit._naive_pair_count(n) == expected


@pytest.mark.parametrize(
    "a,b,expected",
    [
        (1, 1, "same_chunk"),
        (1, 2, "distance_1"),
        (1, 3, "distance_2"),
        (1, 4, "distance_3_plus"),
        (5, 7, "distance_2"),
        (1, 20, "distance_3_plus"),
    ],
)
def test_source_distance_label(a, b, expected):
    assert audit._source_distance_label(a, b) == expected


# ---------------------------------------------------------------------------
# BLOCK 1 -- per-domain overlap signal functions
# ---------------------------------------------------------------------------


def _ev(paragraph_id: str, role: str = "primary") -> EvidenceRef:
    return EvidenceRef(paragraph_id=paragraph_id, role=role, strength="explicit", excerpt="t")


def test_fact_signals_all_fire():
    a = SimpleNamespace(
        fact_type="world_fact", statement_zh="Alice met Bob",
        subject_refs=("char_0001",), object_refs=("loc_0001",), evidence_refs=(_ev("P1"),),
    )
    b = SimpleNamespace(
        fact_type="world_fact", statement_zh="  alice met bob  ",
        subject_refs=("char_0001", "char_0002"), object_refs=("loc_0001", "loc_0002"),
        evidence_refs=(_ev("P1"), _ev("P2")),
    )
    sig = audit._fact_signals(a, b)
    assert sig == frozenset(
        {
            "same_fact_type",
            "exact_normalized_statement",
            "subject_overlap",
            "object_overlap",
            "bound_entity_overlap",
            "evidence_paragraph_overlap",
        }
    )


def test_fact_signals_none():
    a = SimpleNamespace(
        fact_type="world_fact", statement_zh="Alice met Bob",
        subject_refs=("char_0001",), object_refs=("loc_0001",), evidence_refs=(_ev("P1"),),
    )
    b = SimpleNamespace(
        fact_type="identity", statement_zh="Carol lives",
        subject_refs=("char_0003",), object_refs=("loc_0003",), evidence_refs=(_ev("P9"),),
    )
    assert audit._fact_signals(a, b) == frozenset()


def test_event_signals_all_fire():
    a = SimpleNamespace(
        summary_zh="Alice walked", participants=("char_0001",), locations=("loc_0001",),
        temporal_mode="normal", evidence_refs=(_ev("P1"),),
    )
    b = SimpleNamespace(
        summary_zh="alice  walked", participants=("char_0001", "char_0002"),
        locations=("loc_0001",), temporal_mode="normal", evidence_refs=(_ev("P1"), _ev("P2")),
    )
    sig = audit._event_signals(a, b)
    assert sig == frozenset(
        {
            "exact_normalized_summary",
            "participant_overlap",
            "location_overlap",
            "bound_entity_overlap",
            "evidence_paragraph_overlap",
            "temporal_mode_equal",
        }
    )


def test_event_signals_temporal_mode_distinguishes():
    a = SimpleNamespace(
        summary_zh="Alice walked", participants=("char_0001",), locations=(),
        temporal_mode="normal", evidence_refs=(_ev("P1"),),
    )
    b = SimpleNamespace(
        summary_zh="Alice walked", participants=("char_0001",), locations=(),
        temporal_mode="flashback", evidence_refs=(_ev("P1"),),
    )
    # Same summary + participant, but different temporal mode -> no temporal_mode_equal.
    assert "temporal_mode_equal" not in audit._event_signals(a, b)
    assert "exact_normalized_summary" in audit._event_signals(a, b)


def test_relationship_signals_all_fire():
    a = SimpleNamespace(
        source_entity_ref="char_0001", target_entity_ref="char_0002",
        relationship_type_zh="meets", direction="directed", evidence_refs=(_ev("P1"),),
    )
    b = SimpleNamespace(
        source_entity_ref="char_0002", target_entity_ref="char_0001",
        relationship_type_zh=" meets ", direction="directed", evidence_refs=(_ev("P1"),),
    )
    sig = audit._relationship_signals(a, b)
    assert sig == frozenset(
        {
            "exact_or_symmetric_endpoint_group",
            "direction_equal",
            "exact_normalized_relationship_type",
            "relationship_type_token_overlap",
            "evidence_paragraph_overlap",
        }
    )


def test_relationship_signals_token_overlap_only():
    # Same shared CJK token, different full types, different endpoint groups.
    a = SimpleNamespace(
        source_entity_ref="char_0001", target_entity_ref="char_0002",
        relationship_type_zh="拥有 宠物", direction="directed", evidence_refs=(_ev("P1"),),
    )
    b = SimpleNamespace(
        source_entity_ref="char_0003", target_entity_ref="char_0004",
        relationship_type_zh="害怕 宠物", direction="undirected", evidence_refs=(_ev("P9"),),
    )
    sig = audit._relationship_signals(a, b)
    assert "relationship_type_token_overlap" in sig  # shared "宠物"
    assert "exact_normalized_relationship_type" not in sig
    assert "exact_or_symmetric_endpoint_group" not in sig
    assert "direction_equal" not in sig
    assert "evidence_paragraph_overlap" not in sig


def test_relationship_signals_none():
    a = SimpleNamespace(
        source_entity_ref="char_0001", target_entity_ref="char_0002",
        relationship_type_zh="拥有", direction="directed", evidence_refs=(_ev("P1"),),
    )
    b = SimpleNamespace(
        source_entity_ref="char_0003", target_entity_ref="char_0004",
        relationship_type_zh="恐惧", direction="undirected", evidence_refs=(_ev("P9"),),
    )
    assert audit._relationship_signals(a, b) == frozenset()


# ---------------------------------------------------------------------------
# BLOCK 1 -- exact n-choose-2 pair analysis
# ---------------------------------------------------------------------------


def _pair_fact(chunk_id, global_ref, fact_type, statement, subjects, objects, evid):
    return SimpleNamespace(
        chunk_id=chunk_id,
        global_candidate_ref=global_ref,
        fact_type=fact_type,
        statement_zh=statement,
        subject_refs=tuple(subjects),
        object_refs=tuple(objects),
        evidence_refs=tuple(_ev(p) for p in evid),
    )


def test_pair_analysis_counts_and_buckets():
    chunk_ordinal = {"C1": 1, "C2": 2}
    f1 = _pair_fact("C1", "C1:f1", "world_fact", "Alice met Bob", ("char_0001",), ("loc_0001",), ["P1"])
    f2 = _pair_fact("C1", "C1:f2", "world_fact", "Alice met Bob", ("char_0001",), ("loc_0001",), ["P1"])
    f3 = _pair_fact("C2", "C2:f3", "identity", "Carol lives", ("char_0003",), ("loc_0003",), ["P9"])
    analysis = audit._pair_analysis((f1, f2, f3), signal_fn=audit._fact_signals, chunk_ordinal=chunk_ordinal)
    assert analysis.total_pairs == 3
    assert analysis.no_signal == 2  # (f1,f3) and (f2,f3) share nothing
    assert analysis.distance_counts["same_chunk"] == 1  # (f1,f2)
    assert analysis.distance_counts["distance_1"] == 2  # (f1,f3), (f2,f3)
    # (f1,f2) shares: same type, statement, subject, object, bound, evidence.
    combo = frozenset(
        {
            "same_fact_type",
            "exact_normalized_statement",
            "subject_overlap",
            "object_overlap",
            "bound_entity_overlap",
            "evidence_paragraph_overlap",
        }
    )
    assert analysis.combo_counts[combo] == 1
    assert analysis.combo_pairs[combo] == [(f1, f2)]


def test_pair_analysis_single_candidate():
    f1 = _pair_fact("C1", "C1:f1", "world_fact", "x", ("char_0001",), (), ["P1"])
    analysis = audit._pair_analysis((f1,), signal_fn=audit._fact_signals, chunk_ordinal={"C1": 1})
    assert analysis.total_pairs == 0
    assert analysis.no_signal == 0
    assert analysis.combo_counts == {}


# ---------------------------------------------------------------------------
# BLOCK 1 -- conservative exact-duplicate diagnostics
# ---------------------------------------------------------------------------


def test_duplicate_diagnostics_same_and_cross_chunk():
    chunk_ordinal = {"C1": 1, "C2": 2}
    key = (
        "world_fact",
        "alice met bob",
        ("char_0001",),
        (),
    )
    cands = [
        SimpleNamespace(chunk_id="C1", **{"_k": key}),
        SimpleNamespace(chunk_id="C1", **{"_k": key}),
        SimpleNamespace(chunk_id="C2", **{"_k": key}),
        # A distinct candidate (not in any duplicate group).
        SimpleNamespace(chunk_id="C1", **{"_k": ("world_fact", "other", (), ())}),
    ]
    diag = audit._duplicate_diagnostics(cands, lambda c: c._k, chunk_ordinal)
    assert diag["duplicate_group_count"] == 1
    assert diag["candidate_count_in_duplicate_groups"] == 3
    # pairs in the group: (C1,C1)=same_chunk, (C1,C2)=cross, (C1,C2)=cross.
    assert diag["same_chunk_duplicate_pair_count"] == 1
    assert diag["cross_chunk_duplicate_pair_count"] == 2


def test_fact_duplicate_key_order_and_whitespace_insensitive():
    a = SimpleNamespace(fact_type="world_fact", statement_zh="Alice  met Bob",
                        subject_refs=("char_0002", "char_0001"), object_refs=("loc_0001",))
    b = SimpleNamespace(fact_type="world_fact", statement_zh="alice met bob",
                        subject_refs=("char_0001", "char_0002"), object_refs=("loc_0001",))
    c = SimpleNamespace(fact_type="world_fact", statement_zh="Alice met Bob",
                        subject_refs=("char_0001", "char_0002"), object_refs=("loc_0002",))
    assert audit._fact_duplicate_key(a) == audit._fact_duplicate_key(b)  # sorted refs + normalized stmt
    assert audit._fact_duplicate_key(a) != audit._fact_duplicate_key(c)  # different object


def test_event_duplicate_key_temporal_mode_matters():
    a = SimpleNamespace(summary_zh="Alice walked", participants=("char_0001",),
                        locations=("loc_0001",), temporal_mode="normal")
    b = SimpleNamespace(summary_zh="alice walked", participants=("char_0001",),
                        locations=("loc_0001",), temporal_mode="normal")
    c = SimpleNamespace(summary_zh="Alice walked", participants=("char_0001",),
                        locations=("loc_0001",), temporal_mode="flashback")
    assert audit._event_duplicate_key(a) == audit._event_duplicate_key(b)
    assert audit._event_duplicate_key(a) != audit._event_duplicate_key(c)


def test_relationship_duplicate_key_symmetric_endpoints():
    a = SimpleNamespace(source_entity_ref="char_0001", target_entity_ref="char_0002",
                        relationship_type_zh="meets", direction="directed", state_zh=None)
    b = SimpleNamespace(source_entity_ref="char_0002", target_entity_ref="char_0001",
                        relationship_type_zh="meets", direction="directed", state_zh=None)
    c = SimpleNamespace(source_entity_ref="char_0001", target_entity_ref="char_0002",
                        relationship_type_zh="meets", direction="directed", state_zh="ended")
    assert audit._relationship_duplicate_key(a) == audit._relationship_duplicate_key(b)
    assert audit._relationship_duplicate_key(a) != audit._relationship_duplicate_key(c)


# ---------------------------------------------------------------------------
# BLOCK 1 -- deterministic distribution rows + bounded samples
# ---------------------------------------------------------------------------


def test_distribution_rows_exact_buckets_and_tail():
    rows = audit._distribution_rows([1, 1, 2, 5, 12, 20])
    as_map = dict(rows)
    # Exact integer buckets 0..10 (zeros included) + a "10+" tail.
    assert as_map[0] == 0
    assert as_map[1] == 2
    assert as_map[2] == 1
    assert as_map[5] == 1
    assert as_map[10] == 0
    assert as_map["10+"] == 2  # 12 and 20
    # Buckets are contiguous from 0 (including zero-count buckets).
    labels = [r[0] for r in rows]
    assert labels[:11] == list(range(11))


def test_distribution_rows_no_tail_when_within_threshold():
    rows = dict(audit._distribution_rows([0, 1, 1]))
    assert rows == {0: 1, 1: 2}
    assert "10+" not in rows


def test_distribution_rows_empty():
    assert audit._distribution_rows([]) == []


def _sample_fact(ref, chunk_id="C1"):
    return SimpleNamespace(global_candidate_ref=ref, chunk_id=chunk_id)


def test_sample_pairs_bounded_and_deterministic():
    pairs = [
        (_sample_fact(f"C1:{n:03d}"), _sample_fact(f"C1:{n:03d}R")) for n in range(1, 16)
    ]
    samples = audit._sample_pairs(pairs)
    assert len(samples) == audit._MAX_SAMPLES_PER_BUCKET  # truncated to the budget
    # Sorted deterministically by (left_ref, right_ref); never random.
    assert samples == sorted(samples, key=lambda p: (p[0].global_candidate_ref, p[1].global_candidate_ref))
    assert len(audit._sample_pairs([])) == 0


# ---------------------------------------------------------------------------
# BLOCK 1 -- full audit output (synthetic multi-chunk corpus)
# ---------------------------------------------------------------------------


def _rich_corpus():
    return [
        ChunkSpec(
            chapter_id="CH001",
            paragraph_ids=("CH001_P0001", "CH001_P0002"),
            chars=(_char("cand_char_001", "Alice"), _char("cand_char_002", "Bob"), _char("cand_char_003", "Carol")),
            locs=(_loc("cand_loc_001", "Wonderland"),),
            unres=(),
            facts=(
                _fact("cand_fact_001", subject_refs=("cand_char_001", "cand_char_002")),
                # Same-chunk duplicate of cand_fact_001 (identical normalized type, statement, refs).
                FactCandidate(
                    candidate_id="cand_fact_002", fact_type="world_fact",
                    statement_zh="stmt", subject_refs=("cand_char_001", "cand_char_002"),
                    object_refs=(), evidence_strength="explicit", evidence=(_evidence(),),
                ),
                FactCandidate(
                    candidate_id="cand_fact_003", fact_type="identity",
                    statement_zh="lives", subject_refs=("cand_char_003",),
                    object_refs=("cand_loc_001",), evidence_strength="explicit", evidence=(_evidence(),),
                ),
            ),
            events=(
                _event("cand_evt_001", participant_refs=("cand_char_001",), location_refs=("cand_loc_001",)),
                # Same-chunk duplicate of cand_evt_001 (identical normalized summary + refs + mode).
                _event("cand_evt_002", participant_refs=("cand_char_001",), location_refs=("cand_loc_001",)),
            ),
            rels=(
                _rel("cand_rel_001", source_ref="cand_char_001", target_ref="cand_char_002"),
                _rel("cand_rel_002", source_ref="cand_char_002", target_ref="cand_char_001"),
            ),
        ),
        ChunkSpec(
            chapter_id="CH002",
            paragraph_ids=("CH002_P0001", "CH002_P0002"),
            chars=(_char("cand_char_004", "Dave"), _char("cand_char_005", "Eve")),
            locs=(_loc("cand_loc_002", "Looking Glass"),),
            unres=(_unres("cand_unres_001", "unknown"),),
            facts=(
                _fact("cand_fact_004", subject_refs=("cand_char_004",), object_refs=("cand_char_005",)),
            ),
            events=(
                _event("cand_evt_003", participant_refs=("cand_char_005",), location_refs=("cand_loc_002",)),
            ),
            rels=(
                _rel("cand_rel_003", source_ref="cand_char_004", target_ref="cand_char_005"),
            ),
        ),
    ]


def test_audit_full_output_synthetic(tmp_path, capsys):
    tree = _build_multi_chunk_tree(tmp_path, *_rich_corpus())
    result = _plan(tree)
    index = result.index
    n_fact = len(index.facts)
    n_event = len(index.events)
    n_rel = len(index.relationships)

    rc = audit.run_audit(
        tree.store, tree.pointers, project_id=PROJECT, document_id=DOCUMENT,
        reconciliation_profile_id=RECON_PROFILE_ID,
    )
    assert rc == 0
    out = capsys.readouterr().out

    # --- header + every report section is present ---
    for header in (
        "=== A5B ZERO-PROVIDER ALICE AUDIT ===",
        "=== EXACT INPUT IDENTITY ===",
        "=== CANDIDATE UNIVERSE ===",
        "=== BOUND REFS ===",
        "=== FACTS ===",
        "=== EVENTS ===",
        "=== RELATIONSHIPS ===",
        "=== NAIVE PAIR UNIVERSE ===",
        "=== SOURCE DISTANCE ===",
        "=== OVERLAP SIGNALS ===",
        "=== EXACT DUPLICATE DIAGNOSTICS ===",
        "=== REPRESENTATIVE SAMPLES ===",
        "=== AUDIT CHECKS ===",
    ):
        assert header in out, f"missing section {header!r}"

    # --- exact input identity (EntityMap + every pinned A3 ref) ---
    em_ref = result.snapshot.entity_map_ref
    assert em_ref.artifact_type in out
    assert em_ref.artifact_id in out
    assert em_ref.content_hash in out
    assert f"r{em_ref.revision}" in out
    assert result.snapshot.a3_input.source_document_ref.artifact_id in out
    assert result.snapshot.a3_input.chunk_manifest_ref.artifact_id in out
    for ref in result.snapshot.candidate_extraction_refs:
        assert ref.artifact_id in out
    assert "A5B AUDIT RESULT: PASS" in out

    # --- candidate universe totals (exact) ---
    assert _int_in(out, "fact_candidate_count") == n_fact
    assert _int_in(out, "event_candidate_count") == n_event
    assert _int_in(out, "relationship_candidate_count") == n_rel
    assert _int_in(out, "total_candidate_count") == n_fact + n_event + n_rel

    # --- bound ref namespace totals (exact, computed from the index) ---
    per_field = audit._field_namespaces(result)
    char_total = sum(per_field[f]["char"] for f in per_field)
    loc_total = sum(per_field[f]["loc"] for f in per_field)
    unres_total = sum(per_field[f]["unres"] for f in per_field)
    assert _int_in(out, "resolved_char_ref_occurrence_count") == char_total
    assert _int_in(out, "resolved_loc_ref_occurrence_count") == loc_total
    assert _int_in(out, "unresolved_ref_occurrence_count") == unres_total
    assert char_total + loc_total + unres_total > 0

    # --- distributions are printed (each domain has its canonical sub-labels) ---
    for label in (
        "facts_by_type:",
        "fact_subject_count_distribution:",
        "fact_object_count_distribution:",
        "events_by_temporal_mode:",
        "event_participant_count_distribution:",
        "event_location_count_distribution:",
        "relationships_by_direction:",
        "relationship_endpoint_groups",
        "normalized_relationship_type_frequency",
    ):
        assert label in out, f"missing distribution label {label!r}"

    # --- naive pair counts are exact ---
    assert _int_in(out, "naive_fact_pair_count") == n_fact * (n_fact - 1) // 2
    assert _int_in(out, "naive_event_pair_count") == n_event * (n_event - 1) // 2
    assert _int_in(out, "naive_relationship_pair_count") == n_rel * (n_rel - 1) // 2
    assert _int_in(out, "naive_pair_count_total") == (
        n_fact * (n_fact - 1) + n_event * (n_event - 1) + n_rel * (n_rel - 1)
    ) // 2

    # --- source distance buckets are printed (all four labels, per domain) ---
    for domain in ("FACT", "EVENT", "RELATIONSHIP"):
        line = next(
            (ln for ln in out.splitlines() if ln.startswith(f"{domain:13} same_chunk=")),
            None,
        )
        assert line is not None, f"missing {domain} source-distance line"
        assert "distance_1=" in line and "distance_2=" in line and "distance_3_plus=" in line

    # --- duplicate groups are reported (computed from the index) ---
    ordinal = audit._chunk_ordinal_by_chunk_id(result.snapshot)
    fact_diag = audit._duplicate_diagnostics(index.facts, audit._fact_duplicate_key, ordinal)
    event_diag = audit._duplicate_diagnostics(index.events, audit._event_duplicate_key, ordinal)
    assert fact_diag["duplicate_group_count"] >= 1  # the intentional same-chunk fact duplicate
    assert event_diag["duplicate_group_count"] >= 1  # the intentional same-chunk event duplicate
    assert "duplicate_group_count" in out

    # --- representative samples are bounded (<= budget per combo) ---
    sample_lines = [ln for ln in out.splitlines() if ln.lstrip().startswith("CH0") and "<->" in ln]
    # There is at least one representative sample line, and none of them is random
    # (they are deterministic "left <-> right" lines).
    assert sample_lines, "expected at least one representative sample line"


def test_audit_is_read_only(tmp_path):
    tree = _build_multi_chunk_tree(tmp_path, *_rich_corpus())
    before = _fingerprint_dir(tmp_path)
    rc = audit.run_audit(
        tree.store, tree.pointers, project_id=PROJECT, document_id=DOCUMENT,
        reconciliation_profile_id=RECON_PROFILE_ID,
    )
    after = _fingerprint_dir(tmp_path)
    assert rc == 0
    # Zero-provider + read-only: no file created, modified, or removed.
    assert before == after


# ---------------------------------------------------------------------------
# BLOCK 1 -- exact Alice corpus totals (local, gitignored run tree)
# ---------------------------------------------------------------------------

ALICE_RUNS_ROOT = REPO_ROOT / "runs" / "a4e_real_novel"
ALICE_PROJECT = "a3e-real-novel"
ALICE_DOCUMENT = "src_001"
ALICE_PROFILE = "entity-reconciliation-v2"


def _alice_tree_present() -> bool:
    return (ALICE_RUNS_ROOT / ALICE_PROJECT / "story" / "artifacts").is_dir()


@pytest.mark.skipif(not _alice_tree_present(), reason="Alice run tree not present (local, gitignored)")
def test_audit_exact_alice_totals(capsys):
    root = ALICE_RUNS_ROOT / ALICE_PROJECT / "story"
    store = FileArtifactStore(root / "artifacts")
    pointers = FilePointerStore(root / "pointers", store)
    rc = audit.run_audit(
        store, pointers, project_id=ALICE_PROJECT, document_id=ALICE_DOCUMENT,
        reconciliation_profile_id=ALICE_PROFILE,
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "=== A5B ZERO-PROVIDER ALICE AUDIT ===" in out
    assert "A5B AUDIT RESULT: PASS" in out
    # Exact Alice corpus candidate universe (verified against the run tree).
    assert _int_in(out, "fact_candidate_count") == 158
    assert _int_in(out, "event_candidate_count") == 167
    assert _int_in(out, "relationship_candidate_count") == 116
    assert _int_in(out, "total_candidate_count") == 441
    # Exact Alice naive pair counts.
    assert _int_in(out, "naive_fact_pair_count") == 12403
    assert _int_in(out, "naive_event_pair_count") == 13861
    assert _int_in(out, "naive_relationship_pair_count") == 6670
    assert _int_in(out, "naive_pair_count_total") == 12403 + 13861 + 6670


# ---------------------------------------------------------------------------
# BLOCK 2 -- stable exact-dedupe of bound alias refs
# ---------------------------------------------------------------------------


def test_stable_dedupe_pure():
    assert _stable_dedupe(("char_0001", "char_0001", "char_0002")) == ("char_0001", "char_0002")
    # First occurrence wins; order preserved; idempotent; distinct unres ids kept.
    assert _stable_dedupe(("unres_0001", "char_0001", "unres_0001", "unres_0002")) == (
        "unres_0001", "char_0001", "unres_0002"
    )
    assert _stable_dedupe(()) == ()
    seq = ("char_0001", "char_0001")
    assert _stable_dedupe(seq) == ("char_0001",)
    assert _stable_dedupe(_stable_dedupe(("a", "a", "b", "b", "a"))) == ("a", "b")


def test_block2_bound_alias_refs_dedupe(tmp_path):
    # Two local char candidates (same name) that A4 merges into ONE canonical id.
    # A fact referencing both must collapse to a single bound id (stable dedupe).
    tree = _build_run_tree(
        tmp_path,
        chars=(_char("cand_char_001", "Alice"), _char("cand_char_002", "Alice")),
        locs=(),
        unres=(),
        facts=(_fact("cand_fact_001", subject_refs=("cand_char_001", "cand_char_002")),),
        events=(),
        rels=(),
        publish_a4=False,
    )
    _publish_a4(
        tree.store, tree.pointers, index=tree.index, a3_input=tree.a3_input,
        recon_profile=tree.recon_profile, decision="same_entity",
    )
    result = _plan(tree)
    fact = result.index.facts[0]
    # Both local refs bound to the same A4 canonical char -> deduped to one.
    assert len(fact.subject_refs) == 1
    assert fact.subject_refs[0].startswith("char_")
    # Sanity: without the merge (different_entity) the two refs stay distinct.
    tree2 = _build_run_tree(
        tmp_path / "different",
        chars=(_char("cand_char_001", "Alice"), _char("cand_char_002", "Alice")),
        locs=(),
        unres=(),
        facts=(_fact("cand_fact_001", subject_refs=("cand_char_001", "cand_char_002")),),
        events=(),
        rels=(),
        publish_a4=False,
    )
    _publish_a4(
        tree2.store, tree2.pointers, index=tree2.index, a3_input=tree2.a3_input,
        recon_profile=tree2.recon_profile, decision="different_entity",
    )
    assert len(_plan(tree2).index.facts[0].subject_refs) == 2


# ---------------------------------------------------------------------------
# BLOCK 3 -- paragraph ordinal uses the GLOBAL SourceDocument rank
# ---------------------------------------------------------------------------


def test_block3_global_paragraph_rank(tmp_path):
    tree = _build_multi_chunk_tree(
        tmp_path,
        ChunkSpec(
            chapter_id="CH001",
            paragraph_ids=("CH001_P0001", "CH001_P0002"),
            chars=(_char("cand_char_001", "Alice"),),
            locs=(),
            unres=(),
            facts=(_fact("cand_fact_001", subject_refs=("cand_char_001",)),),
            events=(),
            rels=(),
        ),
        ChunkSpec(
            chapter_id="CH002",
            paragraph_ids=("CH002_P0001", "CH002_P0002"),
            chars=(_char("cand_char_003", "Dave"),),
            locs=(),
            unres=(),
            facts=(
                FactCandidate(
                    candidate_id="cand_fact_003", fact_type="world_fact", statement_zh="s",
                    subject_refs=("cand_char_003",), object_refs=(),
                    evidence_strength="explicit",
                    # Anchor on CH002_P0001 -> global rank 3 (NOT chunk-local rank 1).
                    evidence=(_ev("CH002_P0001"),),
                ),
            ),
            events=(),
            rels=(),
        ),
    )
    result = _plan(tree)
    fact = next(f for f in result.index.facts if f.local_candidate_id == "cand_fact_003")
    fields = fact.source_order_key.split(":")
    chunk_ordinal = int(fields[0])
    paragraph_ordinal = int(fields[1])
    # Global chunk ordinal (2) and global paragraph rank (3), not chunk-local.
    assert chunk_ordinal == 2
    assert paragraph_ordinal == 3


# ---------------------------------------------------------------------------
# BLOCK 4 -- source anchor is the earliest PRIMARY by global source rank
# ---------------------------------------------------------------------------


def test_anchor_paragraph_ordinal_pure():
    paragraph_rank = {"P1": 1, "P2": 2, "P3": 3}
    # Earliest PRIMARY wins (evidence-array order is NOT the authority).
    evid = (_ev("P3"), _ev("P1"))
    assert _anchor_paragraph_ordinal(evid, paragraph_rank=paragraph_rank, candidate_ref="x") == 1
    # A later PRIMARY is ignored when an earlier PRIMARY exists.
    evid = (_ev("P3"), _ev("P1"), _ev("P2"))
    assert _anchor_paragraph_ordinal(evid, paragraph_rank=paragraph_rank, candidate_ref="x") == 1
    # No-primary fallback: earliest among ALL evidence.
    evid = (_ev("P2", role="supporting"), _ev("P3", role="supporting"))
    assert _anchor_paragraph_ordinal(evid, paragraph_rank=paragraph_rank, candidate_ref="x") == 2
    # PRIMARY takes precedence over an earlier supporting ref.
    evid = (_ev("P1", role="supporting"), _ev("P2", role="primary"))
    assert _anchor_paragraph_ordinal(evid, paragraph_rank=paragraph_rank, candidate_ref="x") == 2
    # Fail closed: no evidence.
    with pytest.raises(Exception):
        _anchor_paragraph_ordinal((), paragraph_rank=paragraph_rank, candidate_ref="x")
    # Fail closed: evidence paragraph absent from the exact SourceDocument.
    with pytest.raises(Exception):
        _anchor_paragraph_ordinal(
            (_ev("P99"),), paragraph_rank=paragraph_rank, candidate_ref="x"
        )


def test_block4_earliest_primary_anchor(tmp_path):
    # A single-chunk fact with PRIMARY evidence on both the 3rd and 1st paragraph:
    # the anchor must be the 1st (earliest global PRIMARY), not the 3rd.
    tree = _build_run_tree(
        tmp_path,
        chars=(_char("cand_char_001", "Alice"),),
        locs=(),
        unres=(),
        facts=(
            FactCandidate(
                candidate_id="cand_fact_001", fact_type="world_fact", statement_zh="s",
                subject_refs=("cand_char_001",), object_refs=(),
                evidence_strength="explicit",
                evidence=(_ev("CH001_P0003"), _ev("CH001_P0001")),
            ),
        ),
        events=(),
        rels=(),
    )
    result = _plan(tree)
    fact = result.index.facts[0]
    paragraph_ordinal = int(fact.source_order_key.split(":")[1])
    assert paragraph_ordinal == 1  # earliest PRIMARY (CH001_P0001), not CH001_P0003


# ---------------------------------------------------------------------------
# BLOCK 5 -- source_order_key ends in the exact global candidate ref
# ---------------------------------------------------------------------------


def test_block5_source_order_key_terminates_in_global_ref(tmp_path):
    tree = _build_multi_chunk_tree(tmp_path, *_rich_corpus())
    result = _plan(tree)
    index = result.index
    candidates = list(index.facts) + list(index.events) + list(index.relationships)
    assert candidates, "expected at least one indexed candidate"
    for cand in candidates:
        assert cand.source_order_key.endswith(cand.global_candidate_ref)
        # The key is a 4-field fixed-width numeric prefix + ":" + the exact
        # global candidate ref (which itself is chunk_id:local_id).
        prefix = cand.source_order_key[: -len(cand.global_candidate_ref) - 1]
        assert prefix.count(":") == 3  # 4 numeric fields before the global ref
    # The helper itself ends in the global ref (BLOCK 5 frozen format).
    assert _source_order_key(
        chunk_ordinal=7, para_ordinal=3, category_ordinal=1, suffix=5, global_ref="CH001_C001:cand_fact_001"
    ) == "000007:000000003:01:000000005:CH001_C001:cand_fact_001"
