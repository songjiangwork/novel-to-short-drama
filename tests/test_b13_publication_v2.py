"""In-memory B4A v2 only; all authority and provenance are synthetic."""
from copy import deepcopy
import hashlib
from types import SimpleNamespace

import pytest

from short_drama.adaptation.disposition_checkpoint_v2 import encode_feedback_checkpoint_v2, verify_and_decode_feedback_checkpoint_v2
from short_drama.adaptation.disposition_execution import B13DispositionExecutionResult
from short_drama.adaptation.disposition_feedback import build_feedback_request
from short_drama.adaptation.disposition_publication import (
    B13DispositionPublicationError, B13_DISPOSITION_PUBLICATION_V2_MAX_BYTES,
    _accepted_to_dict, _publication_identity, prepare_b13_disposition_publication,
    verify_b13_disposition_publication,
)
from short_drama.adaptation.disposition_assembly import assemble_b13_disposition_plan
from short_drama.adaptation.disposition_reconciliation import (
    B13ReconciliationDecisionSet, build_b13_reconciliation_inventory, review_b13_reconciliation,
)
from short_drama.artifacts import content_hash, strict_json_loads
from short_drama.story.story_analysis import StoryAnalysisManifest
from test_b13_disposition_checkpoint import _accepted, _fixture, _provenance, _session
from test_b13_disposition_publication_persistence import _real_two_window_preparation
from test_b13_disposition_publication import _publication
from test_adaptation_strategy import _b12_test_runtime_config
from test_b13_window_semantic import _proposal


def test_v1_synthetic_publication_matches_base_bytes_and_identities(tmp_path):
    publication, *_ = _publication(tmp_path)
    assert hashlib.sha256(publication.canonical_bytes()).hexdigest() == 'ee3206d2ebb65a2c5622eb9be34918b83de09310d2f2d5ad89762caeffa1f28c'
    assert publication.semantic_identity == 'c0c3c8e1e90b6480cd08579323e7316cc7c669f149864d2eb4b5c88afdb70a81'
    assert publication.publication_identity == '9351db8bb1d9de84b906f427bc8fde42e8905922b6eb79708433586d6421acf4'


@pytest.fixture(scope='module')
def case(tmp_path_factory):
    tree, session, packet, semantic = _fixture(tmp_path_factory.mktemp('b4a-v2'))
    source = session.preparation.source_snapshot
    manifest = StoryAnalysisManifest.from_dict(tree.store.get_ref(source.story_analysis_manifest_ref).payload)
    refs = {'a5_manifest': source.consolidation_manifest_ref, 'a6_manifest': source.story_analysis_manifest_ref,
            'entity_map': source.entity_map_ref, 'characters': source.canonical_character_registry_ref,
            'locations': source.canonical_location_registry_ref, 'facts': source.canonical_fact_set_ref,
            'events': source.canonical_event_set_ref, 'relationships': source.canonical_relationship_set_ref,
            'arc_analysis': source.arc_analysis_ref, 'event_analysis': manifest.global_event_analysis_ref}
    authority = {name: tree.store.get_ref(ref).to_dict() for name, ref in refs.items()}
    bad = _proposal(packet)
    bad['adaptation_mappings'] = []
    feedback = build_feedback_request(semantic, bad)
    second = SimpleNamespace(request=feedback.request, semantic_profile=semantic.semantic_profile,
                             output_schema=semantic.output_schema)
    raw = encode_feedback_checkpoint_v2(session=session, window_id=packet.window_id, rejected_candidate=bad,
        first_provenance=_provenance(semantic, session.runtime_config), first_attempts=1,
        accepted_proposal=_proposal(packet), second_provenance=_provenance(second, session.runtime_config),
        second_attempts=1)
    accepted = verify_and_decode_feedback_checkpoint_v2(raw, session)
    prep = session.preparation
    execution = B13DispositionExecutionResult(prep.source_snapshot, prep.strategy_selection,
                                              prep.preparation_identity, (accepted,))
    inventory = build_b13_reconciliation_inventory(execution)
    review = review_b13_reconciliation(inventory, B13ReconciliationDecisionSet(inventory.inventory_identity, ()))
    assembly = assemble_b13_disposition_plan(review)
    publication = prepare_b13_disposition_publication(review, assembly, execution, historical_authority=authority)
    return tree, session, packet, execution, review, assembly, publication, authority


def test_v2_self_contained_without_runtime_session_or_store(case, monkeypatch):
    _tree, _session, _packet, execution, _review, assembly, publication, _authority = case
    import short_drama.adaptation.disposition_checkpoint_v2 as codec
    import short_drama.adaptation.disposition_semantic as semantic_module
    def forbidden(*_a, **_kw): pytest.fail('live runtime or session used')
    monkeypatch.setattr(codec, '_trusted_session', forbidden)
    monkeypatch.setattr(semantic_module, '_mapped_provider_request_bytes', forbidden)
    assert publication.to_dict()['schema_version'] == 2
    assert verify_b13_disposition_publication(publication) == assembly
    assert 'feedback_evidence' in publication.to_dict()['evidence']['accepted_windows'][0]
    with pytest.raises(B13DispositionPublicationError):
        _accepted_to_dict(execution.accepted_windows[0])


def test_tampered_v2_evidence_is_rejected_without_leaking(case):
    _tree, _session, _packet, _execution, _review, _assembly, publication, _authority = case
    def candidate(p): p['evidence']['accepted_windows'][0]['feedback_evidence']['round1']['candidate']['event_dispositions'][0]['rationale_zh'] = 'SECRET SOURCE'
    def diagnostic(p): p['evidence']['accepted_windows'][0]['feedback_evidence']['round1']['diagnostic']['events'] = []
    def second(p): p['evidence']['accepted_windows'][0]['feedback_evidence']['round2']['provenance']['request_hash'] = '0' * 64
    def template(p): p['evidence']['accepted_windows'][0]['feedback_evidence']['round2']['prompt_content_hash'] = '0' * 64
    def checkpoint_id(p): p['evidence']['accepted_windows'][0]['feedback_evidence']['checkpoint_identity'] = '0' * 64
    def source(p): p['evidence']['source_snapshot']['canonical_event_set_ref']['content_hash'] = '0' * 64
    def no_carrier(p): del p['evidence']['accepted_windows'][0]['feedback_evidence']
    def final(p): p['evidence']['accepted_windows'][0]['proposal']['event_dispositions'][0]['disposition'] = 'omit'
    def authority(p): p['evidence']['historical_authority']['events']['payload']['events'][0]['summary_zh'] = 'SECRET SOURCE'
    def prep(p): p['evidence']['preparation_identity'] = '0' * 64
    def packet(p): p['evidence']['accepted_windows'][0]['packet']['window']['window_id'] = 'window_9999'
    def strategy(p): p['evidence']['strategy_selection']['strategy_ref']['content_hash'] = '0' * 64
    def reconciliation(p): p['evidence']['reconciliation']['inventory']['preparation_identity'] = '0' * 64
    def lineage(p): p['lineage'] = []
    def plan(p): p['evidence']['assembly']['preparation_identity'] = '0' * 64
    def schema(p): p['schema_version'] = 99
    for mutate in (candidate, diagnostic, second, template, checkpoint_id, source, no_carrier, final, authority, prep, packet, strategy, reconciliation, lineage, plan, schema):
        data = deepcopy(publication.to_dict())
        mutate(data)
        data['publication_identity'] = _publication_identity(data)
        with pytest.raises(B13DispositionPublicationError) as err:
            verify_b13_disposition_publication(data)
        assert 'SECRET SOURCE' not in str(err.value) and 'evt_' not in str(err.value)


def test_v2_size_and_downgrade_fail_closed(case):
    _tree, _session, _packet, _execution, _review, _assembly, publication, _authority = case
    for size in (262_145, 1_000_001):
        data = publication.to_dict()
        data['evidence']['accepted_windows'][0]['feedback_evidence']['round1']['candidate']['event_dispositions'][0]['rationale_zh'] = 'x' * size
        data['publication_identity'] = _publication_identity(data)
        with pytest.raises(B13DispositionPublicationError):
            verify_b13_disposition_publication(data)
    data = publication.to_dict()
    data['evidence']['historical_authority']['events']['payload']['oversize'] = 'x' * B13_DISPOSITION_PUBLICATION_V2_MAX_BYTES
    with pytest.raises(B13DispositionPublicationError): verify_b13_disposition_publication(data)


def test_mixed_windows_remain_bound_to_their_own_history(tmp_path, monkeypatch):
    tree, prep, requests = _real_two_window_preparation(tmp_path, monkeypatch)
    session = _session(prep, requests, _b12_test_runtime_config())
    source = prep.source_snapshot
    manifest = StoryAnalysisManifest.from_dict(tree.store.get_ref(source.story_analysis_manifest_ref).payload)
    refs = {'a5_manifest': source.consolidation_manifest_ref, 'a6_manifest': source.story_analysis_manifest_ref,
            'entity_map': source.entity_map_ref, 'characters': source.canonical_character_registry_ref,
            'locations': source.canonical_location_registry_ref, 'facts': source.canonical_fact_set_ref,
            'events': source.canonical_event_set_ref, 'relationships': source.canonical_relationship_set_ref,
            'arc_analysis': source.arc_analysis_ref, 'event_analysis': manifest.global_event_analysis_ref}
    authority = {name: tree.store.get_ref(ref).to_dict() for name, ref in refs.items()}
    packet, semantic = prep.packets[1], requests[1]
    bad = _proposal(packet)
    bad['adaptation_mappings'] = []
    feedback = build_feedback_request(semantic, bad)
    second = SimpleNamespace(request=feedback.request, semantic_profile=semantic.semantic_profile,
                             output_schema=semantic.output_schema)
    raw = encode_feedback_checkpoint_v2(session=session, window_id=packet.window_id, rejected_candidate=bad,
        first_provenance=_provenance(semantic, session.runtime_config), first_attempts=1,
        accepted_proposal=_proposal(packet), second_provenance=_provenance(second, session.runtime_config),
        second_attempts=1)
    accepted = (_accepted(prep.packets[0], requests[0], session.runtime_config),
                verify_and_decode_feedback_checkpoint_v2(raw, session))
    execution = B13DispositionExecutionResult(source, prep.strategy_selection, prep.preparation_identity, accepted)
    inventory = build_b13_reconciliation_inventory(execution)
    review = review_b13_reconciliation(inventory, B13ReconciliationDecisionSet(inventory.inventory_identity, ()))
    assembly = assemble_b13_disposition_plan(review)
    publication = prepare_b13_disposition_publication(review, assembly, execution, historical_authority=authority)
    assert verify_b13_disposition_publication(publication) == assembly
    data = publication.to_dict()
    windows = data['evidence']['accepted_windows']
    assert 'feedback_evidence' not in windows[0] and 'feedback_evidence' in windows[1]
    windows[0]['feedback_evidence'] = windows[1].pop('feedback_evidence')
    data['publication_identity'] = _publication_identity(data)
    with pytest.raises(B13DispositionPublicationError):
        verify_b13_disposition_publication(data)


def test_b4b_rejects_v2_before_request_or_storage_access(case):
    from short_drama.adaptation.disposition_persistence import (B13DispositionPersistenceService,
        B13DispositionPersistenceError, _require_v1_publication_shape)
    publication = case[6]
    service = object.__new__(B13DispositionPersistenceService)
    with pytest.raises(B13DispositionPersistenceError, match='only v1'):
        service.publish_validated(publication=publication, preparation=None, semantic_preparations=())
    data = publication.to_dict()
    data['schema_version'] = 1
    del data['evidence']['accepted_windows'][0]['feedback_evidence']
    data['publication_identity'] = _publication_identity(data)
    with pytest.raises(B13DispositionPersistenceError):
        _require_v1_publication_shape(data)
    with pytest.raises(B13DispositionPublicationError): verify_b13_disposition_publication(data)
