"""Unratified v2 offline codec; only synthetic packets and fake provenance."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest

from short_drama.adaptation.disposition_checkpoint import B13CheckpointEvidenceError, encode_checkpoint, verify_and_decode_checkpoint
from short_drama.adaptation.disposition_checkpoint_inspection_v2 import B13OfflineCheckpointConflict, inspect_checkpoint_versions_offline
from short_drama.adaptation.disposition_checkpoint_v2 import (
    B13FeedbackCheckpointError, FeedbackEvidenceLimits, encode_feedback_checkpoint_v2,
    encode_verified_feedback_checkpoint_v2, verify_and_decode_feedback_checkpoint_v2,
)
from short_drama.adaptation.disposition_execution import B13WindowExecutionError
from short_drama.adaptation.disposition_execution import _MissingMappingDetail
from short_drama.adaptation import B13DispositionWindowPacket, build_b13_window_semantic_preparation
from short_drama.adaptation.disposition_feedback import B13FeedbackError, build_feedback_request
import short_drama.adaptation.disposition_feedback as feedback_module
from short_drama.artifacts import canonical_json_bytes, content_hash, strict_json_loads
from short_drama.llm import GenerationExecutionOptions
from test_b13_disposition_checkpoint import _accepted, _fixture, _provenance
from test_b13_disposition_execution import _two_window_inputs
from test_b13_window_semantic import _proposal


@pytest.fixture(scope='module')
def case(tmp_path_factory):
    _tree, session, packet, semantic = _fixture(tmp_path_factory.mktemp('synthetic-feedback'))
    candidate = _proposal(packet)
    candidate['adaptation_mappings'] = []
    feedback = build_feedback_request(semantic, candidate)
    feedback_semantic = SimpleNamespace(request=feedback.request, semantic_profile=semantic.semantic_profile,
                                        output_schema=semantic.output_schema)
    first = _provenance(semantic, session.runtime_config)
    second = _provenance(feedback_semantic, session.runtime_config)
    kwargs = dict(session=session, window_id=packet.window_id, rejected_candidate=candidate,
                  first_provenance=first, first_attempts=1, accepted_proposal=_proposal(packet),
                  second_provenance=second, second_attempts=2)
    raw = encode_feedback_checkpoint_v2(**kwargs)
    return session, packet, semantic, kwargs, raw


def _rewrap(value):
    value['checkpoint_identity'] = content_hash({k: v for k, v in value.items() if k != 'checkpoint_identity'})
    return canonical_json_bytes(value)


def _tamper_candidate(payload):
    payload['round1']['candidate']['event_dispositions'][0]['rationale_zh'] = 'SECRET SOURCE'
    payload['round1']['candidate_hash'] = content_hash(payload['round1']['candidate'])


def _tamper_final(payload):
    payload['proposal']['event_dispositions'][0]['disposition'] = 'omit'
    payload['proposal_hash'] = content_hash(payload['proposal'])


def test_offline_round_trip_has_independent_requests_and_bound_carrier(case):
    session, packet, semantic, kwargs, raw = case
    accepted = verify_and_decode_feedback_checkpoint_v2(raw, session)
    data = strict_json_loads(raw)
    assert data['round1']['request_hash'] != data['round2']['request_hash']
    assert data['round1']['rendered_prompt_hash'] != data['round2']['rendered_prompt_hash']
    assert accepted.feedback_evidence.canonical_bytes() == raw
    assert encode_verified_feedback_checkpoint_v2(accepted, session) == raw
    assert accepted.request_identity == semantic.request_identity
    assert accepted.invocation_provenance.request_hash == data['round2']['request_hash']
    assert accepted.semantic_rounds == 2 and accepted.provider_attempts == 3
    assert encode_feedback_checkpoint_v2(**kwargs) == raw
    with pytest.raises(B13CheckpointEvidenceError):
        encode_checkpoint(accepted, session)
    with pytest.raises(B13WindowExecutionError):
        replace(accepted, feedback_evidence=None)
    with pytest.raises(B13WindowExecutionError):
        replace(accepted, proposal=_accepted(packet, semantic, session.runtime_config).proposal,
                request_identity='0' * 64)


def test_rebuild_preserves_bound_execution_options(case):
    session, packet, original, kwargs, _raw = case
    options = GenerationExecutionOptions(prompt_context_reuse='disabled')
    semantic = build_b13_window_semantic_preparation(
        packet, runtime_config=session.runtime_config, execution_options=options,
    )
    assert semantic.request_identity == original.request_identity
    assert semantic.provider_request_bytes != original.provider_request_bytes
    assert semantic.request.request_hash == original.request.request_hash
    session = replace(session, semantic_preparations=(semantic,))
    feedback = build_feedback_request(semantic, kwargs['rejected_candidate'])
    feedback_semantic = SimpleNamespace(
        request=feedback.request, semantic_profile=semantic.semantic_profile,
        output_schema=semantic.output_schema,
    )
    updated = {**kwargs, 'session': session,
               'first_provenance': _provenance(semantic, session.runtime_config),
               'second_provenance': _provenance(feedback_semantic, session.runtime_config)}
    raw = encode_feedback_checkpoint_v2(**updated)
    accepted = verify_and_decode_feedback_checkpoint_v2(raw, session)
    assert accepted.request_identity == semantic.request_identity
    assert encode_verified_feedback_checkpoint_v2(accepted, session) == raw


def test_tampering_and_limits_fail_closed_without_content_leak(case):
    session, _packet, semantic, kwargs, raw = case
    mutations = [
        _tamper_candidate,
        lambda p: p['round1']['diagnostic']['events'][0].update(mapping_state='ONLY_INACTIVE_EVENT_MAPPINGS'),
        lambda p: p['round1']['diagnostic'].update(events=[]),
        lambda p: p['round1']['diagnostic'].update(truncated=True),
        lambda p: p['round2'].update(prompt_content_hash='0' * 64),
        lambda p: p['round2']['provenance'].update(request_hash='0' * 64),
        lambda p: p['round1']['provenance'].update(request_hash='0' * 64),
        lambda p: p['window'].update(window_ordinal=True),
        _tamper_final,
        lambda p: p.update(unexpected='SECRET SOURCE'),
    ]
    for change in mutations:
        payload = deepcopy(strict_json_loads(raw))
        change(payload)
        with pytest.raises(B13FeedbackCheckpointError) as exc:
            verify_and_decode_feedback_checkpoint_v2(_rewrap(payload), session)
        assert 'SECRET SOURCE' not in str(exc.value) and 'evt_' not in str(exc.value)
    with pytest.raises(B13FeedbackCheckpointError):
        verify_and_decode_feedback_checkpoint_v2(raw, session, limits=FeedbackEvidenceLimits(evidence_bytes=len(raw)-1))
    with pytest.raises(B13FeedbackCheckpointError):
        encode_feedback_checkpoint_v2(**kwargs, limits=FeedbackEvidenceLimits(candidate_bytes=1))
    with pytest.raises(B13FeedbackError) as exc:
        build_feedback_request(semantic, {'secret': 'SECRET SOURCE evt_999999'})
    assert 'SECRET' not in str(exc.value) and 'evt_' not in str(exc.value)
    with pytest.raises(B13FeedbackError):
        build_feedback_request(semantic, kwargs['accepted_proposal'])


def test_inactive_mapping_and_truncated_diagnostics_fail_closed(case, monkeypatch):
    _session, _packet, semantic, kwargs, _raw = case
    inactive = deepcopy(kwargs['accepted_proposal'])
    inactive['adaptation_mappings'][0]['active'] = False
    diagnostic = strict_json_loads(build_feedback_request(semantic, inactive).diagnostic_bytes)
    assert [event['mapping_state'] for event in diagnostic['events']] == ['ONLY_INACTIVE_EVENT_MAPPINGS']
    monkeypatch.setattr(feedback_module, '_missing_mapping_detail',
                        lambda *_: _MissingMappingDetail(33, ((1, 'NO_EVENT_MAPPING'),), True))
    with pytest.raises(B13FeedbackError):
        build_feedback_request(semantic, kwargs['rejected_candidate'])


def test_feedback_asset_hash_and_two_round_attempt_caps_are_pinned(case, monkeypatch):
    _session, _packet, semantic, kwargs, _raw = case
    monkeypatch.setattr(feedback_module, 'FEEDBACK_PROMPT_HASH', '0' * 64)
    with pytest.raises(B13FeedbackError):
        build_feedback_request(semantic, kwargs['rejected_candidate'])
    monkeypatch.undo()
    with pytest.raises(B13FeedbackCheckpointError):
        encode_feedback_checkpoint_v2(**{**kwargs, 'first_attempts': 4})


def test_feedback_lists_all_missing_owned_events_in_packet_order(tmp_path):
    _tree, _preparation, _requests, _budgets, runtime, first, _second = _two_window_inputs(tmp_path)
    view = first.to_dict()
    view['window']['owned_event_refs'] = ['evt_000001', 'evt_000002']
    view['owned_source_events'] = [
        {'event_id': 'evt_000001', 'narrative_order': 1},
        {'event_id': 'evt_000002', 'narrative_order': 2},
    ]
    packet = B13DispositionWindowPacket(first.source_snapshot, first.strategy_selection,
                                        first.window_id, first.window_ordinal, canonical_json_bytes(view))
    semantic = build_b13_window_semantic_preparation(packet, runtime_config=runtime)
    candidate = _proposal(packet)
    candidate['adaptation_mappings'].pop(0)
    candidate['adaptation_mappings'][0]['active'] = False
    diagnostic = strict_json_loads(build_feedback_request(semantic, candidate).diagnostic_bytes)
    assert [(event['owned_ordinal'], event['mapping_state']) for event in diagnostic['events']] == [
        (1, 'NO_EVENT_MAPPING'), (2, 'ONLY_INACTIVE_EVENT_MAPPINGS')]


def test_offline_version_conflicts_and_v1_bytes(case, tmp_path):
    session, packet, semantic, _kwargs, raw = case
    accepted = _accepted(packet, semantic, session.runtime_config)
    v1 = encode_checkpoint(accepted, session)
    assert verify_and_decode_checkpoint(v1, session) == accepted and accepted.feedback_evidence is None
    prep = session.preparation.preparation_identity
    base = semantic.request_identity
    suffix = f'-p{prep}-w{packet.window_ordinal:08d}-q{base}.json'
    one, two = tmp_path / f'b13-checkpoint-v1{suffix}', tmp_path / f'b13-checkpoint-v2{suffix}'
    assert inspect_checkpoint_versions_offline(tmp_path, session, packet.window_id) is None
    one.write_bytes(v1)
    assert inspect_checkpoint_versions_offline(tmp_path, session, packet.window_id) == accepted
    two.write_bytes(raw)
    with pytest.raises(B13OfflineCheckpointConflict):
        inspect_checkpoint_versions_offline(tmp_path, session, packet.window_id)
    assert one.read_bytes() == v1
    one.unlink()
    assert inspect_checkpoint_versions_offline(tmp_path, session, packet.window_id).feedback_evidence.canonical_bytes() == raw
    two.unlink()
    unknown = tmp_path / f'b13-checkpoint-v99{suffix}'
    unknown.write_bytes(raw)
    with pytest.raises(B13OfflineCheckpointConflict):
        inspect_checkpoint_versions_offline(tmp_path, session, packet.window_id)
    unknown.unlink()
    other = tmp_path / (f'b13-checkpoint-v2-p{prep}-w{packet.window_ordinal:08d}-q' + ('0' * 64) + '.json')
    other.write_bytes(raw)
    with pytest.raises(B13OfflineCheckpointConflict):
        inspect_checkpoint_versions_offline(tmp_path, session, packet.window_id)
    other.unlink()
    two.write_bytes(b'not valid')
    with pytest.raises(B13OfflineCheckpointConflict):
        inspect_checkpoint_versions_offline(tmp_path, session, packet.window_id)
