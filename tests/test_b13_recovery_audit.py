import hashlib
import os
from dataclasses import replace

import pytest

from short_drama.artifacts import content_hash, strict_json_loads
from short_drama.adaptation import disposition_execution_claim as claim
from short_drama.adaptation import disposition_recovery_audit as audit
from short_drama.adaptation.disposition_checkpoint import encode_checkpoint
from short_drama.adaptation.disposition_checkpoint_storage import save_checkpoint
from test_adaptation_strategy import _b12_test_runtime_config
from test_b13_disposition_checkpoint import _accepted, _fixture, _session


def setup_case(tmp_path):
    tree, session, packet, semantic = _fixture(tmp_path)
    root, checkpoints = tmp_path / 'execution', tmp_path / 'checkpoints'
    with claim.execution_guard(root, store=tree.store, pointers=tree.pointers) as guard:
        claim.create_claim(guard=guard, store=tree.store, pointers=tree.pointers, session=session,
                           checkpoint_root=checkpoints, window_id=packet.window_id)
    accepted = _accepted(packet, semantic, _b12_test_runtime_config())
    save_checkpoint(checkpoints, accepted, session)
    args = dict(store=tree.store, pointers=tree.pointers, session=session, execution_root=root,
                checkpoint_root=checkpoints, window_id=packet.window_id)
    ident = claim._identity(session, packet.window_id)
    original = claim._encoded(ident)
    binding = dict(claim_identity=content_hash(ident), claim_bytes_hash=hashlib.sha256(original).hexdigest(),
                   preparation_identity=ident['preparation_identity'], window_id=packet.window_id,
                   window_ordinal=ident['window_ordinal'], request_identity=ident['request_identity'],
                   semantic_request_hash=semantic.request.request_hash,
                   authority_hash=content_hash({'source_snapshot': session.preparation.source_snapshot.to_dict(),
                       'strategy_ref': session.preparation.strategy_selection.strategy_ref.to_dict(),
                       'strategy_hash': ident['strategy_hash'], 'selection_mode': 'CURRENT',
                       'current_pointer_ref': session.preparation.strategy_selection.current_pointer_ref.to_dict()}))
    roots = {k: audit._root_identity(p) for k, p in dict(execution=root, checkpoint=checkpoints,
                   artifact=tree.store.root, pointer=tree.pointers.root).items()}
    return args, binding, roots, accepted


def put(args, binding, kind, fields):
    raw = audit.encode_record(kind, dict(binding=binding, **fields))
    path = args['execution_root'] / audit.record_name(binding['claim_identity'], kind)
    path.write_bytes(raw); path.chmod(0o600)
    assert audit.decode_record(raw, kind)['content_hash'] == strict_json_loads(raw)['content_hash']
    return strict_json_loads(raw)


def decision(args, binding, roots):
    return put(args, binding, 'decision', dict(decision_id='decision-1', operator_id='operator',
        recorded_at='2026-01-01T00:00:00Z', reason='inspection', evidence=['a' * 64],
        prior_outcome='unknown', ack_prior_requests_may_have_completed=True,
        ack_duplicate_generation_possible=True, roots=roots, root_result_identity=None,
        policy_reference='independently-approved-test-policy', caps={'semantic_rounds': 2, 'technical_attempts_per_round': 3,
        'total_attempts': 6}, attempt_id='attempt-1'))


def consumption(args, binding, roots, d):
    return put(args, binding, 'consumption', dict(decision_identity=d['content_hash'], decision_id=d['decision_id'],
        attempt_id=d['attempt_id'], recorded_at='2026-01-01T00:00:00Z', bindings_hash=content_hash(
            {'binding': binding, 'roots': roots, 'caps': d['caps'], 'policy_reference': d['policy_reference']})))


def test_ordinary_and_complete(tmp_path, monkeypatch):
    monkeypatch.setattr('short_drama.llm.LLMClient.generate_structured', lambda *_a, **_k: pytest.fail('provider called'))
    args, binding, roots, accepted = setup_case(tmp_path)
    from test_b13_disposition_execution_claim import _snapshot
    before = _snapshot(tmp_path)
    assert audit.inspect_recovery(**args) == audit.Audit.ORDINARY_AUDIT
    assert _snapshot(tmp_path) == before
    d = decision(args, binding, roots)
    assert audit.inspect_recovery(**args) == audit.Audit.ORDINARY_AUDIT
    c = consumption(args, binding, roots, d)
    assert audit.inspect_recovery(**args) == audit.Audit.AUDIT_INCOMPLETE
    raw = encode_checkpoint(accepted, args['session'])
    put(args, binding, 'terminal', dict(decision_identity=d['content_hash'], consumption_identity=c['content_hash'],
        attempt_id=d['attempt_id'], recorded_at='2026-01-01T00:00:00Z', outcome='CHECKPOINTED', failure_code=None,
        semantic_rounds=accepted.semantic_rounds, provider_attempts=accepted.provider_attempts,
        original_semantic_rounds=None, original_provider_attempts=None, lineage_semantic_rounds=None,
        lineage_provider_attempts=None, checkpoint_identity=strict_json_loads(raw)['checkpoint_identity'],
        checkpoint_bytes_hash=hashlib.sha256(raw).hexdigest()))
    assert audit.inspect_recovery(**args) == audit.Audit.AUDIT_INCOMPLETE
    assert audit.inspect_recovery(**args, approved_policy_reference='independently-approved-test-policy') == audit.Audit.RECOVERY_AUDIT_COMPLETE


@pytest.mark.parametrize('bad', ['duplicate', 'symlink', 'oversize', 'version', 'partial', 'unknown_slot', 'root'])
def test_bad_evidence_blocks(tmp_path, bad):
    args, binding, roots, _ = setup_case(tmp_path)
    path = args['execution_root'] / audit.record_name(binding['claim_identity'], 'decision')
    if bad == 'root':
        roots['execution']['ino'] += 1
        decision(args, binding, roots)
    elif bad == 'symlink': path.symlink_to(tmp_path / 'missing')
    elif bad == 'unknown_slot':
        (args['execution_root'] / (path.name + '.tmp')).touch(mode=0o600)
    else:
        d = decision(args, binding, roots)
        raw = path.read_bytes()
        if bad == 'duplicate': raw = raw[:-1] + b',"schema_version":1}'
        elif bad == 'oversize': raw = b'x' * (audit.MAX_RECORD_BYTES + 1)
        elif bad == 'version': raw = raw.replace(b'"schema_version":1', b'"schema_version":2')
        else: raw = raw[:-1]
        path.write_bytes(raw)
    assert audit.inspect_recovery(**args) == audit.Audit.AUDIT_INCOMPLETE


def test_no_checkpoint_blocks_and_no_roots_created(tmp_path):
    args, binding, roots, _ = setup_case(tmp_path)
    for path in args['checkpoint_root'].iterdir(): path.unlink()
    args['checkpoint_root'].rmdir()
    assert audit.inspect_recovery(**args) == audit.Audit.BLOCKED
    assert not args['checkpoint_root'].exists()


def test_slot_replacement_fails_closed(tmp_path, monkeypatch):
    args, binding, roots, _ = setup_case(tmp_path)
    decision(args, binding, roots)
    real = os.open
    def replace(name, flags, *a, **kw):
        fd = real(name, flags, *a, **kw)
        if name == audit.record_name(binding['claim_identity'], 'decision'):
            path = args['execution_root'] / name
            path.unlink(); path.write_bytes(b'{}'); path.chmod(0o600)
        return fd
    monkeypatch.setattr(audit.os, 'open', replace)
    assert audit.inspect_recovery(**args) == audit.Audit.AUDIT_INCOMPLETE


def test_wrong_claim_and_authority_fail_closed(tmp_path):
    from test_b13_disposition_resume import _move_a6
    args, binding, roots, _ = setup_case(tmp_path)
    path = claim.claim_path(args['execution_root'], args['session'], args['window_id'])
    raw = path.read_bytes()
    path.write_bytes(raw[:-1])
    assert audit.inspect_recovery(**args) == audit.Audit.AUDIT_INCOMPLETE
    path.write_bytes(raw)
    _move_a6(type('Tree', (), {'store': args['store'], 'pointers': args['pointers']})(), args['session'])
    # An authority failure before checkpoint verification blocks all use.
    assert audit.inspect_recovery(**args) == audit.Audit.BLOCKED


def test_incomplete_initialization_blocks(tmp_path):
    args, binding, roots, _ = setup_case(tmp_path)
    put(args, binding, 'root_intent', dict(operator_id='operator', recorded_at='2026-01-01T00:00:00Z',
        reason='initialization', parent=audit._root_identity(args['checkpoint_root'].parent),
        desired_path=str(args['checkpoint_root']), historically_absent=True))
    assert audit.inspect_recovery(**args) == audit.Audit.AUDIT_INCOMPLETE


def test_recovery_binding_rejects_cross_selection_modes(tmp_path):
    from short_drama.adaptation import prepare_b13_disposition_windows, build_b13_window_semantic_preparation

    args, binding, roots, _ = setup_case(tmp_path)
    current = args['session']
    selection = current.preparation.strategy_selection
    explicit_prep = prepare_b13_disposition_windows(
        store=args['store'], pointers=args['pointers'], source_snapshot=current.preparation.source_snapshot,
        adaptation_target_id=selection.artifact.strategy.resolved_target.adaptation_target_id,
        strategy_ref=selection.strategy_ref)
    requests = tuple(build_b13_window_semantic_preparation(p, runtime_config=current.runtime_config)
                     for p in explicit_prep.packets)
    explicit = _session(explicit_prep, requests, current.runtime_config)
    assert explicit_prep.strategy_selection.current_pointer_ref is None
    assert explicit_prep.strategy_selection.strategy_ref == selection.strategy_ref
    assert explicit_prep.preparation_identity == current.preparation.preparation_identity
    assert claim._encoded(claim._identity(explicit, args['window_id'])) == claim._encoded(
        claim._identity(current, args['window_id']))
    assert requests[0].request_identity == current.semantic_preparations[0].request_identity
    decision(args, binding, roots)
    assert audit.inspect_recovery(**args) == audit.Audit.ORDINARY_AUDIT
    explicit_args = dict(args, session=explicit)
    assert audit.inspect_recovery(**explicit_args) == audit.Audit.AUDIT_INCOMPLETE
    explicit_binding = dict(binding, authority_hash=content_hash({
        'source_snapshot': explicit_prep.source_snapshot.to_dict(),
        'strategy_ref': selection.strategy_ref.to_dict(), 'strategy_hash': selection.strategy_hash,
        'selection_mode': 'EXPLICIT', 'current_pointer_ref': None}))
    decision(args, explicit_binding, roots)
    assert audit.inspect_recovery(**explicit_args) == audit.Audit.ORDINARY_AUDIT
    assert audit.inspect_recovery(**args) == audit.Audit.AUDIT_INCOMPLETE


def test_recovery_binding_rejects_other_current_revision(tmp_path):
    args, binding, roots, _ = setup_case(tmp_path)
    preparation = args['session'].preparation
    selection = preparation.strategy_selection
    pointer = selection.current_pointer_ref
    other = replace(pointer, revision=pointer.revision + 1)
    altered = dict(binding, authority_hash=content_hash({
        'source_snapshot': preparation.source_snapshot.to_dict(),
        'strategy_ref': selection.strategy_ref.to_dict(), 'strategy_hash': selection.strategy_hash,
        'selection_mode': 'CURRENT', 'current_pointer_ref': other.to_dict()}))
    decision(args, altered, roots)
    assert audit.inspect_recovery(**args) == audit.Audit.AUDIT_INCOMPLETE
    decision(args, binding, roots)
    assert audit.inspect_recovery(**args) == audit.Audit.ORDINARY_AUDIT


def terminal_fields():
    return dict(binding={'claim_identity': 'a' * 64, 'claim_bytes_hash': 'b' * 64,
        'preparation_identity': 'c' * 64, 'window_id': 'window_0001', 'window_ordinal': 1,
        'request_identity': 'd' * 64, 'semantic_request_hash': 'e' * 64, 'authority_hash': 'f' * 64},
        decision_identity='a' * 64, consumption_identity='b' * 64, attempt_id='attempt-1',
        recorded_at='2026-01-01T00:00:00Z', outcome='FAILED', failure_code=None,
        semantic_rounds=None, provider_attempts=None, original_semantic_rounds=None,
        original_provider_attempts=None, lineage_semantic_rounds=None, lineage_provider_attempts=None,
        checkpoint_identity=None, checkpoint_bytes_hash=None)


@pytest.mark.parametrize('code', [None, 'PROVIDER_TIMEOUT', 'PROVIDER_TRANSPORT_FAILURE',
    'PROVIDER_HTTP_FAILURE', 'STRUCTURED_OUTPUT_FAILURE', 'SEMANTIC_GENERATION_EXHAUSTED',
    'PROVENANCE_FAILURE', 'AUTHORITY_FAILURE', 'CHECKPOINT_STORAGE_FAILURE', 'UNKNOWN_FAILURE'])
def test_safe_terminal_failure_codes(code):
    fields = dict(terminal_fields(), failure_code=code)
    assert audit.decode_record(audit.encode_record('terminal', fields), 'terminal')['failure_code'] == code


@pytest.mark.parametrize('code', ['SOME_UNKNOWN_FAILURE', 'TimeoutError token secret', 'api-key-12345'])
def test_rejects_untrusted_failure_text(code):
    with pytest.raises(audit.RecoveryEvidenceError):
        audit.encode_record('terminal', dict(terminal_fields(), failure_code=code))


@pytest.mark.parametrize('changes', [
    {'outcome': 'CHECKPOINTED', 'failure_code': 'PROVIDER_TIMEOUT', 'semantic_rounds': 1,
     'provider_attempts': 1, 'checkpoint_identity': 'a' * 64, 'checkpoint_bytes_hash': 'b' * 64},
    {'checkpoint_identity': 'a' * 64}, {'checkpoint_bytes_hash': 'b' * 64},
])
def test_terminal_rejects_conflicting_outcome_evidence(changes):
    with pytest.raises(audit.RecoveryEvidenceError):
        audit.encode_record('terminal', dict(terminal_fields(), **changes))
