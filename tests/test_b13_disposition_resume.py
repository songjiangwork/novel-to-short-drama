"""Offline resume planning and fail-closed filesystem boundaries."""
from dataclasses import replace
import os

import pytest

import short_drama.adaptation.disposition_resume as resume
from short_drama.adaptation.disposition_checkpoint import B13_CHECKPOINT_MAX_BYTES
from short_drama.adaptation.disposition_checkpoint_storage import checkpoint_path, save_checkpoint
from short_drama.adaptation.disposition_execution import B13WindowRejectedSemanticRound
from short_drama.artifacts import ImmutableArtifactEnvelope
from short_drama.foundation import PointerKind
from short_drama.story.story_analysis import StoryAnalysisManifest
from short_drama.story.story_analysis_persistence import a6_pointer_id
from test_b13_disposition_checkpoint import _accepted, _fixture, _session, _provenance, _payload, _reencode, _broken_session
from test_b13_disposition_execution import _two_window_inputs, _allow_synthetic_two_window_preflight
from test_adaptation_strategy import _b12_test_runtime_config


def _plan(tree, session, root):
    return resume.plan_b13_resume(store=tree.store, pointers=tree.pointers, session=session, checkpoint_root=root)


def test_missing_and_empty_roots_no_writes(tmp_path, monkeypatch):
    tree, session, packet, _ = _fixture(tmp_path)
    root = tmp_path / 'missing'
    before = {p: p.read_bytes() for base in (tree.store.root, tree.pointers.root) for p in base.rglob('*') if p.is_file()}
    monkeypatch.setattr('short_drama.llm.LLMClient.generate_structured', lambda *_a, **_kw: pytest.fail('generation'))
    plan = _plan(tree, session, root)
    assert not root.exists()
    assert plan.verdict == 'READY' and plan.windows_total == plan.needs_generation_count == 1
    assert plan.verified_reuse_count == 0
    assert plan.windows[0].window_id == packet.window_id
    assert plan.windows[0].decision == resume.B13ResumeDecision.NEEDS_GENERATION
    assert plan.windows[0].accepted_proposal is None
    root.mkdir()
    assert _plan(tree, session, root) == plan
    assert list(root.iterdir()) == []
    assert all(p.read_bytes() == content for p, content in before.items())


@pytest.mark.parametrize('chosen', [(1,), (0, 1)])
def test_non_prefix_and_full_reuse(tmp_path, monkeypatch, chosen):
    tree, preparation, requests, _, runtime, _, _ = _two_window_inputs(tmp_path)
    _allow_synthetic_two_window_preflight(monkeypatch, preparation)
    session = _session(preparation, requests, runtime)
    root = tmp_path / 'checkpoints'
    expected = {}
    for i in chosen:
        rejected = (B13WindowRejectedSemanticRound(1, 2, _provenance(requests[i], runtime), 'rejected'),)
        accepted = _accepted(preparation.packets[i], requests[i], runtime, rejected=rejected, accepted_attempts=3)
        save_checkpoint(root, accepted, session)
        expected[i] = accepted
    plan = _plan(tree, session, root)
    assert [w.window_id for w in plan.windows] == [p.window_id for p in preparation.packets]
    assert [w.request_identity for w in plan.windows] == [s.request_identity for s in requests]
    assert plan.verified_reuse_count == len(chosen) and plan.needs_generation_count == 2 - len(chosen)
    for i, w in enumerate(plan.windows):
        assert w.accepted_proposal == expected.get(i)
        assert w.decision == (resume.B13ResumeDecision.VERIFIED_REUSE if i in chosen else resume.B13ResumeDecision.NEEDS_GENERATION)


@pytest.mark.parametrize('bad', ['corrupt', 'truncated', 'oversize', 'symlink', 'fifo', 'identity', 'provenance', 'accounting'])
def test_bad_exact_checkpoint_fails_closed(tmp_path, bad):
    tree, session, packet, semantic = _fixture(tmp_path)
    root = tmp_path / 'root'
    accepted = _accepted(packet, semantic, _b12_test_runtime_config())
    path = save_checkpoint(root, accepted, session)
    if bad in ('identity', 'provenance', 'accounting'):
        payload = _payload(accepted, session)
        if bad == 'identity': payload['request_identity'] = '0' * 64
        if bad == 'provenance': payload['invocation_provenance']['request_hash'] = '0' * 64
        if bad == 'accounting': payload['provider_attempts'] += 1
        path.write_bytes(_reencode(payload))
    elif bad == 'corrupt': path.write_bytes(b'{}')
    elif bad == 'truncated': path.write_bytes(path.read_bytes()[:-1])
    elif bad == 'oversize': path.write_bytes(b'x' * (B13_CHECKPOINT_MAX_BYTES + 1))
    elif bad == 'symlink':
        path.unlink()
        path.symlink_to(tmp_path / 'elsewhere')
    else:
        path.unlink()
        os.mkfifo(path)
    with pytest.raises(resume.B13ResumeError): _plan(tree, session, root)


def test_unsafe_root_fails(tmp_path):
    tree, session, _, _ = _fixture(tmp_path)
    root = tmp_path / 'root'
    root.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(resume.B13ResumeError): _plan(tree, session, root)


@pytest.mark.parametrize('name', ['non_pass_preflight', 'wrong_model', 'wrong_strategy_ref', 'missing_budgets',
    'missing_evidence', 'evidence_request_mismatch', 'tokenizer_unknown', 'budget_context_mismatch',
    'semantic_runtime_mismatch', 'missing_semantic'])
def test_bad_session_fails_before_loading(tmp_path, monkeypatch, name):
    tree, session, _, _ = _fixture(tmp_path)
    monkeypatch.setattr(resume, 'load_checkpoint', lambda *_args: pytest.fail('must not load'))
    with pytest.raises(resume.B13ResumeError): _plan(tree, _broken_session(session, name), tmp_path / 'root')


def test_tampered_semantic_and_budget_fail(tmp_path):
    tree, session, _, _ = _fixture(tmp_path)
    altered = replace(session.semantic_preparations[0], request_identity='0' * 64)
    row = replace(session.window_evidence[0], request_identity='0' * 64)
    for broken in (replace(session, semantic_preparations=(altered,), window_evidence=(row,)),
                   replace(session, input_ceiling_tokens=1, budgets=(replace(session.budgets[0], input_ceiling_tokens=1),))):
        with pytest.raises(resume.B13ResumeError): _plan(tree, broken, tmp_path / 'root')


def _move_a6(tree, session):
    source = session.preparation.source_snapshot
    manifest = StoryAnalysisManifest.from_dict(tree.store.get_ref(source.story_analysis_manifest_ref).payload)
    ref = source.story_analysis_manifest_ref
    replacement = tree.store.put(ImmutableArtifactEnvelope.create(artifact_type=ref.artifact_type,
        artifact_id=ref.artifact_id, revision=ref.revision + 1, schema_version=1, payload=manifest.to_dict()))
    pointer_id = a6_pointer_id(manifest.project_id, manifest.document_id, manifest.semantic_identity.story_analysis_profile_id)
    old = tree.pointers.resolve_current_pointer_ref(pointer_id)
    tree.pointers.compare_and_set(pointer_id=pointer_id, pointer_kind=PointerKind.CURRENT,
                                  expected_pointer_ref=old, target_ref=replacement)


@pytest.mark.parametrize('during', [False, True])
def test_a6_change_fails_before_or_after_loading(tmp_path, monkeypatch, during):
    tree, session, _, _ = _fixture(tmp_path)
    if during:
        original = resume.load_checkpoint
        def move(*args):
            result = original(*args)
            _move_a6(tree, session)
            return result
        monkeypatch.setattr(resume, 'load_checkpoint', move)
    else:
        _move_a6(tree, session)
    with pytest.raises(resume.B13ResumeError): _plan(tree, session, tmp_path / 'root')


def test_a5_current_change_fails(tmp_path):
    from short_drama.story.consolidation import ConsolidationManifest
    from short_drama.story.consolidation_persistence import a5_pointer_id
    tree, session, _, _ = _fixture(tmp_path)
    ref = session.preparation.source_snapshot.consolidation_manifest_ref
    envelope = tree.store.get_ref(ref)
    manifest = ConsolidationManifest.from_dict(envelope.payload)
    replacement = tree.store.put(ImmutableArtifactEnvelope.create(artifact_type=ref.artifact_type,
        artifact_id=ref.artifact_id, revision=ref.revision + 1, schema_version=1, payload=envelope.payload))
    pointer_id = a5_pointer_id(manifest.project_id, manifest.document_id, manifest.semantic_identity.consolidation_profile_id)
    old = tree.pointers.resolve_current_pointer_ref(pointer_id)
    tree.pointers.compare_and_set(pointer_id=pointer_id, pointer_kind=PointerKind.CURRENT,
                                  expected_pointer_ref=old, target_ref=replacement)
    with pytest.raises(resume.B13ResumeError): _plan(tree, session, tmp_path / 'root')


def test_explicit_pin_survives_b12_current_advance(tmp_path):
    from short_drama.adaptation import prepare_b13_disposition_windows, build_b13_window_semantic_preparation
    from test_b13_disposition_execution import _execution_inputs
    from test_adaptation_strategy_execution import _move_current_to_payload
    tree, semantic, publication, _, _, _, runtime = _execution_inputs(tmp_path)
    prep = prepare_b13_disposition_windows(store=tree.store, pointers=tree.pointers,
        source_snapshot=semantic.preparation.packet.source_snapshot,
        adaptation_target_id=semantic.preparation.packet.resolved_target.adaptation_target_id,
        strategy_ref=publication.strategy_ref)
    requests = tuple(build_b13_window_semantic_preparation(p, runtime_config=runtime) for p in prep.packets)
    session = _session(prep, requests, runtime)
    _move_current_to_payload(tree, semantic, publication, publication.artifact.to_dict())
    assert _plan(tree, session, tmp_path / 'root').strategy_ref == publication.strategy_ref


def test_unrelated_checkpoint_name_is_not_scanned(tmp_path):
    tree, session, packet, _ = _fixture(tmp_path)
    root = tmp_path / 'root'
    target = checkpoint_path(root, session, packet.window_id)
    (root / target.name.replace(session.preparation.preparation_identity, '0' * 64)).write_bytes(b'corrupt historical bytes')
    assert _plan(tree, session, root).needs_generation_count == 1


@pytest.mark.parametrize('order', ['missing', 'duplicate', 'reordered'])
def test_bad_window_sequence_fails_before_loading(tmp_path, monkeypatch, order):
    tree, preparation, requests, _, runtime, _, _ = _two_window_inputs(tmp_path)
    _allow_synthetic_two_window_preflight(monkeypatch, preparation)
    session = _session(preparation, requests, runtime)
    invalid = {'missing': requests[:1], 'duplicate': (requests[0], requests[0]),
               'reordered': tuple(reversed(requests))}[order]
    monkeypatch.setattr(resume, 'load_checkpoint', lambda *_: pytest.fail('must not load'))
    with pytest.raises(resume.B13ResumeError):
        _plan(tree, replace(session, semantic_preparations=invalid), tmp_path / 'root')


def test_missing_explicit_strategy_fails_closed(tmp_path, monkeypatch):
    tree, session, _, _ = _fixture(tmp_path)
    ref = session.strategy_ref
    original = tree.store.get_ref
    def get_ref(value):
        if value == ref:
            raise ValueError('invalid selected strategy')
        return original(value)
    monkeypatch.setattr(tree.store, 'get_ref', get_ref)
    with pytest.raises(resume.B13ResumeError): _plan(tree, session, tmp_path / 'root')


def test_several_non_prefix_reuses_keep_complete_order(tmp_path, monkeypatch):
    from short_drama.adaptation import B12StrategySelection, B13DispositionPreparation, B13DispositionWindowPacket, B13DispositionPacketMeasurement, build_b13_window_semantic_preparation
    from short_drama.adaptation.target import CanonicalTargetReference
    from short_drama.artifacts import canonical_json_bytes
    tree, base, _, _, runtime, _, _ = _two_window_inputs(tmp_path)
    source = replace(base.source_snapshot, canonical_references=(*base.source_snapshot.canonical_references,
                     CanonicalTargetReference('event', 'evt_000003')))
    strategy = replace(base.strategy_selection.artifact.strategy,
                       resolved_target=replace(base.strategy_selection.artifact.strategy.resolved_target, source_snapshot=source))
    selection = B12StrategySelection(base.strategy_selection.strategy_ref,
        replace(base.strategy_selection.artifact, source_snapshot=source, strategy=strategy),
        base.strategy_selection.current_pointer_ref)
    packets = []
    for i in range(1, 4):
        event = f'evt_{i:06d}'
        raw = base.packets[0].to_dict()
        raw['source_snapshot'] = source.to_dict()
        raw['global_adaptation_strategy']['strategy'] = strategy.to_dict()
        raw['global_adaptation_strategy']['strategy_hash'] = strategy.content_hash()
        raw['window'] = {'window_id': f'window_{i:04d}', 'window_ordinal': i,
                         'owned_event_refs': [event], 'context_event_refs': []}
        raw['owned_source_events'] = [{'event_id': event, 'narrative_order': i}]
        raw['context_source_events'] = []
        packets.append(B13DispositionWindowPacket(source, selection, f'window_{i:04d}', i, canonical_json_bytes(raw)))
    prep = B13DispositionPreparation(source, selection, tuple(packets),
        tuple(B13DispositionPacketMeasurement(p.window_id, len(p.canonical_bytes()), 1, p.content_hash()) for p in packets),
        tuple(f'evt_{i:06d}' for i in range(1, 4)), base.preparation_identity)
    requests = tuple(build_b13_window_semantic_preparation(p, runtime_config=runtime) for p in packets)
    session = _session(prep, requests, runtime)
    _allow_synthetic_two_window_preflight(monkeypatch, prep)
    root = tmp_path / 'private'
    accepted = {i: _accepted(packets[i], requests[i], runtime) for i in (0, 2)}
    for item in accepted.values(): save_checkpoint(root, item, session)
    plan = _plan(tree, session, root)
    assert [w.window_id for w in plan.windows] == [p.window_id for p in packets]
    assert [w.accepted_proposal for w in plan.windows] == [accepted[0], None, accepted[2]]
    assert (plan.verified_reuse_count, plan.needs_generation_count) == (2, 1)


def test_consistently_changed_budget_during_loading_fails(tmp_path, monkeypatch):
    tree, session, _, _ = _fixture(tmp_path)
    original = resume.load_checkpoint
    def change(*args):
        result = original(*args)
        new_ceiling = session.input_ceiling_tokens - 1
        object.__setattr__(session, 'input_ceiling_tokens', new_ceiling)
        object.__setattr__(session, 'budgets', tuple(replace(b, input_ceiling_tokens=new_ceiling) for b in session.budgets))
        return result
    monkeypatch.setattr(resume, 'load_checkpoint', change)
    with pytest.raises(resume.B13ResumeError): _plan(tree, session, tmp_path / 'root')


def test_root_removed_after_load_root_check_is_not_recreated(tmp_path, monkeypatch):
    import short_drama.adaptation.disposition_checkpoint_storage as storage
    tree, session, _, _ = _fixture(tmp_path)
    root = tmp_path / 'root'
    root.mkdir()
    original = storage._root
    def removed(*args, **kwargs):
        result = original(*args, **kwargs)
        if result is not None:
            root.rmdir()
        return result
    monkeypatch.setattr(storage, '_root', removed)
    # A disappeared root is unsafe, and must never be recreated during a read.
    with pytest.raises(resume.B13ResumeError): _plan(tree, session, root)
    assert not root.exists()
