"""Fresh orchestration uses only offline discovery and tokenization."""
import sys
import uuid
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts'))
import b13_alice_resume_preflight as cli
import test_b13_alice_token_budget_preflight as pft


@pytest.fixture
def report_path():
    path = Path('/tmp') / f'b13-resume-test-{uuid.uuid4().hex}.json'
    yield path
    path.unlink(missing_ok=True)


def test_fresh_preflight_and_no_writes(tmp_path):
    tree, source, publication, target = pft._preparation(tmp_path)
    client = pft._client(pft._responses())
    before = {p: p.read_bytes() for base in (tree.store.root, tree.pointers.root) for p in base.rglob('*') if p.is_file()}
    root = tmp_path / 'missing'
    plan = cli.prepare_b13_alice_resume(store=tree.store, pointers=tree.pointers, source_snapshot=source,
        runtime_config=pft._runtime(), client=client, input_ceiling_tokens=65536,
        strategy_ref=publication.strategy_ref, adaptation_target_id=target, checkpoint_root=root)
    assert cli._report(plan)['verdict'] == 'READY'
    assert plan.windows_total == plan.needs_generation_count and not root.exists()
    assert client.generation_calls == []
    assert {p: p.read_bytes() for base in (tree.store.root, tree.pointers.root) for p in base.rglob('*') if p.is_file()} == before


def test_unknown_capacity_never_plans(tmp_path, monkeypatch):
    tree, source, publication, target = pft._preparation(tmp_path)
    responses = pft._responses()
    responses[('GET', '/slots')] = 500
    monkeypatch.setattr(cli, 'plan_b13_resume', lambda **kw: pytest.fail('must not plan'))
    with pytest.raises(Exception):
        cli.prepare_b13_alice_resume(store=tree.store, pointers=tree.pointers, source_snapshot=source,
            runtime_config=pft._runtime(), client=pft._client(responses), input_ceiling_tokens=65536,
            strategy_ref=publication.strategy_ref, adaptation_target_id=target, checkpoint_root=tmp_path / 'root')


def test_cli_rejects_execution_and_report_outside_tmp(capsys):
    with pytest.raises(SystemExit):
        cli.main(['--execute', '--input-ceiling-tokens', '100', '--checkpoint-root', '/tmp/checkpoints'])
    assert cli.main(['--input-ceiling-tokens', '100', '--checkpoint-root', '/tmp/checkpoints', '--report', '/var/report']) == 1
    assert 'under /tmp' in capsys.readouterr().err
    assert cli.main(['--input-ceiling-tokens', '100', '--checkpoint-root', '/tmp/private',
                     '--report', '/tmp/private/report.json']) == 1
    assert 'outside' in capsys.readouterr().err


def test_cli_runs_fresh_pipeline_and_emits_bounded_report(tmp_path, monkeypatch, capsys, report_path):
    tree, source, publication, target = pft._preparation(tmp_path)
    client = pft._client(pft._responses())
    monkeypatch.setattr(cli.pf, 'load_alice_authority', lambda _: (tree.store, tree.pointers, source))
    monkeypatch.setattr(cli.pf, 'LlamaClient', lambda **_: client)
    original = cli.prepare_b13_alice_resume
    monkeypatch.setattr(cli, 'prepare_b13_alice_resume', lambda **kw: original(
        **kw, strategy_ref=publication.strategy_ref, adaptation_target_id=target))
    report = report_path
    assert cli.main(['--input-ceiling-tokens', '65536', '--request-model', pft.REQUEST_MODEL,
                     '--base-url', 'http://srv/v1', '--provider-family', 'qwen',
                     '--checkpoint-root', str(tmp_path / 'missing'), '--report', str(report)]) == 0
    import json
    assert report.stat().st_size <= cli._MAX_REPORT_BYTES
    payload = json.loads(report.read_text())
    assert json.loads(capsys.readouterr().out) == payload
    assert payload['verdict'] == 'READY' and payload['provider_generation_calls'] == 0
    assert payload['needs_generation_count'] == payload['windows_total']
    assert len(payload['preparation_identity']) == 64
    assert payload['strategy_ref'] == publication.strategy_ref.to_dict()
    assert payload['source']['a5_consolidation_manifest_ref'] == source.consolidation_manifest_ref.to_dict()
    assert payload['source']['a6_story_analysis_manifest_ref'] == source.story_analysis_manifest_ref.to_dict()
    assert len(payload['windows']) == payload['windows_total']
    for i, window in enumerate(payload['windows'], 1):
        assert set(window) == {'window_ordinal', 'window_id', 'request_identity', 'decision'}
        assert window['window_ordinal'] == i and len(window['request_identity']) == 64
        assert window['decision'] == 'NEEDS_GENERATION'
    assert client.generation_calls == []


def test_report_is_exclusive_and_rejects_symlinked_parents(tmp_path, report_path):
    report = report_path
    report.write_bytes(b'original')
    with pytest.raises(OSError):
        cli._write_report(report, b'new')
    assert report.read_bytes() == b'original'
    real = tmp_path / 'real'
    real.mkdir()
    link = tmp_path / 'link'
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(ValueError):
        cli._report_path(link / 'report.json', ())
    assert not (real / 'report.json').exists()


def test_report_nested_parent_is_rejected_before_substitution(tmp_path):
    parent = tmp_path / 'parent'
    parent.mkdir()
    with pytest.raises(ValueError):
        cli._report_path(parent / 'report.json', ())


def test_cli_rejects_report_symlink_inserted_after_validation(tmp_path, monkeypatch, report_path):
    tree, source, publication, target = pft._preparation(tmp_path)
    client = pft._client(pft._responses())
    artifact = next(p for p in tree.store.root.rglob('*') if p.is_file())
    original_bytes = artifact.read_bytes()
    report = report_path
    monkeypatch.setattr(cli.pf, 'load_alice_authority', lambda _: (tree.store, tree.pointers, source))
    monkeypatch.setattr(cli.pf, 'LlamaClient', lambda **_: client)
    original = cli.prepare_b13_alice_resume
    def prepare_and_substitute(**kwargs):
        plan = original(**kwargs, strategy_ref=publication.strategy_ref, adaptation_target_id=target)
        report.symlink_to(artifact)
        return plan
    monkeypatch.setattr(cli, 'prepare_b13_alice_resume', prepare_and_substitute)
    before = {p: p.read_bytes() for base in (tree.store.root, tree.pointers.root) for p in base.rglob('*') if p.is_file()}
    assert cli.main(['--input-ceiling-tokens', '65536', '--request-model', pft.REQUEST_MODEL,
                     '--base-url', 'http://srv/v1', '--provider-family', 'qwen',
                     '--checkpoint-root', str(tmp_path / 'missing'), '--report', str(report)]) == 1
    assert artifact.read_bytes() == original_bytes
    assert {p: p.read_bytes() for base in (tree.store.root, tree.pointers.root) for p in base.rglob('*') if p.is_file()} == before
    assert client.generation_calls == []


def test_report_rejects_nested_parent_paths(tmp_path):
    for parent in (tmp_path / 'missing', tmp_path):
        with pytest.raises(ValueError):
            cli._write_report(parent / 'report.json', b'diagnostic')


def test_audit_report_preserves_exact_reuse_decision_without_checkpoint_body(tmp_path):
    from short_drama.adaptation.disposition_checkpoint_storage import save_checkpoint
    from short_drama.adaptation.disposition_resume import plan_b13_resume
    from test_b13_disposition_checkpoint import _fixture, _accepted
    from test_adaptation_strategy import _b12_test_runtime_config
    tree, session, packet, semantic = _fixture(tmp_path)
    root = tmp_path / 'checkpoints'
    save_checkpoint(root, _accepted(packet, semantic, _b12_test_runtime_config()), session)
    plan = plan_b13_resume(store=tree.store, pointers=tree.pointers, session=session, checkpoint_root=root)
    report = cli._report(plan)
    assert report == cli._report(plan)
    assert report['preparation_identity'] == session.preparation.preparation_identity
    assert report['windows'] == [{'window_ordinal': packet.window_ordinal, 'window_id': packet.window_id,
                                  'request_identity': semantic.request_identity, 'decision': 'VERIFIED_REUSE'}]
    assert report['verified_reuse_count'] == 1 and report['needs_generation_count'] == 0
    assert 'proposal' not in str(report) and 'provenance' not in str(report)
