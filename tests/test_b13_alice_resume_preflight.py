"""Fresh orchestration uses only offline discovery and tokenization."""
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts'))
import b13_alice_resume_preflight as cli
import test_b13_alice_token_budget_preflight as pft


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


def test_cli_runs_fresh_pipeline_and_emits_bounded_report(tmp_path, monkeypatch, capsys):
    tree, source, publication, target = pft._preparation(tmp_path)
    client = pft._client(pft._responses())
    monkeypatch.setattr(cli.pf, 'load_alice_authority', lambda _: (tree.store, tree.pointers, source))
    monkeypatch.setattr(cli.pf, 'LlamaClient', lambda **_: client)
    original = cli.prepare_b13_alice_resume
    monkeypatch.setattr(cli, 'prepare_b13_alice_resume', lambda **kw: original(
        **kw, strategy_ref=publication.strategy_ref, adaptation_target_id=target))
    report = tmp_path / 'report.json'
    assert cli.main(['--input-ceiling-tokens', '65536', '--request-model', pft.REQUEST_MODEL,
                     '--base-url', 'http://srv/v1', '--provider-family', 'qwen',
                     '--checkpoint-root', str(tmp_path / 'missing'), '--report', str(report)]) == 0
    import json
    payload = json.loads(report.read_text())
    assert json.loads(capsys.readouterr().out) == payload
    assert payload['verdict'] == 'READY' and payload['provider_generation_calls'] == 0
    assert payload['needs_generation_count'] == payload['windows_total']
    assert client.generation_calls == []
