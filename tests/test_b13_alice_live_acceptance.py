"""Offline CLI acceptance tests; every provider client is fake."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from dataclasses import replace

import pytest

import b13_alice_live_acceptance as cli
import b13_alice_bounded_runner as alice
import short_drama.adaptation.disposition_runner as runner
import short_drama.adaptation.disposition_checkpoint_storage as storage
from short_drama.adaptation.disposition_execution_claim import claim_path, create_claim, execution_guard
from test_adaptation_strategy_execution import _FakeClient
from test_b13_disposition_checkpoint import _fixture
from test_b13_window_semantic import _proposal


@pytest.fixture
def setup_cli(tmp_path, monkeypatch):
    tree, session, packet, _ = _fixture(tmp_path)
    fake = _FakeClient([(_proposal(packet), False)], session.runtime_config)
    monkeypatch.setattr(cli, '_load_authority_read_only', lambda _: (tree.store, tree.pointers, session.preparation.source_snapshot))
    monkeypatch.setattr(cli, 'RuntimeConfig', lambda **_: session.runtime_config)
    monkeypatch.setattr(cli.pf, 'LlamaClient', lambda **_: object())
    monkeypatch.setattr(cli.preparation, 'prepare_b13_alice_execution_session', lambda **_: session)
    monkeypatch.setattr(alice, 'prepare_b13_alice_execution_session', lambda **_: session)
    monkeypatch.setattr(cli, 'OpenAICompatibleLLMClient', lambda _: fake)
    args = ['--execution-root', str(tmp_path / 'execution'), '--checkpoint-root', str(tmp_path / 'checkpoints'),
            '--input-ceiling-tokens', '65536']
    return tree, session, packet, fake, args


def invoke(args, capsys):
    code = cli.main(args)
    lines = capsys.readouterr().out.splitlines()
    return code, [json.loads(line) for line in lines]


def test_default_is_read_only(setup_cli, tmp_path, capsys):
    _, _, _, fake, args = setup_cli
    code, lines = invoke(args, capsys)
    assert code == 0 and lines[-1]['status'] == 'READY'
    assert lines[-1]['observed_provider_attempts'] == 0 and not fake.calls
    assert not (tmp_path / 'execution').exists() and not (tmp_path / 'checkpoints').exists()


def test_live_is_one_window_and_durable_repeat_reuses(setup_cli, capsys, monkeypatch):
    _, _, _, fake, args = setup_cli
    real = alice.run_b13_alice_bounded
    def checked(**kwargs):
        assert kwargs['max_new_windows'] == 1
        return real(**kwargs)
    monkeypatch.setattr(cli.bounded, 'run_b13_alice_bounded', checked)
    code, lines = invoke(args + ['--execute-live'], capsys)
    assert code == 0 and lines[0]['max_provider_attempts'] == 6
    assert lines[-1]['generated_count'] == 1 and lines[-1]['observed_provider_attempts'] == 1
    assert len(lines[-1]['checkpoint_identity']) == 64
    code, lines = invoke(args + ['--execute-live'], capsys)
    assert code == 0 and lines[-1]['reused_count'] == 1 and lines[-1]['generated_count'] == 0
    assert len(fake.calls) == 1


@pytest.mark.parametrize('failure', ['preflight', 'mismatch', 'root', 'claim', 'provider', 'fsync'])
def test_fail_closed(setup_cli, tmp_path, capsys, monkeypatch, failure):
    tree, session, packet, fake, args = setup_cli
    if failure == 'preflight':
        monkeypatch.setattr(cli.preparation, 'prepare_b13_alice_execution_session', lambda **_: (_ for _ in ()).throw(ValueError('secret')))
    elif failure == 'mismatch':
        fake.runtime_config = replace(session.runtime_config, request_model='wrong')
    elif failure == 'root':
        args[1] = str(tree.store.root / 'bad')
    elif failure == 'claim':
        with execution_guard(tmp_path / 'execution', store=tree.store, pointers=tree.pointers) as guard:
            create_claim(guard=guard, store=tree.store, pointers=tree.pointers, session=session,
                         checkpoint_root=tmp_path / 'checkpoints', window_id=packet.window_id)
    elif failure == 'provider':
        monkeypatch.setattr(fake, 'generate_structured', lambda *a, **kw: (_ for _ in ()).throw(RuntimeError('secret')))
    else:
        original_save, original_fsync = runner.save_checkpoint, storage.os.fsync
        def fail_save(*args):
            def fail_fsync(_fd):
                raise OSError('fsync secret')
            monkeypatch.setattr(storage.os, 'fsync', fail_fsync)
            try:
                return original_save(*args)
            finally:
                monkeypatch.setattr(storage.os, 'fsync', original_fsync)
        monkeypatch.setattr(runner, 'save_checkpoint', fail_save)
    code, lines = invoke(args + ['--execute-live'], capsys)
    assert code == 1 and lines[-1]['status'] == 'FAILED'
    assert 'secret' not in str(lines)
    if failure in ('provider', 'fsync'):
        assert lines[-1]['observed_provider_attempts'] is None
        assert claim_path(tmp_path / 'execution', session, packet.window_id).exists()
        code, _ = invoke(args + ['--execute-live'], capsys)
        assert code == 1
    else:
        assert not fake.calls


def test_unrecognized_limit_and_missing_roots(capsys):
    with pytest.raises(SystemExit):
        cli.main(['--max-new-windows', '2'])


def test_missing_foundation_layout_is_not_created(tmp_path):
    with pytest.raises(ValueError):
        cli._load_authority_read_only(tmp_path / "missing")
    assert not (tmp_path / "missing").exists()
