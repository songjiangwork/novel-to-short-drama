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
    code, lines = invoke(args + ['--execute-live', '--generation-timeout', '1800'], capsys)
    assert code == 0 and lines[0]['max_provider_attempts'] == 6
    assert lines[-1]['generated_count'] == 1 and lines[-1]['observed_provider_attempts'] == 1
    assert len(lines[-1]['checkpoint_identity']) == 64
    code, lines = invoke(args + ['--execute-live', '--generation-timeout', '1800'], capsys)
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
    code, lines = invoke(args + ['--execute-live', '--generation-timeout', '1800'], capsys)
    assert code == 1 and lines[-1]['status'] == 'FAILED'
    assert 'secret' not in str(lines)
    if failure in ('provider', 'fsync'):
        assert lines[-1]['error_code'] == ('CHECKPOINT_STORAGE_FAILURE' if failure == 'fsync' else 'UNKNOWN_FAILURE')
        assert lines[-1]['observed_provider_attempts'] is None
        assert claim_path(tmp_path / 'execution', session, packet.window_id).exists()
        code, _ = invoke(args + ['--execute-live', '--generation-timeout', '1800'], capsys)
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


def test_missing_live_timeout_stops_before_authority(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, '_load_authority_read_only', lambda _: pytest.fail('authority accessed'))
    args = ['--execution-root', str(tmp_path / 'execution'), '--checkpoint-root', str(tmp_path / 'checkpoints'),
            '--input-ceiling-tokens', '65536', '--execute-live']
    with pytest.raises(SystemExit) as error:
        cli.main(args)
    assert error.value.code == 2
    assert '--generation-timeout is required' in capsys.readouterr().err
    assert not (tmp_path / 'execution').exists() and not (tmp_path / 'checkpoints').exists()


@pytest.mark.parametrize('option', ['--preflight-timeout', '--generation-timeout'])
@pytest.mark.parametrize('value', ['invalid', '0', '-1', '3601', 'nan', 'inf'])
def test_invalid_timeouts_stop_before_authority(tmp_path, monkeypatch, option, value):
    monkeypatch.setattr(cli, '_load_authority_read_only', lambda _: pytest.fail('authority accessed'))
    args = ['--execution-root', str(tmp_path / 'execution'), '--checkpoint-root', str(tmp_path / 'checkpoints'),
            '--input-ceiling-tokens', '65536', option, value]
    with pytest.raises(SystemExit) as error:
        cli.main(args)
    assert error.value.code == 2
    assert not (tmp_path / 'execution').exists() and not (tmp_path / 'checkpoints').exists()


def test_timeouts_reach_separate_clients(setup_cli, capsys, monkeypatch):
    from types import SimpleNamespace
    from short_drama.llm import RuntimeConfig
    _, _, _, _, args = setup_cli
    seen = {}
    monkeypatch.setattr(cli, 'RuntimeConfig', RuntimeConfig)
    monkeypatch.setattr(cli.pf, 'LlamaClient', lambda **kw: seen.setdefault('preflight', kw))
    def adapter(runtime):
        seen['runtime'] = runtime
        return object()
    monkeypatch.setattr(cli, 'OpenAICompatibleLLMClient', adapter)
    monkeypatch.setattr(cli.bounded, 'run_b13_alice_bounded', lambda **kw: SimpleNamespace(
        status=runner.B13RunnerStatus.BOUNDED_PROGRESS, reused_windows=0, generated_windows=0,
        new_semantic_calls=0, new_provider_attempts=0))
    code, lines = invoke(args + ['--execute-live', '--preflight-timeout', '123',
                                 '--generation-timeout', '1800'], capsys)
    assert code == 0
    assert seen['preflight']['timeout_seconds'] == 123
    assert seen['runtime'].timeout_seconds == 1800
    assert lines[0]['preflight_timeout_seconds'] == 123
    assert lines[0]['generation_timeout_seconds'] == 1800
    assert lines[0]['max_provider_attempts'] == 6


def test_read_only_uses_default_preflight_timeout_and_no_adapter(setup_cli, capsys, monkeypatch):
    _, _, _, _, args = setup_cli
    seen = {}
    monkeypatch.setattr(cli.pf, 'LlamaClient', lambda **kw: seen.update(kw) or object())
    monkeypatch.setattr(cli, 'OpenAICompatibleLLMClient', lambda _: pytest.fail('adapter constructed'))
    code, _ = invoke(args, capsys)
    assert code == 0 and seen['timeout_seconds'] == 120


def test_generation_timeout_does_not_change_request_identity(setup_cli):
    from short_drama.adaptation import build_b13_window_semantic_preparation
    _, session, packet, _, _ = setup_cli
    original = build_b13_window_semantic_preparation(packet, runtime_config=session.runtime_config)
    long_timeout = build_b13_window_semantic_preparation(
        packet, runtime_config=replace(session.runtime_config, timeout_seconds=1800))
    assert original.request_identity == long_timeout.request_identity
    assert original.semantic_profile == long_timeout.semantic_profile
    assert original.request == long_timeout.request


@pytest.mark.parametrize(('error', 'expected'), [
    (cli.LLMTimeoutError('secret'), 'PROVIDER_TIMEOUT'),
    (cli.LLMTransportError('secret'), 'PROVIDER_TRANSPORT_FAILURE'),
    (cli.LLMHTTPError('secret', status=401, retryable=False), 'PROVIDER_HTTP_FAILURE'),
    (cli.LLMResponseError('secret'), 'STRUCTURED_OUTPUT_FAILURE'),
    (cli.LLMStructuredOutputError('secret'), 'STRUCTURED_OUTPUT_FAILURE'),
    (RuntimeError('secret'), 'UNKNOWN_FAILURE'),
])
def test_live_failure_code_from_fake_provider(setup_cli, capsys, monkeypatch, error, expected):
    _, _, _, fake, args = setup_cli
    monkeypatch.setattr(fake, 'generate_structured', lambda *a, **kw: (_ for _ in ()).throw(error))
    code, lines = invoke(args + ['--execute-live', '--generation-timeout', '1800'], capsys)
    assert code == 1 and lines[-1]['error_code'] == expected
    assert lines[-1]['semantic_rounds'] is None and lines[-1]['observed_provider_attempts'] is None
    assert 'semantic_rejection_reasons' not in lines[-1]
    assert 'secret' not in str(lines)


def test_retry_exhaustion_reports_only_final_technical_failure(setup_cli, capsys, monkeypatch):
    _, _, _, fake, args = setup_cli
    def fail(*a, **kw):
        try:
            raise cli.LLMStructuredOutputError('secret response')
        except cli.LLMStructuredOutputError as exc:
            raise cli.LLMRetryExhaustedError(attempts=3, message='secret') from exc
    monkeypatch.setattr(fake, 'generate_structured', fail)
    code, lines = invoke(args + ['--execute-live', '--generation-timeout', '1800'], capsys)
    assert code == 1 and lines[-1]['error_code'] == 'STRUCTURED_OUTPUT_FAILURE'
    assert lines[-1]['observed_provider_attempts'] is None
    assert 'secret' not in str(lines)


@pytest.mark.parametrize(('kind', 'expected'), [
    ('semantic', 'SEMANTIC_GENERATION_EXHAUSTED'),
    ('provenance', 'PROVENANCE_FAILURE'),
    ('schema', 'STRUCTURED_OUTPUT_FAILURE'),
])
def test_post_response_failure_codes(setup_cli, capsys, monkeypatch, kind, expected):
    _, session, packet, _, args = setup_cli
    invalid = _proposal(packet)
    invalid['event_dispositions'][0]['disposition'] = 'omit'
    responses = {'semantic': [(invalid, False), (invalid, False)],
                 'provenance': [(_proposal(packet), True)], 'schema': [({}, False)]}
    fake = _FakeClient(responses[kind], session.runtime_config)
    monkeypatch.setattr(cli, 'OpenAICompatibleLLMClient', lambda _: fake)
    code, lines = invoke(args + ['--execute-live', '--generation-timeout', '1800'], capsys)
    assert code == 1 and lines[-1]['error_code'] == expected
    assert len(fake.calls) == (2 if kind == 'semantic' else 1)
    assert lines[-1]['semantic_rounds'] is None and lines[-1]['observed_provider_attempts'] is None


def test_two_semantic_rejections_report_only_safe_reasons(setup_cli, capsys, monkeypatch):
    _, session, packet, _, args = setup_cli
    first, second = _proposal(packet), _proposal(packet)
    first['event_dispositions'][0]['disposition'] = 'omit'
    first['event_dispositions'][0]['rationale_zh'] = 'secret novel text'
    second['event_dispositions'] = []
    fake = _FakeClient([(first, False), (second, False)], session.runtime_config)
    monkeypatch.setattr(cli, 'OpenAICompatibleLLMClient', lambda _: fake)
    code, lines = invoke(args + ['--execute-live', '--generation-timeout', '1800'], capsys)
    assert code == 1 and len(fake.calls) == 2
    assert lines[-1]['error_code'] == 'SEMANTIC_GENERATION_EXHAUSTED'
    assert lines[-1]['semantic_rejection_reasons'] == [
        'omit owned event forbids active mapping candidates',
        'owned event dispositions must cover exactly every owned event',
    ]
    assert lines[-1]['semantic_rounds'] is None and lines[-1]['observed_provider_attempts'] is None
    assert 'secret' not in str(lines)


def test_missing_mapping_details_survive_cli_without_source_data(setup_cli, capsys, monkeypatch):
    _, session, packet, _, args = setup_cli
    first, second = _proposal(packet), _proposal(packet)
    event_id = first['event_dispositions'][0]['source_event_id']
    first['event_dispositions'][0]['rationale_zh'] = 'secret response text'
    first['adaptation_mappings'] = []
    second['adaptation_mappings'][0]['active'] = False
    fake = _FakeClient([(first, False), (second, False)], session.runtime_config)
    monkeypatch.setattr(cli, 'OpenAICompatibleLLMClient', lambda _: fake)
    code, lines = invoke(args + ['--execute-live', '--generation-timeout', '1800'], capsys)
    assert code == 1
    summary = lines[-1]
    assert summary['semantic_rejection_reasons'] == [cli._MISSING_MAPPING_REASON] * 2
    assert summary['semantic_rejection_details'] == [
        {'round': 1, 'missing_count': 1, 'events': [{'owned_ordinal': 1, 'mapping_state': 'NO_EVENT_MAPPING'}], 'truncated': False},
        {'round': 2, 'missing_count': 1, 'events': [{'owned_ordinal': 1, 'mapping_state': 'ONLY_INACTIVE_EVENT_MAPPINGS'}], 'truncated': False},
    ]
    assert event_id not in str(lines) and 'secret response text' not in str(lines)


def test_malformed_rejection_details_fail_closed():
    exc = cli.B13WindowGenerationExhaustedError(
        window_id='secret window', request_identity='secret id', rounds_attempted=2,
        failures=(cli._MISSING_MAPPING_REASON, cli._MISSING_MAPPING_REASON),
        rejection_details=(cli._MissingMappingDetail(1, ((1, 'secret source id'),), False), 'secret response'))
    details = cli._semantic_rejection_details(exc)
    assert details == [{'round': n, 'missing_count': None, 'events': [], 'status': 'UNKNOWN'} for n in (1, 2)]
    assert 'secret' not in str(details)


def test_untrusted_semantic_diagnostics_do_not_leak(setup_cli, capsys, monkeypatch):
    _, _, _, _, args = setup_cli
    def fail(**_):
        try:
            raise cli.B13WindowGenerationExhaustedError(
                window_id='secret window', request_identity='secret request', rounds_attempted=2,
                failures=('owned event dispositions must not be duplicated',
                          'owned event dispositions must not be duplicated: secret prompt and response'))
        except cli.B13WindowGenerationExhaustedError as exc:
            raise cli.B13RunnerError('secret wrapper') from exc
    monkeypatch.setattr(cli.bounded, 'run_b13_alice_bounded', fail)
    code, lines = invoke(args + ['--execute-live', '--generation-timeout', '1800'], capsys)
    assert code == 1 and lines[-1]['error_code'] == 'SEMANTIC_GENERATION_EXHAUSTED'
    assert lines[-1]['semantic_rejection_reasons'] == [
        'owned event dispositions must not be duplicated', 'UNKNOWN_SEMANTIC_REJECTION']
    assert 'secret' not in str(lines)
    assert lines[-1]['observed_provider_attempts'] is None


def test_malformed_semantic_failure_counts_fail_closed():
    exc = cli.B13WindowGenerationExhaustedError(
        window_id='synthetic', request_identity='synthetic', rounds_attempted=1, failures=('secret',))
    assert cli._semantic_rejection_reasons(exc) == ['UNKNOWN_SEMANTIC_REJECTION'] * 2
    assert cli._semantic_rejection_reasons(RuntimeError('secret')) is None


def test_authority_and_checkpoint_codes_do_not_unwrap_unrelated_causes():
    assert cli._failure_code(cli.B13WindowAuthorityError('secret'), 'live') == 'AUTHORITY_FAILURE'
    assert cli._failure_code(cli.B13CheckpointStorageError('secret'), 'live') == 'CHECKPOINT_STORAGE_FAILURE'
    try:
        try:
            raise cli.LLMTimeoutError('secret')
        except cli.LLMTimeoutError as exc:
            raise RuntimeError('secret') from exc
    except RuntimeError as exc:
        assert cli._failure_code(exc, 'live') == 'UNKNOWN_FAILURE'
