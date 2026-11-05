"""Synthetic-only opt-in through the actual B2B loop; never live authority."""
from dataclasses import replace
import json

import pytest

from short_drama.adaptation.disposition_execution import (
    B13WindowGenerationExhaustedError, B13WindowProvenanceError,
    _execute_b13_feedback_for_test,
)
from short_drama.adaptation.disposition_feedback import build_feedback_request
from short_drama.adaptation.disposition_semantic import (
    B13WindowSchemaError, B13WindowSemanticBudgetError, _mapped_provider_request_bytes,
)
from short_drama.adaptation.disposition_assembly import assemble_b13_disposition_plan
from short_drama.adaptation.disposition_publication import prepare_b13_disposition_publication, verify_b13_disposition_publication
from short_drama.adaptation.disposition_reconciliation import B13ReconciliationDecisionSet, build_b13_reconciliation_inventory, review_b13_reconciliation
from short_drama.story.chunking import estimate_tokens
from short_drama.story.story_analysis import StoryAnalysisManifest
from test_adaptation_strategy_execution import _FakeClient
from test_b13_disposition_checkpoint import _fixture
from test_b13_window_semantic import _proposal


class RecordingFakeClient(_FakeClient):
    def __init__(self, results, runtime):
        super().__init__(results, runtime)
        self.prompts = []
        self.responses = []

    def generate_structured(self, prompt, *args, **kwargs):
        self.prompts.append(prompt)
        result = super().generate_structured(prompt, *args, **kwargs)
        self.responses.append(result)
        return result


class SyntheticTokenizer:
    """Test double only: never evidence of any real model/tokenizer capacity."""
    def __init__(self, count=10):
        self.count = count
        self.messages = None

    def apply_template(self, messages, **_kwargs):
        self.messages = messages
        return 'synthetic rendered prompt'

    def tokenize(self, _prompt):
        return [1] * self.count


def run_fake(tree, session, fake, tokenizer):
    return _execute_b13_feedback_for_test(store=tree.store, pointers=tree.pointers,
        session=session, llm_client=fake, tokenizer=tokenizer)


def authority(tree, source):
    manifest = StoryAnalysisManifest.from_dict(tree.store.get_ref(source.story_analysis_manifest_ref).payload)
    refs = {'a5_manifest': source.consolidation_manifest_ref, 'a6_manifest': source.story_analysis_manifest_ref,
        'entity_map': source.entity_map_ref, 'characters': source.canonical_character_registry_ref,
        'locations': source.canonical_location_registry_ref, 'facts': source.canonical_fact_set_ref,
        'events': source.canonical_event_set_ref, 'relationships': source.canonical_relationship_set_ref,
        'arc_analysis': source.arc_analysis_ref, 'event_analysis': manifest.global_event_analysis_ref}
    return {name: tree.store.get_ref(ref).to_dict() for name, ref in refs.items()}


def test_real_execution_loop_feedback_to_b4a(tmp_path):
    tree, session, packet, semantic = _fixture(tmp_path)
    bad = _proposal(packet)
    bad['adaptation_mappings'] = []
    fake = RecordingFakeClient([(bad, False), (_proposal(packet), False)], session.runtime_config)
    tokenizer = SyntheticTokenizer()
    execution = run_fake(tree, session, fake, tokenizer)
    accepted = execution.accepted_windows[0]
    assert len(fake.calls) == 2 and accepted.provider_attempts == accepted.semantic_rounds == 2
    assert fake.prompts[0] == semantic.request.rendered_prompt
    assert fake.prompts[0].prompt_version == 3
    feedback = build_feedback_request(semantic, bad)
    assert fake.prompts[1] == feedback.request.rendered_prompt
    assert fake.prompts[0].rendered_prompt_hash != fake.prompts[1].rendered_prompt_hash
    assert fake.responses[0].provenance.request_hash != fake.responses[1].provenance.request_hash
    assert accepted.request_identity == semantic.request_identity
    assert accepted.semantic_request_hash == fake.responses[0].provenance.request_hash
    assert accepted.invocation_provenance == fake.responses[1].provenance
    assert tokenizer.messages == json.loads(_mapped_provider_request_bytes(feedback.request, session.runtime_config, fake.calls[1]))['messages']
    assert accepted.feedback_evidence is not None
    from short_drama.adaptation.disposition_checkpoint_v2 import verify_and_decode_feedback_checkpoint_v2
    assert verify_and_decode_feedback_checkpoint_v2(accepted.feedback_evidence.canonical_bytes(), session) == accepted
    inventory = build_b13_reconciliation_inventory(execution)
    review = review_b13_reconciliation(inventory, B13ReconciliationDecisionSet(inventory.inventory_identity, ()))
    assembly = assemble_b13_disposition_plan(review)
    publication = prepare_b13_disposition_publication(review, assembly, execution, historical_authority=authority(tree, session.preparation.source_snapshot))
    assert publication.to_dict()['schema_version'] == 2
    assert verify_b13_disposition_publication(publication) == assembly
    print('SYNTHETIC_REQUESTS', fake.responses[0].provenance.request_hash, fake.responses[1].provenance.request_hash)
    print('SYNTHETIC_RENDERED', fake.prompts[0].rendered_prompt_hash, fake.prompts[1].rendered_prompt_hash, 'CALLS', len(fake.calls))


def test_first_round_success_stays_v1(tmp_path):
    from short_drama.adaptation.disposition_checkpoint import encode_checkpoint, verify_and_decode_checkpoint
    tree, session, packet, _ = _fixture(tmp_path)
    fake = RecordingFakeClient([(_proposal(packet), False)], session.runtime_config)
    accepted = run_fake(tree, session, fake, SyntheticTokenizer()).accepted_windows[0]
    assert len(fake.calls) == 1 and accepted.feedback_evidence is None and accepted.semantic_rounds == 1
    assert verify_and_decode_checkpoint(encode_checkpoint(accepted, session), session) == accepted


def test_missing_context_evidence_stops_before_first_call(tmp_path):
    from short_drama.adaptation.disposition_checkpoint import B13CheckpointEvidenceError
    tree, session, packet, _ = _fixture(tmp_path)
    fake = RecordingFakeClient([(_proposal(packet), False)], session.runtime_config)
    with pytest.raises(B13CheckpointEvidenceError):
        run_fake(tree, replace(session, effective_n_ctx=None), fake, SyntheticTokenizer())
    assert fake.calls == []


@pytest.mark.parametrize('kind', ['unsupported', 'schema', 'budget', 'tokenizer', 'authority', 'provenance', 'second_failure'])
def test_fail_closed_without_extra_dispatch(tmp_path, kind, monkeypatch):
    tree, session, packet, _ = _fixture(tmp_path)
    bad = _proposal(packet)
    bad['adaptation_mappings'] = []
    if kind == 'unsupported':
        bad = _proposal(packet)
        bad['event_dispositions'][0]['disposition'] = 'omit'
    if kind == 'schema':
        bad = {}
    fake = RecordingFakeClient([(bad, False), ((_proposal(packet) if kind != 'second_failure' else bad), kind == 'provenance')], session.runtime_config)
    tokenizer = SyntheticTokenizer(2_000_000 if kind == 'tokenizer' else 10)
    if kind == 'authority':
        import short_drama.adaptation.disposition_execution as execution
        original = execution._revalidate_authority
        after_first = 0
        def fail_before_second(*args):
            nonlocal after_first
            if len(fake.calls) == 1:
                after_first += 1
                if after_first == 2:
                    raise execution.B13WindowAuthorityError('synthetic authority changed')
            return original(*args)
        monkeypatch.setattr(execution, '_revalidate_authority', fail_before_second)
    if kind == 'budget':
        from short_drama.llm import GenerationExecutionOptions
        semantic = session.semantic_preparations[0]
        opts = GenerationExecutionOptions(prompt_context_reuse='disabled')
        first = estimate_tokens(_mapped_provider_request_bytes(semantic.request, session.runtime_config, opts).decode())
        feedback = build_feedback_request(semantic, bad)
        second = estimate_tokens(_mapped_provider_request_bytes(feedback.request, session.runtime_config, opts).decode())
        ceiling = max(first, semantic.measurement.provider_request_estimated_tokens)
        assert second > ceiling
        budget = replace(session.budgets[0], input_ceiling_tokens=ceiling)
        session = replace(session, budgets=(budget,), input_ceiling_tokens=ceiling)
    from short_drama.adaptation.disposition_execution import B13WindowAuthorityError
    expected = {'unsupported': B13WindowGenerationExhaustedError, 'schema': B13WindowSchemaError,
                'budget': B13WindowSemanticBudgetError, 'tokenizer': B13WindowSemanticBudgetError,
                'authority': B13WindowAuthorityError,
                'provenance': B13WindowProvenanceError, 'second_failure': B13WindowGenerationExhaustedError}[kind]
    with pytest.raises(expected):
        run_fake(tree, session, fake, tokenizer)
    assert len(fake.calls) == (2 if kind in ('provenance', 'second_failure') else 1)
