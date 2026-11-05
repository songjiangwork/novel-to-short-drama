"""Synthetic-only opt-in through the actual B2B loop; never live authority."""
from dataclasses import replace
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts'))
import b13_alice_token_budget_preflight as pf
from short_drama.adaptation.disposition_checkpoint_v2 import (
    encode_feedback_checkpoint_v2, verify_and_decode_feedback_checkpoint_v2,
)
from short_drama.adaptation.disposition_execution import (
    B13DispositionExecutionResult, B13WindowGenerationExhaustedError,
    B13WindowProvenanceError, B13WindowSemanticGenerationPolicy,
    _execute_one_b13_window, _verify_client_runtime, _verify_preflight,
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


class SyntheticFeedback:
    """Opt-in callback limited to this test module; no production dispatch."""
    def __init__(self, session, tokenizer):
        self.session = session
        self.tokenizer = tokenizer
        self.first = None
        self.feedback = None
        self.report = None

    def prepare(self, semantic, result, options, budget):
        self.first = result
        self.feedback = build_feedback_request(semantic, result.parsed_json)
        body = _mapped_provider_request_bytes(self.feedback.request, semantic.runtime_config, options)
        measurement = replace(semantic.measurement, provider_request_bytes=len(body),
                              provider_request_estimated_tokens=estimate_tokens(body.decode('utf-8')))
        request = replace(semantic, request=self.feedback.request, execution_options=options,
                          provider_request_bytes=body, measurement=measurement)
        tokens, reason = pf.count_prompt_tokens(self.tokenizer, json.loads(body)['messages'])
        self.report = pf.assess_window(packet=semantic.packet, initial=request, regen=request,
            model_context_tokens=budget.model_context_tokens, input_ceiling_tokens=budget.input_ceiling_tokens,
            tokenizer_prompt_tokens=tokens, tokenizer_reason=reason)
        if self.report['verdict'] != 'PASS':
            raise B13WindowSemanticBudgetError('synthetic feedback budget did not pass')
        return request

    def accept(self, semantic, result, rejected, proposal):
        raw = encode_feedback_checkpoint_v2(session=self.session, window_id=semantic.packet.window_id,
            rejected_candidate=self.first.parsed_json, first_provenance=rejected.invocation_provenance,
            first_attempts=rejected.provider_attempts, accepted_proposal=proposal.to_dict(),
            second_provenance=result.provenance, second_attempts=result.attempts)
        return verify_and_decode_feedback_checkpoint_v2(raw, self.session)


def run_fake(tree, session, fake, adapter):
    prep, semantic, budget = session.preparation, session.semantic_preparations[0], session.budgets[0]
    _verify_preflight(store=tree.store, pointers=tree.pointers, preparation=prep,
        semantic_preparations=(semantic,), budgets=(budget,), runtime=session.runtime_config)
    _verify_client_runtime(fake, session.runtime_config)
    accepted = _execute_one_b13_window(store=tree.store, pointers=tree.pointers, preparation=prep,
        semantic=semantic, budget=budget, runtime_config=session.runtime_config, llm_client=fake,
        policy=B13WindowSemanticGenerationPolicy(), _synthetic_feedback=adapter)
    return B13DispositionExecutionResult(prep.source_snapshot, prep.strategy_selection, prep.preparation_identity, (accepted,))


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
    adapter = SyntheticFeedback(session, SyntheticTokenizer())
    execution = run_fake(tree, session, fake, adapter)
    accepted = execution.accepted_windows[0]
    assert len(fake.calls) == 2 and accepted.provider_attempts == accepted.semantic_rounds == 2
    assert fake.prompts[0] == semantic.request.rendered_prompt
    assert fake.prompts[0].prompt_version == 3
    assert fake.prompts[1] == adapter.feedback.request.rendered_prompt
    assert fake.prompts[0].rendered_prompt_hash != fake.prompts[1].rendered_prompt_hash
    assert fake.responses[0].provenance.request_hash != fake.responses[1].provenance.request_hash
    assert adapter.tokenizer.messages == json.loads(_mapped_provider_request_bytes(adapter.feedback.request, session.runtime_config, fake.calls[1]))['messages']
    assert accepted.feedback_evidence is not None
    inventory = build_b13_reconciliation_inventory(execution)
    review = review_b13_reconciliation(inventory, B13ReconciliationDecisionSet(inventory.inventory_identity, ()))
    assembly = assemble_b13_disposition_plan(review)
    publication = prepare_b13_disposition_publication(review, assembly, execution, historical_authority=authority(tree, session.preparation.source_snapshot))
    assert publication.to_dict()['schema_version'] == 2
    assert verify_b13_disposition_publication(publication) == assembly
    print('SYNTHETIC_REQUESTS', fake.responses[0].provenance.request_hash, fake.responses[1].provenance.request_hash)
    print('SYNTHETIC_RENDERED', fake.prompts[0].rendered_prompt_hash, fake.prompts[1].rendered_prompt_hash, 'CALLS', len(fake.calls))


def test_first_round_success_stays_v1(tmp_path):
    tree, session, packet, _ = _fixture(tmp_path)
    fake = RecordingFakeClient([(_proposal(packet), False)], session.runtime_config)
    accepted = run_fake(tree, session, fake, SyntheticFeedback(session, SyntheticTokenizer())).accepted_windows[0]
    assert len(fake.calls) == 1 and accepted.feedback_evidence is None and accepted.semantic_rounds == 1


@pytest.mark.parametrize('kind', ['unsupported', 'schema', 'budget', 'tokenizer', 'provenance', 'second_failure'])
def test_fail_closed_without_extra_dispatch(tmp_path, kind):
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
    adapter = SyntheticFeedback(session, tokenizer)
    if kind == 'budget':
        # The first request fits; force the feedback's actual mapped request over its ceiling.
        old_prepare = adapter.prepare
        def too_small(semantic, result, options, budget):
            return old_prepare(semantic, result, options, replace(budget, input_ceiling_tokens=1))
        adapter.prepare = too_small
    expected = {'unsupported': B13WindowGenerationExhaustedError, 'schema': B13WindowSchemaError,
                'budget': B13WindowSemanticBudgetError, 'tokenizer': B13WindowSemanticBudgetError,
                'provenance': B13WindowProvenanceError, 'second_failure': B13WindowGenerationExhaustedError}[kind]
    with pytest.raises(expected):
        run_fake(tree, session, fake, adapter)
    assert len(fake.calls) == (2 if kind in ('provenance', 'second_failure') else 1)
