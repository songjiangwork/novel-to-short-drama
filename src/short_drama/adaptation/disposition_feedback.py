"""Unratified, offline-only B1.3 feedback request reconstruction.

No invocation or persistence is performed. Errors never include candidate data.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from short_drama.artifacts import canonical_json_bytes
from short_drama.llm import PromptRegistry, StructuredGenerationRequest, build_structured_request
from .disposition_execution import _missing_mapping_detail, _MISSING_MAPPING_REASON
from .disposition_semantic import (
    B13WindowProposal, B13WindowProposalError, B13WindowSemanticPreparation, B13_WINDOW_PROMPTS_DIR,
    build_b13_window_request, build_b13_window_semantic_preparation,
)

FEEDBACK_PROMPT_ID = 'b13.window-disposition-feedback'
FEEDBACK_PROMPT_VERSION = 1
FEEDBACK_PROMPT_HASH = '4522394c7618b88e7c6149a9971d70951f7a915d1ea7940b1dcf5c5b6cefcfa9'
FEEDBACK_DIAGNOSTIC_VERSION = 1
DEFAULT_CANDIDATE_LIMIT = 262_144


class B13FeedbackError(ValueError):
    """The offline feedback input cannot be verified."""


@dataclass(frozen=True, slots=True)
class FeedbackPreparation:
    request: StructuredGenerationRequest = field(repr=False)
    diagnostic_bytes: bytes = field(repr=False)


def build_historical_feedback_request(packet, candidate: dict, *,
                                      candidate_limit: int = DEFAULT_CANDIDATE_LIMIT) -> FeedbackPreparation:
    """Rebuild feedback from a caller-verified historical packet, without runtime state."""
    try:
        base_request = build_b13_window_request(packet)
        schema = base_request.output_schema
        if type(candidate_limit) is not int or not 0 < candidate_limit <= DEFAULT_CANDIDATE_LIMIT:
            raise ValueError
        raw = canonical_json_bytes(candidate)
        if len(raw) > candidate_limit:
            raise ValueError
        try:
            B13WindowProposal.from_dict(packet, candidate, schema)
        except B13WindowProposalError as exc:
            if str(exc) != _MISSING_MAPPING_REASON:
                raise ValueError from None
        else:
            raise ValueError
        detail = _missing_mapping_detail(packet, candidate, schema)
        if (detail is None or detail.missing_count is None or detail.truncated
                or detail.missing_count != len(detail.events) or not detail.events):
            raise ValueError
        owned = packet.to_dict()['window']['owned_event_refs']
        decisions = {d['source_event_id']: d['disposition'] for d in candidate['event_dispositions']}
        items = []
        for ordinal, state in detail.events:
            event_id = owned[ordinal - 1]
            beats = [beat for beat in candidate['adapted_story_beats']
                     if any(ref['kind'] == 'event' and ref['canonical_id'] == event_id
                            for ref in beat['source_event_refs'])]
            mappings = [item for item in candidate['adaptation_mappings']
                        if item['source_ref']['kind'] == 'event'
                        and item['source_ref']['canonical_id'] == event_id]
            items.append({'owned_ordinal': ordinal, 'source_event_ref': {'kind': 'event', 'canonical_id': event_id},
                          'disposition': decisions[event_id], 'related_beats': beats,
                          'existing_event_mappings': mappings, 'mapping_state': state})
        diagnostic = {'version': FEEDBACK_DIAGNOSTIC_VERSION, 'rule': _MISSING_MAPPING_REASON, 'events': items}
        diagnostic_bytes = canonical_json_bytes(diagnostic)
        template = PromptRegistry(B13_WINDOW_PROMPTS_DIR).load(FEEDBACK_PROMPT_ID, version=FEEDBACK_PROMPT_VERSION)
        if template.content_hash != FEEDBACK_PROMPT_HASH:
            raise ValueError
        base = base_request.rendered_prompt
        rendered = template.render({'base_system': base.system_text, 'base_user': base.user_text,
                                    'candidate_json': raw.decode('utf-8'),
                                    'feedback_json': diagnostic_bytes.decode('utf-8')})
        request = build_structured_request(rendered_prompt=rendered, output_schema=schema,
                                           semantic_profile=base_request.semantic_profile)
        if request.request_hash == base_request.request_hash:
            raise ValueError
        return FeedbackPreparation(request, diagnostic_bytes)
    except Exception:
        raise B13FeedbackError('feedback request cannot be verified') from None


def build_feedback_request(semantic: B13WindowSemanticPreparation, candidate: dict, *,
                           candidate_limit: int = DEFAULT_CANDIDATE_LIMIT) -> FeedbackPreparation:
    """Verify the supplied session preparation and reuse historical reconstruction."""
    try:
        if not isinstance(semantic, B13WindowSemanticPreparation):
            raise ValueError
        rebuilt = build_b13_window_semantic_preparation(
            semantic.packet, runtime_config=semantic.runtime_config,
            execution_options=semantic.execution_options)
        if semantic != rebuilt:
            raise ValueError
        return build_historical_feedback_request(semantic.packet, candidate, candidate_limit=candidate_limit)
    except Exception:
        raise B13FeedbackError('feedback request cannot be verified') from None
