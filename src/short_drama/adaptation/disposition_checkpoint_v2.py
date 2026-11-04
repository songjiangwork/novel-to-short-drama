"""PROPOSED, offline-only B1.3 feedback evidence codec; not dispatch authority.

This codec performs no storage, provider calls, live preflight or publication.
Its caller supplies a structurally trusted session and must separately verify
live authority when appropriate. Limits and this wire contract are not ratified.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

from short_drama.artifacts import canonical_json_bytes, content_hash, strict_json_loads
from short_drama.llm import LLMInvocationProvenance
from .disposition_checkpoint import _trusted_session, _locate_window, _provenance, _verify_provenance
from .disposition_execution import B13WindowAcceptedProposal, B13WindowRejectedSemanticRound, _MISSING_MAPPING_REASON
from .disposition_feedback import build_feedback_request, DEFAULT_CANDIDATE_LIMIT
from .disposition_feedback_evidence import _from_verified_bytes
from .disposition_semantic import build_b13_window_semantic_preparation

DEFAULT_EVIDENCE_LIMIT = 1_000_000
_ARTIFACT = 'b13_window_feedback_checkpoint_evidence'


class B13FeedbackCheckpointError(ValueError):
    """Offline evidence is invalid; never exposes candidate or source content."""


@dataclass(frozen=True, slots=True)
class FeedbackEvidenceLimits:
    candidate_bytes: int = DEFAULT_CANDIDATE_LIMIT
    evidence_bytes: int = DEFAULT_EVIDENCE_LIMIT

    def __post_init__(self):
        if (type(self.candidate_bytes) is not int or not 0 < self.candidate_bytes <= DEFAULT_CANDIDATE_LIMIT
                or type(self.evidence_bytes) is not int or not 0 < self.evidence_bytes <= DEFAULT_EVIDENCE_LIMIT):
            raise B13FeedbackCheckpointError('invalid offline evidence limits')


def _rebuild(session, window_id):
    prep, semantics, runtime = _trusted_session(session)
    packet, semantic = _locate_window(prep, semantics, window_id)
    rebuilt = build_b13_window_semantic_preparation(
        packet, runtime_config=runtime, execution_options=semantic.execution_options,
    )
    if semantic != rebuilt:
        raise ValueError
    return prep, packet, rebuilt, runtime


def _build(session, window_id, candidate, first_provenance, first_attempts,
           proposal, second_provenance, second_attempts, limits):
    if not isinstance(limits, FeedbackEvidenceLimits):
        raise ValueError
    prep, packet, semantic, runtime = _rebuild(session, window_id)
    if any(type(n) is not int or not 1 <= n <= 3 for n in (first_attempts, second_attempts)):
        raise ValueError
    if not isinstance(first_provenance, LLMInvocationProvenance) or not isinstance(second_provenance, LLMInvocationProvenance):
        raise ValueError
    _verify_provenance(first_provenance, semantic=semantic, runtime=runtime, label='first')
    feedback = build_feedback_request(semantic, candidate, candidate_limit=limits.candidate_bytes)
    round2 = SimpleNamespace(request=feedback.request, semantic_profile=semantic.semantic_profile,
                             output_schema=semantic.output_schema)
    _verify_provenance(second_provenance, semantic=round2, runtime=runtime, label='second')
    final = semantic.decode_and_validate_proposal(proposal)
    base = semantic.request.rendered_prompt
    rendered = feedback.request.rendered_prompt
    material = {
        'schema_version': 2, 'artifact_type': _ARTIFACT,
        'preparation_identity': prep.preparation_identity,
        'source_snapshot_hash': content_hash(prep.source_snapshot.to_dict()),
        'strategy_ref': prep.strategy_selection.strategy_ref.to_dict(),
        'strategy_hash': prep.strategy_selection.strategy_hash,
        'window': {'window_id': packet.window_id, 'window_ordinal': packet.window_ordinal, 'packet_hash': packet.content_hash()},
        'base_request_identity': semantic.request_identity,
        'round1': {'prompt_id': base.prompt_id, 'prompt_version': base.prompt_version,
                   'prompt_content_hash': base.prompt_content_hash, 'rendered_prompt_hash': base.rendered_prompt_hash,
                   'request_hash': semantic.request.request_hash, 'provenance': first_provenance.to_dict(),
                   'provider_attempts': first_attempts, 'candidate': candidate,
                   'candidate_hash': content_hash(candidate), 'diagnostic': strict_json_loads(feedback.diagnostic_bytes)},
        'round2': {'prompt_id': rendered.prompt_id, 'prompt_version': rendered.prompt_version,
                   'prompt_content_hash': rendered.prompt_content_hash, 'rendered_prompt_hash': rendered.rendered_prompt_hash,
                   'request_hash': feedback.request.request_hash, 'provenance': second_provenance.to_dict(),
                   'provider_attempts': second_attempts},
        'semantic_profile_identity': {'profile_id': semantic.semantic_profile.profile_id, 'profile_hash': semantic.semantic_profile.semantic_profile_hash},
        'output_schema_identity': {'schema_id': semantic.output_schema.schema_id, 'schema_version': semantic.output_schema.schema_version,
                                   'schema_hash': semantic.output_schema.schema_hash},
        'semantic_rounds': 2, 'provider_attempts': first_attempts + second_attempts,
        'proposal': final.to_dict(), 'proposal_hash': content_hash(final.to_dict()),
    }
    return {**material, 'checkpoint_identity': content_hash(material)}


def encode_feedback_checkpoint_v2(*, session, window_id, rejected_candidate, first_provenance,
                                  first_attempts, accepted_proposal, second_provenance,
                                  second_attempts, limits=FeedbackEvidenceLimits()):
    """Encode and independently verify one synthetic/in-memory result; no I/O."""
    try:
        payload = _build(session, window_id, rejected_candidate, first_provenance, first_attempts,
                         accepted_proposal, second_provenance, second_attempts, limits)
        raw = canonical_json_bytes(payload)
        if len(raw) > limits.evidence_bytes:
            raise ValueError
        verify_and_decode_feedback_checkpoint_v2(raw, session, limits=limits)
        return raw
    except Exception:
        raise B13FeedbackCheckpointError('feedback checkpoint cannot be encoded') from None


def verify_and_decode_feedback_checkpoint_v2(raw, session, *, limits=FeedbackEvidenceLimits()):
    """Verify and reconstruct both requests and both semantic results in memory."""
    try:
        if not isinstance(limits, FeedbackEvidenceLimits) or type(raw) is not bytes or not 0 < len(raw) <= limits.evidence_bytes:
            raise ValueError
        payload = strict_json_loads(raw)
        if type(payload) is not dict or canonical_json_bytes(payload) != raw:
            raise ValueError
        if type(payload.get('schema_version')) is not int or payload['schema_version'] != 2 or payload.get('artifact_type') != _ARTIFACT:
            raise ValueError
        first, second = payload['round1'], payload['round2']
        expected = _build(session, payload['window']['window_id'], first['candidate'], _provenance(first['provenance']),
                          first['provider_attempts'], payload['proposal'], _provenance(second['provenance']),
                          second['provider_attempts'], limits)
        if canonical_json_bytes(payload) != canonical_json_bytes(expected):
            raise ValueError
        prep, packet, semantic, _runtime = _rebuild(session, payload['window']['window_id'])
        proposal = semantic.decode_and_validate_proposal(payload['proposal'])
        rejected = B13WindowRejectedSemanticRound(1, first['provider_attempts'], _provenance(first['provenance']), _MISSING_MAPPING_REASON)
        prompt = semantic.request.rendered_prompt
        return B13WindowAcceptedProposal(
            packet.window_id, packet.window_ordinal, semantic.request_identity, semantic.request.request_hash,
            {'profile_id': semantic.semantic_profile.profile_id, 'profile_hash': semantic.semantic_profile.semantic_profile_hash},
            {'prompt_id': prompt.prompt_id, 'prompt_version': prompt.prompt_version, 'prompt_content_hash': prompt.prompt_content_hash},
            {'schema_id': semantic.output_schema.schema_id, 'schema_version': semantic.output_schema.schema_version, 'schema_hash': semantic.output_schema.schema_hash},
            prompt.rendered_prompt_hash, _provenance(second['provenance']), 2, payload['provider_attempts'],
            second['provider_attempts'], (_MISSING_MAPPING_REASON,), (rejected,), proposal,
            _from_verified_bytes(raw))
    except Exception:
        raise B13FeedbackCheckpointError('feedback checkpoint verification failed') from None


def encode_verified_feedback_checkpoint_v2(accepted, session, *, limits=FeedbackEvidenceLimits()):
    """Revalidate an opaque carrier before returning its original bytes, with no recursion."""
    try:
        if not isinstance(accepted, B13WindowAcceptedProposal) or accepted.feedback_evidence is None:
            raise ValueError
        raw = accepted.feedback_evidence.canonical_bytes()
        if verify_and_decode_feedback_checkpoint_v2(raw, session, limits=limits) != accepted:
            raise ValueError
        return raw
    except Exception:
        raise B13FeedbackCheckpointError('feedback checkpoint cannot be encoded') from None
