"""PROPOSED offline B1.3 feedback evidence; never dispatch authority.

Historical verification assumes its packet and preparation identity have been
independently authenticated by the caller. It proves no live authority,
tokenizer freshness, or Provider/model authenticity.
"""
from __future__ import annotations

from dataclasses import dataclass
import re

from short_drama.artifacts import canonical_json_bytes, content_hash, strict_json_loads
from short_drama.llm import LLMInvocationProvenance
from .disposition_checkpoint import _trusted_session, _locate_window, _provenance
from .disposition_execution import B13WindowAcceptedProposal, B13WindowRejectedSemanticRound, _MISSING_MAPPING_REASON
from .disposition_feedback import build_historical_feedback_request, DEFAULT_CANDIDATE_LIMIT
from .disposition_feedback_evidence import _from_verified_bytes
from .disposition_preparation import B13DispositionWindowPacket
from .disposition_semantic import (
    B13WindowProposal, b13_window_request_identity, build_b13_window_request,
    build_b13_window_semantic_preparation,
)

DEFAULT_EVIDENCE_LIMIT = 1_000_000
_ARTIFACT = 'b13_window_feedback_checkpoint_evidence'
_HASH = re.compile(r'^[0-9a-f]{64}$')


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


def _verify_request_provenance(provenance, request):
    """Check request-bound fields, without claiming to authenticate the backend."""
    if not isinstance(provenance, LLMInvocationProvenance):
        raise ValueError
    prompt, profile, schema = request.rendered_prompt, request.semantic_profile, request.output_schema
    checks = (
        (provenance.semantic_profile_id, profile.profile_id),
        (provenance.semantic_profile_hash, profile.semantic_profile_hash),
        (provenance.prompt_id, prompt.prompt_id),
        (provenance.prompt_version, prompt.prompt_version),
        (provenance.prompt_content_hash, prompt.prompt_content_hash),
        (provenance.rendered_prompt_hash, prompt.rendered_prompt_hash),
        (provenance.output_schema_id, schema.schema_id),
        (provenance.output_schema_version, schema.schema_version),
        (provenance.output_schema_hash, schema.schema_hash),
        (provenance.request_hash, request.request_hash),
    )
    if any(actual != expected for actual, expected in checks):
        raise ValueError


def _build_historical(packet, preparation_identity, candidate, first_provenance, first_attempts,
                      proposal, second_provenance, second_attempts, limits):
    if (not isinstance(limits, FeedbackEvidenceLimits) or not isinstance(packet, B13DispositionWindowPacket)
            or type(preparation_identity) is not str or _HASH.fullmatch(preparation_identity) is None):
        raise ValueError
    if any(type(n) is not int or not 1 <= n <= 3 for n in (first_attempts, second_attempts)):
        raise ValueError
    base_request = build_b13_window_request(packet)
    _verify_request_provenance(first_provenance, base_request)
    feedback = build_historical_feedback_request(packet, candidate, candidate_limit=limits.candidate_bytes)
    _verify_request_provenance(second_provenance, feedback.request)
    if (first_provenance.provider_family != second_provenance.provider_family
            or first_provenance.model != second_provenance.model):
        raise ValueError
    final = B13WindowProposal.from_dict(packet, proposal, base_request.output_schema)
    profile, schema = base_request.semantic_profile, base_request.output_schema
    base, rendered = base_request.rendered_prompt, feedback.request.rendered_prompt
    base_identity = b13_window_request_identity(packet=packet, profile=profile, schema=schema, request=base_request)
    material = {
        'schema_version': 2, 'artifact_type': _ARTIFACT,
        'preparation_identity': preparation_identity,
        'source_snapshot_hash': content_hash(packet.source_snapshot.to_dict()),
        'strategy_ref': packet.strategy_selection.strategy_ref.to_dict(),
        'strategy_hash': packet.strategy_selection.strategy_hash,
        'window': {'window_id': packet.window_id, 'window_ordinal': packet.window_ordinal, 'packet_hash': packet.content_hash()},
        'base_request_identity': base_identity,
        'round1': {'prompt_id': base.prompt_id, 'prompt_version': base.prompt_version,
                   'prompt_content_hash': base.prompt_content_hash, 'rendered_prompt_hash': base.rendered_prompt_hash,
                   'request_hash': base_request.request_hash, 'provenance': first_provenance.to_dict(),
                   'provider_attempts': first_attempts, 'candidate': candidate,
                   'candidate_hash': content_hash(candidate), 'diagnostic': strict_json_loads(feedback.diagnostic_bytes)},
        'round2': {'prompt_id': rendered.prompt_id, 'prompt_version': rendered.prompt_version,
                   'prompt_content_hash': rendered.prompt_content_hash, 'rendered_prompt_hash': rendered.rendered_prompt_hash,
                   'request_hash': feedback.request.request_hash, 'provenance': second_provenance.to_dict(),
                   'provider_attempts': second_attempts},
        'semantic_profile_identity': {'profile_id': profile.profile_id, 'profile_hash': profile.semantic_profile_hash},
        'output_schema_identity': {'schema_id': schema.schema_id, 'schema_version': schema.schema_version, 'schema_hash': schema.schema_hash},
        'semantic_rounds': 2, 'provider_attempts': first_attempts + second_attempts,
        'proposal': final.to_dict(), 'proposal_hash': content_hash(final.to_dict()),
    }
    return {**material, 'checkpoint_identity': content_hash(material)}


def _build(session, window_id, candidate, first_provenance, first_attempts,
           proposal, second_provenance, second_attempts, limits):
    prep, packet, semantic, runtime = _rebuild(session, window_id)
    payload = _build_historical(packet, prep.preparation_identity, candidate, first_provenance,
                                first_attempts, proposal, second_provenance, second_attempts, limits)
    if (payload['base_request_identity'] != semantic.request_identity
            or first_provenance.provider_family != runtime.provider_family
            or first_provenance.model != runtime.request_model
            or second_provenance.provider_family != runtime.provider_family
            or second_provenance.model != runtime.request_model):
        raise ValueError
    return payload


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


def verify_historical_feedback_checkpoint_v2(raw, *, packet, preparation_identity,
                                             limits=FeedbackEvidenceLimits()):
    """Verify a self-contained record using a caller-verified packet and preparation.

    The preparation identity must be derived by the caller from independently
    verified historical authority and the complete historical packet set. This
    function checks consistency only; it does not itself authenticate authority,
    establish live preflight, or prove Provider identity or response origin.
    The unkeyed checkpoint identity is not an external authentication anchor;
    callers must bind it to independently trusted immutable publication evidence.
    """
    try:
        if not isinstance(limits, FeedbackEvidenceLimits) or type(raw) is not bytes or not 0 < len(raw) <= limits.evidence_bytes:
            raise ValueError
        payload = strict_json_loads(raw)
        if type(payload) is not dict or canonical_json_bytes(payload) != raw:
            raise ValueError
        if type(payload.get('schema_version')) is not int or payload['schema_version'] != 2 or payload.get('artifact_type') != _ARTIFACT:
            raise ValueError
        first, second = payload['round1'], payload['round2']
        expected = _build_historical(packet, preparation_identity, first['candidate'], _provenance(first['provenance']),
                                     first['provider_attempts'], payload['proposal'], _provenance(second['provenance']),
                                     second['provider_attempts'], limits)
        if canonical_json_bytes(payload) != canonical_json_bytes(expected):
            raise ValueError
        base = build_b13_window_request(packet)
        proposal = B13WindowProposal.from_dict(packet, payload['proposal'], base.output_schema)
        rejected = B13WindowRejectedSemanticRound(1, first['provider_attempts'], _provenance(first['provenance']), _MISSING_MAPPING_REASON)
        prompt, profile, schema = base.rendered_prompt, base.semantic_profile, base.output_schema
        return B13WindowAcceptedProposal(
            packet.window_id, packet.window_ordinal, payload['base_request_identity'], base.request_hash,
            {'profile_id': profile.profile_id, 'profile_hash': profile.semantic_profile_hash},
            {'prompt_id': prompt.prompt_id, 'prompt_version': prompt.prompt_version, 'prompt_content_hash': prompt.prompt_content_hash},
            {'schema_id': schema.schema_id, 'schema_version': schema.schema_version, 'schema_hash': schema.schema_hash},
            prompt.rendered_prompt_hash, _provenance(second['provenance']), 2, payload['provider_attempts'],
            second['provider_attempts'], (_MISSING_MAPPING_REASON,), (rejected,), proposal,
            _from_verified_bytes(raw))
    except Exception:
        raise B13FeedbackCheckpointError('feedback checkpoint verification failed') from None


def verify_and_decode_feedback_checkpoint_v2(raw, session, *, limits=FeedbackEvidenceLimits()):
    """Apply session checks and the same historical validation, without live I/O."""
    try:
        if type(raw) is not bytes or not isinstance(limits, FeedbackEvidenceLimits) or not 0 < len(raw) <= limits.evidence_bytes:
            raise ValueError
        payload = strict_json_loads(raw)
        prep, packet, semantic, runtime = _rebuild(session, payload['window']['window_id'])
        accepted = verify_historical_feedback_checkpoint_v2(
            raw, packet=packet, preparation_identity=prep.preparation_identity, limits=limits)
        first, second = accepted.rejected_semantic_rounds[0].invocation_provenance, accepted.invocation_provenance
        if (accepted.request_identity != semantic.request_identity
                or first.provider_family != runtime.provider_family or first.model != runtime.request_model
                or second.provider_family != runtime.provider_family or second.model != runtime.request_model):
            raise ValueError
        return accepted
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
