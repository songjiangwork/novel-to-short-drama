"""Opaque carrier for offline-verified feedback checkpoint evidence.

This is not an authorization or a cryptographic Provider attestation.
"""
from dataclasses import dataclass, field
from short_drama.artifacts import canonical_json_bytes, content_hash, strict_json_loads

_TOKEN = object()


@dataclass(frozen=True, slots=True, init=False)
class VerifiedFeedbackEvidenceV2:
    """Immutable canonical bytes; created only after the v2 verifier succeeds."""
    _canonical_bytes: bytes = field(repr=False)

    def __init__(self, raw: bytes, *, _token=None):
        if _token is not _TOKEN or type(raw) is not bytes or not raw:
            raise ValueError('invalid feedback evidence carrier')
        object.__setattr__(self, '_canonical_bytes', raw)

    def canonical_bytes(self) -> bytes:
        return self._canonical_bytes

    def matches(self, accepted) -> bool:
        """Check the evidence-to-window binding without serializing the carrier."""
        try:
            p = strict_json_loads(self._canonical_bytes)
            first, second = p['round1'], p['round2']
            material = {k: v for k, v in p.items() if k != 'checkpoint_identity'}
            packet = accepted.proposal.packet
            return (
                canonical_json_bytes(p) == self._canonical_bytes
                and p['schema_version'] == 2 and p['artifact_type'] == 'b13_window_feedback_checkpoint_evidence'
                and p['checkpoint_identity'] == content_hash(material)
                and p['base_request_identity'] == accepted.request_identity
                and p['window'] == {'window_id': accepted.window_id, 'window_ordinal': accepted.window_ordinal,
                                    'packet_hash': packet.content_hash()}
                and p['source_snapshot_hash'] == content_hash(packet.source_snapshot.to_dict())
                and p['strategy_ref'] == packet.strategy_selection.strategy_ref.to_dict()
                and p['strategy_hash'] == packet.strategy_selection.strategy_hash
                and p['proposal'] == accepted.proposal.to_dict()
                and p['proposal_hash'] == content_hash(p['proposal'])
                and p['semantic_profile_identity'] == accepted.semantic_profile_identity
                and p['output_schema_identity'] == accepted.output_schema_identity
                and accepted.prompt_identity == {'prompt_id': first['prompt_id'],
                                                  'prompt_version': first['prompt_version'],
                                                  'prompt_content_hash': first['prompt_content_hash']}
                and first['rendered_prompt_hash'] == accepted.rendered_prompt_hash
                and first['request_hash'] == accepted.semantic_request_hash
                and second['provenance'] == accepted.invocation_provenance.to_dict()
                and second['request_hash'] == accepted.invocation_provenance.request_hash
                and p['semantic_rounds'] == accepted.semantic_rounds == 2
                and p['provider_attempts'] == accepted.provider_attempts
                and second['provider_attempts'] == accepted.accepted_round_provider_attempts
                and len(accepted.rejected_semantic_rounds) == 1
                and first['provenance'] == accepted.rejected_semantic_rounds[0].invocation_provenance.to_dict()
                and first['provider_attempts'] == accepted.rejected_semantic_rounds[0].provider_attempts
                and first['diagnostic']['rule'] == accepted.rejected_semantic_rounds[0].diagnostic
            )
        except Exception:
            return False


def _from_verified_bytes(raw: bytes) -> VerifiedFeedbackEvidenceV2:
    """Internal construction seam: caller must have run the complete verifier."""
    return VerifiedFeedbackEvidenceV2(raw, _token=_TOKEN)
