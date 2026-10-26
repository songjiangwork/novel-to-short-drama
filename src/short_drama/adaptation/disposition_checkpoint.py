"""B1.3 single-window checkpoint evidence codec and verifier (zero-provider):
serializes one accepted B1.3 window into a bounded, deterministic, canonical
checkpoint and verifies it back into a typed ``B13WindowAcceptedProposal``.
Rebuilds the packet, semantic request, and pinned assets from a trusted session
and revalidates the proposal and provenance.  No I/O, generation, publication,
or CURRENT resolution/advance."""
from __future__ import annotations

from typing import Any

from short_drama.artifacts import canonical_json_bytes, content_hash, strict_json_loads
from short_drama.llm import LLMInvocationProvenance, RuntimeConfig

from .disposition_execution import B13WindowAcceptedProposal, B13WindowRejectedSemanticRound
from .disposition_preparation import B13DispositionPreparation
from .disposition_semantic import (
    B13WindowProposal, B13WindowSemanticPreparation, build_b13_window_semantic_preparation)


B13_CHECKPOINT_EVIDENCE_SCHEMA_VERSION = 1
B13_CHECKPOINT_EVIDENCE_ARTIFACT_TYPE = "b13_window_checkpoint_evidence"
B13_CHECKPOINT_MAX_BYTES = 1_000_000

_TOP_FIELDS = {
    "schema_version", "artifact_type", "checkpoint_identity",
    "preparation_identity", "source_snapshot_hash", "strategy_ref", "strategy_hash",
    "window", "request_identity", "semantic_request_hash",
    "semantic_profile_identity", "prompt_identity", "output_schema_identity",
    "rendered_prompt_hash", "invocation_provenance",
    "semantic_rounds", "provider_attempts", "accepted_round_provider_attempts",
    "semantic_failures", "rejected_semantic_rounds", "proposal",
}
_WINDOW_FIELDS = {"window_id", "window_ordinal", "packet_hash"}
_PROVENANCE_FIELDS = {
    "provider_family", "model", "semantic_profile_id", "semantic_profile_hash",
    "prompt_id", "prompt_version", "prompt_content_hash", "rendered_prompt_hash",
    "output_schema_id", "output_schema_version", "output_schema_hash", "request_hash",
    "provider_response_id", "finish_reason", "usage",
}
_REJECTED_FIELDS = {"semantic_round", "provider_attempts", "diagnostic", "invocation_provenance"}


class B13CheckpointEvidenceError(ValueError):
    """A B1.3 single-window checkpoint is malformed or mismatches its session."""


def _session_parts(session: Any):
    preparation = getattr(session, "preparation", None)
    semantic_preparations = getattr(session, "semantic_preparations", None)
    runtime_config = getattr(session, "runtime_config", None)
    if not isinstance(preparation, B13DispositionPreparation):
        raise B13CheckpointEvidenceError("checkpoint requires a trusted B1.3 execution session")
    if (not isinstance(semantic_preparations, (tuple, list)) or not semantic_preparations
            or any(not isinstance(item, B13WindowSemanticPreparation) for item in semantic_preparations)):
        raise B13CheckpointEvidenceError("checkpoint requires one semantic preparation per window")
    if not isinstance(runtime_config, RuntimeConfig):
        raise B13CheckpointEvidenceError("checkpoint requires an exact RuntimeConfig")
    return preparation, tuple(semantic_preparations), runtime_config


def _provenance(value: Any) -> LLMInvocationProvenance:
    if not isinstance(value, dict) or set(value) != _PROVENANCE_FIELDS:
        raise B13CheckpointEvidenceError("invocation provenance has an invalid shape")
    try:
        return LLMInvocationProvenance(**value)
    except Exception as exc:
        raise B13CheckpointEvidenceError("invocation provenance is invalid") from exc


def _verify_provenance(provenance, *, semantic: B13WindowSemanticPreparation, runtime: RuntimeConfig, label: str) -> None:
    request = semantic.request
    checks = (
        (provenance.semantic_profile_id, semantic.semantic_profile.profile_id),
        (provenance.semantic_profile_hash, semantic.semantic_profile.semantic_profile_hash),
        (provenance.prompt_id, request.rendered_prompt.prompt_id),
        (provenance.prompt_version, request.rendered_prompt.prompt_version),
        (provenance.prompt_content_hash, request.rendered_prompt.prompt_content_hash),
        (provenance.rendered_prompt_hash, request.rendered_prompt.rendered_prompt_hash),
        (provenance.output_schema_id, semantic.output_schema.schema_id),
        (provenance.output_schema_version, semantic.output_schema.schema_version),
        (provenance.output_schema_hash, semantic.output_schema.schema_hash),
        (provenance.request_hash, request.request_hash),
        (provenance.provider_family, runtime.provider_family),
        (provenance.model, runtime.request_model),
    )
    if any(got != expected for got, expected in checks):
        raise B13CheckpointEvidenceError(
            f"{label} invocation provenance does not match the exact reconstructed request"
        )


def _locate_window(preparation: B13DispositionPreparation, semantic_preparations, window_id: Any):
    if not isinstance(window_id, str):
        raise B13CheckpointEvidenceError("checkpoint window identity is invalid")
    packets = [packet for packet in preparation.packets if packet.window_id == window_id]
    if len(packets) != 1:
        raise B13CheckpointEvidenceError("checkpoint does not match exactly one session window")
    semantics = [item for item in semantic_preparations if item.packet.window_id == window_id]
    if len(semantics) != 1 or semantics[0].packet is not packets[0]:
        raise B13CheckpointEvidenceError("checkpoint window has no matching semantic preparation")
    return packets[0], semantics[0]


def _rejected_rounds(raw: Any, rebuilt: B13WindowSemanticPreparation, runtime: RuntimeConfig):
    if not isinstance(raw, list):
        raise B13CheckpointEvidenceError("rejected semantic rounds must be an array")
    rounds = []
    for item in raw:
        if not isinstance(item, dict) or set(item) != _REJECTED_FIELDS:
            raise B13CheckpointEvidenceError("rejected semantic round has an invalid shape")
        provenance = _provenance(item["invocation_provenance"])
        _verify_provenance(provenance, semantic=rebuilt, runtime=runtime, label="rejected semantic round")
        rounds.append(B13WindowRejectedSemanticRound(
            item["semantic_round"], item["provider_attempts"], provenance, item["diagnostic"]))
    return tuple(rounds)


def _accepted_to_payload(accepted: B13WindowAcceptedProposal, preparation: B13DispositionPreparation, packet) -> dict:
    selection = preparation.strategy_selection
    payload = {
        "schema_version": B13_CHECKPOINT_EVIDENCE_SCHEMA_VERSION,
        "artifact_type": B13_CHECKPOINT_EVIDENCE_ARTIFACT_TYPE,
        "preparation_identity": preparation.preparation_identity,
        "source_snapshot_hash": content_hash(preparation.source_snapshot.to_dict()),
        "strategy_ref": selection.strategy_ref.to_dict(),
        "strategy_hash": selection.strategy_hash,
        "window": {"window_id": packet.window_id, "window_ordinal": packet.window_ordinal,
                   "packet_hash": packet.content_hash()},
        "request_identity": accepted.request_identity,
        "semantic_request_hash": accepted.semantic_request_hash,
        "semantic_profile_identity": accepted.semantic_profile_identity,
        "prompt_identity": accepted.prompt_identity,
        "output_schema_identity": accepted.output_schema_identity,
        "rendered_prompt_hash": accepted.rendered_prompt_hash,
        "invocation_provenance": accepted.invocation_provenance.to_dict(),
        "semantic_rounds": accepted.semantic_rounds,
        "provider_attempts": accepted.provider_attempts,
        "accepted_round_provider_attempts": accepted.accepted_round_provider_attempts,
        "semantic_failures": list(accepted.semantic_failures),
        "rejected_semantic_rounds": [
            {"semantic_round": item.semantic_round, "provider_attempts": item.provider_attempts,
             "diagnostic": item.diagnostic, "invocation_provenance": item.invocation_provenance.to_dict()}
            for item in accepted.rejected_semantic_rounds],
        "proposal": accepted.proposal.to_dict(),
    }
    payload["checkpoint_identity"] = content_hash(
        {key: value for key, value in payload.items() if key != "checkpoint_identity"})
    return payload


def encode_checkpoint(accepted: B13WindowAcceptedProposal, session: Any) -> bytes:
    """Canonicalize one accepted B1.3 window bound to its trusted session."""
    preparation, semantic_preparations, _runtime = _session_parts(session)
    packet, semantic = _locate_window(preparation, semantic_preparations, accepted.window_id)
    if (packet.window_ordinal != accepted.window_ordinal
            or accepted.proposal.packet.content_hash() != packet.content_hash()
            or accepted.request_identity != semantic.request_identity
            or accepted.semantic_request_hash != semantic.request.request_hash):
        raise B13CheckpointEvidenceError("accepted window does not match its trusted session window")
    return canonical_json_bytes(_accepted_to_payload(accepted, preparation, packet))


def verify_and_decode_checkpoint(raw: bytes, session: Any) -> B13WindowAcceptedProposal:
    """Reconstruct a typed accepted proposal only if the checkpoint is valid."""
    if not isinstance(raw, (bytes, bytearray)):
        raise B13CheckpointEvidenceError("checkpoint payload must be bytes")
    raw = bytes(raw)
    if not raw or len(raw) > B13_CHECKPOINT_MAX_BYTES:
        raise B13CheckpointEvidenceError("checkpoint payload exceeds the bounded size")
    preparation, semantic_preparations, runtime = _session_parts(session)
    try:
        payload = strict_json_loads(raw)
    except Exception as exc:
        raise B13CheckpointEvidenceError("checkpoint payload is not strict canonical JSON") from exc
    if not isinstance(payload, dict) or set(payload) != _TOP_FIELDS:
        raise B13CheckpointEvidenceError("checkpoint payload has an invalid shape")
    if (payload["schema_version"] != B13_CHECKPOINT_EVIDENCE_SCHEMA_VERSION
            or payload["artifact_type"] != B13_CHECKPOINT_EVIDENCE_ARTIFACT_TYPE):
        raise B13CheckpointEvidenceError("checkpoint schema version or artifact type is invalid")
    material = {key: value for key, value in payload.items() if key != "checkpoint_identity"}
    if payload["checkpoint_identity"] != content_hash(material):
        raise B13CheckpointEvidenceError("checkpoint identity does not bind its retained evidence")
    if (payload["preparation_identity"] != preparation.preparation_identity
            or payload["source_snapshot_hash"] != content_hash(preparation.source_snapshot.to_dict())):
        raise B13CheckpointEvidenceError("checkpoint preparation/source authority mismatch")
    selection = preparation.strategy_selection
    if (payload["strategy_ref"] != selection.strategy_ref.to_dict()
            or payload["strategy_hash"] != selection.strategy_hash):
        raise B13CheckpointEvidenceError("checkpoint B1.2 strategy binding mismatch")
    window = payload["window"]
    if not isinstance(window, dict) or set(window) != _WINDOW_FIELDS:
        raise B13CheckpointEvidenceError("checkpoint window binding has an invalid shape")
    packet, session_semantic = _locate_window(preparation, semantic_preparations, window["window_id"])
    if packet.window_ordinal != window["window_ordinal"] or window["packet_hash"] != packet.content_hash():
        raise B13CheckpointEvidenceError("checkpoint window identity mismatch")
    try:
        rebuilt = build_b13_window_semantic_preparation(packet, runtime_config=runtime)
    except Exception as exc:
        raise B13CheckpointEvidenceError("cannot rebuild the window semantic request") from exc
    if (rebuilt.request_identity != session_semantic.request_identity
            or rebuilt.request.request_hash != session_semantic.request.request_hash):
        raise B13CheckpointEvidenceError("trusted session semantic preparation is inconsistent")
    if (payload["request_identity"] != rebuilt.request_identity
            or payload["semantic_request_hash"] != rebuilt.request.request_hash
            or payload["rendered_prompt_hash"] != rebuilt.request.rendered_prompt.rendered_prompt_hash):
        raise B13CheckpointEvidenceError("checkpoint semantic request identity does not match the rebuilt request")
    expected = {
        "semantic_profile_identity": {"profile_id": rebuilt.semantic_profile.profile_id,
                                       "profile_hash": rebuilt.semantic_profile.semantic_profile_hash},
        "prompt_identity": {"prompt_id": rebuilt.request.rendered_prompt.prompt_id,
                            "prompt_version": rebuilt.request.rendered_prompt.prompt_version,
                            "prompt_content_hash": rebuilt.request.rendered_prompt.prompt_content_hash},
        "output_schema_identity": {"schema_id": rebuilt.output_schema.schema_id,
                                   "schema_version": rebuilt.output_schema.schema_version,
                                   "schema_hash": rebuilt.output_schema.schema_hash},
    }
    if any(payload[key] != expected[key] for key in expected):
        raise B13CheckpointEvidenceError("checkpoint profile/prompt/schema identity mismatch")
    try:
        proposal = B13WindowProposal.from_dict(packet, payload["proposal"], rebuilt.output_schema)
        provenance = _provenance(payload["invocation_provenance"])
        _verify_provenance(provenance, semantic=rebuilt, runtime=runtime, label="accepted")
        rejected = _rejected_rounds(payload["rejected_semantic_rounds"], rebuilt, runtime)
        return B13WindowAcceptedProposal(
            window["window_id"], window["window_ordinal"], payload["request_identity"],
            payload["semantic_request_hash"], payload["semantic_profile_identity"],
            payload["prompt_identity"], payload["output_schema_identity"], payload["rendered_prompt_hash"],
            provenance, payload["semantic_rounds"], payload["provider_attempts"],
            payload["accepted_round_provider_attempts"], tuple(payload["semantic_failures"]),
            rejected, proposal,
        )
    except B13CheckpointEvidenceError:
        raise
    except Exception as exc:
        raise B13CheckpointEvidenceError("checkpoint proposal or accounting is invalid") from exc


__all__ = [
    "B13_CHECKPOINT_EVIDENCE_SCHEMA_VERSION", "B13_CHECKPOINT_EVIDENCE_ARTIFACT_TYPE",
    "B13_CHECKPOINT_MAX_BYTES", "B13CheckpointEvidenceError",
    "encode_checkpoint", "verify_and_decode_checkpoint",
]
