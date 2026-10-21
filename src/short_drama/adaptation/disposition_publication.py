"""B1.3-B4A immutable publication evidence, with no persistence side effects.

The contract intentionally retains the B2B material needed to recreate the
reconciliation input.  B4B may store this payload, but this module neither
writes an artifact nor resolves a CURRENT pointer.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from short_drama.artifacts import (
    ArtifactRef, artifact_content_hash, canonical_json_bytes, content_hash,
    strict_json_loads,
)
from short_drama.foundation import LineageRef
from short_drama.llm import LLMInvocationProvenance, PromptRegistry, build_structured_request

from .disposition_assembly import (
    B13AssemblyWindowAudit, B13DispositionAssemblyResult,
    B13GlobalIdentityBinding, assemble_b13_disposition_plan,
)
from .disposition_execution import (
    B13DispositionExecutionResult, B13WindowAcceptedProposal,
    B13WindowRejectedSemanticRound,
)
from .disposition_preparation import B12StrategySelection, B13DispositionWindowPacket
from .disposition_reconciliation import (
    B13CandidateRef, B13ReconciliationDecision, B13ReconciliationDecisionSet,
    B13ReconciliationReview,
    build_b13_reconciliation_inventory, review_b13_reconciliation,
)
from .disposition_semantic import (
    B13_WINDOW_PROMPT_ID, B13_WINDOW_PROMPT_VERSION, B13_WINDOW_PROMPTS_DIR, B13WindowProposal,
    b13_window_request_identity, load_b13_window_semantic_assets,
)
from .strategy_execution import B12_STRATEGY_ARTIFACT_SCHEMA_VERSION, B12StrategyArtifact
from .target import AdaptationSourceSnapshot


B13_DISPOSITION_PUBLICATION_SCHEMA_VERSION = 1
B13_DISPOSITION_PUBLICATION_ARTIFACT_TYPE = "b13_disposition_publication"


class B13DispositionPublicationError(ValueError):
    """Publication evidence is malformed, incomplete, or cannot be rederived."""


def b13_disposition_publication_artifact_id(source: AdaptationSourceSnapshot, strategy: B12StrategyArtifact) -> str:
    """Return the stable B1.3 stage-local identity (not a CURRENT pointer)."""
    if not isinstance(source, AdaptationSourceSnapshot) or not isinstance(strategy, B12StrategyArtifact):
        raise B13DispositionPublicationError("publication artifact identity requires source and B1.2 strategy")
    return "{}.{}.b13.disposition".format(
        source.story_analysis_manifest_ref.artifact_id,
        strategy.strategy.resolved_target.adaptation_target_id,
    )


def _hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise B13DispositionPublicationError(f"{label} must be a SHA-256 hash")
    return value


def _provenance(value: Any) -> LLMInvocationProvenance:
    fields = {
        "provider_family", "model", "semantic_profile_id", "semantic_profile_hash",
        "prompt_id", "prompt_version", "prompt_content_hash", "rendered_prompt_hash",
        "output_schema_id", "output_schema_version", "output_schema_hash", "request_hash",
        "provider_response_id", "finish_reason", "usage",
    }
    if not isinstance(value, dict) or set(value) != fields:
        raise B13DispositionPublicationError("invocation provenance has an invalid shape")
    try:
        return LLMInvocationProvenance(**value)
    except Exception as exc:
        raise B13DispositionPublicationError("invocation provenance is invalid") from exc


def _verify_reconstructed_provenance(
    provenance: LLMInvocationProvenance, *, request: Any, semantic_profile: Any,
    output_schema: Any, label: str,
) -> None:
    """Apply B2B's request-bound provenance checks to retained invocation audit.

    B4A intentionally does not retain a runtime endpoint or credentials, so it
    cannot attest which backend actually served an invocation.  The declared
    routing fields are retained as audit data and compared between semantic
    rounds below; every content-bound field is instead checked against the
    independently rebuilt request and pinned B2A assets.
    """
    checks = (
        (provenance.semantic_profile_id, semantic_profile.profile_id),
        (provenance.semantic_profile_hash, semantic_profile.semantic_profile_hash),
        (provenance.prompt_id, request.rendered_prompt.prompt_id),
        (provenance.prompt_version, request.rendered_prompt.prompt_version),
        (provenance.prompt_content_hash, request.rendered_prompt.prompt_content_hash),
        (provenance.rendered_prompt_hash, request.rendered_prompt.rendered_prompt_hash),
        (provenance.output_schema_id, output_schema.schema_id),
        (provenance.output_schema_version, output_schema.schema_version),
        (provenance.output_schema_hash, output_schema.schema_hash),
        (provenance.request_hash, request.request_hash),
    )
    if any(actual != expected for actual, expected in checks):
        raise B13DispositionPublicationError(
            f"{label} invocation provenance does not match the exact reconstructed request"
        )


def _accepted_to_dict(value: B13WindowAcceptedProposal) -> dict[str, Any]:
    return {
        "packet": value.proposal.packet.to_dict(),
        "request_identity": value.request_identity,
        "semantic_request_hash": value.semantic_request_hash,
        "semantic_profile_identity": value.semantic_profile_identity,
        "prompt_identity": value.prompt_identity,
        "output_schema_identity": value.output_schema_identity,
        "rendered_prompt_hash": value.rendered_prompt_hash,
        "invocation_provenance": value.invocation_provenance.to_dict(),
        "semantic_rounds": value.semantic_rounds,
        "provider_attempts": value.provider_attempts,
        "accepted_round_provider_attempts": value.accepted_round_provider_attempts,
        "semantic_failures": list(value.semantic_failures),
        "rejected_semantic_rounds": [
            {"semantic_round": item.semantic_round, "provider_attempts": item.provider_attempts,
             "diagnostic": item.diagnostic, "invocation_provenance": item.invocation_provenance.to_dict()}
            for item in value.rejected_semantic_rounds
        ],
        "proposal": value.proposal.to_dict(),
    }


def _decision_from_dict(value: Any) -> B13ReconciliationDecision:
    fields = {"kind", "candidate_refs", "finding_ids", "rationale_zh", "provenance_zh"}
    if not isinstance(value, dict) or set(value) != fields or not isinstance(value["candidate_refs"], list) or not isinstance(value["finding_ids"], list):
        raise B13DispositionPublicationError("reconciliation decision has an invalid shape")
    try:
        return B13ReconciliationDecision(
            value["kind"], tuple(B13CandidateRef(**item) for item in value["candidate_refs"]),
            tuple(value["finding_ids"]), value["rationale_zh"], value["provenance_zh"],
        )
    except Exception as exc:
        raise B13DispositionPublicationError("reconciliation decision is invalid") from exc


def _bindings_to_dict(bindings: tuple[B13GlobalIdentityBinding, ...]) -> list[dict[str, Any]]:
    return [
        {"candidate_ref": item.candidate_ref.to_dict(), "identity_key": item.identity_key,
         "adapted_id": item.adapted_id}
        for item in bindings
    ]


def _audit_to_dict(audit: tuple[B13AssemblyWindowAudit, ...]) -> list[dict[str, Any]]:
    return [
        {"window_id": item.window_id, "proposal_hash": item.proposal_hash,
         "request_identity": item.request_identity,
         "candidate_refs": [ref.to_dict() for ref in item.candidate_refs]}
        for item in audit
    ]


def _assembly_to_dict(value: B13DispositionAssemblyResult) -> dict[str, Any]:
    return {
        "plan": value.plan.to_dict(), "inventory_identity": value.inventory_identity,
        "decision_identity": value.decision_identity, "strategy_ref": value.strategy_ref,
        "strategy_hash": value.strategy_hash, "preparation_identity": value.preparation_identity,
        "identity_bindings": _bindings_to_dict(value.identity_bindings),
        "window_audit": _audit_to_dict(value.window_audit),
    }


def _lineage(source: AdaptationSourceSnapshot, selection: B12StrategySelection) -> list[dict[str, Any]]:
    source_roles = (
        ("a6_story_analysis_manifest", source.story_analysis_manifest_ref),
        ("a5_consolidation_manifest", source.consolidation_manifest_ref),
        ("a5_entity_map", source.entity_map_ref),
        ("a5_canonical_character_registry", source.canonical_character_registry_ref),
        ("a5_canonical_location_registry", source.canonical_location_registry_ref),
        ("a5_canonical_fact_set", source.canonical_fact_set_ref),
        ("a5_canonical_event_set", source.canonical_event_set_ref),
        ("a5_canonical_relationship_set", source.canonical_relationship_set_ref),
        ("a6_arc_analysis", source.arc_analysis_ref),
        ("b12_strategy", selection.strategy_ref),
    )
    values = [LineageRef(role, ref) for role, ref in source_roles]
    if selection.current_pointer_ref is not None:
        values.append(LineageRef("b12_current_pointer_observed", selection.current_pointer_ref))
    return [item.to_dict() for item in sorted(values, key=LineageRef.sort_key)]


def _semantic_identity(source: AdaptationSourceSnapshot, selection: B12StrategySelection,
                       execution: B13DispositionExecutionResult, decisions: B13ReconciliationDecisionSet) -> str:
    return content_hash({
        "stage": "B1.3-B4A", "schema_version": B13_DISPOSITION_PUBLICATION_SCHEMA_VERSION,
        "source_snapshot": source.to_dict(), "strategy_ref": selection.strategy_ref.to_dict(),
        "strategy_artifact_hash": selection.strategy_ref.content_hash,
        "strategy_hash": selection.strategy_hash, "preparation_identity": execution.preparation_identity,
        "accepted_windows": [
            {"window_id": item.window_id, "window_ordinal": item.window_ordinal,
             "packet_hash": item.proposal.packet.content_hash(), "request_identity": item.request_identity,
             "semantic_request_hash": item.semantic_request_hash,
             "semantic_profile_identity": item.semantic_profile_identity,
             "prompt_identity": item.prompt_identity, "output_schema_identity": item.output_schema_identity,
             "rendered_prompt_hash": item.rendered_prompt_hash, "proposal": item.proposal.to_dict()}
            for item in execution.accepted_windows
        ],
        "decisions": [item.to_dict() for item in decisions.decisions],
    })


def _publication_identity(value: dict[str, Any]) -> str:
    material = dict(value)
    material.pop("publication_identity", None)
    return content_hash(material)


@dataclass(frozen=True, slots=True)
class B13DispositionPublication:
    """Canonical versioned payload intended for B4B immutable persistence."""

    _canonical_payload: bytes

    def __post_init__(self) -> None:
        if not isinstance(self._canonical_payload, bytes):
            raise B13DispositionPublicationError("publication payload must be canonical bytes")
        try:
            payload = strict_json_loads(self._canonical_payload)
        except Exception as exc:
            raise B13DispositionPublicationError("publication payload is invalid JSON") from exc
        if not isinstance(payload, dict) or canonical_json_bytes(payload) != self._canonical_payload:
            raise B13DispositionPublicationError("publication payload must be canonical JSON")

    def to_dict(self) -> dict[str, Any]:
        value = strict_json_loads(self._canonical_payload)
        assert isinstance(value, dict)
        return value

    def canonical_bytes(self) -> bytes:
        return self._canonical_payload

    @property
    def publication_identity(self) -> str:
        return self.to_dict()["publication_identity"]

    @property
    def semantic_identity(self) -> str:
        return self.to_dict()["semantic_identity"]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "B13DispositionPublication":
        payload = cls(canonical_json_bytes(value))
        verify_b13_disposition_publication(payload)
        return payload


def _reconstruct_execution(payload: dict[str, Any]) -> tuple[B13DispositionExecutionResult, B13ReconciliationDecisionSet]:
    evidence = payload["evidence"]
    try:
        source = AdaptationSourceSnapshot.from_dict(evidence["source_snapshot"])
        strategy_ref = ArtifactRef.from_dict(evidence["strategy_selection"]["strategy_ref"])
        strategy = B12StrategyArtifact.from_dict(evidence["strategy_selection"]["strategy_artifact"])
        pointer_data = evidence["strategy_selection"]["current_pointer_ref"]
        pointer = None if pointer_data is None else ArtifactRef.from_dict(pointer_data)
    except Exception as exc:
        raise B13DispositionPublicationError("retained source or B1.2 strategy evidence is invalid") from exc
    if strategy_ref.content_hash != artifact_content_hash(
        schema_version=B12_STRATEGY_ARTIFACT_SCHEMA_VERSION, payload=strategy.to_dict(),
    ):
        raise B13DispositionPublicationError("retained B1.2 strategy artifact does not match its immutable ref")
    try:
        selection = B12StrategySelection(strategy_ref, strategy, pointer)
    except Exception as exc:
        raise B13DispositionPublicationError("retained B1.2 selection is invalid") from exc
    if source != strategy.source_snapshot:
        raise B13DispositionPublicationError("retained A5/A6 snapshot disagrees with B1.2 strategy")
    assets = load_b13_window_semantic_assets()
    prompts = PromptRegistry(B13_WINDOW_PROMPTS_DIR)
    accepted: list[B13WindowAcceptedProposal] = []
    raw_windows = evidence["accepted_windows"]
    if not isinstance(raw_windows, list):
        raise B13DispositionPublicationError("accepted window evidence must be an array")
    for raw in raw_windows:
        fields = {
            "packet", "request_identity", "semantic_request_hash", "semantic_profile_identity",
            "prompt_identity", "output_schema_identity", "rendered_prompt_hash", "invocation_provenance",
            "semantic_rounds", "provider_attempts", "accepted_round_provider_attempts", "semantic_failures",
            "rejected_semantic_rounds", "proposal",
        }
        if not isinstance(raw, dict) or set(raw) != fields:
            raise B13DispositionPublicationError("accepted window evidence has an invalid shape")
        packet_data = raw["packet"]
        try:
            window = packet_data["window"]
            packet = B13DispositionWindowPacket(
                source, selection, window["window_id"], window["window_ordinal"], canonical_json_bytes(packet_data),
            )
            proposal = B13WindowProposal.from_dict(packet, raw["proposal"], assets.output_schema)
            rendered = prompts.load(B13_WINDOW_PROMPT_ID, version=B13_WINDOW_PROMPT_VERSION).render(
                {"window_packet_json": packet.canonical_bytes().decode("utf-8")}
            )
            request = build_structured_request(rendered_prompt=rendered, output_schema=assets.output_schema,
                                               semantic_profile=assets.semantic_profile)
            provenance = _provenance(raw["invocation_provenance"])
            rejected = tuple(B13WindowRejectedSemanticRound(
                item["semantic_round"], item["provider_attempts"], _provenance(item["invocation_provenance"]),
                item["diagnostic"],
            ) for item in raw["rejected_semantic_rounds"])
            item = B13WindowAcceptedProposal(
                packet.window_id, packet.window_ordinal, raw["request_identity"], raw["semantic_request_hash"],
                raw["semantic_profile_identity"], raw["prompt_identity"], raw["output_schema_identity"],
                raw["rendered_prompt_hash"], provenance, raw["semantic_rounds"], raw["provider_attempts"],
                raw["accepted_round_provider_attempts"], tuple(raw["semantic_failures"]), rejected, proposal,
            )
        except Exception as exc:
            raise B13DispositionPublicationError("accepted window evidence is invalid") from exc
        if (item.semantic_request_hash != request.request_hash
                or item.rendered_prompt_hash != rendered.rendered_prompt_hash
                or item.request_identity != b13_window_request_identity(
                    packet=packet, profile=assets.semantic_profile, schema=assets.output_schema, request=request,
                )):
            raise B13DispositionPublicationError("accepted window request identity cannot be independently rederived")
        if item.semantic_profile_identity != {"profile_id": assets.semantic_profile.profile_id, "profile_hash": assets.semantic_profile.semantic_profile_hash}:
            raise B13DispositionPublicationError("accepted window semantic profile identity is invalid")
        if item.output_schema_identity != {"schema_id": assets.output_schema.schema_id, "schema_version": assets.output_schema.schema_version, "schema_hash": assets.output_schema.schema_hash}:
            raise B13DispositionPublicationError("accepted window output schema identity is invalid")
        _verify_reconstructed_provenance(
            item.invocation_provenance, request=request,
            semantic_profile=assets.semantic_profile, output_schema=assets.output_schema,
            label="accepted",
        )
        for rejected_round in item.rejected_semantic_rounds:
            _verify_reconstructed_provenance(
                rejected_round.invocation_provenance, request=request,
                semantic_profile=assets.semantic_profile, output_schema=assets.output_schema,
                label="rejected semantic round",
            )
            if (
                rejected_round.invocation_provenance.provider_family != item.invocation_provenance.provider_family
                or rejected_round.invocation_provenance.model != item.invocation_provenance.model
            ):
                raise B13DispositionPublicationError(
                    "rejected semantic round declared routing differs from its accepted invocation"
                )
        accepted.append(item)
    try:
        execution = B13DispositionExecutionResult(source, selection, evidence["preparation_identity"], tuple(accepted))
        decisions_raw = evidence["reconciliation"]["decisions"]
        decisions = B13ReconciliationDecisionSet(decisions_raw["inventory_identity"], tuple(_decision_from_dict(x) for x in decisions_raw["decisions"]))
    except Exception as exc:
        raise B13DispositionPublicationError("retained execution or decision evidence is invalid") from exc
    return execution, decisions


def verify_b13_disposition_publication(value: B13DispositionPublication | dict[str, Any]) -> B13DispositionAssemblyResult:
    """Independently reconstruct every B2B/B3A/B3B result from retained evidence."""
    payload = value.to_dict() if isinstance(value, B13DispositionPublication) else value
    fields = {"schema_version", "artifact_type", "stage_artifact_id", "semantic_identity", "publication_identity", "lineage", "evidence"}
    if not isinstance(payload, dict) or set(payload) != fields:
        raise B13DispositionPublicationError("publication payload has an invalid shape")
    if payload["schema_version"] != B13_DISPOSITION_PUBLICATION_SCHEMA_VERSION:
        raise B13DispositionPublicationError("unsupported B1.3 publication schema version")
    if payload["artifact_type"] != B13_DISPOSITION_PUBLICATION_ARTIFACT_TYPE:
        raise B13DispositionPublicationError("publication artifact type is invalid")
    _hash(payload["semantic_identity"], "semantic_identity")
    _hash(payload["publication_identity"], "publication_identity")
    if payload["publication_identity"] != _publication_identity(payload):
        raise B13DispositionPublicationError("publication identity does not bind exact retained evidence")
    execution, decisions = _reconstruct_execution(payload)
    inventory = build_b13_reconciliation_inventory(execution)
    evidence = payload["evidence"]
    if evidence.get("reconciliation", {}).get("inventory") != inventory.to_dict():
        raise B13DispositionPublicationError("retained reconciliation inventory cannot be independently rederived")
    if decisions.inventory_identity != inventory.inventory_identity:
        raise B13DispositionPublicationError("reconciliation decisions have stale inventory identity")
    review = review_b13_reconciliation(inventory, decisions)
    try:
        review.require_resolved()
    except Exception as exc:
        raise B13DispositionPublicationError("publication reconciliation is not resolved") from exc
    assembled = assemble_b13_disposition_plan(review)
    if evidence.get("assembly") != _assembly_to_dict(assembled):
        raise B13DispositionPublicationError("retained final plan or scoped-to-global assembly evidence is forged")
    source = execution.source_snapshot
    selection = execution.strategy_selection
    if payload["stage_artifact_id"] != b13_disposition_publication_artifact_id(source, selection.artifact):
        raise B13DispositionPublicationError("publication stage artifact identity is invalid")
    if payload["lineage"] != _lineage(source, selection):
        raise B13DispositionPublicationError("publication immutable lineage is incomplete or inconsistent")
    if payload["semantic_identity"] != _semantic_identity(source, selection, execution, decisions):
        raise B13DispositionPublicationError("publication semantic identity is inconsistent")
    return assembled


def prepare_b13_disposition_publication(
    review: B13ReconciliationReview, assembly: B13DispositionAssemblyResult,
    execution: B13DispositionExecutionResult,
) -> B13DispositionPublication:
    """Prepare, validate, and serialize B4A evidence without providers or writes."""
    if (not isinstance(review, B13ReconciliationReview)
            or not isinstance(assembly, B13DispositionAssemblyResult)
            or not isinstance(execution, B13DispositionExecutionResult)):
        raise B13DispositionPublicationError("publication requires exact B3A review, B3B assembly, and B2B execution evidence")
    # Both operations revalidate the full review and call require_resolved.
    expected = assemble_b13_disposition_plan(review)
    if assembly != expected:
        raise B13DispositionPublicationError("supplied B3B assembly disagrees with an independently reassembled plan")
    inventory = build_b13_reconciliation_inventory(execution)
    if review.inventory != inventory:
        raise B13DispositionPublicationError("B3A review does not bind the supplied complete B2B execution evidence")
    if (assembly.strategy_ref != inventory.strategy_ref or assembly.strategy_hash != inventory.strategy_hash
            or assembly.preparation_identity != inventory.preparation_identity):
        raise B13DispositionPublicationError("assembly B1.2 or preparation identity is inconsistent")
    if execution.source_snapshot != review.inventory.source_snapshot:
        raise B13DispositionPublicationError("execution source snapshot disagrees with reconciliation review")
    semantic_identity = _semantic_identity(execution.source_snapshot, execution.strategy_selection, execution, review.decisions)
    evidence = {
        "source_snapshot": execution.source_snapshot.to_dict(),
        "strategy_selection": {
            "strategy_ref": execution.strategy_selection.strategy_ref.to_dict(),
            "current_pointer_ref": None if execution.strategy_selection.current_pointer_ref is None else execution.strategy_selection.current_pointer_ref.to_dict(),
            "strategy_artifact": execution.strategy_selection.artifact.to_dict(),
        },
        "preparation_identity": execution.preparation_identity,
        "accepted_windows": [_accepted_to_dict(item) for item in execution.accepted_windows],
        "reconciliation": {
            "inventory": inventory.to_dict(),
            "decisions": {"inventory_identity": review.decisions.inventory_identity,
                          "decisions": [item.to_dict() for item in review.decisions.decisions]},
        },
        "assembly": _assembly_to_dict(assembly),
    }
    payload = {
        "schema_version": B13_DISPOSITION_PUBLICATION_SCHEMA_VERSION,
        "artifact_type": B13_DISPOSITION_PUBLICATION_ARTIFACT_TYPE,
        "stage_artifact_id": b13_disposition_publication_artifact_id(execution.source_snapshot, execution.strategy_selection.artifact),
        "semantic_identity": semantic_identity,
        "publication_identity": "0" * 64,
        "lineage": _lineage(execution.source_snapshot, execution.strategy_selection),
        "evidence": evidence,
    }
    payload["publication_identity"] = _publication_identity(payload)
    return B13DispositionPublication.from_dict(payload)


__all__ = [
    "B13_DISPOSITION_PUBLICATION_SCHEMA_VERSION", "B13_DISPOSITION_PUBLICATION_ARTIFACT_TYPE",
    "B13DispositionPublicationError", "B13DispositionPublication",
    "b13_disposition_publication_artifact_id", "prepare_b13_disposition_publication",
    "verify_b13_disposition_publication",
]
