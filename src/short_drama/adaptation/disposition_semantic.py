"""B1.3-B2A versioned, zero-provider window proposal preparation.

The proposal is deliberately *not* a ``B13DispositionPlan`` fragment.  Qwen
uses window-local candidate keys; B1.3-B3 alone may reconcile candidates and
assign the Python-owned adapted IDs.  This module neither calls a provider nor
persists or publishes anything.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any

from short_drama.artifacts import canonical_json_bytes, content_hash
from short_drama.io import load_json
from short_drama.llm import (
    GenerationExecutionOptions, OpenAICompatibleLLMClient, OutputSchema,
    PromptRegistry, RuntimeConfig, SemanticLLMProfile, StructuredGenerationRequest,
    build_structured_request, load_semantic_profile, validate_against_output_schema,
)
from short_drama.paths import REPO_ROOT
from short_drama.story.chunking import estimate_tokens

from .disposition import (
    AdaptedCharacterOrigin, AdaptedEndpointKind, AdaptedStoryBeatOrigin,
    CanonCompatibility, CanonicalEventDisposition, InventedBeatPurpose,
    MappingContributionRole, AnonymousRoleKind, ParticipantKind,
)
from .disposition_preparation import B13DispositionWindowPacket, B13PreparationError
from .target import CanonicalTargetReference


B13_WINDOW_PROMPT_ID = "b13.window-disposition"
B13_WINDOW_PROMPT_VERSION = 1
B13_WINDOW_OUTPUT_SCHEMA_ID = "b13-window-disposition-proposal"
B13_WINDOW_PROMPTS_DIR = REPO_ROOT / "prompts" / "adaptation"
B13_WINDOW_PROMPT_CONTENT_HASH_V1 = "01052d4b6a202506851b7e560bb8cb9fadc209aa24fb1272b5d86cd2bf0e0c04"
_LOCAL_KEY = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
_RESERVED_LOCAL_PREFIXES = ("adapted_", "adaptation_", "anonymous_", "char_")


class B13WindowSemanticError(B13PreparationError):
    """A B1.3 window semantic asset, proposal, or budget is invalid."""


class B13WindowSchemaError(B13WindowSemanticError):
    """A provider value did not satisfy the pinned output JSON Schema.

    This is deliberately distinct from a schema-valid proposal which violates
    a B1.3 semantic invariant.  The common adapter owns retry of malformed
    structured output; B1.3 only regenerates the latter.
    """


class B13WindowProposalError(B13WindowSemanticError):
    """A schema-valid window proposal violates a B1.3 semantic invariant."""


class B13WindowSemanticBudgetError(B13WindowSemanticError):
    """An explicit future provider context declaration is insufficient."""


@dataclass(frozen=True, slots=True)
class B13WindowSemanticAssetSelection:
    version: int
    semantic_profile: SemanticLLMProfile
    output_schema: OutputSchema


# A closed reviewed pair.  The output reserve is a versioned semantic profile
# setting, while input ceiling and n_ctx intentionally remain operator-supplied
# runtime declarations; this repository does not guess either production value.
_B13_WINDOW_ASSETS: dict[int, tuple[str, str, str, str, str]] = {
    1: (
        "b13_window_disposition_llm_v1.yaml",
        "b13-window-disposition-proposal-v1.schema.json",
        "b13-window-disposition-llm-v1",
        "680ea8b52a5d0a04cee995b08156debe9dc199483c8e6fe6ac733a75d9745d85",
        "59fdff8bbfb7633c6a3a4a9296eb1c0513daa2f8ee52e01fa8abd824cc23c0b3",
    ),
}


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\0" in value:
        raise B13WindowSemanticError(f"{name} must be non-empty text without NUL")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise B13WindowSemanticError(f"{name} must be UTF-8 text") from exc
    return value


def _local_key(value: Any, name: str) -> str:
    value = _text(value, name)
    if _LOCAL_KEY.fullmatch(value) is None or value.startswith(_RESERVED_LOCAL_PREFIXES):
        raise B13WindowSemanticError(f"{name} must be a non-authoritative window-local key")
    return value


def _canonical_ref(value: Any, name: str, allowed: set[CanonicalTargetReference], *, kind: str | None = None) -> CanonicalTargetReference:
    try:
        ref = CanonicalTargetReference.from_dict(value)
    except Exception as exc:
        raise B13WindowSemanticError(f"{name} must be a canonical source reference") from exc
    if kind is not None and ref.kind != kind:
        raise B13WindowSemanticError(f"{name} must reference a {kind}")
    if ref not in allowed:
        raise B13WindowSemanticError(f"{name} references source outside the exact A5/A6 snapshot")
    return ref


def _enum(value: Any, enum: type, name: str) -> str:
    try:
        return enum(value).value
    except (TypeError, ValueError) as exc:
        raise B13WindowSemanticError(f"{name} is not a permitted value") from exc


def _exact(value: Any, fields: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise B13WindowSemanticError(f"{name} must contain exactly: {', '.join(sorted(fields))}")
    return value


@dataclass(frozen=True, slots=True)
class B13WindowProposal:
    """Validated provider proposal with local keys only, ready for B1.3-B3."""

    packet: B13DispositionWindowPacket
    _canonical_payload: bytes

    @classmethod
    def from_dict(cls, packet: B13DispositionWindowPacket, value: dict[str, Any], schema: OutputSchema) -> "B13WindowProposal":
        if not isinstance(packet, B13DispositionWindowPacket):
            raise B13WindowSemanticError("proposal requires an exact B1.3 window packet")
        if not isinstance(value, dict):
            raise B13WindowSemanticError("window proposal must be an object")
        # This mirrors the client acceptance seam for zero-provider callers.
        try:
            validate_against_output_schema(value, schema)
        except Exception as exc:
            raise B13WindowSchemaError("window proposal violates the selected output schema") from exc
        cls._validate(packet, value)
        return cls(packet, canonical_json_bytes(value))

    @staticmethod
    def _validate(packet: B13DispositionWindowPacket, value: dict[str, Any]) -> None:
        # All failures below occur after JSON Schema acceptance.  Keeping the
        # classification explicit lets B2B retry only semantic invalidity.
        try:
            B13WindowProposal._validate_semantics(packet, value)
        except B13WindowSemanticError as exc:
            raise B13WindowProposalError(str(exc)) from exc

    @staticmethod
    def _validate_semantics(packet: B13DispositionWindowPacket, value: dict[str, Any]) -> None:
        _exact(value, {"schema_version", "event_dispositions", "adapted_characters", "anonymous_roles", "adapted_story_beats", "adaptation_mappings", "continuity_overlay_candidates"}, "window proposal")
        if value["schema_version"] != 1:
            raise B13WindowSemanticError("unsupported B1.3 window proposal schema_version")
        for name in ("event_dispositions", "adapted_characters", "anonymous_roles", "adapted_story_beats", "adaptation_mappings"):
            if not isinstance(value[name], list):
                raise B13WindowSemanticError(f"{name} must be an array")
        if not isinstance(value["continuity_overlay_candidates"], dict):
            raise B13WindowSemanticError("continuity_overlay_candidates must be an object")
        view = packet.to_dict()
        owned = tuple(view["window"]["owned_event_refs"])
        allowed = set(packet.source_snapshot.canonical_references)
        decisions: set[str] = set()
        for item in value["event_dispositions"]:
            item = _exact(item, {"source_event_id", "disposition", "rationale_zh"}, "event disposition")
            event_id = _text(item["source_event_id"], "source_event_id")
            _enum(item["disposition"], CanonicalEventDisposition, "event disposition")
            _text(item["rationale_zh"], "event disposition rationale_zh")
            if event_id not in owned:
                raise B13WindowSemanticError("only this window's owned events may receive authoritative dispositions")
            if event_id in decisions:
                raise B13WindowSemanticError("owned event dispositions must not be duplicated")
            decisions.add(event_id)
        if decisions != set(owned):
            raise B13WindowSemanticError("owned event dispositions must cover exactly every owned event")

        characters: dict[str, set[CanonicalTargetReference]] = {}
        for item in value["adapted_characters"]:
            item = _exact(item, {"local_key", "display_name_zh", "origin", "function_provenance", "invention_rationale_zh"}, "adapted character candidate")
            key = _local_key(item["local_key"], "adapted character local_key")
            if key in characters:
                raise B13WindowSemanticError("adapted character local_key must be unique")
            _text(item["display_name_zh"], "adapted character display_name_zh")
            origin = _enum(item["origin"], AdaptedCharacterOrigin, "adapted character origin")
            if not isinstance(item["function_provenance"], list):
                raise B13WindowSemanticError("function_provenance must be an array")
            refs: set[CanonicalTargetReference] = set()
            for provenance in item["function_provenance"]:
                provenance = _exact(provenance, {"source_character_ref", "function_zh", "provenance_zh"}, "character function provenance")
                ref = _canonical_ref(provenance["source_character_ref"], "source_character_ref", allowed, kind="character")
                if ref in refs:
                    raise B13WindowSemanticError("character function provenance must not repeat a source character")
                refs.add(ref)
                _text(provenance["function_zh"], "character function_zh")
                _text(provenance["provenance_zh"], "character provenance_zh")
            rationale = item["invention_rationale_zh"]
            if origin == "source_single" and len(refs) != 1:
                raise B13WindowSemanticError("source_single character requires exactly one function provenance")
            if origin == "source_composite" and len(refs) < 2:
                raise B13WindowSemanticError("source_composite character requires at least two function provenances")
            if origin == "adaptation_invented":
                if refs:
                    raise B13WindowSemanticError("adaptation_invented character must not claim source provenance")
                _text(rationale, "invented character rationale")
            elif rationale is not None:
                raise B13WindowSemanticError("invention_rationale_zh is only permitted for adaptation_invented characters")
            characters[key] = refs

        anonymous_roles: dict[str, str] = {}
        for item in value["anonymous_roles"]:
            item = _exact(item, {"local_key", "role_kind", "function_zh"}, "anonymous role candidate")
            key = _local_key(item["local_key"], "anonymous role local_key")
            if key in anonymous_roles or key in characters:
                raise B13WindowSemanticError("anonymous role local_key must be unique across local participants")
            anonymous_roles[key] = _enum(item["role_kind"], AnonymousRoleKind, "anonymous role kind")
            _text(item["function_zh"], "anonymous role function_zh")

        beats: dict[str, tuple[str, set[CanonicalTargetReference], set[CanonicalTargetReference]]] = {}
        owned_beat_memberships: dict[str, set[str]] = {event_id: set() for event_id in owned}
        anonymous_role_uses: dict[str, int] = {key: 0 for key in anonymous_roles}
        for item in value["adapted_story_beats"]:
            fields = {"local_key", "summary_zh", "origin", "canon_compatibility", "source_event_refs", "participants", "canonical_anchor_refs", "invention_purpose", "invention_rationale_zh", "invention_provenance_zh"}
            item = _exact(item, fields, "adapted story beat candidate")
            key = _local_key(item["local_key"], "adapted story beat local_key")
            if key in beats:
                raise B13WindowSemanticError("adapted story beat local_key must be unique")
            _text(item["summary_zh"], "adapted story beat summary_zh")
            origin = _enum(item["origin"], AdaptedStoryBeatOrigin, "adapted story beat origin")
            _enum(item["canon_compatibility"], CanonCompatibility, "canon compatibility")
            if not isinstance(item["source_event_refs"], list) or not isinstance(item["canonical_anchor_refs"], list) or not isinstance(item["participants"], list):
                raise B13WindowSemanticError("beat source references and participants must be arrays")
            event_refs = {_canonical_ref(ref, "source_event_ref", allowed, kind="event") for ref in item["source_event_refs"]}
            anchors = {_canonical_ref(ref, "canonical_anchor_ref", allowed) for ref in item["canonical_anchor_refs"]}
            if len(event_refs) != len(item["source_event_refs"]) or len(anchors) != len(item["canonical_anchor_refs"]):
                raise B13WindowSemanticError("beat source references must not repeat")
            if origin in {"source_direct", "source_condensed"} and len(event_refs) != 1:
                raise B13WindowSemanticError("source_direct/source_condensed beat requires exactly one source event")
            if origin == "source_composite" and len(event_refs) < 2:
                raise B13WindowSemanticError("source_composite beat requires at least two source events")
            invented = (item["invention_purpose"], item["invention_rationale_zh"], item["invention_provenance_zh"])
            if origin == "adaptation_invented":
                if event_refs or not anchors:
                    raise B13WindowSemanticError("adaptation_invented beat requires anchors and no source events")
                _enum(item["invention_purpose"], InventedBeatPurpose, "invented beat purpose")
                _text(item["invention_rationale_zh"], "invented beat rationale")
                _text(item["invention_provenance_zh"], "invented beat provenance")
            elif any(part is not None for part in invented):
                raise B13WindowSemanticError("invention fields are only permitted for adaptation_invented beats")
            seen_participants: set[tuple[str, str]] = set()
            for participant in item["participants"]:
                participant = _exact(participant, {"kind", "local_key"}, "beat participant")
                kind = _enum(participant["kind"], ParticipantKind, "beat participant kind")
                participant_key = _local_key(participant["local_key"], "beat participant local_key")
                if (kind, participant_key) in seen_participants:
                    raise B13WindowSemanticError("beat participants must not be duplicated")
                seen_participants.add((kind, participant_key))
                if kind == "adapted_character":
                    if participant_key not in characters:
                        raise B13WindowSemanticError("adapted character participant local_key does not resolve")
                else:
                    if participant_key not in anonymous_roles:
                        raise B13WindowSemanticError("anonymous role participant local_key does not resolve")
                    anonymous_role_uses[participant_key] += 1
            for ref in event_refs:
                if ref.canonical_id in owned_beat_memberships:
                    owned_beat_memberships[ref.canonical_id].add(key)
            beats[key] = (origin, event_refs, anchors)

        active_owned: dict[str, set[tuple[str, str]]] = {event_id: set() for event_id in owned}
        seen_mappings: set[bytes] = set()
        for item in value["adaptation_mappings"]:
            item = _exact(item, {"source_ref", "adapted_endpoint", "contribution_role", "retained_semantics_zh", "transformed_semantics_zh", "provenance_zh", "active"}, "adaptation mapping candidate")
            ref = _canonical_ref(item["source_ref"], "mapping source_ref", allowed)
            endpoint = _exact(item["adapted_endpoint"], {"kind", "local_key"}, "mapping adapted_endpoint")
            kind = _enum(endpoint["kind"], AdaptedEndpointKind, "mapping endpoint kind")
            key = _local_key(endpoint["local_key"], "mapping endpoint local_key")
            _enum(item["contribution_role"], MappingContributionRole, "mapping contribution_role")
            for name in ("retained_semantics_zh", "transformed_semantics_zh", "provenance_zh"):
                _text(item[name], name)
            if not isinstance(item["active"], bool):
                raise B13WindowSemanticError("mapping active must be boolean")
            signature = canonical_json_bytes(item)
            if signature in seen_mappings:
                raise B13WindowSemanticError("adaptation mappings must not be duplicated")
            seen_mappings.add(signature)
            if kind == "story_beat":
                if key not in beats:
                    raise B13WindowSemanticError("mapping adapted story-beat local_key does not resolve")
                _origin, event_refs, anchors = beats[key]
                if ref.kind == "event" and ref not in event_refs:
                    raise B13WindowSemanticError("event mapping must be declared by its beat source_event_refs")
                if ref.kind != "event" and ref not in anchors:
                    raise B13WindowSemanticError("non-event mapping must be declared by its beat canonical_anchor_refs")
            else:
                if key not in characters or ref.kind != "character" or ref not in characters[key]:
                    raise B13WindowSemanticError("character mapping must be backed by function provenance")
            if item["active"] and ref.kind == "event" and ref.canonical_id in active_owned:
                active_owned[ref.canonical_id].add((kind, key))
        disposition_by_event = {item["source_event_id"]: item["disposition"] for item in value["event_dispositions"]}
        for event_id, disposition in disposition_by_event.items():
            active = active_owned[event_id]
            memberships = owned_beat_memberships[event_id]
            if disposition == "omit" and active:
                raise B13WindowSemanticError("omit owned event forbids active mapping candidates")
            if disposition == "omit" and memberships:
                raise B13WindowSemanticError("omit owned event must not materialize in a beat candidate")
            if disposition != "omit" and not active:
                raise B13WindowSemanticError("kept/compressed/merged owned event requires an active mapping candidate")
            if any(("story_beat", beat_key) not in active for beat_key in memberships):
                raise B13WindowSemanticError("owned source-backed beat candidate requires its corresponding active mapping")
            if disposition == "compress" and not any(beats[key][0] in {"source_direct", "source_condensed"} for kind, key in active if kind == "story_beat"):
                raise B13WindowSemanticError("compress owned event requires a single-source beat candidate")
            if disposition == "merge_member" and not active:
                raise B13WindowSemanticError("merge_member owned event requires an active mapping candidate")
            if disposition == "merge_member" and any(kind != "story_beat" or beats[key][0] != "source_composite" for kind, key in active):
                raise B13WindowSemanticError("merge_member owned event must map only to source_composite beat candidates")
        if any(uses != 1 for uses in anonymous_role_uses.values()):
            raise B13WindowSemanticError("anonymous role candidate must participate in exactly one beat")

        overlay = _exact(value["continuity_overlay_candidates"], {"factual_divergences", "relationship_requirements", "story_state_requirements"}, "continuity_overlay_candidates")
        for name in overlay:
            if not isinstance(overlay[name], list):
                raise B13WindowSemanticError(f"continuity overlay {name} must be an array")
        factual_refs: set[CanonicalTargetReference] = set()
        for item in overlay["factual_divergences"]:
            item = _exact(item, {"source_fact_ref", "adapted_assertion_zh", "provenance_zh"}, "factual divergence candidate")
            ref = _canonical_ref(item["source_fact_ref"], "source_fact_ref", allowed, kind="fact")
            if ref in factual_refs:
                raise B13WindowSemanticError("factual divergence candidates must not repeat a source fact")
            factual_refs.add(ref)
            _text(item["adapted_assertion_zh"], "adapted_assertion_zh"); _text(item["provenance_zh"], "provenance_zh")
        relationship_refs: set[CanonicalTargetReference] = set()
        for item in overlay["relationship_requirements"]:
            item = _exact(item, {"source_relationship_ref", "requirement_zh", "provenance_zh"}, "relationship requirement candidate")
            ref = _canonical_ref(item["source_relationship_ref"], "source_relationship_ref", allowed, kind="relationship")
            if ref in relationship_refs:
                raise B13WindowSemanticError("relationship requirement candidates must not repeat a source relationship")
            relationship_refs.add(ref)
            _text(item["requirement_zh"], "requirement_zh"); _text(item["provenance_zh"], "provenance_zh")
        state_keys: set[str] = set()
        for item in overlay["story_state_requirements"]:
            item = _exact(item, {"adapted_story_beat_local_key", "assertion_zh", "provenance_zh"}, "story state candidate")
            key = _local_key(item["adapted_story_beat_local_key"], "story state local_key")
            if key not in beats:
                raise B13WindowSemanticError("story state candidate must reference a local beat candidate")
            if key in state_keys:
                raise B13WindowSemanticError("story state candidates must not repeat a local beat")
            state_keys.add(key)
            _text(item["assertion_zh"], "assertion_zh"); _text(item["provenance_zh"], "provenance_zh")

    def to_dict(self) -> dict[str, Any]:
        from short_drama.artifacts import strict_json_loads
        value = strict_json_loads(self._canonical_payload)
        assert isinstance(value, dict)
        return value

    def canonical_bytes(self) -> bytes:
        return self._canonical_payload

    def content_hash(self) -> str:
        return content_hash(self.to_dict())


@dataclass(frozen=True, slots=True)
class B13WindowSemanticBudget:
    runtime_model_id: str
    model_context_tokens: int
    input_ceiling_tokens: int

    def __post_init__(self) -> None:
        _text(self.runtime_model_id, "runtime_model_id")
        for name in ("model_context_tokens", "input_ceiling_tokens"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise B13WindowSemanticBudgetError(f"{name} must be a positive integer")


@dataclass(frozen=True, slots=True)
class B13WindowSemanticMeasurement:
    packet_bytes: int
    packet_estimated_tokens: int
    provider_request_bytes: int
    provider_request_estimated_tokens: int
    reserved_output_tokens: int


@dataclass(frozen=True, slots=True)
class B13WindowSemanticPreparation:
    packet: B13DispositionWindowPacket
    semantic_profile: SemanticLLMProfile
    output_schema: OutputSchema
    request: StructuredGenerationRequest
    runtime_config: RuntimeConfig
    execution_options: GenerationExecutionOptions | None
    provider_request_bytes: bytes
    measurement: B13WindowSemanticMeasurement
    request_identity: str

    def enforce_budget(self, budget: B13WindowSemanticBudget | None) -> None:
        if budget is None:
            raise B13WindowSemanticBudgetError("B1.3 provider execution requires an explicit semantic budget")
        if budget.runtime_model_id != self.runtime_config.request_model:
            raise B13WindowSemanticBudgetError("semantic budget runtime_model_id does not match the exact runtime request_model")
        if self.measurement.provider_request_estimated_tokens > budget.input_ceiling_tokens:
            raise B13WindowSemanticBudgetError("B1.3 mapped provider request exceeds input ceiling; refusing to truncate")
        if self.measurement.provider_request_estimated_tokens + self.measurement.reserved_output_tokens > budget.model_context_tokens:
            raise B13WindowSemanticBudgetError("B1.3 mapped provider request plus reserved output exceeds runtime model context")

    def decode_and_validate_proposal(self, proposal: dict[str, Any]) -> B13WindowProposal:
        return B13WindowProposal.from_dict(self.packet, proposal, self.output_schema)


def load_b13_window_semantic_assets(*, version: int = 1) -> B13WindowSemanticAssetSelection:
    try:
        profile_name, schema_name, profile_id, profile_hash, schema_hash = _B13_WINDOW_ASSETS[version]
    except (KeyError, TypeError):
        raise B13WindowSemanticError(f"unsupported B1.3 window semantic asset version {version!r}; supported versions are 1") from None
    if isinstance(version, bool) or not isinstance(version, int):
        raise B13WindowSemanticError(f"unsupported B1.3 window semantic asset version {version!r}; supported versions are 1")
    try:
        profile = load_semantic_profile(REPO_ROOT / "profiles" / profile_name)
        data = load_json(REPO_ROOT / "schemas" / schema_name)
    except Exception as exc:
        raise B13WindowSemanticError(f"cannot load B1.3 window semantic assets for version {version}") from exc
    if profile.profile_id != profile_id or profile.semantic_profile_hash != profile_hash:
        raise B13WindowSemanticError(f"B1.3 window semantic profile content hash does not match pinned v{version} asset")
    if profile.structured_output_mode != "json_schema":
        raise B13WindowSemanticError(f"B1.3 window semantic profile v{version} requires structured_output_mode=json_schema")
    schema = OutputSchema.create(
        schema_id=B13_WINDOW_OUTPUT_SCHEMA_ID, schema_version=version, schema=data,
    )
    if schema.schema_hash != schema_hash:
        raise B13WindowSemanticError(f"B1.3 window proposal schema content hash does not match pinned v{version} asset")
    return B13WindowSemanticAssetSelection(version, profile, schema)


def _resolve_assets(version: int, profile: SemanticLLMProfile | None, schema: OutputSchema | None) -> B13WindowSemanticAssetSelection:
    selected = load_b13_window_semantic_assets(version=version)
    if (profile is None) != (schema is None):
        raise B13WindowSemanticError("B1.3 window semantic profile and output schema must be supplied together or selected by version")
    if profile is not None and (profile != selected.semantic_profile or schema != selected.output_schema):
        raise B13WindowSemanticError(f"B1.3 window semantic assets do not match the pinned v{version} profile/schema pair")
    return selected


def _mapped_provider_request_bytes(request: StructuredGenerationRequest, runtime: RuntimeConfig, options: GenerationExecutionOptions | None) -> bytes:
    body = OpenAICompatibleLLMClient(runtime).build_request_body(request.rendered_prompt, request.output_schema, request.semantic_profile, execution_options=options)
    return json.dumps(body, ensure_ascii=False).encode("utf-8")


def b13_window_request_identity(*, packet: B13DispositionWindowPacket, profile: SemanticLLMProfile, schema: OutputSchema, request: StructuredGenerationRequest) -> str:
    window = packet.to_dict()["window"]
    return content_hash({
        "stage": "B1.3-B2A", "source_snapshot": packet.source_snapshot.to_dict(),
        "strategy_ref": packet.strategy_selection.strategy_ref.to_dict(),
        "strategy_hash": packet.strategy_selection.strategy_hash,
        "window_id": packet.window_id, "window_ordinal": packet.window_ordinal,
        "owned_event_ids": window["owned_event_refs"], "window_packet_hash": packet.content_hash(),
        "prompt": {"prompt_id": request.rendered_prompt.prompt_id, "prompt_version": request.rendered_prompt.prompt_version, "prompt_content_hash": request.rendered_prompt.prompt_content_hash},
        "output_schema": {"schema_id": schema.schema_id, "schema_version": schema.schema_version, "schema_content_hash": schema.schema_hash},
        "semantic_profile": {"profile_id": profile.profile_id, "profile_hash": profile.semantic_profile_hash},
        "rendered_request_hash": request.request_hash,
    })


def build_b13_window_semantic_preparation(packet: B13DispositionWindowPacket, *, prompts: PromptRegistry | None = None, semantic_asset_version: int = 1, semantic_profile: SemanticLLMProfile | None = None, output_schema: OutputSchema | None = None, runtime_config: RuntimeConfig | None = None, execution_options: GenerationExecutionOptions | None = None) -> B13WindowSemanticPreparation:
    if not isinstance(packet, B13DispositionWindowPacket):
        raise B13WindowSemanticError("B1.3 window semantic preparation requires B13DispositionWindowPacket")
    if not isinstance(runtime_config, RuntimeConfig):
        raise B13WindowSemanticError("B1.3 window semantic preparation requires an explicit exact RuntimeConfig")
    if execution_options is not None and not isinstance(execution_options, GenerationExecutionOptions):
        raise B13WindowSemanticError("B1.3 execution_options must be GenerationExecutionOptions or None")
    assets = _resolve_assets(semantic_asset_version, semantic_profile, output_schema)
    prompts = PromptRegistry(B13_WINDOW_PROMPTS_DIR) if prompts is None else prompts
    prompt = prompts.load(B13_WINDOW_PROMPT_ID, version=B13_WINDOW_PROMPT_VERSION)
    if prompt.content_hash != B13_WINDOW_PROMPT_CONTENT_HASH_V1:
        raise B13WindowSemanticError(
            "B1.3 window prompt content hash does not match pinned v1 asset"
        )
    rendered = prompt.render({"window_packet_json": packet.canonical_bytes().decode("utf-8")})
    request = build_structured_request(rendered_prompt=rendered, output_schema=assets.output_schema, semantic_profile=assets.semantic_profile)
    provider_bytes = _mapped_provider_request_bytes(request, runtime_config, execution_options)
    measurement = B13WindowSemanticMeasurement(len(packet.canonical_bytes()), estimate_tokens(packet.canonical_bytes().decode("utf-8")), len(provider_bytes), estimate_tokens(provider_bytes.decode("utf-8")), assets.semantic_profile.max_output_tokens)
    return B13WindowSemanticPreparation(packet, assets.semantic_profile, assets.output_schema, request, runtime_config, execution_options, provider_bytes, measurement, b13_window_request_identity(packet=packet, profile=assets.semantic_profile, schema=assets.output_schema, request=request))


__all__ = [
    "B13_WINDOW_PROMPT_ID", "B13_WINDOW_PROMPT_VERSION", "B13_WINDOW_OUTPUT_SCHEMA_ID",
    "B13_WINDOW_PROMPT_CONTENT_HASH_V1",
    "B13WindowSemanticError", "B13WindowSchemaError", "B13WindowProposalError",
    "B13WindowSemanticBudgetError", "B13WindowSemanticAssetSelection",
    "B13WindowProposal", "B13WindowSemanticBudget", "B13WindowSemanticMeasurement",
    "B13WindowSemanticPreparation", "load_b13_window_semantic_assets",
    "b13_window_request_identity", "build_b13_window_semantic_preparation",
]
