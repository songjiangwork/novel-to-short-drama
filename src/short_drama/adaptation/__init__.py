"""B1 adaptation-domain contracts.

This package deliberately does not plan chronology, presentation entries,
episodes, or publish an AdaptationPlan.
"""

from .target import (
    AdaptationTarget,
    AdaptationTargetDefaults,
    AdaptationTargetError,
    AdaptationTargetOverrides,
    AdaptationSourceSnapshot,
    CanonicalTargetReference,
    ConstraintKind,
    ConstraintStrength,
    FidelityProfile,
    NumericConstraint,
    Provenance,
    ResolvedAdaptationTarget,
    StableIdIndex,
    TargetElement,
    TargetValue,
    UnresolvedTargetReference,
    resolve_adaptation_target,
)
from .strategy import (
    AdaptationStrategy, AdaptationStrategyError, CharacterDisposition, CharacterStrategy,
    DeliberateDivergence, EndingDisposition, PlotlineDisposition, PlotlineStrategy,
    PreservationPriority, PriorityEntry, RelationshipDisposition, RelationshipStrategy,
    RevealDisposition, RevealStrategy, SnapshotLocalPlotline, StrategyConflict,
)
from .strategy_packet import B12Packet, B12PacketMeasurement, B12Preparation, B12SemanticView, derive_snapshot_local_plotlines, prepare_b12_packet
from .strategy_semantic import (
    B12SemanticAssetSelection, B12SemanticBudget, B12SemanticBudgetError, B12SemanticMeasurement,
    B12SemanticPreparation, b12_request_identity, build_b12_semantic_preparation,
    load_b12_output_schema, load_b12_semantic_assets, load_b12_semantic_profile,
)
from .strategy_execution import (
    B12_MAX_SEMANTIC_GENERATION_ROUNDS, B12_STRATEGY_ARTIFACT_TYPE,
    B12SemanticExecutionError, B12SemanticGenerationExhaustedError,
    B12SemanticGenerationPolicy, B12SemanticProvenanceError, B12StrategyArtifact,
    B12StrategyPublication, b12_strategy_artifact_id,
    b12_strategy_pointer_id, execute_b12_strategy,
)
from .disposition import (
    B13_DISPOSITION_SCHEMA_VERSION, B13DispositionError, B13DispositionPlan,
    AdaptationMapping, AdaptedCharacter, AdaptedCharacterOrigin,
    AdaptedContinuityOverlay, AdaptedEndpoint, AdaptedEndpointKind,
    AdaptedFactDivergence, AdaptedStoryBeat, AdaptedStoryBeatOrigin,
    AnonymousRole, AnonymousRoleKind, BeatParticipant, CanonCompatibility,
    CanonicalEventDisposition, CharacterFunctionProvenance, ContinuityStoryState,
    EventDisposition, InventedBeatPurpose, MappingContributionRole,
    ParticipantKind, RelationshipArcRequirement,
)
from .disposition_preparation import (
    B13_DISPOSITION_PACKET_SCHEMA_VERSION, B13PreparationError,
    B12StrategySelection, resolve_b12_strategy_selection,
    B13DispositionWindowPacket, B13DispositionPacketMeasurement,
    B13DispositionPreparation, prepare_b13_disposition_windows,
)

__all__ = [
    "AdaptationTarget", "AdaptationTargetDefaults", "AdaptationTargetError",
    "AdaptationTargetOverrides", "AdaptationSourceSnapshot", "CanonicalTargetReference", "ConstraintKind",
    "ConstraintStrength", "FidelityProfile", "NumericConstraint", "Provenance",
    "ResolvedAdaptationTarget", "StableIdIndex", "TargetElement", "TargetValue",
    "UnresolvedTargetReference", "resolve_adaptation_target",
    "AdaptationStrategy", "AdaptationStrategyError", "CharacterDisposition", "CharacterStrategy",
    "DeliberateDivergence", "EndingDisposition", "PlotlineDisposition", "PlotlineStrategy",
    "PreservationPriority", "PriorityEntry", "RelationshipDisposition", "RelationshipStrategy",
    "RevealDisposition", "RevealStrategy", "SnapshotLocalPlotline", "StrategyConflict",
    "B12Packet", "B12PacketMeasurement", "B12Preparation", "B12SemanticView", "derive_snapshot_local_plotlines", "prepare_b12_packet",
    "B12SemanticAssetSelection", "B12SemanticBudget", "B12SemanticBudgetError", "B12SemanticMeasurement", "B12SemanticPreparation",
    "b12_request_identity", "build_b12_semantic_preparation", "load_b12_output_schema", "load_b12_semantic_assets", "load_b12_semantic_profile",
    "B12_MAX_SEMANTIC_GENERATION_ROUNDS", "B12_STRATEGY_ARTIFACT_TYPE",
    "B12SemanticExecutionError", "B12SemanticGenerationExhaustedError",
    "B12SemanticGenerationPolicy", "B12SemanticProvenanceError", "B12StrategyArtifact",
    "B12StrategyPublication", "b12_strategy_artifact_id",
    "b12_strategy_pointer_id", "execute_b12_strategy",
    "B13_DISPOSITION_SCHEMA_VERSION", "B13DispositionError", "B13DispositionPlan",
    "AdaptationMapping", "AdaptedCharacter", "AdaptedCharacterOrigin",
    "AdaptedContinuityOverlay", "AdaptedEndpoint", "AdaptedEndpointKind",
    "AdaptedFactDivergence", "AdaptedStoryBeat", "AdaptedStoryBeatOrigin",
    "AnonymousRole", "AnonymousRoleKind", "BeatParticipant", "CanonCompatibility",
    "CanonicalEventDisposition", "CharacterFunctionProvenance", "ContinuityStoryState",
    "EventDisposition", "InventedBeatPurpose", "MappingContributionRole",
    "ParticipantKind", "RelationshipArcRequirement",
    "B13_DISPOSITION_PACKET_SCHEMA_VERSION", "B13PreparationError",
    "B12StrategySelection", "resolve_b12_strategy_selection",
    "B13DispositionWindowPacket", "B13DispositionPacketMeasurement",
    "B13DispositionPreparation", "prepare_b13_disposition_windows",
]
