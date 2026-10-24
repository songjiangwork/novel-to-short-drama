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
from .disposition_semantic import (
    B13WindowSemanticError, B13WindowSchemaError, B13WindowProposalError,
    B13WindowSemanticBudgetError, B13WindowSemanticAssetSelection,
    B13WindowProposal, B13WindowSemanticBudget, B13WindowSemanticMeasurement,
    B13WindowSemanticPreparation, load_b13_window_semantic_assets,
    b13_window_request_identity, build_b13_window_semantic_preparation,
)
from .disposition_execution import (
    B13_MAX_SEMANTIC_GENERATION_ROUNDS, B13WindowExecutionError,
    B13WindowProvenanceError, B13WindowGenerationExhaustedError,
    B13WindowSemanticGenerationPolicy, B13WindowRejectedSemanticRound,
    B13WindowAcceptedProposal,
    B13DispositionExecutionResult, execute_b13_disposition_windows,
)
from .disposition_reconciliation import (
    B13ReconciliationError, B13CandidateKind, B13CandidateRef,
    B13ReconciliationCandidate, B13InventoryWindow,
    B13ReconciliationFindingKind, B13ReconciliationFinding,
    B13ReconciliationInventory, B13ReconciliationDecisionKind,
    B13ReconciliationDecision, B13ReconciliationDecisionSet,
    B13ReconciliationReview, build_b13_reconciliation_inventory,
    review_b13_reconciliation,
)
from .disposition_assembly import (
    B13DispositionAssemblyError, B13GlobalIdentityBinding, B13AssemblyWindowAudit,
    B13DispositionAssemblyResult, assemble_b13_disposition_plan,
)
from .disposition_publication import (
    B13_DISPOSITION_PUBLICATION_SCHEMA_VERSION, B13_DISPOSITION_PUBLICATION_ARTIFACT_TYPE,
    B13DispositionPublicationError, B13DispositionPublication,
    b13_disposition_publication_artifact_id, prepare_b13_disposition_publication,
    verify_b13_disposition_publication,
)
from .disposition_persistence import (
    B13DispositionPersistenceError, B13DispositionPersistenceService,
    B13DispositionPublicationResult, B13VerifiedCurrentSnapshot,
    b13_disposition_pointer_id,
)
from .skeleton import (
    B14_SKELETON_SCHEMA_VERSION, B14SkeletonError, AdaptedStorySkeleton,
    ArcCheckpoint, ArcCheckpointKind, ArcDimension, AudienceKnowledgeState,
    BeatAccounting, BeatPresentationTreatment, CharacterKnowledgeState,
    CharacterKnowledgeTransition, GapAnchor, GapAnchorKind,
    ChronologyPrecedence, ContinuityStateKind, GapDisposition,
    KnowledgeStatus, KnowledgeSubjectKind, NarrativeDependency, NarrativeOutcome,
    PresentationEntry, PresentationSemantics, RevealTransition, SetupPayoffLink,
    SkeletonGap, SkeletonGapKind, WithholdingBoundary, WorldTruthState,
)
from .skeleton_validation import (
    B14SkeletonAuthority, B14SkeletonValidationError, B14SkeletonValidationResult,
    B14ValidationFinding, PrimaryPresentationIdentity, resolve_b14_skeleton_authority,
    resolve_b14_skeleton_semantic_authority,
    validate_adapted_story_skeleton, validate_stored_adapted_story_skeleton,
)
from .skeleton_semantic import (
    B14SkeletonSemanticError, B14SkeletonSemanticBudgetError, B14SkeletonSemanticAssetSelection,
    B14SkeletonSemanticBudget, B14SkeletonSemanticMeasurement, B14SkeletonPacket,
    B14SkeletonSemanticPreparation, prepare_b14_skeleton_packet,
    load_b14_skeleton_semantic_assets, b14_skeleton_request_identity,
    build_b14_skeleton_semantic_preparation,
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
    "B13WindowSemanticError", "B13WindowSchemaError", "B13WindowProposalError",
    "B13WindowSemanticBudgetError", "B13WindowSemanticAssetSelection",
    "B13WindowProposal", "B13WindowSemanticBudget", "B13WindowSemanticMeasurement",
    "B13WindowSemanticPreparation", "load_b13_window_semantic_assets",
    "b13_window_request_identity", "build_b13_window_semantic_preparation",
    "B13_MAX_SEMANTIC_GENERATION_ROUNDS", "B13WindowExecutionError",
    "B13WindowProvenanceError", "B13WindowGenerationExhaustedError",
    "B13WindowSemanticGenerationPolicy", "B13WindowRejectedSemanticRound",
    "B13WindowAcceptedProposal",
    "B13DispositionExecutionResult", "execute_b13_disposition_windows",
    "B13ReconciliationError", "B13CandidateKind", "B13CandidateRef",
    "B13ReconciliationCandidate", "B13InventoryWindow",
    "B13ReconciliationFindingKind", "B13ReconciliationFinding",
    "B13ReconciliationInventory", "B13ReconciliationDecisionKind",
    "B13ReconciliationDecision", "B13ReconciliationDecisionSet",
    "B13ReconciliationReview", "build_b13_reconciliation_inventory",
    "review_b13_reconciliation",
    "B13DispositionAssemblyError", "B13GlobalIdentityBinding", "B13AssemblyWindowAudit",
    "B13DispositionAssemblyResult", "assemble_b13_disposition_plan",
    "B13_DISPOSITION_PUBLICATION_SCHEMA_VERSION", "B13_DISPOSITION_PUBLICATION_ARTIFACT_TYPE",
    "B13DispositionPublicationError", "B13DispositionPublication",
    "b13_disposition_publication_artifact_id", "prepare_b13_disposition_publication",
    "verify_b13_disposition_publication",
    "B13DispositionPersistenceError", "B13DispositionPersistenceService",
    "B13DispositionPublicationResult", "B13VerifiedCurrentSnapshot",
    "b13_disposition_pointer_id",
    "B14_SKELETON_SCHEMA_VERSION", "B14SkeletonError", "AdaptedStorySkeleton",
    "ArcCheckpoint", "ArcCheckpointKind", "ArcDimension", "AudienceKnowledgeState",
    "BeatAccounting", "BeatPresentationTreatment", "CharacterKnowledgeState",
    "CharacterKnowledgeTransition", "GapAnchor", "GapAnchorKind",
    "ChronologyPrecedence", "ContinuityStateKind", "GapDisposition",
    "KnowledgeStatus", "KnowledgeSubjectKind", "NarrativeDependency", "NarrativeOutcome",
    "PresentationEntry", "PresentationSemantics", "RevealTransition", "SetupPayoffLink",
    "SkeletonGap", "SkeletonGapKind", "WithholdingBoundary", "WorldTruthState",
    "B14SkeletonAuthority", "B14SkeletonValidationError", "B14SkeletonValidationResult",
    "B14ValidationFinding", "PrimaryPresentationIdentity", "resolve_b14_skeleton_authority",
    "resolve_b14_skeleton_semantic_authority",
    "validate_adapted_story_skeleton", "validate_stored_adapted_story_skeleton",
    "B14SkeletonSemanticError", "B14SkeletonSemanticBudgetError", "B14SkeletonSemanticAssetSelection",
    "B14SkeletonSemanticBudget", "B14SkeletonSemanticMeasurement", "B14SkeletonPacket",
    "B14SkeletonSemanticPreparation", "prepare_b14_skeleton_packet",
    "load_b14_skeleton_semantic_assets", "b14_skeleton_request_identity",
    "build_b14_skeleton_semantic_preparation",
]
