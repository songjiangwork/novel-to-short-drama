"""B1 adaptation-domain contracts.

Only B1.1 target normalization belongs here.  This package deliberately does
not plan events, episodes, or publish an AdaptationPlan.
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
    B12SemanticBudget, B12SemanticBudgetError, B12SemanticMeasurement, B12SemanticPreparation,
    b12_request_identity, build_b12_semantic_preparation, load_b12_output_schema, load_b12_semantic_profile,
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
    "B12SemanticBudget", "B12SemanticBudgetError", "B12SemanticMeasurement", "B12SemanticPreparation",
    "b12_request_identity", "build_b12_semantic_preparation", "load_b12_output_schema", "load_b12_semantic_profile",
]
