"""B1 adaptation-domain contracts.

Only B1.1 target normalization belongs here.  This package deliberately does
not plan events, episodes, or publish an AdaptationPlan.
"""

from .target import (
    AdaptationTarget,
    AdaptationTargetDefaults,
    AdaptationTargetError,
    AdaptationTargetOverrides,
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

__all__ = [
    "AdaptationTarget", "AdaptationTargetDefaults", "AdaptationTargetError",
    "AdaptationTargetOverrides", "CanonicalTargetReference", "ConstraintKind",
    "ConstraintStrength", "FidelityProfile", "NumericConstraint", "Provenance",
    "ResolvedAdaptationTarget", "StableIdIndex", "TargetElement", "TargetValue",
    "UnresolvedTargetReference", "resolve_adaptation_target",
]
