from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from short_drama.artifacts import ArtifactError, ArtifactRef, FileArtifactStore, ImmutableArtifactEnvelope

from ._common import artifact_ref_sort_key, require_artifact_ref, require_exact_keys, require_text
from .errors import LineageModelError, LineageResolutionError


@dataclass(frozen=True, slots=True)
class LineageRef:
    """An exact immutable upstream artifact pin plus its semantic role."""

    role: str
    artifact_ref: ArtifactRef

    def __post_init__(self) -> None:
        require_text(self.role, "role", LineageModelError)
        require_artifact_ref(self.artifact_ref, "artifact_ref", LineageModelError)

    def to_dict(self) -> dict[str, object]:
        return {"role": self.role, "artifact_ref": self.artifact_ref.to_dict()}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "LineageRef":
        require_exact_keys(value, {"role", "artifact_ref"}, "LineageRef", LineageModelError)
        try:
            artifact_ref = ArtifactRef.from_dict(value["artifact_ref"])
        except (TypeError, ValueError) as exc:
            raise LineageModelError(f"invalid LineageRef artifact_ref: {exc}") from exc
        return cls(role=value["role"], artifact_ref=artifact_ref)

    def sort_key(self) -> tuple[object, ...]:
        return (self.role, *artifact_ref_sort_key(self.artifact_ref))


def resolve_lineage_ref(store: FileArtifactStore, lineage_ref: LineageRef) -> ImmutableArtifactEnvelope:
    if not isinstance(store, FileArtifactStore):
        raise LineageResolutionError("store must be a FileArtifactStore")
    if not isinstance(lineage_ref, LineageRef):
        raise LineageResolutionError("lineage_ref must be a LineageRef")
    try:
        return store.get_ref(lineage_ref.artifact_ref)
    except ArtifactError as exc:
        raise LineageResolutionError(
            f"failed to resolve lineage role {lineage_ref.role!r}: {lineage_ref.artifact_ref!r}"
        ) from exc


def verify_lineage_refs(
    store: FileArtifactStore,
    refs: Iterable[LineageRef],
) -> tuple[ImmutableArtifactEnvelope, ...]:
    ordered = tuple(sorted(tuple(refs), key=LineageRef.sort_key))
    return tuple(resolve_lineage_ref(store, ref) for ref in ordered)
