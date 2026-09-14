from __future__ import annotations

import os
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterator

from short_drama.artifacts import (
    ArtifactError,
    ArtifactNotFoundError,
    ArtifactRef,
    FileArtifactStore,
    ImmutableArtifactEnvelope,
    canonical_json_bytes,
    strict_json_loads,
)

from ._common import (
    require_artifact_ref,
    require_exact_keys,
    require_storage_id,
)
from .errors import (
    PointerConflictError,
    PointerError,
    PointerIntegrityError,
    PointerLockedError,
    PointerNotFoundError,
    SupersessionError,
)

CURRENT_POINTER_ARTIFACT_TYPE = "current_pointer"
SUPERSESSION_RECORD_ARTIFACT_TYPE = "supersession_record"
POINTER_SCHEMA_VERSION = 1


class PointerKind(str, Enum):
    CURRENT = "CURRENT"
    CURRENT_APPROVED = "CURRENT_APPROVED"


def _coerce_pointer_kind(
    value: Any,
    error_type: type[PointerError] | type[SupersessionError] = PointerError,
) -> PointerKind:
    if isinstance(value, PointerKind):
        return value
    try:
        return PointerKind(value)
    except (TypeError, ValueError) as exc:
        raise error_type("pointer_kind must be CURRENT or CURRENT_APPROVED") from exc


@dataclass(frozen=True, slots=True)
class CurrentPointer:
    pointer_id: str
    pointer_kind: PointerKind
    target_ref: ArtifactRef
    authority_ref: ArtifactRef | None = None
    previous_pointer_ref: ArtifactRef | None = None

    def __post_init__(self) -> None:
        require_storage_id(self.pointer_id, "pointer_id", PointerError)
        object.__setattr__(self, "pointer_kind", _coerce_pointer_kind(self.pointer_kind))
        require_artifact_ref(self.target_ref, "target_ref", PointerError)
        if self.authority_ref is not None:
            require_artifact_ref(self.authority_ref, "authority_ref", PointerError)
        if self.previous_pointer_ref is not None:
            require_artifact_ref(self.previous_pointer_ref, "previous_pointer_ref", PointerError)
            if self.previous_pointer_ref.artifact_type != CURRENT_POINTER_ARTIFACT_TYPE:
                raise PointerError("previous_pointer_ref must reference a current_pointer artifact")
            if self.previous_pointer_ref.artifact_id != self.pointer_id:
                raise PointerError("previous_pointer_ref must belong to the same pointer_id")
        if self.pointer_kind is PointerKind.CURRENT and self.authority_ref is not None:
            raise PointerError("CURRENT pointer must not have authority_ref")
        if self.pointer_kind is PointerKind.CURRENT_APPROVED and self.authority_ref is None:
            raise PointerError("CURRENT_APPROVED pointer requires authority_ref")

    def to_dict(self) -> dict[str, object]:
        return {
            "pointer_id": self.pointer_id,
            "pointer_kind": self.pointer_kind.value,
            "target_ref": self.target_ref.to_dict(),
            "authority_ref": None if self.authority_ref is None else self.authority_ref.to_dict(),
            "previous_pointer_ref": (
                None if self.previous_pointer_ref is None else self.previous_pointer_ref.to_dict()
            ),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CurrentPointer":
        require_exact_keys(
            value,
            {"pointer_id", "pointer_kind", "target_ref", "authority_ref", "previous_pointer_ref"},
            "CurrentPointer",
            PointerError,
        )
        try:
            return cls(
                pointer_id=value["pointer_id"],
                pointer_kind=value["pointer_kind"],
                target_ref=ArtifactRef.from_dict(value["target_ref"]),
                authority_ref=(
                    None
                    if value["authority_ref"] is None
                    else ArtifactRef.from_dict(value["authority_ref"])
                ),
                previous_pointer_ref=(
                    None
                    if value["previous_pointer_ref"] is None
                    else ArtifactRef.from_dict(value["previous_pointer_ref"])
                ),
            )
        except (TypeError, ValueError) as exc:
            raise PointerError(f"invalid CurrentPointer ArtifactRef: {exc}") from exc


@dataclass(frozen=True, slots=True)
class SupersessionRecord:
    pointer_id: str
    pointer_kind: PointerKind
    superseded_pointer_ref: ArtifactRef
    superseding_pointer_ref: ArtifactRef
    superseded_target_ref: ArtifactRef
    superseding_target_ref: ArtifactRef

    def __post_init__(self) -> None:
        require_storage_id(self.pointer_id, "pointer_id", SupersessionError)
        object.__setattr__(
            self, "pointer_kind", _coerce_pointer_kind(self.pointer_kind, SupersessionError)
        )
        for field_name in (
            "superseded_pointer_ref",
            "superseding_pointer_ref",
            "superseded_target_ref",
            "superseding_target_ref",
        ):
            require_artifact_ref(getattr(self, field_name), field_name, SupersessionError)
        if self.superseded_pointer_ref.artifact_type != CURRENT_POINTER_ARTIFACT_TYPE:
            raise SupersessionError("superseded_pointer_ref must reference current_pointer")
        if self.superseding_pointer_ref.artifact_type != CURRENT_POINTER_ARTIFACT_TYPE:
            raise SupersessionError("superseding_pointer_ref must reference current_pointer")
        if self.superseded_pointer_ref.artifact_id != self.pointer_id:
            raise SupersessionError("superseded_pointer_ref belongs to another pointer_id")
        if self.superseding_pointer_ref.artifact_id != self.pointer_id:
            raise SupersessionError("superseding_pointer_ref belongs to another pointer_id")
        if self.superseding_pointer_ref.revision <= self.superseded_pointer_ref.revision:
            raise SupersessionError(
                "superseding_pointer_ref revision must be greater than superseded_pointer_ref revision"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "pointer_id": self.pointer_id,
            "pointer_kind": self.pointer_kind.value,
            "superseded_pointer_ref": self.superseded_pointer_ref.to_dict(),
            "superseding_pointer_ref": self.superseding_pointer_ref.to_dict(),
            "superseded_target_ref": self.superseded_target_ref.to_dict(),
            "superseding_target_ref": self.superseding_target_ref.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SupersessionRecord":
        require_exact_keys(
            value,
            {
                "pointer_id",
                "pointer_kind",
                "superseded_pointer_ref",
                "superseding_pointer_ref",
                "superseded_target_ref",
                "superseding_target_ref",
            },
            "SupersessionRecord",
            SupersessionError,
        )
        try:
            return cls(
                pointer_id=value["pointer_id"],
                pointer_kind=value["pointer_kind"],
                superseded_pointer_ref=ArtifactRef.from_dict(value["superseded_pointer_ref"]),
                superseding_pointer_ref=ArtifactRef.from_dict(value["superseding_pointer_ref"]),
                superseded_target_ref=ArtifactRef.from_dict(value["superseded_target_ref"]),
                superseding_target_ref=ArtifactRef.from_dict(value["superseding_target_ref"]),
            )
        except (TypeError, ValueError) as exc:
            raise SupersessionError(f"invalid SupersessionRecord ArtifactRef: {exc}") from exc


class FilePointerStore:
    """File-backed current-pointer control plane with exact compare-and-swap semantics.

    The mutable head is only a locator for an immutable CurrentPointer snapshot. Current
    authority is never derived by scanning artifact revisions or timestamps. The root
    follows the same trusted-local-filesystem threat model as FileArtifactStore.
    """

    def __init__(self, root: str | os.PathLike[str], artifact_store: FileArtifactStore) -> None:
        if not isinstance(artifact_store, FileArtifactStore):
            raise PointerError("artifact_store must be a FileArtifactStore")
        raw_root = Path(root).expanduser()
        if raw_root.is_symlink():
            raise PointerError(f"pointer store root must not be a symlink: {raw_root}")
        raw_root.mkdir(parents=True, exist_ok=True)
        if raw_root.is_symlink():
            raise PointerError(f"pointer store root must not be a symlink: {raw_root}")
        self.root = raw_root.resolve()
        self.artifact_store = artifact_store
        self._heads_dir = self.root / "heads"
        self._locks_dir = self.root / "locks"
        self._heads_dir.mkdir(exist_ok=True)
        self._locks_dir.mkdir(exist_ok=True)
        for path, label in ((self._heads_dir, "heads"), (self._locks_dir, "locks")):
            if path.is_symlink() or not path.is_dir():
                raise PointerError(f"pointer store {label} path must be a real directory: {path}")

    def compare_and_set(
        self,
        *,
        pointer_id: str,
        pointer_kind: PointerKind,
        expected_pointer_ref: ArtifactRef | None,
        target_ref: ArtifactRef,
        authority_ref: ArtifactRef | None = None,
    ) -> ArtifactRef:
        pointer_id = require_storage_id(pointer_id, "pointer_id", PointerError)
        pointer_kind = _coerce_pointer_kind(pointer_kind)
        require_artifact_ref(target_ref, "target_ref", PointerError)
        if expected_pointer_ref is not None:
            require_artifact_ref(expected_pointer_ref, "expected_pointer_ref", PointerError)
            self._require_pointer_ref_identity(pointer_id, expected_pointer_ref)
        if authority_ref is not None:
            require_artifact_ref(authority_ref, "authority_ref", PointerError)

        with self._exclusive_lock(pointer_id):
            current_ref = self._read_head_ref(pointer_id, missing_ok=True)
            if current_ref != expected_pointer_ref:
                if self._is_exact_retry(
                    current_ref=current_ref,
                    expected_pointer_ref=expected_pointer_ref,
                    pointer_kind=pointer_kind,
                    target_ref=target_ref,
                    authority_ref=authority_ref,
                ):
                    return current_ref  # type: ignore[return-value]
                raise PointerConflictError(
                    f"stale pointer update for {pointer_id!r}: expected {expected_pointer_ref!r}, "
                    f"current is {current_ref!r}"
                )

            try:
                self.artifact_store.get_ref(target_ref)
            except ArtifactError as exc:
                raise PointerError(f"pointer target cannot be resolved exactly: {target_ref!r}") from exc

            if current_ref is not None:
                current_pointer = self._load_pointer_ref(current_ref)
                if current_pointer.pointer_kind is not pointer_kind:
                    raise PointerConflictError(
                        "pointer_kind cannot change for an existing pointer_id"
                    )
                if (
                    current_pointer.target_ref == target_ref
                    and current_pointer.authority_ref == authority_ref
                ):
                    return current_ref

            candidate = CurrentPointer(
                pointer_id=pointer_id,
                pointer_kind=pointer_kind,
                target_ref=target_ref,
                authority_ref=authority_ref,
                previous_pointer_ref=current_ref,
            )
            self._verify_pointer_authority(candidate)

            revision = self._next_pointer_revision(pointer_id, current_ref)
            envelope = ImmutableArtifactEnvelope.create(
                artifact_type=CURRENT_POINTER_ARTIFACT_TYPE,
                artifact_id=pointer_id,
                revision=revision,
                schema_version=POINTER_SCHEMA_VERSION,
                payload=candidate.to_dict(),
            )
            try:
                new_ref = self.artifact_store.put(envelope)
            except ArtifactError as exc:
                raise PointerError(f"failed to persist CurrentPointer snapshot: {exc}") from exc

            if current_ref is not None:
                previous = self._load_pointer_ref(current_ref)
                supersession = SupersessionRecord(
                    pointer_id=pointer_id,
                    pointer_kind=pointer_kind,
                    superseded_pointer_ref=current_ref,
                    superseding_pointer_ref=new_ref,
                    superseded_target_ref=previous.target_ref,
                    superseding_target_ref=target_ref,
                )
                supersession_envelope = ImmutableArtifactEnvelope.create(
                    artifact_type=SUPERSESSION_RECORD_ARTIFACT_TYPE,
                    artifact_id=pointer_id,
                    revision=revision,
                    schema_version=POINTER_SCHEMA_VERSION,
                    payload=supersession.to_dict(),
                )
                try:
                    self.artifact_store.put(supersession_envelope)
                except ArtifactError as exc:
                    raise SupersessionError(
                        f"failed to persist SupersessionRecord before head commit: {exc}"
                    ) from exc

            self._write_head(pointer_id, new_ref)
            return new_ref

    def resolve_current_pointer_ref(self, pointer_id: str) -> ArtifactRef:
        pointer_id = require_storage_id(pointer_id, "pointer_id", PointerError)
        ref = self._read_head_ref(pointer_id, missing_ok=False)
        assert ref is not None
        return ref

    def resolve_current(self, pointer_id: str) -> CurrentPointer:
        return self._load_pointer_ref(self.resolve_current_pointer_ref(pointer_id))

    def resolve_current_approved(self, pointer_id: str):
        """Return the exact ApprovalRef that authorizes the current-approved target."""

        pointer = self.resolve_current(pointer_id)
        if pointer.pointer_kind is not PointerKind.CURRENT_APPROVED:
            raise PointerError(f"pointer {pointer_id!r} is not CURRENT_APPROVED")
        assert pointer.authority_ref is not None
        from .approval import ApprovalRef, resolve_approval_ref

        approval_ref = ApprovalRef(
            target_ref=pointer.target_ref,
            approval_record_ref=pointer.authority_ref,
        )
        resolve_approval_ref(self.artifact_store, approval_ref)
        return approval_ref

    def is_current(self, pointer_id: str, artifact_ref: ArtifactRef) -> bool:
        require_artifact_ref(artifact_ref, "artifact_ref", PointerError)
        try:
            return self.resolve_current(pointer_id).target_ref == artifact_ref
        except PointerNotFoundError:
            return False

    def require_current(self, pointer_id: str, artifact_ref: ArtifactRef) -> CurrentPointer:
        require_artifact_ref(artifact_ref, "artifact_ref", PointerError)
        pointer = self.resolve_current(pointer_id)
        if pointer.target_ref != artifact_ref:
            raise PointerConflictError(
                f"artifact is not current for pointer {pointer_id!r}: {artifact_ref!r}"
            )
        return pointer

    def active_history(self, pointer_id: str) -> tuple[ArtifactRef, ...]:
        pointer_id = require_storage_id(pointer_id, "pointer_id", PointerError)
        current_ref = self.resolve_current_pointer_ref(pointer_id)
        history: list[ArtifactRef] = []
        seen: set[ArtifactRef] = set()
        ref: ArtifactRef | None = current_ref
        while ref is not None:
            if ref in seen:
                raise PointerIntegrityError(f"cycle detected in pointer history for {pointer_id!r}")
            seen.add(ref)
            pointer = self._load_pointer_ref(ref)
            history.append(ref)
            ref = pointer.previous_pointer_ref
        return tuple(history)

    def resolve_supersession(
        self,
        pointer_id: str,
        superseding_pointer_ref: ArtifactRef,
    ) -> SupersessionRecord:
        pointer_id = require_storage_id(pointer_id, "pointer_id", PointerError)
        self._require_pointer_ref_identity(pointer_id, superseding_pointer_ref)
        pointer = self._load_pointer_ref(superseding_pointer_ref)
        if pointer.previous_pointer_ref is None:
            raise SupersessionError("initial CurrentPointer has no SupersessionRecord")
        try:
            envelope = self.artifact_store.get(
                SUPERSESSION_RECORD_ARTIFACT_TYPE,
                pointer_id,
                superseding_pointer_ref.revision,
            )
        except ArtifactError as exc:
            raise SupersessionError("matching SupersessionRecord cannot be resolved") from exc
        if envelope.schema_version != POINTER_SCHEMA_VERSION:
            raise SupersessionError(
                "unsupported SupersessionRecord schema_version: "
                f"{envelope.schema_version}; supported={POINTER_SCHEMA_VERSION}"
            )
        payload = envelope.payload
        if not isinstance(payload, dict):
            raise SupersessionError("persisted SupersessionRecord payload must be an object")
        record = SupersessionRecord.from_dict(payload)
        previous = self._load_pointer_ref(record.superseded_pointer_ref)
        if (
            record.superseding_pointer_ref != superseding_pointer_ref
            or record.superseded_pointer_ref != pointer.previous_pointer_ref
            or record.pointer_kind is not pointer.pointer_kind
            or record.superseded_target_ref != previous.target_ref
            or record.superseding_target_ref != pointer.target_ref
        ):
            raise SupersessionError("SupersessionRecord does not match active pointer transition")
        return record

    def _is_exact_retry(
        self,
        *,
        current_ref: ArtifactRef | None,
        expected_pointer_ref: ArtifactRef | None,
        pointer_kind: PointerKind,
        target_ref: ArtifactRef,
        authority_ref: ArtifactRef | None,
    ) -> bool:
        if current_ref is None:
            return False
        current = self._load_pointer_ref(current_ref)
        return (
            current.previous_pointer_ref == expected_pointer_ref
            and current.pointer_kind is pointer_kind
            and current.target_ref == target_ref
            and current.authority_ref == authority_ref
        )

    def _next_pointer_revision(
        self,
        pointer_id: str,
        current_ref: ArtifactRef | None,
    ) -> int:
        revision = 1 if current_ref is None else current_ref.revision + 1
        while True:
            try:
                self.artifact_store.get(CURRENT_POINTER_ARTIFACT_TYPE, pointer_id, revision)
            except ArtifactNotFoundError:
                return revision
            except ArtifactError as exc:
                raise PointerIntegrityError(
                    f"cannot inspect pointer revision {revision} for allocation"
                ) from exc
            revision += 1

    def _decode_pointer_snapshot(self, ref: ArtifactRef) -> CurrentPointer:
        self._require_pointer_ref_identity(ref.artifact_id, ref)
        try:
            envelope = self.artifact_store.get_ref(ref)
        except ArtifactError as exc:
            raise PointerIntegrityError(f"failed to resolve CurrentPointer: {ref!r}") from exc
        if envelope.schema_version != POINTER_SCHEMA_VERSION:
            raise PointerIntegrityError(
                "unsupported CurrentPointer schema_version: "
                f"{envelope.schema_version}; supported={POINTER_SCHEMA_VERSION}"
            )
        payload = envelope.payload
        if not isinstance(payload, dict):
            raise PointerIntegrityError("persisted CurrentPointer payload must be an object")
        try:
            pointer = CurrentPointer.from_dict(payload)
        except PointerError as exc:
            raise PointerIntegrityError(f"invalid persisted CurrentPointer: {exc}") from exc
        if pointer.pointer_id != ref.artifact_id:
            raise PointerIntegrityError("CurrentPointer payload pointer_id mismatches artifact identity")
        return pointer

    def _load_pointer_ref(self, ref: ArtifactRef) -> CurrentPointer:
        pointer = self._decode_pointer_snapshot(ref)
        if pointer.previous_pointer_ref is not None:
            if pointer.previous_pointer_ref.revision >= ref.revision:
                raise PointerIntegrityError(
                    "CurrentPointer previous_pointer_ref must reference an earlier pointer revision"
                )
            previous = self._load_pointer_ref(pointer.previous_pointer_ref)
            if previous.pointer_kind is not pointer.pointer_kind:
                raise PointerIntegrityError(
                    "CurrentPointer previous_pointer_ref must preserve pointer_kind"
                )
        try:
            self.artifact_store.get_ref(pointer.target_ref)
        except ArtifactError as exc:
            raise PointerIntegrityError(
                f"CurrentPointer target does not resolve exactly: {pointer.target_ref!r}"
            ) from exc
        self._verify_pointer_authority(pointer)
        return pointer

    @staticmethod
    def _require_pointer_ref_identity(pointer_id: str, ref: ArtifactRef) -> None:
        if ref.artifact_type != CURRENT_POINTER_ARTIFACT_TYPE or ref.artifact_id != pointer_id:
            raise PointerError(
                f"pointer ref must target {CURRENT_POINTER_ARTIFACT_TYPE}/{pointer_id}"
            )

    def _verify_pointer_authority(self, pointer: CurrentPointer) -> None:
        if pointer.pointer_kind is PointerKind.CURRENT:
            if pointer.authority_ref is not None:
                raise PointerIntegrityError("CURRENT pointer unexpectedly has authority_ref")
            return
        assert pointer.authority_ref is not None
        from .approval import ApprovalRef, resolve_approval_ref

        try:
            resolve_approval_ref(
                self.artifact_store,
                ApprovalRef(
                    target_ref=pointer.target_ref,
                    approval_record_ref=pointer.authority_ref,
                ),
            )
        except Exception as exc:
            raise PointerIntegrityError(
                "CURRENT_APPROVED pointer authority does not resolve to a valid APPROVE record"
            ) from exc

    def _head_path(self, pointer_id: str) -> Path:
        return self._heads_dir / f"{pointer_id}.json"

    def _lock_path(self, pointer_id: str) -> Path:
        return self._locks_dir / f"{pointer_id}.lock"

    def _read_head_ref(self, pointer_id: str, *, missing_ok: bool) -> ArtifactRef | None:
        path = self._head_path(pointer_id)
        if path.is_symlink():
            raise PointerIntegrityError(f"pointer head must not be a symlink: {path}")
        if not path.exists():
            if missing_ok:
                return None
            raise PointerNotFoundError(f"current pointer does not exist: {pointer_id}")
        if not path.is_file():
            raise PointerIntegrityError(f"pointer head is not a file: {path}")
        try:
            raw = path.read_bytes()
            value = strict_json_loads(raw)
            if canonical_json_bytes(value) != raw:
                raise PointerIntegrityError("pointer head is not canonical JSON")
            require_exact_keys(
                value,
                {"current_pointer_ref"},
                "PointerHead",
                PointerIntegrityError,
            )
            ref = ArtifactRef.from_dict(value["current_pointer_ref"])
            self._require_pointer_ref_identity(pointer_id, ref)
            self._load_pointer_ref(ref)
            return ref
        except PointerError:
            raise
        except Exception as exc:
            raise PointerIntegrityError(f"invalid pointer head for {pointer_id!r}: {exc}") from exc

    def _write_head(self, pointer_id: str, ref: ArtifactRef) -> None:
        path = self._head_path(pointer_id)
        if path.is_symlink():
            raise PointerIntegrityError(f"pointer head must not be a symlink: {path}")
        data = canonical_json_bytes({"current_pointer_ref": ref.to_dict()})
        temp_path: Path | None = None
        try:
            fd, raw_temp = tempfile.mkstemp(prefix=f".{pointer_id}.", suffix=".tmp", dir=self._heads_dir)
            temp_path = Path(raw_temp)
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, path)
            temp_path = None
            self._fsync_directory(self._heads_dir)
        except OSError as exc:
            raise PointerError(f"failed to atomically update pointer head: {exc}") from exc
        finally:
            if temp_path is not None:
                try:
                    temp_path.unlink(missing_ok=True)
                except OSError:
                    pass

    @contextmanager
    def _exclusive_lock(self, pointer_id: str) -> Iterator[None]:
        path = self._lock_path(pointer_id)
        if path.is_symlink():
            raise PointerLockedError(f"pointer lock path must not be a symlink: {path}")
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
        try:
            fd = os.open(path, flags, 0o600)
        except FileExistsError as exc:
            raise PointerLockedError(
                f"pointer {pointer_id!r} is locked; stale locks are not auto-reclaimed"
            ) from exc
        except OSError as exc:
            raise PointerError(f"failed to acquire pointer lock: {exc}") from exc
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(f"pid={os.getpid()}\n")
                handle.flush()
                os.fsync(handle.fileno())
            yield
        finally:
            try:
                path.unlink(missing_ok=True)
            except OSError as exc:
                raise PointerError(f"failed to release pointer lock: {exc}") from exc

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        try:
            fd = os.open(path, os.O_RDONLY)
        except OSError:
            return
        try:
            try:
                os.fsync(fd)
            except OSError:
                pass
        finally:
            os.close(fd)
