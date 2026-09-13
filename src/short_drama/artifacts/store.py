from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path

from .canonical import canonical_json_bytes, strict_json_loads
from .errors import (
    ArtifactConflictError,
    ArtifactIntegrityError,
    ArtifactNotFoundError,
    ArtifactPathError,
    ArtifactStoreError,
    ArtifactValidationError,
    CanonicalSerializationError,
)
from .models import ArtifactRef, ImmutableArtifactEnvelope

_SAFE_COMPONENT_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
_WINDOWS_RESERVED_BASENAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


class FileArtifactStore:
    """Lightweight immutable filesystem store for canonical v1.2 artifacts.

    Security boundary: the configured store root and its parent path are expected to be
    controlled by trusted code/users and must not be concurrently rewritten by an
    untrusted process. The store rejects symlinked roots, artifact directories, and
    revision targets, but it is not intended to sandbox a hostile shared filesystem.

    Storage path components are deliberately restricted to lowercase ASCII so artifact
    identities map one-to-one on both case-sensitive and case-insensitive filesystems.
    """

    def __init__(self, root: str | os.PathLike[str]) -> None:
        raw_root = Path(root).expanduser()
        if raw_root.is_symlink():
            raise ArtifactPathError(f"artifact store root must not be a symlink: {raw_root}")
        raw_root.mkdir(parents=True, exist_ok=True)
        if raw_root.is_symlink():
            raise ArtifactPathError(f"artifact store root must not be a symlink: {raw_root}")
        self.root = raw_root.resolve()
        if not self.root.is_dir():
            raise ArtifactStoreError(f"artifact store root is not a directory: {self.root}")

    @staticmethod
    def _safe_component(value: str, field_name: str) -> str:
        if not isinstance(value, str) or _SAFE_COMPONENT_RE.fullmatch(value) is None:
            raise ArtifactPathError(
                f"{field_name} must match {_SAFE_COMPONENT_RE.pattern!r} for filesystem storage"
            )
        if value in {".", ".."} or value.endswith("."):
            raise ArtifactPathError(f"unsafe {field_name}: {value!r}")
        windows_basename = value.split(".", 1)[0].upper()
        if windows_basename in _WINDOWS_RESERVED_BASENAMES:
            raise ArtifactPathError(
                f"{field_name} uses a Windows-reserved filesystem name: {value!r}"
            )
        return value

    @staticmethod
    def _reject_symlink(path: Path, field_name: str) -> None:
        if path.is_symlink():
            raise ArtifactPathError(f"{field_name} must not be a symlink: {path}")

    def _path(self, artifact_type: str, artifact_id: str, revision: int) -> Path:
        safe_type = self._safe_component(artifact_type, "artifact_type")
        safe_id = self._safe_component(artifact_id, "artifact_id")
        if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
            raise ArtifactPathError("revision must be an integer >= 1")

        type_dir = self.root / safe_type
        id_dir = type_dir / safe_id
        path = id_dir / f"r{revision:08d}.json"
        self._reject_symlink(type_dir, "artifact_type directory")
        self._reject_symlink(id_dir, "artifact_id directory")
        self._reject_symlink(path, "artifact revision target")

        try:
            path.resolve().relative_to(self.root)
        except ValueError as exc:
            raise ArtifactPathError("artifact path escapes configured store root") from exc
        return path

    def put(self, envelope: ImmutableArtifactEnvelope) -> ArtifactRef:
        if not isinstance(envelope, ImmutableArtifactEnvelope):
            raise ArtifactStoreError("put() requires an ImmutableArtifactEnvelope")

        target = self._path(envelope.artifact_type, envelope.artifact_id, envelope.revision)
        type_dir = target.parent.parent
        id_dir = target.parent
        type_dir.mkdir(exist_ok=True)
        self._reject_symlink(type_dir, "artifact_type directory")
        if not type_dir.is_dir():
            raise ArtifactPathError(f"artifact_type storage path is not a directory: {type_dir}")
        id_dir.mkdir(exist_ok=True)
        self._reject_symlink(id_dir, "artifact_id directory")
        if not id_dir.is_dir():
            raise ArtifactPathError(f"artifact_id storage path is not a directory: {id_dir}")

        # Revalidate after directory creation. The trusted-root contract above excludes a
        # hostile concurrent rename/symlink swap between this check and subsequent I/O.
        target = self._path(envelope.artifact_type, envelope.artifact_id, envelope.revision)
        data = envelope.canonical_bytes()

        if target.exists():
            return self._resolve_existing_put(target, envelope, data)

        temp_path: Path | None = None
        try:
            fd, raw_temp = tempfile.mkstemp(
                prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
            )
            temp_path = Path(raw_temp)
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())

            os.link(temp_path, target)
            self._fsync_directory(target.parent)
            return envelope.ref
        except FileExistsError:
            return self._resolve_existing_put(target, envelope, data)
        except OSError as exc:
            raise ArtifactStoreError(f"failed to atomically persist artifact: {exc}") from exc
        finally:
            if temp_path is not None:
                try:
                    temp_path.unlink(missing_ok=True)
                except OSError:
                    pass

    def get(self, artifact_type: str, artifact_id: str, revision: int) -> ImmutableArtifactEnvelope:
        path = self._path(artifact_type, artifact_id, revision)
        if not path.is_file():
            raise ArtifactNotFoundError(
                f"artifact not found: {artifact_type}/{artifact_id} revision {revision}"
            )
        envelope = self._read_verified(path)
        expected_identity = (artifact_type, artifact_id, revision)
        actual_identity = (envelope.artifact_type, envelope.artifact_id, envelope.revision)
        if actual_identity != expected_identity:
            raise ArtifactIntegrityError(
                f"persisted artifact identity mismatch: expected {expected_identity!r}, "
                f"got {actual_identity!r}"
            )
        return envelope

    def get_ref(self, ref: ArtifactRef) -> ImmutableArtifactEnvelope:
        if not isinstance(ref, ArtifactRef):
            raise ArtifactStoreError("get_ref() requires an ArtifactRef")
        envelope = self.get(ref.artifact_type, ref.artifact_id, ref.revision)
        if envelope.ref != ref:
            raise ArtifactIntegrityError(
                "persisted artifact does not match requested ArtifactRef content_hash"
            )
        return envelope

    def _resolve_existing_put(
        self,
        target: Path,
        envelope: ImmutableArtifactEnvelope,
        data: bytes,
    ) -> ArtifactRef:
        try:
            existing = self._read_verified(target)
        except ArtifactStoreError as exc:
            raise ArtifactConflictError(
                f"immutable target already exists but is not the same valid artifact: {target}"
            ) from exc
        if existing.ref == envelope.ref and existing.canonical_bytes() == data:
            return existing.ref
        raise ArtifactConflictError(
            "conflicting content already exists for immutable artifact identity/revision"
        )

    @staticmethod
    def _read_verified(path: Path) -> ImmutableArtifactEnvelope:
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise ArtifactStoreError(f"failed to read artifact: {path}") from exc
        try:
            value = strict_json_loads(raw)
            if not isinstance(value, dict):
                raise ArtifactIntegrityError("persisted artifact envelope must be a JSON object")
            if canonical_json_bytes(value) != raw:
                raise ArtifactIntegrityError("persisted artifact envelope is not canonical JSON")
            return ImmutableArtifactEnvelope.from_dict(value)
        except ArtifactIntegrityError:
            raise
        except (CanonicalSerializationError, ArtifactValidationError, TypeError, ValueError) as exc:
            raise ArtifactIntegrityError(
                f"persisted artifact failed integrity verification: {exc}"
            ) from exc

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
