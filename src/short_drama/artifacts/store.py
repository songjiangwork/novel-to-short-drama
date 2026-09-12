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

_SAFE_COMPONENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class FileArtifactStore:
    """Lightweight immutable filesystem store for canonical v1.2 artifacts."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        if not self.root.is_dir():
            raise ArtifactStoreError(f"artifact store root is not a directory: {self.root}")

    @staticmethod
    def _safe_component(value: str, field_name: str) -> str:
        if not isinstance(value, str) or _SAFE_COMPONENT_RE.fullmatch(value) is None:
            raise ArtifactPathError(
                f"{field_name} must match {_SAFE_COMPONENT_RE.pattern!r} for filesystem storage"
            )
        if value in {".", ".."}:
            raise ArtifactPathError(f"unsafe {field_name}: {value!r}")
        return value

    def _path(self, artifact_type: str, artifact_id: str, revision: int) -> Path:
        safe_type = self._safe_component(artifact_type, "artifact_type")
        safe_id = self._safe_component(artifact_id, "artifact_id")
        if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
            raise ArtifactPathError("revision must be an integer >= 1")
        path = self.root / safe_type / safe_id / f"r{revision:08d}.json"
        try:
            path.resolve().relative_to(self.root)
        except ValueError as exc:  # Defensive; safe components should make this unreachable.
            raise ArtifactPathError("artifact path escapes configured store root") from exc
        return path

    def put(self, envelope: ImmutableArtifactEnvelope) -> ArtifactRef:
        if not isinstance(envelope, ImmutableArtifactEnvelope):
            raise ArtifactStoreError("put() requires an ImmutableArtifactEnvelope")
        target = self._path(envelope.artifact_type, envelope.artifact_id, envelope.revision)
        target.parent.mkdir(parents=True, exist_ok=True)
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

            # Hard-link publication is an atomic no-replace operation on the same filesystem.
            # It therefore cannot silently clobber an immutable revision in a writer race.
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
        # Directory fsync improves durability on POSIX. Some platforms/filesystems do not
        # support it; publication is already atomic, so lack of directory fsync is non-fatal.
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
