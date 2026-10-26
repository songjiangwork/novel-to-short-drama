"""Private, atomic storage for one verified B1.3 window checkpoint.

The configured root is an operator-controlled private directory, separate from
the A5/A6 artifact and pointer trees.  This module rejects symlink roots and
targets, but does not defend against an adversary concurrently changing trusted
parent directories or a hostile filesystem.  It verifies bytes against the
caller-supplied session only; that is not authorization to resume execution.
"""
from __future__ import annotations

import os
import re
import stat
import tempfile
from pathlib import Path
from typing import Any

from . import disposition_checkpoint as evidence
from .disposition_execution import B13WindowAcceptedProposal


_HASH = re.compile(r"^[0-9a-f]{64}$")
_FILE_PREFIX = "b13-checkpoint-v1"


class B13CheckpointStorageError(RuntimeError):
    """A checkpoint filesystem operation was unsafe, corrupt, or not durable."""


def _fail(message: str, exc: BaseException | None = None) -> None:
    if exc is None:
        raise B13CheckpointStorageError(message)
    raise B13CheckpointStorageError(message) from exc


def _lstat(path: Path, *, missing_ok: bool = False) -> os.stat_result | None:
    try:
        return os.lstat(path)
    except FileNotFoundError:
        if missing_ok:
            return None
        _fail(f"checkpoint path disappeared: {path}")
    except OSError as exc:
        _fail(f"cannot inspect checkpoint path: {path}", exc)


def _root(root: Path | str | os.PathLike[str], *, create: bool) -> Path | None:
    path = Path(root).expanduser()
    before = _lstat(path, missing_ok=True)
    if before is None:
        if not create:
            return None
        try:
            path.mkdir(parents=True, mode=0o700)
        except OSError as exc:
            _fail(f"cannot create checkpoint root: {path}", exc)
    elif stat.S_ISLNK(before.st_mode):
        _fail(f"checkpoint root must not be a symlink: {path}")
    current = _lstat(path)
    assert current is not None
    if stat.S_ISLNK(current.st_mode):
        _fail(f"checkpoint root must not be a symlink: {path}")
    if not stat.S_ISDIR(current.st_mode):
        _fail(f"checkpoint root is not a directory: {path}")
    return path.resolve()


def _session_window(session: Any, window_id: str):
    """Use the codec's complete trusted-session gate before deriving a path."""
    try:
        preparation, semantics, _runtime = evidence._trusted_session(session)
    except evidence.B13CheckpointEvidenceError:
        raise
    except Exception as exc:  # defensive: never turn malformed session into a path
        _fail("checkpoint session is invalid", exc)
    if not isinstance(window_id, str):
        _fail("checkpoint window id must be a string")
    matches = [(packet, semantic) for packet, semantic in zip(preparation.packets, semantics)
               if packet.window_id == window_id]
    if len(matches) != 1:
        _fail("checkpoint window does not identify exactly one trusted session window")
    packet, semantic = matches[0]
    preparation_identity = preparation.preparation_identity
    request_identity = semantic.request_identity
    if (not isinstance(preparation_identity, str) or not isinstance(request_identity, str)
            or _HASH.fullmatch(preparation_identity) is None
            or _HASH.fullmatch(request_identity) is None):
        _fail("checkpoint path identity is not a SHA-256 hash")
    if (isinstance(packet.window_ordinal, bool) or not isinstance(packet.window_ordinal, int)
            or not 1 <= packet.window_ordinal <= 99_999_999):
        _fail("checkpoint window ordinal cannot be encoded safely")
    return preparation_identity, packet.window_ordinal, request_identity


def checkpoint_path(root: Path | str | os.PathLike[str], session: Any, window_id: str) -> Path:
    """Return the only permitted path for a trusted session-window identity.

    This creates the private root when necessary; callers cannot supply a
    filename.  The filename includes full preparation and semantic-request
    identities, so distinct preparations or requests never alias.
    """
    directory = _root(root, create=True)
    assert directory is not None
    preparation, ordinal, request = _session_window(session, window_id)
    return directory / f"{_FILE_PREFIX}-p{preparation}-w{ordinal:08d}-q{request}.json"


def _target_stat(path: Path, *, missing_ok: bool = False) -> os.stat_result | None:
    value = _lstat(path, missing_ok=missing_ok)
    if value is None:
        return None
    if stat.S_ISLNK(value.st_mode):
        _fail(f"checkpoint target must not be a symlink: {path}")
    if not stat.S_ISREG(value.st_mode):
        _fail(f"checkpoint target must be a regular file: {path}")
    return value


def _same_file(left: os.stat_result, right: os.stat_result) -> bool:
    return (left.st_dev, left.st_ino, stat.S_IFMT(left.st_mode)) == (
        right.st_dev, right.st_ino, stat.S_IFMT(right.st_mode))


def _stable_file(left: os.stat_result, right: os.stat_result) -> bool:
    return _same_file(left, right) and (left.st_size, left.st_mtime_ns, left.st_ctime_ns) == (
        right.st_size, right.st_mtime_ns, right.st_ctime_ns)


def _read_bounded(path: Path) -> bytes:
    before = _target_stat(path)
    assert before is not None
    if before.st_size > evidence.B13_CHECKPOINT_MAX_BYTES:
        _fail("checkpoint file exceeds the maximum size")
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        _fail("platform cannot open checkpoint files without following symlinks")
    flags = os.O_RDONLY | nofollow | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        _fail(f"cannot safely open checkpoint file: {path}", exc)
    try:
        opened = os.fstat(fd)
        if not stat.S_ISREG(opened.st_mode) or not _same_file(before, opened):
            _fail("checkpoint target changed or is not a regular file")
        if opened.st_size > evidence.B13_CHECKPOINT_MAX_BYTES:
            _fail("checkpoint file exceeds the maximum size")
        parts: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(fd, min(65_536, evidence.B13_CHECKPOINT_MAX_BYTES + 1 - total))
            if not chunk:
                break
            total += len(chunk)
            if total > evidence.B13_CHECKPOINT_MAX_BYTES:
                _fail("checkpoint file exceeds the maximum size")
            parts.append(chunk)
        after = os.fstat(fd)
        if not _stable_file(opened, after):
            _fail("checkpoint file changed while being read")
        return b"".join(parts)
    except OSError as exc:
        _fail(f"cannot read checkpoint file: {path}", exc)
    finally:
        try:
            os.close(fd)
        except OSError:
            pass


def _fsync_directory(directory: Path) -> None:
    try:
        fd = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError as exc:
        _fail("cannot open checkpoint directory for durability sync", exc)
    try:
        os.fsync(fd)
    except OSError as exc:
        _fail("cannot fsync checkpoint directory", exc)
    finally:
        os.close(fd)


def _write_all(fd: int, raw: bytes) -> None:
    view = memoryview(raw)
    while view:
        try:
            written = os.write(fd, view)
        except OSError as exc:
            _fail("cannot write temporary checkpoint", exc)
        if not isinstance(written, int) or not 0 < written <= len(view):
            _fail("temporary checkpoint write was short")
        view = view[written:]


def _existing_matches(path: Path, raw: bytes, session: Any) -> bool:
    try:
        existing = _read_bounded(path)
        return existing == raw and evidence.verify_and_decode_checkpoint(existing, session) is not None
    except (B13CheckpointStorageError, evidence.B13CheckpointEvidenceError) as exc:
        _fail("existing checkpoint is not the same verified checkpoint", exc)


def save_checkpoint(root: Path | str | os.PathLike[str], accepted: B13WindowAcceptedProposal, session: Any) -> Path:
    """Durably save one accepted window without replacing an existing file.

    Canonical evidence is verified in memory before it reaches persistent
    storage.  Publication is a same-directory hard link, followed by a required
    directory fsync; existing byte-identical verified evidence is idempotent.
    """
    try:
        raw = evidence.encode_checkpoint(accepted, session)
        decoded = evidence.verify_and_decode_checkpoint(raw, session)
    except evidence.B13CheckpointEvidenceError:
        raise
    if decoded != accepted or len(raw) > evidence.B13_CHECKPOINT_MAX_BYTES:
        _fail("checkpoint evidence did not round-trip exactly before persistence")
    target = checkpoint_path(root, session, accepted.window_id)
    if _target_stat(target, missing_ok=True) is not None:
        if _existing_matches(target, raw, session):
            _fsync_directory(target.parent)
            return target
        _fail("conflicting checkpoint already exists")
    temporary: Path | None = None
    try:
        fd, name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
        temporary = Path(name)
        try:
            os.fchmod(fd, 0o600)
            _write_all(fd, raw)
            os.fsync(fd)
        finally:
            os.close(fd)
        try:
            os.link(temporary, target)
        except FileExistsError:
            if _existing_matches(target, raw, session):
                _fsync_directory(target.parent)
                return target
            _fail("conflicting checkpoint appeared during atomic publish")
        _fsync_directory(target.parent)
        return target
    except B13CheckpointStorageError:
        raise
    except OSError as exc:
        _fail("cannot atomically persist checkpoint", exc)
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def load_checkpoint(root: Path | str | os.PathLike[str], session: Any, window_id: str) -> B13WindowAcceptedProposal | None:
    """Load exactly one expected checkpoint, or ``None`` only when it is absent."""
    directory = _root(root, create=False)
    _session_window(session, window_id)
    if directory is None:
        return None
    target = directory / checkpoint_path(directory, session, window_id).name
    if _target_stat(target, missing_ok=True) is None:
        return None
    raw = _read_bounded(target)
    try:
        decoded = evidence.verify_and_decode_checkpoint(raw, session)
    except evidence.B13CheckpointEvidenceError:
        raise
    if decoded.window_id != window_id:
        _fail("checkpoint decoded to a different window")
    return decoded


__all__ = [
    "B13CheckpointStorageError", "checkpoint_path", "save_checkpoint", "load_checkpoint",
]
