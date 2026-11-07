"""Private, atomic storage for one verified B1.3 window checkpoint.

The configured root is an operator-controlled private directory, separate from
the A5/A6 artifact and pointer trees.  This module rejects symlink roots and
targets, but does not defend against an adversary concurrently changing trusted
parent directories or a hostile filesystem.  It verifies bytes against the
caller-supplied session only; that is not authorization to resume execution.
The root parent must already be a trusted non-symlink directory; only the final
root component may be created, and that parent is fsynced before save succeeds.
"""
from __future__ import annotations

import os
import re
import secrets
import stat
import tempfile
from contextlib import contextmanager
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
        parent = path.parent
        parent_stat = _lstat(parent, missing_ok=True)
        if parent_stat is None or stat.S_ISLNK(parent_stat.st_mode) or not stat.S_ISDIR(parent_stat.st_mode):
            _fail(f"checkpoint root parent must be an existing non-symlink directory: {parent}")
        try:
            os.mkdir(path, mode=0o700)
        except FileExistsError:
            pass
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
    if before is None:
        _fsync_directory(path.parent)
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
    return directory / _checkpoint_name(session, window_id)


def _checkpoint_name(session: Any, window_id: str) -> str:
    preparation, ordinal, request = _session_window(session, window_id)
    return f"{_FILE_PREFIX}-p{preparation}-w{ordinal:08d}-q{request}.json"


def _related_checkpoint_names(directory: Path, preparation: str, ordinal: int) -> tuple[str, ...]:
    """Find all attributable version namespaces, including malformed names.

    This is a point-in-time observation, not a lock against older writers.
    """
    marker = f"-p{preparation}-w{ordinal:08d}"
    try:
        return tuple(sorted(name for name in os.listdir(directory)
                            if name.startswith("b13-checkpoint-") and marker in name))
    except OSError as exc:
        _fail("cannot inspect checkpoint directory", exc)


def _reject_other_window_checkpoint(target: Path) -> None:
    # Never treat another version or request for the same window as absent.
    try:
        name = target.name
        if not name.startswith(("b13-checkpoint-v1-p", "b13-checkpoint-v2-p")):
            raise ValueError
        preparation, rest = name.split("-p", 1)[1].split("-w", 1)
        ordinal = int(rest.split("-q", 1)[0])
        if _HASH.fullmatch(preparation) is None:
            raise ValueError
        if any(item != name for item in _related_checkpoint_names(target.parent, preparation, ordinal)):
            _fail("checkpoint for another request exists for this window")
    except (ValueError, IndexError):
        _fail("invalid checkpoint target identity")


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
        if not stat.S_ISREG(opened.st_mode) or not _stable_file(before, opened):
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
        if total != opened.st_size:
            _fail("checkpoint file is incomplete")
        after = os.fstat(fd)
        if not _stable_file(opened, after):
            _fail("checkpoint file changed while being read")
        named = _target_stat(path)
        if named is None or not _stable_file(after, named):
            _fail("checkpoint target changed while being read")
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


@contextmanager
def _staged_checkpoint(target: Path, raw: bytes):
    """Stage bytes in the target directory; the caller owns validation/publish.

    This private primitive is not a checkpoint save API or authorization.
    """
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
        yield temporary
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


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
    _reject_other_window_checkpoint(target)
    if _target_stat(target, missing_ok=True) is not None:
        if _existing_matches(target, raw, session):
            _fsync_directory(target.parent)
            _reject_other_window_checkpoint(target)
            return target
        _fail("conflicting checkpoint already exists")
    try:
        with _staged_checkpoint(target, raw) as temporary:
            try:
                _reject_other_window_checkpoint(target)
                os.link(temporary, target)
            except FileExistsError:
                if _existing_matches(target, raw, session):
                    _fsync_directory(target.parent)
                    _reject_other_window_checkpoint(target)
                    return target
                _fail("conflicting checkpoint appeared during atomic publish")
            _fsync_directory(target.parent)
            _reject_other_window_checkpoint(target)
            return target
    except B13CheckpointStorageError:
        raise
    except OSError as exc:
        _fail("cannot atomically persist checkpoint", exc)


@contextmanager
def _private_v2_directory(directory: Path):
    """Pin one owner-only directory inode for all v2 filesystem operations."""
    before = _lstat(directory)
    if (before is None or not stat.S_ISDIR(before.st_mode)
            or before.st_uid != os.geteuid() or stat.S_IMODE(before.st_mode) != 0o700):
        _fail("v2 checkpoint root must be a private owned directory")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    nofollow = getattr(os, "O_NOFOLLOW", None)
    directory_flag = getattr(os, "O_DIRECTORY", None)
    if nofollow is None or directory_flag is None:
        _fail("platform cannot safely pin v2 checkpoint directory")
    try:
        fd = os.open(directory, flags | nofollow | directory_flag)
    except OSError as exc:
        _fail("cannot safely open v2 checkpoint directory", exc)
    try:
        _verify_private_v2_directory(directory, fd)
        if not _same_file(before, os.fstat(fd)):
            _fail("v2 checkpoint root changed while being opened")
        yield fd
        _verify_private_v2_directory(directory, fd)
    finally:
        os.close(fd)


def _verify_private_v2_directory(directory: Path, fd: int) -> None:
    try:
        by_name, opened = os.lstat(directory), os.fstat(fd)
    except OSError as exc:
        _fail("v2 checkpoint root disappeared", exc)
    if (not _same_file(by_name, opened) or not stat.S_ISDIR(opened.st_mode)
            or not stat.S_ISDIR(by_name.st_mode)
            or by_name.st_uid != os.geteuid() or opened.st_uid != os.geteuid()
            or stat.S_IMODE(by_name.st_mode) != 0o700
            or stat.S_IMODE(opened.st_mode) != 0o700):
        _fail("v2 checkpoint root identity or confidentiality changed")


def _related_v2_names_at(fd: int, target_name: str) -> tuple[str, ...]:
    """Reject every other checkpoint namespace for the same window and request."""
    try:
        if not target_name.startswith("b13-checkpoint-v2-p"):
            raise ValueError
        preparation, rest = target_name.split("-p", 1)[1].split("-w", 1)
        ordinal = int(rest.split("-q", 1)[0])
        if _HASH.fullmatch(preparation) is None or ordinal < 1:
            raise ValueError
        marker = f"-p{preparation}-w{ordinal:08d}"
        related = tuple(sorted(item for item in os.listdir(fd)
                               if item.startswith("b13-checkpoint-") and marker in item))
    except (ValueError, OSError) as exc:
        _fail("cannot inspect v2 checkpoint namespace", exc)
    if any(item != target_name for item in related):
        _fail("v2 checkpoint conflicts with another version or request")
    return related


def _read_private_v2_at(fd: int, name: str) -> bytes:
    """Bounded nofollow read with inode, byte-size and privacy revalidation."""
    try:
        before = os.stat(name, dir_fd=fd, follow_symlinks=False)
        if (not stat.S_ISREG(before.st_mode) or before.st_uid != os.geteuid()
                or stat.S_IMODE(before.st_mode) != 0o600 or before.st_nlink != 1
                or before.st_size > evidence.B13_CHECKPOINT_MAX_BYTES):
            raise ValueError
        flags = (os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
                 | getattr(os, "O_NONBLOCK", 0))
        file_fd = os.open(name, flags, dir_fd=fd)
        try:
            opened = os.fstat(file_fd)
            if not _stable_file(before, opened):
                raise ValueError
            parts = []
            total = 0
            while True:
                chunk = os.read(file_fd, min(65_536, evidence.B13_CHECKPOINT_MAX_BYTES + 1 - total))
                if not chunk:
                    break
                parts.append(chunk)
                total += len(chunk)
                if total > evidence.B13_CHECKPOINT_MAX_BYTES:
                    raise ValueError
            if total != opened.st_size or not _stable_file(opened, os.fstat(file_fd)):
                raise ValueError
        finally:
            os.close(file_fd)
        after = os.stat(name, dir_fd=fd, follow_symlinks=False)
        if not _stable_file(before, after) or after.st_nlink != 1:
            raise ValueError
        return b"".join(parts)
    except (OSError, ValueError) as exc:
        _fail("v2 checkpoint failed private read-back", exc)


def confirm_feedback_checkpoint_directory(root: Path | str | os.PathLike[str]) -> None:
    """Durably confirm any visible v2 checkpoint before accepting its reuse.

    A previous post-link directory fsync may have failed after exposing a
    valid file. This guarded directory fsync closes that indeterminate state
    before the caller can report verified reuse or COMPLETE.
    """
    directory = Path(root).expanduser()
    with _private_v2_directory(directory) as fd:
        try:
            os.fsync(fd)
        except OSError as exc:
            _fail("cannot confirm v2 checkpoint directory durability", exc)
        _verify_private_v2_directory(directory, fd)


def save_feedback_checkpoint_v2(root: Path | str | os.PathLike[str],
                                accepted: B13WindowAcceptedProposal, session: Any) -> Path:
    """Publish verified feedback evidence through one pinned private directory.

    Requires a caller-held execution guard and an approved retention policy.
    Once staging starts all file operations use dir_fd; never follow a swapped
    directory symlink or accept an orphaned temporary hard link as success.
    """
    from .disposition_checkpoint_v2 import (
        DEFAULT_EVIDENCE_LIMIT, encode_verified_feedback_checkpoint_v2,
        verify_and_decode_feedback_checkpoint_v2,
    )
    try:
        if accepted.feedback_evidence is None:
            raise ValueError
        raw = encode_verified_feedback_checkpoint_v2(accepted, session)
        if (len(raw) > DEFAULT_EVIDENCE_LIMIT
                or verify_and_decode_feedback_checkpoint_v2(raw, session) != accepted):
            raise ValueError
    except Exception:
        _fail("v2 checkpoint evidence is invalid")

    legacy = checkpoint_path(root, session, accepted.window_id)
    target = legacy.with_name(legacy.name.replace("b13-checkpoint-v1-", "b13-checkpoint-v2-", 1))
    directory, name = target.parent, target.name
    with _private_v2_directory(directory) as dirfd:
        def verify_published() -> None:
            _verify_private_v2_directory(directory, dirfd)
            _related_v2_names_at(dirfd, name)
            if (_read_private_v2_at(dirfd, name) != raw
                    or verify_and_decode_feedback_checkpoint_v2(raw, session) != accepted):
                _fail("published v2 checkpoint failed exact read-back")
            _verify_private_v2_directory(directory, dirfd)

        _related_v2_names_at(dirfd, name)
        try:
            existing = os.stat(name, dir_fd=dirfd, follow_symlinks=False)
        except FileNotFoundError:
            existing = None
        except OSError as exc:
            _fail("cannot inspect existing v2 checkpoint", exc)
        if existing is not None:
            verify_published()
            try:
                os.fsync(dirfd)
            except OSError as exc:
                _fail("cannot confirm existing v2 checkpoint durability", exc)
            verify_published()
            return target

        staged_name = None
        try:
            # Open temp and destination relative to the same pinned directory
            # descriptor. EEXIST is never a reason to overwrite existing data.
            for _ in range(3):
                candidate = f".{name}.{secrets.token_hex(16)}.tmp"
                try:
                    staged_fd = os.open(candidate, os.O_WRONLY | os.O_CREAT | os.O_EXCL
                                        | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
                                        0o600, dir_fd=dirfd)
                    staged_name = candidate
                    break
                except FileExistsError:
                    continue
            if staged_name is None:
                _fail("cannot allocate private v2 checkpoint staging file")
            try:
                os.fchmod(staged_fd, 0o600)
                _write_all(staged_fd, raw)
                os.fsync(staged_fd)
            finally:
                os.close(staged_fd)

            _verify_private_v2_directory(directory, dirfd)
            _related_v2_names_at(dirfd, name)
            try:
                os.link(staged_name, name, src_dir_fd=dirfd, dst_dir_fd=dirfd,
                        follow_symlinks=False)
            except FileExistsError:
                # A concurrent same-name writer cannot grant us overwrite
                # authority. The exact final file is checked after cleanup.
                pass
            try:
                os.fsync(dirfd)
            except OSError as exc:
                _fail("post-link v2 directory fsync failed", exc)
            _verify_private_v2_directory(directory, dirfd)
        finally:
            if staged_name is not None:
                try:
                    os.unlink(staged_name, dir_fd=dirfd)
                    os.fsync(dirfd)
                except OSError as exc:
                    _fail("v2 staging cleanup or directory fsync failed", exc)

        # Only after cleanup is it possible for final st_nlink == 1. If
        # cleanup failed, there is no success-shaped return even if visible.
        verify_published()
        return target


def load_checkpoint(root: Path | str | os.PathLike[str], session: Any, window_id: str) -> B13WindowAcceptedProposal | None:
    """Load exactly one expected checkpoint, or ``None`` only when it is absent."""
    directory = _root(root, create=False)
    _session_window(session, window_id)
    if directory is None:
        return None
    # Do not call checkpoint_path here: it may recreate a root removed after
    # the read-only root check. Loading must never create filesystem objects.
    target = directory / _checkpoint_name(session, window_id)
    root_before = os.stat(directory, follow_symlinks=False)
    if not stat.S_ISDIR(root_before.st_mode):
        _fail("checkpoint root changed during inspection")
    _reject_other_window_checkpoint(target)
    before = _target_stat(target, missing_ok=True)
    if before is None:
        # A root that vanished after inspection is not an ordinary cache miss.
        if _root(directory, create=False) is None:
            _fail("checkpoint root disappeared during load")
        _reject_other_window_checkpoint(target)
        if (_target_stat(target, missing_ok=True) is not None
                or not _stable_file(root_before, os.stat(directory, follow_symlinks=False))):
            _fail("checkpoint appeared during inspection")
        return None
    raw = _read_bounded(target)
    _reject_other_window_checkpoint(target)
    try:
        decoded = evidence.verify_and_decode_checkpoint(raw, session)
    except evidence.B13CheckpointEvidenceError:
        raise
    if decoded.window_id != window_id:
        _fail("checkpoint decoded to a different window")
    after = _target_stat(target)
    if (after is None or not _stable_file(before, after)
            or not _stable_file(root_before, os.stat(directory, follow_symlinks=False))):
        _fail("checkpoint changed during verification")
    _reject_other_window_checkpoint(target)
    return decoded


__all__ = [
    "B13CheckpointStorageError", "checkpoint_path", "save_checkpoint",
    "save_feedback_checkpoint_v2", "confirm_feedback_checkpoint_directory", "load_checkpoint",
]
