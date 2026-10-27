"""Private, at-most-once B1.3 window execution claims (no provider calls).

All cooperating executors for one story must use the same private execution
root, outside Foundation artifact and pointer trees. The advisory flock cannot
stop non-cooperating processes and requires reliable local filesystem locking.
Root parents and protected trees must be trusted and stable; only the final
root directory may be created. The guard requires the same store instances
later passed to claim creation. Privileged filesystem manipulation is outside
this boundary.
Hold the guard through execution and checkpoint persistence. A claim does not
prove dispatch or count provider requests; unresolved claims are never reclaimed.
The caller must freshly discover runtime/tokenizer limits and validate the
client before execution; declared session evidence alone cannot prove freshness.
"""
from __future__ import annotations

import fcntl
import os
import stat
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from short_drama.artifacts import FileArtifactStore, canonical_json_bytes, content_hash, strict_json_loads
from short_drama.foundation import FilePointerStore
from . import disposition_checkpoint_storage as storage
from .disposition_resume import _verify

MAX_CLAIM_BYTES = 16_384
_LOCK = '.b13-execution.lock'


class B13ExecutionClaimError(RuntimeError):
    """Execution is unsafe or claim/lock evidence cannot be verified."""


class ClaimState(str, Enum):
    ABSENT = 'ABSENT'
    CLAIMED_UNRESOLVED = 'CLAIMED_UNRESOLVED'
    INVALID = 'INVALID'


@dataclass(frozen=True, slots=True)
class ClaimInspection:
    state: ClaimState
    path: Path


def _identity(session: Any, window_id: str) -> dict:
    prep, ordinal, request = storage._session_window(session, window_id)
    preparation = session.preparation
    return {'schema_version': 1, 'state': ClaimState.CLAIMED_UNRESOLVED.value,
            'preparation_identity': prep, 'window_id': window_id, 'window_ordinal': ordinal,
            'request_identity': request,
            'source_snapshot_hash': content_hash(preparation.source_snapshot.to_dict()),
            'strategy_ref': preparation.strategy_selection.strategy_ref.to_dict(),
            'strategy_hash': preparation.strategy_selection.artifact.strategy.content_hash()}


def _encoded(identity: dict) -> bytes:
    raw = canonical_json_bytes({**identity, 'claim_identity': content_hash(identity)})
    if len(raw) > MAX_CLAIM_BYTES:
        raise B13ExecutionClaimError('claim exceeds maximum size')
    return raw


def claim_path(root: Path | str, session: Any, window_id: str) -> Path:
    """Derive the exact path without creating or searching the filesystem."""
    identity = _identity(session, window_id)
    name = (f"b13-claim-v1-p{identity['preparation_identity']}-"
            f"w{identity['window_ordinal']:08d}-q{identity['request_identity']}.json")
    return Path(root).expanduser().absolute() / name


def _private(st: os.stat_result) -> bool:
    return st.st_uid == os.geteuid() and not (st.st_mode & 0o077)


def _same(a: os.stat_result, b: os.stat_result) -> bool:
    return (a.st_dev, a.st_ino, stat.S_IFMT(a.st_mode)) == (b.st_dev, b.st_ino, stat.S_IFMT(b.st_mode))


def _open_root(root: Path | str) -> tuple[Path, int] | None:
    path = Path(root).expanduser().absolute()
    seen = False
    try:
        before = os.lstat(path)
        seen = True
        if not stat.S_ISDIR(before.st_mode) or not _private(before):
            raise B13ExecutionClaimError('execution root must be a private owned directory')
        if not hasattr(os, 'O_NOFOLLOW') or not hasattr(os, 'O_DIRECTORY'):
            raise B13ExecutionClaimError('safe directory opening is unsupported')
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
        if not _same(before, os.fstat(fd)):
            os.close(fd)
            raise B13ExecutionClaimError('execution root changed')
        return path, fd
    except FileNotFoundError as exc:
        if not seen and not path.is_symlink():
            return None
        raise B13ExecutionClaimError('execution root disappeared') from exc
    except (OSError, storage.B13CheckpointStorageError) as exc:
        raise B13ExecutionClaimError('cannot safely open execution root') from exc


def _check_root(path: Path, fd: int) -> None:
    try:
        current, opened = os.lstat(path), os.fstat(fd)
        if not _same(current, opened) or not stat.S_ISDIR(current.st_mode) or not _private(current):
            raise B13ExecutionClaimError('execution root changed or is unsafe')
    except OSError as exc:
        raise B13ExecutionClaimError('execution root disappeared') from exc


def _open_isolated_root(root: Path | str, store: FileArtifactStore,
                        pointers: FilePointerStore) -> tuple[Path, int, tuple]:
    """Validate against real protected paths before creating the final component.

    Parents and protected trees must be trusted and stable, as in the stores.
    Directory descriptors and inode checks detect ordinary replacement; they do
    not defend against privileged manipulation or hostile mount changes.
    """
    if (not isinstance(store, FileArtifactStore) or not isinstance(pointers, FilePointerStore)
            or pointers.artifact_store is not store):
        raise B13ExecutionClaimError('guard requires the matching artifact and pointer stores')
    raw = Path(root).expanduser().absolute()
    if raw.name in ('', '.', '..'):
        raise B13ExecutionClaimError('invalid execution root component')
    parent_fd = None
    root_fd = None
    try:
        # Strict resolution handles '..' and intermediate symlinks and requires
        # every parent to exist. The final component must never be a symlink.
        parent = raw.parent.resolve(strict=True)
        path = parent / raw.name
        protected = tuple(Path(item.root).resolve(strict=True) for item in (store, pointers))
        protected_stats = tuple(os.stat(item) for item in protected)
        if any(not stat.S_ISDIR(st.st_mode) for st in protected_stats):
            raise B13ExecutionClaimError('protected root is not a directory')
        if any(path.is_relative_to(item) or item.is_relative_to(path) for item in protected):
            raise B13ExecutionClaimError('execution root must be outside artifact and pointer trees')
        if not hasattr(os, 'O_NOFOLLOW') or not hasattr(os, 'O_DIRECTORY'):
            raise B13ExecutionClaimError('safe directory opening is unsupported')
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
        parent_before = os.lstat(parent)
        parent_fd = os.open(parent, flags)
        if (not stat.S_ISDIR(parent_before.st_mode) or not _same(parent_before, os.fstat(parent_fd))
                or any(_same(parent_before, st) for st in protected_stats)):
            raise B13ExecutionClaimError('execution root parent changed')
        try:
            before = os.stat(raw.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            before = None
        if before is not None:
            if (not stat.S_ISDIR(before.st_mode) or not _private(before)
                    or any(_same(before, st) for st in protected_stats)):
                raise B13ExecutionClaimError('execution root is unsafe or aliases a protected root')
        # Recheck both names before the first mutation. No recursive mkdir.
        if not _same(parent_before, os.lstat(parent)) or any(
                not _same(st, os.stat(item)) for item, st in zip(protected, protected_stats)):
            raise B13ExecutionClaimError('execution root parent or protected root changed')
        if before is None:
            os.mkdir(raw.name, mode=0o700, dir_fd=parent_fd)
            before = os.stat(raw.name, dir_fd=parent_fd, follow_symlinks=False)
        root_fd = os.open(raw.name, flags, dir_fd=parent_fd)
        opened = os.fstat(root_fd)
        if (not stat.S_ISDIR(opened.st_mode) or not _private(opened)
                or not _same(before, opened) or not _same(opened, os.lstat(path))):
            raise B13ExecutionClaimError('execution root changed during initialization')
        # Always sync the parent, including after a prior failed creation sync.
        os.fsync(parent_fd)
        _check_root(path, root_fd)
        result = (path, root_fd, tuple(zip(protected, protected_stats)))
        root_fd = None
        return result
    except B13ExecutionClaimError:
        raise
    except (OSError, RuntimeError, ValueError) as exc:
        raise B13ExecutionClaimError('cannot safely establish isolated execution root') from exc
    finally:
        if root_fd is not None:
            os.close(root_fd)
        if parent_fd is not None:
            os.close(parent_fd)


class ExecutionGuard:
    """Nonblocking process guard; acquire via ``execution_guard`` context manager."""
    def __init__(self, root: Path | str, *, store: FileArtifactStore, pointers: FilePointerStore):
        self._root_arg = root
        self._store = store
        self._pointers = pointers
        self._fd = self._dirfd = None
        self._pid = None
        self._locked = False

    def __enter__(self):
        if self._fd is not None:
            raise B13ExecutionClaimError('guard already active')
        self.path, self._dirfd, self._protected = _open_isolated_root(
            self._root_arg, self._store, self._pointers)
        try:
            _check_root(self.path, self._dirfd)
            self._check_protected()
            flags = os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK
            self._fd = os.open(_LOCK, flags, 0o600, dir_fd=self._dirfd)
            self._pid = os.getpid()
            self._check_identity()
            fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._locked = True
            self.verify()
            return self
        except Exception as exc:
            self.__exit__(None, None, None)
            if isinstance(exc, B13ExecutionClaimError):
                raise
            raise B13ExecutionClaimError('cannot acquire exclusive execution lock') from exc

    def _check_protected(self):
        try:
            if self._pointers.artifact_store is not self._store or any(
                    Path(item.root).resolve(strict=True) != path or not _same(original, os.stat(path))
                    for item, (path, original) in zip((self._store, self._pointers), self._protected)):
                raise B13ExecutionClaimError('guard protected roots changed')
        except B13ExecutionClaimError:
            raise
        except (OSError, RuntimeError) as exc:
            raise B13ExecutionClaimError('cannot verify guard protected roots') from exc

    def _check_identity(self):
        if self._fd is None or self._dirfd is None or self._pid != os.getpid():
            raise B13ExecutionClaimError('execution guard is not active in this process')
        _check_root(self.path, self._dirfd)
        self._check_protected()
        try:
            opened = os.fstat(self._fd)
            named = os.stat(_LOCK, dir_fd=self._dirfd, follow_symlinks=False)
            if (not stat.S_ISREG(opened.st_mode) or not _private(opened)
                    or opened.st_nlink != 1 or not _same(opened, named)):
                raise B13ExecutionClaimError('execution lock target changed or is unsafe')
        except OSError as exc:
            raise B13ExecutionClaimError('execution lock target disappeared') from exc

    def verify(self) -> None:
        """Fail closed if this process no longer owns the expected guard."""
        if not self._locked:
            raise B13ExecutionClaimError('execution guard is not active')
        self._check_identity()

    def __exit__(self, *_args):
        if self._fd is not None:
            os.close(self._fd)
        if self._dirfd is not None:
            os.close(self._dirfd)
        self._fd = self._dirfd = self._pid = None
        self._locked = False


def execution_guard(root: Path | str, *, store: FileArtifactStore,
                    pointers: FilePointerStore) -> ExecutionGuard:
    """Acquire a guard only after isolating its root from both supplied stores."""
    return ExecutionGuard(root, store=store, pointers=pointers)


def _inspect_at(path: Path, dirfd: int, expected: bytes) -> ClaimInspection:
    _check_root(path.parent, dirfd)
    try:
        before = os.stat(path.name, dir_fd=dirfd, follow_symlinks=False)
    except FileNotFoundError:
        _check_root(path.parent, dirfd)
        return ClaimInspection(ClaimState.ABSENT, path)
    except OSError as exc:
        raise B13ExecutionClaimError('cannot inspect claim') from exc
    if not stat.S_ISREG(before.st_mode) or not _private(before) or before.st_nlink != 1 or before.st_size > MAX_CLAIM_BYTES:
        return ClaimInspection(ClaimState.INVALID, path)
    try:
        fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC, dir_fd=dirfd)
        try:
            opened = os.fstat(fd)
            if not storage._stable_file(before, opened):
                return ClaimInspection(ClaimState.INVALID, path)
            chunks = []
            remaining = MAX_CLAIM_BYTES + 1
            while remaining:
                chunk = os.read(fd, remaining)
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            raw = b''.join(chunks)
            after = os.fstat(fd)
            named = os.stat(path.name, dir_fd=dirfd, follow_symlinks=False)
            valid = (len(raw) == opened.st_size and storage._stable_file(opened, after)
                     and storage._stable_file(after, named) and raw == expected
                     and canonical_json_bytes(strict_json_loads(raw)) == raw)
        finally:
            os.close(fd)
        _check_root(path.parent, dirfd)
        return ClaimInspection(ClaimState.CLAIMED_UNRESOLVED if valid else ClaimState.INVALID, path)
    except (OSError, ValueError, TypeError):
        return ClaimInspection(ClaimState.INVALID, path)


def inspect_claim(root: Path | str, session: Any, window_id: str) -> ClaimInspection:
    path = claim_path(root, session, window_id)
    expected = _encoded(_identity(session, window_id))
    opened = _open_root(root)
    if opened is None:
        return ClaimInspection(ClaimState.ABSENT, path)
    directory, fd = opened
    try:
        return _inspect_at(directory / path.name, fd, expected)
    finally:
        os.close(fd)


def create_claim(*, guard: ExecutionGuard, store, pointers, session: Any,
                 checkpoint_root: Path | str, window_id: str) -> ClaimInspection:
    """Verify authority and exact checkpoint, then durably publish once.

    Caller must hold the same guard until the accepted checkpoint is durable.
    Fresh runtime and tokenizer discovery remains the caller's responsibility.
    Existing verified checkpoints and every existing claim refuse creation.
    """
    if not isinstance(guard, ExecutionGuard):
        raise B13ExecutionClaimError('active execution guard required')
    guard.verify()
    if guard._store is not store or guard._pointers is not pointers:
        raise B13ExecutionClaimError('guard was acquired for different stores')
    try:
        initial = _verify(store, pointers, session)
        evidence = (session.runtime_config, tuple(session.budgets), session.preflight_verdict,
                    session.request_model, session.effective_n_ctx, session.input_ceiling_tokens,
                    session.strategy_ref, tuple(session.window_evidence))
        path = claim_path(guard.path, session, window_id)
        raw = _encoded(_identity(session, window_id))
        checkpoint = storage.load_checkpoint(checkpoint_root, session, window_id)
        state = _inspect_at(path, guard._dirfd, raw)
        final = _verify(store, pointers, session)
        final_evidence = (session.runtime_config, tuple(session.budgets), session.preflight_verdict,
                          session.request_model, session.effective_n_ctx, session.input_ceiling_tokens,
                          session.strategy_ref, tuple(session.window_evidence))
        if final != initial or evidence != final_evidence:
            raise B13ExecutionClaimError('authority changed during claim verification')
        guard.verify()
        if checkpoint is not None:
            raise B13ExecutionClaimError('verified checkpoint already exists')
        if state.state is not ClaimState.ABSENT:
            raise B13ExecutionClaimError('existing claim is unresolved or invalid')
        fd, temp = tempfile.mkstemp(prefix='.b13-claim-', suffix='.tmp', dir=guard.path)
        try:
            try:
                os.fchmod(fd, 0o600)
                storage._write_all(fd, raw)
                os.fsync(fd)
            finally:
                os.close(fd)
            guard.verify()
            os.link(Path(temp).name, path.name, src_dir_fd=guard._dirfd, dst_dir_fd=guard._dirfd, follow_symlinks=False)
            os.fsync(guard._dirfd)
            guard.verify()
            return ClaimInspection(ClaimState.CLAIMED_UNRESOLVED, path)
        finally:
            os.unlink(Path(temp).name, dir_fd=guard._dirfd)
    except B13ExecutionClaimError:
        raise
    except Exception as exc:
        raise B13ExecutionClaimError('claim verification or durable publication failed') from exc
