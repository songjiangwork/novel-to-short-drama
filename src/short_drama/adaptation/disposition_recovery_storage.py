"""Private offline persistence seam for recovery decisions; never dispatch authority.

Only existing roots are accepted. The proposed policy is not ratified. This
module deliberately exposes no consumption, initialization, or execution API.
It assumes the guard's trusted-parent and cooperating-filesystem boundary.
"""
from __future__ import annotations

import os
import secrets
import stat
from enum import Enum
from pathlib import Path

from short_drama.artifacts import content_hash
from . import disposition_execution_claim as claim
from . import disposition_recovery_audit as audit
from .disposition_resume import _verify


class _Result(str, Enum):
    PERSISTED = 'PERSISTED'
    EXISTS_VERIFIED = 'EXISTS_VERIFIED'


class _StorageError(RuntimeError):
    """Failure; publication may be uncertain once link has been attempted."""

    def __init__(self, message, *, link_attempted=False):
        super().__init__(message)
        self.link_attempted = link_attempted


def _persist_decision(*, guard, store, pointers, session, checkpoint_root, window_id, raw):
    """Persist one canonical, session-bound decision in its fixed slot.

    This internal seam grants no authorization. On failure, tracked temporary
    evidence is retained; link_attempted distinguishes possible publication
    from failures before link. An existing slot is never a new consumption.
    """
    attempted = False
    fd = None
    try:
        if not isinstance(guard, claim.ExecutionGuard):
            raise ValueError('active guard required')
        guard.verify()
        if not claim._same(guard._parent_stat, os.lstat(guard.path.parent)):
            raise ValueError('guard parent changed')
        if guard._store is not store or guard._pointers is not pointers:
            raise ValueError('guard store mismatch')
        record = audit.decode_record(raw, 'decision')
        first = _verify(store, pointers, session)
        def session_evidence():
            return (session.runtime_config, tuple(session.budgets), session.preflight_verdict,
                    session.request_model, session.effective_n_ctx, session.input_ceiling_tokens,
                    session.strategy_ref, tuple(session.window_evidence))
        initial_evidence = session_evidence()
        identity = claim._identity(session, window_id)
        expected = claim._encoded(identity)
        claim_id = content_hash(identity)
        semantic = next(s for s in first[1] if s.packet.window_id == window_id)
        binding = dict(claim_identity=claim_id, claim_bytes_hash=audit._hash(expected),
                       preparation_identity=identity['preparation_identity'], window_id=window_id,
                       window_ordinal=identity['window_ordinal'], request_identity=identity['request_identity'],
                       semantic_request_hash=semantic.request.request_hash, authority_hash=audit._authority_hash(first[0]))
        if record['binding'] != binding:
            raise ValueError('decision binding mismatch')
        roots = {k: audit._root_identity(p) for k, p in
                 dict(execution=guard.path, checkpoint=checkpoint_root, artifact=store.root, pointer=pointers.root).items()}
        paths = [Path(v['path']) for v in roots.values()]
        if (any(a == b or a.is_relative_to(b) or b.is_relative_to(a)
                for i, a in enumerate(paths) for b in paths[i + 1:])
                or len({(v['dev'], v['ino']) for v in roots.values()}) != 4
                or any(not claim._private(os.lstat(p)) for p in paths[:2])
                or record['roots'] != roots or record['root_result_identity'] is not None):
            raise ValueError('unsafe or mismatched roots')
        # A separate R1B2 will validate initialization evidence. Do not adopt it here.
        claim_name = claim.claim_path(guard.path, session, window_id).name
        original_stat = os.stat(claim_name, dir_fd=guard._dirfd, follow_symlinks=False)
        parents = {str(p.parent): audit._root_identity(p.parent) for p in paths}
        name = audit.record_name(claim_id, 'decision')
        prefix = f'b13-recovery-v1-{claim_id}-'
        allowed = {audit.record_name(claim_id, t) for t in audit._TYPES}

        def check():
            guard.verify()
            if (not claim._same(guard._parent_stat, os.lstat(guard.path.parent))
                    or _verify(store, pointers, session) != first
                    or session_evidence() != initial_evidence
                    or {k: audit._root_identity(p) for k, p in
                        dict(execution=guard.path, checkpoint=checkpoint_root, artifact=store.root, pointer=pointers.root).items()} != roots
                    or {str(p.parent): audit._root_identity(p.parent) for p in paths} != parents
                    or not claim._same(original_stat, os.stat(claim_name, dir_fd=guard._dirfd, follow_symlinks=False))
                    or claim._inspect_at(guard.path / claim_name, guard._dirfd, expected).state is not claim.ClaimState.CLAIMED_UNRESOLVED):
                raise ValueError('guard, authority, roots or claim changed')

        check()
        names = os.listdir(guard._dirfd)
        if any(prefix in n and n not in allowed for n in names):
            raise ValueError('ambiguous recovery remnant')
        if any(audit._read_slot(guard.path, guard._dirfd, audit.record_name(claim_id, t)) is not None
               for t in audit._TYPES if t != 'decision'):
            raise ValueError('other recovery evidence requires separate review')
        existing = audit._read_slot(guard.path, guard._dirfd, name)
        if existing is not None:
            if existing != raw or audit.decode_record(existing, 'decision') != record:
                raise ValueError('conflicting slot')
            slot_stat = os.stat(name, dir_fd=guard._dirfd, follow_symlinks=False)
            check()
            if (audit._read_slot(guard.path, guard._dirfd, name) != raw
                    or not claim._same(slot_stat, os.stat(name, dir_fd=guard._dirfd, follow_symlinks=False))
                    or any(prefix in n and n not in allowed for n in os.listdir(guard._dirfd))):
                raise ValueError('existing slot changed')
            check()
            return _Result.EXISTS_VERIFIED
        check()
        temp = f'.{name}.{secrets.token_hex(16)}.tmp'
        fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                     0o600, dir_fd=guard._dirfd)
        temp_stat = os.fstat(fd)
        check()
        os.fchmod(fd, 0o600)
        if not stat.S_ISREG(temp_stat.st_mode) or not claim._private(os.fstat(fd)) or temp_stat.st_nlink != 1:
            raise ValueError('unsafe temporary file')
        remaining = memoryview(raw)
        while remaining:
            check()
            written = os.write(fd, remaining)
            if not isinstance(written, int) or not 0 < written <= len(remaining):
                raise ValueError('invalid temporary write')
            remaining = remaining[written:]
            check()
        os.fsync(fd)
        check()
        current = os.fstat(fd)
        named = os.stat(temp, dir_fd=guard._dirfd, follow_symlinks=False)
        if not claim._same(temp_stat, current) or not claim._same(current, named) or current.st_size != len(raw) or current.st_nlink != 1:
            raise ValueError('temporary file changed')
        attempted = True
        os.link(temp, name, src_dir_fd=guard._dirfd, dst_dir_fd=guard._dirfd, follow_symlinks=False)
        check()
        os.fsync(guard._dirfd)
        check()
        final_stat = os.stat(name, dir_fd=guard._dirfd, follow_symlinks=False)
        named = os.stat(temp, dir_fd=guard._dirfd, follow_symlinks=False)
        if not claim._same(current, final_stat) or not claim._same(current, named) or final_stat.st_nlink != 2:
            raise ValueError('published file changed')
        check()
        os.unlink(temp, dir_fd=guard._dirfd)
        check()
        os.fsync(guard._dirfd)
        check()
        if (not claim._same(current, os.stat(name, dir_fd=guard._dirfd, follow_symlinks=False))
                or audit._read_slot(guard.path, guard._dirfd, name) != raw
                or any(prefix in n and n not in allowed for n in os.listdir(guard._dirfd))):
            raise ValueError('final verification failed')
        check()
        return _Result.PERSISTED
    except Exception as exc:
        raise _StorageError('recovery record persistence failed', link_attempted=attempted) from exc
    finally:
        if fd is not None:
            os.close(fd)
