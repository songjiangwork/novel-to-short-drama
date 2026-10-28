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
from . import disposition_checkpoint_storage as checkpoint_storage
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
    fd = parent_fd = root_fd = None
    temp = None
    try:
        if not isinstance(guard, claim.ExecutionGuard):
            raise ValueError('active guard required')
        guard.verify()
        if not claim._same(guard._parent_stat, os.lstat(guard.path.parent)):
            raise ValueError('guard parent changed')
        if guard._store is not store or guard._pointers is not pointers:
            raise ValueError('guard store mismatch')
        checkpoint_path = Path(checkpoint_root)
        if (not checkpoint_path.is_absolute() or str(checkpoint_path) != os.fspath(checkpoint_root)
                or checkpoint_path != Path(os.path.normpath(checkpoint_path))):
            raise ValueError('checkpoint path must be absolute and canonical')
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
                or record['roots'] != roots):
            raise ValueError('unsafe or mismatched roots')
        claim_name = claim.claim_path(guard.path, session, window_id).name
        original_stat = os.stat(claim_name, dir_fd=guard._dirfd, follow_symlinks=False)
        parents = {str(p.parent): audit._root_identity(p.parent) for p in paths}
        name = audit.record_name(claim_id, 'decision')
        prefix = f'b13-recovery-v1-{claim_id}-'
        allowed = {audit.record_name(claim_id, t) for t in audit._TYPES}
        init_names = {t: audit.record_name(claim_id, t) for t in ('root_intent', 'root_result')}
        parent = paths[1].parent
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
        parent_fd = os.open(parent, flags)
        root_fd = os.open(paths[1].name, flags, dir_fd=parent_fd)
        parent_stat, root_stat = os.fstat(parent_fd), os.fstat(root_fd)
        ancestors = tuple(reversed(parent.parents)) + (parent,)
        ancestor_stats = tuple(os.lstat(p) for p in ancestors)
        init_snapshot = None
        own_temp = None
        require_absence = False

        def verify_initialization():
            nonlocal init_snapshot
            for component, initial in zip(ancestors, ancestor_stats):
                current = os.lstat(component)
                if (not stat.S_ISDIR(current.st_mode) or not claim._same(initial, current)
                        or current.st_uid not in (0, os.geteuid())
                        or (current.st_mode & 0o022 and not (current.st_uid == 0 and current.st_mode & stat.S_ISVTX))):
                    raise ValueError('untrustworthy checkpoint ancestor')
            parent_now, root_now = os.fstat(parent_fd), os.fstat(root_fd)
            if (not claim._same(parent_stat, parent_now) or not claim._same(parent_now, os.lstat(parent))
                    or parent_now.st_uid != os.geteuid() or parent_now.st_mode & 0o022
                    or not claim._same(root_stat, root_now)
                    or not claim._same(root_now, os.stat(paths[1].name, dir_fd=parent_fd, follow_symlinks=False))
                    or not claim._same(root_now, os.lstat(paths[1]))
                    or not stat.S_ISDIR(root_now.st_mode) or root_now.st_uid != os.geteuid()
                    or stat.S_IMODE(root_now.st_mode) != 0o700
                    or (root_now.st_dev, root_now.st_ino) == (parent_now.st_dev, parent_now.st_ino)):
                raise ValueError('checkpoint root or parent changed')
            if any(prefix in n and n not in allowed and n != own_temp for n in os.listdir(guard._dirfd)):
                raise ValueError('ambiguous recovery remnant')
            if any(audit._read_slot(guard.path, guard._dirfd, audit.record_name(claim_id, t)) is not None
                   for t in ('consumption', 'terminal')):
                raise ValueError('other recovery evidence requires separate review')
            def slot_stat(name):
                try:
                    return os.stat(name, dir_fd=guard._dirfd, follow_symlinks=False)
                except FileNotFoundError:
                    return None

            before = {t: slot_stat(n) for t, n in init_names.items()}
            raws = {t: audit._read_slot(guard.path, guard._dirfd, n) for t, n in init_names.items()}
            stats = {t: os.stat(init_names[t], dir_fd=guard._dirfd, follow_symlinks=False)
                     for t in init_names if raws[t] is not None}
            if any((before[t] is None) != (raws[t] is None)
                   or (before[t] is not None and not checkpoint_storage._stable_file(before[t], stats[t]))
                   for t in init_names):
                raise ValueError('initialization slot changed during read')
            if (raws['root_intent'] is None) != (raws['root_result'] is None):
                raise ValueError('incomplete initialization evidence')
            result_id = None
            if raws['root_intent'] is not None:
                intent = audit.decode_record(raws['root_intent'], 'root_intent')
                result = audit.decode_record(raws['root_result'], 'root_result')
                if (intent['binding'] != binding or result['binding'] != binding
                        or intent['parent'] != parents[str(parent)] or intent['desired_path'] != str(paths[1])
                        or result['intent_identity'] != intent['content_hash'] or result['parent'] != intent['parent']
                        or result['root'] != roots['checkpoint'] or result['root_uid'] != root_now.st_uid
                        or result['root_mode'] != stat.S_IMODE(root_now.st_mode)):
                    raise ValueError('initialization evidence mismatch')
                result_id = result['content_hash']
            if record['root_result_identity'] != result_id:
                raise ValueError('root result reference mismatch')
            if init_snapshot is None:
                init_snapshot = (raws, stats)
            elif (raws != init_snapshot[0] or stats.keys() != init_snapshot[1].keys()
                  or any(not checkpoint_storage._stable_file(st, stats[t]) for t, st in init_snapshot[1].items())):
                raise ValueError('initialization evidence changed')

        def check():
            guard.verify()
            for parent in (paths[0].parent, paths[1].parent):
                st = os.lstat(parent)
                if not stat.S_ISDIR(st.st_mode) or st.st_uid != os.geteuid() or st.st_mode & 0o022:
                    raise ValueError('unsafe execution or checkpoint parent')
            if (not claim._same(guard._parent_stat, os.lstat(guard.path.parent))
                    or _verify(store, pointers, session) != first
                    or session_evidence() != initial_evidence
                    or {k: audit._root_identity(p) for k, p in
                        dict(execution=guard.path, checkpoint=checkpoint_root, artifact=store.root, pointer=pointers.root).items()} != roots
                    or {str(p.parent): audit._root_identity(p.parent) for p in paths} != parents
                    or not claim._same(original_stat, os.stat(claim_name, dir_fd=guard._dirfd, follow_symlinks=False))
                    or claim._inspect_at(guard.path / claim_name, guard._dirfd, expected).state is not claim.ClaimState.CLAIMED_UNRESOLVED):
                raise ValueError('guard, authority, roots or claim changed')
            verify_initialization()
            if require_absence and checkpoint_storage.load_checkpoint(checkpoint_root, session, window_id) is not None:
                raise ValueError('verified checkpoint takes precedence')

        check()
        names = os.listdir(guard._dirfd)
        if any(prefix in n and n not in allowed for n in names):
            raise ValueError('ambiguous recovery remnant')
        existing = audit._read_slot(guard.path, guard._dirfd, name)
        if existing is not None:
            existing_checkpoint = checkpoint_storage.load_checkpoint(checkpoint_root, session, window_id)
            if existing != raw or audit.decode_record(existing, 'decision') != record:
                raise ValueError('conflicting slot')
            slot_stat = os.stat(name, dir_fd=guard._dirfd, follow_symlinks=False)
            check()
            if (audit._read_slot(guard.path, guard._dirfd, name) != raw
                    or not claim._same(slot_stat, os.stat(name, dir_fd=guard._dirfd, follow_symlinks=False))
                    or any(prefix in n and n not in allowed for n in os.listdir(guard._dirfd))):
                raise ValueError('existing slot changed')
            check()
            if checkpoint_storage.load_checkpoint(checkpoint_root, session, window_id) != existing_checkpoint:
                raise ValueError('checkpoint changed during inspection')
            check()
            return _Result.EXISTS_VERIFIED
        def require_checkpoint_absent():
            if checkpoint_storage.load_checkpoint(checkpoint_root, session, window_id) is not None:
                raise ValueError('verified checkpoint takes precedence')

        require_absence = True
        check()
        require_checkpoint_absent()
        check()
        temp = f'.{name}.{secrets.token_hex(16)}.tmp'
        own_temp = temp
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
        check()
        require_checkpoint_absent()
        check()
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
        require_checkpoint_absent()
        check()
        return _Result.PERSISTED
    except Exception as exc:
        # If a failure follows removal of the temporary name, restore a
        # tracked remnant when possible. A failed restoration is still an
        # uncertain publication and must never be reported as success.
        if attempted and temp is not None:
            try:
                os.stat(temp, dir_fd=guard._dirfd, follow_symlinks=False)
            except FileNotFoundError:
                try:
                    os.link(name, temp, src_dir_fd=guard._dirfd, dst_dir_fd=guard._dirfd, follow_symlinks=False)
                    os.fsync(guard._dirfd)
                except OSError:
                    pass
            except OSError:
                pass
        raise _StorageError('recovery record persistence failed', link_attempted=attempted) from exc
    finally:
        if fd is not None:
            os.close(fd)
        if root_fd is not None:
            os.close(root_fd)
        if parent_fd is not None:
            os.close(parent_fd)
