"""Guarded, offline checkpoint-root initialization; never dispatch authority.

Requires trustworthy, cooperating filesystem users and reliable local fsync and
flock. Readable evidence cannot prove that an earlier caller returned success.
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
from . import disposition_checkpoint_storage as checkpoints
from .disposition_resume import _verify


class _RootState(str, Enum):
    INITIALIZED = 'INITIALIZED'
    EVIDENCE_VERIFIED = 'EVIDENCE_VERIFIED'
    EXISTING = 'EXISTING'
    VERIFIED_REUSE = 'VERIFIED_REUSE'


class _RootError(RuntimeError):
    """Initialization failed; any created evidence must be retained."""


def _initialize_root(*, guard, store, pointers, session, checkpoint_root, window_id,
                     operator_id, recorded_at, reason, historically_absent):
    """Initialize only an absent final component; existing evidence is read-only."""
    parent_fd = root_fd = None
    try:
        if not isinstance(guard, claim.ExecutionGuard) or guard._store is not store or guard._pointers is not pointers:
            raise ValueError('matching active guard required')
        guard.verify()
        if historically_absent is not True:
            raise ValueError('historical observation must be explicitly supplied')
        path = Path(checkpoint_root)
        if not path.is_absolute() or str(path) != os.fspath(checkpoint_root) or path.name in ('', '.', '..'):
            raise ValueError('checkpoint path must be absolute and canonical')
        parent = path.parent
        if parent.resolve(strict=True) != parent or path != Path(os.path.normpath(path)):
            raise ValueError('checkpoint parent must be canonical and non-symlink')
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
        before = os.lstat(parent)
        parent_fd = os.open(parent, flags)
        if not claim._same(before, os.fstat(parent_fd)):
            raise ValueError('checkpoint parent changed')
        roots = [Path(guard.path), Path(store.root), Path(pointers.root)]
        root_ids = [audit._root_identity(p) for p in roots]
        parent_id = audit._root_identity(parent)
        ancestors = tuple(reversed(parent.parents)) + (parent,)
        ancestor_stats = tuple(os.lstat(p) for p in ancestors)

        def check_ancestors():
            for component, initial in zip(ancestors, ancestor_stats):
                current = os.lstat(component)
                if (not stat.S_ISDIR(current.st_mode) or not claim._same(initial, current)
                        or current.st_uid not in (0, os.geteuid())
                        or (current.st_mode & 0o022 and not (current.st_uid == 0 and current.st_mode & stat.S_ISVTX))):
                    raise ValueError('untrustworthy checkpoint ancestor')

        check_ancestors()
        if (any(parent == p or parent.is_relative_to(p) or p.is_relative_to(parent) for p in roots)
                or any(a == b or a.is_relative_to(b) or b.is_relative_to(a)
                       for index, a in enumerate(roots) for b in roots[index + 1:])):
            raise ValueError('checkpoint parent or protected roots overlap')
        if len({(r['dev'], r['ino']) for r in root_ids + [parent_id]}) != 4:
            raise ValueError('root alias')
        first = _verify(store, pointers, session)
        session_evidence = (session.runtime_config, tuple(session.budgets), session.preflight_verdict,
                            session.request_model, session.effective_n_ctx, session.input_ceiling_tokens,
                            session.strategy_ref, tuple(session.window_evidence))
        ident = claim._identity(session, window_id)
        expected = claim._encoded(ident)
        claim_id = content_hash(ident)
        semantic = next(s for s in first[1] if s.packet.window_id == window_id)
        binding = dict(claim_identity=claim_id, claim_bytes_hash=audit._hash(expected),
                       preparation_identity=ident['preparation_identity'], window_id=window_id,
                       window_ordinal=ident['window_ordinal'], request_identity=ident['request_identity'],
                       semantic_request_hash=semantic.request.request_hash,
                       authority_hash=audit._authority_hash(first[0]))
        claim_name = claim.claim_path(guard.path, session, window_id).name
        claim_stat = os.stat(claim_name, dir_fd=guard._dirfd, follow_symlinks=False)
        prefix = f'b13-recovery-v1-{claim_id}-'
        names = {t: audit.record_name(claim_id, t) for t in audit._TYPES}
        parent_record = parent_id

        def check():
            guard.verify()
            check_ancestors()
            st = os.fstat(parent_fd)
            execution_parent = os.lstat(guard.path.parent)
            if (not stat.S_ISDIR(execution_parent.st_mode) or execution_parent.st_uid != os.geteuid()
                    or execution_parent.st_mode & 0o022 or not stat.S_ISDIR(st.st_mode) or st.st_uid != os.geteuid() or st.st_mode & 0o022
                    or not claim._same(before, st) or not claim._same(st, os.lstat(parent))
                    or parent.resolve(strict=True) != parent or audit._root_identity(parent) != parent_id
                    or not claim._same(guard._parent_stat, os.lstat(guard.path.parent))
                    or [audit._root_identity(p) for p in roots] != root_ids
                    or _verify(store, pointers, session) != first
                    or (session.runtime_config, tuple(session.budgets), session.preflight_verdict,
                        session.request_model, session.effective_n_ctx, session.input_ceiling_tokens,
                        session.strategy_ref, tuple(session.window_evidence)) != session_evidence
                    or not claim._same(claim_stat, os.stat(claim_name, dir_fd=guard._dirfd, follow_symlinks=False))
                    or claim._inspect_at(guard.path / claim_name, guard._dirfd, expected).state is not claim.ClaimState.CLAIMED_UNRESOLVED):
                raise ValueError('guard, parent, claim or authority changed')

        def slots():
            if any(prefix in n and n not in names.values() for n in os.listdir(guard._dirfd)):
                raise ValueError('ambiguous recovery remnant')
            return {t: audit._read_slot(guard.path, guard._dirfd, name) for t, name in names.items()}

        def root_stat():
            try:
                return os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
            except FileNotFoundError:
                return None

        def verify_root(fd, expected_stat):
            opened = os.fstat(fd)
            named = root_stat()
            if (named is None or not stat.S_ISDIR(opened.st_mode) or opened.st_uid != os.geteuid()
                    or stat.S_IMODE(opened.st_mode) != 0o700 or not claim._same(expected_stat, opened)
                    or not claim._same(opened, named) or not claim._same(opened, os.lstat(path))
                    or (opened.st_dev, opened.st_ino) in {(r['dev'], r['ino']) for r in root_ids + [parent_id]}):
                raise ValueError('root changed or unsafe')
            return opened

        def absent_checkpoint():
            if checkpoints.load_checkpoint(path, session, window_id) is not None:
                raise ValueError('verified checkpoint takes precedence')

        def publish(kind, raw, transition_check):
            # Retain temporary evidence on every error, including link-then-error.
            name = names[kind]
            temp = f'.{name}.{secrets.token_hex(16)}.tmp'
            fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                         0o600, dir_fd=guard._dirfd)
            try:
                st = os.fstat(fd)
                if not stat.S_ISREG(st.st_mode) or not claim._private(st) or st.st_nlink != 1:
                    raise ValueError('unsafe temporary record')
                checkpoints._write_all(fd, raw)
                os.fsync(fd)
                transition_check()
                now = os.fstat(fd)
                named = os.stat(temp, dir_fd=guard._dirfd, follow_symlinks=False)
                if not claim._same(st, now) or not claim._same(now, named) or now.st_size != len(raw) or now.st_nlink != 1:
                    raise ValueError('temporary record changed')
                os.link(temp, name, src_dir_fd=guard._dirfd, dst_dir_fd=guard._dirfd, follow_symlinks=False)
                transition_check()
                os.fsync(guard._dirfd)
                transition_check()
                final = os.stat(name, dir_fd=guard._dirfd, follow_symlinks=False)
                if not claim._same(now, final) or final.st_nlink != 2:
                    raise ValueError('published record changed')
                os.unlink(temp, dir_fd=guard._dirfd)
                os.fsync(guard._dirfd)
                transition_check()
                if (not claim._same(now, os.stat(name, dir_fd=guard._dirfd, follow_symlinks=False))
                        or audit._read_slot(guard.path, guard._dirfd, name) != raw):
                    raise ValueError('record readback failed')
                transition_check()
                return now
            finally:
                os.close(fd)

        check()
        records = slots()
        if any(records[t] is not None for t in ('decision', 'consumption', 'terminal')):
            raise ValueError('other recovery evidence requires review')
        record_stats = {t: os.stat(names[t], dir_fd=guard._dirfd, follow_symlinks=False)
                        for t in names if records[t] is not None}
        i, r = records['root_intent'], records['root_result']
        if bool(i) != bool(r):
            raise ValueError('incomplete initialization evidence')
        existing = root_stat()
        if existing is not None:
            root_fd = os.open(path.name, flags, dir_fd=parent_fd)
            verify_root(root_fd, existing)
            if i is not None:
                intent = audit.decode_record(i, 'root_intent')
                result = audit.decode_record(r, 'root_result')
                st = verify_root(root_fd, existing)
                if (intent['binding'] != binding or result['binding'] != binding
                        or intent['parent'] != parent_record or intent['desired_path'] != str(path)
                        or result['intent_identity'] != intent['content_hash'] or result['parent'] != parent_record
                        or result['root'] != audit._root_identity(path) or result['root_uid'] != st.st_uid
                        or result['root_mode'] != stat.S_IMODE(st.st_mode)):
                    raise ValueError('initialization evidence mismatch')
            accepted = checkpoints.load_checkpoint(path, session, window_id)
            check()
            verify_root(root_fd, existing)
            if (checkpoints.load_checkpoint(path, session, window_id) != accepted
                    or slots() != records or any(not claim._same(st, os.stat(names[t], dir_fd=guard._dirfd, follow_symlinks=False))
                                                for t, st in record_stats.items())):
                raise ValueError('recovery evidence changed')
            if accepted is not None:
                return _RootState.VERIFIED_REUSE
            return _RootState.EVIDENCE_VERIFIED if i else _RootState.EXISTING
        if i is not None:
            raise ValueError('initialized root disappeared')
        absent_checkpoint()
        intent_raw = audit.encode_record('root_intent', dict(binding=binding, operator_id=operator_id,
            recorded_at=recorded_at, reason=reason, parent=parent_record, desired_path=str(path), historically_absent=True))
        intent = audit.decode_record(intent_raw, 'root_intent')
        def before_create():
            check()
            absent_checkpoint()
            if root_stat() is not None:
                raise ValueError('checkpoint root appeared')
        intent_stat = publish('root_intent', intent_raw, before_create)
        before_create()
        if slots()['root_intent'] != intent_raw:
            raise ValueError('intent changed')
        os.mkdir(path.name, mode=0o700, dir_fd=parent_fd)
        created = root_stat()
        if created is None:
            raise ValueError('created root missing')
        root_fd = os.open(path.name, flags, dir_fd=parent_fd)
        verify_root(root_fd, created)
        check()
        os.fsync(parent_fd)
        def after_create():
            check()
            verify_root(root_fd, created)
            absent_checkpoint()
            if os.listdir(root_fd):
                raise ValueError('new checkpoint root is unexpectedly nonempty')
            if (not claim._same(intent_stat, os.stat(names['root_intent'], dir_fd=guard._dirfd, follow_symlinks=False))
                    or audit._read_slot(guard.path, guard._dirfd, names['root_intent']) != intent_raw):
                raise ValueError('intent changed')
        after_create()
        st = verify_root(root_fd, created)
        result_raw = audit.encode_record('root_result', dict(binding=binding, intent_identity=intent['content_hash'],
            parent=parent_record, root=audit._root_identity(path), root_uid=st.st_uid,
            root_mode=stat.S_IMODE(st.st_mode), absence_observed=True, parent_synced=True))
        result_stat = publish('root_result', result_raw, after_create)
        final = slots()
        if (not claim._same(result_stat, os.stat(names['root_result'], dir_fd=guard._dirfd, follow_symlinks=False))
                or final['root_intent'] != intent_raw or final['root_result'] != result_raw or any(
                final[t] is not None for t in ('decision', 'consumption', 'terminal'))):
            raise ValueError('final evidence changed')
        after_create()
        return _RootState.INITIALIZED
    except Exception as exc:
        raise _RootError('checkpoint root verification or initialization failed') from exc
    finally:
        if root_fd is not None:
            os.close(root_fd)
        if parent_fd is not None:
            os.close(parent_fd)
