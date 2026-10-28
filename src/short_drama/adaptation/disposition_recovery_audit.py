"""Offline recovery evidence inspection. No record grants execution authority.

This is a proposed v1 wire contract, not a policy approval or a persistence API.
The caller must apply the audit result before authoritative downstream use.
As with existing stores, trusted parents and cooperating filesystem users are
required; the checks cannot exclude privileged manipulation or ABA changes.
"""
from __future__ import annotations

import hashlib
import os
import re
import stat
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from short_drama.artifacts import canonical_json_bytes, content_hash, strict_json_loads
from . import disposition_execution_claim as claim
from . import disposition_checkpoint as checkpoint
from . import disposition_checkpoint_storage as storage
from .disposition_resume import _verify

MAX_RECORD_BYTES = 32768
# Matches the safe diagnostics emitted by b13_alice_live_acceptance (#159).
FAILURE_CODES = frozenset({
    'PROVIDER_TIMEOUT', 'PROVIDER_TRANSPORT_FAILURE', 'PROVIDER_HTTP_FAILURE',
    'STRUCTURED_OUTPUT_FAILURE', 'SEMANTIC_GENERATION_EXHAUSTED',
    'PROVENANCE_FAILURE', 'AUTHORITY_FAILURE', 'CHECKPOINT_STORAGE_FAILURE',
    'UNKNOWN_FAILURE',
})
_TYPES = ('root_intent', 'root_result', 'decision', 'consumption', 'terminal')
_FIELDS = {
    'root_intent': {'binding', 'operator_id', 'recorded_at', 'reason', 'parent', 'desired_path', 'historically_absent'},
    'root_result': {'binding', 'intent_identity', 'parent', 'root', 'root_uid', 'root_mode', 'absence_observed', 'parent_synced'},
    'decision': {'binding', 'decision_id', 'operator_id', 'recorded_at', 'reason', 'evidence', 'prior_outcome',
                 'ack_prior_requests_may_have_completed', 'ack_duplicate_generation_possible', 'roots',
                 'root_result_identity', 'policy_reference', 'caps', 'attempt_id'},
    'consumption': {'binding', 'decision_identity', 'decision_id', 'attempt_id', 'recorded_at', 'bindings_hash'},
    'terminal': {'binding', 'decision_identity', 'consumption_identity', 'attempt_id', 'recorded_at', 'outcome',
                 'failure_code', 'semantic_rounds', 'provider_attempts', 'original_semantic_rounds',
                 'original_provider_attempts', 'lineage_semantic_rounds', 'lineage_provider_attempts',
                 'checkpoint_identity', 'checkpoint_bytes_hash'},
}
_HASH = re.compile(r'^[0-9a-f]{64}$')
_SAFE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._:/@+ -]{0,255}$')


class RecoveryEvidenceError(ValueError):
    """Evidence is malformed, inconsistent, or unsafe to inspect."""


class Audit(str, Enum):
    ORDINARY_AUDIT = 'ORDINARY_AUDIT'
    RECOVERY_AUDIT_COMPLETE = 'RECOVERY_AUDIT_COMPLETE'
    AUDIT_INCOMPLETE = 'AUDIT_INCOMPLETE'
    BLOCKED = 'BLOCKED'


def _hash(raw):
    return hashlib.sha256(raw).hexdigest()


def _text(value):
    return isinstance(value, str) and _SAFE.fullmatch(value) is not None


def _digest(value):
    return isinstance(value, str) and _HASH.fullmatch(value) is not None


def _root_value(value):
    return (isinstance(value, dict) and set(value) == {'path', 'dev', 'ino', 'type'}
            and isinstance(value['path'], str) and Path(value['path']).is_absolute()
            and type(value['dev']) is int and value['dev'] >= 0
            and type(value['ino']) is int and value['ino'] > 0 and value['type'] == 'directory')


def encode_record(record_type: str, fields: dict) -> bytes:
    if record_type not in _FIELDS or not isinstance(fields, dict) or set(fields) != _FIELDS[record_type]:
        raise RecoveryEvidenceError('invalid record fields')
    material = {'schema_version': 1, 'record_type': record_type, **fields}
    raw = canonical_json_bytes({**material, 'content_hash': content_hash(material)})
    decode_record(raw, record_type)
    return raw


def decode_record(raw: bytes, record_type: str) -> dict:
    if record_type not in _FIELDS or not isinstance(raw, bytes) or not 0 < len(raw) <= MAX_RECORD_BYTES:
        raise RecoveryEvidenceError('invalid record size or type')
    try:
        obj = strict_json_loads(raw)
        if (not isinstance(obj, dict) or set(obj) != _FIELDS[record_type] | {'schema_version', 'record_type', 'content_hash'}
                or type(obj['schema_version']) is not int or obj['schema_version'] != 1
                or obj['record_type'] != record_type or canonical_json_bytes(obj) != raw
                or obj['content_hash'] != content_hash({k: v for k, v in obj.items() if k != 'content_hash'})):
            raise RecoveryEvidenceError('invalid record identity or shape')
        b = obj['binding']
        if (not isinstance(b, dict) or set(b) != {'claim_identity', 'claim_bytes_hash', 'preparation_identity',
                'window_id', 'window_ordinal', 'request_identity', 'semantic_request_hash', 'authority_hash'}
                or any(not _digest(b[k]) for k in ('claim_identity', 'claim_bytes_hash', 'preparation_identity',
                    'request_identity', 'semantic_request_hash', 'authority_hash'))
                or not _text(b['window_id']) or type(b['window_ordinal']) is not int or b['window_ordinal'] < 1):
            raise RecoveryEvidenceError('invalid binding')
        for key in ('operator_id', 'recorded_at', 'reason', 'decision_id', 'attempt_id', 'policy_reference'):
            if key in obj and obj[key] is not None and not _text(obj[key]):
                raise RecoveryEvidenceError('invalid text')
        for key in ('intent_identity', 'decision_identity', 'consumption_identity', 'bindings_hash', 'checkpoint_identity', 'checkpoint_bytes_hash', 'root_result_identity'):
            if key in obj and obj[key] is not None and not _digest(obj[key]):
                raise RecoveryEvidenceError('invalid reference')
        if record_type == 'root_intent':
            if not _root_value(obj['parent']) or not isinstance(obj['desired_path'], str) or not Path(obj['desired_path']).is_absolute() or obj['historically_absent'] is not True:
                raise RecoveryEvidenceError('invalid root intent')
        if record_type == 'root_result':
            if not _root_value(obj['parent']) or not _root_value(obj['root']) or obj['absence_observed'] is not True or obj['parent_synced'] is not True or not _digest(obj['intent_identity']) or type(obj['root_uid']) is not int or obj['root_uid'] < 0 or type(obj['root_mode']) is not int or obj['root_mode'] != 0o700:
                raise RecoveryEvidenceError('invalid root result')
        if record_type == 'decision':
            if (obj['prior_outcome'] != 'unknown' or obj['ack_prior_requests_may_have_completed'] is not True
                    or obj['ack_duplicate_generation_possible'] is not True or not _text(obj['policy_reference'])
                    or not _text(obj['decision_id']) or not _text(obj['attempt_id'])
                    or not isinstance(obj['evidence'], list) or not 1 <= len(obj['evidence']) <= 16
                    or any(not _digest(x) for x in obj['evidence'])
                    or obj['caps'] != {'semantic_rounds': 2, 'technical_attempts_per_round': 3, 'total_attempts': 6}
                    or not isinstance(obj['roots'], dict) or set(obj['roots']) != {'execution', 'checkpoint', 'artifact', 'pointer'}
                    or any(not _root_value(x) for x in obj['roots'].values())):
                raise RecoveryEvidenceError('invalid decision')
        if record_type == 'consumption' and (not _digest(obj['decision_identity']) or not _digest(obj['bindings_hash']) or not _text(obj['decision_id']) or not _text(obj['attempt_id'])):
            raise RecoveryEvidenceError('invalid consumption')
        if record_type == 'terminal':
            if (not _digest(obj['decision_identity']) or not _digest(obj['consumption_identity']) or not _text(obj['attempt_id'])
                    or obj['outcome'] not in ('CHECKPOINTED', 'FAILED')
                    or any(obj[k] is not None for k in ('original_semantic_rounds', 'original_provider_attempts', 'lineage_semantic_rounds', 'lineage_provider_attempts'))):
                raise RecoveryEvidenceError('invalid terminal')
            rounds, attempts = obj['semantic_rounds'], obj['provider_attempts']
            if not ((rounds is None and attempts is None) or (type(rounds) is int and 1 <= rounds <= 2 and type(attempts) is int and rounds <= attempts <= rounds * 3)):
                raise RecoveryEvidenceError('invalid attempt counts')
            if obj['outcome'] == 'CHECKPOINTED':
                if not _digest(obj['checkpoint_identity']) or not _digest(obj['checkpoint_bytes_hash']) or obj['failure_code'] is not None or rounds is None:
                    raise RecoveryEvidenceError('invalid checkpoint terminal')
            elif (obj['checkpoint_identity'] is not None or obj['checkpoint_bytes_hash'] is not None
                  or (obj['failure_code'] is not None and
                      (not isinstance(obj['failure_code'], str) or obj['failure_code'] not in FAILURE_CODES))):
                raise RecoveryEvidenceError('invalid failure terminal')
        for k in ('operator_id', 'recorded_at', 'reason'):
            if k in obj and not _text(obj[k]):
                raise RecoveryEvidenceError('missing audit metadata')
        if 'recorded_at' in obj:
            stamp = obj['recorded_at']
            if not stamp.endswith('Z') or datetime.fromisoformat(stamp[:-1] + '+00:00').utcoffset() != timezone.utc.utcoffset(None):
                raise RecoveryEvidenceError('invalid UTC timestamp')
        return obj
    except RecoveryEvidenceError:
        raise
    except Exception as exc:
        raise RecoveryEvidenceError('invalid canonical record') from exc


def record_name(claim_identity: str, record_type: str) -> str:
    if not _digest(claim_identity) or record_type not in _TYPES:
        raise RecoveryEvidenceError('invalid slot identity')
    return f'b13-recovery-v1-{claim_identity}-{record_type}.json'


def _read_slot(directory, fd, name):
    claim._check_root(directory, fd)
    try:
        before = os.stat(name, dir_fd=fd, follow_symlinks=False)
    except FileNotFoundError:
        claim._check_root(directory, fd)
        return None
    if not stat.S_ISREG(before.st_mode) or not claim._private(before) or before.st_nlink != 1 or before.st_size > MAX_RECORD_BYTES:
        raise RecoveryEvidenceError('unsafe recovery slot')
    file = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC, dir_fd=fd)
    try:
        opened = os.fstat(file)
        if not storage._stable_file(before, opened):
            raise RecoveryEvidenceError('recovery slot changed')
        chunks, remaining = [], MAX_RECORD_BYTES + 1
        while remaining:
            chunk = os.read(file, remaining)
            if not chunk: break
            chunks.append(chunk); remaining -= len(chunk)
        raw = b''.join(chunks)
        after = os.fstat(file)
        named = os.stat(name, dir_fd=fd, follow_symlinks=False)
        if len(raw) != opened.st_size or not storage._stable_file(opened, after) or not storage._stable_file(after, named):
            raise RecoveryEvidenceError('recovery slot changed during read')
        claim._check_root(directory, fd)
        return raw
    finally:
        os.close(file)


def _root_identity(path):
    path = Path(path).expanduser().absolute()
    if path != path.resolve(strict=True):
        raise RecoveryEvidenceError('noncanonical root')
    st = os.lstat(path)
    if not stat.S_ISDIR(st.st_mode):
        raise RecoveryEvidenceError('invalid root')
    return {'path': str(path), 'dev': st.st_dev, 'ino': st.st_ino, 'type': 'directory'}


def _authority_hash(preparation):
    """Bind verified source and the exact B1.2 selection path and authority."""
    selection = preparation.strategy_selection
    pointer = selection.current_pointer_ref
    return content_hash({
        'source_snapshot': preparation.source_snapshot.to_dict(),
        'strategy_ref': selection.strategy_ref.to_dict(),
        'strategy_hash': selection.strategy_hash,
        'selection_mode': 'CURRENT' if pointer is not None else 'EXPLICIT',
        'current_pointer_ref': None if pointer is None else pointer.to_dict(),
    })


def inspect_recovery(*, store, pointers, session, execution_root, checkpoint_root, window_id,
                     approved_policy_reference=None) -> Audit:
    """Read only; never permits generation.

    A complete recovery audit additionally needs a caller-supplied policy
    reference established by an independently approved trust mechanism. This
    function cannot authenticate that mechanism or approve a policy itself.
    """
    verified = False
    try:
        first = _verify(store, pointers, session)
        accepted = storage.load_checkpoint(checkpoint_root, session, window_id)
        verified = accepted is not None
        identity = claim._identity(session, window_id)
        expected = claim._encoded(identity)
        claim_id = content_hash(identity)
        opened = claim._open_root(execution_root)
        if opened is None:
            return Audit.AUDIT_INCOMPLETE if verified else Audit.BLOCKED
        directory, fd = opened
        try:
            if claim._inspect_at(directory / claim.claim_path(directory, session, window_id).name, fd, expected).state is not claim.ClaimState.CLAIMED_UNRESOLVED:
                raise RecoveryEvidenceError('missing or invalid original claim')
            # Any leftover recovery temporary/unknown slot makes absence ambiguous.
            prefix = f'b13-recovery-v1-{claim_id}-'
            names = os.listdir(fd)
            allowed = {record_name(claim_id, t) for t in _TYPES}
            if any(prefix in name and name not in allowed for name in names):
                raise RecoveryEvidenceError('ambiguous recovery slot')
            records = {t: _read_slot(directory, fd, record_name(claim_id, t)) for t in _TYPES}
            records = {t: decode_record(raw, t) if raw is not None else None for t, raw in records.items()}
            preparation, semantics = first
            semantic = next(s for s in semantics if s.packet.window_id == window_id)
            binding = {'claim_identity': claim_id, 'claim_bytes_hash': _hash(expected),
                       'preparation_identity': identity['preparation_identity'], 'window_id': window_id,
                       'window_ordinal': identity['window_ordinal'], 'request_identity': identity['request_identity'],
                       'semantic_request_hash': semantic.request.request_hash,
                       'authority_hash': _authority_hash(preparation)}
            if any(r is not None and r['binding'] != binding for r in records.values()):
                raise RecoveryEvidenceError('binding mismatch')
            i, r, d, c, t = (records[x] for x in _TYPES)
            if bool(i) != bool(r): raise RecoveryEvidenceError('incomplete root initialization')
            if i and (r['intent_identity'] != i['content_hash'] or r['parent'] != i['parent'] or r['root']['path'] != i['desired_path'] or r['root'] != _root_identity(checkpoint_root) or r['parent'] != _root_identity(Path(checkpoint_root).parent)):
                raise RecoveryEvidenceError('root initialization mismatch')
            if i:
                parent_stat = os.lstat(Path(checkpoint_root).parent)
                root_stat = os.lstat(checkpoint_root)
                root_fd = os.open(checkpoint_root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
                try:
                    opened = os.fstat(root_fd)
                    if (parent_stat.st_uid != os.geteuid() or parent_stat.st_mode & 0o022
                            or not stat.S_ISDIR(opened.st_mode) or not claim._private(opened)
                            or not claim._same(root_stat, opened)
                            or not claim._same(opened, os.lstat(checkpoint_root))
                            or opened.st_uid != r['root_uid']
                            or stat.S_IMODE(opened.st_mode) != r['root_mode']):
                        raise RecoveryEvidenceError('unsafe initialized root or parent')
                finally:
                    os.close(root_fd)
            if d:
                roots = {'execution': _root_identity(execution_root), 'checkpoint': _root_identity(checkpoint_root),
                         'artifact': _root_identity(store.root), 'pointer': _root_identity(pointers.root)}
                paths = [Path(item['path']) for item in roots.values()]
                if any(a == b or a.is_relative_to(b) or b.is_relative_to(a)
                       for index, a in enumerate(paths) for b in paths[index + 1:]):
                    raise RecoveryEvidenceError('overlapping roots')
                if len({(item['dev'], item['ino']) for item in roots.values()}) != len(roots):
                    raise RecoveryEvidenceError('aliased roots')
                if any(not claim._private(os.lstat(p)) for p in paths[:2]):
                    raise RecoveryEvidenceError('unsafe recovery root')
                if d['roots'] != roots or d['root_result_identity'] != (r['content_hash'] if r else None):
                    raise RecoveryEvidenceError('root binding mismatch')
            if c and (not d or c['decision_identity'] != d['content_hash'] or c['decision_id'] != d['decision_id'] or c['attempt_id'] != d['attempt_id'] or c['bindings_hash'] != content_hash({'binding': binding, 'roots': d['roots'], 'caps': d['caps'], 'policy_reference': d['policy_reference']})):
                raise RecoveryEvidenceError('consumption mismatch')
            if t and (not c or t['decision_identity'] != d['content_hash'] or t['consumption_identity'] != c['content_hash'] or t['attempt_id'] != c['attempt_id']):
                raise RecoveryEvidenceError('terminal mismatch')
            if _verify(store, pointers, session) != first or storage.load_checkpoint(checkpoint_root, session, window_id) != accepted:
                raise RecoveryEvidenceError('authority or checkpoint changed')
            if (claim._inspect_at(directory / claim.claim_path(directory, session, window_id).name, fd, expected).state
                    is not claim.ClaimState.CLAIMED_UNRESOLVED
                    or set(os.listdir(fd)) != set(names)
                    or any(_read_slot(directory, fd, record_name(claim_id, kind)) !=
                           (canonical_json_bytes(records[kind]) if records[kind] is not None else None)
                           for kind in _TYPES)):
                raise RecoveryEvidenceError('claim or recovery slots changed')
            if i:
                parent_after = os.lstat(Path(checkpoint_root).parent)
                root_after = os.lstat(checkpoint_root)
                if (r['root'] != _root_identity(checkpoint_root)
                        or r['parent'] != _root_identity(Path(checkpoint_root).parent)
                        or parent_after.st_uid != os.geteuid() or parent_after.st_mode & 0o022
                        or root_after.st_uid != r['root_uid']
                        or stat.S_IMODE(root_after.st_mode) != r['root_mode']):
                    raise RecoveryEvidenceError('initialized root changed during inspection')
            if d and roots != {'execution': _root_identity(execution_root), 'checkpoint': _root_identity(checkpoint_root),
                               'artifact': _root_identity(store.root), 'pointer': _root_identity(pointers.root)}:
                raise RecoveryEvidenceError('roots changed during inspection')
            claim._check_root(directory, fd)
            if not verified: return Audit.BLOCKED
            if not c: return Audit.ORDINARY_AUDIT
            if not _text(approved_policy_reference) or approved_policy_reference != d['policy_reference']:
                return Audit.AUDIT_INCOMPLETE
            if not t or t['outcome'] != 'CHECKPOINTED': return Audit.AUDIT_INCOMPLETE
            checkpoint_directory = storage._root(checkpoint_root, create=False)
            if checkpoint_directory is None:
                raise RecoveryEvidenceError('checkpoint root disappeared')
            raw = storage._read_bounded(checkpoint_directory / storage._checkpoint_name(session, window_id))
            if (checkpoint.verify_and_decode_checkpoint(raw, session) != accepted
                    or raw != checkpoint.encode_checkpoint(accepted, session)):
                raise RecoveryEvidenceError('checkpoint bytes changed')
            payload = strict_json_loads(raw)
            if (t['checkpoint_identity'] != payload['checkpoint_identity'] or t['checkpoint_bytes_hash'] != _hash(raw)
                    or t['semantic_rounds'] != accepted.semantic_rounds or t['provider_attempts'] != accepted.provider_attempts):
                return Audit.AUDIT_INCOMPLETE
            return Audit.RECOVERY_AUDIT_COMPLETE
        finally:
            os.close(fd)
    except Exception:
        return Audit.AUDIT_INCOMPLETE if verified else Audit.BLOCKED
