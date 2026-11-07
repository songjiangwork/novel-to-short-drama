"""Read-only v1/v2 conflict inspection, used by the opt-in feedback runner.

Older readers are unaware of v2. Before any production rollout, stop all old
writers, fence their access, complete version cutover, and under the original
execution guard revalidate roots, authority, claims and every checkpoint. A
directory scan is not an atomic snapshot and cannot constrain old processes.
This helper itself neither enables v2 writes nor supplies dispatch authorization.
The runner supplies an execution guard and explicit selected-window policy.
"""
from __future__ import annotations

import os
import stat
from pathlib import Path

from . import disposition_checkpoint_storage as storage
from .disposition_checkpoint import verify_and_decode_checkpoint
from .disposition_checkpoint_v2 import verify_and_decode_feedback_checkpoint_v2


class B13OfflineCheckpointConflict(ValueError):
    """An expected checkpoint is missing, ambiguous or invalid."""


def inspect_checkpoint_versions_offline(root, session, window_id):
    """Read exactly one v1/v2 evidence file, or None when none is present.

    Unknown versions, malformed attributable names, different base requests and
    multiple files are errors. This does not create directories or write files.
    """
    try:
        preparation, ordinal, request = storage._session_window(session, window_id)
        directory = storage._root(root, create=False)
        if directory is None:
            return None
        root_before = os.stat(directory, follow_symlinks=False)
        if not stat.S_ISDIR(root_before.st_mode):
            raise ValueError
        candidates = storage._related_checkpoint_names(directory, preparation, ordinal)
        if not candidates:
            if (storage._related_checkpoint_names(directory, preparation, ordinal) != candidates
                    or not storage._stable_file(root_before, os.stat(directory, follow_symlinks=False))):
                raise ValueError
            return None
        if len(candidates) != 1:
            raise ValueError
        name = candidates[0]
        suffix = f'-p{preparation}-w{ordinal:08d}-q{request}.json'
        if name not in (f'b13-checkpoint-v1{suffix}', f'b13-checkpoint-v2{suffix}'):
            raise ValueError
        path = Path(directory) / name
        before = storage._target_stat(path)
        is_feedback_v2 = name.startswith('b13-checkpoint-v2-')
        # v2 persists a rejected, schema-valid Provider candidate. Unlike
        # historical v1 evidence, that carrier requires strict confidentiality
        # on both its directory and its hard-link-free final file.
        if is_feedback_v2 and (
            root_before.st_uid != os.geteuid() or stat.S_IMODE(root_before.st_mode) != 0o700
            or before.st_uid != os.geteuid() or stat.S_IMODE(before.st_mode) != 0o600
            or before.st_nlink != 1
        ):
            raise ValueError
        raw = storage._read_bounded(path)
        if name.startswith('b13-checkpoint-v1-'):
            accepted = verify_and_decode_checkpoint(raw, session)
        else:
            accepted = verify_and_decode_feedback_checkpoint_v2(raw, session)
        if (accepted.window_id != window_id or accepted.window_ordinal != ordinal
                or accepted.request_identity != request):
            raise ValueError
        after = storage._target_stat(path)
        if (before is None or after is None or not storage._stable_file(before, after)
                or (is_feedback_v2 and (
                    after.st_uid != os.geteuid() or stat.S_IMODE(after.st_mode) != 0o600
                    or after.st_nlink != 1))
                or storage._related_checkpoint_names(directory, preparation, ordinal) != candidates
                or not storage._stable_file(root_before, os.stat(directory, follow_symlinks=False))):
            raise ValueError
        return accepted
    except Exception:
        raise B13OfflineCheckpointConflict('checkpoint evidence is invalid or conflicts') from None
