"""Experimental read-only v1/v2 conflict inspection; not wired to the runner.

Older readers are unaware of v2. Production rollout must first gate all writers
and integrate version-aware inspection; this helper cannot make them safe.
"""
from __future__ import annotations

import os
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
        marker = f'-p{preparation}-w{ordinal:08d}'
        candidates = [name for name in os.listdir(directory)
                      if name.startswith('b13-checkpoint-') and marker in name]
        if not candidates:
            return None
        if len(candidates) != 1:
            raise ValueError
        name = candidates[0]
        suffix = f'-p{preparation}-w{ordinal:08d}-q{request}.json'
        if name not in (f'b13-checkpoint-v1{suffix}', f'b13-checkpoint-v2{suffix}'):
            raise ValueError
        raw = storage._read_bounded(Path(directory) / name)
        if name.startswith('b13-checkpoint-v1-'):
            return verify_and_decode_checkpoint(raw, session)
        return verify_and_decode_feedback_checkpoint_v2(raw, session)
    except Exception:
        raise B13OfflineCheckpointConflict('checkpoint evidence is invalid or conflicts') from None
