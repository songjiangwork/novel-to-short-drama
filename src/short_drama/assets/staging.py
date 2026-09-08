from __future__ import annotations
from copy import deepcopy
import os
import re
import shutil
from pathlib import Path
from typing import Any

_WINDOWS_DRIVE = re.compile(r"^([A-Za-z]):[\\/](.*)$")

def host_path(path: str | Path) -> Path:
    """Resolve canonical Windows paths when the runtime is executing under WSL."""
    s = str(path)
    if os.name != "nt":
        match = _WINDOWS_DRIVE.match(s)
        if match:
            drive, tail = match.groups()
            parts = [p for p in re.split(r"[\\/]+", tail) if p]
            return Path("/mnt") / drive.lower() / Path(*parts)
    return Path(s)

def staged_name(asset_id: str, source_path: str) -> str:
    return f"{asset_id.replace('.', '__')}{host_path(source_path).suffix.lower()}"

def stage_references(refs: dict[str, Any], input_dir: str | Path, project_id: str, shot_id: str) -> dict[str, Any]:
    input_root = host_path(input_dir)
    destination = input_root / "short_drama" / project_id / shot_id / "assets"
    staged = deepcopy(refs)
    for kind in ("images", "audios"):
        for ref in staged.get(kind, []):
            src = host_path(ref["source_path"])
            if not src.is_file():
                raise FileNotFoundError(f"Canonical asset not found: {src}")
            dst = destination / staged_name(ref["asset_id"], str(src))
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            ref["runtime_filename"] = dst.relative_to(input_root).as_posix()
    return staged
