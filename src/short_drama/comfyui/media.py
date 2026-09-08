from __future__ import annotations
from pathlib import Path
from typing import Any

VIDEO_EXTENSIONS = {".mp4", ".webm", ".mkv", ".mov"}

def extract_video_refs(history_entry: dict[str, Any]) -> list[dict[str, str]]:
    """Find SavedResult-like dicts regardless of whether history labels them videos/images/etc."""
    found: list[dict[str, str]] = []
    seen = set()

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            filename = value.get("filename")
            if isinstance(filename, str) and Path(filename).suffix.lower() in VIDEO_EXTENSIONS:
                item = {
                    "filename": filename,
                    "subfolder": str(value.get("subfolder", "")),
                    "type": str(value.get("type", "output")),
                }
                key = tuple(item.values())
                if key not in seen:
                    seen.add(key); found.append(item)
            for child in value.values(): walk(child)
        elif isinstance(value, list):
            for child in value: walk(child)

    walk(history_entry.get("outputs", {}))
    return found

def resolve_local_output(media_ref: dict[str, str], output_dir: str | Path) -> Path:
    """Resolve a ComfyUI output reference on the local host.

    ComfyUI running on Windows may return ``subfolder`` using backslashes even
    when the adapter is running under WSL/Linux. Normalize both separator
    styles before composing the local POSIX path.
    """
    subfolder = str(media_ref.get("subfolder", "")).replace("\\", "/").strip("/")
    filename = str(media_ref["filename"]).replace("\\", "/").split("/")[-1]
    return Path(output_dir) / Path(subfolder) / filename
