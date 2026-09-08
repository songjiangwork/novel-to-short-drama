from __future__ import annotations
import math

ASPECT_RATIOS = {
    "1:1": (1, 1),
    "2:3": (2, 3),
    "3:2": (3, 2),
    "3:4": (3, 4),
    "4:3": (4, 3),
    "9:16": (9, 16),
    "16:9": (16, 9),
    "21:9": (21, 9),
}

def resolution_from_profile(megapixels: float, aspect_ratio: str, multiple: int = 32) -> tuple[int, int]:
    """Mirror ComfyUI core ResolutionSelector semantics (1024*1024 pixels per MP)."""
    if aspect_ratio not in ASPECT_RATIOS:
        raise ValueError(f"Unsupported aspect ratio: {aspect_ratio}")
    wr, hr = ASPECT_RATIOS[aspect_ratio]
    total_pixels = float(megapixels) * 1024 * 1024
    scale = math.sqrt(total_pixels / (wr * hr))
    width = round(wr * scale / multiple) * multiple
    height = round(hr * scale / multiple) * multiple
    return int(width), int(height)

def h3_length_for_seconds(seconds: float, fps: int = 24) -> int:
    """Mirror the frozen H3 17k+5 frame-grid formula from the validated workflow."""
    frames = max(5, round(float(seconds) * fps))
    return int(frames + (5 - (frames % 17)) % 17)
