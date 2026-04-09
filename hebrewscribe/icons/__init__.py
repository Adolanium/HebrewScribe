"""Icon loader for toolbar PNG icons.

Loads pre-rendered PNG icons at the appropriate DPI scale.
Falls back gracefully to None if icons are missing (callers
show text-only buttons in that case).

Icons are rendered from SVGs by scripts/render_icons.py at build time.
"""

import logging
import tkinter as tk
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger("HebrewScribe")

# Module-level cache — PhotoImage instances must be kept alive or tkinter
# garbage-collects them and shows blank widgets.
_icon_cache: Dict[str, "tk.PhotoImage"] = {}

# Directory containing the rendered PNG icons (same directory as this __init__.py)
_ICON_DIR = Path(__file__).parent

# Available scale factors and their filename suffixes
_SCALES = [
    (1.0, "1x"),
    (1.5, "1_5x"),
    (2.0, "2x"),
    (3.0, "3x"),
]


def load_icon(name: str, dpi_scale: float = 1.0) -> Optional["tk.PhotoImage"]:
    """Load a toolbar icon PNG at the appropriate DPI scale.

    Selects the nearest available scale factor to match the current
    display DPI. Returns None if the icon file is missing (graceful
    degradation to text-only buttons).

    Args:
        name: Icon base name without extension (e.g., "add-file")
        dpi_scale: Current display DPI scale factor (1.0 = 96 DPI)

    Returns:
        tk.PhotoImage instance (cached) or None if not found
    """
    # Pick the nearest scale
    best_suffix = "1x"
    best_dist = float("inf")
    for scale_val, suffix in _SCALES:
        dist = abs(dpi_scale - scale_val)
        if dist < best_dist:
            best_dist = dist
            best_suffix = suffix

    cache_key = f"{name}_{best_suffix}"
    if cache_key in _icon_cache:
        return _icon_cache[cache_key]

    # Try the best scale, then fall back to 1x
    for suffix in (best_suffix, "1x"):
        icon_path = _ICON_DIR / f"{name}_{suffix}.png"
        if icon_path.exists():
            try:
                img = tk.PhotoImage(file=str(icon_path))
                _icon_cache[cache_key] = img
                return img
            except Exception:
                logger.debug("Failed to load icon %s", icon_path, exc_info=True)
                return None

    # Icon not found — caller falls back to text-only
    return None
