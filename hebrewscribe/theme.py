"""Modern flat ttk theme and color system (self-contained, no external dependency)."""

import logging
import sys

logger = logging.getLogger("HebrewScribe")


# --- Platform-adaptive font sizing (Phase 1.1) ---
# SF Pro (macOS) renders ~8% taller than Segoe UI (Windows) at the same pt size.
_FONT_SIZE_OFFSET = -1 if sys.platform == "darwin" else 0

def _fs(base: int) -> int:
    """Adjust a font point size for the current platform."""
    return max(7, base + _FONT_SIZE_OFFSET)


# --- Platform-adaptive layout (Phase 1.2) ---
if sys.platform == "darwin":
    PAD_CARD_INNER_X = 10
    PAD_CARD_INNER_Y = 6
    PAD_TOOLBAR_GAP = 6
    PAD_SECTION_BELOW = 6
else:
    PAD_CARD_INNER_X = 12
    PAD_CARD_INNER_Y = 8
    PAD_TOOLBAR_GAP = 8
    PAD_SECTION_BELOW = 8

# --- Platform-adaptive row height (Phase 1.3) ---
_BASE_ROWHEIGHT = 26 if sys.platform == "darwin" else 28


class Colors:
    """Centralized color palette for HebrewScribe UI.

    All GUI colors are defined here. No hex literals should appear in app.py
    or widgets.py — use Colors.CONSTANT_NAME instead.

    Brand colors are from context.md and must not be changed without
    updating the project branding documentation.
    """

    # --- Surface hierarchy (light to dark) ---
    BG_PRIMARY = "#fafafa"         # window/frame background
    BG_CARD = "#f6f7f9"            # card/panel fill (warmed from #f5f7fa)
    BG_CARD_ALT = "#f0f2f5"        # alternate card (hover, nested)
    BG_ELEVATED = "#ffffff"        # treeview, inputs (highest surface)
    BG_STRIPE = "#f7f7f7"          # treeview alternating row

    # --- Borders ---
    BORDER_CARD = "#e0e4ea"        # card outlines
    BORDER_SEPARATOR = "#d0d4da"   # explicit dividers
    BORDER_LIGHT = "#e0e0e0"       # separator, progress track background
    CARD_ACCENT_STRIPE = "#c0c8d4" # left-edge accent stripe (neutral grey)

    # --- Sash grip (panel splitter affordance) ---
    SASH_GRIP = "#c8cdd5"            # pill at rest — cool grey, border family
    SASH_GRIP_HOVER = "#a8b0bc"      # pill on hover — darker to confirm interactivity
    SASH_BG_HOVER = "#e8ecf0"        # sash zone on hover
    # Sash background at rest = BG_PRIMARY (invisible seam, Apple Split View pattern)

    # --- Text hierarchy ---
    TEXT_PRIMARY = "#1a1a1a"       # headings, body
    TEXT_HEADING = "#333333"       # heading variant (slightly lighter)
    TEXT_SECONDARY = "#555555"     # labels, descriptions
    TEXT_TERTIARY = "#636363"      # hints, low-emphasis
    TEXT_MUTED = "#666666"         # idle/muted text
    TEXT_DISABLED = "#a0a0a0"      # truly disabled
    TEXT_PLACEHOLDER = "#c0c0c0"   # placeholder icons

    # --- Brand accent (from context.md — do not change) ---
    ACCENT = "#2a5cdb"
    ACCENT_HOVER = "#4a7aef"
    ACCENT_PRESS = "#1a44b0"

    # --- Button styles ---
    BTN_SECONDARY_BG = "#e8e8e8"
    BTN_SECONDARY_HOVER = "#d4d4d4"
    BTN_SECONDARY_PRESS = "#c0c0c0"
    BTN_DISABLED_BG = "#f0f0f0"
    BTN_DISABLED_FG = "#a0a0a0"

    BTN_PRIMARY_DISABLED_BG = "#94a3b8"
    BTN_PRIMARY_DISABLED_FG = "#e2e8f0"

    BTN_DESTRUCTIVE_FG = "#dc2626"
    BTN_DESTRUCTIVE_HOVER = "#fee2e2"
    BTN_DESTRUCTIVE_PRESS = "#fecaca"

    # --- Semantic status ---
    STATUS_IDLE_BG = "#e8ecf0"     # refined from #f0f0f0 for better contrast
    STATUS_IDLE_FG = "#666666"

    STATUS_RUNNING_BG = "#d6e2f9"
    STATUS_RUNNING_FG = "#1a3a7a"

    STATUS_PAUSED_BG = "#fef3c7"
    STATUS_PAUSED_FG = "#92400e"

    STATUS_SUCCESS_BG = "#dcfce7"
    STATUS_SUCCESS_FG = "#166534"

    STATUS_ERROR_BG = "#fee2e2"
    STATUS_ERROR_FG = "#991b1b"

    STATUS_WARNING_FG = "#996600"

    # --- Activity tree / log ---
    LOG_GROUP_FG = "#4a5568"       # group headers in activity tree
    LOG_ERROR_FG = "#cc0000"       # error in activity tree and raw log
    LOG_PASS_FG = "#22883a"        # selftest pass
    LOG_FAIL_FG = "#cc3333"        # selftest fail

    # --- Interactive highlights ---
    HIGHLIGHT_FLASH = "#e8f0fe"    # row flash after reorder
    HIGHLIGHT_HOVER = "#eef2ff"    # row hover in treeview

    # --- Card interactive states (model cards) ---
    CARD_SELECTED_BG = "#e0ecff"
    CARD_SELECTED_BORDER = "#2a5cdb"
    CARD_HOVER_BG = "#eef2f7"
    CARD_DEFAULT_BG = "#f6f7f9"    # matches BG_CARD
    CARD_DEFAULT_BORDER = "#e0e4ea"
    CARD_RECOMMENDED_BG = "#eff6ff"

    # --- Tooltip ---
    TOOLTIP_BG = "#1a1a1a"
    TOOLTIP_FG = "#f0f0f0"

    # --- Scrollbar ---
    SCROLLBAR_BG = "#e8e8e8"
    SCROLLBAR_TROUGH = "#f5f5f5"
    SCROLLBAR_ACTIVE = "#c0c0c0"

    # --- Action bar shadow gradient (top to bottom, painted above action bar) ---
    SHADOW_LIGHT = "#e8e8e8"
    SHADOW_MED = "#e0e0e0"
    SHADOW_DARK = "#d8d8d8"


def install_modern_theme(dpi_scale: float = 1.0) -> None:
    """Register a flat, modern ttk theme and apply it.

    Inspired by Azure/Sun Valley but hand-rolled to avoid external deps.
    Transforms comboboxes, buttons, scrollbars, notebook tabs, treeview,
    progress bars, and checkbuttons from Win32-era to contemporary flat look.

    Args:
        dpi_scale: DPI scale factor (1.0 = 96 DPI). Pixel values in the
            theme (rowheight, scrollbar width, sash thickness, progress bar
            thickness) are multiplied by this factor.
    """
    # Import tkinter here so Colors can be imported without a display
    from tkinter import ttk

    from hebrewscribe.utils import SYSTEM_FONT

    def _s(px: int) -> int:
        return round(px * dpi_scale)

    style = ttk.Style()

    # Base theme: clam on all platforms.
    # Aqua (macOS native) was tested but conflicts with system dark mode:
    # aqua inherits the system foreground colors (light text for dark mode)
    # while our theme forces light backgrounds, producing invisible text.
    # Deferred to a future PR that adds full dark mode support.
    base_theme = "clam"
    use_aqua = False

    # Use 'aqua' (macOS) or 'clam' as base — most customizable
    try:
        style.theme_create("modern", parent=base_theme, settings={

            # --- Global defaults ---
            ".": {
                "configure": {
                    "background": Colors.BG_PRIMARY,
                    "foreground": Colors.TEXT_PRIMARY,
                    "borderwidth": 0,
                    "focusthickness": 0,
                    "font": (SYSTEM_FONT, _fs(10)),
                },
            },

            # --- Frames ---
            "TFrame": {
                "configure": {"background": Colors.BG_PRIMARY},
            },
            "TLabelframe": {
                "configure": {"background": Colors.BG_PRIMARY, "borderwidth": 0},
            },

            # --- Labels ---
            "TLabel": {
                "configure": {
                    "background": Colors.BG_PRIMARY,
                    "foreground": Colors.TEXT_PRIMARY,
                    "padding": (0, 2),
                },
            },

            # --- Buttons ---
            "TButton": {
                "configure": {
                    "background": Colors.BTN_SECONDARY_BG,
                    "foreground": Colors.TEXT_PRIMARY,
                    "borderwidth": 1,
                    "relief": "flat",
                    "padding": (12, 5),
                    "anchor": "center",
                    "font": (SYSTEM_FONT, _fs(10)),
                },
                "map": {
                    "background": [
                        ("active", Colors.BTN_SECONDARY_HOVER),
                        ("disabled", Colors.BTN_DISABLED_BG),
                    ],
                    "foreground": [
                        ("disabled", Colors.BTN_DISABLED_FG),
                    ],
                    "relief": [
                        ("pressed", "flat"),
                        ("active", "flat"),
                    ],
                },
            },

            # --- Combobox ---
            "TCombobox": {
                "configure": {
                    "selectbackground": Colors.ACCENT,
                    "selectforeground": "white",
                    "fieldbackground": Colors.BG_ELEVATED,
                    "background": Colors.BG_ELEVATED,
                    "borderwidth": 1,
                    "padding": (6, 4),
                    "arrowsize": 14,
                },
                "map": {
                    "fieldbackground": [("readonly", Colors.BG_ELEVATED)],
                    "selectbackground": [("readonly", Colors.BG_ELEVATED)],
                    "selectforeground": [("readonly", Colors.TEXT_PRIMARY)],
                    "foreground": [("readonly", Colors.TEXT_PRIMARY)],
                },
            },

            # --- Entry ---
            "TEntry": {
                "configure": {
                    "fieldbackground": Colors.BG_ELEVATED,
                    "borderwidth": 1,
                    "padding": (6, 4),
                    "selectbackground": Colors.ACCENT,
                    "selectforeground": "white",
                },
            },

            # --- Checkbutton ---
            "TCheckbutton": {
                "configure": {
                    "background": Colors.BG_PRIMARY,
                    "foreground": Colors.TEXT_PRIMARY,
                    "indicatormargin": (1, 1, 4, 1),
                    "padding": (4, 2),
                },
                "map": {
                    "background": [("active", Colors.BG_CARD_ALT)],
                },
            },

            # --- Notebook (tabs) ---
            "TNotebook": {
                "configure": {
                    "background": Colors.BG_PRIMARY,
                    "borderwidth": 0,
                    "tabmargins": (2, 4, 2, 0),
                },
            },
            "TNotebook.Tab": {
                "configure": {
                    "background": Colors.BTN_SECONDARY_BG,
                    "foreground": Colors.TEXT_SECONDARY,
                    "padding": (14, 6),
                    "font": (SYSTEM_FONT, _fs(10)),
                },
                "map": {
                    "background": [
                        ("selected", Colors.BG_PRIMARY),
                        ("active", Colors.BG_CARD_ALT),
                    ],
                    "foreground": [
                        ("selected", Colors.TEXT_PRIMARY),
                    ],
                    "expand": [
                        ("selected", (1, 1, 1, 0)),
                    ],
                },
            },

            # --- Treeview ---
            "Treeview": {
                "configure": {
                    "background": Colors.BG_ELEVATED,
                    "foreground": Colors.TEXT_PRIMARY,
                    "fieldbackground": Colors.BG_ELEVATED,
                    "borderwidth": 0,
                    "rowheight": _s(_BASE_ROWHEIGHT),
                    "font": (SYSTEM_FONT, _fs(10)),
                },
                "map": {
                    "background": [("selected", Colors.STATUS_RUNNING_BG)],
                    "foreground": [("selected", Colors.STATUS_RUNNING_FG)],
                },
            },
            "Treeview.Heading": {
                "configure": {
                    "background": Colors.BG_CARD_ALT,
                    "foreground": Colors.TEXT_SECONDARY,
                    "borderwidth": 0,
                    "relief": "flat",
                    "font": (SYSTEM_FONT, _fs(9), "bold"),
                    "padding": (8, 4),
                },
                "map": {
                    "background": [("active", Colors.BTN_SECONDARY_BG)],
                },
            },

            # --- Scrollbar ---
            "Vertical.TScrollbar": {
                "configure": {
                    "background": Colors.SCROLLBAR_BG,
                    "troughcolor": Colors.SCROLLBAR_TROUGH,
                    "borderwidth": 0,
                    "arrowsize": 0,
                    "width": _s(10),
                },
                "map": {
                    "background": [
                        ("active", Colors.SCROLLBAR_ACTIVE),
                        ("disabled", Colors.BG_CARD_ALT),
                    ],
                },
            },
            "Horizontal.TScrollbar": {
                "configure": {
                    "background": Colors.SCROLLBAR_BG,
                    "troughcolor": Colors.SCROLLBAR_TROUGH,
                    "borderwidth": 0,
                    "arrowsize": 0,
                    "width": _s(10),
                },
                "map": {
                    "background": [
                        ("active", Colors.SCROLLBAR_ACTIVE),
                        ("disabled", Colors.BG_CARD_ALT),
                    ],
                },
            },

            # --- Progressbar ---
            "Horizontal.TProgressbar": {
                "configure": {
                    "background": Colors.ACCENT,
                    "troughcolor": Colors.SCROLLBAR_BG,
                    "borderwidth": 0,
                    "thickness": _s(8),
                },
            },

            # --- Separator ---
            "TSeparator": {
                "configure": {
                    "background": Colors.BORDER_LIGHT,
                },
            },

            # --- Panedwindow ---
            "TPanedwindow": {
                "configure": {
                    "background": Colors.BG_PRIMARY,
                },
            },
            "Sash": {
                "configure": {
                    "sashthickness": _s(8),
                    "gripcount": 0,
                },
            },
        })

    except Exception:
        logger.error("Failed to create modern ttk theme", exc_info=True)
        raise

    style.theme_use("modern")
    logger.debug("Applied modern ttk theme (base: clam)")
