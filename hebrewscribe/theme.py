"""Modern flat ttk theme and color system (self-contained, no external dependency)."""

import logging
import sys

logger = logging.getLogger("HebrewScribe")


# --- Platform-adaptive font sizing ---
# SF Pro (macOS) renders ~8% taller than Segoe UI (Windows) at the same pt size.
_FONT_SIZE_OFFSET = -1 if sys.platform == "darwin" else 0

def _fs(base: int) -> int:
    """Adjust a font point size for the current platform."""
    return max(7, base + _FONT_SIZE_OFFSET)


# --- Platform-adaptive layout ---
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

# --- Platform-adaptive row height ---
_BASE_ROWHEIGHT = 26 if sys.platform == "darwin" else 28


class Colors:
    """Centralized color palette for HebrewScribe UI.

    All GUI colors are defined here. No hex literals should appear in app.py
    or widgets.py — use Colors.CONSTANT_NAME instead.

    Brand accent colors are fixed project branding — do not change them.
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

    # --- Brand accent (fixed branding — do not change) ---
    ACCENT = "#2a5cdb"
    ACCENT_HOVER = "#4a7aef"
    ACCENT_PRESS = "#1a44b0"

    # --- Button styles ---
    BTN_SECONDARY_BG = "#e8e8e8"
    BTN_SECONDARY_HOVER = "#d4d4d4"
    BTN_SECONDARY_PRESS = "#c0c0c0"
    BTN_DISABLED_BG = "#f0f0f0"
    BTN_DISABLED_FG = "#a0a0a0"

    # Disabled-primary must read as "unavailable" like every other disabled
    # button — the old steel-blue #94a3b8 looked like a live control.
    BTN_PRIMARY_DISABLED_BG = "#dfe5ee"
    BTN_PRIMARY_DISABLED_FG = "#a0a0a0"

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

    # Base theme: clam on all platforms. Aqua (macOS native) conflicts with
    # system dark mode: aqua inherits the system foreground colors (light text
    # for dark mode) while this theme forces light backgrounds, producing
    # invisible text — so clam is used everywhere.
    base_theme = "clam"

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
                    # Kill clam's 3D bevel: flat 1px border, muted arrow
                    "bordercolor": Colors.BORDER_SEPARATOR,
                    "lightcolor": Colors.BG_ELEVATED,
                    "darkcolor": Colors.BG_ELEVATED,
                    "arrowcolor": Colors.TEXT_SECONDARY,
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
                    "bordercolor": Colors.BORDER_SEPARATOR,
                    "lightcolor": Colors.BG_ELEVATED,
                    "darkcolor": Colors.BG_ELEVATED,
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
                    "bordercolor": Colors.BORDER_CARD,
                    "lightcolor": Colors.BG_PRIMARY,
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
                    "bordercolor": Colors.BORDER_CARD,
                    "lightcolor": Colors.BG_ELEVATED,
                    "darkcolor": Colors.BG_ELEVATED,
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
                    "bordercolor": Colors.SCROLLBAR_TROUGH,
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
                    "bordercolor": Colors.SCROLLBAR_TROUGH,
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
    try:
        _install_check_indicator(style, _s)
    except Exception:
        # Non-fatal: clam's stock indicator remains.
        logger.warning("Could not install modern checkbutton indicator",
                       exc_info=True)
    logger.debug("Applied modern ttk theme (base: clam)")


# PhotoImages must outlive this function or Tk silently drops them.
_CHECK_IMAGES: list = []


def _install_check_indicator(style, _s) -> None:
    """Replace clam's X-in-a-box Checkbutton indicator with a check-in-square.

    An X in a box reads as "excluded", not "on". Images are drawn in code
    with PhotoImage.put — self-contained, DPI-scaled, no asset pipeline.
    Requires a live Tk root (install_modern_theme runs with one).
    """
    import tkinter as tk

    size = _s(15)
    gap = _s(5)  # transparent spacing between the box and the label

    def _draw(fill, border, check=None):
        img = tk.PhotoImage(width=size + gap, height=size)
        img.put(fill, to=(0, 0, size, size))
        img.put(border, to=(0, 0, size, 1))
        img.put(border, to=(0, size - 1, size, size))
        img.put(border, to=(0, 0, 1, size))
        img.put(border, to=(size - 1, 0, size, size))
        if check:
            t = max(2, size // 7)
            x0, y0 = round(size * 0.22), round(size * 0.48)
            x1, y1 = round(size * 0.42), round(size * 0.68)
            x2, y2 = round(size * 0.78), round(size * 0.28)
            for xa, ya, xb, yb in ((x0, y0, x1, y1), (x1, y1, x2, y2)):
                steps = max(1, xb - xa)
                for i in range(steps + 1):
                    x = xa + i
                    y = ya + round(i * (yb - ya) / steps)
                    img.put(check, to=(x, max(0, y - t + 1), x + 1, min(size, y + 1)))
        _CHECK_IMAGES.append(img)
        return img

    img_off = _draw(Colors.BG_ELEVATED, Colors.BORDER_SEPARATOR)
    img_on = _draw(Colors.ACCENT, Colors.ACCENT, check="white")
    img_dis_off = _draw(Colors.BTN_DISABLED_BG, Colors.BORDER_LIGHT)
    img_dis_on = _draw(Colors.TEXT_PLACEHOLDER, Colors.TEXT_PLACEHOLDER,
                       check="white")

    style.element_create(
        "Modern.Checkbutton.indicator", "image", img_off,
        ("disabled selected", img_dis_on),
        ("disabled", img_dis_off),
        ("selected", img_on),
        sticky="",
    )
    style.layout("TCheckbutton", [
        ("Checkbutton.padding", {"sticky": "nswe", "children": [
            ("Modern.Checkbutton.indicator", {"side": "left", "sticky": ""}),
            ("Checkbutton.focus", {"side": "left", "sticky": "w", "children": [
                ("Checkbutton.label", {"sticky": "nswe"}),
            ]}),
        ]}),
    ])
