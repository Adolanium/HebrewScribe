"""Reusable tkinter widgets: ToolTip, TaskbarProgress."""

import logging
import sys
import tkinter as tk
from typing import Optional

from hebrewscribe.theme import Colors as C, _fs
from hebrewscribe.utils import SYSTEM_FONT

logger = logging.getLogger("HebrewScribe")


class ToolTip:
    """Lightweight hover tooltip for any tkinter widget.

    *text* can be a string or a callable returning a string.  When callable,
    the text is resolved fresh each time the tooltip is shown, enabling
    dynamic content that reflects current application state.
    """

    def __init__(self, widget: tk.Widget, text, delay: int = 400) -> None:
        self.widget = widget
        self._text = text
        self.delay = delay
        self._tipwindow: Optional[tk.Toplevel] = None
        self._after_id: Optional[str] = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._cancel, add="+")
        widget.bind("<ButtonPress>", self._cancel, add="+")

    def _schedule(self, event=None) -> None:
        self._cancel()
        self._after_id = self.widget.after(self.delay, self._show)

    def _cancel(self, event=None) -> None:
        if self._after_id:
            self.widget.after_cancel(self._after_id)
            self._after_id = None
        self._hide()

    def _dpi_scale(self) -> float:
        """Read the DPI scale from the root window, falling back to 1.0."""
        try:
            # _root(), not winfo_toplevel(): for widgets inside a dialog the
            # nearest toplevel is the dialog, which has no _dpi_scale — the
            # attribute lives on the application root.
            root = self.widget._root()
            return getattr(root, "_dpi_scale", 1.0)
        except Exception:
            return 1.0

    @property
    def text(self) -> str:
        """Resolve text, calling it if it's a callable."""
        return self._text() if callable(self._text) else self._text

    @text.setter
    def text(self, value) -> None:
        self._text = value

    def _show(self) -> None:
        if self._tipwindow:
            return
        resolved = self.text
        if not resolved:
            return
        scale = self._dpi_scale()
        offset_x = round(20 * scale)
        wrap = round(340 * scale)
        x = self.widget.winfo_rootx() + offset_x
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + round(4 * scale)
        tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        label = tk.Label(
            tw, text=resolved, justify="left", wraplength=wrap,
            background=C.TOOLTIP_BG, foreground=C.TOOLTIP_FG, relief="flat", borderwidth=0,
            font=(SYSTEM_FONT, _fs(9)), padx=round(10 * scale), pady=round(6 * scale),
        )
        label.pack()
        # Clamp to the screen so tips near the right/bottom edges stay visible.
        # Never slide the tip up over the cursor (the <Leave> binding would
        # hide it and cause a show/hide flicker loop) — flip above the widget
        # instead. Tk reports only the primary monitor's size on Windows, so
        # on secondary monitors this clamp may not trigger; acceptable.
        tw.update_idletasks()
        tip_w, tip_h = tw.winfo_reqwidth(), tw.winfo_reqheight()
        screen_w = self.widget.winfo_screenwidth()
        screen_h = self.widget.winfo_screenheight()
        x = max(0, min(x, screen_w - tip_w))
        if y + tip_h > screen_h:
            y = self.widget.winfo_rooty() - tip_h - round(4 * scale)
        tw.wm_geometry(f"+{x}+{y}")
        self._tipwindow = tw

    def _hide(self) -> None:
        if self._tipwindow:
            self._tipwindow.destroy()
            self._tipwindow = None


class TaskbarProgress:
    """Thin wrapper around ITaskbarList3 for Windows taskbar progress overlay.

    Falls back to no-ops on non-Windows platforms or when COM init fails.
    """

    TBPF_NOPROGRESS = 0x00
    TBPF_INDETERMINATE = 0x01
    TBPF_NORMAL = 0x02
    TBPF_ERROR = 0x04
    TBPF_PAUSED = 0x08

    def __init__(self) -> None:
        self._taskbar = None
        self._hwnd: Optional[int] = None

    def bind(self, tk_window: tk.Tk) -> None:
        """Attempt to acquire the COM interface. Safe to call on any platform."""
        if sys.platform != "win32":
            return
        try:
            import ctypes
            self._init_ctypes(tk_window)
        except Exception:
            logger.debug("Taskbar progress not available", exc_info=True)

    def _init_ctypes(self, tk_window: tk.Tk) -> None:
        """Pure-ctypes ITaskbarList3 via raw COM vtable — no comtypes needed."""
        import ctypes

        # Little-endian GUID byte layouts (data1/2/3 byte-swapped, data4 raw):
        # CLSID_TaskbarList  {56FDF344-FD6D-11D0-958A-006097C9A090}
        # IID_ITaskbarList3  {EA1AFB91-9E28-4B86-90E9-9E9F8A5EEFAF}
        # The IID previously encoded here was a wrong GUID — CoCreateInstance
        # answered E_NOINTERFACE on every machine and taskbar progress was
        # silently dead (fixed 2026-08-12, verified via ctypes probe).
        CLSID_TaskbarList = b"\x44\xf3\xfd\x56\x6d\xfd\xd0\x11\x95\x8a\x00\x60\x97\xc9\xa0\x90"
        IID_ITaskbarList3 = b"\x91\xfb\x1a\xea\x28\x9e\x86\x4b\x90\xe9\x9e\x9f\x8a\x5e\xef\xaf"

        ole32 = ctypes.windll.ole32
        ole32.CoInitialize(None)

        clsid = (ctypes.c_byte * 16)(*CLSID_TaskbarList)
        iid = (ctypes.c_byte * 16)(*IID_ITaskbarList3)
        p = ctypes.c_void_p()
        hr = ole32.CoCreateInstance(
            ctypes.byref(clsid), None, 1,  # CLSCTX_INPROC_SERVER
            ctypes.byref(iid), ctypes.byref(p)
        )
        if hr != 0 or not p.value:
            raise OSError(f"CoCreateInstance failed: 0x{hr:08X}")

        self._vtable = ctypes.cast(
            ctypes.cast(p, ctypes.POINTER(ctypes.c_void_p))[0],
            ctypes.POINTER(ctypes.c_void_p)
        )
        self._p = p
        self._hwnd = ctypes.windll.user32.GetParent(tk_window.winfo_id())

        # HrInit is vtable index 3
        ctypes.WINFUNCTYPE(ctypes.HRESULT, ctypes.c_void_p)(self._vtable[3])(self._p)
        logger.debug("Taskbar progress: COM interface acquired")

    def set_progress(self, current: int, total: int) -> None:
        """Set taskbar progress value (0..total)."""
        if not hasattr(self, "_vtable") or self._hwnd is None:
            return
        import ctypes
        try:
            fn = ctypes.WINFUNCTYPE(
                ctypes.HRESULT, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_ulonglong, ctypes.c_ulonglong
            )(self._vtable[9])  # SetProgressValue
            fn(self._p, self._hwnd, current, total)
        except Exception:
            pass

    def set_state(self, flag: int) -> None:
        """Set taskbar progress state (TBPF_* constants)."""
        if not hasattr(self, "_vtable") or self._hwnd is None:
            return
        import ctypes
        try:
            fn = ctypes.WINFUNCTYPE(
                ctypes.HRESULT, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_int
            )(self._vtable[10])  # SetProgressState
            fn(self._p, self._hwnd, flag)
        except Exception:
            pass

    def clear(self) -> None:
        self.set_state(self.TBPF_NOPROGRESS)
