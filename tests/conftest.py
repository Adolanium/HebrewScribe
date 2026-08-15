"""Shared test configuration: stubs tkinter before any test imports hebrewscribe.app."""

import sys
import types

# Prevent tkinter from launching a GUI during import.
# Must run before any test module imports hebrewscribe.app.
if "tkinter" not in sys.modules:
    _tk_stub = types.ModuleType("tkinter")
    _tk_stub.Tk = type("Tk", (), {"__init__": lambda *a, **kw: None})
    _tk_stub.Widget = object
    _tk_stub.Toplevel = object
    _tk_stub.Label = object
    _tk_stub.Text = object
    _tk_stub.Entry = object
    _tk_stub.Menu = object
    _tk_stub.Button = object
    _tk_stub.Frame = object
    _tk_stub.Canvas = object
    _tk_stub.PhotoImage = type("PhotoImage", (), {"__init__": lambda *a, **kw: None})
    _tk_stub.StringVar = lambda *a, **kw: None
    _tk_stub.BooleanVar = lambda *a, **kw: None
    _tk_stub.filedialog = types.ModuleType("tkinter.filedialog")
    _tk_stub.messagebox = types.ModuleType("tkinter.messagebox")
    # No-op dialogs so app methods exercised headless (e.g. _append_files'
    # duplicate notice, clear_files' confirm) don't need a display.
    _tk_stub.messagebox.showinfo = lambda *a, **kw: None
    _tk_stub.messagebox.showwarning = lambda *a, **kw: None
    _tk_stub.messagebox.showerror = lambda *a, **kw: None
    _tk_stub.messagebox.askyesno = lambda *a, **kw: True
    _tk_stub.ttk = types.ModuleType("tkinter.ttk")
    for _name in ("Frame", "LabelFrame", "Panedwindow", "Label", "Button", "Entry",
                   "Combobox", "Checkbutton", "Treeview", "Scrollbar", "Progressbar",
                   "Notebook", "Separator"):
        setattr(_tk_stub.ttk, _name, type(_name, (), {}))
    # Style stub for theme.py
    _style_stub = type("Style", (), {
        "__init__": lambda *a, **kw: None,
        "theme_create": lambda *a, **kw: None,
        "theme_use": lambda *a, **kw: None,
        "theme_names": lambda *a, **kw: (),
    })
    _tk_stub.ttk.Style = _style_stub
    sys.modules["tkinter"] = _tk_stub
    sys.modules["tkinter.filedialog"] = _tk_stub.filedialog
    sys.modules["tkinter.messagebox"] = _tk_stub.messagebox
    sys.modules["tkinter.ttk"] = _tk_stub.ttk
