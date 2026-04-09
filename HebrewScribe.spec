# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for HebrewScribe.

Usage:
    pyinstaller HebrewScribe.spec

Produces:  dist/HebrewScribe/HebrewScribe.exe  (one-dir bundle)
"""

import importlib.util
import os as _os

from PyInstaller.utils.hooks import collect_all, collect_data_files

# Read version from hebrewscribe/__init__.py so it stays in sync automatically.
_init_spec = importlib.util.spec_from_file_location(
    "hebrewscribe",
    _os.path.join(SPECPATH, "hebrewscribe", "__init__.py"),
)
_init_mod = importlib.util.module_from_spec(_init_spec)
_init_spec.loader.exec_module(_init_mod)
_APP_VERSION = _init_mod.__version__

block_cipher = None

# ---------------------------------------------------------------------------
# Collect everything for heavy dependencies that have dynamic imports / data
# ---------------------------------------------------------------------------
faster_datas, faster_binaries, faster_hiddens = collect_all("faster_whisper")
ct2_datas, ct2_binaries, ct2_hiddens = collect_all("ctranslate2")
tkdnd_datas, tkdnd_binaries, tkdnd_hiddens = collect_all("tkinterdnd2")
onnx_datas, onnx_binaries, onnx_hiddens = collect_all("onnxruntime")

all_datas = (
    faster_datas + ct2_datas + tkdnd_datas + onnx_datas
    + [("LICENSE", "."), ("THIRD-PARTY-LICENSES.md", "."), ("tests/fixtures/speech_6s.wav", "."), ("icon.ico", ".")]
    + [("hebrewscribe/icons/*.png", "hebrewscribe/icons"), ("hebrewscribe/icon-64.png", "hebrewscribe")]
)
all_binaries = faster_binaries + ct2_binaries + tkdnd_binaries + onnx_binaries
all_hiddenimports = (
    faster_hiddens + ct2_hiddens + tkdnd_hiddens + onnx_hiddens
    + ["hebrewscribe", "hebrewscribe.app", "hebrewscribe.utils",
       "hebrewscribe.models", "hebrewscribe.worker", "hebrewscribe.outputs",
       "hebrewscribe.power", "hebrewscribe.theme", "hebrewscribe.widgets",
       "hebrewscribe.icons"]
)

a = Analysis(
    ["HebrewScribe.pyw"],
    pathex=[],
    binaries=all_binaries,
    datas=all_datas,
    hiddenimports=all_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="HebrewScribe",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,          # no console window (Tkinter GUI app)
    disable_windowed_traceback=False,
    icon="icon.ico",
    version="version_info.py",
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="HebrewScribe",
)
