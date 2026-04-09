# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for HebrewScribe — macOS .app bundle.

Usage:
    pyinstaller HebrewScribe-mac.spec

Produces:  dist/HebrewScribe.app  (macOS application bundle)

Prerequisites:
    - ffmpeg and ffprobe must be in installer/ffmpeg-mac/
      Source binaries live in ../build-assets/ffmpeg-mac/ (outside the repo).
      Copy them before building: cp ../build-assets/ffmpeg-mac/* installer/ffmpeg-mac/
    - All Python dependencies installed in the active environment
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
onnx_datas, onnx_binaries, onnx_hiddens = collect_all("onnxruntime")

# tkinterdnd2 — optional, may not be installed on macOS
try:
    tkdnd_datas, tkdnd_binaries, tkdnd_hiddens = collect_all("tkinterdnd2")
except Exception:
    tkdnd_datas, tkdnd_binaries, tkdnd_hiddens = [], [], []

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
    noarchive=False,
    cipher=block_cipher,
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
    upx=False,               # UPX not recommended on macOS
    console=False,            # no terminal window (Tkinter GUI app)
    disable_windowed_traceback=False,
    icon="icon.icns",
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="HebrewScribe",
)

app = BUNDLE(
    coll,
    name="HebrewScribe.app",
    icon="icon.icns",
    bundle_identifier="com.hebrewscribe.app",
    info_plist={
        "CFBundleName": "HebrewScribe",
        "CFBundleDisplayName": "HebrewScribe",
        "CFBundleShortVersionString": _APP_VERSION,
        "CFBundleVersion": _APP_VERSION,
        "NSHighResolutionCapable": True,
        "LSMinimumSystemVersion": "12.0",
        "NSMicrophoneUsageDescription": "HebrewScribe does not use the microphone.",
    },
)
