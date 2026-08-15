# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for HebrewScribe — macOS .app bundle.

Usage:
    pyinstaller HebrewScribe-mac.spec

Produces:  dist/HebrewScribe.app  (macOS application bundle)

Prerequisites:
    - ffmpeg and ffprobe must be in installer/ffmpeg-mac/
      Source binaries live in build-assets/ffmpeg-mac/ (inside the repo root,
      gitignored). Copy them before building:
          cp build-assets/ffmpeg-mac/* installer/ffmpeg-mac/
    - All Python dependencies installed in the active environment

Diarization (optional but included in release builds):
    - Freeze env must be Python >= 3.10 on arm64 (torch >= 2.3 ships
      Apple-Silicon-only macOS wheels — a diarization-capable build is
      arm64-only): pip install "pyannote.audio>=4,<5" torch>=2.8 torchaudio>=2.8
      then: pip uninstall torchcodec -y  (the app feeds in-memory waveform
      dicts; pyannote degrades gracefully without torchcodec)
    - Model weights staged at installer/models/diarization/community-1/
      (source: build-assets/models/diarization/community-1/ — build-assets
      lives INSIDE the repo root, gitignored):
          mkdir -p installer/models/diarization
          cp -R build-assets/models/diarization/community-1 installer/models/diarization/
      They land in the .app under Contents/Frameworks/models/... (sys._MEIPASS).
"""

import importlib.util
import os as _os

from PyInstaller.utils.hooks import (collect_all, collect_data_files,
                                     collect_submodules, copy_metadata)

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

# --- Speaker diarization stack (pyannote.audio 4.x) -----------------------
# Guarded per package so a freeze venv without the diarization deps still
# builds a working (diarization-less) app. torch itself is NOT collect_all'd:
# the official PyInstaller torch hook handles its libraries correctly.
_diar_datas, _diar_binaries, _diar_hiddens = [], [], []
for _pkg in (
    "pyannote.audio", "pyannote.core", "pyannote.database",
    "pyannote.metrics", "pyannote.pipeline",
    "asteroid_filterbanks", "torchaudio", "lightning", "torchmetrics",
    "pytorch_metric_learning", "torch_audiomentations", "julius",
    "omegaconf", "safetensors",
):
    try:
        _d, _b, _h = collect_all(_pkg)
        _diar_datas += _d
        _diar_binaries += _b
        _diar_hiddens += _h
    except Exception:
        pass

for _pkg in ("torch", "torchaudio", "lightning", "lightning_utilities",
             "torchmetrics", "pyannote.audio"):
    try:
        _diar_datas += copy_metadata(_pkg)
    except Exception:
        pass

# scipy >= 1.18 vendors array_api_compat under scipy._external; the stock
# scipy hook misses its dynamically-imported submodules (numpy.fft et al.),
# which torchmetrics trips over via `import scipy.signal` at pyannote import
# time (ModuleNotFoundError caught by the v2.1.0 Windows smoke freeze).
try:
    _diar_hiddens += collect_submodules("scipy._external")
except Exception:
    pass

# --- Live recording stack (optional) ---------------------------------------
# silero-vad loads its bundled VAD model through importlib.resources on
# silero_vad.data — a dynamic reference PyInstaller can't trace and no stock
# hook covers, so a frozen app raises "No module named 'silero_vad.data'"
# the moment recording starts. Guarded: a venv without the recording extra
# still builds a working (recording-less) app.
try:
    _d, _b, _h = collect_all("silero_vad")
    _diar_datas += _d
    _diar_binaries += _b
    _diar_hiddens += _h
except Exception:
    pass

# Bundled ffmpeg/ffprobe (staged in installer/ffmpeg-mac/ — see the
# Prerequisites in this spec's docstring).
# REQUIRED for diarization builds: diarize.load_waveform and duration probing
# shell out to _bundled_bin("ffmpeg"/"ffprobe"), and a Finder-launched .app
# gets launchd's bare PATH — even a Homebrew ffmpeg is invisible. Without
# these, speaker identification silently degrades to no labels for every
# input that isn't already a 16 kHz 16-bit WAV. BINARY entries (not datas)
# so PyInstaller ad-hoc-signs them on arm64; dest "ffmpeg" lands at
# Contents/Frameworks/ffmpeg with the Resources cross-symlink _bundled_bin
# already searches.
_ffm = _os.path.join(SPECPATH, "installer", "ffmpeg-mac")
for _tool in ("ffmpeg", "ffprobe"):
    _p = _os.path.join(_ffm, _tool)
    if _os.path.isfile(_p):
        _diar_binaries += [(_p, "ffmpeg")]
# GPL license text accompanying the (L)GPL ffmpeg binaries (data, not binary).
_ffm_license = _os.path.join(_ffm, "LICENSE.txt")
if _os.path.isfile(_ffm_license):
    _diar_datas += [(_ffm_license, "ffmpeg")]

# Bundled diarization model weights (staged per the docstring above).
_weights_src = _os.path.join(SPECPATH, "installer", "models",
                             "diarization", "community-1")
if _os.path.isdir(_weights_src):
    _diar_datas += [(_weights_src, "models/diarization/community-1")]

all_datas = (
    faster_datas + ct2_datas + tkdnd_datas + onnx_datas + _diar_datas
    + [("LICENSE", "."), ("THIRD-PARTY-LICENSES.md", "."), ("tests/fixtures/speech_6s.wav", "."), ("icon.ico", ".")]
    + [("hebrewscribe/icons/*.png", "hebrewscribe/icons"), ("hebrewscribe/icon-64.png", "hebrewscribe")]
)
all_binaries = (faster_binaries + ct2_binaries + tkdnd_binaries
                + onnx_binaries + _diar_binaries)
all_hiddenimports = (
    faster_hiddens + ct2_hiddens + tkdnd_hiddens + onnx_hiddens + _diar_hiddens
    + ["hebrewscribe", "hebrewscribe.app", "hebrewscribe.utils",
       "hebrewscribe.models", "hebrewscribe.worker", "hebrewscribe.outputs",
       "hebrewscribe.power", "hebrewscribe.theme", "hebrewscribe.widgets",
       "hebrewscribe.icons", "hebrewscribe.diarize"]
    + ["pyannote.audio.pipelines.speaker_diarization",
       "pyannote.audio.models.segmentation",
       "pyannote.audio.models.embedding",
       "einops", "scipy.signal", "scipy.cluster.hierarchy", "rich"]
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
    # torchcodec: deliberately excluded (see docstring). matplotlib: pulled by
    # pyannote.core/metrics for notebook plotting only — if the frozen import
    # chain trips on it, drop it from this list and re-freeze.
    excludes=["torchcodec", "matplotlib", "IPython", "jupyter",
              "tensorboard", "triton"],
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
        "NSMicrophoneUsageDescription": "HebrewScribe uses the microphone for live speech-to-text transcription.",
    },
)
