# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for HebrewScribe.

Usage:
    pyinstaller HebrewScribe.spec

Produces:  dist/HebrewScribe/HebrewScribe.exe  (one-dir bundle)

Freeze-venv prerequisites for the diarization feature (unified installer):
  - Python >= 3.10 (pyannote.audio 4.x floor)
  - CPU-only torch wheels (a CUDA wheel balloons the bundle by GBs):
        pip install torch>=2.8 torchaudio>=2.8 --index-url https://download.pytorch.org/whl/cpu
        pip install "pyannote.audio>=4,<5"
        pip uninstall torchcodec -y
    torchcodec is excluded deliberately: the app feeds pyannote in-memory
    waveform dicts, so torchcodec (and its Windows FFmpeg-DLL matrix) is
    never needed; pyannote 4.0.7 degrades gracefully when it is absent.
  - PyInstaller >= 6.10 with pyinstaller-hooks-contrib >= 2025.8 (torch hook).
Model weights are NOT collected here — the Inno installer ships them beside
the exe (installer/HebrewScribeSetup.iss), mirroring the ffmpeg pattern.
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
tkdnd_datas, tkdnd_binaries, tkdnd_hiddens = collect_all("tkinterdnd2")
onnx_datas, onnx_binaries, onnx_hiddens = collect_all("onnxruntime")

# --- Speaker diarization stack (pyannote.audio 4.x) -----------------------
# Guarded per package so a freeze venv without the diarization deps still
# builds a working (diarization-less) app. torch itself is NOT collect_all'd:
# the official PyInstaller torch hook handles its DLL forest correctly.
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

# lightning/pyannote probe package versions via importlib.metadata at import
# time; missing dist-info is a classic frozen-app crash.
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

# UPX over torch's DLLs is slow, risks corruption, and gains nothing (Inno
# recompresses with lzma2/ultra64).
_UPX_EXCLUDE = [
    "torch_cpu.dll", "torch_python.dll", "torch_global_deps.dll",
    "c10.dll", "fbgemm.dll", "libiomp5md.dll", "dnnl.dll", "asmjit.dll",
    "shm.dll", "uv.dll",
]

a = Analysis(
    ["HebrewScribe.pyw"],
    pathex=[],
    binaries=all_binaries,
    datas=all_datas,
    hiddenimports=all_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # torchcodec: deliberately excluded (see module docstring). matplotlib:
    # pulled by pyannote.core/metrics for notebook plotting only — if the
    # frozen import chain trips on it, drop it from this list and re-freeze.
    excludes=["torchcodec", "matplotlib", "IPython", "jupyter",
              "tensorboard", "triton"],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# Drop sounddevice's ASIO-enabled PortAudio DLL: the app never requests ASIO
# (WASAPI/WDM-KS/DirectSound suffice), and the ASIO SDK inside it carries
# Steinberg's proprietary license, not an OSI one.
_no_asio = lambda entries: [e for e in entries
                            if not ("portaudio" in e[0].lower() and "asio" in e[0].lower())]
a.datas = _no_asio(a.datas)
a.binaries = _no_asio(a.binaries)

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
    upx_exclude=_UPX_EXCLUDE,
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
    upx_exclude=_UPX_EXCLUDE,
    name="HebrewScribe",
)
