# HebrewScribe

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20macOS-lightgrey.svg)](#quick-start)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![Microsoft Store](https://img.shields.io/badge/Microsoft%20Store-HebrewScribe-0078D4)](https://apps.microsoft.com/detail/9PBS32VWZPBB)

A free, offline desktop application for batch transcription of Hebrew audio using local Whisper models.

HebrewScribe turns Hebrew audio files into text on your own computer, with no cloud services, no accounts, and no data leaving your machine. Queue a handful of recordings or an entire folder, and let it run. Powered by [faster-whisper](https://github.com/SYSTRAN/faster-whisper) and Whisper models fine-tuned for Hebrew by [ivrit-ai](https://huggingface.co/ivrit-ai).

## Why HebrewScribe?

- **Hebrew-first:** For Hebrew it selects [ivrit-ai large-v3](https://huggingface.co/ivrit-ai/whisper-large-v3-ct2) — ivrit-ai's Hebrew fine-tune of Whisper large-v3 — and downloads it on first use. English and other languages are also selectable, each with its own model
- **Private and open source:** All processing happens locally. No cloud, no telemetry, no accounts. The only network call is downloading a model the first time. [Source code on GitHub](https://github.com/yevgeniyglider/HebrewScribe) for full transparency
- **Batch-oriented:** Queue files and folders, reorder the queue mid-run, get per-file and batch ETAs with real-time progress
- **Free:** No subscriptions, no usage limits, no expiry

The process is entirely local:

- 14 audio formats supported (MP3, WAV, M4A, FLAC, OGG, and more)
- Voice Activity Detection filters silence for faster processing
- GPU acceleration when available (CUDA on Windows)
- Sleep prevention keeps your machine awake during long batches
- Live recording (experimental): dictate into the microphone and watch the transcript appear

![HebrewScribe main window](screenshot.png)

## Quick Start

**Windows** — install from the Microsoft Store (recommended: always the current version, updates automatically, no security prompts):

<a href="https://apps.microsoft.com/detail/9PBS32VWZPBB"><img src="https://get.microsoft.com/images/en-us%20dark.svg" width="200" alt="Get HebrewScribe from the Microsoft Store"/></a>

**macOS** — install from source (below).

Other options, including a direct Windows installer for machines where the Store is unavailable, are listed at [yevgeniyglider.com/hebrewscribe](https://yevgeniyglider.com/hebrewscribe/).

## Installation from Source

Requires Python 3.10+ and [FFmpeg](https://ffmpeg.org).

```bash
pip install -e ".[faster]"
```

For speaker diarization ("Identify speakers"), add the `diarization` extra —
it pulls PyTorch, so prefer the CPU wheel index unless you have a CUDA GPU:

```bash
pip install -e ".[faster,diarization]" --extra-index-url https://download.pytorch.org/whl/cpu
```

The speaker model (~33 MB) downloads automatically on first use and runs
fully offline afterwards. On macOS this requires Apple Silicon.

For experimental live recording (dictation from the microphone), add the
`recording` extra:

```bash
pip install -e ".[faster,recording]"
```

Run with `python -m hebrewscribe`, or double-click `HebrewScribe.pyw` (Windows) / `HebrewScribe.command` (macOS).

## Architecture

HebrewScribe is a Python desktop application using tkinter for the GUI:

- **GUI:** tkinter with a custom flat theme, DPI-aware on Windows
- **Transcription:** [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2-based)
- **Speaker diarization (optional):** [pyannote.audio](https://github.com/pyannote/pyannote-audio) community-1 pipeline, fully offline with self-hosted weights
- **Audio:** FFmpeg for format conversion
- **Models:** Curated selection from [ivrit-ai](https://huggingface.co/ivrit-ai) (Hebrew) and [Systran](https://huggingface.co/Systran) (multilingual), downloaded from Hugging Face Hub

The codebase is a single Python package (`hebrewscribe/`) with 10 focused modules: app, worker, recorder, diarize, models, outputs, power, theme, utils, and widgets. 360+ tests cover the core logic.

## Project Map

```
HebrewScribe/
  LICENSE                          # MIT license text
  README.md                        # Project readme
  CHANGELOG.md                     # Release notes
  THIRD-PARTY-LICENSES.md         # Bundled dependency licenses
  pyproject.toml                   # Package metadata
  version_info.py                  # Windows EXE version metadata
  .editorconfig                    # Editor formatting defaults
  HebrewScribe.pyw                 # Windows launcher
  HebrewScribe.command             # macOS launcher
  HebrewScribe.spec                # PyInstaller Windows spec
  HebrewScribe-mac.spec            # PyInstaller macOS spec
  scripts/install-hooks.sh         # Git hook installer
  scripts/render_icons.py          # SVG-to-PNG build script
  icon.ico / icon.icns             # App icons
  screenshot.png                   # README screenshot
  hebrewscribe/                    # Source package (10 modules + icons)
  tests/                           # 360+ tests
  assets/                          # SVG source icons
```

## Acknowledgments

- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) by SYSTRAN for fast CTranslate2-based Whisper inference
- [Whisper](https://github.com/openai/whisper) by OpenAI for the speech recognition models
- [ivrit-ai](https://huggingface.co/ivrit-ai) for Hebrew-tuned Whisper models
- [FFmpeg](https://ffmpeg.org) for audio format handling

## License

MIT License. See [LICENSE](LICENSE) for the full text, and [THIRD-PARTY-LICENSES.md](THIRD-PARTY-LICENSES.md) for bundled dependency licenses.

---

[yevgeniyglider.com/hebrewscribe](https://yevgeniyglider.com/hebrewscribe/) · Built by [Yevgeniy Glider](https://yevgeniyglider.com)