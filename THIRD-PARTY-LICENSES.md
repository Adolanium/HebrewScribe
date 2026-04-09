# Third-Party Licenses

HebrewScribe bundles the following open-source software in the Windows
installer. Their inclusion does not imply endorsement by their respective
authors.

## Redistributed Components

### CTranslate2
- License: MIT
- Copyright: (c) OpenNMT
- URL: https://github.com/OpenNMT/CTranslate2

### faster-whisper
- License: MIT
- Copyright: (c) 2023 SYSTRAN
- URL: https://github.com/SYSTRAN/faster-whisper

### ONNX Runtime
- License: MIT
- Copyright: (c) Microsoft Corporation
- URL: https://github.com/microsoft/onnxruntime

### OpenAI Whisper
- License: MIT
- Copyright: (c) 2022 OpenAI
- URL: https://github.com/openai/whisper

### PyYAML
- License: MIT
- Copyright: (c) 2017-2021 Ingy dot Net; 2006-2016 Kirill Simonov
- URL: https://pyyaml.org

### tkinterdnd2
- License: MIT
- Copyright: (c) 2020 Philippe Gagne / Eliav2
- URL: https://github.com/Eliav2/tkinterdnd2

### filelock
- License: MIT
- Copyright: (c) Bernat Gabor and contributors
- URL: https://github.com/tox-dev/filelock

### certifi
- License: MPL-2.0
- Copyright: (c) Kenneth Reitz and contributors
- URL: https://github.com/certifi/python-certifi

### colorama
- License: BSD-3-Clause
- Copyright: (c) 2010 Jonathan Hartley
- URL: https://github.com/tartley/colorama

### NumPy
- License: BSD-3-Clause
- Copyright: (c) 2005-2026 NumPy Developers
- URL: https://numpy.org

### PyAV
- License: BSD-3-Clause
- Copyright: (c) PyAV contributors
- URL: https://github.com/PyAV-Org/PyAV

### Protocol Buffers
- License: BSD-3-Clause
- Copyright: (c) Google LLC
- URL: https://github.com/protocolbuffers/protobuf

### fsspec
- License: BSD-3-Clause
- Copyright: (c) 2018 Martin Durant
- URL: https://github.com/fsspec/filesystem_spec

### Hugging Face Hub
- License: Apache-2.0
- Copyright: (c) Hugging Face, Inc.
- URL: https://github.com/huggingface/huggingface_hub

### Tokenizers
- License: Apache-2.0
- Copyright: (c) Hugging Face, Inc.
- URL: https://github.com/huggingface/tokenizers

### Bootstrap Icons
- License: MIT
- Copyright: (c) 2019-2024 The Bootstrap Authors
- URL: https://github.com/twbs/icons
- Note: 10 toolbar icons used (file-earmark-plus, folder-plus, dash-circle,
  trash, chevron-bar-up, chevron-up, chevron-down, chevron-bar-down,
  folder2-open, gear). Source SVGs adapted and rendered to PNG at build time.

### FFmpeg
- License: GPL-2.0-or-later (see note below)
- Copyright: (c) FFmpeg developers
- URL: https://ffmpeg.org
- Bundled version: N-123522-gac4d50cb26-20260317 (git commit ac4d50cb26)
- Bundled build source: https://github.com/BtbN/FFmpeg-Builds (win64-gpl build, 2026-03-17)
- Source code: https://github.com/BtbN/FFmpeg-Builds and https://ffmpeg.org/download.html
- Note: FFmpeg is invoked as a separate executable, not linked into the
  application. The currently bundled build is compiled with --enable-gpl
  --enable-version3 (including libx264 and libx265). A future release will
  replace this with an LGPL-only build, since HebrewScribe only requires audio
  decoding and probing. Users who need the corresponding source code for the
  bundled FFmpeg build can obtain it from the BtbN builds repository above.

### Python
- License: PSF-2.0
- Copyright: (c) 2001-2026 Python Software Foundation
- URL: https://docs.python.org/3/license.html

## Components Downloaded at Runtime

The following components are downloaded by the user at runtime from Hugging
Face Hub and are not distributed with HebrewScribe.

### ivrit-ai/whisper-large-v3-ct2
- License: Apache-2.0
- Copyright: (c) ivrit.ai
- URL: https://huggingface.co/ivrit-ai/whisper-large-v3-ct2

### Systran/faster-distil-whisper-large-v3
- License: MIT
- Copyright: (c) SYSTRAN
- URL: https://huggingface.co/Systran/faster-distil-whisper-large-v3

### Systran/faster-whisper-large-v3
- License: MIT
- Copyright: (c) SYSTRAN
- URL: https://huggingface.co/Systran/faster-whisper-large-v3

### Distil-Whisper (model architecture)
- License: MIT
- Copyright: (c) 2023 Hugging Face
- URL: https://github.com/huggingface/distil-whisper

## License Texts

MIT License: https://opensource.org/licenses/MIT
BSD-3-Clause: https://opensource.org/licenses/BSD-3-Clause
Apache License 2.0: https://www.apache.org/licenses/LICENSE-2.0
Mozilla Public License 2.0: https://www.mozilla.org/en-US/MPL/2.0/
GNU General Public License 2.0: https://www.gnu.org/licenses/old-licenses/gpl-2.0.html
Python Software Foundation License: https://docs.python.org/3/license.html
