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

### sounddevice
- License: MIT
- Copyright: (c) 2015-2023 Matthias Geier
- URL: https://github.com/spatialaudio/python-sounddevice
- Note: used for microphone capture in the experimental live-recording
  feature. Ships with bundled PortAudio binaries (see below).

### PortAudio
- License: PortAudio license (MIT-style)
- Copyright: (c) 1999-2011 Ross Bencina and Phil Burk
- URL: https://www.portaudio.com
- Note: bundled via the sounddevice package.

### Silero VAD
- License: MIT
- Copyright: (c) Silero Team
- URL: https://github.com/snakers4/silero-vad
- Note: voice-activity-detection code and bundled model weights, used by the
  experimental live-recording feature.

### FFmpeg
- License: GPL-3.0 (effective license of this build — compiled with
  --enable-gpl --enable-version3)
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
  A full copy of the GPL v3 license text is installed alongside the binaries
  (ffmpeg\LICENSE.txt in the Windows install directory).

### pyannote.audio
- License: MIT
- Copyright: (c) 2017- CNRS / Hervé Bredin and contributors
- URL: https://github.com/pyannote/pyannote-audio

### PyTorch
- License: BSD-3-Clause
- Copyright: (c) 2016- PyTorch contributors; Meta Platforms, Inc.
- URL: https://github.com/pytorch/pytorch

### torchaudio
- License: BSD-2-Clause
- Copyright: (c) 2017- PyTorch contributors
- URL: https://github.com/pytorch/audio

### Lightning
- License: Apache-2.0
- Copyright: (c) Lightning AI
- URL: https://github.com/Lightning-AI/pytorch-lightning

### pyannote speaker-diarization-community-1 (model weights)
- License: CC-BY-4.0
- Copyright: (c) pyannoteAI and contributors
- URL: https://huggingface.co/pyannote/speaker-diarization-community-1
- Note: Bundled in the Windows installer and macOS app; downloaded at first
  use for source installs. Redistributed with attribution per CC-BY-4.0.
  The pipeline builds on the following work:
  - A. Plaquet and H. Bredin, "Powerset multi-class cross entropy loss for
    neural speaker diarization," Interspeech 2023 (segmentation model).
  - H. Wang et al., "WeSpeaker: A research and production oriented speaker
    embedding learning toolkit," ICASSP 2023 (speaker embedding model).
  - F. Landini et al., "Bayesian HMM clustering of x-vector sequences (VBx)
    in speaker diarization," Computer Speech & Language, 2022 (VBx/PLDA;
    PLDA courtesy of BUT Speech@FIT, Brno University of Technology).

### torchmetrics
- License: Apache-2.0
- Copyright: (c) Lightning AI
- URL: https://github.com/Lightning-AI/torchmetrics

### omegaconf
- License: BSD-3-Clause
- Copyright: (c) 2018 Omry Yadan
- URL: https://github.com/omry/omegaconf

### safetensors
- License: Apache-2.0
- Copyright: (c) Hugging Face, Inc.
- URL: https://github.com/huggingface/safetensors

### einops
- License: MIT
- Copyright: (c) 2018 Alex Rogozhnikov
- URL: https://github.com/arogozhnikov/einops

### asteroid-filterbanks
- License: MIT
- Copyright: (c) Asteroid team
- URL: https://github.com/asteroid-team/asteroid-filterbanks

### pytorch-metric-learning
- License: MIT
- Copyright: (c) Kevin Musgrave
- URL: https://github.com/KevinMusgrave/pytorch-metric-learning

### torch-audiomentations
- License: MIT
- Copyright: (c) Asteroid team
- URL: https://github.com/asteroid-team/torch-audiomentations

### julius
- License: MIT
- Copyright: (c) 2020 Alexandre Défossez
- URL: https://github.com/adefossez/julius

### Rich
- License: MIT
- Copyright: (c) 2020 Will McGugan
- URL: https://github.com/Textualize/rich

### SciPy
- License: BSD-3-Clause
- Copyright: (c) 2001-2026 SciPy Developers
- URL: https://scipy.org

### Python
- License: PSF-2.0
- Copyright: (c) 2001-2026 Python Software Foundation
- URL: https://docs.python.org/3/license.html

### Tcl/Tk
- License: Tcl/Tk license (BSD-style)
- Copyright: (c) Regents of the University of California, Sun Microsystems,
  Scriptics Corporation, ActiveState Corporation and other parties
- URL: https://www.tcl.tk/software/tcltk/license.html

### OpenSSL
- License: Apache-2.0 (OpenSSL 3.x)
- Copyright: (c) The OpenSSL Project Authors
- URL: https://www.openssl.org

## Additional Bundled Python Dependencies

Transitive dependencies also present in the frozen Windows bundle. This table is
**generated** by `scripts/gen_third_party_licenses.py` from the licence metadata of
the packages that actually ship — regenerate it whenever dependencies change rather
than editing it by hand.

| Package | License |
|---|---|
| aiohappyeyeballs | PSF-2.0 |
| aiohttp | Apache-2.0 AND MIT |
| aiosignal | Apache 2.0 |
| alembic | MIT |
| annotated-doc | MIT |
| anyio | MIT |
| attrs | MIT |
| autocommand | LGPLv3 |
| av | BSD-3-Clause |
| backports.tarfile | MIT License |
| cffi | MIT |
| charset-normalizer | MIT |
| click | BSD-3-Clause |
| colorlog | MIT License |
| flatbuffers | Apache 2.0 |
| frozenlist | Apache-2.0 |
| googleapis-common-protos | Apache 2.0 |
| greenlet | MIT AND PSF-2.0 |
| grpcio | Apache-2.0 |
| h11 | MIT |
| hf-xet | Apache-2.0 |
| httpcore | BSD-3-Clause |
| httpx | BSD-3-Clause |
| idna | BSD-3-Clause |
| importlib_metadata | Apache-2.0 |
| jaraco.context | MIT |
| jaraco.functools | MIT |
| jaraco.text | MIT License |
| Jinja2 | BSD License |
| joblib | BSD-3-Clause |
| lightning-utilities | Apache-2.0 |
| Mako | MIT |
| markdown-it-py | MIT License |
| MarkupSafe | BSD-3-Clause |
| mdurl | MIT License |
| more-itertools | MIT |
| mpmath | BSD |
| multidict | Apache License 2.0 |
| narwhals | MIT |
| networkx | BSD-3-Clause |
| opentelemetry-api | Apache-2.0 |
| opentelemetry-exporter-otlp | Apache-2.0 |
| opentelemetry-exporter-otlp-proto-common | Apache-2.0 |
| opentelemetry-exporter-otlp-proto-grpc | Apache-2.0 |
| opentelemetry-exporter-otlp-proto-http | Apache-2.0 |
| opentelemetry-proto | Apache-2.0 |
| opentelemetry-sdk | Apache-2.0 |
| opentelemetry-semantic-conventions | Apache-2.0 |
| optuna | MIT License |
| packaging | Apache-2.0 OR BSD-2-Clause |
| pandas | BSD 3-Clause License |
| pillow | MIT-CMU |
| platformdirs | MIT |
| primePy | MIT License |
| propcache | Apache-2.0 |
| protobuf | 3-Clause BSD License |
| pyannote-core | The MIT License (MIT) |
| pyannote-database | The MIT License (MIT) |
| pyannote-metrics | The MIT License (MIT) |
| pyannote-pipeline | The MIT License (MIT) |
| pyannoteai-sdk | MIT License |
| pycparser | BSD-3-Clause |
| Pygments | BSD-2-Clause |
| pyparsing | MIT |
| python-dateutil | Dual License |
| pytorch-lightning | Apache-2.0 |
| requests | Apache-2.0 |
| scikit-learn | BSD-3-Clause |
| setuptools | MIT |
| shellingham | ISC License |
| six | MIT |
| sortedcontainers | Apache 2.0 |
| SQLAlchemy | MIT |
| sympy | BSD |
| threadpoolctl | BSD-3-Clause |
| tomli | MIT |
| torch | BSD-3-Clause |
| torch_pitch_shift | MIT License |
| tqdm | MPL-2.0 AND MIT |
| typer | MIT |
| typing_extensions | PSF-2.0 |
| tzdata | Apache-2.0 |
| urllib3 | MIT |
| yarl | Apache-2.0 |
| zipp | MIT |

Packages excluded from the bundle and therefore from this table: `torchcodec`,
`matplotlib` (and its dependency cone), `IPython`, `jupyter`, `tensorboard`, `triton`
(see `excludes` in `HebrewScribe.spec`), and the ASIO-enabled PortAudio DLL, which is
dropped because the ASIO SDK inside it carries Steinberg's proprietary licence.

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
BSD-2-Clause: https://opensource.org/license/bsd-2-clause
BSD-3-Clause: https://opensource.org/licenses/BSD-3-Clause
Apache License 2.0: https://www.apache.org/licenses/LICENSE-2.0
Creative Commons Attribution 4.0: https://creativecommons.org/licenses/by/4.0/
Mozilla Public License 2.0: https://www.mozilla.org/en-US/MPL/2.0/
GNU General Public License 3.0: https://www.gnu.org/licenses/gpl-3.0.html
GNU General Public License 2.0: https://www.gnu.org/licenses/old-licenses/gpl-2.0.html
Python Software Foundation License: https://docs.python.org/3/license.html

The Windows installer additionally ships a full copy of the GPL v3 text next
to the FFmpeg binaries it covers (ffmpeg\LICENSE.txt).
