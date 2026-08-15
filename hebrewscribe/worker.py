"""Transcription worker: model loading, transcription backends, output writing.

This module contains all transcription logic, decoupled from the GUI.
Communication with the GUI happens through the WorkerHost protocol.
"""

import atexit
import logging
import os
import shutil
import subprocess
import tempfile
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Protocol, runtime_checkable

from hebrewscribe.utils import (
    _bundled_bin, _selftest_audio_path, ffmpeg_available,
    probe_duration_seconds, safe_stem, unique_stem,
)
from hebrewscribe.outputs import (format_speaker_transcript, write_txt,
                                  write_srt, write_json)

logger = logging.getLogger("HebrewScribe")


# ---------------------------------------------------------------------------
# Shared types
# ---------------------------------------------------------------------------

class CancelledByUser(Exception):
    pass


@dataclass
class RunOptions:
    backend: str
    model_spec: str
    output_dir: str
    language: str
    task: str
    device: str
    compute_type: str
    beam_size: int
    vad_filter: bool
    condition_on_previous_text: bool
    batch_size: int
    formats: List[str]
    # Diarization fields default off: RunOptions is also constructed by
    # app.get_run_options and run_selftest — defaults keep old call sites valid.
    diarize: bool = False
    num_speakers: int = 0        # 0 = auto-detect
    diarize_device: str = "auto"  # "auto" | "cpu" (macOS device preference)


# ---------------------------------------------------------------------------
# WorkerHost protocol — what the worker needs from the GUI (or a test mock)
# ---------------------------------------------------------------------------

_VALID_BACKENDS = {"openai-whisper", "faster-whisper"}


def _validate_opts(opts: RunOptions) -> None:
    """Validate RunOptions fields early, before any heavy work."""
    if opts.backend not in _VALID_BACKENDS:
        raise ValueError(f"Unknown backend '{opts.backend}'. "
                         f"Expected one of: {', '.join(sorted(_VALID_BACKENDS))}")
    if not opts.model_spec:
        raise ValueError("model_spec must not be empty")
    if opts.beam_size < 1:
        raise ValueError(f"beam_size must be >= 1, got {opts.beam_size}")
    if not opts.formats:
        raise ValueError("formats must contain at least one output format")
    if opts.num_speakers < 0:
        raise ValueError(f"num_speakers must be >= 0, got {opts.num_speakers}")
    out_dir = Path(opts.output_dir)
    if not out_dir.is_dir():
        raise ValueError(f"output_dir does not exist: {opts.output_dir}")


@runtime_checkable
class WorkerHost(Protocol):
    cancel_requested: bool
    stop_requested: bool
    pause_event: threading.Event

    def post_event(self, kind: str, **payload) -> None: ...
    def next_file(self) -> Optional[str]: ...


def check_for_cancel(host: WorkerHost) -> None:
    if host.cancel_requested:
        raise CancelledByUser("Cancelled by user.")


def wait_if_paused(host: WorkerHost) -> None:
    """Block until the host's pause_event is set. Check for cancel while waiting."""
    while not host.pause_event.is_set():
        check_for_cancel(host)
        host.post_event("paused")
        time.sleep(0.15)
    # Cancel can land between the last in-loop check and the resume; without
    # this, "Cancel" pressed while paused let the worker start (and
    # preconvert) the next file before taking effect.
    check_for_cancel(host)
    host.post_event("resumed")


# ---------------------------------------------------------------------------
# Audio pre-conversion
# ---------------------------------------------------------------------------

_PRECONVERT_DIR: Optional[str] = None
_preconvert_lock = threading.Lock()


def _get_preconvert_dir() -> str:
    global _PRECONVERT_DIR
    with _preconvert_lock:
        if _PRECONVERT_DIR is None or not Path(_PRECONVERT_DIR).exists():
            _PRECONVERT_DIR = tempfile.mkdtemp(prefix="whisper_preconvert_")
        return _PRECONVERT_DIR


def validate_audio_file(path: Path) -> None:
    """Check that a file exists, is non-empty, and looks like a valid audio file.

    Raises ValueError with a human-readable message on failure.
    """
    if not path.exists():
        raise ValueError(f"File not found: {path}")
    if not path.is_file():
        raise ValueError(f"Not a file: {path}")
    size = path.stat().st_size
    if size == 0:
        raise ValueError(f"File is empty (0 bytes): {path.name}")
    # Minimum plausible audio size: even a 1-second mono 8kHz WAV is ~16 KB.
    # Files under 128 bytes are almost certainly corrupt headers or placeholders.
    if size < 128:
        raise ValueError(
            f"File too small to be valid audio ({size} bytes): {path.name}")
    # If ffprobe is available, verify the file is actually decodable
    if ffmpeg_available():
        try:
            completed = subprocess.run(
                [
                    _bundled_bin("ffprobe"),
                    "-v", "error",
                    "-select_streams", "a:0",
                    "-show_entries", "stream=codec_type",
                    "-of", "csv=p=0",
                    str(path),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=15,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            if completed.returncode != 0:
                stderr = completed.stderr.strip()
                raise ValueError(
                    f"Cannot read audio from {path.name}: "
                    f"{stderr[:200] if stderr else 'ffprobe returned an error'}")
            if "audio" not in (completed.stdout or ""):
                raise ValueError(
                    f"No audio stream found in {path.name}. "
                    f"The file may be corrupt or not an audio file.")
        except subprocess.TimeoutExpired:
            logger.warning("ffprobe timed out validating %s, proceeding anyway", path.name)
        except ValueError:
            raise  # re-raise our own ValueErrors
        except Exception:
            logger.debug("ffprobe validation failed for %s, proceeding anyway",
                         path.name, exc_info=True)


def preconvert_audio(source: Path) -> Path:
    """Convert audio to 16kHz mono WAV for faster Whisper ingestion."""
    if source.suffix.lower() == ".wav":
        return source
    if not ffmpeg_available():
        return source
    out_dir = _get_preconvert_dir()
    out_path = Path(out_dir) / f"{safe_stem(source)}_{hash(str(source)) & 0xFFFFFFFF:08x}.wav"
    if out_path.exists():
        return out_path
    try:
        completed = subprocess.run(
            [
                _bundled_bin("ffmpeg"), "-y", "-i", str(source),
                "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
                str(out_path),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=120,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if completed.returncode == 0 and out_path.exists():
            logger.debug("Pre-converted %s -> %s", source.name, out_path.name)
            return out_path
    except Exception:
        logger.debug("Audio pre-conversion failed for %s", source, exc_info=True)
    return source


def _cleanup_preconvert_dir() -> None:
    if _PRECONVERT_DIR and Path(_PRECONVERT_DIR).exists():
        try:
            shutil.rmtree(_PRECONVERT_DIR, ignore_errors=True)
        except Exception:
            pass


atexit.register(_cleanup_preconvert_dir)


# ---------------------------------------------------------------------------
# GPU helpers
# ---------------------------------------------------------------------------

def cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        logger.debug("CUDA availability check failed", exc_info=True)
        return False


def mps_available() -> bool:
    """Check if Apple Silicon MPS backend is available (macOS only)."""
    try:
        import torch
        return bool(torch.backends.mps.is_available())
    except Exception:
        return False


def _auto_batch_size(device_str: str) -> int:
    """Pick a batch_size based on available device memory."""
    actual_device = device_str
    if actual_device == "auto":
        try:
            import torch
            actual_device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            actual_device = "cpu"

    if actual_device == "cpu":
        return 1

    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            total_gb = props.total_mem / (1024 ** 3)
            free_mem = torch.cuda.mem_get_info(0)[0]
            free_gb = free_mem / (1024 ** 3)
            if free_gb >= 6:
                return 16
            elif free_gb >= 3:
                return 8
            elif free_gb >= 1.5:
                return 4
            else:
                return 1
    except Exception:
        logger.debug("GPU memory detection failed, batch_size=1", exc_info=True)

    return 1


# ---------------------------------------------------------------------------
# Model corruption detection & recovery
# ---------------------------------------------------------------------------

def _is_ct2_corruption_error(exc: Exception) -> bool:
    """Return True if *exc* looks like a CTranslate2 model-data corruption error.

    Two families are matched:

    1. **Inference-time JSON errors** — CTranslate2 uses the nlohmann/json C++
       library internally.  When a model file is truncated or corrupt, the C++
       layer raises a RuntimeError whose message contains the
       ``json.exception.type_error`` family.

    2. **Load-time "Unable to open file" errors** — raised by
       ``ctranslate2.models.Whisper.__init__`` when ``model.bin`` is missing or
       has a binary format incompatible with the installed ctranslate2 version.
       This happens after a ctranslate2 major-version upgrade (e.g. building the
       installer against Python 3.14 bundles a newer ctranslate2 that cannot
       read a model.bin written by an older version).  Re-downloading forces
       huggingface_hub to fetch files compatible with the current library.

    3. **Load-time "is incomplete" errors** — raised when ``model.bin`` exists
       but is shorter than its header claims (download killed, disk full,
       antivirus truncation).  CTranslate2 reports the byte offset it could not
       reach rather than a JSON or open failure, so this needs its own match.
    """
    if not isinstance(exc, RuntimeError):
        return False
    msg = str(exc)
    return (
        "json.exception.type_error" in msg
        or "Unable to open file" in msg
        or "is incomplete" in msg
    )


# Minimum size for a usable CT2 Whisper weight file (tiny is ~75 MB).
_MIN_MODEL_BIN_BYTES = 10 * 1024 * 1024

# Files required in a CTranslate2 Whisper model directory.
_CT2_REQUIRED_FILES = (
    "model.bin",
    "config.json",
    "tokenizer.json",
)


def _model_bin_status(model_dir: Path) -> tuple[bool, str]:
    """Check whether *model_dir* looks like a readable CT2 Whisper model.

    Returns ``(ok, detail)``.  ``ok`` is True only when required files exist,
    ``model.bin`` is readable, and its size is plausible.
    """
    if not model_dir.is_dir():
        return False, f"not a directory: {model_dir}"

    missing = [name for name in _CT2_REQUIRED_FILES
               if not (model_dir / name).exists()]
    if missing:
        return False, f"missing files: {', '.join(missing)}"

    model_bin = model_dir / "model.bin"
    try:
        size = model_bin.stat().st_size
    except OSError as exc:
        return False, f"cannot stat model.bin: {exc}"

    if size < _MIN_MODEL_BIN_BYTES:
        return False, f"model.bin too small ({size} bytes) — download incomplete?"

    try:
        with open(model_bin, "rb") as fh:
            header = fh.read(16)
        if not header:
            return False, "model.bin is empty / unreadable"
    except OSError as exc:
        return False, f"cannot read model.bin: {exc}"

    return True, f"model.bin ok ({size} bytes)"


def _discard_short_model_bin(model_dir: Path) -> bool:
    """Delete ``model.bin`` in *model_dir* if it is present but too short.

    huggingface_hub returns early for any file that already exists on disk —
    it compares presence, not length — so a truncated weight file survives a
    re-download and the caller is told the fetch succeeded.  Removing it first
    makes the retry fetch the file for real.

    Only a *short* file is removed.  A full-size file that merely cannot be
    opened (antivirus, backup, another process) is left alone: deleting a
    complete model over a transient lock is exactly the damage this code path
    exists to avoid.  Returns True if a file was deleted.
    """
    model_bin = model_dir / "model.bin"
    try:
        if model_bin.stat().st_size >= _MIN_MODEL_BIN_BYTES:
            return False
        model_bin.unlink()
    except OSError:
        logger.debug("Could not discard short model.bin: %s", model_bin,
                     exc_info=True)
        return False
    logger.warning("Discarded truncated model.bin: %s", model_bin)
    return True


def _find_hf_cache_dir(model_spec: str) -> Optional[Path]:
    """Locate the HuggingFace cache directory for *model_spec*.

    Returns the ``models--org--name`` directory if found, or ``None`` if the
    model is outside the HF cache (user-managed directory — we won't touch it).

    Two resolution paths:

    1. **Hub ID** (e.g. ``ivrit-ai/whisper-large-v3-ct2``):
       Derive ``models--ivrit-ai--whisper-large-v3-ct2`` and search known
       HF cache roots.
    2. **Local path** already inside a HF cache (e.g.
       ``…/models--ivrit-ai--whisper-large-v3-ct2/snapshots/abc123``):
       Walk up the path looking for a parent whose name starts with
       ``models--``.
    """
    from hebrewscribe.models import get_cache_dirs

    spec = model_spec.strip()

    # --- Path 1: hub ID ---
    is_hub_id = (
        "/" in spec
        and not Path(spec).is_absolute()
        and spec.count("/") == 1
        and not Path(spec).exists()
    )
    if is_hub_id:
        dir_name = "models--" + spec.replace("/", "--")
        for root in get_cache_dirs():
            candidate = root / dir_name
            if candidate.is_dir():
                return candidate
        return None

    # --- Path 2: local path inside HF cache ---
    p = Path(spec)
    for parent in [p] + list(p.parents):
        if parent.name.startswith("models--"):
            return parent

    return None


def _purge_and_reload(host: WorkerHost, opts: RunOptions):
    """Delete a corrupt model cache, re-download, and reload.

    Returns ``(model, resolved_device)`` — same as :func:`load_model`.
    Raises if the cache directory cannot be found (model is user-managed)
    or if re-download / reload fails.
    """
    cache_dir = _find_hf_cache_dir(opts.model_spec)
    if cache_dir is None:
        raise RuntimeError(
            f"Cannot locate HuggingFace cache for '{opts.model_spec}'. "
            f"Manual recovery required: delete the cached model and restart."
        )

    host.post_event("log",
                     message=f"Model cache appears corrupt. "
                             f"Deleting {cache_dir} and re-downloading...")
    host.post_event("job_status", message="Re-downloading model (cache was corrupt)...")

    shutil.rmtree(str(cache_dir), ignore_errors=True)
    logger.warning("Purged corrupt model cache: %s", cache_dir)

    # Clear stale batched-pipeline references.
    _batched_pipeline_cache.clear()

    model, device = load_model(host, opts)
    host.post_event("log", message=f"Model reloaded on device: {device}")
    return model, device


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(host: WorkerHost, opts: RunOptions):
    """Load a Whisper model. Returns (model, resolved_device)."""
    backend = opts.backend

    if backend == "openai-whisper":
        import whisper
        device = opts.device
        if device == "auto":
            try:
                import torch
                if torch.cuda.is_available():
                    device = "cuda"
                elif torch.backends.mps.is_available():
                    device = "mps"
                else:
                    device = "cpu"
            except Exception:
                logger.debug("torch not available, falling back to CPU", exc_info=True)
                device = "cpu"
        host.post_event("log", message=f"Loading openai-whisper model: {opts.model_spec}")
        model = whisper.load_model(opts.model_spec, device=device)
        return model, device

    import faster_whisper
    device = opts.device
    if device == "auto":
        # CTranslate2 (faster-whisper backend) does not support MPS;
        # on macOS Apple Silicon it runs on CPU with int8 acceleration.
        device = "cuda" if cuda_available() else "cpu"

    compute_type = opts.compute_type
    if compute_type == "auto":
        compute_type = "float16" if device == "cuda" else "int8"

    # Pre-download hub models with progress reporting.
    # A HuggingFace repo ID looks like "org/model" (one slash, no path separators).
    # Local filesystem paths are absolute, deeply nested, or already exist on disk.
    model_path = opts.model_spec
    _spec = opts.model_spec
    _is_hub_id = (
        "/" in _spec
        and not Path(_spec).is_absolute()
        and _spec.count("/") == 1
        and not Path(_spec).exists()
    )
    if _is_hub_id:
        model_path = _download_with_progress(host, _spec)

    # Shorten model spec for log display (full path shown only in debug log)
    _display_spec = opts.model_spec
    if len(_display_spec) > 60 or os.sep in _display_spec:
        _p = Path(_display_spec)
        for _part in _p.parts:
            if _part.startswith("models--"):
                _display_spec = _part.replace("models--", "").replace("--", "/", 1)
                break
        else:
            _display_spec = _p.name
    host.post_event("log", message=f"Loading faster-whisper model: {_display_spec}")
    model = faster_whisper.WhisperModel(model_path, device=device, compute_type=compute_type)
    batch_label = "auto" if opts.batch_size == 0 else str(opts.batch_size)
    host.post_event("log", message=f"Device: {device}, compute: {compute_type}, batch_size: {batch_label}")
    return model, device


def _download_with_progress(host: WorkerHost, repo_id: str) -> str:
    """Download a HuggingFace model with progress events posted to the GUI.

    Returns the local path to use with WhisperModel().
    If the model is already cached and ``model.bin`` looks usable, returns
    immediately with no download events.

    A folder created before ``model.bin`` finished is not treated as a cache
    hit: the download is retried rather than raising, so a half-finished
    fetch completes itself instead of failing the batch.

    A ``model.bin`` that exists but is too short is deleted first.
    huggingface_hub returns early for any file that merely exists (it compares
    presence, not length), so without the delete the retry would transfer zero
    bytes and still report success.
    """
    import huggingface_hub

    _allow = [
        "config.json", "preprocessor_config.json",
        "model.bin", "tokenizer.json", "vocabulary.*",
    ]

    # Check if already cached (fast path — no download events emitted)
    try:
        local = huggingface_hub.snapshot_download(
            repo_id, local_files_only=True,
            allow_patterns=_allow,
        )
        ok, detail = _model_bin_status(Path(local))
        if ok:
            host.post_event("log", message=f"Model already cached: {repo_id}")
            return local
        # Remove a short model.bin before retrying: huggingface_hub skips any
        # file that already exists, so leaving it in place would make the
        # retry a no-op that still reports "Download complete".
        _discard_short_model_bin(Path(local))
        host.post_event(
            "log",
            message=(
                f"Cached model is incomplete ({detail}). "
                f"Resuming download..."
            ),
        )
    except Exception:
        pass  # Not cached — proceed with download

    host.post_event("download_start", repo_id=repo_id)
    host.post_event("log", message=f"Downloading model: {repo_id}")

    _progress_lock = threading.Lock()

    class _ProgressTqdm:
        """Minimal tqdm-compatible class that posts download_progress events.

        Must satisfy the tqdm API surface used by huggingface_hub's
        thread_map: get_lock/set_lock class methods, set_description,
        display, moveto, and context-manager protocol.
        """

        _lock = _progress_lock

        def __init__(self, *args, **kwargs):
            self.total = kwargs.get("total", 0) or 0
            self.desc = kwargs.get("desc", "")
            self.n = kwargs.get("initial", 0) or 0
            self.pos = kwargs.get("position", 0)
            self.disable = kwargs.get("disable", False)
            self._last_post = 0.0

        # --- Class-level lock API required by tqdm.contrib.concurrent ---
        @classmethod
        def get_lock(cls):
            return cls._lock

        @classmethod
        def set_lock(cls, lock):
            cls._lock = lock

        # --- Instance methods ---
        def update(self, n=1):
            if self.disable:
                return
            self.n += n
            now = time.monotonic()
            if now - self._last_post < 0.25 and self.n < self.total:
                return
            self._last_post = now
            host.post_event("download_progress",
                            downloaded=self.n, total=self.total,
                            file_desc=self.desc)

        def set_description(self, desc="", refresh=True):
            self.desc = desc or ""

        def set_description_str(self, desc="", refresh=True):
            self.desc = desc or ""

        def display(self, msg=None, pos=None):
            pass

        def moveto(self, n):
            pass

        def clear(self, nolock=False):
            pass

        def refresh(self, nolock=False, lock_args=None):
            pass

        def reset(self, total=None):
            self.n = 0
            if total is not None:
                self.total = total

        def close(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

        def __iter__(self):
            return self

        def __next__(self):
            raise StopIteration

    local = huggingface_hub.snapshot_download(
        repo_id,
        allow_patterns=_allow,
        tqdm_class=_ProgressTqdm,
    )
    host.post_event("download_done", repo_id=repo_id)
    host.post_event("log", message=f"Download complete: {repo_id}")
    return local


# ---------------------------------------------------------------------------
# Batched pipeline
# ---------------------------------------------------------------------------

_batched_pipeline_cache: dict = {}


def _get_batched_pipeline(model) -> object:
    """Wrap a WhisperModel in BatchedInferencePipeline for parallel chunk decoding."""
    model_id = id(model)
    if model_id in _batched_pipeline_cache:
        return _batched_pipeline_cache[model_id]
    try:
        from faster_whisper import BatchedInferencePipeline
        pipeline = BatchedInferencePipeline(model=model)
        logger.debug("BatchedInferencePipeline created")
        _batched_pipeline_cache[model_id] = pipeline
        return pipeline
    except (ImportError, AttributeError):
        logger.warning("BatchedInferencePipeline not available; falling back to sequential")
        _batched_pipeline_cache[model_id] = model
        return model


# ---------------------------------------------------------------------------
# Shared transcription helpers
# ---------------------------------------------------------------------------

_PROGRESS_THROTTLE_SEC = 0.10


def _emit_progress(host: WorkerHost, file_path: Path, phase: str,
                   percent: float, elapsed: float, eta,
                   speed, processed_seconds: float,
                   total_seconds) -> None:
    """Emit a current_progress event with standard field layout."""
    host.post_event(
        "current_progress",
        path=str(file_path),
        phase=phase,
        percent=percent,
        elapsed=elapsed,
        eta=eta,
        speed=speed,
        processed_seconds=processed_seconds,
        total_seconds=total_seconds,
        row_status="Running",
    )


def _normalize_segment(seg, from_dict: bool = False) -> dict:
    """Build a canonical segment dict from openai (dict) or faster-whisper (object).

    When faster-whisper ran with word_timestamps=True the dict also carries
    "words": [{"start", "end", "word"}]. The key is absent otherwise, so
    diarization-off outputs stay byte-identical. The openai path never carries
    words — that backend gets segment-level speaker assignment only.
    """
    if from_dict:
        text = (seg.get("text") or "").strip()
        return {
            "id": seg.get("id"),
            "start": float(seg.get("start", 0.0)),
            "end": float(seg.get("end", 0.0)),
            "text": text,
        }
    text = (seg.text or "").strip()
    normed = {
        "id": getattr(seg, "id", None),
        "start": float(seg.start),
        "end": float(seg.end),
        "text": text,
    }
    words = getattr(seg, "words", None)
    if words:
        normed["words"] = [
            {"start": float(w.start), "end": float(w.end), "word": w.word}
            for w in words
        ]
    return normed


def _completion_progress(host: WorkerHost, file_path: Path,
                         start_wall: float, total_seconds,
                         segments: list) -> None:
    """Emit the final 100% progress event after transcription."""
    elapsed = time.time() - start_wall
    processed = segments[-1]["end"] if segments else 0.0
    speed = (processed / elapsed) if elapsed > 0 else None
    final_total = total_seconds or processed
    _emit_progress(host, file_path, "Transcribing complete",
                   100, elapsed, 0, speed, final_total, final_total)


# ---------------------------------------------------------------------------
# Transcription backends
# ---------------------------------------------------------------------------

def transcribe_openai(host: WorkerHost, model, file_path: Path, opts: RunOptions, device: str,
                      display_path: Optional[Path] = None) -> dict:
    # display_path: the original source file for events/UI. file_path may be a
    # pre-converted temp WAV whose path the GUI doesn't know (rows are keyed by
    # the original path), so events must never carry it.
    display_path = display_path or file_path
    check_for_cancel(host)
    wait_if_paused(host)

    duration = probe_duration_seconds(file_path)
    start_wall = time.time()
    host.post_event("current_file", name=display_path.name, phase="Transcribing (openai-whisper)")
    _emit_progress(host, display_path, "Transcribing (openai-whisper)",
                   0, 0, None, None, 0, duration)

    fp16 = device == "cuda"
    language = None if opts.language == "auto" else opts.language

    result = model.transcribe(
        str(file_path),
        language=language,
        task=opts.task,
        verbose=False,
        fp16=fp16,
        condition_on_previous_text=opts.condition_on_previous_text,
        beam_size=opts.beam_size,
    )

    segments = []
    preview_lines = []
    for seg in result.get("segments", []):
        normed = _normalize_segment(seg, from_dict=True)
        segments.append(normed)
        if normed["text"]:
            preview_lines.append(normed["text"])

    _completion_progress(host, display_path, start_wall, duration, segments)

    if preview_lines:
        host.post_event("preview", text="\n".join(preview_lines[-8:]))

    return {
        "text": (result.get("text") or "").strip(),
        "segments": segments,
        "language": result.get("language"),
        "raw": result,
    }


def transcribe_faster(host: WorkerHost, model, file_path: Path, opts: RunOptions,
                      display_path: Optional[Path] = None) -> dict:
    # See transcribe_openai: events carry display_path (the original source
    # file), while file_path may be a pre-converted temp WAV.
    display_path = display_path or file_path
    check_for_cancel(host)
    wait_if_paused(host)

    language = None if opts.language == "auto" else opts.language
    kwargs = {
        "beam_size": opts.beam_size,
        "language": language,
        "task": opts.task,
        "condition_on_previous_text": opts.condition_on_previous_text,
    }
    if opts.vad_filter:
        kwargs["vad_filter"] = True
    if opts.diarize:
        # Word-level speaker assignment needs word timestamps (~10-30% CPU
        # cost). NOTE: combined with vad_filter, faster-whisper rewrites
        # segment start/end to the first/last word times.
        kwargs["word_timestamps"] = True

    effective_batch = opts.batch_size
    if effective_batch == 0:
        effective_batch = _auto_batch_size(opts.device)
    if effective_batch > 1:
        kwargs["batch_size"] = effective_batch

    host.post_event("current_file", name=display_path.name, phase="Initializing (faster-whisper)")
    transcriber = _get_batched_pipeline(model) if effective_batch > 1 else model
    segments_iter, info = transcriber.transcribe(str(file_path), **kwargs)

    total_seconds = getattr(info, "duration", None) or probe_duration_seconds(file_path)
    start_wall = time.time()
    last_emit = start_wall
    segments = []
    texts = []

    detected_language = getattr(info, "language", None)
    host.post_event("current_file", name=display_path.name, phase="Transcribing", language=detected_language)

    for seg in segments_iter:
        check_for_cancel(host)
        wait_if_paused(host)

        normed = _normalize_segment(seg, from_dict=False)
        texts.append(normed["text"])
        segments.append(normed)

        if normed["text"]:
            host.post_event("preview", text=normed["text"])

        now = time.time()
        if now - last_emit >= _PROGRESS_THROTTLE_SEC:
            processed_seconds = float(seg.end)
            percent = (processed_seconds / total_seconds * 100.0) if total_seconds and total_seconds > 0 else 0.0
            elapsed = now - start_wall
            speed = (processed_seconds / elapsed) if elapsed > 0 else None
            eta = ((total_seconds - processed_seconds) / speed) if (speed and total_seconds and speed > 0) else None
            _emit_progress(host, display_path, "Transcribing",
                           percent, elapsed, eta, speed,
                           processed_seconds, total_seconds)
            last_emit = now

    _completion_progress(host, display_path, start_wall, total_seconds, segments)

    return {
        "text": " ".join([t for t in texts if t]).strip(),
        "segments": segments,
        "language": getattr(info, "language", None),
        "language_probability": getattr(info, "language_probability", None),
        "duration": getattr(info, "duration", None),
        "raw": {
            "language": getattr(info, "language", None),
            "language_probability": getattr(info, "language_probability", None),
            "duration": getattr(info, "duration", None),
            "segments": segments,
        },
    }


# ---------------------------------------------------------------------------
# Speaker diarization (optional post-transcription pass)
# ---------------------------------------------------------------------------

def _diarize_file(host: WorkerHost, dia_pipeline, audio_path: Path,
                  display_path: Path, opts: RunOptions, result: dict,
                  file_duration) -> None:
    """Run diarization on one file and label the result's segments in place.

    Failure policy: any per-file diarization error —
    including MemoryError from pyannote's reconstruction memory spike on
    multi-hour audio — keeps the transcript and drops only the speaker
    labels; the batch continues. Cancellation propagates as CancelledByUser.
    """
    from hebrewscribe import diarize as dz

    check_for_cancel(host)
    wait_if_paused(host)
    host.post_event("current_file", name=display_path.name,
                    phase="Identifying speakers…",
                    language=result.get("language"))
    if file_duration and file_duration > dz.LONG_AUDIO_WARN_SECONDS:
        host.post_event(
            "log", level="warning",
            message=f"Warning: {display_path.name} is over 2 hours long — "
                    "speaker identification may use several GB of RAM.")
    waveform = None
    try:
        waveform = dz.load_waveform(audio_path)
        annotation = dz.diarize_waveform(
            dia_pipeline, waveform, host, display_path,
            num_speakers=opts.num_speakers, total_seconds=file_duration)
        turns = dz.annotation_to_turns(annotation)
        segments = dz.assign_speakers_to_segments(result["segments"], turns)
        # Label language follows the OUTPUT text: task=translate produces
        # English text, which must not carry Hebrew "דובר N" labels.
        label_lang = ("en" if opts.task == "translate"
                      else (result.get("language") or opts.language))
        speakers = dz.build_speaker_map(segments, label_lang)
        result["segments"] = segments
        result["speakers"] = speakers
        n = len(speakers)
        host.post_event("log", message=f"Identified {n} speaker"
                                       f"{'s' if n != 1 else ''}")
        if speakers:
            # Refresh the live preview with the labeled, turn-grouped text —
            # exactly what write_txt will produce for this file. The preview
            # holds only the current file (cleared on every file_started), so
            # replace semantics are safe.
            host.post_event("preview_replace",
                            text=format_speaker_transcript(segments, speakers),
                            language=("en" if opts.task == "translate"
                                      else result.get("language")))
    except dz.DiarizationCancelled:
        raise CancelledByUser("Cancelled by user.")
    except CancelledByUser:
        raise
    except Exception as exc:
        # Transcript is preserved; only the labels are lost for this file.
        host.post_event(
            "log", level="warning",
            message=f"Warning: speaker identification failed — keeping the "
                    f"transcript without speaker labels ({exc})")
        logger.warning("Diarization failed for %s", display_path,
                       exc_info=True)
    finally:
        # Free multi-hour tensors before the next file.
        del waveform


# ---------------------------------------------------------------------------
# Output writing
# ---------------------------------------------------------------------------

def write_outputs(host: WorkerHost, source_file: Path, opts: RunOptions, result: dict, used_stems: set) -> None:
    out_dir = Path(opts.output_dir)
    stem = unique_stem(source_file, out_dir, opts.formats, used_stems)

    if stem != safe_stem(source_file):
        host.post_event("log", message=f"Output renamed to '{stem}.*' to avoid collision")

    speakers = result.get("speakers")
    if "txt" in opts.formats:
        write_txt(out_dir / f"{stem}.txt", result["text"],
                  segments=result["segments"], speakers=speakers)
    if "srt" in opts.formats:
        write_srt(out_dir / f"{stem}.srt", result["segments"],
                  speakers=speakers)
    if "json" in opts.formats:
        write_json(out_dir / f"{stem}.json", str(source_file),
                   opts.backend, opts.model_spec, result)


# ---------------------------------------------------------------------------
# Main worker loop
# ---------------------------------------------------------------------------

def run_transcription_worker(host: WorkerHost, opts: RunOptions, total: int) -> None:
    """Run the transcription batch. Called from a background thread.

    The worker pulls files one at a time via host.next_file() instead of
    iterating a frozen snapshot. This allows the GUI to reorder the queue
    while the worker is running.
    """
    done_count = 0
    failed_count = 0
    running_count = 0
    started_any = False
    used_stems: set = set()

    try:
        _validate_opts(opts)
        host.post_event("job_status", message="Loading model...")
        try:
            model, resolved_device = load_model(host, opts)
        except Exception as load_exc:
            # Auto-recover from a corrupt or format-incompatible model cache.
            # This fires when model.bin is missing, truncated, or was written by
            # a different ctranslate2 major version (e.g. after rebuilding the
            # installer against a newer Python/ctranslate2).
            if _is_ct2_corruption_error(load_exc) and opts.backend == "faster-whisper":
                host.post_event("log",
                                message="Model failed to load (cache corrupt or "
                                        "incompatible format). Attempting recovery...")
                model, resolved_device = _purge_and_reload(host, opts)
            else:
                raise
        host.post_event("log", message=f"Model loaded on device: {resolved_device}")

        # --- Optional diarization pipeline (loaded once per batch) ---
        # Any load failure degrades the whole batch to no-diarization; it
        # never kills the run. Cancel during the first-use weights download
        # propagates as a normal cancellation.
        dia_pipeline = None
        if opts.diarize:
            host.post_event("job_status",
                            message="Loading speaker identification model...")
            try:
                from hebrewscribe import diarize as _dz
            except Exception:
                _dz = None
                logger.warning("hebrewscribe.diarize failed to import",
                               exc_info=True)
            if _dz is not None:
                try:
                    dia_pipeline = _dz.load_diarization_pipeline(
                        host, opts.diarize_device)
                except CancelledByUser:
                    raise
                except _dz.DiarizationCancelled:
                    raise CancelledByUser("Cancelled by user.")
                except Exception as dia_exc:
                    host.post_event(
                        "log", level="warning",
                        message="Warning: speaker identification unavailable — "
                                f"continuing without speaker labels ({dia_exc})")
                    logger.warning("Diarization pipeline load failed",
                                   exc_info=True)
                    dia_pipeline = None
            else:
                host.post_event(
                    "log", level="warning",
                    message="Warning: speaker identification unavailable — "
                            "continuing without speaker labels.")
        host.post_event("job_status", message="Model loaded. Ready to transcribe.")

        model_recovered = False  # True after one successful purge+reload

        idx = 0
        while True:
            check_for_cancel(host)
            wait_if_paused(host)
            if host.stop_requested:
                break

            file_path = host.next_file()
            if file_path is None:
                break

            idx += 1
            started_any = True
            running_count = 1
            p = Path(file_path)
            file_duration = probe_duration_seconds(p)
            host.post_event("file_started", path=file_path, name=p.name, phase=f"Preparing file {idx}/{total}", idx=idx, total=total, duration=file_duration)
            host.post_event("overall_progress", value=done_count, done=done_count, failed=failed_count, running=running_count)
            host.post_event("job_status", message=f"{idx}/{total}: {p.name}")
            host.post_event("log", message=f"Transcribing: {p.name}")

            try:
                validate_audio_file(p)

                audio_path = preconvert_audio(p)
                if audio_path != p:
                    host.post_event("log", message=f"Pre-converted to WAV: {audio_path.name}")
                if opts.backend == "openai-whisper":
                    result = transcribe_openai(host, model, audio_path, opts, resolved_device,
                                               display_path=p)
                else:
                    result = transcribe_faster(host, model, audio_path, opts, display_path=p)

                if dia_pipeline is not None:
                    # Keep in sync with the corruption-retry call site below.
                    _diarize_file(host, dia_pipeline, audio_path, p, opts,
                                  result, file_duration)

                check_for_cancel(host)
                wait_if_paused(host)
                host.post_event("current_file", name=p.name, phase="Writing outputs", language=result.get("language"))
                write_outputs(host, p, opts, result, used_stems)
                done_count += 1
                running_count = 0
                host.post_event("file_finished", path=file_path, language=result.get("language"), audio_seconds=file_duration)
                host.post_event("overall_progress", value=done_count, done=done_count, failed=failed_count, running=running_count)
                host.post_event("log", message=f"Done: {p.name}")
            except CancelledByUser:
                running_count = 0
                host.post_event("file_failed", path=file_path, status="Cancelled", progress_text="--")
                raise
            except Exception as file_exc:
                # --- Auto-recovery for corrupt CTranslate2 models ---
                if (
                    _is_ct2_corruption_error(file_exc)
                    and not model_recovered
                    and opts.backend == "faster-whisper"
                ):
                    host.post_event("log",
                                    message=f"Detected model corruption in {p.name}, "
                                            f"attempting automatic recovery...")
                    try:
                        model, resolved_device = _purge_and_reload(host, opts)
                        model_recovered = True

                        # Retry the same file with the fresh model.
                        host.post_event("log", message=f"Retrying: {p.name}")
                        host.post_event("job_status",
                                        message=f"{idx}/{total}: {p.name} (retry)")
                        host.post_event("file_started", path=file_path,
                                        name=p.name,
                                        phase=f"Retrying {idx}/{total}",
                                        idx=idx, total=total,
                                        duration=file_duration)

                        audio_path = preconvert_audio(p)
                        result = transcribe_faster(host, model, audio_path, opts, display_path=p)

                        if dia_pipeline is not None:
                            # Keep in sync with the main call site above.
                            _diarize_file(host, dia_pipeline, audio_path, p,
                                          opts, result, file_duration)

                        check_for_cancel(host)
                        wait_if_paused(host)
                        host.post_event("current_file", name=p.name,
                                        phase="Writing outputs",
                                        language=result.get("language"))
                        write_outputs(host, p, opts, result, used_stems)
                        done_count += 1
                        running_count = 0
                        host.post_event("file_finished", path=file_path,
                                        language=result.get("language"),
                                        audio_seconds=file_duration)
                        host.post_event("overall_progress", value=done_count,
                                        done=done_count, failed=failed_count,
                                        running=running_count)
                        host.post_event("log", message=f"Done (after recovery): {p.name}")
                        continue  # skip the failure path below
                    except CancelledByUser:
                        running_count = 0
                        host.post_event("file_failed", path=file_path,
                                        status="Cancelled", progress_text="--")
                        raise
                    except Exception as retry_exc:
                        # Recovery or retry failed — fall through to normal
                        # failure handling with the *retry* exception.
                        file_exc = retry_exc
                        host.post_event("log",
                                        message=f"Recovery failed for {p.name}: "
                                                f"{retry_exc}")

                failed_count += 1
                running_count = 0
                error_detail = f"{file_exc}\n\n{traceback.format_exc()}"
                host.post_event("file_failed", path=file_path, status="Failed", progress_text="--", error_message=error_detail)
                host.post_event("overall_progress", value=done_count, done=done_count, failed=failed_count, running=running_count)
                host.post_event("log", message=f"ERROR in {p.name}: {file_exc}")
                host.post_event("log", message=traceback.format_exc())

        show_warning = failed_count > 0
        if host.cancel_requested:
            message = f"Cancelled. Completed: {done_count}. Failed: {failed_count}."
            host.post_event("done", message=message, cancelled=True, stopped=False, show_warning=show_warning or started_any, done_count=done_count, failed_count=failed_count)
        elif host.stop_requested:
            message = f"Stopped after current file. Completed: {done_count}. Failed: {failed_count}."
            host.post_event("done", message=message, cancelled=False, stopped=True, show_warning=show_warning, done_count=done_count, failed_count=failed_count)
        else:
            message = f"Finished. Completed: {done_count}. Failed: {failed_count}."
            host.post_event("done", message=message, cancelled=False, stopped=False, show_warning=show_warning, done_count=done_count, failed_count=failed_count)
    except CancelledByUser:
        message = f"Cancelled. Completed: {done_count}. Failed: {failed_count}."
        host.post_event("done", message=message, cancelled=True, stopped=False, show_warning=(failed_count > 0 or started_any), done_count=done_count, failed_count=failed_count)
    except Exception as e:
        host.post_event("failed", message=f"{e}\n\n{traceback.format_exc()}")


# ---------------------------------------------------------------------------
# Self-test: validate the full pipeline with a tiny model and short clip
# ---------------------------------------------------------------------------

SELFTEST_MODEL = "Systran/faster-whisper-tiny"


# Number of checks run_selftest performs — keep in sync with its docstring
# list; the GUI's "Running N checks…" label reads this.
SELFTEST_CHECK_COUNT = 7


@dataclass
class SelfTestResult:
    """Result of run_selftest()."""
    passed: bool
    checks: List[str]        # human-readable lines: "[OK] ..." or "[FAIL] ..."
    error: Optional[str]     # traceback if a fatal error occurred
    elapsed: float           # total wall-clock seconds


def run_selftest(host: WorkerHost, backend: str, device: str,
                 compute_type: str) -> SelfTestResult:
    """Run a quick self-test using the user's current engine settings.

    Tests:
      1. Test audio fixture is available
      2. ffmpeg is reachable
      3. Backend package imports
      4. Model loads on the selected device/compute_type
      5. Transcription produces non-empty output
      6. Output files can be written
      7. Speaker identification loads (optional — skipped when not installed)

    Uses Systran/faster-whisper-tiny (~75 MB) regardless of the user's
    model selection, to keep the test fast (~2-5 seconds on CPU).
    """
    checks: List[str] = []
    start = time.time()

    try:
        # --- 1. Fixture ---
        audio = _selftest_audio_path()
        if audio is None or not audio.exists():
            checks.append("[FAIL] Test audio clip not found")
            return SelfTestResult(False, checks, None, time.time() - start)
        checks.append(f"[OK] Test audio clip found ({audio.name})")

        # --- 2. ffmpeg ---
        if ffmpeg_available():
            checks.append("[OK] ffmpeg is available")
        else:
            checks.append("[WARN] ffmpeg not found (pre-conversion disabled)")

        # --- 3. Backend import ---
        if backend == "faster-whisper":
            try:
                import faster_whisper  # noqa: F401
                checks.append("[OK] faster-whisper is installed")
            except ImportError:
                checks.append("[FAIL] faster-whisper is not installed")
                return SelfTestResult(False, checks, None, time.time() - start)
        else:
            try:
                import whisper  # noqa: F401
                checks.append("[OK] openai-whisper is installed")
            except ImportError:
                checks.append("[FAIL] openai-whisper is not installed")
                return SelfTestResult(False, checks, None, time.time() - start)

        # --- 4. Model load ---
        host.post_event("selftest_progress", step="Loading test model...")
        test_opts = RunOptions(
            backend=backend,
            model_spec=SELFTEST_MODEL if backend == "faster-whisper" else "tiny",
            output_dir="",  # placeholder, overridden below
            language="en",
            task="transcribe",
            device=device,
            compute_type=compute_type,
            beam_size=1,
            vad_filter=False,
            condition_on_previous_text=False,
            batch_size=0,
            formats=["txt"],
        )
        model, resolved_device = load_model(host, test_opts)
        checks.append(f"[OK] Model loaded on {resolved_device}")

        # --- 5. Transcription ---
        host.post_event("selftest_progress", step="Transcribing test clip...")
        if backend == "faster-whisper":
            result = transcribe_faster(host, model, audio, test_opts)
        else:
            result = transcribe_openai(host, model, audio, test_opts, resolved_device)

        text = (result.get("text") or "").strip()
        seg_count = len(result.get("segments", []))
        if text and seg_count > 0:
            checks.append(f"[OK] Transcription produced {seg_count} segment(s)")
        elif text:
            checks.append("[OK] Transcription produced text (no segments)")
        else:
            checks.append("[FAIL] Transcription produced no output")
            return SelfTestResult(False, checks, None, time.time() - start)

        # --- 6. Output writing ---
        host.post_event("selftest_progress", step="Testing output writing...")
        with tempfile.TemporaryDirectory() as tmpdir:
            test_opts_out = RunOptions(
                backend=test_opts.backend,
                model_spec=test_opts.model_spec,
                output_dir=tmpdir,
                language="en", task="transcribe",
                device=device, compute_type=compute_type,
                beam_size=1, vad_filter=False,
                condition_on_previous_text=False,
                batch_size=0, formats=["txt", "srt", "json"],
            )
            used: set = set()
            write_outputs(host, audio, test_opts_out, result, used)
            written = list(Path(tmpdir).glob("*"))
            if len(written) >= 3:
                checks.append(f"[OK] Output files written ({len(written)} files)")
            else:
                checks.append(f"[FAIL] Expected 3 output files, got {len(written)}")
                return SelfTestResult(False, checks, None, time.time() - start)

        # --- 7. Speaker identification (optional) ---
        try:
            from hebrewscribe.diarize import (is_diarization_available,
                                              load_diarization_pipeline,
                                              resolve_weights_dir)
            if not is_diarization_available():
                checks.append(
                    "[--] Speaker identification not installed (optional)")
            elif resolve_weights_dir() is None:
                checks.append("[WARN] Speaker identification installed; "
                              "model will download on first use")
            else:
                host.post_event("selftest_progress",
                                step="Loading speaker model...")
                try:
                    load_diarization_pipeline(host, "cpu")
                    checks.append("[OK] Speaker identification pipeline loads")
                except Exception as dia_exc:
                    checks.append(
                        f"[FAIL] Speaker identification broken: {dia_exc}")
                    return SelfTestResult(False, checks, None,
                                          time.time() - start)
        except CancelledByUser:
            raise
        except Exception as dia_exc:
            checks.append(f"[WARN] Speaker identification check skipped: "
                          f"{dia_exc}")

        elapsed = time.time() - start
        checks.append(f"\nAll checks passed in {elapsed:.1f}s")
        return SelfTestResult(True, checks, None, elapsed)

    except CancelledByUser:
        checks.append("[--] Cancelled by user")
        return SelfTestResult(False, checks, None, time.time() - start)
    except Exception as e:
        checks.append(f"[FAIL] {e}")
        return SelfTestResult(False, checks, traceback.format_exc(),
                              time.time() - start)
