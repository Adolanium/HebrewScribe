"""Transcription worker: model loading, transcription backends, output writing.

This module contains all transcription logic, decoupled from the GUI.
Communication with the GUI happens through the WorkerHost protocol.
"""

import atexit
import logging
import os
import shutil
import subprocess
import sys
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
from hebrewscribe.outputs import write_txt, write_srt, write_json

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
    """Return True if a CUDA GPU is usable for faster-whisper/CTranslate2.

    Prefer ``ctranslate2.get_cuda_device_count()`` — the packaged app does not
    bundle PyTorch, so a torch-only check always reported False and forced CPU
    even when CTranslate2's own CUDA runtime was available.
    """
    try:
        import ctranslate2
        if int(ctranslate2.get_cuda_device_count()) > 0:
            return True
    except Exception:
        logger.debug("ctranslate2 CUDA probe failed", exc_info=True)

    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        logger.debug("torch CUDA probe failed", exc_info=True)
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
       ``ctranslate2.models.Whisper.__init__`` when ``model.bin`` is missing,
       unreadable, or incompatible.  Not every occurrence means the HF cache
       is corrupt (Windows symlink / device issues can produce the same
       message), so callers should validate the on-disk model before purging.
    """
    if not isinstance(exc, RuntimeError):
        return False
    msg = str(exc)
    return (
        "json.exception.type_error" in msg
        or "Unable to open file" in msg
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


def _materialize_model_dir(model_path: str, host: Optional[WorkerHost] = None) -> str:
    """Return a model directory free of broken Windows HF-cache symlinks.

    HuggingFace hub stores snapshot files as symlinks into ``blobs/``.  Some
    native loaders (and some Windows permission setups) fail to open those
    symlinks even when Python can read them.  When any required file is a
    symlink, hardlink (or copy) the resolved targets into a sibling
    ``_materialized`` directory and return that path.
    """
    root = Path(model_path)
    if not root.is_dir():
        return model_path

    # Already a flat/materialized tree — nothing to do.
    try:
        needs_materialize = any(
            (root / name).is_symlink()
            for name in ("model.bin", "config.json", "tokenizer.json",
                         "vocabulary.json", "vocabulary.txt",
                         "preprocessor_config.json")
            if (root / name).exists()
        )
    except OSError:
        needs_materialize = False

    if not needs_materialize:
        return model_path

    dest = root.parent / f"{root.name}__materialized"
    try:
        dest.mkdir(parents=True, exist_ok=True)
        for src in root.iterdir():
            if not src.is_file() and not src.is_symlink():
                continue
            target = dest / src.name
            if target.exists():
                # Reuse existing hardlink/copy if size matches.
                try:
                    if target.stat().st_size == src.stat().st_size:
                        continue
                except OSError:
                    pass
                try:
                    target.unlink()
                except OSError:
                    pass

            real = src.resolve()
            linked = False
            try:
                os.link(str(real), str(target))
                linked = True
            except OSError:
                linked = False
            if not linked:
                shutil.copy2(str(real), str(target))

        ok, detail = _model_bin_status(dest)
        if not ok:
            logger.warning("Materialized model dir still invalid: %s", detail)
            return model_path

        if host is not None:
            host.post_event(
                "log",
                message="Using materialized model directory (resolved HF symlinks).",
            )
        logger.info("Materialized CT2 model dir: %s -> %s", root, dest)
        return str(dest)
    except Exception:
        logger.warning("Failed to materialize model dir %s", root, exc_info=True)
        return model_path


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
    """Recover from a failed model load: materialize, then purge+re-download.

    Returns ``(model, resolved_device)`` — same as :func:`load_model`.
    Raises if recovery fails.
    """
    # Clear stale batched-pipeline references before any reload.
    _batched_pipeline_cache.clear()

    # --- Step 1: if the on-disk model looks intact, avoid destructive purge ---
    # "Unable to open file 'model.bin'" is often a Windows symlink / path issue,
    # not a corrupt download.  Materialize and retry first.
    candidate_dirs: list[Path] = []
    spec = opts.model_spec.strip()
    if Path(spec).is_dir():
        candidate_dirs.append(Path(spec))
    cache_dir = _find_hf_cache_dir(spec)
    if cache_dir is not None:
        snapshots = cache_dir / "snapshots"
        if snapshots.is_dir():
            candidate_dirs.extend(
                p for p in snapshots.iterdir() if p.is_dir()
            )

    for model_dir in candidate_dirs:
        ok, detail = _model_bin_status(model_dir)
        if not ok:
            logger.debug("Skip materialize for %s: %s", model_dir, detail)
            continue
        host.post_event(
            "log",
            message=("Model files look intact on disk; resolving symlinks and "
                     f"retrying load ({detail})..."),
        )
        host.post_event("job_status", message="Retrying model load...")
        materialized = _materialize_model_dir(str(model_dir), host=host)
        try:
            # Load directly from the local path (bypass hub download).
            local_opts = RunOptions(
                backend=opts.backend,
                model_spec=materialized,
                output_dir=opts.output_dir,
                language=opts.language,
                task=opts.task,
                device=opts.device,
                compute_type=opts.compute_type,
                beam_size=opts.beam_size,
                vad_filter=opts.vad_filter,
                condition_on_previous_text=opts.condition_on_previous_text,
                batch_size=opts.batch_size,
                formats=list(opts.formats),
            )
            model, device = load_model(host, local_opts)
            host.post_event("log", message=f"Model reloaded on device: {device}")
            return model, device
        except Exception as exc:
            logger.warning(
                "Materialized reload failed for %s: %s", model_dir, exc,
            )

    # --- Step 2: destructive purge + re-download (hub models only) ---
    if cache_dir is None:
        raise RuntimeError(
            f"Cannot locate HuggingFace cache for '{opts.model_spec}'. "
            f"Manual recovery required: delete the cached model and restart."
        )

    host.post_event("log",
                     message=f"Model cache appears corrupt or unreadable. "
                             f"Deleting {cache_dir} and re-downloading...")
    host.post_event("job_status", message="Re-downloading model (cache was corrupt)...")

    # Also drop any materialized sibling dirs left from step 1.
    snapshots = cache_dir / "snapshots"
    if snapshots.is_dir():
        for p in snapshots.iterdir():
            if p.is_dir() and p.name.endswith("__materialized"):
                shutil.rmtree(str(p), ignore_errors=True)

    shutil.rmtree(str(cache_dir), ignore_errors=True)
    logger.warning("Purged corrupt model cache: %s", cache_dir)

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
    elif device == "cuda" and not cuda_available():
        host.post_event(
            "log",
            message="CUDA requested but no GPU is available to CTranslate2; using CPU.",
        )
        device = "cpu"

    compute_type = opts.compute_type
    if compute_type == "auto":
        compute_type = "float16" if device == "cuda" else "int8"
    elif compute_type == "default":
        compute_type = "default"

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

    # Resolve HF snapshot symlinks on Windows before handing the path to CT2.
    if Path(model_path).is_dir():
        ok, detail = _model_bin_status(Path(model_path))
        if not ok:
            raise RuntimeError(
                f"Model directory is not ready for loading ({detail}): {model_path}"
            )
        model_path = _materialize_model_dir(model_path, host=host)

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

    def _try_load(dev: str, ctype: str):
        return faster_whisper.WhisperModel(
            model_path, device=dev, compute_type=ctype,
        )

    try:
        model = _try_load(device, compute_type)
    except Exception as first_exc:
        # If CUDA load fails for any reason, fall back to CPU before giving up.
        # A common misreport is "Unable to open file 'model.bin'" when the real
        # problem is a broken/partial CUDA runtime in the frozen app.
        if device == "cuda":
            host.post_event(
                "log",
                message=(
                    f"CUDA load failed ({first_exc}). Falling back to CPU/int8..."
                ),
            )
            try:
                model = _try_load("cpu", "int8")
                device = "cpu"
                compute_type = "int8"
            except Exception:
                raise first_exc from None
        else:
            raise

    batch_label = "auto" if opts.batch_size == 0 else str(opts.batch_size)
    host.post_event("log", message=f"Device: {device}, compute: {compute_type}, batch_size: {batch_label}")
    return model, device


def _download_with_progress(host: WorkerHost, repo_id: str) -> str:
    """Download a HuggingFace model with progress events posted to the GUI.

    Returns the local path to use with WhisperModel().
    If the model is already cached, returns immediately with no download events.

    After download (or cache hit), validates that ``model.bin`` is present,
    readable, and large enough — hub metadata can report "complete" while the
    weight file is still missing or a broken symlink.
    """
    import huggingface_hub

    # On Windows, prefer real files over hub snapshot symlinks when possible.
    # (Safe no-op on other platforms / older hub versions.)
    if sys.platform == "win32":
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")

    _allow = [
        "config.json", "preprocessor_config.json",
        "model.bin", "tokenizer.json", "vocabulary.*",
    ]

    def _validate_or_raise(local_path: str, context: str) -> str:
        ok, detail = _model_bin_status(Path(local_path))
        if not ok:
            raise RuntimeError(
                f"Model download {context} but files are not usable ({detail}). "
                f"Path: {local_path}"
            )
        return local_path

    # Check if already cached (fast path — no download events emitted)
    try:
        local = huggingface_hub.snapshot_download(
            repo_id, local_files_only=True,
            allow_patterns=_allow,
        )
        host.post_event("log", message=f"Model already cached: {repo_id}")
        return _validate_or_raise(local, "reported as cached")
    except RuntimeError:
        raise
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
    local = _validate_or_raise(local, "finished")
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
    """Build a canonical segment dict from openai (dict) or faster-whisper (object)."""
    if from_dict:
        text = (seg.get("text") or "").strip()
        return {
            "id": seg.get("id"),
            "start": float(seg.get("start", 0.0)),
            "end": float(seg.get("end", 0.0)),
            "text": text,
        }
    text = (seg.text or "").strip()
    return {
        "id": getattr(seg, "id", None),
        "start": float(seg.start),
        "end": float(seg.end),
        "text": text,
    }


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

def transcribe_openai(host: WorkerHost, model, file_path: Path, opts: RunOptions, device: str) -> dict:
    check_for_cancel(host)
    wait_if_paused(host)

    duration = probe_duration_seconds(file_path)
    start_wall = time.time()
    host.post_event("current_file", name=file_path.name, phase="Transcribing (openai-whisper)")
    _emit_progress(host, file_path, "Transcribing (openai-whisper)",
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

    _completion_progress(host, file_path, start_wall, duration, segments)

    if preview_lines:
        host.post_event("preview", text="\n".join(preview_lines[-8:]))

    return {
        "text": (result.get("text") or "").strip(),
        "segments": segments,
        "language": result.get("language"),
        "raw": result,
    }


def transcribe_faster(host: WorkerHost, model, file_path: Path, opts: RunOptions) -> dict:
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

    effective_batch = opts.batch_size
    if effective_batch == 0:
        effective_batch = _auto_batch_size(opts.device)
    if effective_batch > 1:
        kwargs["batch_size"] = effective_batch

    host.post_event("current_file", name=file_path.name, phase="Initializing (faster-whisper)")
    transcriber = _get_batched_pipeline(model) if effective_batch > 1 else model
    segments_iter, info = transcriber.transcribe(str(file_path), **kwargs)

    total_seconds = getattr(info, "duration", None) or probe_duration_seconds(file_path)
    start_wall = time.time()
    last_emit = start_wall
    segments = []
    texts = []

    detected_language = getattr(info, "language", None)
    host.post_event("current_file", name=file_path.name, phase="Transcribing", language=detected_language)

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
            _emit_progress(host, file_path, "Transcribing",
                           percent, elapsed, eta, speed,
                           processed_seconds, total_seconds)
            last_emit = now

    _completion_progress(host, file_path, start_wall, total_seconds, segments)

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
# Output writing
# ---------------------------------------------------------------------------

def write_outputs(host: WorkerHost, source_file: Path, opts: RunOptions, result: dict, used_stems: set) -> None:
    out_dir = Path(opts.output_dir)
    stem = unique_stem(source_file, out_dir, opts.formats, used_stems)

    if stem != safe_stem(source_file):
        host.post_event("log", message=f"Output renamed to '{stem}.*' to avoid collision")

    if "txt" in opts.formats:
        write_txt(out_dir / f"{stem}.txt", result["text"])
    if "srt" in opts.formats:
        write_srt(out_dir / f"{stem}.srt", result["segments"])
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
                    result = transcribe_openai(host, model, audio_path, opts, resolved_device)
                else:
                    result = transcribe_faster(host, model, audio_path, opts)

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
                        result = transcribe_faster(host, model, audio_path, opts)

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
      3. Model loads on the selected device/compute_type
      4. Transcription produces non-empty output
      5. Output files can be written

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
