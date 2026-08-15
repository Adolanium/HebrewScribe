"""Speaker diarization via pyannote.audio 4.x (speaker-diarization-community-1).

Design constraints:

- Fully offline: the pipeline loads from a local weights directory
  (``Pipeline.from_pretrained(<dir>)`` resolves the config's relative
  ``$model/`` placeholders — no HuggingFace token, no network).
- ``PYANNOTE_METRICS_ENABLED`` must be set to ``"false"`` BEFORE any pyannote
  import: pyannote 4.x ships default-on OpenTelemetry that posts to
  otel.pyannote.ai on pipeline init and per run.
- Audio is fed as an in-memory waveform dict
  ``{"waveform": (1, N) float32 tensor, "sample_rate": 16000}`` which bypasses
  torchcodec entirely (frozen builds exclude torchcodec).
- macOS defaults to MPS with a CPU fallback; clustering runs on CPU regardless.
- Cancellation is implemented by raising :class:`DiarizationCancelled` from the
  pipeline progress hook — the only cancel path pyannote offers. The hook only
  fires during the segmentation/embedding phases; clustering and reconstruction
  are uninterruptible, so a cancel may lag by a phase.
- The openai-whisper backend never requests word timestamps, so it gets
  segment-level speaker assignment only (the word-level path needs
  ``word_timestamps=True`` from faster-whisper).

No pyannote/torch/numpy imports at module level — everything heavy is lazy so
the app imports this module freely even without the diarization extra.
"""

import hashlib
import logging
import math
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
import wave
from bisect import bisect_right
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

from hebrewscribe.utils import _bundled_bin, get_models_dir, is_package_available

if TYPE_CHECKING:  # pragma: no cover - typing only
    from hebrewscribe.worker import WorkerHost

logger = logging.getLogger("HebrewScribe")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COMMUNITY1_DIR_NAME = "community-1"

# Weight files expected inside the community-1 snapshot directory;
# weights_complete() checks all of these before any pipeline load.
EXPECTED_FILES = (
    "config.yaml",
    "segmentation/pytorch_model.bin",
    "embedding/pytorch_model.bin",
    "plda/plda.npz",
    "plda/xvec_transform.npz",
)

# Self-hosted weights release asset (CC-BY-4.0 — attribution ships in
# THIRD-PARTY-LICENSES.md and inside the tarball itself).
DIARIZATION_WEIGHTS_VERSION = "v1"
DIARIZATION_WEIGHTS_URL = (
    "https://github.com/yevgeniyglider/HebrewScribe/releases/download/"
    "models-diarization-v1/hebrewscribe-diarization-community-1-v1.tar.gz"
)
# SHA-256 of the release tarball above; downloads that do not match are
# rejected and deleted.
DIARIZATION_WEIGHTS_SHA256: Optional[str] = (
    "d0b2190e8cd88ed58762bfc042976828c42c53bbafd9155643f4dc20c0cfcfdb"
)

LONG_AUDIO_WARN_SECONDS = 2 * 3600
MIN_FLIP_SECONDS = 0.5

# Speaker display labels per transcript language. Hebrew gets the RTL-safe
# native label (a Latin "Speaker 1:" prefix flips the line's base direction in
# renderers that auto-detect direction). Writers append the ":".
SPEAKER_LABEL_FORMATS = {"he": "דובר {n}"}
DEFAULT_SPEAKER_LABEL_FORMAT = "Speaker {n}"

_DOWNLOAD_REPO_LABEL = "diarization/community-1"
_PROGRESS_THROTTLE_SEC = 0.25

# Percent ranges for mapping per-phase hook progress onto one overall bar.
_HOOK_PHASE_RANGES = {
    "segmentation": (0.0, 60.0),
    "embeddings": (60.0, 95.0),
}


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class DiarizationCancelled(Exception):
    """Sentinel raised from the pipeline hook / downloader on user cancel."""


class DiarizationUnavailable(Exception):
    """pyannote not installed, weights unavailable, or audio unreadable."""


# ---------------------------------------------------------------------------
# Availability & weights resolution
# ---------------------------------------------------------------------------

def _set_offline_env() -> None:
    """Disable pyannote's default-on telemetry. MUST run before any pyannote
    import. setdefault: an explicit user opt-in via the env var is respected."""
    os.environ.setdefault("PYANNOTE_METRICS_ENABLED", "false")


def is_diarization_available() -> bool:
    """True if pyannote.audio imports. Sets the telemetry kill-switch FIRST
    because the probe performs a real import."""
    _set_offline_env()
    return is_package_available("pyannote.audio")


def weights_complete(weights_dir: Path) -> bool:
    """True if every expected weight file exists and is non-empty."""
    for rel in EXPECTED_FILES:
        f = Path(weights_dir) / rel
        try:
            if not f.is_file() or f.stat().st_size == 0:
                return False
        except OSError:
            return False
    return True


def resolve_weights_dir() -> Optional[Path]:
    """Locate a complete local weights directory, or None.

    Search order (first complete dir wins):
      1. beside the executable (Windows installed layout, sibling of ffmpeg/)
      2. the PyInstaller data dir / macOS Resources (mac .app layout)
      3. the user models dir (~/.hebrewscribe/models/...), where source
         installs land after the first-use download.
    """
    bundled_rel = Path("models") / "diarization" / COMMUNITY1_DIR_NAME
    candidates: List[Path] = []
    if getattr(sys, "frozen", False):
        app_dir = Path(sys.executable).parent
        candidates.append(app_dir / bundled_rel)
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            candidates.append(Path(meipass) / bundled_rel)
        if sys.platform == "darwin":
            candidates.append(app_dir.parent / "Resources" / bundled_rel)
    candidates.append(get_models_dir() / "diarization" / COMMUNITY1_DIR_NAME)
    for d in candidates:
        if weights_complete(d):
            return d
    return None


# ---------------------------------------------------------------------------
# Weights download (plain HTTPS, not HuggingFace)
# ---------------------------------------------------------------------------

def _safe_extract_tar(tar_path: Path, dest: Path) -> None:
    """Extract a tarball, refusing path-traversal members.

    Uses the stdlib "data" extraction filter when available (3.11.4+ /
    backports); otherwise sanitizes member paths by hand.
    """
    with tarfile.open(tar_path, "r:gz") as tf:
        try:
            tf.extractall(dest, filter="data")
        except TypeError:  # Python without the filter= parameter
            base = dest.resolve()
            for member in tf.getmembers():
                # Mirror filter="data": no links, no devices/FIFOs — a
                # symlink member could redirect later writes outside dest.
                if member.issym() or member.islnk():
                    raise DiarizationUnavailable(
                        f"Link member in weights archive: {member.name}")
                if not (member.isreg() or member.isdir()):
                    raise DiarizationUnavailable(
                        f"Unsupported member type in weights archive: "
                        f"{member.name}")
                target = (dest / member.name).resolve()
                if base not in target.parents and target != base:
                    raise DiarizationUnavailable(
                        f"Unsafe path in weights archive: {member.name}")
            tf.extractall(dest)


def download_weights(host: "WorkerHost", dest_dir: Path) -> Path:
    """Download + verify + atomically install the weights tarball.

    Posts the app's standard download_start/download_progress/download_done
    events (the GUI renders them like a model download). Cancel is honored
    per chunk. A partial or corrupt download never becomes dest_dir.
    """
    dest_dir = Path(dest_dir)
    dest_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp_root = Path(tempfile.mkdtemp(
        prefix=".tmp-diarization-", dir=str(dest_dir.parent)))
    host.post_event("download_start", repo_id=_DOWNLOAD_REPO_LABEL)
    host.post_event("log", message="Downloading speaker model: community-1")
    try:
        tar_path = tmp_root / "weights.tar.gz"
        digest = hashlib.sha256()
        request = urllib.request.Request(
            DIARIZATION_WEIGHTS_URL, headers={"User-Agent": "HebrewScribe"})
        with urllib.request.urlopen(request, timeout=60) as resp, \
                open(tar_path, "wb") as out:
            total = int(resp.headers.get("Content-Length") or 0)
            downloaded = 0
            last_post = 0.0
            while True:
                if host.cancel_requested:
                    raise DiarizationCancelled()
                chunk = resp.read(64 * 1024)
                if not chunk:
                    break
                out.write(chunk)
                digest.update(chunk)
                downloaded += len(chunk)
                now = time.monotonic()
                if now - last_post >= _PROGRESS_THROTTLE_SEC or downloaded == total:
                    last_post = now
                    host.post_event("download_progress",
                                    downloaded=downloaded, total=total,
                                    file_desc="Downloading speaker model")
        if DIARIZATION_WEIGHTS_SHA256:
            if digest.hexdigest().lower() != DIARIZATION_WEIGHTS_SHA256.lower():
                raise DiarizationUnavailable(
                    "Speaker model download failed checksum verification")
        else:
            logger.warning("Diarization weights checksum not pinned; "
                           "skipping verification")
        extract_dir = tmp_root / "extract"
        extract_dir.mkdir()
        _safe_extract_tar(tar_path, extract_dir)
        candidate = extract_dir / COMMUNITY1_DIR_NAME
        if not weights_complete(candidate):
            raise DiarizationUnavailable(
                "Speaker model archive is missing expected files")
        if dest_dir.exists():
            shutil.rmtree(dest_dir, ignore_errors=True)
        os.replace(candidate, dest_dir)
    except DiarizationCancelled:
        host.post_event("log", message="Speaker model download cancelled.")
        raise
    except DiarizationUnavailable:
        raise
    except Exception as exc:
        raise DiarizationUnavailable(
            f"Speaker model download failed: {exc}") from exc
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
        # Terminal event on EVERY exit (success, failure, cancel): the GUI's
        # download_done handler restores the status-bar device line, which
        # download_progress overwrote. Without this, a failed/cancelled
        # download leaves "Downloading model: N%" stuck for the whole run.
        host.post_event("download_done", repo_id=_DOWNLOAD_REPO_LABEL)
    host.post_event("log", message="Speaker model download complete.")
    return dest_dir


def ensure_weights(host: "WorkerHost") -> Path:
    """Return a complete local weights dir, downloading it on first use."""
    found = resolve_weights_dir()
    if found is not None:
        return found
    dest = get_models_dir() / "diarization" / COMMUNITY1_DIR_NAME
    return download_weights(host, dest)


# ---------------------------------------------------------------------------
# Pipeline load & inference
# ---------------------------------------------------------------------------

def load_diarization_pipeline(host: "WorkerHost",
                              device_preference: str = "auto"):
    """Load the community-1 pipeline from local weights, fully offline.

    device_preference: "auto" (MPS on Apple Silicon, else CUDA, else CPU)
    or "cpu". Every accelerator attempt falls back to CPU on failure.
    """
    _set_offline_env()
    try:
        from pyannote.audio import Pipeline
    except ImportError as exc:
        raise DiarizationUnavailable(
            "pyannote.audio is not installed "
            '(pip install -e ".[diarization]" from the project root)') from exc

    weights = ensure_weights(host)
    pipeline = Pipeline.from_pretrained(str(weights))
    if pipeline is None:
        raise DiarizationUnavailable(
            f"Could not load speaker model from {weights}")

    import torch

    def _rehome_cpu():
        # pyannote's Pipeline.to moves sub-models sequentially with no
        # rollback — after a failed move, force everything back to CPU.
        try:
            pipeline.to(torch.device("cpu"))
        except Exception:
            logger.warning("Could not re-home pipeline to CPU", exc_info=True)

    chosen = "cpu"
    if device_preference != "cpu":
        if sys.platform == "darwin" and torch.backends.mps.is_available():
            try:
                pipeline.to(torch.device("mps"))
                chosen = "mps"
            except Exception:
                logger.warning("MPS unavailable for diarization; using CPU",
                               exc_info=True)
                _rehome_cpu()
        elif torch.cuda.is_available():
            try:
                pipeline.to(torch.device("cuda"))
                chosen = "cuda"
            except Exception:
                logger.warning("CUDA unavailable for diarization; using CPU",
                               exc_info=True)
                _rehome_cpu()
    host.post_event(
        "log", message=f"Speaker identification ready (device: {chosen})")
    return pipeline


def _read_wave_file(path: Path):
    """Read a 16-bit PCM WAV via the stdlib. Returns (float32 mono ndarray, rate).

    Raises wave.Error / ValueError for shapes the stdlib reader can't handle
    (float/24-bit/ADPCM WAVs land here and go through ffmpeg instead).
    """
    import numpy as np

    with wave.open(str(path), "rb") as wf:
        nch = wf.getnchannels()
        width = wf.getsampwidth()
        rate = wf.getframerate()
        if width != 2:
            raise ValueError(f"Unsupported WAV sample width: {width}")
        frames = wf.readframes(wf.getnframes())
    arr = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if nch > 1:
        arr = arr.reshape(-1, nch).mean(axis=1)
    return arr, rate


def load_waveform(path: Path) -> dict:
    """Load audio as {"waveform": (1, N) float32 tensor, "sample_rate": 16000}.

    Must self-guarantee 16 kHz mono float32: preconvert_audio() passes .wav
    sources through unconverted, so arbitrary WAV shapes arrive here.
    Non-conforming audio is converted with the bundled ffmpeg.
    """
    import torch

    path = Path(path)
    arr = None
    try:
        arr, rate = _read_wave_file(path)
        if rate != 16000:
            arr = None  # wrong rate → ffmpeg
    except (wave.Error, ValueError, EOFError):
        arr = None
    if arr is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        try:
            completed = subprocess.run(
                [
                    _bundled_bin("ffmpeg"), "-y", "-i", str(path),
                    "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
                    tmp.name,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=600,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            if completed.returncode != 0:
                raise DiarizationUnavailable(
                    f"ffmpeg could not convert {path.name} for speaker "
                    "identification")
            arr, rate = _read_wave_file(Path(tmp.name))
            if rate != 16000:
                raise DiarizationUnavailable(
                    f"Converted audio for {path.name} has unexpected rate {rate}")
        except (OSError, subprocess.SubprocessError, wave.Error, ValueError,
                EOFError) as exc:
            raise DiarizationUnavailable(
                f"Could not read audio for speaker identification: "
                f"{path.name} ({exc})") from exc
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass
    waveform = torch.from_numpy(arr).unsqueeze(0)
    return {"waveform": waveform, "sample_rate": 16000}


def diarize_waveform(pipeline, waveform: dict, host: "WorkerHost",
                     display_path: Path, num_speakers: int = 0,
                     total_seconds: Optional[float] = None):
    """Run the pipeline with progress + cancel wired through the hook.

    Returns the exclusive (non-overlapping) diarization Annotation, which is
    purpose-built for mapping speakers onto STT timestamps.

    Progress contract with the GUI (reuses current_progress): row_status stays
    "Running" and processed_seconds/total_seconds are OMITTED so the batch-ETA
    audio accounting isn't double-counted by the diarization pass.
    """
    # Lazy import avoids a module-level cycle; worker is always loaded by the
    # time a diarization run exists.
    from hebrewscribe.worker import wait_if_paused

    started = time.time()
    state = {"last_post": 0.0}

    def _hook(step_name, step_artifact=None, *, file=None, total=None,
              completed=None):
        # Cancel check first — the ONLY cancellation path pyannote offers.
        # Clustering/reconstruction never call the hook, so a cancel there
        # takes effect only when the next hooked phase (or the caller) runs.
        if host.cancel_requested:
            raise DiarizationCancelled()
        # Honor Pause between hooked batches (same per-unit granularity as
        # transcription's per-segment wait). A cancel while paused raises
        # CancelledByUser from inside wait_if_paused, which _diarize_file
        # re-raises unchanged.
        wait_if_paused(host)
        if completed is None or not total:
            return
        span = _HOOK_PHASE_RANGES.get(step_name)
        if span is None:
            return
        percent = span[0] + (span[1] - span[0]) * (completed / total)
        now = time.monotonic()
        if now - state["last_post"] < _PROGRESS_THROTTLE_SEC and completed < total:
            return
        state["last_post"] = now
        host.post_event(
            "current_progress",
            path=str(display_path),
            phase="Identifying speakers…",
            percent=percent,
            elapsed=time.time() - started,
            eta=None,
            speed=None,
            row_status="Running",
        )

    result = pipeline(waveform, num_speakers=(num_speakers or None),
                      hook=_hook)
    return getattr(result, "exclusive_speaker_diarization", result)


# ---------------------------------------------------------------------------
# Pure assignment logic (no heavy imports — unit-testable without pyannote)
# ---------------------------------------------------------------------------

def annotation_to_turns(annotation) -> List[dict]:
    """Flatten a pyannote Annotation into sorted plain-dict speaker turns.

    The sole function that touches pyannote types; everything downstream
    operates on [{"start", "end", "speaker"}] dicts.
    """
    turns = []
    for segment, _track, label in annotation.itertracks(yield_label=True):
        turns.append({"start": float(segment.start),
                      "end": float(segment.end),
                      "speaker": str(label)})
    turns.sort(key=lambda t: (t["start"], t["end"]))
    return turns


def _speaker_at(turns: List[dict], starts: List[float], t: float) -> Optional[str]:
    """Speaker of the turn covering time t; nearest turn when t is in a gap."""
    if not turns:
        return None
    if not math.isfinite(t):
        # NaN/inf poison every comparison below (bisect included) and would
        # walk off the turn list — treat as unassignable.
        return None
    idx = bisect_right(starts, t) - 1
    if idx >= 0 and turns[idx]["end"] >= t:
        return turns[idx]["speaker"]
    # In a gap: nearest neighbor by boundary distance (ties → earlier turn).
    prev_dist = t - turns[idx]["end"] if idx >= 0 else float("inf")
    next_dist = turns[idx + 1]["start"] - t if idx + 1 < len(turns) else float("inf")
    if prev_dist <= next_dist:
        return turns[idx]["speaker"]
    return turns[idx + 1]["speaker"]


def _overlap_speaker(turns: List[dict], starts: List[float],
                     start: float, end: float) -> Optional[str]:
    """Speaker with max total overlap in [start, end]; ties → earliest-seen
    in the turn list; zero overlap → nearest turn to the midpoint."""
    if not turns:
        return None
    totals: "OrderedDict[str, float]" = OrderedDict()
    for turn in turns:
        if turn["end"] <= start:
            continue
        if turn["start"] >= end:
            break
        overlap = min(end, turn["end"]) - max(start, turn["start"])
        if overlap > 0:
            totals[turn["speaker"]] = totals.get(turn["speaker"], 0.0) + overlap
    if not totals:
        return _speaker_at(turns, starts, (start + end) / 2.0)
    return max(totals.items(), key=lambda kv: kv[1])[0]


def _words_usable(seg: dict) -> bool:
    """Sanity-check word timestamps before trusting them for assignment.

    Whisper word times can be grossly wrong right after silences; broken
    words demote the segment to the segment-level fallback.
    """
    words = seg.get("words")
    if not words:
        return False
    prev_mid = None
    for w in words:
        start, end = w.get("start"), w.get("end")
        if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            return False
        # NaN passes isinstance and every ordered comparison below evaluates
        # False, which would wave broken words straight through this gate.
        if not (math.isfinite(start) and math.isfinite(end)):
            return False
        if start > end:
            return False
        mid = (start + end) / 2.0
        if prev_mid is not None and mid < prev_mid:
            return False
        if mid < seg["start"] - 2.0 or mid > seg["end"] + 2.0:
            return False
        prev_mid = mid
    return True


def _absorb_short_flips(runs: List[dict], min_flip_seconds: float) -> List[dict]:
    """Absorb single-word runs shorter than the threshold into a neighbor.

    Shared-neighbor speaker wins; else the previous run; else the next.
    Repeats until stable so newly-adjacent same-speaker runs re-merge.
    """
    changed = True
    while changed and len(runs) > 1:
        changed = False
        for i, run in enumerate(runs):
            if len(run["words"]) != 1:
                continue
            w = run["words"][0]
            if (w["end"] - w["start"]) >= min_flip_seconds:
                continue
            prev_run = runs[i - 1] if i > 0 else None
            next_run = runs[i + 1] if i + 1 < len(runs) else None
            if prev_run is None and next_run is None:
                continue
            if prev_run is not None and next_run is not None \
                    and prev_run["speaker"] == next_run["speaker"]:
                prev_run["words"] += run["words"] + next_run["words"]
                del runs[i:i + 2]
            elif prev_run is not None:
                prev_run["words"].append(w)
                del runs[i]
            else:
                next_run["words"].insert(0, w)
                del runs[i]
            changed = True
            break
        # Re-merge adjacent same-speaker runs created by an absorption.
        merged: List[dict] = []
        for run in runs:
            if merged and merged[-1]["speaker"] == run["speaker"]:
                merged[-1]["words"] += run["words"]
            else:
                merged.append(run)
        runs = merged
    return runs


def assign_speakers_to_segments(segments: List[dict], turns: List[dict],
                                min_flip_seconds: float = MIN_FLIP_SECONDS
                                ) -> List[dict]:
    """Assign a speaker to every segment; split segments at speaker changes.

    Pure function: inputs are not mutated; output is a new segment list with
    ids renumbered 0..n-1. Word-level path (midpoint lookup on the exclusive
    turns) when the segment carries usable words; segment-level max-overlap
    fallback otherwise (always the case for the openai-whisper backend).
    """
    starts = [t["start"] for t in turns]
    out: List[dict] = []
    for seg in segments:
        if not _words_usable(seg):
            speaker = _overlap_speaker(turns, starts, seg["start"], seg["end"])
            new_seg = {"id": None, "start": seg["start"], "end": seg["end"],
                       "text": seg.get("text", ""), "speaker": speaker}
            if seg.get("words"):
                new_seg["words"] = [dict(w) for w in seg["words"]]
            out.append(new_seg)
            continue
        words = [dict(w) for w in seg["words"]]
        runs: List[dict] = []
        for w in words:
            speaker = _speaker_at(turns, starts, (w["start"] + w["end"]) / 2.0)
            if runs and runs[-1]["speaker"] == speaker:
                runs[-1]["words"].append(w)
            else:
                runs.append({"speaker": speaker, "words": [w]})
        runs = _absorb_short_flips(runs, min_flip_seconds)
        if len(runs) == 1:
            # No split: keep the original whisper text verbatim.
            out.append({"id": None, "start": seg["start"], "end": seg["end"],
                        "text": seg.get("text", ""),
                        "speaker": runs[0]["speaker"], "words": words})
            continue
        for i, run in enumerate(runs):
            sub_start = seg["start"] if i == 0 else run["words"][0]["start"]
            sub_end = (seg["end"] if i == len(runs) - 1
                       else runs[i + 1]["words"][0]["start"])
            text = "".join(w.get("word", "") for w in run["words"]).strip()
            out.append({"id": None, "start": sub_start, "end": sub_end,
                        "text": text, "speaker": run["speaker"],
                        "words": run["words"]})
    for i, seg in enumerate(out):
        seg["id"] = i
    return out


def build_speaker_map(segments: List[dict],
                      language: Optional[str]) -> "OrderedDict[str, str]":
    """Map raw speaker ids to display labels in first-appearance order.

    "he" → "דובר 1", anything else → "Speaker 1" (writers append the colon).
    Segments with speaker=None are skipped; empty map when nothing is labeled.
    """
    fmt = SPEAKER_LABEL_FORMATS.get((language or "").lower(),
                                    DEFAULT_SPEAKER_LABEL_FORMAT)
    mapping: "OrderedDict[str, str]" = OrderedDict()
    n = 0
    for seg in segments:
        speaker = seg.get("speaker")
        if speaker is None or speaker in mapping:
            continue
        n += 1
        mapping[speaker] = fmt.format(n=n)
    return mapping
