"""Pure utility functions: formatting, paths, config I/O, ffmpeg helpers."""

import json
import logging
import logging.handlers
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger("HebrewScribe")


def get_log_dir() -> Path:
    """Return the log directory (~/.hebrewscribe/logs/), creating it if needed."""
    log_dir = Path.home() / ".hebrewscribe" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def get_models_dir() -> Path:
    """Return the app-managed models directory (~/.hebrewscribe/models/),
    creating it if needed."""
    models_dir = Path.home() / ".hebrewscribe" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir


def bidi_name(name: str) -> str:
    """Wrap a filename for DISPLAY when its first strong character is RTL.

    Mixed Hebrew + digits + ".ext" names render scrambled under Tk's LTR
    paragraph direction; an RLE…PDF embedding gives the whole name an RTL
    base, matching how Explorer/Finder render it. Display-only — the result
    must never be used as a lookup key or written to disk.
    """
    for ch in name:
        o = ord(ch)
        if 0x0590 <= o <= 0x08FF or 0xFB1D <= o <= 0xFDFF or 0xFE70 <= o <= 0xFEFF:
            # RLE … PDF, then LRM: without the trailing mark, neutrals that
            # FOLLOW the name in a larger LTR string (" (42%)" in the banner)
            # got pulled into the RTL run and rendered scrambled.
            return chr(0x202B) + name + chr(0x202C) + chr(0x200E)
        if (0x41 <= o <= 0x5A) or (0x61 <= o <= 0x7A) or o >= 0xC0:
            return name  # first strong character is LTR
    return name


class _ResilientRotatingFileHandler(logging.handlers.RotatingFileHandler):
    """Rotation that survives another process holding the log file open.

    On Windows a second app instance keeps hebrewscribe.log locked, so the
    stock doRollover's rename raises PermissionError with the stream already
    closed — after which EVERY subsequent record is silently dropped. Keep
    appending unrotated instead; rotation is retried on a later emit once
    the other instance exits.
    """

    def doRollover(self):
        try:
            super().doRollover()
        except OSError:
            if self.stream is None or self.stream.closed:
                self.stream = self._open()


def setup_logging() -> None:
    """Configure the HebrewScribe logger with console + rotating file handlers.

    Console: INFO level.
    File:    DEBUG level, 2 MB per file, 3 backups (~8 MB max).

    Safe to call multiple times — skips if handlers are already attached.
    """
    root_logger = logging.getLogger("HebrewScribe")
    # Avoid duplicating handlers on repeated calls
    if any(isinstance(h, logging.handlers.RotatingFileHandler) for h in root_logger.handlers):
        return

    root_logger.setLevel(logging.DEBUG)

    # Console handler (INFO)
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
               for h in root_logger.handlers):
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
        ))
        root_logger.addHandler(console)

    # Rotating file handler (DEBUG)
    try:
        log_path = get_log_dir() / "hebrewscribe.log"
        file_handler = _ResilientRotatingFileHandler(
            log_path,
            maxBytes=2 * 1024 * 1024,  # 2 MB
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        root_logger.addHandler(file_handler)
    except Exception:
        # Non-fatal: if we can't write logs, the app still works
        root_logger.warning("Could not set up file logging", exc_info=True)

# Platform-native UI font family.
# Segoe UI on Windows, Helvetica Neue on macOS (SF Pro is not accessible via
# tkinter's Tcl bridge — .AppleSystemFont triggers CoreText warnings),
# generic sans-serif fallback on Linux.
if sys.platform == "win32":
    SYSTEM_FONT = "Segoe UI"
elif sys.platform == "darwin":
    SYSTEM_FONT = "Helvetica Neue"
else:
    SYSTEM_FONT = "sans-serif"


def _bundled_bin(name: str) -> str:
    """Return the full path to a bundled binary (ffmpeg/ffprobe) if running
    inside a PyInstaller bundle, otherwise return the bare name (relies on PATH).
    """
    if getattr(sys, "frozen", False):
        # PyInstaller sets sys._MEIPASS for --onefile, but for --onedir
        # the exe lives in the app folder.  We ship ffmpeg beside the exe.
        app_dir = Path(sys.executable).parent
        # On Windows binaries have .exe extension; on macOS/Linux they don't
        ext = ".exe" if sys.platform == "win32" else ""
        # macOS .app bundle: binary is inside Contents/MacOS, resources
        # are in Contents/Resources (or Contents/Frameworks).  Check the
        # Resources/ffmpeg subfolder first (where we put them in the spec).
        if sys.platform == "darwin":
            resources = app_dir.parent / "Resources"
            for search_dir in [resources / "ffmpeg", resources, app_dir]:
                candidate = search_dir / (name + ext)
                if candidate.exists():
                    return str(candidate)
        # Windows / Linux: check ffmpeg subfolder, then beside the exe
        for search_dir in [app_dir / "ffmpeg", app_dir]:
            candidate = search_dir / (name + ext)
            if candidate.exists():
                return str(candidate)
    return name  # fall back to system PATH


def _selftest_audio_path() -> Optional[Path]:
    """Return the path to the bundled self-test audio clip.

    When running from source, it lives in tests/fixtures/.
    When running from a PyInstaller bundle, it's in the app directory
    (or macOS Resources folder).
    """
    filename = "speech_6s.wav"
    # PyInstaller bundle
    if getattr(sys, "frozen", False):
        app_dir = Path(sys.executable).parent
        # sys._MEIPASS points to _internal/ in one-dir mode
        meipass = Path(getattr(sys, "_MEIPASS", app_dir))
        if sys.platform == "darwin":
            resources = app_dir.parent / "Resources"
            for d in [resources, meipass, app_dir]:
                p = d / filename
                if p.exists():
                    return p
        for d in [meipass, app_dir, app_dir / "tests" / "fixtures"]:
            p = d / filename
            if p.exists():
                return p
        return None
    # Running from source
    src_root = Path(__file__).parent.parent
    p = src_root / "tests" / "fixtures" / filename
    return p if p.exists() else None


AUDIO_EXTENSIONS = {
    ".mp3", ".wav", ".m4a", ".mp4", ".mpeg", ".mpga", ".webm", ".ogg", ".flac", ".aac", ".wma", ".mkv",
    ".opus", ".aiff",
}

LANGUAGE_OPTIONS = [
    "he", "en", "ru", "fr", "pl", "zh",
]

# Human-readable labels for the language dropdown.
# The value stored/used internally is still the ISO code.
LANGUAGE_DISPLAY = {
    "he":   "Hebrew",
    "en":   "English",
    "ru":   "Russian",
    "fr":   "French",
    "pl":   "Polish",
    "zh":   "Chinese",
}

OUTPUT_FORMATS = ["txt", "srt", "json"]


def load_json(path: Path, default):
    if not path.exists():
        return default
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        # Transient I/O failure (lock, permissions): the file may be perfectly
        # valid — leave it alone so a later read can succeed.
        logger.warning("Could not read %s", path, exc_info=True)
        return default
    except UnicodeDecodeError:
        raw = None  # not text — corrupt content, back it up below
    if raw is not None:
        try:
            return json.loads(raw)
        except ValueError:
            logger.warning("Failed to parse JSON from %s", path, exc_info=True)
    # Corrupt content: preserve the file — the next save_json would otherwise
    # overwrite it with defaults, silently destroying its contents.
    try:
        backup = path.with_name(path.name + ".corrupt.bak")
        path.replace(backup)
        logger.warning("Backed up unreadable JSON to %s", backup)
    except OSError:
        logger.warning("Could not back up %s", path, exc_info=True)
    return default


def save_json(path: Path, payload) -> None:
    try:
        # Write-then-rename so a crash mid-write can't truncate the file
        # (os.replace is atomic on both POSIX and Windows).
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)
    except Exception:
        logger.warning("Failed to save JSON to %s", path, exc_info=True)


def is_package_available(module_name: str) -> bool:
    try:
        __import__(module_name)
        return True
    except ImportError:
        return False
    except Exception:
        logger.debug("Unexpected error checking for package %s", module_name, exc_info=True)
        return False


def ffmpeg_available() -> bool:
    try:
        completed = subprocess.run(
            [_bundled_bin("ffmpeg"), "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return completed.returncode == 0
    except Exception:
        logger.debug("ffmpeg not found or not working", exc_info=True)
        return False


def probe_duration_seconds(path: Path) -> Optional[float]:
    try:
        completed = subprocess.run(
            [
                _bundled_bin("ffprobe"),
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=15,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if completed.returncode == 0:
            raw = completed.stdout.strip()
            if raw:
                value = float(raw)
                if value > 0:
                    return value
    except Exception:
        logger.debug("ffprobe failed for %s", path, exc_info=True)

    try:
        if path.suffix.lower() == ".wav":
            import wave
            with wave.open(str(path), "rb") as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                if rate > 0:
                    return frames / float(rate)
    except Exception:
        logger.debug("WAV duration fallback failed for %s", path, exc_info=True)

    return None


def _open_path(path: Path) -> None:
    """Open a file or folder using the OS default handler. Cross-platform."""
    import sys as _sys
    if _sys.platform == "win32":
        os.startfile(str(path))  # type: ignore[attr-defined]
    elif _sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])


def format_timestamp(seconds: float) -> str:
    if seconds is None:
        seconds = 0.0
    total_ms = int(round(seconds * 1000))
    hours = total_ms // 3_600_000
    minutes = (total_ms % 3_600_000) // 60_000
    secs = (total_ms % 60_000) // 1000
    ms = total_ms % 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def format_hms(seconds: Optional[float]) -> str:
    if seconds is None or seconds < 0:
        return "--:--"
    whole = int(seconds)
    h = whole // 3600
    m = (whole % 3600) // 60
    s = whole % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def progress_bar_text(percent: float, width: int = 14) -> str:
    """Render a compact unicode progress bar with sub-character resolution.

    Uses Unicode block elements for smooth 1/8-character granularity:
    \u2588 (full), \u2589-\u258f (7/8..1/8), \u2591 (empty).
    """
    if percent < 0:
        return ""
    percent = min(100.0, percent)
    # Total units = width * 8 (each character has 8 sub-steps)
    total_units = width * 8
    filled_units = int(round(percent / 100.0 * total_units))
    full_chars = filled_units // 8
    remainder = filled_units % 8
    # Sub-character blocks: index 0=empty, 1=\u258f(1/8), ... 7=\u2589(7/8)
    _PARTIALS = ["", "\u258f", "\u258e", "\u258d", "\u258c", "\u258b", "\u258a", "\u2589"]
    partial = _PARTIALS[remainder] if remainder > 0 else ""
    empty = width - full_chars - (1 if partial else 0)
    bar = "\u2588" * full_chars + partial + "\u2591" * max(0, empty)
    return f"{bar} {percent:.0f}%"


def safe_stem(path: Path) -> str:
    stem = path.stem.strip()
    if not stem or stem.startswith("."):
        return "transcript"
    return stem


def unique_stem(source_file: Path, out_dir: Path, formats: List[str], used_stems: set) -> str:
    """Return a stem guaranteed not to collide with existing output files or earlier batch items."""
    base = safe_stem(source_file)

    def _collides(candidate: str) -> bool:
        if candidate.lower() in used_stems:
            return True
        for fmt in formats:
            if (out_dir / f"{candidate}.{fmt}").exists():
                return True
        return False

    if not _collides(base):
        used_stems.add(base.lower())
        return base

    parent_name = source_file.parent.name.strip()
    if parent_name:
        candidate = f"{base}_{parent_name}"
        if not _collides(candidate):
            used_stems.add(candidate.lower())
            return candidate

    counter = 2
    while True:
        suffix = f"{parent_name}_{counter}" if parent_name else str(counter)
        candidate = f"{base}_{suffix}"
        if not _collides(candidate):
            used_stems.add(candidate.lower())
            return candidate
        counter += 1
