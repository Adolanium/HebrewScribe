import logging
import os
import queue
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from hebrewscribe.utils import (
    AUDIO_EXTENSIONS, LANGUAGE_OPTIONS, LANGUAGE_DISPLAY, OUTPUT_FORMATS,
    SYSTEM_FONT, bidi_name,
    load_json, save_json, is_package_available,
    ffmpeg_available, probe_duration_seconds, _open_path,
    format_hms, get_log_dir, setup_logging,
)
from hebrewscribe.models import (
    OPENAI_MODELS, FASTER_WHISPER_MODELS, RECOMMENDED_MODEL,
    detect_openai_cached_models,
    detect_faster_whisper_local_models,
    friendly_label, model_lang,
    get_primary_model_cache_dir,
)
from hebrewscribe.worker import (
    CancelledByUser, RunOptions, SelfTestResult, SELFTEST_CHECK_COUNT,
    run_selftest, run_transcription_worker,
)
from hebrewscribe.power import SleepInhibitor
from hebrewscribe.theme import install_modern_theme, Colors as C, _fs, PAD_CARD_INNER_X, PAD_CARD_INNER_Y, PAD_TOOLBAR_GAP, PAD_SECTION_BELOW
from hebrewscribe.icons import load_icon
from hebrewscribe.widgets import ToolTip, TaskbarProgress

_HAS_SOUNDDEVICE = False
try:
    import sounddevice as _sd  # noqa: F401 — availability check only
    _HAS_SOUNDDEVICE = True
except (ImportError, OSError):
    # OSError: sounddevice imports but raises when the PortAudio binary
    # fails to load (e.g. a broken frozen bundle) — degrade to no recording
    # instead of killing the whole windowed app at startup.
    pass

# Disable pyannote's default-on telemetry before anything can import it
# (pyannote 4.x posts to otel.pyannote.ai unless this is set). setdefault:
# an explicit user opt-in via the env var is respected.
os.environ.setdefault("PYANNOTE_METRICS_ENABLED", "false")

# Cheap metadata probe, deliberately NOT a real import: pyannote.audio pulls
# torch, which would add seconds to every launch of the bundled app. A
# present-but-broken install surfaces at batch start via the worker's
# degrade-to-no-labels path.
import importlib.util as _importlib_util
try:
    # find_spec("pyannote.audio") imports the parent package, which raises
    # ModuleNotFoundError when pyannote is absent entirely.
    _HAS_PYANNOTE = _importlib_util.find_spec("pyannote.audio") is not None
except (ImportError, ValueError):
    _HAS_PYANNOTE = False

APP_TITLE = "HebrewScribe"

logger = logging.getLogger(APP_TITLE)
CONFIG_PATH = Path.home() / ".hebrewscribe_config.json"
_OLD_CONFIG_PATH = Path.home() / ".whisper_hebrew_transcriber_config.json"
# Migrate config from old name (zero-friction rename)
if not CONFIG_PATH.exists() and _OLD_CONFIG_PATH.exists():
    try:
        import shutil
        shutil.copy2(_OLD_CONFIG_PATH, CONFIG_PATH)
        logging.getLogger("HebrewScribe").info(
            "Migrated config from %s to %s", _OLD_CONFIG_PATH, CONFIG_PATH
        )
    except OSError:
        pass  # Fall through to defaults


# ---------------------------------------------------------------------------
# Structured log data model
# ---------------------------------------------------------------------------

@dataclass
class LogEntry:
    """A single log event with semantic metadata for structured display."""
    timestamp: float
    time_str: str
    message: str
    level: str = "info"          # "debug" | "info" | "warning" | "error"
    phase: str = "user"          # "setup" | "file" | "results" | "user" | "system"
    run_id: int = 0              # which run (1-based)
    file_path: Optional[str] = None  # associated file, if any
    raw_only: bool = False       # True = show only in Raw view


# Messages matching these patterns are suppressed from Activity view
_SYSTEM_NOISE_PATTERNS = (
    "Model list refreshed for backend:",
)


def _classify_log_phase(message: str, current_phase: str) -> tuple:
    """Classify a log message into (phase, raw_only) based on content.

    Returns a tuple of (phase_str, raw_only_bool).
    """
    low = message.lower()

    # System noise — suppress from Activity
    for pattern in _SYSTEM_NOISE_PATTERNS:
        if pattern.lower() in low:
            return "system", True

    # Tracebacks — raw-only (too verbose for Activity view)
    if low.startswith("traceback (most recent"):
        return current_phase, True

    # Setup-phase messages
    if (low.startswith("loading ") or low.startswith("device:") or
            low.startswith("model loaded") or low.startswith("model already cached") or
            low.startswith("downloading model") or low.startswith("download complete")):
        return "setup", False

    # File-phase messages (including recovery sub-events)
    if (low.startswith("transcribing:") or low.startswith("pre-converted") or
            low.startswith("output renamed") or low.startswith("done:") or
            low.startswith("done (after recovery)") or
            low.startswith("error in ") or low.startswith("retrying:") or
            low.startswith("detected model corruption") or
            low.startswith("model cache appears corrupt") or
            low.startswith("model reloaded") or
            low.startswith("recovery failed")):
        return "file", False

    # Results-phase messages
    if low.startswith("finished.") or low.startswith("cancelled.") or low.startswith("stopped"):
        return "results", False

    return current_phase, False


class TranscriberApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)

        # Set window icon (title bar + taskbar)
        self._icon_path: Optional[str] = None
        try:
            _base = getattr(sys, '_MEIPASS', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            _ico = os.path.join(_base, 'icon.ico')
            if os.path.isfile(_ico):
                self.iconbitmap(_ico)
                self._icon_path = _ico
        except Exception:
            logger.debug("Failed to set window icon", exc_info=True)

        # Compute DPI scale factor relative to the standard 96 DPI baseline.
        # On a 150% scaled Windows display this yields 1.5, on macOS Retina
        # typically 2.0 (but tkinter already handles Retina), on standard
        # displays 1.0.  Used to scale hardcoded pixel values (geometry,
        # column widths) so the app looks correct at any system scaling.
        self._dpi_scale: float = self.winfo_fpixels("1i") / 96.0
        # Clamp to reasonable range to avoid pathological values.
        # Never shrink below the 96-DPI design baseline: Aqua Tk reports
        # ~72 px/inch (points), which would otherwise scale the whole UI
        # down ~25% on every macOS launch.
        if self._dpi_scale < 1.0:
            self._dpi_scale = 1.0
        elif self._dpi_scale > 4.0:
            self._dpi_scale = 4.0

        def _s(px: int) -> int:
            """Scale a pixel value by the DPI factor."""
            return round(px * self._dpi_scale)

        self._s = _s
        # Clamp to the usable screen: on 1366x768 laptops (or 150% scaling on
        # 1920x1080) the scaled 1260x860 design size would exceed the display,
        # and a minsize larger than the screen makes the window unfittable.
        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()
        win_w = min(_s(1260), screen_w - _s(40))
        win_h = min(_s(860), screen_h - _s(80))
        self.geometry(f"{win_w}x{win_h}")
        self.minsize(min(_s(1100), win_w), min(_s(640), win_h))

        # Apply modern flat theme before building any widgets
        try:
            install_modern_theme(dpi_scale=self._dpi_scale)
        except Exception:
            logger.debug("Modern theme failed, using default", exc_info=True)

        # Set base window background to match theme
        self.configure(background=C.BG_PRIMARY)

        self.config_data = load_json(CONFIG_PATH, {
            "extra_model_roots": [],
            "last_output_dir": "",
            "last_backend": "",
            "last_language": "he",
            "last_task": "transcribe",
            "last_device": "auto",
            "last_compute_type": "auto",
            "last_beam_size": 5,
            "last_vad_filter": True,
            "last_condition_on_previous_text": False,
            "last_batch_size": 0,
            "last_speed_preset": "quality",
            "last_formats": ["txt", "srt"],
            "experimental_recording": False,
            "last_diarize": False,
            "last_num_speakers": 0,
            "diarize_device": "auto",
        })

        self.worker_thread: Optional[threading.Thread] = None
        self.event_queue: "queue.Queue[Tuple[str, dict]]" = queue.Queue()
        self.pause_event = threading.Event()
        self.pause_event.set()
        self.stop_requested = False
        self.cancel_requested = False

        # Live recording state
        self._recorder = None  # LiveRecorder instance (created on first use)

        self.file_paths: List[str] = []
        self._queue_lock = threading.Lock()  # guards file_paths access between GUI and worker
        # Plain-Python mirror of each row's display status (path -> status),
        # maintained on the GUI thread by set_tree_row/refresh_file_tree.
        # The worker's next_file() reads THIS under _queue_lock instead of
        # the Treeview: a Tcl call from the worker could hit a mid-rebuild
        # tree (TclError kills the batch) or deadlock against a GUI thread
        # waiting on the same lock.
        self.file_status: Dict[str, str] = {}
        self.model_map: Dict[str, str] = {}
        self.file_items: Dict[str, str] = {}
        self.file_errors: Dict[str, str] = {}  # path -> error message for failed files
        self.file_durations: Dict[str, Optional[float]] = {}  # path -> audio seconds
        self._file_wall_starts: Dict[str, float] = {}  # path -> wall-clock start of transcription
        self._job_wall_start: float = 0.0  # wall-clock start of current job

        self.backend_var = tk.StringVar(value="")
        self.model_var = tk.StringVar(value="")
        self.output_dir_var = tk.StringVar(value=self.config_data.get("last_output_dir", ""))
        _saved_lang = self.config_data.get("last_language", "he")
        if _saved_lang not in LANGUAGE_OPTIONS:
            _saved_lang = "he"
        self.language_var = tk.StringVar(value=_saved_lang)
        self.task_var = tk.StringVar(value=self.config_data.get("last_task", "transcribe"))
        self.device_var = tk.StringVar(value=self.config_data.get("last_device", "auto"))
        self.compute_type_var = tk.StringVar(value=self.config_data.get("last_compute_type", "auto"))
        self.beam_size_var = tk.StringVar(value=str(self.config_data.get("last_beam_size", 5)))
        self.vad_var = tk.BooleanVar(value=self.config_data.get("last_vad_filter", True))
        self.condition_on_previous_text_var = tk.BooleanVar(
            value=self.config_data.get("last_condition_on_previous_text", False))
        self.batch_size_var = tk.StringVar(
            value=str(self.config_data.get("last_batch_size", 0)))
        self.speed_preset_var = tk.StringVar(
            value=self.config_data.get("last_speed_preset", "quality"))
        self.show_all_models_var = tk.BooleanVar(value=False)

        # Speaker diarization controls. The checkbox state is forced off when
        # the engine is absent so readiness can never block Start over an
        # invisible control. num_speakers is a StringVar (like beam/batch) so
        # mid-edit garbage can't raise inside Tk callbacks.
        self.diarize_var = tk.BooleanVar(value=self._diarize_enabled_initial(
            self.config_data.get("last_diarize", False), _HAS_PYANNOTE))
        self.num_speakers_var = tk.StringVar(
            value=str(self.config_data.get("last_num_speakers", 0)))
        _saved_ddev = self.config_data.get("diarize_device", "auto")
        if _saved_ddev not in ("auto", "cpu"):
            _saved_ddev = "auto"
        self.diarize_device_var = tk.StringVar(value=_saved_ddev)

        self.job_status_var = tk.StringVar(value="Ready.")
        self.current_file_var = tk.StringVar(value="No file running")
        self.current_phase_var = tk.StringVar(value="Idle")  # internal state, no visible label
        self.overall_summary_var = tk.StringVar(value="0 files queued")

        # Breathing line vars (single composed line per card)
        self.file_breathing_var = tk.StringVar(value="")
        self.batch_identity_var = tk.StringVar(value="0 files")
        self.batch_breathing_var = tk.StringVar(value="")

        # Internal state for compose functions (no visible labels)
        self._current_lang: str = ""
        self._last_file_speed: Optional[float] = None
        self._last_file_eta: Optional[str] = None
        self._last_file_elapsed_s: float = 0.0
        self._last_batch_speed: Optional[float] = None
        self._last_batch_eta_s: Optional[float] = None
        self._last_job_wall: float = 0.0
        self._last_finished_file_path: Optional[str] = None

        self.format_vars = {fmt: tk.BooleanVar(value=fmt in self.config_data.get("last_formats", ["txt", "srt"])) for fmt in OUTPUT_FORMATS}
        self.prevent_sleep_var = tk.BooleanVar(value=self.config_data.get("prevent_sleep", True))
        self._experimental_recording_var = tk.BooleanVar(
            value=self.config_data.get("experimental_recording", False))
        self._sleep_inhibitor = SleepInhibitor()
        self.readiness_var = tk.StringVar(value="")
        self.advanced_visible = tk.BooleanVar(value=self.config_data.get("advanced_visible", False))

        # Batch-level ETA tracking
        self._batch_audio_done: float = 0.0    # cumulative audio seconds completed
        self._batch_audio_total: float = 0.0   # total audio seconds in queue
        self._batch_wall_start: float = 0.0    # wall-clock start of batch

        # Session-level accumulators (across multiple runs)
        self._session_audio_total: float = 0.0
        self._session_wall_total: float = 0.0
        self._session_done_count: int = 0
        self._session_failed_count: int = 0
        self._session_run_count: int = 0

        # Elapsed ticker (1-second independent refresh)
        self._elapsed_ticker_id: Optional[str] = None
        self._vram_ticker_id: Optional[str] = None
        self._current_file_path: Optional[str] = None  # path of file being transcribed

        # Live segment & word counters (internal, shown in tooltip)
        self._segments_done: int = 0
        self._words_transcribed: int = 0

        # GPU / device info
        self.device_info_var = tk.StringVar(value="")

        # Structured log state
        self._log_entries: List[LogEntry] = []
        self._log_run_id: int = 0           # incremented each start_transcription
        self._log_current_phase: str = "user"  # tracks phase context for ambiguous messages
        self._log_current_file: Optional[str] = None  # file_path of file being transcribed
        # Activity tree node tracking (populated by the _activity_* handlers)
        self._activity_setup_node: str = ""
        self._activity_files_node: str = ""
        self._activity_file_nodes: Dict[str, str] = {}
        self._activity_results_node: str = ""
        self._activity_files_done: int = 0
        self._activity_files_failed: int = 0
        self._activity_files_total: int = 0
        self._activity_log_view: str = "activity"  # "activity" or "raw"

        self._build_ui()
        self._apply_speed_preset()  # set initial preset description
        self._populate_backends()
        # Build the model list AND restore the saved per-language model. Calling
        # _on_language_changed (rather than bare refresh_models) applies the
        # saved-pref > recommended > affinity ladder at startup, so a returning
        # user's last model choice is honoured instead of silently reset to the
        # first-sorted entry.
        self._on_language_changed()
        self._wire_readiness_checks()
        self._check_readiness()
        self.after(120, self.process_event_queue)

        # Taskbar progress (Windows only, degrades silently)
        self._taskbar = TaskbarProgress()
        self.after(500, lambda: self._taskbar.bind(self))

        # Detect GPU/device at startup (deferred to avoid slowing init)
        self.after(600, self._detect_device_info)

        # Keyboard shortcuts
        self.bind_all("<Control-Return>", self._on_ctrl_enter)
        self.bind_all("<Escape>", self._on_escape)

        # Exit confirmation when job is running
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        if sys.platform == "darwin":
            # Cmd+Q / app-menu Quit terminates Tk directly without firing
            # WM_DELETE_WINDOW — route it through the same close path so the
            # running-job confirm, config save, and caffeinate release happen.
            try:
                self.createcommand("::tk::mac::Quit", self._on_close)
            except Exception:
                logger.debug("Could not register mac Quit handler", exc_info=True)

        # Auto-refresh model list when window regains focus (debounced)
        self._last_focus_refresh: float = 0.0
        self.bind("<FocusIn>", self._on_focus_in)

    # --- Font constants for typographic hierarchy ---
    FONT_SECTION = (SYSTEM_FONT, _fs(13), "bold")
    FONT_LABEL = (SYSTEM_FONT, _fs(10))
    FONT_VALUE = (SYSTEM_FONT, _fs(10))
    FONT_SMALL = (SYSTEM_FONT, _fs(9))
    FONT_BUTTON = (SYSTEM_FONT, _fs(10))
    FONT_START = (SYSTEM_FONT, _fs(11), "bold")

    # --- Color system: semantic palette from Colors class ---
    CLR_IDLE_BG = C.STATUS_IDLE_BG
    CLR_IDLE_FG = C.STATUS_IDLE_FG
    CLR_RUNNING_BG = C.STATUS_RUNNING_BG
    CLR_RUNNING_FG = C.STATUS_RUNNING_FG
    CLR_RUNNING_ACCENT = C.ACCENT
    CLR_PAUSED_BG = C.STATUS_PAUSED_BG
    CLR_PAUSED_FG = C.STATUS_PAUSED_FG
    CLR_SUCCESS_BG = C.STATUS_SUCCESS_BG
    CLR_SUCCESS_FG = C.STATUS_SUCCESS_FG
    CLR_ERROR_BG = C.STATUS_ERROR_BG
    CLR_ERROR_FG = C.STATUS_ERROR_FG
    CLR_WARNING_FG = C.STATUS_WARNING_FG

    # --- Global exception handling ---
    _crash_dialog_shown = False

    def report_callback_exception(self, exc_type, exc_value, exc_tb):
        """Override Tk's default handler to log + show a crash dialog."""
        import traceback as _tb
        logger.critical(
            "Unhandled exception in UI callback:\n%s",
            "".join(_tb.format_exception(exc_type, exc_value, exc_tb)),
        )
        self._show_crash_dialog(exc_type, exc_value, exc_tb)

    def _show_crash_dialog(self, exc_type, exc_value, exc_tb) -> None:
        """Show a modal dialog with the traceback and action buttons."""
        import traceback as _tb
        if self._crash_dialog_shown:
            return  # prevent recursive crash dialogs
        self._crash_dialog_shown = True
        try:
            tb_text = "".join(_tb.format_exception(exc_type, exc_value, exc_tb))
            dlg = tk.Toplevel(self)
            dlg.title(f"{APP_TITLE} \u2014 Unexpected Error")
            dlg.resizable(True, True)
            dlg.transient(self)
            dlg.grab_set()
            if self._icon_path:
                dlg.iconbitmap(self._icon_path)

            _s = self._s
            frame = ttk.Frame(dlg, padding=_s(20))
            frame.pack(fill="both", expand=True)

            ttk.Label(
                frame, text="Something went wrong.",
                font=(SYSTEM_FONT, _fs(13), "bold"),
            ).pack(anchor="w")
            ttk.Label(
                frame,
                text="The error has been logged. You can copy the details below.",
                font=self.FONT_SMALL, foreground=C.TEXT_TERTIARY,
            ).pack(anchor="w", pady=(_s(4), _s(12)))

            # Scrollable traceback
            text_frame = ttk.Frame(frame)
            text_frame.pack(fill="both", expand=True)
            scrollbar = ttk.Scrollbar(text_frame, orient="vertical")
            scrollbar.pack(side="right", fill="y")
            text_widget = tk.Text(
                text_frame, wrap="word", font=("Consolas" if sys.platform == "win32" else "Menlo", 9),
                bg=C.TOOLTIP_BG, fg=C.TOOLTIP_FG, insertbackground=C.TOOLTIP_FG,
                relief="flat", borderwidth=0,
                yscrollcommand=scrollbar.set,
                height=16, width=80,
            )
            text_widget.pack(fill="both", expand=True)
            scrollbar.config(command=text_widget.yview)
            text_widget.insert("1.0", tb_text)
            text_widget.config(state="disabled")

            # Action buttons
            btn_frame = ttk.Frame(frame)
            btn_frame.pack(fill="x", pady=(_s(12), 0))

            def _copy():
                self.clipboard_clear()
                self.clipboard_append(tb_text)

            def _open_logs():
                try:
                    _open_path(get_log_dir())
                except Exception:
                    pass

            copy_btn = self._make_button(btn_frame, text="Copy to clipboard", command=_copy)
            copy_btn.pack(side="left", padx=(0, _s(8)))
            log_btn = self._make_button(btn_frame, text="Open log folder", command=_open_logs)
            log_btn.pack(side="left")
            close_btn = self._make_button(btn_frame, text="Close", command=dlg.destroy)
            close_btn.pack(side="right")

            def _on_dismiss(_event=None):
                self._crash_dialog_shown = False
                dlg.destroy()

            # Route every dismissal path (Escape, window close, Close button)
            # through _on_dismiss so the guard flag is always reset — otherwise
            # dismissing with Escape suppresses all later crash dialogs.
            dlg.bind("<Escape>", _on_dismiss)
            dlg.protocol("WM_DELETE_WINDOW", _on_dismiss)
            close_btn.config(command=_on_dismiss)

            # Center on parent
            dlg.update_idletasks()
            x = self.winfo_x() + (self.winfo_width() - dlg.winfo_width()) // 2
            y = self.winfo_y() + (self.winfo_height() - dlg.winfo_height()) // 2
            dlg.geometry(f"+{max(0, x)}+{max(0, y)}")
        except Exception:
            # If the crash dialog itself crashes, reset the flag and log
            self._crash_dialog_shown = False
            logger.critical("Failed to show crash dialog", exc_info=True)

    # --- Button style definitions ---
    _BTN_STYLES = {
        "primary": {
            "bg": C.ACCENT, "fg": "white",
            "hover_bg": C.ACCENT_HOVER, "press_bg": C.ACCENT_PRESS,
            "disabled_bg": C.BTN_PRIMARY_DISABLED_BG, "disabled_fg": C.BTN_PRIMARY_DISABLED_FG,
        },
        "secondary": {
            "bg": C.BTN_SECONDARY_BG, "fg": C.TEXT_PRIMARY,
            "hover_bg": C.BTN_SECONDARY_HOVER, "press_bg": C.BTN_SECONDARY_PRESS,
            "disabled_bg": C.BTN_DISABLED_BG, "disabled_fg": C.BTN_DISABLED_FG,
        },
        "destructive": {
            "bg": C.BG_PRIMARY, "fg": C.BTN_DESTRUCTIVE_FG,
            "hover_bg": C.BTN_DESTRUCTIVE_HOVER, "press_bg": C.BTN_DESTRUCTIVE_PRESS,
            "disabled_bg": C.BTN_DISABLED_BG, "disabled_fg": C.BTN_DISABLED_FG,
        },
    }

    def _make_button(self, parent: tk.Widget, text: str,
                     command=None, variant: str = "secondary",
                     font=None, icon_name: str = "", **kwargs) -> tk.Button:
        """Create a styled button with hover feedback, optionally with an icon.

        Variants: 'primary' (filled accent), 'secondary' (neutral),
        'destructive' (red text).

        If icon_name is provided and the corresponding PNG exists, the button
        shows icon + text (compound="left"). If the icon is missing, falls
        back to text-only (zero regression).
        """
        colors = self._BTN_STYLES.get(variant, self._BTN_STYLES["secondary"])
        defaults = dict(
            background=colors["bg"], foreground=colors["fg"],
            activebackground=colors["press_bg"],
            activeforeground=colors["fg"],
            disabledforeground=colors["disabled_fg"],
            font=font or self.FONT_BUTTON,
            relief="flat", borderwidth=0,
            padx=12, pady=5, cursor="hand2",
            takefocus=0,
        )
        defaults.update(kwargs)
        btn = tk.Button(parent, text=text, command=command, **defaults)
        btn._btn_colors = colors  # type: ignore[attr-defined]

        # Load icon if requested
        if icon_name:
            img = load_icon(icon_name, self._dpi_scale)
            if img is not None:
                compound = "left" if text else "center"
                btn.configure(image=img, compound=compound)
                if text:
                    btn.configure(padx=8)  # tighter padding with icon
                btn._icon_ref = img  # type: ignore[attr-defined]  # prevent GC

        def on_enter(e):
            if str(btn.cget("state")) != "disabled":
                btn.configure(background=colors["hover_bg"])

        def on_leave(e):
            if str(btn.cget("state")) != "disabled":
                btn.configure(background=colors["bg"])

        btn.bind("<Enter>", on_enter)
        btn.bind("<Leave>", on_leave)

        return btn

    def _set_button_enabled(self, btn: tk.Button, enabled: bool) -> None:
        """Enable/disable a styled button, updating colors to match."""
        colors = getattr(btn, "_btn_colors", self._BTN_STYLES["secondary"])
        if enabled:
            btn.configure(state="normal", background=colors["bg"],
                         foreground=colors["fg"])
        else:
            btn.configure(state="disabled", background=colors["disabled_bg"],
                         foreground=colors["disabled_fg"])

    def _build_ui(self) -> None:
        # === STATUS BAR: thin bar at the very bottom ===
        status_border = ttk.Frame(self)
        status_border.pack(side="bottom", fill="x")
        ttk.Separator(status_border, orient="horizontal").pack(fill="x")
        self._status_bar = ttk.Frame(status_border, padding=(10, 4))
        self._status_bar.pack(fill="x")
        # RIGHT: About — always rightmost, packed first to claim the edge
        about_label = ttk.Label(self._status_bar, text="About", font=self.FONT_LABEL,
                                foreground=C.TEXT_TERTIARY, cursor="hand2")
        about_label.pack(side="right")
        about_label.bind("<Button-1>", lambda e: self._show_about_dialog())
        about_label.bind("<Enter>", lambda e: about_label.configure(foreground=C.ACCENT))
        about_label.bind("<Leave>", lambda e: about_label.configure(foreground=C.TEXT_TERTIARY))
        # LEFT: device info, readiness issues, file count — all left-anchored
        self._statusbar_left = ttk.Label(self._status_bar, textvariable=self.device_info_var,
                                          font=self.FONT_LABEL, foreground=C.TEXT_SECONDARY)
        self._statusbar_left.pack(side="left")
        # Neutral by default: pre-flight hints ("No audio files added") are
        # not errors; _check_readiness switches to error red only for
        # genuinely invalid input.
        self.readiness_label = ttk.Label(self._status_bar, textvariable=self.readiness_var,
                                          foreground=C.TEXT_TERTIARY, font=self.FONT_LABEL)
        self.readiness_label.pack(side="left", padx=(16, 0))
        self._statusbar_right = ttk.Label(self._status_bar, text="",
                                           font=self.FONT_LABEL, foreground=C.TEXT_SECONDARY)
        self._statusbar_right.pack(side="left", padx=(16, 0))

        # === ACTION BAR (pinned above status bar) ===
        # Action bar shadow (3-frame gradient for depth)
        for shadow_color in (C.SHADOW_LIGHT, C.SHADOW_MED, C.SHADOW_DARK):
            tk.Frame(self, background=shadow_color, height=1).pack(side="bottom", fill="x")
        action_bar = ttk.Frame(self, padding=(16, 10))
        action_bar.pack(side="bottom", fill="x")

        self.start_button = self._make_button(
            action_bar, text="  Start transcription  ",
            command=self.start_transcription, variant="primary",
            font=self.FONT_START, pady=10, padx=20,
        )
        self.start_button.pack(side="left", padx=(0, 8))

        # Record button — live mic transcription (experimental, opt-in via Advanced Settings)
        if _HAS_SOUNDDEVICE:
            self.record_button = self._make_button(
                action_bar, text="  Record  ",
                command=self._toggle_recording, variant="secondary",
                font=self.FONT_START, pady=10, padx=16,
            )
            if self._experimental_recording_var.get():
                self.record_button.pack(side="left", padx=(0, 8))
        else:
            self.record_button = None

        # Run controls — hidden when idle, shown when running
        self._run_controls = ttk.Frame(action_bar)
        # NOT packed initially — shown by _show_run_controls()
        self.pause_button = self._make_button(self._run_controls, text="Pause", command=self.pause_processing)
        self.pause_button.pack(side="left")
        self.resume_button = self._make_button(self._run_controls, text="Resume", command=self.resume_processing)
        self.resume_button.pack(side="left", padx=(6, 0))
        self.stop_button = self._make_button(self._run_controls, text="Stop after current", command=self.request_stop_after_current)
        self.stop_button.pack(side="left", padx=(6, 0))

        # Cancel button — separated from safe controls with extra gap
        self.cancel_button = self._make_button(action_bar, text="Cancel", command=self.request_cancel_now, variant="destructive")
        # NOT packed initially — shown alongside _run_controls when running

        # Post-completion controls — hidden until batch finishes
        self._done_controls = ttk.Frame(action_bar)
        # NOT packed initially — shown after batch completes
        self._retry_btn = self._make_button(self._done_controls, text="Retry failed", command=self._retry_failed)
        self._retry_btn.pack(side="left", padx=(0, 6))

        self._make_button(action_bar, text="Open output folder", command=self.open_output_folder).pack(side="right")

        # === MAIN CONTENT ===
        self._root_frame = ttk.Frame(self, padding=(16, 12, 16, 0))
        self._root_frame.pack(fill="both", expand=True)
        root = self._root_frame

        # --- Footer: settings bar pinned to the bottom of the content area ---
        # Packed first (side="bottom") so the PanedWindow can expand to fill
        # everything above it.  Settings are secondary to the queue/run panels.
        self._footer_frame = ttk.Frame(root)
        self._footer_frame.pack(side="bottom", fill="x")

        # --- Header: banner only (state indicator, shown during transcription) ---
        self._header_frame = ttk.Frame(root)
        self._header_frame.pack(fill="x")

        # --- Settings: single compact bar, lives inside the footer ---
        self._settings_container = ttk.Frame(self._footer_frame)
        self._settings_container.pack(fill="x")

        _CARD_BG = C.BG_CARD
        self._CARD_BG = _CARD_BG
        settings_card = tk.Frame(self._settings_container, background=_CARD_BG,
                                 highlightbackground=C.BORDER_CARD, highlightthickness=1)
        settings_card.pack(fill="x", pady=(6, 0))
        settings_inner = tk.Frame(settings_card, background=_CARD_BG, padx=4, pady=8)
        settings_inner.pack(fill="x")

        # Single row: Language | model info | Output folder + Browse | Advanced
        settings_row = tk.Frame(settings_inner, background=_CARD_BG)
        settings_row.pack(fill="x")

        # -- Language --
        tk.Label(settings_row, text="Language", font=self.FONT_LABEL,
                 background=_CARD_BG, foreground=C.TEXT_PRIMARY).grid(row=0, column=0, sticky="w", padx=(0, 6))
        self._lang_display_var = tk.StringVar(
            value=LANGUAGE_DISPLAY.get(self.language_var.get(), self.language_var.get()))
        lang_display_values = [LANGUAGE_DISPLAY.get(c, c) for c in LANGUAGE_OPTIONS]
        self.language_combo = ttk.Combobox(
            settings_row, textvariable=self._lang_display_var, state="readonly",
            values=lang_display_values, width=14)
        # ipady lifts entry/combobox to the icon buttons' ~35px height so the
        # options row sits on a single control height.
        self.language_combo.grid(row=0, column=1, sticky="w", ipady=self._s(4))
        self.language_combo.bind("<<ComboboxSelected>>", lambda e: self._on_language_display_changed())

        # -- Separator --
        sep = tk.Frame(settings_row, background=C.BORDER_SEPARATOR, width=1, height=20)
        sep.grid(row=0, column=2, padx=(12, 12), sticky="ns")

        # -- Output folder --
        tk.Label(settings_row, text="Output", font=self.FONT_LABEL,
                 background=_CARD_BG, foreground=C.TEXT_PRIMARY).grid(row=0, column=3, sticky="w", padx=(0, 6))
        ttk.Entry(settings_row, textvariable=self.output_dir_var).grid(row=0, column=4, sticky="we", ipady=self._s(4))
        self._browse_btn = self._make_button(settings_row, text="Browse", command=self.choose_output_dir, icon_name="browse")
        self._browse_btn.grid(row=0, column=5, padx=(6, 0))

        # -- Identify speakers (only when the diarization engine is present) --
        if _HAS_PYANNOTE:
            self.diarize_check = ttk.Checkbutton(
                settings_row, text="Identify speakers",
                variable=self.diarize_var,
            )
            self.diarize_check.grid(row=0, column=6, padx=(12, 0))
        else:
            self.diarize_check = None

        # -- Keep awake --
        self._keep_awake_check = ttk.Checkbutton(
            settings_row, text="Keep awake",
            variable=self.prevent_sleep_var,
        )
        self._keep_awake_check.grid(row=0, column=7, padx=(12, 0))

        # -- Advanced gear --
        self._adv_btn = self._make_button(settings_row, text="Advanced\u2026", command=self._open_advanced_dialog, icon_name="gear")
        self._adv_btn.grid(row=0, column=8, padx=(12, 0))

        # Output entry stretches to fill available space
        settings_row.columnconfigure(4, weight=1)

        # Hidden combobox keeps model_combo["values"] in sync for any code reading it
        self.model_combo = ttk.Combobox(settings_inner, textvariable=self.model_var, state="readonly")
        # cfg_frame alias kept for compatibility with any code referencing it
        self.cfg_frame = settings_inner

        # --- Advanced Settings Dialog (hidden at startup) ---
        self._build_advanced_dialog()

        # --- State banner: colored strip below settings, changes with app state ---
        self._banner_frame = tk.Frame(self._header_frame, background=self.CLR_IDLE_BG, height=0)
        # Hidden initially — shown by _update_banner()
        self._banner_label = tk.Label(
            self._banner_frame, text="", font=self.FONT_VALUE,
            background=self.CLR_IDLE_BG, foreground=self.CLR_IDLE_FG,
            anchor="w", padx=12, pady=4,
        )
        self._banner_label.pack(fill="x")

        # --- Middle: file queue + run details ---
        self._paned = ttk.Panedwindow(root, orient="horizontal")
        self._paned.pack(fill="both", expand=True, pady=(6, 0))
        middle = self._paned
        # Apply the intended 60/40 idle split once real geometry exists —
        # without this the sash sat wherever clam left it (~80/20).
        self._panel_ratio_initialized = False
        self._user_panel_ratio: Optional[float] = None
        self._paned.bind("<Configure>", self._init_panel_ratio_once, add="+")

        # Gutter padding: breathing room between panel content and the sash.
        # Right panel gets more (12px) because its heading text is flush against
        # the sash edge; left panel gets less (8px) because its treeview scrollbar
        # already provides structural separation.
        left = ttk.Frame(middle, padding=(0, 0, self._s(8), 0))
        right = ttk.Frame(middle, padding=(self._s(12), 0, 0, 0))
        middle.add(left, weight=3)   # idle default 60/40
        middle.add(right, weight=2)
        self._build_sash_grip()

        # === LEFT: Audio queue ===
        ttk.Label(left, text="Audio queue", font=self.FONT_SECTION).pack(anchor="w")
        ttk.Label(left, text="Add files or folders to transcribe",
                  font=self.FONT_SMALL, foreground=C.TEXT_TERTIARY).pack(anchor="w", pady=(0, 8))
        # No extra padding: toolbar and tree share the heading's left edge
        # (one content margin everywhere).
        files_frame = ttk.Frame(left)
        files_frame.pack(fill="both", expand=True)

        toolbar = ttk.Frame(files_frame)
        toolbar.pack(fill="x", pady=(0, 10))
        self._btn_add_files = self._make_button(toolbar, text="Add files", command=self.add_files, icon_name="add-file")
        self._btn_add_files.pack(side="left")
        self._btn_add_folder = self._make_button(toolbar, text="Add folder", command=self.add_folder, icon_name="add-folder")
        self._btn_add_folder.pack(side="left", padx=(PAD_TOOLBAR_GAP, 0))
        self._btn_remove = self._make_button(toolbar, text="Remove", command=self.remove_selected_files, icon_name="remove")
        self._btn_remove.pack(side="left", padx=(PAD_TOOLBAR_GAP, 0))
        self._btn_clear = self._make_button(toolbar, text="Clear", command=self.clear_files, icon_name="clear")
        self._btn_clear.pack(side="left", padx=(PAD_TOOLBAR_GAP, 0))
        # Remove/Clear start disabled (no files, no selection)
        self._set_button_enabled(self._btn_remove, False)
        self._set_button_enabled(self._btn_clear, False)

        # Reorder buttons live in the toolbar (right-aligned cluster) so the
        # tree's left edge aligns with the toolbar instead of an off-grid
        # floating gutter. Rightmost = move-bottom; visual order ⤒ ↑ ↓ ⤓.
        _mod = "⌥" if sys.platform == "darwin" else "Alt+"
        self._btn_move_bottom = self._make_button(toolbar, text="", command=self._move_selection_bottom, icon_name="move-bottom")
        self._btn_move_bottom.pack(side="right")
        ToolTip(self._btn_move_bottom, f"Move to bottom ({_mod}End)")
        self._btn_move_down = self._make_button(toolbar, text="", command=self._move_selection_down, icon_name="move-down")
        self._btn_move_down.pack(side="right", padx=(0, 2))
        ToolTip(self._btn_move_down, f"Move down ({_mod}Down)")
        self._btn_move_up = self._make_button(toolbar, text="", command=self._move_selection_up, icon_name="move-up")
        self._btn_move_up.pack(side="right", padx=(0, 2))
        ToolTip(self._btn_move_up, f"Move up ({_mod}Up)")
        self._btn_move_top = self._make_button(toolbar, text="", command=self._move_selection_top, icon_name="move-top")
        self._btn_move_top.pack(side="right", padx=(0, 2))
        ToolTip(self._btn_move_top, f"Move to top ({_mod}Home)")
        for btn in (self._btn_move_top, self._btn_move_up, self._btn_move_down, self._btn_move_bottom):
            self._set_button_enabled(btn, False)

        tree_container = ttk.Frame(files_frame)
        tree_container.pack(fill="both", expand=True)

        self.file_tree = ttk.Treeview(
            tree_container,
            columns=("ordinal", "status", "progress", "duration", "file", "path"),
            show="headings",
            selectmode="extended",
            height=16,
        )
        tree_vsb = ttk.Scrollbar(tree_container, orient="vertical", command=self.file_tree.yview)
        tree_hsb = ttk.Scrollbar(tree_container, orient="horizontal", command=self.file_tree.xview)

        # Auto-hide the horizontal scrollbar: a permanent full-width thumb is
        # dead chrome (and it floated orphaned across the empty state).
        def _on_tree_xscroll(first, last):
            tree_hsb.set(first, last)
            if float(first) <= 0.0 and float(last) >= 1.0:
                tree_hsb.grid_remove()
            else:
                tree_hsb.grid()
        self.file_tree.configure(yscrollcommand=tree_vsb.set,
                                 xscrollcommand=_on_tree_xscroll)

        # Headings anchored to match their cells; fixed-vocabulary columns
        # sized to content with stretch=False so shrinkage never shears
        # "Running" into "Runnin(" — File is the sole stretch column.
        self.file_tree.heading("ordinal", text="#")
        self.file_tree.heading("status", text="Status", anchor="w")
        self.file_tree.heading("progress", text="Progress", anchor="w")
        self.file_tree.heading("duration", text="Duration", anchor="e")
        self.file_tree.heading("file", text="File", anchor="w")
        self.file_tree.heading("path", text="Path", anchor="w")
        _s = self._s
        self.file_tree.column("ordinal", width=_s(32), minwidth=_s(28), anchor="center", stretch=False)
        self.file_tree.column("status", width=_s(72), minwidth=_s(64), anchor="w", stretch=False)
        self.file_tree.column("progress", width=_s(150), minwidth=_s(84), anchor="w", stretch=False)
        self.file_tree.column("duration", width=_s(64), minwidth=_s(64), anchor="e", stretch=False)
        self.file_tree.column("file", width=_s(220), minwidth=_s(100), anchor="w", stretch=True)
        self.file_tree.column("path", width=_s(300), minwidth=_s(100), anchor="w", stretch=False)

        # State colors + alternating row backgrounds
        self.file_tree.tag_configure("queued", foreground=self.CLR_IDLE_FG)
        # The active row gets a subtle background so the eye finds it instantly
        self.file_tree.tag_configure("running", foreground=self.CLR_RUNNING_FG,
                                     background=C.HIGHLIGHT_HOVER)
        self.file_tree.tag_configure("done", foreground=self.CLR_SUCCESS_FG)
        self.file_tree.tag_configure("failed", foreground=self.CLR_ERROR_FG)
        self.file_tree.tag_configure("cancelled", foreground=self.CLR_WARNING_FG)
        self.file_tree.tag_configure("stripe", background=C.BG_STRIPE)

        self.file_tree.grid(row=0, column=1, sticky="nsew")
        tree_vsb.grid(row=0, column=2, sticky="ns")
        tree_hsb.grid(row=1, column=1, sticky="ew")
        tree_hsb.grid_remove()  # hidden until content actually overflows
        tree_container.rowconfigure(0, weight=1)
        tree_container.columnconfigure(1, weight=1)

        # Progress bar overlays drawn as tk.Frame widgets placed on the treeview.
        # Each visible row with progress gets a thin bar widget at the bottom of its
        # "progress" column cell. Bars are children of the treeview (scroll with it)
        # and are non-interactive (no event bindings).
        self._tree_progress: Dict[str, tuple] = {}  # item_id -> (percent, status)
        self._tree_bar_widgets: list = []  # list of placed Frame widgets (reused/recycled)
        # Redraw on scroll and resize
        def _on_tree_scroll(*args):
            tree_vsb.set(*args)
            self._redraw_tree_progress()
        self.file_tree.configure(yscrollcommand=_on_tree_scroll)

        def _on_tree_resize(_e=None):
            self._redraw_tree_progress()
            self._reelide_tree_names()
        self.file_tree.bind("<Configure>", lambda e: self.after(50, _on_tree_resize))
        # Column drag-resizes don't fire <Configure>; catch the release.
        self.file_tree.bind("<ButtonRelease-1>",
                            lambda e: self.after(50, self._reelide_tree_names),
                            add="+")

        # Empty state: clear call-to-action when queue is empty
        self._empty_frame = tk.Frame(tree_container, background=C.BG_PRIMARY)
        self._empty_frame.grid(row=0, column=0, columnspan=3, sticky="nsew")
        self._empty_frame.lift()
        # Center the content vertically
        self._empty_frame.grid_rowconfigure(0, weight=1)
        self._empty_frame.grid_rowconfigure(2, weight=1)
        self._empty_frame.grid_columnconfigure(0, weight=1)
        empty_inner = tk.Frame(self._empty_frame, background=C.BG_PRIMARY)
        empty_inner.grid(row=1, column=0)
        tk.Label(
            empty_inner, text="\u266b", font=(SYSTEM_FONT, _fs(32)),
            foreground=C.TEXT_PLACEHOLDER, background=C.BG_PRIMARY,
        ).pack(pady=(0, 6))
        tk.Label(
            empty_inner, text="No audio files yet",
            font=(SYSTEM_FONT, _fs(12)), foreground=C.TEXT_SECONDARY, background=C.BG_PRIMARY,
        ).pack()
        # Small spacer before the Add files button
        tk.Frame(empty_inner, height=12, background=C.BG_PRIMARY).pack()
        self._empty_add_btn = self._make_button(
            empty_inner, text="  Add files\u2026  ",
            command=self.add_files, variant="primary",
            font=(SYSTEM_FONT, _fs(10), "bold"), icon_name="add-file-white",
        )
        self._empty_add_btn.pack()

        self._setup_dnd(tree_container)

        self._tree_menu = tk.Menu(self.file_tree, tearoff=0)
        self._tree_menu.add_command(label="Show error", command=self._show_file_error)
        self._tree_menu.add_command(label="Retry failed", command=self._retry_failed)
        self._tree_menu.add_separator()
        self._tree_menu.add_command(label="Copy path", command=self._copy_file_path)
        self._tree_menu.add_command(label="Remove selected", command=self.remove_selected_files)
        self._tree_menu.add_separator()
        self._tree_menu.add_command(label="Move to top", command=self._move_selection_top)
        self._tree_menu.add_command(label="Move up", command=self._move_selection_up)
        self._tree_menu.add_command(label="Move down", command=self._move_selection_down)
        self._tree_menu.add_command(label="Move to bottom", command=self._move_selection_bottom)
        self.file_tree.bind("<Button-3>", self._on_tree_right_click)
        if sys.platform == "darwin":
            # macOS Aqua delivers the secondary (right) click as Button-2, so the
            # queue context menu never opens on <Button-3> alone (matches the
            # Activity tree's darwin binding).
            self.file_tree.bind("<Button-2>", self._on_tree_right_click)
        self.file_tree.bind("<Double-1>", self._on_tree_double_click)

        # Keyboard shortcuts for reorder
        self.bind("<Alt-Home>", lambda e: self._move_selection_top())
        self.bind("<Alt-Up>", lambda e: self._move_selection_up())
        self.bind("<Alt-Down>", lambda e: self._move_selection_down())
        self.bind("<Alt-End>", lambda e: self._move_selection_bottom())

        # Selection change triggers button state update
        def _on_tree_select(e):
            self._update_move_button_state()
            self._update_queue_button_state()
        self.file_tree.bind("<<TreeviewSelect>>", _on_tree_select)

        # Flash tag for reorder feedback
        self.file_tree.tag_configure("flash", background=C.HIGHLIGHT_FLASH)

        # Row hover highlight
        self._hover_row_id: Optional[str] = None
        self.file_tree.tag_configure("hover", background=C.HIGHLIGHT_HOVER)
        self.file_tree.bind("<Motion>", self._on_tree_hover)
        self.file_tree.bind("<Leave>", self._on_tree_leave)

        # === RIGHT: Current run ===
        ttk.Label(right, text="Current run", font=self.FONT_SECTION).pack(anchor="w")
        self._run_subtitle = ttk.Label(right, text="",
                  font=self.FONT_SMALL, foreground=C.TEXT_TERTIARY)
        self._run_subtitle.pack(anchor="w", pady=(0, 8))
        details_frame = ttk.Frame(right, padding=(10, 0))
        details_frame.pack(fill="both", expand=True)

        # --- Idle panel: onboarding prompt shown when no job has run ---
        self._idle_panel = tk.Frame(details_frame, background=C.BG_PRIMARY)
        self._idle_panel.pack(fill="both", expand=True)
        # Center vertically
        self._idle_panel.grid_rowconfigure(0, weight=1)
        self._idle_panel.grid_rowconfigure(2, weight=1)
        self._idle_panel.grid_columnconfigure(0, weight=1)
        idle_inner = tk.Frame(self._idle_panel, background=C.BG_PRIMARY)
        idle_inner.grid(row=1, column=0)

        # Idle-state message
        tk.Label(
            idle_inner, text="No active transcription",
            font=(SYSTEM_FONT, _fs(12)), foreground=C.TEXT_SECONDARY, background=C.BG_PRIMARY,
        ).pack(anchor="w")

        # --- Recording panel (hidden initially, shown when recording) ---
        self._recording_panel = tk.Frame(details_frame, background=C.BG_PRIMARY)
        # NOT packed initially — shown by _start_recording()

        rec_inner = tk.Frame(self._recording_panel, background=C.BG_PRIMARY)
        rec_inner.pack(fill="both", expand=True, padx=4, pady=4)

        # Recording status line
        self._rec_status_var = tk.StringVar(value="")
        self._rec_status_label = tk.Label(
            rec_inner, textvariable=self._rec_status_var,
            font=(SYSTEM_FONT, _fs(11)), foreground=C.TEXT_SECONDARY,
            background=C.BG_PRIMARY, anchor="w",
        )
        self._rec_status_label.pack(fill="x", pady=(0, 6))

        # Audio level bar
        self._rec_level_frame = tk.Frame(rec_inner, background=C.BORDER_CARD, height=6)
        self._rec_level_frame.pack(fill="x", pady=(0, 8))
        self._rec_level_frame.pack_propagate(False)
        # Red, not brand blue: the level meter reads as live capture.
        self._rec_level_bar = tk.Frame(self._rec_level_frame, background=C.BTN_DESTRUCTIVE_FG, width=0)
        self._rec_level_bar.place(x=0, y=0, relheight=1.0, width=0)

        # Recording text area (same pattern as preview_text)
        rec_text_frame = tk.Frame(rec_inner, background=C.BG_PRIMARY)
        rec_text_frame.pack(fill="both", expand=True)
        self._rec_text = tk.Text(
            rec_text_frame, height=14, wrap="word",
            font=(SYSTEM_FONT, _fs(12)),
            background=C.BG_ELEVATED, foreground=C.TEXT_PRIMARY,
            insertbackground=C.TEXT_PRIMARY,
            relief="flat", borderwidth=0,
            highlightthickness=1, highlightbackground=C.BORDER_CARD,
            highlightcolor=C.BORDER_CARD,
            padx=self._s(12), pady=self._s(10),
            spacing1=self._s(2), spacing3=self._s(8),
        )
        self._rec_text.tag_configure("rtl", justify="right")
        self._rec_text.tag_configure("ltr", justify="left")
        rec_vsb = ttk.Scrollbar(rec_text_frame, orient="vertical", command=self._rec_text.yview)
        self._rec_text.configure(yscrollcommand=rec_vsb.set)
        self._rec_text.pack(side="left", fill="both", expand=True)
        rec_vsb.pack(side="right", fill="y")

        # Recording toolbar (copy, clear, save)
        rec_toolbar = tk.Frame(rec_inner, background=C.BG_PRIMARY)
        rec_toolbar.pack(fill="x", pady=(6, 0))
        self._make_button(rec_toolbar, text="Copy", command=self._copy_rec_text).pack(side="left", padx=(0, 6))
        self._make_button(rec_toolbar, text="Clear", command=self._clear_rec_text).pack(side="left", padx=(0, 6))
        self._make_button(rec_toolbar, text="Save to file", command=self._save_rec_text).pack(side="left")

        # --- Run details (hidden initially, replace idle panel on first run) ---
        self._run_details_frame = ttk.Frame(details_frame)
        # NOT packed initially — shown by _show_stats() on first run

        summary = ttk.Frame(self._run_details_frame)
        summary.pack(fill="x", pady=(0, 8))
        ttk.Label(summary, textvariable=self.job_status_var, font=self.FONT_SECTION).pack(anchor="w")
        ttk.Label(summary, textvariable=self.overall_summary_var, font=self.FONT_LABEL,
                  foreground=C.TEXT_MUTED).pack(anchor="w", pady=(2, 0))

        # --- Card: Current File (blue left accent border) ---
        self._stats_frame = ttk.Frame(self._run_details_frame)
        self._stats_frame.pack(fill="x")

        self._file_card_outer = tk.Frame(self._stats_frame, background=self.CLR_RUNNING_ACCENT)
        file_card_outer = self._file_card_outer
        file_card_outer.pack(fill="x", pady=(0, 8))
        # 4px blue left border via padx asymmetry
        file_card = tk.Frame(file_card_outer, background=C.BG_CARD,
                             highlightbackground=C.BORDER_CARD, highlightthickness=1)
        file_card.pack(fill="both", padx=(4, 0))
        file_inner = tk.Frame(file_card, background=C.BG_CARD, padx=10, pady=8)
        file_inner.pack(fill="x")

        tk.Label(file_inner, textvariable=self.current_file_var, font=self.FONT_VALUE,
                 background=C.BG_CARD, foreground=C.TEXT_PRIMARY, anchor="w").pack(fill="x")

        # Progress bar inside file card
        bar_frame = tk.Frame(file_inner, background=C.BG_CARD)
        bar_frame.pack(fill="x", pady=(6, 0))
        self.current_progress = ttk.Progressbar(bar_frame, mode="determinate")
        self.current_progress.pack(fill="x")

        # Breathing line (single composed status line, color-coded by state)
        self._file_breathing_label = tk.Label(
            file_inner, textvariable=self.file_breathing_var, font=self.FONT_SMALL,
            background=C.BG_CARD, foreground=self.CLR_IDLE_FG, anchor="w")
        self._file_breathing_label.pack(fill="x", pady=(4, 0))

        # --- Card: Batch ---
        self._batch_card = tk.Frame(self._stats_frame, background=C.BG_CARD,
                              highlightbackground=C.BORDER_CARD, highlightthickness=1)
        batch_card = self._batch_card
        batch_card.pack(fill="x", pady=(0, 6))
        batch_inner = tk.Frame(batch_card, background=C.BG_CARD, padx=10, pady=8)
        batch_inner.pack(fill="x")

        # Batch identity label (adaptive: "1 of 3 files" → "2 files · 05:31 audio")
        self._batch_identity_label = tk.Label(
            batch_inner, textvariable=self.batch_identity_var, font=self.FONT_VALUE,
            background=C.BG_CARD, foreground=C.TEXT_PRIMARY, anchor="w")
        self._batch_identity_label.pack(fill="x")

        self.overall_progress = ttk.Progressbar(batch_inner, mode="determinate")
        self.overall_progress.pack(fill="x", pady=(4, 0))

        # Batch breathing line
        self._batch_breathing_label = tk.Label(
            batch_inner, textvariable=self.batch_breathing_var, font=self.FONT_SMALL,
            background=C.BG_CARD, foreground=self.CLR_IDLE_FG, anchor="w")
        self._batch_breathing_label.pack(fill="x", pady=(4, 0))

        # Tabbed panel: Preview / Log (inside run details, shown on first run)
        self._detail_notebook = ttk.Notebook(self._run_details_frame)
        self._detail_notebook.pack(fill="both", expand=True)

        preview_tab = ttk.Frame(self._detail_notebook, padding=4)
        self._detail_notebook.add(preview_tab, text="Live preview")
        # Preview toolbar with copy button
        preview_toolbar = ttk.Frame(preview_tab)
        preview_toolbar.pack(fill="x", pady=(0, 4))
        self._copy_preview_btn = self._make_button(
            preview_toolbar, text="Copy to clipboard", command=self._copy_preview_text)
        self._copy_preview_btn.pack(side="right")
        preview_container = ttk.Frame(preview_tab)
        preview_container.pack(fill="both", expand=True)
        self.preview_text = tk.Text(
            preview_container, height=14, wrap="word",
            font=(SYSTEM_FONT, _fs(11)),
            background=C.BG_ELEVATED, foreground=C.TEXT_PRIMARY,
            insertbackground=C.TEXT_PRIMARY,
            relief="flat", borderwidth=0,
            highlightthickness=1, highlightbackground=C.BORDER_CARD,
            highlightcolor=C.BORDER_CARD,
            padx=self._s(12), pady=self._s(8),
            spacing1=self._s(1), spacing3=self._s(4),
        )
        # RTL/LTR tag for Hebrew vs other languages
        self.preview_text.tag_configure("rtl", justify="right")
        self.preview_text.tag_configure("ltr", justify="left")
        preview_vsb = ttk.Scrollbar(preview_container, orient="vertical", command=self.preview_text.yview)
        self.preview_text.configure(yscrollcommand=preview_vsb.set)
        self.preview_text.pack(side="left", fill="both", expand=True)
        preview_vsb.pack(side="right", fill="y")

        log_tab = ttk.Frame(self._detail_notebook, padding=4)
        self._detail_notebook.add(log_tab, text="Log")

        # --- Log view toggle toolbar ---
        log_toolbar = ttk.Frame(log_tab)
        log_toolbar.pack(fill="x", pady=(0, 4))
        self._log_activity_btn = self._make_button(
            log_toolbar, text="Activity",
            command=lambda: self._switch_log_view("activity"))
        self._log_activity_btn.pack(side="left", padx=(0, 2))
        self._log_raw_btn = self._make_button(
            log_toolbar, text="Raw",
            command=lambda: self._switch_log_view("raw"))
        self._log_raw_btn.pack(side="left")

        # --- Activity view (structured Treeview) ---
        self._log_activity_frame = ttk.Frame(log_tab)
        self._activity_tree = ttk.Treeview(
            self._log_activity_frame,
            columns=("wall_time", "timestamp"),
            show="tree",  # show tree column only (no headings)
            selectmode="browse",
        )
        self._activity_tree.column("#0", width=self._s(520), stretch=True)
        self._activity_tree.column("wall_time", width=self._s(60), anchor="e", stretch=False)
        self._activity_tree.column("timestamp", width=self._s(60), anchor="e", stretch=False)

        # Tag styles for activity tree nodes
        self._activity_tree.tag_configure("group_setup", foreground=C.LOG_GROUP_FG,
                                          font=(SYSTEM_FONT, _fs(10), "bold"))
        self._activity_tree.tag_configure("group_files", foreground=C.TEXT_PRIMARY,
                                          font=(SYSTEM_FONT, _fs(10), "bold"))
        self._activity_tree.tag_configure("file_ok", foreground=C.STATUS_SUCCESS_FG)
        self._activity_tree.tag_configure("file_fail", foreground=C.LOG_ERROR_FG)
        self._activity_tree.tag_configure("file_warn", foreground=C.STATUS_WARNING_FG)
        self._activity_tree.tag_configure("file_running", foreground=C.STATUS_RUNNING_FG)
        self._activity_tree.tag_configure("results", foreground=C.LOG_GROUP_FG,
                                          font=(SYSTEM_FONT, _fs(10), "italic"))
        self._activity_tree.tag_configure("detail", foreground=C.TEXT_TERTIARY)
        self._activity_tree.tag_configure("error_detail", foreground=C.LOG_ERROR_FG)
        self._activity_tree.tag_configure("user_action", foreground=C.TEXT_TERTIARY)

        activity_vsb = ttk.Scrollbar(self._log_activity_frame, orient="vertical",
                                     command=self._activity_tree.yview)
        self._activity_tree.configure(yscrollcommand=activity_vsb.set)
        self._activity_tree.pack(side="left", fill="both", expand=True)
        activity_vsb.pack(side="right", fill="y")

        # --- Raw view (existing tk.Text) ---
        self._log_raw_frame = ttk.Frame(log_tab)
        # Deliberate monospace for the raw log (matches the crash dialog);
        # the defect was the implicit TkFixedFont fallback and zero padding.
        self.log_text = tk.Text(
            self._log_raw_frame, height=14, wrap="word",
            font=("Consolas" if sys.platform == "win32" else "Menlo", _fs(9)),
            background=C.BG_ELEVATED, foreground=C.TEXT_PRIMARY,
            relief="flat", borderwidth=0,
            highlightthickness=1, highlightbackground=C.BORDER_CARD,
            highlightcolor=C.BORDER_CARD,
            padx=self._s(10), pady=self._s(8),
        )
        # Color-coded log tags
        self.log_text.tag_configure("info", foreground=C.TEXT_PRIMARY)
        self.log_text.tag_configure("warning", foreground=C.STATUS_WARNING_FG)
        self.log_text.tag_configure("error", foreground=C.LOG_ERROR_FG)
        log_vsb = ttk.Scrollbar(self._log_raw_frame, orient="vertical",
                                command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_vsb.set)
        self.log_text.pack(side="left", fill="both", expand=True)
        log_vsb.pack(side="right", fill="y")

        # Activity tree context menu
        self._activity_menu = tk.Menu(self._activity_tree, tearoff=0)
        self._activity_menu.add_command(label="Expand all",
                                        command=self._activity_expand_all)
        self._activity_menu.add_command(label="Collapse all",
                                        command=self._activity_collapse_all)
        self._activity_menu.add_separator()
        self._activity_menu.add_command(label="Copy all to clipboard",
                                        command=self._activity_copy_all)
        self._activity_tree.bind("<Button-3>" if sys.platform != "darwin" else "<Button-2>",
                                 self._activity_show_context_menu)

        # Default to Activity view
        self._switch_log_view("activity")

        self._apply_tooltips()

    # --- Keyboard shortcuts ---

    def _on_ctrl_enter(self, event=None) -> None:
        """Ctrl+Enter: start transcription (if ready and not running)."""
        if not (self.worker_thread and self.worker_thread.is_alive()):
            if not self.readiness_var.get():  # no issues blocking start
                self.start_transcription()

    def _on_escape(self, event=None) -> None:
        """Escape: cancel transcription if running."""
        if self.worker_thread and self.worker_thread.is_alive():
            self.request_cancel_now()

    def _on_focus_in(self, event=None) -> None:
        """Auto-refresh model list when window regains focus (debounced, 10s)."""
        now = time.monotonic()
        if now - self._last_focus_refresh < 10.0:
            return  # debounce: skip if refreshed within last 10 seconds
        if self.worker_thread and self.worker_thread.is_alive():
            return  # don't refresh during a running job
        self._last_focus_refresh = now
        self.refresh_models()

    def _on_close(self) -> None:
        """Handle window close. Confirm if a transcription job or recording is active."""
        if self.worker_thread and self.worker_thread.is_alive():
            if not messagebox.askyesno(
                "Quit?",
                "A transcription is running. Quit anyway?",
                parent=self,
            ):
                return
        if self._recorder is not None and self._recorder.is_recording():
            # wait=False: never join recorder threads on the main thread
            # while closing — they are daemons, process exit reaps them.
            self._recorder.stop(wait=False)
            self._recorder = None
        # Persist settings changed since the last run start — best
        # effort: a save failure must never block quitting.
        try:
            self.save_config_from_ui()
        except Exception:
            logger.debug("Config save on close failed", exc_info=True)
        self._sleep_inhibitor.release()
        self.destroy()

    def _language_tooltip_text(self) -> str:
        """Build dynamic tooltip text for the language combobox.

        Shows the auto-selected model, its size, download status,
        language optimization info, and the transcription engine.
        """
        label = self.model_var.get()
        if not label:
            return "No model available for this language."
        friendly, lang_tag, size_str = self._parse_model_label(label)

        # Line 1: model name + size
        model_line = friendly
        if size_str:
            model_line += f" ({size_str}"
            if "[hub]" in label:
                model_line += ", downloads on first use"
            else:
                model_line += ", cached"
            model_line += ")"
        elif "[hub]" in label:
            model_line += " (downloads on first use)"

        # Line 2: language affinity
        if lang_tag == "Hebrew":
            affinity = "Hebrew-optimized model"
        elif lang_tag == "English":
            affinity = "English-optimized model"
        elif lang_tag == "Multilingual":
            lang_display = self._lang_display_var.get()
            affinity = f"Multilingual model (supports {lang_display})"
        else:
            affinity = lang_tag

        # Line 3: engine
        backend = self.backend_var.get() if hasattr(self, "backend_var") else ""
        engine = "faster-whisper" if "faster" in backend else backend if backend else ""

        parts = [f"Model: {model_line}", affinity]
        if engine:
            parts.append(f"Engine: {engine}")
        return "\n".join(parts)

    def _copy_preview_text(self) -> None:
        """Copy the preview text content to the system clipboard."""
        text = self.preview_text.get("1.0", "end-1c").strip()
        if text:
            self.clipboard_clear()
            self.clipboard_append(text)

    def _get_preview_text_tag(self) -> str:
        """Return 'rtl' if current language is Hebrew, else 'ltr'."""
        lang = self.language_var.get().strip()
        return "rtl" if lang == "he" else "ltr"

    def _apply_tooltips(self) -> None:
        """Attach hover tooltips to controls that benefit from explanation."""
        ToolTip(self.preset_combo,
                "Speed preset. Overrides beam, batch, VAD, condition settings.\n"
                "fast: beam=1, VAD on, no context carry.\n"
                "balanced: beam=3, VAD on.\n"
                "quality: beam=5, context carry on.\n"
                "custom: set each parameter yourself.")
        ToolTip(self.backend_combo,
                "Transcription engine.\n"
                "faster-whisper: recommended, faster, lower memory.\n"
                "openai-whisper: original implementation, requires FFmpeg.")
        ToolTip(self.language_combo, self._language_tooltip_text)
        ToolTip(self.task_combo,
                "transcribe: output in the source language.\n"
                "translate: output translated to English.")
        ToolTip(self.device_combo,
                "auto: use GPU if available, else CPU.\n"
                "cuda: force GPU (requires NVIDIA + CUDA).\n"
                "cpu: force CPU (slower but always works).")
        ToolTip(self.compute_combo,
                "Numeric precision for inference.\n"
                "auto: float16 on GPU, int8 on CPU.\n"
                "Lower precision = faster + less memory, slightly less accurate.")
        ToolTip(self.beam_entry,
                "Beam search width. Higher = more accurate, slower.\n"
                "Default: 5. Minimum: 1; above 10 gives little benefit.")
        ToolTip(self.vad_check,
                "Voice Activity Detection. Skips silent sections\n"
                "for faster processing. Only works with faster-whisper.")
        ToolTip(self.batch_size_entry,
                "Parallel chunk processing. 0 = auto (picks based on\n"
                "device/memory). Higher = faster on GPU, more memory.\n"
                "Set to 1 to disable batching.")
        ToolTip(self.cond_prev_check,
                "Feed previous segment text as context for the next.\n"
                "Off = faster, fewer hallucination loops.\n"
                "On = better coherence across segments.")
        if self.diarize_check is not None:
            ToolTip(self.diarize_check,
                    "Label each line with the speaker (Speaker 1, Speaker 2, ...).\n"
                    "Runs after transcription and roughly doubles processing time.")
        ToolTip(self.num_speakers_entry,
                "0 = auto-detect. Set the exact count if you know it\n"
                "(e.g. 2 for an interview) — improves accuracy.")
        if getattr(self, "diarize_device_combo", None) is not None:
            ToolTip(self.diarize_device_combo,
                    "Auto uses Apple Silicon GPU acceleration (MPS) when\n"
                    "available. Choose CPU on machines with 8 GB RAM.")
        ToolTip(self._keep_awake_check,
                "Prevent the computer from sleeping during transcription.\n"
                "Recommended for long batches that run unattended.")
        ToolTip(self._adv_btn, "Open advanced settings: backend, task, device,\ncompute type, beam/batch size, VAD, formats, models")
        # Card tooltips (live-updated via _refresh_breathing_lines)
        self._file_card_tooltip = ToolTip(self._file_card_outer, "")
        self._batch_card_tooltip = ToolTip(self._batch_card, "")
        ToolTip(self._btn_clear,
                "Clear the file list and reset this session's\nactivity log and statistics")
        ToolTip(self.start_button, "Start transcription  (Ctrl+Enter)")
        ToolTip(self.pause_button, "Pause at the next safe point")
        ToolTip(self.resume_button, "Resume transcription")
        ToolTip(self.stop_button, "Finish the current file, then stop")
        ToolTip(self.cancel_button, "Cancel immediately  (Escape)")

    def _build_advanced_dialog(self) -> None:
        """Create the advanced settings dialog with tabbed sections."""
        _s = self._s
        self._adv_dialog = tk.Toplevel(self)
        self._adv_dialog.title("Advanced Settings")
        self._adv_dialog.geometry(f"{_s(600)}x{_s(400)}")
        self._adv_dialog.resizable(True, True)
        self._adv_dialog.minsize(_s(500), _s(380))
        if self._icon_path:
            self._adv_dialog.iconbitmap(self._icon_path)
        self._adv_dialog.protocol("WM_DELETE_WINDOW", self._adv_dialog.withdraw)
        self._adv_dialog.withdraw()
        self._adv_dialog.configure(background=C.BG_PRIMARY)

        # --- Header ---
        header = tk.Frame(self._adv_dialog, background=C.BG_PRIMARY)
        header.pack(fill="x", padx=_s(16), pady=(_s(14), 0))
        ttk.Label(header, text="Advanced Settings", font=self.FONT_SECTION).pack(anchor="w")
        ttk.Label(header, text="Engine and performance",
                  font=self.FONT_SMALL, foreground=C.TEXT_TERTIARY).pack(anchor="w", pady=(0, _s(12)))

        # --- Notebook with 3 tabs ---
        notebook = ttk.Notebook(self._adv_dialog)
        notebook.pack(fill="both", expand=True, padx=_s(16))

        # ── Tab 1: Engine ──
        engine_tab = ttk.Frame(notebook, padding=(_s(24), _s(20)))
        notebook.add(engine_tab, text="Engine")

        engine_grid = tk.Frame(engine_tab, background=C.BG_PRIMARY)
        engine_grid.pack(anchor="w")

        # Row 1: Backend + Task
        tk.Label(engine_grid, text="Backend", font=self.FONT_SMALL,
                 background=C.BG_PRIMARY, foreground=C.TEXT_SECONDARY).grid(
            row=0, column=0, sticky="w")
        self.backend_combo = ttk.Combobox(
            engine_grid, textvariable=self.backend_var, state="readonly", width=18)
        self.backend_combo.grid(row=1, column=0, sticky="w", pady=(_s(4), 0))
        self.backend_combo.bind("<<ComboboxSelected>>", lambda e: self.refresh_models())

        tk.Label(engine_grid, text="Task", font=self.FONT_SMALL,
                 background=C.BG_PRIMARY, foreground=C.TEXT_SECONDARY).grid(
            row=0, column=1, sticky="w", padx=(_s(28), 0))
        self.task_combo = ttk.Combobox(
            engine_grid, textvariable=self.task_var, state="readonly",
            values=["transcribe", "translate"], width=18)
        self.task_combo.grid(row=1, column=1, sticky="w", padx=(_s(28), 0), pady=(_s(4), 0))

        # Row 2: Device + Compute type
        tk.Label(engine_grid, text="Device", font=self.FONT_SMALL,
                 background=C.BG_PRIMARY, foreground=C.TEXT_SECONDARY).grid(
            row=2, column=0, sticky="w", pady=(_s(16), 0))
        self.device_combo = ttk.Combobox(
            engine_grid, textvariable=self.device_var, state="readonly",
            values=["auto", "cpu", "cuda"], width=18)
        self.device_combo.grid(row=3, column=0, sticky="w", pady=(_s(4), 0))

        tk.Label(engine_grid, text="Compute type", font=self.FONT_SMALL,
                 background=C.BG_PRIMARY, foreground=C.TEXT_SECONDARY).grid(
            row=2, column=1, sticky="w", padx=(_s(28), 0), pady=(_s(16), 0))
        self.compute_combo = ttk.Combobox(
            engine_grid, textvariable=self.compute_type_var, state="readonly",
            values=["auto", "default", "int8", "int8_float16", "float16", "float32"],
            width=18)
        self.compute_combo.grid(row=3, column=1, sticky="w", padx=(_s(28), 0), pady=(_s(4), 0))

        # ── Tab 2: Quality ──
        quality_tab = ttk.Frame(notebook, padding=(_s(24), _s(20)))
        notebook.add(quality_tab, text="Quality")

        # Speed preset (hero control)
        tk.Label(quality_tab, text="Speed preset", font=self.FONT_SMALL,
                 background=C.BG_PRIMARY, foreground=C.TEXT_SECONDARY).pack(anchor="w")
        self.preset_combo = ttk.Combobox(
            quality_tab, textvariable=self.speed_preset_var, state="readonly",
            values=["fast", "balanced", "quality", "custom"], width=12)
        self.preset_combo.pack(anchor="w", pady=(_s(4), 0))
        self.preset_combo.bind("<<ComboboxSelected>>", lambda e: self._apply_speed_preset())
        self._inline_preset_desc = tk.Label(
            quality_tab, text="", font=self.FONT_SMALL,
            background=C.BG_PRIMARY, foreground=C.TEXT_TERTIARY, anchor="w")
        self._inline_preset_desc.pack(anchor="w", pady=(_s(4), 0))

        # Separator: quick choice above, detail controls below
        ttk.Separator(quality_tab, orient="horizontal").pack(
            fill="x", pady=(_s(16), _s(16)))

        # Detail controls
        detail_grid = tk.Frame(quality_tab, background=C.BG_PRIMARY)
        detail_grid.pack(anchor="w")

        tk.Label(detail_grid, text="Beam size", font=self.FONT_SMALL,
                 background=C.BG_PRIMARY, foreground=C.TEXT_SECONDARY).grid(
            row=0, column=0, sticky="w")
        self.beam_entry = ttk.Entry(detail_grid, textvariable=self.beam_size_var, width=6)
        self.beam_entry.grid(row=1, column=0, sticky="w", pady=(_s(4), 0))

        tk.Label(detail_grid, text="Batch size", font=self.FONT_SMALL,
                 background=C.BG_PRIMARY, foreground=C.TEXT_SECONDARY).grid(
            row=0, column=1, sticky="w", padx=(_s(28), 0))
        self.batch_size_entry = ttk.Entry(
            detail_grid, textvariable=self.batch_size_var, width=6)
        self.batch_size_entry.grid(
            row=1, column=1, sticky="w", padx=(_s(28), 0), pady=(_s(4), 0))

        # Checkboxes below the entries
        self.vad_check = ttk.Checkbutton(
            quality_tab, text="VAD filter", variable=self.vad_var)
        self.vad_check.pack(anchor="w", pady=(_s(12), 0))
        self.cond_prev_check = ttk.Checkbutton(
            quality_tab, text="Condition on previous text",
            variable=self.condition_on_previous_text_var)
        self.cond_prev_check.pack(anchor="w", pady=(_s(6), 0))

        # ── Speakers group (diarization) ──
        # Always built; when the engine is absent the controls are disabled
        # and the hint carries the install line (visible-with-hint precedent
        # of the experimental-recording toggle below).
        ttk.Separator(quality_tab, orient="horizontal").pack(
            fill="x", pady=(_s(16), _s(12)))
        tk.Label(quality_tab, text="Speakers", font=self.FONT_SMALL,
                 background=C.BG_PRIMARY, foreground=C.TEXT_SECONDARY).pack(
            anchor="w")

        speakers_grid = tk.Frame(quality_tab, background=C.BG_PRIMARY)
        speakers_grid.pack(anchor="w", pady=(_s(4), 0))

        tk.Label(speakers_grid, text="Number of speakers", font=self.FONT_SMALL,
                 background=C.BG_PRIMARY, foreground=C.TEXT_SECONDARY).grid(
            row=0, column=0, sticky="w")
        self.num_speakers_entry = ttk.Entry(
            speakers_grid, textvariable=self.num_speakers_var, width=6)
        self.num_speakers_entry.grid(row=1, column=0, sticky="w",
                                     pady=(_s(4), 0))

        if sys.platform == "darwin":
            tk.Label(speakers_grid, text="Processing device",
                     font=self.FONT_SMALL, background=C.BG_PRIMARY,
                     foreground=C.TEXT_SECONDARY).grid(
                row=0, column=1, sticky="w", padx=(_s(28), 0))
            self._diarize_device_display_var = tk.StringVar(
                value=self._DIARIZE_DEVICE_DISPLAY.get(
                    self.diarize_device_var.get(), "Auto (MPS)"))
            self.diarize_device_combo = ttk.Combobox(
                speakers_grid, textvariable=self._diarize_device_display_var,
                state="readonly",
                values=list(self._DIARIZE_DEVICE_DISPLAY.values()), width=12)
            self.diarize_device_combo.grid(
                row=1, column=1, sticky="w", padx=(_s(28), 0), pady=(_s(4), 0))
            self.diarize_device_combo.bind(
                "<<ComboboxSelected>>",
                lambda e: self._on_diarize_device_display_changed())
        else:
            # Non-mac platforms have no device choice: the worker's "auto"
            # resolves to CUDA-if-available, else CPU.
            self.diarize_device_combo = None

        speakers_hint = "0 = detect the number of speakers automatically."
        if not _HAS_PYANNOTE:
            # The project is not on PyPI — quote the README's real command.
            speakers_hint += ('\nRequires: pip install -e ".[diarization]" '
                              "from the project root")
            self.num_speakers_entry.configure(state="disabled")
            if self.diarize_device_combo is not None:
                self.diarize_device_combo.configure(state="disabled")
        tk.Label(quality_tab, text=speakers_hint, font=self.FONT_SMALL,
                 background=C.BG_PRIMARY, foreground=C.TEXT_TERTIARY,
                 justify="left", anchor="w").pack(anchor="w", pady=(_s(4), 0))

        # ── Tab 3: Output ──
        output_tab = ttk.Frame(notebook, padding=(_s(24), _s(20)))
        notebook.add(output_tab, text="Output")

        tk.Label(output_tab,
                 text="Select the output formats for your transcriptions:",
                 font=self.FONT_SMALL, background=C.BG_PRIMARY,
                 foreground=C.TEXT_SECONDARY).pack(anchor="w", pady=(0, _s(14)))
        fmt_row = tk.Frame(output_tab, background=C.BG_PRIMARY)
        fmt_row.pack(anchor="w")
        for fmt in OUTPUT_FORMATS:
            ttk.Checkbutton(
                fmt_row, text=fmt.upper(),
                variable=self.format_vars[fmt]).pack(side="left", padx=(0, _s(20)))

        # --- Experimental features (below tabs, above footer) ---
        ttk.Separator(self._adv_dialog, orient="horizontal").pack(
            fill="x", padx=_s(16), pady=(_s(10), 0))
        exp_frame = tk.Frame(self._adv_dialog, background=C.BG_PRIMARY)
        exp_frame.pack(fill="x", padx=_s(24), pady=(_s(8), _s(4)))
        ttk.Checkbutton(
            exp_frame, text="Enable live recording (experimental)",
            variable=self._experimental_recording_var,
            command=self._update_recording_visibility,
        ).pack(anchor="w")
        exp_desc = "Real-time speech-to-text from your microphone."
        if not _HAS_SOUNDDEVICE:
            exp_desc += "\nRequires: pip install sounddevice silero-vad faster-whisper"
        tk.Label(
            exp_frame, text=exp_desc, font=self.FONT_SMALL,
            background=C.BG_PRIMARY, foreground=C.TEXT_TERTIARY,
            justify="left", anchor="w",
        ).pack(anchor="w", padx=(_s(24), 0))

        # --- Footer (below tabs, always visible) ---
        footer_border = tk.Frame(
            self._adv_dialog, background=C.BORDER_LIGHT, height=1)
        footer_border.pack(fill="x", padx=_s(16))

        footer = tk.Frame(self._adv_dialog, background=C.BG_PRIMARY)
        footer.pack(fill="x", padx=_s(16), pady=(_s(10), _s(12)))

        # Left: self-test
        self._selftest_btn = self._make_button(
            footer, text="\u25b6 Run self-test", command=self._run_selftest,
            font=self.FONT_SMALL, padx=8, pady=2)
        self._selftest_btn.pack(side="left")
        self._selftest_status = tk.Label(
            footer, text="", font=self.FONT_SMALL,
            background=C.BG_PRIMARY, foreground=C.TEXT_TERTIARY)
        self._selftest_status.pack(side="left", padx=(_s(8), 0))

        # Right: OK button — primary variant marks the dialog's main action
        self._make_button(
            footer, text="    OK    ", variant="primary",
            command=self._adv_dialog.withdraw).pack(side="right")

        # Right (before OK): model cache
        self._open_model_dir_btn = self._make_button(
            footer, text="Open model folder",
            command=self._open_model_cache_dir,
            font=self.FONT_SMALL, padx=8, pady=2)
        self._open_model_dir_btn.pack(side="right", padx=(0, _s(8)))
        cache_dir = get_primary_model_cache_dir()
        if cache_dir is None:
            self._open_model_dir_btn.configure(state="disabled")
        ToolTip(self._open_model_dir_btn,
                "Open the folder where downloaded transcription models are stored.\n"
                "You can delete models here to free disk space.")

    def _open_advanced_dialog(self) -> None:
        """Show the advanced settings dialog sized to its content, centered on parent."""
        _s = self._s
        self._adv_dialog.deiconify()
        self._adv_dialog.update_idletasks()
        # Size to content: the tabs and the experimental section have outgrown
        # the old fixed 600x480, which clipped the bottom button row until the
        # user resized by hand. reqheight covers the tallest notebook tab.
        scr_w, scr_h = self.winfo_screenwidth(), self.winfo_screenheight()
        req_w = max(_s(600), self._adv_dialog.winfo_reqwidth())
        req_w = min(req_w, scr_w - _s(40))
        req_h = max(_s(400), self._adv_dialog.winfo_reqheight())
        req_h = min(req_h, scr_h - _s(120))
        px, py = self.winfo_x(), self.winfo_y()
        x = px + (self.winfo_width() - req_w) // 2
        y = py + (self.winfo_height() - req_h) // 2
        # Clamp to the screen rect (max last, so the title bar always stays
        # reachable) — but only when the parent is on the primary display:
        # Tk reports only the primary monitor's size on Windows, and yanking
        # a dialog from a secondary monitor to x=0 is worse than no clamp.
        if 0 <= px < scr_w and 0 <= py < scr_h:
            x = max(0, min(x, scr_w - req_w))
            y = max(0, min(y, scr_h - req_h))
        self._adv_dialog.geometry(f"{req_w}x{req_h}+{x}+{y}")
        self._adv_dialog.lift()
        self._adv_dialog.focus_set()

    def _open_model_cache_dir(self) -> None:
        """Open the model cache directory in the OS file manager."""
        cache_dir = get_primary_model_cache_dir()
        if cache_dir is not None:
            try:
                _open_path(cache_dir)
            except Exception:
                logger.warning("Failed to open model cache dir: %s", cache_dir, exc_info=True)

    # --- Self-test -----------------------------------------------------------

    class _SelfTestHost:
        """Minimal WorkerHost that silently absorbs all events.

        Prevents self-test progress from leaking into the main UI.
        """
        def __init__(self) -> None:
            self.event_queue: queue.Queue = queue.Queue()
            self.pause_event = threading.Event()
            self.pause_event.set()
            self.stop_requested = False
            self.cancel_requested = False

        def post_event(self, kind: str, **payload: object) -> None:
            pass  # swallow all events

    def _run_selftest(self) -> None:
        """Launch the self-test on a background thread."""
        if self.worker_thread and self.worker_thread.is_alive():
            messagebox.showinfo("Busy", "A transcription job is running. "
                                "Wait for it to finish before testing.")
            return

        self._selftest_btn.configure(state="disabled", text="\u231b Testing\u2026")
        self._selftest_status.configure(
            text=f"Running {SELFTEST_CHECK_COUNT} checks\u2026",
            foreground=C.TEXT_TERTIARY)
        self._selftest_thread = threading.Thread(
            target=self._selftest_worker, daemon=True)
        self._selftest_thread.start()
        self._poll_selftest()

    def _selftest_worker(self) -> None:
        """Background thread: run the self-test and store the result."""
        self._selftest_result: Optional[SelfTestResult] = None
        self._selftest_result = run_selftest(
            self._SelfTestHost(),
            backend=self.backend_var.get(),
            device=self.device_var.get(),
            compute_type=self.compute_type_var.get(),
        )

    def _poll_selftest(self) -> None:
        """Poll for self-test completion from the main thread."""
        if self._selftest_thread and self._selftest_thread.is_alive():
            self.after(200, self._poll_selftest)
            return
        # Done — restore button
        self._selftest_btn.configure(state="normal", text="\u25b6 Run self-test")
        result = getattr(self, "_selftest_result", None)
        if result is None:
            self._selftest_status.configure(text="Error", foreground=C.LOG_FAIL_FG)
            return
        passed_count = sum(1 for c in result.checks if c.startswith("[OK]"))
        total_checks = sum(1 for c in result.checks
                           if c.startswith("[OK]") or c.startswith("[FAIL]"))
        if result.passed:
            self._selftest_status.configure(
                text=f"{passed_count}/{total_checks} passed ({result.elapsed:.1f}s)",
                foreground=C.LOG_PASS_FG, cursor="hand2")
        else:
            self._selftest_status.configure(
                text=f"{passed_count}/{total_checks} passed \u2014 see details",
                foreground=C.LOG_FAIL_FG, cursor="hand2")
            # Auto-open detail dialog on failure
            self._show_selftest_result(result)
        # Make status clickable to (re-)view details
        self._selftest_status.bind(
            "<Button-1>", lambda e: self._show_selftest_result(result))

    def _show_selftest_result(self, result: SelfTestResult) -> None:
        """Show a dialog with the full self-test report."""
        dlg = tk.Toplevel(self)
        title = "Self-test passed" if result.passed else "Self-test failed"
        dlg.title(title)
        dlg.resizable(True, True)
        dlg.configure(background=C.BG_PRIMARY)
        dlg.transient(self._adv_dialog)
        if self._icon_path:
            dlg.iconbitmap(self._icon_path)

        frame = ttk.Frame(dlg, padding=16)
        frame.pack(fill="both", expand=True)

        color = C.LOG_PASS_FG if result.passed else C.LOG_FAIL_FG
        ttk.Label(frame, text=title, font=self.FONT_SECTION,
                  foreground=color).pack(anchor="w")

        # Explicit width/height: an unbounded Text requests 24 rows, which
        # pushed the Close button below the old fixed 480x320 dialog.
        text = tk.Text(frame, wrap="word", font=self.FONT_SMALL,
                       background=C.BG_CARD, foreground=C.TEXT_HEADING,
                       relief="flat", borderwidth=0, padx=8, pady=8,
                       width=64, height=14)
        text.pack(fill="both", expand=True, pady=(8, 0))
        report = "\n".join(result.checks)
        if result.error:
            report += f"\n\nDetails:\n{result.error}"
        text.insert("1.0", report)
        text.configure(state="disabled")

        self._make_button(frame, text="Close",
                          command=dlg.destroy).pack(anchor="e", pady=(8, 0))

        # Size to content, centered on the Advanced dialog, clamped to the
        # primary screen (Tk can't see other monitors' bounds on Windows).
        dlg.update_idletasks()
        scr_w, scr_h = self.winfo_screenwidth(), self.winfo_screenheight()
        w = min(max(self._s(480), dlg.winfo_reqwidth()), scr_w - self._s(40))
        h = min(dlg.winfo_reqheight(), scr_h - self._s(80))
        parent = self._adv_dialog
        x = parent.winfo_x() + (parent.winfo_width() - w) // 2
        y = parent.winfo_y() + (parent.winfo_height() - h) // 2
        if 0 <= parent.winfo_x() < scr_w and 0 <= parent.winfo_y() < scr_h:
            x = max(0, min(x, scr_w - w))
            y = max(0, min(y, scr_h - h))
        dlg.geometry(f"{w}x{h}+{x}+{y}")

    def _show_stats(self) -> None:
        """Reveal the run details panel (hidden until first run). Hides idle/recording panels."""
        self._hide_idle_panel()
        if self._recording_panel.winfo_manager():
            self._recording_panel.pack_forget()
            self._restore_pre_recording_sash()
        if not self._run_details_frame.winfo_manager():
            self._run_details_frame.pack(fill="both", expand=True)

    @staticmethod
    def _bidi_display(text: str, rtl: bool) -> str:
        """Wrap each line in RLM marks for display when the text is RTL.

        Tk resolves neutral characters (. , ! ? …) by the PARAGRAPH direction,
        which is always LTR — so a sentence-final period on a Hebrew line
        rendered on the wrong (right) side. Surrounding RLMs make trailing/
        leading neutrals resolve as RTL. Display-only: Copy/Save strip them.
        """
        if not rtl:
            return text
        rlm = chr(0x200F)  # RIGHT-TO-LEFT MARK (invisible)
        return "\n".join(
            rlm + line + rlm if line.strip() else line
            for line in text.split("\n"))

    def _widen_for_recording(self) -> None:
        """Give the dictation panel real width: the default split favors the
        queue and left the live transcript a hard-to-read ~20% column."""
        try:
            total = self._paned.winfo_width()
            cur = self._paned.sashpos(0)
            if total > 50 and (total - cur) < int(total * 0.42):
                self._pre_recording_sash = cur
                self._set_panel_ratio(0.55)
            else:
                self._pre_recording_sash = None
        except Exception:
            self._pre_recording_sash = None

    def _restore_pre_recording_sash(self) -> None:
        """Undo the sash widening done for the dictation panel — but only if
        the sash is still where recording put it (never clobber a user drag)."""
        prev = getattr(self, "_pre_recording_sash", None)
        self._pre_recording_sash = None
        if prev is None:
            return
        try:
            total = self._paned.winfo_width()
            if total > 50 and abs(self._paned.sashpos(0) - int(total * 0.55)) <= self._s(24):
                self._paned.sashpos(0, prev)
                self._reposition_grip()
        except Exception:
            pass

    def _hide_idle_panel(self) -> None:
        """Hide the idle onboarding panel."""
        if self._idle_panel.winfo_manager():
            self._idle_panel.pack_forget()
        self._run_subtitle.configure(text="Live progress and preview will appear here")

    def _show_idle_panel(self) -> None:
        """Show the idle onboarding panel, hide run details."""
        if not self._idle_panel.winfo_manager():
            self._idle_panel.pack(fill="both", expand=True)
        self._run_subtitle.configure(text="")

    def _init_panel_ratio_once(self, _event=None) -> None:
        """One-shot: set the idle 60/40 split as soon as the paned window
        has real width (a plain <Map> can fire while width is still 1)."""
        if self._panel_ratio_initialized or self._paned.winfo_width() <= 50:
            return
        self._panel_ratio_initialized = True
        self._set_panel_ratio(0.6)

    def _apply_panel_ratio(self, default: float) -> None:
        """State-transition ratio changes honor a user-dragged sash."""
        ratio = self._user_panel_ratio
        self._set_panel_ratio(ratio if ratio is not None else default)

    def _clamp_sash(self, sash_x: int, total: int) -> int:
        """Keep both panes usable: the queue toolbar (buttons + reorder
        cluster) needs ~600px before controls collide, and the run panel
        needs room to breathe."""
        return max(self._s(600), min(sash_x, total - self._s(360)))

    def _set_panel_ratio(self, left_pct: float) -> None:
        """Set the paned window sash position as a percentage for the left pane."""
        try:
            total = self._paned.winfo_width()
            if total < 50:
                return  # window not yet mapped
            sash_x = self._clamp_sash(int(total * left_pct), total)
            self._paned.sashpos(0, sash_x)
            self._reposition_grip()
        except Exception:
            pass  # sash not yet available

    # --- Sash pill grip (panel splitter affordance) ---

    def _build_sash_grip(self) -> None:
        """Create a small pill-shaped grip indicator centered on the paned sash.

        Apple Split View pattern: invisible seam, floating capsule grip,
        cursor change on hover.  The pill is drawn as a single Canvas line
        with round capstyle (produces a perfect capsule with zero polygon math).
        """
        _s = self._s

        self._grip_w = _s(8)   # cached for fast reposition
        self._grip_h = _s(32)
        self._grip_heading_offset = _s(20)
        self._grip_canvas = tk.Canvas(
            self._paned, width=self._grip_w, height=self._grip_h,
            background=C.BG_PRIMARY, highlightthickness=0, borderwidth=0,
            takefocus=0,  # prevent tab-navigation focus ring
        )
        self._drag_paned_x0 = 0  # initialise; set properly on Button-1
        # Draw pill: vertical line with round caps
        cx = self._grip_w // 2
        pad_y = _s(4)  # inset from canvas top/bottom
        self._grip_pill_id = self._grip_canvas.create_line(
            cx, pad_y, cx, self._grip_h - pad_y,
            fill=C.SASH_GRIP, width=_s(3), capstyle="round",
        )

        # Hover feedback
        self._grip_canvas.bind("<Enter>", self._on_grip_enter)
        self._grip_canvas.bind("<Leave>", self._on_grip_leave)

        # Drag forwarding: clicks on the pill must drag the sash
        self._grip_canvas.bind("<Button-1>", self._on_grip_press)
        self._grip_canvas.bind("<B1-Motion>", self._on_grip_drag)
        self._grip_canvas.bind("<ButtonRelease-1>", self._on_grip_release)

        # Reposition on every layout change — called synchronously because the
        # method is just five cheap tkinter reads + one place() call.
        # An earlier debounce via after_idle caused the pill to freeze during
        # continuous window resize (idle never fires until resize stops).
        self._paned.bind("<Configure>", lambda e: self._reposition_grip(), add="+")

        # Initial position (deferred — geometry not yet computed at build time)
        self.after(100, self._reposition_grip)

    def _reposition_grip(self) -> None:
        """Move the pill grip Canvas to sit centered on the current sash position.

        Called synchronously on every <Configure> and after sash moves.
        Uses cached grip dimensions to keep the call as cheap as possible.
        """
        try:
            sash_x = self._paned.sashpos(0)
            paned_h = self._paned.winfo_height()
            if paned_h < 50:
                return  # window minimised or not yet mapped
            # Offset downward by half the heading area so the pill
            # centers on the content zone, not the full panel height.
            y = (paned_h - self._grip_h) // 2 + self._grip_heading_offset
            # Clamp to keep pill within the paned area
            y = max(0, min(y, paned_h - self._grip_h))
            self._grip_canvas.place(
                in_=self._paned,
                x=sash_x - self._grip_w // 2,
                y=y,
            )
        except Exception:
            pass  # sash or widget not yet available

    def _on_grip_enter(self, _event) -> None:
        self._grip_canvas.configure(background=C.SASH_BG_HOVER, cursor="sb_h_double_arrow")
        self._grip_canvas.itemconfigure(self._grip_pill_id, fill=C.SASH_GRIP_HOVER)

    def _on_grip_leave(self, _event) -> None:
        self._grip_canvas.configure(background=C.BG_PRIMARY, cursor="")
        self._grip_canvas.itemconfigure(self._grip_pill_id, fill=C.SASH_GRIP)

    def _on_grip_press(self, event) -> None:
        """Cache paned x-origin at drag start for smooth tracking."""
        self._drag_paned_x0 = self._paned.winfo_rootx()

    def _on_grip_drag(self, event) -> None:
        """Forward drag from the pill Canvas to the PanedWindow sash."""
        x = event.x_root - self._drag_paned_x0
        x = self._clamp_sash(x, self._paned.winfo_width())
        self._paned.sashpos(0, x)
        self._reposition_grip()

    def _on_grip_release(self, _event) -> None:
        self._reposition_grip()
        # Remember the user's chosen split so state transitions honor it.
        try:
            total = self._paned.winfo_width()
            if total > 50:
                self._user_panel_ratio = self._paned.sashpos(0) / total
        except Exception:
            pass

    def _update_statusbar_file_info(self) -> None:
        """Update the right side of the status bar with file count + total duration."""
        n = len(self.file_paths)
        if n == 0:
            self._statusbar_right.config(text="")
            return
        dur_str = self._batch_duration_str()
        text = f"{n} file{'s' if n != 1 else ''}"
        if dur_str:
            text += f"  \u00b7  {dur_str}"
        self._statusbar_right.config(text=text)

    # --- Speed presets ---------------------------------------------------

    SPEED_PRESETS = {
        "fast":     {"beam_size": 1, "vad_filter": True,  "condition_on_previous_text": False, "batch_size": 0},
        "balanced": {"beam_size": 3, "vad_filter": True,  "condition_on_previous_text": False, "batch_size": 0},
        "quality":  {"beam_size": 5, "vad_filter": True,  "condition_on_previous_text": True,  "batch_size": 0},
    }

    PRESET_DESCRIPTIONS = {
        "fast":     "Beam=1, VAD on, no context carry  --  fastest",
        "balanced": "Beam=3, VAD on  --  good accuracy/speed trade-off",
        "quality":  "Beam=5, VAD on, context carry  --  best accuracy",
        "custom":   "Manual control over all parameters",
    }

    _DIARIZE_DEVICE_DISPLAY = {"auto": "Auto (MPS)", "cpu": "CPU"}

    @staticmethod
    def _diarize_enabled_initial(config_value: object,
                                 engine_available: bool) -> bool:
        """Initial diarize-checkbox state: the saved preference, forced off
        when the engine is absent (a stale saved True must never yield a
        checked-but-hidden checkbox that blocks Start via readiness)."""
        return bool(config_value) and engine_available

    def _on_diarize_device_display_changed(self) -> None:
        """Translate the device display name back to its config code."""
        display = self._diarize_device_display_var.get()
        for code, name in self._DIARIZE_DEVICE_DISPLAY.items():
            if name == display:
                self.diarize_device_var.set(code)
                break

    def _apply_speed_preset(self, _event=None) -> None:
        """Apply the selected speed preset to the advanced controls."""
        preset = self.speed_preset_var.get()
        desc = self.PRESET_DESCRIPTIONS.get(preset, "")
        if hasattr(self, "_inline_preset_desc"):
            self._inline_preset_desc.configure(text=desc)
        vals = self.SPEED_PRESETS.get(preset)
        if vals is None:
            return  # "custom" -- leave everything alone
        self.beam_size_var.set(str(vals["beam_size"]))
        self.vad_var.set(vals["vad_filter"])
        self.condition_on_previous_text_var.set(vals["condition_on_previous_text"])
        self.batch_size_var.set(str(vals["batch_size"]))

    def _detect_preset_from_values(self) -> str:
        """Return the preset name matching current advanced values, or 'custom'."""
        try:
            current = {
                "beam_size": int(self.beam_size_var.get()),
                "vad_filter": self.vad_var.get(),
                "condition_on_previous_text": self.condition_on_previous_text_var.get(),
                "batch_size": int(self.batch_size_var.get()),
            }
        except (ValueError, TypeError):
            return "custom"
        for name, vals in self.SPEED_PRESETS.items():
            if current == vals:
                return name
        return "custom"

    def _on_language_display_changed(self) -> None:
        """Translate the human-readable language display back to ISO code."""
        display = self._lang_display_var.get()
        # Reverse lookup: find the ISO code for this display name
        for code, name in LANGUAGE_DISPLAY.items():
            if name == display:
                self.language_var.set(code)
                break
        self._on_language_changed()

    def _on_language_changed(self) -> None:
        """Auto-select the best model when the user changes the language.

        Rebuilds the model list (language-matching first), then picks the
        best model: saved preference > recommended > any matching affinity.
        """
        lang = self.language_var.get().strip()
        # Rebuild model list so language-matching models come first
        self.refresh_models()

        labels = list(self.model_map.keys())
        if not labels:
            return

        # 1. Saved per-language preference
        saved = self.config_data.get(f"last_model_{lang}", "")
        if saved and saved in labels:
            self.model_var.set(saved)
            return

        # 2. Recommended model for this language
        recommended_substr = RECOMMENDED_MODEL.get(lang, "")
        if recommended_substr:
            for label in labels:
                if recommended_substr in label:
                    self.model_var.set(label)
                    return

        # 3. Any model with matching language affinity
        for label in labels:
            if model_lang(label) == lang:
                self.model_var.set(label)
                return

        # 4. For 'auto' or no match, keep current selection

    def _update_banner(self, state: str, text: str = "") -> None:
        """Update the state banner: color + text based on app state.

        Uses a brief fade animation when transitioning between states.
        """
        color_map = {
            "idle":     (self.CLR_IDLE_BG,    self.CLR_IDLE_FG),
            "ready":    (self.CLR_IDLE_BG,    self.CLR_IDLE_FG),
            "running":  (self.CLR_RUNNING_BG, self.CLR_RUNNING_FG),
            "paused":   (self.CLR_PAUSED_BG,  self.CLR_PAUSED_FG),
            "complete": (self.CLR_SUCCESS_BG,  self.CLR_SUCCESS_FG),
            "error":    (self.CLR_ERROR_BG,    self.CLR_ERROR_FG),
        }
        target_bg, fg = color_map.get(state, color_map["idle"])
        if not self._banner_frame.winfo_manager():
            # Always packed (invisible strip when idle) so the banner's
            # appearance never shifts the whole layout down by a row.
            self._banner_frame.pack(fill="x", pady=(4, 0), in_=self._header_frame)
        if state == "idle" and not text:
            self._banner_last_state = "idle"
            self._banner_frame.configure(background=C.BG_PRIMARY)
            self._banner_label.configure(text="", background=C.BG_PRIMARY)
            return
        self._banner_label.configure(text=text, foreground=fg)
        # Animate background transition
        prev_state = getattr(self, "_banner_last_state", "idle")
        self._banner_last_state = state
        if prev_state != state:
            self._animate_banner_bg(target_bg, steps=4, interval=40)
        else:
            self._banner_frame.configure(background=target_bg)
            self._banner_label.configure(background=target_bg)

    def _animate_banner_bg(self, target_hex: str, steps: int = 4, interval: int = 40) -> None:
        """Fade the banner background to target color over `steps` frames."""
        try:
            current_hex = self._banner_frame.cget("background")
            cr, cg, cb = self._hex_to_rgb(current_hex)
            tr, tg, tb = self._hex_to_rgb(target_hex)
        except (ValueError, TypeError):
            self._banner_frame.configure(background=target_hex)
            self._banner_label.configure(background=target_hex)
            return
        for i in range(1, steps + 1):
            frac = i / steps
            r = int(cr + (tr - cr) * frac)
            g = int(cg + (tg - cg) * frac)
            b = int(cb + (tb - cb) * frac)
            color = f"#{r:02x}{g:02x}{b:02x}"
            self.after(i * interval, self._set_banner_color, color)

    def _set_banner_color(self, color: str) -> None:
        """Apply a single color to both banner frame and label."""
        try:
            self._banner_frame.configure(background=color)
            self._banner_label.configure(background=color)
        except Exception:
            pass

    @staticmethod
    def _hex_to_rgb(hex_color: str) -> tuple:
        """Convert a hex color string to (r, g, b) tuple."""
        h = hex_color.lstrip("#")
        return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))

    def _count_by_status(self) -> dict:
        """Scan the tree and return counts by status: Queued, Done, Failed, Running, Cancelled."""
        counts = {"Queued": 0, "Done": 0, "Failed": 0, "Running": 0, "Cancelled": 0}
        for item_id in self.file_items.values():
            if not self.file_tree.exists(item_id):
                continue
            status = self.file_tree.item(item_id, "values")[1]
            if status in counts:
                counts[status] += 1
        return counts

    def _update_contextual_banner(self, worker_finished: bool = False) -> None:
        """Update banner and start button to reflect the queue's current state.

        Called after any mutation that changes file statuses outside of a running
        job: _append_files, remove_selected_files, clear_files, _retry_failed,
        and the 'done' event handler.

        worker_finished=True bypasses the is_alive() guard: during the 'done'
        event drain the worker thread may still be tearing down (model unload
        can outlast the 120 ms poll), yet the run is logically over and the
        Continue button text must be set now, not never.
        """
        if not worker_finished and self.worker_thread and self.worker_thread.is_alive():
            return  # running state is managed by _apply_app_state("running")

        counts = self._count_by_status()
        # Start requeues Cancelled rows (see _requeue_cancelled), so they count
        # toward what a Start/Continue press would actually run.
        queued = counts["Queued"] + counts["Cancelled"]
        done = counts["Done"]
        failed = counts["Failed"]
        has_history = done > 0 or failed > 0

        if queued > 0 and has_history:
            parts = []
            if done > 0:
                parts.append(f"{done} done")
            if failed > 0:
                parts.append(f"{failed} failed")
            if counts["Cancelled"] > 0:
                parts.append(f"{counts['Cancelled']} cancelled")
            if counts["Queued"] > 0:
                parts.append(f"{counts['Queued']} queued")
            self._update_banner("ready", " \u00b7 ".join(parts) + " \u2014 ready to continue")
            self.title(APP_TITLE)
            self._taskbar.clear()
            self.start_button.config(text=f"  Continue ({queued} file{'s' if queued != 1 else ''})  ")
        elif queued > 0:
            self._update_banner("idle", f"{queued} file{'s' if queued != 1 else ''} queued \u2014 ready to start")
            self.title(APP_TITLE)
            self._taskbar.clear()
            self.start_button.config(text="  Start transcription  ")
        elif done > 0 and failed == 0:
            # All done, nothing pending — keep completion banner
            pass
        elif failed > 0:
            # Has failures, nothing queued — keep error banner
            pass
        else:
            # Empty queue
            self._apply_app_state("idle")
            self.start_button.config(text="  Start transcription  ")

    def _apply_app_state(self, state: str, **kwargs) -> None:
        """Transform the UI based on application state: 'idle', 'running', or 'complete'.

        kwargs for 'running': idx (int), total (int), filename (str), file_pct (float)
        kwargs for 'complete': message (str), done (int), failed (int), wall_seconds (float),
                               cancelled (bool), stopped (bool)
        """
        if state == "running":
            idx = kwargs.get("idx", 0)
            total = kwargs.get("total", 0)
            filename = kwargs.get("filename", "")
            file_pct = kwargs.get("file_pct", -1)
            if idx == 0 and file_pct < 0:
                # Pre-file phase (model load/download; file_started always has
                # idx >= 1) \u2014 "Transcribing 0/N: loading model..." reads like a
                # stall, so name the phase instead.
                self.title(f"Loading model\u2026 \u2014 {APP_TITLE}")
                self._taskbar.set_state(TaskbarProgress.TBPF_NORMAL)
                self._update_banner("running", "Loading model\u2026")
                self._hide_idle_panel()
                self._apply_panel_ratio(0.40)
                return
            pct_str = f" \u00b7 {file_pct:.0f}%" if file_pct >= 0 else ""
            filename = bidi_name(filename)  # Hebrew names + digits scramble without an RTL base
            self.title(f"[{idx}/{total}{pct_str}] {filename} \u2014 {APP_TITLE}")
            self._taskbar.set_state(TaskbarProgress.TBPF_NORMAL)
            if total > 0:
                done_frac = max(0, idx - 1) + (file_pct / 100.0 if file_pct >= 0 else 0)
                self._taskbar.set_progress(int(done_frac * 1000), total * 1000)
            # Banner: show running state with file info
            banner_text = f"Transcribing {idx}/{total}: {filename}" + (f" ({file_pct:.0f}%)" if file_pct >= 0 else "")
            self._update_banner("running", banner_text)
            # Hide idle panel if visible; shift panels to favor run details
            self._hide_idle_panel()
            self._apply_panel_ratio(0.40)
        elif state == "complete":
            done = kwargs.get("done", 0)
            failed = kwargs.get("failed", 0)
            wall = kwargs.get("wall_seconds", 0.0)
            wall_str = format_hms(wall) if wall > 0 else ""
            if kwargs.get("cancelled") or kwargs.get("stopped"):
                # Interrupted by the user — neutral render, never the green
                # success banner (and not the error one: nothing went wrong).
                word = "Cancelled" if kwargs.get("cancelled") else "Stopped"
                title_tag = f"[{word}: {done} done]"
                if wall_str:
                    title_tag += f" ({wall_str})"
                self.title(f"{title_tag} — {APP_TITLE}")
                self._taskbar.clear()
                banner_text = f"{word}: {done} file{'s' if done != 1 else ''} done"
                if failed > 0:
                    banner_text += f", {failed} failed"
                self._update_banner("paused", banner_text)
                self._notify_completion(kwargs.get("message", "Transcription complete."))
                self._apply_panel_ratio(0.50)
                return
            if failed > 0:
                title_tag = f"[Done: {done}, {failed} failed]"
            else:
                title_tag = f"[Done: {done}]"
            if wall_str:
                title_tag += f" ({wall_str})"
            self.title(f"{title_tag} \u2014 {APP_TITLE}")
            if failed > 0:
                self._taskbar.set_state(TaskbarProgress.TBPF_ERROR)
                self._taskbar.set_progress(done * 1000, (done + failed) * 1000)
                banner_text = f"Completed: {done} done, {failed} failed"
                if wall_str:
                    banner_text += f" ({wall_str})"
                self._update_banner("error", banner_text)
            else:
                self._taskbar.set_progress(1000, 1000)
                self._taskbar.clear()
                banner_text = f"Completed: {done} file{'s' if done != 1 else ''} transcribed"
                if wall_str:
                    banner_text += f" in {wall_str}"
                self._update_banner("complete", banner_text)
            self._notify_completion(kwargs.get("message", "Transcription complete."))
            self._apply_panel_ratio(0.50)  # balanced on completion
        elif state == "failed":
            # Fatal run failure (setup/model error or an unexpected worker crash).
            # Render an explicit error state — never the green success banner that
            # the 'complete' path shows when failed == 0.
            message = kwargs.get("message", "Run failed.")
            wall = kwargs.get("wall_seconds", 0.0)
            wall_str = format_hms(wall) if wall > 0 else ""
            title_tag = "[Failed]" + (f" ({wall_str})" if wall_str else "")
            self.title(f"{title_tag} — {APP_TITLE}")
            self._taskbar.set_state(TaskbarProgress.TBPF_ERROR)
            banner_text = "Run failed — see the log for details"
            if wall_str:
                banner_text += f" ({wall_str})"
            self._update_banner("error", banner_text)
            # First line only: the message may carry a full traceback (already
            # in the raw log), which must not be pushed into an OS notification.
            self._notify_completion(message.split("\n", 1)[0].strip() or "Run failed.")
            self._apply_panel_ratio(0.50)
        else:  # idle
            self.title(APP_TITLE)
            self._taskbar.clear()
            self._update_banner("idle")

    def _notify_completion(self, message: str) -> None:
        """Send an OS-level notification if the window is not focused.

        The *message* typically includes completion stats
        (e.g. "Done. 3 files\\nAudio: 01:23 | Real time: 00:15 | Speed: 5.5x").
        """
        import sys as _sys
        try:
            if self.focus_displayof() is not None:
                return  # app is focused, messagebox is enough
        except Exception:
            logger.debug("Could not check window focus", exc_info=True)

        # Flatten to a single line for notification systems that
        # don't handle newlines well.
        flat = message.replace("\n", "  —  ").strip()

        # Windows: flash taskbar + sound + balloon tip
        if _sys.platform == "win32":
            try:
                import ctypes
                # Flash the app's taskbar button. GetConsoleWindow() is NULL in
                # the windowed build (pythonw / PyInstaller windowed exe); the
                # app HWND comes from GetParent(winfo_id()), same as
                # TaskbarProgress in widgets.py.
                hwnd = ctypes.windll.user32.GetParent(self.winfo_id())
                if hwnd:
                    ctypes.windll.user32.FlashWindow(hwnd, True)
                import winsound
                winsound.MessageBeep(winsound.MB_ICONASTERISK)
            except Exception:
                logger.debug("Windows flash/sound failed", exc_info=True)
            # Best-effort toast via PowerShell (.NET BalloonTip)
            try:
                ps_title = APP_TITLE.replace("'", "''")
                ps_msg = flat.replace("'", "''")
                ps_script = (
                    "[void][System.Reflection.Assembly]"
                    "::LoadWithPartialName('System.Windows.Forms');"
                    "$n = New-Object System.Windows.Forms.NotifyIcon;"
                    "$n.Icon = [System.Drawing.SystemIcons]::Information;"
                    "$n.Visible = $true;"
                    f"$n.ShowBalloonTip(5000, '{ps_title}', '{ps_msg}', "
                    "[System.Windows.Forms.ToolTipIcon]::Info);"
                    "Start-Sleep -Seconds 6; $n.Dispose()"
                )
                subprocess.Popen(
                    ["powershell", "-WindowStyle", "Hidden", "-Command", ps_script],
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
            except Exception:
                logger.debug("Windows balloon tip failed", exc_info=True)
            return
        # macOS: osascript notification
        elif _sys.platform == "darwin":
            try:
                safe = flat.replace("\\", "\\\\").replace('"', '\\"')
                subprocess.Popen([
                    "osascript", "-e",
                    f'display notification "{safe}" with title "{APP_TITLE}"',
                ])
                return
            except Exception:
                logger.debug("macOS notification failed", exc_info=True)
        # Linux: notify-send
        else:
            try:
                subprocess.Popen(["notify-send", APP_TITLE, flat])
                return
            except Exception:
                logger.debug("Linux notification failed", exc_info=True)
        # Universal fallback: bell
        try:
            self.bell()
        except Exception:
            logger.debug("Bell fallback failed", exc_info=True)

    def _setup_dnd(self, target: tk.Widget) -> None:
        """Try to enable native drag-and-drop. Logs install hint if tkinterdnd2 is absent."""
        try:
            import tkinterdnd2
            # TranscriberApp subclasses tk.Tk, so the tkdnd Tcl package is never
            # auto-loaded (that only happens for a TkinterDnD.Tk root). Load it
            # explicitly once — otherwise drop_target_register raises TclError on
            # every platform and drag-and-drop silently does nothing.
            if not getattr(self, "_tkdnd_loaded", False):
                tkinterdnd2.TkinterDnD._require(self)
                self._tkdnd_loaded = True
            target.drop_target_register("DND_Files")  # type: ignore[attr-defined]
            target.dnd_bind("<<Drop>>", self._on_dnd_drop)  # type: ignore[attr-defined]
            logger.debug("Drag-and-drop enabled via tkinterdnd2")
        except ImportError:
            logger.info("Drag-and-drop disabled: install tkinterdnd2 to enable "
                        "(pip install tkinterdnd2)")
        except Exception:
            logger.debug("Drag-and-drop setup failed (tkinterdnd2 present but unsupported)",
                         exc_info=True)

    def _on_dnd_drop(self, event) -> None:
        """Handle files dropped via native drag-and-drop."""
        raw = event.data if hasattr(event, "data") else ""
        # tkinterdnd2 returns space-separated paths; paths with spaces are wrapped in {}
        files = []
        if "{" in raw:
            files = re.findall(r"\{([^}]+)\}", raw)
            remaining = re.sub(r"\{[^}]+\}", "", raw).strip()
            if remaining:
                files.extend(remaining.split())
        else:
            files = raw.split()
        audio_files = [f for f in files if Path(f).suffix.lower() in AUDIO_EXTENSIONS and Path(f).is_file()]
        if audio_files:
            self._append_files(audio_files)

    def _update_empty_state(self) -> None:
        """Show/hide the empty-state overlay depending on whether files exist."""
        if self.file_paths:
            self._empty_frame.grid_remove()
        else:
            self._empty_frame.grid()
            self._empty_frame.lift()

    # --- Right-click context menu for file tree ---

    def _on_tree_right_click(self, event) -> None:
        """Show context menu on right-click and select the row under cursor."""
        row_id = self.file_tree.identify_row(event.y)
        if row_id:
            # Select the right-clicked row if not already selected
            if row_id not in self.file_tree.selection():
                self.file_tree.selection_set(row_id)
            # Enable/disable "Show error" based on whether selection has errors
            selected = self.file_tree.selection()
            item_to_path = {v: k for k, v in self.file_items.items()}
            has_error = any(
                item_to_path.get(item, "") in self.file_errors
                for item in selected
            )
            self._tree_menu.entryconfig("Show error", state="normal" if has_error else "disabled")
            # Enable/disable move actions based on selection
            has_queued = any(
                self.file_tree.item(item, "values")[1] == "Queued"
                for item in selected
            )
            move_state = "normal" if has_queued else "disabled"
            for label in ("Move to top", "Move up", "Move down", "Move to bottom"):
                self._tree_menu.entryconfig(label, state=move_state)
            # Enable "Retry failed" only when failed/cancelled rows exist anywhere
            retryable = any(
                self.file_tree.exists(iid)
                and self.file_tree.item(iid, "values")[1] in ("Failed", "Cancelled")
                for iid in self.file_items.values()
            )
            self._tree_menu.entryconfig("Retry failed",
                                        state="normal" if retryable else "disabled")
            self._tree_menu.tk_popup(event.x_root, event.y_root)

    def _show_file_error(self) -> None:
        """Display the stored error message for the selected file."""
        selected = self.file_tree.selection()
        if not selected:
            return
        item_to_path = {v: k for k, v in self.file_items.items()}
        path_str = item_to_path.get(selected[0], "")
        error = self.file_errors.get(path_str)
        if error:
            messagebox.showerror(f"Error: {Path(path_str).name}", error)
        else:
            messagebox.showinfo("No error", "No error recorded for this file.")

    def _on_tree_double_click(self, event) -> None:
        """Double-click on a failed row shows its full error."""
        row_id = self.file_tree.identify_row(event.y)
        if not row_id:
            return
        item_to_path = {v: k for k, v in self.file_items.items()}
        path_str = item_to_path.get(row_id, "")
        if path_str and path_str in self.file_errors:
            messagebox.showerror(f"Error: {Path(path_str).name}", self.file_errors[path_str])

    def _on_tree_hover(self, event) -> None:
        """Highlight the row under cursor with a subtle tint."""
        row_id = self.file_tree.identify_row(event.y)
        if row_id == self._hover_row_id:
            return
        # Remove hover from previous row
        if self._hover_row_id and self.file_tree.exists(self._hover_row_id):
            tags = self.file_tree.item(self._hover_row_id, "tags")
            new_tags = tuple(t for t in tags if t != "hover")
            self.file_tree.item(self._hover_row_id, tags=new_tags)
        # Apply hover to new row
        self._hover_row_id = row_id
        if row_id:
            tags = self.file_tree.item(row_id, "tags")
            if "hover" not in tags:
                self.file_tree.item(row_id, tags=tags + ("hover",))

    def _on_tree_leave(self, event) -> None:
        """Remove hover highlight when cursor leaves the tree."""
        if self._hover_row_id and self.file_tree.exists(self._hover_row_id):
            tags = self.file_tree.item(self._hover_row_id, "tags")
            new_tags = tuple(t for t in tags if t != "hover")
            self.file_tree.item(self._hover_row_id, tags=new_tags)
        self._hover_row_id = None

    def _retry_failed(self) -> None:
        """Re-queue all 'Failed' or 'Cancelled' files and re-run transcription."""
        failed_paths = []
        for path_str, item_id in self.file_items.items():
            if not self.file_tree.exists(item_id):
                continue
            status = self.file_tree.item(item_id, "values")[1]
            if status in ("Failed", "Cancelled"):
                failed_paths.append(path_str)
        if not failed_paths:
            messagebox.showinfo("Nothing to retry", "No failed or cancelled files in the queue.")
            return
        # Reset failed files to Queued — next_file() will pick them up
        for path_str in failed_paths:
            self.set_tree_row(path_str, status="Queued", progress_text="")
            self.file_errors.pop(path_str, None)
        self.refresh_file_tree()
        self.update_overall_summary()
        self.log(f"Retrying {len(failed_paths)} failed file(s).")
        self.start_transcription()

    def _copy_file_path(self) -> None:
        """Copy the path(s) of all selected files to the clipboard, one per line."""
        selected = self.file_tree.selection()
        if not selected:
            return
        item_to_path = {v: k for k, v in self.file_items.items()}
        paths = [item_to_path.get(item, "") for item in selected]
        paths = [p for p in paths if p]
        if paths:
            self.clipboard_clear()
            self.clipboard_append("\n".join(paths))
            n = len(paths)
            self.log(f"Copied {n} file path{'s' if n != 1 else ''} to clipboard.")

    # --- Queue reorder ---

    def _move_selection_top(self) -> None:
        self._move_selected("top")

    def _move_selection_up(self) -> None:
        self._move_selected("up")

    def _move_selection_down(self) -> None:
        self._move_selected("down")

    def _move_selection_bottom(self) -> None:
        self._move_selected("bottom")

    def _is_movable_index(self, idx: int) -> bool:
        """Return True if the file at index idx has 'Queued' status."""
        path_str = self.file_paths[idx]
        item_id = self.file_items.get(path_str)
        if not item_id or not self.file_tree.exists(item_id):
            return False
        return self.file_tree.item(item_id, "values")[1] == "Queued"

    def _move_selected(self, direction: str) -> None:
        """Move selected Queued rows in the given direction.

        direction: 'top', 'up', 'down', 'bottom'.
        Non-Queued rows are silently ignored. Immovable rows block movement.
        """
        selected = self.file_tree.selection()
        if not selected:
            return

        # Map selected item_ids to their indices in file_paths
        item_to_path = {v: k for k, v in self.file_items.items()}
        path_to_idx = {p: i for i, p in enumerate(self.file_paths)}

        # Collect indices of selected Queued rows only
        movable_indices = []
        for item_id in selected:
            path = item_to_path.get(item_id)
            if path is None:
                continue
            idx = path_to_idx.get(path)
            if idx is None:
                continue
            if self._is_movable_index(idx):
                movable_indices.append(idx)

        if not movable_indices:
            return

        movable_indices.sort()
        movable_set = set(movable_indices)
        moving_paths = [self.file_paths[i] for i in movable_indices]

        with self._queue_lock:
            if direction == "up":
                # Process ascending: each row swaps with the row above if it's
                # not immovable. Track actual positions as we mutate.
                positions = list(movable_indices)  # current positions, updated after each swap
                for pi, pos in enumerate(positions):
                    target = pos - 1
                    if target < 0:
                        continue
                    # Check if target is occupied by another movable item (already moved)
                    if target in set(positions[:pi]):
                        continue  # blocked by sibling that couldn't move
                    # Check if target is immovable
                    target_path = self.file_paths[target]
                    target_item = self.file_items.get(target_path)
                    if target_item and self.file_tree.exists(target_item):
                        if self.file_tree.item(target_item, "values")[1] != "Queued":
                            continue  # immovable wall
                    self.file_paths[pos], self.file_paths[target] = self.file_paths[target], self.file_paths[pos]
                    positions[pi] = target

            elif direction == "down":
                # Process descending: each row swaps with the row below
                positions = list(movable_indices)
                for pi in range(len(positions) - 1, -1, -1):
                    pos = positions[pi]
                    target = pos + 1
                    if target >= len(self.file_paths):
                        continue
                    if target in set(positions[pi + 1:]):
                        continue  # blocked by sibling that couldn't move
                    target_path = self.file_paths[target]
                    target_item = self.file_items.get(target_path)
                    if target_item and self.file_tree.exists(target_item):
                        if self.file_tree.item(target_item, "values")[1] != "Queued":
                            continue  # immovable wall
                    self.file_paths[pos], self.file_paths[target] = self.file_paths[target], self.file_paths[pos]
                    positions[pi] = target

            elif direction == "top":
                remaining = [p for i, p in enumerate(self.file_paths) if i not in movable_set]
                # Insert after any leading immovable rows
                insert_at = 0
                for i, p in enumerate(remaining):
                    item_id = self.file_items.get(p)
                    if item_id and self.file_tree.exists(item_id):
                        if self.file_tree.item(item_id, "values")[1] == "Queued":
                            insert_at = i
                            break
                        else:
                            insert_at = i + 1
                self.file_paths = remaining[:insert_at] + moving_paths + remaining[insert_at:]

            elif direction == "bottom":
                remaining = [p for i, p in enumerate(self.file_paths) if i not in movable_set]
                # Insert before any trailing immovable rows
                insert_at = len(remaining)
                for i in range(len(remaining) - 1, -1, -1):
                    item_id = self.file_items.get(remaining[i])
                    if item_id and self.file_tree.exists(item_id):
                        if self.file_tree.item(item_id, "values")[1] == "Queued":
                            insert_at = i + 1
                            break
                        else:
                            insert_at = i
                self.file_paths = remaining[:insert_at] + moving_paths + remaining[insert_at:]
            else:
                return

        # Refresh and re-select
        moved_paths = moving_paths
        self.refresh_file_tree()
        self.update_overall_summary()
        # Re-select the moved rows
        new_selection = []
        for mp in moved_paths:
            item_id = self.file_items.get(mp)
            if item_id:
                new_selection.append(item_id)
        if new_selection:
            self.file_tree.selection_set(*new_selection)
            self.file_tree.see(new_selection[0])
        self._flash_rows(moved_paths)
        self._update_move_button_state()

    def _update_move_button_state(self) -> None:
        """Enable/disable move buttons based on selection state."""

        selected = self.file_tree.selection()
        if not selected:
            for btn in (self._btn_move_top, self._btn_move_up, self._btn_move_down, self._btn_move_bottom):
                self._set_button_enabled(btn, False)
            return

        # Map selected items to indices
        item_to_path = {v: k for k, v in self.file_items.items()}
        path_to_idx = {p: i for i, p in enumerate(self.file_paths)}
        queued_indices = []
        for item_id in selected:
            path = item_to_path.get(item_id)
            if path is None:
                continue
            idx = path_to_idx.get(path)
            if idx is not None and self._is_movable_index(idx):
                queued_indices.append(idx)

        if not queued_indices:
            for btn in (self._btn_move_top, self._btn_move_up, self._btn_move_down, self._btn_move_bottom):
                self._set_button_enabled(btn, False)
            return

        queued_indices.sort()

        # Check top boundary: can any selected queued row move up?
        can_up = False
        for qi in queued_indices:
            if qi > 0 and (self._is_movable_index(qi - 1) or (qi - 1) in queued_indices):
                # Not already at top or blocked
                can_up = True
                break
            elif qi > 0 and not self._is_movable_index(qi - 1) and (qi - 1) not in queued_indices:
                # Blocked by immovable wall
                continue
            # At index 0 — can't move up
        # Check top boundary more precisely: is the block already at the topmost possible position?
        first_possible = 0
        for i in range(len(self.file_paths)):
            if self._is_movable_index(i):
                first_possible = i
                break
            first_possible = i + 1
        at_top = queued_indices[0] <= first_possible

        # Check bottom boundary
        last_possible = len(self.file_paths) - 1
        for i in range(len(self.file_paths) - 1, -1, -1):
            if self._is_movable_index(i):
                last_possible = i
                break
            last_possible = i - 1
        at_bottom = queued_indices[-1] >= last_possible

        can_down = False
        for qi in reversed(queued_indices):
            if qi < len(self.file_paths) - 1 and (self._is_movable_index(qi + 1) or (qi + 1) in queued_indices):
                can_down = True
                break

        self._set_button_enabled(self._btn_move_top, not at_top)
        self._set_button_enabled(self._btn_move_up, can_up and not at_top)
        self._set_button_enabled(self._btn_move_down, can_down and not at_bottom)
        self._set_button_enabled(self._btn_move_bottom, not at_bottom)

    def _update_queue_button_state(self) -> None:
        """Enable/disable Remove and Clear based on selection and queue state."""
        has_files = bool(self.file_paths)
        has_selection = bool(self.file_tree.selection())
        is_running = self.worker_thread and self.worker_thread.is_alive()
        self._set_button_enabled(self._btn_remove, has_selection)
        self._set_button_enabled(self._btn_clear, has_files and not is_running)

    def _flash_rows(self, paths: list, color: str = C.HIGHLIGHT_FLASH, duration: int = 350) -> None:
        """Briefly highlight moved rows for visual feedback."""
        for path_str in paths:
            item_id = self.file_items.get(path_str)
            if item_id and self.file_tree.exists(item_id):
                tags = list(self.file_tree.item(item_id, "tags"))
                if "flash" not in tags:
                    tags.append("flash")
                    self.file_tree.item(item_id, tags=tuple(tags))
        self.after(duration, self._clear_flash)

    def _clear_flash(self) -> None:
        """Remove flash highlight from all rows."""
        for item_id in self.file_tree.get_children():
            tags = self.file_tree.item(item_id, "tags")
            if "flash" in tags:
                new_tags = tuple(t for t in tags if t != "flash")
                self.file_tree.item(item_id, tags=new_tags)

    def _wire_readiness_checks(self) -> None:
        """Set up traces so _check_readiness fires whenever relevant state changes."""
        for var in (self.backend_var, self.model_var, self.output_dir_var,
                    self.beam_size_var, self.batch_size_var,
                    self.diarize_var, self.num_speakers_var):
            var.trace_add("write", lambda *_: self._check_readiness())
        for fmt_var in self.format_vars.values():
            fmt_var.trace_add("write", lambda *_: self._check_readiness())
        # When advanced params change manually, detect if they still match a preset
        for var in (self.beam_size_var, self.batch_size_var):
            var.trace_add("write", lambda *_: self._sync_preset_from_values())
        self.vad_var.trace_add("write", lambda *_: self._sync_preset_from_values())
        self.condition_on_previous_text_var.trace_add("write", lambda *_: self._sync_preset_from_values())

    def _check_readiness(self) -> None:
        """Evaluate whether the app is ready to start, update label and button state."""
        if self.worker_thread and self.worker_thread.is_alive():
            return  # don't overwrite running-state controls

        issues: List[str] = []

        if not self.file_paths:
            issues.append("No audio files added")
        if not self.model_var.get() or not self.model_map.get(self.model_var.get(), ""):
            issues.append("No model selected")
        if not self.output_dir_var.get().strip():
            issues.append("No output folder")
        if not any(v.get() for v in self.format_vars.values()):
            issues.append("No output format selected")

        beam_text = self.beam_size_var.get()
        try:
            if int(beam_text) < 1:
                issues.append("Beam size must be >= 1")
        except (ValueError, TypeError):
            issues.append("Beam size must be an integer")

        batch_text = self.batch_size_var.get()
        try:
            bs = int(batch_text)
            if bs < 0:
                issues.append("Batch size must be >= 0")
        except (ValueError, TypeError):
            issues.append("Batch size must be an integer")

        # Belt-and-braces: the checkbox is hidden and forced off when the
        # engine is absent, so this only fires on programmatic mutation.
        if self.diarize_var.get() and not _HAS_PYANNOTE:
            issues.append("Speaker identification requires pyannote.audio "
                          "(not installed)")
        try:
            if int(self.num_speakers_var.get()) < 0:
                issues.append("Number of speakers must be >= 0")
        except (ValueError, TypeError):
            issues.append("Number of speakers must be an integer")

        if issues:
            self.readiness_var.set(" · ".join(issues))
            # Missing prerequisites are neutral guidance; only genuinely
            # invalid input (unparseable/out-of-range values) earns red.
            neutral = {"No audio files added", "No model selected",
                       "No output folder", "No output format selected"}
            invalid = [i for i in issues if i not in neutral]
            self.readiness_label.configure(
                foreground=C.STATUS_ERROR_FG if invalid else C.TEXT_TERTIARY)
            self._set_button_enabled(self.start_button, False)
        else:
            self.readiness_var.set("")
            self._set_button_enabled(self.start_button, True)

    def _sync_preset_from_values(self) -> None:
        """Update the preset dropdown to reflect whether current values match a preset."""
        detected = self._detect_preset_from_values()
        if detected != self.speed_preset_var.get():
            self.speed_preset_var.set(detected)
            desc = self.PRESET_DESCRIPTIONS.get(detected, "")
            if hasattr(self, "_inline_preset_desc"):
                self._inline_preset_desc.configure(text=desc)

    def _populate_backends(self) -> None:
        backends = []
        if is_package_available("whisper"):
            backends.append("openai-whisper")
        if is_package_available("faster_whisper"):
            backends.append("faster-whisper")
        if not backends:
            backends = ["faster-whisper", "openai-whisper"]

        self.backend_combo["values"] = backends
        preferred = self.config_data.get("last_backend")
        self.backend_var.set(preferred if preferred in backends else backends[0])

    # --- Card colors ---
    _CARD_SELECTED_BG = C.CARD_SELECTED_BG
    _CARD_SELECTED_BORDER = C.CARD_SELECTED_BORDER
    _CARD_HOVER_BG = C.CARD_HOVER_BG
    _CARD_DEFAULT_BG = C.CARD_DEFAULT_BG
    _CARD_DEFAULT_BORDER = C.CARD_DEFAULT_BORDER
    _CARD_RECOMMENDED_BG = C.CARD_RECOMMENDED_BG

    def refresh_models(self) -> None:
        """Rebuild model_map and model card widgets.

        Deduplicates models: if a model exists both locally and in the hub list,
        show it once with source='local'.  Hub-only models show source='hub'.
        """
        backend = self.backend_var.get()
        lang = self.language_var.get().strip()
        self.model_map = {}

        # Phase 1: collect raw data
        # local_raw: {friendly_name -> (label, actual_path)}
        # hub_raw:   {friendly_name -> (label, repo)}
        local_raw: dict[str, tuple[str, str]] = {}
        hub_raw: dict[str, tuple[str, str]] = {}

        if backend == "openai-whisper":
            cached = detect_openai_cached_models()
            for model in cached:
                label = friendly_label(model, "local")
                local_raw[model] = (label, model)
            for model in OPENAI_MODELS:
                if model not in cached:
                    label = friendly_label(model, "hub")
                    hub_raw[model] = (label, model)
        else:
            extra_roots = self.config_data.get("extra_model_roots", [])
            local_models = detect_faster_whisper_local_models(extra_roots)
            for label, actual in local_models:
                # Extract a dedup key from the label's friendly name
                friendly, _, _ = self._parse_model_label(label)
                local_raw[friendly] = (label, actual)
            for repo in FASTER_WHISPER_MODELS:
                label = friendly_label(repo, "hub")
                friendly, _, _ = self._parse_model_label(label)
                # Only add hub entry if not already local
                if friendly not in local_raw:
                    hub_raw[friendly] = (label, repo)

        # Phase 2: build deduplicated entry list
        # Each entry: (label, source) where source is "local" or "hub"
        entries: list[tuple[str, str]] = []
        for _key, (label, actual) in local_raw.items():
            self.model_map[label] = actual
            entries.append((label, "local"))
        for _key, (label, repo) in hub_raw.items():
            self.model_map[label] = repo
            entries.append((label, "hub"))

        # Sort: language-matching first, then multilingual, then others
        def _lang_sort_key(entry: tuple) -> tuple:
            mlang = model_lang(entry[0])
            if mlang == lang:
                return (0, entry[0].lower())
            if mlang is None:
                return (1, entry[0].lower())
            return (2, entry[0].lower())

        entries.sort(key=_lang_sort_key)

        # Keep hidden combobox in sync for any code reading model_combo["values"]
        all_labels = [e[0] for e in entries]
        self.model_combo["values"] = all_labels

        # If current selection is invalid, pick the first entry
        if all_labels:
            current = self.model_var.get()
            if current not in all_labels:
                self.model_var.set(all_labels[0])
        else:
            self.model_var.set("")

        logger.debug("Model list refreshed for backend: %s", backend)

    def _parse_model_label(self, label: str) -> tuple:
        """Extract (friendly_name, language_tag, size) from a model label.

        Handles labels like:
          "★ ivrit-ai large-v3 [local] — Hebrew · 3.1 GB"
          "distil-large-v3 [hub] — English · 1.5 GB"
        Returns (friendly, lang_tag, size_str).
        """
        # Strip star prefix
        text = label.lstrip("\u2605 ")
        # Split on " — " to separate name+source from metadata
        if " \u2014 " in text:
            name_part, meta_part = text.split(" \u2014 ", 1)
        else:
            name_part = text
            meta_part = ""

        # Remove [local], [hub], [cached], [download] tags from name
        friendly = re.sub(r"\s*\[(local|hub|cached|download)\]\s*", " ", name_part).strip()

        # Parse metadata: "Hebrew · 3.1 GB" or "Multilingual"
        lang_tag = "Multilingual"
        size_str = ""
        if meta_part:
            parts = [p.strip() for p in meta_part.split("\u00b7")]
            if parts:
                lang_tag = parts[0]
            if len(parts) > 1:
                size_str = parts[1]

        return friendly, lang_tag, size_str

    def choose_output_dir(self) -> None:
        initial = self.output_dir_var.get() or str(Path.home())
        path = filedialog.askdirectory(initialdir=initial)
        if path:
            self.output_dir_var.set(path)


    def add_files(self) -> None:
        # Build the filter straight from AUDIO_EXTENSIONS so it can never drift
        # out of sync (previously it omitted .opus and .aiff).
        audio_patterns = " ".join(f"*{ext}" for ext in sorted(AUDIO_EXTENSIONS))
        files = filedialog.askopenfilenames(
            title="Select audio files",
            filetypes=[("Audio/video", audio_patterns), ("All files", "*.*")],
        )
        if files:
            self._append_files(list(files))

    def add_folder(self) -> None:
        folder = filedialog.askdirectory(initialdir=str(Path.home()))
        if not folder:
            return
        p = Path(folder)
        found = []
        for child in p.rglob("*"):
            if child.is_file() and child.suffix.lower() in AUDIO_EXTENSIONS:
                found.append(str(child))
        if not found:
            # Before the first run the log pane isn't visible yet, so a
            # log-only notice would be invisible.
            messagebox.showinfo(
                "No files added",
                f"No supported audio files were found in:\n{folder}",
                parent=self)
            return
        self._append_files(sorted(found))

    def _append_files(self, new_files: List[str]) -> None:
        # Adding while a run is active corrupts the batch accounting: idx/
        # total and the progress maxima are frozen at start ("Transcribing
        # 4/3"). Guards every entry path, including drag-and-drop.
        if self.worker_thread and self.worker_thread.is_alive():
            self.log("Cannot add files while transcription is running.")
            self.bell()
            return
        was_empty = len(self.file_paths) == 0
        existing = set(self.file_paths)
        added = 0
        for f in new_files:
            # Normalize path separators so file_paths always uses OS-native
            # format.  tkinter's filedialog on Windows returns forward-slash
            # paths (C:/…) while Path(...) normalises to backslashes (C:\…).
            # Without normalisation, treeview values[5] (which stores
            # str(Path(…))) would differ from file_paths entries, breaking
            # remove_selected_files, error lookups, etc.
            f = str(Path(f))
            if f not in existing:
                self.file_paths.append(f)
                existing.add(f)
                added += 1
        if new_files and added == 0:
            # Everything was a duplicate — say so visibly (the log pane may
            # not be shown yet, see add_folder).
            if len(new_files) == 1:
                dup_msg = "That file is already in the queue."
            else:
                dup_msg = f"All {len(new_files)} selected files are already in the queue."
            messagebox.showinfo("No files added", dup_msg, parent=self)
        self.refresh_file_tree()
        self.update_overall_summary()
        self._update_contextual_banner()
        self._update_queue_button_state()
        self.log(f"Added {added} file(s).")
        # Auto-fill output folder from first file's parent directory
        if was_empty and self.file_paths and not self.output_dir_var.get().strip():
            first_parent = str(Path(self.file_paths[0]).parent)
            self.output_dir_var.set(first_parent)
            self.log(f"Output folder set to: {first_parent}")
        # Probe durations in background to keep UI responsive
        # Normalize paths so probe results match file_items / file_paths keys
        to_probe = [str(Path(f)) for f in new_files if str(Path(f)) not in self.file_durations]
        if to_probe:
            threading.Thread(target=self._probe_durations_bg, args=(list(to_probe),), daemon=True).start()

    def _probe_durations_bg(self, paths: List[str]) -> None:
        """Probe audio durations in a background thread; post results via event queue."""
        for path_str in paths:
            dur = probe_duration_seconds(Path(path_str))
            if dur is not None:
                self.event_queue.put(("file_duration", {"path": path_str, "duration": dur}))

    @staticmethod
    def _status_tag(status: str) -> str:
        """Map a display status string to a Treeview tag for color coding."""
        s = status.lower()
        if s == "done":
            return "done"
        elif s == "failed":
            return "failed"
        elif s in ("running", "transcribing"):
            return "running"
        elif s == "cancelled":
            return "cancelled"
        return "queued"

    def _tree_file_display(self, name: str) -> str:
        """Elide a filename to the File column and give RTL names a proper
        bidi base — Treeview hard-clips mid-glyph and scrambles mixed
        Hebrew/digits/extension names otherwise. Display-only."""
        return bidi_name(self._elide_middle(name))

    def _elide_middle(self, name: str) -> str:
        """Middle-ellipsize a filename to fit the current File column width."""
        try:
            width = int(self.file_tree.column("file", "width")) - self._s(14)
            if width <= 40:
                return name
            if not hasattr(self, "_tree_font"):
                import tkinter.font as tkfont
                self._tree_font = tkfont.Font(font=(SYSTEM_FONT, _fs(10)))
            f = self._tree_font
            if f.measure(name) <= width:
                return name
            # Keep the extension plus the stem's last characters so the
            # ellipsis never lands mid-number ("wav.8…." reads as garbage).
            stem, dot, ext = name.rpartition(".")
            if dot and stem and 0 < len(ext) <= 5:
                tail = stem[-4:] + "." + ext
            else:
                tail = name[-8:]
            head = name[:len(name) - len(tail)]
            while head and f.measure(head + "…" + tail) > width:
                head = head[:-1]
            return head + "…" + tail
        except Exception:
            return name

    def _reelide_tree_names(self) -> None:
        """Recompute File-column display names after a resize."""
        try:
            for path_str, item_id in self.file_items.items():
                if not self.file_tree.exists(item_id):
                    continue
                vals = list(self.file_tree.item(item_id, "values"))
                disp = self._tree_file_display(Path(path_str).name)
                if len(vals) >= 5 and vals[4] != disp:
                    vals[4] = disp
                    self.file_tree.item(item_id, values=tuple(vals))
        except Exception:
            logger.debug("Tree name re-elision failed", exc_info=True)

    def refresh_file_tree(self) -> None:
        current_rows = {path: self.file_tree.item(item_id, "values") for path, item_id in self.file_items.items() if self.file_tree.exists(item_id)}
        # Preserve progress data across refresh (keyed by path)
        old_progress = {}
        for path_str, item_id in self.file_items.items():
            if item_id in self._tree_progress:
                old_progress[path_str] = self._tree_progress[item_id]
        self._tree_progress.clear()
        for item in self.file_tree.get_children():
            self.file_tree.delete(item)
        self.file_items = {}
        new_status: Dict[str, str] = {}
        for idx, path_str in enumerate(self.file_paths):
            p = Path(path_str)
            previous = current_rows.get(path_str)
            if previous and len(previous) >= 6:
                _ord, status, progress, duration_str, _, _ = previous
            elif previous and len(previous) >= 5:
                status, progress, _, _ = previous[1:5]
                duration_str = format_hms(self.file_durations.get(path_str))
            else:
                status, progress = "Queued", ""
                duration_str = format_hms(self.file_durations.get(path_str))
            tag = self._status_tag(status)
            # alternating row stripe
            tags = (tag, "stripe") if idx % 2 == 1 else (tag,)
            ordinal = idx + 1
            item_id = self.file_tree.insert("", "end", values=(ordinal, status, progress, duration_str, self._tree_file_display(p.name), str(p)), tags=tags)
            self.file_items[path_str] = item_id
            new_status[path_str] = status
            # Restore progress data
            if path_str in old_progress:
                self._tree_progress[item_id] = old_progress[path_str]
        with self._queue_lock:
            self.file_status.clear()
            self.file_status.update(new_status)
        self._update_empty_state()
        self._redraw_tree_progress()

    def set_tree_row(self, path_str: str, status: Optional[str] = None,
                     progress_text: Optional[str] = None, duration_text: Optional[str] = None,
                     progress_pct: Optional[float] = None) -> None:
        item_id = self.file_items.get(path_str)
        if not item_id or not self.file_tree.exists(item_id):
            return
        values = list(self.file_tree.item(item_id, "values"))
        if status is not None:
            values[1] = status
        if progress_text is not None:
            values[2] = progress_text
        if duration_text is not None:
            values[3] = duration_text
        tag = self._status_tag(values[1])
        self.file_tree.item(item_id, values=tuple(values), tags=(tag,))
        with self._queue_lock:
            self.file_status[path_str] = values[1]
        # Track progress percent for canvas bar overlay
        if progress_pct is not None:
            self._tree_progress[item_id] = (progress_pct, values[1])
        elif status and status.lower() == "done":
            self._tree_progress[item_id] = (100.0, "Done")
        elif status and status.lower() in ("failed", "cancelled"):
            self._tree_progress.pop(item_id, None)
        self._redraw_tree_progress()

    def _redraw_tree_progress(self) -> None:
        """Redraw progress bar widgets for all visible treeview rows.

        Places a thin tk.Frame at the bottom of each row's "progress" column cell.
        Color: blue for running, green for done.
        Frames are children of file_tree so they scroll with it automatically.
        """
        # Hide all existing bar widgets
        for w in self._tree_bar_widgets:
            w.place_forget()

        if not self._tree_progress or not self.file_paths:
            return

        BAR_HEIGHT = self._s(3)
        BAR_YPAD = self._s(2)

        color_map = {
            "running":     self.CLR_RUNNING_ACCENT,
            "transcribing": self.CLR_RUNNING_ACCENT,
            "done":        self.CLR_SUCCESS_FG,
        }
        bg_color = C.BORDER_LIGHT

        widget_idx = 0
        for item_id, (pct, status) in self._tree_progress.items():
            if not self.file_tree.exists(item_id):
                continue
            try:
                bbox = self.file_tree.bbox(item_id, column="progress")
            except Exception:
                continue
            if not bbox:
                continue  # row not visible
            x, y, w, h = bbox
            if w < 8 or h < 4:
                continue

            bar_y = y + h - BAR_HEIGHT - BAR_YPAD
            track_x = x + 4
            track_w = w - 8
            fill_w = max(0, int(track_w * min(100.0, max(0.0, pct)) / 100.0))

            # Ensure we have enough recycled widgets (track + fill = 2 per row)
            while len(self._tree_bar_widgets) < widget_idx + 2:
                f = tk.Frame(self.file_tree, height=BAR_HEIGHT, borderwidth=0,
                             highlightthickness=0)
                self._tree_bar_widgets.append(f)

            # Track (grey background)
            track = self._tree_bar_widgets[widget_idx]
            track.configure(background=bg_color)
            track.place(x=track_x, y=bar_y, width=track_w, height=BAR_HEIGHT)
            widget_idx += 1

            # Fill (colored portion)
            fill = self._tree_bar_widgets[widget_idx]
            if fill_w > 0:
                fill_color = color_map.get(status.lower(), self.CLR_RUNNING_ACCENT)
                fill.configure(background=fill_color)
                fill.place(x=track_x, y=bar_y, width=fill_w, height=BAR_HEIGHT)
            else:
                fill.place_forget()
            widget_idx += 1

    def remove_selected_files(self) -> None:
        selected = self.file_tree.selection()
        if not selected:
            return
        # Use file_items reverse lookup (item_id → path) so we match the
        # exact strings stored in file_paths, avoiding any normalisation
        # mismatch between treeview values and file_paths entries.
        item_to_path = {v: k for k, v in self.file_items.items()}
        selected_paths = {item_to_path[item] for item in selected if item in item_to_path}
        if not selected_paths:
            return
        with self._queue_lock:
            self.file_paths = [p for p in self.file_paths if p not in selected_paths]
        self.refresh_file_tree()
        self.update_overall_summary()
        self._update_contextual_banner()
        self._update_queue_button_state()
        self.log(f"Removed {len(selected_paths)} file(s).")

    def clear_files(self) -> None:
        # Defence in depth: button is disabled during transcription,
        # but guard against programmatic calls.
        if self.worker_thread and self.worker_thread.is_alive():
            return
        # Clearing is a full session reset (queue + activity log + stats), which
        # is broader than the button label implies — confirm when run history
        # would be lost. Not gated on _log_entries: adding files already logs,
        # so that would make the confirm fire on every click.
        counts = self._count_by_status()
        if (self._session_run_count > 0 or counts["Done"] > 0
                or counts["Failed"] > 0 or counts["Cancelled"] > 0):
            if not messagebox.askyesno(
                    "Clear session",
                    "Clear the file list and this session's activity log and statistics?",
                    parent=self):
                return
        self.file_paths = []
        self.file_errors.clear()
        self.file_durations.clear()
        self._tree_progress.clear()
        # Reset session accumulators — this is the explicit "start over"
        self._session_audio_total = 0.0
        self._session_wall_total = 0.0
        self._session_done_count = 0
        self._session_failed_count = 0
        self._session_run_count = 0
        # Clear structured log
        self._log_entries.clear()
        self._log_run_id = 0
        if hasattr(self, "_activity_tree"):
            self._activity_tree.delete(*self._activity_tree.get_children())
        self.log_text.delete("1.0", "end")
        # Revert the right pane to the onboarding idle panel — the run details
        # describe a session that no longer exists. Leave the recording
        # panel alone if it is the one currently shown.
        if self._run_details_frame.winfo_manager():
            self._run_details_frame.pack_forget()
            self._show_idle_panel()
        self.refresh_file_tree()
        self.update_overall_summary()
        self._update_contextual_banner()
        self._update_queue_button_state()
        self.log("Cleared file list.")

    def update_overall_summary(self, done: int = -1, failed: int = -1, running: int = -1) -> None:
        total = len(self.file_paths)
        self._check_readiness()
        self._update_statusbar_file_info()
        if total == 0:
            self.overall_summary_var.set("0 files queued")
            return
        # If caller didn't provide counts, scan the tree for truth
        cancelled = 0
        if done < 0 or failed < 0 or running < 0:
            counts = self._count_by_status()
            done = counts["Done"]
            failed = counts["Failed"]
            running = counts["Running"]
            cancelled = counts["Cancelled"]
        queued = max(0, total - done - failed - running - cancelled)
        dur_str = self._batch_duration_str()
        size_str = self._estimate_output_size_str()
        extra_parts = [x for x in (dur_str, size_str) if x]
        extra = f"  ({', '.join(extra_parts)})" if extra_parts else ""
        summary = f"Queued: {queued} | Done: {done} | Failed: {failed} | Running: {running}"
        if cancelled > 0:
            summary += f" | Cancelled: {cancelled}"
        self.overall_summary_var.set(summary + extra)

    def _batch_duration_str(self) -> str:
        """Return total audio duration of queued files as a formatted string, or '' if unknown."""
        known = [self.file_durations[p] for p in self.file_paths if self.file_durations.get(p)]
        if not known:
            return ""
        total_sec = sum(known)
        suffix = "" if len(known) == len(self.file_paths) else "+"
        return f"{format_hms(total_sec)}{suffix} audio"

    def _estimate_output_size_str(self) -> str:
        """Estimate total output file size based on audio duration and selected formats.

        Empirical ratios (bytes per audio-second) for Hebrew speech:
          txt: ~12 B/s (plain text, ~2-3 words/sec)
          srt: ~30 B/s (timestamps + indices + text)
          json: ~80 B/s (full metadata + segments)
        These are rough estimates -- actual size depends on speech density.
        """
        BYTES_PER_SEC = {"txt": 12, "srt": 30, "json": 80}
        known = [self.file_durations[p] for p in self.file_paths if self.file_durations.get(p)]
        if not known:
            return ""
        total_sec = sum(known)
        selected_formats = [fmt for fmt, var in self.format_vars.items() if var.get()]
        if not selected_formats:
            return ""
        bytes_total = sum(BYTES_PER_SEC.get(f, 20) * total_sec for f in selected_formats)
        if bytes_total < 1024:
            return f"~{bytes_total:.0f} B"
        elif bytes_total < 1024 * 1024:
            return f"~{bytes_total / 1024:.0f} KB"
        else:
            return f"~{bytes_total / (1024 * 1024):.1f} MB"

    def _update_batch_duration_summary(self) -> None:
        """Refresh the summary line with updated duration info (called after duration probe)."""
        if not (self.worker_thread and self.worker_thread.is_alive()):
            self.update_overall_summary()

    def get_selected_model_spec(self) -> str:
        label = self.model_var.get()
        return self.model_map.get(label, "")

    def get_run_options(self) -> RunOptions:
        selected_formats = [fmt for fmt, var in self.format_vars.items() if var.get()]
        if not selected_formats:
            raise ValueError("Select at least one output format.")

        try:
            beam_size = int(self.beam_size_var.get())
        except ValueError:
            raise ValueError("Beam size must be an integer.")
        if beam_size < 1:
            raise ValueError("Beam size must be >= 1.")

        try:
            batch_size = int(self.batch_size_var.get())
        except ValueError:
            raise ValueError("Batch size must be an integer.")

        try:
            num_speakers = int(self.num_speakers_var.get())
        except ValueError:
            raise ValueError("Number of speakers must be an integer.")
        if num_speakers < 0:
            raise ValueError("Number of speakers must be >= 0.")

        return RunOptions(
            backend=self.backend_var.get(),
            model_spec=self.get_selected_model_spec(),
            output_dir=self.output_dir_var.get().strip(),
            language=self.language_var.get().strip(),
            task=self.task_var.get().strip(),
            device=self.device_var.get().strip(),
            compute_type=self.compute_type_var.get().strip(),
            beam_size=beam_size,
            vad_filter=self.vad_var.get(),
            condition_on_previous_text=self.condition_on_previous_text_var.get(),
            batch_size=batch_size,
            formats=selected_formats,
            diarize=self.diarize_var.get(),
            num_speakers=num_speakers,
            diarize_device=self.diarize_device_var.get().strip(),
        )

    def validate_before_run(self) -> Optional[RunOptions]:
        if self.worker_thread and self.worker_thread.is_alive():
            messagebox.showinfo("Already running", "A transcription job is already running.")
            return None
        if self._recorder is not None and self._recorder.is_recording():
            messagebox.showinfo("Recording active", "Stop the live recording first.")
            return None

        # Re-run readiness check (defensive — button should already be disabled if issues exist)
        self._check_readiness()
        if self.readiness_var.get():
            messagebox.showerror("Not ready", self.readiness_var.get())
            return None

        try:
            opts = self.get_run_options()
        except Exception as e:
            messagebox.showerror("Invalid settings", str(e))
            return None

        out_path = Path(opts.output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        if opts.backend == "openai-whisper" and not ffmpeg_available():
            messagebox.showwarning(
                "FFmpeg not found",
                "openai-whisper usually needs FFmpeg available in PATH.\n\nYou can still continue, but this backend may fail until FFmpeg is installed.",
            )

        return opts

    def _show_run_controls(self) -> None:
        """Show Pause/Resume/Stop buttons and Cancel (separated)."""
        if not self._run_controls.winfo_manager():
            self._run_controls.pack(side="left", padx=(16, 0))
        if not self.cancel_button.winfo_manager():
            self.cancel_button.pack(side="left", padx=(24, 0))

    def _hide_run_controls(self) -> None:
        """Hide Pause/Resume/Stop/Cancel buttons when idle."""
        if self._run_controls.winfo_manager():
            self._run_controls.pack_forget()
        if self.cancel_button.winfo_manager():
            self.cancel_button.pack_forget()

    def _show_done_controls(self, has_failures: bool = False) -> None:
        """Show post-completion controls."""
        if has_failures and not self._done_controls.winfo_manager():
            self._done_controls.pack(side="left", padx=(16, 0))
        elif not has_failures and self._done_controls.winfo_manager():
            self._done_controls.pack_forget()

    def _hide_done_controls(self) -> None:
        """Hide post-completion controls."""
        if self._done_controls.winfo_manager():
            self._done_controls.pack_forget()

    def set_controls_running(self, is_running: bool) -> None:
        if is_running:
            self._show_run_controls()
            self._hide_done_controls()
            self._set_button_enabled(self.pause_button, True)
            self._set_button_enabled(self.resume_button, False)
            self._set_button_enabled(self.stop_button, True)
            self._set_button_enabled(self.cancel_button, True)
            self._set_button_enabled(self.start_button, False)
            if self.record_button:
                self._set_button_enabled(self.record_button, False)
            self.start_button.config(text="  Start transcription  ")
            self.readiness_var.set("")
            self._update_queue_button_state()
            # _update_queue_button_state derives is_running from the worker
            # thread, which hasn't started yet when this runs — disable Clear
            # explicitly so it isn't left enabled-looking during the run.
            self._set_button_enabled(self._btn_clear, False)
            # Adding files mid-run is blocked (_append_files guard); grey the
            # buttons so the affordance matches. Drag-and-drop hits the same
            # guard.
            self._set_button_enabled(self._btn_add_files, False)
            self._set_button_enabled(self._btn_add_folder, False)
        else:
            self._hide_run_controls()
            self._stop_pause_blink()
            self._sleep_inhibitor.release()
            # Restore settings panel in the footer area
            if not self._settings_container.winfo_manager():
                self._settings_container.pack(fill="x", in_=self._footer_frame)
            has_failures = bool(self.file_errors)
            self._show_done_controls(has_failures)
            self._check_readiness()  # re-evaluate and set start button state
            if self.record_button:
                self._set_button_enabled(self.record_button, True)
            self._set_button_enabled(self._btn_add_files, True)
            self._set_button_enabled(self._btn_add_folder, True)
            # Let contextual banner set button text based on queue state.
            # worker_finished: this branch only runs from the done/failed/
            # fatal_error handlers, where the thread may still be tearing down.
            self._update_contextual_banner(worker_finished=True)
            self._update_queue_button_state()

    def _requeue_cancelled(self) -> None:
        """Reset any 'Cancelled' rows back to 'Queued' so they can be re-run."""
        changed = False
        for path_str, item_id in list(self.file_items.items()):
            if not self.file_tree.exists(item_id):
                continue
            if self.file_tree.item(item_id, "values")[1] == "Cancelled":
                self.set_tree_row(path_str, status="Queued", progress_text="")
                self.file_errors.pop(path_str, None)
                changed = True
        if changed:
            self.refresh_file_tree()

    def start_transcription(self) -> None:
        opts = self.validate_before_run()
        if not opts:
            return

        # Fold any 'Cancelled' rows from a previous run back into the queue so
        # Start re-runs them — otherwise a cancelled file is stranded (re-runnable
        # only by Remove + re-Add) while the summary still counts it as queued.
        self._requeue_cancelled()

        # Count only Queued files — don't re-process Done/Failed
        counts = self._count_by_status()
        queued_count = counts["Queued"]
        if queued_count == 0:
            messagebox.showinfo("Nothing to transcribe",
                                "No queued files. Add files or retry failed ones.")
            return

        self._show_stats()  # reveal stats panel on first run
        self.stop_requested = False
        self.cancel_requested = False
        # Only clear errors for files that are Queued (being re-run)
        for path_str in list(self.file_errors.keys()):
            item_id = self.file_items.get(path_str)
            if item_id and self.file_tree.exists(item_id):
                status = self.file_tree.item(item_id, "values")[1]
                if status == "Queued":
                    self.file_errors.pop(path_str, None)
        self._job_wall_start = time.time()
        self._batch_audio_done = 0.0
        self._batch_audio_total = 0.0
        self._batch_wall_start = time.time()
        self._current_file_pct = 0.0
        self._current_file_idx = 0
        self._current_file_total = queued_count
        self.pause_event.set()
        self.set_controls_running(True)
        # Hide entire settings panel during transcription
        if self._settings_container.winfo_manager():
            self._settings_container.pack_forget()
        self._apply_app_state("running", idx=0, total=queued_count, filename="loading model...")
        # Indeterminate progress during model loading
        self.current_progress.config(mode="indeterminate")
        self.current_progress.start(20)
        self.overall_progress["value"] = 0
        self.overall_progress["maximum"] = max(1, queued_count)
        self.job_status_var.set("Starting...")
        self.current_file_var.set("Preparing...")
        self.current_phase_var.set("Loading model")
        self._current_lang = ""
        self._last_file_speed = None
        self._last_file_eta = None
        self._last_file_elapsed_s = 0.0
        self._last_batch_speed = None
        self._last_batch_eta_s = None
        self._last_job_wall = 0.0
        self._last_finished_file_path = None
        self._segments_done = 0
        self._words_transcribed = 0
        self._current_file_path = None
        self._refresh_breathing_lines()
        self.preview_text.delete("1.0", "end")
        # Auto-switch to preview tab when transcription starts
        self._detail_notebook.select(0)
        self._tree_progress.clear()
        self._redraw_tree_progress()
        # Structured log: new run
        self._log_run_id += 1
        self._log_current_phase = "setup"
        self._log_current_file = None
        self._activity_files_done = 0
        self._activity_files_failed = 0
        self._activity_files_total = queued_count
        self._activity_setup_node = ""
        self._activity_files_node = ""
        self._activity_file_nodes = {}
        self._activity_results_node = ""
        # Create activity tree groups for this run
        if hasattr(self, "_activity_tree"):
            self._activity_begin_run(queued_count)
        # Log run info for multi-run sessions
        run_num = self._session_run_count + 1
        lang = self.language_var.get()
        model_label = self.model_var.get()
        if run_num > 1:
            self.log(f"--- Run {run_num}: {queued_count} file(s), language={lang}, model={model_label} ---")
        # Taskbar: indeterminate during model load
        self._taskbar.set_state(TaskbarProgress.TBPF_INDETERMINATE)
        self.save_config_from_ui()
        # Prevent sleep if user opted in
        if self.prevent_sleep_var.get():
            if not self._sleep_inhibitor.acquire():
                self.log("Warning: could not prevent system sleep. Your computer may sleep during long batches.")
        # Detect device info and start tickers
        self._detect_device_info()
        self._start_elapsed_ticker()
        self._refresh_vram()

        def _guarded_worker():
            try:
                run_transcription_worker(self, opts, queued_count)
            except Exception:
                logger.error("Worker thread crashed unexpectedly", exc_info=True)
                self.post_event("fatal_error", message="Unexpected internal error. Check the raw log for details.")

        self.worker_thread = threading.Thread(target=_guarded_worker, daemon=True)
        self.worker_thread.start()

    def pause_processing(self) -> None:
        if not (self.worker_thread and self.worker_thread.is_alive()):
            return
        self.pause_event.clear()
        self._set_button_enabled(self.pause_button, False)
        self._set_button_enabled(self.resume_button, True)
        self.job_status_var.set("Pause requested. Will pause at the next safe point.")
        self.log("Pause requested.")
        self._start_pause_blink()

    def resume_processing(self) -> None:
        if not (self.worker_thread and self.worker_thread.is_alive()):
            return
        self.pause_event.set()
        self._set_button_enabled(self.pause_button, True)
        self._set_button_enabled(self.resume_button, False)
        self.job_status_var.set("Resuming...")
        self._stop_pause_blink()
        self.log("Resume requested.")

    def _start_pause_blink(self) -> None:
        """Begin blinking '[Paused]' in the title bar for visual pause feedback."""
        self._pause_blink_visible = True
        self._pause_blink_id: Optional[str] = None
        self._do_pause_blink()

    def _do_pause_blink(self) -> None:
        """Toggle title between '[Paused]' and plain to create a blink effect."""
        if self.pause_event.is_set():
            return  # no longer paused
        if self._pause_blink_visible:
            self.title(f"[PAUSED] \u2014 {APP_TITLE}")
        else:
            self.title(f"         \u2014 {APP_TITLE}")
        self._pause_blink_visible = not self._pause_blink_visible
        self._pause_blink_id = self.after(800, self._do_pause_blink)

    def _stop_pause_blink(self) -> None:
        """Stop the pause blink and restore normal title."""
        if hasattr(self, "_pause_blink_id") and self._pause_blink_id:
            self.after_cancel(self._pause_blink_id)
            self._pause_blink_id = None

    # --- Elapsed ticker (1-second independent refresh) ---

    def _start_elapsed_ticker(self) -> None:
        """Begin updating elapsed/speed every second, independent of worker events."""
        self._stop_elapsed_ticker()
        self._tick_elapsed()

    def _tick_elapsed(self) -> None:
        """Update elapsed time and speed from wall-clock. Reschedules every 1 second."""
        if not (self.worker_thread and self.worker_thread.is_alive()):
            return
        # Per-file elapsed
        if self._current_file_path and self._current_file_path in self._file_wall_starts:
            file_wall = time.time() - self._file_wall_starts[self._current_file_path]
            self._last_file_elapsed_s = file_wall
            # Compute speed from file duration and progress percent
            dur = self.file_durations.get(self._current_file_path)
            pct = getattr(self, "_current_file_pct", 0.0)
            if dur and dur > 0 and pct > 0 and file_wall > 0:
                processed = dur * (pct / 100.0)
                self._last_file_speed = processed / file_wall
        # Job-level elapsed
        elif self._job_wall_start > 0:
            self._last_file_elapsed_s = time.time() - self._job_wall_start
        self._refresh_breathing_lines()
        self._elapsed_ticker_id = self.after(1000, self._tick_elapsed)

    def _stop_elapsed_ticker(self) -> None:
        """Stop the 1-second elapsed ticker."""
        if self._elapsed_ticker_id:
            self.after_cancel(self._elapsed_ticker_id)
            self._elapsed_ticker_id = None

    # --- GPU / device info ---

    def _detect_device_info(self) -> None:
        """Detect GPU/device and set device_info_var. Safe to call on any platform."""
        try:
            import torch
            if torch.cuda.is_available():
                idx = torch.cuda.current_device()
                name = torch.cuda.get_device_name(idx)
                total = torch.cuda.get_device_properties(idx).total_mem / (1024 ** 3)
                free = (total - torch.cuda.memory_allocated(idx) / (1024 ** 3))
                self.device_info_var.set(f"GPU: {name}  ({free:.1f}/{total:.1f} GB free)")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                # Apple Silicon — CTranslate2 uses CPU, but we show the chip info
                import platform
                chip = platform.processor() or "Apple Silicon"
                self.device_info_var.set(f"{chip} (CPU mode — MPS not supported by CTranslate2)")
            else:
                self.device_info_var.set("CPU mode")
        except ImportError:
            self.device_info_var.set("CPU mode")
        except Exception:
            logger.debug("Device detection failed", exc_info=True)
            self.device_info_var.set("")

    def _refresh_vram(self) -> None:
        """Update VRAM usage during transcription. Reschedules every 5 seconds."""
        if not (self.worker_thread and self.worker_thread.is_alive()):
            return
        try:
            import torch
            if torch.cuda.is_available():
                idx = torch.cuda.current_device()
                name = torch.cuda.get_device_name(idx)
                total = torch.cuda.get_device_properties(idx).total_mem / (1024 ** 3)
                allocated = torch.cuda.memory_allocated(idx) / (1024 ** 3)
                self.device_info_var.set(
                    f"GPU: {name}  ({allocated:.1f}/{total:.1f} GB used)")
        except Exception:
            pass  # keep whatever was last shown
        self._vram_ticker_id = self.after(5000, self._refresh_vram)

    def _stop_vram_ticker(self) -> None:
        """Stop periodic VRAM refresh."""
        if self._vram_ticker_id:
            self.after_cancel(self._vram_ticker_id)
            self._vram_ticker_id = None

    def request_stop_after_current(self) -> None:
        if not (self.worker_thread and self.worker_thread.is_alive()):
            return
        self.stop_requested = True
        self.job_status_var.set("Will stop after the current file.")
        self.log("Stop requested. The app will stop after the current file finishes.")

    def request_cancel_now(self) -> None:
        if not (self.worker_thread and self.worker_thread.is_alive()):
            return
        self.cancel_requested = True
        self.pause_event.set()
        self._stop_pause_blink()
        self.job_status_var.set("Cancel requested. Stopping at the next safe point.")
        self.log("Immediate cancel requested.")

    def check_for_cancel(self) -> None:
        if self.cancel_requested:
            raise CancelledByUser("Cancelled by user.")

    def wait_if_paused(self) -> None:
        while not self.pause_event.is_set():
            self.check_for_cancel()
            self.post_event("paused")
            time.sleep(0.15)
        # Mirror worker.wait_if_paused: a cancel issued while paused wins
        # over the resume.
        self.check_for_cancel()
        self.post_event("resumed")

    def post_event(self, kind: str, **payload) -> None:
        self.event_queue.put((kind, payload))

    # ------------------------------------------------------------------
    # Live recording
    # ------------------------------------------------------------------

    def _update_recording_visibility(self) -> None:
        """Show or hide the Record button and recording panel based on the experimental toggle."""
        enabled = self._experimental_recording_var.get()
        if enabled:
            # Show Record button (if deps available and not already packed)
            if self.record_button and not self.record_button.winfo_manager():
                self.record_button.pack(side="left", padx=(0, 8), after=self.start_button)
        else:
            # Stop any active recording
            if self._recorder is not None and self._recorder.is_recording():
                self._stop_recording()
            # Hide recording panel if visible, restore idle panel
            if self._recording_panel.winfo_manager():
                self._recording_panel.pack_forget()
                self._run_subtitle.configure(text="")
                if not self._idle_panel.winfo_manager() and not self._run_details_frame.winfo_manager():
                    self._idle_panel.pack(fill="both", expand=True)
            # Hide Record button
            if self.record_button and self.record_button.winfo_manager():
                self.record_button.pack_forget()
        self.save_config_from_ui()

    def _toggle_recording(self) -> None:
        """Start or stop live recording."""
        if self._recorder is not None and self._recorder.is_recording():
            self._stop_recording()
        else:
            self._start_recording()

    def _start_recording(self) -> None:
        """Begin live speech-to-text recording."""
        # Block if batch transcription is running
        if self.worker_thread and self.worker_thread.is_alive():
            from tkinter import messagebox
            messagebox.showinfo("Busy", "Cannot record while batch transcription is running.")
            return

        from hebrewscribe.recorder import LiveRecorder

        # Resolve model spec from current language selection
        model_spec = self.get_selected_model_spec()
        if not model_spec:
            from tkinter import messagebox
            messagebox.showerror("No model", "Please select a language first.")
            return

        language = self.language_var.get().strip()
        device = self.device_var.get().strip()
        compute_type = self.compute_type_var.get().strip()

        # Show recording panel
        if self._idle_panel.winfo_manager():
            self._idle_panel.pack_forget()
        if self._run_details_frame.winfo_manager():
            self._run_details_frame.pack_forget()
        self._recording_panel.pack(fill="both", expand=True)
        self._widen_for_recording()

        # Update button appearance
        if self.record_button:
            self.record_button.configure(
                text="  Stop recording  ",
                background=self._BTN_STYLES["destructive"]["bg"],
                foreground=self._BTN_STYLES["destructive"]["fg"],
                activebackground=self._BTN_STYLES["destructive"]["press_bg"],
            )
            self.record_button._btn_colors = self._BTN_STYLES["destructive"]

        # Disable start button during recording
        self._set_button_enabled(self.start_button, False)

        # Update right panel heading
        self._run_subtitle.configure(text="Live speech-to-text")

        self._rec_status_var.set("Starting...")

        # Create and start recorder
        self._recorder = LiveRecorder(
            event_callback=self.post_event,
            language=language,
            model_spec=model_spec,
            device=device,
            compute_type=compute_type,
        )
        self._recorder.start()

    def _stop_recording(self) -> None:
        """Stop live recording without blocking the GUI.

        recorder.stop() joins worker threads — up to ~10 s when a segment is
        mid-transcription — so it runs off the Tk main thread. The button
        reset and status happen in the rec_stopped handler; until then the
        Record button stays disabled so a re-start can't race the old
        threads.
        """
        recorder, self._recorder = self._recorder, None
        if recorder is None:
            return
        self._rec_status_var.set("Stopping...")
        if self.record_button:
            self._set_button_enabled(self.record_button, False)
        threading.Thread(target=recorder.stop, daemon=True,
                         name="rec-stop").start()

    def _copy_rec_text(self) -> None:
        """Copy recorded text to clipboard (minus display-only RLM marks)."""
        text = self._rec_text.get("1.0", "end-1c").replace(chr(0x200F), "").strip()
        if text:
            self.clipboard_clear()
            self.clipboard_append(text)

    def _clear_rec_text(self) -> None:
        """Clear the recorded text area (confirmed — dictation has no undo)."""
        if self._rec_text.get("1.0", "end-1c").strip():
            from tkinter import messagebox
            if not messagebox.askyesno(
                "Clear text?",
                "Clear the dictated text? This cannot be undone.",
                parent=self,
            ):
                return
        self._rec_text.delete("1.0", "end")

    def _save_rec_text(self) -> None:
        """Save recorded text to a file (minus display-only RLM marks)."""
        text = self._rec_text.get("1.0", "end-1c").replace(chr(0x200F), "").strip()
        if not text:
            return
        from tkinter import filedialog
        path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            title="Save recording text",
        )
        if path:
            try:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(text)
            except OSError as e:
                from tkinter import messagebox
                messagebox.showerror("Save failed", str(e))

    def next_file(self) -> Optional[str]:
        """Return the next Queued file path, or None if none remain.

        Called by the worker thread. Reads only plain-Python state
        (file_paths + file_status) under _queue_lock — deliberately no Tk
        calls: a Treeview read from this thread could hit a mid-rebuild
        tree (TclError aborts the batch) or deadlock against a GUI thread
        blocked on the same lock.
        """
        with self._queue_lock:
            for path_str in self.file_paths:
                if self.file_status.get(path_str, "Queued") == "Queued":
                    return path_str
            return None

    def _ensure_determinate_progress(self) -> None:
        """Switch current-file bar back to determinate if it's pulsing."""
        if str(self.current_progress.cget("mode")) == "indeterminate":
            self._cancel_progress_anim(self.current_progress)
            self.current_progress.stop()
            self.current_progress.config(mode="determinate", maximum=100, value=0)

    def _cancel_progress_anim(self, bar: ttk.Progressbar) -> None:
        """Cancel every pending animation frame scheduled for this bar."""
        anim_key = f"_anim_{id(bar)}"
        for aid in getattr(self, anim_key, ()):
            try:
                self.after_cancel(aid)
            except Exception:
                pass
        setattr(self, anim_key, [])

    def _smooth_set_progress(self, bar: ttk.Progressbar, target: float,
                             steps: int = 6, interval: int = 30) -> None:
        """Animate a progress bar from its current value to target over steps frames."""
        # Cancel ALL pending frames of the previous animation — keeping only
        # the last after-id let stale frames land after a newer value and
        # drag the bar backward.
        self._cancel_progress_anim(bar)
        try:
            current = float(bar["value"])
        except (tk.TclError, ValueError):
            current = 0.0
        if abs(target - current) < 0.5:
            bar["value"] = target
            return
        anim_key = f"_anim_{id(bar)}"
        delta = target - current
        ids = []
        for i in range(1, steps + 1):
            val = current + delta * (i / steps)
            ids.append(self.after(i * interval, lambda v=val: bar.configure(value=v)))
        setattr(self, anim_key, ids)

    # --- Breathing line compose & refresh ---

    def _compose_file_breathing(self) -> tuple:
        """Return (text, foreground_color) for the file card breathing line."""
        phase = self.current_phase_var.get()

        if phase in ("Loading model", "Loading model\u2026"):
            return ("Loading model\u2026", self.CLR_IDLE_FG)
        if phase.startswith("Downloading") or phase.startswith("Download"):
            return (phase, self.CLR_RUNNING_FG)
        if phase == "Paused":
            elapsed = self._get_file_elapsed_str()
            return (f"Paused \u00b7 {elapsed} elapsed", self.CLR_PAUSED_FG)
        if phase == "Finished":
            elapsed = self._get_file_elapsed_str()
            speed = self._last_file_speed
            parts = [f"{elapsed} real time"]
            if speed and speed > 0:
                parts.append(f"{speed:.1f}x speed")
            return (" \u00b7 ".join(parts), self.CLR_SUCCESS_FG)
        if phase == "Failed":
            elapsed = self._get_file_elapsed_str()
            return (f"Failed after {elapsed} \u2014 see log", self.CLR_ERROR_FG)
        if phase in ("Cancelled", "Stopped"):
            elapsed = self._get_file_elapsed_str()
            return (f"{phase} after {elapsed}", self.CLR_WARNING_FG)

        # Transcribing — progressive reveal
        elapsed_s = self._last_file_elapsed_s
        speed = self._last_file_speed
        eta = self._last_file_eta

        # Before we have reliable metrics, show phase name
        if elapsed_s < 3 or not speed or speed <= 0 or not eta:
            # Use the actual phase if it's informative (e.g., "Converting",
            # "Starting"), otherwise default to "Transcribing"
            label = phase if phase and phase not in ("--", "Running") else "Transcribing"
            return (label, self.CLR_RUNNING_FG)

        return (f"{eta} left \u00b7 {speed:.1f}x speed", self.CLR_RUNNING_FG)

    def _compose_batch_identity(self) -> str:
        """Return text for the batch card identity label."""
        phase = self.current_phase_var.get()
        done = self._activity_files_done
        failed = self._activity_files_failed
        total = self._activity_files_total
        audio_str = format_hms(self._batch_audio_done) if self._batch_audio_done > 0 else ""

        if phase in ("Finished", "Cancelled", "Stopped", "Failed"):
            if failed > 0:
                base = f"{done} done, {failed} failed"
            else:
                base = f"{done} file{'s' if done != 1 else ''}"
            return f"{base} \u00b7 {audio_str} audio" if audio_str else base

        # Running / loading
        lang = self._current_lang
        if total == 1:
            base = "1 file"
        else:
            base = f"{done} of {total} files"
        return f"{base} \u00b7 {lang}" if lang else base

    def _compose_batch_breathing(self) -> tuple:
        """Return (text, foreground_color) for the batch card breathing line."""
        phase = self.current_phase_var.get()

        if phase in ("Loading model", "Loading model\u2026"):
            return ("Loading model\u2026", self.CLR_IDLE_FG)
        if phase.startswith("Downloading") or phase.startswith("Download"):
            return ("Downloading model\u2026", self.CLR_RUNNING_FG)
        if phase == "Paused":
            return ("Paused", self.CLR_PAUSED_FG)
        if phase in ("Finished", "Failed"):
            wall = time.time() - self._batch_wall_start if self._batch_wall_start else 0
            # Use the stored job wall time if available (more accurate post-completion)
            if self._last_job_wall > 0:
                wall = self._last_job_wall
            speed = self._last_batch_speed
            parts = [f"{format_hms(wall)} real time"]
            if speed and speed > 0:
                parts.append(f"{speed:.1f}x speed")
            color = self.CLR_SUCCESS_FG if phase == "Finished" else self.CLR_ERROR_FG
            if self._activity_files_failed > 0 and phase == "Finished":
                color = self.CLR_ERROR_FG
            return (" \u00b7 ".join(parts), color)
        if phase in ("Cancelled", "Stopped"):
            wall = time.time() - self._batch_wall_start if self._batch_wall_start else 0
            if self._last_job_wall > 0:
                wall = self._last_job_wall
            return (f"{phase} after {format_hms(wall)} real time", self.CLR_WARNING_FG)

        # Transcribing
        speed = self._last_batch_speed
        eta_s = self._last_batch_eta_s
        if not speed or speed <= 0 or eta_s is None:
            return ("Transcribing\u2026", self.CLR_RUNNING_FG)

        return (f"{format_hms(eta_s)} left \u00b7 {speed:.1f}x speed", self.CLR_RUNNING_FG)

    def _compose_file_tooltip(self) -> str:
        """Build multi-line tooltip for the file card.

        Field order: audio duration, elapsed/real time, ETA, speed, language,
        segments+words. Consistent with batch tooltip (context first, time,
        speed).
        """
        phase = self.current_phase_var.get()
        lines = []

        # Audio duration (use current or last-known file)
        file_path = self._current_file_path or self._last_finished_file_path
        if file_path:
            dur = self.file_durations.get(file_path)
            if dur and dur > 0:
                lines.append(f"Audio: {format_hms(dur)}")

        # Elapsed / Real time
        elapsed = self._get_file_elapsed_str()
        if phase in ("Finished", "Cancelled", "Stopped", "Failed"):
            lines.append(f"Real time: {elapsed}")
        else:
            lines.append(f"Elapsed: {elapsed}")
            eta = self._last_file_eta
            if eta:
                lines.append(f"ETA: {eta}")

        # Speed
        speed = self._last_file_speed
        if speed and speed > 0:
            lines.append(f"Speed: {speed:.1f}x")

        # Language
        if self._current_lang:
            lines.append(f"Language: {self._current_lang}")

        # Segments & words
        if self._segments_done > 0 or self._words_transcribed > 0:
            lines.append(f"Segments: {self._segments_done} \u00b7 Words: {self._words_transcribed}")

        return "\n".join(lines) if lines else ""

    def _compose_batch_tooltip(self) -> str:
        """Build multi-line tooltip for the batch card.

        Field order: files, audio, real time, speed. Consistent with
        file tooltip (context first, time, speed).
        """
        done = self._activity_files_done
        failed = self._activity_files_failed
        total = self._activity_files_total
        lines = []

        phase = self.current_phase_var.get()

        # Files
        if phase in ("Finished", "Cancelled", "Stopped", "Failed"):
            lines.append(f"Files: {done} done, {failed} failed")
        else:
            lines.append(f"Files: {done} of {total} ({failed} failed)")

        # Audio — use _batch_audio_done (excludes in-progress partial) for
        # post-completion accuracy; include current partial during a run.
        if phase in ("Finished", "Cancelled", "Stopped", "Failed"):
            audio_done = self._batch_audio_done
            if audio_done > 0:
                lines.append(f"Total audio: {format_hms(audio_done)}")
        else:
            audio_done = self._batch_audio_done + getattr(self, "_batch_audio_done_current", 0.0)
            audio_total_s = self._batch_audio_total
            if audio_total_s > 0:
                lines.append(f"Audio: {format_hms(audio_done)} of {format_hms(audio_total_s)}")
            elif audio_done > 0:
                lines.append(f"Audio: {format_hms(audio_done)}")

        # Real time
        wall = time.time() - self._batch_wall_start if self._batch_wall_start else 0
        if self._last_job_wall > 0 and phase in ("Finished", "Cancelled", "Stopped", "Failed"):
            wall = self._last_job_wall
        if wall > 0:
            lines.append(f"Real time: {format_hms(wall)}")

        # Speed
        speed = self._last_batch_speed
        if speed and speed > 0:
            lines.append(f"Speed: {speed:.1f}x")

        return "\n".join(lines) if lines else ""

    def _get_file_elapsed_str(self) -> str:
        """Return formatted elapsed time for the current file."""
        if self._current_file_path and self._current_file_path in self._file_wall_starts:
            elapsed = time.time() - self._file_wall_starts[self._current_file_path]
            return format_hms(elapsed)
        # Post-completion: use last stored elapsed
        if self._last_file_elapsed_s > 0:
            return format_hms(self._last_file_elapsed_s)
        return "--:--"

    def _refresh_breathing_lines(self) -> None:
        """Recompute and apply all breathing line text and colors."""
        # File card
        text, color = self._compose_file_breathing()
        self.file_breathing_var.set(text)
        self._file_breathing_label.configure(foreground=color)

        # Batch card
        self.batch_identity_var.set(self._compose_batch_identity())
        text, color = self._compose_batch_breathing()
        self.batch_breathing_var.set(text)
        self._batch_breathing_label.configure(foreground=color)

        # Tooltips
        if hasattr(self, "_file_card_tooltip"):
            self._file_card_tooltip.text = self._compose_file_tooltip()
        if hasattr(self, "_batch_card_tooltip"):
            self._batch_card_tooltip.text = self._compose_batch_tooltip()

    def _update_batch_eta(self) -> None:
        """Recompute batch-level ETA from cumulative audio processed vs wall time."""
        wall_elapsed = time.time() - self._batch_wall_start
        # Include both completed files and the in-progress file's contribution
        audio_done = self._batch_audio_done + getattr(self, "_batch_audio_done_current", 0.0)
        # Warm-up gate: during model load, wall time grows while audio_done
        # stays ~0, so the speed sample explodes into absurd estimates
        # ("~57424:19:04"). Publish nothing until the sample is trustworthy.
        if wall_elapsed < 10.0 or audio_done < 3.0:
            self._last_batch_speed = None
            self._last_batch_eta_s = None
            self._clear_queued_estimates()
            return
        batch_speed = audio_done / wall_elapsed  # audio-sec per wall-sec
        self._last_batch_speed = batch_speed
        remaining_audio = max(0, self._batch_audio_total - audio_done)
        if batch_speed > 0:
            self._last_batch_eta_s = remaining_audio / batch_speed
        else:
            self._last_batch_eta_s = None
        # Update per-file estimates for queued files
        self._update_queued_estimates(batch_speed)

    def _clear_queued_estimates(self) -> None:
        """Blank any stale estimate text on still-queued rows."""
        for path_str, item_id in self.file_items.items():
            if not self.file_tree.exists(item_id):
                continue
            values = self.file_tree.item(item_id, "values")
            if values[1] == "Queued" and values[2]:
                self.set_tree_row(path_str, progress_text="")

    def _update_queued_estimates(self, batch_speed: float) -> None:
        """Show estimated processing time on queued rows based on current batch speed."""
        if batch_speed <= 0:
            return
        for path_str, item_id in self.file_items.items():
            if not self.file_tree.exists(item_id):
                continue
            values = self.file_tree.item(item_id, "values")
            if values[1] != "Queued":
                continue
            dur = self.file_durations.get(path_str)
            if dur and dur > 0:
                est_wall = dur / batch_speed
                # Sanity cap: a wrong number destroys trust — blank is better.
                if est_wall > 8 * 3600:
                    if values[2]:
                        self.set_tree_row(path_str, progress_text="")
                    continue
                self.set_tree_row(path_str, progress_text=f"~{format_hms(est_wall)}")

    def _build_completion_stats(self, wall_seconds: float, done: int, failed: int) -> str:
        """Build a one-line stats summary after batch completion.

        When multiple runs have occurred in the session, appends cumulative
        session totals after the current run's stats.
        """
        parts = []
        audio_total = self._batch_audio_done
        if audio_total > 0:
            parts.append(f"Audio: {format_hms(audio_total)}")
        if wall_seconds > 0:
            parts.append(f"Real time: {format_hms(wall_seconds)}")
        if audio_total > 0 and wall_seconds > 0:
            speed = audio_total / wall_seconds
            parts.append(f"Speed: {speed:.1f}x")
        run_stats = "  |  ".join(parts)

        # Append cumulative session stats when there have been multiple runs
        if self._session_run_count > 1:
            s_parts = []
            s_audio = self._session_audio_total
            s_wall = self._session_wall_total
            if s_audio > 0:
                s_parts.append(f"Audio: {format_hms(s_audio)}")
            if s_wall > 0:
                s_parts.append(f"Real time: {format_hms(s_wall)}")
            if s_audio > 0 and s_wall > 0:
                s_speed = s_audio / s_wall
                s_parts.append(f"Speed: {s_speed:.1f}x")
            if s_parts:
                session_stats = "  |  ".join(s_parts)
                return f"{run_stats}  \u00b7  Session: {session_stats}"

        return run_stats

    def process_event_queue(self) -> None:
        try:
            while True:
                kind, payload = self.event_queue.get_nowait()
                try:
                    self._handle_event(kind, payload)
                except Exception:
                    logger.error("Error handling event %s", kind, exc_info=True)
        except queue.Empty:
            pass
        self.after(120, self.process_event_queue)

    def _handle_event(self, kind: str, payload: dict) -> None:
        """Process a single worker event. Exceptions are caught by the caller."""
        if kind == "log":
            # phase auto-classified by _classify_log_phase; workers may pass
            # an explicit level (e.g. diarization warnings)
            self.log(payload["message"], level=payload.get("level", "info"))
        elif kind == "file_duration":
            path_str = payload["path"]
            dur = payload["duration"]
            self.file_durations[path_str] = dur
            self.set_tree_row(path_str, duration_text=format_hms(dur))
            self._update_batch_duration_summary()
        elif kind == "job_status":
            self.job_status_var.set(payload["message"])
        elif kind == "download_start":
            repo = payload.get("repo_id", "")
            self.job_status_var.set(f"Downloading model: {repo}")
            self.current_phase_var.set("Downloading model\u2026")
            self.current_file_var.set(repo)
            self._show_stats()
            self.current_progress.config(mode="indeterminate")
            self.current_progress.start(20)
            self._update_banner("running", f"Downloading {repo}\u2026")
            self._refresh_breathing_lines()
        elif kind == "download_progress":
            downloaded = payload.get("downloaded", 0)
            total = payload.get("total", 0)
            desc = payload.get("file_desc", "")
            if total > 0:
                self._ensure_determinate_progress()
                pct = min(100.0, downloaded / total * 100)
                self._smooth_set_progress(self.current_progress, pct)
                mb_done = downloaded / (1024 * 1024)
                mb_total = total / (1024 * 1024)
                phase_desc = desc if desc.lower().startswith("download") else f"Downloading {desc}"
                self.current_phase_var.set(
                    f"{phase_desc}  ({mb_done:.0f} / {mb_total:.0f} MB)")
                # Show download progress in status bar. The label is bound to
                # device_info_var via textvariable, which overrides any direct
                # .config(text=...); drive the var so the line actually appears.
                self.device_info_var.set(
                    f"Downloading model: {pct:.0f}% ({mb_done:.0f}/{mb_total:.0f} MB)")
            self._refresh_breathing_lines()
        elif kind == "download_done":
            self._ensure_determinate_progress()
            self._cancel_progress_anim(self.current_progress)
            self.current_progress["value"] = 100
            self.current_phase_var.set("Download complete. Loading model\u2026")
            self.job_status_var.set("Loading model\u2026")
            # Restore status bar device info
            self._detect_device_info()
            self._refresh_breathing_lines()
        elif kind == "current_file":
            self.current_file_var.set(payload.get("name", "--"))
            self.current_phase_var.set(payload.get("phase", "--"))
            if payload.get("language"):
                self._current_lang = payload["language"]
            # If entering a blocking phase (openai-whisper), pulse the bar
            phase = payload.get("phase", "")
            if "openai-whisper" in phase.lower():
                if str(self.current_progress.cget("mode")) != "indeterminate":
                    self.current_progress.config(mode="indeterminate")
                    self.current_progress.start(20)
            self._refresh_breathing_lines()
        elif kind == "current_progress":
            self._ensure_determinate_progress()
            percent = max(0.0, min(100.0, float(payload.get("percent", 0.0))))
            self._smooth_set_progress(self.current_progress, percent)
            self._current_file_pct = percent
            self.current_phase_var.set(payload.get("phase", self.current_phase_var.get()))
            # Store typed values for breathing line compose
            speed = payload.get("speed")
            self._last_file_speed = float(speed) if isinstance(speed, (int, float)) and speed > 0 else None
            eta_val = payload.get("eta")
            self._last_file_eta = format_hms(eta_val) if eta_val is not None else None
            elapsed_val = payload.get("elapsed")
            if isinstance(elapsed_val, (int, float)):
                self._last_file_elapsed_s = float(elapsed_val)
            processed = payload.get("processed_seconds")
            total_sec = payload.get("total_seconds")
            path_str = payload.get("path")
            if path_str:
                bar = f"{percent:.0f}%"
                if processed is not None and total_sec is not None and total_sec > 0:
                    bar = f"{percent:.0f}%  {format_hms(processed)}/{format_hms(total_sec)}"
                self.set_tree_row(path_str, status=payload.get("row_status", "Running"),
                                  progress_text=bar, progress_pct=percent)
            # Update title with per-file percent
            self._apply_app_state("running",
                                  idx=self._current_file_idx,
                                  total=self._current_file_total,
                                  filename=self.current_file_var.get(),
                                  file_pct=percent)
            # Update batch ETA: current file's contribution
            if processed is not None and total_sec is not None and total_sec > 0:
                self._batch_audio_done_current = float(processed)
                self._update_batch_eta()
            self._refresh_breathing_lines()
        elif kind == "overall_progress":
            self._smooth_set_progress(self.overall_progress, float(payload.get("value", 0)))
            # No args: rescan the tree for truth. The worker's payload counts
            # are per-run, so on Continue/retry runs they mixed with the
            # whole-session totals shown everywhere else.
            self.update_overall_summary()
        elif kind == "file_started":
            self._ensure_determinate_progress()
            path_str = payload["path"]
            self._current_file_path = path_str
            self._log_current_file = path_str
            self._log_current_phase = "file"
            self._activity_on_file_started(path_str, payload["name"])
            self._file_wall_starts[path_str] = time.time()
            self.set_tree_row(path_str, status="Running", progress_text="0%", progress_pct=0.0)
            self.current_file_var.set(payload["name"])
            self.current_phase_var.set(payload.get("phase", "Starting"))
            self._cancel_progress_anim(self.current_progress)
            self.current_progress["value"] = 0
            self._current_file_pct = 0.0
            self._current_file_idx = payload.get("idx", 0)
            self._current_file_total = payload.get("total", 0)
            self._batch_audio_done_current = 0.0
            # Reset per-file state for breathing line
            self._last_file_speed = None
            self._last_file_eta = None
            self._last_file_elapsed_s = 0.0
            self._segments_done = 0
            self._words_transcribed = 0
            self.preview_text.delete("1.0", "end")
            # Store total audio duration for batch ETA
            file_duration = payload.get("duration")
            if file_duration and file_duration > 0:
                self._batch_audio_total += file_duration
            self._apply_app_state("running",
                                  idx=payload.get("idx", 0),
                                  total=payload.get("total", 0),
                                  filename=payload.get("name", ""))
            self._refresh_breathing_lines()
        elif kind == "preview":
            text = payload.get("text", "").strip()
            if text:
                tag = self._get_preview_text_tag()
                self.preview_text.insert(
                    "end", self._bidi_display(text, tag == "rtl") + "\n", tag)
                self.preview_text.see("end")
                # Update segment & word counters (shown in tooltip)
                self._segments_done += 1
                self._words_transcribed += len(text.split())
        elif kind == "preview_replace":
            # Post-diarization refresh: swap the streamed transcript for the
            # final speaker-labeled text of the current file. Display-only —
            # the segment/word counters keep their transcription-time values.
            text = payload.get("text", "").strip()
            if text:
                tag = self._get_preview_text_tag()
                self.preview_text.delete("1.0", "end")
                self.preview_text.insert(
                    "end", self._bidi_display(text, tag == "rtl") + "\n", tag)
                self.preview_text.see("end")
        elif kind == "file_finished":
            path_str = payload["path"]
            self._last_finished_file_path = path_str
            self._activity_on_file_finished(path_str)
            self._log_current_file = None
            # Show wall-clock elapsed time instead of bare "100%"
            file_wall = time.time() - self._file_wall_starts.get(path_str, time.time())
            elapsed_str = format_hms(file_wall) if file_wall > 0 else "Done"
            self.set_tree_row(path_str, status="Done", progress_text=elapsed_str, progress_pct=100.0)
            self._cancel_progress_anim(self.current_progress)
            self.current_progress["value"] = 100
            self._current_file_pct = 100.0
            language = payload.get("language")
            if language:
                self._current_lang = language
            # Accumulate audio seconds for batch ETA
            file_audio = payload.get("audio_seconds", 0.0)
            if file_audio and file_audio > 0:
                self._batch_audio_done += file_audio
                self._batch_audio_done_current = 0.0
                self._update_batch_eta()
        elif kind == "file_failed":
            path_str = payload["path"]
            self._activity_on_file_failed(path_str, payload.get("error_message", ""),
                                          status=payload.get("status", "Failed"))
            self._log_current_file = None
            error_msg = payload.get("error_message", "")
            if error_msg:
                self.file_errors[path_str] = error_msg
            # Show truncated error inline in the progress column
            inline_err = ""
            if error_msg:
                first_line = error_msg.split("\n")[0].strip()[:60]
                inline_err = f"\u26a0 {first_line}"
            self.set_tree_row(path_str, status=payload.get("status", "Failed"),
                              progress_text=inline_err or payload.get("progress_text", "--"))
        elif kind == "paused":
            self.current_phase_var.set("Paused")
            self._taskbar.set_state(TaskbarProgress.TBPF_PAUSED)
            self._update_banner("paused", "Paused \u2014 waiting to resume\u2026")
            self._refresh_breathing_lines()
        elif kind == "resumed":
            if self.pause_event.is_set() and self.current_phase_var.get() == "Paused":
                self.current_phase_var.set("Running")
                self._taskbar.set_state(TaskbarProgress.TBPF_NORMAL)
                self._update_banner("running", "Resumed")
                self._refresh_breathing_lines()
        elif kind == "done":
            self._log_current_phase = "results"
            self._log_current_file = None
            self._ensure_determinate_progress()
            # Restore the status-bar device line in case a download_progress
            # overwrote it and the run ended before download_done (e.g. a
            # cancelled weights download) — mirrors the failed/fatal handlers.
            self._detect_device_info()
            self._stop_elapsed_ticker()
            self._stop_vram_ticker()
            self._current_file_path = None
            self.set_controls_running(False)
            wall_seconds = time.time() - self._job_wall_start if self._job_wall_start else 0.0
            done_n = payload.get("done_count", 0)
            failed_n = payload.get("failed_count", 0)
            # Accumulate session stats
            self._session_audio_total += self._batch_audio_done
            self._session_wall_total += wall_seconds
            self._session_done_count += done_n
            self._session_failed_count += failed_n
            self._session_run_count += 1
            msg = payload["message"]
            # Build completion stats
            stats = self._build_completion_stats(wall_seconds, done_n, failed_n)
            if stats:
                msg += f"\n{stats}"
            self.job_status_var.set(payload["message"] + (f"  ({format_hms(wall_seconds)} elapsed)" if wall_seconds > 0 else ""))
            self._last_job_wall = wall_seconds
            if payload.get("cancelled"):
                self.current_phase_var.set("Cancelled")
            elif payload.get("stopped"):
                self.current_phase_var.set("Stopped")
            else:
                self.current_phase_var.set("Finished")
            self._refresh_breathing_lines()
            self.update_overall_summary()
            self._apply_app_state("complete",
                                  message=msg,
                                  done=done_n,
                                  failed=failed_n,
                                  wall_seconds=wall_seconds,
                                  cancelled=payload.get("cancelled", False),
                                  stopped=payload.get("stopped", False))
            # Activity tree: results node (summary already in the tree label)
            self._activity_on_results(
                wall_seconds, done_n, failed_n,
                cancelled=payload.get("cancelled", False),
                stopped=payload.get("stopped", False))
            # Log completion to raw view only (activity tree already has the Results summary)
            self.log(msg, phase="results", raw_only=True)
            # Reset phase context so subsequent user actions don't nest under Results
            self._log_current_phase = "user"
        elif kind == "failed":
            self._ensure_determinate_progress()
            self._stop_elapsed_ticker()
            self._stop_vram_ticker()
            self._current_file_path = None
            self.set_controls_running(False)
            wall_seconds = time.time() - self._job_wall_start if self._job_wall_start else 0.0
            self.job_status_var.set("Failed.")
            self.current_phase_var.set("Failed")
            self._last_job_wall = wall_seconds
            self._refresh_breathing_lines()
            self.update_overall_summary()
            # A failure mid-download leaves "Downloading model: N%" in the
            # status bar (only download_done restores it) — re-detect now.
            self._detect_device_info()
            self._apply_app_state("failed",
                                  message=f"Transcription failed: {payload['message']}",
                                  wall_seconds=wall_seconds)
            # Auto-switch to log tab on fatal failure, show raw view for traceback
            self._detail_notebook.select(1)
            self._switch_log_view("raw")
            self.log(f"Transcription failed: {payload['message']}")
        elif kind == "fatal_error":
            # The worker thread crashed with an unhandled exception. Without this
            # branch the event is silently dropped: controls stay in the running
            # state forever and the sleep inhibitor is never released. Recover the
            # UI exactly like a fatal 'failed' run.
            self._ensure_determinate_progress()
            self._stop_elapsed_ticker()
            self._stop_vram_ticker()
            self._current_file_path = None
            self.set_controls_running(False)
            wall_seconds = time.time() - self._job_wall_start if self._job_wall_start else 0.0
            self.job_status_var.set("Failed.")
            self.current_phase_var.set("Failed")
            self._last_job_wall = wall_seconds
            self._refresh_breathing_lines()
            self.update_overall_summary()
            self._detect_device_info()
            msg = payload.get("message", "Unexpected internal error.")
            self._apply_app_state("failed", message=msg, wall_seconds=wall_seconds)
            self._detail_notebook.select(1)
            self._switch_log_view("raw")
            self.log(f"Fatal error: {msg}")

        # --- Recording events ---
        elif kind == "rec_text":
            text = payload.get("text", "").strip()
            if text:
                lang = payload.get("language", "")
                tag = "rtl" if lang == "he" else "ltr"
                self._rec_text.insert(
                    "end", self._bidi_display(text, tag == "rtl") + "\n", tag)
                self._rec_text.see("end")
        elif kind == "rec_status":
            msg = payload.get("message", "")
            # A hot microphone is capture, not success/progress — it gets a
            # red recording dot, never green. "Recording stopped" stays
            # neutral (no em-dash, so the prefix check excludes it).
            hot = msg.startswith(("Recording —", "Transcribing",
                                  "Preparing model"))
            self._rec_status_var.set(("● " + msg) if hot else msg)
            if "error" in msg.lower() or "failed" in msg.lower():
                self._rec_status_label.configure(foreground=self.CLR_ERROR_FG)
            elif hot:
                self._rec_status_label.configure(foreground=C.BTN_DESTRUCTIVE_FG)
            else:
                self._rec_status_label.configure(foreground=C.TEXT_SECONDARY)
        elif kind == "rec_error":
            msg = payload.get("message", "Unknown error")
            self._rec_status_var.set(f"Error: {msg}")
            self._rec_status_label.configure(foreground=self.CLR_ERROR_FG)
            logger.error("Recording error: %s", msg)
        elif kind == "rec_level":
            level = payload.get("level", 0.0)
            try:
                bar_width = max(0, int(self._rec_level_frame.winfo_width() * level))
                self._rec_level_bar.place_configure(width=bar_width)
            except (tk.TclError, ValueError):
                pass
        elif kind == "rec_stopped":
            # Recording finished — reset the Record button (kept disabled by
            # _stop_recording while the recorder threads wound down) and the
            # status, unless an error message is showing.
            if self.record_button:
                colors = self._BTN_STYLES["secondary"]
                self.record_button.configure(
                    text="  Record  ",
                    background=colors["bg"],
                    foreground=colors["fg"],
                    activebackground=colors["press_bg"],
                )
                self.record_button._btn_colors = colors
                self._set_button_enabled(self.record_button, True)
            self._check_readiness()
            if self._rec_status_var.get() in ("Stopping...", "Starting..."):
                self._rec_status_var.set("Recording stopped")
                self._rec_status_label.configure(foreground=C.TEXT_SECONDARY)
            self._rec_level_bar.place_configure(width=0)

    # Worker methods (run_transcription_worker, load_model, transcribe_*,
    # write_outputs, cuda_available) extracted to hebrewscribe.worker.
    # TranscriberApp implements WorkerHost via: post_event, check_for_cancel,
    # wait_if_paused, cancel_requested, stop_requested, pause_event.

    def open_output_folder(self) -> None:
        path = self.output_dir_var.get().strip()
        if not path:
            return
        p = Path(path)
        if not p.exists():
            messagebox.showerror("Folder not found", "Output folder does not exist.")
            return
        try:
            _open_path(p)
        except Exception:
            logger.debug("Could not open folder %s", p, exc_info=True)
            messagebox.showinfo("Output folder", str(p))

    def _record_log(self, message: str, level: str = "info",
                    phase: Optional[str] = None, file_path: Optional[str] = None,
                    raw_only: bool = False) -> LogEntry:
        """Create a LogEntry and append it to the structured log.

        If phase is None, auto-classify from message content.
        Returns the created entry.
        """
        now = time.time()
        time_str = time.strftime("%H:%M:%S")

        # Auto-detect level from content if not explicitly set. Match only the
        # message head (text before the first colon, quoted spans removed):
        # worker templates put filenames after a colon ("Transcribing: <name>"),
        # and a filename that happens to contain "error"/"failed" must not turn
        # the line red or force-expand its activity node.
        if level == "info":
            head = re.sub(r"'[^']*'", "", message).split(":", 1)[0].lower()
            if "error" in head or "failed" in head or "traceback" in head:
                level = "error"
            elif "warning" in head or "warn" in head:
                level = "warning"

        # Auto-classify phase if not provided
        if phase is None:
            phase, auto_raw = _classify_log_phase(message, self._log_current_phase)
            if auto_raw:
                raw_only = True

        entry = LogEntry(
            timestamp=now,
            time_str=time_str,
            message=message.rstrip(),
            level=level,
            phase=phase,
            run_id=self._log_run_id,
            file_path=file_path or self._log_current_file,
            raw_only=raw_only,
        )
        self._log_entries.append(entry)

        # Update activity tree (if widget exists and entry is visible)
        if hasattr(self, "_activity_tree") and not raw_only:
            self._activity_insert(entry)

        return entry

    def log(self, message: str, level: str = "info",
            phase: Optional[str] = None, file_path: Optional[str] = None,
            raw_only: bool = False) -> None:
        """Write a message to both the raw log text and the structured log."""
        entry = self._record_log(message, level=level, phase=phase,
                                 file_path=file_path, raw_only=raw_only)
        # Always write to raw log
        self.log_text.insert("end", f"[{entry.time_str}] {entry.message}\n", entry.level)
        self.log_text.see("end")

    # --- Log view switching ---

    def _switch_log_view(self, view: str) -> None:
        """Toggle between 'activity' (structured) and 'raw' (flat text) log views."""
        self._activity_log_view = view
        if view == "activity":
            self._log_raw_frame.pack_forget()
            self._log_activity_frame.pack(fill="both", expand=True)
            # Visual toggle state
            self._log_activity_btn.configure(state="disabled")
            self._log_raw_btn.configure(state="normal")
        else:
            self._log_activity_frame.pack_forget()
            self._log_raw_frame.pack(fill="both", expand=True)
            self._log_raw_btn.configure(state="disabled")
            self._log_activity_btn.configure(state="normal")

    # --- Activity tree management ---

    def _activity_begin_run(self, file_count: int) -> None:
        """Create the Setup and Files group nodes for a new transcription run."""
        tree = self._activity_tree
        run_label = f"Run {self._log_run_id}" if self._log_run_id > 1 else ""

        # Setup group (collapsed by default)
        setup_text = "Setup" + (f" ({run_label})" if run_label else "")
        self._activity_setup_node = tree.insert(
            "", "end", text=setup_text,
            values=("", ""),
            tags=("group_setup",), open=False)

        # Files group (expanded)
        files_text = f"Files \u2014 0/{file_count}"
        self._activity_files_node = tree.insert(
            "", "end", text=files_text,
            values=("", ""),
            tags=("group_files",), open=True)

    def _activity_insert(self, entry: LogEntry) -> None:
        """Insert a LogEntry into the activity tree at the appropriate location."""
        tree = self._activity_tree
        if not tree.winfo_exists():
            return

        msg = entry.message
        ts = entry.time_str

        if entry.phase == "setup":
            # Nest under Setup group
            if self._activity_setup_node:
                # Shorten model paths for display
                display_msg = self._shorten_log_message(msg)
                tree.insert(self._activity_setup_node, "end",
                            text=f"  {display_msg}",
                            values=("", ts),
                            tags=("detail",))
                # Update Setup summary from key messages
                self._activity_update_setup_summary(msg)
            return

        if entry.phase == "file":
            # If this is a "Transcribing:" message, the file node was already
            # created by _activity_on_file_started. Nest detail under it.
            if entry.file_path and entry.file_path in self._activity_file_nodes:
                display_msg = self._shorten_log_message(msg)
                tag = "error_detail" if entry.level == "error" else "detail"
                tree.insert(self._activity_file_nodes[entry.file_path], "end",
                            text=f"  {display_msg}",
                            values=("", ts),
                            tags=(tag,))
                # Auto-expand file node on error
                if entry.level == "error":
                    tree.item(self._activity_file_nodes[entry.file_path], open=True)
            return

        if entry.phase == "results":
            # Create or update results node
            if not self._activity_results_node:
                self._activity_results_node = tree.insert(
                    "", "end", text=msg,
                    values=("", ts),
                    tags=("results",))
            else:
                # Append additional result lines (e.g., stats)
                tree.insert(self._activity_results_node, "end",
                            text=f"  {msg}",
                            values=("", ts),
                            tags=("detail",))
            tree.see(self._activity_results_node)
            return

        if entry.phase == "user":
            # Standalone user action (Added files, Pause, etc.)
            tree.insert("", "end",
                        text=msg,
                        values=("", ts),
                        tags=("user_action",))
            return

    def _activity_on_file_started(self, path_str: str, name: str) -> None:
        """Create a file node in the activity tree when a file starts processing."""
        if not hasattr(self, "_activity_tree") or not self._activity_files_node:
            return
        tree = self._activity_tree

        # Auto-collapse older file nodes (keep last 5 expanded)
        visible_file_nodes = list(self._activity_file_nodes.values())
        if len(visible_file_nodes) >= 5:
            for old_node in visible_file_nodes[:-4]:
                try:
                    tree.item(old_node, open=False)
                except tk.TclError:
                    pass

        node_id = tree.insert(
            self._activity_files_node, "end",
            text=f"  \u2022 {name}",
            values=("...", ""),
            tags=("file_running",), open=False)
        self._activity_file_nodes[path_str] = node_id
        tree.see(node_id)

    def _activity_on_file_finished(self, path_str: str) -> None:
        """Update a file node to show success with wall time."""
        if path_str not in self._activity_file_nodes:
            return
        tree = self._activity_tree
        node_id = self._activity_file_nodes[path_str]
        name = Path(path_str).name
        file_wall = time.time() - self._file_wall_starts.get(path_str, time.time())
        wall_str = format_hms(file_wall) if file_wall > 0 else "Done"

        tree.item(node_id, text=f"  \u2713 {name}", values=(wall_str, ""),
                  tags=("file_ok",))

        self._activity_files_done += 1
        self._activity_update_files_summary()

    def _activity_on_file_failed(self, path_str: str, error_msg: str,
                                status: str = "Failed") -> None:
        """Update a file node to show failure/cancellation and expand on error."""
        if path_str not in self._activity_file_nodes:
            return
        tree = self._activity_tree
        node_id = self._activity_file_nodes[path_str]
        name = Path(path_str).name

        if status == "Cancelled":
            # Cancelled files get a neutral dash, not a failure X
            tree.item(node_id, text=f"  \u2014 {name}",
                      values=("\u2014", ""),
                      tags=("user_action",))
        else:
            # Show first line of error
            first_line = ""
            if error_msg:
                first_line = error_msg.split("\n")[0].strip()[:80]
            tree.item(node_id, text=f"  \u2717 {name}",
                      values=("", ""),
                      tags=("file_fail",))
            if first_line:
                tree.insert(node_id, "end",
                            text=f"    {first_line}",
                            values=("", ""),
                            tags=("error_detail",))
                tree.item(node_id, open=True)

        self._activity_files_failed += 1
        self._activity_update_files_summary()

    def _activity_update_files_summary(self) -> None:
        """Update the 'Files' group node summary text."""
        if not self._activity_files_node:
            return
        done = self._activity_files_done
        failed = self._activity_files_failed
        total = self._activity_files_total
        parts = []
        if done > 0:
            parts.append(f"{done} done")
        if failed > 0:
            parts.append(f"{failed} failed")
        in_progress = total - done - failed
        if in_progress > 0 and (done > 0 or failed > 0):
            parts.append(f"{in_progress} queued")
        summary = ", ".join(parts) if parts else f"0/{total}"
        if not parts:
            summary = f"0/{total}"
        else:
            summary = f"{done + failed}/{total} \u2014 " + ", ".join(parts)
        self._activity_tree.item(self._activity_files_node,
                                 text=f"Files \u2014 {summary}")

    def _activity_update_setup_summary(self, msg: str) -> None:
        """Update the Setup group summary from model load messages."""
        if not self._activity_setup_node:
            return
        low = msg.lower()
        # Extract model + device info for summary
        if low.startswith("model loaded on device:"):
            device = msg.split(":", 1)[1].strip()
            # Try to get model name from earlier entries
            model_name = ""
            for e in reversed(self._log_entries):
                if e.run_id == self._log_run_id and e.phase == "setup":
                    if "loading faster-whisper" in e.message.lower():
                        # Extract short model name
                        model_name = self._shorten_model_name(e.message)
                        break
                    elif "loading openai-whisper" in e.message.lower():
                        model_name = e.message.split(":", 1)[1].strip() if ":" in e.message else ""
                        break
            summary = f"Setup \u2014 {model_name} \u00b7 {device}" if model_name else f"Setup \u2014 {device}"
            self._activity_tree.item(self._activity_setup_node, text=summary)

    def _activity_on_results(self, wall_seconds: float, done_n: int, failed_n: int,
                             cancelled: bool = False, stopped: bool = False) -> None:
        """Create the Results node at the end of a run."""
        if not hasattr(self, "_activity_tree"):
            return
        tree = self._activity_tree

        if cancelled:
            label = f"Results \u2014 Cancelled ({done_n} done, {failed_n} failed)"
        elif stopped:
            label = f"Results \u2014 Stopped ({done_n} done, {failed_n} failed)"
        else:
            # Build a short summary for the tree label
            parts = [f"{done_n} done"]
            if failed_n > 0:
                parts.append(f"{failed_n} failed")
            if wall_seconds > 0:
                parts.append(f"{format_hms(wall_seconds)} real time")
                speed = self._batch_audio_done / wall_seconds if self._batch_audio_done > 0 else 0
                if speed > 0:
                    parts.append(f"{speed:.1f}x speed")
            summary = " \u00b7 ".join(parts)
            label = f"Results \u2014 {summary}"

        self._activity_results_node = tree.insert(
            "", "end", text=label,
            values=(format_hms(wall_seconds) if wall_seconds > 0 else "", time.strftime("%H:%M:%S")),
            tags=("results",))
        tree.see(self._activity_results_node)

    @staticmethod
    def _shorten_model_name(msg: str) -> str:
        """Extract a short model name from a 'Loading ...' log message."""
        # "Loading faster-whisper model: C:\Users\...\models--Systran--faster-distil-whisper-large-v3\..."
        # or "Loading faster-whisper model: Systran/faster-distil-whisper-large-v3"
        if ":" not in msg:
            return ""
        spec = msg.split(":", 1)[1].strip()
        # Hub IDs are short: "org/name" (1 slash, not absolute, no backslash)
        if spec.count("/") == 1 and not Path(spec).is_absolute() and "\\" not in spec and len(spec) < 80:
            return spec
        # Long path — extract the model directory name
        if len(spec) > 60 or os.sep in spec or "\\" in spec:
            p = Path(spec)
            for part in p.parts:
                if part.startswith("models--"):
                    return part.replace("models--", "").replace("--", "/", 1)
            return p.name
        return spec

    @staticmethod
    def _shorten_log_message(msg: str) -> str:
        """Shorten filesystem paths in a log message for display."""
        # Replace any path-like string (>40 chars with path separators) with filename
        # Common patterns: "Transcribing: C:\long\path\file.wav" -> "Transcribing: file.wav"
        #                  "Loading faster-whisper model: C:\long\..." -> "Loading model: distil-large-v3"
        if ":" in msg:
            prefix, rest = msg.split(":", 1)
            rest = rest.strip()
            # Only shorten if it looks like an actual filesystem path (long, with separators)
            # Hub IDs like "Systran/name" are short and should NOT be shortened
            is_long_path = len(rest) > 40 and (
                ("\\" in rest and rest.count("\\") > 1)
                or ("/" in rest and rest.count("/") > 1)
            )
            if is_long_path:
                p = Path(rest)
                # For model paths, try to extract friendly name
                if "models--" in rest:
                    for part in p.parts:
                        if part.startswith("models--"):
                            short = part.replace("models--", "").replace("--", "/", 1)
                            return f"{prefix}: {short}"
                return f"{prefix}: {bidi_name(p.name)}"
        return msg

    def _activity_show_context_menu(self, event) -> None:
        """Show the activity tree context menu."""
        try:
            self._activity_menu.tk_popup(event.x_root, event.y_root)
        finally:
            self._activity_menu.grab_release()

    def _activity_expand_all(self) -> None:
        """Expand all nodes in the activity tree."""
        for item in self._activity_tree.get_children():
            self._activity_tree.item(item, open=True)
            for child in self._activity_tree.get_children(item):
                self._activity_tree.item(child, open=True)

    def _activity_collapse_all(self) -> None:
        """Collapse all nodes in the activity tree."""
        for item in self._activity_tree.get_children():
            self._activity_tree.item(item, open=False)
            for child in self._activity_tree.get_children(item):
                self._activity_tree.item(child, open=False)

    def _activity_copy_all(self) -> None:
        """Copy the entire activity tree content to clipboard as indented text."""
        lines = []
        def _walk(node, depth=0):
            text = self._activity_tree.item(node, "text").strip()
            vals = self._activity_tree.item(node, "values")
            wall = vals[0] if vals and vals[0] else ""
            ts = vals[1] if vals and len(vals) > 1 and vals[1] else ""
            indent = "  " * depth
            parts = [f"{indent}{text}"]
            if wall:
                parts.append(f"  {wall}")
            if ts:
                parts.append(f"  [{ts}]")
            lines.append("".join(parts))
            for child in self._activity_tree.get_children(node):
                _walk(child, depth + 1)
        for top in self._activity_tree.get_children():
            _walk(top)
        if lines:
            self.clipboard_clear()
            self.clipboard_append("\n".join(lines))

    def _show_about_dialog(self) -> None:
        """Show a simple About dialog with app name, version, and author."""
        from hebrewscribe import __version__
        dlg = tk.Toplevel(self)
        dlg.title(f"About {APP_TITLE}")
        dlg.resizable(False, False)
        dlg.transient(self)
        dlg.grab_set()
        if self._icon_path:
            dlg.iconbitmap(self._icon_path)
        frame = ttk.Frame(dlg, padding=30)
        frame.pack(fill="both", expand=True)
        # App icon
        _icon_base = getattr(sys, '_MEIPASS', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        _icon_png = os.path.join(_icon_base, 'hebrewscribe', 'icon-64.png')
        if not os.path.isfile(_icon_png):
            _icon_png = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'icon-64.png')
        if os.path.isfile(_icon_png):
            try:
                dlg._about_icon = tk.PhotoImage(file=_icon_png)  # prevent GC
                tk.Label(frame, image=dlg._about_icon).pack(pady=(0, 8))
            except Exception:
                pass
        ttk.Label(frame, text=APP_TITLE, font=(SYSTEM_FONT, _fs(18), "bold")).pack(pady=(0, 4))
        ttk.Label(frame, text=f"Version {__version__}",
                  font=self.FONT_LABEL, foreground=C.TEXT_MUTED).pack()
        ttk.Label(frame, text="by Yevgeniy Glider",
                  font=self.FONT_LABEL, foreground=C.TEXT_MUTED).pack(pady=(2, 0))
        # Homepage link
        _home_link = tk.Label(frame, text="yevgeniyglider.com/hebrewscribe",
                              font=(SYSTEM_FONT, _fs(9), "underline"),
                              foreground=C.ACCENT, cursor="hand2")
        _home_link.pack(pady=(2, 0))
        _home_link.bind("<Button-1>",
                        lambda e: __import__("webbrowser").open("https://yevgeniyglider.com/hebrewscribe/"))
        _home_link.bind("<Enter>", lambda e: _home_link.configure(foreground=C.ACCENT_PRESS))
        _home_link.bind("<Leave>", lambda e: _home_link.configure(foreground=C.ACCENT))
        ttk.Label(frame, text="Batch Hebrew speech-to-text using local Whisper models",
                  font=self.FONT_SMALL, foreground=C.TEXT_TERTIARY, justify="center").pack(pady=(12, 0))
        # License link
        def _open_license(event=None):
            base = getattr(sys, '_MEIPASS', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            path = os.path.join(base, 'LICENSE')
            if os.path.isfile(path):
                _open_path(path)
            else:
                __import__("webbrowser").open("https://github.com/yevgeniyglider/HebrewScribe/blob/main/LICENSE")
        _mit_link = tk.Label(frame, text="Free and open source (MIT)",
                             font=(SYSTEM_FONT, _fs(9), "underline"),
                             foreground=C.ACCENT, cursor="hand2")
        _mit_link.pack(pady=(6, 0))
        _mit_link.bind("<Button-1>", _open_license)
        _mit_link.bind("<Enter>", lambda e: _mit_link.configure(foreground=C.ACCENT_PRESS))
        _mit_link.bind("<Leave>", lambda e: _mit_link.configure(foreground=C.ACCENT))
        # Third-party licenses link
        def _open_licenses(event=None):
            # PyInstaller bundle: file is next to the exe
            base = getattr(sys, '_MEIPASS', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            path = os.path.join(base, 'THIRD-PARTY-LICENSES.md')
            if os.path.isfile(path):
                _open_path(path)
            else:
                __import__("webbrowser").open("https://github.com/yevgeniyglider/HebrewScribe/blob/main/THIRD-PARTY-LICENSES.md")
        _lic_link = tk.Label(frame, text="Third-party licenses",
                             font=(SYSTEM_FONT, _fs(9), "underline"),
                             foreground=C.ACCENT, cursor="hand2")
        _lic_link.pack(pady=(2, 0))
        _lic_link.bind("<Button-1>", _open_licenses)
        _lic_link.bind("<Enter>", lambda e: _lic_link.configure(foreground=C.ACCENT_PRESS))
        _lic_link.bind("<Leave>", lambda e: _lic_link.configure(foreground=C.ACCENT))
        ok_btn = self._make_button(frame, text="OK", command=dlg.destroy)
        ok_btn.pack(pady=(18, 0))
        dlg.bind("<Escape>", lambda e: dlg.destroy())
        dlg.bind("<Return>", lambda e: dlg.destroy())
        # Center on parent, clamped to the screen: this dialog holds a grab,
        # so if it landed fully off-screen the app would look frozen. min
        # before max so the top-left corner wins when both bounds conflict.
        dlg.update_idletasks()
        x = self.winfo_x() + (self.winfo_width() - dlg.winfo_width()) // 2
        y = self.winfo_y() + (self.winfo_height() - dlg.winfo_height()) // 2
        x = max(0, min(x, self.winfo_screenwidth() - dlg.winfo_width()))
        y = max(0, min(y, self.winfo_screenheight() - dlg.winfo_height()))
        dlg.geometry(f"+{x}+{y}")
        dlg.focus_set()

    def save_config_from_ui(self) -> None:
        self.config_data["last_output_dir"] = self.output_dir_var.get().strip()
        self.config_data["last_backend"] = self.backend_var.get().strip()
        self.config_data["last_language"] = self.language_var.get().strip()
        # Persist per-language model choice for smart restoration
        lang = self.language_var.get().strip()
        if lang and self.model_var.get():
            self.config_data[f"last_model_{lang}"] = self.model_var.get()
        self.config_data["last_task"] = self.task_var.get().strip()
        self.config_data["last_device"] = self.device_var.get().strip()
        self.config_data["last_compute_type"] = self.compute_type_var.get().strip()
        # A mid-edit or non-numeric entry keeps the previously saved value —
        # this runs from tk callbacks (recording toggle, window close) where a
        # bare int() would raise into the crash dialog.
        try:
            self.config_data["last_beam_size"] = int(self.beam_size_var.get())
        except (ValueError, TypeError):
            pass
        self.config_data["last_vad_filter"] = self.vad_var.get()
        self.config_data["last_condition_on_previous_text"] = self.condition_on_previous_text_var.get()
        try:
            self.config_data["last_batch_size"] = int(self.batch_size_var.get())
        except (ValueError, TypeError):
            pass
        self.config_data["last_speed_preset"] = self.speed_preset_var.get()
        self.config_data["last_formats"] = [fmt for fmt, var in self.format_vars.items() if var.get()]
        self.config_data["prevent_sleep"] = self.prevent_sleep_var.get()
        self.config_data["experimental_recording"] = self._experimental_recording_var.get()
        self.config_data["last_diarize"] = self.diarize_var.get()
        try:
            self.config_data["last_num_speakers"] = int(
                self.num_speakers_var.get())
        except (ValueError, TypeError):
            pass
        self.config_data["diarize_device"] = self.diarize_device_var.get().strip()
        self.save_config()

    def save_config(self) -> None:
        save_json(CONFIG_PATH, self.config_data)


def _enable_dpi_awareness() -> None:
    """Tell Windows this process renders at native DPI (no bitmap scaling).

    Must be called before any tkinter window is created.  No-op on
    non-Windows platforms or if the call fails (e.g. older Windows).
    """
    if sys.platform != "win32":
        return
    try:
        import ctypes
        # PROCESS_PER_MONITOR_DPI_AWARE = 2 (best; per-monitor scaling)
        # Falls back to PROCESS_SYSTEM_DPI_AWARE = 1 on older builds.
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except (AttributeError, OSError):
            ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass  # non-critical; app just stays bitmap-scaled


def main() -> None:
    setup_logging()
    _enable_dpi_awareness()

    # Global exception hooks — catch anything that slips past try/except
    _original_excepthook = sys.excepthook

    def _global_excepthook(exc_type, exc_value, exc_tb):
        import traceback as _tb
        logger.critical(
            "Unhandled exception:\n%s",
            "".join(_tb.format_exception(exc_type, exc_value, exc_tb)),
        )
        _original_excepthook(exc_type, exc_value, exc_tb)

    sys.excepthook = _global_excepthook

    def _thread_excepthook(args):
        if args.exc_type is SystemExit:
            return
        logger.critical(
            "Unhandled exception in thread %s",
            args.thread.name if args.thread else "unknown",
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
        )

    threading.excepthook = _thread_excepthook

    app = TranscriberApp()
    app.mainloop()


if __name__ == "__main__":
    main()
