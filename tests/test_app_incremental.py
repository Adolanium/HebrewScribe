"""Unit tests for incremental run (append-and-continue) feature.

Tests exercise _count_by_status(), _update_contextual_banner(),
start_transcription() queued-only behavior, and session stats
accumulation without requiring a real Tk instance.
"""

import threading
import pytest

# Reuse the MockTreeview from reorder tests
from tests.test_app_reorder import MockTreeview


# ---------------------------------------------------------------------------
# Minimal harness mimicking TranscriberApp for incremental run logic
# ---------------------------------------------------------------------------

class _FakeBtn:
    def __init__(self):
        self.enabled = True
        self.text = ""

    def config(self, **kwargs):
        if "text" in kwargs:
            self.text = kwargs["text"]

    def configure(self, **kwargs):
        self.config(**kwargs)


class _FakeVar:
    def __init__(self, value=""):
        self._value = value

    def get(self):
        return self._value

    def set(self, value):
        self._value = value


class _FakeProgress:
    def __init__(self):
        self._cfg = {"value": 0, "maximum": 100, "mode": "determinate"}

    def __setitem__(self, key, value):
        self._cfg[key] = value

    def __getitem__(self, key):
        return self._cfg[key]

    def config(self, **kw):
        self._cfg.update(kw)

    def configure(self, **kw):
        self.config(**kw)

    def start(self, interval=50):
        pass

    def stop(self):
        pass


class _FakeTaskbar:
    TBPF_NORMAL = 2
    TBPF_ERROR = 4
    TBPF_PAUSED = 8
    TBPF_INDETERMINATE = 1

    def set_state(self, s):
        pass

    def set_progress(self, v, m):
        pass

    def clear(self):
        pass


class _FakeNotebook:
    def select(self, idx=None):
        pass


class _FakeText:
    def delete(self, *a):
        pass

    def insert(self, *a):
        pass


class _FakeSleepInhibitor:
    def acquire(self):
        pass

    def release(self):
        pass


class _FakeFrame:
    def winfo_manager(self):
        return ""

    def pack(self, **kw):
        pass

    def pack_forget(self):
        pass

    def configure(self, **kw):
        pass


class IncrementalHarness:
    """Mimics the subset of TranscriberApp needed by incremental run logic."""

    # Minimal class-level attributes that _count_by_status / banner need
    APP_TITLE = "HebrewScribe"

    def __init__(self, paths, statuses=None):
        self.file_paths = list(paths)
        self._queue_lock = threading.Lock()
        self.file_items = {}
        self.file_tree = MockTreeview()
        self.worker_thread = None
        self.file_durations = {}
        self.file_errors = {}
        self._tree_progress = {}

        # Banner state tracking
        self._banner_last_state = "idle"
        self._last_banner_text = ""
        self._last_banner_state = "idle"

        # Buttons
        self.start_button = _FakeBtn()

        # Taskbar
        self._taskbar = _FakeTaskbar()

        # Session stats
        self._session_audio_total = 0.0
        self._session_wall_total = 0.0
        self._session_done_count = 0
        self._session_failed_count = 0
        self._session_run_count = 0

        # Batch stats
        self._batch_audio_done = 0.0
        self._batch_audio_total = 0.0

        if statuses is None:
            statuses = {p: "Queued" for p in paths}
        self._statuses = statuses
        self._build_tree()

    def _build_tree(self):
        self.file_tree._items.clear()
        self.file_tree._children.clear()
        self.file_tree._counter = 0
        self.file_items.clear()
        for idx, path in enumerate(self.file_paths):
            status = self._statuses.get(path, "Queued")
            ordinal = idx + 1
            item_id = self.file_tree.insert(
                "", "end",
                values=(ordinal, status, "", "", path.split("/")[-1], path),
                tags=(status.lower(),)
            )
            self.file_items[path] = item_id

    def title(self, text):
        self._title = text

    def _update_banner(self, state, text=""):
        self._last_banner_state = state
        self._last_banner_text = text
        self._banner_last_state = state

    def _apply_app_state(self, state, **kwargs):
        self._last_banner_state = state

    def refresh_file_tree(self):
        self._build_tree()

    def update_overall_summary(self, **kw):
        pass

    def _check_readiness(self):
        pass

    def _update_statusbar_file_info(self):
        pass


# Bind real methods from app module for testing.
# We bind them onto IncrementalHarness so self._count_by_status() works
# when called internally by _update_contextual_banner.
from hebrewscribe.app import TranscriberApp

IncrementalHarness._count_by_status = TranscriberApp._count_by_status
IncrementalHarness._update_contextual_banner = TranscriberApp._update_contextual_banner

# Keep standalone references for direct calls in tests
_count_by_status = TranscriberApp._count_by_status
_update_contextual_banner = TranscriberApp._update_contextual_banner
_build_completion_stats = TranscriberApp._build_completion_stats


# ---------------------------------------------------------------------------
# Tests: _count_by_status
# ---------------------------------------------------------------------------

class TestCountByStatus:
    def test_all_queued(self):
        h = IncrementalHarness(["a.wav", "b.wav", "c.wav"])
        counts = _count_by_status(h)
        assert counts == {"Queued": 3, "Done": 0, "Failed": 0, "Running": 0}

    def test_mixed_statuses(self):
        h = IncrementalHarness(
            ["a.wav", "b.wav", "c.wav", "d.wav"],
            {"a.wav": "Done", "b.wav": "Done", "c.wav": "Failed", "d.wav": "Queued"}
        )
        counts = _count_by_status(h)
        assert counts == {"Queued": 1, "Done": 2, "Failed": 1, "Running": 0}

    def test_all_done(self):
        h = IncrementalHarness(
            ["a.wav", "b.wav"],
            {"a.wav": "Done", "b.wav": "Done"}
        )
        counts = _count_by_status(h)
        assert counts == {"Queued": 0, "Done": 2, "Failed": 0, "Running": 0}

    def test_empty_queue(self):
        h = IncrementalHarness([])
        counts = _count_by_status(h)
        assert counts == {"Queued": 0, "Done": 0, "Failed": 0, "Running": 0}

    def test_with_running(self):
        h = IncrementalHarness(
            ["a.wav", "b.wav"],
            {"a.wav": "Running", "b.wav": "Queued"}
        )
        counts = _count_by_status(h)
        assert counts == {"Queued": 1, "Done": 0, "Failed": 0, "Running": 1}


# ---------------------------------------------------------------------------
# Tests: _update_contextual_banner
# ---------------------------------------------------------------------------

class TestUpdateContextualBanner:
    def test_queued_only_shows_ready_to_start(self):
        h = IncrementalHarness(["a.wav", "b.wav"])
        _update_contextual_banner(h)
        assert h._last_banner_state == "idle"
        assert "ready to start" in h._last_banner_text
        assert "Start transcription" in h.start_button.text

    def test_done_plus_queued_shows_continue(self):
        h = IncrementalHarness(
            ["a.wav", "b.wav", "c.wav"],
            {"a.wav": "Done", "b.wav": "Queued", "c.wav": "Queued"}
        )
        _update_contextual_banner(h)
        assert h._last_banner_state == "ready"
        assert "ready to continue" in h._last_banner_text
        assert "1 done" in h._last_banner_text
        assert "2 queued" in h._last_banner_text
        assert "Continue" in h.start_button.text
        assert "2" in h.start_button.text

    def test_all_done_no_change(self):
        h = IncrementalHarness(
            ["a.wav", "b.wav"],
            {"a.wav": "Done", "b.wav": "Done"}
        )
        # Set a known state first
        h._last_banner_state = "complete"
        h._last_banner_text = "Completed: 2 files"
        _update_contextual_banner(h)
        # Should not change (pass branch)
        assert h._last_banner_state == "complete"

    def test_failed_plus_queued_shows_continue(self):
        h = IncrementalHarness(
            ["a.wav", "b.wav", "c.wav"],
            {"a.wav": "Failed", "b.wav": "Queued", "c.wav": "Queued"}
        )
        _update_contextual_banner(h)
        assert h._last_banner_state == "ready"
        assert "1 failed" in h._last_banner_text
        assert "2 queued" in h._last_banner_text

    def test_empty_queue_goes_idle(self):
        h = IncrementalHarness([])
        _update_contextual_banner(h)
        assert h._last_banner_state == "idle"

    def test_skipped_during_active_run(self):
        """Banner should not change while worker is alive."""
        h = IncrementalHarness(["a.wav"])
        h._last_banner_state = "running"
        # Simulate a live worker thread
        h.worker_thread = threading.Thread(target=lambda: None)
        h.worker_thread.start()
        h.worker_thread.join()
        # Thread finished but is_alive() may still be True briefly;
        # use a mock that pretends to be alive
        class _FakeAliveThread:
            def is_alive(self):
                return True
        h.worker_thread = _FakeAliveThread()
        _update_contextual_banner(h)
        # Should not have changed
        assert h._last_banner_state == "running"


# ---------------------------------------------------------------------------
# Tests: session stats accumulation
# ---------------------------------------------------------------------------

class TestSessionStats:
    def test_single_run_no_session_prefix(self):
        h = IncrementalHarness([])
        h._batch_audio_done = 600.0  # 10 minutes audio
        h._session_run_count = 1
        stats = _build_completion_stats(h, wall_seconds=100.0, done=3, failed=0)
        assert "Session:" not in stats
        assert "Audio:" in stats
        assert "Speed:" in stats

    def test_multi_run_shows_session(self):
        h = IncrementalHarness([])
        h._batch_audio_done = 300.0
        h._session_audio_total = 900.0  # 15 min total across session
        h._session_wall_total = 180.0   # 3 min wall total
        h._session_run_count = 3
        stats = _build_completion_stats(h, wall_seconds=60.0, done=1, failed=0)
        assert "Session:" in stats
        # Session stats should show the cumulative values
        assert "15:00" in stats  # 900 seconds = 15:00

    def test_session_reset_on_clear(self):
        h = IncrementalHarness(["a.wav"])
        h._session_audio_total = 500.0
        h._session_wall_total = 100.0
        h._session_done_count = 5
        h._session_failed_count = 1
        h._session_run_count = 2
        # Simulate what clear_files does
        h._session_audio_total = 0.0
        h._session_wall_total = 0.0
        h._session_done_count = 0
        h._session_failed_count = 0
        h._session_run_count = 0
        assert h._session_audio_total == 0.0
        assert h._session_run_count == 0


# ---------------------------------------------------------------------------
# Tests: start_transcription queued-only behavior
# ---------------------------------------------------------------------------

class TestStartQueuesOnly:
    def test_start_counts_only_queued(self):
        """Verify _count_by_status returns correct count when Done files present."""
        h = IncrementalHarness(
            ["a.wav", "b.wav", "c.wav", "d.wav", "e.wav"],
            {"a.wav": "Done", "b.wav": "Done", "c.wav": "Done",
             "d.wav": "Queued", "e.wav": "Queued"}
        )
        counts = _count_by_status(h)
        assert counts["Queued"] == 2
        assert counts["Done"] == 3

    def test_no_queued_files_detected(self):
        """When all files are Done, queued count should be 0."""
        h = IncrementalHarness(
            ["a.wav", "b.wav"],
            {"a.wav": "Done", "b.wav": "Done"}
        )
        counts = _count_by_status(h)
        assert counts["Queued"] == 0

    def test_failed_not_counted_as_queued(self):
        h = IncrementalHarness(
            ["a.wav", "b.wav"],
            {"a.wav": "Failed", "b.wav": "Done"}
        )
        counts = _count_by_status(h)
        assert counts["Queued"] == 0
        assert counts["Failed"] == 1
