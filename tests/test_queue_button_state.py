"""Unit tests for _update_queue_button_state (Remove/Clear toolbar button states).

Verifies that:
- Remove button is disabled when nothing is selected, enabled when selected
- Clear button is disabled when queue is empty or a job is running
- Clear button is enabled when queue has files and no job is running
- State updates correctly after append, remove, clear, and run transitions
"""

import threading
from pathlib import Path

import pytest

from tests.test_app_reorder import MockTreeview
from hebrewscribe.app import TranscriberApp


# ---------------------------------------------------------------------------
# Fake button that records its enabled/disabled state
# ---------------------------------------------------------------------------

class _TrackingBtn:
    """Minimal tk.Button stand-in that records state changes."""

    def __init__(self):
        self._state = "normal"
        self._btn_colors = TranscriberApp._BTN_STYLES["secondary"]

    def configure(self, **kw):
        if "state" in kw:
            self._state = kw["state"]

    def config(self, **kw):
        self.configure(**kw)

    def cget(self, key):
        if key == "state":
            return self._state
        return ""

    @property
    def enabled(self):
        return self._state == "normal"

    @property
    def disabled(self):
        return self._state == "disabled"


class _FakeThread:
    """Simulates a worker thread that is alive or dead."""

    def __init__(self, alive: bool):
        self._alive = alive

    def is_alive(self):
        return self._alive


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

class QueueButtonHarness:
    """Minimal harness for testing _update_queue_button_state."""

    _BTN_STYLES = TranscriberApp._BTN_STYLES

    def __init__(self, paths=None):
        self.file_paths = list(paths or [])
        self.file_items = {}
        self.file_tree = MockTreeview()
        self.worker_thread = None
        self._btn_remove = _TrackingBtn()
        self._btn_clear = _TrackingBtn()

        # Build tree if paths provided
        for idx, p in enumerate(self.file_paths):
            item_id = self.file_tree.insert(
                "", "end",
                values=(idx + 1, "Queued", "", "", Path(p).name, p),
                tags=("queued",),
            )
            self.file_items[p] = item_id


# Bind the real methods
QueueButtonHarness._set_button_enabled = TranscriberApp._set_button_enabled
QueueButtonHarness._update_queue_button_state = TranscriberApp._update_queue_button_state


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRemoveButtonState:
    """Remove button should be enabled iff something is selected."""

    def test_disabled_when_empty_queue(self):
        h = QueueButtonHarness([])
        h._update_queue_button_state()
        assert h._btn_remove.disabled

    def test_disabled_when_files_but_no_selection(self):
        h = QueueButtonHarness(["a.wav", "b.wav"])
        h._update_queue_button_state()
        assert h._btn_remove.disabled

    def test_enabled_when_file_selected(self):
        h = QueueButtonHarness(["a.wav", "b.wav"])
        h.file_tree.selection_set(h.file_items["a.wav"])
        h._update_queue_button_state()
        assert h._btn_remove.enabled

    def test_enabled_with_multiple_selection(self):
        h = QueueButtonHarness(["a.wav", "b.wav", "c.wav"])
        h.file_tree.selection_set(h.file_items["a.wav"], h.file_items["c.wav"])
        h._update_queue_button_state()
        assert h._btn_remove.enabled

    def test_disabled_after_deselect(self):
        h = QueueButtonHarness(["a.wav"])
        h.file_tree.selection_set(h.file_items["a.wav"])
        h._update_queue_button_state()
        assert h._btn_remove.enabled
        # Deselect
        h.file_tree._selection.clear()
        h._update_queue_button_state()
        assert h._btn_remove.disabled

    def test_enabled_during_transcription_if_selected(self):
        """Remove is allowed during transcription (thread-safe via _queue_lock)."""
        h = QueueButtonHarness(["a.wav"])
        h.worker_thread = _FakeThread(alive=True)
        h.file_tree.selection_set(h.file_items["a.wav"])
        h._update_queue_button_state()
        assert h._btn_remove.enabled


class TestClearButtonState:
    """Clear button should be enabled iff queue is non-empty and no job running."""

    def test_disabled_when_empty_queue(self):
        h = QueueButtonHarness([])
        h._update_queue_button_state()
        assert h._btn_clear.disabled

    def test_enabled_when_files_exist_no_job(self):
        h = QueueButtonHarness(["a.wav", "b.wav"])
        h._update_queue_button_state()
        assert h._btn_clear.enabled

    def test_disabled_during_transcription(self):
        h = QueueButtonHarness(["a.wav"])
        h.worker_thread = _FakeThread(alive=True)
        h._update_queue_button_state()
        assert h._btn_clear.disabled

    def test_enabled_after_transcription_ends(self):
        h = QueueButtonHarness(["a.wav"])
        h.worker_thread = _FakeThread(alive=True)
        h._update_queue_button_state()
        assert h._btn_clear.disabled
        # Transcription ends
        h.worker_thread = _FakeThread(alive=False)
        h._update_queue_button_state()
        assert h._btn_clear.enabled

    def test_disabled_after_all_files_removed(self):
        h = QueueButtonHarness(["a.wav"])
        h._update_queue_button_state()
        assert h._btn_clear.enabled
        # Simulate clearing
        h.file_paths.clear()
        h._update_queue_button_state()
        assert h._btn_clear.disabled

    def test_enabled_after_files_added_to_empty_queue(self):
        h = QueueButtonHarness([])
        h._update_queue_button_state()
        assert h._btn_clear.disabled
        # Add a file
        h.file_paths.append("new.wav")
        h._update_queue_button_state()
        assert h._btn_clear.enabled

    def test_worker_thread_none_treated_as_not_running(self):
        h = QueueButtonHarness(["a.wav"])
        h.worker_thread = None
        h._update_queue_button_state()
        assert h._btn_clear.enabled


class TestBothButtonsInteraction:
    """Combined state transitions across both buttons."""

    def test_empty_queue_both_disabled(self):
        h = QueueButtonHarness([])
        h._update_queue_button_state()
        assert h._btn_remove.disabled
        assert h._btn_clear.disabled

    def test_files_no_selection_clear_enabled_remove_disabled(self):
        h = QueueButtonHarness(["a.wav"])
        h._update_queue_button_state()
        assert h._btn_remove.disabled
        assert h._btn_clear.enabled

    def test_files_with_selection_both_enabled(self):
        h = QueueButtonHarness(["a.wav"])
        h.file_tree.selection_set(h.file_items["a.wav"])
        h._update_queue_button_state()
        assert h._btn_remove.enabled
        assert h._btn_clear.enabled

    def test_running_with_selection_remove_enabled_clear_disabled(self):
        h = QueueButtonHarness(["a.wav"])
        h.worker_thread = _FakeThread(alive=True)
        h.file_tree.selection_set(h.file_items["a.wav"])
        h._update_queue_button_state()
        assert h._btn_remove.enabled
        assert h._btn_clear.disabled
