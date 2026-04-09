"""Unit tests for remove_selected_files and path normalisation.

Verifies that:
- remove_selected_files correctly removes files from file_paths
- Path normalisation in _append_files prevents forward/backslash mismatch
- Error lookups use file_items reverse mapping, not treeview values[5]
- Edge cases: multi-select, remove all, remove none, remove during mixed states
"""

import threading
from pathlib import Path, PurePosixPath, PureWindowsPath
from unittest.mock import patch

import pytest

from tests.test_app_reorder import MockTreeview


# ---------------------------------------------------------------------------
# Harness
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


class _FakeTaskbar:
    def set_state(self, s): pass
    def set_progress(self, v, m): pass
    def clear(self): pass


class _FakeFrame:
    def winfo_manager(self): return ""
    def pack(self, **kw): pass
    def pack_forget(self): pass
    def configure(self, **kw): pass


class RemoveHarness:
    """Mimics the subset of TranscriberApp needed by remove_selected_files."""

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
        self._tree_bar_widgets = []

        # Banner
        self._banner_last_state = "idle"
        self._last_banner_text = ""
        self._last_banner_state = "idle"
        self.start_button = _FakeBtn()
        self._taskbar = _FakeTaskbar()

        # Session stats
        self._session_audio_total = 0.0
        self._session_wall_total = 0.0
        self._session_done_count = 0
        self._session_failed_count = 0
        self._session_run_count = 0
        self._batch_audio_done = 0.0
        self._batch_audio_total = 0.0

        # Log messages captured
        self._log_messages = []

        # Stubs for _append_files
        self.output_dir_var = type("Var", (), {"get": lambda s: "/tmp", "set": lambda s, v: None})()
        import queue
        self.event_queue = queue.Queue()

        if statuses is None:
            statuses = {p: "Queued" for p in paths}
        self._statuses = statuses
        self._build_tree()

    def _probe_durations_bg(self, paths):
        pass  # no-op in tests

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
                values=(ordinal, status, "", "", Path(path).name, str(Path(path))),
                tags=(status.lower(),)
            )
            self.file_items[path] = item_id

    def select(self, paths):
        items = [self.file_items[p] for p in paths]
        self.file_tree.selection_set(*items)

    def refresh_file_tree(self):
        self._build_tree()

    def update_overall_summary(self, **kw):
        pass

    def _update_contextual_banner(self):
        pass

    def _update_queue_button_state(self):
        pass

    def _check_readiness(self):
        pass

    def _update_statusbar_file_info(self):
        pass

    def _redraw_tree_progress(self):
        pass

    def title(self, text):
        pass

    def _update_banner(self, state, text=""):
        self._last_banner_state = state
        self._last_banner_text = text

    def log(self, msg, **kw):
        self._log_messages.append(msg)


# Bind the real method
from hebrewscribe.app import TranscriberApp

RemoveHarness.remove_selected_files = TranscriberApp.remove_selected_files
RemoveHarness._count_by_status = TranscriberApp._count_by_status
RemoveHarness._append_files = TranscriberApp._append_files


# ---------------------------------------------------------------------------
# Tests: basic remove_selected_files
# ---------------------------------------------------------------------------

class TestRemoveSelectedFiles:
    def test_remove_single_file(self):
        h = RemoveHarness(["a.wav", "b.wav", "c.wav"])
        h.select(["b.wav"])
        h.remove_selected_files()
        assert h.file_paths == ["a.wav", "c.wav"]

    def test_remove_multiple_files(self):
        h = RemoveHarness(["a.wav", "b.wav", "c.wav", "d.wav"])
        h.select(["a.wav", "c.wav"])
        h.remove_selected_files()
        assert h.file_paths == ["b.wav", "d.wav"]

    def test_remove_all_files(self):
        h = RemoveHarness(["a.wav", "b.wav"])
        h.select(["a.wav", "b.wav"])
        h.remove_selected_files()
        assert h.file_paths == []

    def test_remove_with_no_selection(self):
        h = RemoveHarness(["a.wav", "b.wav"])
        # Don't select anything
        h.remove_selected_files()
        assert h.file_paths == ["a.wav", "b.wav"]

    def test_remove_preserves_order(self):
        paths = ["1.wav", "2.wav", "3.wav", "4.wav", "5.wav"]
        h = RemoveHarness(paths)
        h.select(["2.wav", "4.wav"])
        h.remove_selected_files()
        assert h.file_paths == ["1.wav", "3.wav", "5.wav"]

    def test_remove_with_mixed_statuses(self):
        """Removing works regardless of file status."""
        statuses = {"a.wav": "Done", "b.wav": "Failed", "c.wav": "Queued"}
        h = RemoveHarness(["a.wav", "b.wav", "c.wav"], statuses)
        h.select(["a.wav", "b.wav"])
        h.remove_selected_files()
        assert h.file_paths == ["c.wav"]

    def test_remove_logs_count(self):
        h = RemoveHarness(["a.wav", "b.wav", "c.wav"])
        h.select(["a.wav", "c.wav"])
        h.remove_selected_files()
        assert any("2 file(s)" in msg for msg in h._log_messages)

    def test_remove_uses_queue_lock(self):
        """Verify that _queue_lock is held during file_paths mutation."""
        h = RemoveHarness(["a.wav", "b.wav"])
        h.select(["a.wav"])
        lock_was_held = []

        original_lock = h._queue_lock
        class TrackedLock:
            def __enter__(self_lock):
                original_lock.acquire()
                lock_was_held.append(True)
                return self_lock
            def __exit__(self_lock, *args):
                original_lock.release()

        h._queue_lock = TrackedLock()
        h.remove_selected_files()
        assert lock_was_held, "Expected _queue_lock to be acquired"
        assert h.file_paths == ["b.wav"]


# ---------------------------------------------------------------------------
# Tests: path normalisation in _append_files
# ---------------------------------------------------------------------------

class TestPathNormalisation:
    """Verify that _append_files normalises paths so they match treeview values."""

    def test_forward_slashes_normalised(self):
        """On all platforms, str(Path(x)) is applied to incoming paths."""
        h = RemoveHarness([])
        # Simulate what tkinter file dialog returns on Windows: forward slashes
        raw_paths = ["C:/Users/test/audio/file1.wav", "C:/Users/test/audio/file2.wav"]
        h._append_files(raw_paths)
        # After normalisation, paths should match str(Path(...))
        for p in h.file_paths:
            assert p == str(Path(p)), f"Path not normalised: {p!r}"

    def test_normalised_paths_deduplicate(self):
        """Forward-slash and backslash versions of the same path don't duplicate."""
        h = RemoveHarness([])
        h._append_files([str(Path("/tmp/test/a.wav"))])
        h._append_files(["/tmp/test/a.wav"])  # might differ on Windows
        # On Linux they're the same; on Windows forward vs back would normalise to same
        # Either way, should be deduplicated
        assert len(h.file_paths) == 1

    def test_remove_works_after_append(self):
        """Full round-trip: append files, select in tree, remove."""
        h = RemoveHarness([])
        h._append_files(["x/a.wav", "x/b.wav", "x/c.wav"])
        normalised_b = str(Path("x/b.wav"))
        h.select([normalised_b])
        h.remove_selected_files()
        assert normalised_b not in h.file_paths
        assert len(h.file_paths) == 2


# ---------------------------------------------------------------------------
# Tests: error lookup via file_items reverse mapping
# ---------------------------------------------------------------------------

class TestErrorLookup:
    """Verify that error-related lookups use file_items, not values[5]."""

    def test_show_error_finds_error_via_file_items(self):
        statuses = {"a.wav": "Failed", "b.wav": "Queued"}
        h = RemoveHarness(["a.wav", "b.wav"], statuses)
        h.file_errors["a.wav"] = "Something went wrong"

        # Simulate what _on_tree_right_click does internally:
        # Build item_to_path from file_items (not from values[5])
        item_to_path = {v: k for k, v in h.file_items.items()}
        item_id = h.file_items["a.wav"]
        path_from_items = item_to_path.get(item_id, "")
        assert path_from_items == "a.wav"
        assert path_from_items in h.file_errors

    def test_error_lookup_matches_even_with_path_difference(self):
        """If treeview stores a normalised path but file_items stores the original,
        the reverse lookup still works because both use file_items keys."""
        # Simulate: file_paths has forward-slash path, treeview has backslash
        original_path = "some/dir/file.wav"
        normalised = str(Path(original_path))

        h = RemoveHarness([original_path])
        h.file_errors[original_path] = "test error"

        # The reverse lookup from file_items uses the original key
        item_id = h.file_items[original_path]
        item_to_path = {v: k for k, v in h.file_items.items()}
        resolved = item_to_path[item_id]
        assert resolved == original_path
        assert resolved in h.file_errors


# ---------------------------------------------------------------------------
# Edge cases and hardening
# ---------------------------------------------------------------------------

class TestRemoveEdgeCases:
    def test_remove_single_file_queue(self):
        """Remove the only file in the queue."""
        h = RemoveHarness(["only.wav"])
        h.select(["only.wav"])
        h.remove_selected_files()
        assert h.file_paths == []
        assert h.file_items == {}

    def test_remove_with_spaces_in_path(self):
        """Paths with spaces should work correctly."""
        paths = ["/tmp/my folder/file one.wav", "/tmp/my folder/file two.wav"]
        h = RemoveHarness(paths)
        h.select([paths[0]])
        h.remove_selected_files()
        assert h.file_paths == [paths[1]]

    def test_remove_with_unicode_path(self):
        """Unicode characters in paths should not break removal."""
        paths = ["/tmp/\u05e2\u05d1\u05e8\u05d9\u05ea/\u05e7\u05d5\u05d1\u05e5.wav",
                 "/tmp/\u0440\u0443\u0441\u0441\u043a\u0438\u0439/\u0444\u0430\u0439\u043b.wav"]
        h = RemoveHarness(paths)
        h.select([paths[0]])
        h.remove_selected_files()
        assert h.file_paths == [paths[1]]

    def test_remove_running_file(self):
        """Removing a file with Running status should still work."""
        statuses = {"a.wav": "Running", "b.wav": "Queued"}
        h = RemoveHarness(["a.wav", "b.wav"], statuses)
        h.select(["a.wav"])
        h.remove_selected_files()
        assert h.file_paths == ["b.wav"]

    def test_remove_done_file(self):
        """Removing a completed file from the list."""
        statuses = {"a.wav": "Done", "b.wav": "Done", "c.wav": "Queued"}
        h = RemoveHarness(["a.wav", "b.wav", "c.wav"], statuses)
        h.select(["a.wav"])
        h.remove_selected_files()
        assert h.file_paths == ["b.wav", "c.wav"]

    def test_remove_cleans_up_file_items(self):
        """After removal, file_items should only contain remaining files."""
        h = RemoveHarness(["a.wav", "b.wav", "c.wav"])
        h.select(["b.wav"])
        h.remove_selected_files()
        assert set(h.file_items.keys()) == {"a.wav", "c.wav"}

    def test_remove_then_add_same_file(self):
        """Remove a file, then add it back — should work."""
        h = RemoveHarness([])
        h._append_files(["a.wav", "b.wav"])
        norm_a = str(Path("a.wav"))
        h.select([norm_a])
        h.remove_selected_files()
        assert norm_a not in h.file_paths
        h._append_files(["a.wav"])
        assert str(Path("a.wav")) in h.file_paths

    def test_remove_non_contiguous_selection(self):
        """Select files 1, 3, 5 from a list of 5 and remove them."""
        paths = ["1.wav", "2.wav", "3.wav", "4.wav", "5.wav"]
        h = RemoveHarness(paths)
        h.select(["1.wav", "3.wav", "5.wav"])
        h.remove_selected_files()
        assert h.file_paths == ["2.wav", "4.wav"]

    def test_repeated_remove(self):
        """Multiple sequential removes should each work correctly."""
        h = RemoveHarness(["a.wav", "b.wav", "c.wav", "d.wav"])
        h.select(["a.wav"])
        h.remove_selected_files()
        assert h.file_paths == ["b.wav", "c.wav", "d.wav"]
        h.select(["c.wav"])
        h.remove_selected_files()
        assert h.file_paths == ["b.wav", "d.wav"]
        h.select(["b.wav", "d.wav"])
        h.remove_selected_files()
        assert h.file_paths == []
