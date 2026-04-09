"""Unit tests for queue reorder feature in TranscriberApp.

Tests exercise _move_selected() logic using a lightweight mock that
simulates file_paths, file_items, and Treeview status without requiring
a real Tk instance.
"""

import pytest


# ---------------------------------------------------------------------------
# Mock infrastructure
# ---------------------------------------------------------------------------

class MockTreeview:
    """Minimal Treeview mock that stores items as dicts and supports
    selection, item(), exists(), get_children(), and selection_set()."""

    def __init__(self):
        self._items = {}      # item_id -> {"values": tuple, "tags": tuple}
        self._children = []   # ordered list of item_ids
        self._selection = []  # currently selected item_ids
        self._counter = 0

    def insert(self, parent, position, values=(), tags=()):
        self._counter += 1
        item_id = f"I{self._counter:03d}"
        self._items[item_id] = {"values": tuple(values), "tags": tuple(tags)}
        self._children.append(item_id)
        return item_id

    def item(self, item_id, key=None, **kwargs):
        if key is not None:
            return self._items[item_id][key]
        if kwargs:
            for k, v in kwargs.items():
                self._items[item_id][k] = v
        return self._items[item_id]

    def exists(self, item_id):
        return item_id in self._items

    def selection(self):
        return tuple(self._selection)

    def selection_set(self, *items):
        self._selection = list(items)

    def get_children(self, parent=""):
        return tuple(self._children)

    def delete(self, item_id):
        self._items.pop(item_id, None)
        if item_id in self._children:
            self._children.remove(item_id)

    def see(self, item_id):
        pass  # no-op in test

    def tag_configure(self, *args, **kwargs):
        pass

    def bind(self, *args, **kwargs):
        pass

    def heading(self, *args, **kwargs):
        pass

    def column(self, *args, **kwargs):
        pass

    def configure(self, **kwargs):
        pass

    def config(self, **kwargs):
        pass

    def grid(self, **kwargs):
        pass

    def bbox(self, *args, **kwargs):
        return None


class ReorderHarness:
    """Mimics the subset of TranscriberApp needed by _move_selected().

    Attributes:
        file_paths: The queue order.
        file_items: Maps path -> item_id.
        statuses: Maps path -> status string (e.g. "Queued", "Running").
    """

    def __init__(self, paths, statuses=None):
        """
        paths: list of path strings
        statuses: dict path -> status (default all "Queued")
        """
        self.file_paths = list(paths)
        self._queue_lock = __import__("threading").Lock()
        self.file_items = {}
        self.file_tree = MockTreeview()
        self.worker_thread = None  # no batch running
        self.file_durations = {}
        self.file_errors = {}
        self._tree_progress = {}
        self._btn_move_top = _FakeBtn()
        self._btn_move_up = _FakeBtn()
        self._btn_move_down = _FakeBtn()
        self._btn_move_bottom = _FakeBtn()

        if statuses is None:
            statuses = {p: "Queued" for p in paths}
        self._statuses = statuses

        self._build_tree()

    def _build_tree(self):
        """Populate the mock Treeview from file_paths and statuses."""
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

    def select(self, paths):
        """Set treeview selection to the given paths."""
        items = [self.file_items[p] for p in paths]
        self.file_tree.selection_set(*items)

    # --- Methods called by _move_selected ---

    def refresh_file_tree(self):
        """Rebuild the tree from current file_paths (simplified)."""
        self._build_tree()

    def update_overall_summary(self, **kw):
        pass

    def _flash_rows(self, paths, **kw):
        pass

    def _update_move_button_state(self):
        pass

    def _set_button_enabled(self, btn, enabled):
        btn.enabled = enabled

    def _redraw_tree_progress(self):
        pass

    def _update_empty_state(self):
        pass

    # --- Imported from TranscriberApp ---

    def _is_movable_index(self, idx):
        path_str = self.file_paths[idx]
        item_id = self.file_items.get(path_str)
        if not item_id or not self.file_tree.exists(item_id):
            return False
        return self.file_tree.item(item_id, "values")[1] == "Queued"

    def _move_selected(self, direction):
        # Import the actual method logic — we bind it from the real class
        from hebrewscribe.app import TranscriberApp
        TranscriberApp._move_selected(self, direction)


class _FakeBtn:
    def __init__(self):
        self.enabled = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def paths_order(harness):
    """Return the current file_paths (just filenames for readability)."""
    return list(harness.file_paths)


def ordinals(harness):
    """Return the ordinal values from the treeview for each row."""
    result = []
    for path in harness.file_paths:
        item_id = harness.file_items.get(path)
        if item_id:
            result.append(int(harness.file_tree.item(item_id, "values")[0]))
    return result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMoveUp:
    def test_move_single_up(self):
        h = ReorderHarness(["a", "b", "c"])
        h.select(["b"])
        h._move_selected("up")
        assert paths_order(h) == ["b", "a", "c"]

    def test_move_up_at_boundary(self):
        h = ReorderHarness(["a", "b", "c"])
        h.select(["a"])
        h._move_selected("up")
        assert paths_order(h) == ["a", "b", "c"]

    def test_multi_select_block_up(self):
        h = ReorderHarness(["a", "b", "c", "d"])
        h.select(["c", "d"])
        h._move_selected("up")
        assert paths_order(h) == ["a", "c", "d", "b"]

    def test_blocked_by_immovable(self):
        h = ReorderHarness(["a", "b", "c"], statuses={"a": "Running", "b": "Queued", "c": "Queued"})
        h.select(["b"])
        h._move_selected("up")
        # b can't move past Running row a
        assert paths_order(h) == ["a", "b", "c"]


class TestMoveDown:
    def test_move_single_down(self):
        h = ReorderHarness(["a", "b", "c"])
        h.select(["b"])
        h._move_selected("down")
        assert paths_order(h) == ["a", "c", "b"]

    def test_move_down_at_boundary(self):
        h = ReorderHarness(["a", "b", "c"])
        h.select(["c"])
        h._move_selected("down")
        assert paths_order(h) == ["a", "b", "c"]

    def test_multi_select_block_down(self):
        h = ReorderHarness(["a", "b", "c", "d"])
        h.select(["a", "b"])
        h._move_selected("down")
        assert paths_order(h) == ["c", "a", "b", "d"]


class TestMoveToTop:
    def test_move_to_top(self):
        h = ReorderHarness(["a", "b", "c"])
        h.select(["c"])
        h._move_selected("top")
        assert paths_order(h) == ["c", "a", "b"]

    def test_move_to_top_with_immovable_above(self):
        h = ReorderHarness(["a", "b", "c", "d"],
                           statuses={"a": "Done", "b": "Queued", "c": "Queued", "d": "Queued"})
        h.select(["d"])
        h._move_selected("top")
        # d should land just after the immovable Done row
        assert paths_order(h) == ["a", "d", "b", "c"]


class TestMoveToBottom:
    def test_move_to_bottom(self):
        h = ReorderHarness(["a", "b", "c"])
        h.select(["a"])
        h._move_selected("bottom")
        assert paths_order(h) == ["b", "c", "a"]

    def test_move_to_bottom_with_immovable_below(self):
        h = ReorderHarness(["a", "b", "c", "d"],
                           statuses={"a": "Queued", "b": "Queued", "c": "Queued", "d": "Failed"})
        h.select(["a"])
        h._move_selected("bottom")
        # a should land just before the immovable Failed row
        assert paths_order(h) == ["b", "c", "a", "d"]


class TestEdgeCases:
    def test_mixed_selection(self):
        """Only Queued rows move; Done rows in the selection are ignored."""
        h = ReorderHarness(["a", "b", "c"],
                           statuses={"a": "Done", "b": "Queued", "c": "Queued"})
        h.select(["a", "c"])
        h._move_selected("up")
        # a is Done (ignored), c moves up past b
        assert paths_order(h) == ["a", "c", "b"]

    def test_all_immovable(self):
        h = ReorderHarness(["a", "b", "c"],
                           statuses={"a": "Running", "b": "Done", "c": "Failed"})
        h.select(["a", "b", "c"])
        h._move_selected("up")
        assert paths_order(h) == ["a", "b", "c"]

    def test_empty_selection(self):
        h = ReorderHarness(["a", "b", "c"])
        # No selection
        h._move_selected("up")
        assert paths_order(h) == ["a", "b", "c"]

    def test_noncontiguous_multi_select_up(self):
        h = ReorderHarness(["a", "b", "c", "d", "e"])
        h.select(["b", "d"])
        h._move_selected("up")
        # b moves to index 0, d moves to index 2
        assert paths_order(h) == ["b", "a", "d", "c", "e"]

    def test_ordinal_values_after_move(self):
        h = ReorderHarness(["a", "b", "c"])
        h.select(["c"])
        h._move_selected("top")
        assert paths_order(h) == ["c", "a", "b"]
        assert ordinals(h) == [1, 2, 3]

    def test_single_file_queue(self):
        """Moving the only file in the queue is a no-op for all directions."""
        for direction in ("up", "down", "top", "bottom"):
            h = ReorderHarness(["a"])
            h.select(["a"])
            h._move_selected(direction)
            assert paths_order(h) == ["a"], f"Failed for direction={direction}"

    def test_all_selected_move_up_noop(self):
        """Selecting every Queued row and moving up — no room, no-op."""
        h = ReorderHarness(["a", "b", "c"])
        h.select(["a", "b", "c"])
        h._move_selected("up")
        assert paths_order(h) == ["a", "b", "c"]

    def test_all_selected_move_down_noop(self):
        """Selecting every Queued row and moving down — no room, no-op."""
        h = ReorderHarness(["a", "b", "c"])
        h.select(["a", "b", "c"])
        h._move_selected("down")
        assert paths_order(h) == ["a", "b", "c"]

    def test_noncontiguous_multi_select_down(self):
        """Non-contiguous selection moving down — each moves independently."""
        h = ReorderHarness(["a", "b", "c", "d", "e"])
        h.select(["b", "d"])
        h._move_selected("down")
        # b→index 2, d→index 4
        assert paths_order(h) == ["a", "c", "b", "e", "d"]

    def test_immovable_wall_between_selected(self):
        """Immovable row between two selected Queued rows partitions movement."""
        h = ReorderHarness(["a", "b", "c", "d", "e"],
                           statuses={"a": "Queued", "b": "Queued", "c": "Running",
                                     "d": "Queued", "e": "Queued"})
        h.select(["b", "d"])
        h._move_selected("up")
        # b moves up past a; d can't move past Running c
        assert paths_order(h) == ["b", "a", "c", "d", "e"]

    def test_blocked_by_immovable_down(self):
        """Queued row can't move down past an immovable row."""
        h = ReorderHarness(["a", "b", "c"],
                           statuses={"a": "Queued", "b": "Done", "c": "Queued"})
        h.select(["a"])
        h._move_selected("down")
        # a can't move past Done row b
        assert paths_order(h) == ["a", "b", "c"]

    def test_already_at_top_move_top_noop(self):
        """Moving to top when already at topmost valid position is a no-op."""
        h = ReorderHarness(["a", "b", "c"])
        h.select(["a"])
        h._move_selected("top")
        assert paths_order(h) == ["a", "b", "c"]

    def test_already_at_bottom_move_bottom_noop(self):
        """Moving to bottom when already at bottommost valid position."""
        h = ReorderHarness(["a", "b", "c"])
        h.select(["c"])
        h._move_selected("bottom")
        assert paths_order(h) == ["a", "b", "c"]

    def test_multi_select_move_to_top(self):
        """Multi-select move to top preserves internal order."""
        h = ReorderHarness(["a", "b", "c", "d", "e"])
        h.select(["d", "e"])
        h._move_selected("top")
        assert paths_order(h) == ["d", "e", "a", "b", "c"]

    def test_multi_select_move_to_bottom(self):
        """Multi-select move to bottom preserves internal order."""
        h = ReorderHarness(["a", "b", "c", "d", "e"])
        h.select(["a", "b"])
        h._move_selected("bottom")
        assert paths_order(h) == ["c", "d", "e", "a", "b"]

    def test_repeated_move_up(self):
        """Two consecutive move-ups accumulate correctly."""
        h = ReorderHarness(["a", "b", "c", "d"])
        h.select(["d"])
        h._move_selected("up")
        assert paths_order(h) == ["a", "b", "d", "c"]
        # After refresh, re-select d and move again
        h.select(["d"])
        h._move_selected("up")
        assert paths_order(h) == ["a", "d", "b", "c"]

    def test_multiple_immovable_walls_at_top(self):
        """Move to top with multiple immovable rows at the start."""
        h = ReorderHarness(["a", "b", "c", "d", "e"],
                           statuses={"a": "Running", "b": "Done", "c": "Queued",
                                     "d": "Queued", "e": "Queued"})
        h.select(["e"])
        h._move_selected("top")
        # e lands right after the two immovable rows
        assert paths_order(h) == ["a", "b", "e", "c", "d"]

    def test_move_to_bottom_with_multiple_immovable_at_end(self):
        """Move to bottom with multiple immovable rows at the end."""
        h = ReorderHarness(["a", "b", "c", "d", "e"],
                           statuses={"a": "Queued", "b": "Queued", "c": "Queued",
                                     "d": "Done", "e": "Failed"})
        h.select(["a"])
        h._move_selected("bottom")
        # a lands just before the immovable tail
        assert paths_order(h) == ["b", "c", "a", "d", "e"]
