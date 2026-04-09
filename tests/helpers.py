"""Shared test utilities: MockApp and helpers used across test modules."""

import queue
import threading


class MockApp:
    """Minimal mock implementing the WorkerHost protocol.

    Used by test_worker.py (unit tests with mocked models) and
    test_e2e.py (integration test with real models).
    """

    def __init__(self, files=None):
        self.event_queue = queue.Queue()
        self.pause_event = threading.Event()
        self.pause_event.set()  # not paused
        self.stop_requested = False
        self.cancel_requested = False
        self.events = []
        self._files = list(files) if files else []

    def post_event(self, kind, **payload):
        self.events.append((kind, payload))

    def next_file(self):
        """Return the next file from the queue, or None if empty."""
        return self._files.pop(0) if self._files else None
