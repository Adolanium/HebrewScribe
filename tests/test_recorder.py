"""Regression tests for the live recorder's shutdown and lifecycle invariants.

- Stop must not drop speech segments already queued or being flushed.
- The VAD thread must close the microphone stream on every exit path.
- stop(wait=False) (window close) must never block on thread joins.
"""

import threading

import numpy as np


class _EventSink:
    def __init__(self):
        self.events = []

    def __call__(self, kind, **payload):
        self.events.append((kind, payload))

    def kinds(self):
        return [k for k, _ in self.events]


class _FakeSeg:
    def __init__(self, text):
        self.text = text


class _FakeModel:
    def transcribe(self, audio, **kwargs):
        return iter([_FakeSeg("hello")]), None


class _FakeStream:
    def __init__(self):
        self.stopped = False
        self.closed = False

    def stop(self):
        self.stopped = True

    def close(self):
        self.closed = True


def _make_recorder(sink):
    from hebrewscribe.recorder import LiveRecorder
    return LiveRecorder(event_callback=sink, language="en",
                        model_spec="dummy", device="cpu", compute_type="int8")


# ---------------------------------------------------------------------------
# Stop must not drop queued/flushed segments
# ---------------------------------------------------------------------------

def test_transcriber_drains_queue_after_stop():
    """Segments already in the queue when Stop hits must still transcribe:
    the loop exits on the sentinel, never on stop_flag."""
    sink = _EventSink()
    rec = _make_recorder(sink)
    rec._load_transcription_model = lambda: setattr(rec, "_model", _FakeModel())

    rec._transcribe_queue.put(np.zeros(1600, dtype=np.float32))
    rec._transcribe_queue.put(np.zeros(1600, dtype=np.float32))
    rec._transcribe_queue.put(None)  # sentinel (normally from the VAD thread)
    rec._stop_flag.set()             # Stop was already clicked
    rec._recording = False

    rec._transcribe_thread_func()

    texts = [p for k, p in sink.events if k == "rec_text"]
    assert len(texts) == 2, f"queued segments dropped: {sink.events}"


def test_stop_sends_no_transcribe_sentinel():
    """stop() must only wake the VAD thread (audio sentinel); the transcribe
    sentinel comes from the VAD thread's finally AFTER it flushed."""
    sink = _EventSink()
    rec = _make_recorder(sink)
    rec._recording = True

    rec.stop()

    assert rec._audio_queue.get_nowait() is None
    assert rec._transcribe_queue.empty(), \
        "stop() queued a transcribe sentinel ahead of the VAD flush"
    assert "rec_stopped" in sink.kinds()


def test_stop_wait_false_skips_join():
    """stop(wait=False) (window close) must not join threads."""
    sink = _EventSink()
    rec = _make_recorder(sink)
    rec._recording = True

    never_joined = threading.Thread(target=lambda: threading.Event().wait(30),
                                    daemon=True)
    never_joined.start()
    rec._vad_thread = never_joined

    import time
    t0 = time.monotonic()
    rec.stop(wait=False)
    assert time.monotonic() - t0 < 1.0, "stop(wait=False) blocked on a join"
    assert "rec_stopped" in sink.kinds()


# ---------------------------------------------------------------------------
# The VAD thread must close the mic on every exit path
# ---------------------------------------------------------------------------

def test_vad_thread_closes_stream_on_exit():
    sink = _EventSink()
    rec = _make_recorder(sink)
    stream = _FakeStream()
    rec._load_vad = lambda: setattr(rec, "_vad_iterator", object())
    rec._start_audio_stream = lambda: setattr(rec, "_stream", stream)

    rec._audio_queue.put(None)  # break the loop immediately
    rec._vad_thread_func()

    assert stream.closed, "mic stream left open after VAD thread exit"
    assert rec._transcribe_queue.get_nowait() is None  # sentinel forwarded


def test_vad_thread_skips_mic_when_stop_already_set():
    """A fast model-load failure sets stop_flag before the mic opens —
    the VAD thread must then never start the stream."""
    sink = _EventSink()
    rec = _make_recorder(sink)
    opened = []
    rec._load_vad = lambda: None
    rec._start_audio_stream = lambda: opened.append(True)

    rec._stop_flag.set()
    rec._vad_thread_func()

    assert not opened, "microphone opened even though stop_flag was set"
    assert rec._transcribe_queue.get_nowait() is None
