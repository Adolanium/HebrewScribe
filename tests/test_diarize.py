"""Tests for hebrewscribe.diarize — pure assignment logic and IO layer.

The assignment core is tested with faked turns/annotations (deterministic;
no pyannote needed). IO tests patch urllib/subprocess; torch-dependent
waveform tests skip cleanly when torch is not installed.
"""

import hashlib
import importlib.util
import io
import os
import tarfile
import wave as wave_mod
from pathlib import Path

import pytest

from hebrewscribe import diarize
from hebrewscribe.diarize import (
    DiarizationCancelled,
    DiarizationUnavailable,
    annotation_to_turns,
    assign_speakers_to_segments,
    build_speaker_map,
    download_weights,
    is_diarization_available,
    resolve_weights_dir,
    weights_complete,
)
from tests.helpers import MockApp

_HAS_TORCH = importlib.util.find_spec("torch") is not None


def _w(word, start, end):
    return {"start": start, "end": end, "word": word}


def _seg(start, end, text, words=None):
    seg = {"id": 0, "start": start, "end": end, "text": text}
    if words is not None:
        seg["words"] = words
    return seg


TWO_TURNS = [
    {"start": 0.0, "end": 3.0, "speaker": "SPEAKER_00"},
    {"start": 3.0, "end": 6.0, "speaker": "SPEAKER_01"},
]


# ---------------------------------------------------------------------------
# Word-level assignment
# ---------------------------------------------------------------------------

class TestWordLevelAssignment:
    def test_two_speaker_segment_splits(self):
        words = [_w("Hello", 0.2, 0.8), _w(" there", 1.0, 1.6),
                 _w(" friend", 2.0, 2.6), _w(" thanks", 3.4, 4.0),
                 _w(" bye", 4.5, 5.1)]
        out = assign_speakers_to_segments(
            [_seg(0.0, 6.0, "Hello there friend thanks bye", words)], TWO_TURNS)
        assert len(out) == 2
        first, second = out
        assert first["speaker"] == "SPEAKER_00"
        assert second["speaker"] == "SPEAKER_01"
        assert first["text"] == "Hello there friend"
        assert second["text"] == "thanks bye"
        # First sub keeps segment start; last keeps segment end; the
        # boundary is the next run's first word start.
        assert first["start"] == 0.0
        assert first["end"] == 3.4
        assert second["start"] == 3.4
        assert second["end"] == 6.0

    def test_word_midpoint_rule(self):
        # Word 2.8-3.4: midpoint 3.1 lands in the second turn.
        words = [_w("x", 2.8, 3.4)]
        out = assign_speakers_to_segments([_seg(2.8, 3.4, "x", words)], TWO_TURNS)
        assert out[0]["speaker"] == "SPEAKER_01"

    def test_short_single_word_flip_absorbed(self):
        turns = [
            {"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"},
            {"start": 2.0, "end": 2.9, "speaker": "SPEAKER_01"},
            {"start": 2.9, "end": 6.0, "speaker": "SPEAKER_00"},
        ]
        words = [_w("a", 0.2, 0.8), _w(" b", 1.0, 1.6),
                 _w(" c", 2.1, 2.4),  # 0.3 s, midpoint 2.25 -> SPEAKER_01
                 _w(" d", 3.0, 3.6), _w(" e", 4.0, 4.6)]
        out = assign_speakers_to_segments([_seg(0.0, 6.0, "a b c d e", words)],
                                          turns)
        assert len(out) == 1
        assert out[0]["speaker"] == "SPEAKER_00"
        # No split: original whisper text kept verbatim.
        assert out[0]["text"] == "a b c d e"

    def test_long_single_word_flip_splits(self):
        turns = [
            {"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"},
            {"start": 2.0, "end": 2.9, "speaker": "SPEAKER_01"},
            {"start": 2.9, "end": 6.0, "speaker": "SPEAKER_00"},
        ]
        words = [_w("a", 0.2, 0.8), _w(" b", 1.0, 1.6),
                 _w(" c", 2.0, 2.9),  # 0.9 s >= threshold -> real flip
                 _w(" d", 3.0, 3.6), _w(" e", 4.0, 4.6)]
        out = assign_speakers_to_segments([_seg(0.0, 6.0, "a b c d e", words)],
                                          turns)
        assert [s["speaker"] for s in out] == [
            "SPEAKER_00", "SPEAKER_01", "SPEAKER_00"]
        assert out[1]["text"] == "c"

    def test_flip_at_segment_edge_absorbs_forward(self):
        turns = [
            {"start": 0.0, "end": 0.5, "speaker": "SPEAKER_01"},
            {"start": 0.5, "end": 6.0, "speaker": "SPEAKER_00"},
        ]
        words = [_w("a", 0.1, 0.3),  # 0.2 s, no previous run -> absorbs forward
                 _w(" b", 0.6, 1.2), _w(" c", 1.5, 2.0)]
        out = assign_speakers_to_segments([_seg(0.0, 6.0, "a b c", words)],
                                          turns)
        assert len(out) == 1
        assert out[0]["speaker"] == "SPEAKER_00"

    def test_gap_midpoint_nearest_turn(self):
        turns = [
            {"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"},
            {"start": 4.0, "end": 6.0, "speaker": "SPEAKER_01"},
        ]
        words = [_w("a", 2.3, 2.7)]  # midpoint 2.5: 0.5 past S00, 1.5 to S01
        out = assign_speakers_to_segments([_seg(2.3, 2.7, "a", words)], turns)
        assert out[0]["speaker"] == "SPEAKER_00"
        words = [_w("b", 3.7, 4.1)]  # midpoint 3.9: nearer to S01
        out = assign_speakers_to_segments([_seg(3.7, 4.1, "b", words)], turns)
        assert out[0]["speaker"] == "SPEAKER_01"

    def test_output_monotonic_nonoverlapping_renumbered(self):
        words = [_w("a", 0.2, 0.8), _w(" b", 3.4, 4.0), _w(" c", 4.5, 5.1)]
        out = assign_speakers_to_segments([_seg(0.0, 6.0, "a b c", words)],
                                          TWO_TURNS)
        assert [s["id"] for s in out] == list(range(len(out)))
        for i, seg in enumerate(out):
            assert seg["start"] <= seg["end"]
            if i:
                assert out[i - 1]["end"] == seg["start"]


# ---------------------------------------------------------------------------
# Segment-level fallback
# ---------------------------------------------------------------------------

class TestSegmentFallback:
    def test_no_words_max_overlap(self):
        out = assign_speakers_to_segments([_seg(0.0, 2.0, "hi")], TWO_TURNS)
        assert out[0]["speaker"] == "SPEAKER_00"
        out = assign_speakers_to_segments([_seg(3.5, 5.5, "bye")], TWO_TURNS)
        assert out[0]["speaker"] == "SPEAKER_01"

    def test_overlap_tie_earliest_seen(self):
        out = assign_speakers_to_segments([_seg(1.0, 5.0, "even")], TWO_TURNS)
        # 2.0 s overlap with each turn: tie goes to the earliest-seen speaker.
        assert out[0]["speaker"] == "SPEAKER_00"

    def test_broken_word_times_fall_back(self):
        words = [_w("a", 0.2, 0.8), _w(" b", 500.0, 500.4)]  # way outside seg
        out = assign_speakers_to_segments([_seg(0.0, 6.0, "a b", words)],
                                          TWO_TURNS)
        assert len(out) == 1  # no split — fallback path
        assert out[0]["speaker"] == "SPEAKER_00"
        assert out[0]["text"] == "a b"

    def test_nan_word_times_fall_back_not_crash(self):
        # NaN passes isinstance checks and poisons ordered comparisons; the
        # usability gate must reject it so the segment demotes cleanly to
        # segment-level assignment instead of crashing the whole file.
        nan = float("nan")
        words = [_w("a", 0.2, 0.8), _w(" b", nan, nan)]
        out = assign_speakers_to_segments([_seg(0.0, 6.0, "a b", words)],
                                          TWO_TURNS)
        assert len(out) == 1
        assert out[0]["speaker"] == "SPEAKER_00"
        assert out[0]["text"] == "a b"
        # NaN end only
        words = [_w("a", 0.2, 0.8), _w(" b", 1.0, nan)]
        out = assign_speakers_to_segments([_seg(0.0, 6.0, "a b", words)],
                                          TWO_TURNS)
        assert len(out) == 1
        assert out[0]["speaker"] == "SPEAKER_00"

    def test_no_turns_speaker_none_and_empty_map(self):
        out = assign_speakers_to_segments([_seg(0.0, 2.0, "hi")], [])
        assert out[0]["speaker"] is None
        assert build_speaker_map(out, "he") == {}

    def test_zero_overlap_nearest_turn(self):
        turns = [{"start": 10.0, "end": 12.0, "speaker": "SPEAKER_00"}]
        out = assign_speakers_to_segments([_seg(0.0, 2.0, "early")], turns)
        assert out[0]["speaker"] == "SPEAKER_00"

    def test_inputs_not_mutated(self):
        words = [_w("a", 0.2, 0.8), _w(" b", 3.4, 4.0)]
        seg = _seg(0.0, 6.0, "a b", words)
        segments = [seg]
        assign_speakers_to_segments(segments, TWO_TURNS)
        assert seg["text"] == "a b"
        assert "speaker" not in seg
        assert len(seg["words"]) == 2


# ---------------------------------------------------------------------------
# Labels & annotation flattening
# ---------------------------------------------------------------------------

class TestSpeakerMap:
    def test_first_appearance_order_and_hebrew(self):
        segs = [
            {"speaker": "SPEAKER_01"},
            {"speaker": "SPEAKER_00"},
            {"speaker": "SPEAKER_01"},
            {"speaker": None},
        ]
        mapping = build_speaker_map(segs, "he")
        assert list(mapping.items()) == [
            ("SPEAKER_01", "דובר 1"), ("SPEAKER_00", "דובר 2")]

    def test_default_label_for_other_languages(self):
        segs = [{"speaker": "SPEAKER_00"}]
        assert build_speaker_map(segs, "en")["SPEAKER_00"] == "Speaker 1"
        assert build_speaker_map(segs, None)["SPEAKER_00"] == "Speaker 1"


class _FakeTurn:
    def __init__(self, start, end):
        self.start = start
        self.end = end


class _FakeAnnotation:
    def __init__(self, rows):
        self._rows = rows

    def itertracks(self, yield_label=False):
        assert yield_label
        for start, end, label in self._rows:
            yield _FakeTurn(start, end), "_", label


class TestAnnotationToTurns:
    def test_flatten_and_sort(self):
        ann = _FakeAnnotation([
            (3.0, 6.0, "SPEAKER_01"), (0.0, 3.0, "SPEAKER_00")])
        turns = annotation_to_turns(ann)
        assert turns == TWO_TURNS


# ---------------------------------------------------------------------------
# Weights resolution & download
# ---------------------------------------------------------------------------

def _make_weights_dir(root):
    d = root / "diarization" / diarize.COMMUNITY1_DIR_NAME
    for rel in diarize.EXPECTED_FILES:
        f = d / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(b"stub-weights")
    return d


class TestWeightsResolution:
    def test_resolve_source_install(self, tmp_path, monkeypatch):
        monkeypatch.setattr(diarize, "get_models_dir", lambda: tmp_path)
        d = _make_weights_dir(tmp_path)
        assert resolve_weights_dir() == d

    def test_resolve_incomplete_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setattr(diarize, "get_models_dir", lambda: tmp_path)
        d = _make_weights_dir(tmp_path)
        (d / diarize.EXPECTED_FILES[1]).write_bytes(b"")  # empty file
        assert resolve_weights_dir() is None
        assert not weights_complete(d)


def _weights_tarball(pad_bytes=0):
    """An in-memory tar.gz with the community-1 layout."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for rel in diarize.EXPECTED_FILES:
            payload = b"stub-weights" + b"\x00" * pad_bytes
            info = tarfile.TarInfo(
                name=f"{diarize.COMMUNITY1_DIR_NAME}/{rel}")
            info.size = len(payload)
            tf.addfile(info, io.BytesIO(payload))
    return buf.getvalue()


class _FakeResponse:
    def __init__(self, data, on_read=None):
        self._buf = io.BytesIO(data)
        self.headers = {"Content-Length": str(len(data))}
        self._on_read = on_read

    def read(self, n=-1):
        chunk = self._buf.read(n)
        if self._on_read:
            self._on_read()
        return chunk

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class TestDownloadWeights:
    def _patch_urlopen(self, monkeypatch, data, on_read=None):
        monkeypatch.setattr(
            diarize.urllib.request, "urlopen",
            lambda *a, **k: _FakeResponse(data, on_read=on_read))

    def test_download_events_and_result(self, tmp_path, monkeypatch):
        data = _weights_tarball()
        self._patch_urlopen(monkeypatch, data)
        # Match the pin to this test's tarball so verification passes.
        monkeypatch.setattr(diarize, "DIARIZATION_WEIGHTS_SHA256",
                            hashlib.sha256(data).hexdigest())
        host = MockApp()
        dest = tmp_path / "diarization" / diarize.COMMUNITY1_DIR_NAME
        result = download_weights(host, dest)
        assert result == dest
        assert weights_complete(dest)
        kinds = [k for k, _ in host.events]
        assert "download_start" in kinds
        assert "download_progress" in kinds
        assert "download_done" in kinds
        prog = [p for k, p in host.events if k == "download_progress"][-1]
        assert prog["downloaded"] == prog["total"] > 0

    def test_corrupt_archive_raises_and_no_dest(self, tmp_path, monkeypatch):
        data = b"this is not a tarball"
        self._patch_urlopen(monkeypatch, data)
        # Match the pin so the failure exercised is tar extraction, not checksum.
        monkeypatch.setattr(diarize, "DIARIZATION_WEIGHTS_SHA256",
                            hashlib.sha256(data).hexdigest())
        host = MockApp()
        dest = tmp_path / "diarization" / diarize.COMMUNITY1_DIR_NAME
        with pytest.raises(DiarizationUnavailable):
            download_weights(host, dest)
        assert not dest.exists()
        # temp dirs cleaned up (mkdtemp uses dir=dest.parent)
        assert list(dest.parent.glob(".tmp-diarization-*")) == []

    def test_checksum_mismatch_raises(self, tmp_path, monkeypatch):
        self._patch_urlopen(monkeypatch, _weights_tarball())
        monkeypatch.setattr(diarize, "DIARIZATION_WEIGHTS_SHA256", "0" * 64)
        host = MockApp()
        dest = tmp_path / "diarization" / diarize.COMMUNITY1_DIR_NAME
        with pytest.raises(DiarizationUnavailable):
            download_weights(host, dest)
        assert not dest.exists()

    def test_cancel_mid_stream(self, tmp_path, monkeypatch):
        host = MockApp()

        def cancel_after_read():
            host.cancel_requested = True

        # Pad so the payload spans multiple 64 KiB chunks.
        self._patch_urlopen(monkeypatch, _weights_tarball(pad_bytes=200_000),
                            on_read=cancel_after_read)
        dest = tmp_path / "diarization" / diarize.COMMUNITY1_DIR_NAME
        with pytest.raises(DiarizationCancelled):
            download_weights(host, dest)
        assert not dest.exists()
        # temp dirs cleaned up (mkdtemp uses dir=dest.parent)
        assert list(dest.parent.glob(".tmp-diarization-*")) == []

    def test_failed_download_still_posts_download_done(self, tmp_path,
                                                       monkeypatch):
        """The terminal event must fire on failure too, or the GUI's
        status-bar device line stays stuck on download progress."""
        self._patch_urlopen(monkeypatch, b"this is not a tarball")
        host = MockApp()
        dest = tmp_path / "diarization" / diarize.COMMUNITY1_DIR_NAME
        with pytest.raises(DiarizationUnavailable):
            download_weights(host, dest)
        kinds = [k for k, _ in host.events]
        assert "download_done" in kinds

    def test_symlink_member_rejected_in_fallback_extract(self, tmp_path):
        """The manual (pre-filter=) extraction path must reject link members."""
        evil = io.BytesIO()
        with tarfile.open(fileobj=evil, mode="w:gz") as tf:
            info = tarfile.TarInfo(name="community-1/config.yaml")
            info.type = tarfile.SYMTYPE
            info.linkname = "/etc/passwd"
            tf.addfile(info)
        tar_path = tmp_path / "evil.tar.gz"
        tar_path.write_bytes(evil.getvalue())
        dest = tmp_path / "extract"
        dest.mkdir()

        class _NoFilterTar:
            """Force the fallback branch regardless of Python version."""

            def __init__(self, real):
                self._real = real

            def extractall(self, path, filter=None):
                if filter is not None:
                    raise TypeError("filter not supported")
                self._real.extractall(path)

            def __getattr__(self, name):
                return getattr(self._real, name)

        import contextlib

        orig_open = tarfile.open

        @contextlib.contextmanager
        def fake_open(path, mode):
            with orig_open(path, mode) as real:
                yield _NoFilterTar(real)

        tarfile.open = fake_open
        try:
            with pytest.raises(DiarizationUnavailable, match="Link member"):
                diarize._safe_extract_tar(tar_path, dest)
        finally:
            tarfile.open = orig_open


class TestDiarizeHookPause:
    def test_hook_waits_while_paused(self):
        """The pipeline hook must honor Pause between hooked batches — the
        only pause point inside a potentially minutes-long diarize pass."""
        import threading

        class _FakePipeline:
            def __call__(self, waveform, num_speakers=None, hook=None):
                hook("segmentation", None, total=10, completed=5)
                return "ann"

        host = MockApp()
        host.pause_event.clear()  # paused
        threading.Timer(0.3, host.pause_event.set).start()
        out = diarize.diarize_waveform(_FakePipeline(), {"waveform": None},
                                       host, Path("/a/x.wav"))
        assert out == "ann"
        kinds = [k for k, _ in host.events]
        assert "paused" in kinds
        assert "resumed" in kinds

    def test_hook_cancel_while_paused_raises(self):
        from hebrewscribe.worker import CancelledByUser

        class _FakePipeline:
            def __call__(self, waveform, num_speakers=None, hook=None):
                hook("segmentation", None, total=10, completed=5)
                return "ann"

        host = MockApp()
        host.pause_event.clear()

        def _cancel():
            host.cancel_requested = True

        import threading
        threading.Timer(0.3, _cancel).start()
        with pytest.raises(CancelledByUser):
            diarize.diarize_waveform(_FakePipeline(), {"waveform": None},
                                     host, Path("/a/x.wav"))


class TestAvailability:
    def test_env_set_before_probe(self, monkeypatch):
        monkeypatch.delenv("PYANNOTE_METRICS_ENABLED", raising=False)
        seen = {}

        def fake_probe(name):
            seen["env"] = os.environ.get("PYANNOTE_METRICS_ENABLED")
            return False

        monkeypatch.setattr(diarize, "is_package_available", fake_probe)
        assert is_diarization_available() is False
        assert seen["env"] == "false"


# ---------------------------------------------------------------------------
# Waveform loading (needs torch)
# ---------------------------------------------------------------------------

def _write_wav(path, rate=16000, channels=1, seconds=0.25):
    import struct
    n = int(rate * seconds)
    with wave_mod.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        frames = b"".join(
            struct.pack("<" + "h" * channels, *([1000] * channels))
            for _ in range(n))
        wf.writeframes(frames)


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
class TestLoadWaveform:
    def test_16k_mono_pcm_fast_path(self, tmp_path):
        wav = tmp_path / "a.wav"
        _write_wav(wav)
        out = diarize.load_waveform(wav)
        assert out["sample_rate"] == 16000
        assert tuple(out["waveform"].shape)[0] == 1
        assert out["waveform"].shape[1] == 4000
        assert str(out["waveform"].dtype) == "torch.float32"
        assert float(out["waveform"].abs().max()) <= 1.0

    def test_16k_stereo_downmixed(self, tmp_path):
        wav = tmp_path / "st.wav"
        _write_wav(wav, channels=2)
        out = diarize.load_waveform(wav)
        assert tuple(out["waveform"].shape)[0] == 1
        assert out["waveform"].shape[1] == 4000

    def test_non_16k_uses_ffmpeg(self, tmp_path, monkeypatch):
        wav = tmp_path / "hi.wav"
        _write_wav(wav, rate=44100)
        calls = {}

        def fake_run(cmd, **kwargs):
            calls["cmd"] = cmd
            _write_wav(cmd[-1])  # ffmpeg "writes" a conforming file

            class R:
                returncode = 0
            return R()

        monkeypatch.setattr(diarize.subprocess, "run", fake_run)
        out = diarize.load_waveform(wav)
        assert out["sample_rate"] == 16000
        assert "-ar" in calls["cmd"] and "16000" in calls["cmd"]

    def test_unreadable_raises_unavailable(self, tmp_path, monkeypatch):
        bad = tmp_path / "bad.wav"
        bad.write_bytes(b"not audio")

        def fail_run(cmd, **kwargs):
            class R:
                returncode = 1
            return R()

        monkeypatch.setattr(diarize.subprocess, "run", fail_run)
        with pytest.raises(DiarizationUnavailable):
            diarize.load_waveform(bad)
