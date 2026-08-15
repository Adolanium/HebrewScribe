"""Tests for the transcription worker control flow using mocked models."""

import tempfile
import threading
import time
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from tests.helpers import MockApp

from hebrewscribe.worker import (
    RunOptions, run_transcription_worker,
    _validate_opts, _emit_progress, _normalize_segment,
    _completion_progress, _auto_batch_size, _get_batched_pipeline,
    _batched_pipeline_cache, preconvert_audio,
    _is_ct2_corruption_error, _find_hf_cache_dir,
    _model_bin_status, _download_with_progress,
)


WORKER_MODULE = "hebrewscribe.worker"

_FAKE_RESULT = {
    "text": "transcribed text",
    "segments": [{"start": 0.0, "end": 1.0, "text": "transcribed text"}],
    "language": "he",
}


def _make_opts(tmpdir, backend="faster-whisper", formats=None):
    return RunOptions(
        backend=backend,
        model_spec="tiny",
        output_dir=tmpdir,
        language="he",
        task="transcribe",
        device="cpu",
        compute_type="int8",
        beam_size=1,
        vad_filter=True,
        condition_on_previous_text=False,
        batch_size=0,
        formats=formats or ["txt"],
    )


def _make_fake_audio(tmpdir, name="test.wav"):
    """Create a fake audio file (just needs to exist, not be valid audio)."""
    p = Path(tmpdir) / name
    p.write_bytes(b"RIFF" + b"\x00" * 100)
    return str(p)


def _patch_worker(**overrides):
    """Return a stack of patches for worker module functions.

    Defaults: load_model returns a fake model, transcribe_faster returns
    _FAKE_RESULT, preconvert_audio returns the input path.
    """
    defaults = {
        "load_model": MagicMock(return_value=(MagicMock(name="fake_model"), "cpu")),
        "transcribe_faster": MagicMock(return_value=_FAKE_RESULT.copy()),
        "transcribe_openai": MagicMock(return_value=_FAKE_RESULT.copy()),
        "preconvert_audio": MagicMock(side_effect=lambda p: p),
        "probe_duration_seconds": MagicMock(return_value=10.0),
        "validate_audio_file": MagicMock(return_value=None),
    }
    defaults.update(overrides)
    patches = {}
    for name, mock in defaults.items():
        patches[name] = patch(f"{WORKER_MODULE}.{name}", mock)
    return patches, defaults


class TestWorkerHappyPath(unittest.TestCase):
    def test_single_file_success(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            opts = _make_opts(tmpdir)
            audio = _make_fake_audio(tmpdir)
            app = MockApp(files=[audio])
            patches, mocks = _patch_worker()
            with patches["load_model"], patches["transcribe_faster"], \
                 patches["transcribe_openai"], patches["preconvert_audio"], \
                 patches["probe_duration_seconds"], \
                 patches["validate_audio_file"]:
                run_transcription_worker(app, opts, 1)
            kinds = [e[0] for e in app.events]
            self.assertIn("file_started", kinds)
            self.assertIn("file_finished", kinds)
            self.assertIn("done", kinds)
            done_event = [e for e in app.events if e[0] == "done"][0]
            self.assertIn("Finished", done_event[1]["message"])
            self.assertEqual(done_event[1]["done_count"], 1)
            self.assertEqual(done_event[1]["failed_count"], 0)

    def test_multiple_files_success(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            opts = _make_opts(tmpdir)
            files = [_make_fake_audio(tmpdir, f"test{i}.wav") for i in range(3)]
            app = MockApp(files=files)
            patches, mocks = _patch_worker()
            with patches["load_model"], patches["transcribe_faster"], \
                 patches["transcribe_openai"], patches["preconvert_audio"], \
                 patches["probe_duration_seconds"], \
                 patches["validate_audio_file"]:
                run_transcription_worker(app, opts, len(files))
            done_event = [e for e in app.events if e[0] == "done"][0]
            self.assertEqual(done_event[1]["done_count"], 3)
            self.assertEqual(done_event[1]["failed_count"], 0)

    def test_output_files_created(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            opts = _make_opts(tmpdir, formats=["txt", "srt"])
            audio = _make_fake_audio(tmpdir, "hello.wav")
            app = MockApp(files=[audio])
            patches, mocks = _patch_worker()
            with patches["load_model"], patches["transcribe_faster"], \
                 patches["transcribe_openai"], patches["preconvert_audio"], \
                 patches["probe_duration_seconds"], \
                 patches["validate_audio_file"]:
                run_transcription_worker(app, opts, 1)
            self.assertTrue((Path(tmpdir) / "hello.txt").exists())
            self.assertTrue((Path(tmpdir) / "hello.srt").exists())


class TestWorkerStopAfterCurrent(unittest.TestCase):
    def test_stop_after_first_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            call_count = [0]

            def transcribe_then_stop(host, model, file_path, opts, display_path=None):
                call_count[0] += 1
                if call_count[0] >= 1:
                    app.stop_requested = True
                return _FAKE_RESULT.copy()

            patches, mocks = _patch_worker(
                transcribe_faster=MagicMock(side_effect=transcribe_then_stop),
            )
            opts = _make_opts(tmpdir)
            files = [_make_fake_audio(tmpdir, f"f{i}.wav") for i in range(3)]
            app = MockApp(files=files)
            with patches["load_model"], patches["transcribe_faster"], \
                 patches["transcribe_openai"], patches["preconvert_audio"], \
                 patches["probe_duration_seconds"], \
                 patches["validate_audio_file"]:
                run_transcription_worker(app, opts, len(files))
            done_event = [e for e in app.events if e[0] == "done"][0]
            self.assertTrue(done_event[1].get("stopped"))
            self.assertEqual(done_event[1]["done_count"], 1)


class TestWorkerCancel(unittest.TestCase):
    def test_cancel_before_any_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            opts = _make_opts(tmpdir)
            audio = _make_fake_audio(tmpdir)
            app = MockApp(files=[audio])
            app.cancel_requested = True
            patches, mocks = _patch_worker()
            with patches["load_model"], patches["transcribe_faster"], \
                 patches["transcribe_openai"], patches["preconvert_audio"], \
                 patches["probe_duration_seconds"], \
                 patches["validate_audio_file"]:
                run_transcription_worker(app, opts, 1)
            done_event = [e for e in app.events if e[0] == "done"][0]
            self.assertTrue(done_event[1].get("cancelled"))
            self.assertEqual(done_event[1]["done_count"], 0)

    def test_cancel_during_transcription(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            def transcribe_and_cancel(host, model, file_path, opts, display_path=None):
                app.cancel_requested = True
                return _FAKE_RESULT.copy()

            patches, mocks = _patch_worker(
                transcribe_faster=MagicMock(side_effect=transcribe_and_cancel),
            )
            opts = _make_opts(tmpdir)
            files = [_make_fake_audio(tmpdir, f"f{i}.wav") for i in range(3)]
            app = MockApp(files=files)
            with patches["load_model"], patches["transcribe_faster"], \
                 patches["transcribe_openai"], patches["preconvert_audio"], \
                 patches["probe_duration_seconds"], \
                 patches["validate_audio_file"]:
                run_transcription_worker(app, opts, len(files))
            done_event = [e for e in app.events if e[0] == "done"][0]
            self.assertTrue(done_event[1].get("cancelled"))


class TestWorkerFileError(unittest.TestCase):
    def test_single_file_failure_continues_batch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            call_count = [0]

            def fail_first(host, model, file_path, opts, display_path=None):
                call_count[0] += 1
                if call_count[0] == 1:
                    raise RuntimeError("Simulated transcription error")
                return _FAKE_RESULT.copy()

            patches, mocks = _patch_worker(
                transcribe_faster=MagicMock(side_effect=fail_first),
            )
            opts = _make_opts(tmpdir)
            files = [_make_fake_audio(tmpdir, f"f{i}.wav") for i in range(3)]
            app = MockApp(files=files)
            with patches["load_model"], patches["transcribe_faster"], \
                 patches["transcribe_openai"], patches["preconvert_audio"], \
                 patches["probe_duration_seconds"], \
                 patches["validate_audio_file"]:
                run_transcription_worker(app, opts, len(files))
            done_event = [e for e in app.events if e[0] == "done"][0]
            self.assertEqual(done_event[1]["done_count"], 2)
            self.assertEqual(done_event[1]["failed_count"], 1)
            failed_events = [e for e in app.events if e[0] == "file_failed"]
            self.assertEqual(len(failed_events), 1)
            self.assertIn("error_message", failed_events[0][1])

    def test_all_files_fail(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            def always_fail(host, model, file_path, opts, display_path=None):
                raise RuntimeError("always fails")

            patches, mocks = _patch_worker(
                transcribe_faster=MagicMock(side_effect=always_fail),
            )
            opts = _make_opts(tmpdir)
            files = [_make_fake_audio(tmpdir, f"f{i}.wav") for i in range(2)]
            app = MockApp(files=files)
            with patches["load_model"], patches["transcribe_faster"], \
                 patches["transcribe_openai"], patches["preconvert_audio"], \
                 patches["probe_duration_seconds"], \
                 patches["validate_audio_file"]:
                run_transcription_worker(app, opts, len(files))
            done_event = [e for e in app.events if e[0] == "done"][0]
            self.assertEqual(done_event[1]["done_count"], 0)
            self.assertEqual(done_event[1]["failed_count"], 2)
            self.assertTrue(done_event[1].get("show_warning"))

    def test_model_load_failure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            patches, mocks = _patch_worker(
                load_model=MagicMock(side_effect=RuntimeError("Model not found")),
            )
            opts = _make_opts(tmpdir)
            audio = _make_fake_audio(tmpdir)
            app = MockApp(files=[audio])
            with patches["load_model"], patches["transcribe_faster"], \
                 patches["transcribe_openai"], patches["preconvert_audio"], \
                 patches["probe_duration_seconds"], \
                 patches["validate_audio_file"]:
                run_transcription_worker(app, opts, 1)
            failed_events = [e for e in app.events if e[0] == "failed"]
            self.assertEqual(len(failed_events), 1)
            self.assertIn("Model not found", failed_events[0][1]["message"])


class TestWorkerOpenaiBackend(unittest.TestCase):
    def test_uses_openai_transcribe(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            opts = _make_opts(tmpdir, backend="openai-whisper")
            audio = _make_fake_audio(tmpdir)
            app = MockApp(files=[audio])
            patches, mocks = _patch_worker()
            with patches["load_model"], patches["transcribe_faster"], \
                 patches["transcribe_openai"], patches["preconvert_audio"], \
                 patches["probe_duration_seconds"], \
                 patches["validate_audio_file"]:
                run_transcription_worker(app, opts, 1)
            done_event = [e for e in app.events if e[0] == "done"][0]
            self.assertEqual(done_event[1]["done_count"], 1)


class TestWorkerEventSequence(unittest.TestCase):
    def test_event_order(self):
        """Verify that events arrive in the expected order for a single file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            opts = _make_opts(tmpdir)
            audio = _make_fake_audio(tmpdir)
            app = MockApp(files=[audio])
            patches, mocks = _patch_worker()
            with patches["load_model"], patches["transcribe_faster"], \
                 patches["transcribe_openai"], patches["preconvert_audio"], \
                 patches["probe_duration_seconds"], \
                 patches["validate_audio_file"]:
                run_transcription_worker(app, opts, 1)
            kinds = [e[0] for e in app.events]
            job_status_idx = kinds.index("job_status")
            file_started_idx = kinds.index("file_started")
            file_finished_idx = kinds.index("file_finished")
            done_idx = kinds.index("done")
            self.assertLess(job_status_idx, file_started_idx)
            self.assertLess(file_started_idx, file_finished_idx)
            self.assertLess(file_finished_idx, done_idx)


# ---------------------------------------------------------------------------
# _validate_opts
# ---------------------------------------------------------------------------

class TestValidateOpts(unittest.TestCase):
    """Tests for _validate_opts — early RunOptions validation."""

    def _good_opts(self, tmpdir, **overrides):
        base = dict(
            backend="faster-whisper", model_spec="tiny",
            output_dir=tmpdir, language="he", task="transcribe",
            device="cpu", compute_type="int8", beam_size=1,
            vad_filter=True, condition_on_previous_text=False,
            batch_size=0, formats=["txt"],
        )
        base.update(overrides)
        return RunOptions(**base)

    def test_valid_opts_passes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _validate_opts(self._good_opts(tmpdir))

    def test_invalid_backend(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError) as ctx:
                _validate_opts(self._good_opts(tmpdir, backend="unknown"))
            self.assertIn("Unknown backend", str(ctx.exception))

    def test_empty_model_spec(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError) as ctx:
                _validate_opts(self._good_opts(tmpdir, model_spec=""))
            self.assertIn("model_spec", str(ctx.exception))

    def test_beam_size_zero(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError) as ctx:
                _validate_opts(self._good_opts(tmpdir, beam_size=0))
            self.assertIn("beam_size", str(ctx.exception))

    def test_empty_formats(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError) as ctx:
                _validate_opts(self._good_opts(tmpdir, formats=[]))
            self.assertIn("formats", str(ctx.exception))

    def test_nonexistent_output_dir(self):
        with self.assertRaises(ValueError) as ctx:
            _validate_opts(self._good_opts("/no/such/path/here"))
        self.assertIn("output_dir", str(ctx.exception))

    def test_openai_backend_valid(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _validate_opts(self._good_opts(tmpdir, backend="openai-whisper"))

    def test_validation_runs_in_worker(self):
        """run_transcription_worker rejects bad opts before loading model."""
        app = MockApp(files=["/fake.wav"])
        opts = RunOptions(
            backend="bogus", model_spec="tiny", output_dir="/tmp",
            language="he", task="transcribe", device="cpu",
            compute_type="int8", beam_size=1, vad_filter=True,
            condition_on_previous_text=False, batch_size=0,
            formats=["txt"],
        )
        patches, _ = _patch_worker()
        with patches["load_model"], patches["transcribe_faster"], \
             patches["transcribe_openai"], patches["preconvert_audio"], \
             patches["probe_duration_seconds"]:
            run_transcription_worker(app, opts, 1)
        kinds = [e[0] for e in app.events]
        self.assertIn("failed", kinds)
        failed = [e for e in app.events if e[0] == "failed"][0]
        self.assertIn("Unknown backend", failed[1]["message"])


# ---------------------------------------------------------------------------
# Shared transcription helpers
# ---------------------------------------------------------------------------

class TestEmitProgress(unittest.TestCase):
    """Tests for _emit_progress helper."""

    def test_emits_current_progress_event(self):
        app = MockApp()
        p = Path("/fake/audio.wav")
        _emit_progress(app, p, "Transcribing", 42.5, 10.0, 5.0, 2.0, 20.0, 50.0)
        self.assertEqual(len(app.events), 1)
        kind, payload = app.events[0]
        self.assertEqual(kind, "current_progress")
        self.assertEqual(payload["path"], str(p))
        self.assertEqual(payload["phase"], "Transcribing")
        self.assertAlmostEqual(payload["percent"], 42.5)
        self.assertEqual(payload["row_status"], "Running")

    def test_none_eta_and_speed(self):
        app = MockApp()
        _emit_progress(app, Path("/f.wav"), "Init", 0, 0, None, None, 0, 100)
        payload = app.events[0][1]
        self.assertIsNone(payload["eta"])
        self.assertIsNone(payload["speed"])


class TestNormalizeSegment(unittest.TestCase):
    """Tests for _normalize_segment from both openai dicts and faster-whisper objects."""

    def test_from_dict(self):
        seg = {"id": 1, "start": 1.5, "end": 3.2, "text": "  hello  "}
        normed = _normalize_segment(seg, from_dict=True)
        self.assertEqual(normed["id"], 1)
        self.assertAlmostEqual(normed["start"], 1.5)
        self.assertAlmostEqual(normed["end"], 3.2)
        self.assertEqual(normed["text"], "hello")

    def test_from_dict_missing_fields(self):
        seg = {}
        normed = _normalize_segment(seg, from_dict=True)
        self.assertIsNone(normed["id"])
        self.assertAlmostEqual(normed["start"], 0.0)
        self.assertAlmostEqual(normed["end"], 0.0)
        self.assertEqual(normed["text"], "")

    def test_from_dict_none_text(self):
        seg = {"id": 0, "start": 0, "end": 1, "text": None}
        normed = _normalize_segment(seg, from_dict=True)
        self.assertEqual(normed["text"], "")

    def test_from_object(self):
        seg = types.SimpleNamespace(id=2, start=4.0, end=6.5, text="  world  ")
        normed = _normalize_segment(seg, from_dict=False)
        self.assertEqual(normed["id"], 2)
        self.assertAlmostEqual(normed["start"], 4.0)
        self.assertAlmostEqual(normed["end"], 6.5)
        self.assertEqual(normed["text"], "world")

    def test_from_object_no_id(self):
        seg = types.SimpleNamespace(start=0, end=1, text="ok")
        normed = _normalize_segment(seg, from_dict=False)
        self.assertIsNone(normed["id"])

    def test_from_object_none_text(self):
        seg = types.SimpleNamespace(id=0, start=0, end=1, text=None)
        normed = _normalize_segment(seg, from_dict=False)
        self.assertEqual(normed["text"], "")


class TestCompletionProgress(unittest.TestCase):
    """Tests for _completion_progress helper."""

    def test_emits_100_percent(self):
        app = MockApp()
        segs = [{"end": 30.0}]
        start = time.time() - 10.0
        _completion_progress(app, Path("/f.wav"), start, 30.0, segs)
        kind, payload = app.events[0]
        self.assertEqual(kind, "current_progress")
        self.assertEqual(payload["percent"], 100)
        self.assertEqual(payload["phase"], "Transcribing complete")
        self.assertGreater(payload["elapsed"], 0)

    def test_empty_segments(self):
        app = MockApp()
        start = time.time()
        _completion_progress(app, Path("/f.wav"), start, 0, [])
        payload = app.events[0][1]
        self.assertEqual(payload["percent"], 100)
        self.assertEqual(payload["processed_seconds"], 0)


# ---------------------------------------------------------------------------
# preconvert_audio paths
# ---------------------------------------------------------------------------

class TestPreconvertAudio(unittest.TestCase):
    """Tests for preconvert_audio: WAV passthrough, no-ffmpeg fallback, success."""

    def test_wav_returns_unchanged(self):
        p = Path("/some/file.wav")
        self.assertEqual(preconvert_audio(p), p)

    def test_wav_case_insensitive(self):
        p = Path("/some/file.WAV")
        self.assertEqual(preconvert_audio(p), p)

    @patch(f"{WORKER_MODULE}.ffmpeg_available", return_value=False)
    def test_no_ffmpeg_returns_source(self, _mock):
        p = Path("/some/file.mp3")
        self.assertEqual(preconvert_audio(p), p)

    @patch(f"{WORKER_MODULE}.ffmpeg_available", return_value=True)
    @patch(f"{WORKER_MODULE}.subprocess.run")
    def test_successful_conversion(self, mock_run, _mock_ff):
        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "test.mp3"
            src.write_bytes(b"\x00" * 50)
            # Make subprocess.run simulate a successful conversion
            def fake_run(cmd, **kw):
                out_path = Path(cmd[-1])
                out_path.write_bytes(b"RIFF" + b"\x00" * 50)
                return MagicMock(returncode=0)
            mock_run.side_effect = fake_run
            result = preconvert_audio(src)
            self.assertTrue(result.suffix == ".wav")
            self.assertNotEqual(result, src)

    @patch(f"{WORKER_MODULE}.ffmpeg_available", return_value=True)
    @patch(f"{WORKER_MODULE}.subprocess.run", side_effect=OSError("ffmpeg died"))
    def test_ffmpeg_crash_returns_source(self, _mock_run, _mock_ff):
        p = Path("/some/file.mp3")
        self.assertEqual(preconvert_audio(p), p)


# ---------------------------------------------------------------------------
# _auto_batch_size branches
# ---------------------------------------------------------------------------

class TestAutoBatchSize(unittest.TestCase):
    """Tests for _auto_batch_size GPU memory thresholds."""

    def test_cpu_returns_1(self):
        self.assertEqual(_auto_batch_size("cpu"), 1)

    @patch(f"{WORKER_MODULE}.torch", create=True)
    def test_auto_without_torch_returns_1(self, mock_torch):
        """When device is 'auto' and torch import fails, fall back to cpu."""
        # Simulate torch import failure by making the function's import raise
        with patch.dict("sys.modules", {"torch": None}):
            result = _auto_batch_size("auto")
            self.assertEqual(result, 1)

    def _mock_torch(self, free_gb):
        """Create a mock torch module with given free GPU memory."""
        mock = MagicMock()
        mock.cuda.is_available.return_value = True
        mock.cuda.get_device_properties.return_value = MagicMock(
            total_mem=int(free_gb * 2 * (1024 ** 3)))
        mock.cuda.mem_get_info.return_value = (
            int(free_gb * (1024 ** 3)),
            int(free_gb * 2 * (1024 ** 3)),
        )
        return mock

    def test_cuda_high_memory_returns_16(self):
        with patch.dict("sys.modules", {"torch": self._mock_torch(8.0)}):
            self.assertEqual(_auto_batch_size("cuda"), 16)

    def test_cuda_medium_memory_returns_8(self):
        with patch.dict("sys.modules", {"torch": self._mock_torch(4.0)}):
            self.assertEqual(_auto_batch_size("cuda"), 8)

    def test_cuda_low_memory_returns_4(self):
        with patch.dict("sys.modules", {"torch": self._mock_torch(2.0)}):
            self.assertEqual(_auto_batch_size("cuda"), 4)

    def test_cuda_very_low_memory_returns_1(self):
        with patch.dict("sys.modules", {"torch": self._mock_torch(1.0)}):
            self.assertEqual(_auto_batch_size("cuda"), 1)


# ---------------------------------------------------------------------------
# _get_batched_pipeline fallback
# ---------------------------------------------------------------------------

class TestGetBatchedPipeline(unittest.TestCase):
    """Tests for _get_batched_pipeline caching and ImportError fallback."""

    def setUp(self):
        _batched_pipeline_cache.clear()

    def test_import_error_returns_model(self):
        """When faster_whisper.BatchedInferencePipeline is missing, return model."""
        model = MagicMock(name="fake_model")
        with patch.dict("sys.modules", {"faster_whisper": MagicMock(spec=[])}):
            result = _get_batched_pipeline(model)
        self.assertIs(result, model)

    def test_cache_hit(self):
        model = MagicMock(name="fake_model")
        sentinel = object()
        _batched_pipeline_cache[id(model)] = sentinel
        result = _get_batched_pipeline(model)
        self.assertIs(result, sentinel)

    def test_cache_stores_result(self):
        model = MagicMock(name="fake_model")
        _batched_pipeline_cache.clear()
        with patch.dict("sys.modules", {"faster_whisper": MagicMock(spec=[])}):
            _get_batched_pipeline(model)
        self.assertIn(id(model), _batched_pipeline_cache)


# ---------------------------------------------------------------------------
# Pause / resume mid-file
# ---------------------------------------------------------------------------

class TestPauseResume(unittest.TestCase):
    """Tests for pause/resume behavior during transcription."""

    def test_pause_then_resume_completes(self):
        """Simulate pause then resume — job finishes and emits pause/resume events."""
        with tempfile.TemporaryDirectory() as tmpdir:
            opts = _make_opts(tmpdir)
            audio = _make_fake_audio(tmpdir)
            app = MockApp(files=[audio])
            app.pause_event.clear()  # start paused

            # Unpause from a background thread before the worker starts
            def unpause():
                time.sleep(0.05)
                app.pause_event.set()
            threading.Thread(target=unpause, daemon=True).start()

            patches, _ = _patch_worker()
            with patches["load_model"], patches["transcribe_faster"], \
                 patches["transcribe_openai"], patches["preconvert_audio"], \
                 patches["probe_duration_seconds"], \
                 patches["validate_audio_file"]:
                run_transcription_worker(app, opts, 1)
            kinds = [e[0] for e in app.events]
            self.assertIn("paused", kinds)
            self.assertIn("resumed", kinds)
            done_event = [e for e in app.events if e[0] == "done"][0]
            self.assertEqual(done_event[1]["done_count"], 1)

    def test_cancel_while_paused(self):
        """Cancel request during pause should result in cancelled done event."""
        with tempfile.TemporaryDirectory() as tmpdir:
            opts = _make_opts(tmpdir)
            audio = _make_fake_audio(tmpdir)
            app = MockApp(files=[audio])
            app.pause_event.clear()  # start paused

            # Cancel quickly from another thread so wait_if_paused exits
            def do_cancel():
                time.sleep(0.02)
                app.cancel_requested = True
            threading.Thread(target=do_cancel, daemon=True).start()

            patches, _ = _patch_worker()
            with patches["load_model"], patches["transcribe_faster"], \
                 patches["transcribe_openai"], patches["preconvert_audio"], \
                 patches["probe_duration_seconds"], \
                 patches["validate_audio_file"]:
                run_transcription_worker(app, opts, 1)
            done_event = [e for e in app.events if e[0] == "done"][0]
            self.assertTrue(done_event[1]["cancelled"])


# ---------------------------------------------------------------------------
# Model corruption detection & recovery
# ---------------------------------------------------------------------------

class TestIsCt2CorruptionError(unittest.TestCase):
    """Tests for _is_ct2_corruption_error."""

    def test_matches_type_error_305(self):
        exc = RuntimeError(
            "[json.exception.type_error.305] cannot use operator[] "
            "with a string argument with null"
        )
        self.assertTrue(_is_ct2_corruption_error(exc))

    def test_matches_other_type_error_codes(self):
        exc = RuntimeError("[json.exception.type_error.302] type must be a number")
        self.assertTrue(_is_ct2_corruption_error(exc))

    def test_rejects_unrelated_runtime_error(self):
        exc = RuntimeError("CUDA out of memory")
        self.assertFalse(_is_ct2_corruption_error(exc))

    def test_rejects_non_runtime_error(self):
        exc = ValueError("[json.exception.type_error.305] wrong type")
        self.assertFalse(_is_ct2_corruption_error(exc))

    def test_matches_unable_to_open_file(self):
        # Load-time error from ctranslate2 when model.bin is missing or its
        # binary format is incompatible with the installed ctranslate2 version.
        exc = RuntimeError(
            "Unable to open file 'model.bin' in model "
            "'C:\\Users\\testuser\\.cache\\huggingface\\hub\\"
            "models--ivrit-ai--whisper-large-v3-ct2\\snapshots\\abc123'"
        )
        self.assertTrue(_is_ct2_corruption_error(exc))


class TestModelBinStatus(unittest.TestCase):
    """Tests for _model_bin_status validation helper."""

    def test_ok_for_complete_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir)
            (d / "model.bin").write_bytes(b"x" * (11 * 1024 * 1024))
            (d / "config.json").write_text("{}", encoding="utf-8")
            (d / "tokenizer.json").write_text("{}", encoding="utf-8")
            ok, detail = _model_bin_status(d)
            self.assertTrue(ok, detail)
            self.assertIn("ok", detail)

    def test_fails_when_model_bin_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir)
            (d / "config.json").write_text("{}", encoding="utf-8")
            (d / "tokenizer.json").write_text("{}", encoding="utf-8")
            ok, detail = _model_bin_status(d)
            self.assertFalse(ok)
            self.assertIn("missing", detail)

    def test_fails_when_model_bin_too_small(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir)
            (d / "model.bin").write_bytes(b"tiny")
            (d / "config.json").write_text("{}", encoding="utf-8")
            (d / "tokenizer.json").write_text("{}", encoding="utf-8")
            ok, detail = _model_bin_status(d)
            self.assertFalse(ok)
            self.assertIn("too small", detail)


class TestDownloadCacheCheck(unittest.TestCase):
    """An unfinished cache must resume, not raise and not count as done."""

    def test_incomplete_cache_resumes_download(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            incomplete = Path(tmpdir) / "incomplete"
            incomplete.mkdir()
            (incomplete / "config.json").write_text("{}", encoding="utf-8")
            (incomplete / "tokenizer.json").write_text("{}", encoding="utf-8")

            complete = Path(tmpdir) / "complete"
            complete.mkdir()
            (complete / "model.bin").write_bytes(b"x" * (11 * 1024 * 1024))
            (complete / "config.json").write_text("{}", encoding="utf-8")
            (complete / "tokenizer.json").write_text("{}", encoding="utf-8")

            calls = []

            def fake_snapshot(repo_id, local_files_only=False, **kwargs):
                calls.append(bool(local_files_only))
                if local_files_only:
                    return str(incomplete)
                return str(complete)

            host = MagicMock()
            fake_hub = types.ModuleType("huggingface_hub")
            fake_hub.snapshot_download = fake_snapshot
            with patch.dict("sys.modules", {"huggingface_hub": fake_hub}):
                result = _download_with_progress(host, "org/model")

            self.assertEqual(result, str(complete))
            self.assertEqual(calls, [True, False])
            messages = [
                c.kwargs.get("message", "")
                for c in host.post_event.call_args_list
                if c.args and c.args[0] == "log"
            ]
            self.assertTrue(any("incomplete" in m.lower() for m in messages))
            self.assertFalse(any("already cached" in m.lower() for m in messages))

    def test_complete_cache_is_used(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            complete = Path(tmpdir) / "complete"
            complete.mkdir()
            (complete / "model.bin").write_bytes(b"x" * (11 * 1024 * 1024))
            (complete / "config.json").write_text("{}", encoding="utf-8")
            (complete / "tokenizer.json").write_text("{}", encoding="utf-8")

            calls = []

            def fake_snapshot(repo_id, local_files_only=False, **kwargs):
                calls.append(bool(local_files_only))
                return str(complete)

            host = MagicMock()
            fake_hub = types.ModuleType("huggingface_hub")
            fake_hub.snapshot_download = fake_snapshot
            with patch.dict("sys.modules", {"huggingface_hub": fake_hub}):
                result = _download_with_progress(host, "org/model")

            self.assertEqual(result, str(complete))
            self.assertEqual(calls, [True])
            messages = [
                c.kwargs.get("message", "")
                for c in host.post_event.call_args_list
                if c.args and c.args[0] == "log"
            ]
            self.assertTrue(any("already cached" in m.lower() for m in messages))


class TestFindHfCacheDir(unittest.TestCase):
    """Tests for _find_hf_cache_dir."""

    def test_hub_id_found(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "models--ivrit-ai--whisper-large-v3-ct2"
            cache_dir.mkdir()
            with patch("hebrewscribe.models.get_cache_dirs",
                       return_value=[Path(tmpdir)]):
                result = _find_hf_cache_dir("ivrit-ai/whisper-large-v3-ct2")
            self.assertEqual(result, cache_dir)

    def test_hub_id_not_found(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("hebrewscribe.models.get_cache_dirs",
                       return_value=[Path(tmpdir)]):
                result = _find_hf_cache_dir("ivrit-ai/whisper-large-v3-ct2")
            self.assertIsNone(result)

    def test_local_path_inside_hf_cache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = (Path(tmpdir) / "models--org--name" /
                         "snapshots" / "abc123")
            model_dir.mkdir(parents=True)
            result = _find_hf_cache_dir(str(model_dir))
            self.assertIsNotNone(result)
            self.assertEqual(result.name, "models--org--name")

    def test_local_path_outside_hf_cache(self):
        result = _find_hf_cache_dir("/usr/local/models/my_model")
        self.assertIsNone(result)


class TestModelCorruptionRecovery(unittest.TestCase):
    """Integration test: corrupt model triggers purge+reload+retry."""

    def test_recovery_succeeds_and_file_completes(self):
        """First transcribe raises CT2 corruption error, recovery succeeds,
        retry transcribes successfully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            opts = _make_opts(tmpdir)
            audio = _make_fake_audio(tmpdir)
            app = MockApp(files=[audio])

            call_count = [0]

            def transcribe_fail_then_succeed(host, model, file_path, opts, display_path=None):
                call_count[0] += 1
                if call_count[0] == 1:
                    raise RuntimeError(
                        "[json.exception.type_error.305] cannot use operator[] "
                        "with a string argument with null"
                    )
                return _FAKE_RESULT.copy()

            fresh_model = MagicMock(name="fresh_model")
            patches, mocks = _patch_worker(
                transcribe_faster=MagicMock(
                    side_effect=transcribe_fail_then_succeed),
            )

            # Mock _purge_and_reload to return a fresh model.
            with patches["load_model"], patches["transcribe_faster"], \
                 patches["transcribe_openai"], patches["preconvert_audio"], \
                 patches["probe_duration_seconds"], \
                 patches["validate_audio_file"], \
                 patch(f"{WORKER_MODULE}._purge_and_reload",
                       return_value=(fresh_model, "cpu")) as mock_purge:
                run_transcription_worker(app, opts, 1)

            # Recovery was attempted.
            mock_purge.assert_called_once()
            # transcribe_faster was called twice (fail + retry).
            self.assertEqual(call_count[0], 2)
            # File completed successfully after retry.
            done_event = [e for e in app.events if e[0] == "done"][0]
            self.assertEqual(done_event[1]["done_count"], 1)
            self.assertEqual(done_event[1]["failed_count"], 0)
            # Log mentions recovery.
            log_msgs = [e[1]["message"] for e in app.events if e[0] == "log"]
            self.assertTrue(any("recovery" in m.lower() for m in log_msgs))

    def test_recovery_only_attempted_once_per_batch(self):
        """If recovery already happened and a second file hits the same error,
        it fails normally without a second purge."""
        with tempfile.TemporaryDirectory() as tmpdir:
            opts = _make_opts(tmpdir)
            files = [_make_fake_audio(tmpdir, f"f{i}.wav") for i in range(3)]
            app = MockApp(files=files)

            call_count = [0]

            def transcribe_fail_or_succeed(host, model, file_path, opts, display_path=None):
                call_count[0] += 1
                # Calls 1 (file1 fail) and 2 (file1 retry) and 3 (file2 fail)
                if call_count[0] in (1, 4):
                    raise RuntimeError(
                        "[json.exception.type_error.305] ...")
                return _FAKE_RESULT.copy()

            fresh_model = MagicMock(name="fresh_model")
            patches, mocks = _patch_worker(
                transcribe_faster=MagicMock(
                    side_effect=transcribe_fail_or_succeed),
            )
            with patches["load_model"], patches["transcribe_faster"], \
                 patches["transcribe_openai"], patches["preconvert_audio"], \
                 patches["probe_duration_seconds"], \
                 patches["validate_audio_file"], \
                 patch(f"{WORKER_MODULE}._purge_and_reload",
                       return_value=(fresh_model, "cpu")) as mock_purge:
                run_transcription_worker(app, opts, len(files))

            # Recovery attempted only once (for file 1).
            self.assertEqual(mock_purge.call_count, 1)
            done_event = [e for e in app.events if e[0] == "done"][0]
            # file1: recovered, file2: succeeded, file3: failed (no second recovery)
            self.assertEqual(done_event[1]["done_count"], 2)
            self.assertEqual(done_event[1]["failed_count"], 1)

    def test_no_recovery_for_openai_backend(self):
        """CT2 corruption error with openai-whisper backend is not recoverable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            opts = _make_opts(tmpdir, backend="openai-whisper")
            audio = _make_fake_audio(tmpdir)
            app = MockApp(files=[audio])

            patches, mocks = _patch_worker(
                transcribe_openai=MagicMock(side_effect=RuntimeError(
                    "[json.exception.type_error.305] ...")),
            )
            with patches["load_model"], patches["transcribe_faster"], \
                 patches["transcribe_openai"], patches["preconvert_audio"], \
                 patches["probe_duration_seconds"], \
                 patches["validate_audio_file"], \
                 patch(f"{WORKER_MODULE}._purge_and_reload") as mock_purge:
                run_transcription_worker(app, opts, 1)

            mock_purge.assert_not_called()
            done_event = [e for e in app.events if e[0] == "done"][0]
            self.assertEqual(done_event[1]["failed_count"], 1)

    def test_recovery_failure_falls_through(self):
        """If _purge_and_reload itself fails, the file fails normally."""
        with tempfile.TemporaryDirectory() as tmpdir:
            opts = _make_opts(tmpdir)
            audio = _make_fake_audio(tmpdir)
            app = MockApp(files=[audio])

            patches, mocks = _patch_worker(
                transcribe_faster=MagicMock(side_effect=RuntimeError(
                    "[json.exception.type_error.305] ...")),
            )
            with patches["load_model"], patches["transcribe_faster"], \
                 patches["transcribe_openai"], patches["preconvert_audio"], \
                 patches["probe_duration_seconds"], \
                 patches["validate_audio_file"], \
                 patch(f"{WORKER_MODULE}._purge_and_reload",
                       side_effect=RuntimeError("Download failed")):
                run_transcription_worker(app, opts, 1)

            done_event = [e for e in app.events if e[0] == "done"][0]
            self.assertEqual(done_event[1]["done_count"], 0)
            self.assertEqual(done_event[1]["failed_count"], 1)

    def test_recovery_at_load_time(self):
        """When load_model raises an 'Unable to open file' CT2 error, the worker
        purges the cache, reloads, and then transcribes successfully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            opts = _make_opts(tmpdir)
            audio = _make_fake_audio(tmpdir)
            app = MockApp(files=[audio])

            fresh_model = MagicMock(name="fresh_model")
            corrupt_load_error = RuntimeError(
                "Unable to open file 'model.bin' in model '/some/cache/path'"
            )

            patches, mocks = _patch_worker()
            # load_model raises the CT2 load-time error on first call.
            patches["load_model"] = patch(
                f"{WORKER_MODULE}.load_model",
                side_effect=corrupt_load_error,
            )
            with patches["load_model"], patches["transcribe_faster"], \
                 patches["transcribe_openai"], patches["preconvert_audio"], \
                 patches["probe_duration_seconds"], \
                 patches["validate_audio_file"], \
                 patch(f"{WORKER_MODULE}._purge_and_reload",
                       return_value=(fresh_model, "cpu")) as mock_purge:
                run_transcription_worker(app, opts, 1)

            # Recovery was triggered at load time.
            mock_purge.assert_called_once()
            # Transcription completed successfully after reload.
            done_event = [e for e in app.events if e[0] == "done"][0]
            self.assertEqual(done_event[1]["done_count"], 1)
            self.assertEqual(done_event[1]["failed_count"], 0)
            # Log mentions recovery.
            log_msgs = [e[1]["message"] for e in app.events if e[0] == "log"]
            self.assertTrue(any("recover" in m.lower() for m in log_msgs))

    def test_no_load_time_recovery_for_openai_backend(self):
        """CT2 load-time error with openai-whisper backend is not recoverable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            opts = _make_opts(tmpdir, backend="openai-whisper")
            audio = _make_fake_audio(tmpdir)
            app = MockApp(files=[audio])

            patches, mocks = _patch_worker()
            patches["load_model"] = patch(
                f"{WORKER_MODULE}.load_model",
                side_effect=RuntimeError(
                    "Unable to open file 'model.bin' in model '/some/cache'"
                ),
            )
            with patches["load_model"], patches["transcribe_faster"], \
                 patches["transcribe_openai"], patches["preconvert_audio"], \
                 patches["probe_duration_seconds"], \
                 patches["validate_audio_file"], \
                 patch(f"{WORKER_MODULE}._purge_and_reload") as mock_purge:
                run_transcription_worker(app, opts, 1)

            mock_purge.assert_not_called()
            # Worker posts "failed" (not "done") because the job never started.
            failed_events = [e for e in app.events if e[0] == "failed"]
            self.assertTrue(len(failed_events) > 0)


class TestDisplayPathInEvents(unittest.TestCase):
    """Progress/current_file events must carry the original source path even
    when the audio was pre-converted to a temp WAV — GUI rows are keyed by the
    original path, so a temp path would silently kill per-row progress."""

    def test_preconverted_events_use_original_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            opts = _make_opts(tmpdir)
            audio = _make_fake_audio(tmpdir, "song.mp3")
            temp_wav = Path(tmpdir) / "song_abc123.wav"
            temp_wav.write_bytes(b"RIFF" + b"\x00" * 50)

            seg = types.SimpleNamespace(id=0, start=0.0, end=1.0, text="hi")
            info = types.SimpleNamespace(duration=1.0, language="he",
                                         language_probability=1.0)
            model = MagicMock(name="fake_model")
            model.transcribe.return_value = (iter([seg]), info)

            app = MockApp(files=[audio])
            patches, mocks = _patch_worker(
                load_model=MagicMock(return_value=(model, "cpu")),
                preconvert_audio=MagicMock(return_value=temp_wav),
            )
            # transcribe_faster is deliberately NOT patched — the real one
            # must route display_path into its events.
            with patches["load_model"], patches["preconvert_audio"], \
                 patches["probe_duration_seconds"], \
                 patches["validate_audio_file"]:
                run_transcription_worker(app, opts, 1)

            # The model was fed the temp WAV...
            model.transcribe.assert_called_once()
            self.assertEqual(model.transcribe.call_args[0][0], str(temp_wav))
            # ...but every event names the original file.
            prog_paths = {e[1]["path"] for e in app.events
                          if e[0] == "current_progress"}
            self.assertEqual(prog_paths, {audio})
            names = {e[1]["name"] for e in app.events if e[0] == "current_file"}
            self.assertIn("song.mp3", names)
            self.assertNotIn(temp_wav.name, names)


class TestPullBasedFileOrder(unittest.TestCase):
    """Verify that the pull-based worker processes files in next_file() order."""

    def test_files_processed_in_queue_order(self):
        """Worker processes files in the order next_file() returns them."""
        with tempfile.TemporaryDirectory() as tmpdir:
            opts = _make_opts(tmpdir)
            files = [_make_fake_audio(tmpdir, f"file{i}.wav") for i in range(4)]
            app = MockApp(files=files)
            patches, mocks = _patch_worker()
            with patches["load_model"], patches["transcribe_faster"], \
                 patches["transcribe_openai"], patches["preconvert_audio"], \
                 patches["probe_duration_seconds"], \
                 patches["validate_audio_file"]:
                run_transcription_worker(app, opts, len(files))

            # Extract the order files were started
            started_paths = [
                e[1]["path"] for e in app.events if e[0] == "file_started"
            ]
            self.assertEqual(started_paths, files)

    def test_reordered_queue_respected(self):
        """If next_file() returns files in a different order, worker follows."""
        with tempfile.TemporaryDirectory() as tmpdir:
            opts = _make_opts(tmpdir)
            files = [_make_fake_audio(tmpdir, f"file{i}.wav") for i in range(3)]
            # Reverse the order the mock returns files
            reordered = list(reversed(files))
            app = MockApp(files=reordered)
            patches, mocks = _patch_worker()
            with patches["load_model"], patches["transcribe_faster"], \
                 patches["transcribe_openai"], patches["preconvert_audio"], \
                 patches["probe_duration_seconds"], \
                 patches["validate_audio_file"]:
                run_transcription_worker(app, opts, len(files))

            started_paths = [
                e[1]["path"] for e in app.events if e[0] == "file_started"
            ]
            self.assertEqual(started_paths, reordered)

    def test_next_file_none_stops_worker(self):
        """Worker stops cleanly when next_file() returns None immediately."""
        with tempfile.TemporaryDirectory() as tmpdir:
            opts = _make_opts(tmpdir)
            app = MockApp(files=[])  # empty queue
            patches, mocks = _patch_worker()
            with patches["load_model"], patches["transcribe_faster"], \
                 patches["transcribe_openai"], patches["preconvert_audio"], \
                 patches["probe_duration_seconds"], \
                 patches["validate_audio_file"]:
                run_transcription_worker(app, opts, 0)

            done_event = [e for e in app.events if e[0] == "done"][0]
            self.assertEqual(done_event[1]["done_count"], 0)
            self.assertEqual(done_event[1]["failed_count"], 0)
            self.assertIn("Finished", done_event[1]["message"])


class TestWorkerDiarization(unittest.TestCase):
    """Diarization integration: RunOptions, kwargs gating, flow, degrade paths."""

    DIARIZE_MODULE = "hebrewscribe.diarize"

    def _diarize_patches(self, **overrides):
        """Patches on hebrewscribe.diarize used by the worker's lazy imports."""
        from types import SimpleNamespace
        turns = [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}]
        defaults = {
            "load_diarization_pipeline": MagicMock(return_value="fake-pipeline"),
            "load_waveform": MagicMock(
                return_value={"waveform": SimpleNamespace(), "sample_rate": 16000}),
            "diarize_waveform": MagicMock(return_value="fake-annotation"),
            "annotation_to_turns": MagicMock(return_value=turns),
        }
        defaults.update(overrides)
        return {name: patch(f"{self.DIARIZE_MODULE}.{name}", mock)
                for name, mock in defaults.items()}, defaults

    def _run(self, opts, app, dia_overrides=None, worker_overrides=None):
        patches, mocks = _patch_worker(**(worker_overrides or {}))
        dia_patches, dia_mocks = self._diarize_patches(**(dia_overrides or {}))
        with patches["load_model"], patches["transcribe_faster"], \
             patches["transcribe_openai"], patches["preconvert_audio"], \
             patches["probe_duration_seconds"], \
             patches["validate_audio_file"], \
             dia_patches["load_diarization_pipeline"], \
             dia_patches["load_waveform"], dia_patches["diarize_waveform"], \
             dia_patches["annotation_to_turns"]:
            run_transcription_worker(app, opts, 1)
        return mocks, dia_mocks

    def test_runoptions_new_fields_default_off(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            opts = _make_opts(tmpdir)  # constructed without the new kwargs
            self.assertFalse(opts.diarize)
            self.assertEqual(opts.num_speakers, 0)
            self.assertEqual(opts.diarize_device, "auto")

    def test_validate_opts_rejects_negative_num_speakers(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            opts = _make_opts(tmpdir)
            opts.num_speakers = -1
            with self.assertRaises(ValueError):
                _validate_opts(opts)

    def test_normalize_segment_words_carried_and_absent(self):
        seg = types.SimpleNamespace(
            id=1, start=0.0, end=2.0, text="hi there",
            words=[types.SimpleNamespace(start=0.0, end=0.9, word="hi"),
                   types.SimpleNamespace(start=1.0, end=1.9, word=" there")])
        normed = _normalize_segment(seg)
        self.assertEqual(normed["words"], [
            {"start": 0.0, "end": 0.9, "word": "hi"},
            {"start": 1.0, "end": 1.9, "word": " there"}])
        seg_no_words = types.SimpleNamespace(id=1, start=0.0, end=2.0,
                                             text="hi", words=None)
        self.assertNotIn("words", _normalize_segment(seg_no_words))

    def test_transcribe_faster_word_timestamps_only_when_diarizing(self):
        from hebrewscribe.worker import transcribe_faster
        for diarize_on in (False, True):
            with tempfile.TemporaryDirectory() as tmpdir:
                opts = _make_opts(tmpdir)
                opts.batch_size = 1  # avoid the batched-pipeline path
                opts.diarize = diarize_on
                seen = {}

                def fake_transcribe(path, **kwargs):
                    seen.update(kwargs)
                    info = types.SimpleNamespace(
                        duration=1.0, language="he", language_probability=1.0)
                    return iter([]), info

                model = types.SimpleNamespace(transcribe=fake_transcribe)
                app = MockApp()
                transcribe_faster(app, model, Path(tmpdir) / "a.wav", opts)
                self.assertEqual("word_timestamps" in seen, diarize_on,
                                 f"diarize={diarize_on}: kwargs={sorted(seen)}")

    def test_diarize_happy_flow_labels_output(self):
        import json as json_mod
        with tempfile.TemporaryDirectory() as tmpdir:
            opts = _make_opts(tmpdir, formats=["txt", "srt", "json"])
            opts.diarize = True
            audio = _make_fake_audio(tmpdir, "hello.wav")
            app = MockApp(files=[audio])
            self._run(opts, app)
            txt = (Path(tmpdir) / "hello.txt").read_text(encoding="utf-8")
            self.assertEqual(txt, "דובר 1:\ntranscribed text\n")
            # srt/json must receive speakers through the REAL dispatcher —
            # guards the write_outputs speakers= passthrough for each writer.
            srt = (Path(tmpdir) / "hello.srt").read_text(encoding="utf-8")
            self.assertIn("דובר 1: transcribed text", srt)
            self.assertIn("-->", srt)
            data = json_mod.loads(
                (Path(tmpdir) / "hello.json").read_text(encoding="utf-8"))
            self.assertEqual(data["segments"][0]["speaker"], "SPEAKER_00")
            self.assertEqual(data["segments"][0]["speaker_label"], "דובר 1")
            self.assertEqual(data["meta"]["speakers"],
                             {"SPEAKER_00": "דובר 1"})
            phases = [p.get("phase") for k, p in app.events
                      if k == "current_file"]
            self.assertIn("Identifying speakers…", phases)
            logs = [p["message"] for k, p in app.events if k == "log"]
            self.assertIn("Identified 1 speaker", logs)
            # Live-preview refresh: the labeled text is posted and matches
            # the .txt content exactly.
            replaces = [p for k, p in app.events if k == "preview_replace"]
            self.assertEqual(len(replaces), 1)
            self.assertEqual(replaces[0]["text"] + "\n", txt)
            done = [p for k, p in app.events if k == "done"][0]
            self.assertEqual(done["done_count"], 1)
            self.assertEqual(done["failed_count"], 0)

    def test_no_turns_keeps_plain_output(self):
        """Diarizer succeeds but finds no speech turns (music/silence):
        outputs must be identical to a plain transcript — no labels invented."""
        with tempfile.TemporaryDirectory() as tmpdir:
            opts = _make_opts(tmpdir, formats=["txt"])
            opts.diarize = True
            audio = _make_fake_audio(tmpdir, "hello.wav")
            app = MockApp(files=[audio])
            self._run(opts, app, dia_overrides={
                "annotation_to_turns": MagicMock(return_value=[])})
            txt = (Path(tmpdir) / "hello.txt").read_text(encoding="utf-8")
            self.assertEqual(txt, "transcribed text\n")
            logs = [p["message"] for k, p in app.events if k == "log"]
            self.assertIn("Identified 0 speakers", logs)
            # No labels → no preview refresh (nothing to show over the stream).
            kinds = [e[0] for e in app.events]
            self.assertNotIn("preview_replace", kinds)
            done = [p for k, p in app.events if k == "done"][0]
            self.assertEqual(done["done_count"], 1)
            self.assertEqual(done["failed_count"], 0)

    def test_cancel_during_pipeline_load_cancels_run(self):
        """Cancel during the batch-start weights download must cancel the
        whole run (the DiarizationCancelled -> CancelledByUser conversion in
        the pipeline-load block), never degrade into a label-less batch."""
        from hebrewscribe.diarize import DiarizationCancelled
        with tempfile.TemporaryDirectory() as tmpdir:
            opts = _make_opts(tmpdir, formats=["txt"])
            opts.diarize = True
            audio = _make_fake_audio(tmpdir, "hello.wav")
            app = MockApp(files=[audio])
            _, dia_mocks = self._run(opts, app, dia_overrides={
                "load_diarization_pipeline": MagicMock(
                    side_effect=DiarizationCancelled())})
            done = [p for k, p in app.events if k == "done"][0]
            self.assertTrue(done["cancelled"])
            self.assertEqual(done["done_count"], 0)
            kinds = [e[0] for e in app.events]
            self.assertNotIn("file_finished", kinds)

    def test_pipeline_load_failure_degrades_batch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            opts = _make_opts(tmpdir, formats=["txt"])
            opts.diarize = True
            audio = _make_fake_audio(tmpdir, "hello.wav")
            app = MockApp(files=[audio])
            _, dia_mocks = self._run(opts, app, dia_overrides={
                "load_diarization_pipeline": MagicMock(
                    side_effect=RuntimeError("no weights"))})
            txt = (Path(tmpdir) / "hello.txt").read_text(encoding="utf-8")
            self.assertEqual(txt, "transcribed text\n")  # no labels
            warns = [p for k, p in app.events if k == "log"
                     and "without speaker labels" in p["message"]]
            self.assertTrue(warns)
            self.assertEqual(warns[0].get("level"), "warning")
            done = [p for k, p in app.events if k == "done"][0]
            self.assertEqual(done["done_count"], 1)
            # The per-file diarize path must never have run.
            dia_mocks["load_waveform"].assert_not_called()

    def test_per_file_diarize_failure_keeps_transcript(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            opts = _make_opts(tmpdir, formats=["txt"])
            opts.diarize = True
            audio = _make_fake_audio(tmpdir, "hello.wav")
            app = MockApp(files=[audio])
            self._run(opts, app, dia_overrides={
                "load_waveform": MagicMock(side_effect=MemoryError("oom"))})
            txt = (Path(tmpdir) / "hello.txt").read_text(encoding="utf-8")
            self.assertEqual(txt, "transcribed text\n")
            kinds = [e[0] for e in app.events]
            self.assertIn("file_finished", kinds)
            self.assertNotIn("preview_replace", kinds)  # failed → no refresh
            warns = [p for k, p in app.events if k == "log"
                     and "keeping the transcript" in p["message"]]
            self.assertTrue(warns)
            self.assertEqual(warns[0].get("level"), "warning")
            done = [p for k, p in app.events if k == "done"][0]
            self.assertEqual(done["done_count"], 1)
            self.assertEqual(done["failed_count"], 0)

    def test_cancel_mid_diarize(self):
        from hebrewscribe.diarize import DiarizationCancelled
        with tempfile.TemporaryDirectory() as tmpdir:
            opts = _make_opts(tmpdir, formats=["txt"])
            opts.diarize = True
            audio = _make_fake_audio(tmpdir, "hello.wav")
            app = MockApp(files=[audio])
            self._run(opts, app, dia_overrides={
                "diarize_waveform": MagicMock(
                    side_effect=DiarizationCancelled())})
            done = [p for k, p in app.events if k == "done"][0]
            self.assertTrue(done["cancelled"])
            failed = [p for k, p in app.events if k == "file_failed"]
            self.assertEqual(failed[0]["status"], "Cancelled")

    def test_retry_path_diarizes_too(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            opts = _make_opts(tmpdir, formats=["txt"])
            opts.diarize = True
            audio = _make_fake_audio(tmpdir, "hello.wav")
            app = MockApp(files=[audio])
            corrupt = RuntimeError(
                "[json.exception.type_error.305] cannot use operator[]")
            transcribe_mock = MagicMock(
                side_effect=[corrupt, _FAKE_RESULT.copy()])
            with patch(f"{WORKER_MODULE}._purge_and_reload",
                       return_value=(MagicMock(), "cpu")):
                _, dia_mocks = self._run(
                    opts, app,
                    worker_overrides={"transcribe_faster": transcribe_mock})
            txt = (Path(tmpdir) / "hello.txt").read_text(encoding="utf-8")
            self.assertEqual(txt, "דובר 1:\ntranscribed text\n")
            # Diarization ran exactly once — on the successful retry.
            self.assertEqual(dia_mocks["load_waveform"].call_count, 1)
            done = [p for k, p in app.events if k == "done"][0]
            self.assertEqual(done["done_count"], 1)

    def test_long_file_ram_warning(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            opts = _make_opts(tmpdir, formats=["txt"])
            opts.diarize = True
            audio = _make_fake_audio(tmpdir, "long.wav")
            app = MockApp(files=[audio])
            self._run(opts, app, worker_overrides={
                "probe_duration_seconds": MagicMock(return_value=3 * 3600.0)})
            warns = [p for k, p in app.events if k == "log"
                     and "over 2 hours" in p["message"]]
            self.assertTrue(warns)
            self.assertEqual(warns[0].get("level"), "warning")
            done = [p for k, p in app.events if k == "done"][0]
            self.assertEqual(done["done_count"], 1)

    def test_diarize_progress_events_omit_audio_seconds(self):
        """Diarize-phase current_progress must not carry processed/total
        seconds (would double-count batch audio in the GUI's ETA)."""
        from hebrewscribe.diarize import diarize_waveform

        class _FakePipeline:
            def __call__(self, waveform, num_speakers=None, hook=None):
                hook("segmentation", None, total=10, completed=10)
                return types.SimpleNamespace(
                    exclusive_speaker_diarization="ann")

        app = MockApp()
        out = diarize_waveform(_FakePipeline(), {"waveform": None},
                               app, Path("/a/x.wav"), num_speakers=0,
                               total_seconds=100.0)
        self.assertEqual(out, "ann")
        prog = [p for k, p in app.events if k == "current_progress"]
        self.assertTrue(prog)
        self.assertNotIn("processed_seconds", prog[0])
        self.assertNotIn("total_seconds", prog[0])
        self.assertEqual(prog[0]["phase"], "Identifying speakers…")
        self.assertEqual(prog[0]["row_status"], "Running")

    def test_diarize_num_speakers_zero_maps_to_none(self):
        from hebrewscribe.diarize import diarize_waveform
        seen = {}

        class _FakePipeline:
            def __call__(self, waveform, num_speakers=None, hook=None):
                seen["num_speakers"] = num_speakers
                return "ann"

        app = MockApp()
        diarize_waveform(_FakePipeline(), {"waveform": None}, app,
                         Path("/a/x.wav"), num_speakers=0)
        self.assertIsNone(seen["num_speakers"])
        diarize_waveform(_FakePipeline(), {"waveform": None}, app,
                         Path("/a/x.wav"), num_speakers=3)
        self.assertEqual(seen["num_speakers"], 3)


class TestSelfTestDiarization(unittest.TestCase):
    """The optional speaker-identification self-test step."""

    def _run_selftest(self, dia_patches):
        from contextlib import ExitStack

        from hebrewscribe.worker import run_selftest
        with ExitStack() as stack:
            stack.enter_context(patch(f"{WORKER_MODULE}.load_model",
                                      return_value=(MagicMock(), "cpu")))
            stack.enter_context(patch(f"{WORKER_MODULE}.transcribe_faster",
                                      return_value=_FAKE_RESULT.copy()))
            stack.enter_context(patch(f"{WORKER_MODULE}.ffmpeg_available",
                                      return_value=True))
            for p in dia_patches:
                stack.enter_context(p)
            return run_selftest(MockApp(), "faster-whisper", "cpu", "int8")

    def test_clean_skip_when_not_installed(self):
        result = self._run_selftest([
            patch("hebrewscribe.diarize.is_diarization_available",
                  return_value=False)])
        self.assertTrue(result.passed)
        self.assertTrue(any("optional" in c for c in result.checks))

    def test_warn_when_weights_missing(self):
        result = self._run_selftest([
            patch("hebrewscribe.diarize.is_diarization_available",
                  return_value=True),
            patch("hebrewscribe.diarize.resolve_weights_dir",
                  return_value=None)])
        self.assertTrue(result.passed)
        self.assertTrue(any("download on first use" in c
                            for c in result.checks))

    def test_ok_when_weights_present_and_pipeline_loads(self):
        result = self._run_selftest([
            patch("hebrewscribe.diarize.is_diarization_available",
                  return_value=True),
            patch("hebrewscribe.diarize.resolve_weights_dir",
                  return_value=Path("/fake/weights")),
            patch("hebrewscribe.diarize.load_diarization_pipeline",
                  return_value=MagicMock())])
        self.assertTrue(result.passed)
        self.assertTrue(any("pipeline loads" in c for c in result.checks))

    def test_fail_when_pipeline_broken(self):
        result = self._run_selftest([
            patch("hebrewscribe.diarize.is_diarization_available",
                  return_value=True),
            patch("hebrewscribe.diarize.resolve_weights_dir",
                  return_value=Path("/fake/weights")),
            patch("hebrewscribe.diarize.load_diarization_pipeline",
                  side_effect=RuntimeError("broken install"))])
        self.assertFalse(result.passed)
        self.assertTrue(any("Speaker identification broken" in c
                            for c in result.checks))


class _PausedHost:
    def __init__(self):
        self.pause_event = threading.Event()
        self.pause_event.set()  # running (not paused)
        self.cancel_requested = True
        self.stop_requested = False
        self.events = []

    def post_event(self, kind, **payload):
        self.events.append(kind)


class TestWaitIfPausedCancel(unittest.TestCase):
    def test_pending_cancel_wins_over_resume(self):
        """A cancel requested while paused must take effect on resume,
        not after the next file has already started."""
        from hebrewscribe.worker import wait_if_paused, CancelledByUser
        host = _PausedHost()
        with self.assertRaises(CancelledByUser):
            wait_if_paused(host)
        self.assertNotIn("resumed", host.events)


if __name__ == "__main__":
    unittest.main()
