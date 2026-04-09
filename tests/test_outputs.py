"""Tests for output writing, JSON format, RunOptions, and write_outputs dispatcher."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from hebrewscribe.worker import RunOptions, CancelledByUser, write_outputs
from hebrewscribe.utils import safe_stem


class TestWriteJson(unittest.TestCase):
    """Test the JSON output format directly (not covered by test_utils)."""

    def _write_json_via_dispatcher(self, source_name, result, formats=None):
        """Helper: call write_outputs and return the JSON content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            opts = RunOptions(
                backend="faster-whisper",
                model_spec="large-v3",
                output_dir=tmpdir,
                language="he",
                task="transcribe",
                device="cpu",
                compute_type="int8",
                beam_size=1,
                vad_filter=True,
                condition_on_previous_text=False,
                batch_size=0,
                formats=formats or ["json"],
            )
            source = Path(f"/audio/{source_name}")
            mock_app = MagicMock()
            used_stems = set()
            write_outputs(mock_app, source, opts, result, used_stems)
            json_path = Path(tmpdir) / f"{safe_stem(source)}.json"
            self.assertTrue(json_path.exists(), f"Expected {json_path} to exist")
            return json.loads(json_path.read_text(encoding="utf-8"))

    def test_json_contains_required_fields(self):
        result = {
            "text": "Hello world",
            "segments": [{"start": 0.0, "end": 1.0, "text": "Hello world"}],
            "language": "he",
        }
        data = self._write_json_via_dispatcher("test.mp3", result)
        self.assertEqual(data["source_file"], "/audio/test.mp3")
        self.assertEqual(data["backend"], "faster-whisper")
        self.assertEqual(data["model"], "large-v3")
        self.assertEqual(data["language"], "he")
        self.assertEqual(data["text"], "Hello world")
        self.assertEqual(len(data["segments"]), 1)

    def test_json_meta_excludes_text_segments_raw(self):
        result = {
            "text": "Hello",
            "segments": [],
            "language": "he",
            "language_probability": 0.98,
            "duration": 5.0,
            "raw": {"should_be_excluded": True},
        }
        data = self._write_json_via_dispatcher("test.mp3", result)
        self.assertIn("language_probability", data["meta"])
        self.assertIn("duration", data["meta"])
        self.assertNotIn("text", data["meta"])
        self.assertNotIn("segments", data["meta"])
        self.assertNotIn("raw", data["meta"])

    def test_json_hebrew_text(self):
        result = {
            "text": "שלום עולם",
            "segments": [{"start": 0.0, "end": 2.0, "text": "שלום עולם"}],
            "language": "he",
        }
        data = self._write_json_via_dispatcher("hebrew.wav", result)
        self.assertEqual(data["text"], "שלום עולם")
        self.assertEqual(data["segments"][0]["text"], "שלום עולם")


class TestWriteOutputsDispatcher(unittest.TestCase):
    """Test that write_outputs creates the right files for each format."""

    def _run_dispatcher(self, formats, source_name="test.mp3"):
        result = {
            "text": "Hello world",
            "segments": [
                {"start": 0.0, "end": 1.5, "text": "Hello"},
                {"start": 1.5, "end": 3.0, "text": "world"},
            ],
            "language": "en",
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            opts = RunOptions(
                backend="faster-whisper",
                model_spec="tiny",
                output_dir=tmpdir,
                language="en",
                task="transcribe",
                device="cpu",
                compute_type="int8",
                beam_size=1,
                vad_filter=True,
                condition_on_previous_text=False,
                batch_size=0,
                formats=formats,
            )
            mock_app = MagicMock()
            used_stems = set()
            write_outputs(mock_app, Path(f"/a/{source_name}"), opts, result, used_stems)
            out = Path(tmpdir)
            stem = safe_stem(Path(source_name))
            return out, stem, {f.name for f in out.iterdir()}

    def test_txt_only(self):
        out, stem, files = self._run_dispatcher(["txt"])
        self.assertEqual(files, {f"{stem}.txt"})

    def test_srt_only(self):
        out, stem, files = self._run_dispatcher(["srt"])
        self.assertEqual(files, {f"{stem}.srt"})

    def test_json_only(self):
        out, stem, files = self._run_dispatcher(["json"])
        self.assertEqual(files, {f"{stem}.json"})

    def test_all_formats(self):
        out, stem, files = self._run_dispatcher(["txt", "srt", "json"])
        expected = {f"{stem}.txt", f"{stem}.srt", f"{stem}.json"}
        self.assertEqual(files, expected)

    def test_no_extra_files(self):
        """Only requested formats should be written."""
        out, stem, files = self._run_dispatcher(["txt"])
        self.assertNotIn(f"{stem}.srt", files)
        self.assertNotIn(f"{stem}.json", files)


class TestRunOptions(unittest.TestCase):
    def test_dataclass_creation(self):
        opts = RunOptions(
            backend="faster-whisper",
            model_spec="large-v3",
            output_dir="/tmp/out",
            language="he",
            task="transcribe",
            device="auto",
            compute_type="auto",
            beam_size=1,
            vad_filter=True,
            condition_on_previous_text=False,
            batch_size=0,
            formats=["txt", "srt"],
        )
        self.assertEqual(opts.backend, "faster-whisper")
        self.assertEqual(opts.beam_size, 1)
        self.assertEqual(opts.formats, ["txt", "srt"])
        self.assertTrue(opts.vad_filter)
        self.assertFalse(opts.condition_on_previous_text)
        self.assertEqual(opts.batch_size, 0)

    def test_fields_are_accessible(self):
        opts = RunOptions(
            backend="openai-whisper",
            model_spec="tiny",
            output_dir="/tmp",
            language="auto",
            task="translate",
            device="cpu",
            compute_type="float32",
            beam_size=1,
            vad_filter=False,
            condition_on_previous_text=False,
            batch_size=0,
            formats=["json"],
        )
        self.assertEqual(opts.task, "translate")
        self.assertEqual(opts.compute_type, "float32")


class TestWriteOutputsEdgeCases(unittest.TestCase):
    def test_empty_segments(self):
        """Should handle empty segments list without crashing."""
        result = {"text": "", "segments": [], "language": "he"}
        with tempfile.TemporaryDirectory() as tmpdir:
            opts = RunOptions(
                backend="faster-whisper", model_spec="tiny", output_dir=tmpdir,
                language="he", task="transcribe", device="cpu", compute_type="int8",
                beam_size=1, vad_filter=True, condition_on_previous_text=False, batch_size=0, formats=["txt", "srt", "vtt", "json"],
            )
            mock_app = MagicMock()
            write_outputs(mock_app, Path("/a/empty.wav"), opts, result, set())
            self.assertTrue((Path(tmpdir) / "empty.txt").exists())
            self.assertTrue((Path(tmpdir) / "empty.srt").exists())

    def test_segment_with_none_text(self):
        """Segments with None text should be treated as empty string."""
        result = {
            "text": "",
            "segments": [{"start": 0.0, "end": 1.0, "text": None}],
            "language": "he",
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            opts = RunOptions(
                backend="faster-whisper", model_spec="tiny", output_dir=tmpdir,
                language="he", task="transcribe", device="cpu", compute_type="int8",
                beam_size=1, vad_filter=True, condition_on_previous_text=False, batch_size=0, formats=["srt", "vtt"],
            )
            mock_app = MagicMock()
            # Should not crash
            write_outputs(mock_app, Path("/a/test.wav"), opts, result, set())
            srt = (Path(tmpdir) / "test.srt").read_text(encoding="utf-8")
            self.assertIn("00:00:00,000 --> 00:00:01,000", srt)

    def test_collision_rename_logs(self):
        """When stem is renamed to avoid collision, post_event should be called."""
        result = {"text": "hi", "segments": [], "language": "en"}
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create existing file to force collision
            (Path(tmpdir) / "test.txt").write_text("existing")
            opts = RunOptions(
                backend="faster-whisper", model_spec="tiny", output_dir=tmpdir,
                language="en", task="transcribe", device="cpu", compute_type="int8",
                beam_size=1, vad_filter=True, condition_on_previous_text=False, batch_size=0, formats=["txt"],
            )
            mock_app = MagicMock()
            write_outputs(mock_app, Path("/a/b/test.wav"), opts, result, set())
            # Should have called post_event with rename message
            mock_app.post_event.assert_called()
            call_args = mock_app.post_event.call_args_list
            rename_calls = [c for c in call_args if "renamed" in str(c).lower()]
            self.assertTrue(len(rename_calls) > 0, "Expected a rename log event")


class TestCancelledByUser(unittest.TestCase):
    def test_is_exception(self):
        self.assertTrue(issubclass(CancelledByUser, Exception))

    def test_can_be_raised_and_caught(self):
        with self.assertRaises(CancelledByUser):
            raise CancelledByUser("test cancel")


if __name__ == "__main__":
    unittest.main()
