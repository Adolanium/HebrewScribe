"""Smoke tests for utility functions and output writers."""

import json
import unittest
from pathlib import Path

from hebrewscribe.utils import (
    format_timestamp, format_hms, safe_stem, unique_stem,
    load_json, save_json,
)
from hebrewscribe.outputs import write_txt, write_srt


class TestFormatTimestamp(unittest.TestCase):
    def test_zero(self):
        self.assertEqual(format_timestamp(0.0), "00:00:00,000")

    def test_normal(self):
        self.assertEqual(format_timestamp(3661.5), "01:01:01,500")

    def test_none_treated_as_zero(self):
        self.assertEqual(format_timestamp(None), "00:00:00,000")

    def test_sub_second(self):
        self.assertEqual(format_timestamp(0.001), "00:00:00,001")

    def test_large_value(self):
        result = format_timestamp(36000.0)
        self.assertEqual(result, "10:00:00,000")


class TestFormatHms(unittest.TestCase):
    def test_none(self):
        self.assertEqual(format_hms(None), "--:--")

    def test_negative(self):
        self.assertEqual(format_hms(-1), "--:--")

    def test_zero(self):
        self.assertEqual(format_hms(0), "00:00")

    def test_minutes_only(self):
        self.assertEqual(format_hms(125), "02:05")

    def test_with_hours(self):
        self.assertEqual(format_hms(3661), "01:01:01")


class TestSafeStem(unittest.TestCase):
    def test_normal(self):
        self.assertEqual(safe_stem(Path("/foo/bar.mp3")), "bar")

    def test_empty_stem(self):
        self.assertEqual(safe_stem(Path("/foo/.mp3")), "transcript")

    def test_whitespace_stem(self):
        self.assertEqual(safe_stem(Path("/foo/  .mp3")), "transcript")


class TestUniqueStem(unittest.TestCase):
    def test_no_collision(self):
        used = set()
        result = unique_stem(Path("/a/b/file.mp3"), Path("/out"), ["txt"], used)
        self.assertEqual(result, "file")
        self.assertIn("file", used)

    def test_collision_with_used_stems(self):
        used = {"file"}
        result = unique_stem(Path("/a/b/file.mp3"), Path("/out"), ["txt"], used)
        self.assertEqual(result, "file_b")

    def test_collision_with_existing_file(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir)
            (out / "file.txt").write_text("x")
            used = set()
            result = unique_stem(Path("/a/b/file.mp3"), out, ["txt"], used)
            self.assertEqual(result, "file_b")

    def test_double_collision(self):
        used = {"file", "file_b"}
        result = unique_stem(Path("/a/b/file.mp3"), Path("/out"), ["txt"], used)
        self.assertEqual(result, "file_b_2")


class TestLoadSaveJson(unittest.TestCase):
    def test_roundtrip(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "test.json"
            data = {"key": "value", "num": 42}
            save_json(p, data)
            loaded = load_json(p, {})
            self.assertEqual(loaded, data)

    def test_load_missing_returns_default(self):
        result = load_json(Path("/nonexistent/path.json"), {"default": True})
        self.assertEqual(result, {"default": True})

    def test_load_corrupt_returns_default(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "bad.json"
            p.write_text("not json{{{", encoding="utf-8")
            result = load_json(p, "fallback")
            self.assertEqual(result, "fallback")

    def test_load_corrupt_backs_up_original(self):
        """A corrupt file is renamed aside, not left to be overwritten."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "bad.json"
            p.write_text("not json{{{", encoding="utf-8")
            load_json(p, {})
            backup = Path(tmpdir) / "bad.json.corrupt.bak"
            self.assertFalse(p.exists())
            self.assertTrue(backup.exists())
            self.assertEqual(backup.read_text(encoding="utf-8"), "not json{{{")

    def test_save_leaves_no_temp_file(self):
        """Atomic save: the .tmp intermediate must not survive."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "out.json"
            save_json(p, {"a": 1})
            self.assertEqual(load_json(p, {}), {"a": 1})
            self.assertEqual(sorted(f.name for f in Path(tmpdir).iterdir()),
                             ["out.json"])


class TestWriteTxt(unittest.TestCase):
    def test_writes_text(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "out.txt"
            write_txt(p, "  hello world  ")
            self.assertEqual(p.read_text(encoding="utf-8"), "hello world\n")


class TestWriteSrt(unittest.TestCase):
    def test_writes_valid_srt(self):
        import tempfile
        segments = [
            {"start": 0.0, "end": 1.5, "text": "Hello"},
            {"start": 1.5, "end": 3.0, "text": "World"},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "out.srt"
            write_srt(p, segments)
            content = p.read_text(encoding="utf-8")
            self.assertIn("1\n00:00:00,000 --> 00:00:01,500\nHello", content)
            self.assertIn("2\n00:00:01,500 --> 00:00:03,000\nWorld", content)


def test_resilient_rotation_survives_oserror(tmp_path, monkeypatch):
    """Log rotation must survive a locked log file (second app instance):
    the handler reopens its stream instead of silently going dark."""
    import logging
    import logging.handlers
    from hebrewscribe.utils import _ResilientRotatingFileHandler

    log_file = tmp_path / "test.log"
    handler = _ResilientRotatingFileHandler(str(log_file), maxBytes=50,
                                            backupCount=2, encoding="utf-8")

    def failing_rollover(self):
        # Mimic the stock behaviour on Windows: stream closed, then the
        # rename fails because another process holds the file open.
        if self.stream:
            self.stream.close()
            self.stream = None
        raise OSError(32, "The process cannot access the file")

    monkeypatch.setattr(logging.handlers.RotatingFileHandler,
                        "doRollover", failing_rollover)

    handler.doRollover()  # must not raise
    assert handler.stream is not None and not handler.stream.closed

    record = logging.LogRecord("t", logging.INFO, __file__, 1,
                               "still logging", None, None)
    handler.emit(record)
    handler.close()
    assert "still logging" in log_file.read_text(encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
