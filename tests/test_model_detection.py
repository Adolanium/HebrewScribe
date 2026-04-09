"""Tests for model detection and cache directory logic."""

import os
import tempfile
import unittest
from pathlib import Path
from hebrewscribe.models import (
    _looks_like_ct2_model_dir, detect_openai_cached_models,
    detect_faster_whisper_local_models, get_cache_dirs,
)


class TestLooksLikeCt2ModelDir(unittest.TestCase):
    def test_valid_model_dir_with_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir) / "model"
            d.mkdir()
            (d / "model.bin").write_bytes(b"fake")
            (d / "config.json").write_text("{}")
            self.assertTrue(_looks_like_ct2_model_dir(d))

    def test_valid_model_dir_with_tokenizer(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir) / "model"
            d.mkdir()
            (d / "model.bin").write_bytes(b"fake")
            (d / "tokenizer.json").write_text("{}")
            self.assertTrue(_looks_like_ct2_model_dir(d))

    def test_missing_model_bin(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir) / "model"
            d.mkdir()
            (d / "config.json").write_text("{}")
            self.assertFalse(_looks_like_ct2_model_dir(d))

    def test_missing_config_and_tokenizer(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir) / "model"
            d.mkdir()
            (d / "model.bin").write_bytes(b"fake")
            self.assertFalse(_looks_like_ct2_model_dir(d))

    def test_file_not_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            f = Path(tmpdir) / "model.bin"
            f.write_bytes(b"fake")
            self.assertFalse(_looks_like_ct2_model_dir(f))

    def test_nonexistent_path(self):
        self.assertFalse(_looks_like_ct2_model_dir(Path("/nonexistent/path")))


class TestDetectOpenaiCachedModels(unittest.TestCase):
    def test_finds_cached_pt_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = Path(tmpdir)
            (cache / "tiny.pt").write_bytes(b"fake")
            (cache / "base.pt").write_bytes(b"fake")
            # Patch the env to point to our temp cache
            old_env = os.environ.get("WHISPER_CACHE_DIR")
            os.environ["WHISPER_CACHE_DIR"] = str(cache)
            try:
                result = detect_openai_cached_models()
                self.assertIn("tiny", result)
                self.assertIn("base", result)
            finally:
                if old_env is None:
                    os.environ.pop("WHISPER_CACHE_DIR", None)
                else:
                    os.environ["WHISPER_CACHE_DIR"] = old_env

    def test_empty_cache_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            old_env = os.environ.get("WHISPER_CACHE_DIR")
            os.environ["WHISPER_CACHE_DIR"] = str(tmpdir)
            try:
                result = detect_openai_cached_models()
                # May contain models from default ~/.cache/whisper if it exists,
                # but at minimum should not crash
                self.assertIsInstance(result, list)
            finally:
                if old_env is None:
                    os.environ.pop("WHISPER_CACHE_DIR", None)
                else:
                    os.environ["WHISPER_CACHE_DIR"] = old_env


class TestDetectFasterWhisperLocalModels(unittest.TestCase):
    def test_finds_model_in_extra_roots(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "my-model"
            model_dir.mkdir()
            (model_dir / "model.bin").write_bytes(b"fake")
            (model_dir / "config.json").write_text("{}")
            result = detect_faster_whisper_local_models([str(model_dir)])
            labels = [r[0] for r in result]
            self.assertTrue(any("my-model" in label for label in labels))

    def test_finds_model_as_child_of_root(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "whisper-large-v3"
            model_dir.mkdir()
            (model_dir / "model.bin").write_bytes(b"fake")
            (model_dir / "config.json").write_text("{}")
            result = detect_faster_whisper_local_models([str(root)])
            labels = [r[0] for r in result]
            self.assertTrue(any("whisper-large-v3" in label for label in labels))

    def test_finds_huggingface_snapshot_model(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "models--Systran--faster-whisper-large-v3" / "snapshots" / "abc123"
            model_dir.mkdir(parents=True)
            (model_dir / "model.bin").write_bytes(b"fake")
            (model_dir / "config.json").write_text("{}")
            result = detect_faster_whisper_local_models([str(root)])
            labels = [r[0] for r in result]
            # friendly_label maps Systran/faster-whisper-large-v3 → "whisper large-v3"
            self.assertTrue(any("whisper large-v3" in label for label in labels))

    def test_empty_root_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = detect_faster_whisper_local_models([str(tmpdir)])
            # Should be a list (might have models from system cache dirs)
            self.assertIsInstance(result, list)

    def test_nonexistent_root_does_not_crash(self):
        result = detect_faster_whisper_local_models(["/nonexistent/model/path"])
        self.assertIsInstance(result, list)

    def test_deduplication(self):
        """Same model dir referenced via two roots should appear only once."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "model"
            model_dir.mkdir()
            (model_dir / "model.bin").write_bytes(b"fake")
            (model_dir / "config.json").write_text("{}")
            result = detect_faster_whisper_local_models([str(model_dir), str(model_dir)])
            paths = [r[1] for r in result]
            model_resolved = str(model_dir.resolve())
            self.assertEqual(paths.count(model_resolved), 1)


class TestGetCacheDirs(unittest.TestCase):
    def test_returns_list_of_paths(self):
        result = get_cache_dirs()
        self.assertIsInstance(result, list)
        for item in result:
            self.assertIsInstance(item, Path)

    def test_no_duplicates(self):
        result = get_cache_dirs()
        lowered = [str(p).lower() for p in result]
        self.assertEqual(len(lowered), len(set(lowered)))

    def test_includes_default_whisper_cache(self):
        result = get_cache_dirs()
        default = Path.home() / ".cache" / "whisper"
        self.assertIn(default, result)


if __name__ == "__main__":
    unittest.main()
