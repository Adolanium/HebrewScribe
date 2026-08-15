"""End-to-end smoke test: real model, real audio, real output files.

Marked 'slow' so the normal test suite (pytest) skips it.
Run explicitly with:  pytest -m slow
                  or:  pytest -m slow --timeout=120

Requires:
  - faster-whisper installed
  - Internet access on first run (downloads Systran/faster-whisper-tiny, ~75 MB)
  - ffmpeg on PATH (for pre-conversion; the fixture is already WAV so optional)

The test downloads the model once; subsequent runs use the HuggingFace cache.
"""

import json
import tempfile
import unittest
from pathlib import Path

import pytest

from tests.helpers import MockApp

from hebrewscribe.worker import RunOptions, run_transcription_worker

FIXTURE_DIR = Path(__file__).parent / "fixtures"
SPEECH_WAV = FIXTURE_DIR / "speech_6s.wav"

# The smallest available faster-whisper model (~75 MB).
MODEL_REPO = "Systran/faster-whisper-tiny"

# Known transcript (lowercased, for substring matching).
# The fixture is synthesized speech (macOS `say`, voice Samantha):
# "This is a short recording used to test the transcription pipeline,
#  with a few seconds of spoken English."
# Exact wording may vary slightly across model versions.
EXPECTED_FRAGMENTS = ["recording", "pipeline"]


def _requires_faster_whisper():
    """Skip the test if faster-whisper is not installed."""
    try:
        import faster_whisper  # noqa: F401
        return False
    except ImportError:
        return True


@pytest.mark.slow
@pytest.mark.skipif(_requires_faster_whisper(),
                    reason="faster-whisper not installed")
class TestEndToEnd(unittest.TestCase):
    """Transcribe a real 6-second speech clip and verify output correctness."""

    def test_transcribe_speech_produces_all_outputs(self):
        """Full pipeline: load tiny model, transcribe speech, check outputs."""
        self.assertTrue(SPEECH_WAV.exists(),
                        f"Fixture missing: {SPEECH_WAV}")

        with tempfile.TemporaryDirectory() as tmpdir:
            opts = RunOptions(
                backend="faster-whisper",
                model_spec=MODEL_REPO,
                output_dir=tmpdir,
                language="en",
                task="transcribe",
                device="cpu",
                compute_type="int8",
                beam_size=1,
                vad_filter=False,
                condition_on_previous_text=False,
                batch_size=0,
                formats=["txt", "srt", "json"],
            )

            app = MockApp(files=[str(SPEECH_WAV)])
            run_transcription_worker(app, opts, 1)

            # --- Worker completed without fatal error ---
            kinds = [e[0] for e in app.events]
            self.assertNotIn("failed", kinds,
                             "Worker raised a fatal error: "
                             + str([e for e in app.events if e[0] == "failed"]))

            # --- Done event: 1 completed, 0 failed ---
            done_events = [e for e in app.events if e[0] == "done"]
            self.assertEqual(len(done_events), 1, "Expected exactly one 'done' event")
            done = done_events[0][1]
            self.assertEqual(done["done_count"], 1,
                             f"Expected 1 done, got {done['done_count']}")
            self.assertEqual(done["failed_count"], 0,
                             f"Expected 0 failed, got {done['failed_count']}")

            # --- Output files exist and are non-empty ---
            out_dir = Path(tmpdir)
            for ext in ["txt", "srt", "json"]:
                out_file = out_dir / f"speech_6s.{ext}"
                self.assertTrue(out_file.exists(),
                                f"Missing output: {out_file.name}")
                self.assertGreater(out_file.stat().st_size, 0,
                                   f"Empty output: {out_file.name}")

            # --- TXT contains recognizable speech ---
            txt_text = (out_dir / "speech_6s.txt").read_text("utf-8").lower()
            for fragment in EXPECTED_FRAGMENTS:
                self.assertIn(fragment, txt_text,
                              f"Expected '{fragment}' in transcript, got: {txt_text!r}")

            # --- SRT has at least one subtitle entry ---
            srt_text = (out_dir / "speech_6s.srt").read_text("utf-8")
            self.assertIn("-->", srt_text,
                          "SRT file has no timestamp arrows")

            # --- JSON has expected structure ---
            payload = json.loads((out_dir / "speech_6s.json").read_text("utf-8"))
            self.assertIn("text", payload)
            self.assertIn("segments", payload)
            self.assertIsInstance(payload["segments"], list)
            self.assertGreater(len(payload["segments"]), 0,
                               "JSON has zero segments")
            self.assertEqual(payload["backend"], "faster-whisper")
            self.assertEqual(payload["model"], MODEL_REPO)

    def test_model_loaded_event_emitted(self):
        """Verify the worker emits a model-loaded log event."""
        with tempfile.TemporaryDirectory() as tmpdir:
            opts = RunOptions(
                backend="faster-whisper",
                model_spec=MODEL_REPO,
                output_dir=tmpdir,
                language="en",
                task="transcribe",
                device="cpu",
                compute_type="int8",
                beam_size=1,
                vad_filter=False,
                condition_on_previous_text=False,
                batch_size=0,
                formats=["txt"],
            )

            app = MockApp(files=[str(SPEECH_WAV)])
            run_transcription_worker(app, opts, 1)

            log_messages = [e[1].get("message", "") for e in app.events
                           if e[0] == "log"]
            model_loaded = any("Model loaded" in m or "model" in m.lower()
                               for m in log_messages)
            self.assertTrue(model_loaded,
                            f"No model-loaded log event. Logs: {log_messages}")


if __name__ == "__main__":
    unittest.main()
