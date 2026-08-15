"""Tests for the structured log system (LogEntry, phase classification, path shortening)."""

import os
import pytest

from hebrewscribe.app import LogEntry, _classify_log_phase, TranscriberApp


class TestLogEntry:
    """LogEntry dataclass creation and defaults."""

    def test_defaults(self):
        entry = LogEntry(timestamp=1.0, time_str="20:00:00", message="test")
        assert entry.level == "info"
        assert entry.phase == "user"
        assert entry.run_id == 0
        assert entry.file_path is None
        assert entry.raw_only is False

    def test_all_fields(self):
        entry = LogEntry(
            timestamp=1.0, time_str="20:00:00", message="err",
            level="error", phase="file", run_id=2,
            file_path="/tmp/a.wav", raw_only=True)
        assert entry.level == "error"
        assert entry.phase == "file"
        assert entry.run_id == 2
        assert entry.file_path == "/tmp/a.wav"
        assert entry.raw_only is True


class TestPhaseClassification:
    """_classify_log_phase maps messages to (phase, raw_only) tuples."""

    def test_model_refresh_is_system_noise(self):
        phase, raw = _classify_log_phase(
            "Model list refreshed for backend: faster-whisper", "user")
        assert phase == "system"
        assert raw is True

    def test_loading_model_is_setup(self):
        phase, raw = _classify_log_phase(
            "Loading faster-whisper model: distil-large-v3", "user")
        assert phase == "setup"
        assert raw is False

    def test_device_info_is_setup(self):
        phase, raw = _classify_log_phase(
            "Device: cpu, compute: int8, batch_size: auto", "user")
        assert phase == "setup"
        assert raw is False

    def test_model_loaded_is_setup(self):
        phase, raw = _classify_log_phase(
            "Model loaded on device: cpu", "user")
        assert phase == "setup"
        assert raw is False

    def test_model_cached_is_setup(self):
        phase, raw = _classify_log_phase(
            "Model already cached: Systran/faster-distil-whisper-large-v3", "user")
        assert phase == "setup"
        assert raw is False

    def test_downloading_is_setup(self):
        phase, raw = _classify_log_phase(
            "Downloading model: Systran/faster-distil-whisper-large-v3", "user")
        assert phase == "setup"
        assert raw is False

    def test_download_complete_is_setup(self):
        phase, raw = _classify_log_phase(
            "Download complete: Systran/faster-distil-whisper-large-v3", "user")
        assert phase == "setup"
        assert raw is False

    def test_transcribing_is_file(self):
        phase, raw = _classify_log_phase(
            "Transcribing: Recording 4.wav", "user")
        assert phase == "file"
        assert raw is False

    def test_pre_converted_is_file(self):
        phase, raw = _classify_log_phase(
            "Pre-converted to WAV: Recording 4.tmp.wav", "user")
        assert phase == "file"
        assert raw is False

    def test_output_renamed_is_file(self):
        phase, raw = _classify_log_phase(
            "Output renamed to 'Recording 4_RØDE Connect.*' to avoid collision", "user")
        assert phase == "file"
        assert raw is False

    def test_done_file_is_file(self):
        phase, raw = _classify_log_phase("Done: Recording 4.wav", "user")
        assert phase == "file"
        assert raw is False

    def test_error_in_file_is_file(self):
        phase, raw = _classify_log_phase(
            "ERROR in Recording 5.wav: Audio file has no audio stream", "user")
        assert phase == "file"
        assert raw is False

    def test_retrying_is_file(self):
        phase, raw = _classify_log_phase("Retrying: Recording 5.wav", "user")
        assert phase == "file"
        assert raw is False

    def test_corruption_detected_is_file(self):
        phase, raw = _classify_log_phase(
            "Detected model corruption in Recording 5.wav, attempting recovery...", "user")
        assert phase == "file"
        assert raw is False

    def test_finished_is_results(self):
        phase, raw = _classify_log_phase(
            "Finished. Completed: 3. Failed: 0.", "user")
        assert phase == "results"
        assert raw is False

    def test_cancelled_is_results(self):
        phase, raw = _classify_log_phase(
            "Cancelled. Completed: 1. Failed: 0.", "user")
        assert phase == "results"
        assert raw is False

    def test_unknown_inherits_current_phase(self):
        phase, raw = _classify_log_phase("Some random message", "file")
        assert phase == "file"
        assert raw is False

    def test_user_action_preserves_user_phase(self):
        phase, raw = _classify_log_phase("Added 3 file(s).", "user")
        assert phase == "user"
        assert raw is False


class TestPathShortening:
    """TranscriberApp._shorten_log_message and _shorten_model_name."""

    def test_short_message_unchanged(self):
        assert TranscriberApp._shorten_log_message("Done: file.wav") == "Done: file.wav"

    def test_long_path_shortened_to_filename(self):
        if os.name == "nt":
            msg = "Transcribing: C:\\Users\\testuser\\Documents\\Recordings\\Recording 4.wav"
        else:
            msg = "Transcribing: /home/testuser/Documents/Recordings/Recording 4.wav"
        result = TranscriberApp._shorten_log_message(msg)
        assert "Recording 4.wav" in result
        assert "Transcribing:" in result
        # Should NOT contain the full path
        assert "Users" not in result and "home" not in result

    def test_model_path_shortened_to_friendly_name(self):
        msg = "Loading faster-whisper model: /home/user/.cache/huggingface/hub/models--Systran--faster-distil-whisper-large-v3/snapshots/abc123"
        result = TranscriberApp._shorten_log_message(msg)
        assert "Systran/faster-distil-whisper-large-v3" in result

    def test_shorten_model_name_hub_id(self):
        msg = "Loading faster-whisper model: Systran/faster-distil-whisper-large-v3"
        assert TranscriberApp._shorten_model_name(msg) == "Systran/faster-distil-whisper-large-v3"

    def test_shorten_model_name_cache_path(self):
        msg = "Loading faster-whisper model: /home/user/.cache/huggingface/hub/models--Systran--faster-distil-whisper-large-v3/snapshots/abc123"
        result = TranscriberApp._shorten_model_name(msg)
        assert result == "Systran/faster-distil-whisper-large-v3"

    def test_no_colon_returns_empty(self):
        assert TranscriberApp._shorten_model_name("no colon here") == ""


class TestEdgeCases:
    """Edge cases for phase classification and structured log."""

    def test_traceback_is_raw_only(self):
        phase, raw = _classify_log_phase(
            "Traceback (most recent call last):\n  File ...", "file")
        assert phase == "file"
        assert raw is True

    def test_model_cache_corrupt_is_file(self):
        phase, raw = _classify_log_phase(
            "Model cache appears corrupt. Deleting /path/to/cache...", "file")
        assert phase == "file"
        assert raw is False

    def test_model_reloaded_is_file(self):
        phase, raw = _classify_log_phase(
            "Model reloaded on device: cpu", "file")
        assert phase == "file"
        assert raw is False

    def test_recovery_failed_is_file(self):
        phase, raw = _classify_log_phase(
            "Recovery failed for Recording 5.wav: some error", "file")
        assert phase == "file"
        assert raw is False

    def test_run_separator_is_user(self):
        phase, raw = _classify_log_phase(
            "--- Run 2: 3 file(s), language=en, model=distil-large-v3 ---", "user")
        assert phase == "user"
        assert raw is False

    def test_pause_is_user(self):
        phase, raw = _classify_log_phase("Pause requested.", "user")
        assert phase == "user"
        assert raw is False

    def test_added_files_is_user(self):
        phase, raw = _classify_log_phase("Added 3 file(s).", "user")
        assert phase == "user"
        assert raw is False

    def test_cleared_is_user(self):
        phase, raw = _classify_log_phase("Cleared file list.", "user")
        assert phase == "user"
        assert raw is False

    def test_stopped_is_results(self):
        phase, raw = _classify_log_phase(
            "Stopped after current file. Completed: 2. Failed: 0.", "user")
        assert phase == "results"
        assert raw is False


# ---------------------------------------------------------------------------
# Level auto-detection in _record_log (LOG-02/08): only the message head
# decides the level, so filenames containing "error"/"failed" stay info.
# ---------------------------------------------------------------------------

class _RecordLogHarness:
    """Bare-minimum stand-in for TranscriberApp for _record_log."""

    def __init__(self):
        self._log_entries = []
        self._log_run_id = 0
        self._log_current_phase = "user"
        self._log_current_file = None


_RecordLogHarness._record_log = TranscriberApp._record_log


class TestLevelAutoDetection:
    def _level(self, message):
        return _RecordLogHarness()._record_log(message).level

    def test_error_head_detected(self):
        assert self._level("ERROR in a.mp3: boom") == "error"

    def test_failed_head_detected(self):
        assert self._level("Transcription failed: something broke") == "error"

    def test_traceback_detected(self):
        assert self._level("Traceback (most recent call last):\n  ...") == "error"

    def test_warning_head_detected(self):
        assert self._level("Warning: ffmpeg not found") == "warning"

    def test_filename_with_error_stays_info(self):
        assert self._level("Transcribing: error_analysis.wav") == "info"

    def test_filename_with_failed_stays_info(self):
        assert self._level("Done: interview_failed_mic.mp3") == "info"

    def test_renamed_output_with_error_stem_stays_info(self):
        assert self._level("Output renamed to 'error_take2.*' to avoid collision") == "info"

    def test_quoted_filename_with_colon_stays_info(self):
        # macOS filenames may contain ':' — quoted spans are stripped before
        # the colon split so the keyword inside the name can't leak into the head.
        assert self._level(
            "Output renamed to 'failed interview: part 2.*' to avoid collision") == "info"

    def test_explicit_level_not_overridden(self):
        h = _RecordLogHarness()
        assert h._record_log("Transcribing: x.wav", level="error").level == "error"

    def test_identified_speakers_is_info(self):
        # Diarization completion line must never render as warning/error.
        assert self._level("Identified 3 speakers") == "info"

    def test_diarize_degrade_warning_level(self):
        # The worker's degrade message: head-only matching flags "Warning"
        # while a filename containing "error" after the colon can't escalate.
        assert self._level(
            "Warning: speaker identification failed — keeping the "
            "transcript without speaker labels (error_take1.wav)") == "warning"

    def test_explicit_warning_level_from_event_passthrough(self):
        h = _RecordLogHarness()
        assert h._record_log("speaker labels lost",
                             level="warning").level == "warning"
