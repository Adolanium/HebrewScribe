"""GUI-slice tests for the diarization feature: config keys, run options,
readiness gating, device display mapping, and event level passthrough.

Headless: harness classes carry only the attributes each unbound
TranscriberApp method reads (established pattern of test_app_incremental.py).
"""

import pytest

import hebrewscribe.app as app_mod
from hebrewscribe.app import TranscriberApp
from hebrewscribe.utils import load_json, save_json


class _FakeVar:
    def __init__(self, value=""):
        self._value = value

    def get(self):
        return self._value

    def set(self, value):
        self._value = value


def _run_options_harness(**overrides):
    h = type("RunOptsHarness", (), {})()
    h.format_vars = {"txt": _FakeVar(True), "srt": _FakeVar(False),
                     "json": _FakeVar(False)}
    h.beam_size_var = _FakeVar("5")
    h.batch_size_var = _FakeVar("0")
    h.num_speakers_var = _FakeVar("0")
    h.backend_var = _FakeVar("faster-whisper")
    h.output_dir_var = _FakeVar("/tmp/out")
    h.language_var = _FakeVar("he")
    h.task_var = _FakeVar("transcribe")
    h.device_var = _FakeVar("cpu")
    h.compute_type_var = _FakeVar("int8")
    h.vad_var = _FakeVar(True)
    h.condition_on_previous_text_var = _FakeVar(False)
    h.diarize_var = _FakeVar(False)
    h.diarize_device_var = _FakeVar("auto")
    h.get_selected_model_spec = lambda: "tiny"
    for name, value in overrides.items():
        setattr(h, name, value)
    return h


class TestGetRunOptions:
    def _opts(self, harness):
        return TranscriberApp.get_run_options(harness)

    def test_defaults_off(self):
        opts = self._opts(_run_options_harness())
        assert opts.diarize is False
        assert opts.num_speakers == 0
        assert opts.diarize_device == "auto"

    def test_exact_count_passthrough(self):
        h = _run_options_harness(diarize_var=_FakeVar(True),
                                 num_speakers_var=_FakeVar("3"),
                                 diarize_device_var=_FakeVar("cpu"))
        opts = self._opts(h)
        assert opts.diarize is True
        assert opts.num_speakers == 3
        assert opts.diarize_device == "cpu"

    def test_zero_is_auto_not_error(self):
        opts = self._opts(_run_options_harness(
            diarize_var=_FakeVar(True), num_speakers_var=_FakeVar("0")))
        assert opts.num_speakers == 0

    def test_non_int_raises(self):
        with pytest.raises(ValueError, match="Number of speakers"):
            self._opts(_run_options_harness(num_speakers_var=_FakeVar("abc")))

    def test_negative_raises(self):
        with pytest.raises(ValueError, match=">= 0"):
            self._opts(_run_options_harness(num_speakers_var=_FakeVar("-1")))


class _ReadinessHarness:
    def __init__(self):
        self.worker_thread = None
        self.file_paths = ["/a/x.wav"]
        self.model_var = _FakeVar("Tiny")
        self.model_map = {"Tiny": "tiny"}
        self.output_dir_var = _FakeVar("/tmp/out")
        self.format_vars = {"txt": _FakeVar(True)}
        self.beam_size_var = _FakeVar("5")
        self.batch_size_var = _FakeVar("0")
        self.diarize_var = _FakeVar(False)
        self.num_speakers_var = _FakeVar("0")
        self.readiness_var = _FakeVar("")
        self.readiness_label = _FakeLabel()
        self.start_button = object()
        self.enabled_calls = []

    def _set_button_enabled(self, btn, enabled):
        self.enabled_calls.append(enabled)


class _FakeLabel:
    def __init__(self):
        self.configured = {}

    def configure(self, **kw):
        self.configured.update(kw)


_ReadinessHarness._check_readiness = TranscriberApp._check_readiness


class TestReadiness:
    def test_clean_when_diarize_off(self):
        h = _ReadinessHarness()
        h._check_readiness()
        assert h.readiness_var.get() == ""
        assert h.enabled_calls[-1] is True

    def test_issue_when_pyannote_missing_and_on(self, monkeypatch):
        monkeypatch.setattr(app_mod, "_HAS_PYANNOTE", False)
        h = _ReadinessHarness()
        h.diarize_var = _FakeVar(True)
        h._check_readiness()
        assert "pyannote" in h.readiness_var.get()
        assert h.enabled_calls[-1] is False

    def test_clean_when_pyannote_present_and_on(self, monkeypatch):
        monkeypatch.setattr(app_mod, "_HAS_PYANNOTE", True)
        h = _ReadinessHarness()
        h.diarize_var = _FakeVar(True)
        h._check_readiness()
        assert h.readiness_var.get() == ""

    def test_num_speakers_validation(self):
        h = _ReadinessHarness()
        h.num_speakers_var = _FakeVar("x")
        h._check_readiness()
        assert "must be an integer" in h.readiness_var.get()
        h2 = _ReadinessHarness()
        h2.num_speakers_var = _FakeVar("-2")
        h2._check_readiness()
        assert ">= 0" in h2.readiness_var.get()
        h3 = _ReadinessHarness()
        h3.num_speakers_var = _FakeVar("5")
        h3._check_readiness()
        assert h3.readiness_var.get() == ""


class TestInitialGating:
    def test_stale_saved_true_forced_off_when_engine_absent(self):
        assert TranscriberApp._diarize_enabled_initial(True, False) is False

    def test_saved_true_kept_when_engine_present(self):
        assert TranscriberApp._diarize_enabled_initial(True, True) is True

    def test_saved_false_stays_false(self):
        assert TranscriberApp._diarize_enabled_initial(False, True) is False

    def test_non_bool_config_coerced(self):
        assert TranscriberApp._diarize_enabled_initial("yes", True) is True
        assert TranscriberApp._diarize_enabled_initial(0, True) is False


class _ConfigHarness:
    def __init__(self):
        self.config_data = {}
        self.output_dir_var = _FakeVar("/tmp/out")
        self.backend_var = _FakeVar("faster-whisper")
        self.language_var = _FakeVar("he")
        self.model_var = _FakeVar("Tiny")
        self.task_var = _FakeVar("transcribe")
        self.device_var = _FakeVar("auto")
        self.compute_type_var = _FakeVar("auto")
        self.beam_size_var = _FakeVar("5")
        self.vad_var = _FakeVar(True)
        self.condition_on_previous_text_var = _FakeVar(False)
        self.batch_size_var = _FakeVar("0")
        self.speed_preset_var = _FakeVar("quality")
        self.format_vars = {"txt": _FakeVar(True)}
        self.prevent_sleep_var = _FakeVar(True)
        self._experimental_recording_var = _FakeVar(False)
        self.diarize_var = _FakeVar(False)
        self.num_speakers_var = _FakeVar("0")
        self.diarize_device_var = _FakeVar("auto")
        self.saved = False

    def save_config(self):
        self.saved = True


_ConfigHarness.save_config_from_ui = TranscriberApp.save_config_from_ui


class TestConfigPersistence:
    def test_round_trip_values(self):
        h = _ConfigHarness()
        h.diarize_var = _FakeVar(True)
        h.num_speakers_var = _FakeVar("4")
        h.diarize_device_var = _FakeVar("cpu")
        h.save_config_from_ui()
        assert h.config_data["last_diarize"] is True
        assert h.config_data["last_num_speakers"] == 4
        assert h.config_data["diarize_device"] == "cpu"
        assert h.saved

    def test_bad_num_speakers_keeps_previous(self):
        h = _ConfigHarness()
        h.config_data["last_num_speakers"] = 2
        h.num_speakers_var = _FakeVar("garbage")
        h.save_config_from_ui()
        assert h.config_data["last_num_speakers"] == 2

    def test_disk_round_trip_types(self, tmp_path):
        p = tmp_path / "cfg.json"
        save_json(p, {"last_diarize": True, "last_num_speakers": 3,
                      "diarize_device": "cpu"})
        loaded = load_json(p, {})
        assert loaded["last_diarize"] is True
        assert loaded["last_num_speakers"] == 3
        assert loaded["diarize_device"] == "cpu"


class TestDeviceDisplayMapping:
    def test_display_to_code(self):
        h = type("H", (), {})()
        h._DIARIZE_DEVICE_DISPLAY = TranscriberApp._DIARIZE_DEVICE_DISPLAY
        h._diarize_device_display_var = _FakeVar("CPU")
        h.diarize_device_var = _FakeVar("auto")
        TranscriberApp._on_diarize_device_display_changed(h)
        assert h.diarize_device_var.get() == "cpu"
        h._diarize_device_display_var = _FakeVar("Auto (MPS)")
        TranscriberApp._on_diarize_device_display_changed(h)
        assert h.diarize_device_var.get() == "auto"


class TestPresetOrthogonality:
    def test_detect_preset_ignores_diarize_vars(self):
        h = type("H", (), {"SPEED_PRESETS": TranscriberApp.SPEED_PRESETS})()
        h.beam_size_var = _FakeVar("5")
        h.vad_var = _FakeVar(True)
        h.condition_on_previous_text_var = _FakeVar(True)
        h.batch_size_var = _FakeVar("0")
        h.diarize_var = _FakeVar(True)          # must be invisible to presets
        h.num_speakers_var = _FakeVar("7")
        assert TranscriberApp._detect_preset_from_values(h) == "quality"


class _FakeText:
    """Records the tk.Text calls the preview handlers make."""

    def __init__(self):
        self.calls = []
        self.content = ""

    def delete(self, start, end):
        self.calls.append(("delete", start, end))
        self.content = ""

    def insert(self, index, text, tag=None):
        self.calls.append(("insert", index, text, tag))
        self.content += text

    def see(self, index):
        self.calls.append(("see", index))


class TestPreviewReplaceHandler:
    class _Harness:
        _get_preview_text_tag = TranscriberApp._get_preview_text_tag
        _bidi_display = staticmethod(TranscriberApp._bidi_display)

        def __init__(self, lang="he"):
            self.language_var = _FakeVar(lang)
            self.preview_text = _FakeText()
            self.preview_text.content = "streamed line 1\nstreamed line 2\n"

    def test_replaces_streamed_text_with_labeled(self):
        h = self._Harness()
        TranscriberApp._handle_event(
            h, "preview_replace",
            {"text": "דובר 1:\nשלום", "language": "he"})
        # RTL display wraps each line in invisible RLM marks so trailing
        # punctuation renders on the correct (left) side.
        rlm = chr(0x200F)
        assert h.preview_text.content == (
            f"{rlm}דובר 1:{rlm}\n{rlm}שלום{rlm}\n")
        ops = [c[0] for c in h.preview_text.calls]
        assert ops == ["delete", "insert", "see"]
        # Hebrew UI language → rtl tag on the inserted text
        assert h.preview_text.calls[1][3] == "rtl"

    def test_empty_text_leaves_preview_untouched(self):
        h = self._Harness()
        before = h.preview_text.content
        TranscriberApp._handle_event(h, "preview_replace", {"text": "  "})
        assert h.preview_text.content == before
        assert h.preview_text.calls == []

    def test_ltr_language_gets_ltr_tag(self):
        h = self._Harness(lang="en")
        TranscriberApp._handle_event(
            h, "preview_replace", {"text": "Speaker 1:\nhello"})
        assert h.preview_text.calls[1][3] == "ltr"


class TestLogEventLevelPassthrough:
    class _Harness:
        def __init__(self):
            self.calls = []

        def log(self, message, level="info"):
            self.calls.append((message, level))

    def test_level_forwarded(self):
        h = self._Harness()
        TranscriberApp._handle_event(h, "log",
                                     {"message": "m", "level": "warning"})
        assert h.calls == [("m", "warning")]

    def test_missing_level_defaults_info(self):
        h = self._Harness()
        TranscriberApp._handle_event(h, "log", {"message": "m"})
        assert h.calls == [("m", "info")]
