"""Tests for hebrewscribe.power.SleepInhibitor."""

import subprocess
import sys
import unittest
from unittest.mock import MagicMock, patch

from hebrewscribe.power import SleepInhibitor, ES_CONTINUOUS, ES_SYSTEM_REQUIRED


class TestSleepInhibitorIdempotency(unittest.TestCase):
    """Platform-independent behavioural tests."""

    def test_release_without_acquire_is_noop(self):
        inhibitor = SleepInhibitor()
        inhibitor.release()  # should not raise
        self.assertFalse(inhibitor.active)

    def test_double_release_is_noop(self):
        inhibitor = SleepInhibitor()
        inhibitor._active = True  # fake an active state
        # First release clears it
        with patch.object(inhibitor, f"_release_{'windows' if sys.platform == 'win32' else 'macos' if sys.platform == 'darwin' else 'linux'}"):
            inhibitor.release()
        self.assertFalse(inhibitor.active)
        # Second release is a no-op
        inhibitor.release()
        self.assertFalse(inhibitor.active)

    def test_acquire_sets_active(self):
        inhibitor = SleepInhibitor()
        method = f"_acquire_{'windows' if sys.platform == 'win32' else 'macos' if sys.platform == 'darwin' else 'linux'}"
        with patch.object(inhibitor, method):
            inhibitor.acquire()
        self.assertTrue(inhibitor.active)

    def test_double_acquire_skips_second(self):
        inhibitor = SleepInhibitor()
        method = f"_acquire_{'windows' if sys.platform == 'win32' else 'macos' if sys.platform == 'darwin' else 'linux'}"
        with patch.object(inhibitor, method) as mock_acq:
            inhibitor.acquire()
            inhibitor.acquire()
        mock_acq.assert_called_once()

    def test_acquire_failure_does_not_set_active(self):
        inhibitor = SleepInhibitor()
        method = f"_acquire_{'windows' if sys.platform == 'win32' else 'macos' if sys.platform == 'darwin' else 'linux'}"
        with patch.object(inhibitor, method, side_effect=OSError("fail")):
            inhibitor.acquire()
        self.assertFalse(inhibitor.active)


class TestSleepInhibitorWindows(unittest.TestCase):
    """Test Windows codepath with mocked ctypes."""

    @patch("hebrewscribe.power.sys")
    def test_acquire_calls_set_execution_state(self, mock_sys):
        mock_sys.platform = "win32"
        inhibitor = SleepInhibitor()
        mock_kernel32 = MagicMock()
        with patch("ctypes.windll", create=True) as mock_windll:
            mock_windll.kernel32 = mock_kernel32
            inhibitor._acquire_windows()
        mock_kernel32.SetThreadExecutionState.assert_called_once_with(
            ES_CONTINUOUS | ES_SYSTEM_REQUIRED
        )

    def test_release_calls_continuous_only(self):
        inhibitor = SleepInhibitor()
        mock_kernel32 = MagicMock()
        with patch("ctypes.windll", create=True) as mock_windll:
            mock_windll.kernel32 = mock_kernel32
            inhibitor._release_windows()
        mock_kernel32.SetThreadExecutionState.assert_called_once_with(ES_CONTINUOUS)


class TestSleepInhibitorMacOS(unittest.TestCase):
    """Test macOS codepath with mocked subprocess."""

    def test_acquire_starts_caffeinate(self):
        inhibitor = SleepInhibitor()
        mock_proc = MagicMock()
        with patch("hebrewscribe.power.subprocess.Popen", return_value=mock_proc) as mock_popen:
            inhibitor._acquire_macos()
        mock_popen.assert_called_once_with(
            ["caffeinate", "-i"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self.assertIs(inhibitor._process, mock_proc)

    def test_release_terminates_caffeinate(self):
        inhibitor = SleepInhibitor()
        mock_proc = MagicMock()
        inhibitor._process = mock_proc
        inhibitor._release_macos()
        mock_proc.terminate.assert_called_once()
        mock_proc.wait.assert_called_once_with(timeout=5)

    def test_release_kills_if_timeout(self):
        inhibitor = SleepInhibitor()
        mock_proc = MagicMock()
        mock_proc.wait.side_effect = subprocess.TimeoutExpired(cmd="caffeinate", timeout=5)
        inhibitor._process = mock_proc
        inhibitor._release_macos()
        mock_proc.kill.assert_called_once()

    def test_release_no_process_is_safe(self):
        inhibitor = SleepInhibitor()
        inhibitor._process = None
        inhibitor._release_macos()  # should not raise


class TestSleepInhibitorLinux(unittest.TestCase):
    """Test Linux codepath with mocked subprocess."""

    def test_acquire_starts_systemd_inhibit(self):
        inhibitor = SleepInhibitor()
        mock_proc = MagicMock()
        with patch("hebrewscribe.power.subprocess.Popen", return_value=mock_proc) as mock_popen:
            inhibitor._acquire_linux()
        mock_popen.assert_called_once()
        args = mock_popen.call_args[0][0]
        self.assertEqual(args[0], "systemd-inhibit")
        self.assertIs(inhibitor._process, mock_proc)

    def test_acquire_handles_missing_systemd_inhibit(self):
        inhibitor = SleepInhibitor()
        with patch("hebrewscribe.power.subprocess.Popen", side_effect=FileNotFoundError):
            inhibitor._acquire_linux()  # should not raise
        self.assertIsNone(inhibitor._process)


if __name__ == "__main__":
    unittest.main()
