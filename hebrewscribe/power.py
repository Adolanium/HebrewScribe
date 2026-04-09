"""Cross-platform sleep prevention.

Windows: SetThreadExecutionState via ctypes (kernel32).
macOS:   caffeinate subprocess (ships with every macOS since 10.8).
Linux:   systemd-inhibit or no-op.

All failures are logged and swallowed — sleep prevention is best-effort.
"""

import logging
import subprocess
import sys

logger = logging.getLogger("HebrewScribe")

# Windows constants
ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000002


class SleepInhibitor:
    """Prevents the OS from sleeping while acquired.

    Usage::

        inhibitor = SleepInhibitor()
        inhibitor.acquire()   # prevent sleep
        # ... long-running work ...
        inhibitor.release()   # allow sleep again

    Idempotent: calling acquire() twice refreshes the hold,
    calling release() when not acquired is a no-op.
    """

    def __init__(self) -> None:
        self._active = False
        self._process: "subprocess.Popen[bytes] | None" = None

    @property
    def active(self) -> bool:
        """Whether sleep prevention is currently held."""
        return self._active

    def acquire(self) -> bool:
        """Prevent the OS from sleeping. Returns True on success, False on failure."""
        if self._active:
            return True
        try:
            if sys.platform == "win32":
                self._acquire_windows()
            elif sys.platform == "darwin":
                self._acquire_macos()
            else:
                self._acquire_linux()
            self._active = True
            logger.info("Sleep prevention acquired")
            return True
        except Exception:
            logger.warning("Failed to acquire sleep prevention", exc_info=True)
            return False

    def release(self) -> None:
        """Allow the OS to sleep again."""
        if not self._active:
            return
        try:
            if sys.platform == "win32":
                self._release_windows()
            elif sys.platform == "darwin":
                self._release_macos()
            else:
                self._release_linux()
            logger.info("Sleep prevention released")
        except Exception:
            logger.warning("Failed to release sleep prevention", exc_info=True)
        finally:
            self._active = False
            self._process = None

    # --- Windows -------------------------------------------------------

    def _acquire_windows(self) -> None:
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(
            ES_CONTINUOUS | ES_SYSTEM_REQUIRED
        )

    def _release_windows(self) -> None:
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)

    # --- macOS ---------------------------------------------------------

    def _acquire_macos(self) -> None:
        # caffeinate -i: prevent idle sleep; dies when parent kills it
        self._process = subprocess.Popen(
            ["caffeinate", "-i"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def _release_macos(self) -> None:
        if self._process is not None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()

    # --- Linux ---------------------------------------------------------

    def _acquire_linux(self) -> None:
        # systemd-inhibit: prevent idle sleep; dies when child exits
        # We use 'sleep infinity' as the child — terminated on release
        try:
            self._process = subprocess.Popen(
                [
                    "systemd-inhibit",
                    "--what=idle",
                    "--who=HebrewScribe",
                    "--why=Transcription in progress",
                    "sleep", "infinity",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            logger.debug("systemd-inhibit not available, skipping")

    def _release_linux(self) -> None:
        if self._process is not None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
