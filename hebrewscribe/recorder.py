"""Live speech-to-text recorder: microphone capture, VAD, and transcription.

This module handles continuous recording with voice activity detection.
Audio is captured from the microphone, speech boundaries are detected
using Silero VAD, and each speech segment is transcribed using the
same faster-whisper backend as batch mode.

Communication with the GUI happens through an event callback.
"""

import collections
import logging
import queue
import threading
from typing import Callable, Optional

import numpy as np

from hebrewscribe.worker import cuda_available

logger = logging.getLogger("HebrewScribe")

# Audio constants
SAMPLE_RATE = 16000
CHUNK_SAMPLES = 512  # 32ms at 16kHz — required by Silero VAD
MAX_SPEECH_SECONDS = 30  # Force-transcribe if speech exceeds this
MIN_SPEECH_SECONDS = 0.3  # Ignore segments shorter than this


class LiveRecorder:
    """Continuous speech-to-text recorder using VAD + faster-whisper.

    Parameters
    ----------
    event_callback : callable(kind: str, **payload)
        Thread-safe callback to post events to the GUI.
        Event kinds: rec_text, rec_status, rec_error, rec_level, rec_stopped.
    language : str
        Language code (e.g., "he", "en").
    model_spec : str
        HuggingFace repo ID or local path for faster-whisper model.
    device : str
        "cuda", "cpu", or "auto".
    compute_type : str
        "float16", "int8", "auto", or "default".
    """

    def __init__(
        self,
        event_callback: Callable,
        language: str,
        model_spec: str,
        device: str = "auto",
        compute_type: str = "auto",
    ):
        self._event = event_callback
        self._language = language
        self._model_spec = model_spec
        self._device_pref = device
        self._compute_type_pref = compute_type

        self._model = None
        self._vad_model = None
        self._vad_iterator = None
        self._stream = None

        self._audio_queue: "queue.Queue[Optional[np.ndarray]]" = queue.Queue()
        self._transcribe_queue: "queue.Queue[Optional[np.ndarray]]" = queue.Queue()

        self._stop_flag = threading.Event()
        self._recording = False

        # "Recording — speak now" is only truthful once BOTH the mic is
        # capturing and the transcription model is loaded; whichever thread
        # finishes second posts it.
        self._mic_ready = threading.Event()
        self._model_ready = threading.Event()

        self._vad_thread: Optional[threading.Thread] = None
        self._transcribe_thread: Optional[threading.Thread] = None

    def is_recording(self) -> bool:
        return self._recording

    def start(self) -> None:
        """Start recording, VAD, and transcription threads."""
        if self._recording:
            return

        self._stop_flag.clear()
        self._mic_ready.clear()
        self._model_ready.clear()
        self._recording = True

        # Clear queues
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                break
        while not self._transcribe_queue.empty():
            try:
                self._transcribe_queue.get_nowait()
            except queue.Empty:
                break

        # Start threads in order: transcriber first (so it's ready), then VAD, then audio
        self._transcribe_thread = threading.Thread(
            target=self._transcribe_thread_func, daemon=True, name="rec-transcribe"
        )
        self._transcribe_thread.start()

        self._vad_thread = threading.Thread(
            target=self._vad_thread_func, daemon=True, name="rec-vad"
        )
        self._vad_thread.start()

        # Audio stream starts inside VAD thread after VAD model is loaded
        # (sounddevice callback pushes to _audio_queue)

    def stop(self, wait: bool = True) -> None:
        """Stop recording and shut down all threads.

        wait=False skips joining the worker threads — used on window close,
        where joining on the Tk main thread would freeze the closing GUI
        (the threads are daemons; process exit reaps them).
        """
        if not self._recording:
            return

        self._stop_flag.set()

        # Stop sounddevice stream
        self._close_stream()

        # Wake the VAD thread; it flushes the in-progress utterance in its
        # finally block and only THEN forwards the sentinel to the
        # transcriber. Putting a transcribe sentinel here would jump the
        # queue ahead of that flush and drop the user's last sentence.
        self._audio_queue.put(None)

        if wait:
            for t in (self._vad_thread, self._transcribe_thread):
                if t is not None and t.is_alive():
                    t.join(timeout=5.0)

        self._recording = False
        self._event(kind="rec_stopped")

    # ------------------------------------------------------------------
    # Audio capture (runs in sounddevice's own callback thread)
    # ------------------------------------------------------------------

    def _audio_callback(self, indata, frames, time_info, status):
        """Called by sounddevice for each audio chunk."""
        if status:
            logger.debug("sounddevice status: %s", status)
        # Copy data — sounddevice reuses the buffer
        self._audio_queue.put(indata[:, 0].copy())

    # ------------------------------------------------------------------
    # VAD thread: detect speech boundaries, accumulate speech buffer
    # ------------------------------------------------------------------

    def _vad_thread_func(self) -> None:
        try:
            self._event(kind="rec_status", message="Loading VAD model...")
            self._load_vad()
            if self._stop_flag.is_set():
                # Model load already failed (or Stop arrived) — never open
                # the microphone at all.
                self._transcribe_queue.put(None)
                return
            self._event(kind="rec_status", message="Starting microphone...")
            self._start_audio_stream()
            self._mic_ready.set()
            if self._model_ready.is_set():
                self._event(kind="rec_status", message="Recording — speak now")
            else:
                # The transcriber posts "speak now" once the model is loaded;
                # claiming it earlier invites dictation that is only heard,
                # not yet transcribed.
                self._event(kind="rec_status",
                            message="Preparing model — audio is being captured…")
        except Exception as e:
            logger.error("Recording setup failed: %s", e, exc_info=True)
            self._event(kind="rec_error", message=str(e))
            self._stop_flag.set()
            self._close_stream()
            self._transcribe_queue.put(None)
            return

        speech_buffer = []
        speech_active = False
        speech_samples = 0
        # Rolling pre-speech context: Silero reports 'start' only after its
        # trigger window, so without a pre-roll the first ~0.2 s of every
        # utterance was cut off (speech_pad_ms cannot restore audio that was
        # never buffered). 8 chunks x 32 ms = 256 ms.
        preroll = collections.deque(maxlen=8)

        try:
            while not self._stop_flag.is_set():
                try:
                    chunk = self._audio_queue.get(timeout=0.5)
                except queue.Empty:
                    continue

                if chunk is None:
                    break

                # Compute RMS for level meter
                rms = float(np.sqrt(np.mean(chunk ** 2)))
                # Convert to 0-1 range (speech typically 0.01-0.3 RMS)
                level = min(1.0, rms / 0.15)
                self._event(kind="rec_level", level=level)

                preroll.append(chunk)

                # Feed chunk to VAD
                import torch
                chunk_tensor = torch.from_numpy(chunk).float()
                result = self._vad_iterator(chunk_tensor, return_seconds=False)

                if result is not None:
                    if "start" in result:
                        speech_active = True
                        # Seed with the pre-roll (minus the current chunk —
                        # the append below adds it) so the first syllables
                        # of the utterance are not clipped.
                        speech_buffer = list(preroll)[:-1]
                        speech_samples = sum(len(c) for c in speech_buffer)
                    elif "end" in result:
                        if speech_active and speech_samples >= int(MIN_SPEECH_SECONDS * SAMPLE_RATE):
                            # Speech ended — send buffer for transcription
                            full_audio = np.concatenate(speech_buffer)
                            self._transcribe_queue.put(full_audio)
                        speech_active = False
                        speech_buffer.clear()
                        speech_samples = 0

                if speech_active:
                    speech_buffer.append(chunk)
                    speech_samples += len(chunk)

                    # Force-transcribe if speech is too long without a pause
                    if speech_samples >= int(MAX_SPEECH_SECONDS * SAMPLE_RATE):
                        full_audio = np.concatenate(speech_buffer)
                        self._transcribe_queue.put(full_audio)
                        speech_buffer.clear()
                        speech_samples = 0
                        self._vad_iterator.reset_states()

        except Exception as e:
            logger.error("VAD thread error: %s", e, exc_info=True)
            self._event(kind="rec_error", message=f"VAD error: {e}")
        finally:
            # The mic must never outlive this thread: when the transcriber's
            # model load fails it only sets stop_flag — without this close,
            # the stream stayed open and the audio queue grew unbounded.
            self._close_stream()
            # Flush remaining speech buffer on stop
            if speech_active and speech_buffer and speech_samples >= int(MIN_SPEECH_SECONDS * SAMPLE_RATE):
                full_audio = np.concatenate(speech_buffer)
                self._transcribe_queue.put(full_audio)
            # Signal transcriber to finish
            self._transcribe_queue.put(None)

    def _load_vad(self) -> None:
        """Load Silero VAD model."""
        from silero_vad import load_silero_vad, VADIterator
        self._vad_model = load_silero_vad()
        self._vad_iterator = VADIterator(
            self._vad_model,
            threshold=0.5,
            sampling_rate=SAMPLE_RATE,
            min_silence_duration_ms=500,
            speech_pad_ms=100,
        )

    def _close_stream(self) -> None:
        """Stop and close the mic stream; tolerates concurrent closers."""
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                logger.debug("Error closing audio stream", exc_info=True)

    def _start_audio_stream(self) -> None:
        """Start the sounddevice input stream."""
        import sounddevice as sd
        try:
            self._stream = sd.InputStream(
                channels=1,
                samplerate=SAMPLE_RATE,
                dtype=np.float32,
                blocksize=CHUNK_SAMPLES,
                callback=self._audio_callback,
            )
            self._stream.start()
        except sd.PortAudioError as e:
            raise RuntimeError(
                f"Could not open microphone: {e}\n\n"
                "Check that a microphone is connected and that HebrewScribe "
                "has microphone permission in System Settings."
            ) from e

    # ------------------------------------------------------------------
    # Transcription thread: transcribe speech segments
    # ------------------------------------------------------------------

    def _transcribe_thread_func(self) -> None:
        try:
            self._event(kind="rec_status", message="Loading transcription model...")
            self._load_transcription_model()
        except Exception as e:
            logger.error("Model loading failed: %s", e, exc_info=True)
            self._event(kind="rec_error", message=f"Model loading failed: {e}")
            self._stop_flag.set()
            return

        self._model_ready.set()
        if self._mic_ready.is_set() and not self._stop_flag.is_set():
            self._event(kind="rec_status", message="Recording — speak now")

        try:
            # Drain until the sentinel — never exit on stop_flag: segments
            # queued (or flushed by the VAD thread's finally) before Stop
            # must still be transcribed, or the last sentence is lost.
            while True:
                try:
                    audio_buffer = self._transcribe_queue.get(timeout=1.0)
                except queue.Empty:
                    continue

                if audio_buffer is None:
                    break

                # Transcribe the speech segment
                self._event(kind="rec_status", message="Transcribing...")
                try:
                    language = None if self._language == "auto" else self._language
                    segments_iter, info = self._model.transcribe(
                        audio_buffer,
                        language=language,
                        beam_size=1,
                        vad_filter=False,  # VAD already applied
                        condition_on_previous_text=False,
                    )
                    texts = []
                    for seg in segments_iter:
                        text = seg.text.strip()
                        if text:
                            texts.append(text)

                    full_text = " ".join(texts)
                    if full_text:
                        self._event(kind="rec_text", text=full_text, language=self._language)

                except Exception as e:
                    logger.error("Transcription error: %s", e, exc_info=True)
                    self._event(kind="rec_error", message=f"Transcription error: {e}")
                else:
                    # Only after a successful segment — an error status must
                    # stay visible until the next segment succeeds.
                    if self._recording and not self._stop_flag.is_set():
                        self._event(kind="rec_status", message="Recording — speak now")

        except Exception as e:
            logger.error("Transcribe thread error: %s", e, exc_info=True)
            self._event(kind="rec_error", message=f"Transcription thread error: {e}")

    def _load_transcription_model(self) -> None:
        """Load faster-whisper model for transcription."""
        import faster_whisper

        device = self._device_pref
        if device == "auto":
            device = "cuda" if cuda_available() else "cpu"

        compute_type = self._compute_type_pref
        if compute_type == "auto":
            compute_type = "float16" if device == "cuda" else "int8"

        # Pre-download if hub ID
        model_path = self._model_spec
        _spec = self._model_spec
        from pathlib import Path
        _is_hub_id = (
            "/" in _spec
            and not Path(_spec).is_absolute()
            and _spec.count("/") == 1
            and not Path(_spec).exists()
        )
        if _is_hub_id:
            from huggingface_hub import snapshot_download
            self._event(kind="rec_status", message=f"Downloading model: {_spec}...")
            model_path = snapshot_download(repo_id=_spec)

        logger.info("Recording: loading model %s (device=%s, compute=%s)",
                     self._model_spec, device, compute_type)
        self._model = faster_whisper.WhisperModel(
            model_path, device=device, compute_type=compute_type
        )
