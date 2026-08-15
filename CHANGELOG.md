# Changelog

## 2.1.0 (2026-08-15)

- **Speaker diarization**: new "Identify speakers" checkbox labels each
  transcript line by speaker (e.g., `Speaker 1:`, and in Hebrew: `דובר 1:`)
  in txt/srt outputs; JSON carries raw speaker ids and display labels.
  Powered by pyannote.audio (speaker-diarization-community-1), fully offline —
  the Windows installer bundles the ~33 MB speaker model; source installs
  download it on first use. Advanced Settings gains a "Number of speakers"
  hint (0 = auto) and, on macOS, a processing-device choice (Apple Silicon
  GPU acceleration via MPS, or CPU).
- Word-level speaker attribution: segments that span a speaker change are
  split at the change (requires word timestamps; enabled automatically when
  diarization is on — expect roughly 10-30% slower transcription).
- Live preview refresh: once speaker identification finishes for a file, the
  Live preview pane is replaced with the final speaker-labeled transcript —
  exactly what the .txt output will contain. (During transcription the
  preview streams unlabeled text as before; speakers are computed only after
  the whole file is heard.)
- **Live recording**: transcribe straight from the
  microphone — speak, and each phrase appears in the app within a few
  seconds, also fully offline (voice activity detection segments the speech
  locally). Copy or save the text when you stop. Off by default: tick
  "Enable live recording (experimental)" in Advanced Settings to show
  the Record button.
- The Windows installer grows by roughly 160 MB (bundled PyTorch CPU
  runtime; 318 MB total, up from 155 MB). Source installs now require
  Python 3.10+ (3.9 is end-of-life).
- macOS: diarization-capable builds are Apple-Silicon-only (PyTorch no longer
  ships Intel-Mac wheels).
- Fixed: Windows taskbar progress overlay bug - the taskbar now shows batch progress.
- Fixed: "Cancel" pressed while paused no longer starts the next file before
  taking effect.
- Fixed: batch summary counts no longer mix per-run and whole-session totals
  on "Continue" runs; adding files mid-run is now blocked with a message.
- Fixed: progress bars can no longer briefly jump backwards; some tooltips and
  dialogs that had bugs (About, self-test report, Advanced Settings) stay on-screen near
  screen edges and no longer open with clipped buttons.
- Fixed: with two app instances running, log rotation no longer silently
  stops the file log.
- Visual refresh: modern checkboxes and flat control styling; aligned layout
  grid; readable live-transcript and preview typography; smarter queue table
  (content-fit columns, filename ellipsis, better time estimates,
  the active row is highlighted); correct right-to-left rendering of Hebrew
  punctuation and filenames everywhere (preview, dictation, queue, banner);
  the dictation panel widens automatically while recording and uses a red
  "recording" status language.

## 2.0.0 (2026-04-09)

- Initial public release. Fully redesigned engine and layout.

## 1.0.0

- Initial internal release.