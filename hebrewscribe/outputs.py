"""Output file writers: txt, srt, json."""

import json
from pathlib import Path
from typing import List, Optional

from hebrewscribe.utils import format_timestamp


def format_speaker_transcript(segments: List[dict], speakers: dict) -> str:
    """Render segments as speaker-labeled, turn-grouped paragraphs.

    Consecutive same-speaker segments group into one paragraph:

        דובר 1:
        <text of the speaker's turn>

        דובר 2:
        ...

    Segments whose speaker is None form unlabeled paragraphs — text is never
    dropped. This is the single canonical rendering: write_txt writes it and
    the GUI's post-diarization preview refresh shows it, so what the user
    previews is exactly what lands in the .txt file.
    """
    paragraphs = []
    group_speaker = object()  # sentinel != any real value on first iteration
    for seg in segments:
        seg_text = (seg.get("text") or "").strip()
        if not seg_text:
            continue
        speaker = seg.get("speaker")
        if speaker == group_speaker and paragraphs:
            paragraphs[-1][1].append(seg_text)
        else:
            paragraphs.append((speaker, [seg_text]))
            group_speaker = speaker
    lines = []
    for speaker, texts in paragraphs:
        body = " ".join(texts)
        label = speakers.get(speaker) if speaker is not None else None
        if label:
            lines.append(f"{label}:\n{body}")
        else:
            lines.append(body)
    return "\n\n".join(lines).strip()


def write_txt(out_path: Path, text: str,
              segments: Optional[List[dict]] = None,
              speakers: Optional[dict] = None) -> None:
    """Write the plain-text transcript.

    Without speaker data the output is exactly the historical format (the
    joined transcript text). With speakers, the turn-grouped rendering of
    format_speaker_transcript().
    """
    if not speakers or not segments:
        out_path.write_text(text.strip() + "\n", encoding="utf-8")
        return
    out_path.write_text(
        format_speaker_transcript(segments, speakers) + "\n",
        encoding="utf-8")


def write_srt(out_path: Path, segments: List[dict],
              speakers: Optional[dict] = None) -> None:
    """Write an SRT subtitle file.

    With speaker data, each cue's text line is prefixed with the display
    label as plain text ("דובר 1: שלום"). Never emits WebVTT <v> voice tags —
    they are not valid SRT and players handle them inconsistently.
    """
    lines = []
    for idx, seg in enumerate(segments, start=1):
        text = (seg.get("text") or "").strip()
        if speakers:
            label = speakers.get(seg.get("speaker"))
            if label:
                text = f"{label}: {text}"
        lines.append(str(idx))
        lines.append(f"{format_timestamp(seg['start'])} --> {format_timestamp(seg['end'])}")
        lines.append(text)
        lines.append("")
    out_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def write_json(out_path: Path, source_file: str, backend: str, model: str,
               result: dict) -> None:
    """Write a full JSON transcript with metadata.

    When diarization ran, each segment carries the raw engine speaker id in
    "speaker" and the display label in "speaker_label", and the raw-id → label
    map lands in meta["speakers"] (via the generic meta sweep). Without
    diarization the payload is byte-identical to the historical format.
    """
    segments = result.get("segments", [])
    speakers = result.get("speakers")
    if speakers:
        enriched = []
        for seg in segments:
            seg = dict(seg)
            if seg.get("speaker") is not None:
                label = speakers.get(seg["speaker"])
                if label:
                    seg["speaker_label"] = label
            enriched.append(seg)
        segments = enriched
    payload = {
        "source_file": source_file,
        "backend": backend,
        "model": model,
        "language": result.get("language"),
        "text": result.get("text", ""),
        "segments": segments,
        "meta": {k: v for k, v in result.items()
                 if k not in {"text", "segments", "raw"}},
    }
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
