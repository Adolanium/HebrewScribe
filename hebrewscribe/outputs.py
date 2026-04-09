"""Output file writers: txt, srt, json."""

import json
from pathlib import Path
from typing import List

from hebrewscribe.utils import format_timestamp


def write_txt(out_path: Path, text: str) -> None:
    out_path.write_text(text.strip() + "\n", encoding="utf-8")


def write_srt(out_path: Path, segments: List[dict]) -> None:
    lines = []
    for idx, seg in enumerate(segments, start=1):
        lines.append(str(idx))
        lines.append(f"{format_timestamp(seg['start'])} --> {format_timestamp(seg['end'])}")
        lines.append((seg.get("text") or "").strip())
        lines.append("")
    out_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def write_json(out_path: Path, source_file: str, backend: str, model: str,
               result: dict) -> None:
    """Write a full JSON transcript with metadata."""
    payload = {
        "source_file": source_file,
        "backend": backend,
        "model": model,
        "language": result.get("language"),
        "text": result.get("text", ""),
        "segments": result.get("segments", []),
        "meta": {k: v for k, v in result.items()
                 if k not in {"text", "segments", "raw"}},
    }
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
