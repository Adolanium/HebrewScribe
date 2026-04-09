"""Model detection: find locally cached Whisper models for both backends."""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("HebrewScribe")

OPENAI_MODELS = [
    "tiny.en", "tiny",
    "base.en", "base",
    "small.en", "small",
    "medium.en", "medium",
    "large", "turbo",
]

# Curated repo IDs offered for download.
# Order: Hebrew-specific, English-specific, multilingual.
FASTER_WHISPER_MODELS = [
    "ivrit-ai/whisper-large-v3-ct2",
    "Systran/faster-distil-whisper-large-v3",
    "Systran/faster-whisper-large-v3",
]

# ---------------------------------------------------------------------------
# Language affinity: which language each known model is best for.
# "he" = Hebrew, "en" = English, None = multilingual / unknown.
# Keys match against: repo slug (ivrit-ai/whisper-large-v3-ct2), directory
# name (whisper-large-v3-ct2), or openai model name (tiny.en).
# ---------------------------------------------------------------------------

KNOWN_MODELS: Dict[str, dict] = {
    # --- Hebrew ---
    "ivrit-ai/whisper-large-v3-ct2": {
        "friendly": "ivrit-ai large-v3",
        "lang": "he",
    },
    "ivrit-ai/whisper-large-v2-ct2": {
        "friendly": "ivrit-ai large-v2",
        "lang": "he",
    },
    # directory-name variants that appear as [local]
    "whisper-large-v3-ct2": {
        "friendly": "ivrit-ai large-v3",
        "lang": "he",
    },
    "whisper-large-v2-ct2": {
        "friendly": "ivrit-ai large-v2",
        "lang": "he",
    },
    # --- English (distil = English-only by design) ---
    "Systran/faster-distil-whisper-large-v3": {
        "friendly": "distil-large-v3",
        "lang": "en",
    },
    "faster-distil-whisper-large-v3": {
        "friendly": "distil-large-v3",
        "lang": "en",
    },
    "Systran/faster-whisper-large-v3": {
        "friendly": "whisper large-v3",
        "lang": None,
    },
    "faster-whisper-large-v3": {
        "friendly": "whisper large-v3",
        "lang": None,
    },
    "Systran/faster-distil-whisper-large-v2": {
        "friendly": "distil-large-v2",
        "lang": "en",
    },
    "Systran/faster-distil-whisper-medium.en": {
        "friendly": "distil-medium.en",
        "lang": "en",
    },
}

# Recommended model per language (used for auto-selection).
# Values are substrings matched against the model label.
RECOMMENDED_MODEL: Dict[str, str] = {
    "he": "ivrit-ai large-v3",
    "en": "distil-large-v3",
    "ru": "whisper large-v3",
    "fr": "whisper large-v3",
    "pl": "whisper large-v3",
    "zh": "whisper large-v3",
}

# Approximate model sizes in GB (for display).  Key = friendly name or raw name.
MODEL_SIZES: Dict[str, str] = {
    "ivrit-ai large-v3": "3.1 GB",
    "ivrit-ai large-v2": "3.1 GB",
    "distil-large-v3": "1.5 GB",
    "distil-large-v2": "1.5 GB",
    "distil-medium.en": "0.8 GB",
    "whisper large-v3": "3.1 GB",
    # openai-whisper models
    "tiny.en": "0.07 GB",
    "tiny": "0.07 GB",
    "base.en": "0.14 GB",
    "base": "0.14 GB",
    "small.en": "0.46 GB",
    "small": "0.46 GB",
    "medium.en": "1.4 GB",
    "medium": "1.4 GB",
    "large": "3.1 GB",
    "turbo": "1.6 GB",
}


_LANG_NAMES: Dict[str, str] = {
    "he": "Hebrew",
    "en": "English",
    "ru": "Russian",
    "fr": "French",
    "pl": "Polish",
    "zh": "Chinese",
}

def _lang_tag(lang: Optional[str]) -> str:
    """Return a language tag for display."""
    return _LANG_NAMES.get(lang, "Multilingual") if lang else "Multilingual"


def _infer_lang(name: str) -> Optional[str]:
    """Infer language affinity from a model name string using heuristics."""
    low = name.lower()
    if "ivrit" in low:
        return "he"
    if low.endswith(".en") or "distil" in low:
        return "en"
    return None


def _size_suffix(friendly_name: str) -> str:
    """Return ' · 3.1 GB' if size is known, else empty string."""
    size = MODEL_SIZES.get(friendly_name, "")
    return f" \u00b7 {size}" if size else ""


def _recommended_prefix(friendly_name: str) -> str:
    """Return a star prefix if this model is recommended for any language."""
    for _lang, rec_name in RECOMMENDED_MODEL.items():
        if rec_name == friendly_name:
            return "\u2605 "  # filled star
    return ""


def friendly_label(raw_name: str, source_tag: str = "local") -> str:
    """Build a human-readable label for *raw_name* (repo slug or dir name).

    Returns e.g. ``"\u2605 ivrit-ai large-v3 [local] \u2014 Hebrew \u00b7 3.1 GB"``
    """
    info = KNOWN_MODELS.get(raw_name)
    if info:
        fname = info["friendly"]
        prefix = _recommended_prefix(fname)
        size = _size_suffix(fname)
        return f"{prefix}{fname} [{source_tag}] \u2014 {_lang_tag(info['lang'])}{size}"
    # Fallback: use raw name + heuristic language tag
    lang = _infer_lang(raw_name)
    size = _size_suffix(raw_name)
    return f"{raw_name} [{source_tag}] \u2014 {_lang_tag(lang)}{size}"


def model_lang(label: str) -> Optional[str]:
    """Extract the language affinity from a model *label* string.

    Checks KNOWN_MODELS first, then falls back to heuristic.
    Returns 'he', 'en', or None (multilingual).
    """
    # Check by known friendly names embedded in the label
    for info in KNOWN_MODELS.values():
        if info["friendly"] in label:
            return info.get("lang")
    return _infer_lang(label)


def get_cache_dirs() -> List[Path]:
    candidates = []

    env_whisper = os.environ.get("WHISPER_CACHE_DIR")
    if env_whisper:
        candidates.append(Path(env_whisper))

    candidates.append(Path.home() / ".cache" / "whisper")

    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        candidates.append(Path(hf_home) / "hub")
        candidates.append(Path(hf_home))

    candidates.append(Path.home() / ".cache" / "huggingface" / "hub")

    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        candidates.append(Path(local_app_data) / "huggingface" / "hub")

    app_data = os.environ.get("APPDATA")
    if app_data:
        candidates.append(Path(app_data) / "huggingface" / "hub")

    out = []
    seen = set()
    for p in candidates:
        key = str(p).lower()
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def get_primary_model_cache_dir() -> Optional[Path]:
    """Return the first existing model cache directory, or None."""
    for p in get_cache_dirs():
        if p.exists():
            return p
    return None


def detect_openai_cached_models() -> List[str]:
    found = set()
    possible_roots = []

    env_whisper = os.environ.get("WHISPER_CACHE_DIR")
    if env_whisper:
        possible_roots.append(Path(env_whisper))
    possible_roots.append(Path.home() / ".cache" / "whisper")

    for root in possible_roots:
        if not root.exists():
            continue
        for model in OPENAI_MODELS:
            if (root / f"{model}.pt").exists():
                found.add(model)
        for pt in root.glob("*.pt"):
            found.add(pt.stem)

    return sorted(found, key=lambda x: OPENAI_MODELS.index(x) if x in OPENAI_MODELS else 999)


def _looks_like_ct2_model_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    has_model = (path / "model.bin").exists()
    has_config = (path / "config.json").exists() or (path / "tokenizer.json").exists()
    return has_model and has_config


def _dir_size_gb(path: Path) -> Optional[str]:
    """Return total size of files in *path* formatted as 'X.X GB', or None."""
    try:
        total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        gb = total / (1024 ** 3)
        if gb >= 0.01:
            return f"{gb:.1f} GB"
    except OSError:
        pass
    return None


def detect_faster_whisper_local_models(extra_roots: List[str]) -> List[Tuple[str, str]]:
    results: List[Tuple[str, str]] = []
    seen_paths = set()

    def _add(raw_name: str, resolved_path: str) -> None:
        if resolved_path not in seen_paths:
            seen_paths.add(resolved_path)
            label = friendly_label(raw_name, "local")
            # If label has no size from the hardcoded table, measure on disk
            if "\u00b7" not in label:
                measured = _dir_size_gb(Path(resolved_path))
                if measured:
                    label = f"{label} \u00b7 {measured}"
            results.append((label, resolved_path))

    roots = get_cache_dirs() + [Path(x) for x in extra_roots if x]
    for root in roots:
        if not root.exists():
            continue

        if _looks_like_ct2_model_dir(root):
            _add(root.name, str(root.resolve()))

        try:
            for child in root.iterdir():
                if _looks_like_ct2_model_dir(child):
                    _add(child.name, str(child.resolve()))
        except Exception:
            logger.debug("Error scanning model dir %s", root, exc_info=True)

        try:
            for model_dir in root.glob("models--*"):
                snapshots = model_dir / "snapshots"
                if not snapshots.exists():
                    continue
                for snap in snapshots.iterdir():
                    if _looks_like_ct2_model_dir(snap):
                        repo_name = model_dir.name.replace("models--", "").replace("--", "/")
                        _add(repo_name, str(snap.resolve()))
        except Exception:
            logger.debug("Error scanning HuggingFace model cache in %s", root, exc_info=True)

        for depth1 in list(root.glob("*"))[:200]:
            if depth1.is_dir() and _looks_like_ct2_model_dir(depth1):
                _add(depth1.name, str(depth1.resolve()))
            if depth1.is_dir():
                try:
                    for depth2 in list(depth1.glob("*"))[:100]:
                        if depth2.is_dir() and _looks_like_ct2_model_dir(depth2):
                            _add(depth2.name, str(depth2.resolve()))
                except Exception:
                    logger.debug("Error scanning subdirectory %s", depth1, exc_info=True)

    results.sort(key=lambda x: x[0].lower())
    return results
