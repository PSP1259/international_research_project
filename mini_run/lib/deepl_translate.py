import os
import json
from pathlib import Path
import hashlib

from .errors import ConfigError, APIError, DataError

CACHE_PATH = Path(__file__).resolve().parent / "deepl_cache.json"
USAGE_PATH = Path(__file__).resolve().parent / "deepl_usage.json"

PERSONAL_CHAR_LIMIT = 20_000  # DeepL monthly Limit is 500'000

# json-Datei laden sonst DataError
def _load_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise DataError(f"Could not read {path} ({e})")

# Python-Objekt als JSON-Datei speichern
def _save_json(path: Path, obj):
    try:
        path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        raise DataError(f"Could not write {path} ({e})")

# Erzeugt Cache-Key für "dieser Text in dieser Ziel-Sprache"
def _key(text: str, target_lang: str) -> str:
    h = hashlib.sha1((target_lang + "|" + text).encode("utf-8")).hexdigest()
    return h

# liest used_chars aus usage file
def _get_used_chars() -> int:
    usage = _load_json(USAGE_PATH, {"used_chars": 0})
    return int(usage.get("used_chars", 0))

# addiert n Zeichen zum used_chars counter
def _add_used_chars(n: int) -> None:
    usage = _load_json(USAGE_PATH, {"used_chars": 0})
    usage["used_chars"] = int(usage.get("used_chars", 0)) + int(n)
    _save_json(USAGE_PATH, usage)

# übersetzt Text mit DeepL
def translate_text(text: str, target_lang: str, source_lang: str = "EN") -> str:
    text = text or ""
    if not text.strip():
        return text

    auth = os.getenv("DEEPL_API_KEY")
    if not auth:
        raise ConfigError("DEEPL_API_KEY missing")

    # Cache laden (hash -> translation)
    cache = _load_json(CACHE_PATH, {})
    k = _key(text, target_lang)
    if k in cache:        # wenn bereits übersetzt -> sofort zurück => kein API call
        return cache[k]

    # Budget check BEFORE calling API
    used = _get_used_chars()
    needed = len(text)
    if used + needed > PERSONAL_CHAR_LIMIT:     # API Call verhindern
        return text

    try:
        import deepl                            # API Call: Übersetzung anfordern
        translator = deepl.Translator(auth)
        res = translator.translate_text(
            text,
            source_lang=source_lang,
            target_lang=target_lang
        )
        out = str(res)
    except Exception as e:
        raise APIError(f"DeepL translate failed ({e})")

    # Übersetzung in Cache schreiben
    cache[k] = out
    _save_json(CACHE_PATH, cache)     # Cache Datei speichern
    _add_used_chars(needed)           # Usage Counter erhöhen (lokal)
    return out

# übersetzt Frage + alle Optionen und gibt sie als tuple zurück
def translate_question_and_options(question: str, options: list[str], target_lang: str) -> tuple[str, list[str]]:
    q_t = translate_text(question, target_lang=target_lang)
    opts_t = [translate_text(o, target_lang=target_lang) for o in options]
    return q_t, opts_t
