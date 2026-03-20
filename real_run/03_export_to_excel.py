# 07_export_to_excel.py
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"          # bei dir gemäss repo-structure
OUT_FILE = BASE_DIR / "results_all.xlsx"    # Output im Projektroot anpassen falls nötig
INPUT_FILE = BASE_DIR / "data" / "mmlu_ready_bilingual.csv"
LABEL_RE = re.compile(r"^\W*([A-Z])\b")
OPTION_INDEX_RE = re.compile(r"^\s*OPTION\s+(\d{1,2})\s*$", re.IGNORECASE)


def safe_read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    rows.append(obj)
            except json.JSONDecodeError:
                # kaputte Zeile ignorieren
                continue
    return rows


def infer_meta_from_filename(filename: str) -> Dict[str, Optional[str]]:
    """
    Erwartete Dateinamen (Beispiele):
      results_gpt-4.1_DE.jsonl
      results_gpt-5.2_EN_reasoning.jsonl
      results_gemini-3-flash-preview_DE.jsonl
    """
    name = filename.replace(".jsonl", "")

    # model: alles zwischen "results_" und _EN/_DE...
    m = re.match(r"^results_(.+?)_(EN|DE)(?:_(reasoning))?$", name)
    if not m:
        return {"model": None, "language": None, "reasoning_variant": None, "provider": None}

    model = m.group(1)
    language = m.group(2)
    reasoning_variant = m.group(3)  # "reasoning" oder None

    # provider grob aus model ableiten
    provider = None
    if model.startswith("gpt-"):
        provider = "openai"
    elif model.startswith("gemini-"):
        provider = "google"

    return {
        "model": model,
        "language": language,
        "reasoning_variant": reasoning_variant,  # z.B. "reasoning" bei gpt-5.2_*_reasoning
        "provider": provider,
    }


def flatten_dict(d: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    """
    Flatten nested dicts: {"a":{"b":1}} -> {"a.b":1}
    Listen bleiben als JSON-string (damit Excel nicht explodiert).
    """
    items: Dict[str, Any] = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else str(k)
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        elif isinstance(v, list):
            # Listen als kompakter JSON String
            try:
                items[new_key] = json.dumps(v, ensure_ascii=False)
            except Exception:
                items[new_key] = str(v)
        else:
            items[new_key] = v
    return items


def normalize_label(value: Any) -> Optional[str]:
    if pd.isna(value):
        return None

    text = str(value).strip().upper()
    if not text:
        return None

    match = LABEL_RE.match(text)
    return match.group(1) if match else None


def repair_text(value: Any) -> Optional[str]:
    if pd.isna(value):
        return None

    text = str(value)
    try:
        repaired = text.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text

    def marker_score(candidate: str) -> int:
        return sum(candidate.count(marker) for marker in ("Ã", "â", "\x80", "\x81", "�"))

    return repaired if marker_score(repaired) < marker_score(text) else text


def normalize_text(value: Any) -> Optional[str]:
    text = repair_text(value)
    if text is None:
        return None

    text = " ".join(text.strip().upper().split())
    return text or None


def normalize_lookup_key(value: Any) -> Optional[str]:
    text = normalize_text(value)
    if not text:
        return None

    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r"\\[A-Z]+", "", text)
    text = text.replace("VOLTS", "VOLT").replace("DOLLARS", "DOLLAR")
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^A-Z0-9]+", "", text)
    return text or None


def normalize_question_id(value: Any) -> Any:
    if pd.isna(value):
        return None

    try:
        return int(value)
    except (TypeError, ValueError):
        text = str(value).strip()
        return text or None


def safe_parse_options(value: Any) -> List[Any]:
    if pd.isna(value):
        return []

    if isinstance(value, list):
        return value

    try:
        parsed = json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return []

    return parsed if isinstance(parsed, list) else []


def build_option_lookup(path: Path) -> Dict[tuple[Any, str], Dict[str, Any]]:
    if not path.exists():
        return {}

    df_source = pd.read_csv(path)
    lookup: Dict[tuple[Any, str], Dict[str, Any]] = {}

    for _, row in df_source.iterrows():
        question_id = normalize_question_id(row.get("question_id"))
        if question_id is None:
            continue

        for language, column_name in (("EN", "options_json"), ("DE", "options_de_json")):
            options = safe_parse_options(row.get(column_name))
            if not options:
                continue

            state: Dict[str, Any] = {
                "count": len(options),
                "exact": {},
                "fuzzy": {},
            }
            for index, option in enumerate(options):
                label = chr(ord("A") + index)
                option_text = normalize_text(option)
                if not option_text:
                    continue
                option_key = normalize_lookup_key(option)
                state["exact"].setdefault(option_text, label)
                if option_key:
                    state["fuzzy"].setdefault(option_key, label)

            if state["exact"] or state["fuzzy"]:
                lookup[(question_id, language)] = state

    return lookup


def label_from_position(position: int, option_count: int) -> Optional[str]:
    if position < 1:
        return None

    upper_bound = min(option_count, 26)
    if upper_bound < 1 or position > upper_bound:
        return None

    return chr(ord("A") + position - 1)


def infer_position_label(answer_value: Any, option_count: int) -> Optional[str]:
    text = normalize_text(answer_value)
    if not text:
        return None

    match = OPTION_INDEX_RE.match(text)
    if match:
        return label_from_position(int(match.group(1)), option_count)

    if text.isdigit():
        return label_from_position(int(text), option_count)

    return None


def infer_answer_label(
    row: Dict[str, Any],
    option_lookup: Dict[tuple[Any, str], Dict[str, Any]],
) -> Optional[str]:
    answer_label = normalize_label(row.get("answer_char"))
    if answer_label:
        return answer_label

    question_id = normalize_question_id(row.get("question_id"))
    language = normalize_text(row.get("language") or row.get("_language"))
    if question_id is None or not language:
        return None

    primary_state = option_lookup.get((question_id, language))
    fallback_language = "DE" if language == "EN" else "EN"
    fallback_state = option_lookup.get((question_id, fallback_language))
    option_count = 0
    for state in (primary_state, fallback_state):
        if state:
            option_count = max(option_count, int(state.get("count", 0)))

    position_label = infer_position_label(row.get("answer_char"), option_count)
    if position_label:
        return position_label

    answer_text = normalize_text(row.get("answer_char"))
    answer_key = normalize_lookup_key(row.get("answer_char"))
    if not answer_text and not answer_key:
        return None

    for state in (primary_state, fallback_state):
        if not state:
            continue
        if answer_text and answer_text in state["exact"]:
            return state["exact"][answer_text]
        if answer_key and answer_key in state["fuzzy"]:
            return state["fuzzy"][answer_key]

    return None


def infer_correct(
    row: Dict[str, Any],
    option_lookup: Dict[tuple[Any, str], Dict[str, Any]],
) -> Optional[int]:
    answer_label = infer_answer_label(row, option_lookup)
    correct_label = normalize_label(row.get("correct_answer", row.get("answer")))
    if not answer_label or not correct_label:
        return None
    return int(answer_label == correct_label)


def main() -> None:
    if not RESULTS_DIR.exists():
        raise FileNotFoundError(f"results folder nicht gefunden: {RESULTS_DIR}")

    files = sorted(RESULTS_DIR.glob("results_*.jsonl"))
    if not files:
        raise FileNotFoundError(f"keine results_*.jsonl gefunden in: {RESULTS_DIR}")

    all_rows: List[Dict[str, Any]] = []
    backfilled_correct = 0
    option_lookup = build_option_lookup(INPUT_FILE)

    for f in files:
        meta = infer_meta_from_filename(f.name)
        rows = safe_read_jsonl(f)

        for r in rows:
            flat = flatten_dict(r)
            if flat.get("correct") is None:
                inferred_correct = infer_correct(flat, option_lookup)
                if inferred_correct is not None:
                    flat["correct"] = inferred_correct
                    backfilled_correct += 1
            # Meta-Spalten vorne dran
            flat["_file"] = f.name
            flat["_provider"] = meta["provider"]
            flat["_model"] = meta["model"]
            flat["_language"] = meta["language"]
            flat["_reasoning_variant"] = meta["reasoning_variant"]
            all_rows.append(flat)

    df = pd.DataFrame(all_rows)

    # optional: schöne Spaltenreihenfolge (wenn vorhanden)
    preferred = [
        "_provider", "_model", "_language", "_reasoning_variant", "_file",
        "question_id", "domain", "category",
        "correct_answer", "answer_char", "correct",
        "conf_self", "confidence", "confidence_self",  # je nach naming
        "latency_sec", "latency", "token_count", "tokens",
    ]
    cols = list(df.columns)
    ordered = [c for c in preferred if c in cols] + [c for c in cols if c not in preferred]
    df = df[ordered]

    if "correct" in df.columns:
        df["correct"] = pd.to_numeric(df["correct"], errors="coerce").astype("Int64")

    # Export
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    export_path = OUT_FILE
    try:
        with pd.ExcelWriter(export_path, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="all_results", index=False)
    except PermissionError:
        export_path = OUT_FILE.with_name(f"{OUT_FILE.stem}_updated{OUT_FILE.suffix}")
        with pd.ExcelWriter(export_path, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="all_results", index=False)
        print(f"   Standarddatei gesperrt, schreibe stattdessen: {export_path}")

    print(f"Export fertig: {export_path}")
    print(f"   Dateien: {len(files)} | Zeilen total: {len(df)} | Spalten: {len(df.columns)}")
    print(f"   Fehlende 'correct' Werte nachberechnet: {backfilled_correct}")


if __name__ == "__main__":
    main()
