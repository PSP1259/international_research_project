"""
load_csv: liest dein MMLU-CSV als DataFrame
clean_text: säubert Frage-Text (whitespace, control chars)
parse_options: macht aus options_json (String) -> Liste von Optionen

CSV
 └─ load_csv()
      └─ question row
           └─ parse_options()
                └─ labels_for()
                     └─ gold_letter()
"""

import json
import re
from pathlib import Path

import pandas as pd

from .errors import DataError


# Hat das CSV diese Spalten?
NEEDED_COLS = ["question_id", "category", "question", "options_json", "answer_index", "src"]


def clean_text(s) -> str:       # Basic cleanup for text

    if s is None:
        return ""

    # statt crash bei None => leere Zelle ""
    s = str(s)                 

    # remove control characters
    s = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", s) 

    # Mehrere Spaces/Tabs werden zu einem Space
    s = re.sub(r"[ \t]+", " ", s)

    # Wenn es 3+ neue Zeilen hast → macht nur 2
    s = re.sub(r"\n{3,}", "\n\n", s)

    # Trimmt vorne/hinten whitespace
    return s.strip()


def parse_options(options_json: str) -> list[str]:  # Options: ["Paris", "London", "Berlin"]

    try:
        opts = json.loads(options_json)             # String → Python-Objekt
    except Exception:
        return []

    if not isinstance(opts, list):                  # Muss eine Liste sein
        return []

    return [clean_text(x) for x in opts]            # text cleanen


# Dynamisches Setup statt hardcoden
def labels_for(options: list[str]) -> list[str]:    # Zu jeder Option ein Buchstaben erzeugen
    labels = []                                     # erzeugt eine Liste
    for i in range(len(options)):                   # 0, 1, 2...
        labels.append(chr(ord("A") + i))            # "A", "B", "C"...
    return labels


# Antwort prüfen
def gold_letter(answer_index: int, options: list[str]): 
    if 0 <= answer_index < len(options):                    # Antwortindex muss höher als 0 sein
        return chr(ord("A") + answer_index)                 # Buchstaben für Antwort zurückgeben
    return None


def load_csv(csv_path: str | Path) -> pd.DataFrame:
    csv_path = Path(csv_path)

    # Datei gefunden?
    try:
        df = pd.read_csv(csv_path, encoding="utf-8")                
    except Exception as e:
        raise DataError(f"Could not read CSV: {csv_path} ({e})") 

    # Spalten prüfen
    missing = []
    for c in NEEDED_COLS:
        if c not in df.columns:
            missing.append(c)

    # Abbruch bei Fehler
    if missing:
        raise DataError(f"CSV missing columns: {missing}")

    return df

