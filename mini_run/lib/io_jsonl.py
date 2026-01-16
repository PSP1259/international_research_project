# Persistent

import json
from pathlib import Path

from .errors import DataError           # REF: lib/errors.py




class JsonlStore:      
    def __init__(self, path: Path):     # Erstellt Unter-/Ordner auch wenn bereits existiert
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, row):              # Sicherheitscheck -> verhindert kaputte JSON-Zeilen
        if row is None:
            return

        try:
            with self.path.open("a", encoding="utf-8") as f:         # "a" append (ans Ende hängen)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")  # jede Zeile ein Objekt
        except Exception as e:
            raise DataError(f"Could not write JSONL: {self.path} ({e})")

    def read_all(self):                     # Output immer eine Liste von Dictionaries
        if not self.path.exists():          # Wenn Datei nicht existiert => leere Liste
            return []

        try:
            with self.path.open("r", encoding="utf-8") as f:     # "r" Lesemodus
                result = []                                      # sammelt alle Datensätze
                for line in f:                                   # Datei wird zeilenweise gelesen
                    if line.strip():                             # ignoriert leere Zeilen
                        result.append(json.loads(line))     
                return result
        except Exception as e:
            raise DataError(f"Could not read JSONL: {self.path} ({e})")
