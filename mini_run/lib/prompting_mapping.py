# Structure

import hashlib
from pathlib import Path
from .mmlu import labels_for


def load_text(path: str | Path) -> str:         # Liest den Inhalt einer Datei -> str
    path = Path(path)                           # Path -> Path Objekt
    return path.read_text(encoding="utf-8")     # öffnet die Datei


# Virtueller Fingerprint
def sha1_10(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
"""
1. Kryptografische Hashfunktion =>      hashlib.sha1(...)
2. String -> Bytes =>                   text.encode("utf-8")
3. Bytes -> lesbarer Hex-String =>.     hexdigest()
4. Die ersten 10 Zeichen =>             [:10]

Es gibt sicherer hashes als sha1 in hashlib. 
""" 

def build_prompt(template: str, question: str, options: list[str]) -> str:
    # 1) Erzeuge Labels passend zur Anzahl Optionen ["A", "B",...]
    labels = labels_for(options)
    # 2) Baue den Options Block Zeile für Zeile auf
    option_lines = []


    for label, option_text in zip(labels, options):
        line = f"{label}. {option_text}"                # ["A. Berlin"]
        option_lines.append(line)                       # ["A. Berlin", "B. Paris", "C. Rome"]

    # 3) Verbinde alle Zeilen mit Zeilenumbrüchen
    # A. Berlin
    # B. Paris
    # C. Rome
    options_block = "\n".join(option_lines)

    # 4) Setze alle Platzhalter im Prompt Template ein
    #    {question} -> Frage
    #    {options}  -> Options Block
    #    {labels}   -> "A,B,C"
    prompt = template.format(
        question=question,
        options=options_block,
        labels=",".join(labels),
    )

    return prompt, labels