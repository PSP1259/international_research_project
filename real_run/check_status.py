import json
import pandas as pd
from pathlib import Path

# Pfade
BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
INPUT_FILE = BASE_DIR / "data" / "mmlu_ready_bilingual.csv"



def check():
  print("\n--- MMLU RUN STATUS MONITOR --- ")
    


  # 1. Ziel setzen
  if INPUT_FILE.exists():
    df = pd.read_csv(INPUT_FILE)
    total_goal = len(df)
    print(f"🏁 ZIEL: {total_goal} Fragen pro Modell/Sprache\n")
  else:
    print("Warnung: Input-Datei nicht gefunden. Ziel unbekannt.\n")
    total_goal = 0

  if not RESULTS_DIR.exists():
    print("Noch keine Ergebnisse vorhanden.")
    return




  # 2. Ergebnisse scannen
  files = sorted(list(RESULTS_DIR.glob("*.jsonl")))

  if len(files) == 0:
    print("Noch keine .jsonl Dateien im results Ordner gefunden.")
    return
    
  # Tabellenkopf
  print(f"{'DATEI NAME':<40} | {'FORTSCHRITT':<15} | {'STATUS'}")
  print("-" * 75)

  for f in files:
    count = 0

    # Robust zählen: nur gültige JSON Zeilen zählen, die question_id enthalten
    try:
      with open(f, "r", encoding="utf-8") as file:
        for line in file:
          line = line.strip()
          if not line:
            continue
          try:
            obj = json.loads(line)
            if "question_id" in obj:
              count += 1
          except:
            # Kaputte Zeilen ignorieren, damit Fortschritt nicht verfälscht wird
            continue
    except:
      count = 0
        
    # Prozent berechnen
    percent = 0
    if total_goal > 0:
      percent = (count / total_goal) * 100
        
    # Status
    if count >= total_goal and total_goal > 0:
      status = "FERTIG"
    elif count == 0:
      status = "STARTBEREIT"
    else:
      status = f"🏃 {percent:.1f}%"

    print(f"{f.name:<40} | {count}/{total_goal:<5}       | {status}")

  print("-" * 75)
  print("Starte Jobs im Ordner 'jobs/', um weiterzumachen.\n")

if __name__ == "__main__":
    check()
