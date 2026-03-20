from pathlib import Path
import pandas as pd
import os

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = BASE_DIR.parent / "data" / "mmlu_pro_all_questions.csv"
OUTPUT_FILE = BASE_DIR / "data" / "mmlu_sample_englisch.csv"

SAMPLE_CONFIG = {
  "math": 94,
  "physics": 93,
  "chemistry": 92,
  "law": 92,
  "engineering": 91,
  "other": 91,
  "economics": 90,
  "health": 90,
  "psychology": 89,
  "business": 89,
  "biology": 88,
  "philosophy": 84,
  "computer science": 81,
  "history": 80
}

def create_sample():
  print("---Start Sampling Process---")

  # 1) Check ob INPUT_FILE da ist
  if not INPUT_FILE.exists():
    print(f"FEHLER: Konnte {INPUT_FILE} nicht finden!")
    return  # << wichtig

  # 2) Check ob CSV lesbar ist
  print(f"Lade Quelldatei: {INPUT_FILE} ...")
  try:
    df_all = pd.read_csv(INPUT_FILE, encoding="utf-8")
  except Exception as e:
    print(f"Fehler beim Lesen der CSV: {e}")
    return 

  print(f"Gesamtanzahl Fragen im Original: {len(df_all)}")

  sampled_frames = []

  # 3) Loop durch die Kategorien
  print("Ziehe zufällige Fragen pro Kategorie...")
  for category_name, count in SAMPLE_CONFIG.items():
    df_category = df_all[df_all["category"] == category_name]
    available = len(df_category)

    if available < count:
      print(f"WARNUNG: {category_name}: Wollte {count}, habe nur {available}. Nehme alle.")
      count = available

    if available > 0 and count > 0:
      sample = df_category.sample(n=count, random_state=49)
      sampled_frames.append(sample)
      print(f"Check: {category_name} : {len(sample)}")
    else:
      print(f"Check: {category_name} : 0 (keine Daten)")

  # 4) Speichern
  if sampled_frames:
    final_df = pd.concat(sampled_frames, ignore_index=True)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    final_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")  
    print(f"Sample gespeichert in: {OUTPUT_FILE}")
    print(f"Anzahl Fragen im Sample: {len(final_df)}")
  else:
    print("Fehler: Kein Sample erzeugt (sampled_frames ist leer).")

if __name__ == "__main__":
  create_sample()
