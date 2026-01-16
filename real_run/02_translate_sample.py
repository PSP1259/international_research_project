import os
import json
import time
import pandas as pd
import deepl
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "data" / "mmlu_sample_englisch.csv"
OUTPUT_FILE = BASE_DIR / "data" / "mmlu_ready_bilingual.csv" 

CHAR_LIMIT_WARNING = 450000     # Monatliche Limit: 500'000


def translate_run():
  print("---Starte Übersetzungsprozess---")

  # 1. API Key holen
  api_key = os.getenv("DEEPL_API_KEY")
  if not api_key:
    print("FEHLER: Kein DEEPL_API_KEY in der .env gefunden!")
    return

  # 2. Translater initialisieren
  translator = deepl.Translator(api_key)

  # Check Usage
  try:
    usage = translator.get_usage()
    if usage.character.limit_reached:
      print("WARNUNG: DeepL Limit erreicht!")
      return
    print(f"DeepL Status: {usage.character.count} / {usage.character.limit} Zeichen genutzt.")
  except Exception as e:
    print(f"Warnung: Konnte DeepL Status nicht abrufen: {e}")

  # 3. Datei laden (mmlu_sample_english.csv)
  df = pd.read_csv(INPUT_FILE)
  print(f"Lade {len(df)} Fragen zur Übersetzung...")

  # Neue Spalten vorbereiten
  df["question_de"] = ""
  df["options_de_json"] = ""

  # Zähler für verbrauchte Zeichen
  chars_used_session = 0

  # 4. Translator Loop
  print("---Starte Translator Loop---")

  start_time = time.time()

  # a) Frage übersetzen EN -> DE
  for index, row in df.iterrows():

    # 1. Variablen initialisieren
    q_de = ""
    options_json_de = "[]"

    # A) Frage übersetzen
    q_en = row["question"]

    # Sicherheits-Check: Ist die englische Frage leer?
    if pd.isna(q_en) or str(q_en).strip() == "":
      print(f"Warnung: Leere Frage bei ID {row.get('question_id')}. Überspringe.")
      continue

    # Notbremse
    if chars_used_session > CHAR_LIMIT_WARNING:
      print("NOTBREMSE: Zeichen-Limit nähert sich!")
      break

    try:
      # DeepL Call für die Frage
      res_q = translator.translate_text(str(q_en), target_lang="DE")
      q_de = res_q.text
      chars_used_session += len(str(q_en))
    except Exception as e:
      print(f"Fehler bei Frage ID {row.get('question_id')}: {e}")
      # Wenn die Frage nicht übersetzt werden kann => überspringen!
      continue 

    # B) Optionen übersetzen
    try:
      # 1. String "[...]" in echte Python-Liste umwandeln
      options_list_en = json.loads(row["options_json"])

      # 2. Liste an DeepL senden
      res_opts = translator.translate_text(options_list_en, target_lang="DE")

      # 3. Ergebnisse extrahieren
      if isinstance(res_opts, list):
        options_list_de = [r.text for r in res_opts]
      else:
        options_list_de = [res_opts.text]

      # 4. Zurück in JSON-String wandeln
      options_json_de = json.dumps(options_list_de, ensure_ascii=False)

      # Zeichen zählen
      for opt in options_list_en:
        chars_used_session += len(str(opt))

    except Exception as e:
      print(f"Fehler bei Optionen ID {row.get('question_id')}: {e}")
      # Fallback: Speichert eine leere Liste, damit das Skript nicht abstürzt
      options_json_de = "[]"

    # C) Ins DataFrame schreiben (Jetzt ist q_de sicher definiert!)
    df.at[index, "question_de"] = q_de
    df.at[index, "options_de_json"] = options_json_de

    # Fortschrittsanzeige alle 50 Zeilen
    if (index + 1) % 50 == 0:
      print(f"   ... {index + 1} / {len(df)} erledigt. (Ca. {chars_used_session} Zeichen verbraucht)")

  # 6. Speichern
  output_dir = OUTPUT_FILE.parent
  output_dir.mkdir(parents=True, exist_ok=True)

  df.to_csv(OUTPUT_FILE, index=False)

  duration = time.time() - start_time
  print("-" * 30)
  print(f"FERTIG! Gespeichert unter: {OUTPUT_FILE}")
  print(f"Dauer: {round(duration, 2)} Sekunden")
  print(f"Zeichen in dieser Session: {chars_used_session}")
  print("-" * 30)


if __name__ == "__main__":
  translate_run()


