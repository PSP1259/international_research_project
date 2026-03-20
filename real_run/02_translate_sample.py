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
  print("---Starte Ãœbersetzungsprozess---")

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

  def _is_blank(value):
    return pd.isna(value) or str(value).strip() == ""

  def _has_options(value):
    if _is_blank(value):
      return False
    return str(value).strip() != "[]"

  # 3. Datei laden / Resume
  df_input = pd.read_csv(INPUT_FILE)

  if OUTPUT_FILE.exists():
    df_existing = pd.read_csv(OUTPUT_FILE)
    if "question_id" in df_input.columns and "question_id" in df_existing.columns:
      cols = ["question_id", "question_de", "options_de_json"]
      cols_existing = [c for c in cols if c in df_existing.columns]
      df = df_input.merge(df_existing[cols_existing], on="question_id", how="left")
    elif len(df_existing) == len(df_input):
      df = df_existing
    else:
      df = df_input
    print("Vorhandene Ausgabe gefunden, setze fort.")
  else:
    df = df_input

  if "question_de" not in df.columns:
    df["question_de"] = ""
  if "options_de_json" not in df.columns:
    df["options_de_json"] = ""

  df["question_de"] = df["question_de"].fillna("")
  df["options_de_json"] = df["options_de_json"].fillna("")

  done_mask = (
    df["question_de"].astype(str).str.strip() != ""
  ) & (
    df["options_de_json"].astype(str).str.strip() != ""
  ) & (
    df["options_de_json"].astype(str).str.strip() != "[]"
  )
  done_count = int(done_mask.sum())
  print(f"Lade {len(df)} Fragen zur Ãœbersetzung... Bereits uebersetzt: {done_count}")

  # ZÃ¤hler fÃ¼r verbrauchte Zeichen
  chars_used_session = 0

  # 4. Translator Loop
  print("---Starte Translator Loop---")

  start_time = time.time()

  # a) Frage Ã¼bersetzen EN -> DE
  for index, row in df.iterrows():

    # A) Frage Ã¼bersetzen
    q_en = row["question"]

    # Sicherheits-Check: Ist die englische Frage leer?
    if pd.isna(q_en) or str(q_en).strip() == "":
      print(f"Warnung: Leere Frage bei ID {row.get('question_id')}. Ãœberspringe.")
      continue

    q_de_existing = row.get("question_de", "")
    options_de_existing = row.get("options_de_json", "")

    has_question = not _is_blank(q_de_existing)
    has_options = _has_options(options_de_existing)

    if has_question and has_options:
      continue

    # Notbremse
    if chars_used_session > CHAR_LIMIT_WARNING:
      print("NOTBREMSE: Zeichen-Limit nÃ¤hert sich!")
      break

    if not has_question:
      try:
        # DeepL Call für die Frage
        res_q = translator.translate_text(str(q_en), target_lang="DE")
        df.at[index, "question_de"] = res_q.text
        chars_used_session += len(str(q_en))
      except Exception as e:
        print(f"Fehler bei Frage ID {row.get('question_id')}: {e}")
        # Wenn die Frage nicht übersetzt werden kann => überspringen!
        continue

    if not has_options:
      # Notbremse
      if chars_used_session > CHAR_LIMIT_WARNING:
        print("NOTBREMSE: Zeichen-Limit nÃ¤hert sich!")
        break

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

        # 4. ZurÃ¼ck in JSON-String wandeln
        options_json_de = json.dumps(options_list_de, ensure_ascii=False)

        # Zeichen zÃ¤hlen
        for opt in options_list_en:
          chars_used_session += len(str(opt))

      except Exception as e:
        print(f"Fehler bei Optionen ID {row.get('question_id')}: {e}")
        # Fallback: Speichert eine leere Liste, damit das Skript nicht abstÃ¼rzt
        options_json_de = "[]"

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
