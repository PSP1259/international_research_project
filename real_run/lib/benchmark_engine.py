import pandas as pd
import json
import time
import os
import re
import unicodedata
from pathlib import Path
from lib.ai_client import AIClient


current_file_path = Path(__file__).resolve()

BASE_DIR = current_file_path.parent.parent  

INPUT_FILE = BASE_DIR / "data" / "mmlu_ready_bilingual.csv"


# ----- Prompt Templates -----

PROMPT_TEMPLATES = {
  "EN": """
  Question: {question}
  Options: {options}

  INSTRUCTIONS:
  1. Solve the question.
  3. Determine your confidence (0-100).
  4. Return ONLY a JSON object.
  Example JSON format:
  {{
    "answer_char": "x", 
    "confidence": 0
  }}
  """,

  "DE": """
  Frage: {question}
  Optionen: {options}

  ANLEITUNG:
  1. Löse die Frage.
  2. Bestimme deine Sicherheit (0-100).
  3. Antworte NUR mit einem JSON Objekt.

  Beispiel JSON Format:
  {{
    "answer_char": "X",
    "confidence": 0
  }}
  """
}


LABEL_RE = re.compile(r"^\W*([A-Z])\b")
OPTION_INDEX_RE = re.compile(r"^\s*OPTION\s+(\d{1,2})\s*$", re.IGNORECASE)


def _normalize_label(value):
  if pd.isna(value):
    return None

  text = str(value).strip().upper()
  if not text:
    return None

  match = LABEL_RE.match(text)
  return match.group(1) if match else None


def _repair_text(value):
  if pd.isna(value):
    return None

  text = str(value)
  try:
    repaired = text.encode("latin-1").decode("utf-8")
  except (UnicodeEncodeError, UnicodeDecodeError):
    return text

  def _marker_score(candidate):
    return sum(candidate.count(marker) for marker in ("Ã", "â", "\x80", "\x81", "�"))

  return repaired if _marker_score(repaired) < _marker_score(text) else text


def _normalize_text(value):
  text = _repair_text(value)
  if text is None:
    return None

  text = " ".join(text.strip().upper().split())
  return text or None


def _normalize_lookup_key(value):
  text = _normalize_text(value)
  if not text:
    return None

  text = unicodedata.normalize("NFKD", text)
  text = re.sub(r"\\[A-Z]+", "", text)
  text = text.replace("VOLTS", "VOLT").replace("DOLLARS", "DOLLAR")
  text = "".join(ch for ch in text if not unicodedata.combining(ch))
  text = re.sub(r"[^A-Z0-9]+", "", text)
  return text or None


def _safe_load_options(value):
  if pd.isna(value):
    return []

  if isinstance(value, list):
    return value

  try:
    parsed = json.loads(value)
  except (TypeError, ValueError, json.JSONDecodeError):
    return []

  return parsed if isinstance(parsed, list) else []


def _label_from_position(position, option_count):
  if position < 1:
    return None

  upper_bound = min(option_count, 26)
  if upper_bound < 1 or position > upper_bound:
    return None

  return chr(ord("A") + position - 1)


def _infer_position_label(answer_value, option_count):
  text = _normalize_text(answer_value)
  if not text:
    return None

  match = OPTION_INDEX_RE.match(text)
  if match:
    return _label_from_position(int(match.group(1)), option_count)

  if text.isdigit():
    return _label_from_position(int(text), option_count)

  return None


def _extract_answer_label(answer_value, options_value):
  answer_label = _normalize_label(answer_value)
  if answer_label:
    return answer_label

  options = _safe_load_options(options_value)

  position_label = _infer_position_label(answer_value, len(options))
  if position_label:
    return position_label

  answer_text = _normalize_text(answer_value)
  answer_key = _normalize_lookup_key(answer_value)
  if not answer_text and not answer_key:
    return None

  for index, option in enumerate(options):
    label = chr(ord("A") + index)
    if answer_text and _normalize_text(option) == answer_text:
      return label
    if answer_key and _normalize_lookup_key(option) == answer_key:
      return label

  return None


def run_benchmark(config):

  # Dateiname generieren
  filename = f"results_{config['model_name']}_{config['language']}"
  if config["is_reasoning"]:
    filename += "_reasoning"
  filename += ".jsonl"

  OUTPUT_FILE = BASE_DIR / "results" / filename

  # Start Meldung
  print(f"\nJOB: {config['model_name']} [{config['language']}]")
  if config.get("max_questions"): print(f"LIMIT: {config['max_questions']}")



  # 1. Daten laden
  if not INPUT_FILE.exists():
    print(f"Input Datei fehlt: {INPUT_FILE}")
    return

  df = pd.read_csv(INPUT_FILE)
  total_questions = len(df)



  # 2. Resume Logik (Was haben wir schon?)
  completed_ids = set()
  OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

  if OUTPUT_FILE.exists():
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
      for line in f:
        try:
          data = json.loads(line)
          completed_ids.add(data["question_id"])
        except:
          continue  # Kaputte Zeilen ignorieren
    print(f"Resume: {len(completed_ids)} bereits erledigt.")

  # 3. Client initialisieren
  try:
    bot = AIClient(
      provider=config["provider"],
      model_name=config["model_name"],
      is_reasoning=config["is_reasoning"]
    )
  except Exception as e:
    print(f"Client Crash: {e}")
    return

  # 4. Der Loop
  processed_count = 0
  skipped_by_config = 0
  max_q = config.get("max_questions", 999999)  # Default: kein Limit
  skip_ids = set(config.get("skip_question_ids", []))
  if skip_ids:
    print(f"Zusatz-Skip-Liste aktiv: {len(skip_ids)} IDs.")

  # Datei im Append-Modus öffnen
  with open(OUTPUT_FILE, "a", encoding="utf-8") as f_out:

    for index, row in df.iterrows():

      # LIMIT CHECK (Emergency Break)
      if processed_count >= max_q:
        print(f"Tages-Limit von {max_q} Fragen erreicht. Beende Job.")
        break

      try:
        q_id = int(row["question_id"])
      except Exception:
        q_id = row["question_id"]

      # Skip wenn schon erledigt
      if q_id in completed_ids:
        continue

      # Skip IDs aus der Job-Konfig (z. B. bekannte Problemfragen)
      if q_id in skip_ids:
        skipped_by_config += 1
        continue

      lang = config["language"]

      # Text je nach Sprache wählen
      if lang == "DE":
        q_text = row.get("question_de", "")
        opts_text = row.get("options_de_json", "[]")
      else:
        q_text = row.get("question", "")
        opts_text = row.get("options_json", "[]")

      # Sicherheits-Check: Leere Fragen skippen
      if pd.isna(q_text) or str(q_text).strip() == "":
        print(f"Warnung: Leerer Text für ID {q_id}. Skipping.")
        continue

      # Prompt bauen
      prompt = PROMPT_TEMPLATES[lang].format(
        question=q_text,
        options=opts_text
      )

      print(f"ID {q_id} ({index + 1}/{total_questions}) ... ", end="")
      start_ts = time.time()

      # API Call
      response_data, usage_data = bot.infer(prompt)

      duration = time.time() - start_ts

      # Ergebnis nur speichern, wenn gültiges JSON zurückkommt
      if isinstance(response_data, dict):
        answer_char = _extract_answer_label(response_data.get("answer_char"), opts_text)
        correct_answer = _normalize_label(row.get("answer"))
        correct = 1 if answer_char and correct_answer and answer_char == correct_answer else 0

        result_entry = {
          "question_id": q_id,
          "category": row.get("category"),
          "language": lang,
          "model": config["model_name"],
          "is_reasoning": config["is_reasoning"],
          "question_len_char": len(q_text),
          "input_tokens": usage_data.get("input_tokens", 0),
          "output_tokens": usage_data.get("output_tokens", 0),
          "logprob": usage_data.get("raw_logprob"),
          "top_choices": usage_data.get("top_choices", []),
          "correct_answer": correct_answer,
          "answer_char": answer_char,
          "correct": correct,
          "confidence": response_data.get("confidence"),
          "duration_s": round(duration, 2),
          "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        # Sofort speichern (JSONL)
        f_out.write(json.dumps(result_entry, ensure_ascii=False) + "\n")
        f_out.flush()

        processed_count += 1

        print(f"OK ({duration:.1f}s)")

      else:
        print("Fehlgeschlagen")

      # Rate-Limit Schutz
      time.sleep(config["sleep_seconds"])

  print(f"Job beendet. {processed_count}! (skip IDs: {skipped_by_config})")


if __name__ == "__main__":
  run_benchmark(RUN_CONFIG)
