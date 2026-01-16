import pandas as pd
import json
import time
import os
from pathlib import Path
from lib.ai_client import AIClient


current_file_path = Path(__file__).resolve()

BASE_DIR = current_file_path.parent.parent  

INPUT_FILE = BASE_DIR / "data" / "mmlu_ready_bilingual.csv"


# ----- Prompt Templates -----

PROMPT_TEMPLATES = {
  "EN": """
  Question: {question}
  Options:
  {options}

  Select the correct option letter (A, B, C, ...).
  Return ONLY a JSON object using this format (replace values with your result):
  {{
    "answer_char": "X",
    "confidence": 0
  }}
  """,

  "DE": """
  Frage: {question}
  Optionen:
  {options}

  Waehle den korrekten Buchstaben (A, B, C, ...).
  Antworte NUR mit einem JSON-Objekt in diesem Format (ersetze die Platzhalter!):
  {{
    "answer_char": "X",
    "confidence": 0
  }}
  """
}


def _parse_options(opts_text):
  try:
    opts = json.loads(opts_text)
  except Exception:
    return []
  if not isinstance(opts, list):
    return []
  return [str(x).strip() for x in opts]


def _labels_for(options):
  return [chr(ord("A") + i) for i in range(len(options))]


def _format_options(options, labels):
  lines = []
  for label, option in zip(labels, options):
    lines.append(f"{label}) {option}")
  return "\n".join(lines)


def _normalize_text(value):
  return " ".join(str(value).strip().split()).lower()


def _normalize_label(value):
  if value is None:
    return None
  label = str(value).strip().upper()
  return label if label else None


def _normalize_answer(answer_raw, options, labels):
  if answer_raw is None:
    return None
  raw = str(answer_raw).strip()
  if raw == "":
    return None
  if not labels:
    return raw

  raw_upper = raw.upper()
  for label in labels:
    if raw_upper == label:
      return label

  for label in labels:
    if raw_upper.startswith(label) and len(raw_upper) > 1:
      next_char = raw_upper[1]
      if not next_char.isalnum():
        return label

  norm_raw = _normalize_text(raw)
  for idx, opt in enumerate(options):
    if _normalize_text(opt) == norm_raw:
      return labels[idx]

  return raw


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
  max_q = config.get("max_questions", 999999)  # Default: kein Limit

  # Datei im Append-Modus öffnen
  with open(OUTPUT_FILE, "a", encoding="utf-8") as f_out:

    for index, row in df.iterrows():

      # LIMIT CHECK (Emergency Break)
      if processed_count >= max_q:
        print(f"Tages-Limit von {max_q} Fragen erreicht. Beende Job.")
        break

      q_id = row["question_id"]

      # Skip wenn schon erledigt
      if q_id in completed_ids:
        continue

      lang = config["language"]

      # Text je nach Sprache wählen
      if lang == "DE":
        q_text = row.get("question_de", "")
        opts_text = row.get("options_de_json", "[]")
      else:
        q_text = row.get("question", "")
        opts_text = row.get("options_json", "[]")

      options = _parse_options(opts_text)
      labels = _labels_for(options) if options else []
      options_block = _format_options(options, labels) if options else str(opts_text)

      # Sicherheits-Check: Leere Fragen skippen
      if pd.isna(q_text) or str(q_text).strip() == "":
        print(f"Warnung: Leerer Text für ID {q_id}. Skipping.")
        continue

      # Prompt bauen
      prompt = PROMPT_TEMPLATES[lang].format(
        question=q_text,
        options=options_block
      )

      print(f"ID {q_id} ({index + 1}/{total_questions}) ... ", end="", flush=True)
      start_ts = time.time()

      # API Call
      response_data, usage_data = bot.infer(prompt)

      duration = time.time() - start_ts

      # Ergebnis nur speichern, wenn gültiges JSON zurückkommt
      if isinstance(response_data, dict):
        answer_raw = response_data.get("answer_char")
        answer_char = _normalize_answer(answer_raw, options, labels)
        answer_label = _normalize_label(answer_char)
        correct_label = _normalize_label(row.get("answer"))
        correct = 1 if answer_label and correct_label and answer_label == correct_label else 0
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
          "correct_answer": row.get("answer"),       
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

  print(f"Job beendet. {processed_count}!")


if __name__ == "__main__":
  run_benchmark(RUN_CONFIG)
