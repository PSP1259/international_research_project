from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from lib.mmlu import load_csv, clean_text, parse_options
from lib.deepl_translate import translate_question_and_options
from lib.io_jsonl import JsonlStore

N = 1                   # Sample size
RANDOM_STATE = 99       # Seed
TARGET_LANG = "DE"

# Dateien & Pfade
BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR.parent / "mmlu_pro_all_questions.csv"
OUT_TRANSLATED = BASE_DIR / "mini_translated_de.jsonl"

# Pflichtspalten werden aus lib.mmlu geprüft
df = load_csv(CSV_PATH)

# Mini Sample ziehen
sample = df.sample(n=N, random_state=RANDOM_STATE).copy()

# JSONL-Store initialisieren
store = JsonlStore(OUT_TRANSLATED)

# Läuft Zeile für Zeile durch das df => Säuberung
for _, r in sample.iterrows():
    qid = int(r["question_id"])
    question = clean_text(r["question"])
    options = parse_options(r["options_json"])

    # Deepl Übersetzung für Question und Options
    q_de, opts_de = translate_question_and_options(question, options, TARGET_LANG)
    
    # Schreibt pro Frage eine JSONL-Zeile
    store.append({
        "question_id": qid,
        "lang_src": "EN",
        "lang_tgt": TARGET_LANG,
        "question_en": question,
        "question_de": q_de,
        "options_en": options,
        "options_de": opts_de,
        "deepl_chars_in": len(question) + sum(len(o) for o in options),  # simple budget metric
    })

print("DeepL mini-run done:", OUT_TRANSLATED) # Terminal-Feedback
