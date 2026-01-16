"""

1) CSV laden
2) Sample ziehen
3) Prompt bauen
4) OpenAI und Gemini infer
5) Results JSONL schreiben
6) Failures separat JSONL schreiben


now_ts()
apply_api_result(row, res)
run_tool(client, prompt, row, labels, infer_store, fail_store)
main()
"""

import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from lib.io_jsonl import JsonlStore                                   # REF: lib/io_jsonl.py
from lib.mmlu import load_csv, clean_text, parse_options              # REF: lib/mmlu.py
from lib.clients import OpenAIClient, GeminiClient, extract_json      # REF: lib/clients.py
from lib.prompting_mapping import load_text, sha1_10, build_prompt            # REF: lib/prompting.py
from lib.logrows import base_row, failure_row                         # REF: lib/logrows.py
from lib.errors import IRPError


BASE_DIR = Path(__file__).resolve().parent                   # Flexibler und robuster als "hart" auf Laufwerk (C:\...)
CSV_PATH = BASE_DIR.parent / "mmlu_pro_all_questions.csv"    # Speicherort & -name
PROMPT_PATH = BASE_DIR / "prompts" / "prompt_en.txt"         # Speicherort & -name

OUT_INFER = BASE_DIR / "mini_infer.jsonl"                    # Speicherort & -name
OUT_FAILS = BASE_DIR / "mini_raw_failures.jsonl"             # Speicherort & -name

N = 1                                          # Sample size
RANDOM_STATE = 111                            # Benutzt immer denselben Seed => die exakt selben Werte

OPENAI_MODEL = "gpt-5.2"
GEMINI_MODEL = "gemini-2.5-flash"

LANGUAGE = "EN"
INPUT_MODE = "api"

SCHEMA_VERSION = "v1"
RUN_ID = "2026-01-07"
PROMPT_ID = "prompt_en_v1"
PROMPT_FILE = "prompts/prompt_en.txt"


def now_ts():
    return time.strftime("%Y-%m-%d %H:%M:%S")                       # Timestamp: "2026-01-05 13:39:56"


def apply_api_result(row: dict, res: dict) -> None:
    """
    Übernimmt Messwerte aus der API Antwort in die Log Row.
    Damit run_tool lesbarer bleibt.
    """
    row["latency_s"] = res.get("latency_s")
    row["token_out"] = res.get("token_out")
    row["attempt"] = res.get("attempt")
    row["error"] = res.get("error")
    row["error_type"] = res.get("error_type")
    row["retryable"] = res.get("retryable")
    row["retry_after_s"] = res.get("retry_after_s")


def run_tool(client, prompt: str, row: dict, labels: list[str], infer_store, fail_store):
    """
    Einheitlicher Ablauf pro Tool Call.

    1) client.infer(prompt)
    2) Messwerte in row übernehmen
    3) Wenn ok False, Failure loggen
    4) Wenn ok True, JSON extrahieren
    5) Wenn parse failed, Failure loggen
    6) Wenn parse ok, answer speichern und row schreiben
    """

    # 1) Sendet den Prompt an die API
    res = client.infer(prompt)

    # 2) Übernimmt Messwerte aus der API Antwort
    apply_api_result(row, res)

    # 3) API Call erfolgreich?
    if not res.get("ok"):
        infer_store.append(row)
        fail_store.append(failure_row(row, stage="api_error", raw=res.get("raw")))
        return

    # 4) Rohantwort hole, falls None -> leer lassen
    raw = res.get("raw", "") or ""

    # 5) JSON aus Antwort extrahieren {"answer":"B","confidence":92}
    parsed = extract_json(raw)

    # 6.1) JSON Parsing failed
    if parsed is None:
        row["error"] = "parse_failed"
        row["error_type"] = "parse"
        row["retryable"] = False
        infer_store.append(row)
        fail_store.append(failure_row(row, stage="parse_failed", raw=raw))
        return

    # 6.2) JSON Parsing erfolgreich
    ans = parsed.get("answer")
    row["answer"] = ans
    row["conf_self"] = parsed.get("confidence")
    row["answer_valid"] = bool(ans in labels)       # prüft ob Antwort erlaubt ist
    infer_store.append(row)


def main():

    # 1) Prints im Terminal für Kontrolle
    print("Reading:", CSV_PATH)
    print("Prompt:", PROMPT_PATH)
    print("Writing infer:", OUT_INFER)
    print("Writing fails:", OUT_FAILS)

    # 2) Alte JSONL-Dateien löschen
    if OUT_INFER.exists(): OUT_INFER.unlink()
    if OUT_FAILS.exists(): OUT_FAILS.unlink()

    # 3) JSONL Writer initialisieren
    infer_store = JsonlStore(OUT_INFER)
    fail_store = JsonlStore(OUT_FAILS)

    # 4) Daten und Template laden
    df = load_csv(CSV_PATH)
    tpl = load_text(PROMPT_PATH)
    p_sha = sha1_10(tpl)

    # 5) Mini Sample ziehen
    sample = df.sample(n=N, random_state=RANDOM_STATE).copy()

    # 6) Clients erstellen
    openai = OpenAIClient(OPENAI_MODEL)
    gemini = GeminiClient(GEMINI_MODEL)

    schema_cfg = {
        "SCHEMA_VERSION": SCHEMA_VERSION,
        "RUN_ID": RUN_ID,
        "PROMPT_ID": PROMPT_ID,
        "PROMPT_FILE": PROMPT_FILE,
        "LANGUAGE": LANGUAGE,
        "INPUT_MODE": INPUT_MODE,
        "now_ts": now_ts,
    }

    # 7) Jede Frage im Sample wiederholen
    for _, r in sample.iterrows():              # _ wäre ein Index, r eine Zeile von df

        # 8) Meta Felder pro Frage holen
        qid = int(r["question_id"])
        domain = str(r["category"])
        src = str(r["src"])

        # 9) Frage und Optionen aufbereiten
        question = clean_text(r["question"])
        options = parse_options(r["options_json"])

        # 10) Prompt zusammenbauen
        prompt, labels = build_prompt(tpl, question, options)

        # 11) Schutz via Prompt-Länge
        if len(prompt.strip()) < 50:
            raise ValueError("Prompt too short - template formatting failed")

        # 12) Überprüfung ob Antwort wirklich nicht in prompt, if True = AssertionError
        assert "answer_index" not in prompt
        assert "label_answer" not in prompt

        # 13) Log Row für OpenAI bauen
        row = base_row("openai", OPENAI_MODEL, qid, domain, src, labels, p_sha, schema_cfg)
        run_tool(openai, prompt, row, labels, infer_store, fail_store)

        # 14) Log Row für Gemini bauen
        row = base_row("gemini", GEMINI_MODEL, qid, domain, src, labels, p_sha, schema_cfg)
        run_tool(gemini, prompt, row, labels, infer_store, fail_store)

    print("Done:", OUT_INFER)


if __name__ == "__main__":
    try:
        main()
    except IRPError as e:
        print("Project error:", e)
        raise




