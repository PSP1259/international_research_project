import os, time, json
from dotenv import load_dotenv

load_dotenv()

from google import genai

MODEL = "gemini-3-flash-preview"
OUT_JSONL = "results_gemini.jsonl"

# ---- safety knobs ----
MAX_RETRIES = 6
BASE_SLEEP = 1.0          # seconds
MAX_SLEEP = 60.0
MAX_CALLS_PER_RUN = 50    # harte Bremse fürs Testen (später hochsetzen)

def append_jsonl(path, obj):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def now_ts():
    return time.strftime("%Y-%m-%d %H:%M:%S")

def call_gemini_safe(prompt, client):
    """
    Returns dict with ok/text/latency and error info.
    Retries on 429 + 5xx with exponential backoff.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        t0 = time.time()
        try:
            resp = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                # wenn dein SDK config unterstützt, können wir hier später max_output_tokens setzen
            )
            latency = time.time() - t0
            text = getattr(resp, "text", None)

            return {
                "ok": True,
                "model": MODEL,
                "latency_s": round(latency, 3),
                "text": text,
                "attempt": attempt,
            }

        except Exception as e:
            latency = time.time() - t0
            msg = str(e)

            # sehr einfache Klassifikation (reicht für Start)
            is_429 = "429" in msg
            is_5xx = any(code in msg for code in ["500", "502", "503", "504"])

            err_obj = {
                "ok": False,
                "model": MODEL,
                "latency_s": round(latency, 3),
                "attempt": attempt,
                "error": msg[:600],
                "ts": now_ts(),
            }

            # wenn nicht retrybar -> sofort zurück
            if not (is_429 or is_5xx):
                return err_obj

            # retrybar -> backoff
            sleep_s = min(MAX_SLEEP, BASE_SLEEP * (2 ** (attempt - 1)))
            time.sleep(sleep_s)

    return {
        "ok": False,
        "model": MODEL,
        "attempt": MAX_RETRIES,
        "error": "max retries exceeded",
        "ts": now_ts(),
    }

def main():
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

    # Mini-Testloop (später ersetzen wir das durch MMLU-CSV)
    prompts = [
        "Reply with exactly: OK",
        "Reply with exactly: OK",
        "Reply with exactly: OK",
    ]

    for i, p in enumerate(prompts[:MAX_CALLS_PER_RUN], start=1):
        res = call_gemini_safe(p, client)
        row = {"i": i, "ts": now_ts(), "prompt": p, **res}
        print(row)
        append_jsonl(OUT_JSONL, row)

if __name__ == "__main__":
    main()
  