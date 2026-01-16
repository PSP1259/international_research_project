import os
import time
import json
from dotenv import load_dotenv

load_dotenv()                                             # MUSS vor OpenAI-Client passieren

def log_result(name, ok, latency_s=None, extra=None):
    row = {
        "api": name,
        "ok": ok,
        "latency_s": latency_s,
        "extra": extra,
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    print(json.dumps(row, ensure_ascii=False))


def test_openai():
    try:
        from openai import OpenAI

        client = OpenAI()                                 # OpenAI-Client -> KEINE Argumente allowed

        t0 = time.time()
        resp = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": "Reply with exactly: OK"}],
            temperature=0
        )
        latency = time.time() - t0

        text = resp.choices[0].message.content
        log_result("openai", True, latency_s=round(latency, 3), extra={"text": text})

    except Exception as e:
        log_result("openai", False, extra={"error": str(e)})


test_openai()

