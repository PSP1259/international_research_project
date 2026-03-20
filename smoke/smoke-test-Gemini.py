import os, time, json
from dotenv import load_dotenv

load_dotenv()

def log_result(name, ok, latency_s=None, extra=None):
    print(json.dumps({
        "api": name, "ok": ok, "latency_s": latency_s,
        "extra": extra, "ts": time.strftime("%Y-%m-%d %H:%M:%S")
    }, ensure_ascii=False))

def test_gemini_genai():
    try:
        from google import genai
        client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

        model = "gemini-2.0-flash"

        t0 = time.time()
        resp = client.models.generate_content(
            model=model,
            contents="Reply with exactly: OK"
        )
        latency = time.time() - t0

        # resp.text ist meistens verfügbar
        text = getattr(resp, "text", None)
        log_result("gemini", True, round(latency, 3), {"model": model, "text": text})

    except Exception as e:
        log_result("gemini", False, extra={"error": str(e)})

test_gemini_genai()



