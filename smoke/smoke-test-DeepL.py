import os, time, json
import requests
from dotenv import load_dotenv

load_dotenv()

OUT = "deepl_smoke.jsonl"

def append_jsonl(path, obj):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def log_result(name, ok, latency_s=None, extra=None):
    row = {
        "api": name,
        "ok": ok,
        "latency_s": latency_s,
        "extra": extra,
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    print(json.dumps(row, ensure_ascii=False))
    append_jsonl(OUT, row)

def test_deepl():
    key = os.getenv("DEEPL_API_KEY")
    if not key:
        log_result("deepl", False, extra={"error": "DEEPL_API_KEY missing"})
        return

    params = {"auth_key": key, "text": "Hello world", "target_lang": "DE"}
    urls = ["https://api-free.deepl.com/v2/translate", "https://api.deepl.com/v2/translate"]

    last = None
    for url in urls:
        try:
            t0 = time.time()
            r = requests.post(url, data=params, timeout=30)
            latency = time.time() - t0

            if r.status_code == 200:
                data = r.json()
                out = data["translations"][0]["text"]
                log_result("deepl", True, round(latency, 3), extra={"url": url, "out": out})
                return
            else:
                last = {"url": url, "status": r.status_code, "body": r.text[:300]}
        except Exception as e:
            last = {"url": url, "error": str(e)}

    log_result("deepl", False, extra=last)

print("CWD =", os.getcwd())
print("Will write to:", os.path.abspath(OUT))
test_deepl()

