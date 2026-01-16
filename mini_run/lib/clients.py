# API Communication

import time
import os
import json
from openai import OpenAI
from google import genai

from .errors import ConfigError


MAX_RETRIES = 5          # Max. Wiederholungen
BASE_SLEEP = 1.0         # Start-Wartezeit (Sek.)
MAX_SLEEP = 30.0         # Max. Wartezeit
ERR_LIMIT = 240          # Max. Länge Error-Text


def sleep_backoff(attempt: int):
    seconds = min(MAX_SLEEP, BASE_SLEEP * (2 ** (attempt - 1)))
    time.sleep(seconds)
"""
Exponentialer Backoff pro Versuch.
attempt 1 -> 1.0 Sekunde
attempt 2 -> 2.0 Sekunden
attempt 3 -> 4.0 Sekunden
Wartezeit wächst mit Potenz von 2.
min(MAX_SLEEP, ...) setzt eine Obergrenze von 30.0 Sekunden.
"""


def extract_json(text: str):
    if not text:
        return None

    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    # Try to extract the first JSON object from surrounding text.
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                try:
                    return json.loads(candidate)
                except Exception:
                    return None

    return None


def _openai_temperature(model: str):
    # Some models (e.g. gpt-5-nano) only accept the default temperature.
    if model.startswith("gpt-5"):
        return None
    return 0.0


def _gemini_text(resp) -> str:
    text = getattr(resp, "text", None)
    if text:
        return text

    candidates = getattr(resp, "candidates", None)
    if candidates:
        cand0 = candidates[0]
        content = getattr(cand0, "content", None)
        if content is not None:
            content_text = getattr(content, "text", None)
            if content_text:
                return content_text
            parts = getattr(content, "parts", None)
            if parts:
                texts = []
                for part in parts:
                    part_text = getattr(part, "text", None)
                    if part_text:
                        texts.append(part_text)
                if texts:
                    return "".join(texts)

    choices = getattr(resp, "choices", None)
    if choices:
        try:
            return choices[0].message.content
        except Exception:
            pass

    return ""


# OpenAI Client:
class OpenAIClientEasy:
    def __init__(self, model: str):
        self.client = OpenAI()      # Holt Key selbständig
        self.model = model

    # OpenAI SDK => Role Based Prompting (System - User) -> chat.completions.create()
    def infer(self, prompt: str) -> dict:
        for attempt in range(1, MAX_RETRIES + 1):
            t0 = time.time()
            try:
                req = {
                    "model": self.model,                          # welches modell soll antworten
                    "messages": [
                        # System Message
                        {"role": "system", "content": "Return only the final answer."},
                        # User Message
                        {"role": "user", "content": prompt},
                    ],
                }
                temp = _openai_temperature(self.model)
                if temp is not None:
                    req["temperature"] = temp

                resp = self.client.chat.completions.create(**req)    # API request

                return {
                    "ok": True,
                    "attempt": attempt,
                    "latency_s": round(time.time() - t0, 3),
                    "raw": resp.choices[0].message.content,
                    "token_out": getattr(getattr(resp, "usage", None), "completion_tokens", None),
                    "error": None,
                    "error_type": None,
                    "retryable": None,
                    "retry_after_s": None,
                }

            except Exception as e:
                if attempt == MAX_RETRIES:
                    return {
                        "ok": False,
                        "attempt": attempt,
                        "latency_s": round(time.time() - t0, 3),
                        "raw": None,
                        "token_out": None,
                        "error": str(e)[:ERR_LIMIT],
                        "error_type": type(e).__name__,
                        "retryable": None,
                        "retry_after_s": None,
                    }

                sleep_backoff(attempt)


# Gemini Client:
class GeminiClientEasy:
    def __init__(self, model: str):
        key = os.getenv("GOOGLE_API_KEY")             # Ich muss den Key selbst holen
        if not key:
            raise ConfigError("GOOGLE_API_KEY missing")

        self.client = genai.Client(api_key=key)
        self.model = model

    # Gemini SDK => Single Prompt API -> models.generate_content()
    def infer(self, prompt: str) -> dict:
        for attempt in range(1, MAX_RETRIES + 1):
            t0 = time.time()
            try:
                resp = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=genai.types.GenerateContentConfig(
                        temperature=0.0,
                    )
                )

                return {
                    "ok": True,
                    "attempt": attempt,
                    "latency_s": round(time.time() - t0, 3),
                    "raw": _gemini_text(resp),
                    "token_out": None,
                    "error": None,
                    "error_type": None,
                    "retryable": None,
                    "retry_after_s": None,
                }

            except Exception as e:
                if attempt == MAX_RETRIES:
                    return {
                        "ok": False,
                        "attempt": attempt,
                        "latency_s": round(time.time() - t0, 3),
                        "raw": None,
                        "token_out": None,
                        "error": str(e)[:ERR_LIMIT],
                        "error_type": type(e).__name__,
                        "retryable": None,
                        "retry_after_s": None,
                    }

                sleep_backoff(attempt)


class OpenAIClient(OpenAIClientEasy):
    pass


class GeminiClient(GeminiClientEasy):
    pass
