from lib.ai_client import AIClient
import os

# 1. Test: Google
print("\n--- Test 1: Google Gemini ---")
try:
    bot = AIClient(provider="google", model_name="gemini-2.0-flash")
    # Einfacher Test-Prompt
    res = bot.infer("Gib mir ein JSON: {'status': 'Google läuft'}")
    print(f"✅ Ergebnis: {res}")
except Exception as e:
    print(f"❌ Fehler: {e}")

# 2. Test: OpenAI Standard (GPT-4o-mini oder dein 4.1)
# Nimm hier ein Modell, das du sicher hast (z.B. gpt-4o-mini zum Testen des Codes)
print("\n--- Test 2: OpenAI Standard (Kein Reasoning) ---")
try:
    # is_reasoning=False -> Sollte temperature=0 nutzen
    bot = AIClient(provider="openai", model_name="gpt-4o-mini", is_reasoning=False)
    res = bot.infer("Gib mir ein JSON: {'status': 'OpenAI Standard läuft'}")
    print(f"✅ Ergebnis: {res}")
except Exception as e:
    print(f"❌ Fehler: {e}")

# 3. Test: OpenAI Reasoning (Wenn du Zugriff auf o1/o3/gpt-5 hast)
# Falls du noch keinen Zugriff hast, wird das hier crashen -> das ist OK!
print("\n--- Test 3: OpenAI Reasoning ---")
try:
    # is_reasoning=True -> Sollte reasoning_effort nutzen
    # ACHTUNG: Nutze hier 'o1-mini' oder 'o3-mini' falls gpt-5.2 noch nicht public ist
    # Zum Code-Testen reicht auch ein Fake-Name, um zu sehen ob der Parameter gesetzt wird
    bot = AIClient(provider="openai", model_name="o3-mini", is_reasoning=True)
    res = bot.infer("Löse: 2+2. JSON {'result': 4}")
    print(f"✅ Ergebnis: {res}")
except Exception as e:
    print(f"⚠️ Hinweis (erwartet bei fehlendem Zugriff): {e}")