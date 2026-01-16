import os
import json
import logging
import math

from google import genai
from google.genai import types
from openai import OpenAI
from dotenv import load_dotenv


# Einheitliches Logging-Verhalten
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AIClient")


# Universal-Client
class AIClient:

  """
  - Konstruktor: Wird aufgerufen sobald ein neues Objekt erstellt wird.

  - *Args: Not sure how many arguments might be passed to a function
      - provider (str): "google" oder "openai"
      - model_name (str): z.B. "chatgpt-5.2" oder "gemini-2.0-flash"
      - is_reasoning (bool): Wenn True -> aktivieren
  """

  def __init__(self, provider: str, model_name: str, is_reasoning: bool = False):

    # Ladet die API Keys neu
    load_dotenv()

    # Speichert die Einstellungen in "self"
    self.provider = provider.lower()
    self.model_name = model_name
    self.is_reasoning = is_reasoning
    self._google_logprobs_supported = True

    # Setup der Clients
    if self.provider == "google":
      api_key = os.getenv("GOOGLE_API_KEY")
      if not api_key:
        raise ValueError("CRITICAL: GOOGLE_API_KEY fehlt in .env!")
      # HttpOptions.timeout is in milliseconds.
      http_options = types.HttpOptions(timeout=60000)
      self.client = genai.Client(api_key=api_key, http_options=http_options)
      # Gemini 3 preview models do not support logprobs in this setup.
      if self.model_name in {"gemini-3-flash-preview", "gemini-3-pro-preview"}:
        self._google_logprobs_supported = False

    elif self.provider == "openai":
      api_key = os.getenv("OPENAI_API_KEY")
      if not api_key:
        raise ValueError("CRITICAL: OPENAI_API_KEY fehlt in .env!")
      self.client = OpenAI(api_key=api_key)

    else:
      raise ValueError(f"Unbekannter Provider: {provider}")

    logger.info("Client gestartet: %s - %s", self.provider.upper(), self.model_name)


  def infer(self, prompt: str):
    try:
      if self.provider == "google":
        return self._call_google(prompt)
      elif self.provider == "openai":
        return self._call_openai(prompt)
    except Exception as e:
      logger.error(f"Fehler im API Call: {e}")
      return None, None
    

  def _calculate_prob(self, logprob_val):
  # Hilfsfunktion: Wandelt logprob (-0.1) in Prozent (90.4) um
    try:
      if logprob_val is None: return None
      # e hoch logprob = Wahrscheinlichkeit (0.0 bis 1.0)
      return math.exp(logprob_val) * 100
    except:
      return None


  def _extract_top_choices(self, target_char, logprobs_data, provider):
    """
    Sucht den Token, der die Antwort (A, B, C...) enthält, und holt dessen Konkurrenten.
    """
    if not target_char or not logprobs_data:
      return []

    found_choices = []

    def _clean_token_text(text):
      if text is None:
        return None
      t = str(text).strip()
      if len(t) >= 2 and t[0] == t[-1] and t[0] in ("\"", "'"):
        t = t[1:-1]
      return t

    def _candidate_text(candidate):
      if candidate is None:
        return None
      for attr in ("token", "text"):
        val = getattr(candidate, attr, None)
        if isinstance(val, str) and val.strip():
          return val
      content = getattr(candidate, "content", None)
      parts = getattr(content, "parts", None)
      if parts and len(parts) > 0:
        text = getattr(parts[0], "text", None)
        if isinstance(text, str) and text.strip():
          return text
      token_id = getattr(candidate, "token_id", None)
      if isinstance(token_id, str) and token_id.strip():
        return token_id
      return None
    
    
    
    # --- OPENAI LOGIC ---

    try:
      if provider == "openai":
        # logprobs_data ist hier: response.choices[0].logprobs.content
        for token_data in logprobs_data:
          # Bereinigen (Leerzeichen entfernen) und vergleichen
          clean_token = token_data.token.strip()
          
          # Wenn wir den Antwort-Buchstaben finden (z.B. "A")
          if clean_token == target_char:
            # Top-Logprobs (die Alternativen) für DIESEN Token
            if token_data.top_logprobs:
              for alt in token_data.top_logprobs:
                prob_percent = math.exp(alt.logprob) * 100
                found_choices.append({
                  "token": alt.token,
                  "prob": round(prob_percent, 2)
                })
            break # Haben ihn gefunden, Loop beenden

    # --- GOOGLE LOGIC ---
      elif provider == "google":
        # Google liefert top_candidates pro Schritt. 
        # wo der gewählte Kandidat unser Buchstabe ist.
        # logprobs_data ist hier: response.candidates[0].logprobs_result
        
        chosen_tokens = logprobs_data.chosen_candidates
        top_candidates_list = logprobs_data.top_candidates

        for i, chosen in enumerate(chosen_tokens):
          chosen_text = _candidate_text(chosen)
          if not chosen_text:
            continue
          if _clean_token_text(chosen_text) != target_char:
            continue

          if len(top_candidates_list) > i:
            top_entry = top_candidates_list[i]
            candidates = getattr(top_entry, "candidates", None) or top_entry
            if candidates:
              for cand in candidates:
                cand_text = _candidate_text(cand)
                logp = getattr(cand, "log_probability", None)
                if cand_text is None or logp is None:
                  continue
                p = math.exp(logp) * 100
                found_choices.append({
                  "token": cand_text,
                  "prob": round(p, 2)
                })
              break
    except Exception as e:
      logger.warning(f"Konnte Top-Choices nicht extrahieren: {e}")
      
    return found_choices           








  # ------------------ GOOGLE ----------------------

  # Gemini SDK => Single Prompt API -> models.generate_content()
  def _call_google(self, prompt):
    use_logprobs = self._google_logprobs_supported
    if use_logprobs:
      config = types.GenerateContentConfig(
        temperature=0.0,                              # Deterministisch
        response_mime_type="application/json",
        response_logprobs=True,
        logprobs=4
      )
    else:
      config = types.GenerateContentConfig(
        temperature=0.0,
        response_mime_type="application/json"
      )
    try:
      response = self.client.models.generate_content(
        model=self.model_name,
        contents=prompt,
        config=config
      )




      # 1. Antwort Parsen
      json_resp = json.loads(response.text)
      answer_char = json_resp.get("answer_char")



      # 2. Logprobs extrahieren (Google Logic)
      # Wir nehmen den Durchschnitt der Logprobs aller Tokens im Output
      avg_logprob = None
      top_choices_extracted = []
      if use_logprobs and response.candidates and response.candidates[0].logprobs_result:
        # Hole alle logprobs der gewählten Tokens
        tokens = response.candidates[0].logprobs_result.chosen_candidates
        if tokens:
          # Durchschnitt berechnen
          vals = [t.log_probability for t in tokens if t.log_probability is not None]
          if vals:
            avg_logprob = sum(vals) / len(vals)

        top_choices_extracted = self._extract_top_choices(
          answer_char,
          response.candidates[0].logprobs_result,
          "google"
        )

      # 3. Usage
      usage = {
        "input_tokens": response.usage_metadata.prompt_token_count if response.usage_metadata else 0,
        "output_tokens": response.usage_metadata.candidates_token_count if response.usage_metadata else 0,
        "raw_logprob": avg_logprob,
        "prob_percent": self._calculate_prob(avg_logprob),
        "top_choices": top_choices_extracted
      }

      return json_resp, usage

    except Exception as e:
      if use_logprobs and ("Logprobs is not enabled" in str(e) or "logprobs" in str(e)):
        logger.warning("Google Logprobs nicht unterstuetzt. Deaktiviere fuer diesen Lauf.")
        self._google_logprobs_supported = False
        return self._call_google(prompt)

      logger.error(f"Google Fehler: {e}")
      return None, None





  # ------------------ OpenAI ----------------------
  
  # OpenAI SDK => Role Based Prompting (System - User) -> chat.completions.create()

  def _call_openai(self, prompt):

    # 1. Basis-Parameter
    params = {
      "model": self.model_name,
      "messages": [{"role": "user", "content": prompt}],
      "response_format": {"type": "json_object"},
      "logprobs": True,
      "top_logprobs": 4
    }

    # 2. Reasoning True/False?
    if self.is_reasoning:
      # ---FALL A: REASONING AKTIV---
      # Temperature ist hier VERBOTEN (fuehrt zu API Error).
      # none (default), low, medium, high, xhigh
      params["reasoning_effort"] = "xhigh"

    else:
      # ---FALL B: STANDARD / KEIN REASONING---
      # Das gilt fuer GPT-4.1 UND fuer GPT-5.2 im Modus "none".
      # Hier MUESSEN wir Temperature 0 setzen fuer reproduzierbare Ergebnisse.
      params["temperature"] = 0.0

      # Speziell fuer GPT-5.2: Wenn wir 'is_reasoning=False' haben,
      # sagen wir der API explizit: "Bitte nicht nachdenken".
      if self.model_name.startswith("gpt-5"):
        params["reasoning_effort"] = "none"

    # 3. Der Call
    try:
      # **params entpackt das Dictionary automatisch in die richtigen Argumente
      response = self.client.chat.completions.create(**params)

      # 1. Antwort extrahieren & JSON Parsen
      content = response.choices[0].message.content
      json_response = json.loads(content) # Wir brauchen das JSON jetzt schon!
      answer_char = json_response.get("answer_char") # Wonach wir suchen (z.B. "A")

      # 2. Logprobs Analyse
      avg_logprob = None
      top_choices_extracted = [] # <--- NEU
      
      if response.choices[0].logprobs and response.choices[0].logprobs.content:
        tokens = response.choices[0].logprobs.content
        
        # Durchschnitt (wie bisher)
        vals = [t.logprob for t in tokens if t.logprob is not None]
        if vals: avg_logprob = sum(vals) / len(vals)

        # <--- NEU: Die Alternativen holen
        top_choices_extracted = self._extract_top_choices(answer_char, tokens, "openai")


      # 3. Usage
      u = response.usage
      usage = {
        "input_tokens": u.prompt_tokens if u else 0,
        "output_tokens": u.completion_tokens if u else 0,
        "raw_logprob": avg_logprob,
        "prob_percent": self._calculate_prob(avg_logprob),
        "top_choices": top_choices_extracted  # <--- HIER SPEICHERN
      }

      return json_response, usage



    except Exception as e:
    # Fallback: Wenn Logprobs nicht erlaubt sind 
      if "logprobs" in str(e) and params.get("logprobs"):
        logger.warning("Logprobs nicht unterstuetzt. Versuche ohne...")
        params.pop("logprobs")
        params.pop("top_logprobs")
        try:
          response = self.client.chat.completions.create(**params)
          content = response.choices[0].message.content
          u = response.usage
          usage = {
            "input_tokens": u.prompt_tokens if u else 0,
            "output_tokens": u.completion_tokens if u else 0,
            "raw_logprob": None,
            "prob_percent": None
          }
          return json.loads(content), usage
        except Exception as e2:
          logger.error(f"OpenAI Retry failed: {e2}")
          return None, None
            
      logger.error(f"OpenAI Fehler: {e}")
      return None, None
