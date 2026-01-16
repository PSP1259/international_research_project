"""
1. Schritt in Powershell (im Python)
C:/Users/psp/AppData/Local/Python/pythoncore-3.14-64/python.exe -m pip install python-dotenv openai google-generativeai
"""

# 2. Virtuelle / externe Umgebung freischalten
import os                      # Importiert os-Modul für .env
from dotenv import load_dotenv # Impotiert Funktion für .env

# 3. Lädt die Variablen aus der .env Datei in die Umgebung
load_dotenv()

# 4. Keys abrufen
openai_key = os.getenv("OPENAI_API_KEY")
google_key = os.getenv("GOOGLE_API_KEY")
deepl_key = os.getenv("DEEPL_API_KEY")

# 5 Test-Ausgabe (Sicherheits-Check: Nur die ersten 4 Zeichen zeigen)
print(f"OpenAI Key geladen: {openai_key[:4] if openai_key else 'NICHT GEFUNDEN'}...")
print(f"Google Key geladen: {google_key[:4] if google_key else 'NICHT GEFUNDEN'}...")
print(f"DeepL Key geladen: {deepl_key[:4] if deepl_key else 'NICHT GEFUNDEN'}...")
