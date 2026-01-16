import pandas as pd

# 1. Datensatz laden
# Nutze die lokale Datei, die du bereits hast
df = pd.read_parquet("test-00000-of-00001.parquet")

# 2. Fokus-Kategorien definieren 
target_categories = ['math', 'computer science', 'law', 'history']

# 3. Filtern mit .isin()
research_df = df[df['category'].isin(target_categories)].copy()

# 4. Spalten für dein Projekt hinzufügen [cite: 25, 26, 250, 251]
research_df['Model'] = None # Gemini / ChatGPT
research_df['Language'] = 'EN' # Später 'DE' nach DeepL-Durchlauf [cite: 246]
research_df['Answer'] = None
research_df['Correct'] = None # Binär 1/0 
research_df['Confidence_Self_Report'] = None # Was das Modell behauptet (0-100%) 

# --- ERGÄNZUNGEN FÜR TIEFERE ANALYSE ---
research_df['Logprobs_Confidence'] = None  # Die mathematische Wahrscheinlichkeit der API (Log-Likelihood)
research_df['Reasoning_Chain'] = None      # Der Denkprozess (Chain of Thought), um Halluzinationen zu finden [cite: 287]
research_df['Token_Usage'] = None         # Komplexität der Antwort
research_df['Latency_Sec'] = None         # Zeitaufwand (Performance-Indikator) 
research_df['Prompt_Version'] = "Zero-Shot-CoT" # Dokumentation der Methode
research_df['Ground_Truth'] = df['answer']

# 5. Zeichen-Check für DeepL
# Wir summieren die Länge der Fragen und aller Antwort-Optionen
total_chars = research_df['question'].str.len().sum()
print(f"Gesamtanzahl Zeichen für Fragen: {total_chars}")

# 6. Pilot-Set speichern (10 Fragen pro Kategorie)
pilot_set = research_df.groupby('category').sample(n=10, random_state=42)
pilot_set.to_csv("pilot_study_ready.csv", index=False)

print("\n--- Statistik ---")
print(research_df['category'].value_counts())
print("\n'pilot_study_ready.csv' wurde erstellt.")