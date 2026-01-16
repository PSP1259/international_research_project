from pathlib import Path

from lib.io_jsonl import JsonlStore
from lib.mmlu import load_csv, parse_options, gold_letter


BASE_DIR = Path(__file__).resolve().parent

CSV_PATH = BASE_DIR.parent / "mmlu_pro_all_questions.csv"
INFER_JSONL = BASE_DIR / "mini_infer.jsonl"
OUT_JSONL = BASE_DIR / "mini_scored.jsonl"


def main():

    # 1) Lade die Modell Resultate (Predictions) aus mini_infer.jsonl
    infer_rows = JsonlStore(INFER_JSONL).read_all()

    # 2) Sammle alle question_ids, die in den Predictions vorkommen
    needed_ids = set()
    for row in infer_rows:                              # Loop für jede Prediction aus dem Log
        if "question_id" in row:
            needed_ids.add(int(row["question_id"]))     # ID Str -> Int

    # 3) Lade das CSV und baue eine Map: question_id -> gold_answer_letter
    df = load_csv(CSV_PATH)
    gold_by_qid = {}

    for _, r in df.iterrows():             # Loop für jede Zeile des df CSV
        qid = int(r["question_id"])        # IDs sammeln

        # Wenn diese Frage nicht im Set der Predictions ist -> überspringen
        if qid not in needed_ids:
            continue

        options = parse_options(r["options_json"])              # JSON str -> Python Liste
        answer_index = int(r["answer_index"])                   # Option als Zahl korrekt?
        gold_by_qid[qid] = gold_letter(answer_index, options)   # Speichert Buchstaben in die Map       

    # 4) Löscht altes scored File, erstellt neues
    if OUT_JSONL.exists():
        OUT_JSONL.unlink()
    out_store = JsonlStore(OUT_JSONL)

    # 5) Jede Prediction bewerten und speichern
    for row in infer_rows:
        qid = int(row["question_id"])
        pred = row.get("answer")
        gold = gold_by_qid.get(qid)
        row["label_answer_letter"] = gold

        if gold is None or pred is None:
            row["correct"] = None
        else:
            row["correct"] = int(pred == gold)  # Vergleicht Prediction mit Gold: 1 oder 0

        out_store.append(row)                   # Schreibt die fertige Zeile ins JSONL File

    print("Wrote:", OUT_JSONL)                  # Informativ fürs Terminal 


if __name__ == "__main__":
    main()




