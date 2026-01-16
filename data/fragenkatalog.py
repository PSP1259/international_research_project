import json
import numpy as np
import pandas as pd

url = "https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/resolve/main/data/test-00000-of-00001.parquet"
df = pd.read_parquet(url, engine="pyarrow")

def to_jsonable(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    if isinstance(x, np.ndarray):
        x = x.tolist()
    return x

df["options_json"] = df["options"].apply(lambda x: json.dumps(to_jsonable(x), ensure_ascii=False))

cols = ["question_id", "category", "question", "options_json", "answer", "answer_index", "src"]
df[cols].to_csv("mmlu_pro_all_questions.csv", index=False, encoding="utf-8")

summary = (
    df.groupby("category")
      .size()
      .reset_index(name="n_questions")
      .sort_values("n_questions", ascending=False)
)
summary.to_csv("mmlu_pro_category_overview.csv", index=False, encoding="utf-8")

print("CSV erzeugt:")
print("mmlu_pro_all_questions.csv")
print("mmlu_pro_category_overview.csv")
print(summary.head(20))











