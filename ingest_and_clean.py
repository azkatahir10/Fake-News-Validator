import pandas as pd
import json
from pathlib import Path

def load_and_label(fake_path="Fake.csv", true_path="True.csv"):
    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)

    fake_df["label"] = "FAKE"
    true_df["label"] = "REAL"

    df = pd.concat([fake_df, true_df], ignore_index=True)
    df = df[["title", "text", "label"]].dropna()
    return df

def save_jsonl(df, out_path="corpus.jsonl"):
    with open(out_path, "w", encoding="utf8") as f:
        for i, row in df.iterrows():
            obj = {
                "id": f"doc_{i}",
                "title": str(row["title"])[:300],
                "text": str(row["text"]),
                "label": row["label"]
            }
            f.write(json.dumps(obj) + "\n")

if __name__ == "__main__":
    df = load_and_label("Fake.csv", "True.csv")
    save_jsonl(df, "corpus.jsonl")
    print("Saved corpus.jsonl with", len(df), "docs")
