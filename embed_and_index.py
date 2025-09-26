import json, pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from pathlib import Path

MODEL_NAME = "all-MiniLM-L6-v2"
EMB_DIM = 384

def load_corpus(path="corpus.jsonl"):
    return [json.loads(line) for line in open(path, encoding="utf8")]

if __name__ == "__main__":
    model = SentenceTransformer(MODEL_NAME)
    docs = load_corpus("corpus.jsonl")
    texts = [(d["title"] or "") + " . " + (d["text"] or "") for d in docs]

    embs = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    faiss.normalize_L2(embs)

    index = faiss.IndexFlatIP(EMB_DIM)
    index.add(embs)

    np.save("embeddings.npy", embs)
    with open("meta.pkl", "wb") as f:
        pickle.dump(docs, f)

    print("Indexed", len(docs))
