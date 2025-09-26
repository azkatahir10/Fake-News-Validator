import pandas as pd
import numpy as np
from fastapi import FastAPI, Query
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize FastAPI app
app = FastAPI(title="Fake News Validator API")

# Load embeddings + dataset
print("🔄 Loading embeddings and dataset...")
embs = np.load("embeddings.npy")
articles = pd.read_csv("articles.csv")

# Recreate vectorizer so we can embed queries
vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
vectorizer.fit(articles["text"])
print("✅ Data loaded successfully!")

# Pydantic model for request
class QueryRequest(BaseModel):
    query: str


@app.get("/")
def root():
    return {"message": "Fake News Validator Backend running!"}


@app.post("/query")
def query_news(request: QueryRequest):
    query = request.query

    # Embed query
    q_vec = vectorizer.transform([query]).toarray()

    # Compute cosine similarities
    sims = cosine_similarity(q_vec, embs)[0]

    # Get top 5 most similar articles
    top_idx = sims.argsort()[-5:][::-1]
    top_articles = articles.iloc[top_idx]

    # Count true vs fake
    true_count = int(sum(top_articles["label"] == 1))
    fake_count = int(sum(top_articles["label"] == 0))

    if true_count > fake_count:
        verdict = "Likely True"
        confidence = true_count / len(top_articles)
    elif fake_count > true_count:
        verdict = "Likely Fake"
        confidence = fake_count / len(top_articles)
    else:
        verdict = "Uncertain"
        confidence = 0.5

    return {
        "verdict": verdict,
        "confidence": round(confidence * 100, 2),
        "checked_articles": len(top_articles),
        "articles": top_articles[["title", "text", "label"]].to_dict(orient="records")
    }
