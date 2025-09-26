import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Load datasets
true_df = pd.read_csv("True.csv")
fake_df = pd.read_csv("Fake.csv")

# Add labels
true_df["label"] = 1
fake_df["label"] = 0

# Combine
df = pd.concat([true_df, fake_df]).reset_index(drop=True)

# Use TF-IDF for embeddings
vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
X = vectorizer.fit_transform(df["text"])

# Save embeddings + metadata
np.save("embeddings.npy", X.toarray())
df.to_csv("articles.csv", index=False)
print("✅ Embeddings and articles saved!")
