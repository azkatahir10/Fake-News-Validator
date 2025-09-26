import numpy as np, pickle, joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

embs = np.load("embeddings.npy")
with open("meta.pkl", "rb") as f:
    docs = pickle.load(f)

labels = np.array([1 if d["label"] == "REAL" else 0 for d in docs])
X_train, X_test, y_train, y_test = train_test_split(embs, labels, test_size=0.2, random_state=42)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred))

joblib.dump(clf, "clf.joblib")
print("Classifier saved")
