import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from joblib import dump

df = pd.read_csv("data.csv")
X = df["text"].astype(str)
y = df["label"].astype(str)

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1,2))),
    ("clf", LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced"))
])

pipe.fit(X_tr, y_tr)
dump(pipe, "model.joblib")
print("Saved model.joblib")