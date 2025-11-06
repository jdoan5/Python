import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("data.csv")
X = df["text"].astype(str)
y = df["label"].astype(str)

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1,2))),
    ("clf", LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced"))
])
# pipe = Pipeline([
#     ("tfidf", TfidfVectorizer(stop_words="english")),
#     ("clf", LogisticRegression(max_iter=2000))
# ])

pipe.fit(X_tr, y_tr)
y_pr = pipe.predict(X_te)

print("Accuracy:", round(accuracy_score(y_te, y_pr), 3))
print(classification_report(y_te, y_pr))

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

fig, ax = plt.subplots()
ConfusionMatrixDisplay.from_predictions(y_te, y_pr, ax=ax)
ax.set_title("Confusion matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
print("Saved: confusion_matrix.png")