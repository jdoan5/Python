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

import json
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report

# 1) Get probabilities for the FAKE class
#    (works for LogisticRegression; for LinearSVC switch to CalibratedClassifierCV or skip thresholding)
clf = pipe.named_steps["clf"]
classes = list(clf.classes_)            # e.g., ['FAKE', 'REAL'] or ['REAL','FAKE']
fake_idx = classes.index("FAKE")        # we treat FAKE as the "positive" class for cost

proba_fake = pipe.predict_proba(X_te)[:, fake_idx]

# 2) Baseline at 0.50 threshold
y_pred_50 = np.where(proba_fake >= 0.50, "FAKE", "REAL")
cm_50 = confusion_matrix(y_te, y_pred_50, labels=["REAL","FAKE"])  # rows=actual, cols=pred
tn50, fp50, fn50, tp50 = cm_50.ravel()
recall50 = tp50 / (tp50 + fn50) if (tp50 + fn50) else 0.0

# 3) Cost-aware threshold search (penalize accusing REAL as FAKE more)
C_FP = 5.0   # cost of falsely accusing REAL (pred=FAKE, actual=REAL)
C_FN = 1.0   # cost of missing FAKE (pred=REAL, actual=FAKE)
grid = np.linspace(0.05, 0.95, 19)

def expected_cost(th):
    yp = np.where(proba_fake >= th, "FAKE", "REAL")
    cm = confusion_matrix(y_te, yp, labels=["REAL","FAKE"])
    tn, fp, fn, tp = cm.ravel()
    return C_FP*fp + C_FN*fn

best_t = float(grid[np.argmin([expected_cost(t) for t in grid])])

# 4) Metrics at best threshold
y_pred_best = np.where(proba_fake >= best_t, "FAKE", "REAL")
cm_best = confusion_matrix(y_te, y_pred_best, labels=["REAL","FAKE"])
tnb, fpb, fnb, tpb = cm_best.ravel()
recall_best = tpb / (tpb + fnb) if (tpb + fnb) else 0.0

# 5) Compute the % reduction in false accusations of REAL (FP drop)
if fp50 > 0:
    fp_reduction_pct = 100.0 * (fp50 - fpb) / fp50
else:
    fp_reduction_pct = 0.0

print(f"Cost-aware threshold = {best_t:.2f}")
print(f"False accusations of REAL (FP): {fp50} -> {fpb}  (reduction {fp_reduction_pct:.1f}%)")
print(f"Recall (FAKE): {recall50:.3f} -> {recall_best:.3f}")

# 6) Save a small metrics file you can cite on your resume
Path("runs").mkdir(exist_ok=True)
metrics = {
    "threshold_default": 0.50,
    "threshold_cost_aware": best_t,
    "false_positives_default": int(fp50),
    "false_positives_cost_aware": int(fpb),
    "false_positive_reduction_pct": round(fp_reduction_pct, 1),
    "recall_fake_default": round(recall50, 3),
    "recall_fake_cost_aware": round(recall_best, 3),
    "cost_weights": {"FP": C_FP, "FN": C_FN},
}
Path("runs/metrics.json").write_text(json.dumps(metrics, indent=2))
print("Wrote runs/metrics.json")