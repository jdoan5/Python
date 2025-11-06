# train.py (hardened)
import json, re, numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, roc_curve, auc, average_precision_score
import matplotlib.pyplot as plt
from joblib import dump

# ---------- load & clean ----------
df = pd.read_csv("data.csv", encoding="utf-8-sig")
# normalize column names
df.columns = [c.strip() for c in df.columns]
lc = {c: c.strip().lower() for c in df.columns}

# locate churn column (case-insensitive)
churn_col = None
for c in df.columns:
    if lc[c] == "churn":
        churn_col = c
        break
if churn_col is None:
    raise SystemExit("No 'churn' column found. Please ensure your CSV has a 'churn' column.")

# robust map of churn labels
vals = df[churn_col].astype(str).str.strip().str.lower()
label_map = {
    "no":0, "yes":1,
    "0":0, "1":1,
    "false":0, "true":1,
    "n":0, "y":1
}
y = vals.map(label_map)
if y.isna().any():
    bad = sorted(set(vals[y.isna()]))
    raise SystemExit(f"Unknown churn values: {bad}. "
                     f"Add to label_map or clean the CSV.")

# drop target from features
X = df.drop(columns=[churn_col])

# try to coerce obvious numeric-looking columns (e.g., "89.10", "1,234", "$99")
for c in X.columns:
    if X[c].dtype == object:
        sample = X[c].astype(str).str.replace(r"[,$% ]", "", regex=True)
        if sample.str.match(r"^-?\d+(\.\d+)?$").mean() > 0.9:
            X[c] = pd.to_numeric(sample, errors="coerce")

num_cols = X.select_dtypes(include=["number"]).columns.tolist()
cat_cols = X.columns.difference(num_cols).tolist()

# ---------- pipeline ----------
pre = ColumnTransformer([
    ("num", Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler())
    ]), num_cols),
    ("cat", Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]), cat_cols)
])

clf = Pipeline([
    ("pre", pre),
    ("model", LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear"))
])

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

clf.fit(Xtr, ytr)

proba = clf.predict_proba(Xte)[:,1]
pred50 = (proba >= 0.5).astype(int)

print("ROC-AUC:", round(roc_auc_score(yte, proba), 3))
print("\nReport @ 0.50 threshold:")
print(classification_report(yte, pred50, digits=3, zero_division=0))

# ---------- cost-based threshold ----------
C_FN, C_FP = 5.0, 1.0   # tune for your context
def expected_cost(t):
    yp = (proba >= t).astype(int)
    fn = ((yp == 0) & (yte == 1)).sum()
    fp = ((yp == 1) & (yte == 0)).sum()
    return C_FN*fn + C_FP*fp

grid = np.linspace(0.05, 0.95, 19)
costs = np.array([expected_cost(t) for t in grid])
best_t = float(grid[np.argmin(costs)])

pred_best = (proba >= best_t).astype(int)
print(f"\nChosen threshold={best_t:.2f} (min expected cost)")
print(classification_report(yte, pred_best, digits=3, zero_division=0))

# ---------- plots ----------
Path("runs").mkdir(exist_ok=True, parents=True)
fpr, tpr, _ = roc_curve(yte, proba)
plt.figure(); plt.plot(fpr, tpr); plt.plot([0,1],[0,1], "--")
plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC (AUC={auc(fpr,tpr):.3f})")
plt.tight_layout(); plt.savefig("runs/roc_curve.png"); plt.close()

prec, rec, _ = precision_recall_curve(yte, proba)
ap = average_precision_score(yte, proba)
plt.figure(); plt.plot(rec, prec)
plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"PR (AP={ap:.3f})")
plt.tight_layout(); plt.savefig("runs/pr_curve.png"); plt.close()

# ---------- save ----------
Path("models").mkdir(exist_ok=True)
dump(clf, "models/churn_model.joblib")
Path("runs/threshold.json").write_text(json.dumps({"threshold": best_t}, indent=2))
print("\nSaved: models/churn_model.joblib, runs/threshold.json, runs/*_curve.png")