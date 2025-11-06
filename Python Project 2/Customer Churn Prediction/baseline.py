import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv("data.csv", encoding="utf-8-sig")
df.columns = [c.strip().lower() for c in df.columns]
if "churn" not in df.columns:
    raise SystemExit("CSV needs a 'churn' column (Yes/No).")

y = df["churn"].map({"No":0, "Yes":1}).astype(int)
X = df.drop(columns=["churn"])

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
majority = int(ytr.value_counts().idxmax())
yp = [majority] * len(yte)
print(f"Baseline accuracy={accuracy_score(yte, yp):.3f} (predict all {majority})")
print("Test distribution:\n", yte.value_counts())