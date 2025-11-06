import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv("data.csv")
X = df["text"].astype(str)
y = df["label"].astype(str)

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

majority = y_tr.value_counts().idxmax()
y_pred = [majority] * len(y_te)

acc = accuracy_score(y_te, y_pred)
print(f"Majority baseline: predict '{majority}' → accuracy={acc:.3f}")
print("Test labels:\n", y_te.value_counts())