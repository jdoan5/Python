import pandas as pd
from joblib import load
from sklearn.inspection import permutation_importance

pipe = load("models/churn_model.joblib")
pre  = pipe.named_steps["pre"]
clf  = pipe.named_steps["model"]

df = pd.read_csv("data.csv", encoding="utf-8-sig")
y = df["churn"].astype(str).str.strip().str.lower().map({"no":0,"yes":1,"0":0,"1":1})
X = df.drop(columns=[c for c in df.columns if c.lower()=="churn"][0])

Xtr = pre.transform(X)
r = permutation_importance(clf, Xtr, y, n_repeats=20, random_state=42, n_jobs=-1)

# feature names
num_cols = pre.transformers_[0][2]
oh = pre.transformers_[1][1].named_steps["onehot"]
cat_cols = pre.transformers_[1][2]
feat_names = list(num_cols) + list(oh.get_feature_names_out(cat_cols))

imp = pd.DataFrame({"feature": feat_names, "importance": r.importances_mean}) \
        .sort_values("importance", ascending=False).head(20)
print(imp)
imp.to_csv("runs/perm_importance.csv", index=False)
print("Saved runs/perm_importance.csv")