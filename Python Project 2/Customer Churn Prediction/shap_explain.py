# shap_explain.py
import pandas as pd
import matplotlib.pyplot as plt
from joblib import load
import shap
from pathlib import Path
import numpy as np
import scipy.sparse as sp

# ---- load model (Pipeline with 'pre' and 'model') ----
pipe = load("models/churn_model.joblib")
pre  = pipe.named_steps["pre"]      # ColumnTransformer (already FITTED during training)
clf  = pipe.named_steps["model"]    # LogisticRegression

# ---- load some data to explain (features only) ----
df = pd.read_csv("data.csv", encoding="utf-8-sig")
# find the churn column case-insensitively and drop it
churn_col = next(c for c in df.columns if c.strip().lower() == "churn")
X = df.drop(columns=[churn_col])

# IMPORTANT: use the fitted preprocessor -> transform (NOT fit_transform)
X_trans = pre.transform(X)

# ---- feature names that align with the trained model ----
num_cols = pre.transformers_[0][2]  # numeric feature names
oh       = pre.transformers_[1][1].named_steps["onehot"]
cat_cols = pre.transformers_[1][2]
oh_names = oh.get_feature_names_out(cat_cols)
feature_names = list(num_cols) + list(oh_names)

# ---- pick a small background set for SHAP (same transformed space) ----
if sp.issparse(X_trans):
    X_bg = X_trans[:min(100, X_trans.shape[0])]
else:
    X_bg = X_trans[:min(100, X_trans.shape[0]), :]

# Use a masker (modern API) to avoid deprecation warning
masker = shap.maskers.Independent(X_bg)

# Linear model + transformed inputs -> LinearExplainer is appropriate
explainer = shap.LinearExplainer(clf, masker)

# Get SHAP values for all rows we passed in (in transformed space)
shap_values = explainer.shap_values(X_trans)

# ---- plots ----
Path("runs").mkdir(exist_ok=True)

# Summary (global) plot
# Try new API first; fall back to classic summary_plot signature.
try:
    expl = shap.Explanation(values=shap_values,
                            base_values=explainer.expected_value,
                            data=X_trans,
                            feature_names=feature_names)
    shap.plots.beeswarm(expl, max_display=20, show=False)
except Exception:
    shap.summary_plot(shap_values, features=X_trans,
                      feature_names=feature_names, show=False)

plt.tight_layout()
plt.savefig("runs/shap_summary.png")
plt.close()

# Single-row force plot (row 0)
try:
    shap.initjs()
    fp = shap.force_plot(explainer.expected_value, shap_values[0, :],
                         matplotlib=True)
    plt.savefig("runs/shap_force_row0.png")
    plt.close()
except Exception:
    pass

print("Saved: runs/shap_summary.png", "and (if supported) runs/shap_force_row0.png")