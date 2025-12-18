from joblib import load
import numpy as np

pipe = load("model.joblib")
vec = pipe.named_steps["tfidf"]
clf = pipe.named_steps["clf"]

if not hasattr(clf, "coef_"):
    raise SystemExit("Use a linear model (e.g., LogisticRegression).")

feature_names = np.array(vec.get_feature_names_out())
coefs = clf.coef_[0]      # binary: 1 vector
k = 15
top_fake = feature_names[np.argsort(coefs)[:k]]            # most negative
top_real = feature_names[np.argsort(coefs)[-k:]][::-1]     # most positive

print("Top terms for FAKE:\n", ", ".join(top_fake))
print("\nTop terms for REAL:\n", ", ".join(top_real))