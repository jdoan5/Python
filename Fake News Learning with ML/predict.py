import argparse
from joblib import load

parser = argparse.ArgumentParser()
parser.add_argument("--text", required=True)
args = parser.parse_args()

pipe = load("model.joblib")
pred = pipe.predict([args.text])[0]
proba = None
if hasattr(pipe.named_steps["clf"], "predict_proba"):
    proba = pipe.predict_proba([args.text])[0].max()

print(f"Prediction: {pred}" + (f" (confidence ~ {proba:.2f})" if proba is not None else ""))