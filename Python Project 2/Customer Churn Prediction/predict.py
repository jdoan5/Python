import json, argparse, pandas as pd
from joblib import load
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--json", help="Inline JSON with a single customer row")
ap.add_argument("--file", help="CSV with rows to score; outputs predictions.csv unless --out given")
ap.add_argument("--out", help="Output path for CSV scoring")
args = ap.parse_args()

model = load("models/churn_model.joblib")
t = 0.5
th_path = Path("runs/threshold.json")
if th_path.exists():
    t = json.loads(th_path.read_text())["threshold"]

def score_df(df):
    proba = model.predict_proba(df)[:,1]
    pred = (proba >= t).astype(int)
    return proba, pred

if args.json:
    row = pd.DataFrame([json.loads(args.json)])
    p, y = score_df(row)
    print(json.dumps({"proba": float(p[0]), "pred": int(y[0]), "threshold": t}, indent=2))
elif args.file:
    df = pd.read_csv(args.file)
    p, y = score_df(df)
    df["churn_proba"] = p
    df["churn_pred"] = y
    out = Path(args.out or "predictions.csv")
    df.to_csv(out, index=False)
    print(f"Wrote {out}")
else:
    ap.error("Provide --json or --file")