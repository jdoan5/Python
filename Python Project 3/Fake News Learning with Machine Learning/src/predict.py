#!/usr/bin/env python3
"""
predict.py — Fake News Learning with Machine Learning

Predict Fake/Real for:
- a single text (--text)
- or an input CSV (--input) and write an output CSV (--output)

Outputs:
- pred_label: 0=Real, 1=Fake
- pred_proba_fake: probability of class=1 (Fake), if available
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from joblib import load


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Predict Fake News labels using a trained model.")
    p.add_argument("--model", required=True, help="Path to model .joblib created by train.py")
    p.add_argument("--text", default=None, help="Single text input to score.")
    p.add_argument("--input", default=None, help="Input CSV path to score (must include text column).")
    p.add_argument("--output", default=None, help="Output CSV path (required if --input is used).")
    p.add_argument("--text-col", default="text", help="Text column in input CSV. Default: text")
    return p.parse_args()


def score_texts(model, texts: pd.Series) -> pd.DataFrame:
    preds = model.predict(texts)
    out = pd.DataFrame({"pred_label": preds})

    if hasattr(model, "predict_proba"):
        out["pred_proba_fake"] = model.predict_proba(texts)[:, 1]
    return out


def main() -> None:
    args = parse_args()
    model = load(args.model)

    if args.text:
        s = pd.Series([args.text], name="text")
        out = score_texts(model, s)
        label = int(out.loc[0, "pred_label"])
        print(f"pred_label={label} ({'FAKE' if label==1 else 'REAL'})")
        if "pred_proba_fake" in out.columns:
            print(f"pred_proba_fake={float(out.loc[0, 'pred_proba_fake']):.4f}")
        return

    if args.input:
        if not args.output:
            raise SystemExit("--output is required when using --input.")

        inp = Path(args.input)
        df = pd.read_csv(inp)
        if args.text_col not in df.columns:
            raise SystemExit(f"Missing text column '{args.text_col}'. Found: {list(df.columns)}")

        texts = df[args.text_col].astype(str).fillna("")
        scored = score_texts(model, texts)
        out_df = pd.concat([df, scored], axis=1)
        out_df.to_csv(args.output, index=False)
        print(f"Wrote: {args.output}")
        return

    raise SystemExit("Provide either --text or --input.")


if __name__ == "__main__":
    main()
