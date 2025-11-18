#!/usr/bin/env python3
"""
review_predictions.py — quick summary for predictions.csv (flexible column names)
Accepts either:
- pred_label / pred_proba_yes
- prediction / probability
"""
import argparse
from pathlib import Path
import sys
import pandas as pd

def pick_cols(df):
    # Try standard names first
    candidates = [
        ("pred_label", "pred_proba_yes"),
        ("prediction", "probability"),
    ]
    for lab, proba in candidates:
        if lab in df.columns and proba in df.columns:
            return lab, proba
    return None, None

def main():
    ap = argparse.ArgumentParser(description="Summarize a predictions.csv file (label counts & probability stats).")
    ap.add_argument("--file", default="predictions.csv", help="Path to predictions CSV (default: predictions.csv)")
    ap.add_argument("--out", help="Optional: write a small Markdown report to this path")
    ap.add_argument("--top", type=int, default=5, help="Rows to show from the top (default: 5)")
    args = ap.parse_args()

    path = Path(args.file)
    if not path.exists():
        print(f"[ERROR] file not found: {path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(path)

    label_col, proba_col = pick_cols(df)
    if not label_col:
        print(f"[ERROR] predictions file is missing expected columns.", file=sys.stderr)
        print(f"Looked for one of: ('pred_label','pred_proba_yes') or ('prediction','probability')", file=sys.stderr)
        print(f"Found columns: {list(df.columns)}", file=sys.stderr)
        sys.exit(1)

    head = df[[label_col, proba_col]].head(args.top)
    counts = df[label_col].value_counts(dropna=False).rename_axis("label").to_frame("count")
    desc = df[proba_col].describe()

    print("\n=== Head ===")
    print(head.to_string(index=False))

    print("\n=== Label counts ===")
    print(counts.to_string())

    print("\n=== Probability stats ===")
    print(desc.to_string())

    if args.out:
        md = []
        md.append(f"# Predictions Summary\n")
        md.append(f"- File: `{path.name}`\n")
        md.append(f"- Rows: **{len(df)}**\n")
        md.append(f"- Using columns: `{label_col}`, `{proba_col}`\n")
        md.append("\n## Head\n")
        try:
            md.append(df[[label_col, proba_col]].head(args.top).to_markdown(index=False))
        except Exception:
            md.append("```\n" + head.to_string(index=False) + "\n```")
        md.append("\n## Label counts\n")
        try:
            md.append(counts.reset_index().to_markdown(index=False))
        except Exception:
            md.append("```\n" + counts.to_string() + "\n```")
        md.append("\n## Probability stats\n")
        md.append("```\n" + desc.to_string() + "\n```")

        out_path = Path(args.out)
        out_path.write_text("\n".join(md), encoding="utf-8")
        print(f"\nWrote Markdown report → {out_path}")

if __name__ == "__main__":
    main()
