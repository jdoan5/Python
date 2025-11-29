import argparse
import os
import sys
import json
import pandas as pd
import joblib


def main():
    parser = argparse.ArgumentParser(
        description="Customer churn prediction using TF-IDF + Logistic Regression."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input CSV with at least a 'text' column.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to output CSV with predictions.",
    )
    parser.add_argument(
        "--model",
        default="models/tfidf_logreg_churn.joblib",
        help="Path to trained model .joblib file.",
    )
    parser.add_argument(
        "--meta",
        default="models/threshold.json",
        help="Path to JSON with threshold and cost parameters.",
    )

    args = parser.parse_args()

    if not os.path.exists(args.model) or not os.path.exists(args.meta):
        print(
            "Model or threshold metadata not found. "
            "Make sure you have run train.py and paths are correct.",
            file=sys.stderr,
        )
        sys.exit(1)

    # load model + threshold
    clf = joblib.load(args.model)
    with open(args.meta, "r") as f:
        meta = json.load(f)

    threshold = float(meta["threshold"])
    print(f"Using decision threshold: {threshold:.3f}")

    # read input
    df = pd.read_csv(args.input)

    if "text" not in df.columns:
        print("Input CSV must contain a 'text' column.", file=sys.stderr)
        sys.exit(1)

    # get probabilities and predictions
    proba = clf.predict_proba(df["text"])[:, 1]
    pred = (proba >= threshold).astype(int)

    df_out = df.copy()
    df_out["churn_proba"] = proba
    df_out["churn_pred"] = pred

    df_out.to_csv(args.output, index=False)
    print(f"Saved predictions to {args.output}")


if __name__ == "__main__":
    main()