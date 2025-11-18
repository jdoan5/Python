# baseline.py
import argparse, sys, os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    classification_report, confusion_matrix, roc_auc_score
)
import joblib

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase, remove parentheses text, replace spaces/punct with underscores, strip duplicates."""
    df = df.copy()
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(r"\s*\(.*?\)\s*", "", regex=True)   # drop text in parentheses
        .str.replace(r"[^0-9a-zA-Z]+", "_", regex=True)  # spaces & punctuation -> _
        .str.replace(r"__+", "_", regex=True)
        .str.strip("_")
    )
    # Known header fixes / unifications
    rename_map = {
        "tenture": "tenure",
        "tenture_months": "tenure_months",
        "customer_satisfaction_1_10_scale": "satisfaction_1_10",
        "customer_satisfaction": "satisfaction_1_10",
        "type": "body_type",
        "fuel_type": "fuel_type",  # keep same name, ensures presence
        "msrp_usd": "msrp",
    }
    df = df.rename(columns=rename_map)
    return df

def clean_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Coerce known numeric fields (strip $ and commas)
    for c in ["tenure", "tenure_months", "satisfaction_1_10", "msrp"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(r"[^0-9.\-]", "", regex=True), errors="coerce")

    # Fix common typos
    if "fuel_type" in df.columns:
        df["fuel_type"] = (
            df["fuel_type"]
            .astype(str)
            .str.strip()
            .replace({"Hybird": "Hybrid", "hybird": "Hybrid"})
        )

    return df

def map_target(y: pd.Series) -> pd.Series:
    """Map Yes/No (and variants) to 1/0."""
    mapped = (
        y.astype(str).str.strip().str.lower()
        .map({"yes": 1, "y": 1, "1": 1, "true": 1, "no": 0, "n": 0, "0": 0, "false": 0})
    )
    if mapped.isna().any():
        bad = y[mapped.isna()].unique()
        raise ValueError(f"Unrecognized target values: {bad}. Expected Yes/No (or Y/N/1/0/True/False).")
    return mapped.astype(int)

def load_data(path: str, target: str) -> tuple[pd.DataFrame, pd.Series]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("CSV is empty.")

    df = normalize_columns(df)
    df = clean_values(df)

    if target not in df.columns:
        raise KeyError(f"Target column '{target}' not found. Columns: {list(df.columns)}")

    y = map_target(df[target])
    X = df.drop(columns=[target])

    # Infer numeric vs categorical after coercion
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    return X, y, num_cols, cat_cols

def safe_train_test_split(X, y, test_size=0.33, random_state=42):
    """Use stratify only if both classes have >= 2 samples; otherwise, fall back to non-stratified."""
    vc = y.value_counts()
    try_stratify = (vc.min() >= 2) and (len(vc) == 2)
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state,
        stratify=y if try_stratify else None
    )

def main():
    ap = argparse.ArgumentParser(description="Baseline churn model (cars) with simple cleaning and LR.")
    ap.add_argument("--csv", default="data/data.csv", help="Path to CSV (default: data/data.csv)")
    ap.add_argument("--target", default="churn", help="Target column name (default: churn)")
    ap.add_argument("--save", default="model.joblib", help="Where to save the trained model (default: model.joblib)")
    args = ap.parse_args()

    # 1) Load & split
    X, y, num_cols, cat_cols = load_data(args.csv, args.target)
    X_train, X_test, y_train, y_test = safe_train_test_split(X, y)

    print(f"\nRows: {len(y)} | Train: {len(y_train)} | Test: {len(y_test)}")
    print(f"Numeric: {num_cols}")
    print(f"Categorical: {cat_cols}")

    # 2) Majority-class baseline
    majority = int(y_train.value_counts().idxmax())
    y_base = np.full_like(y_test.to_numpy(), fill_value=majority)
    base_acc = accuracy_score(y_test, y_base)
    base_bacc = balanced_accuracy_score(y_test, y_base)
    print("\n=== Majority-class baseline ===")
    print(f"Accuracy:          {base_acc:.3f}")
    print(f"Balanced accuracy: {base_bacc:.3f}")

    # 3) Simple model: OneHot + Standardize + LogisticRegression
    pre = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ])

    clf = Pipeline([
        ("pre", pre),
        ("lr", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ])

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("\n=== Logistic Regression ===")
    print(f"Accuracy:          {accuracy_score(y_test, y_pred):.3f}")
    print(f"Balanced accuracy: {balanced_accuracy_score(y_test, y_pred):.3f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, digits=3))

    print("Confusion matrix [rows=true, cols=pred]:")
    print(confusion_matrix(y_test, y_pred))

    # Optional: AUC if both classes present in test set
    try:
        y_proba = clf.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
        print(f"AUC: {auc:.3f}")
    except Exception:
        pass

    # 4) Save model (optional)
    if args.save:
        joblib.dump(clf, args.save)
        print(f"\nSaved model to: {args.save}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[ERROR] {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)