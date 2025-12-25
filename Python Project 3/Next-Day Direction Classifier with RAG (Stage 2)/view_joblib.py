# view_joblib.py
from __future__ import annotations

import argparse
from pathlib import Path

import joblib


def summarize(obj, name="obj", max_items=20, indent=0):
    pad = " " * indent
    t = type(obj).__name__
    print(f"{pad}{name}: type={t}")

    # dict
    if isinstance(obj, dict):
        print(f"{pad}  len={len(obj)}")
        keys = list(obj.keys())[:max_items]
        print(f"{pad}  keys(sample)={keys}")
        return

    # list/tuple/set
    if isinstance(obj, (list, tuple, set)):
        try:
            n = len(obj)
        except Exception:
            n = "?"
        print(f"{pad}  len={n}")
        # show sample items
        for i, item in enumerate(list(obj)[:min(max_items, 5)]):
            print(f"{pad}  [{i}] -> {type(item).__name__}")
        return

    # numpy / pandas (optional best effort)
    try:
        import numpy as np  # noqa
        import pandas as pd  # noqa
        if hasattr(obj, "shape"):
            print(f"{pad}  shape={getattr(obj, 'shape', None)}")
        if hasattr(obj, "columns"):
            print(f"{pad}  columns(sample)={list(obj.columns)[:max_items]}")
        return
    except Exception:
        pass

    # scikit-learn estimator / pipeline
    attrs = [
        "named_steps",
        "classes_",
        "coef_",
        "feature_importances_",
        "n_features_in_",
        "vocabulary_",
        "idf_",
    ]
    found = [a for a in attrs if hasattr(obj, a)]
    if found:
        print(f"{pad}  sklearn-like attrs={found}")
        if hasattr(obj, "named_steps"):
            try:
                print(f"{pad}  pipeline steps={list(obj.named_steps.keys())}")
            except Exception:
                pass
        if hasattr(obj, "classes_"):
            try:
                print(f"{pad}  classes_={obj.classes_}")
            except Exception:
                pass
        if hasattr(obj, "n_features_in_"):
            try:
                print(f"{pad}  n_features_in_={obj.n_features_in_}")
            except Exception:
                pass
        return

    # fallback: show public attrs
    public = [a for a in dir(obj) if not a.startswith("_")]
    print(f"{pad}  public attrs(sample)={public[:max_items]}")


def main():
    ap = argparse.ArgumentParser(description="Inspect a .joblib file.")
    ap.add_argument("path", help="Path to .joblib file")
    args = ap.parse_args()

    p = Path(args.path).expanduser().resolve()
    if not p.exists():
        raise SystemExit(f"File not found: {p}")

    obj = joblib.load(p)
    print(f"Loaded: {p}")
    summarize(obj)

    # If it's a dict, also summarize a few nested entries
    if isinstance(obj, dict):
        print("\n--- nested (sample) ---")
        for k in list(obj.keys())[:5]:
            summarize(obj[k], name=f"obj[{repr(k)}]", indent=2)


if __name__ == "__main__":
    main()