# command: streamlit run app.py

#!/usr/bin/env python3
"""
Streamlit dashboard for: Fake News Learning with Machine Learning (Stage 1)

Stage 1 goal:
- A simple, reproducible baseline text classifier:
  TF-IDF vectorization + Logistic Regression (scikit-learn Pipeline)

Expected project layout (relative to this file):
  data/
    fake_and_real_news.csv          # Kaggle-style sample (Text,label) or your own CSV
  artifacts/
    fake_news_tfidf_logreg.joblib   # trained model pipeline (joblib)
    label_map.json                  # optional: mapping used in training
    metrics.json                    # optional: evaluation metrics from training
    preds_scored.csv                # optional: batch predictions from predict.py

Run:
  streamlit run app.py
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# -----------------------------
# Paths + discovery
# -----------------------------
ROOT = Path(__file__).resolve().parent

CANDIDATE_DATA = [
    ROOT / "data" / "fake_and_real_news.csv",
    ROOT / "data" / "data.csv",
]
CANDIDATE_MODEL = [
    ROOT / "artifacts" / "fake_news_tfidf_logreg.joblib",
]
CANDIDATE_LABELMAP = [
    ROOT / "artifacts" / "label_map.json",
]
CANDIDATE_METRICS = [
    ROOT / "artifacts" / "metrics.json",
    ROOT / "reports" / "metrics.json",
]
CANDIDATE_PREDS = [
    ROOT / "artifacts" / "preds_scored.csv",
    ROOT / "artifacts" / "preds.csv",
    ROOT / "reports" / "preds.csv",
]


def first_existing(paths: list[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None


DATA_PATH = first_existing(CANDIDATE_DATA)
MODEL_PATH = first_existing(CANDIDATE_MODEL)
LABEL_MAP_PATH = first_existing(CANDIDATE_LABELMAP)
METRICS_PATH = first_existing(CANDIDATE_METRICS)
PREDS_PATH = first_existing(CANDIDATE_PREDS)


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Fake News Learning with ML (Stage 1)", page_icon="📰", layout="wide")


# -----------------------------
# Helpers
# -----------------------------
def file_stamp(path: Optional[Path]) -> str:
    if path is None:
        return "missing"
    try:
        ts = path.stat().st_mtime
        return pd.to_datetime(ts, unit="s").strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return "unknown"


def read_json(path: Optional[Path]) -> Dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def normalize_label_map(raw: Dict[str, Any]) -> Dict[str, str]:
    """
    Supports either direction:
      A) {"0": "Real", "1": "Fake"}        (int->name)
      B) {"Real": 0, "Fake": 1}           (name->int)
    Returns a display map: {"0": "Real", "1": "Fake"}.
    """
    if not raw:
        return {"0": "Real", "1": "Fake"}

    # Case A: looks like int->name
    keys = list(raw.keys())
    if any(k in ("0", "1") for k in map(str, keys)):
        return {str(k): str(v) for k, v in raw.items()}

    # Case B: name->int (invert)
    inv: Dict[str, str] = {}
    for k, v in raw.items():
        inv[str(v)] = str(k)
    # Ensure defaults exist
    inv.setdefault("0", "Real")
    inv.setdefault("1", "Fake")
    return inv


def prettify_label(label: Any, display_map: Dict[str, str]) -> str:
    if label is None:
        return "Unknown"
    return display_map.get(str(label), str(label))


def infer_columns(df: pd.DataFrame) -> Tuple[str, Optional[str]]:
    """
    Try to infer the text column and label column.
    Returns (text_col, label_col_or_None).
    """
    cols = {c.lower(): c for c in df.columns}

    # Common Kaggle schema: Text,label
    for c in ("text", "content", "article", "headline", "title"):
        if c in cols:
            text_col = cols[c]
            break
    else:
        # fallback: first column
        text_col = df.columns[0]

    label_col = None
    for c in ("label", "target", "y", "class"):
        if c in cols:
            label_col = cols[c]
            break

    return text_col, label_col


@st.cache_data(show_spinner=False)
def load_data(path: Optional[Path]) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_metrics(path: Optional[Path]) -> Dict[str, Any]:
    return read_json(path)


@st.cache_data(show_spinner=False)
def load_preds(path: Optional[Path]) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


@st.cache_resource(show_spinner=False)
def load_model(path: Optional[Path]) -> Tuple[Optional[Any], list[str], Optional[Exception]]:
    if path is None:
        return None, [], FileNotFoundError("Model file not found (expected artifacts/fake_news_tfidf_logreg.joblib).")

    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            m = joblib.load(path)
        warn_msgs = [str(x.message) for x in w]
        return m, warn_msgs, None
    except Exception as e:
        return None, [], e


def safe_predict_one(model: Any, text: str) -> Dict[str, Any]:
    """
    Predict a single sample.
    Returns: {label, proba, proba_by_class, error}
    """
    out: Dict[str, Any] = {"label": None, "proba": None, "proba_by_class": {}, "error": None}
    text = (text or "").strip()
    if not text:
        return out

    try:
        pred = model.predict([text])[0]
        out["label"] = pred

        # Optional probabilities (LogReg supports predict_proba).
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba([text])[0]
            classes = getattr(model, "classes_", list(range(len(probs))))
            out["proba_by_class"] = {str(c): float(p) for c, p in zip(classes, probs)}

            try:
                probs = model.predict_proba([text])[0]
                ...
            except Exception:
                probs = None

            # Binary: probability of class 1 if present
            if len(probs) == 2:
                out["proba"] = float(probs[1])
        return out
    except Exception as e:
        out["error"] = str(e)
        return out


def top_features(model: Any, k: int = 20) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    For a linear text model Pipeline(TfidfVectorizer -> LogisticRegression),
    show the most positive/negative features.
    Returns (top_real_df, top_fake_df).
    """
    try:
        vectorizer = None
        clf = None

        if hasattr(model, "named_steps"):
            for _, step in model.named_steps.items():
                if hasattr(step, "get_feature_names_out"):
                    vectorizer = step
                if hasattr(step, "coef_"):
                    clf = step

        if vectorizer is None or clf is None:
            return pd.DataFrame(), pd.DataFrame()

        feat_names = np.array(vectorizer.get_feature_names_out())
        coef = clf.coef_

        # Binary LR: shape (1, n_features)
        if getattr(coef, "ndim", 1) == 2 and coef.shape[0] == 1:
            coef = coef[0]
        else:
            # Multiclass: take last row as best-effort "positive"
            coef = coef[-1]

        coef = np.array(coef)

        idx_sorted = np.argsort(coef)
        top_neg = idx_sorted[:k]            # most negative -> class 0
        top_pos = idx_sorted[-k:][::-1]     # most positive -> class 1

        real_df = pd.DataFrame({"feature": feat_names[top_neg], "weight": coef[top_neg]})
        fake_df = pd.DataFrame({"feature": feat_names[top_pos], "weight": coef[top_pos]})
        return real_df, fake_df
    except Exception:
        return pd.DataFrame(), pd.DataFrame()


# -----------------------------
# Load artifacts
# -----------------------------
raw_label_map = read_json(LABEL_MAP_PATH)
display_label_map = normalize_label_map(raw_label_map)

df_data = load_data(DATA_PATH)
metrics = load_metrics(METRICS_PATH)
preds_df = load_preds(PREDS_PATH)

model, model_warnings, model_err = load_model(MODEL_PATH)

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.subheader("Stage 1 artifacts")
    st.write(f"**Dataset:** `{DATA_PATH if DATA_PATH else 'missing'}` ({file_stamp(DATA_PATH)})")
    st.write(f"**Model:** `{MODEL_PATH if MODEL_PATH else 'missing'}` ({file_stamp(MODEL_PATH)})")
    st.write(f"**Label map:** `{LABEL_MAP_PATH if LABEL_MAP_PATH else 'missing'}` ({file_stamp(LABEL_MAP_PATH)})")
    st.write(f"**Metrics:** `{METRICS_PATH if METRICS_PATH else 'missing'}` ({file_stamp(METRICS_PATH)})")
    st.write(f"**Predictions:** `{PREDS_PATH if PREDS_PATH else 'missing'}` ({file_stamp(PREDS_PATH)})")

    st.divider()
    st.caption("If you see scikit-learn version warnings or prediction errors, retrain the model in the same venv:")
    st.code("python src/train.py --data data/fake_and_real_news.csv --text-col Text --label-col label", language="bash")

    if model_warnings:
        st.warning("Model loaded with warnings.")
        with st.expander("Show warnings"):
            for msg in model_warnings:
                st.write(f"- {msg}")

    if model_err is not None:
        st.error("Model could not be loaded.")
        st.code(str(model_err))


# -----------------------------
# Header
# -----------------------------
st.title("Fake News Learning with Machine Learning (Stage 1)")
st.caption(
    "Baseline NLP classifier (TF‑IDF + Logistic Regression). "
    "Use this dashboard to explain what the model does, show metrics, and demo predictions."
)

# -----------------------------
# Tabs
# -----------------------------
tab_overview, tab_try, tab_metrics, tab_interpret, tab_batch = st.tabs(
    ["Overview", "Try a prediction", "Metrics", "Interpretability", "Batch scoring"]
)

with tab_overview:
    st.subheader("What this project demonstrates")
    st.write(
        "- **Problem framing:** binary text classification (Fake vs Real).\n"
        "- **Baseline modeling:** TF‑IDF vectors + Logistic Regression.\n"
        "- **Evaluation:** accuracy + per-class precision/recall/F1 + confusion matrix.\n"
        "- **Reproducibility:** saved artifacts (model + metrics), consistent CLI scripts."
    )

    st.markdown("**Tech stack (Stage 1):** Python, pandas, scikit-learn, joblib, Streamlit.")

    st.divider()

    st.subheader("Dataset snapshot")
    if df_data.empty:
        st.info("Dataset not found. Put a CSV in `data/` (e.g., `fake_and_real_news.csv`) and refresh.")
    else:
        text_col, label_col = infer_columns(df_data)
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", f"{len(df_data):,}")
        c2.metric("Columns", f"{len(df_data.columns):,}")
        c3.metric("Text column (auto)", text_col)

        st.caption(f"Detected label column: `{label_col}`" if label_col else "No label column detected (prediction-only dataset).")

        # Label distribution if present
        if label_col and label_col in df_data.columns:
            vc = df_data[label_col].astype(str).value_counts().reset_index()
            vc.columns = ["label", "count"]
            st.dataframe(vc, use_container_width=True, hide_index=True)

        st.dataframe(df_data.head(20), use_container_width=True)

with tab_try:
    st.subheader("Interactive prediction (single text)")
    st.write("Paste a short headline or paragraph. The model returns a label, and (if available) probabilities.")

    sample = st.text_area(
        "Text to score",
        height=160,
        placeholder="Example: 'Breaking: Scientists discover ...'",
    )
    run_pred = st.button("Predict", type="primary")

    if run_pred:
        if model is None:
            st.warning("Model is not available (see sidebar).")
        else:
            out = safe_predict_one(model, sample)

            if out["error"]:
                st.error("Prediction failed (likely environment/version mismatch).")
                st.code(out["error"])
            else:
                pred_label = prettify_label(out["label"], display_label_map)
                st.markdown(f"**Prediction:** `{pred_label}`")

                if out["proba"] is not None:
                    st.metric("Estimated P(Fake)", f"{float(out['proba']):.3f}")

                if out["proba_by_class"]:
                    prob_table = (
                        pd.DataFrame(
                            [
                                {"class": prettify_label(k, display_label_map), "probability": v}
                                for k, v in out["proba_by_class"].items()
                            ]
                        )
                        .sort_values("probability", ascending=False)
                        .reset_index(drop=True)
                    )
                    st.dataframe(prob_table, use_container_width=True, hide_index=True)

with tab_metrics:
    st.subheader("Evaluation metrics")
    if not metrics:
        st.info("No metrics.json found yet. Train the model to generate metrics.")
    else:
        acc = metrics.get("accuracy")
        if acc is not None:
            st.metric("Accuracy", f"{float(acc):.3f}")

        note = metrics.get("note")
        if note:
            st.caption(note)

        report = metrics.get("classification_report", {})
        if isinstance(report, dict) and report:
            rows = []
            for k, v in report.items():
                if isinstance(v, dict) and "precision" in v:
                    rows.append(
                        {
                            "label": prettify_label(k, display_label_map),
                            "precision": v.get("precision"),
                            "recall": v.get("recall"),
                            "f1": v.get("f1-score"),
                            "support": v.get("support"),
                        }
                    )
            if rows:
                st.markdown("**Per-class report**")
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        cm = metrics.get("confusion_matrix")
        if isinstance(cm, list) and cm:
            st.markdown("**Confusion matrix** (rows=true, cols=predicted)")
            st.dataframe(pd.DataFrame(cm), use_container_width=True, hide_index=True)

with tab_interpret:
    st.subheader("Top features (model interpretability)")
    st.caption(
        "For Logistic Regression, each word/phrase gets a coefficient (weight). "
        "Positive weights push the prediction toward class 1, negative toward class 0."
    )
    if model is None:
        st.info("Model not loaded.")
    else:
        k = st.slider("Top features per class", min_value=5, max_value=40, value=20, step=5)
        real_df, fake_df = top_features(model, k=k)

        if real_df.empty and fake_df.empty:
            st.info("Could not extract features/weights. This requires a linear model with coef_ and a vectorizer with feature names.")
        else:
            cA, cB = st.columns(2)
            with cA:
                st.markdown(f"**Most indicative of `{display_label_map.get('0','Real')}`** (negative weights)")
                st.dataframe(real_df.sort_values("weight"), use_container_width=True, hide_index=True)
            with cB:
                st.markdown(f"**Most indicative of `{display_label_map.get('1','Fake')}`** (positive weights)")
                st.dataframe(fake_df.sort_values("weight", ascending=False), use_container_width=True, hide_index=True)

with tab_batch:
    st.subheader("Batch scoring")
    st.write(
        "To score a whole CSV file, use your CLI script:\n\n"
        "`python src/predict.py --model artifacts/fake_news_tfidf_logreg.joblib --data data/fake_and_real_news.csv --text-col Text --out artifacts/preds_scored.csv`\n\n"
        "If `preds_scored.csv` exists, this dashboard will display it below."
    )

    if preds_df.empty:
        st.info("No batch predictions file found yet (preds_scored.csv / preds.csv).")
    else:
        st.write(f"Rows: {len(preds_df):,}")
        st.dataframe(preds_df.head(100), use_container_width=True)

        cols = set(preds_df.columns)
        # Optional misclassification view if y_true exists
        if {"y_true", "y_pred"}.issubset(cols):
            wrong = preds_df[preds_df["y_true"] != preds_df["y_pred"]]
            st.markdown("**Misclassified samples**")
            st.write(f"Count: {len(wrong):,}")
            st.dataframe(wrong.head(100), use_container_width=True)

st.divider()
st.caption(
    "Recruiter-friendly framing: this is a baseline classifier that demonstrates an end-to-end ML workflow "
    "(data -> model -> evaluation -> saved artifacts -> simple UI). The next stages typically improve data, "
    "validation, model comparison, and robustness."
)
