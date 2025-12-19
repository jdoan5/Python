"""
Streamlit dashboard for the "Fake News Learning with Machine Learning" project.

Expected project structure (relative to this file):
  data/data.csv
  models/fake_news_tfidf_logreg.joblib
  models/label_map.json            (optional)
  reports/metrics.json             (optional)
  reports/preds.csv                (optional)

Run:
  streamlit run app.py
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import pandas as pd
import streamlit as st


# -----------------------------
# Paths
# -----------------------------
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "data.csv"
MODEL_PATH = ROOT / "models" / "fake_news_tfidf_logreg.joblib"
LABEL_MAP_PATH = ROOT / "models" / "label_map.json"
METRICS_PATH = ROOT / "reports" / "metrics.json"
PREDS_PATH = ROOT / "reports" / "preds.csv"


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Fake News Learning with ML",
    page_icon="📰",
    layout="wide",
)


# -----------------------------
# Utilities
# -----------------------------
def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def file_stamp(path: Path) -> str:
    if not path.exists():
        return "missing"
    try:
        ts = path.stat().st_mtime
        return pd.to_datetime(ts, unit="s").strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return "unknown"


def prettify_label(label: Any, label_map: Dict[str, str]) -> str:
    """
    Convert model output label to human-friendly label.
    Supports:
      - numeric labels (0/1)
      - string labels ("0"/"1", "Real"/"Fake")
    """
    if label is None:
        return "Unknown"
    key = str(label)
    return label_map.get(key, key)


@st.cache_data(show_spinner=False)
def load_metrics() -> Dict[str, Any]:
    return read_json(METRICS_PATH)


@st.cache_data(show_spinner=False)
def load_label_map() -> Dict[str, str]:
    lm = read_json(LABEL_MAP_PATH)
    # If file exists, trust it. Otherwise, provide a sensible default.
    if lm:
        # Normalize keys to strings
        return {str(k): str(v) for k, v in lm.items()}
    return {"0": "Real", "1": "Fake"}


@st.cache_data(show_spinner=False)
def load_preds() -> pd.DataFrame:
    if not PREDS_PATH.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(PREDS_PATH)
    except Exception:
        return pd.DataFrame()


@st.cache_resource(show_spinner=False)
def load_model() -> Tuple[Optional[Any], list[str], Optional[Exception]]:
    """
    Load the joblib model and capture warnings (e.g., sklearn version mismatch).
    If loading fails, returns (None, [], exception).
    """
    if not MODEL_PATH.exists():
        return None, [], FileNotFoundError(f"Model not found: {MODEL_PATH}")
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            m = joblib.load(MODEL_PATH)
        warn_msgs = [str(x.message) for x in w]
        return m, warn_msgs, None
    except Exception as e:
        return None, [], e


def predict_one(model: Any, text: str) -> Dict[str, Any]:
    """
    Predict a single sample. Works with sklearn Pipeline-like objects.
    Returns {label, proba, proba_by_class}.
    """
    if not text.strip():
        return {"label": None, "proba": None, "proba_by_class": {}}

    # Model label prediction
    pred = model.predict([text])[0]

    # Optional probabilities
    proba = None
    proba_by_class: Dict[str, float] = {}
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba([text])[0]
        classes = getattr(model, "classes_", list(range(len(probs))))
        for c, p in zip(classes, probs):
            proba_by_class[str(c)] = float(p)
        # If binary, show probability for the positive class if possible
        if len(probs) == 2:
            proba = float(probs[1])

    return {"label": pred, "proba": proba, "proba_by_class": proba_by_class}


def top_features(model: Any, k: int = 20) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract top positive/negative features for linear models in a Pipeline:
      Pipeline(TfidfVectorizer -> LogisticRegression/LinearSVC)
    Returns (top_real_df, top_fake_df) as dataframes.
    If not available, returns empty dataframes.
    """
    try:
        # Pipeline step names vary; try common patterns.
        vectorizer = None
        clf = None

        if hasattr(model, "named_steps"):
            for name, step in model.named_steps.items():
                if hasattr(step, "get_feature_names_out"):
                    vectorizer = step
                if hasattr(step, "coef_"):
                    clf = step

        if vectorizer is None or clf is None:
            return pd.DataFrame(), pd.DataFrame()

        feat_names = vectorizer.get_feature_names_out()
        coef = clf.coef_
        if coef.ndim == 2 and coef.shape[0] == 1:
            coef = coef[0]
        else:
            # multiclass: take the last class as "Fake" (best-effort)
            coef = coef[-1]

        # Higher coef -> more likely class 1 (commonly "Fake")
        idx_sorted = coef.argsort()
        top_neg = idx_sorted[:k]          # most negative -> more likely class 0 ("Real")
        top_pos = idx_sorted[-k:][::-1]   # most positive -> more likely class 1 ("Fake")

        real_df = pd.DataFrame({"feature": feat_names[top_neg], "weight": coef[top_neg]})
        fake_df = pd.DataFrame({"feature": feat_names[top_pos], "weight": coef[top_pos]})
        return real_df, fake_df
    except Exception:
        return pd.DataFrame(), pd.DataFrame()


# -----------------------------
# Header
# -----------------------------
st.title("Fake News Learning with Machine Learning")
st.caption(
    "A reproducible baseline NLP classifier: TF-IDF vectorization + Logistic Regression. "
    "This dashboard summarizes evaluation metrics and lets you try interactive predictions."
)

label_map = load_label_map()
metrics = load_metrics()
preds_df = load_preds()

model, model_warnings, model_err = load_model()

with st.sidebar:
    st.subheader("Artifacts")
    st.write(f"**data.csv:** `{DATA_PATH}` ({file_stamp(DATA_PATH)})")
    st.write(f"**model:** `{MODEL_PATH}` ({file_stamp(MODEL_PATH)})")
    st.write(f"**label_map.json:** `{LABEL_MAP_PATH}` ({file_stamp(LABEL_MAP_PATH)})")
    st.write(f"**metrics.json:** `{METRICS_PATH}` ({file_stamp(METRICS_PATH)})")
    st.write(f"**preds.csv:** `{PREDS_PATH}` ({file_stamp(PREDS_PATH)})")

    if model_warnings:
        with st.expander("Model load warnings", expanded=False):
            for msg in model_warnings:
                st.warning(msg)

    if model_err is not None:
        st.error("Model could not be loaded.")
        st.code(str(model_err))
        st.info(
            "If you trained the model with a different Python / numpy / scikit-learn version, "
            "re-run `python src/train.py` in this environment to regenerate the joblib file."
        )


# -----------------------------
# Layout
# -----------------------------
col_left, col_right = st.columns([1.1, 0.9], gap="large")

with col_left:
    st.subheader("1) Try a prediction")

    sample = st.text_area(
        "Paste a headline or short article text",
        height=160,
        placeholder="Example: 'Breaking: Scientists discover...' ",
    )

    c1, c2 = st.columns([0.35, 0.65])
    with c1:
        run_pred = st.button("Predict", type="primary", use_container_width=True)
    with c2:
        st.caption("Predictions use the saved TF-IDF + Logistic Regression model.")

    if run_pred:
        if model is None:
            st.warning("Model is not loaded yet. See the sidebar for details.")
        else:
            out = predict_one(model, sample)
            pred_label = prettify_label(out["label"], label_map)

            st.markdown(f"**Prediction:** `{pred_label}`")
            if out["proba"] is not None:
                st.write(f"**P(Fake)** (best-effort): `{out['proba']:.3f}`")

            if out["proba_by_class"]:
                # Display probability table
                prob_table = (
                    pd.DataFrame(
                        [
                            {"class": prettify_label(k, label_map), "probability": v}
                            for k, v in out["proba_by_class"].items()
                        ]
                    )
                    .sort_values("probability", ascending=False)
                    .reset_index(drop=True)
                )
                st.dataframe(prob_table, use_container_width=True, hide_index=True)

    st.divider()

    st.subheader("2) Metrics summary (from reports/metrics.json)")
    if metrics:
        st.metric("Accuracy", f"{float(metrics.get('accuracy', 0.0)):.3f}")

        note = metrics.get("note")
        if note:
            st.caption(note)

        report = metrics.get("classification_report", {})
        if isinstance(report, dict) and report:
            # Flatten only per-class rows ("0", "1", etc.)
            class_rows = []
            for k, v in report.items():
                if isinstance(v, dict) and "precision" in v:
                    class_rows.append(
                        {
                            "label": prettify_label(k, label_map),
                            "precision": v.get("precision"),
                            "recall": v.get("recall"),
                            "f1": v.get("f1-score"),
                            "support": v.get("support"),
                        }
                    )
            if class_rows:
                st.dataframe(pd.DataFrame(class_rows), use_container_width=True, hide_index=True)

        cm = metrics.get("confusion_matrix")
        if isinstance(cm, list) and cm:
            st.caption("Confusion matrix (rows = true, cols = predicted)")
            st.dataframe(pd.DataFrame(cm), use_container_width=True, hide_index=True)
    else:
        st.info("No metrics.json found yet. Run `python src/train.py` to generate metrics into `reports/metrics.json`.")

with col_right:
    st.subheader("3) Optional: Prediction report (reports/preds.csv)")
    if preds_df.empty:
        st.info("No preds.csv found yet. If your training script writes predictions, they will appear here.")
    else:
        st.write(f"Rows: {len(preds_df):,}")
        # Try to detect common columns
        cols = preds_df.columns.tolist()
        st.caption(f"Columns detected: {', '.join(cols)}")
        st.dataframe(preds_df.head(50), use_container_width=True)

        # Misclassification filter if possible
        if {"y_true", "y_pred"}.issubset(set(cols)):
            st.markdown("**Misclassified samples**")
            wrong = preds_df[preds_df["y_true"] != preds_df["y_pred"]]
            st.write(f"Count: {len(wrong):,}")
            st.dataframe(wrong.head(50), use_container_width=True)

    st.divider()

    st.subheader("4) Optional: Model interpretability (top features)")
    if model is None:
        st.info("Model not loaded.")
    else:
        k = st.slider("Top features per class", min_value=5, max_value=40, value=20, step=5)
        real_df, fake_df = top_features(model, k=k)

        if real_df.empty and fake_df.empty:
            st.info("Top-feature extraction not available (model may not expose coef_ or feature names).")
        else:
            cA, cB = st.columns(2)
            with cA:
                st.markdown("**Most indicative of `Real`** (negative weights)")
                st.dataframe(real_df, use_container_width=True, hide_index=True)
            with cB:
                st.markdown("**Most indicative of `Fake`** (positive weights)")
                st.dataframe(fake_df, use_container_width=True, hide_index=True)


# Footer
st.divider()
st.caption(
    "Tip: keep your environment consistent between training and serving (Python, numpy, scikit-learn). "
    "If you see version warnings, re-train the model in the same venv used to run Streamlit."
)
