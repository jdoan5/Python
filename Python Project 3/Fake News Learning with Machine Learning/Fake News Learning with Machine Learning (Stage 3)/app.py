# command: streamlit run app.py
#!/usr/bin/env python3
"""
Streamlit dashboard for: Fake News Learning with Machine Learning (Stage 3)

Stage 3 goals (vs Stage 2):
- Optional probability calibration (so predict_proba is available more often).
- Threshold tuning on a validation split (e.g., optimize F1 for "Fake").
- Clearer, recruiter-friendly reporting artifacts (metrics + model comparison + threshold sweep).

Expected project layout:
  data/
    fake_and_real_news.csv
  artifacts/
    best_model.joblib
    label_map.json
  reports/
    metrics.json
    model_comparison.csv
    preds.csv
    threshold_sweep.csv        (optional, produced when probabilities exist)

Run:
  streamlit run app.py
"""

from __future__ import annotations

import json
import math
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
    ROOT / "artifacts" / "best_model.joblib",
    ROOT / "artifacts" / "fake_news_tfidf_logreg.joblib",  # fallback (Stage 1)
]

CANDIDATE_LABELMAP = [
    ROOT / "artifacts" / "label_map.json",
]

CANDIDATE_METRICS = [
    ROOT / "reports" / "metrics.json",
    ROOT / "artifacts" / "metrics.json",
]

CANDIDATE_COMPARISON = [
    ROOT / "reports" / "model_comparison.csv",
]

CANDIDATE_PREDS = [
    ROOT / "reports" / "preds.csv",
    ROOT / "reports" / "preds_scored.csv",
    ROOT / "artifacts" / "preds.csv",
    ROOT / "artifacts" / "preds_scored.csv",
]

CANDIDATE_THRESHOLD_SWEEP = [
    ROOT / "reports" / "threshold_sweep.csv",
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
COMPARISON_PATH = first_existing(CANDIDATE_COMPARISON)
PREDS_PATH = first_existing(CANDIDATE_PREDS)
THRESHOLD_SWEEP_PATH = first_existing(CANDIDATE_THRESHOLD_SWEEP)


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Fake News Learning with ML (Stage 3)", page_icon="📰", layout="wide")


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
    Accepts either:
      A) {"0": "Real", "1": "Fake"}   (id->name)
      B) {"Real": 0, "Fake": 1}      (name->id)
    Returns display map: {"0": "Real", "1": "Fake"}.
    """
    if not raw:
        return {"0": "Real", "1": "Fake"}

    keys = list(raw.keys())
    if all(str(k).strip().isdigit() for k in keys):
        out = {str(k): str(v) for k, v in raw.items()}
        out.setdefault("0", "Real")
        out.setdefault("1", "Fake")
        return out

    inv: Dict[str, str] = {}
    for k, v in raw.items():
        inv[str(v)] = str(k)
    inv.setdefault("0", "Real")
    inv.setdefault("1", "Fake")
    return inv


def prettify_label(label: Any, display_map: Dict[str, str]) -> str:
    if label is None:
        return "Unknown"
    return display_map.get(str(label), str(label))


def infer_columns(df: pd.DataFrame) -> Tuple[str, Optional[str]]:
    cols = {c.lower(): c for c in df.columns}

    for c in ("text", "content", "article", "headline", "title"):
        if c in cols:
            text_col = cols[c]
            break
    else:
        text_col = df.columns[0]

    label_col = None
    for c in ("label", "target", "y", "class"):
        if c in cols:
            label_col = cols[c]
            break

    return text_col, label_col


def sigmoid(x: float) -> float:
    # Safe-ish sigmoid for large magnitude values
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def get_final_estimator(model: Any) -> Any:
    """If Pipeline-like, return last step estimator; else return model."""
    if hasattr(model, "named_steps"):
        try:
            return list(model.named_steps.values())[-1]
        except Exception:
            return model
    return model


def pos_index_from_estimator(estimator: Any, pos_label: int = 1) -> Optional[int]:
    """Find which proba column corresponds to label=1 using estimator.classes_."""
    classes = getattr(estimator, "classes_", None)
    if classes is None:
        return 1  # best-effort for binary
    classes = np.asarray(classes)
    for i, c in enumerate(classes):
        if c == pos_label or str(c) == str(pos_label):
            return i
    if len(classes) == 2:
        return 1
    return None


def coerce_pred_to_int(y_pred: Any) -> Optional[int]:
    """Best-effort cast for display; returns None if not coercible."""
    try:
        if isinstance(y_pred, (np.integer, int)):
            return int(y_pred)
        s = str(y_pred).strip()
        if s.isdigit():
            return int(s)
    except Exception:
        pass
    return None


def read_chosen_threshold(metrics: Dict[str, Any]) -> Optional[float]:
    """
    Stage 3 metrics.json stores chosen_threshold.
    If not available, return None.
    """
    if not metrics:
        return None
    for k in ("chosen_threshold", "best_threshold", "threshold"):
        if k in metrics:
            try:
                return float(metrics[k])
            except Exception:
                pass
    return None


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
def load_comparison(path: Optional[Path]) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_preds(path: Optional[Path]) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_threshold_sweep(path: Optional[Path]) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


@st.cache_resource(show_spinner=False)
def load_model(path: Optional[Path]) -> Tuple[Optional[Any], list[str], Optional[Exception]]:
    if path is None:
        return None, [], FileNotFoundError("Model file not found (expected artifacts/best_model.joblib).")
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            m = joblib.load(path)
        warn_msgs = [str(x.message) for x in w]
        return m, warn_msgs, None
    except Exception as e:
        return None, [], e


def safe_predict_one_stage3(model: Any, text: str, threshold: float) -> Dict[str, Any]:
    """
    Predict a single sample with Stage-3-aware behavior:
    - If predict_proba exists: compute P(Fake) for class label=1 and apply threshold to produce y_pred_thresholded.
    - Else if decision_function exists: show score (+ sigmoid for readability) and use model.predict as label.

    Returns: {
        pred_id, pred_id_thresholded, proba_fake, proba_by_class, decision_score, threshold_used, error
    }
    """
    out: Dict[str, Any] = {
        "pred_id": None,
        "pred_id_thresholded": None,
        "proba_fake": None,
        "proba_by_class": {},
        "decision_score": None,
        "threshold_used": threshold,
        "error": None,
    }

    text = (text or "").strip()
    if not text:
        return out

    try:
        est = get_final_estimator(model)

        # Probabilities are preferred (calibrated models often provide them).
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba([text])[0]
            classes = getattr(est, "classes_", list(range(len(probs))))
            out["proba_by_class"] = {str(c): float(p) for c, p in zip(classes, probs)}

            idx = pos_index_from_estimator(est, pos_label=1)
            if idx is not None and len(probs) >= 2:
                p_fake = float(probs[idx])
                out["proba_fake"] = p_fake
                out["pred_id_thresholded"] = 1 if p_fake >= threshold else 0

            # Also capture the model's default discrete class (for debugging)
            try:
                pred_default = model.predict([text])[0]
                out["pred_id"] = coerce_pred_to_int(pred_default)
            except Exception:
                out["pred_id"] = None

            # If we have thresholded pred, prefer it as the displayed "pred_id"
            if out["pred_id_thresholded"] is not None:
                out["pred_id"] = out["pred_id_thresholded"]

            return out

        # Decision score fallback
        pred = model.predict([text])[0]
        out["pred_id"] = coerce_pred_to_int(pred)

        if hasattr(model, "decision_function"):
            score = model.decision_function([text])
            score = float(np.asarray(score).ravel()[0])
            out["decision_score"] = score
            out["proba_fake"] = sigmoid(score)  # readability only (not calibrated)

        return out

    except Exception as e:
        out["error"] = str(e)
        return out


def pick_best_row(df: pd.DataFrame) -> Optional[pd.Series]:
    """
    Stage-3-aware best pick: macro_f1 -> f1_fake -> accuracy
    """
    if df.empty:
        return None

    sort_cols = [c for c in ("macro_f1", "f1_macro", "f1_fake", "accuracy") if c in df.columns]
    if not sort_cols:
        return None

    try:
        tmp = df.copy()
        for c in sort_cols:
            tmp[c] = pd.to_numeric(tmp[c], errors="coerce").fillna(-1.0)
        tmp = tmp.sort_values(sort_cols, ascending=[False] * len(sort_cols), kind="mergesort")
        return tmp.iloc[0]
    except Exception:
        return None


# -----------------------------
# Load artifacts
# -----------------------------
raw_label_map = read_json(LABEL_MAP_PATH)
display_label_map = normalize_label_map(raw_label_map)

df_data = load_data(DATA_PATH)
metrics = load_metrics(METRICS_PATH)
comparison_df = load_comparison(COMPARISON_PATH)
preds_df = load_preds(PREDS_PATH)
threshold_df = load_threshold_sweep(THRESHOLD_SWEEP_PATH)

model, model_warnings, model_err = load_model(MODEL_PATH)

chosen_threshold = read_chosen_threshold(metrics)


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.subheader("Stage 3 artifacts")
    st.write(f"**Dataset:** `{DATA_PATH if DATA_PATH else 'missing'}` ({file_stamp(DATA_PATH)})")
    st.write(f"**Best model:** `{MODEL_PATH if MODEL_PATH else 'missing'}` ({file_stamp(MODEL_PATH)})")
    st.write(f"**Label map:** `{LABEL_MAP_PATH if LABEL_MAP_PATH else 'missing'}` ({file_stamp(LABEL_MAP_PATH)})")
    st.write(f"**Metrics:** `{METRICS_PATH if METRICS_PATH else 'missing'}` ({file_stamp(METRICS_PATH)})")
    st.write(f"**Model comparison:** `{COMPARISON_PATH if COMPARISON_PATH else 'missing'}` ({file_stamp(COMPARISON_PATH)})")
    st.write(f"**Predictions:** `{PREDS_PATH if PREDS_PATH else 'missing'}` ({file_stamp(PREDS_PATH)})")
    st.write(f"**Threshold sweep:** `{THRESHOLD_SWEEP_PATH if THRESHOLD_SWEEP_PATH else 'missing'}` ({file_stamp(THRESHOLD_SWEEP_PATH)})")

    if chosen_threshold is not None:
        st.success(f"Chosen threshold (metrics.json): {chosen_threshold:.3f}")
    else:
        st.info("Chosen threshold not found (will default to 0.50 for probability models).")

    st.divider()
    st.caption("Recommended Stage 3 run order (same venv):")
    st.code(
        "python src/train.py --data data/fake_and_real_news.csv --text-col Text --label-col label\n"
        "# optional: batch scoring (auto-uses chosen_threshold from metrics.json if you use stage-3 predict.py)\n"
        "python src/predict.py --model artifacts/best_model.joblib --metrics reports/metrics.json "
        "--data data/fake_and_real_news.csv --text-col Text --label-col label --out reports/preds_scored.csv",
        language="bash",
    )

    if model_warnings:
        st.warning("Model loaded with warnings (often sklearn version mismatch).")
        with st.expander("Show warnings"):
            for msg in model_warnings[:12]:
                st.write(f"- {msg}")
            if len(model_warnings) > 12:
                st.write(f"- ... ({len(model_warnings) - 12} more)")

    if model_err is not None:
        st.error("Model could not be loaded.")
        st.code(str(model_err))


# -----------------------------
# Header
# -----------------------------
st.title("Fake News Learning with Machine Learning (Stage 3)")
st.caption(
    "Stage 3 extends the Stage-2 baseline by adding probability calibration and threshold tuning, "
    "so the demo can show a clearer 'confidence' signal and a reproducible decision rule. "
    "Source: https://www.kaggle.com/datasets/vishakhdapat/fake-news-detection"
)


# -----------------------------
# Tabs
# -----------------------------
tab_overview, tab_try, tab_compare, tab_metrics, tab_threshold, tab_batch = st.tabs(
    ["Overview", "Try a prediction", "Model comparison", "Metrics", "Threshold tuning", "Batch scoring"]
)

with tab_overview:
    st.subheader("What Stage 3 demonstrates")
    st.write(
        "- **Model selection:** compare multiple baselines and pick the best.\n"
        "- **Calibration (optional):** improve probability usability for thresholding and UI display.\n"
        "- **Threshold tuning:** choose a threshold on a validation split to optimize a target metric (e.g., F1 on Fake).\n"
        "- **Reproducibility:** save model + label map + metrics + predictions + (optional) threshold sweep table."
    )
    st.markdown("**Tech stack (Stage 3):** Python, pandas, scikit-learn, joblib, Streamlit.")
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
        st.caption(f"Detected label column: `{label_col}`" if label_col else "No label column detected.")

        if label_col and label_col in df_data.columns:
            vc = df_data[label_col].astype(str).value_counts().reset_index()
            vc.columns = ["label", "count"]
            st.dataframe(vc, use_container_width=True, hide_index=True)

        show_rows = st.slider(
            "Rows to display",
            min_value=10,
            max_value=min(5000, len(df_data)),
            value=min(50, len(df_data)),
            step=10,
        )
        st.dataframe(df_data.head(show_rows), use_container_width=True)

with tab_try:
    st.subheader("Interactive prediction (single text)")
    st.write("Paste a headline or short paragraph. If probabilities exist, the prediction uses a threshold.")

    # Choose default threshold for the UI:
    default_thr = float(chosen_threshold) if chosen_threshold is not None else 0.50
    thr = st.slider("Decision threshold for Fake (used when probabilities exist)", 0.05, 0.95, float(default_thr), 0.01)

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
            out = safe_predict_one_stage3(model, sample, threshold=thr)

            if out["error"]:
                st.error("Prediction failed (often an environment/version mismatch).")
                st.code(out["error"])
            else:
                pred_id = out["pred_id"]
                pred_label = prettify_label(pred_id, display_label_map) if pred_id is not None else "Unknown"
                st.markdown(f"**Prediction:** `{pred_label}`")

                if out["proba_fake"] is not None:
                    st.metric("P(Fake)", f"{float(out['proba_fake']):.3f}")
                    st.caption(f"Threshold used: {float(out['threshold_used']):.3f}")

                if out["decision_score"] is not None and not hasattr(model, "predict_proba"):
                    st.caption(
                        "This model does not output calibrated probabilities. "
                        "Showing a decision score + sigmoid mapping for readability (not a true probability)."
                    )
                    st.metric("Decision score", f"{float(out['decision_score']):.3f}")
                    if out["proba_fake"] is not None:
                        st.metric("Sigmoid(score) (readability only)", f"{float(out['proba_fake']):.3f}")

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

with tab_compare:
    st.subheader("Model comparison (reports/model_comparison.csv)")
    if comparison_df.empty:
        st.info("No model_comparison.csv found yet. Run Stage 3 train.py to generate it.")
    else:
        best = pick_best_row(comparison_df)
        if best is not None:
            st.success("Best model (ranked by macro_f1, then f1_fake, then accuracy):")
            st.dataframe(best.to_frame("value"), use_container_width=True)
        st.dataframe(comparison_df, use_container_width=True)

        if COMPARISON_PATH and COMPARISON_PATH.exists():
            st.download_button(
                "Download model_comparison.csv",
                data=COMPARISON_PATH.read_bytes(),
                file_name=COMPARISON_PATH.name,
                mime="text/csv",
                use_container_width=True,
            )

with tab_metrics:
    st.subheader("Evaluation metrics (metrics.json)")

    if not metrics:
        st.info("No metrics.json found yet. Run Stage 3 train.py to generate metrics.")
    else:
        st.caption(f"Stage: `{metrics.get('stage', 'unknown')}`  |  Best experiment: `{metrics.get('best_experiment', 'unknown')}`")

        calibrated = metrics.get("calibrated")
        if calibrated is not None:
            st.caption(f"Calibrated: `{bool(calibrated)}`  |  Method: `{metrics.get('calibration_method')}`")

        if chosen_threshold is not None:
            st.caption(f"Chosen threshold: `{chosen_threshold:.3f}`  |  Metric: `{metrics.get('threshold_metric', 'n/a')}`")

        # Stage 3 schema: test_metrics is nested
        test_metrics = metrics.get("test_metrics", {})
        acc = test_metrics.get("accuracy", metrics.get("accuracy"))
        macro_f1 = test_metrics.get("macro_f1", metrics.get("macro_f1", metrics.get("f1_macro")))
        f1_fake = test_metrics.get("f1_fake", metrics.get("f1_fake"))

        c1, c2, c3 = st.columns(3)
        if acc is not None:
            c1.metric("Accuracy (test)", f"{float(acc):.3f}")
        if macro_f1 is not None:
            c2.metric("Macro F1 (test)", f"{float(macro_f1):.3f}")
        if f1_fake is not None:
            c3.metric("F1 (Fake) (test)", f"{float(f1_fake):.3f}")

        val_at_thr = metrics.get("val_metrics_at_chosen_threshold")
        if isinstance(val_at_thr, dict) and val_at_thr:
            st.markdown("**Validation metrics at chosen threshold**")
            st.dataframe(pd.DataFrame([val_at_thr]), use_container_width=True, hide_index=True)

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
                st.markdown("**Per-class report (test)**")
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        cm = metrics.get("confusion_matrix")
        if isinstance(cm, list) and cm:
            st.markdown("**Confusion matrix** (rows=true, cols=predicted)")
            st.dataframe(pd.DataFrame(cm), use_container_width=True, hide_index=True)

        if METRICS_PATH and METRICS_PATH.exists():
            st.download_button(
                "Download metrics.json",
                data=METRICS_PATH.read_bytes(),
                file_name=METRICS_PATH.name,
                mime="application/json",
                use_container_width=True,
            )

with tab_threshold:
    st.subheader("Threshold tuning artifacts (optional)")
    st.write(
        "If your trained model produces probabilities, Stage 3 can sweep thresholds and save "
        "`reports/threshold_sweep.csv` so you can justify the chosen operating point."
    )

    if threshold_df.empty:
        st.info("No threshold_sweep.csv found. This is expected if the winning model does not provide probabilities.")
    else:
        st.dataframe(threshold_df, use_container_width=True)

        # Best-effort highlight
        sort_col = None
        for c in ("f1_fake", "macro_f1", "accuracy"):
            if c in threshold_df.columns:
                sort_col = c
                break

        if sort_col:
            tmp = threshold_df.copy()
            tmp[sort_col] = pd.to_numeric(tmp[sort_col], errors="coerce").fillna(-1.0)
            best_row = tmp.sort_values(sort_col, ascending=False, kind="mergesort").iloc[0]
            st.success(f"Best threshold by `{sort_col}` (from sweep table): {float(best_row.get('threshold', 0.5)):.3f}")

        if THRESHOLD_SWEEP_PATH and THRESHOLD_SWEEP_PATH.exists():
            st.download_button(
                "Download threshold_sweep.csv",
                data=THRESHOLD_SWEEP_PATH.read_bytes(),
                file_name=THRESHOLD_SWEEP_PATH.name,
                mime="text/csv",
                use_container_width=True,
            )

with tab_batch:
    st.subheader("Batch scoring results")
    st.write(
        "Stage 3 typically generates `reports/preds.csv` (test split predictions) from training. "
        "You can also score a full CSV using the Stage-3 `predict.py`, which can auto-load the chosen threshold:\n\n"
        "`python src/predict.py --model artifacts/best_model.joblib --metrics reports/metrics.json "
        "--data data/fake_and_real_news.csv --text-col Text --label-col label --out reports/preds_scored.csv`"
    )

    if preds_df.empty:
        st.info("No predictions file found yet (reports/preds.csv / reports/preds_scored.csv).")
    else:
        st.write(f"Rows: {len(preds_df):,}")

        show_preds = st.slider(
            "Prediction rows to display",
            min_value=10,
            max_value=min(5000, len(preds_df)),
            value=min(100, len(preds_df)),
            step=10,
        )
        st.dataframe(preds_df.head(show_preds), use_container_width=True)

        if PREDS_PATH and PREDS_PATH.exists():
            st.download_button(
                f"Download {PREDS_PATH.name}",
                data=PREDS_PATH.read_bytes(),
                file_name=PREDS_PATH.name,
                mime="text/csv",
                use_container_width=True,
            )

        cols = set(preds_df.columns)
        if {"y_true", "y_pred"}.issubset(cols):
            wrong = preds_df[preds_df["y_true"] != preds_df["y_pred"]]
            st.markdown("**Misclassified samples**")
            st.write(f"Count: {len(wrong):,}")
            st.dataframe(wrong.head(min(200, len(wrong))), use_container_width=True)

st.divider()
st.caption(
    "Tip: keep training and demo environments aligned (Python/numpy/scikit-learn). "
    "If you see version mismatch warnings, retrain the model in the same venv used to run Streamlit."
)
