from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import date

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from src.data_fetch import fetch_from_yahoo
from src.features import feature_columns


ROOT = Path(__file__).resolve().parent
RUNS_DIR = ROOT / "runs"


def list_runs(runs_dir: Path) -> List[Path]:
    if not runs_dir.exists():
        return []
    runs = [p for p in runs_dir.iterdir() if p.is_dir() and p.name.startswith("run_")]
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return runs


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def read_text(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None


def load_model(path: Path):
    if not path.exists():
        return None
    try:
        return joblib.load(path)
    except Exception:
        return None


def extract_lr_coefficients(model, feature_names: List[str]) -> Optional[pd.DataFrame]:
    """
    Supports your Pipeline:
      ("scaler", StandardScaler()), ("clf", LogisticRegression(...))
    """
    try:
        clf = model.named_steps.get("clf", None)
        if clf is None or not hasattr(clf, "coef_"):
            return None

        coefs = clf.coef_.ravel()
        if len(coefs) != len(feature_names):
            return None

        df = pd.DataFrame({"feature": feature_names, "coef": coefs})
        df["abs_coef"] = df["coef"].abs()
        df = df.sort_values("abs_coef", ascending=False).reset_index(drop=True)
        return df
    except Exception:
        return None


def plot_confusion_matrix(cm: List[List[int]]):
    fig, ax = plt.subplots()
    ax.imshow(cm)
    ax.set_title("Confusion Matrix (TEST)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    # annotate cells
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i][j]), ha="center", va="center")

    st.pyplot(fig)


def build_features_for_inference(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Same feature math as src/features.py, but WITHOUT creating target_up,
    so we can predict the next day from the latest available trading day.
    """
    out = raw.copy()

    out["ret_1"] = out["Close"].pct_change(1)
    out["ret_5"] = out["Close"].pct_change(5)

    out["ret_1_ma_5"] = out["ret_1"].rolling(5).mean()
    out["ret_1_ma_10"] = out["ret_1"].rolling(10).mean()

    out["ret_1_vol_5"] = out["ret_1"].rolling(5).std()
    out["ret_1_vol_10"] = out["ret_1"].rolling(10).std()

    out["hl_range"] = (out["High"] - out["Low"]) / out["Close"]
    out["oc_change"] = (out["Close"] - out["Open"]) / out["Open"]

    out["vol_chg_1"] = out["Volume"].pct_change(1).replace([np.inf, -np.inf], np.nan)
    out["log_vol"] = np.log1p(out["Volume"])

    feats = feature_columns()
    out = out.dropna(subset=feats).reset_index(drop=True)
    return out


def latest_next_day_prediction(symbol: str, model) -> Dict[str, Any]:
    """
    Fetches recent data, builds inference features (no target),
    and predicts next-day direction from the most recent trading day.
    """
    # Keep it reasonably fast: last 5 years is usually enough for rolling features + context
    start = f"{date.today().year - 5}-01-01"
    raw = fetch_from_yahoo(symbol, start=start, end=None)

    df = build_features_for_inference(raw)
    feats = feature_columns()

    if df.empty:
        raise RuntimeError("No rows after feature engineering for inference.")

    x_last = df[feats].iloc[[-1]]
    proba_up = float(model.predict_proba(x_last)[0, 1])
    pred_up = proba_up >= 0.5

    last_date = pd.to_datetime(df["Date"].iloc[-1])
    last_close = float(df["Close"].iloc[-1])

    # For chart: show last ~180 trading days from RAW
    chart_df = raw.copy()
    chart_df["Date"] = pd.to_datetime(chart_df["Date"])
    chart_df = chart_df.sort_values("Date").tail(180).set_index("Date")

    return {
        "last_date": last_date,
        "last_close": last_close,
        "proba_up": proba_up,
        "pred_label": "UP" if pred_up else "DOWN",
        "chart_close": chart_df["Close"],
    }


st.set_page_config(page_title="Finance Direction — Run Dashboard", layout="wide")

st.title("Next-Day Direction Classifier — Run Dashboard")
st.caption("Reads artifacts from ./runs (metrics.json, report.md, model.joblib, test_metrics.json).")
st.caption("Disclaimer: This project is for educational and portfolio use. It does not provide financial advice and should not be used as a sole basis for investment decisions.")

# Sidebar
st.sidebar.header("Run Selection")

runs = list_runs(RUNS_DIR)
if not runs:
    st.warning("No runs found in ./runs. Run training first:\n\n`python -m src.train`")
    st.stop()

run_names = [p.name for p in runs]
selected_name = st.sidebar.selectbox("Select a run", run_names, index=0)
run_dir = RUNS_DIR / selected_name

st.sidebar.write("Selected:", str(run_dir))

metrics = read_json(run_dir / "metrics.json")
test_metrics = read_json(run_dir / "test_metrics.json")
report_md = read_text(run_dir / "report.md")
model = load_model(run_dir / "model.joblib")

# Top summary row
c1, c2, c3, c4 = st.columns(4)

symbol = (metrics or {}).get("symbol", "Unknown")
stamp = (metrics or {}).get("timestamp", "Unknown")
baseline_val = (metrics or {}).get("baseline_val_accuracy", None)
model_val = (metrics or {}).get("model_val_accuracy", None)
test_acc = (test_metrics or {}).get("test_accuracy", None)

c1.metric("Symbol", symbol)
c2.metric("Run timestamp", stamp)
c3.metric("Baseline VAL acc", f"{baseline_val:.3f}" if isinstance(baseline_val, (int, float)) else "—")
c4.metric("Model VAL acc", f"{model_val:.3f}" if isinstance(model_val, (int, float)) else "—")

if isinstance(test_acc, (int, float)):
    st.metric("TEST accuracy", f"{test_acc:.3f}")

st.divider()

# NEW: Latest UP/DOWN indicator + probability + recent Close chart
st.subheader("Latest Next-Day Direction (from latest trading day)")

if model is None or symbol == "Unknown":
    st.info("Need a valid model.joblib + symbol (run training first).")
else:
    try:
        pred = latest_next_day_prediction(symbol, model)

        a, b, c = st.columns([1.2, 1, 1])
        a.metric("Last market date", pred["last_date"].date().isoformat())
        b.metric("Last Close", f"{pred['last_close']:.2f}")
        c.metric("Prediction", pred["pred_label"], delta=f"{pred['proba_up']:.3f} P(UP)")

        st.progress(min(max(pred["proba_up"], 0.0), 1.0))
        st.caption("Interpretation: UP means the model predicts Close[t+1] > Close[t].")

        st.write("**Recent Close (last ~180 trading days)**")
        st.line_chart(pred["chart_close"])

    except Exception as e:
        st.warning(f"Could not compute latest prediction: {e}")

st.divider()

# Layout: left = metrics/report, right = model insights
left, right = st.columns([1.2, 1])

with left:
    st.subheader("Run Artifacts")

    if metrics:
        st.write("**metrics.json**")
        st.json(metrics)
    else:
        st.info("metrics.json not found for this run.")

    if test_metrics:
        st.write("**test_metrics.json**")
        st.json(test_metrics)
    else:
        st.info("test_metrics.json not found yet. Run evaluation:\n\n`python -m src.evaluate <run_dir>`")

    if report_md:
        st.write("**report.md**")
        st.markdown(report_md)
    else:
        st.info("report.md not found for this run.")

with right:
    st.subheader("Model Insights")

    if test_metrics and isinstance(test_metrics.get("confusion_matrix", None), list):
        cm = test_metrics["confusion_matrix"]
        if len(cm) == 2 and len(cm[0]) == 2:
            plot_confusion_matrix(cm)
        else:
            st.info("Confusion matrix present but not 2x2.")

    # Feature coefficients (Logistic Regression)
    if metrics and model:
        feats = metrics.get("features", [])
        coef_df = extract_lr_coefficients(model, feats if isinstance(feats, list) else [])
        if coef_df is not None and not coef_df.empty:
            st.write("**Logistic Regression coefficients (standardized features)**")
            st.dataframe(coef_df[["feature", "coef"]], use_container_width=True)

            top_n = st.slider("Show top N coefficients", 5, min(25, len(coef_df)), 10)
            top = coef_df.head(top_n).iloc[::-1]
            fig, ax = plt.subplots()
            ax.barh(top["feature"], top["coef"])
            ax.set_title(f"Top {top_n} coefficients")
            ax.set_xlabel("Coefficient (standardized)")
            st.pyplot(fig)
        else:
            st.info("Could not extract coefficients (ensure model is LogisticRegression in a Pipeline).")
    else:
        st.info("model.joblib not found (or metrics missing).")

st.divider()
st.caption("Tip: Train/evaluate to generate artifacts, then refresh the page.")