# app.py (Stage 2) — Streamlit dashboard
# Run: streamlit run app.py

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

from src.config import CFG

# ----------------------------
# Optional: live quote (yfinance)
# ----------------------------
try:
    import yfinance as yf  # type: ignore
except Exception:
    yf = None

# ----------------------------
# Optional: timezone support (Py 3.9+)
# ----------------------------
try:
    from zoneinfo import ZoneInfo  # Py 3.9+
except Exception:
    ZoneInfo = None  # type: ignore


@st.cache_data(ttl=60)
def get_latest_quote(symbol: str) -> Dict[str, Any]:
    """
    Best-effort "current" price using yfinance.
    - Try 1-minute intraday (may be delayed/unavailable).
    - Fallback to last daily close (and prior close for delta).
    """
    sym = str(symbol).strip().upper()
    if not sym:
        return {"symbol": sym, "price": None, "prev": None, "source": "unavailable"}

    if yf is None:
        return {"symbol": sym, "price": None, "prev": None, "source": "yfinance-not-installed"}

    # Intraday (best effort)
    try:
        df = yf.download(sym, period="1d", interval="1m", progress=False, auto_adjust=False)
        if isinstance(df, pd.DataFrame) and not df.empty and "Close" in df.columns:
            closes = df["Close"].dropna()
            if len(closes) >= 1:
                last = float(closes.iloc[-1])
                prev = float(closes.iloc[0])
                return {"symbol": sym, "price": last, "prev": prev, "source": "intraday-1m"}
    except Exception:
        pass

    # Daily fallback
    try:
        df = yf.download(sym, period="5d", interval="1d", progress=False, auto_adjust=False)
        if isinstance(df, pd.DataFrame) and not df.empty and "Close" in df.columns:
            closes = df["Close"].dropna()
            if len(closes) >= 1:
                last = float(closes.iloc[-1])
                prev = float(closes.iloc[-2]) if len(closes) >= 2 else last
                return {"symbol": sym, "price": last, "prev": prev, "source": "daily-close"}
    except Exception:
        pass

    return {"symbol": sym, "price": None, "prev": None, "source": "unavailable"}


# ----------------------------
# Utilities
# ----------------------------
def _p(x: Any) -> Path:
    return x if isinstance(x, Path) else Path(str(x))


PROJECT_DIR = _p(getattr(CFG, "project_dir", Path.cwd()))
RUNS_DIR = _p(getattr(CFG, "runs_dir", PROJECT_DIR / "runs"))
DATA_PROCESSED = _p(getattr(CFG, "data_processed", PROJECT_DIR / "data" / "processed"))


def safe_read_text(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return None


def safe_read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def list_runs(runs_dir: Path) -> List[Path]:
    if not runs_dir.exists():
        return []
    runs = [p for p in runs_dir.glob("run_*") if p.is_dir()]
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return runs


def run_id_from_dir(run_dir: Path) -> str:
    name = run_dir.name
    return name.replace("run_", "", 1) if name.startswith("run_") else name


def find_processed_csv(run_dir: Path) -> Optional[Path]:
    """
    Prefer metrics.json 'processed_data_path' (saved by train.py).
    Otherwise fallback to newest matching <SYMBOL>_*_features.csv.
    """
    meta = safe_read_json(run_dir / "metrics.json") or {}
    processed_name = meta.get("processed_data_path")
    if processed_name:
        p = DATA_PROCESSED / str(processed_name)
        if p.exists():
            return p

    rid = run_id_from_dir(run_dir)
    symbol = rid.split("_", 1)[0].strip()
    matches = sorted(
        DATA_PROCESSED.glob(f"{symbol}_*_features.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return matches[0] if matches else None


def extract_features_from_metrics(run_dir: Path) -> List[str]:
    meta = safe_read_json(run_dir / "metrics.json") or {}
    feats = meta.get("features")
    if isinstance(feats, list) and feats:
        return [str(c) for c in feats]
    cfg_feats = getattr(CFG, "feature_cols", None) or getattr(CFG, "FEATURE_COLS", None) or []
    return [str(c) for c in cfg_feats]


def extract_target_from_metrics(run_dir: Path) -> str:
    meta = safe_read_json(run_dir / "metrics.json") or {}
    return str(
        meta.get("target_col")
        or getattr(CFG, "target_col", None)
        or getattr(CFG, "TARGET_COL", None)
        or "target_up"
    )


def extract_run_timestamp(meta: Dict[str, Any]) -> str:
    """
    train.py writes "timestamp" like YYYYMMDD_HHMMSS.
    If present, show it; otherwise return "—".
    """
    ts = meta.get("timestamp")
    if not ts:
        return "—"
    return str(ts)


def now_string(tz_name: str) -> str:
    """
    Return "YYYY-MM-DD HH:MM:SS TZ".
    If zoneinfo isn't available, return local time without TZ label.
    """
    if ZoneInfo is None:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        dt = datetime.now(ZoneInfo(tz_name))
        return dt.strftime("%Y-%m-%d %H:%M:%S %Z")
    except Exception:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def render_confusion_matrix(cm: np.ndarray, title: str) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(cm)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["0", "1"])
    ax.set_yticklabels(["0", "1"])

    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(int(v)), ha="center", va="center")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)


def model_feature_contributions(model: Any, feature_names: List[str]) -> Optional[pd.DataFrame]:
    """
    Returns a dataframe with columns: feature, contribution
    - LogisticRegression: coef_
    - RandomForest / GradientBoosting: feature_importances_
    """
    # Pipeline(LogReg)
    try:
        if hasattr(model, "named_steps") and "clf" in model.named_steps:
            clf = model.named_steps["clf"]
            if hasattr(clf, "coef_"):
                coef = np.asarray(clf.coef_).reshape(-1)
                n = min(len(feature_names), len(coef))
                return pd.DataFrame({"feature": feature_names[:n], "contribution": coef[:n]})
    except Exception:
        pass

    # Direct estimator with coef_
    try:
        if hasattr(model, "coef_"):
            coef = np.asarray(model.coef_).reshape(-1)
            n = min(len(feature_names), len(coef))
            return pd.DataFrame({"feature": feature_names[:n], "contribution": coef[:n]})
    except Exception:
        pass

    # Feature importances
    try:
        if hasattr(model, "feature_importances_"):
            imp = np.asarray(model.feature_importances_).reshape(-1)
            n = min(len(feature_names), len(imp))
            return pd.DataFrame({"feature": feature_names[:n], "contribution": imp[:n]})
    except Exception:
        pass

    return None


def barh_top_n(df: pd.DataFrame, n: int, title: str, is_importance: bool = False) -> None:
    if df is None or df.empty:
        st.info("No contributions available.")
        return

    vals = df["contribution"].astype(float).values
    if is_importance:
        order = np.argsort(vals)[::-1][:n]
    else:
        order = np.argsort(np.abs(vals))[::-1][:n]

    top = df.iloc[order].copy()
    top = top.sort_values("contribution", ascending=True)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.barh(top["feature"], top["contribution"])
    ax.set_title(title)
    ax.set_xlabel("Contribution")
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)

def autorefresh(seconds: int = 60) -> None:
    # Browser-level refresh; triggers a Streamlit rerun
    components.html(f"<meta http-equiv='refresh' content='{int(seconds)}'>", height=0)
# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Next-Day Direction Classifier with RAG — Stage 2", layout="wide")

# Auto-refresh (UI + "Now" + live quote). 60s.
with st.sidebar:
    st.header("Auto-refresh")
    enable_refresh = st.toggle("Refresh every 60 seconds", value=True)

if enable_refresh:
    autorefresh(60)

st.title("Next-Day Direction Classifier with RAG — Stage 2")
st.caption("Run browser + evaluation artifacts + model interpretability (coefficients / feature importances).")

runs = list_runs(RUNS_DIR)

with st.sidebar:
    st.header("Maintenance")

    if st.button("Clear Streamlit cache"):
        try:
            st.cache_data.clear()
        except Exception:
            pass
        try:
            st.cache_resource.clear()
        except Exception:
            pass
        st.success("Cache cleared.")
        st.rerun()

    if st.button("Reset UI (session state)"):
        st.session_state.clear()
        st.success("Session reset.")
        st.rerun()

with st.sidebar:
    st.header("Runs")
    if not runs:
        st.error(f"No runs found under: {RUNS_DIR}")
        st.stop()

    run_labels = [run_id_from_dir(r) for r in runs]
    selected_label = st.selectbox("Select a run", run_labels, index=0)
    selected_run = runs[run_labels.index(selected_label)]

    st.divider()
    st.write("**Timezone / Now**")
    tz = st.selectbox("Timezone", ["America/New_York", "America/Los_Angeles", "UTC"], index=0)

    st.divider()
    st.write("**Paths**")
    st.code(str(PROJECT_DIR))
    st.code(str(RUNS_DIR))
    st.code(str(DATA_PROCESSED))

    st.divider()
    st.write("**Live quote** (optional)")
    if yf is None:
        st.warning("yfinance not installed. Install: `python -m pip install yfinance`")

run_dir = selected_run
rid = run_id_from_dir(run_dir)

meta_path = run_dir / "metrics.json"
report_path = run_dir / "report.md"
model_path = run_dir / "model.joblib"
val_metrics_path = run_dir / "val_metrics.json"
test_metrics_path = run_dir / "test_metrics.json"
cm_png_path = run_dir / "confusion_matrix_test.png"

meta = safe_read_json(meta_path) or {}
processed_csv = find_processed_csv(run_dir)
features = extract_features_from_metrics(run_dir)
target_col = extract_target_from_metrics(run_dir)

symbol = str(meta.get("symbol") or rid.split("_", 1)[0]).strip().upper()
run_ts = extract_run_timestamp(meta)

# Live quote panel (best-effort)
q = get_latest_quote(symbol)
qc1, qc2, qc3, qc4, qc5 = st.columns(5)
qc1.metric("Symbol", symbol)
if q.get("price") is None:
    qc2.metric("Price", "—")
    qc3.metric("Δ", "—")
else:
    price = float(q["price"])
    prev = float(q.get("prev") or price)
    delta = price - prev
    pct = (delta / prev * 100.0) if prev else 0.0
    qc2.metric("Price (best-effort)", f"${price:.2f}")
    qc3.metric("Δ vs prev", f"{delta:+.2f} ({pct:+.2f}%)")
qc4.metric("Quote source", str(q.get("source", "—")))
qc5.metric("Now", now_string(tz))

# Top summary
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Run ID", rid)
c2.metric("Best model", meta.get("best_model", "—"))
c3.metric("Rows after features", int(meta.get("n_rows_after_features", 0) or 0))
c4.metric("Target", target_col)
c5.metric("Run timestamp", run_ts)

st.divider()

tab1, tab2, tab3, tab4 = st.tabs(["Run Artifacts", "Evaluation", "Model Insights", "Data Preview"])

# ----------------------------
# Tab: Run Artifacts
# ----------------------------
with tab1:
    left, right = st.columns([1, 1])

    with left:
        st.subheader("metrics.json")
        if meta:
            st.json(meta)
        else:
            st.warning(f"Missing or unreadable: {meta_path}")

    with right:
        st.subheader("report.md")
        report_txt = safe_read_text(report_path)
        if report_txt:
            st.markdown(report_txt)
        else:
            st.warning(f"Missing or unreadable: {report_path}")

    st.subheader("Candidate models (VAL) — from metrics.json")
    cand = meta.get("candidates_val")
    if isinstance(cand, dict) and cand:
        rows = []
        for name, d in cand.items():
            rows.append(
                {
                    "model": name,
                    "accuracy": float(d.get("accuracy", 0.0)),
                    "macro_f1": float(d.get("macro_f1", 0.0)),
                    "f1_up": float(d.get("f1_up", 0.0)),
                    "f1_down": float(d.get("f1_down", 0.0)),
                }
            )
        df_cand = pd.DataFrame(rows).sort_values(["macro_f1", "accuracy"], ascending=False)
        st.dataframe(df_cand, use_container_width=True)
    else:
        st.info("No candidates_val found in metrics.json (train may not have saved comparison results).")

# ----------------------------
# Tab: Evaluation
# ----------------------------
with tab2:
    st.subheader("Evaluation outputs")

    if not model_path.exists():
        st.error(f"Missing model artifact: {model_path}")
        st.stop()

    cols = st.columns(2)

    with cols[0]:
        st.write("**val_metrics.json**")
        valm = safe_read_json(val_metrics_path)
        if valm:
            st.json(valm)
        else:
            st.warning("Not found yet. Run:")
            st.code(f"python -m src.evaluate --run-id {rid} --verbose")

    with cols[1]:
        st.write("**test_metrics.json**")
        testm = safe_read_json(test_metrics_path)
        if testm:
            st.json(testm)
        else:
            st.warning("Not found yet. Run:")
            st.code(f"python -m src.evaluate --run-id {rid} --verbose")

    st.divider()
    st.subheader("Confusion matrix (TEST)")

    testm = safe_read_json(test_metrics_path)
    if testm and "confusion_matrix" in testm:
        cm = np.array(testm["confusion_matrix"]["matrix"], dtype=int)
        render_confusion_matrix(cm, "Confusion Matrix (TEST)")
    elif cm_png_path.exists():
        st.image(str(cm_png_path), caption="confusion_matrix_test.png")
    else:
        st.info("No confusion matrix found yet. Run evaluation first.")

# ----------------------------
# Tab: Model Insights
# ----------------------------
with tab3:
    st.subheader("Model interpretability")

    st.write("**Features used**")
    st.code(", ".join(features) if features else "(none)")

    try:
        model = joblib.load(model_path)
    except Exception as e:
        st.error(f"Failed to load model.joblib: {e}")
        st.stop()

    contrib = model_feature_contributions(model, features)
    if contrib is None:
        st.info("This model does not expose coefficients or feature_importances_.")
    else:
        vals = contrib["contribution"].astype(float)
        is_importance = bool(vals.min() >= 0.0 and vals.max() <= 1.0)

        n_max = max(5, len(contrib))
        n = st.slider("Top N", min_value=5, max_value=min(25, n_max), value=min(10, min(25, n_max)))
        title = "Top feature importances" if is_importance else "Top coefficients (standardized if LogReg pipeline)"
        barh_top_n(contrib, n=n, title=title, is_importance=is_importance)

        st.write("All contributions")
        st.dataframe(contrib.sort_values("contribution", ascending=False), use_container_width=True)

# ----------------------------
# Tab: Data Preview
# ----------------------------
with tab4:
    st.subheader("Processed dataset preview")

    if processed_csv and processed_csv.exists():
        st.write(f"Using: `{processed_csv}`")
        try:
            df = pd.read_csv(processed_csv)
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            st.write("Columns:")
            st.code(", ".join(df.columns))
            st.dataframe(df.head(50), use_container_width=True)
        except Exception as e:
            st.error(f"Failed to read processed CSV: {e}")
    else:
        st.warning("No processed CSV located for this run.")
        st.write("Train should create one in `data/processed/` and write its filename into `runs/<run>/metrics.json`.")

st.divider()
st.caption(
    "Workflow: 1) python -m src.train  2) python -m src.evaluate --run-id <RID>  "
    "3) python -m src.rag_index  4) python -m src.rag_chat  5) streamlit run app.py"
)