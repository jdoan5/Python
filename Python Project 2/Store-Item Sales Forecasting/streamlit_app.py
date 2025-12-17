# Store-Item Sales Forecasting — Streamlit Dashboard
# Run: streamlit run streamlit_app.py

from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Store-Item Sales Forecasting", layout="wide")

ROOT = Path(__file__).resolve().parent

DATA_PATH = ROOT / "data.csv"
FORECAST_PATH = ROOT / "forecast.csv"

st.title("Store-Item Sales Forecasting Dashboard")
st.caption("Lightweight dashboard for inspecting historical demand and generated forecasts.")

# ---------- Helpers ----------
def load_csv(path: Path, parse_dates=None) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path, parse_dates=parse_dates)

def pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first matching column from candidates (case-insensitive)."""
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None

def ensure_datetime(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df = df.copy()
    df[col] = pd.to_datetime(df[col], errors="coerce")
    return df.dropna(subset=[col])

# ---------- Load ----------
df_hist = load_csv(DATA_PATH)
df_fcst = load_csv(FORECAST_PATH)

if df_hist is not None:
    date_col = pick_col(df_hist, ["date", "ds", "timestamp"])
    if date_col:
        df_hist = ensure_datetime(df_hist, date_col)

if df_fcst is not None:
    fcst_date_col = pick_col(df_fcst, ["date", "ds", "timestamp"])
    if fcst_date_col:
        df_fcst = ensure_datetime(df_fcst, fcst_date_col)

# ---------- Sidebar filters ----------
st.sidebar.header("Filters")

store_col = pick_col(df_hist, ["store", "store_id"]) if df_hist is not None else None
item_col = pick_col(df_hist, ["item", "item_id", "sku"]) if df_hist is not None else None

selected_store = None
selected_item = None

if df_hist is not None and not df_hist.empty:
    if store_col:
        stores = sorted(df_hist[store_col].dropna().unique().tolist())
        selected_store = st.sidebar.selectbox("Store", ["All"] + stores, index=0)

    if item_col:
        items = sorted(df_hist[item_col].dropna().unique().tolist())
        selected_item = st.sidebar.selectbox("Item", ["All"] + items, index=0)

# ---------- Apply filters ----------
def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    if store_col and selected_store and selected_store != "All":
        out = out[out[store_col] == selected_store]
    if item_col and selected_item and selected_item != "All":
        out = out[out[item_col] == selected_item]
    return out

df_hist_f = apply_filters(df_hist) if df_hist is not None else None

# Forecast may have same store/item cols, but detect separately
if df_fcst is not None and not df_fcst.empty:
    fcst_store_col = pick_col(df_fcst, ["store", "store_id"])
    fcst_item_col = pick_col(df_fcst, ["item", "item_id", "sku"])

    df_fcst_f = df_fcst.copy()
    if fcst_store_col and selected_store and selected_store != "All":
        df_fcst_f = df_fcst_f[df_fcst_f[fcst_store_col] == selected_store]
    if fcst_item_col and selected_item and selected_item != "All":
        df_fcst_f = df_fcst_f[df_fcst_f[fcst_item_col] == selected_item]
else:
    df_fcst_f = None

# ---------- KPI row ----------
k1, k2, k3, k4 = st.columns(4)

if df_hist_f is not None and not df_hist_f.empty:
    hist_date_col = pick_col(df_hist_f, ["date", "ds", "timestamp"])
    y_col = pick_col(df_hist_f, ["sales", "y", "demand", "units", "qty"])
    min_d = df_hist_f[hist_date_col].min() if hist_date_col else None
    max_d = df_hist_f[hist_date_col].max() if hist_date_col else None
    total_rows = len(df_hist_f)
else:
    y_col = None
    min_d = max_d = None
    total_rows = None

if df_fcst_f is not None and not df_fcst_f.empty:
    fcst_y_col = pick_col(df_fcst_f, ["forecast", "yhat", "pred", "prediction", "y_pred", "sales_forecast"])
    horizon = len(df_fcst_f)
else:
    fcst_y_col = None
    horizon = None

with k1:
    st.metric("Rows (filtered)", f"{total_rows:,}" if total_rows is not None else "—")
with k2:
    st.metric("Date range", f"{min_d.date()} → {max_d.date()}" if min_d is not None and max_d is not None else "—")
with k3:
    st.metric("Forecast horizon", f"{horizon:,} rows" if horizon is not None else "—")
with k4:
    st.metric("Data sources", "data.csv + forecast.csv")

st.divider()

# ---------- Charts ----------
left, right = st.columns([2, 1])

with left:
    st.subheader("Historical Sales Trend")

    if df_hist_f is None or df_hist_f.empty:
        st.warning(f"Missing or empty {DATA_PATH.name}. Generate it first (e.g., `python make_data.py`).")
    else:
        hist_date_col = pick_col(df_hist_f, ["date", "ds", "timestamp"])
        y_col = pick_col(df_hist_f, ["sales", "y", "demand", "units", "qty"])

        if not hist_date_col or not y_col:
            st.error("Could not detect required columns in data.csv. Expected a date column and a sales/y column.")
            st.write("Columns found:", list(df_hist_f.columns))
        else:
            fig = px.line(df_hist_f.sort_values(hist_date_col), x=hist_date_col, y=y_col)
            fig.update_layout(height=420, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig, width="stretch")

with right:
    st.subheader("Forecast Preview")

    if df_fcst_f is None or df_fcst_f.empty:
        st.warning(f"Missing or empty {FORECAST_PATH.name}. Run forecasting first (e.g., `python forecast.py`).")
    else:
        st.dataframe(df_fcst_f.head(25), width="stretch", hide_index=True)

        fcst_date_col = pick_col(df_fcst_f, ["date", "ds", "timestamp"])
        fcst_y_col = pick_col(df_fcst_f, ["forecast", "yhat", "pred", "prediction", "y_pred", "sales_forecast"])

        if fcst_date_col and fcst_y_col:
            fig2 = px.line(df_fcst_f.sort_values(fcst_date_col), x=fcst_date_col, y=fcst_y_col)
            fig2.update_layout(height=260, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig2, width="stretch")
        else:
            st.info("Could not auto-detect forecast columns for a chart. Showing table only.")
            st.write("Columns found:", list(df_fcst_f.columns))

st.divider()

with st.expander("Debug (paths & columns)"):
    st.write("Project root:", str(ROOT))
    st.write("data.csv exists:", DATA_PATH.exists(), str(DATA_PATH))
    st.write("forecast.csv exists:", FORECAST_PATH.exists(), str(FORECAST_PATH))
    if df_hist is not None:
        st.write("data.csv columns:", list(df_hist.columns))
    if df_fcst is not None:
        st.write("forecast.csv columns:", list(df_fcst.columns))