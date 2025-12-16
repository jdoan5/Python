# Optional 2: Streamlit Dashboard
# Command to run: streamlit run streamlit_app.py

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Support Ticket Analytics", layout="wide")

ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "data" / "processed"

VOLUME_PATH = PROCESSED / "kpi_ticket_volume_daily.csv"
PRIORITY_PATH = PROCESSED / "kpi_by_priority.csv"

st.title("Support Ticket Analytics Dashboard")
st.caption("Ops KPIs + lightweight text analytics (synthetic data + ETL + mini mart)")

# ---------- Helpers ----------
def load_csv(path: Path, parse_dates=None):
    if not path.exists():
        return None
    return pd.read_csv(path, parse_dates=parse_dates)

# ---------- Top row: KPI cards ----------
colA, colB, colC, colD = st.columns(4)

df_volume = load_csv(VOLUME_PATH, parse_dates=["date"])
df_priority = load_csv(PRIORITY_PATH)

# Basic headline KPIs from daily volume (if present)
if df_volume is not None and not df_volume.empty:
    total_tickets = int(df_volume["tickets_count"].sum()) if "tickets_count" in df_volume.columns else int(df_volume.iloc[:, 1].sum())
    avg_daily = float(df_volume["tickets_count"].mean()) if "tickets_count" in df_volume.columns else float(df_volume.iloc[:, 1].mean())

    # Use safe column detection
    tickets_col = "tickets_count" if "tickets_count" in df_volume.columns else df_volume.columns[1]
    df_volume_sorted = df_volume.sort_values("date")
    last_7 = df_volume_sorted.tail(7)[tickets_col].sum()
    last_14 = df_volume_sorted.tail(14)[tickets_col].sum()
    trend = (last_7 - (last_14 - last_7))  # rough “delta vs prior 7 days”
else:
    total_tickets = None
    avg_daily = None
    trend = None

with colA:
    st.metric("Total tickets", f"{total_tickets:,}" if total_tickets is not None else "—")
with colB:
    st.metric("Avg tickets/day", f"{avg_daily:,.2f}" if avg_daily is not None else "—")
with colC:
    st.metric("7-day delta (rough)", f"{trend:+,}" if trend is not None else "—")
with colD:
    st.metric("Data sources", "processed CSVs")

st.divider()

# ---------- Charts ----------
left, right = st.columns([2, 1])

with left:
    st.subheader("Ticket Volume Trend")

    if df_volume is None or df_volume.empty:
        st.warning(
            f"Missing {VOLUME_PATH.relative_to(ROOT)}. Run: `python etl_build_mart.py`"
        )
    else:
        tickets_col = "tickets_count" if "tickets_count" in df_volume.columns else df_volume.columns[1]
        fig = px.line(df_volume.sort_values("date"), x="date", y=tickets_col)
        fig.update_layout(height=420, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("KPI by Priority (SLA proxy + resolution time)")

    if df_priority is None or df_priority.empty:
        st.warning(
            f"Missing {PRIORITY_PATH.relative_to(ROOT)}. Run: `python etl_build_mart.py`"
        )
    else:
        # Expect: priority, total_tickets, met_sla_pct, avg_resolution_hours
        cols = df_priority.columns.str.lower().tolist()

        # Show table (clean and screenshot-friendly)
        st.dataframe(df_priority, use_container_width=True, hide_index=True)

        # Optional bar chart if expected columns exist
        if "priority" in cols and ("met_sla_pct" in cols or "met_sla_pct".upper() in df_priority.columns):
            # Try common names
            priority_col = next((c for c in df_priority.columns if c.lower() == "priority"), df_priority.columns[0])
            sla_col = next((c for c in df_priority.columns if c.lower() in {"met_sla_pct", "sla_pct", "sla_percent"}), None)

            if sla_col:
                fig2 = px.bar(df_priority, x=priority_col, y=sla_col)
                fig2.update_layout(height=260, margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ---------- Footer / Debug (optional) ----------
with st.expander("Debug (paths)"):
    st.write("Project root:", str(ROOT))
    st.write("Processed folder:", str(PROCESSED))
    st.write("Volume CSV exists:", VOLUME_PATH.exists(), str(VOLUME_PATH))
    st.write("Priority CSV exists:", PRIORITY_PATH.exists(), str(PRIORITY_PATH))