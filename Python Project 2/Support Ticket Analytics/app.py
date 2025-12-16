from pathlib import Path

import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html
from dash import dash_table

BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

VOLUME_CSV = PROCESSED_DIR / "kpi_ticket_volume_daily.csv"
PRIORITY_CSV = PROCESSED_DIR / "kpi_by_priority.csv"

app = Dash(__name__)
app.title = "Support Ticket Analytics"


def load_volume_df() -> pd.DataFrame | None:
    if not VOLUME_CSV.exists():
        return None
    df = pd.read_csv(VOLUME_CSV)
    # Expect: date, tickets
    if "date" not in df.columns or "tickets" not in df.columns:
        return None
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")
    return df


def load_priority_df() -> pd.DataFrame | None:
    if not PRIORITY_CSV.exists():
        return None
    df = pd.read_csv(PRIORITY_CSV)
    # Expect: priority, total_tickets, met_sla_pct, avg_resolution_hours, ...
    if "priority" not in df.columns:
        return None
    return df


vol_df = load_volume_df()
prio_df = load_priority_df()

left_panel = (
    dcc.Graph(
        figure=px.line(vol_df, x="date", y="tickets", title="Ticket Volume Trend"),
    )
    if vol_df is not None
    else html.Div("Ticket volume KPI not found. Run: python etl_build_mart.py", style={"color": "crimson"})
)

right_panel = (
    html.Div(
        children=[
            html.H3("KPI by Priority (SLA proxy + resolution time)"),
            dash_table.DataTable(
                data=prio_df.to_dict("records"),
                columns=[{"name": c, "id": c} for c in prio_df.columns],
                page_size=10,
                style_table={"overflowX": "auto"},
                style_cell={"padding": "8px", "fontFamily": "system-ui", "fontSize": "13px"},
            ),
            dcc.Graph(
                figure=px.bar(prio_df, x="priority", y="met_sla_pct", title="Met SLA % by Priority")
                if "met_sla_pct" in prio_df.columns
                else px.bar(prio_df, x="priority", y="total_tickets", title="Total Tickets by Priority"),
            ),
        ]
    )
    if prio_df is not None
    else html.Div("Priority KPI not found. Run: python etl_build_mart.py", style={"color": "crimson"})
)

app.layout = html.Div(
    style={"maxWidth": "1200px", "margin": "24px auto", "fontFamily": "system-ui"},
    children=[
        html.H1("Support Ticket Analytics Dashboard"),
        html.Div(
            style={"display": "grid", "gridTemplateColumns": "2fr 1fr", "gap": "16px", "alignItems": "start"},
            children=[left_panel, right_panel],
        ),
        html.Hr(),
        html.Details(
            children=[
                html.Summary("Debug (paths)"),
                html.Pre(
                    f"BASE_DIR: {BASE_DIR}\n"
                    f"PROCESSED_DIR: {PROCESSED_DIR}\n"
                    f"VOLUME_CSV exists: {VOLUME_CSV.exists()} -> {VOLUME_CSV}\n"
                    f"PRIORITY_CSV exists: {PRIORITY_CSV.exists()} -> {PRIORITY_CSV}\n"
                ),
            ]
        ),
    ],
)

if __name__ == "__main__":
    app.run(debug=True)