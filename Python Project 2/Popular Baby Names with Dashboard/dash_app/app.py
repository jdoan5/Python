import pathlib

import pandas as pd
from dash import Dash, dcc, html, Input, Output
import plotly.express as px

# --- Paths & data -----------------------------------------------------------------

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent  # go up from dash_app/ to project folder
DATA_PATH = BASE_DIR / "baby_names_trend.csv"

df = pd.read_csv(DATA_PATH)


def _norm(s: str) -> str:
    """Normalize a column name for matching."""
    return (
        str(s)
        .strip()
        .lower()
        .replace(" ", "")
        .replace("_", "")
        .replace("'", "")
    )


def choose_col(candidates, required=True):
    """
    Return the actual column name in df that best matches any of the
    normalized candidate keys (e.g., 'year', 'yearofbirth').
    """
    norm_map = {_norm(col): col for col in df.columns}
    for key in candidates:
        nkey = _norm(key)
        if nkey in norm_map:
            return norm_map[nkey]
    if required:
        raise KeyError(
            f"None of {candidates} found in CSV columns: {list(df.columns)}"
        )
    return None


# Map friendly names to whatever is in the CSV
Year = choose_col(["year", "year of birth", "yearofbirth"])
FirstName = choose_col(["firstname", "child's first name", "childsfirstname", "name"])
Gender = choose_col(["gender"])
Ethnicity = choose_col(["ethnicity", "ethnicgroup"])
Count = choose_col(["count", "babies", "number"])
Trend = choose_col(["trend_label", "trendlabel", "trend"])

# Normalize year to numeric once
df[Year] = pd.to_numeric(df[Year], errors="coerce")

# Pre-compute option lists
year_options = sorted(df[Year].dropna().astype(int).unique())
gender_options = sorted(df[Gender].dropna().unique())
trend_options = sorted(df[Trend].dropna().unique())
ethnicity_options = sorted(df[Ethnicity].dropna().unique())

# Default filters
default_year = max(year_options) if year_options else None

# --- App setup --------------------------------------------------------------------

app = Dash(__name__, title="NYC Popular Baby Names – Dash")

app.layout = html.Div(
    className="container",
    children=[
        html.H1("NYC Popular Baby Names – Dash", style={"marginTop": "20px"}),
        html.P(
            "Interactive view on top of the prepared baby_names_trend.csv. "
            "Filter by year, gender, trend label, or ethnicity and explore top names.",
            style={"maxWidth": "60rem"},
        ),

        # Filters
        html.Div(
            className="row",
            style={"marginTop": "20px", "marginBottom": "10px"},
            children=[
                html.Div(
                    className="col-md-3 col-12 mb-2",
                    children=[
                        html.Label("Year"),
                        dcc.Dropdown(
                            id="year-filter",
                            options=[{"label": str(y), "value": int(y)} for y in year_options],
                            value=int(default_year) if default_year is not None else None,
                            clearable=True,
                        ),
                    ],
                ),
                html.Div(
                    className="col-md-3 col-12 mb-2",
                    children=[
                        html.Label("Gender"),
                        dcc.Dropdown(
                            id="gender-filter",
                            options=[{"label": g, "value": g} for g in gender_options],
                            value=None,
                            placeholder="All",
                            clearable=True,
                        ),
                    ],
                ),
                html.Div(
                    className="col-md-3 col-12 mb-2",
                    children=[
                        html.Label("Trend label"),
                        dcc.Dropdown(
                            id="trend-filter",
                            options=[{"label": t, "value": t} for t in trend_options],
                            value=None,
                            placeholder="All",
                            clearable=True,
                        ),
                    ],
                ),
                html.Div(
                    className="col-md-3 col-12 mb-2",
                    children=[
                        html.Label("Ethnicity"),
                        dcc.Dropdown(
                            id="ethnicity-filter",
                            options=[{"label": e, "value": e} for e in ethnicity_options],
                            value=None,
                            placeholder="All",
                            clearable=True,
                        ),
                    ],
                ),
            ],
        ),

        html.Hr(),

        # Row: top names bar + name trend line
        html.Div(
            className="row",
            children=[
                html.Div(
                    className="col-md-6 col-12",
                    children=[
                        html.H4("Top 10 names (by count)"),
                        dcc.Graph(id="top-names-chart"),
                    ],
                ),
                html.Div(
                    className="col-md-6 col-12",
                    children=[
                        html.H4("Trend over time for a selected name"),
                        html.Div(
                            style={"maxWidth": "20rem", "marginBottom": "10px"},
                            children=[
                                html.Label("Name"),
                                dcc.Dropdown(
                                    id="name-filter",
                                    options=[
                                        {"label": n, "value": n}
                                        for n in sorted(df[FirstName].dropna().unique())
                                    ],
                                    value=None,
                                    placeholder="Choose a name",
                                    clearable=True,
                                ),
                            ],
                        ),
                        dcc.Graph(id="name-trend-chart"),
                    ],
                ),
            ],
        ),
    ],
)

# --- Helpers & Callbacks --------------------------------------------------------------------


def _apply_filters(data, year, gender, trend, ethnicity):
    dff = data.copy()
    if year is not None:
        dff = dff[dff[Year] == year]
    if gender:
        dff = dff[dff[Gender] == gender]
    if trend:
        dff = dff[dff[Trend] == trend]
    if ethnicity:
        dff = dff[dff[Ethnicity] == ethnicity]
    return dff


@app.callback(
    Output("top-names-chart", "figure"),
    Input("year-filter", "value"),
    Input("gender-filter", "value"),
    Input("trend-filter", "value"),
    Input("ethnicity-filter", "value"),
)
def update_top_names(year, gender, trend, ethnicity):
    dff = _apply_filters(df, year, gender, trend, ethnicity)
    if dff.empty:
        fig = px.bar(title="No data for current filters")
        fig.update_layout(xaxis_title="", yaxis_title="")
        return fig

    top = (
        dff.groupby(FirstName, as_index=False)[Count]
        .sum()
        .sort_values(Count, ascending=False)
        .head(10)
    )

    fig = px.bar(
        top,
        x=FirstName,
        y=Count,
        title="Top 10 names",
        text=Count,
    )
    fig.update_layout(xaxis_title="", yaxis_title="Count")
    fig.update_traces(textposition="outside")
    return fig


@app.callback(
    Output("name-trend-chart", "figure"),
    Input("name-filter", "value"),
)
def update_name_trend(name):
    if not name:
        return px.line(title="Select a name to see its trend over time")

    dff = df[df[FirstName] == name]
    if dff.empty:
        return px.line(title=f"No data for name: {name}")

    trend_df = (
        dff.dropna(subset=[Year])
        .groupby(Year, as_index=False)[Count]
        .sum()
        .sort_values(Year)
    )

    # Use discrete string labels for the x-axis
    trend_df["Year_label"] = trend_df[Year].astype(int).astype(str)

    fig = px.line(
        trend_df,
        x="Year_label",
        y=Count,
        markers=True,
        title=f"Trend over time: {name}",
    )
    fig.update_layout(xaxis_title="Year", yaxis_title="Count")
    return fig


# --- Entry point ------------------------------------------------------------------

if __name__ == '__main__':
    app.run(debug=True)