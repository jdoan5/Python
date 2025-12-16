from dash import Dash, html, dcc
import plotly.express as px
import pandas as pd

df = pd.DataFrame({
    "date": pd.date_range("2025-01-01", periods=10),
    "tickets": [12, 9, 15, 14, 20, 18, 16, 22, 19, 25],
})

fig = px.line(df, x="date", y="tickets", title="Ticket Volume Trend")

app = Dash(__name__)
app.layout = html.Div(
    style={"maxWidth": "1100px", "margin": "24px auto", "fontFamily": "system-ui"},
    children=[
        html.H1("Support Ticket Analytics Dashboard"),
        dcc.Graph(figure=fig),
    ],
)

if __name__ == "__main__":
    app.run_server(debug=True)