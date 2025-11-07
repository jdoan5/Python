import numpy as np, pandas as pd
from joblib import load

model, X_cols = load("model.joblib")
df = pd.read_csv("data.csv", parse_dates=["date"]).sort_values(["store","item","date"])
last_date = df["date"].max()
h = 14

def add_features(g):
    g = g.sort_values("date").copy()
    for L in [1,7,14,28]:
        g[f"lag_{L}"] = g["sales"].shift(L)
    for W in [7,14,28]:
        g[f"rollmean_{W}"] = g["sales"].shift(1).rolling(W).mean()
        g[f"rollstd_{W}"]  = g["sales"].shift(1).rolling(W).std()
    g["dow"] = g["date"].dt.dayofweek
    g["dom"] = g["date"].dt.day
    return g

history = df.copy()
fc_rows = []
for step in range(1, h+1):
    future_day = last_date + pd.Timedelta(days=step)
    tmp = history.copy()
    feat = tmp.groupby(["store","item"], group_keys=False).apply(add_features)
    feat = feat[feat["date"] == history["date"].max()]  # last available features per series
    X = feat[X_cols]
    yhat = model.predict(X)
    new_day = feat[["store","item"]].copy()
    new_day["date"] = future_day
    new_day["sales"] = yhat
    history = pd.concat([history, new_day], ignore_index=True)
    fc_rows.append(new_day)

forecast = pd.concat(fc_rows).reset_index(drop=True)
forecast.to_csv("forecast.csv", index=False)
print("Wrote forecast.csv", forecast.shape)