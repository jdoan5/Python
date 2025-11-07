import numpy as np, pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from joblib import dump

df = pd.read_csv("data.csv", parse_dates=["date"]).sort_values(["store","item","date"])

# feature builder per group
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

feat = df.groupby(["store","item"], group_keys=False).apply(add_features)
feat = feat.dropna().reset_index(drop=True)

# split by time (last 28 days = test)
horizon = 28
cut = feat["date"].max() - pd.Timedelta(days=horizon)
train = feat[feat["date"] <= cut]
test  = feat[feat["date"] >  cut]

X_cols = [c for c in feat.columns if c not in ["date","sales","store","item"]]
Xtr, ytr = train[X_cols], train["sales"]
Xte, yte = test[X_cols],  test["sales"]

model = GradientBoostingRegressor(random_state=42)
model.fit(Xtr, ytr)
pred = model.predict(Xte)
mae  = mean_absolute_error(yte, pred)
print(f"GBR MAE={mae:.2f} (must beat seasonal-naive)")

dump((model, X_cols), "model.joblib")
print("Saved model.joblib")