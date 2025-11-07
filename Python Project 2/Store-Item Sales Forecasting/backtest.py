import numpy as np, pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

df = pd.read_csv("data.csv", parse_dates=["date"]).sort_values(["store","item","date"])

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

feat = df.groupby(["store","item"], group_keys=False).apply(add_features).dropna()
X_cols = [c for c in feat.columns if c not in ["date","sales","store","item"]]

# backtest: evaluate 4 weekly folds at end of series
h = 7
ends = sorted(feat["date"].unique())[-(4*h):][::h]  # 4 checkpoints, one week apart
maes = []

for end in ends:
    train = feat[feat["date"] <= end]
    test  = feat[(feat["date"] > end) & (feat["date"] <= end + pd.Timedelta(days=h))]
    if test.empty: continue
    m = GradientBoostingRegressor(random_state=42).fit(train[X_cols], train["sales"])
    p = m.predict(test[X_cols])
    maes.append(mean_absolute_error(test["sales"], p))

print(f"Walk-forward MAE mean={np.mean(maes):.2f}  std={np.std(maes):.2f}  over {len(maes)} folds")