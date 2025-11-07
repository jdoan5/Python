import pandas as pd, numpy as np

df = pd.read_csv("data.csv", parse_dates=["date"])
df = df.sort_values(["store","item","date"])

# define test horizon = last 28 days
horizon = 28
last_day = df["date"].max()
cutoff = last_day - pd.Timedelta(days=horizon)

# seasonal naive: predict day t by using value from t-7
def smape(y_true, y_pred):
    denom = (np.abs(y_true) + np.abs(y_pred)).replace(0, 1e-8)
    return (100 * np.mean(2 * np.abs(y_pred - y_true) / denom))

test = df[df["date"] > cutoff].copy()
hist = df.copy()
hist["date_lag7"] = hist["date"] + pd.Timedelta(days=7)

pred = test.merge(
    hist[["store","item","date_lag7","sales"]].rename(columns={"date_lag7":"date","sales":"pred"}),
    on=["store","item","date"], how="left"
)
pred["pred"] = pred["pred"].ffill()  # fallback if first week missing (not in our synth data)
mae = (pred["sales"] - pred["pred"]).abs().mean()
print(f"Seasonal-naive (lag 7) MAE={mae:.2f}, sMAPE={smape(pred['sales'], pred['pred']):.2f}% (horizon={horizon}d)")