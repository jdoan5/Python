import numpy as np, pandas as pd
rng = np.random.default_rng(42)

dates = pd.date_range("2022-01-01", "2022-12-31", freq="D")
stores = [f"S{i}" for i in range(1, 4)]
items  = [f"I{i}" for i in range(1, 6)]

rows = []
for s in stores:
    for it in items:
        base = rng.normal(50, 5, len(dates))
        weekly = 10*np.sin(2*np.pi*dates.dayofweek/7)           # weekly seasonality
        trend  = np.linspace(0, 8, len(dates))                  # small upward trend
        noise  = rng.normal(0, 3, len(dates))
        y = np.clip(base + weekly + trend + noise, 0, None).round(1)
        rows += list(zip([s]*len(dates), [it]*len(dates), dates, y))

df = pd.DataFrame(rows, columns=["store","item","date","sales"])
df.to_csv("data.csv", index=False)
print("Wrote data.csv", df.shape)