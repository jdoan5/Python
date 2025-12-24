from __future__ import annotations

import numpy as np
import pandas as pd


def feature_columns() -> list[str]:
    return [
        "ret_1", "ret_5",
        "ret_1_ma_5", "ret_1_ma_10",
        "ret_1_vol_5", "ret_1_vol_10",
        "hl_range", "oc_change",
        "vol_chg_1", "log_vol",
    ]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Time-series safe features (only current/past information).
    Label is next-day direction (Close[t+1] > Close[t]).

    Returns a DataFrame containing original columns + features + target_up,
    with rows containing NaNs in (features + target_up) removed.
    """
    out = df.copy()

    # Validate required columns
    required = {"Open", "High", "Low", "Close"}
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise RuntimeError(f"Missing required columns for feature engineering: {missing}")

    # Volume is usually present, but some tickers may not have it reliably
    if "Volume" not in out.columns:
        out["Volume"] = np.nan

    # returns
    out["ret_1"] = out["Close"].pct_change(1)
    out["ret_5"] = out["Close"].pct_change(5)

    # rolling means of returns
    out["ret_1_ma_5"] = out["ret_1"].rolling(5).mean()
    out["ret_1_ma_10"] = out["ret_1"].rolling(10).mean()

    # rolling volatility of returns
    out["ret_1_vol_5"] = out["ret_1"].rolling(5).std()
    out["ret_1_vol_10"] = out["ret_1"].rolling(10).std()

    # price range features
    out["hl_range"] = (out["High"] - out["Low"]) / out["Close"]
    out["oc_change"] = (out["Close"] - out["Open"]) / out["Open"]

    # volume features (log + % change)
    out["vol_chg_1"] = out["Volume"].pct_change(1)
    out["log_vol"] = np.log1p(out["Volume"])

    # Clean inf/-inf that can appear in pct_change/divisions
    out = out.replace([np.inf, -np.inf], np.nan)

    # LABEL: next-day direction
    next_close = out["Close"].shift(-1)
    out["target_up"] = np.where(next_close.isna(), np.nan, (next_close > out["Close"]).astype(int))

    feats = feature_columns()

    # Drop rows with NaNs ONLY in features + target (keeps unrelated cols like Adj Close from causing drops)
    out = out.dropna(subset=feats + ["target_up"]).reset_index(drop=True)

    # Ensure label is int after dropping NaNs
    out["target_up"] = out["target_up"].astype(int)

    return out