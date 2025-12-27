from __future__ import annotations

import numpy as np
import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Stage 2 (trend-based) feature set.
    Time-series safe:
      - only uses current/past info for features
      - label uses next-day close direction
    Requires columns: Date, Open, High, Low, Close, Volume
    """
    out = df.copy()

    # Basic returns
    out["ret_1"] = out["Close"].pct_change(1)
    out["ret_5"] = out["Close"].pct_change(5)
    out["ret_10"] = out["Close"].pct_change(10)

    # Rolling return means
    out["ret_1_ma_5"] = out["ret_1"].rolling(5).mean()
    out["ret_1_ma_10"] = out["ret_1"].rolling(10).mean()
    out["ret_1_ma_20"] = out["ret_1"].rolling(20).mean()

    # Rolling vol
    out["ret_1_vol_5"] = out["ret_1"].rolling(5).std()
    out["ret_1_vol_10"] = out["ret_1"].rolling(10).std()
    out["ret_1_vol_20"] = out["ret_1"].rolling(20).std()

    # Price range / candle features
    out["hl_range"] = (out["High"] - out["Low"]) / out["Close"]
    out["oc_change"] = (out["Close"] - out["Open"]) / out["Open"]

    # Trend proxies: moving averages of price
    out["close_ma_10"] = out["Close"].rolling(10).mean()
    out["close_ma_20"] = out["Close"].rolling(20).mean()
    out["close_ma_50"] = out["Close"].rolling(50).mean()

    # Ratios relative to moving averages (scale-free)
    out["close_vs_ma_10"] = out["Close"] / out["close_ma_10"] - 1.0
    out["close_vs_ma_20"] = out["Close"] / out["close_ma_20"] - 1.0
    out["close_vs_ma_50"] = out["Close"] / out["close_ma_50"] - 1.0

    # MA cross (simple trend signal)
    out["ma_10_20_diff"] = out["close_ma_10"] / out["close_ma_20"] - 1.0

    # Volume features
    out["vol_chg_1"] = out["Volume"].pct_change(1).replace([np.inf, -np.inf], np.nan)
    out["log_vol"] = np.log1p(out["Volume"])

    # LABEL: next-day direction
    out["target_up"] = (out["Close"].shift(-1) > out["Close"]).astype(int)

    # Drop rows with NaNs from rolling windows + final label row
    out = out.dropna().reset_index(drop=True)
    return out


def feature_columns() -> list[str]:
    return [
        # returns
        "ret_1", "ret_5", "ret_10",
        # return means
        "ret_1_ma_5", "ret_1_ma_10", "ret_1_ma_20",
        # return vol
        "ret_1_vol_5", "ret_1_vol_10", "ret_1_vol_20",
        # candle/range
        "hl_range", "oc_change",
        # trend (price MAs + ratios)
        "close_vs_ma_10", "close_vs_ma_20", "close_vs_ma_50",
        "ma_10_20_diff",
        # volume
        "vol_chg_1", "log_vol",
    ]