from __future__ import annotations

from datetime import datetime
import pandas as pd
import yfinance as yf

from src.config import CFG


REQUIRED_OHLC = ["Open", "High", "Low", "Close"]


def _pick_ohlc_level(cols: pd.MultiIndex) -> int:
    """
    yfinance can return MultiIndex in different orientations, e.g.:
      - ('Open','High',...) x ('SPY')   -> OHLC is level 0
      - ('SPY') x ('Open','High',...)   -> OHLC is level 1
    Choose the level that contains OHLC names.
    """
    levels = [list(cols.get_level_values(i)) for i in range(cols.nlevels)]

    def norm(x) -> str:
        return str(x).strip().lower().replace(" ", "")

    required_norm = {norm(x) for x in REQUIRED_OHLC}

    best_level = 0
    best_hits = -1
    for i, vals in enumerate(levels):
        vals_norm = {norm(v) for v in vals}
        hits = len(required_norm & vals_norm)
        if hits > best_hits:
            best_hits = hits
            best_level = i

    return best_level


def _normalize_yahoo_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize yfinance output to columns:
      Date, Open, High, Low, Close, Volume (if present), Adj Close (if present)
    """
    if df is None or df.empty:
        raise RuntimeError(
            "Yahoo Finance returned no rows (empty DataFrame). "
            "Check symbol/date range/internet."
        )

    # Handle MultiIndex columns robustly
    if isinstance(df.columns, pd.MultiIndex):
        lvl = _pick_ohlc_level(df.columns)
        df.columns = df.columns.get_level_values(lvl)

    # Reset index so we get Date column
    df = df.reset_index()

    if "Date" not in df.columns:
        if "Datetime" in df.columns:
            df = df.rename(columns={"Datetime": "Date"})
        else:
            raise RuntimeError(f"Yahoo data missing Date column. Columns: {list(df.columns)}")

    # Case-insensitive rename mapping
    rename_map = {}
    for col in df.columns:
        key = str(col).strip().lower().replace(" ", "")
        if key == "open":
            rename_map[col] = "Open"
        elif key == "high":
            rename_map[col] = "High"
        elif key == "low":
            rename_map[col] = "Low"
        elif key == "close":
            rename_map[col] = "Close"
        elif key in ("adjclose", "adj_close"):
            rename_map[col] = "Adj Close"
        elif key == "volume":
            rename_map[col] = "Volume"

    df = df.rename(columns=rename_map)

    # Validate required OHLC
    missing = [c for c in REQUIRED_OHLC if c not in df.columns]
    if missing:
        raise RuntimeError(
            "Yahoo Finance response did not include required OHLC columns.\n"
            f"Missing: {missing}\n"
            f"Columns received: {list(df.columns)}\n"
            "Tip: ensure CFG.symbol is a valid Yahoo ticker like SPY, AAPL, MSFT."
        )

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # Drop rows missing OHLC
    df = df.dropna(subset=REQUIRED_OHLC)

    return df


def fetch_from_yahoo(symbol: str, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """
    No API key. Pulls daily OHLCV from Yahoo Finance via yfinance.
    """
    df = yf.download(
        tickers=symbol,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    return _normalize_yahoo_columns(df)


def save_raw(df: pd.DataFrame, symbol: str) -> str:
    """
    Save raw Yahoo data to data/raw/<SYMBOL>_<timestamp>.csv
    """
    CFG.data_raw.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = CFG.data_raw / f"{symbol}_{stamp}.csv"
    df.to_csv(out, index=False)
    return str(out)


if __name__ == "__main__":
    df = fetch_from_yahoo(
        CFG.symbol,
        start=getattr(CFG, "start_date", None),
        end=getattr(CFG, "end_date", None),
    )
    path = save_raw(df, CFG.symbol)
    print(f"Saved raw data to {path}")
    print("Columns:", list(df.columns))
    print(df.tail(5))