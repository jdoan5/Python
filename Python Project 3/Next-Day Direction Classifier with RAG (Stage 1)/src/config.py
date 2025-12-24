from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List


@dataclass(frozen=True)
class Config:
    # Data provider
    provider: str = "yahoo"  # was "stooq"

    # Yahoo Finance tickers (examples: "SPY", "AAPL", "MSFT")
    symbol: str = "AAPL"
    symbols: Optional[List[str]] = None  # optional: multi-ticker later

    # Optional date range (None means "max available")
    start_date: Optional[str] = "2015-01-01"
    end_date: Optional[str] = None

    root: Path = Path(__file__).resolve().parents[1]
    data_raw: Path = root / "data" / "raw"
    data_processed: Path = root / "data" / "processed"
    runs_dir: Path = root / "runs"

    # Time split ratios (no shuffling)
    train_ratio: float = 0.70
    val_ratio: float = 0.15

    random_state: int = 42


CFG = Config()