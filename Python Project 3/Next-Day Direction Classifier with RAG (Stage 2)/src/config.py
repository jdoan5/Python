# src/config.py (Stage 2)

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class Config:
    # paths
    project_dir: Path = Path(__file__).resolve().parents[1]
    runs_dir: Path = project_dir / "runs"
    data_raw: Path = project_dir / "data" / "raw"
    data_processed: Path = project_dir / "data" / "processed"

    # data
    # If you want multi-stock runs, edit this list:
    symbols: List[str] = field(default_factory=lambda: ["SPY", "VOO", "SCHD"])  # e.g. ["AAPL", "SPY", "MSFT"]
    # Optional single-symbol fallback (used only if code chooses CFG.symbol)
    symbol: str = "SPY", "VOO", "SCHD"

    start_date: str = "2015-01-01"
    end_date: str = "2025-12-31"

    # split
    train_ratio: float = 0.70
    val_ratio: float = 0.15

    # learning
    random_state: int = 42

    # IMPORTANT FOR evaluate.py
    target_col: str = "target_up"

    # MUST MATCH the columns produced by src/features.py / your processed CSV.
    # Based on your printed CSV columns:
    feature_cols: List[str] = field(default_factory=lambda: [
        "ret_1",
        "ret_5",
        "ret_10",
        "ret_1_ma_5",
        "ret_1_ma_10",
        "ret_1_ma_20",
        "ret_1_vol_5",
        "ret_1_vol_10",
        "ret_1_vol_20",
        "hl_range",
        "oc_change",
        "close_ma_10",
        "close_ma_20",
        "close_ma_50",
        "close_vs_ma_10",
        "close_vs_ma_20",
        "close_vs_ma_50",
        "ma_10_20_diff",
        "vol_chg_1",
        "log_vol",
    ])

    def ensure_dirs(self) -> None:
        """Create required directories."""
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.data_raw.mkdir(parents=True, exist_ok=True)
        self.data_processed.mkdir(parents=True, exist_ok=True)


CFG = Config()