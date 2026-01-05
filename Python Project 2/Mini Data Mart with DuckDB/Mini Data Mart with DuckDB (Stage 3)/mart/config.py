from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import duckdb


@dataclass(frozen=True)
class MartConfig:
    """Central configuration for the Mini Data Mart (Stage 3).

    Repo layout (defaults):

    <project-root>/
      mart/
      data/
        input/
          fact_sales.csv
        export/
          CSV-export/
          JSON-export/
        analytics/
        quality/
      mini_data_mart.duckdb
    """

    BASE_DIR: Path
    DB_PATH: Path

    DATA_DIR: Path
    INPUT_DIR: Path
    EXPORT_DIR: Path
    OUTPUT_CSV_DIR: Path
    OUTPUT_JSON_DIR: Path
    ANALYTICS_DIR: Path
    QUALITY_DIR: Path

    FACT_SALES_CSV: Path


def load_config() -> MartConfig:
    """Build a MartConfig with optional environment overrides.

    Supported env vars (optional):
      MART_DB_PATH       e.g. /abs/path/mini_data_mart.duckdb
      MART_FACT_SALES    e.g. /abs/path/fact_sales.csv
    """
    base_dir = Path(__file__).resolve().parents[1]  # <project-root>
    data_dir = base_dir / "data"
    input_dir = data_dir / "input"

    export_dir = data_dir / "export"
    output_csv = export_dir / "CSV-export"
    output_json = export_dir / "JSON-export"

    analytics_dir = data_dir / "analytics"
    quality_dir = data_dir / "quality"

    db_path = Path(os.getenv("MART_DB_PATH", (base_dir / "mini_data_mart.duckdb").as_posix()))
    fact_csv = Path(os.getenv("MART_FACT_SALES", (input_dir / "fact_sales.csv").as_posix()))

    return MartConfig(
        BASE_DIR=base_dir,
        DB_PATH=db_path,
        DATA_DIR=data_dir,
        INPUT_DIR=input_dir,
        EXPORT_DIR=export_dir,
        OUTPUT_CSV_DIR=output_csv,
        OUTPUT_JSON_DIR=output_json,
        ANALYTICS_DIR=analytics_dir,
        QUALITY_DIR=quality_dir,
        FACT_SALES_CSV=fact_csv,
    )


def ensure_dirs(cfg: MartConfig) -> None:
    cfg.DATA_DIR.mkdir(parents=True, exist_ok=True)
    cfg.INPUT_DIR.mkdir(parents=True, exist_ok=True)

    cfg.EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    cfg.OUTPUT_CSV_DIR.mkdir(parents=True, exist_ok=True)
    cfg.OUTPUT_JSON_DIR.mkdir(parents=True, exist_ok=True)

    cfg.ANALYTICS_DIR.mkdir(parents=True, exist_ok=True)
    cfg.QUALITY_DIR.mkdir(parents=True, exist_ok=True)


def connect(cfg: MartConfig) -> duckdb.DuckDBPyConnection:
    """Connect to DuckDB file (created if missing)."""
    return duckdb.connect(cfg.DB_PATH.as_posix())
