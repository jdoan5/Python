# mart/config.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import duckdb


@dataclass(frozen=True)
class MartConfig:
    """
    Central configuration for the Mini Data Mart (Stage 4).

    Expected repo layout:

    <project-root>/
      mart/
        __main__.py
        build.py
        explore.py
        export.py
        quality.py
        config.py
      data/
        input/
          fact_sales.csv          # required
          dim_customer.csv        # optional (Stage 4)
          dim_product.csv         # optional (Stage 4)
          dim_date.csv            # optional (Stage 4) OR computed from fact
        export/
          CSV-export/
          JSON-export/
        analytics/                # optional outputs (Stage 3/4)
        quality/                  # optional outputs (Stage 3/4)
      mini_data_mart.duckdb
      ui/
        streamlit_app.py          # Stage 4 UI (optional)

    Env overrides (optional):
      MART_DB_PATH         e.g. /abs/path/mini_data_mart.duckdb
      MART_FACT_SALES      e.g. /abs/path/fact_sales.csv
      MART_DIM_CUSTOMER    e.g. /abs/path/dim_customer.csv
      MART_DIM_PRODUCT     e.g. /abs/path/dim_product.csv
      MART_DIM_DATE        e.g. /abs/path/dim_date.csv
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
    DIM_CUSTOMER_CSV: Path
    DIM_PRODUCT_CSV: Path
    DIM_DATE_CSV: Path


def load_config() -> MartConfig:
    base_dir = Path(__file__).resolve().parents[1]  # <project-root>

    data_dir = base_dir / "data"
    input_dir = data_dir / "input"

    export_dir = data_dir / "export"
    output_csv = export_dir / "CSV-export"
    output_json = export_dir / "JSON-export"

    analytics_dir = data_dir / "analytics"
    quality_dir = data_dir / "quality"

    db_path = Path(os.getenv("MART_DB_PATH", str(base_dir / "mini_data_mart.duckdb")))

    fact_csv = Path(os.getenv("MART_FACT_SALES", str(input_dir / "fact_sales.csv")))
    dim_customer_csv = Path(os.getenv("MART_DIM_CUSTOMER", str(input_dir / "dim_customer.csv")))
    dim_product_csv = Path(os.getenv("MART_DIM_PRODUCT", str(input_dir / "dim_product.csv")))
    dim_date_csv = Path(os.getenv("MART_DIM_DATE", str(input_dir / "dim_date.csv")))

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
        DIM_CUSTOMER_CSV=dim_customer_csv,
        DIM_PRODUCT_CSV=dim_product_csv,
        DIM_DATE_CSV=dim_date_csv,
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