from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

from .config import load_config
from .build import build as build_mart
from .explore import explore as explore_mart
from .export import export_all as export_all_mart, export_csv as export_csv_mart, export_json as export_json_mart
from .quality import run_checks


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m mart",
        description="Mini Data Mart CLI (DuckDB + CSV-driven inputs)",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    p_build = sub.add_parser("build", help="Build/rebuild DuckDB tables from CSV inputs")
    p_build.add_argument("--no-reset", action="store_true", help="Do not drop/recreate tables (incremental experiments)")

    sub.add_parser("explore", help="Run example analytics queries against the mart")

    p_export = sub.add_parser("export", help="Export mart tables to CSV/JSON outputs")
    p_export.add_argument("--format", choices=["csv", "json", "all"], default="all")

    sub.add_parser("quality", help="Run data quality checks (row counts, nulls, RI)")

    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    cfg = load_config()

    if args.cmd == "build":
        counts = build_mart(cfg, reset=(not args.no_reset))
        print("\nBuild complete. Row counts:")
        for k, v in counts.items():
            print(f"  {k}: {v}")
        return 0

    if args.cmd == "explore":
        explore_mart(cfg)
        return 0

    if args.cmd == "export":
        if args.format == "csv":
            export_csv_mart(cfg)
        elif args.format == "json":
            export_json_mart(cfg)
        else:
            export_all_mart(cfg)
        return 0

    if args.cmd == "quality":
        ok = run_checks(cfg)
        return 0 if ok else 2

    raise RuntimeError("Unhandled command")


def _csv_columns(con, csv_path: Path) -> List[str]:
    """Return column names detected by DuckDB for the CSV (no pandas needed)."""
    rows: List[Tuple[str, str, str, str, str]] = con.execute(
        "DESCRIBE SELECT * FROM read_csv_auto(?, header=True);",
        [csv_path.as_posix()],
    ).fetchall()
    return [r[0] for r in rows]


def _drop_existing(con) -> None:
    con.execute("DROP TABLE IF EXISTS fact_sales;")
    con.execute("DROP TABLE IF EXISTS dim_customer;")
    con.execute("DROP TABLE IF EXISTS dim_product;")
    con.execute("DROP TABLE IF EXISTS dim_date;")
    con.execute("DROP VIEW IF EXISTS stg_sales;")
    con.execute("DROP VIEW IF EXISTS stg_dim_customer;")
    con.execute("DROP VIEW IF EXISTS stg_dim_product;")
    con.execute("DROP VIEW IF EXISTS stg_dim_date;")


def _create_tables(con, cfg) -> None:
    """Create tables + views in DuckDB based on CSV inputs."""

    def _require_any(cols: List[str], names: tuple[str, ...], csv_name: str) -> None:
        if not any(n in cols for n in names):
            raise ValueError(
                f"Must provide at least one of {names} columns in '{csv_name}' source CSV, but only have: {cols}"
            )

    # -------------------------
    # FACT: sales
    # -------------------------
    sales_cols = _csv_columns(con, cfg.FACT_SALES_CSV)
    has_sale_id = "sale_id" in sales_cols
    has_customer_id = "customer_id" in sales_cols
    has_customer_name = "customer_name" in sales_cols
    has_region = "region" in sales_cols
    has_product_id = "product_id" in sales_cols
    has_product_name = "product_name" in sales_cols
    has_category = "category" in sales_cols

    _require_any(sales_cols, ("order_date",), "fact_sales.csv")
    _require_any(sales_cols, ("quantity",), "fact_sales.csv")
    _require_any(sales_cols, ("unit_price",), "fact_sales.csv")

    con.execute("DROP VIEW IF EXISTS stg_sales;")
    # Create a normalized staging view.
    # Keep both ID and name fields if present to support flexible joins.
    stg_sql = f"""
        CREATE VIEW stg_sales AS
        SELECT
            {"TRY_CAST(sale_id AS BIGINT) AS sale_id," if has_sale_id else ""}
            {"TRY_CAST(customer_id AS BIGINT) AS customer_id," if has_customer_id else "NULL::BIGINT AS customer_id,"}
            {"customer_name AS customer_name," if has_customer_name else "NULL::VARCHAR AS customer_name,"}
            {"region AS region," if has_region else "NULL::VARCHAR AS region,"}
            {"TRY_CAST(product_id AS BIGINT) AS product_id," if has_product_id else "NULL::BIGINT AS product_id,"}
            {"product_name AS product_name," if has_product_name else "NULL::VARCHAR AS product_name,"}
            {"category AS category," if has_category else "NULL::VARCHAR AS category,"}
            TRY_CAST(order_date AS DATE) AS order_date,
            TRY_CAST(quantity AS INTEGER) AS quantity,
            TRY_CAST(unit_price AS DOUBLE) AS unit_price
        FROM read_csv_auto('{cfg.FACT_SALES_CSV.as_posix()}', header=True);
    """
    con.execute(stg_sql)

    # -------------------------
    # DIMENSION: customer
    # -------------------------
    dim_cols = _csv_columns(con, cfg.DIM_CUSTOMER_CSV)
    _require_any(dim_cols, ("customer_id", "customer_name"), "dim_customer.csv")
    con.execute("DROP VIEW IF EXISTS stg_dim_customer;")
    con.execute(f"""
        CREATE VIEW stg_dim_customer AS
        SELECT *
        FROM read_csv_auto('{cfg.DIM_CUSTOMER_CSV.as_posix()}', header=True);
    """)

    # If customer_id missing, generate stable surrogate keys
    if "customer_id" not in dim_cols:
        con.execute("""
            CREATE TABLE dim_customer AS
            SELECT
                ROW_NUMBER() OVER (ORDER BY customer_name) AS customer_id,
                customer_name
            FROM (SELECT DISTINCT customer_name FROM stg_sales)
            WHERE customer_name IS NOT NULL;
        """)
    else:
        con.execute("""
            CREATE TABLE dim_customer AS
            SELECT DISTINCT customer_id, customer_name FROM stg_dim_customer;
        """)

    # -------------------------
    # DIMENSION: product
    # -------------------------
    dim_cols = _csv_columns(con, cfg.DIM_PRODUCT_CSV)
    _require_any(dim_cols, ("product_id", "product_name"), "dim_product.csv")
    con.execute("DROP VIEW IF EXISTS stg_dim_product;")
    con.execute(f"""
        CREATE VIEW stg_dim_product AS
        SELECT *
        FROM read_csv_auto('{cfg.DIM_PRODUCT_CSV.as_posix()}', header=True);
    """)

    if "product_id" in dim_cols:
        con.execute("""
            CREATE TABLE dim_product AS
            SELECT DISTINCT product_id, product_name, category FROM stg_dim_product;
        """)
    else:
        # TODO: can generate IDs from product_name if needed
        raise ValueError("Missing product_id in dim_product.csv, need lookup key to join with fact_sales")

    # -------------------------
    # DIMENSION: date
    # -------------------------
    if cfg.DIM_DATE_CSV.exists():
        dim_cols = set(_csv_columns(con, cfg.DIM_DATE_CSV))
        _require_any(dim_cols, ("date",), "dim_date.csv")
        con.execute("DROP VIEW IF EXISTS stg_dim_date;")
        con.execute(f"""
            CREATE VIEW stg_dim_date AS
            SELECT *
            FROM read_csv_auto('{cfg.DIM_DATE_CSV.as_posix()}', header=True);
        """)

        # Compute missing attributes if not provided
        if "year" not in dim_cols:
            con.execute("ALTER TABLE stg_dim_date ADD COLUMN year INTEGER;")
            con.execute("UPDATE stg_dim_date SET year = date_part('year', date);")
        if "month" not in dim_cols:
            con.execute("ALTER TABLE stg_dim_date ADD COLUMN month VARCHAR;")
            con.execute("UPDATE stg_dim_date SET month = date_part('month', date);")

        con.execute("""
            CREATE TABLE dim_date AS
            SELECT DISTINCT date, year, month FROM stg_dim_date;
        """)

    # -------------------------
    # FINAL: fact_sales
    # -------------------------
    con.execute("""
        CREATE TABLE fact_sales AS
        SELECT
            s.order_date,
            s.quantity,
            s.unit_price,
            COALESCE(c.customer_id, -1) AS customer_id,
            COALESCE(p.product_id, -1) AS product_id,
            COALESCE(d.date, DATE '1900-01-01') AS date
        FROM stg_sales s
        LEFT JOIN dim_customer c ON s.customer_name = c.customer_name
        LEFT JOIN dim_product p ON s.product_name = p.product_name
        LEFT JOIN dim_date d ON s.order_date = d.date;
    """)


if __name__ == "__main__":
    raise SystemExit(main())
