from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

from .config import MartConfig, connect, ensure_dirs


def _csv_columns(con, csv_path: Path) -> List[str]:
    """Return column names detected by DuckDB for the CSV."""
    rows: List[Tuple[str, str, str, str, str]] = con.execute(
        "DESCRIBE SELECT * FROM read_csv_auto(?);",
        [csv_path.as_posix()],
    ).fetchall()
    return [r[0] for r in rows]


def _drop_existing(con) -> None:
    con.execute("DROP TABLE IF EXISTS fact_sales;")
    con.execute("DROP TABLE IF EXISTS dim_customer;")
    con.execute("DROP TABLE IF EXISTS dim_product;")
    con.execute("DROP TABLE IF EXISTS dim_date;")


def build(cfg: MartConfig, reset: bool = True) -> Dict[str, int]:
    """Build/rebuild the DuckDB mini data mart from CSV inputs."""
    ensure_dirs(cfg)

    if not cfg.FACT_SALES_CSV.exists():
        raise FileNotFoundError(
            f"Missing input CSV: {cfg.FACT_SALES_CSV}\n"
            f"Expected location: data/input/fact_sales.csv"
        )

    con = connect(cfg)
    try:
        if reset:
            _drop_existing(con)

        cols = set(_csv_columns(con, cfg.FACT_SALES_CSV))

        # Create a normalized staging view.
        con.execute("DROP VIEW IF EXISTS stg_sales;")

        has_sale_id = "sale_id" in cols

        has_customer_id = "customer_id" in cols
        has_customer_name = "customer_name" in cols
        has_region = "region" in cols

        has_product_id = "product_id" in cols
        has_product_name = "product_name" in cols
        has_category = "category" in cols

        if "order_date" not in cols:
            raise ValueError("fact_sales.csv must include 'order_date' column (YYYY-MM-DD).")

        for req in ("quantity", "unit_price"):
            if req not in cols:
                raise ValueError(f"fact_sales.csv must include '{req}' column.")

        stg_select = f"""
            CREATE VIEW stg_sales AS
            SELECT
                {"TRY_CAST(sale_id AS BIGINT) AS sale_id," if has_sale_id else ""}
                {"TRY_CAST(customer_id AS BIGINT) AS customer_id," if has_customer_id else ""}
                {"customer_name AS customer_name," if has_customer_name else "NULL::VARCHAR AS customer_name,"}
                {"region AS region," if has_region else "NULL::VARCHAR AS region,"}
                {"TRY_CAST(product_id AS BIGINT) AS product_id," if has_product_id else ""}
                {"product_name AS product_name," if has_product_name else "NULL::VARCHAR AS product_name,"}
                {"category AS category," if has_category else "NULL::VARCHAR AS category,"}
                TRY_CAST(order_date AS DATE) AS order_date,
                TRY_CAST(quantity AS INTEGER) AS quantity,
                TRY_CAST(unit_price AS DOUBLE) AS unit_price
            FROM read_csv_auto('{cfg.FACT_SALES_CSV.as_posix()}');
        """
        con.execute(stg_select)

        # -------------------------
        # DIMENSIONS
        # -------------------------

        if has_customer_id:
            con.execute("""
                CREATE TABLE dim_customer AS
                SELECT
                    customer_id,
                    COALESCE(customer_name, 'Customer ' || customer_id::VARCHAR) AS customer_name,
                    COALESCE(region, 'Unknown') AS region
                FROM (
                    SELECT DISTINCT customer_id, customer_name, region
                    FROM stg_sales
                    WHERE customer_id IS NOT NULL
                )
                ORDER BY customer_id;
            """)
        else:
            con.execute("""
                CREATE TABLE dim_customer AS
                SELECT
                    DENSE_RANK() OVER (ORDER BY customer_name, region) AS customer_id,
                    customer_name,
                    COALESCE(region, 'Unknown') AS region
                FROM (
                    SELECT DISTINCT
                        COALESCE(customer_name, 'Unknown Customer') AS customer_name,
                        COALESCE(region, 'Unknown') AS region
                    FROM stg_sales
                )
                ORDER BY customer_id;
            """)

        if has_product_id:
            con.execute("""
                CREATE TABLE dim_product AS
                SELECT
                    product_id,
                    COALESCE(product_name, 'Product ' || product_id::VARCHAR) AS product_name,
                    COALESCE(category, 'Unknown') AS category
                FROM (
                    SELECT DISTINCT product_id, product_name, category
                    FROM stg_sales
                    WHERE product_id IS NOT NULL
                )
                ORDER BY product_id;
            """)
        else:
            con.execute("""
                CREATE TABLE dim_product AS
                SELECT
                    DENSE_RANK() OVER (ORDER BY product_name, category) AS product_id,
                    product_name,
                    COALESCE(category, 'Unknown') AS category
                FROM (
                    SELECT DISTINCT
                        COALESCE(product_name, 'Unknown Product') AS product_name,
                        COALESCE(category, 'Unknown') AS category
                    FROM stg_sales
                )
                ORDER BY product_id;
            """)

        con.execute("""
            CREATE TABLE dim_date AS
            SELECT
                order_date AS date,
                EXTRACT(YEAR FROM order_date)::INTEGER AS year,
                EXTRACT(MONTH FROM order_date)::INTEGER AS month,
                STRFTIME(order_date, '%A') AS day_of_week
            FROM (
                SELECT DISTINCT order_date
                FROM stg_sales
                WHERE order_date IS NOT NULL
            )
            ORDER BY date;
        """)

        # -------------------------
        # FACT TABLE
        # -------------------------

        if not has_customer_id:
            customer_join = """
                JOIN dim_customer c
                  ON c.customer_name = COALESCE(s.customer_name, 'Unknown Customer')
                 AND c.region = COALESCE(s.region, 'Unknown')
            """
            customer_key = "c.customer_id"
        else:
            customer_join = ""
            customer_key = "s.customer_id"

        if not has_product_id:
            product_join = """
                JOIN dim_product p
                  ON p.product_name = COALESCE(s.product_name, 'Unknown Product')
                 AND p.category = COALESCE(s.category, 'Unknown')
            """
            product_key = "p.product_id"
        else:
            product_join = ""
            product_key = "s.product_id"

        sale_id_expr = "s.sale_id" if has_sale_id else "ROW_NUMBER() OVER ()::BIGINT"

        con.execute("DROP TABLE IF EXISTS fact_sales;")
        con.execute(f"""
            CREATE TABLE fact_sales AS
            SELECT
                {sale_id_expr} AS sale_id,
                {customer_key} AS customer_id,
                {product_key} AS product_id,
                s.order_date AS order_date,
                s.quantity AS quantity,
                s.unit_price AS unit_price
            FROM stg_sales s
            {customer_join}
            {product_join}
            WHERE s.order_date IS NOT NULL
              AND s.quantity IS NOT NULL
              AND s.unit_price IS NOT NULL;
        """)

        counts = {
            "dim_customer": con.execute("SELECT COUNT(*) FROM dim_customer;").fetchone()[0],
            "dim_product": con.execute("SELECT COUNT(*) FROM dim_product;").fetchone()[0],
            "dim_date": con.execute("SELECT COUNT(*) FROM dim_date;").fetchone()[0],
            "fact_sales": con.execute("SELECT COUNT(*) FROM fact_sales;").fetchone()[0],
        }
        return counts

    finally:
        con.close()
