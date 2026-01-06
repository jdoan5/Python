# mart/build.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

from .config import MartConfig, connect, ensure_dirs


def _csv_columns(con, csv_path: Path) -> List[str]:
    """Return column names detected by DuckDB for the CSV (no pandas needed)."""
    rows: List[Tuple[str, str, str, str, str]] = con.execute(
        "DESCRIBE SELECT * FROM read_csv_auto(?);",
        [csv_path.as_posix()],
    ).fetchall()
    return [r[0] for r in rows]


def _drop_existing(con) -> None:
    con.execute("DROP VIEW IF EXISTS stg_sales;")
    con.execute("DROP VIEW IF EXISTS stg_dim_customer;")
    con.execute("DROP VIEW IF EXISTS stg_dim_product;")
    con.execute("DROP VIEW IF EXISTS stg_dim_date;")

    con.execute("DROP TABLE IF EXISTS fact_sales;")
    con.execute("DROP TABLE IF EXISTS dim_customer;")
    con.execute("DROP TABLE IF EXISTS dim_product;")
    con.execute("DROP TABLE IF EXISTS dim_date;")


def _require_any(cols: set[str], options: tuple[str, ...], context: str) -> None:
    if not any(o in cols for o in options):
        raise ValueError(f"{context}: expected at least one of {options}. Found: {sorted(cols)}")


def _require_all(cols: set[str], required: tuple[str, ...], context: str) -> None:
    missing = [c for c in required if c not in cols]
    if missing:
        raise ValueError(f"{context}: missing required column(s): {missing}. Found: {sorted(cols)}")


def _build_dims_from_fact(con) -> None:
    """Fallback dim build when dim CSVs are not provided."""
    # dim_customer from stg_sales (name-based)
    con.execute("""
        CREATE TABLE dim_customer AS
        SELECT
            DENSE_RANK() OVER (ORDER BY customer_name, region) AS customer_id,
            customer_name,
            region
        FROM (
            SELECT DISTINCT
                COALESCE(customer_name, 'Unknown Customer') AS customer_name,
                COALESCE(region, 'Unknown') AS region
            FROM stg_sales
        )
        ORDER BY customer_id;
    """)

    # dim_product from stg_sales (name-based)
    con.execute("""
        CREATE TABLE dim_product AS
        SELECT
            DENSE_RANK() OVER (ORDER BY product_name, category) AS product_id,
            product_name,
            category
        FROM (
            SELECT DISTINCT
                COALESCE(product_name, 'Unknown Product') AS product_name,
                COALESCE(category, 'Unknown') AS category
            FROM stg_sales
        )
        ORDER BY product_id;
    """)

    # dim_date computed from stg_sales.order_date
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


def build(cfg: MartConfig, reset: bool = True, strict: bool = True) -> Dict[str, int]:
    """
    Stage 4 build/rebuild:

    Inputs:
      - data/input/fact_sales.csv (required)
      - data/input/dim_customer.csv (optional)
      - data/input/dim_product.csv (optional)
      - data/input/dim_date.csv (optional; else computed)

    Behavior:
      - Creates a staging view from fact_sales.csv with robust casting.
      - Uses dim CSVs if present; otherwise derives dims from the fact.
      - Builds fact_sales with resolved keys.
      - If strict=True: fail-fast when keys cannot be resolved.
    """
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

        # -------------------------
        # Stage fact: stg_sales
        # -------------------------
        fact_cols = set(_csv_columns(con, cfg.FACT_SALES_CSV))

        # Required base fields
        _require_all(fact_cols, ("order_date", "quantity", "unit_price"), "fact_sales.csv")

        # We need either IDs or Names to resolve customer/product keys
        _require_any(fact_cols, ("customer_id", "customer_name"), "fact_sales.csv (customer)")
        _require_any(fact_cols, ("product_id", "product_name"), "fact_sales.csv (product)")

        has_sale_id = "sale_id" in fact_cols

        has_customer_id = "customer_id" in fact_cols
        has_customer_name = "customer_name" in fact_cols
        has_region = "region" in fact_cols

        has_product_id = "product_id" in fact_cols
        has_product_name = "product_name" in fact_cols
        has_category = "category" in fact_cols

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
            FROM read_csv_auto('{cfg.FACT_SALES_CSV.as_posix()}');
        """
        con.execute(stg_sql)

        # -------------------------
        # DIMENSIONS (CSV if present, else derived)
        # -------------------------

        # dim_customer
        if cfg.DIM_CUSTOMER_CSV.exists():
            dim_cols = set(_csv_columns(con, cfg.DIM_CUSTOMER_CSV))
            _require_any(dim_cols, ("customer_id", "customer_name"), "dim_customer.csv")
            con.execute("DROP VIEW IF EXISTS stg_dim_customer;")
            con.execute(f"""
                CREATE VIEW stg_dim_customer AS
                SELECT *
                FROM read_csv_auto('{cfg.DIM_CUSTOMER_CSV.as_posix()}');
            """)

            # If customer_id missing, generate stable surrogate keys
            if "customer_id" in dim_cols:
                con.execute("""
                    CREATE TABLE dim_customer AS
                    SELECT
                        TRY_CAST(customer_id AS BIGINT) AS customer_id,
                        COALESCE(customer_name, 'Unknown Customer') AS customer_name,
                        COALESCE(region, 'Unknown') AS region
                    FROM stg_dim_customer
                    WHERE customer_id IS NOT NULL
                    GROUP BY 1,2,3
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
                        FROM stg_dim_customer
                    )
                    ORDER BY customer_id;
                """)
        else:
            # derive later (after product/date decision), but easiest is: derive all dims from fact
            pass

        # dim_product
        if cfg.DIM_PRODUCT_CSV.exists():
            dim_cols = set(_csv_columns(con, cfg.DIM_PRODUCT_CSV))
            _require_any(dim_cols, ("product_id", "product_name"), "dim_product.csv")
            con.execute("DROP VIEW IF EXISTS stg_dim_product;")
            con.execute(f"""
                CREATE VIEW stg_dim_product AS
                SELECT *
                FROM read_csv_auto('{cfg.DIM_PRODUCT_CSV.as_posix()}');
            """)

            if "product_id" in dim_cols:
                con.execute("""
                    CREATE TABLE dim_product AS
                    SELECT
                        TRY_CAST(product_id AS BIGINT) AS product_id,
                        COALESCE(product_name, 'Unknown Product') AS product_name,
                        COALESCE(category, 'Unknown') AS category
                    FROM stg_dim_product
                    WHERE product_id IS NOT NULL
                    GROUP BY 1,2,3
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
                        FROM stg_dim_product
                    )
                    ORDER BY product_id;
                """)
        else:
            pass

        # dim_date
        if cfg.DIM_DATE_CSV.exists():
            dim_cols = set(_csv_columns(con, cfg.DIM_DATE_CSV))
            _require_any(dim_cols, ("date",), "dim_date.csv")
            con.execute("DROP VIEW IF EXISTS stg_dim_date;")
            con.execute(f"""
                CREATE VIEW stg_dim_date AS
                SELECT *
                FROM read_csv_auto('{cfg.DIM_DATE_CSV.as_posix()}');
            """)

            # Compute missing attributes if not provided
            has_year = "year" in dim_cols
            has_month = "month" in dim_cols
            has_dow = "day_of_week" in dim_cols

            con.execute(f"""
                CREATE TABLE dim_date AS
                SELECT
                    TRY_CAST(date AS DATE) AS date,
                    {("TRY_CAST(year AS INTEGER)" if has_year else "EXTRACT(YEAR FROM TRY_CAST(date AS DATE))::INTEGER")} AS year,
                    {("TRY_CAST(month AS INTEGER)" if has_month else "EXTRACT(MONTH FROM TRY_CAST(date AS DATE))::INTEGER")} AS month,
                    {("day_of_week" if has_dow else "STRFTIME(TRY_CAST(date AS DATE), '%A')")} AS day_of_week
                FROM stg_dim_date
                WHERE date IS NOT NULL
                GROUP BY 1,2,3,4
                ORDER BY date;
            """)
        else:
            pass

        # If any dimension was not created from CSV, derive missing ones from fact.
        # For simplicity: if ANY dim missing, we derive ALL missing dims from stg_sales.
        # (This keeps logic predictable and avoids partial mismatches.)
        existing_tables = {r[0] for r in con.execute("SHOW TABLES;").fetchall()}
        if "dim_customer" not in existing_tables or "dim_product" not in existing_tables or "dim_date" not in existing_tables:
            # If some dims already exist from CSV, we keep them and only create the missing ones.
            if "dim_customer" not in existing_tables:
                con.execute("""
                    CREATE TABLE dim_customer AS
                    SELECT
                        DENSE_RANK() OVER (ORDER BY customer_name, region) AS customer_id,
                        customer_name,
                        region
                    FROM (
                        SELECT DISTINCT
                            COALESCE(customer_name, 'Unknown Customer') AS customer_name,
                            COALESCE(region, 'Unknown') AS region
                        FROM stg_sales
                    )
                    ORDER BY customer_id;
                """)
            if "dim_product" not in existing_tables:
                con.execute("""
                    CREATE TABLE dim_product AS
                    SELECT
                        DENSE_RANK() OVER (ORDER BY product_name, category) AS product_id,
                        product_name,
                        category
                    FROM (
                        SELECT DISTINCT
                            COALESCE(product_name, 'Unknown Product') AS product_name,
                            COALESCE(category, 'Unknown') AS category
                        FROM stg_sales
                    )
                    ORDER BY product_id;
                """)
            if "dim_date" not in existing_tables:
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
        # FACT TABLE (resolve keys)
        # -------------------------
        con.execute("DROP TABLE IF EXISTS fact_sales;")

        # Resolve customer key
        if has_customer_id:
            customer_id_expr = "s.customer_id"
            customer_join = ""
        else:
            customer_id_expr = "c.customer_id"
            customer_join = """
                LEFT JOIN dim_customer c
                  ON c.customer_name = COALESCE(s.customer_name, 'Unknown Customer')
                 AND c.region = COALESCE(s.region, 'Unknown')
            """

        # Resolve product key
        if has_product_id:
            product_id_expr = "s.product_id"
            product_join = ""
        else:
            product_id_expr = "p.product_id"
            product_join = """
                LEFT JOIN dim_product p
                  ON p.product_name = COALESCE(s.product_name, 'Unknown Product')
                 AND p.category = COALESCE(s.category, 'Unknown')
            """

        # Always validate date exists in dim_date
        date_join = """
            LEFT JOIN dim_date d
              ON d.date = s.order_date
        """

        # Build a resolved view to run strict checks before creating fact table
        con.execute("DROP VIEW IF EXISTS stg_resolved;")
        con.execute(f"""
            CREATE VIEW stg_resolved AS
            SELECT
                {"s.sale_id AS sale_id," if has_sale_id else "NULL::BIGINT AS sale_id,"}
                {customer_id_expr} AS customer_id,
                {product_id_expr} AS product_id,
                s.order_date AS order_date,
                s.quantity AS quantity,
                s.unit_price AS unit_price,
                -- flags for diagnostics
                ({customer_id_expr} IS NULL) AS missing_customer_key,
                ({product_id_expr} IS NULL) AS missing_product_key,
                (d.date IS NULL) AS missing_date_key
            FROM stg_sales s
            {customer_join}
            {product_join}
            {date_join}
            WHERE s.order_date IS NOT NULL
              AND s.quantity IS NOT NULL
              AND s.unit_price IS NOT NULL;
        """)

        # Strict checks (fail-fast)
        missing_customer = con.execute("SELECT COUNT(*) FROM stg_resolved WHERE missing_customer_key;").fetchone()[0]
        missing_product = con.execute("SELECT COUNT(*) FROM stg_resolved WHERE missing_product_key;").fetchone()[0]
        missing_date = con.execute("SELECT COUNT(*) FROM stg_resolved WHERE missing_date_key;").fetchone()[0]

        if strict and (missing_customer or missing_product or missing_date):
            raise ValueError(
                "Build failed due to unresolved dimension keys.\n"
                f"  Missing customer keys: {missing_customer}\n"
                f"  Missing product keys:  {missing_product}\n"
                f"  Missing date keys:     {missing_date}\n"
                "Fix by:\n"
                "  - ensuring dim_*.csv includes the referenced keys/names, OR\n"
                "  - ensuring fact_sales.csv has matching customer/product fields, OR\n"
                "  - removing dim_date.csv so date dim is computed from fact, OR\n"
                "  - rerun with strict=False to drop unresolved rows."
            )

        # Create fact_sales (drop unresolved rows if not strict)
        filter_clause = "WHERE NOT missing_customer_key AND NOT missing_product_key AND NOT missing_date_key" if not strict else ""

        sale_id_expr = "sale_id" if has_sale_id else "ROW_NUMBER() OVER (ORDER BY order_date, customer_id, product_id)::BIGINT"

        con.execute(f"""
            CREATE TABLE fact_sales AS
            SELECT
                {sale_id_expr} AS sale_id,
                customer_id,
                product_id,
                order_date,
                quantity,
                unit_price
            FROM stg_resolved
            {filter_clause};
        """)

        # -------------------------
        # Row counts
        # -------------------------
        counts = {
            "dim_customer": con.execute("SELECT COUNT(*) FROM dim_customer;").fetchone()[0],
            "dim_product": con.execute("SELECT COUNT(*) FROM dim_product;").fetchone()[0],
            "dim_date": con.execute("SELECT COUNT(*) FROM dim_date;").fetchone()[0],
            "fact_sales": con.execute("SELECT COUNT(*) FROM fact_sales;").fetchone()[0],
        }
        return counts

    finally:
        con.close()


def build_default(reset: bool = True, strict: bool = True) -> Dict[str, int]:
    """Convenience wrapper for early-stage work."""
    from .config import load_config
    cfg = load_config()
    return build(cfg, reset=reset, strict=strict)