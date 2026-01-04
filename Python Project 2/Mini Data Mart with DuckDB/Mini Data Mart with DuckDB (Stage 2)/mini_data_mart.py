"""
Mini Data Mart with DuckDB — Stage 2 (CSV-driven)

Goal:
- Build a small star schema in DuckDB from CSV inputs (synthetic data for learning).
- Updating the CSV should change what gets loaded, without editing Python code.

Expected fact_sales.csv (recommended columns):
- sale_id (optional)
- customer_name (required)
- region (optional; defaults to 'Unknown')
- product_name (required)
- category (optional; defaults to 'Unknown')
- order_date (required; ISO date like 2025-01-03)
- quantity (required; integer)
- unit_price (required; number)

Alternative supported (IDs-based) fact CSV:
- sale_id, customer_id, product_id, order_date, quantity, unit_price
(If you use IDs-based, you must also ensure dim tables exist or modify this script accordingly.)
"""


from __future__ import annotations

from pathlib import Path
import duckdb


# --- Paths ---
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "mini_data_mart.duckdb"
FACT_CSV = BASE_DIR / "fact_sales.csv"

print(f"Creating / using database at: {DB_PATH}")
print(f"Loading fact_sales from CSV: {FACT_CSV}")

if not FACT_CSV.exists():
    raise FileNotFoundError(
        f"Missing {FACT_CSV.name}. Put fact_sales.csv next to this script, then re-run."
    )

con = duckdb.connect(DB_PATH.as_posix())

# --- Clean rebuild ---
con.execute("DROP TABLE IF EXISTS stg_sales;")
con.execute("DROP TABLE IF EXISTS fact_sales;")
con.execute("DROP TABLE IF EXISTS dim_customer;")
con.execute("DROP TABLE IF EXISTS dim_product;")
con.execute("DROP TABLE IF EXISTS dim_date;")


def _csv_columns() -> list[str]:
    # DuckDB can introspect the CSV schema without creating a table.
    # DESCRIBE returns: column_name, column_type, null, key, default, extra
    rows = con.execute(
        f"DESCRIBE SELECT * FROM read_csv_auto('{FACT_CSV.as_posix()}', header=true);"
    ).fetchall()
    return [r[0] for r in rows]


cols = {c.lower() for c in _csv_columns()}

# --- Stage 2: preferred "names-based" CSV ---
if {"customer_name", "product_name", "order_date", "quantity", "unit_price"}.issubset(cols):
    con.execute(
        f"""
        CREATE TABLE stg_sales AS
        SELECT
            COALESCE(TRY_CAST(sale_id AS INTEGER), ROW_NUMBER() OVER()) AS sale_id,
            CAST(customer_name AS VARCHAR) AS customer_name,
            COALESCE(CAST(region AS VARCHAR), 'Unknown') AS region,
            CAST(product_name AS VARCHAR) AS product_name,
            COALESCE(CAST(category AS VARCHAR), 'Unknown') AS category,
            TRY_CAST(order_date AS DATE) AS order_date,
            TRY_CAST(quantity AS INTEGER) AS quantity,
            TRY_CAST(unit_price AS DOUBLE) AS unit_price
        FROM read_csv_auto('{FACT_CSV.as_posix()}', header=true);
        """
    )

    # --- Validate required fields after TRY_CAST ---
    bad = con.execute(
        """
        SELECT
            SUM(CASE WHEN customer_name IS NULL OR customer_name = '' THEN 1 ELSE 0 END) AS bad_customer,
            SUM(CASE WHEN product_name  IS NULL OR product_name  = '' THEN 1 ELSE 0 END) AS bad_product,
            SUM(CASE WHEN order_date   IS NULL THEN 1 ELSE 0 END) AS bad_date,
            SUM(CASE WHEN quantity     IS NULL THEN 1 ELSE 0 END) AS bad_qty,
            SUM(CASE WHEN unit_price   IS NULL THEN 1 ELSE 0 END) AS bad_price
        FROM stg_sales;
        """
    ).fetchone()

    if any(x and x > 0 for x in bad):
        raise ValueError(
            "fact_sales.csv has invalid/missing values. Fix the CSV then re-run.\n"
            f"Bad counts: customer_name={bad[0]}, product_name={bad[1]}, "
            f"order_date={bad[2]}, quantity={bad[3]}, unit_price={bad[4]}"
        )

    # --- Build dimensions from staging ---
    con.execute(
        """
        CREATE TABLE dim_customer AS
        SELECT
            ROW_NUMBER() OVER (ORDER BY customer_name, region) AS customer_id,
            customer_name,
            region
        FROM (SELECT DISTINCT customer_name, region FROM stg_sales) t;
        """
    )

    con.execute(
        """
        CREATE TABLE dim_product AS
        SELECT
            ROW_NUMBER() OVER (ORDER BY product_name, category) AS product_id,
            product_name,
            category
        FROM (SELECT DISTINCT product_name, category FROM stg_sales) t;
        """
    )

    con.execute(
        """
        CREATE TABLE dim_date AS
        SELECT
            date AS date,
            EXTRACT(year  FROM date) AS year,
            EXTRACT(month FROM date) AS month,
            STRFTIME(date, '%A')     AS day_of_week
        FROM (
            SELECT DISTINCT order_date AS date
            FROM stg_sales
        ) t;
        """
    )

    # --- Build fact table by joining back to dims ---
    con.execute(
        """
        CREATE TABLE fact_sales AS
        SELECT
            s.sale_id,
            c.customer_id,
            p.product_id,
            s.order_date,
            s.quantity,
            s.unit_price
        FROM stg_sales s
        JOIN dim_customer c
          ON s.customer_name = c.customer_name
         AND s.region        = c.region
        JOIN dim_product p
          ON s.product_name = p.product_name
         AND s.category     = p.category;
        """
    )

# --- Alternative: IDs-based CSV (less friendly, but supported) ---
elif {"customer_id", "product_id", "order_date", "quantity", "unit_price"}.issubset(cols):
    # In this mode, you are providing foreign keys directly.
    # This script creates dim_date, and loads fact_sales as-is.
    con.execute(
        f"""
        CREATE TABLE fact_sales AS
        SELECT
            COALESCE(TRY_CAST(sale_id AS INTEGER), ROW_NUMBER() OVER()) AS sale_id,
            TRY_CAST(customer_id AS INTEGER) AS customer_id,
            TRY_CAST(product_id  AS INTEGER) AS product_id,
            TRY_CAST(order_date  AS DATE)    AS order_date,
            TRY_CAST(quantity    AS INTEGER) AS quantity,
            TRY_CAST(unit_price  AS DOUBLE)  AS unit_price
        FROM read_csv_auto('{FACT_CSV.as_posix()}', header=true);
        """
    )

    con.execute(
        """
        CREATE TABLE dim_date AS
        SELECT
            date AS date,
            EXTRACT(year  FROM date) AS year,
            EXTRACT(month FROM date) AS month,
            STRFTIME(date, '%A')     AS day_of_week
        FROM (
            SELECT DISTINCT order_date AS date
            FROM fact_sales
        ) t;
        """
    )

    print(
        "\nNOTE: IDs-based CSV detected. dim_customer and dim_product are NOT created in this mode.\n"
        "If you want the analytics queries in explore_mini_data_mart.py to work, use the names-based CSV format."
    )

else:
    raise ValueError(
        "Unsupported fact_sales.csv format.\n"
        "Use names-based columns: customer_name, product_name, order_date, quantity, unit_price (recommended),\n"
        "optionally sale_id, region, category.\n"
        "OR use IDs-based columns: customer_id, product_id, order_date, quantity, unit_price."
    )


# --- Show table counts ---
print("\n== TABLE COUNTS ==")
tables = con.execute("SHOW TABLES;").fetchall()
for (tname,) in tables:
    n = con.execute(f"SELECT COUNT(*) FROM {tname};").fetchone()[0]
    print(f"{tname:12s} {n}")

con.close()
print("\nStage 2 build complete.")
