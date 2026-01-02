"""
mini_data_mart_loaded_csv.py
Option 2: Build the DuckDB mini data mart FROM CSV files (instead of hardcoded rows).

Expected CSV (minimum):
  - fact_sales.csv with columns:
      sale_id, customer_id, product_id, order_date, quantity, unit_price

Optional CSV (if you have them):
  - dim_customer.csv with columns: customer_id, customer_name, region
  - dim_product.csv  with columns: product_id, product_name, category

Behavior:
  - Drops & rebuilds tables each run (clean rebuild).
  - Loads fact_sales from CSV.
  - Creates dim_date from distinct order_date values (derived).
  - Loads dims from CSV if present; otherwise creates placeholders from IDs.
"""

from pathlib import Path
import duckdb


def find_csv(base_dir: Path, filename: str) -> Path | None:
    """Look for filename in common locations."""
    candidates = [
        base_dir / filename,
        base_dir / "data" / filename,
        base_dir / "data" / "input" / filename,
        base_dir / "data" / "raw" / filename,
    ]
    for p in candidates:
        if p.exists() and p.is_file():
            return p
    return None


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    db_path = base_dir / "mini_data_mart.duckdb"

    fact_path = find_csv(base_dir, "fact_sales.csv")
    if not fact_path:
        raise FileNotFoundError(
            "Could not find fact_sales.csv. Place it next to this script, or under data/."
        )

    dim_customer_path = find_csv(base_dir, "dim_customer.csv")
    dim_product_path = find_csv(base_dir, "dim_product.csv")

    print(f"Building DuckDB mini data mart from CSV")
    print(f"Database: {db_path}")
    print(f"fact_sales.csv: {fact_path}")
    print(f"dim_customer.csv: {dim_customer_path if dim_customer_path else '(not found → generate placeholders)'}")
    print(f"dim_product.csv:  {dim_product_path if dim_product_path else '(not found → generate placeholders)'}")

    con = duckdb.connect(db_path.as_posix())

    # Clean rebuild
    con.execute("DROP TABLE IF EXISTS fact_sales;")
    con.execute("DROP TABLE IF EXISTS dim_customer;")
    con.execute("DROP TABLE IF EXISTS dim_product;")
    con.execute("DROP TABLE IF EXISTS dim_date;")

    # --- Stage fact CSV into a temp table ---
    # read_csv_auto infers types; we explicitly cast order_date to DATE later.
    con.execute(f"""
        CREATE TEMP TABLE fact_sales_stage AS
        SELECT *
        FROM read_csv_auto('{fact_path.as_posix().replace("'", "''")}', header=true);
    """)

    # --- Create dim_customer ---
    con.execute("""
        CREATE TABLE dim_customer (
            customer_id   INTEGER PRIMARY KEY,
            customer_name VARCHAR,
            region        VARCHAR
        );
    """)

    if dim_customer_path:
        con.execute(f"""
            INSERT INTO dim_customer
            SELECT
                CAST(customer_id AS INTEGER) AS customer_id,
                CAST(customer_name AS VARCHAR) AS customer_name,
                CAST(region AS VARCHAR) AS region
            FROM read_csv_auto('{dim_customer_path.as_posix().replace("'", "''")}', header=true);
        """)
    else:
        # Placeholder dims derived from fact IDs
        con.execute("""
            INSERT INTO dim_customer
            SELECT
                CAST(customer_id AS INTEGER) AS customer_id,
                'Customer ' || CAST(customer_id AS VARCHAR) AS customer_name,
                'Unknown' AS region
            FROM (SELECT DISTINCT customer_id FROM fact_sales_stage)
            ORDER BY customer_id;
        """)

    # --- Create dim_product ---
    con.execute("""
        CREATE TABLE dim_product (
            product_id    INTEGER PRIMARY KEY,
            product_name  VARCHAR,
            category      VARCHAR
        );
    """)

    if dim_product_path:
        con.execute(f"""
            INSERT INTO dim_product
            SELECT
                CAST(product_id AS INTEGER) AS product_id,
                CAST(product_name AS VARCHAR) AS product_name,
                CAST(category AS VARCHAR) AS category
            FROM read_csv_auto('{dim_product_path.as_posix().replace("'", "''")}', header=true);
        """)
    else:
        con.execute("""
            INSERT INTO dim_product
            SELECT
                CAST(product_id AS INTEGER) AS product_id,
                'Product ' || CAST(product_id AS VARCHAR) AS product_name,
                'Unknown' AS category
            FROM (SELECT DISTINCT product_id FROM fact_sales_stage)
            ORDER BY product_id;
        """)

    # --- Create fact_sales ---
    con.execute("""
        CREATE TABLE fact_sales (
            sale_id     INTEGER PRIMARY KEY,
            customer_id INTEGER,
            product_id  INTEGER,
            order_date  DATE,
            quantity    INTEGER,
            unit_price  DOUBLE,
            FOREIGN KEY (customer_id) REFERENCES dim_customer (customer_id),
            FOREIGN KEY (product_id) REFERENCES dim_product (product_id),
            FOREIGN KEY (order_date) REFERENCES dim_date (date)
        );
    """)

    # Insert fact rows (cast types; cast order_date to DATE)
    con.execute("""
        INSERT INTO fact_sales
        SELECT
            CAST(sale_id AS INTEGER) AS sale_id,
            CAST(customer_id AS INTEGER) AS customer_id,
            CAST(product_id AS INTEGER) AS product_id,
            CAST(order_date AS DATE) AS order_date,
            CAST(quantity AS INTEGER) AS quantity,
            CAST(unit_price AS DOUBLE) AS unit_price
        FROM fact_sales_stage;
    """)

    # --- Create dim_date derived from fact_sales.order_date ---
    con.execute("""
        CREATE TABLE dim_date AS
        SELECT
            d AS date,
            EXTRACT(year FROM d) AS year,
            EXTRACT(month FROM d) AS month,
            STRFTIME(d, '%A') AS day_of_week
        FROM (
            SELECT DISTINCT order_date AS d
            FROM fact_sales
            WHERE order_date IS NOT NULL
        )
        ORDER BY date;
    """)

    # Verification
    tables = con.execute("SHOW TABLES;").fetchdf()
    print("\nTables in the mini data mart:")
    print(tables)

    counts = con.execute("""
        SELECT
            (SELECT COUNT(*) FROM dim_customer) AS dim_customer_rows,
            (SELECT COUNT(*) FROM dim_product)  AS dim_product_rows,
            (SELECT COUNT(*) FROM dim_date)     AS dim_date_rows,
            (SELECT COUNT(*) FROM fact_sales)   AS fact_sales_rows;
    """).fetchdf()
    print("\nRow counts:")
    print(counts)

    con.close()
    print("\nMini data mart build complete (CSV-driven).")


if __name__ == "__main__":
    main()