from pathlib import Path
import duckdb

# --- Paths ---
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "mini_data_mart.duckdb" #DuckDB file
DATA_DIR = BASE_DIR / "data"

SALES_CVS = DATA_DIR /"fact_sales.csv"

print(f"Creating / using database at: {DB_PATH}")

# Connect (file is created if it doesn't exist)
con = duckdb.connect(DB_PATH.as_posix())

# --- Drop tables if they already exist (so rebuild is clean) ---
con.execute("DROP TABLE IF EXISTS fact_sales;")
con.execute("DROP TABLE IF EXISTS dim_customer;")
con.execute("DROP TABLE IF EXISTS dim_product;")
con.execute("DROP TABLE IF EXISTS dim_date;")

# --- Create dimension tables ---

con.execute("""
            CREATE TABLE dim_customer
            (
                customer_id   INTEGER PRIMARY KEY,
                customer_name VARCHAR,
                region        VARCHAR
            );
            """)

con.execute("""
            INSERT INTO dim_customer
            VALUES (1, 'Alice', 'North'),
                   (2, 'Bob', 'South'),
                   (3, 'Carol', 'East');
            """)

con.execute("""
            CREATE TABLE dim_product
            (
                product_id   INTEGER PRIMARY KEY,
                product_name VARCHAR,
                category     VARCHAR
            );
            """)

con.execute("""
            INSERT INTO dim_product
            VALUES (10, 'Laptop', 'Electronics'),
                   (11, 'Headphones', 'Electronics'),
                   (12, 'Desk Chair', 'Furniture');
            """)

con.execute("""
            CREATE TABLE dim_date
            (
                date        DATE PRIMARY KEY,
                year        INTEGER,
                month       INTEGER,
                day_of_week VARCHAR
            );
            """)

con.execute("""
            INSERT INTO dim_date
            VALUES ('2025-01-01', 2025, 1, 'Wednesday'),
                   ('2025-01-02', 2025, 1, 'Thursday'),
                   ('2025-01-03', 2025, 1, 'Friday');
            """)

# --- Create fact table ---

con.execute("""
            CREATE TABLE fact_sales
            (
                sale_id     INTEGER PRIMARY KEY,
                customer_id INTEGER,
                product_id  INTEGER,
                order_date  DATE,
                quantity    INTEGER,
                unit_price DOUBLE,
                FOREIGN KEY (customer_id) REFERENCES dim_customer (customer_id),
                FOREIGN KEY (product_id) REFERENCES dim_product (product_id),
                FOREIGN KEY (order_date) REFERENCES dim_date (date)
            );
            """)

con.execute("""
            INSERT INTO fact_sales
            VALUES (1001, 1, 10, '2025-01-01', 1, 1200.00),
                   (1002, 2, 11, '2025-01-02', 2, 200.00),
                   (1003, 3, 12, '2025-01-03', 1, 300.00),
                   (1004, 1, 11, '2025-01-03', 1, 200.00);
            """)

# Verify tables
tables = con.execute("SHOW TABLES;").fetchdf()
print("\nTables in the mini data mart:")
print(tables)

con.close()
print("\nMini data mart build complete.")
