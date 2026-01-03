# Create and load the data mart
# This is the ETL/setup script

from pathlib import Path
import duckdb

# --- Paths ---
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "mini_data_mart.duckdb"

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
                   (3, 'Carol', 'East'),
                   (4, 'Dan', 'West'),
                   (5, 'Eve', 'North'),
                   (6, 'Frank', 'South'),
                   (7, 'Grace', 'East'),
                   (8, 'Harry', 'West'),
                   (9, 'Ivan', 'North'),
                   (10, 'Judy', 'South');
                
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
                   (12, 'Desk Chair', 'Furniture'),
                   (13, 'Energy Drink', 'Beverages'),
                   (14, 'Coffee Maker', 'Appliances'),
                   (15, 'Toaster', 'Appliances'),
                   (16, 'Television', 'Electronics'),
                   (17, 'Smartphone', 'Electronics'),
                   (18, 'Tablet', 'Electronics'),
                   (19, 'Laptop Bag', 'Accessories'),
                   (20, 'Mouse Pad', 'Accessories');
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
                   ('2025-01-03', 2025, 1, 'Friday'),
                   ('2025-05-12', 2025, 5, 'Tuesday'),
                   ('2025-08-15', 2025, 8, 'Friday'),
                   ('2025-11-28', 2025, 11, 'Monday'),
                   ('2025-12-04', 2025, 12, 'Thursday'),
                   ('2026-01-01', 2026, 1, 'Monday'),
                   ('2026-01-02', 2026, 1, 'Tuesday'),
                   ('2026-01-03', 2026, 1, 'Wednesday'),
                
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
                   (1002, 1, 11, '2025-01-02', 1, 200.00),
                   (1003, 3, 12, '2025-01-03', 1, 300.00),
                   (1004, 1, 11, '2025-01-03', 1, 200.00),
                   (1005, 4, 13, '2026-01-01', 2, 150.00),
                   (1006, 5, 14, '2026-01-02', 1, 100.00),
                   (1007, 7, 15, '2026-01-03', 1, 120.00),
                   (1008, 8, 16, '2026-01-03', 2, 150.00),
                   (1009, 9, 17, '2025-05-12', 5, 180.00),
                   (1010, 10, 18, '2025-11-28', 0, 120.00);
            """)

# Verify tables
tables = con.execute("SHOW TABLES;").fetchdf()
print("\nTables in the mini data mart:")
print(tables)

con.close()
print("\nMini data mart build complete.")
