# Query and inspect the data mart that already exists
# Connect to existing duckdb
# See what's inside, explorer tables, run reports
from pathlib import Path
import duckdb
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "mini_data_mart.duckdb"

print(f"Opening database: {DB_PATH}")

con = duckdb.connect(DB_PATH.as_posix())

# --- 1. Show all tables ---
print("\n== TABLES ==")
tables = con.execute("SHOW TABLES;").fetchdf()
print(tables)

# --- 2. View each table (first few rows) ---
for (table_name,) in con.execute("SHOW TABLES;").fetchall():
    print(f"\n== {table_name} (first 10 rows) ==")
    df = con.execute(f"SELECT * FROM {table_name} LIMIT 10;").fetchdf()
    print(df)

# --- 3. Example analytics query: revenue by customer and category ---
print("\n== Revenue by customer & category ==")
query = """
SELECT
    c.customer_name,
    p.category,
    SUM(f.quantity * f.unit_price) AS total_revenue
FROM fact_sales f
JOIN dim_customer c ON f.customer_id = c.customer_id
JOIN dim_product p ON f.product_id = p.product_id
GROUP BY c.customer_name, p.category
ORDER BY total_revenue DESC;
"""
result = con.execute(query).fetchdf()
print(result)

con.close()
