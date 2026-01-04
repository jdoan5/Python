from pathlib import Path
import duckdb
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "mini_data_mart.duckdb"
FACT_CSV = BASE_DIR / "fact_sales.csv"

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 0)

con = duckdb.connect(DB_PATH.as_posix())

def table_exists(name: str) -> bool:
    q = "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?;"
    return con.execute(q, [name]).fetchone()[0] > 0

print(f"=== DuckDB file: {DB_PATH} ===")
print("\n== TABLES ==")
print(con.sql("SHOW TABLES;").df())

tables = ["dim_customer", "dim_product", "dim_date", "fact_sales"]

for name in tables:
    if not table_exists(name):
        print(f"\n--- {name} (MISSING) ---")
        continue

    print(f"\n=== {name} ===")
    print("Row count:", con.execute(f"SELECT COUNT(*) FROM {name};").fetchone()[0])

    print("\nSchema:")
    print(con.sql(f"PRAGMA table_info('{name}');").df())

    print("\nSample rows:")
    print(con.sql(f"SELECT * FROM {name} LIMIT 10;").df())

# Stage 2-specific: compare CSV vs DB for fact_sales
if FACT_CSV.exists():
    try:
        csv_rows = pd.read_csv(FACT_CSV).shape[0]
        if table_exists("fact_sales"):
            db_rows = con.execute("SELECT COUNT(*) FROM fact_sales;").fetchone()[0]
            print(f"\n== FACT_SALES CSV vs DB ==")
            print(f"CSV rows: {csv_rows}")
            print(f"DB rows : {db_rows}")
            if csv_rows != db_rows:
                print("WARNING: CSV row count != DB row count (check load logic / filters).")
    except Exception as e:
        print(f"\nWARNING: Could not read {FACT_CSV.name}: {e}")

# Quick integrity checks (useful when loading from CSV)
if all(table_exists(t) for t in ["fact_sales", "dim_customer", "dim_product", "dim_date"]):
    print("\n== INTEGRITY CHECKS (fact -> dims) ==")

    bad_customer = con.execute("""
        SELECT COUNT(*)
        FROM fact_sales f
        LEFT JOIN dim_customer c ON c.customer_id = f.customer_id
        WHERE c.customer_id IS NULL;
    """).fetchone()[0]

    bad_product = con.execute("""
        SELECT COUNT(*)
        FROM fact_sales f
        LEFT JOIN dim_product p ON p.product_id = f.product_id
        WHERE p.product_id IS NULL;
    """).fetchone()[0]

    bad_date = con.execute("""
        SELECT COUNT(*)
        FROM fact_sales f
        LEFT JOIN dim_date d ON d.date = f.order_date
        WHERE d.date IS NULL;
    """).fetchone()[0]

    print("Missing dim_customer matches:", bad_customer)
    print("Missing dim_product matches :", bad_product)
    print("Missing dim_date matches    :", bad_date)

con.close()