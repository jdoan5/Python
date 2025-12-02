import duckdb
import pandas as pd
from pathlib import Path

# Paths
CSV_DIR = Path("data/JSON-export")
CSV_DIR.mkdir(exist_ok=True)

con = duckdb.connect("mini_data_mart.duckdb")

tables = ["dim_customer", "dim_product", "dim_date", "fact_sales"]

print("=== Exporting tables to JSON ===")
for t in tables:
    out_path = CSV_DIR / f"{t}.json"
    print(f"Exporting {t} → {out_path}")

    # Simple and reliable way to export
    df = con.sql(f"SELECT * FROM {t};").df()
    df.to_csv(out_path, index=False)

print("\nDone! JSONs saved in data/JSON-export/")