import duckdb
import pandas as pd

# Connect to the DuckDB mini data mart
con = duckdb.connect("mini_data_mart.duckdb")

# Don't truncate columns/width in pandas
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 0)  # use as much width as needed

print("=== Tables in mini_data_mart.duckdb ===")
print(con.sql("SHOW TABLES;").df())

tables = ["dim_customer", "dim_product", "dim_date", "fact_sales"]

for name in tables:
    print(f"\n=== Sample rows from {name} ===")
    df = con.sql(f"SELECT * FROM {name} LIMIT 5;").df()
    print(df)