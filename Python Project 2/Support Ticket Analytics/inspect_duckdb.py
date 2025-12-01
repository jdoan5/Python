import duckdb
import pandas as pd

con = duckdb.connect("tickets.duckdb")

# Don’t truncate columns/width in pandas
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 0)   # 0 = use as much width as needed

df = con.sql("SELECT * FROM fact_tickets LIMIT 10;").df()
print(df)

# Export a sample to CSV
con.sql("COPY (SELECT * FROM fact_tickets LIMIT 100) TO 'fact_sample.csv' WITH (HEADER, DELIMITER ',');")
