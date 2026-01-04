# Check the file path is true
# if exists = false (connecting to a new, empty DB in a different folder)

import os, duckdb

print("Working dir:", os.getcwd())
print("DB path:", os.path.abspath("mini_data_mart.duckdb"))
print("Exists?", os.path.exists("mini_data_mart.duckdb"))

con = duckdb.connect(os.path.abspath("mini_data_mart.duckdb"))
con.execute("SHOW TABLES;").fetchdf()
