#Optional to export .duckdb to csv file
from pathlib import Path
import duckdb

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "mini_data_mart.duckdb"
DATA_DIR = BASE_DIR / "data/CSV"

DATA_DIR.mkdir(exist_ok=True)

con = duckdb.connect(DB_PATH.as_posix())

# Export each table
for (table_name,) in con.execute("SHOW TABLES;").fetchall():
    csv_path = DATA_DIR / f"{table_name}.csv"
    print(f"Exporting {table_name} -> {csv_path}")
    con.execute(f"""
        COPY {table_name}
        TO '{csv_path.as_posix()}'
        (HEADER, DELIMITER ',');
    """)

con.close()
print("Export complete.")
