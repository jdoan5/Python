from pathlib import Path
import duckdb

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "mini_data_mart.duckdb"
DATA_DIR = BASE_DIR / "data/JSON"

DATA_DIR.mkdir(exist_ok=True)

con = duckdb.connect(DB_PATH.as_posix())

# Export each table as JSON (array of records)
for (table_name,) in con.execute("SHOW TABLES;").fetchall():
    json_path = DATA_DIR / f"{table_name}.json"
    print(f"Exporting {table_name} -> {json_path}")

    con.execute(f"""
        COPY {table_name}
        TO '{json_path.as_posix()}'
        (FORMAT JSON);
    """)

con.close()
print("JSON export complete.")
