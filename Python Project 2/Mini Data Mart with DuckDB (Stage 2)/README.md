# Mini Data Mart with DuckDB — Stage 2 (CSV-Driven Load)

Stage 2 upgrades the mini data mart from **hardcoded seed inserts** (Stage 1) to a **repeatable CSV → DuckDB build**.
When you update `fact_sales.csv`, you can rebuild the database and immediately see the changes in analytics queries.

> **Data source:** Synthetic / generated dataset for learning and demonstration purposes (no real customer or production data).

---

## What Stage 2 adds

- **CSV-driven fact load** (and automatic dimension building)
- **Rebuildable pipeline**: build → query → export
- **Input validation** for dates and numeric fields
- A **names-based CSV format** so you can edit the dataset without managing IDs manually

---

## Project structure

```
Mini Data Mart with DuckDB (Stage 2)/
├─ mini_data_mart.py              # Build script: reads CSV → creates star schema in DuckDB
├─ explore_mini_data_mart.py      # Example analytics queries (reads DuckDB)
├─ inspect_duckdb.py              # (Optional) quick schema/table inspection helper
├─ fact_sales.csv                 # Your input data (edit this for new rows)
├─ fact_sales.template.csv        # Template showing expected CSV columns
├─ mini_data_mart.duckdb          # GENERATED (database file)
└─ data/
   ├─ CSV-export/                 # GENERATED (optional exporters)
   └─ JSON-export/                # GENERATED (optional exporters)
```

---

## Requirements

- Python 3.11+ recommended  
- DuckDB Python package  
- (Optional) Pandas, only if you keep exporters/scripts that use it

### Install (virtualenv recommended)

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install duckdb
```

If you maintain a `requirements.txt`, it can be as small as:

```txt
duckdb
```

---

## Input CSV format (Stage 2)

### Preferred: names-based CSV (recommended)

Copy the template and start editing:

```bash
cp fact_sales.template.csv fact_sales.csv
```

**Expected columns:**

- `order_date` (YYYY-MM-DD)
- `customer_name`
- `region` (optional; defaults to `Unknown`)
- `product_name`
- `category` (optional; defaults to `Unknown`)
- `quantity` (integer)
- `unit_price` (number)

Example:

```csv
order_date,customer_name,region,product_name,category,quantity,unit_price
2025-01-01,Alice,North,Laptop,Electronics,1,1200.00
```

**How it loads:**
- `mini_data_mart.py` reads your CSV into a staging table (`stg_sales`)
- It builds:
  - `dim_customer` from distinct `(customer_name, region)`
  - `dim_product` from distinct `(product_name, category)`
  - `dim_date` from distinct `order_date`
- It loads `fact_sales` by joining the staging rows to those dimensions
- `sale_id` is generated automatically (row number) unless you include it yourself in an ID-based format

---

## Run steps

### 1) Build (creates/rebuilds the DuckDB star schema)

```bash
python mini_data_mart.py
```

### 2) Explore (run analytics queries)

```bash
python explore_mini_data_mart.py
```

### 3) Optional: Inspect database quickly

```bash
python inspect_duckdb.py
```

### 4) Optional: Export outputs

If you have `export_to_csv.py` / `export_to_json.py` in the folder:

```bash
python export_to_csv.py
python export_to_json.py
```

If you prefer DuckDB-native exports, you can also do:

```sql
COPY (SELECT * FROM fact_sales) TO 'data/CSV-export/fact_sales.csv' (HEADER, DELIMITER ',');
```

---

## Rebuild from scratch (clean reset)

Use this when you want a “fresh start”:

```bash
rm -f mini_data_mart.duckdb
rm -rf data/CSV-export data/JSON-export
python mini_data_mart.py
python explore_mini_data_mart.py
```

Notes:
- `mini_data_mart.py` also drops/recreates tables each run, so deleting the DB file is not strictly required.
- Deleting the DB is helpful if you want to guarantee there is no leftover state.

---

## If you edit `fact_sales.csv`, what must you rerun?

- To reflect CSV changes in DuckDB: **rerun `mini_data_mart.py`**
- To see results after rebuild: rerun `explore_mini_data_mart.py`

---

## Common errors and fixes

### `Table with name fact_sales does not exist`
You ran the explore script before successfully building the database.

Fix:

```bash
python mini_data_mart.py
python explore_mini_data_mart.py
```

### Binder error mentioning missing columns (e.g., `customer_id`)
Your CSV columns do not match the expected Stage 2 names-based format.

Fix:
- Ensure `fact_sales.csv` includes `customer_name`, `product_name`, `order_date`, `quantity`, `unit_price`
- Start from `fact_sales.template.csv`

---

## Next stage (idea)

Stage 3 can add one or more of:
- Multiple input CSVs (separate dimension CSVs + fact CSV)
- Data quality checks (row counts, null checks, referential integrity checks)
- Parameterized date ranges and richer analytics outputs
- A simple CLI (e.g., `python -m mart build|explore|export`)
