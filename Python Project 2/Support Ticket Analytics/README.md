# Support Ticket Analytics

Small analytics playground for IT support tickets.

The goal is to:

1. Generate a fake but realistic support ticket dataset.
2. Build a tiny analytics “mart” in DuckDB.
3. Run simple queries / KPIs from that mart (for dashboards or reports).

---

## Project Structure

```text
Support Ticket Analytics/
├─ data/
│  ├─ raw/          # Source CSV files (generated or downloaded)
│  └─ processed/    # Outputs from ETL (cleaned / enriched CSVs)
├─ etl_build_mart.py
├─ generate_fake_tickets.py
├─ inspect_duckdb.py
├─ requirements.txt
├─ tickets.duckdb
└─ README.md
```



## Setup
```text
# 1. Create a virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

```
## How to Run
```text
1. Generate fake tickets
This creates the raw CSV dataset:
python generate_fake_tickets.py

Output: data/raw/support_tickets.csv
You can also replace this file with a real CSV from another source if you want to use real data (same columns or adjust the ETL script accordingly).

2. Build the analytics mart (DuckDB + processed CSVs)
python etl_build_mart.py

This will:
Read data/raw/support_tickets.csv

Create an enriched fact table and any KPI tables

Save:
tickets.duckdb (DuckDB database)
Processed CSVs into data/processed/

3. Inspect the DuckDB database (optional)
python inspect_duckdb.py

You should see:
A table list (e.g. fact_tickets)
A small sample of rows from fact_tickets printed in the console
