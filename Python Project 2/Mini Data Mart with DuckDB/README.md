# Mini Data Mart with DuckDB (PyCharm + Python)

This project demonstrates how to build a small **star-schema mini data mart** using **DuckDB** and **Python**, managed from **PyCharm**.

It creates:

- A **DuckDB database file**: `mini_data_mart.duckdb`
- **Dimension tables**: `dim_customer`, `dim_product`, `dim_date`
- **Fact table**: `fact_sales`
- Example **analytics queries** (e.g., revenue by customer and category)
- Optional **CSV exports** OR **JSON exports** for use in Excel or other tools

---

## Project Overview

The project simulates a tiny sales analytics environment:

- **Dimensions**
  - `dim_customer` – who bought (customer name, region)
  - `dim_product` – what was bought (product name, category)
  - `dim_date` – when it was bought (date, year, month, day_of_week)

- **Fact**
  - `fact_sales` – the measurable events (quantity, unit_price, sale_id)

The scripts:

- `build_mini_data_mart.py`  
  Builds the DuckDB file and creates/populates all tables.

- `explore_mini_data_mart.py`  
  Connects to the DuckDB file, shows sample rows, and runs analytics queries.

- `export_to_csv.py` *(optional)*  
  Exports DuckDB tables to CSV files in a `data/CSV` directory.
OR
- `export_to_json.py` *(optional)*  
  Exports DuckDB tables to JSON files in a `data/JSON` directory.
---

## Architecture Diagram (Mermaid)

```mermaid
graph TD
    A[Python scripts in PyCharm] --> B[mini_data_mart.duckdb]

    subgraph Star Schema in DuckDB
        B --> F_FACT[fact_sales]
        B --> C_DIM[dim_customer]
        B --> P_DIM[dim_product]
        B --> D_DIM[dim_date]
        
        F_FACT --> C_DIM
        F_FACT --> P_DIM
        F_FACT --> D_DIM
    end

    B -- "COPY ... TO CSV" --> E_CSV[(data/csv/*.csv)]
    B -- "COPY ... TO JSON" --> E_JSON[(data/json/*.json)]
   

    
