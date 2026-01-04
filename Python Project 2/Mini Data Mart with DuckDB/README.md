# Mini Data Mart with DuckDB (Stages 1–3)

A staged data-platform learning project that builds a small **star-schema mini data mart** using **DuckDB + Python**.  
Each stage adds one concrete capability: repeatable builds, analytics exploration, exports, and (next) data quality + richer inputs.

**Data source:** Synthetic / generated data for learning and demonstration (no real customer or production data).

---

## What this project demonstrates

- **Dimensional modeling (star schema):** `dim_customer`, `dim_product`, `dim_date` → `fact_sales`
- **Repeatable local analytics:** DuckDB file on disk (`mini_data_mart.duckdb`)
- **ETL patterns:** build step creates tables + loads data, explore step runs analytics queries
- **Exports:** optional helpers write tables to `data/CSV-export/` and `data/JSON-export/`

---

## Stages summary

### Stage 1 — Hardcoded seed data (baseline)
**Goal:** Prove the end-to-end workflow with a small star schema.

**What Stage 1 does**
- Creates the star schema tables (`dim_*`, `fact_sales`)
- Loads **seed data hardcoded in Python**
- Runs sample analytics queries (via an explore script)
- Optional exports to CSV/JSON

**Key learning focus**
- Schema design (facts vs. dimensions)
- Joining fact → dims for analytics outputs
- A clean, repeatable “build → explore → export” workflow

---

### Stage 2 — CSV-driven loads (fact table from CSV)
**Goal:** Make the data mart respond to input changes without editing Python inserts.

**What Stage 2 adds**
- Loads **`fact_sales` from `fact_sales.csv`** (and optionally other inputs later)
- Rebuilding the mart after editing CSV **updates DuckDB**
- Adds simple validation patterns (e.g., required columns / referential checks)

**Key learning focus**
- Input-driven ETL (source files → staging → final tables)
- Rebuildability and reproducibility
- Separating “data” (CSV) from “logic” (ETL scripts)

---

### Stage 3 — Next stage (planned)
**Goal:** Move toward a more realistic “mini platform” workflow.

Possible Stage 3 upgrades (pick 1–3 to start):
- **Multiple input CSVs**: separate CSVs for dimensions + fact
- **Data quality checks**:
  - row counts
  - null checks
  - referential integrity checks
  - duplicates / primary key checks
- **Parameterized exploration**:
  - date ranges
  - richer analytic outputs
- **Simple CLI wrapper**:
  - `python -m mart build|explore|export|quality`
- Optional: lightweight UI layer (Streamlit or Dash/Plotly)

---

## Repository layout (recommended)

```
Mini Data Mart with DuckDB/
├─ mini_data_mart.duckdb              # DuckDB database file (generated)
├─ fact_sales.csv                     # Stage 2 input (source)
├─ mini_data_mart.py                  # Build script (Stage 1/2)
├─ explore_mini_data_mart.py          # Analytics exploration queries
├─ export_to_csv.py                   # Optional export helper
├─ export_to_json.py                  # Optional export helper
├─ tools/
│  └─ inspect_duckdb.py               # Utility script (optional)
└─ data/
   ├─ CSV-export/                     # Generated exports (optional)
   ├─ JSON-export/                    # Generated exports (optional)
   └─ quality/                        # Stage 3: generated quality reports
```

---

## How to run (general)

### 1) Create and activate a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install duckdb pandas numpy
```

### 2) Build the data mart
```bash
python3 mini_data_mart.py
```

### 3) Explore analytics
```bash
python3 explore_mini_data_mart.py
```

### 4) Optional exports
```bash
python3 export_to_csv.py
python3 export_to_json.py
```

### 5) Optional inspection utility
```bash
python3 tools/inspect_duckdb.py
```

---

## Stage-specific run notes

### Stage 1
- Data is defined in `mini_data_mart.py` (hardcoded inserts).
- If you want to add more rows, you update the Python inserts and rebuild:
  ```bash
  python3 mini_data_mart.py
  python3 explore_mini_data_mart.py
  ```

### Stage 2
- `fact_sales.csv` is the **source of truth** for the fact table.
- After editing `fact_sales.csv`, rebuild:
  ```bash
  python3 mini_data_mart.py
  python3 explore_mini_data_mart.py
  ```
- The database will only reflect CSV changes **after** the build step reloads the data.

---

## Reset and rebuild from scratch

Use this when you want a clean rebuild:
```bash
rm -f mini_data_mart.duckdb
rm -rf data/CSV-export data/JSON-export data/quality
python3 mini_data_mart.py
python3 explore_mini_data_mart.py
```

---

## Common pitfalls and fixes

### “ModuleNotFoundError: No module named 'duckdb'”
You’re running Python outside your venv or pip isn’t installed in that environment.

Fix:
```bash
source .venv/bin/activate
python3 -m pip install duckdb pandas numpy
python3 -c "import duckdb; print(duckdb.__version__)"
```

### “pip: No module named pip”
Your venv was created without pip. Recreate venv:
```bash
deactivate 2>/dev/null || true
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
python3 -m ensurepip --upgrade
python3 -m pip install --upgrade pip
```

---

## Screenshots to capture (portfolio-friendly)

### Stage 1
- Star schema / workflow diagram
- DuckDB tables created (`SHOW TABLES`)
- Example analytics output query result
- Export output folders (`data/CSV-export`, `data/JSON-export`)

### Stage 2
- `fact_sales.csv` shown as the new input source
- Build run output indicating CSV load succeeded
- Table verification after build (counts / sample rows)
- “Before vs after” query output after editing the CSV
- Export output updated after rebuild

### Stage 3 (future)
- Quality report output (JSON/CSV)
- Referential integrity violations report (if any)
- CLI command demo (`build`, `quality`, `explore`)

---

## Next enhancements (Stage 3+)
- Dim CSVs + fact CSVs (full input-driven star schema)
- Stronger data quality layer with artifacts written to `data/quality/`
- Date-range parameterized analytics outputs
- CLI wrapper (`python -m mart ...`)
- Optional dashboard (Streamlit / Dash) for interactive exploration
