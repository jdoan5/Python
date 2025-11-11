# Data Quality & Integration Checks (Python/CSV/SQLite)

Config-driven data quality checker. Validates **table rules** (row counts, duplicates), **column rules**
(types, nulls, uniqueness, regex, ranges), and **integration rules** (foreign keys to lookup data).
Emits both **JSON** and **Markdown** reports you can share or gate in CI.

## What it checks
- **Table:** minimum rows, duplicate rows, duplicate business keys
- **Columns:** type coercion (int/float/bool/date/datetime), NOT NULL, uniqueness, allowed values, regex patterns
- **Integrations:** foreign-key presence against a parent lookup (CSV or SQLite)


## Project structure
data-quality/
├─ data/
│  ├─ customers.csv
│  ├─ plan_lookup.csv
│  └─ orders.csv
├─ dq_config.yaml
├─ dq_config_orders.yaml
├─ dq_check.py
├─ requirements.txt
└─ README.md

```mermaid
flowchart TD
  CFG[dq_config.yaml] --> S[source]
  CFG --> C[checks]
  CFG --> I[integrations]
  C --> T[table]
  C --> COL[columns]

