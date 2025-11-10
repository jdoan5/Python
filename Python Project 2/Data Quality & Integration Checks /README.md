# Data Quality & Integration Checks (Python/CSV/SQLite)

Config-driven data quality checker. Validates **table rules** (row counts, duplicates), **column rules**
(types, nulls, uniqueness, regex, ranges), and **integration rules** (foreign keys to lookup data).
Emits both **JSON** and **Markdown** reports you can share or gate in CI.

## What it checks
- **Table:** minimum rows, duplicate rows, duplicate business keys
- **Columns:** type coercion (int/float/bool/date/datetime), NOT NULL, uniqueness, allowed values, regex patterns
- **Integrations:** foreign-key presence against a parent lookup (CSV or SQLite)

## Project structure