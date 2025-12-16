# The ETL and modeling script
# Extract read data from raw/tickets_raw.csv
# Transform clean raw data
    # Computes metrics per ticket
    # Writes cleaned/derived CSV into data/processed
# Load

from pathlib import Path
from typing import Tuple

import duckdb
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
RAW_FILE = BASE_DIR / "data" / "raw" / "support_tickets.csv"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

DB_FILE = BASE_DIR / "tickets.duckdb"


def load_raw_tickets() -> pd.DataFrame:
    if not RAW_FILE.exists():
        raise FileNotFoundError(f"Raw CSV not found: {RAW_FILE}")

    df = pd.read_csv(RAW_FILE)

    # Parse datetimes (blank resolved_at becomes NaT)
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
    df["resolved_at"] = pd.to_datetime(df["resolved_at"], errors="coerce", utc=True)

    # Basic validation: drop rows with missing created_at
    df = df.dropna(subset=["created_at"])

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    # Compute resolution duration in hours
    df["resolution_hours"] = (df["resolved_at"] - df["created_at"]).dt.total_seconds() / 3600.0

    # SLA targets by priority (example values)
    sla_by_priority = {
        "High": 24,
        "Medium": 48,
        "Low": 72,
    }
    df["sla_target_hours"] = df["priority"].map(sla_by_priority)

    # Met SLA = resolved & resolution_hours <= target
    df["met_sla"] = (df["resolved_at"].notna()) & (
        df["resolution_hours"] <= df["sla_target_hours"]
    )

    # Extra breakdown fields
    df["created_date"] = df["created_at"].dt.date
    df["created_hour"] = df["created_at"].dt.hour
    df["created_dow"] = df["created_at"].dt.day_name()

    return df


def load_into_duckdb(df: pd.DataFrame) -> duckdb.DuckDBPyConnection:
    # Connect to DuckDB file
    con = duckdb.connect(DB_FILE.as_posix())

    # Replace table each run
    con.execute("DROP TABLE IF EXISTS fact_tickets")

    con.execute("""
        CREATE TABLE fact_tickets AS
        SELECT * FROM df
    """)

    return con


def compute_kpis(con: duckdb.DuckDBPyConnection) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # KPI 1: SLA stats by priority
    kpi_by_priority = con.execute("""
        SELECT
            priority,
            COUNT(*)                      AS total_tickets,
            SUM(CASE WHEN met_sla THEN 1 ELSE 0 END) AS met_sla_count,
            ROUND(100.0 * SUM(CASE WHEN met_sla THEN 1 ELSE 0 END) / COUNT(*), 2)
                AS met_sla_pct,
            ROUND(AVG(resolution_hours), 2) AS avg_resolution_hours
        FROM fact_tickets
        GROUP BY priority
        ORDER BY priority
    """).fetchdf()

    # KPI 2: tickets per day
    kpi_by_day = con.execute("""
        SELECT
            created_date,
            COUNT(*) AS tickets_count
        FROM fact_tickets
        GROUP BY created_date
        ORDER BY created_date
    """).fetchdf()

    return kpi_by_priority, kpi_by_day

def export_kpis(kpi_by_priority: pd.DataFrame, kpi_by_day: pd.DataFrame) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Priority KPI
    out_priority = PROCESSED_DIR / "kpi_by_priority.csv"
    kpi_by_priority.to_csv(out_priority, index=False)

    # Daily volume KPI (normalize column names for the dashboard)
    out_daily = PROCESSED_DIR / "kpi_ticket_volume_daily.csv"
    kpi_by_day_normalized = kpi_by_day.rename(
        columns={
            "created_date": "date",
            "tickets_count": "tickets",
        }
    )
    kpi_by_day_normalized.to_csv(out_daily, index=False)

    print(f"\nExported: {out_priority}")
    print(f"Exported: {out_daily}")

def main() -> None:
    print(f"Loading raw tickets from {RAW_FILE} ...")
    df_raw = load_raw_tickets()

    print("Engineering features (resolution_hours, SLA, etc.) ...")
    df_feat = engineer_features(df_raw)

    print(f"Loading into DuckDB at {DB_FILE} ...")
    con = load_into_duckdb(df_feat)

    print("Computing KPIs ...")
    kpi_by_priority, kpi_by_day = compute_kpis(con)

    print("\n=== KPI by Priority ===")
    print(kpi_by_priority.to_string(index=False))

    print("\n=== Tickets by Day (first 20 rows) ===")
    print(kpi_by_day.head(20).to_string(index=False))

    export_kpis(kpi_by_priority, kpi_by_day)
    con.close()
    print("\nDone. You now have a small ticket mart + KPIs.")


if __name__ == "__main__":
    main()
