from __future__ import annotations

import json
from datetime import datetime

from .config import MartConfig, connect, ensure_dirs


def run_checks(cfg: MartConfig) -> bool:
    """Run basic data quality checks and write a JSON report to data/quality/."""
    ensure_dirs(cfg)
    con = connect(cfg)
    report: dict = {
        "ts_utc": datetime.utcnow().isoformat() + "Z",
        "db_path": cfg.DB_PATH.as_posix(),
        "checks": [],
        "ok": True,
    }

    def add(name: str, ok: bool, details: dict):
        report["checks"].append({"name": name, "ok": ok, "details": details})
        if not ok:
            report["ok"] = False

    try:
        counts = {}
        for t in ("dim_customer", "dim_product", "dim_date", "fact_sales"):
            counts[t] = int(con.execute(f"SELECT COUNT(*) FROM {t};").fetchone()[0])
        add("row_counts", all(v > 0 for v in counts.values()), {"counts": counts})

        nulls = con.execute("""
            SELECT
              SUM(CASE WHEN customer_id IS NULL THEN 1 ELSE 0 END) AS null_customer_id,
              SUM(CASE WHEN product_id  IS NULL THEN 1 ELSE 0 END) AS null_product_id,
              SUM(CASE WHEN order_date  IS NULL THEN 1 ELSE 0 END) AS null_order_date,
              SUM(CASE WHEN quantity    IS NULL THEN 1 ELSE 0 END) AS null_quantity,
              SUM(CASE WHEN unit_price  IS NULL THEN 1 ELSE 0 END) AS null_unit_price
            FROM fact_sales;
        """).fetchone()
        nulls_map = {
            "null_customer_id": int(nulls[0]),
            "null_product_id": int(nulls[1]),
            "null_order_date": int(nulls[2]),
            "null_quantity": int(nulls[3]),
            "null_unit_price": int(nulls[4]),
        }
        add("fact_nulls", all(v == 0 for v in nulls_map.values()), nulls_map)

        missing_customer = int(con.execute("""
            SELECT COUNT(*)
            FROM fact_sales f
            LEFT JOIN dim_customer c ON f.customer_id = c.customer_id
            WHERE c.customer_id IS NULL;
        """).fetchone()[0])

        missing_product = int(con.execute("""
            SELECT COUNT(*)
            FROM fact_sales f
            LEFT JOIN dim_product p ON f.product_id = p.product_id
            WHERE p.product_id IS NULL;
        """).fetchone()[0])

        missing_date = int(con.execute("""
            SELECT COUNT(*)
            FROM fact_sales f
            LEFT JOIN dim_date d ON f.order_date = d.date
            WHERE d.date IS NULL;
        """).fetchone()[0])

        ri = {
            "missing_customer_fk": missing_customer,
            "missing_product_fk": missing_product,
            "missing_date_fk": missing_date,
        }
        add("referential_integrity", all(v == 0 for v in ri.values()), ri)

    finally:
        con.close()

    out = cfg.QUALITY_DIR / "quality_report.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nQuality report written: {out}")
    print("Quality checks: PASS" if report["ok"] else "Quality checks: FAIL")
    return bool(report["ok"])
