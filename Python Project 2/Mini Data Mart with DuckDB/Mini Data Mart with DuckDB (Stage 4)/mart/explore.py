from __future__ import annotations

from .config import MartConfig, connect, ensure_dirs


def _q(path: str) -> str:
    return path.replace("'", "''")


def explore(cfg: MartConfig) -> None:
    """Run a few example analytics queries and write outputs to data/analytics/."""
    ensure_dirs(cfg)
    con = connect(cfg)
    try:
        print("\n== TABLES ==")
        try:
            print(con.execute("SHOW TABLES;").fetchdf())
        except Exception:
            print(con.execute("SHOW TABLES;").fetchall())

        queries: dict[str, str] = {
            "revenue_by_customer_category": """
                SELECT
                    c.customer_name,
                    c.region,
                    p.category,
                    SUM(f.quantity * f.unit_price) AS revenue
                FROM fact_sales f
                JOIN dim_customer c ON f.customer_id = c.customer_id
                JOIN dim_product p  ON f.product_id  = p.product_id
                GROUP BY 1,2,3
                ORDER BY revenue DESC
            """,
            "revenue_by_day": """
                SELECT
                    d.date,
                    d.day_of_week,
                    SUM(f.quantity * f.unit_price) AS revenue
                FROM fact_sales f
                JOIN dim_date d ON f.order_date = d.date
                GROUP BY 1,2
                ORDER BY d.date
            """,
        }

        out_dir = cfg.ANALYTICS_DIR
        print(f"\nWriting analytics outputs to: {out_dir}")

        for name, sql in queries.items():
            out_csv = (out_dir / f"{name}.csv").as_posix()
            print(f"\n== {name} ==")
            # Print sample results
            try:
                df = con.execute(sql).fetchdf()
                print(df)
            except Exception:
                print(con.execute(sql).fetchall())

            con.execute(f"COPY ({sql}) TO '{_q(out_csv)}' (HEADER, DELIMITER ',');")
            print(f"Wrote: {out_csv}")

    finally:
        con.close()
