"""
Explore / query the Mini Data Mart.

Stage 2 note:
- This script intentionally avoids pandas so it works even in minimal environments.
- It prints a few example analytics queries directly from DuckDB.
"""

from __future__ import annotations

from pathlib import Path
import duckdb


BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "mini_data_mart.duckdb"

print(f"Opening database: {DB_PATH}")
con = duckdb.connect(DB_PATH.as_posix())


def print_table(title: str, rows: list[tuple], cols: list[str], max_rows: int = 25) -> None:
    print(f"\n== {title} ==")
    if not rows:
        print("(no rows)")
        return

    rows = rows[:max_rows]
    data = [cols] + [[("" if v is None else str(v)) for v in r] for r in rows]
    widths = [max(len(data[r][c]) for r in range(len(data))) for c in range(len(cols))]

    def fmt_row(r):
        return " | ".join(val.ljust(widths[i]) for i, val in enumerate(r))

    print(fmt_row(data[0]))
    print("-+-".join("-" * w for w in widths))
    for r in data[1:]:
        print(fmt_row(r))

    if len(rows) == max_rows:
        print(f"... (showing first {max_rows} rows)")


# 1) Show tables
tables = [t[0] for t in con.execute("SHOW TABLES;").fetchall()]
print_table("TABLES", [(t,) for t in tables], ["name"])

# 2) Quick sanity: first 10 sales
if "fact_sales" in tables:
    q = """
    SELECT sale_id, customer_id, product_id, order_date, quantity, unit_price,
           quantity * unit_price AS revenue
    FROM fact_sales
    ORDER BY sale_id
    LIMIT 10;
    """
    res = con.execute(q)
    print_table("FACT_SALES (sample)", res.fetchall(), [c[0] for c in res.description], max_rows=10)

# 3) Revenue by customer & category (requires dims)
if all(t in tables for t in ["fact_sales", "dim_customer", "dim_product"]):
    q = """
    SELECT
        c.customer_name,
        p.category,
        SUM(f.quantity * f.unit_price) AS total_revenue
    FROM fact_sales f
    JOIN dim_customer c ON f.customer_id = c.customer_id
    JOIN dim_product  p ON f.product_id  = p.product_id
    GROUP BY c.customer_name, p.category
    ORDER BY total_revenue DESC;
    """
    res = con.execute(q)
    print_table("Revenue by customer & category", res.fetchall(), [c[0] for c in res.description])

# 4) Top products by revenue
if all(t in tables for t in ["fact_sales", "dim_product"]):
    q = """
    SELECT
        p.product_name,
        p.category,
        SUM(f.quantity * f.unit_price) AS total_revenue
    FROM fact_sales f
    JOIN dim_product p ON f.product_id = p.product_id
    GROUP BY p.product_name, p.category
    ORDER BY total_revenue DESC
    LIMIT 10;
    """
    res = con.execute(q)
    print_table("Top products by revenue", res.fetchall(), [c[0] for c in res.description], max_rows=10)

con.close()
print("\nExplore complete.")
