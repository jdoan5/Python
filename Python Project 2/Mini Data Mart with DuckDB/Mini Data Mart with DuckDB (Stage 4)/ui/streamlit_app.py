# ui/streamlit_app.py
from __future__ import annotations

from datetime import date
from pathlib import Path

import duckdb
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parents[1]
DB_PATH = BASE_DIR / "mini_data_mart.duckdb"

st.set_page_config(page_title="Mini Data Mart (Stage 4)", layout="wide")

st.title("Mini Data Mart (Stage 4)")
st.caption("DuckDB-backed star schema with multi-CSV inputs and interactive filters.")

if not DB_PATH.exists():
    st.warning("Database not found. Run `python -m mart build` from the project root first.")
    st.stop()

con = duckdb.connect(DB_PATH.as_posix(), read_only=True)

# Load filter options
customers = con.execute("SELECT customer_name FROM dim_customer ORDER BY 1;").fetchdf()["customer_name"].tolist()
categories = con.execute("SELECT DISTINCT category FROM dim_product ORDER BY 1;").fetchdf()["category"].tolist()
regions = con.execute("SELECT DISTINCT region FROM dim_customer ORDER BY 1;").fetchdf()["region"].tolist()

dmin, dmax = con.execute("SELECT MIN(order_date), MAX(order_date) FROM fact_sales;").fetchone()
if dmin is None or dmax is None:
    st.error("fact_sales is empty. Check your input CSVs and rebuild.")
    st.stop()

with st.sidebar:
    st.subheader("Filters")
    date_range = st.date_input("Order date range", (dmin, dmax))
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date, end_date = dmin, dmax

    sel_regions = st.multiselect("Region", regions, default=regions)
    sel_categories = st.multiselect("Category", categories, default=categories)
    sel_customers = st.multiselect("Customer", customers, default=[])

    st.divider()
    st.write("Tip: leave Customer empty to include all customers.")


where = []
params = []
where.append("f.order_date BETWEEN ? AND ?")
params += [start_date, end_date]

if sel_regions:
    where.append("c.region IN (" + ",".join(["?"] * len(sel_regions)) + ")")
    params += sel_regions

if sel_categories:
    where.append("p.category IN (" + ",".join(["?"] * len(sel_categories)) + ")")
    params += sel_categories

if sel_customers:
    where.append("c.customer_name IN (" + ",".join(["?"] * len(sel_customers)) + ")")
    params += sel_customers

where_sql = " AND ".join(where)

st.subheader("Revenue by customer and category")
sql_rev = f"""
    SELECT
        c.customer_name,
        c.region,
        p.category,
        SUM(f.quantity * f.unit_price) AS revenue
    FROM fact_sales f
    JOIN dim_customer c ON f.customer_id = c.customer_id
    JOIN dim_product p ON f.product_id = p.product_id
    WHERE {where_sql}
    GROUP BY 1,2,3
    ORDER BY revenue DESC;
"""
df_rev = con.execute(sql_rev, params).fetchdf()
st.dataframe(df_rev, use_container_width=True)

st.subheader("Daily revenue trend")
sql_daily = f"""
    SELECT
        f.order_date AS date,
        SUM(f.quantity * f.unit_price) AS revenue
    FROM fact_sales f
    JOIN dim_customer c ON f.customer_id = c.customer_id
    JOIN dim_product p ON f.product_id = p.product_id
    WHERE {where_sql}
    GROUP BY 1
    ORDER BY 1;
"""
df_daily = con.execute(sql_daily, params).fetchdf()
st.line_chart(df_daily.set_index("date")["revenue"])

st.subheader("Top products")
sql_top = f"""
    SELECT
        p.product_name,
        p.category,
        SUM(f.quantity) AS units,
        SUM(f.quantity * f.unit_price) AS revenue
    FROM fact_sales f
    JOIN dim_customer c ON f.customer_id = c.customer_id
    JOIN dim_product p ON f.product_id = p.product_id
    WHERE {where_sql}
    GROUP BY 1,2
    ORDER BY revenue DESC
    LIMIT 15;
"""
df_top = con.execute(sql_top, params).fetchdf()
st.dataframe(df_top, use_container_width=True)

con.close()
