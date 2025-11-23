import duckdb
import pandas as pd
from datetime import datetime

# Path to locate DuckDB file
DB_PATH = "mini_data_mart.duckdb"

# 1 Connect to DuckDB
con = duckdb.connect(DB_PATH)

# 2 Example: Create dimension tables
con.execute("""
            CREATE TABLE IF NOT EXISTS dim_customer AS
            SELECT *
            FROM (VALUES (1, 'Alice', 'North'),
                         (2, 'Bob', 'South'),
                         (3, 'Charlie', 'East')) AS t(customer_id, customer_name, region);
            """)

con.execute("""
            CREATE TABLE IF NOT EXISTS dim_product AS
            SELECT *
            FROM (VALUES (10, 'Laptop', 'Electronics'),
                         (11, 'Headphones', 'Electronics'),
                         (12, 'Desk Chair', 'Furniture')) AS t(product_id, product_name, category);
            """)

con.execute("""
            CREATE TABLE IF NOT EXISTS dim_date AS
            SELECT *
            FROM (VALUES ('2025-11-20', 2025, 11, 'Thursday'),
                         ('2025-11-21', 2025, 11, 'Friday'),
                         ('2025-11-22', 2025, 11, 'Saturday')) AS t(date, year, date_of_week);
            """)

# 3 Creat fact table and insert some sample data
con.execute("""
            CREATE TABLE IF NOT EXISTS fact_sales
            (
                sale_id
                INTEGER,
                customer_id
                INTEGER,
                product_id
                INTEGER,
                order_date
                DATE,
                quantity
                INTEGER,
                unit_price
                DOUBLE
            );
            """)

# 4 Example analytics query: total revenue by customer and category
query = """
        SELECT c.customer_name,
               p.category,
               SUM(f.quantity * f.unit_price) AS total_revenue
        FROM fact_sales f
                 JOIN dim_customer c ON c.customer_id = c.customer_id
                 JOIN dim_product p ON f.product_id = p.product_id
        GROUP BY c.customer_name, p.category
        ORDER BY total_revenue DESC;
        """
df_result = con.execute(query).fetchdf()
print("Revenue by customer and category:")
print(df_result)

# 5 Another example: revenue by day_of_week
query2 = """
         SELECT d.day_of_week,
                SUM(f.quantity * f.unit_price) AS total_revenue
         FROM fact_sales f
                  JOIN dim_date d ON f.order_date = d.date
         GROUP BY d.day_of_week
         ORDER BY total_revenue DESC;
         """
df_result2 = con.execute(query2).fetchdf()
print("\nRevenue by day of week:")
print(df_result2)

con.close()
