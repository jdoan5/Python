# ui_desktop/app_tk.py
from __future__ import annotations

import tkinter as tk
from tkinter import ttk, messagebox
from datetime import date

import duckdb

from mart.config import load_config


def _safe_fetchall(con, sql: str, params=None):
    try:
        if params:
            return con.execute(sql, params).fetchall()
        return con.execute(sql).fetchall()
    except Exception as e:
        raise RuntimeError(str(e))


class MiniMartApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Mini Data Mart (DuckDB) — Desktop")
        self.geometry("1000x650")

        self.cfg = load_config()
        self.con = duckdb.connect(self.cfg.DB_PATH.as_posix())

        self._build_layout()
        self._load_filters()

    def _build_layout(self):
        # Top controls
        top = ttk.Frame(self, padding=10)
        top.pack(fill="x")

        ttk.Label(top, text="Customer").grid(row=0, column=0, sticky="w")
        self.customer_cb = ttk.Combobox(top, width=30, state="readonly")
        self.customer_cb.grid(row=1, column=0, padx=(0, 12), sticky="w")

        ttk.Label(top, text="Region").grid(row=0, column=1, sticky="w")
        self.region_cb = ttk.Combobox(top, width=18, state="readonly")
        self.region_cb.grid(row=1, column=1, padx=(0, 12), sticky="w")

        ttk.Label(top, text="Category").grid(row=0, column=2, sticky="w")
        self.category_cb = ttk.Combobox(top, width=18, state="readonly")
        self.category_cb.grid(row=1, column=2, padx=(0, 12), sticky="w")

        ttk.Label(top, text="Start date (YYYY-MM-DD)").grid(row=0, column=3, sticky="w")
        self.start_entry = ttk.Entry(top, width=16)
        self.start_entry.grid(row=1, column=3, padx=(0, 12), sticky="w")

        ttk.Label(top, text="End date (YYYY-MM-DD)").grid(row=0, column=4, sticky="w")
        self.end_entry = ttk.Entry(top, width=16)
        self.end_entry.grid(row=1, column=4, padx=(0, 12), sticky="w")

        ttk.Button(top, text="Run query", command=self.run_query).grid(row=1, column=5, sticky="w")

        # Defaults
        self.start_entry.insert(0, f"{date.today().year}-01-01")
        self.end_entry.insert(0, f"{date.today().year}-12-31")

        # Results table
        mid = ttk.Frame(self, padding=(10, 0, 10, 10))
        mid.pack(fill="both", expand=True)

        self.tree = ttk.Treeview(mid, columns=("customer", "category", "revenue"), show="headings")
        self.tree.heading("customer", text="Customer")
        self.tree.heading("category", text="Category")
        self.tree.heading("revenue", text="Revenue")

        self.tree.column("customer", width=280, anchor="w")
        self.tree.column("category", width=180, anchor="w")
        self.tree.column("revenue", width=140, anchor="e")

        vsb = ttk.Scrollbar(mid, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)

        self.tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        # Bottom status
        self.status = tk.StringVar(value=f"DB: {self.cfg.DB_PATH.name}")
        ttk.Label(self, textvariable=self.status, padding=10).pack(fill="x")

    def _load_filters(self):
        # Make sure tables exist
        tables = [r[0] for r in _safe_fetchall(self.con, "SHOW TABLES;")]
        if "fact_sales" not in tables:
            messagebox.showwarning(
                "Mart not built",
                "DuckDB tables not found. Run `python -m mart build` first."
            )

        # Load filter values (safe even if empty)
        customers = _safe_fetchall(self.con, "SELECT DISTINCT customer_name FROM dim_customer ORDER BY 1;")
        regions = _safe_fetchall(self.con, "SELECT DISTINCT region FROM dim_customer ORDER BY 1;")
        categories = _safe_fetchall(self.con, "SELECT DISTINCT category FROM dim_product ORDER BY 1;")

        self.customer_cb["values"] = ["(All)"] + [r[0] for r in customers]
        self.region_cb["values"] = ["(All)"] + [r[0] for r in regions]
        self.category_cb["values"] = ["(All)"] + [r[0] for r in categories]

        self.customer_cb.set("(All)")
        self.region_cb.set("(All)")
        self.category_cb.set("(All)")

    def run_query(self):
        # Clear old rows
        for item in self.tree.get_children():
            self.tree.delete(item)

        customer = self.customer_cb.get()
        region = self.region_cb.get()
        category = self.category_cb.get()
        start = self.start_entry.get().strip()
        end = self.end_entry.get().strip()

        # Build WHERE dynamically with parameters
        where = ["d.date BETWEEN ? AND ?"]
        params = [start, end]

        if customer != "(All)":
            where.append("c.customer_name = ?")
            params.append(customer)
        if region != "(All)":
            where.append("c.region = ?")
            params.append(region)
        if category != "(All)":
            where.append("p.category = ?")
            params.append(category)

        sql = f"""
            SELECT
                c.customer_name AS customer,
                p.category AS category,
                ROUND(SUM(f.quantity * f.unit_price), 2) AS revenue
            FROM fact_sales f
            JOIN dim_customer c ON c.customer_id = f.customer_id
            JOIN dim_product p ON p.product_id = f.product_id
            JOIN dim_date d ON d.date = f.order_date
            WHERE {" AND ".join(where)}
            GROUP BY 1, 2
            ORDER BY revenue DESC;
        """

        try:
            rows = _safe_fetchall(self.con, sql, params)
        except Exception as e:
            messagebox.showerror("Query failed", str(e))
            return

        for customer_name, cat, rev in rows:
            self.tree.insert("", "end", values=(customer_name, cat, f"{rev:,.2f}"))

        self.status.set(f"Rows: {len(rows)} | DB: {self.cfg.DB_PATH.name}")

    def on_close(self):
        try:
            self.con.close()
        finally:
            self.destroy()


def main():
    app = MiniMartApp()
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()


if __name__ == "__main__":
    main()