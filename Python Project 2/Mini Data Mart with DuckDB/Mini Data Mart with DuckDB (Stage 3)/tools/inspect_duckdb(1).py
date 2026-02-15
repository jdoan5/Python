from __future__ import annotations

from mart.config import load_config, connect


def main() -> int:
    cfg = load_config()
    con = connect(cfg)
    try:
        print(f"DB: {cfg.DB_PATH}")
        print("\n== TABLES ==")
        try:
            print(con.execute("SHOW TABLES;").fetchdf())
        except Exception:
            print(con.execute("SHOW TABLES;").fetchall())

        for t in ("dim_customer", "dim_product", "dim_date", "fact_sales"):
            try:
                n = con.execute(f"SELECT COUNT(*) FROM {t};").fetchone()[0]
                print(f"{t}: {n} rows")
            except Exception as e:
                print(f"{t}: (missing) {e}")
        return 0
    finally:
        con.close()


if __name__ == "__main__":
    raise SystemExit(main())
