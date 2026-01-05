from __future__ import annotations

from typing import Iterable

from .config import MartConfig, connect, ensure_dirs

_TABLES: tuple[str, ...] = ("dim_customer", "dim_product", "dim_date", "fact_sales")


def _q(path: str) -> str:
    return path.replace("'", "''")


def export_csv(cfg: MartConfig, tables: Iterable[str] = _TABLES) -> None:
    ensure_dirs(cfg)
    con = connect(cfg)
    try:
        print("\n=== Exporting tables to CSV ===")
        for t in tables:
            out = (cfg.OUTPUT_CSV_DIR / f"{t}.csv").as_posix()
            print(f"Exporting {t} -> {out}")
            con.execute(f"COPY (SELECT * FROM {t}) TO '{_q(out)}' (HEADER, DELIMITER ',');")
    finally:
        con.close()


def export_json(cfg: MartConfig, tables: Iterable[str] = _TABLES) -> None:
    ensure_dirs(cfg)
    con = connect(cfg)
    try:
        print("\n=== Exporting tables to JSON ===")
        for t in tables:
            out = (cfg.OUTPUT_JSON_DIR / f"{t}.json").as_posix()
            print(f"Exporting {t} -> {out}")
            con.execute(f"COPY (SELECT * FROM {t}) TO '{_q(out)}' (FORMAT JSON);")
    finally:
        con.close()


def export_all(cfg: MartConfig) -> None:
    export_csv(cfg)
    export_json(cfg)
