from __future__ import annotations

from mart.config import load_config
from mart.build import build
from mart.explore import explore
from mart.export import export_all
from mart.quality import run_checks


def main() -> int:
    cfg = load_config()
    build(cfg, reset=True)
    explore(cfg)
    export_all(cfg)
    run_checks(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
