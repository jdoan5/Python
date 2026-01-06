from __future__ import annotations

from mart.config import load_config
from mart.export import export_csv

if __name__ == "__main__":
    export_csv(load_config())
