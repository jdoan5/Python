from __future__ import annotations

from mart.config import load_config
from mart.export import export_json

if __name__ == "__main__":
    export_json(load_config())
