from __future__ import annotations

import csv
import os
from typing import List, Tuple


def load_currencies(csv_path: str) -> List[Tuple[str, str]]:
    """
    Loads currencies from a CSV with headers:
      currency_name,currency_code,numeric_code

    Returns: list of (currency_name, currency_code)
    """
    if not os.path.isabs(csv_path):
        # Resolve relative to the project folder (where this file lives)
        base_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(base_dir, csv_path)

    currencies: List[Tuple[str, str]] = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = (row.get("currency_name") or "").strip()
            code = (row.get("currency_code") or "").strip().upper()
            if name and code:
                currencies.append((name, code))

    # Sort by name for nicer UX
    currencies.sort(key=lambda x: x[0].lower())
    return currencies