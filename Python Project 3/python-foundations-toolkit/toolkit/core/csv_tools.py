from __future__ import annotations
from pathlib import Path
import csv
from typing import Dict, Any


def summarize_csv(path: Path, delimiter: str = ",", max_preview_rows: int = 5) -> Dict[str, Any]:
    """
    Summarize a CSV file: columns, number of data rows, and a small preview.

    Returns a dict:
        {
            "path": str,
            "columns": [str, ...],
            "num_rows": int,                # excluding header
            "preview_rows": [ [..], ... ],  # up to max_preview_rows
        }
    """
    if not path.is_file():
        raise FileNotFoundError(f"CSV file not found: {path}")

    row_count = 0
    header = None
    preview_rows = []

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter=delimiter)
        for row_index, row in enumerate(reader):
            if row_index == 0:
                header = row
            else:
                row_count += 1
                if len(preview_rows) < max_preview_rows:
                    preview_rows.append(row)

    if header is None:
        # Empty file
        return {
            "path": str(path),
            "columns": [],
            "num_rows": 0,
            "preview_rows": [],
        }

    return {
        "path": str(path),
        "columns": header,
        "num_rows": row_count,
        "preview_rows": preview_rows,
    }