from __future__ import annotations
from pathlib import Path

from toolkit.core.csv_tools import summarize_csv


def test_summarize_csv_counts_rows_and_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text("a,b,c\n1,2,3\n4,5,6\n", encoding="utf-8")

    summary = summarize_csv(csv_path)

    assert summary["columns"] == ["a", "b", "c"]
    assert summary["num_rows"] == 2
    assert len(summary["preview_rows"]) == 2