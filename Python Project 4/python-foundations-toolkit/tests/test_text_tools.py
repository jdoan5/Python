from __future__ import annotations
from pathlib import Path

from toolkit.core.text_tools import summarize_text


def test_summarize_text_basic_counts(tmp_path: Path) -> None:
    txt_path = tmp_path / "sample.txt"
    txt_path.write_text("Hello world\nHello Python foundations\n", encoding="utf-8")

    summary = summarize_text(txt_path, top_n=3)

    assert summary["num_lines"] == 2
    assert summary["num_words"] >= 4
    # "hello" should be the most frequent word
    assert summary["top_words"][0][0] == "hello"