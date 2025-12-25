from __future__ import annotations
from pathlib import Path
from collections import Counter
import re
from typing import Dict, Any, List, Tuple

_WORD_RE = re.compile(r"[\w']+")


def summarize_text(path: Path, top_n: int = 5) -> Dict[str, Any]:
    """
    Summarize a text file: counts and common words.

    Returns a dict:
        {
            "path": str,
            "num_lines": int,
            "num_chars": int,
            "num_words": int,
            "top_words": [(word, count), ...],
        }
    """
    if not path.is_file():
        raise FileNotFoundError(f"Text file not found: {path}")

    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    words: List[str] = _WORD_RE.findall(text.lower())

    counts = Counter(words)
    top_words: List[Tuple[str, int]] = counts.most_common(top_n)

    return {
        "path": str(path),
        "num_lines": len(lines),
        "num_chars": len(text),
        "num_words": len(words),
        "top_words": top_words,
    }