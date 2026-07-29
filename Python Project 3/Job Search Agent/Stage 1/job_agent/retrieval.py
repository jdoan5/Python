"""Retrieval over the experience inventory — BM25 implemented from scratch.

The "RAG" in this project: the drafting stage may only cite experience that
retrieval actually surfaced, which is what keeps bullets grounded instead of
hallucinated. BM25 (Okapi) is implemented in ~60 lines of stdlib Python rather
than pulled in as a dependency — lexical retrieval is a perfectly honest
baseline for a corpus of this size, and writing it yourself is the point.

Upgrade path (documented in README): swap `Bm25Index` for embeddings
(e.g. Voyage AI) behind the same `search()` signature.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import yaml

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9+#.\-]*")

# Okapi BM25 constants (standard defaults).
K1 = 1.5
B = 0.75


class InventoryError(Exception):
    """Raised when the experience inventory is missing or malformed."""


def tokenize(text: str) -> List[str]:
    """Lowercase word tokens.

    Keeps '+'/'#' so 'c++' and 'c#' survive; strips trailing '.'/'-' so a
    sentence-final 'Python.' matches the query 'python'. Leading punctuation
    is never captured ('.NET' -> 'net'), which is consistent between indexed
    text and queries, so matching still works.
    """
    return [t.rstrip(".-") for t in _TOKEN_RE.findall(text.lower())]


@dataclass
class InventoryEntry:
    entry_id: str
    kind: str          # 'achievement' | 'skill' | 'project' | 'education'
    text: str          # the canonical sentence(s) describing the experience
    tags: List[str] = field(default_factory=list)
    context: str = ""  # role/company/date line, e.g. "Data Analyst @ Acme, 2022-2024"

    def searchable_text(self) -> str:
        return " ".join([self.text, self.context, " ".join(self.tags)])


@dataclass
class SearchHit:
    entry: InventoryEntry
    score: float


class Bm25Index:
    """Okapi BM25 over inventory entries. Build once, search many times."""

    def __init__(self, entries: List[InventoryEntry]) -> None:
        if not entries:
            raise InventoryError("Inventory is empty — nothing to index.")
        self.entries = entries
        self._doc_tokens: List[List[str]] = [
            tokenize(e.searchable_text()) for e in entries
        ]
        self._doc_freqs: List[Counter] = [Counter(toks) for toks in self._doc_tokens]
        self._doc_lens = [len(toks) for toks in self._doc_tokens]
        self._avg_len = sum(self._doc_lens) / len(self._doc_lens)
        # Document frequency per term.
        df: Counter = Counter()
        for toks in self._doc_tokens:
            for term in set(toks):
                df[term] += 1
        n = len(entries)
        # BM25+-style floor avoids negative IDF for very common terms.
        self._idf: Dict[str, float] = {
            term: max(0.01, math.log((n - dfi + 0.5) / (dfi + 0.5) + 1.0))
            for term, dfi in df.items()
        }

    def search(self, query: str, top_k: int = 5) -> List[SearchHit]:
        q_terms = tokenize(query)
        if not q_terms:
            return []
        scores = []
        for i, entry in enumerate(self.entries):
            freqs = self._doc_freqs[i]
            doc_len = self._doc_lens[i]
            score = 0.0
            for term in q_terms:
                tf = freqs.get(term, 0)
                if tf == 0:
                    continue
                idf = self._idf.get(term, 0.0)
                denom = tf + K1 * (1 - B + B * doc_len / self._avg_len)
                score += idf * (tf * (K1 + 1)) / denom
            if score > 0:
                scores.append(SearchHit(entry=entry, score=score))
        scores.sort(key=lambda h: h.score, reverse=True)
        return scores[:top_k]


def load_inventory(path: Path) -> List[InventoryEntry]:
    """Load and validate the YAML experience inventory."""
    if not path.is_file():
        raise InventoryError(
            f"Inventory not found (or not a file): {path}\n"
            "Copy data/experience_inventory.yaml, fill in YOUR real experience, "
            "and point --inventory at it."
        )
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (yaml.YAMLError, OSError) as e:
        raise InventoryError(f"Cannot read inventory {path}: {e}") from e

    if not isinstance(raw, dict) or not isinstance(raw.get("entries"), list):
        raise InventoryError(f"{path} must have a top-level 'entries:' list.")

    entries = []
    seen_ids = set()
    for i, item in enumerate(raw["entries"]):
        if not isinstance(item, dict):
            raise InventoryError(f"Entry #{i + 1} is not a mapping.")
        entry_id = str(item.get("id", "")).strip()
        text = str(item.get("text", "")).strip()
        if not entry_id or not text:
            raise InventoryError(f"Entry #{i + 1} needs both 'id' and 'text'.")
        if entry_id in seen_ids:
            raise InventoryError(f"Duplicate entry id: {entry_id}")
        seen_ids.add(entry_id)
        entries.append(
            InventoryEntry(
                entry_id=entry_id,
                kind=str(item.get("kind", "achievement")),
                text=text,
                tags=[str(t) for t in (item.get("tags") or [])],
                context=str(item.get("context") or ""),
            )
        )
    if not entries:
        raise InventoryError(f"{path} has an empty 'entries:' list — add your experience.")
    return entries
