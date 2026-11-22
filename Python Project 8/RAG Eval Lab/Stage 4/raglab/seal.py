"""Sealed stores: what gets committed, with no documentation text in it (PLAN.md, "Reproducibility").

The live store keeps everything the API returned. The sealed copy committed at
freeze points 3 and 5 keeps an allowlist of fields, and replaces every piece of
documentation text with something that can be checked against the pinned
corpus but not read from the repository:
- a citation's `cited_text` becomes the SHA-256 and length of its text with
  whitespace removed (block joins vary, so whitespace never decides a match);
  `document_title` is dropped (the window rebuilds it);
- a judge's quote that holds becomes its offset, length and SHA-256 in the
  whitespace-collapsed source it quotes (the excerpt, or the reference); a
  quote that doesn't hold keeps only its length and SHA-256, since it can be
  near-verbatim documentation too;
- a thinking block's `signature` becomes its SHA-256.
The answers' own text is model output and is committed. The sealed file is
canonical JSONL, gzipped with mtime 0; its SHA-256 is that of the decompressed
bytes, so another zlib can't change it.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
from pathlib import Path
from typing import Callable, Iterable, Mapping

from raglab.blocks import squash
from raglab.store import Record, canonical

RECORD_FIELDS = ("custom_id", "request_key", "condition", "body_sha256", "phase", "outcome", "response", "error",
                 "batch_id", "request_id", "received_at", "group", "timing")
MESSAGE_FIELDS = ("id", "type", "role", "model", "content", "stop_reason", "stop_sequence", "stop_details", "usage")


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def collapse(text: str) -> str:
    """Whitespace runs as single spaces: the form quotes are matched in (quote_holds)."""
    return " ".join(text.split())


def seal_quote(quote: str, source: str) -> dict:
    """An offset into the collapsed source if the quote holds there; otherwise the judge's own text."""
    q, s = collapse(quote), collapse(source)
    at = s.find(q) if q else -1
    if at < 0:
        return {"held": False, "len": len(q), "sha256": sha(q)}
    return {"held": True, "at": at, "len": len(q), "sha256": sha(q)}


def unseal_quote(sealed: Mapping, source: str) -> str | None:
    """The quote's collapsed text, rebuilt from its source and checked against its hash.

    A quote that didn't hold isn't in any source, so its text can't be rebuilt: None.
    """
    if not sealed["held"]:
        return None
    text = collapse(source)[sealed["at"]: sealed["at"] + sealed["len"]]
    if sha(text) != sealed["sha256"]:
        raise ValueError("a sealed quote doesn't match its source: the corpus or the rebuild changed")
    return text


def seal_message(message: Mapping) -> dict:
    """A generation message with its citations' documentation text and thinking signatures replaced by hashes."""
    out = {k: message[k] for k in MESSAGE_FIELDS if k in message}
    content = []
    for block in message.get("content") or ():
        block = dict(block)
        if block.get("type") == "thinking" and "signature" in block:
            block["signature_sha256"] = sha(block.pop("signature") or "")
        elif block.get("type") == "redacted_thinking" and "data" in block:
            block["data_sha256"] = sha(block.pop("data") or "")
        elif block.get("type") == "text" and block.get("citations"):
            cites = []
            for c in block["citations"]:
                c = dict(c)
                text = c.pop("cited_text", "") or ""
                c.pop("document_title", None)
                c["cited_squash_sha256"], c["cited_squash_len"] = sha(squash(text)), len(squash(text))
                cites.append(c)
            block["citations"] = cites
        content.append(block)
    out["content"] = content
    return out


def check_unsealed_citation(sealed: Mapping, blocks_text: str) -> bool:
    """Does a sealed citation match the text of the blocks it names, rebuilt from the corpus?"""
    return sha(squash(blocks_text)) == sealed["cited_squash_sha256"] and len(squash(blocks_text)) == \
        sealed["cited_squash_len"]


def seal_record(rec: Record, seal_response: Callable[[Mapping], dict]) -> dict:
    data = rec.to_json()
    out = {k: data[k] for k in RECORD_FIELDS if k in data}
    if out.get("response") is not None:
        out["response"] = seal_response(out["response"])
    return out


def write_sealed(records: Iterable[Mapping], path: Path) -> dict:
    """Canonical JSONL, gzipped with mtime 0. Returns what SEALED.json records."""
    rows = sorted(records, key=lambda r: r["custom_id"])
    raw = b"".join(canonical(r) + b"\n" for r in rows)
    buf = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buf, mtime=0, compresslevel=9) as gz:
        gz.write(raw)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(buf.getvalue())
    return {"file": path.name, "records": len(rows), "jsonl_sha256": hashlib.sha256(raw).hexdigest()}


def read_sealed(path: Path, expected: Mapping) -> list[dict]:
    raw = gzip.decompress(path.read_bytes())
    if hashlib.sha256(raw).hexdigest() != expected["jsonl_sha256"]:
        raise ValueError(f"{path.name} doesn't match SEALED.json")
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if len(rows) != expected["records"]:
        raise ValueError(f"{path.name} holds {len(rows)} records, SEALED.json says {expected['records']}")
    return rows


def unlisted_files(folder: Path, listed: Iterable[str]) -> list[str]:
    """Files in a sealed folder that SEALED.json doesn't name, such as iCloud's ' 2' copies."""
    names = set(listed) | {"SEALED.json"}
    return sorted(p.name for p in folder.iterdir() if p.is_file() and p.name not in names)
