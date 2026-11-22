"""The smoke test's checks (PLAN.md, "The pilot", "Smoke test").

Five answers go out first, one at a time on the standard API, under the pilot
authorization: an hhr, a baseline and a closed-book answer to one question, and
two very large baseline windows (about 300 documents each). Their two F calls
and one C call go first under the pilot-judging authorization, because a judge
request carries the answer it judges, which doesn't exist until the answer does.

What it records, with no documentation text in it:
- whether every citation maps exactly (`citation_intact`), and the join rule
  that rebuilds `cited_text` from the cited blocks character for character;
- the usage fields each response reports, so the ledger is known to price
  every one of them, and the inference geography and service tier;
- for judges, whether the structured output parsed;
- the ledger's settled cost, for the developer to check against the Console.
"""

from __future__ import annotations

from collections import Counter
from typing import Mapping, Sequence

from raglab.blocks import Citation, CitationError, WindowDoc, cited_blocks, citation_intact
from raglab.ledger import KNOWN_USAGE_FIELDS
from raglab.units import parse_reply

JOINS = {"concatenated": "", "newline": "\n", "blank line": "\n\n", "space": " "}


def join_rule(c: Citation, docs: Sequence[WindowDoc]) -> str:
    """How `cited_text` is built from the blocks it names: a JOINS name, the same with each block's trailing
    whitespace stripped (" (stripped)"), or "whitespace only" when only the squashed texts agree."""
    texts = [b.text for b in cited_blocks(c, docs)]
    for name, sep in JOINS.items():
        if sep.join(texts) == c.cited_text:
            return name
    for name, sep in JOINS.items():
        if sep.join(t.rstrip() for t in texts) == c.cited_text:
            return name + " (stripped)"
    return "whitespace only" if citation_intact(c, docs) else "no match"


def answer_check(message: Mapping, docs: Sequence[WindowDoc]) -> dict:
    """One smoke answer: its status, citation counts and failures, the join rules seen, and its usage fields.

    Only a citation of two or more blocks shows how blocks are joined; a one-block
    citation matches every rule, so it is counted apart and never decides it.
    """
    answer = parse_reply(message, docs)
    rules: Counter[str] = Counter()
    single = 0
    for block in message.get("content") or ():
        for raw in block.get("citations") or () if block.get("type") == "text" else ():
            try:
                c = Citation.from_api(raw)
                rule = join_rule(c, docs)
            except (CitationError, KeyError, TypeError, ValueError):
                rules["unmappable"] += 1
                continue
            if c.end_block - c.start_block >= 2 or rule in ("whitespace only", "no match"):
                rules[rule] += 1
            else:
                single += 1
    usage = message.get("usage") or {}
    return {
        "status": answer.status,
        "documents": len(docs),
        "citations": answer.citations_total,
        "citations_failed": answer.citations_failed,
        "failures": list(answer.failures),
        "join_rules": dict(sorted(rules.items())),
        "single_block_citations": single,
        "max_document_index_cited": max((c.document_index for u in answer.units for c in u.citations), default=None),
        "usage_fields": sorted(usage),
        "unpriced_usage_fields": sorted(set(usage) - KNOWN_USAGE_FIELDS),
        "inference_geo": usage.get("inference_geo"),
        "service_tier": usage.get("service_tier"),
        "model": message.get("model"),
        "stop_reason": message.get("stop_reason"),
    }


def verdict(checks: Sequence[Mapping], us_accepted: bool = False) -> dict:
    """The smoke test passes only if every citation maps and the usage is the ledger's to price. US-only
    inference fails it unless the developer has accepted 1.1x for the pilot (`us_accepted`); the join rule is
    recorded when the multi-block citations agree on one."""
    rules = Counter()
    for c in checks:
        rules.update(c["join_rules"])
    problems = []
    failed = sum(c["citations_failed"] for c in checks)
    if failed:
        problems.append(f"{failed} citation(s) failed their check: a mapping bug (PLAN.md, 'Citation integrity')")
    if not any(c["citations"] for c in checks if c["documents"]):
        problems.append("no answer with documents cited anything, so the mapping is untested")
    unpriced = sorted({f for c in checks for f in c["unpriced_usage_fields"]})
    if unpriced:
        problems.append(f"usage fields the ledger doesn't price: {', '.join(unpriced)}")
    geos = sorted({str(c["inference_geo"]) for c in checks})
    notes = []
    if "us" in geos and not us_accepted:
        problems.append("US-only inference (billed at 1.1x): accept it for the pilot with `budget acknowledge "
                        "--us-inference pilot --reason ...`, then run `gen smoke` again (the stored answers "
                        "stand), and set the workspace's default inference geography to global in the Console "
                        "for later phases")
    elif "us" in geos:
        notes.append("US-only inference accepted for the pilot: every pilot call costs 1.1x")
    exact = [r for r in rules if r not in ("whitespace only", "no match", "unmappable")]
    deepest = max((c["max_document_index_cited"] for c in checks if c["max_document_index_cited"] is not None),
                  default=None)
    return {"passed": not problems, "problems": problems, "notes": notes, "join_rules": dict(sorted(rules.items())),
            "join_rule": exact[0] if len(exact) == 1 else None, "inference_geo": geos,
            "single_block_citations": sum(c.get("single_block_citations", 0) for c in checks),
            "deepest_document_cited": deepest}
