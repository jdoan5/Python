from __future__ import annotations

import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

import httpx

from agent_core.tools import Tool, ToolRegistry

ARXIV_API = "http://export.arxiv.org/api/query"
ATOM_NS = "{http://www.w3.org/2005/Atom}"
ARXIV_NS = "{http://arxiv.org/schemas/atom}"

VALID_CATEGORIES = {
    "cs.AI", "cs.LG", "cs.CL", "cs.CV", "cs.RO", "cs.NE", "cs.IR", "cs.SE",
    "cs.DB", "cs.DS", "cs.CR", "cs.HC", "cs.MA", "stat.ML", "math.OC",
}

MAX_RESULTS_CAP = 25


def _parse_arxiv_feed(xml_text: str) -> list[dict[str, str]]:
    root = ET.fromstring(xml_text)
    papers = []
    for entry in root.findall(f"{ATOM_NS}entry"):
        title = (entry.findtext(f"{ATOM_NS}title") or "").strip().replace("\n ", " ")
        summary = (entry.findtext(f"{ATOM_NS}summary") or "").strip().replace("\n", " ")
        published = entry.findtext(f"{ATOM_NS}published") or ""
        arxiv_id = (entry.findtext(f"{ATOM_NS}id") or "").rsplit("/", 1)[-1]
        authors = [
            (a.findtext(f"{ATOM_NS}name") or "").strip()
            for a in entry.findall(f"{ATOM_NS}author")
        ]
        categories = [
            c.attrib.get("term", "")
            for c in entry.findall(f"{ATOM_NS}category")
        ]
        papers.append(
            {
                "id": arxiv_id,
                "title": title,
                "authors": ", ".join(authors[:5]) + (" et al." if len(authors) > 5 else ""),
                "published": published[:10],
                "categories": ", ".join(categories[:5]),
                "abstract": summary,
            }
        )
    return papers


def build_tools(digest_dir: Path) -> ToolRegistry:
    digest_dir.mkdir(parents=True, exist_ok=True)
    registry = ToolRegistry()

    def fetch_arxiv_recent(category: str, max_results: int = 10) -> str:
        if category not in VALID_CATEGORIES:
            raise ValueError(
                f"Unknown category '{category}'. Try one of: "
                + ", ".join(sorted(VALID_CATEGORIES))
            )
        if not 1 <= max_results <= MAX_RESULTS_CAP:
            raise ValueError(f"max_results must be 1-{MAX_RESULTS_CAP}, got {max_results}")

        params = {
            "search_query": f"cat:{category}",
            "max_results": str(max_results),
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
        try:
            with httpx.Client(timeout=15.0) as client:
                response = client.get(ARXIV_API, params=params)
                response.raise_for_status()
        except httpx.HTTPError as e:
            return f"arXiv fetch failed: {type(e).__name__}: {e}"

        papers = _parse_arxiv_feed(response.text)
        if not papers:
            return f"No papers found for category {category}."

        lines = []
        for i, p in enumerate(papers, 1):
            lines.append(
                f"[{i}] {p['id']} | {p['published']} | {p['categories']}\n"
                f"    Title: {p['title']}\n"
                f"    Authors: {p['authors']}\n"
                f"    Abstract: {p['abstract'][:600]}{'...' if len(p['abstract']) > 600 else ''}\n"
            )
        return "\n".join(lines)

    registry.register(
        Tool(
            name="fetch_arxiv_recent",
            description=(
                "Fetch the most recent submissions from an arXiv category. "
                "category must be one of: cs.AI, cs.LG, cs.CL, cs.CV, cs.RO, cs.NE, cs.IR, "
                "cs.SE, cs.DB, cs.DS, cs.CR, cs.HC, cs.MA, stat.ML, math.OC. "
                f"max_results 1-{MAX_RESULTS_CAP} (default 10)."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "category": {"type": "string", "description": "arXiv category code."},
                    "max_results": {
                        "type": "integer",
                        "description": f"Number of papers, 1-{MAX_RESULTS_CAP}.",
                    },
                },
                "required": ["category"],
            },
            fn=fetch_arxiv_recent,
        )
    )

    def save_digest(category: str, content: str) -> str:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
        safe_cat = category.replace("/", "-").replace(".", "-")
        path = digest_dir / f"digest_{safe_cat}_{timestamp}.md"
        path.write_text(content, encoding="utf-8")
        return f"Saved digest to {path}"

    registry.register(
        Tool(
            name="save_digest",
            description="Save the final digest as a markdown file under the digest directory.",
            input_schema={
                "type": "object",
                "properties": {
                    "category": {"type": "string", "description": "arXiv category for filename."},
                    "content": {"type": "string", "description": "Full markdown content."},
                },
                "required": ["category", "content"],
            },
            fn=save_digest,
        )
    )

    def list_digests() -> str:
        files = sorted(digest_dir.glob("digest_*.md"), reverse=True)[:20]
        if not files:
            return "No digests saved yet."
        return "\n".join(f"{f.name} ({f.stat().st_size} bytes)" for f in files)

    registry.register(
        Tool(
            name="list_digests",
            description="List recent saved digests in the digest directory.",
            input_schema={"type": "object", "properties": {}, "required": []},
            fn=list_digests,
        )
    )

    return registry
