"""The four-stage tailoring pipeline.

    Stage 1  EXTRACT   posting text -> JobRequirements        (structured output)
    Stage 2  EVIDENCE  agentic loop with a search tool        (tool use / RAG)
    Stage 3  DRAFT     evidence -> TailoredDraft              (structured output)
    Stage 4  GATE      human approves or rejects              (HITL)

Design notes worth defending in an interview:

- Stages 1 and 3 are *constrained* LLM calls: `messages.parse` guarantees the
  shape, so downstream code never regex-parses model prose.
- Stage 2 is the only open-ended agentic part — Claude decides what to search
  for and when it has enough evidence. Its tool results are accumulated into
  an evidence pack that Stage 3 must cite from.
- Grounding rule: Stage 3 may only cite evidence IDs that Stage 2 actually
  retrieved. `_validate_grounding` enforces this in code, not in the prompt —
  prompts request behavior, validators guarantee it.
- Stage 4: nothing touches disk until a human approves. The pipeline takes an
  `approve` callback so the gate works identically from CLI, tests, or a
  future web UI.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

from job_agent.llm import DEFAULT_MODEL, get_client
from job_agent.retrieval import Bm25Index, InventoryEntry, SearchHit
from job_agent.schemas import JobRequirements, TailoredDraft

MAX_EVIDENCE_ITERATIONS = 8

EXTRACT_SYSTEM = """You extract structured requirements from job postings.
Be faithful to the posting: do not invent requirements it doesn't state.
Condense boilerplate ("fast-paced environment") into red_flags only when it
hides real information. Seniority: infer from years/scope, else 'unknown'."""

EVIDENCE_SYSTEM = """You are gathering evidence from a candidate's experience
inventory to support a job application.

You have one tool: search_experience(query). It runs BM25 retrieval over the
candidate's canonical experience entries and returns entries with IDs.

Process:
- For each requirement in the job spec, run 1-2 targeted searches using the
  requirement's keywords. Prefer specific technical queries ("airflow dag
  scheduling") over broad ones ("data engineering").
- You do NOT need to search for every nice_to_have; prioritize must_haves.
- Stop when you've covered the must_haves — do not pad with extra searches.

When you are done, reply with a short plain-text summary of which requirements
appear supported and which look like gaps. Do not draft bullets yet."""

DRAFT_SYSTEM = """You draft a tailored job application from retrieved evidence.

HARD RULES:
- Every bullet MUST cite evidence_ids from the provided evidence pack. If no
  evidence supports a requirement, it goes in `gaps` — never invent experience.
- Bullets: <= 30 words, start with a strong verb, include concrete outcomes or
  metrics ONLY if they appear in the cited evidence.
- fit_score is judged against must_haves only; be honest, not generous. A 5/10
  with clear gaps is more useful than a flattering 8.
- cover_note: skip it entirely (null) if fit_score < 5."""


class PipelineError(Exception):
    pass


class GroundingError(PipelineError):
    """Raised when the draft cites evidence that was never retrieved."""


@dataclass
class EvidencePack:
    """Everything Stage 2 retrieved, keyed by entry id."""

    entries: Dict[str, InventoryEntry] = field(default_factory=dict)
    searches: List[str] = field(default_factory=list)
    summary: str = ""

    def add_hits(self, query: str, hits: List[SearchHit]) -> None:
        self.searches.append(query)
        for hit in hits:
            self.entries[hit.entry.entry_id] = hit.entry

    def as_prompt_block(self) -> str:
        if not self.entries:
            return "(no evidence retrieved)"
        lines = []
        for entry in self.entries.values():
            ctx = f" [{entry.context}]" if entry.context else ""
            lines.append(f"- id={entry.entry_id} ({entry.kind}){ctx}: {entry.text}")
        return "\n".join(lines)


@dataclass
class PipelineResult:
    requirements: JobRequirements
    evidence: EvidencePack
    draft: TailoredDraft
    approved: bool
    output_path: Optional[Path] = None


def _check_stop_reason(response, stage: str) -> None:
    """Turn truncation/refusal into actionable errors instead of generic ones."""
    if response.stop_reason == "max_tokens":
        raise PipelineError(
            f"{stage}: output hit the max_tokens cap and is incomplete. "
            "Shorten the posting or raise max_tokens."
        )
    if response.stop_reason == "refusal":
        raise PipelineError(f"{stage}: the model declined to process this content.")


def extract_requirements(posting_text: str, model: str = DEFAULT_MODEL) -> JobRequirements:
    """Stage 1: structured extraction."""
    client = get_client()
    response = client.messages.parse(
        model=model,
        max_tokens=4000,
        system=EXTRACT_SYSTEM,
        messages=[{"role": "user", "content": f"Job posting:\n\n{posting_text}"}],
        output_format=JobRequirements,
    )
    _check_stop_reason(response, "Extraction")
    parsed = response.parsed_output
    if parsed is None:
        raise PipelineError("Extraction returned no parseable output.")
    return parsed


def gather_evidence(
    requirements: JobRequirements,
    index: Bm25Index,
    model: str = DEFAULT_MODEL,
    on_search: Optional[Callable[[str, int], None]] = None,
) -> EvidencePack:
    """Stage 2: agentic loop — Claude drives BM25 searches via tool use."""
    client = get_client()
    pack = EvidencePack()

    search_tool = {
        "name": "search_experience",
        "description": (
            "Search the candidate's experience inventory (BM25). Returns matching "
            "entries with their ids, text, and context. Use short, specific queries."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search terms, e.g. 'fastapi rest deployment'."}
            },
            "required": ["query"],
        },
    }

    messages: List[dict] = [
        {
            "role": "user",
            "content": (
                "Gather evidence for this job spec:\n\n"
                + requirements.model_dump_json(indent=2)
            ),
        }
    ]

    for _ in range(MAX_EVIDENCE_ITERATIONS):
        response = client.messages.create(
            model=model,
            max_tokens=4000,
            system=EVIDENCE_SYSTEM,
            tools=[search_tool],
            messages=messages,
            # Top-level auto-caching: each iteration re-reads the growing
            # prefix (system + tools + prior turns) from cache instead of
            # re-billing it at full price.
            cache_control={"type": "ephemeral"},
        )

        if response.stop_reason != "tool_use":
            pack.summary = next(
                (b.text for b in response.content if b.type == "text"), ""
            )
            if response.stop_reason == "max_tokens":
                pack.summary += "\n(note: evidence summary was truncated at max_tokens)"
            elif response.stop_reason == "refusal":
                raise PipelineError("Evidence stage: the model declined to process this content.")
            return pack

        messages.append({"role": "assistant", "content": response.content})
        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            query = str(block.input.get("query", "")).strip()
            if query:
                hits = index.search(query, top_k=4)
                pack.add_hits(query, hits)
                if on_search:
                    on_search(query, len(hits))
            else:
                hits = []
            if hits:
                payload = "\n".join(
                    f"id={h.entry.entry_id} score={h.score:.2f} ({h.entry.kind}): {h.entry.text}"
                    for h in hits
                )
            else:
                payload = "No matches." if query else "Empty query — provide search terms."
            tool_results.append(
                {"type": "tool_result", "tool_use_id": block.id, "content": payload}
            )
        messages.append({"role": "user", "content": tool_results})

    # Iteration cap hit — keep whatever was gathered rather than failing the run.
    pack.summary = (
        "Evidence gathering stopped at the iteration cap; coverage of the "
        "requirements may be incomplete. Treat unsearched requirements as unknown, "
        "not as gaps."
    )
    return pack


def draft_application(
    requirements: JobRequirements,
    evidence: EvidencePack,
    model: str = DEFAULT_MODEL,
) -> TailoredDraft:
    """Stage 3: structured drafting, grounded in the evidence pack."""
    client = get_client()
    response = client.messages.parse(
        model=model,
        max_tokens=6000,
        system=DRAFT_SYSTEM,
        messages=[
            {
                "role": "user",
                "content": (
                    "Job spec:\n" + requirements.model_dump_json(indent=2)
                    + "\n\nEvidence pack (the ONLY experience you may cite):\n"
                    + evidence.as_prompt_block()
                    + "\n\nEvidence-gathering notes:\n" + (evidence.summary or "(none)")
                ),
            }
        ],
        output_format=TailoredDraft,
    )
    _check_stop_reason(response, "Drafting")
    draft = response.parsed_output
    if draft is None:
        raise PipelineError("Drafting returned no parseable output.")
    _validate_grounding(draft, evidence)
    return draft


def _validate_grounding(draft: TailoredDraft, evidence: EvidencePack) -> None:
    """Code-level enforcement of the no-fabrication rule.

    Covers both the bullets AND the requirement-coverage matches — everything
    that renders in the review screen or persists to disk must cite only
    evidence that retrieval actually returned.
    """
    known = set(evidence.entries.keys())
    for bullet in draft.bullets:
        if not bullet.evidence_ids:
            raise GroundingError(f"Bullet cites no evidence: {bullet.text!r}")
        unknown = [eid for eid in bullet.evidence_ids if eid not in known]
        if unknown:
            raise GroundingError(
                f"Bullet cites evidence ids that were never retrieved: {unknown} "
                f"(bullet: {bullet.text!r})"
            )
    for match in draft.matches:
        # Empty ids are legitimate here (strength='none'); unknown ids are not.
        unknown = [eid for eid in match.evidence_ids if eid not in known]
        if unknown:
            raise GroundingError(
                f"Match cites evidence ids that were never retrieved: {unknown} "
                f"(requirement: {match.requirement!r})"
            )


def save_draft(
    result_dir: Path,
    requirements: JobRequirements,
    evidence: EvidencePack,
    draft: TailoredDraft,
    source: str,
) -> Path:
    """Write the approved draft as markdown + a JSON sidecar. Only called post-approval."""
    result_dir.mkdir(parents=True, exist_ok=True)
    # Microseconds close the same-second collision window between processes
    # (e.g. CLI and web UI approving simultaneously); the probe loop below
    # remains as a second line of defense.
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S_%f")
    safe_company = "".join(c if c.isalnum() or c in "-_" else "_" for c in requirements.company)[:40]
    base = result_dir / f"{stamp}_{safe_company or 'unknown'}"
    # Same-second saves must never clobber an earlier draft.
    candidate, n = base, 1
    while candidate.with_suffix(".md").exists() or candidate.with_suffix(".json").exists():
        candidate = result_dir / f"{base.name}_{n}"
        n += 1
    base = candidate

    md_lines = [
        f"# {requirements.role} — {requirements.company}",
        f"\nSource: {source}",
        f"\n**Fit: {draft.fit_score}/10** — {draft.fit_rationale}",
        "\n## Tailored bullets\n",
    ]
    for bullet in draft.bullets:
        md_lines.append(f"- {bullet.text}")
        md_lines.append(
            f"  - _targets: {bullet.targets_requirement} | evidence: {', '.join(bullet.evidence_ids)}_"
        )
    if draft.gaps:
        md_lines.append("\n## Honest gaps\n")
        md_lines.extend(f"- {gap}" for gap in draft.gaps)
    if draft.cover_note:
        md_lines.append("\n## Cover note\n")
        md_lines.append(draft.cover_note)
    if requirements.red_flags:
        md_lines.append("\n## Posting red flags\n")
        md_lines.extend(f"- {flag}" for flag in requirements.red_flags)

    md_path = base.with_suffix(".md")
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    sidecar = {
        "source": source,
        "requirements": requirements.model_dump(),
        "draft": draft.model_dump(),
        "evidence_searches": evidence.searches,
        "evidence_ids_retrieved": sorted(evidence.entries.keys()),
    }
    base.with_suffix(".json").write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    return md_path


def run_pipeline(
    posting_text: str,
    source: str,
    index: Bm25Index,
    approve: Callable[[JobRequirements, TailoredDraft], bool],
    output_dir: Path,
    model: str = DEFAULT_MODEL,
    on_stage: Optional[Callable[[str], None]] = None,
    on_search: Optional[Callable[[str, int], None]] = None,
) -> PipelineResult:
    """Run all four stages. `approve` is the human-in-the-loop gate."""

    def stage(name: str) -> None:
        if on_stage:
            on_stage(name)

    stage("Extracting requirements")
    requirements = extract_requirements(posting_text, model=model)

    stage("Gathering evidence (agentic)")
    evidence = gather_evidence(requirements, index, model=model, on_search=on_search)

    stage("Drafting application")
    draft = draft_application(requirements, evidence, model=model)

    stage("Awaiting human approval")
    approved = approve(requirements, draft)

    output_path = None
    if approved:
        output_path = save_draft(output_dir, requirements, evidence, draft, source)

    return PipelineResult(
        requirements=requirements,
        evidence=evidence,
        draft=draft,
        approved=approved,
        output_path=output_path,
    )
