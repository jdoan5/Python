# Python Project 6 — AI Agents

A portfolio of three Claude-powered agents built on a shared agent core.
Each agent demonstrates a different real-world use case: triaging GitHub
issues, drafting tailored job applications, and digesting recent research
papers.

## What this project shows

- **LLM tool use / function calling** — agents loop over the Anthropic API,
  reading and acting on real data via custom tools.
- **Production patterns** — prompt caching, adaptive thinking, structured
  logging, retries, eval-driven development.
- **Defense-in-depth tool design** — path traversal protection, file size
  caps, search result limits.
- **Eval harness** — LLM-as-judge scoring against per-case rubrics.

## Architecture

```
agent_core/                  shared agent runtime
  agent.py                   manual agentic loop (tool use → results → repeat)
  tools.py                   Tool / ToolRegistry abstractions
  llm.py                     Anthropic client init
  evals.py                   eval harness with LLM-as-judge

apps/
  github_triage/             Project A — issue triage
  job_copilot/               Project B — job application copilot
  paper_digest/              Project C — arXiv research digest

tests/                       pytest unit tests
```

The agent loop in `agent_core/agent.py` is hand-rolled (not the SDK's
`tool_runner`) so the iteration model is visible: each pass, Claude either
finishes (`stop_reason == "end_turn"`), pauses (`pause_turn` for
server-side tool work), or emits `tool_use` blocks that we execute and feed
back as `tool_result`s. The system prompt is cached via
`cache_control: {"type": "ephemeral"}` so repeated runs share a warm prefix.

## Setup

```bash
cd "Python Project 6 - AI Agents"
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env   # then add your ANTHROPIC_API_KEY
```

## Project A — GitHub Issue Triage Agent

Reads a GitHub issue, searches a real codebase for grounding, and drafts a
maintainer-style reply with file:line references.

### Run it

```bash
# Triage an issue against this very repo
python -m apps.github_triage.cli \
  "Title: search_codebase returns no results for valid queries

The search_codebase tool returns 'No matches' even when I know the string
exists. Tried with 'def' and got nothing." \
  --repo "Python Project 6 - AI Agents"
```

The agent will:
1. Search the codebase for `search_codebase`, `def`, etc.
2. Read `apps/github_triage/tools.py` to inspect the implementation.
3. Classify the issue and draft a reply citing the actual lines it read.

### Run the evals

```bash
PYTHONPATH=. python -m apps.github_triage.evals "Python Project 6 - AI Agents"
```

Three cases covering grounded-bug-report, feature-vs-bug distinction, and
needs-info handling. Scored 1-5 by an LLM judge against per-case rubrics.

### Run the tests

```bash
PYTHONPATH=. pytest tests/ -v
```

## Project B — Job Application Co-pilot

Scrape a job posting, score fit against your resume, and draft a tailored
cover letter. Each application is tracked in a local SQLite database.

### Setup

```bash
mkdir -p ~/.job_copilot
# Save a plain-text version of your resume:
cp ~/Documents/resume.txt ~/.job_copilot/resume.txt
```

### Run it

```bash
# Score a posting and draft a letter
python -m apps.job_copilot.cli apply "https://jobs.example.com/posting/123"

# Review past applications
python -m apps.job_copilot.cli history
python -m apps.job_copilot.cli history --status drafted
```

The agent will:
1. Fetch and clean the job posting HTML.
2. Read your resume.
3. Score the fit 1-10 with honest gap analysis.
4. Draft a 3-paragraph cover letter that cites specific resume content.
5. Save the application to `~/.job_copilot/applications.db`.

### Tools
- `fetch_url(url)` — HTTPS only; refuses `file://` schemes.
- `read_resume()` — reads from the configured path; caps at 100KB.
- `save_application(...)` — validated insert into SQLite.
- `list_applications(status?)` — query the tracker.

## Project C — Research Paper Digest

Pull recent submissions from any arXiv category and produce a themed
weekly digest with critical 2-3 sentence summaries.

### Run it

```bash
# Generate a digest of the 10 most recent cs.AI papers
python -m apps.paper_digest.cli digest cs.AI

# Pick a different category and fetch more
python -m apps.paper_digest.cli digest cs.LG --count 15

# See past digests
python -m apps.paper_digest.cli list-saved
```

Supported categories: `cs.AI, cs.LG, cs.CL, cs.CV, cs.RO, cs.NE, cs.IR,
cs.SE, cs.DB, cs.DS, cs.CR, cs.HC, cs.MA, stat.ML, math.OC`.

The agent will:
1. Hit the arXiv API for recent submissions.
2. Cluster papers into 2-4 emergent themes.
3. Write 2-3 sentence critical summaries (flags overclaims).
4. Save the full digest as markdown under `~/.paper_digest/`.

### Tools
- `fetch_arxiv_recent(category, max_results)` — Atom feed parser, no extra deps.
- `save_digest(category, content)` — timestamped markdown file.
- `list_digests()` — recent saved digests.

## Tests

```bash
PYTHONPATH=. pytest tests/ -v
```

17 unit tests covering tool registries, path traversal protection, HTML
stripping, SQLite tracker validation, arXiv feed parsing, and category
allowlisting.

## Design choices worth noting

- **Manual loop over `tool_runner`** — visibility into iteration count,
  usage stats, and tool call history without subclassing. Easier to
  retrofit human-in-the-loop approval later.
- **Per-app `tools.py`** — domain tools are colocated with the app, not
  shoehorned into a generic registry. The shared `ToolRegistry` is just
  the dispatch layer.
- **LLM-as-judge for evals** — cheaper and faster than human review for
  iteration, and the rubrics are stable enough that scores correlate well
  with my own judgment. The judge runs on Sonnet 4.6 to keep eval costs
  low.
