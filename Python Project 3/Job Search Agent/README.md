# Job Search Agent

**"I automated my own job search."** An agent that reads a job posting,
extracts what it's really asking for, retrieves matching evidence from a
canonical inventory of my actual experience, drafts tailored resume bullets —
and asks a human before anything is written to disk.

Companion to my Node.js ATS resume pipeline: that project formats resumes for
machines; this one decides *what should go in them*, grounded in real
experience so nothing is fabricated.

## The three patterns this demonstrates

1. **Structured output** — extraction and drafting are `messages.parse()`
   calls returning validated Pydantic models. Downstream code never
   string-parses LLM prose.
2. **Agentic tool use (RAG)** — the evidence stage is an agent loop where
   Claude drives a `search_experience` tool (BM25 implemented from scratch,
   ~60 lines, stdlib only) over the experience inventory.
3. **Human-in-the-loop** — the pipeline takes an `approve` callback. The CLI
   renders the full draft (fit score, coverage table, bullets with evidence
   citations, honest gaps) and nothing is saved until a human says yes.

Plus the one that matters most: **grounding enforced in code, not prompts.**
Every drafted bullet must cite evidence IDs that retrieval actually returned;
`_validate_grounding()` raises if the model cites anything else. Prompts
request behavior — validators guarantee it.

## Pipeline

```
posting URL/file
   │
   ▼
[1] EXTRACT    messages.parse → JobRequirements (must-haves, nice-to-haves, red flags)
   │
   ▼
[2] EVIDENCE   agent loop: Claude calls search_experience(query) → BM25 → evidence pack
   │
   ▼
[3] DRAFT      messages.parse → TailoredDraft (bullets cite evidence IDs; gaps stay honest)
   │           └─ _validate_grounding() rejects any uncited claim
   ▼
[4] GATE       human reviews in terminal → approve/reject
   │
   ▼ (only if approved)
output/<date>_<company>.md + .json sidecar
```

## Setup

```bash
cd "Job Search Agent"
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
cp .env.example .env             # add your ANTHROPIC_API_KEY
```

Then **replace the sample entries** in `data/experience_inventory.yaml` with
your real experience. One achievement per entry, real metrics only, tags for
retrieval. This file is the whole point — the agent can only claim what's in
it.

## Usage

```bash
# Full run against a posting URL
python -m job_agent.cli run "https://jobs.example.com/backend-engineer-123"

# Job boards that block bots? Save the posting text to a file:
python -m job_agent.cli run saved_posting.txt

# Test retrieval directly (tune your inventory tags)
python -m job_agent.cli search "airflow dag scheduling"

# Lint the inventory
python -m job_agent.cli validate
```

The run command walks through: requirement extraction → live evidence
searches (you see each query) → draft → a review screen with fit score,
requirement coverage table, bullets + citations, and honest gaps → approval
prompt. Reject and nothing is written.

## Running in PyCharm

1. **File → Open** → this folder (`Job Search Agent`)
2. **Settings → Project → Python Interpreter → Add Existing** → `.venv/bin/python`
3. Run configs:
   - **Tests**: right-click `tests/` → *Run 'pytest in tests'*
   - **CLI**: Run → Edit Configurations → new *Python* config →
     module `job_agent.cli`, parameters `run saved_posting.txt`,
     working directory = project root. The approval prompt runs in the
     PyCharm run console.

## Tests

```bash
python -m pytest -q
```

Covers: BM25 ranking/tag matching, tokenizer edge cases, inventory
validation (duplicates, missing fields), grounding enforcement (the
anti-fabrication rule), evidence-pack dedup, draft saving/sanitization, and
fetcher URL-scheme rejection. No API key needed — LLM stages are exercised
via their validators and contracts.

## Design decisions (interview talking points)

- **Why BM25 from scratch instead of a vector DB?** The corpus is ~20
  entries. Embeddings add a network dependency and cost for zero retrieval
  gain at this scale. The `search()` interface is swappable — the README
  upgrade path is Voyage embeddings behind the same signature. Knowing *when
  not to* use a vector DB is the senior answer.
- **Why is grounding a validator, not a prompt rule?** Prompts reduce the
  rate of fabrication; they can't eliminate it. The validator makes the
  failure loud (typed exception, exit code 2) instead of silent.
- **Why does the approval gate take a callback?** So the HITL contract is
  testable (`approve=lambda: False` must write nothing) and the CLI can be
  swapped for a web UI without touching pipeline code.
- **Why structured output for stages 1 and 3 but an agent loop for stage 2?**
  Extraction and drafting have known output shapes — constrain them. Evidence
  gathering genuinely benefits from model judgment (what to search, when to
  stop) — let it act, but cap iterations and log every query.
