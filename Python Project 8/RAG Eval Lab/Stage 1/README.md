# RAG Eval Lab — Stage 1: corpus, chunking, lexical retrieval

Stage 1 builds everything retrieval needs and nothing it doesn't: a **pinned,
verifiable copy of the FastAPI docs**, **three chunking strategies**, and
**two lexical retrievers** — one of them a faithful reproduction of
[RAG Mini Chat](../../../Python%20Project%202/RAG%20Mini%20Chat/)'s, so every
later improvement is measured against what I actually shipped before.

No API key, no network after the first fetch, no cost. All of it is
deterministic, because Stage 2 will turn these components into numbers.

```
raglab/
  corpus.py       pinned fetch (sparse + shallow git), lock + verify, load, sections
  markdown.py     mkdocs-flavoured Markdown -> clean text + heading map (fence-aware)
  chunking.py     paragraph (baseline) / fixed window / heading-aware chunkers
  retrieval.py    BM25 (from the Job Search Agent) and TF-IDF (from RAG Mini Chat)
  experiments.py  the 7 configurations Stage 2 will score
  cli.py          python -m raglab ...
tests/            150 tests (9 need the fetched corpus and skip without it)
```

## Run it

```bash
cd "Python Project 8/RAG Eval Lab"
python3 -m venv .venv                       # one environment shared by every stage
.venv/bin/pip install -r "Stage 1/requirements.txt"
cd "Stage 1"

../.venv/bin/python -m raglab corpus fetch          # re-fetches exactly what corpus.lock.json pins
../.venv/bin/python -m raglab corpus verify         # byte-for-byte check against the lock
../.venv/bin/python -m raglab corpus stats          # what was loaded, what was excluded and why
../.venv/bin/python -m raglab chunk                 # the three strategies side by side
../.venv/bin/python -m raglab search "how do I declare a query parameter with a default"
../.venv/bin/python -m raglab compare "how do I return a 404 error when an item is not found"
../.venv/bin/python -m pytest -q
```

## Design decisions

**The corpus is pinned like a dependency.** Every number a later stage reports
is only meaningful against a known corpus. `corpus.lock.json` (kept in git)
records the FastAPI tag (`0.165.0`), the commit it resolved to, and a SHA-256
for each of the 464 files. Re-fetching refuses to continue if upstream has
moved the tag, and **every command re-verifies the corpus against the lock
before using it** — an edited page, or an iCloud `index 2.md` conflict copy,
stops the run instead of quietly shifting its numbers. The lock itself is
checked for internal consistency, and is written with `\n` line endings and
fetched with git's line-ending conversion off, so it is byte-identical on
macOS, Linux and Windows. The docs stay out of git: they're third-party
content, reproducible from the lock.

**Only what's needed is downloaded.** A shallow, blob-filtered, sparse `git`
fetch transfers the English docs and their example code — not the history,
the library source, or the dozens of translated doc trees. Of `docs_src/`,
only the 305 files a page actually includes are kept.

**Upstream is treated as untrusted.** Only release tags are accepted (fetched
as `refs/tags/<tag>`, so not a pull request's head, and never something git
would parse as an option). A tree containing any symlink is refused, include
paths that could climb out of `docs_src/` are never copied (POSIX and Windows
forms alike), every write is checked to land inside the staging tree, and
oversized files are rejected. A new corpus is swapped in by two renames — the
old one moved aside, the new one moved in — so a failure or interruption at
any point leaves a complete corpus on disk.

**Six pages are pinned but not indexed, each for a stated reason** (`corpus
stats` prints them). The big one: `release-notes.md` is a changelog of PR
titles and **48% of the documentation by size** — left in, it would dominate
the term statistics that BM25 and TF-IDF are built on.

**Cleaning is fence-aware, and code examples are real.** FastAPI's docs are
mkdocs-flavoured: `{ #anchor }` heading IDs, `///` admonitions, terminal
transcripts styled with HTML, mkdocstrings `:::` directives, Jinja-templated
pages, and code pulled in by `{* ../../docs_src/x.py ln[14:18] *}` directives
(92 of 159 pages use them). The cleaner resolves those includes — line ranges
included — so chunks contain the actual example code. It tracks code fences
by CommonMark's rules, so `# create the app` inside Python is a comment, not a
heading, and it never strips markup inside an inline `` `code` `` span. Real
pages proved why naive filters fail: `{**old_dict}` is dict unpacking, not an
include directive; the WebSockets page embeds HTML inside a Python string,
which must survive; and 81 links like ``[`Depends`](…)`` hide their URL from
any filter that splits on code spans first. All are regression tests.
Author-assigned anchors are never renamed, and generated ones can never
collide — Stage 2's labels depend on both.

**Gold answers will be labelled by section, not by chunk.** This is the
decision the whole project rests on. Labelling "the right chunk" would tie the
golden set to one chunking strategy — useless when chunking is the thing being
compared. Instead every document is split into **sections** (a heading's own
content, up to the next heading), and **96% of the 1,176 sections have an
author-assigned anchor** like `tutorial/handling-errors#raise-an-httpexception-in-your-code`.
In Stage 2 a question's gold label is a set of those locations, and a retrieved
chunk counts as a hit if its span overlaps one. That works for any chunker, and
survives changes to cleaning.

**Every chunk is an exact span.** `doc.text[chunk.start:chunk.end] == chunk.text`
holds for every chunk of every strategy on the real corpus (tested), and each
chunk knows the section it starts in. That's what makes section-overlap scoring
possible.

**The baseline is reproduced, not approximated.** RAG Mini Chat split text on
blank lines and scored it with `TfidfVectorizer(ngram_range=(1, 2),
stop_words="english")` and cosine similarity. The `paragraph` chunker and the
`tfidf` retriever do exactly that — a test asserts the paragraph chunks equal
`[p.strip() for p in text.split("\n\n") if p.strip()]` on all 153 pages.

**Three switches, varied one at a time.** A configuration is *where chunks are
cut* × *how they're scored* × *whether each chunk carries its heading
breadcrumb* (`Handling Errors > Raise an HTTPException in your code`). The
breadcrumb is deliberately a separate switch rather than part of the
heading-aware strategy, so Stage 2 can attribute an improvement to the right
cause. Both retrievers share scikit-learn's analyzer, so BM25 vs TF-IDF
isolates the scoring function (the baseline's bigrams are the one difference,
kept for fidelity). Ties break by corpus order, so every ranking reproduces.

## What the strategies produce

| strategy | params | chunks | mean chars | median | p95 | max |
|---|---|---:|---:|---:|---:|---:|
| paragraph (baseline) | split on blank lines | 9,228 | 99 | 73 | 253 | 4,315 |
| fixed | 1,200 chars, 200 overlap | 1,045 | 1,058 | 1,159 | 1,197 | 1,200 |
| heading | ≤1,200 chars, whole paragraphs | 1,538 | 619 | 566 | 1,180 | 1,200 |

The baseline shreds the docs into fragments with a median of 73 characters:
every heading becomes its own chunk, and every blank line inside a code
example is a cut — a function gets separated from its own comment. Whether
that hurts retrieval as much as it looks like it should is a Stage 2 question.

## One question, seven configurations

`compare "how do I return a 404 error when an item is not found"` — the right
answer lives in `tutorial/handling-errors`:

| configuration | #1 result | error-handling sections in top 3 |
|---|---|:---:|
| baseline (RAG Mini Chat) | `how-to/conditional-openapi#…` | 1 |
| paragraph + TF-IDF | `tutorial/frontend#custom-404-page` | 1 |
| paragraph + BM25 | `tutorial/handling-errors#add-custom-headers` | 1 |
| fixed + TF-IDF | `tutorial/frontend#client-side-routing` | 1 |
| **fixed + BM25** | `tutorial/handling-errors#raise-an-httpexception-in-your-code` | **3** |
| heading + TF-IDF | `tutorial/frontend#custom-404-page` | 0 |
| **heading + BM25** | `tutorial/handling-errors#add-custom-headers` | **3** |

Three of the four TF-IDF configurations rank a `tutorial/frontend` page first
— custom 404 pages, client-side routing — pulled in by the literal token
`404`; the baseline instead leads with a conditional-OpenAPI section. Both
BM25 configurations with real-sized chunks keep all three results on the
error-handling page. **That is one question, which makes it an anecdote, not a
result.** Turning it into a measurement over a labelled question set is
exactly what Stage 2 is for.

## How Stage 1 was verified

Beyond the test suite, Stage 1 went through two independent adversarial
reviews — one hunting correctness bugs by fuzzing the cleaner and chunkers,
one checking every claim in this README and attacking the fetch path. Between
them they confirmed **19 defects** (two high-severity: a crafted upstream tree
could make the fetch write outside the corpus, or follow a symlink out of it).
All 19 are fixed, each with a regression test — with one sub-case kept on
purpose: in `a<b and c>d`, the `<b and c>` is still removed, because CommonMark
parses it as an HTML tag too. The reviewers also showed four
tests were too weak to catch the bugs they targeted; those were rewritten and
mutation-tested — each failure mode the reviewers described was injected into
the code, and the suite now fails on every one.

## Known limitations

- One include can't be resolved: `how-to/configure-swagger-ui.md` pulls lines
  from FastAPI's *library* source rather than its examples. It's reported by
  `corpus stats` instead of silently dropped.
- The 22 API reference pages are stubs. Their content is generated from
  library docstrings by mkdocstrings at build time, so the Markdown holds only
  a title and an import line.
- Chunk sizes are in characters, not model tokens — deliberate for
  determinism (no tokenizer dependency); ~1,200 chars ≈ 300 tokens.
- English docs only.

## Next: Stage 2

A golden set of 50–100 questions, each labelled with the section(s) that
answer it; recall@k, MRR and nDCG for all seven configurations; and versioned
run manifests (corpus fingerprint, configuration, code version) so every number
in the results table can be reproduced.
