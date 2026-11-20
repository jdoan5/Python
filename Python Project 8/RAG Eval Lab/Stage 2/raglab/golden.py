"""The golden set: questions about the FastAPI docs, with graded section-level judgments.

Each question was written from one *target* section, picked by a seeded,
stratified sample rather than by hand, so the set covers the docs in
proportion to their size instead of whatever happens to be easy to ask.
Each sampled section also comes with a pre-assigned question *style*:

- keyword:    uses the docs' own terms, as someone who knows the API name would;
- paraphrase: describes the need in plain words, avoiding the section's
              distinctive terms and identifiers;
- scenario:   describes a situation or symptom, the way people actually ask.

Pre-assigning styles stops the author from drifting towards whichever style
favours a given retriever. Lexical retrievers should do best on keyword
questions and worst on paraphrases. Stage 3's embeddings exist to close that gap.

Judgments are graded per section location (`doc_id#anchor`):
2 = answers the question, 1 = related but not an answer, 0 = not relevant.
The target is always judged 2. Other sections are added by *pooling* from
two sources: every section any evaluated configuration ranks in its top 10,
so a configuration that finds a valid answer elsewhere is credited rather than
penalised; and every section on the question's target page. Explicit 0s
record that a pooled section was looked at and rejected.

A golden set is pinned to the corpus it was labelled against. Its header names
the corpus fingerprint, and evaluation refuses to run against any other corpus.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Iterable, Mapping

import yaml

from raglab.corpus import CorpusLock, project_root
from raglab.sections import SectionIndex, SectionRef

STYLES = ("keyword", "paraphrase", "scenario")
GRADES = (0, 1, 2)
GOLDEN_VERSION = 1
MAX_QUESTION_CHARS = 300

_ID_RE = re.compile(r"^q\d{3}$")
_FINGERPRINT_RE = re.compile(r"^[0-9a-f]{64}$")
_TOP_KEYS_REQUIRED = ("version", "corpus", "questions")
_TOP_KEYS_OPTIONAL = ("sampling", "rubric", "skipped")
_QUESTION_KEYS_REQUIRED = ("id", "question", "style", "target", "judgments")
_QUESTION_KEYS_OPTIONAL = ("notes",)

# --- sampling: which sections questions are written from -------------------------------

SAMPLE_MIN_BODY_CHARS = 150  # one-line sections ("See the reference.") make poor targets
SAMPLE_PER_DOC = 2           # spread questions across pages, not just across areas
# Pages a user of FastAPI would not ask about. They stay in the corpus (and
# can be retrieved and judged); they just aren't used as targets.
SAMPLE_EXCLUDED: dict[str, str] = {
    "reference/": "API reference stubs: their content is generated from docstrings at build time",
    "help-fastapi": "how to support the FastAPI project, not how to use FastAPI",
    "contributing": "how to develop FastAPI itself",
    "management": "how the FastAPI repository is maintained",
    "translations": "the docs' translation workflow",
    "newsletter": "a sign-up link",
    "about/index": "a navigation page",
    "learn/index": "a navigation page",
    "resources/index": "a navigation page",
    "how-to/index": "a navigation page",
    "how-to/general": "a page of pointers to the pages that hold the answers",
}


class GoldenError(Exception):
    """The golden set file is missing, malformed, or doesn't match the corpus."""


def strict_json(text: str):
    """json.loads that refuses a repeated key instead of silently keeping the last one."""

    def pairs(items: list[tuple[str, object]]) -> dict:
        seen: dict = {}
        for key, value in items:
            if key in seen:
                raise ValueError(f"duplicate key {key!r}")
            seen[key] = value
        return seen

    return json.loads(text, object_pairs_hook=pairs)


class _StrictLoader(yaml.SafeLoader):
    """safe_load, minus two YAML features that would let labels go wrong silently.

    A duplicated key would keep only its last value: a section graded twice
    would quietly lose one grade. Anchors and aliases would let one entry stand
    for many, and are the vehicle for "billion laughs" memory blow-ups.
    """

    def compose_node(self, parent, index):  # noqa: ANN001 - PyYAML's signature
        if self.check_event(yaml.events.AliasEvent):
            event = self.peek_event()
            raise yaml.composer.ComposerError(None, None, "anchors and aliases are not allowed", event.start_mark)
        return super().compose_node(parent, index)


def _construct_unique_mapping(loader: _StrictLoader, node: yaml.MappingNode, deep: bool = False) -> dict:
    mapping: dict = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError:
            raise yaml.constructor.ConstructorError(None, None, "a key must be text or a number", key_node.start_mark)
        if duplicate:
            raise yaml.constructor.ConstructorError(None, None, f"duplicate key {key!r}", key_node.start_mark)
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_StrictLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping)


@dataclass(frozen=True)
class Question:
    id: str
    text: str
    style: str
    target: str
    judgments: Mapping[str, int] = field(repr=False)
    notes: str = ""

    @property
    def primary(self) -> frozenset[str]:
        """Sections that answer the question (grade 2)."""
        return frozenset(loc for loc, g in self.judgments.items() if g == 2)

    @property
    def relevant(self) -> dict[str, int]:
        return {loc: g for loc, g in self.judgments.items() if g > 0}


@dataclass(frozen=True)
class GoldenSet:
    version: int
    corpus_ref: str
    corpus_fingerprint: str
    questions: tuple[Question, ...]
    sha256: str        # of the file's bytes: what a run manifest records
    path: Path
    raw: Mapping = field(repr=False, default_factory=dict)

    def by_id(self, qid: str) -> Question:
        for q in self.questions:
            if q.id == qid:
                return q
        raise KeyError(qid)


def questions_fingerprint(questions: Iterable[Question]) -> str:
    """A hash of what was asked (id, text, style, target), but not of the judgments.

    Pooling adds judgments after the questions are written, so the file's hash
    changes as labelling proceeds; this one doesn't. Recording it when the
    questions were frozen proves later labelling never edited a question.
    """
    h = hashlib.sha256()
    for q in sorted(questions, key=lambda q: q.id):
        h.update("\0".join((q.id, q.text, q.style, q.target)).encode("utf-8") + b"\n")
    return h.hexdigest()


def default_golden_path(version: int = GOLDEN_VERSION) -> Path:
    """Shared by every stage, like the corpus lock: all stages answer the same questions."""
    return project_root() / "golden" / f"v{version}.yaml"


def area_of(doc_id: str) -> str:
    return doc_id.split("/", 1)[0] if "/" in doc_id else "(top)"


# --- loading ------------------------------------------------------------------------------


def load_golden(path: Path) -> GoldenSet:
    """Parse and validate a golden set file, reporting every problem at once."""
    try:
        data = path.read_bytes()
    except OSError as e:
        raise GoldenError(f"Cannot read golden set {path}: {e}") from e
    try:
        raw = yaml.load(data.decode("utf-8"), Loader=_StrictLoader)  # noqa: S506 - a SafeLoader subclass
    except (UnicodeDecodeError, yaml.YAMLError) as e:
        raise GoldenError(f"{path.name} is not valid UTF-8 YAML: {e}") from e
    except (RecursionError, ValueError) as e:  # absurd nesting, or an integer too long to convert
        raise GoldenError(f"{path.name} is not a golden set: {type(e).__name__}: {e}") from e

    problems: list[str] = []
    if not isinstance(raw, dict):
        raise GoldenError(f"{path.name}: expected a mapping at the top level")
    _keys(raw, _TOP_KEYS_REQUIRED, _TOP_KEYS_OPTIONAL, "top level", problems)

    version = raw.get("version")
    if not isinstance(version, int) or isinstance(version, bool) or version < 1:
        problems.append("version: must be a positive integer")
    corpus = raw.get("corpus")
    ref = fingerprint = ""
    if not isinstance(corpus, dict):
        problems.append("corpus: must be a mapping with ref and fingerprint")
    else:
        ref, fingerprint = corpus.get("ref"), corpus.get("fingerprint")
        if not isinstance(ref, str) or not ref:
            problems.append("corpus.ref: must be the FastAPI tag the labels were made against")
        if not isinstance(fingerprint, str) or not _FINGERPRINT_RE.fullmatch(fingerprint):
            problems.append("corpus.fingerprint: must be the 64-hex fingerprint from corpus.lock.json")

    questions: list[Question] = []
    entries = raw.get("questions")
    if not isinstance(entries, list) or not entries:
        problems.append("questions: must be a non-empty list")
        entries = []
    seen_ids: set[str] = set()
    seen_text: dict[str, str] = {}
    for n, entry in enumerate(entries, start=1):
        question = _question(entry, n, problems)
        if question is None:
            continue
        if question.id in seen_ids:
            problems.append(f"{question.id}: duplicate id")
        seen_ids.add(question.id)
        key = " ".join(question.text.lower().split())
        if key in seen_text:
            problems.append(f"{question.id}: same question as {seen_text[key]}")
        seen_text.setdefault(key, question.id)
        questions.append(question)

    skipped = raw.get("skipped", [])
    if not isinstance(skipped, list) or not all(
        isinstance(s, dict) and isinstance(s.get("location"), str) and isinstance(s.get("reason"), str) and s["reason"]
        for s in skipped
    ):
        problems.append("skipped: must be a list of {location, reason}")

    if problems:
        shown = "\n  ".join(problems[:25])
        more = f"\n  … and {len(problems) - 25} more" if len(problems) > 25 else ""
        raise GoldenError(f"{path.name} has {len(problems)} problem(s):\n  {shown}{more}")
    return GoldenSet(
        version=version, corpus_ref=ref, corpus_fingerprint=fingerprint, questions=tuple(questions),
        sha256=hashlib.sha256(data).hexdigest(), path=path, raw=raw,
    )


def _keys(entry: dict, required: tuple[str, ...], optional: tuple[str, ...], where: str, problems: list[str]) -> None:
    for key in required:
        if key not in entry:
            problems.append(f"{where}: missing {key!r}")
    for key in entry:
        if key not in required and key not in optional:
            problems.append(f"{where}: unknown key {key!r}")


def _question(entry: object, n: int, problems: list[str]) -> Question | None:
    if not isinstance(entry, dict):
        problems.append(f"question #{n}: must be a mapping")
        return None
    qid = entry.get("id")
    where = qid if isinstance(qid, str) and qid else f"question #{n}"
    before = len(problems)
    _keys(entry, _QUESTION_KEYS_REQUIRED, _QUESTION_KEYS_OPTIONAL, where, problems)
    if not isinstance(qid, str) or not _ID_RE.fullmatch(qid):
        problems.append(f"{where}: id must look like q001")
    text = entry.get("question")
    if not isinstance(text, str) or not text.strip():
        problems.append(f"{where}: question must be non-empty text")
    elif len(text) > MAX_QUESTION_CHARS:
        problems.append(f"{where}: question is longer than {MAX_QUESTION_CHARS} characters")
    style = entry.get("style")
    if style not in STYLES:
        problems.append(f"{where}: style must be one of {', '.join(STYLES)}")
    notes = entry.get("notes", "")
    if not isinstance(notes, str):
        problems.append(f"{where}: notes must be text")
    judgments = entry.get("judgments")
    clean: dict[str, int] = {}
    if not isinstance(judgments, dict) or not judgments:
        problems.append(f"{where}: judgments must be a non-empty mapping of location -> grade")
    else:
        for loc, grade in judgments.items():
            if not isinstance(loc, str) or not loc:
                problems.append(f"{where}: judgment key {loc!r} is not a section location")
            elif isinstance(grade, bool) or grade not in GRADES:
                problems.append(f"{where}: {loc} has grade {grade!r}; grades are 0, 1 or 2")
            else:
                clean[loc] = grade
        if not any(g == 2 for g in clean.values()):
            problems.append(f"{where}: no section judged 2 (answers), so the question can't be scored")
    target = entry.get("target")
    if not isinstance(target, str) or not target:
        problems.append(f"{where}: target must be the section the question was written from")
    elif clean and clean.get(target) != 2:
        problems.append(f"{where}: target {target} must be judged 2")
    if len(problems) > before:
        return None
    return Question(
        id=qid, text=" ".join(text.split()), style=style, target=target,
        judgments=MappingProxyType(clean), notes=notes,
    )


def check_against_corpus(golden: GoldenSet, lock: CorpusLock, index: SectionIndex) -> list[str]:
    """Problems that would make the labels mean something other than what was judged."""
    problems: list[str] = []
    if golden.corpus_ref != lock.ref:
        problems.append(f"labelled against FastAPI {golden.corpus_ref}, but the lock pins {lock.ref}")
    if golden.corpus_fingerprint != lock.fingerprint:
        problems.append(
            f"labelled against corpus {golden.corpus_ref} ({golden.corpus_fingerprint[:12]}…), but the lock pins "
            f"{lock.ref} ({lock.fingerprint[:12]}…): re-judge before comparing numbers"
        )
    for q in golden.questions:
        for loc in q.judgments:
            if loc in index:
                continue
            if loc in index.empty:
                problems.append(f"{q.id}: {loc} has no text under its heading, so nothing can retrieve it")
            else:
                problems.append(f"{q.id}: {loc} is not a section in the corpus")
    for s in golden.raw.get("skipped", []):
        if s["location"] not in index:
            problems.append(f"skipped: {s['location']} is not a section in the corpus")
    return problems


# --- sampling -------------------------------------------------------------------------------


@dataclass(frozen=True)
class SampledSection:
    ref: SectionRef
    area: str
    style: str
    order: str  # the seeded hash that ranked it, for reproducibility


def _excluded(doc_id: str) -> bool:
    return any(doc_id.startswith(p) if p.endswith("/") else doc_id == p for p in SAMPLE_EXCLUDED)


def _seeded_rank(seed: int, location: str) -> str:
    # A hash per section rather than a shuffled list: the order never depends on a
    # library's RNG, and adding or removing one section doesn't reorder the rest.
    return hashlib.sha256(f"{seed}:{location}".encode("utf-8")).hexdigest()


def _allocate(n: int, sizes: Mapping[str, int]) -> dict[str, int]:
    """Largest-remainder proportional allocation of n samples across areas.

    Exact integer arithmetic: equal remainders really are equal, so ties go to
    the area name as specified, not to floating-point noise.
    """
    total = sum(sizes.values())
    alloc = {a: n * s // total for a, s in sizes.items()}
    remainder = {a: n * s % total for a, s in sizes.items()}
    for a in sorted(sizes, key=lambda a: (-remainder[a], a))[: n - sum(alloc.values())]:
        alloc[a] += 1
    return alloc


def sample_targets(index: SectionIndex, n: int, seed: int) -> list[SampledSection]:
    """Stratified by area, in proportion to how many eligible sections each area has.

    May return fewer than n if the per-page cap leaves an area short; callers
    report the count rather than silently topping up from another area.
    """
    eligible: dict[str, list[SectionRef]] = {}
    for ref in index:
        if not _excluded(ref.doc_id) and ref.length >= SAMPLE_MIN_BODY_CHARS:
            eligible.setdefault(area_of(ref.doc_id), []).append(ref)
    available = sum(len(v) for v in eligible.values())
    if not 1 <= n <= available:
        raise ValueError(f"can sample between 1 and {available} sections")

    picked: list[tuple[str, SectionRef]] = []
    for area, quota in _allocate(n, {a: len(v) for a, v in eligible.items()}).items():
        per_doc: dict[str, int] = {}
        taken = 0
        for ref in sorted(eligible[area], key=lambda r: _seeded_rank(seed, r.location)):
            if taken == quota:
                break
            if per_doc.get(ref.doc_id, 0) >= SAMPLE_PER_DOC:
                continue
            per_doc[ref.doc_id] = per_doc.get(ref.doc_id, 0) + 1
            picked.append((_seeded_rank(seed, ref.location), ref))
            taken += 1
    picked.sort()
    return [
        SampledSection(ref=ref, area=area_of(ref.doc_id), style=STYLES[i % len(STYLES)], order=rank)
        for i, (rank, ref) in enumerate(picked)
    ]


# --- writing ------------------------------------------------------------------------------


def merge_judgments(
    raw: Mapping, additions: Mapping[str, Mapping[str, int]], regrade: bool = False
) -> tuple[dict, int, int]:
    """Add judgments to a golden set's data; returns (data, added, changed).

    By default an existing judgment is never changed: a disagreement is an
    error, not an overwrite. `regrade=True` is for a deliberate correction
    (say, sections first judged from a truncated excerpt, re-judged on their
    full text) and may change grades, but never a question's target, which is
    an answer by construction.
    """
    data = yaml.safe_load(yaml.safe_dump(dict(raw), sort_keys=False))  # deep copy, plain types, key order kept
    by_id = {q["id"]: q for q in data["questions"]}
    added = changed = 0
    problems = []
    for qid, judged in additions.items():
        q = by_id.get(qid)
        if q is None:
            problems.append(f"{qid}: no such question")
            continue
        for loc, grade in judged.items():
            if isinstance(grade, bool) or grade not in GRADES:
                problems.append(f"{qid}: {loc} has grade {grade!r}; grades are 0, 1 or 2")
            elif loc not in q["judgments"]:
                q["judgments"][loc] = grade
                added += 1
            elif q["judgments"][loc] == grade:
                continue
            elif not regrade:
                problems.append(f"{qid}: {loc} is already judged {q['judgments'][loc]}, not {grade}")
            elif loc == q["target"]:
                problems.append(f"{qid}: {loc} is the question's target and stays judged 2")
            else:
                q["judgments"][loc] = grade
                changed += 1
    if problems:
        raise GoldenError("judgments not merged:\n  " + "\n  ".join(problems))
    for q in data["questions"]:  # target first, then answers, related, rejected
        q["judgments"] = dict(sorted(q["judgments"].items(), key=lambda kv: (kv[0] != q["target"], -kv[1], kv[0])))
    return data, added, changed

HEADER = """\
# RAG Eval Lab — golden set v{version}
#
# Questions about the FastAPI docs, each with graded judgments of which doc
# sections answer it. Grades: 2 = answers the question, 1 = related but not an
# answer, 0 = judged and rejected. The full rubric is the `rubric` key below.
#
# Questions were written from seeded, stratified samples of sections (see
# `sampling`). Two pools were then judged against each question: every section
# any evaluated configuration ranks in its top 10, and every section on the
# question's target page. Edit by hand if you like, then run
# `python -m raglab golden validate`.
"""

# Keys in the order a reader wants them: what it is, then the questions, each question before its labels.
_KEY_ORDER = ("version", "corpus", "sampling", "rubric", "questions", "skipped")
_NESTED_ORDER = {
    "corpus": ("ref", "fingerprint"),
    "sampling": ("seed", "sections", "method"),
    "question": ("id", "question", "style", "target", "notes", "judgments"),
    "skipped": ("location", "reason"),
}


def _ordered(mapping: Mapping, order: tuple) -> dict:
    return {k: mapping[k] for k in order if k in mapping} | {k: v for k, v in mapping.items() if k not in order}


def dump_golden(raw: Mapping, path: Path) -> None:
    """Write a golden set with its header comment; keys keep their order."""
    ordered = _ordered(raw, _KEY_ORDER)
    for key in ("corpus", "sampling"):
        if isinstance(ordered.get(key), Mapping):
            ordered[key] = _ordered(ordered[key], _NESTED_ORDER[key])
    if isinstance(ordered.get("rubric"), Mapping):  # best grade first
        ordered["rubric"] = dict(sorted(ordered["rubric"].items(), key=lambda kv: -kv[0] if isinstance(kv[0], int) else 0))
    for key, order in (("questions", _NESTED_ORDER["question"]), ("skipped", _NESTED_ORDER["skipped"])):
        if isinstance(ordered.get(key), list):
            ordered[key] = [_ordered(e, order) if isinstance(e, Mapping) else e for e in ordered[key]]
    body = yaml.safe_dump(ordered, sort_keys=False, allow_unicode=True, width=110)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(HEADER.format(version=raw.get("version", "?")) + "\n" + body, encoding="utf-8", newline="\n")
    tmp.replace(path)


# --- label reliability ------------------------------------------------------------------------


def agreement_report(items: list[Mapping], population: Mapping[str, int] | None = None) -> dict:
    """How often a second, blind judge gave the same grade as the first.

    `items` are {question, location, first, second, is_target}. Reported:
    exact agreement, Cohen's kappa (plain and linearly weighted), the 3x3
    confusion matrix, agreement within each first-pass grade, and how many
    question targets the second judge also called answers.

    A stratified sample over-represents some grades. Given `population` (how
    many labels each stratum holds: "answers", "related", "rejected",
    "targets"), agreement and kappa are also reported reweighted to the whole
    label set.
    """
    from raglab.stats import cohen_kappa  # local import keeps golden.py free of numpy for loaders

    if not items:
        raise ValueError("no agreement items")
    for it in items:
        for key in ("first", "second"):
            if isinstance(it[key], bool) or it[key] not in GRADES:
                raise ValueError(f"{it.get('question')}: {key} grade must be 0, 1 or 2, not {it[key]!r}")
    first = [it["first"] for it in items]
    second = [it["second"] for it in items]
    confusion = [[sum(1 for a, b in zip(first, second) if a == i and b == j) for j in GRADES] for i in GRADES]
    by_first = {  # None, not 0%, when the sample has no item of that grade
        g: (sum(1 for a, b in zip(first, second) if a == g and b == g) / first.count(g)) if first.count(g) else None
        for g in GRADES
    }
    targets = [it for it in items if it.get("is_target")]
    # Recall, success, MRR and coverage only ask "is it an answer?": agreement on that split matters most.
    is_answer_1 = [int(g == 2) for g in first]
    is_answer_2 = [int(g == 2) for g in second]
    reweighted = None
    if population is not None:
        stratum = lambda it: "targets" if it.get("is_target") else {2: "answers", 1: "related", 0: "rejected"}[it["first"]]  # noqa: E731
        counts: dict[str, int] = {}
        for it in items:
            counts[stratum(it)] = counts.get(stratum(it), 0) + 1
        weights = [population[stratum(it)] / counts[stratum(it)] for it in items]
        total = sum(weights)
        reweighted = {
            "exact": sum(w for w, a, b in zip(weights, first, second) if a == b) / total,
            "kappa": cohen_kappa(first, second, sample_weight=weights),
            "kappa_linear": cohen_kappa(first, second, weighted=True, sample_weight=weights),
            "answer_kappa": cohen_kappa(is_answer_1, is_answer_2, categories=(0, 1), sample_weight=weights),
        }
    return {
        "items": len(items),
        "exact": sum(a == b for a, b in zip(first, second)) / len(items),
        "kappa": cohen_kappa(first, second),
        "kappa_linear": cohen_kappa(first, second, weighted=True),
        "answer_kappa": cohen_kappa(is_answer_1, is_answer_2, categories=(0, 1)),
        "confusion": confusion,  # rows: first judge 0/1/2; columns: second judge 0/1/2
        "agreement_by_first_grade": by_first,
        "first_counts": {g: first.count(g) for g in GRADES},
        "targets": len(targets),
        "targets_confirmed": sum(1 for it in targets if it["second"] == 2),
        "reweighted": reweighted,
    }
