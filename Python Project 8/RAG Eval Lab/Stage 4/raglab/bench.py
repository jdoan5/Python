"""What each retriever costs on this machine: `raglab bench` (PLAN.md, "Cost").

Measured outside the reproducible results: nothing in the evaluation reads
results/cost.json, and no grade is read here. Only question texts are used.

- Caches off. Every query embedding and every cross-encoder score is computed
  by the model, so the timings are the models', not a cache lookup's.
- Warm timing: after 10 warm-up questions, 5 passes over the questions, with
  the order of retrievers shuffled by seed 20261120 in each pass. Each step is
  timed on its own: question embedding, dense scoring, BM25, fusion, reranking.
- Reranking is timed at batch 1, as evaluated, and with each question's 100
  pairs in one batch, as a deployment would run it.
- Cold start: import, model loading and a first question, in fresh processes.
- Sizes: model files, index arrays, CPU-seconds per 1,000 questions, peak RSS.
"""

from __future__ import annotations

import json
import os
import platform
import random
import resource
import subprocess
import sys
import time
from pathlib import Path
from statistics import median
from typing import Sequence

import numpy as np

from raglab.chunking import chunk_corpus, get_chunker
from raglab.experiments import RetrievalConfig, VARIANTS
from raglab.retrieval import Bm25Retriever, TfidfRetriever

SEED = 20261120
WARMUP = 10
PASSES = 5
COLD_RUNS = 5
# Heading chunks with the breadcrumb: the confirmatory setting, every retriever, and the rerank-depth curve.
BENCH_CONFIGS: tuple[RetrievalConfig, ...] = (
    *(RetrievalConfig("heading", r) for r in ("tfidf", "bm25", "dense", "hybrid", "hybrid+rerank", "bm25+rerank")),
    *(RetrievalConfig("heading", "hybrid+rerank", variant=f"depth {d}") for d in (10, 20, 50)),
)


class _Timed:
    """A model wrapper with no cache that adds the seconds each call takes to a named counter."""

    def __init__(self, inner, method: str, clock: dict, key: str) -> None:
        self.inner, self.settings, self._clock, self._key = inner, inner.settings, clock, key
        setattr(self, method, self._call(getattr(inner, method)))

    def _call(self, fn):
        def timed(*args, **kwargs):
            start = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                self._clock[self._key] = self._clock.get(self._key, 0.0) + time.perf_counter() - start
        return timed


def _time_method(obj, name: str, clock: dict, key: str) -> None:
    fn = getattr(obj, name)

    def timed(*args, **kwargs):
        start = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            clock[key] = clock.get(key, 0.0) + time.perf_counter() - start
    setattr(obj, name, timed)


def machine() -> dict:
    def sysctl(name: str) -> str | None:
        try:
            out = subprocess.run(["sysctl", "-n", name], capture_output=True, text=True, timeout=5).stdout.strip()
            return out or None
        except (OSError, subprocess.SubprocessError):
            return None

    power = None
    if platform.system() == "Darwin":
        try:
            power = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True, timeout=5).stdout.split("'")[1]
        except (OSError, subprocess.SubprocessError, IndexError):
            power = None
    memory = sysctl("hw.memsize")
    return {"chip": sysctl("machdep.cpu.brand_string") or platform.processor() or None,
            "cores": os.cpu_count(), "performance_cores": sysctl("hw.perflevel0.physicalcpu"),
            "ram_gb": round(int(memory) / 2**30) if memory else None, "power": power,
            "os": f"{platform.system()} {platform.release()} {platform.machine()}", "python": platform.python_version()}


def _percentiles(seconds: Sequence[float]) -> dict:
    ms = np.asarray(seconds) * 1000
    return {"n": int(ms.size), "p50_ms": float(np.percentile(ms, 50)), "p95_ms": float(np.percentile(ms, 95)),
            "mean_ms": float(ms.mean())}


def warm_timing(docs, questions: Sequence[str]) -> dict:
    """Per-question latency and per-step time for each configuration, with every cache off."""
    from raglab import neural

    lock = neural.model_lock()
    clock: dict[str, float] = {}
    raw_embedder = neural.SentenceTransformerEmbedder(lock["embedder"])
    raw_scorer = neural.CrossEncoderScorer(lock["reranker"])
    chunks = chunk_corpus(docs, get_chunker("heading"))
    build: dict[str, float] = {}
    start = time.perf_counter()
    bm25 = Bm25Retriever(chunks)
    build["bm25"] = time.perf_counter() - start
    start = time.perf_counter()
    tfidf = TfidfRetriever(chunks)
    build["tfidf"] = time.perf_counter() - start
    start = time.perf_counter()
    dense = neural.DenseRetriever(chunks, embedder=_Timed(raw_embedder, "embed", clock, "question embedding"))
    build["dense (embed every chunk)"] = time.perf_counter() - start
    scorer = _Timed(raw_scorer, "score", clock, "reranking")
    hybrid = neural.HybridRetriever(chunks, lexical=bm25, dense=dense)
    retrievers = {c.name: r for c, r in zip(BENCH_CONFIGS, (
        tfidf, bm25, dense, hybrid,
        neural.RerankRetriever(chunks, first_stage=hybrid, scorer=scorer),
        neural.RerankRetriever(chunks, first_stage=bm25, scorer=scorer),
        *(neural.RerankRetriever(chunks, first_stage=hybrid, scorer=scorer, depth=VARIANTS[c.variant]["depth"])
          for c in BENCH_CONFIGS[6:]),
    ))}
    _time_method(bm25, "scores", clock, "BM25")
    _time_method(dense, "scores", clock, "dense scoring (incl. question embedding)")

    for q in questions[:WARMUP]:
        for r in retrievers.values():
            r.search(q, 10)
    rng = random.Random(SEED)
    per_question: dict[str, list[float]] = {name: [] for name in retrievers}
    steps: dict[str, dict[str, float]] = {name: {} for name in retrievers}
    cpu: dict[str, float] = {name: 0.0 for name in retrievers}
    for _ in range(PASSES):
        order = list(retrievers)
        rng.shuffle(order)
        for name in order:
            cpu_start = time.process_time()
            for q in questions:
                clock.clear()
                start = time.perf_counter()
                retrievers[name].search(q, 10)
                elapsed = time.perf_counter() - start
                per_question[name].append(elapsed)
                for step, seconds in clock.items():
                    steps[name][step] = steps[name].get(step, 0.0) + seconds
                steps[name]["total"] = steps[name].get("total", 0.0) + elapsed
            cpu[name] += time.process_time() - cpu_start
    calls = len(questions) * PASSES
    out = {}
    for name, seconds in per_question.items():
        s = {k: v / calls * 1000 for k, v in steps[name].items()}
        dense_total = s.get("dense scoring (incl. question embedding)", 0.0)
        out[name] = {
            **_percentiles(seconds),
            "cpu_seconds_per_1000_questions": cpu[name] / calls * 1000,
            "steps_mean_ms": {
                "question embedding": s.get("question embedding", 0.0),
                "dense scoring": max(dense_total - s.get("question embedding", 0.0), 0.0),
                "BM25": s.get("BM25", 0.0),
                "reranking": s.get("reranking", 0.0),
                "fusion, ordering and the rest": max(s["total"] - dense_total - s.get("BM25", 0.0)
                                                     - s.get("reranking", 0.0), 0.0),
            },
        }
    batched = []
    for q in questions:
        texts = [h.chunk.index_text(True) for h in hybrid.search(q, neural.RERANK_DEPTH)]
        start = time.perf_counter()
        raw_scorer._load().predict([(q, t) for t in texts], batch_size=len(texts), convert_to_numpy=True,
                                    show_progress_bar=False)
        batched.append(time.perf_counter() - start)
    return {
        "configurations": out,
        "index_build_seconds": build,
        "rerank_100_pairs_in_one_batch": _percentiles(batched),
        "chunks": len(chunks),
        "index_mb": {"dense vectors (float32)": len(chunks) * 384 * 4 / 2**20,
                     "dense matrix as held (float64, distinct texts)": dense.matrix.nbytes / 2**20},
    }


_COLD = """
import time, json, sys
t0 = time.perf_counter()
from raglab import neural
from raglab.corpus import CorpusLock, default_corpus_dir, default_lock_path, load_corpus
import torch, sentence_transformers  # noqa
t1 = time.perf_counter()
lock = neural.model_lock()
e = neural.SentenceTransformerEmbedder(lock["embedder"]); e._load()
c = neural.CrossEncoderScorer(lock["reranker"]); c._load()
t2 = time.perf_counter()
q = sys.argv[1]
v = e.embed([q], neural.QUERY_INSTRUCTION)
s = c.score(q, ["FastAPI validates the request body with a Pydantic model."])
t3 = time.perf_counter()
print(json.dumps({"import_s": t1 - t0, "load_s": t2 - t1, "first_question_s": t3 - t2}))
"""


def cold_start(stage_dir: Path, question: str, runs: int = COLD_RUNS) -> dict:
    """Import, model loading and a first question, each in a fresh Python process."""
    samples = []
    for _ in range(runs):
        out = subprocess.run([sys.executable, "-c", _COLD, question], cwd=stage_dir, capture_output=True, text=True,
                             timeout=600, check=True)
        samples.append(json.loads(out.stdout.strip().splitlines()[-1]))
    return {k: {"median_s": median(s[k] for s in samples), "max_s": max(s[k] for s in samples)}
            for k in samples[0]} | {"runs": runs}


def bench(docs, questions: Sequence[str], stage_dir: Path) -> dict:
    from raglab import neural

    lock = neural.model_lock()
    warm = warm_timing(docs, questions)
    return {
        "machine": machine(),
        "threads": neural.THREADS,
        "libraries": neural.library_versions(),
        "protocol": {"warmup_questions": WARMUP, "passes": PASSES, "questions": len(questions),
                     "order": f"retrievers shuffled per pass, seed {SEED}", "caches": "off", "batch_size": 1,
                     "cold_runs": COLD_RUNS},
        "model_mb": {role: sum(f.bytes for f in lock[role].files.values()) / 2**20 for role in ("embedder", "reranker")},
        "cold_start": cold_start(stage_dir, questions[0]),
        **warm,
        # macOS reports ru_maxrss in bytes (Linux in kilobytes); a whole-process total, not per retriever
        "peak_rss_mb_whole_process": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / (2**20 if platform.system() == "Darwin" else 2**10),
    }
