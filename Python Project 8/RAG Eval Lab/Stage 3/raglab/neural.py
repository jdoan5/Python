"""Stage 3's retrievers: dense embeddings, hybrid fusion and cross-encoder reranking.

- dense: BAAI/bge-small-en-v1.5 embeds every chunk, and each question with the
  model card's retrieval instruction. The score is cosine similarity, and every
  chunk is ranked.
- hybrid: reciprocal rank fusion (RRF) of bm25 and dense over the same chunks.
  A chunk's score is the sum of 1 / (60 + rank) over the two rankings. BM25's
  ranking holds only the chunks it scores above 0; dense ranks them all. Tied
  input scores share the best rank (competition ranking).
- hybrid+rerank (and the bm25+rerank control): the MS MARCO MiniLM
  cross-encoder re-scores the first stage's top 100 chunks, given (question,
  chunk text as indexed). They are re-ordered by its score, ties keeping
  first-stage order; ranks 101 onwards keep first-stage order.

Exact ties in dense or fused scores are broken by a seeded hash of (question,
chunk id): arbitrary but fixed, and neutral between pages, where corpus order
(alphabetical by page) would systematically favour some docs areas.

Both models are pinned in models.lock.json (models.py) and run on the CPU in
float32 with 4 threads. Every text is embedded on its own (a batch of one),
and every (question, chunk) pair is scored on its own, so a number never
depends on what else was computed with it. Each distinct text is computed
once, so chunks with identical text get bit-identical scores.

That makes caching per text exact. Vectors and scores are cached under
data/cache, one file per setting: everything that decides a number (the
model's fingerprint, role, dtype, device, threads, batch size, token limit,
library versions and platform). Each entry is keyed by the text and its
instruction. A model is loaded only when a number isn't cached yet.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import tempfile
import warnings
import zipfile
from importlib.metadata import version
from pathlib import Path
from typing import Callable, Mapping, Protocol, Sequence

import numpy as np

from raglab.chunking import Chunk
from raglab.corpus import project_root
from raglab.models import ModelError, ModelLock, ModelSpec, default_models_lock_path, ensure_not_synced, require
from raglab.retrieval import Bm25Retriever, Hit, RetrievalError

QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
RRF_K = 60                         # Cormack, Clarke & Büttcher (2009); not tuned here
RERANK_DEPTH = 100                 # the first stage's top 100 are re-scored: the common deployment choice
TIE_SEED = "20261120"              # seeds the hash that orders exactly tied scores
THREADS = 4                        # fixed, so the numbers don't depend on how many cores a machine has
MAX_TOKENS = 512                   # both models' limit; a longer input loses the end of its passage
_LIBRARIES = ("torch", "transformers", "sentence-transformers", "tokenizers", "safetensors")


class Embedder(Protocol):
    settings: Mapping[str, object]  # everything that decides a vector

    def embed(self, texts: Sequence[str], instruction: str) -> np.ndarray: ...  # (n, dim) float32, unit length


class PairScorer(Protocol):
    settings: Mapping[str, object]  # everything that decides a score

    def score(self, query: str, texts: Sequence[str]) -> np.ndarray: ...  # (n,) float32


# --- caches ---------------------------------------------------------------------------------------


def _canonical(settings: Mapping[str, object]) -> str:
    return json.dumps(settings, sort_keys=True, separators=(",", ":"))


def _key(*parts: str) -> str:
    return hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()


class CacheError(Exception):
    """A cache can't be used where it is."""


class _Store:
    """float32 rows keyed by SHA-256 hex digests, in one .npz file that also holds its settings.

    Read without pickle, and checked on load: the stored settings must match,
    keys must be unique, and rows must be finite float32 (and unit length, for
    embeddings). A file that fails is moved aside with a warning and rebuilt,
    never trusted. Writes go to a unique temporary file in the same folder,
    are flushed to disk, then replace the old file in one step.
    """

    def __init__(self, path: Path | None, settings: str, unit_rows: bool = False) -> None:
        self.path = path
        self.settings = settings
        self.unit_rows = unit_rows
        self._rows: dict[str, np.ndarray] = {}
        self._unsaved = 0
        if path is not None:
            try:
                ensure_not_synced(path.parent, "cache")
            except ModelError as e:
                raise CacheError(str(e)) from None
            if path.is_file():
                self._load(path)

    def _load(self, path: Path) -> None:
        try:
            with np.load(path, allow_pickle=False) as z:
                stored, keys, rows = str(z["settings"][()]), z["keys"], z["rows"]
            if stored != self.settings:
                raise ValueError("written under different settings")
            if keys.ndim != 1 or rows.ndim != 2 or len(keys) != len(rows) or rows.dtype != np.float32:
                raise ValueError("keys and rows don't line up")
            if len(set(keys.tolist())) != len(keys) or not np.isfinite(rows).all():
                raise ValueError("duplicate keys or non-finite values")
            if self.unit_rows and len(rows) and not np.allclose(np.linalg.norm(rows, axis=1), 1.0, atol=1e-4):
                raise ValueError("vectors are not unit length")
            self._rows = {str(k): rows[i] for i, k in enumerate(keys)}
        except (OSError, ValueError, KeyError, zipfile.BadZipFile) as e:
            aside = path.with_name(path.name + ".corrupt")
            path.replace(aside)
            warnings.warn(f"cache {path.name} was unusable ({e}); moved it to {aside.name} and starting afresh",
                          RuntimeWarning, stacklevel=3)
            self._rows = {}

    def __len__(self) -> int:
        return len(self._rows)

    def get(self, key: str) -> np.ndarray | None:
        return self._rows.get(key)

    def put(self, key: str, row: np.ndarray) -> None:
        row = np.asarray(row, dtype=np.float32).reshape(-1)
        if not np.isfinite(row).all():
            raise ValueError("refusing to cache a non-finite value")
        self._rows[key] = row
        self._unsaved += 1

    def save(self) -> None:
        if self.path is None or not self._unsaved:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        widths = {r.shape[0] for r in self._rows.values()}
        if len(widths) != 1:
            raise ValueError(f"cache rows must share one width, found {sorted(widths)}")
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=self.path.stem + ".", suffix=".tmp.npz")
        try:
            with os.fdopen(fd, "wb") as f:
                np.savez(f, settings=np.array(self.settings), keys=np.array(list(self._rows), dtype="U64"),
                         rows=np.stack(list(self._rows.values())))
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        self._unsaved = 0


def _store_path(cache_dir: Path | None, kind: str, settings: str) -> Path | None:
    if cache_dir is None:
        return None
    return cache_dir / f"{kind}-{hashlib.sha256(settings.encode('utf-8')).hexdigest()[:16]}.npz"


class CachedEmbedder:
    """An Embedder whose vectors are computed once per (instruction, text), then read from disk."""

    def __init__(self, embedder: Embedder, cache_dir: Path | None) -> None:
        self.embedder = embedder
        self.settings = dict(embedder.settings)
        canonical = _canonical(self.settings)
        self._store = _Store(_store_path(cache_dir, "embeddings", canonical), canonical, unit_rows=True)

    def embed(self, texts: Sequence[str], instruction: str = "") -> np.ndarray:
        keys = [_key(instruction, t) for t in texts]
        todo = {k: t for k, t in zip(keys, texts) if self._store.get(k) is None}
        if todo:
            order = sorted(todo)
            vectors = self.embedder.embed([todo[k] for k in order], instruction)
            for k, v in zip(order, vectors):
                self._store.put(k, v)
            self._store.save()
        if not keys:
            return np.zeros((0, 0), dtype=np.float32)
        return np.stack([self._store.get(k) for k in keys])


class CachedScorer:
    """A PairScorer whose scores are computed once per (question, text), then read from disk."""

    def __init__(self, scorer: PairScorer, cache_dir: Path | None) -> None:
        self.scorer = scorer
        self.settings = dict(scorer.settings)
        canonical = _canonical(self.settings)
        self._store = _Store(_store_path(cache_dir, "rerank", canonical), canonical)

    def score(self, query: str, texts: Sequence[str]) -> np.ndarray:
        keys = [_key(query, t) for t in texts]
        todo = {k: t for k, t in zip(keys, texts) if self._store.get(k) is None}
        if todo:
            order = sorted(todo)
            scores = self.scorer.score(query, [todo[k] for k in order])
            for k, s in zip(order, scores):
                self._store.put(k, np.array([s], dtype=np.float32))
            self._store.save()
        return np.array([self._store.get(k)[0] for k in keys], dtype=np.float32)


# --- the pinned models --------------------------------------------------------------------------


class _DropRedirectNote(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "Redirects are currently not supported" not in record.getMessage()


def _import_torch():
    """torch with a fixed thread count, and the Hugging Face libraries without progress bars or telemetry."""
    for name, value in (("TOKENIZERS_PARALLELISM", "false"), ("HF_HUB_DISABLE_TELEMETRY", "1"),
                        ("HF_HUB_DISABLE_PROGRESS_BARS", "1")):
        os.environ.setdefault(name, value)
    # torch.distributed logs a macOS-only note on import; it says nothing about this project.
    logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").addFilter(_DropRedirectNote())
    import torch
    from transformers.utils import logging as hf_logging

    torch.set_num_threads(THREADS)
    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()
    return torch


def library_versions() -> dict:
    """Read from the installed packages' metadata, so nothing has to be imported."""
    return {name: version(name) for name in _LIBRARIES}


def model_settings(spec: ModelSpec, role: str) -> dict:
    """Everything that decides a model's numbers: the cache key, and the run manifest's record."""
    settings = {
        "role": role, "repo": spec.repo, "revision": spec.revision, "fingerprint": spec.fingerprint,
        "device": "cpu", "dtype": "float32", "threads": THREADS, "batch_size": 1, "max_tokens": MAX_TOKENS,
        "platform": f"{platform.system()} {platform.machine()}", **library_versions(),
    }
    if role == "embedder":
        settings.update(pooling="the model's own (CLS)", normalized=True)
    else:
        settings.update(activation="none (raw logits)", truncation="the passage's end, never the question")
    return settings


def check_weights(module, weights: Path) -> None:
    """Every tensor the module holds must be the file's tensor, bit for bit, and none may be missing.

    A model that silently initialised a layer at random would still give
    bit-identical outputs run to run, so determinism alone can't catch it.
    """
    import torch
    from safetensors.torch import load_file

    stored = load_file(str(weights))
    held = module.state_dict()
    missing = sorted(set(held) - set(stored))
    unexpected = sorted(k for k in set(stored) - set(held) if not k.endswith("position_ids"))
    differ = sorted(k for k in set(held) & set(stored) if not torch.equal(held[k], stored[k].to(held[k].dtype)))
    if missing or unexpected or differ:
        raise ModelError(f"{weights.parent.name}: the loaded weights don't match the file "
                         f"(missing {missing[:3]}, unexpected {unexpected[:3]}, different {differ[:3]})")


def _check_loaded(model, torch, what: str) -> None:
    parameter = next(model.parameters())
    if parameter.device.type != "cpu" or parameter.dtype != torch.float32 or model.training:
        raise ModelError(f"{what} loaded on {parameter.device} as {parameter.dtype}"
                         f"{' in training mode' if model.training else ''}; expected the CPU, float32, eval mode")


class SentenceTransformerEmbedder:
    """The pinned bi-encoder, loaded on first use from its verified folder."""

    def __init__(self, spec: ModelSpec, folder: Callable[[ModelSpec], Path] = require) -> None:
        self.spec = spec
        self.settings = model_settings(spec, "embedder")
        self._folder = folder
        self._model = None

    def _load(self):
        if self._model is None:
            torch = _import_torch()
            from sentence_transformers import SentenceTransformer

            folder = self._folder(self.spec)
            model = SentenceTransformer(str(folder), device="cpu", local_files_only=True, trust_remote_code=False,
                                        model_kwargs={"use_safetensors": True, "dtype": torch.float32})
            model.eval()
            if model.max_seq_length != MAX_TOKENS:
                raise ModelError(f"{self.spec.repo}: expected a {MAX_TOKENS}-token limit, got {model.max_seq_length}")
            _check_loaded(model, torch, self.spec.repo)
            check_weights(model[0].auto_model, folder / "model.safetensors")
            self._model = model
        return self._model

    def embed(self, texts: Sequence[str], instruction: str) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        vectors = self._load().encode(list(texts), prompt=instruction or None, batch_size=1, normalize_embeddings=True,
                                      convert_to_numpy=True, show_progress_bar=False)
        return np.asarray(vectors, dtype=np.float32)


class CrossEncoderScorer:
    """The pinned cross-encoder, loaded on first use from its verified folder."""

    def __init__(self, spec: ModelSpec, folder: Callable[[ModelSpec], Path] = require) -> None:
        self.spec = spec
        self.settings = model_settings(spec, "reranker")
        self._folder = folder
        self._model = None

    def _load(self):
        if self._model is None:
            torch = _import_torch()
            from sentence_transformers import CrossEncoder

            folder = self._folder(self.spec)
            model = CrossEncoder(str(folder), device="cpu", local_files_only=True, trust_remote_code=False,
                                 max_length=MAX_TOKENS, activation_fn=torch.nn.Identity(),
                                 model_kwargs={"use_safetensors": True, "dtype": torch.float32})
            model.model.eval()
            _check_loaded(model.model, torch, self.spec.repo)
            if model.tokenizer.truncation_side != "right":
                raise ModelError(f"{self.spec.repo}: expected truncation from the right")
            check_weights(model.model, folder / "model.safetensors")
            self._model = model
        return self._model

    def score(self, query: str, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros(0, dtype=np.float32)
        scores = self._load().predict([(query, t) for t in texts], batch_size=1, convert_to_numpy=True,
                                      show_progress_bar=False)
        return np.asarray(scores, dtype=np.float32).reshape(-1)


def default_cache_dir() -> Path:
    return project_root() / "data" / "cache"


def model_lock() -> ModelLock:
    """models.lock.json, read fresh each time (tests replace this to avoid the real models)."""
    return ModelLock.load(default_models_lock_path())


_DEFAULTS: dict[str, object] = {}


def default_embedder() -> CachedEmbedder:
    """The process's one embedder; its model loads only when a vector isn't cached."""
    if "embedder" not in _DEFAULTS:
        _DEFAULTS["embedder"] = CachedEmbedder(SentenceTransformerEmbedder(model_lock()["embedder"]), default_cache_dir())
    return _DEFAULTS["embedder"]


def default_scorer() -> CachedScorer:
    """The process's one reranker; its model loads only when a score isn't cached."""
    if "reranker" not in _DEFAULTS:
        _DEFAULTS["reranker"] = CachedScorer(CrossEncoderScorer(model_lock()["reranker"]), default_cache_dir())
    return _DEFAULTS["reranker"]


# --- retrievers ---------------------------------------------------------------------------------


def competition_ranks(scores: np.ndarray, members: np.ndarray) -> np.ndarray:
    """For each member: 1 + how many members score strictly higher, so tied scores share the best rank."""
    values = scores[members]
    descending = np.sort(values)[::-1]
    return 1 + np.searchsorted(-descending, -values, side="left")


def tie_order(scores: np.ndarray, query: str, chunks: Sequence[Chunk]) -> np.ndarray:
    """Every chunk index, best score first; exactly tied scores ordered by sha256(seed:question:chunk id)."""
    order = np.argsort(-scores, kind="stable")
    ranked = scores[order]
    starts = np.concatenate(([0], np.flatnonzero(ranked[1:] != ranked[:-1]) + 1))
    ends = np.concatenate((starts[1:], [len(order)]))
    for a, b in zip(starts[ends - starts > 1], ends[ends - starts > 1]):
        run = order[a:b]
        keys = [hashlib.sha256(f"{TIE_SEED}:{query}:{chunks[j].chunk_id}".encode("utf-8")).hexdigest() for j in run]
        order[a:b] = run[np.argsort(keys, kind="stable")]
    return order


def _hits(order: np.ndarray, scores: np.ndarray, chunks: Sequence[Chunk]) -> list[Hit]:
    return [Hit(rank=i + 1, score=float(scores[j]), chunk=chunks[j]) for i, j in enumerate(order)]


class DenseRetriever:
    name = "dense"

    def __init__(
        self,
        chunks: Sequence[Chunk],
        with_context: bool = True,
        embedder: CachedEmbedder | None = None,
        instruction: str = QUERY_INSTRUCTION,
    ) -> None:
        if not chunks:
            raise RetrievalError("no chunks to index")
        self.chunks = list(chunks)
        self.with_context = with_context
        self.instruction = instruction
        self.embedder = embedder if embedder is not None else default_embedder()
        texts = [c.index_text(with_context) for c in self.chunks]
        distinct = list(dict.fromkeys(texts))
        position = {t: i for i, t in enumerate(distinct)}
        self._inverse = np.array([position[t] for t in texts])
        # float64 dot products of float32 vectors: exact products, so ranking never hinges on BLAS kernels
        self.matrix = self.embedder.embed(distinct, "").astype(np.float64)

    def scores(self, query: str) -> np.ndarray:
        q = self.embedder.embed([query], self.instruction)[0].astype(np.float64)
        return (self.matrix @ q)[self._inverse]

    def search(self, query: str, k: int = 10) -> list[Hit]:
        """Every chunk, best first; exact ties (identical texts) ordered by the seeded hash."""
        scores = self.scores(query)
        return _hits(tie_order(scores, query, self.chunks)[: max(k, 0)], scores, self.chunks)


class HybridRetriever:
    name = "hybrid"

    def __init__(
        self,
        chunks: Sequence[Chunk],
        with_context: bool = True,
        lexical: Bm25Retriever | None = None,
        dense: DenseRetriever | None = None,
        k: int = RRF_K,
    ) -> None:
        if not chunks:
            raise RetrievalError("no chunks to index")
        self.chunks = list(chunks)
        self.with_context = with_context
        self.lexical = lexical if lexical is not None else Bm25Retriever(self.chunks, with_context)
        self.dense = dense if dense is not None else DenseRetriever(self.chunks, with_context)
        ids = [c.chunk_id for c in self.chunks]
        for part in (self.lexical, self.dense):
            if [c.chunk_id for c in part.chunks] != ids or part.with_context != with_context:
                raise ValueError("a hybrid's two retrievers must index the same chunks the same way")
        self.k = k

    def fused(self, query: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(fused score, dense rank, BM25 rank or inf) for every chunk, ranks with ties sharing the best."""
        n = len(self.chunks)
        lexical, dense = self.lexical.scores(query), self.dense.scores(query)
        listed = np.flatnonzero(lexical > 0)  # BM25's ranking holds only the chunks it scores above 0
        lexical_rank = np.full(n, np.inf)
        lexical_rank[listed] = competition_ranks(lexical, listed)
        dense_rank = competition_ranks(dense, np.arange(n)).astype(float)
        score = 1.0 / (self.k + dense_rank) + np.where(np.isfinite(lexical_rank), 1.0 / (self.k + lexical_rank), 0.0)
        return score, dense_rank, lexical_rank

    def scores(self, query: str) -> np.ndarray:
        return self.fused(query)[0]

    def search(self, query: str, k: int = 10) -> list[Hit]:
        """Every chunk, best first. A chunk ranked (a, b) and one ranked (b, a) tie exactly; the seeded hash
        orders them, so neither input is favoured."""
        score = self.fused(query)[0]
        return _hits(tie_order(score, query, self.chunks)[: max(k, 0)], score, self.chunks)


class RerankRetriever:
    """A first stage's top `depth` chunks, re-ordered by the cross-encoder; the rest in first-stage order.

    Ranks are the authority: scores below the depth are the first stage's, on
    its own scale, and are never compared with the cross-encoder's.
    """

    def __init__(
        self,
        chunks: Sequence[Chunk],
        with_context: bool = True,
        first_stage=None,
        scorer: CachedScorer | None = None,
        depth: int = RERANK_DEPTH,
    ) -> None:
        if not chunks:
            raise RetrievalError("no chunks to index")
        if depth < 1:
            raise ValueError("rerank depth must be at least 1")
        self.chunks = list(chunks)
        self.with_context = with_context
        self.first_stage = first_stage if first_stage is not None else HybridRetriever(self.chunks, with_context)
        if [c.chunk_id for c in self.first_stage.chunks] != [c.chunk_id for c in self.chunks] \
                or self.first_stage.with_context != with_context:
            raise ValueError("the reranker's first stage must index the same chunks the same way")
        self.scorer = scorer if scorer is not None else default_scorer()
        self.depth = depth
        self.name = f"{self.first_stage.name}+rerank"

    def depth_for(self, first: Sequence[Hit]) -> int:
        """How many of the first stage's chunks get re-scored for this question."""
        return min(self.depth, len(first))

    def search(self, query: str, k: int = 10) -> list[Hit]:
        first = self.first_stage.search(query, len(self.chunks))
        depth = self.depth_for(first)
        head, tail = first[:depth], first[depth:]
        if head:
            ce = self.scorer.score(query, [h.chunk.index_text(self.with_context) for h in head]).astype(np.float64)
            order = sorted(range(len(head)), key=lambda i: (-ce[i], i))  # stable: ties keep first-stage order
            head = [Hit(rank=0, score=float(ce[i]), chunk=head[i].chunk) for i in order]
        return [Hit(rank=i + 1, score=h.score, chunk=h.chunk) for i, h in enumerate((head + tail)[: max(k, 0)])]


def build_neural(name: str, chunks: Sequence[Chunk], with_context: bool = True):
    """A neural retriever built on its own (no shared components): for `raglab search` and tests."""
    if name == "dense":
        return DenseRetriever(chunks, with_context)
    if name == "hybrid":
        return HybridRetriever(chunks, with_context)
    if name == "bm25+rerank":
        return RerankRetriever(chunks, with_context, first_stage=Bm25Retriever(chunks, with_context))
    if name == "hybrid+rerank":
        return RerankRetriever(chunks, with_context, first_stage=HybridRetriever(chunks, with_context))
    raise ValueError(f"unknown neural retriever {name!r}")


NEURAL_NAMES = ("dense", "hybrid", "bm25+rerank", "hybrid+rerank")


def neural_config(name: str, variant: Mapping[str, object] | None = None) -> dict:
    """Parameters for run manifests: everything that decides a neural retriever's ranking."""
    variant = dict(variant or {})
    lock = model_lock()
    ties = f"exact ties ordered by sha256('{TIE_SEED}:<question>:<chunk id>')"
    embedder = {"model": model_settings(lock["embedder"], "embedder"),
                "query_instruction": variant.get("instruction", QUERY_INSTRUCTION), "document_instruction": "",
                "similarity": "cosine (float64 dot product of unit float32 vectors)", "ranks": "every chunk",
                "ties": ties}
    fusion = {"fusion": "reciprocal rank", "k": RRF_K, "inputs": ["bm25", "dense"],
              "input_ranks": "competition ranking (tied scores share the best rank); BM25 lists only chunks it "
                             "scores above 0, dense lists every chunk",
              "ties": ties, "ranks": "every chunk"}

    def rerank(first_stage: str) -> dict:
        return {"first_stage": first_stage, "model": model_settings(lock["reranker"], "reranker"),
                "depth": variant.get("depth", RERANK_DEPTH),
                "input": "(question, chunk text as indexed)", "order": "cross-encoder score; ties keep first-stage "
                "order; chunks below the depth keep first-stage order"}

    if name == "dense":
        return {"name": name, **embedder}
    if name == "hybrid":
        return {"name": name, **fusion, "dense": embedder}
    if name == "bm25+rerank":
        return {"name": name, **rerank("bm25"), "ranks": "every chunk BM25 scores above 0"}
    if name == "hybrid+rerank":
        return {"name": name, **rerank("hybrid"), "hybrid": {**fusion, "dense": embedder}, "ranks": "every chunk"}
    raise ValueError(f"unknown neural retriever {name!r}; choose from {', '.join(NEURAL_NAMES)}")
