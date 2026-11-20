"""Stage 3's retrievers: dense embeddings, hybrid fusion and cross-encoder reranking.

- dense: BAAI/bge-small-en-v1.5 embeds every chunk, and each question with the
  model card's retrieval instruction. The score is their cosine similarity.
- hybrid: reciprocal rank fusion (RRF) of bm25 and dense over the same chunks.
  A chunk's score is the sum of 1 / (60 + rank) over the two rankings; a chunk
  BM25 scores 0 gets nothing from BM25.
- hybrid+rerank: the MS MARCO MiniLM cross-encoder re-scores the hybrid's top
  100 (question, chunk text as indexed). Ranks below 100 keep the hybrid's order.

Both models are pinned in models.lock.json (models.py) and run on the CPU.

Every text is embedded on its own (a batch of one), and every (question,
chunk) pair is scored on its own, so a number never depends on what else was
computed alongside it. That makes caching per text exact. Vectors and scores
are cached under data/cache, in a file per model setting. A setting is
everything that decides a number: the model's fingerprint, the library
versions and the thread count. Within a file, each entry is keyed by the text
and the instruction.
"""

from __future__ import annotations

import functools
import hashlib
import json
import logging
import os
import warnings
import zipfile
from pathlib import Path
from typing import ClassVar, Mapping, Protocol, Sequence

import numpy as np

from raglab.chunking import Chunk
from raglab.corpus import project_root
from raglab.models import ModelLock, ModelSpec, default_models_lock_path, require
from raglab.retrieval import Bm25Retriever, Hit, RetrievalError, _rank

QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
RRF_K = 60           # Cormack, Clarke & Büttcher (2009); not tuned here
RERANK_DEPTH = 100   # covers the 4.8k-character window for every chunker
THREADS = 4          # fixed, so the vectors don't depend on how many cores a machine has
MAX_TOKENS = 512     # both models' limit; longer texts are truncated


class Embedder(Protocol):
    settings: Mapping[str, object]  # everything that decides a vector

    def embed(self, texts: Sequence[str], instruction: str) -> np.ndarray: ...  # (n, dim) float32, unit length


class PairScorer(Protocol):
    settings: Mapping[str, object]  # everything that decides a score

    def score(self, query: str, texts: Sequence[str]) -> np.ndarray: ...  # (n,) float32


# --- caches ---------------------------------------------------------------------------------------


def _fingerprint(settings: Mapping[str, object]) -> str:
    return hashlib.sha256(json.dumps(settings, sort_keys=True).encode("utf-8")).hexdigest()


def _key(*parts: str) -> str:
    return hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()


class _Store:
    """float32 rows keyed by SHA-256 hex digests, kept in one .npz file and written atomically.

    The file is read without pickle. An unreadable or malformed file is moved
    aside with a warning and rebuilt, never trusted.
    """

    def __init__(self, path: Path | None) -> None:
        self.path = path
        self._rows: dict[str, np.ndarray] = {}
        self._unsaved = 0
        if path is not None and path.is_file():
            self._load(path)

    def _load(self, path: Path) -> None:
        try:
            with np.load(path, allow_pickle=False) as z:
                keys, rows = z["keys"], z["rows"]
            if keys.ndim != 1 or rows.ndim != 2 or len(keys) != len(rows) or rows.dtype != np.float32:
                raise ValueError("keys and rows don't line up")
            self._rows = {str(k): rows[i] for i, k in enumerate(keys)}
        except (OSError, ValueError, KeyError, zipfile.BadZipFile) as e:
            aside = path.with_name(path.name + ".corrupt")
            path.replace(aside)
            warnings.warn(f"cache {path.name} was unreadable ({type(e).__name__}); moved it to {aside.name}",
                          RuntimeWarning, stacklevel=3)
            self._rows = {}

    def __len__(self) -> int:
        return len(self._rows)

    def get(self, key: str) -> np.ndarray | None:
        return self._rows.get(key)

    def put(self, key: str, row: np.ndarray) -> None:
        self._rows[key] = np.asarray(row, dtype=np.float32).reshape(-1)
        self._unsaved += 1

    def save(self) -> None:
        if self.path is None or not self._unsaved:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        widths = {r.shape[0] for r in self._rows.values()}
        if len(widths) != 1:
            raise ValueError(f"cache rows must share one width, found {sorted(widths)}")
        tmp = self.path.with_name(self.path.stem + ".tmp.npz")
        np.savez(tmp, keys=np.array(list(self._rows), dtype="U64"), rows=np.stack(list(self._rows.values())))
        tmp.replace(self.path)
        self._unsaved = 0


class CachedEmbedder:
    """An Embedder whose vectors are computed once per (instruction, text) and then read from disk."""

    def __init__(self, embedder: Embedder, cache_dir: Path | None) -> None:
        self.embedder = embedder
        self.settings = dict(embedder.settings)
        name = f"embeddings-{_fingerprint(self.settings)[:16]}.npz"
        self._store = _Store(cache_dir / name if cache_dir is not None else None)

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
    """A PairScorer whose scores are computed once per (question, text) and then read from disk."""

    def __init__(self, scorer: PairScorer, cache_dir: Path | None) -> None:
        self.scorer = scorer
        self.settings = dict(scorer.settings)
        name = f"rerank-{_fingerprint(self.settings)[:16]}.npz"
        self._store = _Store(cache_dir / name if cache_dir is not None else None)

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
    """torch, quietly, with a fixed thread count."""
    # torch.distributed logs a macOS-only note on import; it says nothing about this project.
    logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").addFilter(_DropRedirectNote())
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import torch

    torch.set_num_threads(THREADS)
    from huggingface_hub.utils import disable_progress_bars
    from transformers.utils import logging as hf_logging

    disable_progress_bars()
    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()
    return torch


def _library_versions(torch) -> dict:
    import sentence_transformers
    import transformers

    return {"torch": torch.__version__, "transformers": transformers.__version__,
            "sentence-transformers": sentence_transformers.__version__}


class SentenceTransformerEmbedder:
    def __init__(self, spec: ModelSpec, directory: Path) -> None:
        torch = _import_torch()
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(str(directory), device="cpu", local_files_only=True,
                                         trust_remote_code=False, model_kwargs={"use_safetensors": True})
        self.model.max_seq_length = MAX_TOKENS
        self.dim = self.model.get_embedding_dimension()
        self.settings = {
            "model": spec.manifest(), "device": "cpu", "dtype": "float32", "threads": torch.get_num_threads(),
            "batch_size": 1, "max_tokens": MAX_TOKENS, "normalized": True, **_library_versions(torch),
        }

    def embed(self, texts: Sequence[str], instruction: str) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vectors = self.model.encode(list(texts), prompt=instruction or None, batch_size=1, normalize_embeddings=True,
                                    convert_to_numpy=True, show_progress_bar=False)
        return np.asarray(vectors, dtype=np.float32)


class CrossEncoderScorer:
    def __init__(self, spec: ModelSpec, directory: Path) -> None:
        torch = _import_torch()
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(str(directory), device="cpu", local_files_only=True,
                                  trust_remote_code=False, model_kwargs={"use_safetensors": True})
        self.model.max_seq_length = MAX_TOKENS
        self.settings = {
            "model": spec.manifest(), "device": "cpu", "dtype": "float32", "threads": torch.get_num_threads(),
            "batch_size": 1, "max_tokens": MAX_TOKENS, "activation": "none (raw logits)", **_library_versions(torch),
        }

    def score(self, query: str, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros(0, dtype=np.float32)
        scores = self.model.predict([(query, t) for t in texts], batch_size=1, convert_to_numpy=True,
                                    show_progress_bar=False)
        return np.asarray(scores, dtype=np.float32).reshape(-1)


def default_cache_dir() -> Path:
    return project_root() / "data" / "cache"


def model_lock() -> ModelLock:
    """models.lock.json, read fresh each time (tests replace this to avoid the real models)."""
    return ModelLock.load(default_models_lock_path())


def _spec(role: str) -> ModelSpec:
    return model_lock()[role]


@functools.lru_cache(maxsize=1)
def default_embedder() -> CachedEmbedder:
    spec = _spec("embedder")
    return CachedEmbedder(SentenceTransformerEmbedder(spec, require(spec)), default_cache_dir())


@functools.lru_cache(maxsize=1)
def default_scorer() -> CachedScorer:
    spec = _spec("reranker")
    return CachedScorer(CrossEncoderScorer(spec, require(spec)), default_cache_dir())


# --- retrievers ---------------------------------------------------------------------------------


class DenseRetriever:
    name: ClassVar[str] = "dense"

    def __init__(self, chunks: Sequence[Chunk], with_context: bool = True, embedder: CachedEmbedder | None = None) -> None:
        if not chunks:
            raise RetrievalError("no chunks to index")
        self.chunks = list(chunks)
        self.with_context = with_context
        self.embedder = embedder if embedder is not None else default_embedder()
        # float64 for the dot products, so ranking never depends on float32 summation order
        self.matrix = self.embedder.embed([c.index_text(with_context) for c in self.chunks], "").astype(np.float64)

    def scores(self, query: str) -> np.ndarray:
        q = self.embedder.embed([query], QUERY_INSTRUCTION)[0].astype(np.float64)
        return self.matrix @ q

    def search(self, query: str, k: int = 10) -> list[Hit]:
        return _rank(self.scores(query), self.chunks, k)


class HybridRetriever:
    name: ClassVar[str] = "hybrid"

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
        self._position = {chunk_id: i for i, chunk_id in enumerate(ids)}

    def scores(self, query: str) -> np.ndarray:
        fused = np.zeros(len(self.chunks))
        for part in (self.lexical, self.dense):
            for hit in part.search(query, len(self.chunks)):  # every chunk that retriever scores above 0
                fused[self._position[hit.chunk.chunk_id]] += 1.0 / (self.k + hit.rank)
        return fused

    def search(self, query: str, k: int = 10) -> list[Hit]:
        return _rank(self.scores(query), self.chunks, k)


class RerankRetriever:
    name: ClassVar[str] = "hybrid+rerank"

    def __init__(
        self,
        chunks: Sequence[Chunk],
        with_context: bool = True,
        first_stage: HybridRetriever | None = None,
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

    def search(self, query: str, k: int = 10) -> list[Hit]:
        """The first stage's top `depth`, re-ordered by the cross-encoder, then the rest in first-stage order.

        Scores below the rerank depth are the first stage's, on its own scale.
        Cross-encoder ties keep the first stage's order.
        """
        first = self.first_stage.search(query, len(self.chunks))
        head, tail = first[: self.depth], first[self.depth:]
        if head:
            ce = self.scorer.score(query, [h.chunk.index_text(self.with_context) for h in head]).astype(np.float64)
            head = [Hit(rank=0, score=float(ce[i]), chunk=head[i].chunk)
                    for i in sorted(range(len(head)), key=lambda i: (-ce[i], i))]
        return [Hit(rank=i + 1, score=h.score, chunk=h.chunk) for i, h in enumerate((head + tail)[: max(k, 0)])]


NEURAL_RETRIEVERS: dict[str, type] = {r.name: r for r in (DenseRetriever, HybridRetriever, RerankRetriever)}


def neural_config(name: str) -> dict:
    """Parameters for run manifests: everything that decides a neural retriever's scores."""
    lock = model_lock()
    if name == "dense":
        return {"name": name, "model": lock["embedder"].manifest(), "query_instruction": QUERY_INSTRUCTION,
                "document_instruction": "", "similarity": "cosine", "batch_size": 1, "threads": THREADS,
                "max_tokens": MAX_TOKENS, "device": "cpu"}
    if name == "hybrid":
        return {"name": name, "fusion": "reciprocal rank", "k": RRF_K, "inputs": ["bm25", "dense"],
                "ties": "corpus order"}
    if name == "hybrid+rerank":
        return {"name": name, "first_stage": "hybrid", "model": lock["reranker"].manifest(), "depth": RERANK_DEPTH,
                "input": "(question, chunk text as indexed)", "batch_size": 1, "threads": THREADS,
                "max_tokens": MAX_TOKENS, "device": "cpu", "ties": "first-stage order"}
    raise ValueError(f"unknown neural retriever {name!r}; choose from {sorted(NEURAL_RETRIEVERS)}")
