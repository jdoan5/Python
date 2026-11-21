"""How many inputs the two models cut at 512 tokens (PLAN.md, "Fixed settings"). Reads no grade.

For the embedder: chunk texts as indexed, per chunker and breadcrumb setting.
For the cross-encoder: every (question, chunk) pair it re-scores, that is each
question's hybrid top 100 in every setting and BM25's top 100 for the control,
counted with the question included, because both share the 512 tokens. The
model's tokenizers (from the verified model folders) do the counting, so the
counts are the models' own.

    cd "Stage 3" && ../.venv/bin/python analysis/truncation.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tokenizers import Tokenizer  # noqa: E402

from raglab.chunking import CHUNKERS, chunk_corpus, get_chunker  # noqa: E402
from raglab.corpus import CorpusLock, default_corpus_dir, default_lock_path, load_corpus  # noqa: E402
from raglab.experiments import CONTROL, build_retrievers, RetrievalConfig  # noqa: E402
from raglab.golden import default_golden_path, load_golden  # noqa: E402
from raglab.models import require  # noqa: E402
from raglab.neural import MAX_TOKENS, QUERY_INSTRUCTION, RERANK_DEPTH, model_lock  # noqa: E402


def tokenizer(role: str) -> Tokenizer:
    t = Tokenizer.from_file(str(require(model_lock()[role]) / "tokenizer.json"))
    t.no_truncation()
    t.no_padding()
    return t


def count(docs, questions: list[str]) -> dict:
    embed_tok, pair_tok = tokenizer("embedder"), tokenizer("reranker")
    out = {"limit_tokens": MAX_TOKENS, "embedder": {}, "reranker_pairs": {}}
    for chunker in CHUNKERS:
        chunks = chunk_corpus(docs, get_chunker(chunker))
        for with_context in (True, False):
            setting = f"{chunker}, breadcrumb {'on' if with_context else 'off'}"
            lengths = [len(embed_tok.encode(c.index_text(with_context)).ids) for c in chunks]
            out["embedder"][setting] = {"chunks": len(lengths), "over_limit": sum(n > MAX_TOKENS for n in lengths),
                                        "max_tokens": max(lengths)}
    queries = [len(embed_tok.encode(QUERY_INSTRUCTION + q).ids) for q in questions]
    out["embedder"]["questions (with the instruction)"] = {"questions": len(queries), "over_limit": sum(
        n > MAX_TOKENS for n in queries), "max_tokens": max(queries)}
    configs = [RetrievalConfig(chunker, "hybrid", with_context) for chunker in CHUNKERS for with_context in (True, False)]
    configs.append(RetrievalConfig(CONTROL.chunker, "bm25", True, label="BM25's top 100 (the control's first stage)"))
    retrievers = build_retrievers(docs, tuple(configs))
    for config in configs:
        retriever = retrievers[config.name]
        pairs = over = 0
        longest = 0
        for q in questions:
            for hit in retriever.search(q, RERANK_DEPTH):
                n = len(pair_tok.encode(q, hit.chunk.index_text(config.with_context)).ids)
                pairs += 1
                over += n > MAX_TOKENS
                longest = max(longest, n)
        out["reranker_pairs"][config.name] = {"pairs": pairs, "over_limit": over, "max_tokens": longest}
    return out


def main() -> None:
    lock = CorpusLock.load(default_lock_path())
    docs = list(load_corpus(default_corpus_dir(lock.ref), lock=lock).documents)
    questions = [q.text for q in load_golden(default_golden_path()).questions]
    result = count(docs, questions)
    path = Path(__file__).resolve().parents[1] / "results" / "truncation.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    for model in ("embedder", "reranker_pairs"):
        for setting, c in result[model].items():
            total = c.get("chunks") or c.get("pairs") or c.get("questions")
            print(f"{model:15} {setting:45} {c['over_limit']:5} of {total:6} over {MAX_TOKENS} tokens "
                  f"(longest {c['max_tokens']})")


if __name__ == "__main__":
    main()
