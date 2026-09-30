"""Retrieval layer over pipeline chunks.

Starts as a dependency-free lexical retriever (IDF-weighted term overlap
with Persian normalization so query terms match OCR text). The interface is
deliberately retriever-shaped so an embedding backend can be dropped in
later behind the same `search()` contract.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

from darsio_ingest.models import Chunk
from darsio_ingest.normalize import normalize_text


def tokenize(text: str) -> list[str]:
    """Normalized word tokens for matching (Persian-aware)."""
    normalized = normalize_text(text)
    return [t for t in normalized.split() if len(t) >= 2]


@dataclass
class RetrievalHit:
    chunk: Chunk
    score: float

    def to_dict(self) -> dict:
        return {
            "score": round(self.score, 4),
            "chunk": self.chunk.to_dict(),
        }


@dataclass
class LexicalRetriever:
    chunks: list[Chunk]
    _token_counts: list[Counter] = field(init=False)
    _idf: dict[str, float] = field(init=False)

    def __post_init__(self) -> None:
        self._token_counts = [Counter(tokenize(c.text)) for c in self.chunks]
        doc_freq: Counter = Counter()
        for counts in self._token_counts:
            doc_freq.update(counts.keys())
        total = max(len(self.chunks), 1)
        self._idf = {
            term: math.log((total + 1) / (freq + 0.5))
            for term, freq in doc_freq.items()
        }

    def score_chunk(self, query_tokens: list[str], index: int) -> float:
        counts = self._token_counts[index]
        if not counts:
            return 0.0
        score = 0.0
        for token in query_tokens:
            if token not in counts:
                continue
            tf = counts[token]
            idf = self._idf.get(token, math.log((len(self.chunks) + 1) / 0.5))
            # Sublinear TF damping.
            score += (1.0 + math.log(tf)) * idf
        return score

    def search(self, query: str, k: int = 4) -> list[RetrievalHit]:
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        scored = [
            (self.score_chunk(query_tokens, i), i)
            for i in range(len(self.chunks))
        ]
        scored = [(s, i) for s, i in scored if s > 0]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [
            RetrievalHit(chunk=self.chunks[i], score=score)
            for score, i in scored[:k]
        ]
