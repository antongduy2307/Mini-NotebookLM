"""In-memory BM25 retrieval rebuilt from SQLite chunks."""

from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

from mini_notebooklm_rag.retrieval.models import SparseCandidate
from mini_notebooklm_rag.storage.repositories import ChunkRecord

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def tokenize(text: str) -> list[str]:
    """Return deterministic English-first BM25 tokens."""
    return [match.group(0).lower() for match in _TOKEN_RE.finditer(text)]


class BM25Store:
    """In-memory sparse retrieval store for one workspace chunk corpus."""

    def __init__(self, chunks: list[ChunkRecord]):
        self.chunks = chunks
        self._tokenized_corpus = [tokenize(chunk.text) for chunk in chunks]
        self._bm25 = BM25Okapi(self._tokenized_corpus) if chunks else None

    @classmethod
    def from_chunks(cls, chunks: list[ChunkRecord]) -> BM25Store:
        return cls(chunks)

    def search(
        self,
        query: str,
        top_k: int,
        selected_document_ids: set[str] | None = None,
    ) -> list[SparseCandidate]:
        if self._bm25 is None or top_k <= 0:
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)
        candidates: list[tuple[ChunkRecord, float]] = []
        query_token_set = set(query_tokens)
        for chunk, tokens, score in zip(self.chunks, self._tokenized_corpus, scores, strict=False):
            if selected_document_ids and chunk.document_id not in selected_document_ids:
                continue
            if not query_token_set.intersection(tokens):
                continue
            candidates.append((chunk, float(score)))

        candidates.sort(key=lambda item: (-item[1], item[0].document_id, item[0].chunk_index))
        return [
            SparseCandidate(
                chunk_id=chunk.id,
                document_id=chunk.document_id,
                score=score,
                rank=index + 1,
            )
            for index, (chunk, score) in enumerate(candidates[:top_k])
        ]
