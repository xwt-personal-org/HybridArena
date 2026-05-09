"""Dependency-free retriever for telecom knowledge chunks."""

from __future__ import annotations

import re
from dataclasses import dataclass

from hybrid_arena.scenarios.telecom_rag.corpus import CorpusChunk


@dataclass(frozen=True)
class RetrievalResult:
    chunk: CorpusChunk
    score: float

    def to_dict(self) -> dict:
        return {"chunk": self.chunk.to_dict(), "score": self.score}


def retrieve(query: str, chunks: list[CorpusChunk], top_k: int = 3) -> list[RetrievalResult]:
    query_tokens = _tokens(query)
    scored: list[RetrievalResult] = []
    for chunk in chunks:
        chunk_tokens = _tokens(" ".join([chunk.title, chunk.text, *chunk.tags]))
        overlap = len(query_tokens & chunk_tokens)
        tag_boost = sum(0.5 for tag in chunk.tags if tag.lower() in query_tokens)
        substring_boost = sum(1.0 for token in query_tokens if token and token in chunk.text.lower())
        score = overlap + tag_boost + substring_boost
        if score > 0:
            scored.append(RetrievalResult(chunk=chunk, score=score))
    return sorted(scored, key=lambda result: (-result.score, result.chunk.doc_id))[:top_k]


def _tokens(text: str) -> set[str]:
    tokens: set[str] = set()
    for token in re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]+", text):
        normalized = token.lower()
        tokens.add(normalized)
        if re.fullmatch(r"[\u4e00-\u9fff]+", token):
            tokens.update(token)
            tokens.update(token[index : index + 2] for index in range(max(0, len(token) - 1)))
    return tokens
