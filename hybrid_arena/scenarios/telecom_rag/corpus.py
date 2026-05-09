"""Corpus loading for the local telecom RAG scenario."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

DEFAULT_CORPUS_PATH = Path("datasets/telecom_docs/mini_telecom_kb.jsonl")


@dataclass(frozen=True)
class CorpusChunk:
    doc_id: str
    title: str
    source: str
    text: str
    tags: tuple[str, ...]

    def to_dict(self) -> dict:
        data = asdict(self)
        data["tags"] = list(self.tags)
        return data


def load_corpus(path: str | Path = DEFAULT_CORPUS_PATH) -> list[CorpusChunk]:
    corpus_path = Path(path)
    chunks: list[CorpusChunk] = []
    with corpus_path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            payload = json.loads(line)
            chunks.append(
                CorpusChunk(
                    doc_id=payload["doc_id"],
                    title=payload["title"],
                    source=payload["source"],
                    text=payload["text"],
                    tags=tuple(payload.get("tags", [])),
                )
            )
    return chunks
