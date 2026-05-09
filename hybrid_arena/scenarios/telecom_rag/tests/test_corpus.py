from __future__ import annotations

from pathlib import Path

from hybrid_arena.scenarios.telecom_rag.corpus import load_corpus


def test_load_corpus_reads_telecom_chunks() -> None:
    chunks = load_corpus(Path("datasets/telecom_docs/mini_telecom_kb.jsonl"))

    assert len(chunks) >= 5
    assert chunks[0].doc_id
    assert chunks[0].text
