from __future__ import annotations

from pathlib import Path

from hybrid_arena.scenarios.telecom_rag.corpus import load_corpus
from hybrid_arena.scenarios.telecom_rag.retriever import retrieve


def test_retrieve_ranks_relevant_amf_chunk() -> None:
    chunks = load_corpus(Path("datasets/telecom_docs/mini_telecom_kb.jsonl"))

    results = retrieve("AMF 负责什么", chunks, top_k=2)

    assert results[0].chunk.doc_id == "5gc-amf"
    assert results[0].score > 0
