from __future__ import annotations

from pathlib import Path

from hybrid_arena.scenarios.telecom_rag.corpus import load_corpus
from hybrid_arena.scenarios.telecom_rag.generator import generate_answer
from hybrid_arena.scenarios.telecom_rag.retriever import retrieve


def test_generate_answer_includes_citations() -> None:
    chunks = load_corpus(Path("datasets/telecom_docs/mini_telecom_kb.jsonl"))
    results = retrieve("什么是 RAN", chunks, top_k=1)

    answer = generate_answer("什么是 RAN", results)

    assert answer["citations"][0]["doc_id"] == "ran-overview"
    assert "RAN" in answer["answer"]


def test_generate_answer_refuses_without_evidence() -> None:
    answer = generate_answer("不存在的问题", [])

    assert answer["answer"] == "知识库证据不足，无法可靠回答。"
    assert answer["citations"] == []
