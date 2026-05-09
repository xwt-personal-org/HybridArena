"""Citation-bound answer generator for local RAG."""

from __future__ import annotations

from hybrid_arena.scenarios.telecom_rag.retriever import RetrievalResult


def generate_answer(question: str, results: list[RetrievalResult]) -> dict:
    if not results:
        return {
            "question": question,
            "answer": "知识库证据不足，无法可靠回答。",
            "citations": [],
            "unsupported": True,
        }
    top = results[0].chunk
    citations = [
        {
            "doc_id": result.chunk.doc_id,
            "title": result.chunk.title,
            "source": result.chunk.source,
            "quote": result.chunk.text,
            "score": result.score,
        }
        for result in results
    ]
    return {
        "question": question,
        "answer": f"基于 {top.title}：{top.text}",
        "citations": citations,
        "unsupported": False,
    }
