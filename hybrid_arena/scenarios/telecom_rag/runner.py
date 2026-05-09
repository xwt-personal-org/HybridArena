"""Runner for local telecom RAG tasks."""

from __future__ import annotations

from pathlib import Path

from hybrid_arena.core.schema import TaskInput, TaskRunResult
from hybrid_arena.core.traces import TraceRecorder
from hybrid_arena.scenarios.telecom_rag.corpus import DEFAULT_CORPUS_PATH, load_corpus
from hybrid_arena.scenarios.telecom_rag.generator import generate_answer
from hybrid_arena.scenarios.telecom_rag.retriever import retrieve


class TelecomRagRunner:
    scenario_name = "telecom_rag"

    def run(self, task: TaskInput) -> TaskRunResult:
        run_id = task.metadata.get("run_id", f"{task.task_id}-rag-run")
        question = task.payload["question"]
        top_k = int(task.payload.get("top_k", 3))
        corpus_path = Path(task.payload.get("corpus_path", DEFAULT_CORPUS_PATH))
        recorder = TraceRecorder(run_id=run_id, task_id=task.task_id, scenario=self.scenario_name)

        recorder.start_step("load_corpus", {"path": str(corpus_path)})
        chunks = load_corpus(corpus_path)
        recorder.finish_step({"chunk_count": len(chunks)})

        recorder.start_step("retrieve", {"question": question, "top_k": top_k})
        results = retrieve(question, chunks, top_k=top_k)
        recorder.finish_step({"results": [result.to_dict() for result in results]})

        recorder.start_step("generate_answer", {"question": question})
        output = generate_answer(question, results)
        metrics = {
            "citation_count": len(output["citations"]),
            "unsupported": output["unsupported"],
        }
        recorder.finish_step(output, metrics=metrics)

        return TaskRunResult(
            run_id=run_id,
            task_id=task.task_id,
            scenario=self.scenario_name,
            output=output,
            metrics=metrics,
            trace=recorder.to_trace(metrics=metrics),
        )
