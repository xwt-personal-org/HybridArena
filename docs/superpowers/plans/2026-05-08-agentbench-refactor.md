# AgentBench Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn HybridArena into a job-oriented AgentBench platform with runnable API, scenario runners, trace, evaluation, demo, and interview-ready documentation.

**Architecture:** Keep MiniMOBA/RL as a research branch and add a new AgentBench mainline: `core` schemas/storage/tracing, `scenarios` business workflows, `services/api` HTTP surface, and `demo` presentation. Each scenario produces a `TaskRunResult` with trace and metrics.

**Tech Stack:** Python 3.10+, Pydantic, FastAPI, SQLite, Pytest, Ruff, Streamlit, PyYAML.

---

## Task 1: Core AgentBench Types

**Files:**
- Create: `hybrid_arena/core/__init__.py`
- Create: `hybrid_arena/core/schema.py`
- Create: `hybrid_arena/core/tests/test_schema.py`

- [ ] **Step 1: Write failing schema tests**

Run: `pytest hybrid_arena/core/tests/test_schema.py -v`
Expected: FAIL because files do not exist.

- [ ] **Step 2: Implement Pydantic/dataclass-compatible schema**

Implement `TaskInput`, `ToolCallRecord`, `TaskTrace`, `TaskRunResult`, `BenchmarkResult` with `to_dict()` and `from_dict()` helpers.

- [ ] **Step 3: Verify**

Run: `pytest hybrid_arena/core/tests/test_schema.py -v`
Expected: PASS.

## Task 2: Trace Recorder

**Files:**
- Create: `hybrid_arena/core/traces.py`
- Create: `hybrid_arena/core/tests/test_traces.py`

- [ ] **Step 1: Write failing trace tests**
- [ ] **Step 2: Implement `TraceRecorder` and `JsonlTraceWriter`**
- [ ] **Step 3: Run `pytest hybrid_arena/core/tests/test_traces.py -v`**

## Task 3: SQLite Storage

**Files:**
- Create: `hybrid_arena/core/storage.py`
- Create: `hybrid_arena/core/tests/test_storage.py`

- [ ] **Step 1: Write failing storage tests**
- [ ] **Step 2: Implement `AgentBenchStore` with `init_schema`, `save_run`, `get_run`, `list_runs`**
- [ ] **Step 3: Run `pytest hybrid_arena/core/tests/test_storage.py -v`**

## Task 4: Scenario Registry

**Files:**
- Create: `hybrid_arena/scenarios/__init__.py`
- Create: `hybrid_arena/scenarios/registry.py`
- Create: `hybrid_arena/scenarios/tests/test_registry.py`

- [ ] **Step 1: Write failing registry tests**
- [ ] **Step 2: Implement protocol, placeholder registration, and lazy runner loading**
- [ ] **Step 3: Run `pytest hybrid_arena/scenarios/tests/test_registry.py -v`**

## Task 5: JD Resume Match Scenario

**Files:**
- Create: `datasets/jd_samples/jd_cases.jsonl`
- Create: `datasets/jd_samples/resume_profile.json`
- Create package: `hybrid_arena/scenarios/jd_resume_match/`
- Create tests under: `hybrid_arena/scenarios/jd_resume_match/tests/`

- [ ] **Step 1: Write failing tests for taxonomy, extractor, analyzer, runner, evaluator**
- [ ] **Step 2: Implement deterministic taxonomy and keyword extractor with evidence spans**
- [ ] **Step 3: Implement gap analyzer and interview question generation**
- [ ] **Step 4: Implement runner returning `TaskRunResult`**
- [ ] **Step 5: Implement evaluator metrics `skill_recall`, `evidence_coverage`, `missing_skill_count`**
- [ ] **Step 6: Run `pytest hybrid_arena/scenarios/jd_resume_match/tests -v`**

## Task 6: Telecom RAG Scenario

**Files:**
- Create: `datasets/telecom_docs/mini_telecom_kb.jsonl`
- Create package: `hybrid_arena/scenarios/telecom_rag/`
- Create tests under: `hybrid_arena/scenarios/telecom_rag/tests/`

- [ ] **Step 1: Write failing tests for corpus, retriever, generator, runner, evaluator**
- [ ] **Step 2: Implement JSONL corpus loader**
- [ ] **Step 3: Implement local token-overlap retriever with tag boost**
- [ ] **Step 4: Implement citation-bound answer generator**
- [ ] **Step 5: Implement evaluator metrics `recall_at_k`, `citation_coverage`, `unsupported_answer_rate`**
- [ ] **Step 6: Run `pytest hybrid_arena/scenarios/telecom_rag/tests -v`**

## Task 7: Ticket Triage Scenario

**Files:**
- Create: `datasets/ticket_samples/ticket_cases.jsonl`
- Create package: `hybrid_arena/scenarios/ticket_triage/`
- Create tests under: `hybrid_arena/scenarios/ticket_triage/tests/`

- [ ] **Step 1: Write failing tests for labels, classifier, recommender, runner, evaluator**
- [ ] **Step 2: Implement rule classifier with confidence and evidence keywords**
- [ ] **Step 3: Implement troubleshooting recommender**
- [ ] **Step 4: Implement runner and macro-F1 evaluator**
- [ ] **Step 5: Run `pytest hybrid_arena/scenarios/ticket_triage/tests -v`**

## Task 8: FastAPI Service

**Files:**
- Modify: `pyproject.toml`
- Create package: `hybrid_arena/services/api/`
- Create: `hybrid_arena/services/api/tests/test_app.py`

- [ ] **Step 1: Add `app` optional dependencies**
- [ ] **Step 2: Install app extras with `pip install -e ".[dev,app]"`**
- [ ] **Step 3: Write failing API tests for `/health`, `/scenarios`, `/tasks/run`, `/runs`, `/runs/{run_id}`**
- [ ] **Step 4: Implement FastAPI app and storage integration**
- [ ] **Step 5: Run `pytest hybrid_arena/services/api/tests/test_app.py -v`**

## Task 9: AgentBench CLI and Reporting

**Files:**
- Create: `hybrid_arena/scripts/agentbench_run.py`
- Create: `hybrid_arena/scripts/tests/test_agentbench_run.py`
- Create: `hybrid_arena/core/reporting.py`
- Create: `hybrid_arena/core/tests/test_reporting.py`

- [ ] **Step 1: Write failing CLI/reporting tests**
- [ ] **Step 2: Implement scenario input loading and report writing**
- [ ] **Step 3: Implement Markdown report generation**
- [ ] **Step 4: Run CLI smoke commands for all scenarios**
- [ ] **Step 5: Run `pytest hybrid_arena/scripts/tests/test_agentbench_run.py hybrid_arena/core/tests/test_reporting.py -v`**

## Task 10: Demo and Docs

**Files:**
- Modify: `hybrid_arena/demo/app.py`
- Modify: `README.md`
- Create: `docs/agentbench-demo-script.md`
- Create: `docs/agentbench-architecture.md`
- Create: `docs/agentbench-resume-bullets.md`
- Modify: `docs/progress.md`
- Modify: `docs/issues.md`
- Modify: `docs/report.md`
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Replace Streamlit RL-first page with AgentBench pages**
- [ ] **Step 2: Rewrite README as AgentBench-first**
- [ ] **Step 3: Add interview artifacts**
- [ ] **Step 4: Update progress/issues/report/changelog**
- [ ] **Step 5: Run `python -m compileall hybrid_arena`**

## Task 11: Full Verification

**Files:**
- No new files.

- [ ] **Step 1: Run `pytest hybrid_arena/core hybrid_arena/scenarios hybrid_arena/services/api hybrid_arena/scripts/tests -v`**
- [ ] **Step 2: Run `pytest hybrid_arena/minimoba/tests hybrid_arena/training/tests hybrid_arena/algorithms/tests -v`**
- [ ] **Step 3: Run `ruff check hybrid_arena`**
- [ ] **Step 4: Fix failures without changing scope**
- [ ] **Step 5: Record verification results in final response**
