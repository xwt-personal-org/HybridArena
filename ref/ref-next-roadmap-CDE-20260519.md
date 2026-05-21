# Next Roadmap C/D/E — HybridArena

> Created: 2026-05-19
> Status: Reference only. This file is not machine state.
> State source: `.ai/ledger.json`

This file stores future roadmap topics after the stable display branch plus Topic A/B dispatch. Do not treat this file as active execution state.

## Topic C — Offline RL from smoke to mini-loop

### Goal
Move Offline RL from BC/CQL/IQL CPU-smoke interfaces to a minimal reproducible offline training loop.

### Why later
Topic A must first establish a real checkpoint evidence chain. Without a reliable checkpoint/evaluation artifact path, Offline RL results would be hard to compare and easy to overclaim.

### Prerequisites
- Stable checkpoint save/load path.
- Checkpoint-bound ONNX export metadata.
- QA tournament can evaluate checkpoint policy source.
- Synthetic or licensed replay dataset policy is clarified.

### Suggested branch
`iter/offline-rl-mini-loop-202605XX`

### Target files
- `hybrid_arena/algorithms/offline/bc.py`
- `hybrid_arena/algorithms/offline/cql.py`
- `hybrid_arena/algorithms/offline/iql.py`
- `hybrid_arena/training/offline_pretrain.py`
- `hybrid_arena/data/offline_replay/`
- `hybrid_arena/qa/tournament.py`
- `configs/experiments/offline_rl_mini_loop.yaml`
- `docs/offline-rl-mini-loop.md`
- `results/offline_rl/`

### Validation commands
- `pytest hybrid_arena/algorithms/tests/test_offline_algorithms.py -v`
- `pytest hybrid_arena/data/offline_replay/tests -v`
- `python -m hybrid_arena.training.offline_pretrain --config configs/experiments/offline_rl_mini_loop.yaml`
- `python -m hybrid_arena.scripts.qa_tournament --episodes 4 --seed 7 --output results/qa/offline_rl_eval --policy-source checkpoint --checkpoint results/offline_rl/bc_checkpoint.pt`
- `ruff check hybrid_arena`

### Non-goals
- Do not claim human replay learning unless licensed human replay data is provided.
- Do not claim solved MOBA policy quality from synthetic replay.
- Do not make GPU training mandatory.

## Topic D — LLM planner from stub to controlled external provider

### Goal
Move from strict LLM stub/rule planner to an auditable external-provider integration path.

### Why later
The current macro-action fail-closed boundary must remain stable. External LLM integration should be added only after stable checkpoint and QA evidence paths exist.

### Prerequisites
- Strict macro-action schema remains fail-closed.
- Planner decisions are traceable.
- QA tournament can identify planner source.
- User provides API/provider decision or local model decision.

### Suggested branch
`iter/llm-provider-controlled-planner-202605XX`

### Target files
- `hybrid_arena/inference/llm_planner.py`
- `hybrid_arena/inference/prompt_templates.py`
- `hybrid_arena/inference/planner_memory.py`
- `hybrid_arena/inference/persona.py`
- `hybrid_arena/qa/tournament.py`
- `hybrid_arena/inference/tests/`
- `docs/llm-provider-planner.md`
- `results/planner_eval/`

### Validation commands
- `pytest hybrid_arena/inference/tests -v`
- `pytest hybrid_arena/qa/tests -v`
- `python -m hybrid_arena.scripts.qa_tournament --episodes 4 --seed 7 --output results/qa/llm_stub_eval`
- `ruff check hybrid_arena`

### Non-goals
- Do not require real external API keys for default tests.
- Do not let LLM output low-level actions directly.
- Do not bypass action mask or strict macro validation.
- Do not claim real external LLM gameplay proof unless provider logs and QA reports exist.

## Topic E — QA from smoke tournament to checkpoint league

### Goal
Upgrade QA from smoke tournament to a checkpoint league with registry, round-robin evaluation, Elo trend, objective gates, and regression reports.

### Why later
Topic A should first produce checkpoint artifacts. Topic E needs multiple comparable checkpoints to be meaningful.

### Prerequisites
- Checkpoint registry exists.
- At least two policy artifacts exist.
- QA reports can distinguish rule, random, checkpoint, macro adapter, and planner-backed policies.
- Objective gates are stable.

### Suggested branch
`iter/checkpoint-league-qa-202605XX`

### Target files
- `hybrid_arena/qa/tournament.py`
- `hybrid_arena/qa/rating.py`
- `hybrid_arena/qa/scenario_matrix.py`
- `hybrid_arena/qa/balance_report.py`
- `hybrid_arena/qa/regression_gates.py`
- `hybrid_arena/scripts/qa_tournament.py`
- `hybrid_arena/scripts/checkpoint_league.py`
- `docs/checkpoint-league-qa.md`
- `results/qa/league/`

### Validation commands
- `pytest hybrid_arena/qa/tests -v`
- `python -m hybrid_arena.scripts.checkpoint_league --registry results/checkpoints/registry.json --episodes 4 --seed 7 --output results/qa/league`
- `python -m json.tool results/qa/league/summary.json`
- `ruff check hybrid_arena`

### Non-goals
- Do not require hundreds of episodes in default validation.
- Do not claim production league scale from CPU smoke runs.
- Do not use reward-only metrics as acceptance.
