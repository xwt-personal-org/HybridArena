# Changelog

## Unreleased

### Changed
- Restored MOBA/RL as the project mainline; AgentBench demoted to optional application layer.
- Rewrote README with MOBA/RL architecture, training/evaluation commands, and environment specs as first-class content.
- Archived AgentBench v3 plan to `docs/.archive/plan-agentbench-v3.md`; activated RL roadmap in `docs/plan.md`.
- Updated `docs/progress.md` and `docs/report.md` to reflect RL stages as primary progress tracking.
- Promoted ISSUE-F13 (objective reward shaping) back to mainline blocker in `docs/issues.md`.
- Unified `AGENTS.md` status and commands around MOBA/RL mainline.

### Added
- Added `hybrid_arena/demo/moba_app.py` as the mainline MOBA Streamlit demo.
- Retained `hybrid_arena/demo/app.py` as the AgentBench application layer demo.

## AgentBench v3 milestone

### Added
- Added AgentBench mainline with shared core schemas, trace recording, SQLite run storage, FastAPI API, CLI reports, Streamlit demo, and three business scenarios.
- Added `jd_resume_match`, `telecom_rag`, and `ticket_triage` scenario packages with deterministic offline runners and evaluators.
- Added AgentBench interview artifacts: architecture doc, demo script, resume bullets, and benchmark report.
- Added baseline experiment config support for ablation runner (`--config`, `--dry-run`).
- Added planner trace schema (`PlannerTrace`) and JSONL recorder (`PlannerTraceRecorder`).
- Added evaluator metrics consistency tests with required fields validation.

### Changed
- Expanded CI test coverage from MiniMOBA-only to the full package.
- Updated result tables with `draw_rate`, `avg_towers_destroyed`, `avg_tower_hp_advantage` columns.

### Fixed
- Fixed local Chinese tokenization in the telecom RAG retriever so packet-loss troubleshooting cases are retrieved.
- Fixed pytest import-name collisions in new scenario test packages by adding package initializers.
- Added missing `httpx` app dependency required by FastAPI/Starlette `TestClient`.
- Fixed documentation status inconsistency around GRPO/QLoRA implementation status.
- Fixed `env.close()` crash when pygame is not installed (now handles ImportError gracefully).
- Added missing `episodes` field to evaluator result dict.

## v0.2.0 - training correctness and objective game milestone

### Fixed

- Fixed joint action mask handling by using 324-way joint categorical policy.
- Fixed PPO rollout/update consistency with saved `action_masks` and `old_values`.
- Fixed clipped value loss and DualClipPPO `dual_clip_fraction` metric.

### Added

- Added tower/base objective runtime state, team economy updates, and objective info metrics.
- Added checkpoint utilities, evaluator, train/evaluate/ablation CLI, and planner demo CLI.
- Added SyncParallelEnvRunner, SelfPlayPool, CurriculumManager, RulePlanner, Dummy LLMPlanner, and MacroActionAdapter.

### Changed

- Python support is constrained to `>=3.10,<3.13`.
- README now documents reproducible training/evaluation commands and known limitations.

### Known Issues

- Smoke ablation output is a pipeline check, not a final benchmark.
- GRPO/QLoRA remains out of scope for this milestone.
