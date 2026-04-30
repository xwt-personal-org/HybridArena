# Changelog

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
