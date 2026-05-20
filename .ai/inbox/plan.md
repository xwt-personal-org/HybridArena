# Web Plan Packet

## Meta
- Project: HybridArena
- Packet Type: patch
- Created: 2026-05-19
- Based On Ledger Task: `iter-five-phase-industrial-rlmoba-20260519`
- Base Branch: `iter/phase5-qa-rating-loop-20260519`
- Suggested Patch Branch: `iter-review-findings-claim-boundary-20260519`
- Intended Consumer: local execution agent
- After Consumed: delete or clear this file

## Change Goal

Close the latest Web Review findings across P0-P3 without expanding the feature surface.

Primary goals:
1. P0: Make LLM macro-action validation fail-closed.
2. P0: Make QA tournament reports truthfully represent the evaluated policy source and planner integration.
3. P1: Bind ONNX export/validation metadata to a checkpoint when supplied; clearly label contract-smoke exports.
4. P1: Preserve Phase 3 deployment open items through an explicit deployment capability/status report.
5. P2: Normalize documentation and generated reports to claim only what has evidence.
6. P3: Add claim-boundary tests so smoke scaffolds cannot be presented as production proof.

## Current State From Ledger

- State: `NEEDS_REVIEW`
- Progress: `5 / 5 phases`
- Validation: `PASS`
- Open Items:
  - Phase 3: `cmake` was not installed; `cpp_inference` configure/build was not run.
  - Phase 3: `ONNXRUNTIME_ROOT` was not set; C++ ONNX Runtime headers/libs were unavailable.
  - Phase 3: CUDA/TensorRT acceleration was not validated; no TensorRT speedup claim is made.
- Review Findings To Close:
  - High: invalid LLM macro action is silently normalized instead of rejected.
  - High: QA tournament validates rule-based smoke behavior while names/reports imply current policy or planner integration.
  - Medium: ONNX export is not checkpoint-bound.
  - Medium: C++/ONNX Runtime/CUDA/TensorRT remain environment-gated and must not be overclaimed.
  - Low/Medium: Offline RL is a CPU-smoke foundation, not a completed offline RL training loop.

## Patch Steps

### Step 0 — Prepare review-findings patch branch and preserve state protocol `scope:auto`

- Files:
  - `.ai/ledger.json`
  - `.ai/inbox/plan.md`
  - `.ai/checkpoint.json` if already present
- Actions:
  1. Create branch `iter-review-findings-claim-boundary-20260519` from `iter/phase5-qa-rating-loop-20260519`.
  2. Put this packet at `.ai/inbox/plan.md`.
  3. Set `.ai/ledger.json.state` to `IN_PROGRESS` or record start in `last_run`.
  4. Preserve existing Phase 3 open items unless the environment actually supports CMake, ONNX Runtime C++, CUDA, and TensorRT validation.
  5. Do not use README, docs, results, report files, or chat logs as state.
- Validation:
  - `python -m json.tool .ai/ledger.json`
  - `git branch --show-current`
- Done When:
  - Work is isolated on the suggested patch branch.
  - Ledger remains valid JSON.
  - `.ai/inbox/plan.md` contains this patch packet during execution.

### Step 1 — P0: Make LLM macro-action validation fail-closed `scope:review`

- Files:
  - `hybrid_arena/inference/macro_actions.py`
  - `hybrid_arena/inference/llm_planner.py`
  - `hybrid_arena/inference/tests/test_llm_planner_stub.py`
  - `hybrid_arena/inference/tests/test_macro_action_adapter.py`
- Actions:
  1. Split strict validation from legacy alias normalization.
  2. In `macro_actions.py`, define canonical macro actions as `DEFEND_OBJECTIVE`, `GROUP_MID`, `RETREAT_FARM`, `INVADE_JUNGLE`, `PUSH_LANE`, `BAIT_FIGHT`.
  3. Keep legacy aliases such as `group_mid`, `push_nearest_tower`, `retreat`, `farm_safe`, `protect_support`, `force_teamfight`, `split_push`, but route them through `canonical_macro_action(action: str, *, allow_aliases: bool = True) -> str`.
  4. Change strict validation to `validate_macro_action(action: str, *, allow_aliases: bool = False) -> str`; unknown, blank, non-string, or malformed macro action must raise `ValueError`.
  5. Update `validate_llm_decision()` so LLM/provider output uses `allow_aliases=False`.
  6. Keep deterministic `RulePlanner` and `MacroActionAdapter` compatibility by allowing aliases only in non-LLM legacy paths.
  7. Add tests for `DROP_TABLE`, `UNKNOWN_ACTION`, empty string, valid stub macro, and legacy adapter aliases.
  8. If fallback is used, ensure fallback is explicit in returned metadata or trace; do not silently rewrite invalid LLM output.
- Validation:
  - `python -m compileall hybrid_arena/inference`
  - `pytest hybrid_arena/inference/tests/test_llm_planner_stub.py -v`
  - `pytest hybrid_arena/inference/tests/test_macro_action_adapter.py -v`
  - `pytest hybrid_arena/inference/tests -v`
  - `ruff check hybrid_arena/inference`
- Done When:
  - Invalid LLM macro action fails closed.
  - Legacy rule/adapter flows remain backward compatible.
  - Tests prove both strict rejection and legacy alias compatibility.

### Step 2 — P0: Make QA tournament policy source and planner integration truthful `scope:review`

- Files:
  - `hybrid_arena/qa/tournament.py`
  - `hybrid_arena/qa/scenario_matrix.py`
  - `hybrid_arena/qa/balance_report.py`
  - `hybrid_arena/scripts/qa_tournament.py`
  - `hybrid_arena/qa/tests/test_tournament.py`
  - `hybrid_arena/qa/tests/test_regression_gates.py`
- Actions:
  1. Add explicit policy-source metadata: `policy_source`, `policy_artifact`, `planner_source`, `evaluation_mode`.
  2. Rename misleading default scenarios:
     - `current_policy_vs_rule_bot` -> `rule_policy_vs_random_smoke`
     - `macro_planner_enabled_vs_disabled` -> `macro_adapter_smoke_vs_rule`
     - keep `objective_stress_test` only if report says it is a smoke test.
  3. Refactor `run_scenario()` so it does not hardcode `RuleBasedAgent().act` while calling it current policy.
  4. Add `make_policy_runner(policy_source, checkpoint_path=None, planner_source="none")`.
  5. For `checkpoint`, require `checkpoint_path`; if absent, raise `ValueError`.
  6. For `macro_adapter_smoke`, actually route through `MacroActionAdapter` and count planner/adaptor use.
  7. Replace hardcoded `illegal_action_rate = 0.0` with an action audit helper. If only post-legalization actions are visible, report `illegal_action_rate_source: "post_legalization_only"`.
  8. Replace hardcoded `planner_override_rate = 0.2` with measured macro-adapter call count / total controlled decisions. If no planner is active, use `planner_override_rate_source: "planner_disabled"`.
  9. Update JSON/CSV/Markdown reports to include policy source, planner source, evaluation mode, metric sources, and claim boundary.
  10. Add tests that default tournament report says `rule_based` and `smoke`, checkpoint mode without checkpoint raises, macro adapter smoke reports source metadata, and no generated report calls rule-smoke `current_policy` or `trained_policy`.
- Validation:
  - `python -m compileall hybrid_arena/qa hybrid_arena/scripts/qa_tournament.py`
  - `pytest hybrid_arena/qa/tests/test_tournament.py -v`
  - `pytest hybrid_arena/qa/tests/test_regression_gates.py -v`
  - `python -m hybrid_arena.scripts.qa_tournament --episodes 4 --seed 7 --output results/qa/smoke`
  - `ruff check hybrid_arena/qa hybrid_arena/scripts/qa_tournament.py`
- Done When:
  - QA tournament output truthfully identifies rule-smoke vs checkpoint vs planner smoke.
  - `planner_override_rate` and `illegal_action_rate` have explicit source metadata.
  - Scenario/report names cannot be mistaken for real trained-policy validation.

### Step 3 — P1: Bind ONNX export and parity validation to checkpoint evidence when available `scope:review`

- Files:
  - `hybrid_arena/deployment/export_onnx.py`
  - `hybrid_arena/deployment/onnx_validate.py`
  - `hybrid_arena/deployment/latency_benchmark.py`
  - `hybrid_arena/deployment/tests/test_export_contract.py`
  - `hybrid_arena/deployment/tests/test_checkpoint_export_contract.py` if helpful
- Actions:
  1. Add optional CLI arguments `--checkpoint` and `--export-mode contract_smoke|checkpoint_bound`.
  2. Add `load_actor_critic_checkpoint(policy: ActorCritic, checkpoint_path: Path) -> dict`.
  3. Support common checkpoint formats: raw `state_dict`, `model_state_dict`, `policy_state_dict`, `actor_critic`.
  4. Unsupported format must raise `ValueError`.
  5. Metadata must include `export_mode`, `trained_policy`, `checkpoint_path`, `checkpoint_sha256`, `model_sha256`, `obs_contract`, `action_contract`, `opset`, `provider`, and `device`.
  6. If no checkpoint is supplied, set `export_mode: "contract_smoke"` and `trained_policy: false`.
  7. If checkpoint is supplied, load it, export, set `export_mode: "checkpoint_bound"` and `trained_policy: true`, then validate ONNX parity against the same checkpoint-loaded PyTorch policy.
  8. Update tests to create a temporary ActorCritic checkpoint and prove checkpoint-bound metadata is emitted.
- Validation:
  - `python -m compileall hybrid_arena/deployment`
  - `pytest hybrid_arena/deployment/tests -v`
  - `python -m hybrid_arena.deployment.export_onnx --output artifacts/policy-contract-smoke.onnx --seed 7 --export-mode contract_smoke`
  - `python -m hybrid_arena.deployment.onnx_validate --model artifacts/policy-contract-smoke.onnx --seed 7`
  - `ruff check hybrid_arena/deployment`
- Done When:
  - ONNX metadata separates contract smoke from checkpoint-bound export.
  - Checkpoint-bound export can be tested using a fixture checkpoint.
  - No deployment report can imply a trained policy when no checkpoint was provided.

### Step 4 — P1: Add deployment capability/status report for Phase 3 open items `scope:auto`

- Files:
  - `hybrid_arena/deployment/status.py`
  - `hybrid_arena/deployment/tests/test_deployment_status.py`
  - `.ai/ledger.json`
- Actions:
  1. Add `detect_deployment_capabilities()` returning a JSON-serializable dict with Python ONNX export, Python ONNX Runtime, CMake, `ONNXRUNTIME_ROOT`, C++ headers/libs, CUDA, and TensorRT availability.
  2. Add derived booleans: `cpp_build_verifiable`, `tensorrt_verifiable`, `cpp_inference_verified`, `tensorrt_verified`.
  3. Add CLI: `python -m hybrid_arena.deployment.status --output results/deployment/status.json`.
  4. In CPU-only or missing-tool environments, do not fail the whole patch. Emit clear `missing` / `skipped` fields.
  5. Ensure `.ai/ledger.json.open_items` retains CMake/C++ ONNX Runtime/CUDA/TensorRT items unless actual validation is run and recorded.
- Validation:
  - `python -m compileall hybrid_arena/deployment`
  - `pytest hybrid_arena/deployment/tests/test_deployment_status.py -v`
  - `python -m hybrid_arena.deployment.status --output results/deployment/status.json`
  - `python -m json.tool results/deployment/status.json`
  - `ruff check hybrid_arena/deployment`
- Done When:
  - Deployment capability status is machine-readable.
  - Missing C++/TensorRT dependencies are visible, not hidden.
  - Ledger and generated deployment status agree.

### Step 5 — P2: Normalize docs and generated-report claim boundaries `scope:review`

- Files:
  - `README.md` if it contains updated claims
  - `docs/rlmoba-five-phase-roadmap.md`
  - `docs/qa-rating-loop.md`
  - Create `docs/claim-boundaries.md`
  - `hybrid_arena/qa/balance_report.py`
  - `hybrid_arena/deployment/export_onnx.py`
  - `hybrid_arena/deployment/status.py`
- Actions:
  1. Create `docs/claim-boundaries.md` with a matrix: feature, evidence available, not yet proven, required proof to upgrade claim.
  2. Required claim boundaries:
     - MARL / Offline RL: foundation + CPU smoke, not solved offline RL.
     - Distributed training: local actor/learner skeleton + V-trace smoke, not cluster throughput proof.
     - ONNX: contract-smoke export unless checkpoint-bound metadata exists.
     - C++ inference: skeleton only unless CMake + ONNX Runtime C++ build is recorded.
     - TensorRT: not validated unless CUDA/TensorRT commands are recorded.
     - LLM planner: strict stub/rule planner validation, not real external LLM gameplay proof.
     - QA: rule-smoke tournament unless checkpoint/planner source proves otherwise.
  3. Update generated Markdown QA reports to include `Evaluation Mode`, `Policy Source`, `Planner Source`, `Claim Boundary`, and `Open Items`.
  4. Do not use docs as state. Docs must point back to `.ai/ledger.json` for machine state.
- Validation:
  - `python -m compileall hybrid_arena/qa hybrid_arena/deployment`
  - `pytest hybrid_arena/qa/tests -v`
  - `pytest hybrid_arena/deployment/tests -v`
  - `python -m hybrid_arena.scripts.qa_tournament --episodes 2 --seed 7 --output results/qa/claim_boundary_smoke`
  - `grep -R "Claim Boundary" docs results/qa/claim_boundary_smoke || true`
  - `ruff check hybrid_arena`
- Done When:
  - Docs and generated reports cannot overclaim smoke scaffolds as production proof.
  - Claim boundaries are explicit and consistent with ledger open items.

### Step 6 — P3: Add claim-boundary regression test suite `scope:review`

- Files:
  - `hybrid_arena/qa/tests/test_claim_boundaries.py`
  - `hybrid_arena/deployment/tests/test_claim_boundaries.py` if cleaner
  - Existing tests touched by P0-P2
- Actions:
  1. Add tests that prevent false claims:
     - no checkpoint -> ONNX metadata has `trained_policy is False`.
     - checkpoint export fixture -> metadata has `export_mode == "checkpoint_bound"`.
     - missing C++ build -> deployment status must not set `cpp_inference_verified: true`.
     - missing TensorRT -> deployment status must not set `tensorrt_verified: true`.
     - default QA tournament -> `evaluation_mode == "smoke"` and `policy_source == "rule_based"`.
     - macro adapter smoke -> planner source and override source are explicit.
     - invalid LLM macro action -> raises, not normalizes.
  2. Keep tests CPU-only.
  3. Do not make tests depend on CUDA, CMake, TensorRT, real checkpoint assets, or external LLM APIs.
- Validation:
  - `pytest hybrid_arena/qa/tests/test_claim_boundaries.py -v`
  - `pytest hybrid_arena/deployment/tests -v`
  - `pytest hybrid_arena/inference/tests -v`
  - `pytest hybrid_arena/qa/tests -v`
  - `ruff check hybrid_arena`
- Done When:
  - Future overclaims become test failures.
  - P0-P2 behavior is locked by regression tests.

### Step 7 — Final integration validation, ledger update, and inbox cleanup `scope:review`

- Files:
  - `.ai/ledger.json`
  - `.ai/inbox/plan.md`
  - `.ai/checkpoint.json`
  - optional `results/deployment/status.json`
  - optional `results/qa/claim_boundary_smoke/*`
- Actions:
  1. Run targeted validations from Steps 1-6.
  2. Run broader AgentBench and MiniMOBA/RL regression.
  3. Update `.ai/ledger.json`; final state should be `NEEDS_REVIEW`.
  4. `validation.status` should be `PASS` if all CPU validations pass and environment open items are represented as expected; use `PARTIAL` or `FAIL` only if evidence requires it.
  5. Preserve unresolved CMake/C++ ONNX Runtime/CUDA/TensorRT open items unless proven otherwise.
  6. Update `.ai/checkpoint.json` only after required validations complete.
  7. Delete or clear `.ai/inbox/plan.md` after successful consumption.
- Validation:
  - `python -m json.tool .ai/ledger.json`
  - `python -m compileall hybrid_arena`
  - `pytest hybrid_arena/inference/tests -v`
  - `pytest hybrid_arena/qa/tests -v`
  - `pytest hybrid_arena/deployment/tests -v`
  - `pytest hybrid_arena/core hybrid_arena/scenarios hybrid_arena/services/api hybrid_arena/scripts/tests -v`
  - `pytest hybrid_arena/minimoba/tests hybrid_arena/training/tests hybrid_arena/algorithms/tests -v`
  - `ruff check hybrid_arena`
  - `test ! -s .ai/inbox/plan.md || test ! -f .ai/inbox/plan.md`
- Done When:
  - P0-P3 findings are implemented and tested.
  - Remaining external evidence gaps are explicit open items.
  - Ledger is updated.
  - Inbox plan is consumed.
  - Branch is ready for Web Review.

## Decisions / Risks

- Do not convert this patch into another feature expansion.
- Do not add new MARL, Offline RL, distributed training, TensorRT, or real LLM features.
- Do not hide Phase 3 environment limitations.
- Do not downgrade strict LLM validation for convenience.
- Do not label rule-smoke QA as current trained policy validation.
- Do not mark final state as `ALL_CLEAR` from execution side. Use `NEEDS_REVIEW`.

## Acceptance

The patch is acceptable only if:
1. Unknown LLM macro action raises instead of normalizing to `group_mid`.
2. Default QA tournament reports `rule_based` and `smoke`, not `current_policy` or `trained_policy`.
3. `planner_override_rate` and `illegal_action_rate` include source metadata and are not unexplained constants.
4. ONNX export metadata distinguishes `contract_smoke` from `checkpoint_bound`.
5. Deployment status reports CMake/C++ ONNX Runtime/CUDA/TensorRT availability without overclaiming.
6. Docs and generated reports include claim boundaries.
7. Claim-boundary tests prevent future false claims.
8. Existing AgentBench and MiniMOBA/RL regression tests still pass.
9. `.ai/ledger.json` is the only machine state layer.
10. `.ai/inbox/plan.md` is deleted or cleared after consumption.
