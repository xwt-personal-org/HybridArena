# Next Roadmap C/D/E - HybridArena

> 状态：reference only。本文不是机器状态源，不写入 `.ai/ledger.json.progress`。
> 机器状态源：`.ai/ledger.json`。

本文只保存稳定展示分支与 Topic A/B 之后的下一步候选路线。当前执行范围仍然限制在稳定展示分支、Topic A「真实 checkpoint 证据链」和 Topic B「C++ ONNX Runtime 部署验证」。

## Topic C: Offline RL from smoke to mini-loop

### 目标

把 Offline RL 从 BC/CQL/IQL CPU smoke 接口推进到一个最小可复现 offline training mini-loop。

### 为什么现在不做

Topic A 需要先建立 checkpoint 保存、导出、parity 和 QA tournament 的证据链。没有稳定 artifact 路径时，Offline RL 结果难比较，也容易被过度解释。

### 前置条件

- checkpoint save/load 路径稳定。
- ONNX metadata 能标识 checkpoint-bound artifact。
- QA tournament 可以评估 checkpoint policy source。
- replay 数据来源清楚，不能把 synthetic replay 说成人类经验。

### 建议分支

`iter/offline-rl-mini-loop-202605XX`

### 目标文件

- `hybrid_arena/algorithms/offline/bc.py`
- `hybrid_arena/algorithms/offline/cql.py`
- `hybrid_arena/algorithms/offline/iql.py`
- `hybrid_arena/training/offline_pretrain.py`
- `hybrid_arena/data/offline_replay/`
- `hybrid_arena/qa/tournament.py`
- `configs/experiments/offline_rl_mini_loop.yaml`
- `docs/offline-rl-mini-loop.md`
- `results/offline_rl/`

### 验证命令

```bash
pytest hybrid_arena/algorithms/tests/test_offline_algorithms.py -v
pytest hybrid_arena/data/offline_replay/tests -v
python -m hybrid_arena.training.offline_pretrain --config configs/experiments/offline_rl_mini_loop.yaml
python -m hybrid_arena.scripts.qa_tournament --episodes 4 --seed 7 --output results/qa/offline_rl_eval --policy-source checkpoint --checkpoint results/offline_rl/bc_checkpoint.pt
ruff check hybrid_arena
```

### 非目标

- 不声称已学习人类 replay，除非用户提供授权 replay 数据。
- 不从 synthetic replay smoke 推导 solved MOBA policy quality。
- 不把 GPU training 设为默认验证前提。

## Topic D: LLM planner from stub to controlled external provider

### 目标

把 strict stub / RulePlanner 推进到可审计的外部 provider 集成路径。

### 为什么现在不做

macro action fail-closed 边界必须先保持稳定。外部 LLM 接入应建立在 checkpoint 与 QA 证据路径稳定之后。

### 前置条件

- strict macro-action schema 仍然 fail-closed。
- planner decisions 可追踪、可落盘审计。
- QA tournament 能区分 planner source。
- 用户明确 provider / API key / 本地模型路线。

### 建议分支

`iter/llm-provider-controlled-planner-202605XX`

### 目标文件

- `hybrid_arena/inference/llm_planner.py`
- `hybrid_arena/inference/prompt_templates.py`
- `hybrid_arena/inference/planner_memory.py`
- `hybrid_arena/inference/persona.py`
- `hybrid_arena/qa/tournament.py`
- `hybrid_arena/inference/tests/`
- `docs/llm-provider-planner.md`
- `results/planner_eval/`

### 验证命令

```bash
pytest hybrid_arena/inference/tests -v
pytest hybrid_arena/qa/tests -v
python -m hybrid_arena.scripts.qa_tournament --episodes 4 --seed 7 --output results/qa/llm_stub_eval
ruff check hybrid_arena
```

### 非目标

- 默认测试不依赖真实外部 API key。
- 不允许 LLM 直接输出低层动作绕过 macro/action mask。
- 不声称真实外部 LLM gameplay proof，除非 provider logs 和 QA reports 都存在。

## Topic E: QA from smoke tournament to checkpoint league

### 目标

把 QA 从 smoke tournament 升级为 checkpoint league：registry、round-robin、Elo trend、objective gates 和 regression reports。

### 为什么现在不做

Topic A 需要先产出 checkpoint artifact。没有多个可比较 checkpoint 时，league 只是接口演示，不能支持质量判断。

### 前置条件

- checkpoint registry 存在。
- 至少两个 policy artifact 可评估。
- QA report 能区分 rule、random、checkpoint、macro adapter 和 planner-backed policies。
- objective gates 稳定，不只看 reward。

### 建议分支

`iter/checkpoint-league-qa-202605XX`

### 目标文件

- `hybrid_arena/qa/tournament.py`
- `hybrid_arena/qa/rating.py`
- `hybrid_arena/qa/scenario_matrix.py`
- `hybrid_arena/qa/balance_report.py`
- `hybrid_arena/qa/regression_gates.py`
- `hybrid_arena/scripts/qa_tournament.py`
- `hybrid_arena/scripts/checkpoint_league.py`
- `docs/checkpoint-league-qa.md`
- `results/qa/league/`

### 验证命令

```bash
pytest hybrid_arena/qa/tests -v
python -m hybrid_arena.scripts.checkpoint_league --registry results/checkpoints/registry.json --episodes 4 --seed 7 --output results/qa/league
python -m json.tool results/qa/league/summary.json
ruff check hybrid_arena
```

### 非目标

- 默认验证不要求数百局长评估。
- 不从 CPU smoke 结果声称 production league scale。
- 不使用 reward-only metrics 作为 acceptance。
