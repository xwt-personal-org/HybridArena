# HybridArena 下一步 Codex 执行指令

生成日期：2026-04-30  
目标：在当前模块 0-7 已完成的基础上，进入“验证、修正、正式实验准备”阶段。不要新增大功能，先证明现有实现可靠。

---

## 工作模式

进入连续执行模式。按本文件从上到下执行。  
每完成一个步骤必须运行对应验证命令。  
验证通过后直接进入下一步，不要停下来请求确认。  
只有出现下列情况才停下报告：

1. 验证失败且自行修复 2 次仍失败。
2. 需要外部资源，例如真实 LLM API key、W&B key、GPU 机器。
3. 遇到本文件未覆盖的技术决策。
4. 发现 README / docs 声称已完成但代码中不存在对应实现。

---

## 启动读取顺序

先读取以下文件：

1. `docs/progress.md`
2. `docs/issues.md`
3. `docs/plan.md`
4. `README.md`
5. `pyproject.toml`
6. `.github/workflows/ci.yml`
7. `docs/experiment-report-v0.md`
8. `results/ablation_summary.md`（如果存在）

确认当前仓库状态应为：模块 0-7 已完成，但实验仍处于 smoke 验证阶段。

---

# Phase F：验证、纠偏与正式实验准备

## 模块 F0：建立验证基线

### Step F0.1：安装完整开发依赖

- 操作：
  - 在干净 Python 3.10、3.11 或 3.12 环境中执行：

```bash
python -m pip install -U pip
python -m pip install -e ".[dev,rl]"
```

- 验证：
  - `python -c "import hybrid_arena; print('ok')"` 成功。
  - `python -c "import torch; print(torch.__version__)"` 成功。

### Step F0.2：运行完整静态验证

- 操作：

```bash
ruff check hybrid_arena
python -m compileall hybrid_arena
```

- 修复规则：
  - Ruff 报错必须修。
  - 不要通过扩大 ignore 列表绕过真实代码问题。
  - 若某条 Ruff 规则与算法代码冲突，只允许对具体文件加最小范围 `# noqa`。

- 验证：
  - 两条命令均返回 0。

### Step F0.3：运行完整测试套件

- 操作：

```bash
pytest hybrid_arena -v
```

- 修复规则：
  - 优先修代码，不要删除测试。
  - 只有 pygame/headless 渲染类测试允许 skip，并必须给出明确 reason。
  - 若测试依赖外部 LLM/API，改为 dummy/mock，不允许真实联网。

- 验证：
  - 全部测试通过，或仅存在明确合理的 skip。

---

## 模块 F1：扩展 CI，避免只测 minimoba

### Step F1.1：更新 `.github/workflows/ci.yml`

- 操作：
  - 将当前 CI 的测试命令从：

```bash
pytest hybrid_arena/minimoba/tests -v
```

  - 改为：

```bash
pytest hybrid_arena -v
```

- 保留：
  - Python matrix：3.10、3.11、3.12。
  - `ruff check hybrid_arena`。
  - `python -m compileall hybrid_arena`。

- 验证：

```bash
python -m compileall hybrid_arena
pytest hybrid_arena -v
```

### Step F1.2：确认 optional dependency 边界

- 操作：
  - 如果完整测试需要 `torch`，CI 安装命令必须使用：

```bash
python -m pip install -e ".[dev,rl]"
```

  - 不允许在普通 dev CI 中安装 `llm`，避免 bitsandbytes / transformers 造成 CI 不稳定。

- 验证：
  - CI 文件中包含 `".[dev,rl]"`。
  - CI 文件中不包含 `".[all]"`。

---

## 模块 F2：修正文档状态口径

### Step F2.1：修正 README 中 GRPO/QLoRA 完成度矛盾

- 背景：
  - README 当前亮点写“QLoRA GRPO 训练”。
  - Known Limitations 又写“GRPO/QLoRA 不在本阶段实现”。
  - 两者矛盾，必须统一。

- 操作：
  - 检查 `hybrid_arena/training/grpo_trainer.py`：
    - 如果只是 skeleton/mock/offline smoke，则 README 不得写“已完成”。
    - 如果真实支持 QLoRA + TRL GRPOTrainer + 可训练数据管线，则保留“实验性实现”。
  - 推荐统一口径：
    - Phase D 状态改为：`实验性 / skeleton complete，正式训练待 planner trace 数据`。
    - Known Limitations 改为：`GRPO/QLoRA 当前为离线可测训练管线骨架，尚未完成真实策略数据闭环与正式 benchmark。`

- 验证：

```bash
grep -n "GRPO\|QLoRA\|Phase D" README.md
```

### Step F2.2：更新 `docs/experiment-report-v0.md`

- 操作：
  - 增加“当前实验不是正式 benchmark”的警告。
  - 增加“下一步正式实验矩阵”。
  - 明确 smoke 参数：episodes、max_steps、seeds、opponents。

- 验证：

```bash
grep -n "正式 benchmark\|smoke\|实验矩阵" docs/experiment-report-v0.md
```

---

## 模块 F3：正式 Baseline 实验矩阵

### Step F3.1：新增正式实验配置文件

- 操作：
  - 创建 `configs/experiments/baseline_v1.yaml`。
  - 内容必须包含：

```yaml
experiment:
  name: baseline_v1
  seeds: [42, 123, 456]
  algorithms: [ppo, ppo_dualclip]
  opponents: [random, rule_based]
  episodes: 30
  max_steps: 500

training:
  total_timesteps: 100000
  num_steps: 128
  num_envs: 4
  device: cpu

outputs:
  checkpoint_dir: checkpoints/baseline_v1
  result_dir: results/baseline_v1
```

- 验证：
  - `test -f configs/experiments/baseline_v1.yaml`

### Step F3.2：让 `run_ablation.py` 支持配置文件

- 修改文件：
  - `hybrid_arena/scripts/run_ablation.py`

- 新增 argparse 参数：
  - `--config configs/experiments/baseline_v1.yaml`
  - `--dry-run`

- 行为要求：
  - 无 `--config` 时保留当前 smoke 默认参数。
  - 有 `--config` 时从 YAML 读取 seeds、algorithms、opponents、episodes、max_steps、training 参数。
  - `--dry-run` 只打印将执行的矩阵，不启动训练。

- 验证：

```bash
python -m hybrid_arena.scripts.run_ablation --config configs/experiments/baseline_v1.yaml --dry-run
```

输出必须包含：
  - `ppo`
  - `ppo_dualclip`
  - `random`
  - `rule_based`
  - `seed=42`

### Step F3.3：运行 smoke 版 config 验证

- 操作：
  - 创建 `configs/experiments/baseline_smoke.yaml`：

```yaml
experiment:
  name: baseline_smoke
  seeds: [42]
  algorithms: [ppo, ppo_dualclip]
  opponents: [random, rule_based]
  episodes: 1
  max_steps: 20

training:
  total_timesteps: 512
  num_steps: 32
  num_envs: 1
  device: cpu

outputs:
  checkpoint_dir: checkpoints/baseline_smoke
  result_dir: results/baseline_smoke
```

- 验证：

```bash
python -m hybrid_arena.scripts.run_ablation --config configs/experiments/baseline_smoke.yaml
test -f results/baseline_smoke/ablation_raw.csv
test -f results/baseline_smoke/ablation_summary.md
```

---

## 模块 F4：审计 win_rate 与 objective 指标

### Step F4.1：新增 Evaluator 指标一致性测试

- 修改或新增：
  - `hybrid_arena/training/tests/test_evaluator_metrics.py`

- 测试用例：
  - `test_win_rate_denominator_excludes_draws_or_documents_draws`
  - `test_objective_metrics_present_in_eval_result`
  - `test_eval_result_serialization_contains_required_fields`

- 必须字段：
  - `episodes`
  - `win_rate`
  - `draw_rate`
  - `avg_reward`
  - `avg_episode_length`
  - `avg_kills`
  - `avg_deaths`
  - `avg_towers_destroyed`
  - `avg_tower_hp_advantage`
  - `fps`

- 验证：

```bash
pytest hybrid_arena/training/tests/test_evaluator_metrics.py -v
```

### Step F4.2：更新结果表字段

- 修改文件：
  - `hybrid_arena/scripts/run_ablation.py`

- 输出表增加字段：
  - `draw_rate`
  - `avg_towers_destroyed`
  - `avg_tower_hp_advantage`

- 验证：

```bash
python -m hybrid_arena.scripts.run_ablation --config configs/experiments/baseline_smoke.yaml
grep -n "draw_rate" results/baseline_smoke/ablation_summary.md
grep -n "avg_towers_destroyed" results/baseline_smoke/ablation_summary.md
```

---

## 模块 F5：Planner Trace 数据接口

### Step F5.1：定义 planner trace schema

- 新增文件：
  - `hybrid_arena/inference/planner_trace.py`

- 实现：

```python
from dataclasses import dataclass, asdict
from typing import Any


@dataclass
class PlannerTrace:
    episode_id: str
    step: int
    team: str
    planner_state: dict[str, Any]
    macro_action: str
    reward_delta: float
    win: bool | None
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
```

- 验证：
  - 新增 `hybrid_arena/inference/tests/test_planner_trace.py`
  - 测试 `to_dict()` 可 JSON 序列化。

### Step F5.2：新增 trace recorder

- 新增文件：
  - `hybrid_arena/inference/trace_recorder.py`

- 实现：

```python
class PlannerTraceRecorder:
    def __init__(self, output_path: str):
        ...

    def add(self, trace: PlannerTrace) -> None:
        ...

    def flush(self) -> None:
        ...
```

- 行为：
  - 输出 JSONL。
  - 每行一个 `PlannerTrace.to_dict()`。
  - 父目录不存在时自动创建。

- 验证：
  - `hybrid_arena/inference/tests/test_planner_trace.py::test_trace_recorder_writes_jsonl`

### Step F5.3：`play_planner.py` 支持导出 trace

- 修改文件：
  - `hybrid_arena/scripts/play_planner.py`

- 新增参数：
  - `--trace-output results/planner_traces/rule_trace.jsonl`

- 行为：
  - 每次 planner 产生 macro action 时写入一条 trace。
  - 默认不写 trace，只有传入参数才启用。

- 验证：

```bash
python -m hybrid_arena.scripts.play_planner --planner rule --max-steps 20 --render-mode none --trace-output results/planner_traces/rule_trace.jsonl
test -f results/planner_traces/rule_trace.jsonl
python -c "import json; print(json.loads(open('results/planner_traces/rule_trace.jsonl').readline()).keys())"
```

---

## 模块 F6：更新进度与问题文档

### Step F6.1：更新 `docs/progress.md`

- 操作：
  - 新增 `Phase F：验证、纠偏与正式实验准备`。
  - 将完成的 F0-F5 步骤按实际状态标记为 `[x]` 或 `[ ]`。
  - 不要改写已完成模块 0-7 的历史记录。

- 验证：

```bash
grep -n "Phase F" docs/progress.md
```

### Step F6.2：更新 `docs/issues.md`

- 操作：
  - 如果执行中发现新问题，按格式记录：

```markdown
## 新发现问题

### ISSUE-Fx：标题
- 严重级别：P0/P1/P2
- 影响：
- 复现命令：
- 修复状态：
```

- 若无新问题，写：

```markdown
## 新发现问题

（暂无）
```

- 验证：

```bash
grep -n "新发现问题" docs/issues.md
```

### Step F6.3：更新 CHANGELOG

- 修改文件：
  - `CHANGELOG.md`

- 增加 unreleased section：

```markdown
## Unreleased

### Added
- Added baseline experiment config support for ablation runner.
- Added planner trace schema and JSONL recorder.

### Changed
- Expanded CI test coverage from MiniMOBA-only to the full package.
- Clarified GRPO/QLoRA implementation status.

### Fixed
- Fixed documentation status inconsistency around GRPO/QLoRA.
```

- 验证：

```bash
grep -n "Unreleased" CHANGELOG.md
```

---

## 最终验证

执行：

```bash
ruff check hybrid_arena
python -m compileall hybrid_arena
pytest hybrid_arena -v
python -m hybrid_arena.scripts.run_ablation --config configs/experiments/baseline_smoke.yaml --dry-run
python -m hybrid_arena.scripts.run_ablation --config configs/experiments/baseline_smoke.yaml
python -m hybrid_arena.scripts.play_planner --planner rule --max-steps 20 --render-mode none --trace-output results/planner_traces/rule_trace.jsonl
```

全部通过后输出完成报告，格式：

| 模块 | 状态 | 验证命令 | 结果 |
|---|---|---|---|
| F0 | done/blocked | ... | ... |
| F1 | done/blocked | ... | ... |
| F2 | done/blocked | ... | ... |
| F3 | done/blocked | ... | ... |
| F4 | done/blocked | ... | ... |
| F5 | done/blocked | ... | ... |
| F6 | done/blocked | ... | ... |

同时列出：
1. 修改过的文件。
2. 新增的文件。
3. 失败但已修复的问题。
4. 仍阻塞的问题。
5. 是否建议进入正式 `baseline_v1` 长实验。
