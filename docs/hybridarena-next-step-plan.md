# HybridArena 代码更新后的下一步计划

生成日期：2026-04-30  
仓库：`w2030298-art/HybridArena`  
基准提交：`fd07734e2c0c08dbfebf0f756034d91c1db731d7`（`[全阶段] 完成 HybridArena 计划模块`）

---

## 0. 当前判断

仓库当前已经完成上一轮 `docs/plan.md` 的模块 0–7，`docs/progress.md` 标记为“全部模块完成”。`docs/issues.md` 当前没有执行阻塞。

但当前仓库仍处于“工程闭环已搭好，正式实验与真实性验证未完成”的阶段。下一步不应继续堆功能，应该先做三件事：

1. **验证代码真实可运行**：本地与 CI 都要覆盖完整测试，而不是只跑 MiniMOBA 子集。
2. **验证训练真实有效**：当前 smoke/ablation 只证明流水线跑通，不证明 PPO/DualClipPPO 收敛或胜率有效。
3. **校正文档与版本叙事**：README、CHANGELOG、`pyproject.toml`、实验报告之间存在“v0.2.0 / Phase D / GRPO 是否完成”的口径不一致。

---

## 1. P0：代码真实性与 CI 加固

### Step P0.1：完整运行本地验证矩阵

#### 操作

在仓库根目录执行：

```bash
python -m pip install -e ".[dev,rl]"
ruff check hybrid_arena
python -m compileall hybrid_arena
pytest hybrid_arena/ -v
python -m hybrid_arena.scripts.train --algo ppo --seed 42 --total-timesteps 512 --num-steps 32 --device cpu
python -m hybrid_arena.scripts.train --algo ppo_dualclip --seed 42 --total-timesteps 512 --num-steps 32 --device cpu
python -m hybrid_arena.scripts.evaluate --opponent rule_based --episodes 3 --seed 42 --output results/eval_smoke.json
python -m hybrid_arena.scripts.run_ablation --episodes 1 --max-steps 50
python -m hybrid_arena.scripts.play_planner --planner rule --max-steps 50 --render-mode none
```

#### 验证

- `pytest hybrid_arena/ -v` 全部通过或只跳过明确标注的渲染/GPU测试。
- `results/eval_smoke.json` 存在，并包含 `win_rate`、`avg_reward`、`avg_episode_length`、`fps`。
- `results/ablation_summary.md` 可重新生成。
- 如失败，将错误写入 `docs/issues.md` 的“当前执行阻塞”区。

---

### Step P0.2：扩展 GitHub Actions CI 覆盖范围

#### 修改文件

- `.github/workflows/ci.yml`

#### 操作

将现有 CI 拆成三个 job：

1. `core-test`
   - Python matrix：3.10 / 3.11 / 3.12
   - 安装：`python -m pip install -e ".[dev]"`
   - 运行：
     ```bash
     ruff check hybrid_arena
     python -m compileall hybrid_arena
     pytest hybrid_arena/minimoba/tests -v
     ```

2. `rl-test`
   - Python：3.10
   - 安装：`python -m pip install -e ".[dev,rl]"`
   - 运行：
     ```bash
     pytest hybrid_arena/algorithms -v
     pytest hybrid_arena/training -v
     python -m hybrid_arena.scripts.train --algo ppo --seed 42 --total-timesteps 256 --num-steps 16 --device cpu
     ```

3. `inference-demo-test`
   - Python：3.10
   - 安装：`python -m pip install -e ".[dev,rl]"`
   - 运行：
     ```bash
     pytest hybrid_arena/inference -v
     python -m hybrid_arena.scripts.play_planner --planner rule --max-steps 20 --render-mode none
     ```

#### 验证

- GitHub Actions 在最新 commit 上出现 3 个 job。
- 所有 job 通过。
- `README.md` 的 badge 改为真实 CI badge：
  ```markdown
  [![CI](https://github.com/w2030298-art/HybridArena/actions/workflows/ci.yml/badge.svg)](https://github.com/w2030298-art/HybridArena/actions/workflows/ci.yml)
  ```

---

### Step P0.3：修正文档与版本口径冲突

#### 修改文件

- `README.md`
- `CHANGELOG.md`
- `pyproject.toml`
- `docs/experiment-report-v0.md`

#### 当前冲突

- `CHANGELOG.md` 写了 `v0.2.0`。
- `pyproject.toml` 仍是 `version = "0.1.0"`。
- `README.md` 亮点中写“QLoRA GRPO训练”，开发状态表中写 Phase D 完成。
- `README.md` Known Limitations 又写“GRPO/QLoRA 不在本阶段实现”。
- `docs/experiment-report-v0.md` 写 GRPO/QLoRA 需要 planner trace 数据后再进入下一阶段。

#### 操作

采用以下统一口径：

- 当前版本：`0.2.0`
- 当前发布名：`v0.2.0-engineering-baseline`
- 已完成：环境、PPO/DualClipPPO/MAPPO/QMIX/COMA 框架、Self-Play、Evaluator、Rule/Dummy Planner、Demo skeleton。
- 未完成：真实 LLM API benchmark、planner trace dataset、GRPO/QLoRA 正式训练、长期收敛实验。
- 将 README 中 Phase D 状态改为：`规划中 / scaffold ready`。
- 将 README 亮点中的“QLoRA GRPO训练”改为“QLoRA GRPO 训练接口预留 / 下一阶段目标”。
- 将 `pyproject.toml` 改为：
  ```toml
  version = "0.2.0"
  ```

#### 验证

```bash
grep -n "0.2.0" pyproject.toml CHANGELOG.md README.md
grep -n "scaffold" README.md
grep -n "正式实验" README.md docs/experiment-report-v0.md
```

---

## 2. P1：正式 RL Baseline 实验

### Step P1.1：新增正式实验配置

#### 新增文件

- `configs/experiments/phase_b_baseline.yaml`

#### 内容

```yaml
experiment:
  name: phase_b_baseline
  seeds: [42, 123, 456]
  algorithms: [ppo, ppo_dualclip]
  opponents: [random, rule_based]
  eval_episodes: 50
  max_steps: 1000

environment:
  map_size: 32
  team_size: 4
  max_steps: 1000

training:
  total_timesteps: 300000
  num_steps: 128
  num_envs: 4
  learning_rate: 0.0003
  gamma: 0.99
  gae_lambda: 0.95
  device: cpu

output:
  checkpoint_dir: checkpoints/phase_b_baseline
  result_dir: results/phase_b_baseline
```

#### 验证

```bash
test -f configs/experiments/phase_b_baseline.yaml
python -c "import yaml; yaml.safe_load(open('configs/experiments/phase_b_baseline.yaml'))"
```

---

### Step P1.2：让 `run_ablation.py` 支持配置文件

#### 修改文件

- `hybrid_arena/scripts/run_ablation.py`

#### 操作

新增 argparse 参数：

```python
parser.add_argument("--config", type=str, default=None)
parser.add_argument("--output-dir", type=str, default="results")
parser.add_argument("--dry-run", action="store_true")
```

新增函数：

```python
def load_ablation_config(path: str) -> dict:
    ...

def build_experiment_matrix(config: dict) -> list[dict]:
    ...
```

矩阵元素格式：

```python
{
    "algo": "ppo",
    "seed": 42,
    "opponent": "rule_based",
    "total_timesteps": 300000,
    "eval_episodes": 50,
    "max_steps": 1000,
}
```

#### 验证

```bash
python -m hybrid_arena.scripts.run_ablation --config configs/experiments/phase_b_baseline.yaml --dry-run
```

输出应列出 12 个实验组合：`2 algos × 3 seeds × 2 opponents`。

---

### Step P1.3：运行正式 baseline 训练与评估

#### 操作

先用 CPU smoke 参数验证：

```bash
python -m hybrid_arena.scripts.run_ablation   --config configs/experiments/phase_b_baseline.yaml   --output-dir results/phase_b_baseline_smoke   --episodes 3   --max-steps 100
```

再在 RTX 4060 机器上运行正式实验：

```bash
python -m hybrid_arena.scripts.run_ablation   --config configs/experiments/phase_b_baseline.yaml   --output-dir results/phase_b_baseline
```

#### 验证

生成以下文件：

```text
results/phase_b_baseline/
├── ablation_raw.csv
├── ablation_summary.md
├── checkpoints/
└── metrics/
```

---

### Step P1.4：新增训练有效性判定脚本

#### 新增文件

- `hybrid_arena/scripts/check_training_signal.py`
- `hybrid_arena/training/tests/test_training_signal.py`

#### 实现接口

```python
def load_ablation_csv(path: str) -> list[dict]:
    ...

def check_reward_improvement(rows: list[dict], min_delta: float = 0.1) -> bool:
    ...

def check_nonzero_objective_events(rows: list[dict]) -> bool:
    ...

def check_eval_episode_lengths(rows: list[dict], min_avg_len: float = 50.0) -> bool:
    ...
```

#### CLI

```bash
python -m hybrid_arena.scripts.check_training_signal   --input results/phase_b_baseline/ablation_raw.csv   --min-reward-delta 0.1   --min-avg-len 50
```

#### 验证

- 如果所有 `win_rate=0` 且 `avg_len` 极短，应返回非零退出码。
- 如果 reward 曲线、objective events、episode length 有合理信号，应返回 0。

---

## 3. P1：环境与评估语义审计

### Step P1.5：审计 win_rate 视角

#### 修改文件

- `hybrid_arena/training/evaluator.py`
- `hybrid_arena/training/tests/test_evaluator.py`

#### 操作

明确 `win_rate` 是从哪个 team / policy side 计算：

```python
@dataclass
class EvalResult:
    policy_team: str
    opponent_type: str
    win_rate: float
    draw_rate: float
    loss_rate: float
    ...
```

新增测试：

```python
def test_evaluator_win_rate_perspective_is_policy_team():
    ...
```

#### 验证

```bash
pytest hybrid_arena/training/tests/test_evaluator.py -v
```

---

### Step P1.6：新增 objective event 统计

#### 修改文件

- `hybrid_arena/minimoba/game_engine.py`
- `hybrid_arena/training/evaluator.py`

#### 操作

在 `infos[agent]` 或 episode summary 中增加：

```python
"towers_destroyed_by_red": int
"towers_destroyed_by_blue": int
"base_destroyed": bool
"winner": "red" | "blue" | "draw" | None
"objective_damage_red": float
"objective_damage_blue": float
```

#### 验证

新增测试：

```python
def test_evaluator_tracks_objective_events():
    ...
```

运行：

```bash
pytest hybrid_arena/minimoba/tests/test_objectives.py hybrid_arena/training/tests/test_evaluator.py -v
```

---

## 4. P2：LLM Planner 数据闭环

### Step P2.1：新增 planner trace 数据结构

#### 新增文件

- `hybrid_arena/inference/traces.py`
- `hybrid_arena/inference/tests/test_traces.py`

#### 实现

```python
@dataclass
class PlannerTrace:
    episode_id: str
    step: int
    team: str
    planner_state: PlannerState
    macro_action: str
    reward_after_k_steps: float
    win_label: int
    metadata: dict[str, Any]
```

实现函数：

```python
def save_trace_jsonl(path: str, traces: list[PlannerTrace]) -> None:
    ...

def load_trace_jsonl(path: str) -> list[PlannerTrace]:
    ...
```

#### 验证

```bash
pytest hybrid_arena/inference/tests/test_traces.py -v
```

---

### Step P2.2：让 planner demo 支持 trace 输出

#### 修改文件

- `hybrid_arena/scripts/play_planner.py`

#### 新增参数

```python
parser.add_argument("--trace-output", type=str, default=None)
parser.add_argument("--trace-horizon", type=int, default=20)
```

#### 运行示例

```bash
python -m hybrid_arena.scripts.play_planner   --planner rule   --max-steps 200   --render-mode none   --trace-output data/planner_traces/rule_seed42.jsonl
```

#### 验证

```bash
test -f data/planner_traces/rule_seed42.jsonl
python -c "from hybrid_arena.inference.traces import load_trace_jsonl; print(len(load_trace_jsonl('data/planner_traces/rule_seed42.jsonl')))"
```

---

### Step P2.3：GRPO 前置数据验收

#### 新增文件

- `docs/grpo-readiness-checklist.md`

#### 内容必须包含

- trace 样本数要求：至少 `10,000` 条 planner decision traces。
- 宏观动作分布要求：每个 macro action 至少出现 `100` 次。
- label 质量要求：每条 trace 有 `reward_after_k_steps` 与 `win_label`。
- baseline 要求：RulePlanner 与 DummyLLMClient 至少各有一份 trace 数据。
- 资源要求：RTX 4060 8GB 下只允许 QLoRA 1.5B 级模型，不直接 full fine-tune。

#### 验证

```bash
test -f docs/grpo-readiness-checklist.md
grep -n "10,000" docs/grpo-readiness-checklist.md
grep -n "QLoRA" docs/grpo-readiness-checklist.md
```

---

## 5. P2：Demo 与发布前收口

### Step P2.4：补齐 Streamlit Demo 的真实路径

#### 修改文件

- `hybrid_arena/demo/app.py`

#### 操作

Demo 必须能执行三条路径：

1. Random vs RuleBased 环境演示。
2. RulePlanner 宏观策略演示。
3. 加载 checkpoint 后的 policy 评估演示。

#### 新增启动命令

在 `README.md` 增加：

```bash
streamlit run hybrid_arena/demo/app.py
```

#### 验证

```bash
python -m compileall hybrid_arena/demo
```

---

### Step P2.5：准备 v0.2.0 Release

#### 前置条件

以下全部满足后再打 tag：

- CI 全部通过。
- `pytest hybrid_arena/ -v` 通过。
- `results/phase_b_baseline/ablation_summary.md` 是正式实验，不是 smoke。
- README 实验表从“待正式实验”替换为真实结果。
- `pyproject.toml` 版本为 `0.2.0`。
- `CHANGELOG.md` 有 `v0.2.0` 条目。
- `docs/issues.md` 无当前执行阻塞。

#### Git 命令

```bash
git status --short
git add .
git commit -m "[release] prepare HybridArena v0.2.0 baseline"
git tag v0.2.0
git push origin master --tags
```

---

## 6. 推荐执行顺序

1. P0.1：先完整跑本地验证矩阵。
2. P0.2：扩展 CI，避免后续回归。
3. P0.3：统一版本与文档口径。
4. P1.1–P1.4：跑正式 baseline，并用脚本判定训练信号。
5. P1.5–P1.6：审计 win_rate 与 objective event 语义。
6. P2.1–P2.3：建立 planner trace 数据闭环。
7. P2.4–P2.5：Demo 收口并发布 v0.2.0。

---

## 7. 暂不建议做的事

- 不要马上开始 GRPO/QLoRA 正式训练；先完成 planner trace dataset。
- 不要继续扩算法数量；当前首要问题是正式实验与指标可信度。
- 不要直接发布 v0.2.0；先解决版本口径与 README 实验表。
- 不要把 smoke ablation 当作论文/简历实验结果。
