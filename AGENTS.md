# AGENTS.md — 项目执行规范

> 全局规范见 `~/.codex/AGENTS.md`，定义了自治迭代规则和 Web 模式契约。
> 本文件只补充项目特定信息。

## 项目信息
- **项目名称**：HybridArena
- **技术栈**：Python 3.10+ + PyTorch 2.x + PettingZoo + CleanRL + Qwen2.5 LLM
- **目标硬件**：RTX 4060 Laptop (8GB VRAM)
- **当前状态**：Phase A-E 完成，Phase F 进行中

## 启动协议

每次新对话：
1. 检查 `docs/inbox/` 有无新文件 → 有则 merge-back
2. 读 `docs/progress.md` → 当前进度与阶段
3. 读 `docs/issues.md` → 已知问题
4. 判断任务来源（dispatch / 口头 / 无）→ 对应处理
5. 简短报告状态

## 代码规范
- 格式化：ruff (lint + format)
- 缩进：4 空格
- 命名：snake_case (Python)
- Git 提交：Conventional Commits，中文描述
- 测试框架：pytest
- 测试位置：`hybrid_arena/*/tests/`

## 项目特定规则
- 所有文档使用中文（根目录 `.md` 文件）
- Action space: MultiDiscrete([9, 4, 9]) = move × skill × target = 324
- Observation: Dict with local_map (11,11,11), self_state (20,), teammate_states (3,15), global_info (10,), action_mask (324,)
- Package installed with `pip install -e .` — import as `hybrid_arena`
- Models saved to `models/`, logs to `runs/`, checkpoints to `checkpoints/`

## 关键命令

### 安装
```bash
pip install -e .            # base (env + agents)
pip install -e ".[dev]"     # with test/lint tools
pip install -e ".[rl]"      # with PyTorch
pip install -e ".[all]"     # everything
```

### 测试
```bash
pytest hybrid_arena/ -v                         # 全部测试
pytest hybrid_arena/minimoba/tests/ -v          # 环境测试
pytest hybrid_arena/minimoba/tests/test_api.py -v       # PettingZoo API compliance
pytest hybrid_arena/minimoba/tests/test_env.py -v       # Environment correctness
pytest hybrid_arena/minimoba/tests/test_reward.py -v    # Reward function tests
```

### 静态检查
```bash
ruff check hybrid_arena/
ruff format --check hybrid_arena/
```

### 基准测试
```bash
python hybrid_arena/scripts/benchmark_fps.py            # Target: > 500 FPS
```

### 训练
```bash
python training/train.py --algorithm ppo_dualclip --seed 42
python training/grpo_qlora_trainer.py --model Qwen2.5-1.5B-Instruct
```

### 演示
```bash
python hybrid_arena/scripts/play_human.py               # 键盘控制
streamlit run demo/app.py                                # Web demo
```

## 禁止
- 不直接修改 `game_engine.py` 的核心逻辑而不更新测试
- 不引入未在 `pyproject.toml` 声明的外部依赖
- 不在 MX450 (2GB) 上运行 LLM 推理（目标硬件为 RTX 4060）
- 不提交 `models/`, `runs/`, `checkpoints/` 等生成目录

## 重要文档
- `docs/plan.md` — 活跃计划（含 scope 标签）
- `docs/progress.md` — 步骤完成状态
- `docs/issues.md` — 执行问题记录
- `docs/architecture.md` — 架构文档
- `docs/refs/` — 技术参考（实施方案、适配方案、调研）
- `docs/experiment-report-v0.md` — 实验报告
- `CHANGELOG.md` — 版本变更记录