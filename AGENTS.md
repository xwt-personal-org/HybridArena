# AGENTS.md — 项目执行规范

> 全局规范见 `~/.codex/AGENTS.md`，定义了自治迭代规则和 Web 模式契约。
> 本文件只补充项目特定信息。

## 项目信息
- **项目名称**：HybridArena
- **技术栈**：Python 3.10+ + PyTorch 2.x + PettingZoo + CleanRL + Qwen2.5 LLM
- **目标硬件**：RTX 4060 Laptop (8GB VRAM)
- **当前主线**：MiniMOBA / RL + LLM Planner 混合架构
- **应用层**：AgentBench（JD / RAG / 工单）已完成首版，保留维护
- **当前状态**：RL 工程闭环已完成，进入正式实验阶段；ISSUE-F13 待解决

## 启动协议

每次新对话：
1. 检查 `docs/inbox/` 有无新文件 → 有则 merge-back
2. 读 `docs/progress.md` → 当前进度与阶段
3. 读 `docs/issues.md` → 已知问题（特别关注 ISSUE-F13）
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
pip install -e ".[app]"     # with FastAPI + Streamlit (AgentBench)
pip install -e ".[all]"     # everything
```

### 测试
```bash
pytest hybrid_arena/ -v                                    # 全部测试
pytest hybrid_arena/minimoba/tests/ -v                     # 环境测试
pytest hybrid_arena/training/tests/ -v                     # 训练测试
pytest hybrid_arena/algorithms/tests/ -v                   # 算法测试
pytest hybrid_arena/core hybrid_arena/scenarios -v         # AgentBench 应用层测试
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
python -m hybrid_arena.scripts.train --algo ppo_dualclip --seed 42 --total-timesteps 512 --num-steps 32 --device cpu
```

### 评估
```bash
python -m hybrid_arena.scripts.evaluate --opponent rule_based --episodes 3 --seed 42
```

### 演示
```bash
python hybrid_arena/scripts/play_human.py               # 键盘控制
streamlit run hybrid_arena/demo/moba_app.py              # MOBA 主线 Demo
streamlit run hybrid_arena/demo/app.py                   # AgentBench 应用层 Demo
```

## 禁止
- 不直接修改 `game_engine.py` 的核心逻辑而不更新测试
- 不引入未在 `pyproject.toml` 声明的外部依赖
- 不在 MX450 (2GB) 上运行 LLM 推理（目标硬件为 RTX 4060）
- 不提交 `models/`, `runs/`, `checkpoints/` 等生成目录

## 重要文档
- `docs/plan.md` — 活跃计划（RL 主线路线图）
- `docs/progress.md` — 步骤完成状态
- `docs/issues.md` — 执行问题记录（ISSUE-F13 为主线阻塞）
- `docs/architecture.md` — RL 下一阶段架构设计
- `docs/experiment-report-v0.md` — RL 实验报告
- `docs/refs/` — 技术参考（实施方案、适配方案、调研）
- `docs/agentbench-architecture.md` — AgentBench 应用层架构
- `CHANGELOG.md` — 版本变更记录
