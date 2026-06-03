# HybridArena 问题记录

## 当前执行阻塞

当前无执行阻塞。ISSUE-F13 长训 gate 已判定通过，允许启动下一阶段长训；见下方记录。

## 已修复的问题

### ISSUE-F13：objective reward shaping 未产生 hard win

- 严重级别：**P1（原主线阻塞）**
- 原始现象：tower_damage 提升，但 `hard_win_rate=0.0`、`base_exposed_rate=0.0`、`avg_base_damage=0.0`。
- 判定输入：WEN-44 scripted objective policy smoke，见 PR [#15](https://github.com/xwt-personal-org/HybridArena/pull/15)，证据 commit `4d1595c2ddfe4cd5649dec732368064e6b918b6a`。
- 记录指标：`hard_win_rate=1.0`、`base_exposed_rate=1.0`、`avg_base_damage=2000.0`、`tower_damage=2400.0`、`avg_reward_margin=29.39999999999999`、`avg_length=163.0`、`terminal_reasons=[base_destroyed, base_destroyed, base_destroyed]`、`conclusion=通过`。
- 长训 gate 结论：**通过**。scripted objective policy 已稳定摧毁塔并触达 base，环境 objective/base 路径可达，允许启动下一阶段 300k-500k 长训。
- 后续处理：本次通过，无需拆分“需修环境 / 需修 reward / 需调整训练”跟进 issue；长训结果仍需单独记录 RL 是否学会 hard win。若后续长训复现失败，再按 reward 尺度或训练设置拆分新 issue。
- WEN-44 状态：scripted objective policy 验证证据满足 ISSUE-F13 gate 判定，可进入 In Review / Done。
- 状态：已解除长训 gate 阻塞（2026-06-02）。

### ISSUE-A3：pytest 新测试目录模块名冲突

- 严重级别：P1
- 现象：组合运行 `hybrid_arena/scenarios` 时，多个目录下的 `test_runner.py` / `test_evaluator.py` 被 pytest 当作同名顶层模块导入。
- 修复：为 `hybrid_arena/core/tests`、`hybrid_arena/scenarios/*/tests`、`hybrid_arena/services/api/tests` 增加 `__init__.py`。
- 状态：已修复，组合测试 37 passed。

### ISSUE-C2：中文 RAG 检索无法召回"网络丢包排查"

- 严重级别：P1
- 现象：`telecom_rag` 对中文长句只提取整段 token，导致 `packet-loss` chunk 未召回。
- 修复：retriever 增加中文单字与 bigram token，保留英文 token 与 tag boost。
- 状态：已修复，`telecom_rag` benchmark `recall_at_k=1.0`。

### ISSUE-A4：FastAPI 测试依赖缺失

- 严重级别：P2
- 现象：`fastapi.testclient` 需要 `httpx`，首次 `app` extras 未声明。
- 修复：`pyproject.toml` 的 `app` extras 增加 `httpx>=0.27`。
- 状态：已修复，API 测试 3 passed。
