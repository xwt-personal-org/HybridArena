# QA / Rating / Tournament 闭环

本文是 Phase 5 的使用说明，不作为机器状态源；机器状态以 `.ai/ledger.json` 为准。声明边界见 `docs/claim-boundaries.md`。

默认 rating 使用 Elo。`TrueSkillLikeRating` 只保留接口占位，未引入硬依赖。

运行 CPU smoke：

```bash
python -m hybrid_arena.scripts.qa_tournament --episodes 4 --seed 7 --output results/qa/smoke
```

输出：

- `qa_tournament.json`
- `qa_tournament.csv`
- `qa_tournament.md`

默认 tournament 口径：

- `evaluation_mode`: `smoke`
- `policy_source`: `rule_based`
- `planner_source`: `none` 或 `macro_adapter_smoke`
- `claim_boundary`: rule-smoke / adapter-smoke，不是 current trained policy validation

生成的 Markdown report 必须包含：

- `Evaluation Mode`
- `Policy Source`
- `Planner Source`
- `Claim Boundary`
- `Open Items`

回归门禁必须包含 objective metrics：

- `hard_win_rate`
- `base_exposed_rate`
- `avg_base_damage`
- `avg_tower_damage`

如果 reward 提升但上述 objective metrics 仍全为 0，则门禁失败，不能宣称策略有效。

Metric source 要求：

- `illegal_action_rate` 来自 action audit；当前 runner 使用 pre-step action mask。
- 如果只能看到 post-legalization action，必须写 `illegal_action_rate_source=post_legalization_only`。
- `planner_override_rate` 必须来自 planner/adapter call count；planner 未启用时写 `planner_override_rate_source=planner_disabled`。
