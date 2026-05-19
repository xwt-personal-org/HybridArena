# QA / Rating / Tournament 闭环

本文是 Phase 5 的使用说明，不作为机器状态源；机器状态以 `.ai/ledger.json` 为准。

默认 rating 使用 Elo。`TrueSkillLikeRating` 只保留接口占位，未引入硬依赖。

运行 CPU smoke：

```bash
python -m hybrid_arena.scripts.qa_tournament --episodes 4 --seed 7 --output results/qa/smoke
```

输出：

- `qa_tournament.json`
- `qa_tournament.csv`
- `qa_tournament.md`

回归门禁必须包含 objective metrics：

- `hard_win_rate`
- `base_exposed_rate`
- `avg_base_damage`
- `avg_tower_damage`

如果 reward 提升但上述 objective metrics 仍全为 0，则门禁失败，不能宣称策略有效。
