# RL/MOBA Tactical Runtime — Branch Summary

Branch: `iter/rlmoba-tactical-runtime-l0-l1`

## Implemented Files

| File | Purpose |
|------|---------|
| `tactical_runtime/__init__.py` | Stable public API exports |
| `tactical_runtime/workspace.py` | `BattlefieldAnnotation`, `GameEvent`, `BattlefieldWorkspace` — spatial annotation storage, decay, observation layer generation |
| `tactical_runtime/schema.py` | `GameEffect`, `GameTrigger`, `GameForwardModel`, `GameSkill` — trigger evaluation, skill schema |
| `tactical_runtime/body_schema.py` | `GameBodySchema` — skill affordance tracking, prompt summary, snapshot |
| `tactical_runtime/dispatcher.py` | `TacticalDispatchResult`, `TacticalDispatcher` — bid-based skill selection with fallback and trace |
| `tactical_runtime/skills.py` | `direction_toward`, `nearest_tagged_region`, 4 deterministic tactical skills, controller registry |
| `tactical_runtime/observation.py` | `crop_pheromone_layer`, `append_pheromone_channels` — optional pheromone layer helpers |
| `tactical_runtime/tests/test_workspace.py` | 15 tests: annotation CRUD, decay, observation layer channels, out-of-bounds |
| `tactical_runtime/tests/test_dispatcher.py` | 9 tests: low-health retreat, enemy detection, escalation, fallback, trace |
| `tactical_runtime/tests/test_skills.py` | 25 tests: direction mapping, nearest region, each tactical skill end-to-end |
| `tactical_runtime/tests/test_observation.py` | 8 tests: crop shape, boundary padding, channel append, mismatch error |
| `ref/rlmoba-tactical-runtime-branch-summary.md` | This file |

## Validation Results

```
ruff check hybrid_arena/minimoba/tactical_runtime → All checks passed
pytest hybrid_arena/minimoba/tactical_runtime/tests -v → 57/57 passed
pytest hybrid_arena/minimoba/tests hybrid_arena/training/tests hybrid_arena/algorithms/tests -v → 97 passed, 1 skipped (pygame)
```

## Architecture

```
TacticalDispatcher
  ├── GameBodySchema (skill affordance tracking)
  │     ├── list[GameSkill] (with GameTrigger evaluations)
  │     └── BattlefieldWorkspace (annotations + events)
  └── fallback_planner (injectable callable)
```

**Dispatch flow:**
1. Event arrives → body schema updates workspace
2. Each skill's triggers scored against workspace + game state
3. Bid = trigger_score × salience − no_go_penalty
4. Highest positive bid wins → controller generates `{move, skill, target}` action
5. No positive bid → fallback planner or escalate

## Known Limitations

- **No spatial indexing**: Annotation queries are O(n) linear scan. Fine for L0/L1 (<100 annotations), will need R-tree or grid bucket for production scale.
- **Controller functions are stateless**: No memory of previous actions or trajectory planning.
- **No skill learning**: All skills are deterministic hand-crafted. No gradient-based skill selection.
- **Observation helpers not wired**: `crop_pheromone_layer` / `append_pheromone_channels` are standalone utilities, not integrated into env.py observation.
- **Trigger evaluation couples to game_state internals**: `_eval_health_threshold` and `_eval_enemy_count` access `game_state.heroes` directly via duck typing (getattr). Works but fragile if GameState schema changes.
- **No concurrent dispatch**: Single-agent dispatch only. Multi-agent parallel dispatch not implemented.

## Next Recommended Iteration

1. **L2: Skill learning** — Add gradient-based skill prior update from reward signal (bandit-style or REINFORCE on skill selection).
2. **L2: Spatial indexing** — Grid-bucket or R-tree for annotation queries at scale.
3. **L2: Multi-agent dispatch** — Parallel dispatch for all 4 team members with coordination.
4. **L2: Trajectory memory** — Skill controllers that track recent actions and adapt.
5. **L3: LLM integration** — Replace fallback planner with actual LLM call for novel situations.
