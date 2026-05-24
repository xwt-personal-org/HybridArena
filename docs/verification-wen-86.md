# Verification Report: MiniMOBA PettingZoo 4v4 & Action Mask (WEN-86)

## Purpose

Confirm MiniMOBA 4v4 environment, PettingZoo Parallel API, `MultiDiscrete([9,4,9])` action space, and 324-d joint `action_mask` are usable for M1 RL work.

## Scope Checked

| Area | Status | Evidence |
|------|--------|----------|
| `reset` / `step` / observe flow | ✅ PASS | `test_4v4_smoke_reset_step_observe`, existing `test_env.py` |
| Red/blue 4-hero configuration | ✅ PASS | `test_4v4_agent_configuration`, `test_4v4_default_hero_assignments` |
| `MultiDiscrete([9,4,9])` ↔ 324 joint actions | ✅ PASS | `test_4v4_action_space_multidiscrete`, `test_action_encoding.py` |
| `action_mask` legality & encoding alignment | ✅ PASS | `test_4v4_action_mask_*`, `test_action_mask.py` |
| PettingZoo Parallel API (4v4) | ✅ PASS | `test_4v4_parallel_api` |
| Terminal / timeout behavior | ✅ PASS | `test_4v4_timeout_terminates_episode`, `test_reward.py` |
| Reward output on step | ✅ PASS | `test_4v4_smoke_reset_step_observe`, `test_rewards_include_time_penalty` |

## Key Test Paths

Run the WEN-86 verification suite:

```bash
pytest hybrid_arena/minimoba/tests/test_wen86_verification.py -v
```

Run full MiniMOBA smoke subset (includes 2v2 + 4v4):

```bash
pytest hybrid_arena/minimoba/tests -m smoke -v
```

Run all MiniMOBA tests:

```bash
pytest hybrid_arena/minimoba/tests -v
```

## Fixtures & Entry Points

| Resource | Path |
|----------|------|
| Parallel env factory | `hybrid_arena/minimoba/env.py` → `parallel_env()` |
| Default 4v4 config | `hybrid_arena/configs/default.yaml` (`team_size: 4`) |
| Action encoding | `hybrid_arena/minimoba/action_encoding.py` |
| Mask builder | `hybrid_arena/minimoba/game_engine.py` → `_build_action_mask()` |
| WEN-86 tests | `hybrid_arena/minimoba/tests/test_wen86_verification.py` |

## Known Limitations (Follow-up Issues)

| Item | Severity | Tracking |
|------|----------|----------|
| **ISSUE-F13**: objective reward shaping does not produce hard win / base damage learning | P1 | `docs/issues.md` — blocks RL objective claims, not env API |
| Illegal masked actions are not rejected in `step()` (policy must respect mask) | P2 | Documented in `docs/refs/ref-pettingzoo-action-mask.md` |
| Most legacy unit tests use `team_size=2` for speed; 4v4 now covered by WEN-86 suite | Info | — |

## Decision Log

- **2026-05-23**: Added dedicated 4v4 verification tests and this report. Env API, action space, and action mask verified PASS. ISSUE-F13 remains open for RL training objective path (WEN-89 scope).
