# QA Tournament Report

- Rating system: elo
- Final rating: 1004.00
- Evaluation Mode: checkpoint_bound
- Policy Source: checkpoint
- Planner Source: none
- Claim Boundary: QA reports are smoke evidence unless policy/planner artifacts prove otherwise.
- Open Items: see per-scenario rows

| Scenario | Evaluation Mode | Policy Source | Planner Source | Win | Hard Win | Base Exposed | Tower Damage | Gate |
|---|---|---|---|---:|---:|---:|---:|---|
| checkpoint_policy_vs_random_smoke | checkpoint_bound | checkpoint | none | 0.000 | 0.000 | 0.000 | 150.000 | PASS |

## checkpoint_policy_vs_random_smoke
- Evaluation Mode: checkpoint_bound
- Policy Source: checkpoint
- Policy Artifact: results/checkpoints/checkpoint-evidence-smoke.pt
- Planner Source: none
- Claim Boundary: Checkpoint-bound QA for the supplied artifact only; not high-skill policy proof.
- Open Items: tiny checkpoint evidence chain; strategy quality not established
- Metric Sources: illegal_action_rate=pre_step_action_mask, planner_override_rate=planner_disabled
| checkpoint_objective_stress_smoke | checkpoint_bound | checkpoint | none | 0.250 | 0.000 | 0.000 | 390.000 | PASS |

## checkpoint_objective_stress_smoke
- Evaluation Mode: checkpoint_bound
- Policy Source: checkpoint
- Policy Artifact: results/checkpoints/checkpoint-evidence-smoke.pt
- Planner Source: none
- Claim Boundary: Checkpoint-bound QA for the supplied artifact only; not high-skill policy proof.
- Open Items: tiny checkpoint evidence chain; strategy quality not established
- Metric Sources: illegal_action_rate=pre_step_action_mask, planner_override_rate=planner_disabled
