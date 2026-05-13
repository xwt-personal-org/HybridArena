# Web Review Report - 2026-05-13

## Blocking Findings

1. API dispatch can execute `WRITE_FS` sample controllers against the workspace root.
2. `TacticalDispatcher` treats unknown controllers as successful no-op dispatches.
3. `SkillMemoryStore` can persist `created_at=0.0` for new memory records.
4. `parse_workspace_event()` can misclassify plain events with `kind="event"` as protocol envelopes.
5. `consumed_inbox` recorded `.ai/inbox/plan.md` although the prior consumed packet filename differed.

## Required Patch Direction

- Make skill-runtime API dispatch safe by default with an explicit write-capable opt-in.
- Add policy/tool/advice diagnostics for blocked effects and memory hygiene.
- Return explicit tactical dispatch failures for missing or invalid controllers.
- Preserve `created_at` semantics on insert and update.
- Keep the final local state at `NEEDS_REVIEW` for Web Review.
