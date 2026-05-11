# AgentBench Skill-Runtime L0/L1 Prototype — Branch Summary

## Implemented Files

| File | Description |
|------|-------------|
| `hybrid_arena/skill_runtime/__init__.py` | Stable public API exports |
| `hybrid_arena/skill_runtime/schema.py` | Frozen dataclasses: Effect, Trigger, ForwardModel, TypedSignature, Skill, WorkspaceEvent, Annotation |
| `hybrid_arena/skill_runtime/workspace.py` | SQLite-backed Workspace with annotations, events, traces, body-schema snapshots |
| `hybrid_arena/skill_runtime/body_schema.py` | BodySchema: applicability filter + salience-ranked affordances |
| `hybrid_arena/skill_runtime/dispatcher.py` | ReflexDispatcher with trigger matching, bid ranking, trace recording |
| `hybrid_arena/skill_runtime/sample_skills.py` | 5 deterministic sample skills (no LLM, no external API) |
| `hybrid_arena/scripts/skill_runtime_demo.py` | CLI demo: `python -m hybrid_arena.scripts.skill_runtime_demo --root . --db .skills/state.db --once` |
| `hybrid_arena/skill_runtime/tests/__init__.py` | Test package init |
| `hybrid_arena/skill_runtime/tests/test_schema.py` | 49 tests covering schema dataclasses |
| `hybrid_arena/skill_runtime/tests/test_workspace.py` | Annotation persistence, query_paths, events, traces |
| `hybrid_arena/skill_runtime/tests/test_dispatcher.py` | Trigger matching, escalation, fallback, trace recording, body-schema summary |
| `hybrid_arena/skill_runtime/tests/test_sample_skills.py` | Sample skill creation and end-to-end dispatch |
| `ref/agentbench-skill-runtime-branch-summary.md` | This file |

## Validation Run Output

```
$ ruff check hybrid_arena/skill_runtime
All checks passed!

$ pytest hybrid_arena/skill_runtime/tests -v
49 passed

$ pytest hybrid_arena/core hybrid_arena/scenarios hybrid_arena/scripts/tests -v
34 passed (no regressions)

$ python -m hybrid_arena.scripts.skill_runtime_demo --root . --db .skills/state.db --once
Selected skill: format_on_save
Success: True
Message: Executed skill 'Format on Save'.
Trace count: 1
```

## Architecture

```
schema.py          ← pure data contracts (frozen dataclasses + enum)
workspace.py       ← SQLite persistence layer (stdlib sqlite3 only)
body_schema.py     ← applicability filter + affordance ranking
dispatcher.py      ← trigger eval → bid ranking → execute → trace
sample_skills.py   ← 5 deterministic mock skills
```

## Bid Formula

```
bid = trigger_score * skill.salience - tonic_inhibition - no_go_penalty
```

- `tonic_inhibition = 0.0` (L0 placeholder)
- `no_go_penalty = skill.no_go_traces * 0.1`
- Winner: highest positive bid
- Fallback: `fallback_planner(event)` if no positive bid and planner provided

## Trigger Kinds

| Kind | Match Logic |
|------|-------------|
| `glob` | `fnmatch(event.path, trigger.spec)` |
| `regex` | `re.fullmatch(trigger.spec, event.path)` |
| `annotation` | `workspace.query_paths(any_tag=trigger.spec)` returns non-empty |

## Known Limitations

1. **No real controller execution** — L0 controllers are labels only; no code is actually run.
2. **Tonic inhibition hardcoded to 0.0** — no adaptive inhibition mechanism yet.
3. **Preconditions limited to tag checks** — only `any_tag` lookups, no complex boolean logic.
4. **No decay/expiry enforcement** — `decay_at` is stored but never checked.
5. **No concurrent access** — single-process SQLite; no WAL or locking.
6. **ForwardModel.callables not supported** — only metadata fields; runtime functions are explicitly excluded from serialisation.
7. **annotation trigger is coarse** — checks if *any* path has the tag, not whether the event path has it.
8. **No skill versioning** — skills are identified by string id only.

## Next Recommended Iteration

- **L2: Real controller execution** — integrate with a sandboxed shell or AST transformer.
- **Adaptive tonic inhibition** — track recent dispatch frequency and suppress low-value skills.
- **Precondition DSL** — support `AND`/`OR`/`NOT` over workspace annotations.
- **Annotation decay enforcement** — prune expired annotations on workspace update.
- **Skill versioning + provenance chain** — track which skill version produced which annotation.
- **Integration with AgentBench trace pipeline** — feed skill traces into the existing `TaskTrace` / `BenchmarkResult` reporting.
