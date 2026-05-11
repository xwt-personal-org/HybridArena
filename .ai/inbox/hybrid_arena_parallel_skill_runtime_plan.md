# Web Plan Packet

## Meta
- Project: HybridArena
- Packet Type: patch
- Created: 2026-05-10
- Based On Ledger Task: missing-ledger-bootstrap-required
- Intended Consumer: local execution agent with branch/subagent support
- After Consumed: delete or clear `.ai/inbox/plan.md`
- Final Review State: `NEEDS_REVIEW`

## Context Snapshot

HybridArena 当前仓库主线为 **AgentBench**，MiniMOBA/RL 被保留为 research branch。当前 README 中的模块结构包含：

```text
hybrid_arena/
├── core/                 # TaskInput / TaskRunResult / trace / SQLite / reporting
├── scenarios/            # jd_resume_match / telecom_rag / ticket_triage
├── services/api/         # FastAPI app
├── scripts/              # agentbench_run + research scripts
├── demo/                 # Streamlit AgentBench demo
├── minimoba/             # research branch: PettingZoo MOBA environment
├── algorithms/           # research branch: PPO/MAPPO/QMIX/COMA
└── training/             # research branch: RL trainer/evaluator
```

本轮改动不改变主线定位：  
- AgentBench 继续作为求职展示主线。  
- MiniMOBA/RL 继续作为 research branch。  
- 两条线并行推进，但先不强行抽象共享父类。

## Change Goal

基于两个输入方案生成并执行一次并行 patch：

1. **AgentBench 支线**  
   基于 `skill-runtime-完整技术方案-2026.md`，在 AgentBench 主线中新增一个本地、离线、可测试的代码工程 `skill_runtime` L0/L1 原型：
   - L0: SQLite-backed `Workspace` + annotations + traces
   - L1: `ReflexDispatcher` + trigger/bid/arbitration/escalation
   - `BodySchema`
   - deterministic sample skills
   - CLI smoke demo

2. **RL/MOBA 支线**  
   基于 `skill-runtime-x-hybridarena-融合方案-2026.md`，在 MiniMOBA/RL research branch 中新增一个本地、离线、可测试的 `tactical_runtime` L0/L1 原型：
   - L0: `BattlefieldWorkspace` + pheromone annotations
   - L1: `TacticalDispatcher` + tactical `GameSkill`
   - 3-channel pheromone observation helpers
   - deterministic tactical skills
   - no change to current default MiniMOBA observation contract

3. **最终合并**  
   创建 integration branch 合并两条分支，运行完整验证，更新 `.ai/ledger.json`，并清理 `.ai/inbox/plan.md`。

## Current State From Ledger

`.ai/ledger.json` 当前未在仓库中确认存在。执行端必须先 bootstrap：

```json
{
  "schema_version": "wkstruc-ledger/v9",
  "task_id": "iter-parallel-skill-runtime-20260510",
  "state": "IN_PROGRESS",
  "progress": {"done": 0, "total": 9, "label": "0 / 9 steps"},
  "last_run": {
    "at": "2026-05-10T00:00:00",
    "agent": "local-execution-agent",
    "summary": "Started parallel AgentBench skill runtime and RL/MOBA tactical runtime patch"
  },
  "validation": {"status": "NOT_RUN", "commands": [], "summary": "Not run"},
  "open_items": []
}
```

后续所有状态只写入 `.ai/ledger.json`；必要时可写 `.ai/checkpoint.json`。不要把 README、`docs/report.md`、dashboard 或 `docs/plan.md` 当状态源。

---

# Parallel Execution Strategy

## Branches

从当前 `master` 创建三个分支：

```bash
git checkout master
git checkout -b iter/agentbench-skill-runtime-l0-l1
git checkout master
git checkout -b iter/rlmoba-tactical-runtime-l0-l1
git checkout master
git checkout -b iter/parallel-skill-runtime-integration
```

执行方式：

- Subagent A:
  - branch: `iter/agentbench-skill-runtime-l0-l1`
  - scope: `hybrid_arena/skill_runtime/`, `hybrid_arena/scripts/skill_runtime_demo.py`, relevant tests, `ref/agentbench-skill-runtime-branch-summary.md`

- Subagent B:
  - branch: `iter/rlmoba-tactical-runtime-l0-l1`
  - scope: `hybrid_arena/minimoba/tactical_runtime/`, relevant tests, `ref/rlmoba-tactical-runtime-branch-summary.md`

- Integration:
  - branch: `iter/parallel-skill-runtime-integration`
  - merge both branches with `--no-ff`
  - run all validation commands
  - ledger final state must be `NEEDS_REVIEW`

---

# Patch Steps

## Step 0 — Bootstrap v9 state and branch/subagent setup `scope:auto`

### Files
- `.ai/ledger.json`
- `.ai/inbox/plan.md`
- `.ai/checkpoint.json` only if needed

### Actions
1. Create `.ai/ledger.json` if missing using the bootstrap JSON above.
2. Ensure `.ai/inbox/plan.md` contains this plan packet.
3. Create two feature branches:
   - `iter/agentbench-skill-runtime-l0-l1`
   - `iter/rlmoba-tactical-runtime-l0-l1`
4. Spawn Subagent A and Subagent B with branch-specific scope.
5. Record start in `.ai/ledger.json`.

### Validation
```bash
git branch --list "iter/agentbench-skill-runtime-l0-l1" "iter/rlmoba-tactical-runtime-l0-l1"
python -m json.tool .ai/ledger.json
```

### Done When
- Ledger exists.
- Both feature branches exist.
- Subagents have non-overlapping scopes.
- Ledger progress can be updated.

---

## Branch A — AgentBench Skill Runtime

## Step A1 — Add skill-runtime schema layer `scope:auto`

### Files
- `hybrid_arena/skill_runtime/__init__.py`
- `hybrid_arena/skill_runtime/schema.py`
- `hybrid_arena/skill_runtime/tests/test_schema.py`

### Actions
Create `hybrid_arena/skill_runtime/` and implement:

- `Effect(Enum)`
  - `READ_FS`
  - `WRITE_FS`
  - `RUN_SHELL`
  - `NETWORK`
  - `LLM_CALL`
- `Trigger`
  - fields: `kind`, `spec`, `salience`
- `ForwardModel`
  - fields: `expected_artifacts`, `invariants`, `success_predicate`
  - first version may hold runtime callables, but must not attempt to serialize them
- `TypedSignature`
  - fields: `input_type`, `output_type`, `effects`
- `Skill`
  - fields: `id`, `name`, `triggers`, `salience`, `no_go_traces`, `prior`, `forward_model`, `precision`, `signature`, `controller`, `cost_estimate`, `preconditions`, `repair_skill`, `provenance`
- `WorkspaceEvent`
  - fields: `kind`, `path`, `payload`, `created_at`
- `Annotation`
  - fields: `path`, `tags`, `status`, `last_skill`, `decay_at`, `lineage`

Export stable public names from `__init__.py`.

### Validation
```bash
pytest hybrid_arena/skill_runtime/tests/test_schema.py -v
ruff check hybrid_arena/skill_runtime
```

### Done When
- Dataclasses instantiate cleanly.
- Defaults are deterministic.
- Tests cover enum values, trigger salience, typed signature effects, and annotation metadata.

---

## Step A2 — Add SQLite-backed Workspace `scope:auto`

### Files
- `hybrid_arena/skill_runtime/workspace.py`
- `hybrid_arena/skill_runtime/tests/test_workspace.py`

### Actions
Implement `Workspace(root: Path, db_path: Path)`.

SQLite tables:

```sql
CREATE TABLE IF NOT EXISTS annotations (
    path TEXT PRIMARY KEY,
    tags TEXT,
    status TEXT,
    last_skill TEXT,
    decay_at REAL,
    lineage TEXT
);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY,
    kind TEXT,
    path TEXT,
    payload TEXT,
    created_at REAL
);

CREATE TABLE IF NOT EXISTS traces (
    id INTEGER PRIMARY KEY,
    skill_id TEXT,
    event_kind TEXT,
    input_snapshot TEXT,
    output_snapshot TEXT,
    success INTEGER,
    residual REAL,
    created_at REAL
);

CREATE TABLE IF NOT EXISTS body_schema_snapshots (
    id INTEGER PRIMARY KEY,
    snapshot TEXT,
    created_at REAL
);
```

Implement:

- `annotate(path, tags, status, last_skill, decay_at=None, lineage=None)`
- `query_paths(predicate: dict) -> list[Path]`
  - support `tags_superset`
  - support `status`
  - support `path_glob`
- `emit(event: WorkspaceEvent)`
- `record_trace(skill_id, event_kind, input_snapshot, output_snapshot, success, residual)`
- `snapshot_annotations() -> list[Annotation]`

Constraints:

- Use stdlib `sqlite3`.
- Use JSON encoding for sets/lists/dicts.
- Do not require `watchdog` in first iteration.
- Do not introduce `sqlite-vec` in this iteration.

### Validation
```bash
pytest hybrid_arena/skill_runtime/tests/test_workspace.py -v
ruff check hybrid_arena/skill_runtime
```

### Done When
- Annotations persist after reopening `Workspace`.
- `query_paths({"tags_superset": ["needs_test", "py"]})` works.
- Events and traces are persisted.
- No external dependency is required.

---

## Step A3 — Add BodySchema and ReflexDispatcher `scope:auto`

### Files
- `hybrid_arena/skill_runtime/body_schema.py`
- `hybrid_arena/skill_runtime/dispatcher.py`
- `hybrid_arena/skill_runtime/tests/test_dispatcher.py`

### Actions

Implement `BodySchema`:

- `__init__(skills, workspace)`
- `update(event)`
- `current_affordances(top_k=8)`
- `to_prompt_summary()`
- `snapshot()`

Applicability rules:

- If `skill.preconditions` is empty, skill is considered available.
- If preconditions exist, all must pass.
- Ranking is by current applicability × salience.

Implement `DispatchResult`:

- `skill_id: str | None`
- `action: object | None`
- `escalated: bool`
- `success: bool`
- `residual: float`
- `message: str`

Implement `ReflexDispatcher`:

- Trigger support:
  - `glob`
  - `regex`
  - `annotation`
- Bid formula:
  - `bid = trigger_score * skill.salience - tonic_inhibition - no_go_penalty`
- Winner:
  - highest positive bid
- Escalation:
  - if no positive bid, call injected `fallback_planner` if provided
  - otherwise return escalated result with `action=None`
- Trace:
  - record both winning skill execution and escalation
- No-Go:
  - first version may implement `no_go_penalty = 0.0`
  - keep method boundary for future learning

### Validation
```bash
pytest hybrid_arena/skill_runtime/tests/test_dispatcher.py -v
ruff check hybrid_arena/skill_runtime
```

### Done When
- Matching event selects expected skill.
- Unmatched event escalates.
- BodySchema summary lists only available affordances.
- Trace count increments after dispatch.

---

## Step A4 — Add deterministic sample skills and CLI smoke demo `scope:review`

### Files
- `hybrid_arena/skill_runtime/sample_skills.py`
- `hybrid_arena/scripts/skill_runtime_demo.py`
- `hybrid_arena/skill_runtime/tests/test_sample_skills.py`
- `ref/agentbench-skill-runtime-branch-summary.md`

### Actions

Implement deterministic sample skills:

1. `format_on_save`
   - trigger: `glob` for `*.py`
   - controller: no-op/mock formatting annotation
   - forward predicate: trace exists and annotation status is `passing`

2. `add_test_for_new_function`
   - trigger: `annotation` with tags `["needs_test", "py"]`
   - controller: creates a minimal pytest skeleton under `tests/`
   - no LLM call

3. `fix_failing_test`
   - trigger: `annotation` status `failing`
   - controller: toggles annotation to `passing` in fixture-only test

4. `update_imports_after_rename`
   - trigger: synthetic `rename` event
   - controller: deterministic string-level import update in a temp fixture

5. `escalate_when_stuck`
   - not a normal skill
   - represented only through dispatcher fallback

Add CLI:

```bash
python -m hybrid_arena.scripts.skill_runtime_demo --root . --db .skills/state.db --once
```

The CLI must:

- create `.skills/state.db`
- register sample skills
- run one synthetic event
- print selected skill
- print trace count

Write `ref/agentbench-skill-runtime-branch-summary.md` with:

- implemented files
- validation run
- known limitations
- next recommended iteration

### Validation
```bash
python -m hybrid_arena.scripts.skill_runtime_demo --root . --db .skills/state.db --once
pytest hybrid_arena/skill_runtime/tests -v
pytest hybrid_arena/core hybrid_arena/scenarios hybrid_arena/services/api hybrid_arena/scripts/tests -v
ruff check hybrid_arena
```

### Done When
- Skill runtime package is testable.
- CLI creates SQLite DB and writes at least one trace.
- Existing AgentBench tests still pass.
- Branch summary exists.

---

## Branch B — RL/MOBA Tactical Runtime

## Step B1 — Add BattlefieldWorkspace and pheromone annotations `scope:auto`

### Files
- `hybrid_arena/minimoba/tactical_runtime/__init__.py`
- `hybrid_arena/minimoba/tactical_runtime/workspace.py`
- `hybrid_arena/minimoba/tactical_runtime/tests/test_workspace.py`

### Actions

Create `hybrid_arena/minimoba/tactical_runtime/`.

Implement:

- `BattlefieldAnnotation`
  - `position: tuple[int, int]`
  - `tags: set[str]`
  - `intensity: float`
  - `last_agent: str`
  - `decay_rate: float`
  - `lineage: list[str]`
  - `step_decay() -> bool`

- `GameEvent`
  - `kind: str`
  - `tick: int`
  - `position: tuple[int, int]`
  - `data: dict`

- `BattlefieldWorkspace`
  - `map_size`
  - `annotations`
  - `events`
  - `landmarks`
  - `annotate(position, tags, agent, intensity=1.0, decay_rate=0.05)`
  - `query_regions(predicate)`
  - `tick(game_state)`
  - `to_observation_layer() -> np.ndarray`

Observation channels:

- channel 0: `danger`
  - tags: `dangerous`, `enemy_spotted`, `contested`
- channel 1: `opportunity`
  - tags: `resource_soon`, `our_control`, `objective`
- channel 2: `control`
  - positive for `our_control`
  - negative for `enemy_control` or `contested`

Constraints:

- Do not alter existing MiniMOBA env observation contract.
- Do not wire into training loop yet.
- Keep deterministic behavior.

### Validation
```bash
pytest hybrid_arena/minimoba/tactical_runtime/tests/test_workspace.py -v
ruff check hybrid_arena/minimoba/tactical_runtime
```

### Done When
- Annotation decay is tested.
- Tag query is tested.
- Observation layer shape is `(map_size, map_size, 3)`.
- Channel semantics are tested.

---

## Step B2 — Add tactical schema, body schema, dispatcher `scope:auto`

### Files
- `hybrid_arena/minimoba/tactical_runtime/schema.py`
- `hybrid_arena/minimoba/tactical_runtime/body_schema.py`
- `hybrid_arena/minimoba/tactical_runtime/dispatcher.py`
- `hybrid_arena/minimoba/tactical_runtime/tests/test_dispatcher.py`

### Actions

Implement:

- `GameEffect(Enum)`
  - `MOVE`
  - `ATTACK`
  - `USE_ABILITY`
  - `RETREAT`
  - `LLM_CALL`

- `GameTrigger`
  - `kind`
  - `spec`
  - `salience`
  - `evaluate(workspace, game_state, agent_id) -> float`

Supported trigger kinds:

- `health_threshold`
- `enemy_count`
- `team_state`
- `annotation_query`

Implement:

- `GameForwardModel`
- `GameSkill`
- `GameBodySchema`
- `TacticalDispatchResult`
- `TacticalDispatcher`

Dispatcher rules:

- Candidate skill if any trigger returns > 0.
- Bid:
  - `trigger_score * skill.salience - tonic_inhibition - no_go_penalty`
- Winner:
  - highest positive bid
- Fallback:
  - call injected fake planner only if no positive bid
- Return action compatible with current action semantics:
  ```python
  {"move": int, "skill": int, "target": int}
  ```

No-Go:

- keep method boundary
- first version can return `0.0`

Trace:

- use in-memory list first, or SQLite-compatible dicts
- no W&B or external logging

### Validation
```bash
pytest hybrid_arena/minimoba/tactical_runtime/tests/test_dispatcher.py -v
ruff check hybrid_arena/minimoba/tactical_runtime
```

### Done When
- Low health selects retreat.
- Nearby enemy/team state selects relevant skill when configured.
- No matching trigger escalates through fake planner.
- Returned action keys are exactly `move`, `skill`, `target`.

---

## Step B3 — Add deterministic tactical skills and observation helpers `scope:review`

### Files
- `hybrid_arena/minimoba/tactical_runtime/skills.py`
- `hybrid_arena/minimoba/tactical_runtime/observation.py`
- `hybrid_arena/minimoba/tactical_runtime/tests/test_skills.py`
- `hybrid_arena/minimoba/tactical_runtime/tests/test_observation.py`
- `ref/rlmoba-tactical-runtime-branch-summary.md`

### Actions

Implement helper functions:

- `direction_toward(target_pos, current_pos) -> int`
  - map to current move index semantics as close as possible
  - if exact mapping is unknown, document mapping in test fixture
- `nearest_tagged_region(workspace, current_pos, tag) -> tuple[int, int] | None`

Implement deterministic tactical skills:

1. `retreat_when_low`
   - trigger: `health_threshold below 0.25`
   - action: move toward own base, no skill, no target

2. `farm_resources`
   - trigger: `annotation_query` for `resource_soon` or `opportunity`
   - action: move toward nearest resource/opportunity region

3. `control_vision`
   - trigger: `annotation_query` for `vision_loss` or `dangerous`
   - action: move toward nearest relevant region or fallback to safe patrol

4. `push_objective`
   - trigger: `team_state` or `annotation_query` for `objective`
   - action: move toward objective landmark

5. fallback planner
   - injectable fake planner in tests
   - no real LLM/API calls

Observation helpers:

- `crop_pheromone_layer(workspace_layer, center, view_size=11) -> np.ndarray`
- `append_pheromone_channels(local_map, pheromone_crop) -> np.ndarray`
  - `(11, 11, 11)` + `(11, 11, 3)` -> `(11, 11, 14)`

Do not change current MiniMOBA default observation shape.

Write `ref/rlmoba-tactical-runtime-branch-summary.md` with:

- implemented files
- validation run
- known limitations
- next recommended iteration

### Validation
```bash
pytest hybrid_arena/minimoba/tactical_runtime/tests -v
pytest hybrid_arena/minimoba/tests hybrid_arena/training/tests hybrid_arena/algorithms/tests -v
ruff check hybrid_arena
```

### Done When
- Tactical runtime package is testable.
- Pheromone helper can produce `(11, 11, 14)` from `(11, 11, 11)` + `(11, 11, 3)`.
- Existing MiniMOBA/RL tests still pass.
- Branch summary exists.

---

## Merge Coordination

## Step M1 — Merge both feature branches into integration branch `scope:review`

### Files
- `.ai/ledger.json`
- `.ai/checkpoint.json` if used
- `ref/agentbench-skill-runtime-branch-summary.md`
- `ref/rlmoba-tactical-runtime-branch-summary.md`
- all files from Branch A and Branch B

### Actions

```bash
git checkout master
git checkout -b iter/parallel-skill-runtime-integration
git merge --no-ff iter/agentbench-skill-runtime-l0-l1
git merge --no-ff iter/rlmoba-tactical-runtime-l0-l1
```

Merge rules:

- Resolve only real merge conflicts.
- Keep two runtimes separate:
  - AgentBench: `hybrid_arena/skill_runtime/`
  - RL/MOBA: `hybrid_arena/minimoba/tactical_runtime/`
- Do not prematurely extract shared base classes.
- Do not update README unless required by tests or packaging. If README is updated, it must remain descriptive only, not state-bearing.

### Validation
```bash
python -m compileall hybrid_arena

pytest hybrid_arena/skill_runtime/tests -v
pytest hybrid_arena/minimoba/tactical_runtime/tests -v

pytest hybrid_arena/core hybrid_arena/scenarios hybrid_arena/services/api hybrid_arena/scripts/tests -v
pytest hybrid_arena/minimoba/tests hybrid_arena/training/tests hybrid_arena/algorithms/tests -v

ruff check hybrid_arena
```

### Ledger Update

If all commands pass:

```json
{
  "state": "NEEDS_REVIEW",
  "progress": {"done": 9, "total": 9, "label": "9 / 9 steps"},
  "validation": {
    "status": "PASS",
    "commands": [
      "python -m compileall hybrid_arena",
      "pytest hybrid_arena/skill_runtime/tests -v",
      "pytest hybrid_arena/minimoba/tactical_runtime/tests -v",
      "pytest hybrid_arena/core hybrid_arena/scenarios hybrid_arena/services/api hybrid_arena/scripts/tests -v",
      "pytest hybrid_arena/minimoba/tests hybrid_arena/training/tests hybrid_arena/algorithms/tests -v",
      "ruff check hybrid_arena"
    ],
    "summary": "Parallel AgentBench skill runtime and RL/MOBA tactical runtime integrated; ready for Web Review."
  },
  "open_items": []
}
```

If some commands fail:

- keep `state: NEEDS_REVIEW`
- set `validation.status` to `FAIL` or `PARTIAL`
- record exact failing command and failure summary in `open_items`

### Done When
- Integration branch contains both runtimes.
- Branch summaries exist under `ref/`.
- Full validation result is written to `.ai/ledger.json`.

---

## Step M2 — Consume inbox plan `scope:auto`

### Files
- `.ai/inbox/plan.md`
- `.ai/ledger.json`

### Actions
1. Delete or clear `.ai/inbox/plan.md` after successful final validation.
2. Update `.ai/ledger.json.last_run.summary`.
3. Keep final state as `NEEDS_REVIEW`, not `DONE`.

### Validation
```bash
test ! -s .ai/inbox/plan.md || test ! -f .ai/inbox/plan.md
python -m json.tool .ai/ledger.json
```

### Done When
- Inbox plan is consumed.
- Ledger is valid JSON.
- Final integration branch is ready for Web Review.

---

# Decisions / Risks

## Decision 1 — First iteration only implements L0/L1

This patch intentionally focuses on minimal viable L0/L1:

- AgentBench:
  - `Workspace`
  - annotations
  - traces
  - `BodySchema`
  - `ReflexDispatcher`
- RL/MOBA:
  - `BattlefieldWorkspace`
  - pheromone field
  - `GameSkill`
  - `TacticalDispatcher`
  - observation helper

L2/L3/L4 are preserved as future architecture, not fully implemented now.

## Decision 2 — No heavy dependency expansion

Do not add `watchdog`, `tree-sitter`, `sqlite-vec`, `networkx`, Anthropic SDK, or new LLM dependencies in this patch unless already present and required by failing tests.

Use:

- stdlib `sqlite3`
- stdlib `re`
- stdlib `fnmatch`
- existing `numpy`
- existing pytest/ruff toolchain

## Decision 3 — No real LLM in tests

LLM escalation is represented as injected fallback function/fake planner only.

## Decision 4 — Do not change MiniMOBA default observation contract

RL/MOBA branch may add:

- `to_observation_layer()`
- `crop_pheromone_layer()`
- `append_pheromone_channels()`

But must not alter the default env observation returned by current tests.

## Decision 5 — Do not over-abstract

AgentBench code-agent runtime and RL/MOBA tactical runtime are structurally similar, but this patch should not force a common base package. Validate both first.

# Acceptance

- `.ai/ledger.json` exists and uses v9 minimal fields.
- `.ai/inbox/plan.md` contains this packet during execution and is deleted/cleared after consumption.
- Branch `iter/agentbench-skill-runtime-l0-l1` implements `hybrid_arena/skill_runtime/`.
- Branch `iter/rlmoba-tactical-runtime-l0-l1` implements `hybrid_arena/minimoba/tactical_runtime/`.
- Both branches include deterministic tests.
- Integration branch `iter/parallel-skill-runtime-integration` merges both branches.
- Validation commands are run and recorded in `.ai/ledger.json`.
- Final ledger state is `NEEDS_REVIEW`.
- No external LLM/API call is required.
- No `docs/plan.md` is created.
- No doctor tooling is created or run.

