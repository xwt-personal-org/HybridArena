# HybridArena — Project Workspace

## Project Status
Phase A (environment + baseline) implemented and tested. Phase B (PPO/DualClipPPO + training) implemented, end-to-end validated.

## Hardware Constraint
- **Target hardware: RTX 4060 Laptop (8GB VRAM)**
- Current dev machine: MX450 (2GB) — sufficient for env + DRL training, not for LLM inference
- See `HybridArena_4060适配方案.md` for full hardware adaptation plan

## Architecture
```
hybrid_arena/
├── minimoba/          # PettingZoo 4v4 MOBA environment ✅
│   ├── env.py         #   MiniMOBAEnv (ParallelEnv)
│   ├── game_engine.py #   GameState (fog of war, combat, simultaneous actions)
│   ├── hero.py        #   HeroConfig, HERO_POOL (tank/dps/support), HeroState
│   ├── map_generator.py   #   Map generation (bases, towers, bushes, obstacles)
│   ├── reward_shaper.py   #   RewardConfig + RewardTracker
│   ├── renderer.py    #   Pygame 2D renderer
│   ├── wrappers.py    #   SingleAgentWrapper (Gymnasium Env)
│   ├── agents/        #   RandomAgent, RuleBasedAgent (FSM baseline)
│   └── tests/         #   API compliance, env correctness, reward tests
├── algorithms/        # PPO/MAPPO/QMIX/COMA + dual-clip PPO
│   ├── networks.py    #   MapEncoder, StateEncoder, ActorCritic (164K params) ✅
│   └── ppo/           #   PPO + DualClipPPO (GAE, clipped value, dual-clip) ✅
├── agents/            # Agent registry (TODO: RL agents)
├── training/          # Trainer, buffer, evaluator, GRPO (TODO)
├── inference/         # LLM planner (Qwen2.5-1.5B/3B) (TODO)
├── demo/              # Streamlit demo (TODO)
├── configs/           # YAML configs
│   └── default.yaml   #   Environment + training defaults
└── scripts/           # play_human.py, benchmark_fps.py
```

## Tech Stack
- Python 3.10+, PyTorch 2.x
- RL: CleanRL fork + PettingZoo + SuperSuit
- LLM: Qwen2.5-1.5B-Instruct (FP16, ~3GB) or 3B (4-bit, ~2.5GB)
- LLM Agent: LangGraph + CrewAI + ReAct/Reflexion
- LLM Training: TRL GRPO + QLoRA
- Experiment tracking: W&B

## Key Commands

### Install
```bash
pip install -e .            # base (env + agents)
pip install -e ".[dev]"     # with test/lint tools
pip install -e ".[rl]"      # with PyTorch
pip install -e ".[all]"     # everything
```

### Test
```bash
pytest hybrid_arena/minimoba/tests/ -v
pytest hybrid_arena/minimoba/tests/test_api.py -v       # PettingZoo API compliance
pytest hybrid_arena/minimoba/tests/test_env.py -v       # Environment correctness
pytest hybrid_arena/minimoba/tests/test_reward.py -v    # Reward function tests
```

### Benchmark
```bash
python hybrid_arena/scripts/benchmark_fps.py            # Target: > 500 FPS
```

### Play
```bash
python hybrid_arena/scripts/play_human.py               # Keyboard-controlled demo
```

### Lint
```bash
ruff check hybrid_arena/
ruff format --check hybrid_arena/
```

### DRL Training (to be implemented)
```bash
python training/train.py --algorithm ppo_dualclip --seed 42
```

### LLM GRPO Fine-tuning (to be implemented)
```bash
python training/grpo_qlora_trainer.py --model Qwen2.5-1.5B-Instruct
```

### Demo (to be implemented)
```bash
streamlit run demo/app.py
```

## Important Notes
- All documentation is in **Chinese** (see root `.md` files)
- `docs/HybridArena_完整项目实施方案.md` — full 4-stage implementation plan
- `docs/HybridArena_4060适配方案.md` — RTX 4060-specific adaptations (model sizes, VRAM, training params)
- `docs/招聘项目方案调研.md` — job-market analysis and project rationale
- Action space: MultiDiscrete([9, 4, 9]) = move × skill × target = 324
- Observation: Dict with local_map (11,11,11), self_state (20,), teammate_states (3,15), global_info (10,), action_mask (324,)
- Package installed with `pip install -e .` — import as `hybrid_arena`
- Models saved to `models/`, logs to `runs/`, checkpoints to `checkpoints/`
