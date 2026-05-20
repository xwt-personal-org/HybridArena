# HybridArena Five-Phase Micro-Research — 2026-05-19

## Scope

This micro-research distills the Google Drive job-demand analysis into a staged HybridArena iteration plan. It is a supporting reference only. It is not machine state. `.ai/ledger.json` remains the only machine state layer.

## Source Inputs

1. Google Drive: `RL-MOBA 岗位需求迭代分析`
   - Key conclusion: HybridArena should move beyond Python single-machine RL PoC and demonstrate algorithm depth, distributed training, C++/ONNX/TensorRT deployment, LLM cognition, and automated QA.
   - Five paths extracted:
     1. MARL + Offline RL
     2. Distributed Actor/Learner training
     3. ONNX/C++/TensorRT deployment
     4. LLM + RL hybrid brain
     5. Automated QA and rating loop

2. GitHub: `w2030298-art/HybridArena`
   - Current usable branch: `iter/hybridarena-agent-architecture-integration-20260512`
   - Current ledger task: `iter-agent-architecture-l2-lite-20260512`
   - Current ledger state: `NEEDS_REVIEW`
   - Current validation: `PASS`
   - Important limitation: historical report says tower/objective shaping improved tower damage, but hard win, base exposed, and base damage still remained zero. This means phase 1 must include objective-reachability gates before long-training claims.

## Technical Findings

### 1. MARL / Offline RL

HybridArena should formalize centralized training with decentralized execution. The actor must consume local observations only. The critic and offline learner may consume global state. PettingZoo's Parallel API directly supports simultaneous multi-agent action/observation flow and exposes `state()` as a global view appropriate for CTDE-style methods.

Offline RL should start as a CPU-testable foundation: synthetic expert replay schema, BC pretraining, and minimal CQL/IQL interfaces. Real human replay data should be treated as an external dependency. CQL is relevant because it explicitly targets offline RL value overestimation caused by distribution shift between the static dataset and learned policy.

### 2. Distributed Training

Distributed execution should be staged as a local actor/learner skeleton before real cluster work. IMPALA establishes the key architecture: decoupled acting and learning plus V-trace off-policy correction. SEED RL further supports the value of centralized inference and optimized communication for high-throughput RL.

The first implementation should avoid pretending to be production-distributed. It should validate message contracts, policy-version lag, replay backpressure, V-trace math, and a local two-actor smoke run.

### 3. ONNX / C++ / TensorRT Deployment

The plan should provide a portable default path first: PyTorch export to ONNX, ONNX Runtime CPU parity, then optional ONNX Runtime TensorRT Execution Provider and native TensorRT. ONNX Runtime has a C++ API and TensorRT Execution Provider. NVIDIA TensorRT's official quick-start describes ONNX conversion, `trtexec`, TensorRT engine build, C++ runtime API usage, and lower overhead of C++ compared with Python for deployment.

Actual TensorRT validation requires CUDA/TensorRT environment and should be recorded as an environment-gated open item when unavailable.

### 4. LLM Planner x RL Controller

The current `hybrid_arena/inference/` and skill-runtime memory/dispatcher work make HybridArena a good fit for a high-level planner plus low-level controller design. ReAct supports interleaving reasoning traces and task actions. Reflexion supports episodic verbal feedback/memory. In HybridArena, these should become auditable macro-planning traces, not hidden action overrides.

Default validation should use a deterministic `RulePlanner` and an LLM stub, not real API keys.

### 5. QA / Rating Loop

Reward curves alone are insufficient for this project because previous RL work already showed reward/objective mismatch. Phase 5 must gate on objective metrics: `hard_win_rate`, `base_exposed_rate`, `avg_base_damage`, and `avg_tower_damage`, plus illegal action rate and inference latency.

Elo should be the default simple rating. TrueSkill-like interfaces may be added optionally, but dependency/license and implementation stability should be reviewed before making it mandatory.

## Recommended Plan Shape

The implementation should be five staged `scope:review` phases. Every phase should stop at `NEEDS_REVIEW` because each affects architecture, evaluation semantics, or deployment claims. External data, real GPU training, TensorRT installation, and real LLM providers should be marked `scope:escalate` decisions inside ledger `open_items`.

## Non-Goals

- No heavy GPU long training in default validation.
- No proprietary replay ingestion unless the user supplies licensed data.
- No real LLM API dependency for tests.
- No mandatory TensorRT validation in a CPU-only environment.
- No replacement of AgentBench mainline.
- No report file as state.
