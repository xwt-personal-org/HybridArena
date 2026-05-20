# Checkpoint Evidence Chain

本文记录 Topic A 的真实 checkpoint 证据链，不作为机器状态源；机器状态以 `.ai/ledger.json` 为准。

## 证据链目标

本分支把“训练产出的 checkpoint”接到后续 deployment 与 QA 路径：

1. CPU tiny train 生成 checkpoint。
2. 使用同一 checkpoint 做 `checkpoint_bound` ONNX export。
3. 使用 ONNX Runtime CPU 做 PyTorch / ONNX parity。
4. 使用 checkpoint policy source 跑 QA tournament，并在报告中写明 artifact 与 claim boundary。

## 生成方式

本轮没有使用 fixture checkpoint。checkpoint 来自一个很小的 CPU PPO 训练命令：

```bash
python -m hybrid_arena.scripts.train --algo ppo --seed 7 --total-timesteps 16 --num-steps 8 --num-envs 1 --map-size 16 --team-size 2 --max-steps 20 --device cpu --checkpoint-dir results/checkpoints --save-interval 16
```

随后将训练产物复制为固定证据文件：

```bash
Copy-Item results/checkpoints/ppo_seed7_step16.pt results/checkpoints/checkpoint-evidence-smoke.pt -Force
```

记录配置见 `configs/experiments/checkpoint_evidence_smoke.yaml`。

## 产物

- Checkpoint: `results/checkpoints/checkpoint-evidence-smoke.pt`
- Checkpoint SHA256: `d064713680561a84748313a542d3fc74ef0e288e3ce80dfce421c36bfd79a1e2`
- ONNX: `results/deployment/checkpoint_onnx/policy.onnx`
- ONNX SHA256: `ef69765fa0cf7f6a28edaa57aa8cd2691290d514ff4ea428ba20ae2638198a0d`
- Export metadata: `results/deployment/checkpoint_onnx/policy.onnx.json`
- ONNX parity report: `results/deployment/checkpoint_onnx/onnx_parity.json`
- QA report: `results/qa/checkpoint_eval/qa_tournament.md`

## 验证命令

```bash
python -m hybrid_arena.deployment.export_onnx --checkpoint results/checkpoints/checkpoint-evidence-smoke.pt --export-mode checkpoint_bound --output results/deployment/checkpoint_onnx/policy.onnx --seed 7
python -m hybrid_arena.deployment.onnx_validate --model results/deployment/checkpoint_onnx/policy.onnx --seed 7
python -m hybrid_arena.scripts.qa_tournament --episodes 4 --seed 7 --output results/qa/checkpoint_eval --policy-source checkpoint --checkpoint results/checkpoints/checkpoint-evidence-smoke.pt
```

本轮结果：

- ONNX export metadata: `export_mode=checkpoint_bound`
- Checkpoint load format: `network_state_dict`
- ONNX parity: `passed=true`, `max_abs_diff=5.960464477539063e-08`, `atol=0.0001`
- QA policy source: `checkpoint`
- QA evaluation mode: `checkpoint_bound`
- QA policy artifact: `results/checkpoints/checkpoint-evidence-smoke.pt`
- QA illegal action rate source: `pre_step_action_mask`
- QA planner source: `none`

## 证明了什么

- checkpoint 文件可以被 deployment loader 加载，且没有 missing/unexpected keys。
- ONNX metadata 能把该 artifact 标记为 `checkpoint_bound`，并记录 checkpoint hash。
- ONNX Runtime CPU 与 PyTorch policy wrapper 在固定 dummy input 上 parity 通过。
- QA tournament 能以 checkpoint policy source 运行，并把 artifact path、evaluation mode、metric source 和 claim boundary 写入报告。

## 没证明什么

- 没证明该 checkpoint 是高水平策略；训练预算只有 16 timesteps。
- 没证明多 seed 稳定性、长期收敛、真实 MOBA 策略质量或人类水平表现。
- 没证明 C++ ONNX Runtime 推理；该项在 Topic B 分支单独验证。
- 没证明 TensorRT 加速；仍需要 CUDA/TensorRT engine build 与 latency 对比证据。

## 复现边界

本证据链适合作为 artifact wiring / claim-boundary smoke。后续若要升级为训练质量证据，至少需要更长训练、多 seed、固定 evaluator、checkpoint registry、objective gates 与对照 baseline，并继续保留 `checkpoint_bound` metadata。
