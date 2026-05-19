"""Streamlit demo for HybridArena MOBA/RL mainline.

Run:
    streamlit run hybrid_arena/demo/moba_app.py
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="HybridArena — MiniMOBA", layout="wide")

st.title("HybridArena — MiniMOBA 4v4")
st.caption("LLM Planner × DRL Control · 多智能体混合架构研究平台")

with st.sidebar:
    st.header("导航")
    page = st.radio(
        "选择页面",
        ["概览", "环境规格", "训练与评估", "LLM Planner", "AgentBench 应用层"],
    )
    st.divider()
    st.markdown(
        """
        **技术栈**

        - PettingZoo Parallel API
        - PyTorch 2.x
        - PPO / DualClipPPO
        - RTX 4060 (8GB VRAM)
        """
    )

if page == "概览":
    st.subheader("项目定位")
    st.markdown(
        """
        HybridArena 是一个 **LLM 高层规划 + 深度强化学习微操控制** 的混合智能体研究平台。

        载体为 PettingZoo 标准的 **4v4 简化 MOBA 环境（MiniMOBA）**，支持战争迷雾、英雄技能、
        推塔/基地目标系统和队伍经济。
        """
    )

    cols = st.columns(4)
    cols[0].metric("Agent 数量", "8 (4v4)")
    cols[1].metric("动作空间", "324")
    cols[2].metric("算法", "PPO / MAPPO / QMIX / COMA")
    cols[3].metric("全量测试", "180+")

    st.subheader("核心能力")
    st.markdown(
        """
        | 模块 | 说明 |
        |------|------|
        | MiniMOBA 环境 | PettingZoo Parallel API，324 联合动作，action mask |
        | PPO / DualClipPPO | 训练正确性修复，joint categorical policy |
        | 目标系统 | 推塔/基地/经济/胜负条件 |
        | Self-play | 历史策略池 + ELO + 课程学习 |
        | LLM Planner MVP | 高层战术规划（团战/分推/发育/防守/抓人） |
        | 实验系统 | train / evaluate / run_ablation CLI |
        """
    )

    st.subheader("已知问题")
    st.warning(
        "ISSUE-F13：objective reward shaping 提升了 tower_damage，"
        "但 hard_win_rate / base_exposed_rate / avg_base_damage 仍为 0。"
    )

elif page == "环境规格":
    st.subheader("MiniMOBA 环境规格")

    st.markdown("#### 动作空间")
    st.code("MultiDiscrete([9, 4, 9])  # move(9) × skill(4) × target(9) = 324", language="python")

    st.markdown("#### 观测空间")
    st.json(
        {
            "local_map": "(11, 11, 11) float32",
            "self_state": "(20,) float32",
            "teammate_states": "(3, 15) float32",
            "global_info": "(10,) float32",
            "action_mask": "(324,) binary",
        }
    )

    st.markdown("#### 目标系统")
    st.markdown(
        """
        - 每队 2 座塔 + 1 座基地
        - 塔 HP、基地 HP、队伍经济实时更新
        - 基地被摧毁 → 游戏结束（hard win）
        - 超过 max_steps → 按剩余 HP 判定（soft win / draw）
        """
    )

    st.markdown("#### 英雄角色")
    st.markdown("tank / dps_1 / dps_2 / support — 各有不同属性和技能配置")

elif page == "训练与评估":
    st.subheader("训练命令")
    st.code(
        "\n".join(
            [
                "# PPO smoke 训练",
                "python -m hybrid_arena.scripts.train --algo ppo --seed 42 --total-timesteps 512 --num-steps 32 --device cpu",
                "",
                "# DualClipPPO",
                "python -m hybrid_arena.scripts.train --algo ppo_dualclip --seed 42 --total-timesteps 512 --num-steps 32 --device cpu",
            ]
        ),
        language="bash",
    )

    st.subheader("评估命令")
    st.code(
        "python -m hybrid_arena.scripts.evaluate --opponent rule_based --episodes 3 --seed 42 --output results/eval_smoke.json",
        language="bash",
    )

    st.subheader("Ablation")
    st.code(
        "python -m hybrid_arena.scripts.run_ablation --episodes 1 --max-steps 50",
        language="bash",
    )

    st.subheader("Smoke 实验说明")
    st.info(
        "当前 smoke 参数（total_timesteps=512, episodes=3）仅验证流水线可运行，"
        "不代表算法收敛结论。正式实验需使用 configs/experiments/ 中的配置。"
    )

elif page == "LLM Planner":
    st.subheader("LLM Planner MVP")
    st.markdown(
        """
        LLM Planner 负责高层战术决策，输出宏观指令（团战/分推/发育/防守/抓人），
        由 MacroActionAdapter 转化为底层 RL 策略偏置。

        当前实现：
        - **RulePlanner**：基于规则的对照基线
        - **DummyLLMClient**：固定输出 `group_mid`，用于测试
        - **LLMPlanner**：状态机循环（analyze → decide → reflect），支持接入真实 LLM API
        """
    )

    st.code(
        "python -m hybrid_arena.scripts.play_planner --planner rule --max-steps 50 --render-mode none",
        language="bash",
    )

    st.markdown("#### 下一步")
    st.markdown(
        """
        1. 收集 planner trace dataset（≥10,000 条决策记录）
        2. GRPO 前置数据验收
        3. QLoRA GRPO 训练（RTX 4060 8GB，Qwen2.5-1.5B）
        """
    )

else:
    st.subheader("AgentBench 应用层")
    st.markdown(
        """
        AgentBench 将平台的 planner、evaluator、trace 能力扩展到三个业务场景。
        作为应用层子系统独立运行，不影响 MOBA/RL 主线。
        """
    )

    st.markdown("#### 场景")
    st.markdown(
        """
        | 场景 | 说明 |
        |------|------|
        | JD 解析与简历差距分析 | taxonomy + evidence span + gap report |
        | 通信知识库 RAG Copilot | JSONL corpus + token retriever + citations |
        | 网络工单分诊与评测台 | rule classifier + 排障建议 + Macro-F1 |
        """
    )

    st.markdown("#### 启动")
    st.code(
        "\n".join(
            [
                "# AgentBench Streamlit Demo",
                "streamlit run hybrid_arena/demo/app.py",
                "",
                "# FastAPI",
                "uvicorn hybrid_arena.services.api.app:app --reload",
                "",
                "# CLI",
                "python -m hybrid_arena.scripts.agentbench_run --scenario ticket_triage \\",
                "  --input datasets/ticket_samples/ticket_cases.jsonl \\",
                "  --output results/agentbench/ticket_report.json",
            ]
        ),
        language="bash",
    )
