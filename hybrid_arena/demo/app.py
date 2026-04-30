"""Streamlit demo for HybridArena.

Usage:
    streamlit run demo/app.py

Requirements:
    pip install streamlit
"""

from __future__ import annotations

import time

import streamlit as st

st.set_page_config(page_title="HybridArena", layout="wide")

st.title("HybridArena: LLM x DRL 混合智能体对战平台")

st.markdown(
    """
    在自研的4v4不完全信息MOBA环境中，对比不同算法和混合方案的表现。
    """
)

with st.sidebar:
    st.header("配置")
    algo = st.selectbox(
        "DRL 算法",
        ["PPO", "Dual-clip PPO", "MAPPO", "QMIX", "COMA"],
    )
    llm_mode = st.selectbox(
        "LLM 模式",
        ["关闭", "Prompt-only", "ReAct", "GRPO-trained"],
    )
    opponent = st.selectbox(
        "对手",
        ["Rule-Based", "Self-Play Best"],
    )
    n_episodes = st.slider("对战局数", 1, 20, 5)
    use_llm_every = st.slider("LLM 调用频率 (每 N 步)", 1, 20, 10)

    st.divider()
    st.markdown(
        """
        **关于本项目**
        - 自研 MiniMOBA-4v4 环境
        - 5 种 DRL 算法统一对比
        - LLM 高层规划 + DRL 微操控制
        - QLoRA GRPO 训练 LLM Planner
        """
    )

if st.button("开始对战", type="primary"):
    progress = st.progress(0)
    status = st.empty()
    log_container = st.container()

    # Simulated run (no actual model loading in this skeleton)
    for ep in range(n_episodes):
        status.text(f"对战中... 第 {ep + 1}/{n_episodes} 局")

        # Simulate steps
        for step in range(100):
            time.sleep(0.01)
            progress.progress((ep * 100 + step) / (n_episodes * 100))

        with log_container:
            st.write(f"局 {ep + 1}: {'胜利' if ep % 2 == 0 else '失败'} | 步数: {80 + ep * 5}")

    status.text("对战完成！")
    progress.empty()

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("胜率", f"{60:.0f}%")
    col2.metric("平均步数", f"{95:.0f}")
    col3.metric("平均奖励", f"{2.3:+.2f}")
    col4.metric("KDA", f"{1.2:.1f}")

    if llm_mode != "关闭":
        with st.expander("LLM 战术思考过程", expanded=True):
            st.markdown("**策略**: 团战")
            st.markdown("**理由**: 敌方全员可见，经济领先，适合集合开团")
            st.markdown("- tank: 先手开团")
            st.markdown("- dps_1: 集火敌方后排")
            st.markdown("- dps_2: 侧翼骚扰")
            st.markdown("- support: 治疗保护")

st.divider()
st.subheader("算法对比表")

st.table(
    {
        "算法": ["Rule-Based", "PPO", "Dual-clip PPO", "MAPPO", "QMIX", "COMA", "Hybrid (LLM+GRPO)"],
        "Win Rate": ["50%", "—", "—", "—", "—", "—", "—"],
        "ELO": ["1000", "—", "—", "—", "—", "—", "—"],
        "训练时间": ["N/A", "~2.5h", "~3h", "~3.5h", "~3.5h", "~3.5h", "~3h+"],
    }
)

st.markdown(
    """
    ---
    **GitHub**: [HybridArena](https://github.com/yourname/hybrid-arena)
    | **W&B**: [实验报告](https://wandb.ai/yourname/hybrid-arena)
    | **技术博客**: [知乎](https://zhuanlan.zhihu.com/yourname)
    """
)
