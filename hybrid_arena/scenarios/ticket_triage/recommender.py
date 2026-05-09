"""Troubleshooting recommendation templates."""

from __future__ import annotations

from hybrid_arena.scenarios.ticket_triage.labels import normalize_label

_STEPS = {
    "radio_access": [
        "核查故障地点的基站、小区和最近一次告警。",
        "对比 RSRP/SINR 等无线覆盖指标与历史基线。",
        "检查是否存在室分、邻区或切换参数异常。",
    ],
    "core_network": [
        "核查 AMF/SMF/UDM 相关告警和注册失败原因码。",
        "检查鉴权、会话建立和 PDU session 流程日志。",
        "按 IMSI/SUPI 维度复现控制面失败路径。",
    ],
    "transport": [
        "检查传输链路端口状态、误码、丢包和时延。",
        "核查上下游节点是否存在拥塞或路由抖动。",
        "对比同链路其他站点是否出现同类异常。",
    ],
    "device": [
        "确认终端型号、系统版本、SIM 状态和网络制式设置。",
        "引导用户重启、换卡或交叉验证其他终端。",
        "检查是否存在终端兼容性或黑名单限制。",
    ],
    "billing": [
        "核查套餐、余额、账单周期和扣费明细。",
        "检查计费事件是否重复、延迟或未同步。",
        "比对客服系统与计费系统中的用户状态。",
    ],
    "unknown": ["补充故障时间、地点、业务类型、影响范围和错误现象后再分诊。"],
}


def recommend_troubleshooting_steps(label: str) -> list[str]:
    return list(_STEPS[normalize_label(label)])
