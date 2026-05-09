"""Ticket labels for telecom-oriented triage."""

TICKET_LABELS: dict[str, str] = {
    "radio_access": "无线接入 / 覆盖 / 掉线",
    "core_network": "核心网 / 注册 / 会话",
    "transport": "传输链路 / 丢包 / 时延",
    "device": "终端 / SIM / 设备",
    "billing": "计费 / 账单 / 套餐",
    "unknown": "信息不足",
}


def normalize_label(label: str) -> str:
    normalized = label.strip().lower().replace(" ", "_").replace("-", "_")
    return normalized if normalized in TICKET_LABELS else "unknown"
