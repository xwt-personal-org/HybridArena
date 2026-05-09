"""Deterministic ticket classifier."""

from __future__ import annotations

from dataclasses import asdict, dataclass

LABEL_KEYWORDS: dict[str, tuple[str, ...]] = {
    "radio_access": ("基站", "覆盖", "掉线", "无线", "小区", "信号"),
    "core_network": ("核心网", "注册", "amf", "smf", "鉴权", "pdu"),
    "transport": ("传输", "链路", "丢包", "抖动", "时延", "拥塞"),
    "device": ("手机", "终端", "设备", "sim", "无服务", "重启"),
    "billing": ("话费", "账单", "扣费", "套餐", "计费", "余额"),
}


@dataclass(frozen=True)
class TicketPrediction:
    label: str
    confidence: float
    evidence_keywords: list[str]
    summary: str

    def to_dict(self) -> dict:
        return asdict(self)


def classify_ticket(ticket_text: str) -> TicketPrediction:
    normalized = ticket_text.lower()
    best_label = "unknown"
    best_hits: list[str] = []
    for label, keywords in LABEL_KEYWORDS.items():
        hits = [keyword for keyword in keywords if keyword.lower() in normalized]
        if len(hits) > len(best_hits):
            best_label = label
            best_hits = hits

    if not best_hits:
        return TicketPrediction(
            label="unknown",
            confidence=0.0,
            evidence_keywords=[],
            summary=_summarize(ticket_text),
        )
    return TicketPrediction(
        label=best_label,
        confidence=min(1.0, len(best_hits) / 3),
        evidence_keywords=best_hits,
        summary=_summarize(ticket_text),
    )


def _summarize(ticket_text: str) -> str:
    cleaned = " ".join(ticket_text.split())
    return cleaned[:50]
