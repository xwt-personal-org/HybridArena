from __future__ import annotations

from hybrid_arena.scenarios.ticket_triage.labels import TICKET_LABELS, normalize_label


def test_ticket_labels_cover_network_and_unknown_categories() -> None:
    assert {"radio_access", "core_network", "transport", "device", "billing", "unknown"} <= set(
        TICKET_LABELS
    )


def test_normalize_label_returns_unknown_for_invalid_label() -> None:
    assert normalize_label("Radio Access") == "radio_access"
    assert normalize_label("not-a-label") == "unknown"
