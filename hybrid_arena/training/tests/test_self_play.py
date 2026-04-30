"""Tests for lightweight training self-play pool."""

from hybrid_arena.training.self_play import SelfPlayPool


def test_self_play_pool_add_sample_and_list():
    pool = SelfPlayPool(max_size=2)

    pool.add_checkpoint("a.pt", {"win_rate": 0.4})
    pool.add_checkpoint("b.pt", {"win_rate": 0.6})
    pool.add_checkpoint("c.pt", {"win_rate": 0.5})

    opponents = pool.list_opponents()
    assert [opponent["path"] for opponent in opponents] == ["b.pt", "c.pt"]
    assert pool.sample_opponent(strategy="recent_or_best")["path"] in {"b.pt", "c.pt"}
