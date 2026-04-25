"""Algorithm implementations (lazy-loaded).

Install with: pip install -e ".[rl]" for PyTorch support.
"""


__all__ = ["ActorCritic", "MapEncoder", "StateEncoder", "PPOConfig", "PPO", "DualClipPPO"]


def __getattr__(name):
    if name in ("ActorCritic", "MapEncoder", "StateEncoder"):
        from hybrid_arena.algorithms.networks import (  # noqa: F811
            ActorCritic,
            MapEncoder,
            StateEncoder,
        )

        _map = {
            "ActorCritic": ActorCritic,
            "MapEncoder": MapEncoder,
            "StateEncoder": StateEncoder,
        }
        return _map[name]
    elif name in ("PPO", "DualClipPPO", "PPOConfig"):
        from hybrid_arena.algorithms.ppo import (  # noqa: F811
            PPO,
            DualClipPPO,
            PPOConfig,
        )

        _map = {"PPO": PPO, "DualClipPPO": DualClipPPO, "PPOConfig": PPOConfig}
        return _map[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
