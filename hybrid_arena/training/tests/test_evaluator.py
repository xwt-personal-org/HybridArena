"""Tests for evaluator metrics."""

from hybrid_arena.minimoba.agents.random_agent import RandomAgent
from hybrid_arena.minimoba.agents.rule_based import RuleBasedAgent
from hybrid_arena.training.evaluator import Evaluator


def test_evaluator_rule_vs_random_runs():
    rule_agent = RuleBasedAgent()
    random_agent = RandomAgent()
    evaluator = Evaluator(env_kwargs={"map_size": 16, "team_size": 2, "max_steps": 20})

    result = evaluator.evaluate(
        rule_agent.act,
        opponent_policy=random_agent.act,
        n_episodes=2,
        seeds=[42, 43],
    )

    for key in (
        "win_rate",
        "draw_rate",
        "avg_reward",
        "avg_episode_length",
        "avg_kills",
        "avg_deaths",
        "avg_towers_destroyed",
        "avg_tower_hp_advantage",
        "fps",
    ):
        assert key in result


def test_evaluator_checkpoint_opponent_contract():
    class MockCheckpointPolicy:
        def __call__(self, obs):
            mask = obs["action_mask"]
            flat = int(mask.nonzero()[0][0])
            return [flat // 36, (flat % 36) // 9, flat % 9]

    evaluator = Evaluator(env_kwargs={"map_size": 16, "team_size": 2, "max_steps": 10})
    policy = MockCheckpointPolicy()

    result = evaluator.evaluate(
        policy,
        opponent_policy=policy,
        n_episodes=1,
        seeds=[42],
    )

    assert "win_rate" in result
