"""Tests for evaluator and logger."""

from __future__ import annotations

from hybrid_arena.training.evaluator import Evaluator, evaluate_policy
from hybrid_arena.training.logger import WandbLogger


class TestEvaluatePolicy:
    def test_random_vs_random(self):

        from hybrid_arena.minimoba.env import parallel_env

        env = parallel_env()

        def random_policy(obs, agent_id):
            return env.action_space(agent_id).sample()

        result = evaluate_policy(
            random_policy,
            opponent_fn=random_policy,
            n_episodes=10,
            env_kwargs={},
        )
        assert "win_rate" in result
        assert 0.0 <= result["win_rate"] <= 1.0
        assert result["red_wins"] + result["blue_wins"] + result["draws"] == 10


class TestEvaluator:
    def test_evaluate_records_history(self):
        eval = Evaluator(n_eval_episodes=5)

        def policy(obs, aid):
            import gymnasium

            return gymnasium.spaces.MultiDiscrete([9, 4, 9]).sample()

        result = eval.evaluate(policy, global_step=1000)
        assert result["global_step"] == 1000
        assert len(eval.history) == 1

    def test_summary(self):
        eval = Evaluator(n_eval_episodes=5)
        eval.history = [
            {"win_rate": 0.3, "avg_length": 100},
            {"win_rate": 0.6, "avg_length": 120},
        ]
        summary = eval.summary()
        assert summary["latest_win_rate"] == 0.6
        assert summary["best_win_rate"] == 0.6


class TestWandbLogger:
    def test_disabled_when_no_wandb(self):
        logger = WandbLogger(enabled=False)
        assert not logger.enabled
        logger.log({"x": 1.0}, step=0)
        logger.finish()
