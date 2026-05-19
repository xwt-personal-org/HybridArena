import numpy as np
import pytest
import torch

from hybrid_arena.algorithms.offline.bc import BehaviorCloningTrainer, flatten_observation
from hybrid_arena.algorithms.offline.cql import DiscreteCQLTrainer
from hybrid_arena.algorithms.offline.iql import DiscreteIQLTrainer
from hybrid_arena.data.offline_replay.synthetic_expert import generate_synthetic_expert_replay
from hybrid_arena.training.offline_pretrain import (
    behavior_cloning_smoke,
    objective_reachability_guard,
)


def test_behavior_cloning_loss_decreases_on_synthetic_fixture():
    transitions = generate_synthetic_expert_replay(episodes=1, steps_per_episode=4, seed=7)
    result = behavior_cloning_smoke(transitions, epochs=12)
    assert result["final_loss"] < result["initial_loss"]


def test_behavior_cloning_direct_interface_returns_finite_loss():
    transitions = generate_synthetic_expert_replay(episodes=1, steps_per_episode=2, seed=8)
    obs = np.asarray([flatten_observation(t.observation) for t in transitions], dtype=np.float32)
    actions = np.asarray([BehaviorCloningTrainer.action_to_index(t.action) for t in transitions])
    trainer = BehaviorCloningTrainer(input_dim=obs.shape[1], seed=8)
    loss = trainer.train_epoch(obs, actions)
    assert np.isfinite(loss)


def test_cql_iql_losses_are_finite_and_mask_aware():
    torch.manual_seed(0)
    observations = torch.randn(4, 12)
    actions = torch.tensor([0, 1, 2, 3])
    target_q = torch.ones(4)
    mask = torch.ones(4, 324)
    mask[:, 10:] = 0
    cql_loss = DiscreteCQLTrainer(input_dim=12).compute_loss(observations, actions, target_q, mask)
    iql_loss = DiscreteIQLTrainer(input_dim=12).compute_loss(observations, actions, target_q)
    assert torch.isfinite(cql_loss)
    assert torch.isfinite(iql_loss)


def test_objective_reachability_guard_blocks_zero_objective_claims():
    with pytest.raises(ValueError, match="Objective reachability failed"):
        objective_reachability_guard(
            {
                "hard_win_rate": 0.0,
                "base_exposed_rate": 0.0,
                "avg_base_damage": 0.0,
                "avg_tower_damage": 0.0,
            }
        )
    objective_reachability_guard({"avg_tower_damage": 1.0})
