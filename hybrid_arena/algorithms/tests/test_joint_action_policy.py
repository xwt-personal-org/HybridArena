"""Tests for the ActorCritic joint action policy."""

import torch

from hybrid_arena.algorithms.networks import ActorCritic
from hybrid_arena.minimoba.action_encoding import N_ACTIONS, decode_action, encode_action


def _make_obs(batch_size: int) -> dict[str, torch.Tensor]:
    return {
        "local_map": torch.zeros(batch_size, 11, 11, 11, dtype=torch.float32),
        "self_state": torch.zeros(batch_size, 20, dtype=torch.float32),
        "teammate_states": torch.zeros(batch_size, 3, 15, dtype=torch.float32),
        "global_info": torch.zeros(batch_size, 10, dtype=torch.float32),
    }


def test_policy_respects_single_valid_action_mask():
    torch.manual_seed(0)
    network = ActorCritic(hidden_dim=16)
    obs = _make_obs(batch_size=1)
    valid_index = encode_action(5, 2, 2)
    action_mask = torch.zeros(1, N_ACTIONS, dtype=torch.int8)
    action_mask[0, valid_index] = 1
    expected_action = torch.tensor(decode_action(valid_index))

    for _ in range(20):
        action, _, _, _ = network.get_action_and_value(obs, action_mask=action_mask)
        assert torch.equal(action[0].cpu(), expected_action)


def test_log_prob_recompute_matches_sampled_action():
    torch.manual_seed(1)
    network = ActorCritic(hidden_dim=16)
    obs = _make_obs(batch_size=3)
    action_mask = torch.ones(3, N_ACTIONS, dtype=torch.int8)

    action, _, _, _ = network.get_action_and_value(obs, action_mask=action_mask)
    _, log_prob, _, _ = network.get_action_and_value(
        obs,
        action=action,
        action_mask=action_mask,
    )

    assert log_prob.shape == (3,)
    assert torch.isfinite(log_prob).all()


def test_no_nan_when_mask_has_valid_action():
    torch.manual_seed(2)
    network = ActorCritic(hidden_dim=16)
    obs = _make_obs(batch_size=2)
    action_mask = torch.zeros(2, N_ACTIONS, dtype=torch.int8)
    action_mask[:, encode_action(0, 3, 8)] = 1

    _, log_prob, entropy, value = network.get_action_and_value(obs, action_mask=action_mask)

    assert torch.isfinite(log_prob).all()
    assert torch.isfinite(entropy).all()
    assert torch.isfinite(value).all()
