import numpy as np

from hybrid_arena.inference.adapter import MacroActionAdapter
from hybrid_arena.minimoba.action_encoding import encode_action


def test_macro_action_bias_respects_action_mask():
    mask = np.zeros(324, dtype=np.int8)
    legal = encode_action(3, 0, 8)
    mask[legal] = 1
    bias = MacroActionAdapter("PUSH_LANE").to_action_mask_bias(mask)
    assert bias[legal] > 0
    assert np.count_nonzero(bias) == 1


def test_reward_bias_is_explicit_metadata():
    bias = MacroActionAdapter("DEFEND_OBJECTIVE").to_reward_bias()
    assert "tower_lost" in bias
