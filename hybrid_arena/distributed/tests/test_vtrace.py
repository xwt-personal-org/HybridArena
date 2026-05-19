import numpy as np
import pytest

from hybrid_arena.distributed.vtrace import vtrace_returns


def test_vtrace_matches_one_step_td_when_on_policy():
    vs, pg_adv = vtrace_returns(
        log_rhos=np.zeros(2),
        discounts=np.array([0.9, 0.9]),
        rewards=np.array([1.0, 2.0]),
        values=np.array([0.5, 1.0]),
        bootstrap_value=np.array(0.0),
    )
    np.testing.assert_allclose(vs, np.array([2.8, 2.0], dtype=np.float32), rtol=1e-5)
    np.testing.assert_allclose(pg_adv, np.array([2.3, 1.0], dtype=np.float32), rtol=1e-5)


def test_vtrace_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="matching shape"):
        vtrace_returns([0.0], [0.9, 0.9], [1.0], [0.0], 0.0)
