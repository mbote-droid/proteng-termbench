"""Regression: a trajectory that never visits one state must not crash the generator."""

from __future__ import annotations

import numpy as np

from protengbench.tasks import t03_switch


def test_rates_of_a_one_state_chain_are_zero_not_an_error() -> None:
    for fill in (0.0, 1.0):
        state = np.full(t03_switch.N_FRAMES, fill)
        k_up, k_down, pop = t03_switch._rates(state)
        assert (k_up, k_down) == (0.0, 0.0)
        assert pop == fill


def test_rates_of_a_visited_chain_are_positive() -> None:
    state = np.tile([0.0] * 30 + [1.0] * 10, t03_switch.N_FRAMES // 40 + 1)[: t03_switch.N_FRAMES]
    k_up, k_down, pop = t03_switch._rates(state)
    assert k_up > 0 and k_down > 0 and 0.2 < pop < 0.3
