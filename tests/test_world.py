"""World and data sanity tests."""

import numpy as np

from leparallax.data import (
    build_temporal_samples,
    collate_temporal,
    history_obs,
)
from leparallax.world import (
    generate_episodes,
    history_indices,
    step_state,
)


def test_episode_shapes():
    eps = generate_episodes(1, steps=4, size=32, seed=0)
    ep = eps[0]
    assert ep.frames.shape == (5, 32, 32, 3)
    assert ep.states.shape == (5, 4)
    assert ep.actions.shape == (4, 2)
    assert ep.goal.shape == (2,)
    assert ep.nuisance.shape == (11,)


def test_generation_is_deterministic():
    a = generate_episodes(1, steps=4, size=32, seed=7)
    b = generate_episodes(1, steps=4, size=32, seed=7)
    assert np.allclose(a[0].states, b[0].states)
    assert np.array_equal(a[0].frames, b[0].frames)


def test_step_stays_in_bounds():
    s = np.array([0.5, 0.5, 0.0, 0.0], dtype=np.float32)
    for _ in range(50):
        s = step_state(s, np.array([1.0, 1.0], dtype=np.float32))
        assert 0.05 <= s[0] <= 0.95
        assert 0.05 <= s[1] <= 0.95


def test_history_indices():
    assert history_indices(0) == [0, 0, 0]
    assert history_indices(1) == [0, 0, 1]
    assert history_indices(2) == [0, 1, 2]
    assert history_indices(5) == [3, 4, 5]


def test_temporal_sample_fields():
    eps = generate_episodes(1, steps=4, size=32, seed=0)
    samples = build_temporal_samples(eps)
    assert len(samples) == 3  # t = 0, 1, 2; t + 2 must exist
    s = samples[0]
    assert s.obs.shape == (3, 32, 32, 3)
    assert s.next_obs.shape == s.obs.shape
    assert s.next_next_obs.shape == s.obs.shape

    obs, act, nxt, state, act2, nxt2, state2 = collate_temporal(
        samples, np.arange(len(samples))
    )
    assert obs.shape == (3, 3, 32, 32, 3)
    assert act.shape == (3, 2)
    assert nxt.shape == obs.shape
    assert state.shape == (3, 4)
    assert act2.shape == (3, 2)
    assert nxt2.shape == obs.shape
    assert state2.shape == (3, 4)


def test_history_obs_can_rerender_nuisance():
    eps = generate_episodes(1, steps=4, size=32, seed=0)
    ep = eps[0]
    other = ep.nuisance.copy()
    other[3:6] = 0.5  # change background color
    a = history_obs(ep, 2, ep.nuisance)
    b = history_obs(ep, 2, other)
    assert a.shape == (3, 32, 32, 3)
    assert b.shape == a.shape
    assert not np.array_equal(a, b)
