"""Datasets and batching for the synthetic world."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from .world import Episode, HISTORY, history_indices, render_state, sample_nuisance


@dataclass
class TemporalSample:
    obs: np.ndarray        # (H, h, w, 3) uint8
    action: np.ndarray     # (2,)
    next_obs: np.ndarray   # (H, h, w, 3) uint8
    state: np.ndarray      # (4,)
    goal: np.ndarray       # (2,)
    nuisance: np.ndarray   # (11,)
    action2: np.ndarray    # (2,)  next action
    next_next_obs: np.ndarray  # (H, h, w, 3)
    state2: np.ndarray     # (4,)  state after two steps


@dataclass
class ViewPair:
    obs1: np.ndarray      # (H, h, w, 3) uint8
    obs2: np.ndarray      # (H, h, w, 3) uint8
    state: np.ndarray     # (4,)


@dataclass
class ProbeSample:
    obs: np.ndarray        # (H, h, w, 3) uint8
    state: np.ndarray      # (4,)
    nuisance: np.ndarray   # (11,)


def build_temporal_samples(episodes: list[Episode]) -> list[TemporalSample]:
    samples = []
    for ep in episodes:
        for t in range(len(ep.actions) - 1):
            idx = history_indices(t)
            nxt = history_indices(t + 1)
            nxt2 = history_indices(t + 2)
            samples.append(
                TemporalSample(
                    obs=ep.frames[idx],
                    action=ep.actions[t],
                    next_obs=ep.frames[nxt],
                    state=ep.states[t],
                    goal=ep.goal,
                    nuisance=ep.nuisance,
                    action2=ep.actions[t + 1],
                    next_next_obs=ep.frames[nxt2],
                    state2=ep.states[t + 2],
                )
            )
    return samples


def build_probe_samples(
    episodes: list[Episode], per_episode: int = 16, seed: int = 0
) -> list[ProbeSample]:
    rng = np.random.default_rng(seed)
    samples = []
    for ep in episodes:
        n = min(per_episode, len(ep.states))
        for t in rng.choice(len(ep.states), size=n, replace=False):
            samples.append(
                ProbeSample(
                    obs=ep.frames[history_indices(int(t))],
                    state=ep.states[int(t)],
                    nuisance=ep.nuisance,
                )
            )
    return samples


def collate_temporal(samples: list[TemporalSample], indices: np.ndarray):
    obs = np.stack([samples[i].obs for i in indices]).astype(np.float32) / 255.0
    nxt = np.stack([samples[i].next_obs for i in indices]).astype(np.float32) / 255.0
    act = np.stack([samples[i].action for i in indices]).astype(np.float32)
    state = np.stack([samples[i].state for i in indices]).astype(np.float32)
    act2 = np.stack([samples[i].action2 for i in indices]).astype(np.float32)
    nxt2 = np.stack([samples[i].next_next_obs for i in indices]).astype(np.float32) / 255.0
    state2 = np.stack([samples[i].state2 for i in indices]).astype(np.float32)
    return obs, act, nxt, state, act2, nxt2, state2


def collate_probe(samples: list[ProbeSample], indices: np.ndarray):
    obs = np.stack([samples[i].obs for i in indices]).astype(np.float32) / 255.0
    state = np.stack([samples[i].state for i in indices]).astype(np.float32)
    nuisance = np.stack([samples[i].nuisance for i in indices]).astype(np.float32)
    return obs, state, nuisance


def history_obs(
    episode: Episode, t: int, nuisance: np.ndarray | None = None
) -> np.ndarray:
    """Observation stack at time t, optionally re-rendered with another nuisance."""
    if nuisance is None:
        return episode.frames[history_indices(t)]
    idx = history_indices(t)
    size = episode.frames.shape[1]
    return np.stack(
        [
            render_state(episode.states[i], episode.goal, nuisance, size=size)
            for i in idx
        ]
    )


def goal_obs(
    episode: Episode, nuisance: np.ndarray | None = None
) -> np.ndarray:
    """A 3-frame stack of the goal state, used as the planning target."""
    n = episode.nuisance if nuisance is None else nuisance
    size = episode.frames.shape[1]
    frame = render_state(
        np.array(
            [episode.goal[0], episode.goal[1], 0.0, 0.0], dtype=np.float32
        ),
        episode.goal,
        n,
        size=size,
    )
    return np.stack([frame] * HISTORY)


def init_planning_history(episode: Episode) -> np.ndarray:
    return episode.frames[history_indices(0)]


def build_view_pairs(
    episodes: list[Episode], per_episode: int = 4, seed: int = 0
) -> list[ViewPair]:
    """Good views: same state, independent nuisance."""
    rng = np.random.default_rng(seed)
    pairs = []
    for ep in episodes:
        n = min(per_episode, len(ep.states))
        for t in rng.choice(len(ep.states), size=n, replace=False):
            t = int(t)
            other = sample_nuisance(rng)
            pairs.append(
                ViewPair(
                    obs1=ep.frames[history_indices(t)],
                    obs2=history_obs(ep, t, other),
                    state=ep.states[t],
                )
            )
    return pairs


def _crop_stack(obs: np.ndarray, rng: np.random.Generator, scale: float = 0.72):
    h, w = obs.shape[1:3]
    side = int(min(h, w) * scale)
    x = int(rng.integers(0, w - side + 1))
    y = int(rng.integers(0, h - side + 1))
    out = []
    for frame in obs:
        crop = Image.fromarray(frame[y : y + side, x : x + side])
        out.append(np.asarray(crop.resize((w, h), Image.BILINEAR)))
    return np.stack(out)


def build_bad_view_pairs(
    episodes: list[Episode], per_episode: int = 4, seed: int = 0
) -> list[ViewPair]:
    """Bad views: two crops of one observation (shared nuisance)."""
    rng = np.random.default_rng(seed)
    pairs = []
    for ep in episodes:
        n = min(per_episode, len(ep.states))
        for t in rng.choice(len(ep.states), size=n, replace=False):
            t = int(t)
            obs = ep.frames[history_indices(t)]
            pairs.append(
                ViewPair(
                    obs1=_crop_stack(obs, rng),
                    obs2=_crop_stack(obs, rng),
                    state=ep.states[t],
                )
            )
    return pairs


def collate_views(pairs: list[ViewPair], indices: np.ndarray):
    obs1 = np.stack([pairs[i].obs1 for i in indices]).astype(np.float32) / 255.0
    obs2 = np.stack([pairs[i].obs2 for i in indices]).astype(np.float32) / 255.0
    state = np.stack([pairs[i].state for i in indices]).astype(np.float32)
    return obs1, obs2, state
