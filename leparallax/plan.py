"""CEM planning in the learned latent space."""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from .data import goal_obs
from .model import JEPA
from .world import render_state, step_state


def cem_plan(
    model: JEPA,
    obs: np.ndarray,
    goal: np.ndarray,
    horizon: int = 5,
    pop: int = 128,
    elite: int = 16,
    iters: int = 5,
    seed: int = 0,
    goal_state: np.ndarray | None = None,
) -> np.ndarray:
    obs = obs.astype(np.float32) / 255.0
    goal = goal.astype(np.float32) / 255.0
    z0 = model.encode(mx.array(obs[None]))
    zg = model.encode(mx.array(goal[None]))
    rng = np.random.default_rng(seed)

    mean = np.zeros((horizon, 2), dtype=np.float32)
    std = np.ones((horizon, 2), dtype=np.float32)

    for _ in range(iters):
        acts = mean + std * rng.normal(size=(pop, horizon, 2))
        acts = np.clip(acts, -1.0, 1.0).astype(np.float32)
        z = mx.repeat(z0, pop, axis=0)
        for h in range(horizon):
            z = model.predict(z, mx.array(acts[:, h]))
        if goal_state is not None:
            s = model.state_head(z)
            cost = mx.sum((s[:, :2] - mx.array(goal_state[:2])) ** 2, axis=-1)
        else:
            cost = mx.sum((z - zg) ** 2, axis=-1)
        mx.eval(cost)
        order = np.argsort(np.array(cost))[:elite]
        mean = acts[order].mean(axis=0)
        std = acts[order].std(axis=0) + 0.05

    return mean


def evaluate_planner(
    model: JEPA,
    episodes,
    n_episodes: int = 10,
    steps: int = 32,
    horizon: int = 5,
    seed: int = 0,
) -> dict:
    successes, distances = 0, []
    for i, ep in enumerate(episodes[:n_episodes]):
        s = ep.states[0].copy()
        frame = render_state(s, ep.goal, ep.nuisance)
        history = [frame, frame, frame]
        goal = goal_obs(ep)
        for t in range(steps):
            obs = np.stack(history[-3:])
            goal_state = np.array([ep.goal[0], ep.goal[1], 0.0, 0.0], dtype=np.float32)
            plan = cem_plan(
                model, obs, goal, horizon=horizon, seed=seed + t,
                goal_state=goal_state,
            )
            s = step_state(s, plan[0])
            history.append(render_state(s, ep.goal, ep.nuisance))
            if np.linalg.norm(s[:2] - ep.goal) < 0.08:
                break
        dist = float(np.linalg.norm(s[:2] - ep.goal))
        distances.append(dist)
        successes += dist < 0.08
    return {
        "success": successes / max(len(episodes[:n_episodes]), 1),
        "mean_dist": float(np.mean(distances)),
    }


def evaluate_random(
    episodes, n_episodes: int = 10, steps: int = 32, seed: int = 0
) -> dict:
    rng = np.random.default_rng(seed)
    successes, distances = 0, []
    for ep in episodes[:n_episodes]:
        s = ep.states[0].copy()
        for _ in range(steps):
            s = step_state(s, rng.uniform(-1.0, 1.0, size=2))
            if np.linalg.norm(s[:2] - ep.goal) < 0.08:
                break
        dist = float(np.linalg.norm(s[:2] - ep.goal))
        distances.append(dist)
        successes += dist < 0.08
    return {
        "success": successes / max(len(episodes[:n_episodes]), 1),
        "mean_dist": float(np.mean(distances)),
    }
