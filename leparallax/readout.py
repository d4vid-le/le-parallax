"""Shared readout and state-feedback utilities."""

from __future__ import annotations

import numpy as np
import mlx.core as mx

from .data import build_probe_samples, collate_probe, history_obs
from .world import (
    render_state,
    sample_nuisance,
    sample_nuisance_shift,
    step_state,
)


def encode_batches(model, obs: np.ndarray, batch: int = 64) -> np.ndarray:
    out = []
    for start in range(0, len(obs), batch):
        x = obs[start : start + batch].astype(np.float32) / 255.0
        z = model.encode(mx.array(x))
        mx.eval(z)
        out.append(np.array(z))
    return np.concatenate(out)


def build_readout_data(model, episodes, n_states: int = 16, seed: int = 0):
    rng = np.random.default_rng(seed)
    obs, states = [], []
    for ep in episodes:
        n = min(n_states, len(ep.states))
        for t in rng.choice(len(ep.states), size=n, replace=False):
            t = int(t)
            for nu in (
                ep.nuisance,
                sample_nuisance(rng),
                sample_nuisance_shift(rng),
                sample_nuisance_shift(rng),
            ):
                obs.append(history_obs(ep, t, nu))
                states.append(ep.states[t])
    X = encode_batches(model, np.stack(obs))
    return X, np.stack(states)


def fit_ridge(X: np.ndarray, Y: np.ndarray, alpha: float = 1.0):
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True) + 1e-6
    Xs = (X - mu) / sd
    Xa = np.concatenate([Xs, np.ones((len(Xs), 1), dtype=Xs.dtype)], axis=1)
    d = Xa.shape[1]
    W = np.linalg.solve(Xa.T @ Xa + alpha * np.eye(d), Xa.T @ Y)
    return mu, sd, W


def predict_ridge(X, mu, sd, W):
    Xs = (X - mu) / sd
    Xa = np.concatenate([Xs, np.ones((len(Xs), 1), dtype=Xs.dtype)], axis=1)
    return Xa @ W


def readout_error(model, mu, sd, W, episodes, per_episode: int = 16, seed: int = 0):
    samples = build_probe_samples(episodes, per_episode=per_episode, seed=seed)
    obs, state, _ = collate_probe(samples, np.arange(len(samples)))
    X = encode_batches(model, obs)
    pred = predict_ridge(X, mu, sd, W)
    pos_mse = float(np.mean((pred[:, :2] - state[:, :2]) ** 2))
    vel_mse = float(np.mean((pred[:, 2:] - state[:, 2:]) ** 2))
    ss_res = ((state[:, :2] - pred[:, :2]) ** 2).sum(axis=0)
    ss_tot = ((state[:, :2] - state[:, :2].mean(axis=0)) ** 2).sum(axis=0)
    pos_r2 = float(np.mean(1.0 - ss_res / (ss_tot + 1e-8)))
    return {"pos_mse": pos_mse, "vel_mse": vel_mse, "pos_r2": pos_r2}


def state_feedback(
    model, mu, sd, W, episodes, k: float = 2.0, damp: float = 1.0,
    steps: int = 32,
):
    successes, distances = 0, []
    for ep in episodes:
        s = ep.states[0].copy()
        frame = render_state(s, ep.goal, ep.nuisance)
        hist = [frame, frame, frame]
        for _ in range(steps):
            obs = np.stack(hist[-3:])
            X = encode_batches(model, obs[None])
            pred = predict_ridge(X, mu, sd, W)[0]
            a = np.clip(
                k * (ep.goal - pred[:2]) - damp * pred[2:], -1.0, 1.0
            )
            s = step_state(s, a)
            hist.append(render_state(s, ep.goal, ep.nuisance))
            if np.linalg.norm(s[:2] - ep.goal) < 0.08:
                break
        d = float(np.linalg.norm(s[:2] - ep.goal))
        distances.append(d)
        successes += d < 0.08
    return {
        "success": successes / max(len(episodes), 1),
        "mean_dist": float(np.mean(distances)),
    }
