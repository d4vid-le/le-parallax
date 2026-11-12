"""Linear probes for state and nuisance."""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from .data import ProbeSample, build_probe_samples, collate_probe, history_obs
from .model import JEPA
from .world import sample_nuisance


def collect_latents(
    model: JEPA, samples: list[ProbeSample], batch_size: int = 64
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    latents, states, nuisances = [], [], []
    for start in range(0, len(samples), batch_size):
        idx = np.arange(start, min(start + batch_size, len(samples)))
        obs, state, nuisance = collate_probe(samples, idx)
        z = model.encode(mx.array(obs))
        mx.eval(z)
        latents.append(np.array(z))
        states.append(state)
        nuisances.append(nuisance)
    return (
        np.concatenate(latents),
        np.concatenate(states),
        np.concatenate(nuisances),
    )


def _ridge_eval(
    Xtr: np.ndarray, Ytr: np.ndarray, Xte: np.ndarray, Yte: np.ndarray,
    alpha: float = 1.0,
) -> np.ndarray:
    mu = Xtr.mean(axis=0, keepdims=True)
    sd = Xtr.std(axis=0, keepdims=True) + 1e-6
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd
    Xtr = np.concatenate([Xtr, np.ones((len(Xtr), 1), dtype=Xtr.dtype)], axis=1)
    Xte = np.concatenate([Xte, np.ones((len(Xte), 1), dtype=Xte.dtype)], axis=1)

    d = Xtr.shape[1]
    W = np.linalg.solve(Xtr.T @ Xtr + alpha * np.eye(d), Xtr.T @ Ytr)
    pred = Xte @ W
    ss_res = ((Yte - pred) ** 2).sum(axis=0)
    ss_tot = ((Yte - Yte.mean(axis=0)) ** 2).sum(axis=0)
    return 1.0 - ss_res / (ss_tot + 1e-8)


def probe_report(
    model: JEPA,
    train_episodes,
    test_episodes,
    per_episode: int = 16,
    seed: int = 0,
) -> dict:
    train = build_probe_samples(train_episodes, per_episode=per_episode, seed=seed)
    test = build_probe_samples(test_episodes, per_episode=per_episode, seed=seed + 1)
    Xtr, state_tr, nuis_tr = collect_latents(model, train)
    Xte, state_te, nuis_te = collect_latents(model, test)

    state_r2 = _ridge_eval(Xtr, state_tr, Xte, state_te)
    nuis_r2 = _ridge_eval(Xtr, nuis_tr, Xte, nuis_te)
    return {
        "state": float(np.mean(state_r2)),
        "position": float(np.mean(state_r2[:2])),
        "velocity": float(np.mean(state_r2[2:])),
        "nuisance": float(np.mean(nuis_r2)),
        "state_per_dim": state_r2.tolist(),
        "nuisance_per_dim": nuis_r2.tolist(),
    }


def nuisance_sensitivity(
    model: JEPA, episodes, per_episode: int = 8, seed: int = 0
) -> dict:
    """Same-state distance under nuisance vs distance between states.

    Lower is better: a state-invariant latent should be closer to another view
    of the same state than to a different state.
    """
    rng = np.random.default_rng(seed)
    same, diff = [], []
    for ep in episodes:
        n = min(per_episode, len(ep.states))
        for t in rng.choice(len(ep.states), size=n, replace=False):
            t = int(t)
            t2 = (t + max(1, len(ep.states) // 3)) % len(ep.states)
            other = sample_nuisance(rng)
            o_same = history_obs(ep, t, other).astype(np.float32) / 255.0
            o_base = history_obs(ep, t, ep.nuisance).astype(np.float32) / 255.0
            o_diff = history_obs(ep, t2, ep.nuisance).astype(np.float32) / 255.0
            z_same = model.encode(mx.array(o_same[None]))
            z_base = model.encode(mx.array(o_base[None]))
            z_diff = model.encode(mx.array(o_diff[None]))
            mx.eval(z_same, z_base, z_diff)
            same.append(float(mx.sqrt(mx.sum((z_same - z_base) ** 2))))
            diff.append(float(mx.sqrt(mx.sum((z_base - z_diff) ** 2))))
    same_m = float(np.mean(same))
    diff_m = float(np.mean(diff))
    ratios = [s / (d + 1e-8) for s, d in zip(same, diff)]
    return {
        "same_state": same_m,
        "different_state": diff_m,
        "ratio": same_m / (diff_m + 1e-8),
        "mean_ratio": float(np.mean(ratios)),
    }
