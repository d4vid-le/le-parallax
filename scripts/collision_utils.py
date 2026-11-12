#!/usr/bin/env python3
"""Shared collision utilities: OOD nuisance conditions and CSA training."""

from __future__ import annotations

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from leparallax.data import (
    ProbeSample,
    collate_temporal,
    history_obs,
)
from leparallax.model import SIGReg, covariance_term, make_loss, variance_term
from leparallax.probes import _ridge_eval, collect_latents
from leparallax.world import history_indices, sample_nuisance, sample_nuisance_shift

CORNERS = [
    (1.10, 0.35, 0.20, +0.08),
    (0.80, 0.20, 0.80, -0.08),
    (1.10, 0.20, 0.80, +0.08),
    (0.80, 0.35, 0.20, -0.08),
]


def build_conditions(test_eps, per_episode: int, seed: int):
    rng = np.random.default_rng(seed + 700)
    conditions = {"normal": [], "heldout_swap": [], "shifted": [], "extreme": []}
    counter = 0
    for i, ep in enumerate(test_eps):
        n = min(per_episode, len(ep.states))
        for t in rng.choice(len(ep.states), size=n, replace=False):
            t = int(t)
            conditions["normal"].append(ProbeSample(
                obs=ep.frames[history_indices(t)],
                state=ep.states[t],
                nuisance=ep.nuisance,
            ))
            other = test_eps[(i + 1 + int(rng.integers(0, len(test_eps) - 1))) % len(test_eps)]
            if other is ep:
                other = test_eps[(i + 1) % len(test_eps)]
            conditions["heldout_swap"].append(ProbeSample(
                obs=history_obs(ep, t, other.nuisance),
                state=ep.states[t],
                nuisance=other.nuisance,
            ))
            shift = sample_nuisance_shift(rng)
            conditions["shifted"].append(ProbeSample(
                obs=history_obs(ep, t, shift),
                state=ep.states[t],
                nuisance=shift,
            ))
            strength, bg, puck, cam = CORNERS[counter % len(CORNERS)]
            counter += 1
            angle = rng.uniform(0.0, 2.0 * np.pi)
            extreme = np.concatenate([
                [np.sin(angle), np.cos(angle), strength],
                np.full(3, bg, dtype=np.float32),
                np.full(3, puck, dtype=np.float32),
                np.full(2, cam, dtype=np.float32),
            ]).astype(np.float32)
            conditions["extreme"].append(ProbeSample(
                obs=history_obs(ep, t, extreme),
                state=ep.states[t],
                nuisance=extreme,
            ))
    return conditions


def evaluate(model, train_samples, conditions):
    Xtr, Ytr, _ = collect_latents(model, train_samples)
    out = {}
    for name, samples in conditions.items():
        X, Y, _ = collect_latents(model, samples)
        r2 = _ridge_eval(Xtr, Ytr, X, Y)
        out[name] = {
            "state": float(np.mean(r2)),
            "position": float(np.mean(r2[:2])),
            "velocity": float(np.mean(r2[2:])),
        }
    return out


def build_cross_domain_pairs(normal_eps, shift_eps, per_episode: int, seed: int):
    """Same state, normal vs shifted nuisance, in both directions."""
    rng = np.random.default_rng(seed + 1100)
    pairs = []
    for ep in normal_eps:
        n = min(per_episode, len(ep.states))
        for t in rng.choice(len(ep.states), size=n, replace=False):
            t = int(t)
            pairs.append((
                ep.frames[history_indices(t)],
                history_obs(ep, t, sample_nuisance_shift(rng)),
                ep.states[t],
            ))
    normal_rng = np.random.default_rng(seed + 1200)
    for ep in shift_eps:
        n = min(per_episode, len(ep.states))
        for t in rng.choice(len(ep.states), size=n, replace=False):
            t = int(t)
            pairs.append((
                ep.frames[history_indices(t)],
                history_obs(ep, t, sample_nuisance(normal_rng)),
                ep.states[t],
            ))
    return pairs


def collate_pairs(pairs, indices):
    o1 = np.stack([pairs[i][0] for i in indices]).astype(np.float32) / 255.0
    o2 = np.stack([pairs[i][1] for i in indices]).astype(np.float32) / 255.0
    state = np.stack([pairs[i][2] for i in indices]).astype(np.float32)
    return o1, o2, state


def train_csa(
    model,
    temporal_samples,
    pairs,
    epochs: int = 5,
    batch_size: int = 32,
    lr: float = 1e-4,
    seed: int = 0,
    out_path=None,
    lambda_latent: float = 0.1,
    lambda_state: float = 0.5,
):
    sigreg = SIGReg()
    loss_fn = make_loss(sigreg, 0.09)

    def full_loss(model, obs, act, nxt, state, act2, nxt2, state2,
                  o1, o2, cf_state):
        base = loss_fn(model, obs, act, nxt, state, act2, nxt2, state2)
        z1 = model.encode(o1)
        z2 = model.encode(o2)
        align = mx.mean((z1 - z2) ** 2)
        state_cf = 0.5 * (
            mx.mean((model.state_head(z1) - cf_state) ** 2)
            + mx.mean((model.state_head(z2) - cf_state) ** 2)
        )
        reg = 0.5 * (sigreg(z1) + sigreg(z2))
        var = 0.5 * (variance_term(z1) + variance_term(z2))
        cov = 0.5 * (covariance_term(z1) + covariance_term(z2))
        return (
            base
            + lambda_latent * align
            + lambda_state * state_cf
            + 0.09 * reg
            + 0.10 * var
            + 0.01 * cov
        )

    loss_and_grad = nn.value_and_grad(model, full_loss)
    optimizer = optim.AdamW(learning_rate=lr, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    n = len(temporal_samples)
    for epoch in range(1, epochs + 1):
        perm = rng.permutation(n)
        total, seen = 0.0, 0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            obs, act, nxt, state, act2, nxt2, state2 = collate_temporal(
                temporal_samples, idx
            )
            pi = rng.integers(0, len(pairs), size=len(idx))
            o1, o2, cf_state = collate_pairs(pairs, pi)
            loss, grads = loss_and_grad(
                model, mx.array(obs), mx.array(act), mx.array(nxt),
                mx.array(state), mx.array(act2), mx.array(nxt2), mx.array(state2),
                mx.array(o1), mx.array(o2), mx.array(cf_state),
            )
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
            total += float(loss.item()) * len(idx)
            seen += len(idx)
        print(
            f"    csa epoch {epoch}/{epochs}  loss {total / max(seen, 1):.5f}",
            flush=True,
        )
    if out_path is not None:
        import pathlib
        pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        model.save_weights(str(out_path))
    return model


def bootstrap_ci(values, n_boot: int = 5000, seed: int = 0):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.array([
        rng.choice(values, size=len(values), replace=True).mean()
        for _ in range(n_boot)
    ])
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
