#!/usr/bin/env python3
"""Readout diagnostic: linear vs MLP readout on frozen H1 encoders.

Fresh OOD episodes (seed 5000 + seed), not the H2 or H2-v2 test episodes.
Exploratory; no confirmatory claim.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from leparallax.data import build_probe_samples, collate_probe, history_obs
from leparallax.plan import evaluate_random
from leparallax.train import load_jepa
from leparallax.world import (
    generate_episodes,
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


def fit_linear(X, Y, alpha=1.0):
    mu = X.mean(0, keepdims=True)
    sd = X.std(0, keepdims=True) + 1e-6
    Xs = (X - mu) / sd
    Xa = np.concatenate([Xs, np.ones((len(Xs), 1), dtype=Xs.dtype)], axis=1)
    d = Xa.shape[1]
    W = np.linalg.solve(Xa.T @ Xa + alpha * np.eye(d), Xa.T @ Y)
    return mu, sd, W


def predict_linear(X, mu, sd, W):
    Xs = (X - mu) / sd
    Xa = np.concatenate([Xs, np.ones((len(Xs), 1), dtype=Xs.dtype)], axis=1)
    return Xa @ W


class MLPReadout(nn.Module):
    def __init__(self, in_dim: int = 64, hidden: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, 4)

    def __call__(self, x):
        return self.fc2(nn.relu(self.fc1(x)))


def train_mlp(X, Y, seed: int = 0, epochs: int = 150, batch: int = 64):
    mu = X.mean(0, keepdims=True)
    sd = X.std(0, keepdims=True) + 1e-6
    Xs = ((X - mu) / sd).astype(np.float32)
    Y = Y.astype(np.float32)
    mx.random.seed(seed)
    model = MLPReadout()
    opt = optim.AdamW(learning_rate=1e-3, weight_decay=1e-4)
    rng = np.random.default_rng(seed)

    def loss_fn(model, x, y):
        return mx.mean((model(x) - y) ** 2)

    lg = nn.value_and_grad(model, loss_fn)
    n = len(Xs)
    for _ in range(epochs):
        perm = rng.permutation(n)
        for start in range(0, n, batch):
            idx = perm[start : start + batch]
            loss, grads = lg(model, mx.array(Xs[idx]), mx.array(Y[idx]))
            opt.update(model, grads)
            mx.eval(model.parameters(), opt.state, loss)
    return mu, sd, model


def predict_mlp(model, mu, sd, X):
    Xs = ((X - mu) / sd).astype(np.float32)
    pred = model(mx.array(Xs))
    mx.eval(pred)
    return np.array(pred)


def readout_r2(pred, state):
    ss_res = ((state[:, :2] - pred[:, :2]) ** 2).sum(0)
    ss_tot = ((state[:, :2] - state[:, :2].mean(0)) ** 2).sum(0)
    return float(np.mean(1.0 - ss_res / (ss_tot + 1e-8)))


def state_feedback(model, predict_fn, episodes, k=2.0, damp=1.0, steps=32):
    successes, distances = 0, []
    for ep in episodes:
        s = ep.states[0].copy()
        frame = render_state(s, ep.goal, ep.nuisance)
        hist = [frame, frame, frame]
        for _ in range(steps):
            obs = np.stack(hist[-3:])
            X = encode_batches(model, obs[None])
            pred = predict_fn(X)[0]
            a = np.clip(k * (ep.goal - pred[:2]) - damp * pred[2:], -1.0, 1.0)
            s = step_state(s, a)
            hist.append(render_state(s, ep.goal, ep.nuisance))
            if np.linalg.norm(s[:2] - ep.goal) < 0.08:
                break
        d = float(np.linalg.norm(s[:2] - ep.goal))
        distances.append(d)
        successes += d < 0.08
    return successes / max(len(episodes), 1), float(np.mean(distances))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--readout-episodes", type=int, default=80)
    ap.add_argument("--states-per-episode", type=int, default=16)
    ap.add_argument("--ood-episodes", type=int, default=10)
    ap.add_argument("--id-episodes", type=int, default=5)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--mlp-epochs", type=int, default=150)
    ap.add_argument("--out", default="runs/readout-diagnostic")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in args.seeds:
        print(f"\n=== seed {seed} ===", flush=True)
        pool = generate_episodes(
            n=100, steps=args.steps, size=96, seed=seed, goal_directed=0.5
        )
        readout_eps = pool[: args.readout_episodes]
        ood = generate_episodes(
            n=args.ood_episodes, steps=args.steps, size=96,
            seed=5000 + seed, goal_directed=0.5, nuisance_shift=True,
        )
        id_ = generate_episodes(
            n=args.id_episodes, steps=args.steps, size=96,
            seed=6000 + seed, goal_directed=0.5, nuisance_shift=False,
        )
        for arm in ("temporal", "good"):
            model = load_jepa(
                f"runs/h1-confirm/models/s{seed}-{arm}.safetensors",
                image_size=96,
            )
            X, Y = build_readout_data(
                model, readout_eps, n_states=args.states_per_episode, seed=seed
            )
            lmu, lsd, lW = fit_linear(X, Y)
            mmu, msd, m = train_mlp(X, Y, seed=seed, epochs=args.mlp_epochs)

            # readout error on fresh OOD
            samples = build_probe_samples(ood, per_episode=16, seed=seed)
            obs, state, _ = collate_probe(samples, np.arange(len(samples)))
            Xo = encode_batches(model, obs)
            lr2 = readout_r2(predict_linear(Xo, lmu, lsd, lW), state)
            mr2 = readout_r2(predict_mlp(m, mmu, msd, Xo), state)

            l_ood = state_feedback(
                model, lambda x: predict_linear(x, lmu, lsd, lW), ood,
                steps=args.steps,
            )
            m_ood = state_feedback(
                model, lambda x: predict_mlp(m, mmu, msd, x), ood,
                steps=args.steps,
            )
            l_id = state_feedback(
                model, lambda x: predict_linear(x, lmu, lsd, lW), id_,
                steps=args.steps,
            )
            m_id = state_feedback(
                model, lambda x: predict_mlp(m, mmu, msd, x), id_,
                steps=args.steps,
            )
            row = {
                "seed": seed, "arm": arm,
                "linear_ood_r2": lr2, "mlp_ood_r2": mr2,
                "linear_ood_success": l_ood[0], "linear_ood_dist": l_ood[1],
                "mlp_ood_success": m_ood[0], "mlp_ood_dist": m_ood[1],
                "linear_id_success": l_id[0], "linear_id_dist": l_id[1],
                "mlp_id_success": m_id[0], "mlp_id_dist": m_id[1],
            }
            rows.append(row)
            print(
                f"  {arm}: linear R2 {lr2:.3f} succ {l_ood[0]:.2f} "
                f"| MLP R2 {mr2:.3f} succ {m_ood[0]:.2f}", flush=True
            )

    (out / "per_seed.json").write_text(json.dumps(rows, indent=2) + "\n")
    by = {(r["seed"], r["arm"]): r for r in rows}
    lines = [
        "# Readout diagnostic — linear vs MLP",
        "",
        "Fresh OOD episodes (seeds 5000 + seed); not the H2 or H2-v2 test sets.",
        "",
        "| seed | arm | linear R² | MLP R² | linear succ | MLP succ | linear dist | MLP dist |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['seed']} | {r['arm']} | {r['linear_ood_r2']:.3f} | "
            f"{r['mlp_ood_r2']:.3f} | {r['linear_ood_success']:.2f} | "
            f"{r['mlp_ood_success']:.2f} | {r['linear_ood_dist']:.3f} | "
            f"{r['mlp_ood_dist']:.3f} |"
        )
    lines += ["", "## Means", ""]
    for arm in ("temporal", "good"):
        rs = [r for r in rows if r["arm"] == arm]
        lines.append(
            f"- **{arm}**: linear R² {np.mean([r['linear_ood_r2'] for r in rs]):.3f}, "
            f"MLP R² {np.mean([r['mlp_ood_r2'] for r in rs]):.3f}, "
            f"linear succ {np.mean([r['linear_ood_success'] for r in rs]):.2f}, "
            f"MLP succ {np.mean([r['mlp_ood_success'] for r in rs]):.2f}"
        )
    lines += [
        "",
        "If MLP R² turns positive and control improves, the linear readout was the",
        "limit. If both readouts stay negative, the encoder's OOD support is the",
        "limit; the next experiment is domain randomization during encoder training.",
        "",
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[-8:]))
    print(f"wrote {out / 'report.md'}")


if __name__ == "__main__":
    main()
