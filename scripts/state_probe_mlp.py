#!/usr/bin/env python3
"""Frozen-latent MLP probe: a stop-gradient state head.

The encoder is frozen; the MLP is trained only on the latents. This is the
cleanest test of whether the state is present but the linear readout is the
limit. No gradient from this probe reaches the JEPA.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from leparallax.probes import _ridge_eval
from leparallax.train import load_jepa
from leparallax.world import generate_episodes, history_indices


class Probe(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 4)

    def __call__(self, z: mx.array) -> mx.array:
        z = nn.relu(self.fc1(z))
        z = nn.relu(self.fc2(z))
        return self.fc3(z)


def collect(episodes):
    obs, states = [], []
    for ep in episodes:
        for t in range(len(ep.states)):
            obs.append(ep.frames[history_indices(t)])
            states.append(ep.states[t])
    return np.stack(obs).astype(np.float32) / 255.0, np.stack(states).astype(np.float32)


def encode(model, obs, batch=64):
    out = []
    for start in range(0, len(obs), batch):
        idx = np.arange(start, min(start + batch, len(obs)))
        z = model.encode(mx.array(obs[idx]))
        mx.eval(z)
        out.append(np.array(z))
    return np.concatenate(out)


def r2(pred, target):
    ss_res = ((target - pred) ** 2).sum(axis=0)
    ss_tot = ((target - target.mean(axis=0)) ** 2).sum(axis=0)
    return 1.0 - ss_res / (ss_tot + 1e-8)


def train_probe(Ztr, Ytr, Zte, Yte, seed: int, epochs: int = 300,
                batch: int = 64, lr: float = 1e-3):
    rng = np.random.default_rng(seed)
    mx.random.seed(seed)
    # Standardize on train.
    mu = Ztr.mean(axis=0, keepdims=True)
    sd = Ztr.std(axis=0, keepdims=True) + 1e-6
    Ztr = (Ztr - mu) / sd
    Zte = (Zte - mu) / sd
    # Validation split.
    perm = rng.permutation(len(Ztr))
    n_val = max(64, int(0.1 * len(Ztr)))
    val_idx = perm[:n_val]
    tr_idx = perm[n_val:]
    Zv, Yv = Ztr[val_idx], Ytr[val_idx]
    Zt, Yt = Ztr[tr_idx], Ytr[tr_idx]

    probe = Probe(Ztr.shape[1])
    opt = optim.AdamW(learning_rate=lr, weight_decay=1e-3)

    def loss_fn(probe, z, y):
        return mx.mean((probe(z) - y) ** 2)

    loss_and_grad = nn.value_and_grad(probe, loss_fn)
    best_val = float("inf")
    best_test = None
    for epoch in range(1, epochs + 1):
        lr_t = lr * 0.5 * (
            1.0 + math.cos(math.pi * (epoch - 1) / max(epochs - 1, 1))
        )
        opt.learning_rate = lr_t
        p = rng.permutation(len(Zt))
        for start in range(0, len(Zt), batch):
            idx = p[start:start + batch]
            loss, grads = loss_and_grad(
                probe, mx.array(Zt[idx]), mx.array(Yt[idx])
            )
            opt.update(probe, grads)
            mx.eval(probe.parameters(), opt.state)
        val_pred = np.array(probe(mx.array(Zv)))
        val_loss = float(np.mean((val_pred - Yv) ** 2))
        if val_loss < best_val:
            best_val = val_loss
            test_pred = np.array(probe(mx.array(Zte)))
            best_test = test_pred
    r2_4 = r2(best_test, Yte)
    return {
        "r2_state": float(np.mean(r2_4)),
        "r2_position": float(np.mean(r2_4[:2])),
        "r2_velocity": float(np.mean(r2_4[2:])),
        "best_val": best_val,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/phase2-framestate")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arm", default="asym_clean")
    ap.add_argument("--latent-dim", type=int, default=64)
    ap.add_argument("--width", type=float, default=1.0)
    ap.add_argument("--frame-state-head", action="store_true")
    ap.add_argument("--probe-seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--train-episodes", type=int, default=40)
    ap.add_argument("--test-episodes", type=int, default=20)
    args = ap.parse_args()

    path = pathlib.Path(args.out) / "models" / f"s{args.seed}-{args.arm}.safetensors"
    model = load_jepa(
        str(path), image_size=96,
        frame_state_head=args.frame_state_head,
        latent_dim=args.latent_dim, width=args.width,
    )
    train_eps = generate_episodes(
        n=args.train_episodes, steps=32, size=96, seed=0,
        goal_directed=0.5, nuisance_mode="normal", one_way_door=True,
        occlude_door_right=True,
    )
    test_eps = generate_episodes(
        n=args.test_episodes, steps=32, size=96, seed=9000,
        goal_directed=0.5, nuisance_mode="normal", one_way_door=True,
        occlude_door_right=True,
    )
    tr_obs, tr_state = collect(train_eps)
    te_obs, te_state = collect(test_eps)
    Ztr = encode(model, tr_obs)
    Zte = encode(model, te_obs)
    lin = _ridge_eval(Ztr, tr_state, Zte, te_state)
    print(
        f"{args.arm}: linear R2 {np.mean(lin):.3f} "
        f"(pos {np.mean(lin[:2]):.3f} vel {np.mean(lin[2:]):.3f})",
        flush=True,
    )
    rows = []
    for ps in args.probe_seeds:
        r = train_probe(Ztr, tr_state, Zte, te_state, ps, epochs=args.epochs)
        rows.append(r)
        print(
            f"  mlp seed {ps}: state {r['r2_state']:.3f} "
            f"(pos {r['r2_position']:.3f} vel {r['r2_velocity']:.3f})",
            flush=True,
        )
    result = {
        "arm": args.arm,
        "linear_state": float(np.mean(lin)),
        "linear_position": float(np.mean(lin[:2])),
        "linear_velocity": float(np.mean(lin[2:])),
        "mlp_state": float(np.mean([r["r2_state"] for r in rows])),
        "mlp_position": float(np.mean([r["r2_position"] for r in rows])),
        "mlp_velocity": float(np.mean([r["r2_velocity"] for r in rows])),
        "per_probe_seed": rows,
    }
    out = pathlib.Path(args.out) / f"mlp_probe_{args.arm}.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"mean MLP state {result['mlp_state']:.3f} "
        f"(pos {result['mlp_position']:.3f} vel {result['mlp_velocity']:.3f}); "
        f"wrote {out}"
    )


if __name__ == "__main__":
    main()
