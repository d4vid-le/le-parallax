#!/usr/bin/env python3
"""Hypothesis A: discriminative latent reachability probe.

Train a classifier on frozen dual-stream latents:
    C(z_current, z_goal) -> P(reachable)

Ablations: state-only, JEPA-only, combined. The abort threshold is calibrated on
validation reachable pairs to a false-abort budget, then evaluated on the planner
starts (reachable left / unreachable right after the one-way door).
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
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from oracle_state_cnn import OracleCNN
from leparallax.data import goal_obs
from leparallax.train import load_jepa
from leparallax.world import generate_episodes, history_indices


class Classifier(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden // 2)
        self.fc3 = nn.Linear(hidden // 2, 1)

    def __call__(self, x: mx.array) -> mx.array:
        x = nn.relu(self.fc1(x))
        x = nn.relu(self.fc2(x))
        return self.fc3(x)


def auc(pos, neg):
    x = np.concatenate([pos, neg])
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(x) + 1)
    n_pos, n_neg = len(pos), len(neg)
    return float((ranks[:n_pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def features(state_model, jepa, obs, goal):
    sf = np.array(state_model.features(mx.array(obs[None].astype(np.float32) / 255.0)))
    sg = np.array(state_model.features(mx.array(goal[None].astype(np.float32) / 255.0)))
    zf = np.array(jepa.encode(mx.array(obs[None].astype(np.float32) / 255.0)))
    zg = np.array(jepa.encode(mx.array(goal[None].astype(np.float32) / 255.0)))
    return sf[0], sg[0], zf[0], zg[0]


def build_pairs(episodes, state_model, jepa, starts_only=False):
    rows = []
    for ep in episodes:
        goal = goal_obs(ep)
        sides = ep.states[:, 0] >= 0.5
        cr = np.where((~sides[:-1]) & sides[1:])[0]
        tc = int(cr[0]) if len(cr) else None
        times = []
        if starts_only:
            if tc is None:
                continue
            for pre in (2, 4):
                ta = max(0, tc - pre)
                if ep.door_open[ta] and ep.states[ta, 0] < 0.5:
                    times.append(ta)
            for post in (1, 2):
                tb = min(len(ep.states) - 1, tc + post)
                if not ep.door_open[tb] and ep.states[tb, 0] >= 0.5:
                    times.append(tb)
        else:
            times = list(range(len(ep.states)))
        for t in times:
            obs = ep.frames[history_indices(t)]
            reach = bool(ep.door_open[t]) or (
                bool(ep.states[t, 0] >= 0.5) == bool(ep.goal[0] >= 0.5)
            )
            sf, sg, zf, zg = features(state_model, jepa, obs, goal)
            rows.append({
                "obs": obs, "sf": sf, "sg": sg, "zf": zf, "zg": zg,
                "reach": float(reach),
            })
    return rows


def split_xy(rows, kind):
    def row_features(r):
        if kind == "state":
            return np.concatenate([r["sf"], r["sg"]])
        if kind == "jepa":
            return np.concatenate([r["zf"], r["zg"]])
        return np.concatenate([r["sf"], r["sg"], r["zf"], r["zg"]])

    X = np.stack([row_features(r) for r in rows]).astype(np.float32)
    y = np.array([r["reach"] for r in rows], dtype=np.float32)
    return X, y


def train_clf(Xtr, ytr, seed: int, epochs: int = 50, batch: int = 64,
              lr: float = 1e-3):
    rng = np.random.default_rng(seed)
    mx.random.seed(seed)
    mu = Xtr.mean(axis=0, keepdims=True)
    sd = Xtr.std(axis=0, keepdims=True) + 1e-6
    Xtr = (Xtr - mu) / sd
    model = Classifier(Xtr.shape[1])
    opt = optim.AdamW(learning_rate=lr, weight_decay=1e-3)

    def loss_fn(model, x, y):
        p = mx.sigmoid(model(x))[:, 0]
        return -mx.mean(
            y * mx.log(p + 1e-6) + (1.0 - y) * mx.log(1.0 - p + 1e-6)
        )

    lg = nn.value_and_grad(model, loss_fn)
    for epoch in range(1, epochs + 1):
        lr_t = lr * 0.5 * (
            1.0 + math.cos(math.pi * (epoch - 1) / max(epochs - 1, 1))
        )
        opt.learning_rate = lr_t
        p = rng.permutation(len(Xtr))
        for start in range(0, len(Xtr), batch):
            idx = p[start:start + batch]
            loss, grads = lg(
                model, mx.array(Xtr[idx]), mx.array(ytr[idx])
            )
            opt.update(model, grads)
            mx.eval(model.parameters(), opt.state)
    return model, mu, sd


def predict(model, mu, sd, X):
    X = (X - mu) / sd
    p = mx.sigmoid(model(mx.array(X)))[:, 0]
    mx.eval(p)
    return np.array(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default="runs/dual-stream-state/state_stream_s0.safetensors")
    ap.add_argument("--jepa-dir", default="runs/dual-stream-jepa")
    ap.add_argument("--jepa-arm", default="asym_clean")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", default="runs/reachability-classifier")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    state_model = OracleCNN()
    state_model.load_weights(args.state)
    jepa = load_jepa(
        str(pathlib.Path(args.jepa_dir) / "models" / f"s0-{args.jepa_arm}.safetensors"),
        image_size=96,
    )
    train_eps = generate_episodes(
        n=40, steps=32, size=96, seed=0, goal_directed=0.5,
        nuisance_mode="normal", one_way_door=True, occlude_door_right=True,
    )
    test_eps = generate_episodes(
        n=20, steps=32, size=96, seed=9000, goal_directed=0.5,
        nuisance_mode="normal", one_way_door=True, occlude_door_right=True,
    )
    print("building pairs...", flush=True)
    train_rows = build_pairs(train_eps, state_model, jepa)
    test_rows = build_pairs(test_eps, state_model, jepa)
    start_rows = build_pairs(test_eps, state_model, jepa, starts_only=True)
    print(
        f"train {len(train_rows)} test {len(test_rows)} starts {len(start_rows)}",
        flush=True,
    )

    rows = []
    for kind in ("state", "jepa", "combined"):
        Xtr, ytr = split_xy(train_rows, kind)
        Xte, yte = split_xy(test_rows, kind)
        Xst, yst = split_xy(start_rows, kind)
        for seed in args.seeds:
            # Validation split for threshold calibration.
            perm = np.random.default_rng(seed + 99).permutation(len(Xtr))
            n_val = max(128, int(0.2 * len(Xtr)))
            vi, ti = perm[:n_val], perm[n_val:]
            model, mu, sd = train_clf(
                Xtr[ti], ytr[ti], seed=seed, epochs=args.epochs
            )
            pv = predict(model, mu, sd, Xtr[vi])
            yv = ytr[vi]
            reach_v = pv[yv > 0.5]
            tau = float(np.percentile(reach_v, 17))  # false-abort budget 0.17
            pt = predict(model, mu, sd, Xst)
            reach_t = pt[yst > 0.5]
            unreach_t = pt[yst <= 0.5]
            auc_t = auc(reach_t, unreach_t)
            abort = float(np.mean(unreach_t < tau))
            false = float(np.mean(reach_t < tau))
            row = {
                "kind": kind, "seed": seed, "tau": tau,
                "test_auc": auc_t,
                "abort_unreachable": abort,
                "false_abort_reachable": false,
                "reach_n": len(reach_t), "unreach_n": len(unreach_t),
                "reach_prob_mean": float(reach_t.mean()),
                "unreach_prob_mean": float(unreach_t.mean()),
            }
            rows.append(row)
            print(
                f"{kind} seed {seed}: AUC {auc_t:.3f} abort {abort:.2f} "
                f"false {false:.2f} tau {tau:.3f} "
                f"(reach P {reach_t.mean():.3f}, unreach {unreach_t.mean():.3f})",
                flush=True,
            )

    (out / "per_seed.json").write_text(json.dumps(rows, indent=2) + "\n")
    lines = [
        "# Reachability classifier — Hypothesis A",
        "",
        "| features | seed | AUC | abort unreach | false abort | tau |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['kind']} | {r['seed']} | {r['test_auc']:.3f} | "
            f"{r['abort_unreachable']:.2f} | {r['false_abort_reachable']:.2f} | "
            f"{r['tau']:.3f} |"
        )
    lines += ["", "Per-seed values are in `per_seed.json`."]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    print(f"wrote {out / 'report.md'}")


if __name__ == "__main__":
    main()
