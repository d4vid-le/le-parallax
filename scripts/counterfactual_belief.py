#!/usr/bin/env python3
"""Hypothesis B: same-side counterfactual + recurrent belief probe.

The current one-way door cannot make a right/right goal unreachable. We
construct a counterfactual where a hidden right-side alcove gate is open when the
door stayed open and closed when the puck crossed a closing door. The last three
frames are identical in both conditions; only the earlier history differs.

Static 3-frame encoders therefore see identical inputs and must be at chance.
A GRU over the full sequence can remember the crossing.
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
from leparallax.train import load_jepa
from leparallax.world import render_state, sample_nuisance


class StaticClf(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def __call__(self, x):
        x = nn.relu(self.fc1(x))
        x = nn.relu(self.fc2(x))
        return self.fc3(x)


class GRUClf(nn.Module):
    def __init__(self, in_dim: int = 64, hidden: int = 64):
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden)
        self.head = nn.Linear(hidden, 1)

    def __call__(self, x):
        out = self.gru(x)  # (B, T, H)
        return self.head(out[:, -1])


def auc(pos, neg):
    x = np.concatenate([pos, neg])
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(x) + 1)
    n_pos, n_neg = len(pos), len(neg)
    return float((ranks[:n_pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def make_pair(rng, closed: bool, params=None):
    if params is None:
        x = rng.uniform(0.62, 0.72)
        y = rng.uniform(0.20, 0.80)
        goal = np.array(
            [rng.uniform(0.85, 0.95), rng.uniform(0.20, 0.80)],
            dtype=np.float32,
        )
        nuisance = sample_nuisance(rng)
    else:
        x, y, goal, nuisance = params
    if closed:
        states = [
            np.array([x - 0.25, y, 0, 0], dtype=np.float32),
            np.array([0.50, y, 0, 0], dtype=np.float32),
            np.array([x - 0.15, y, 0, 0], dtype=np.float32),
            np.array([x - 0.10, y, 0, 0], dtype=np.float32),
            np.array([x - 0.05, y, 0, 0], dtype=np.float32),
            np.array([x, y, 0, 0], dtype=np.float32),
        ]
        doors = [True, True, False, False, False, False]
        label = 0.0
    else:
        states = [
            np.array([x - 0.30, y, 0, 0], dtype=np.float32),
            np.array([x - 0.25, y, 0, 0], dtype=np.float32),
            np.array([x - 0.20, y, 0, 0], dtype=np.float32),
            np.array([x - 0.10, y, 0, 0], dtype=np.float32),
            np.array([x - 0.05, y, 0, 0], dtype=np.float32),
            np.array([x, y, 0, 0], dtype=np.float32),
        ]
        doors = [True, True, True, True, True, True]
        label = 1.0
    frames = np.stack([
        render_state(s, goal, nuisance, size=96, door_open=d,
                     occlude_door_right=True)
        for s, d in zip(states, doors)
    ])
    goal_state = np.array([goal[0], goal[1], 0.0, 0.0], dtype=np.float32)
    goal_frames = np.stack([
        render_state(goal_state, goal, nuisance, size=96, door_open=None)
        for _ in range(3)
    ])
    return frames, goal_frames, label


def jepa_frame_feats(jepa, frames):
    feats = []
    for f in frames:
        jf = jepa.encoder.frame(mx.array(f[None].astype(np.float32) / 255.0))
        mx.eval(jf)
        feats.append(np.array(jf)[0])
    return np.stack(feats)


def build(n, seed, state_model, jepa):
    rng = np.random.default_rng(seed)
    state_static, jepa_static, jepa_seq, labels = [], [], [], []
    for _ in range(n):
        x = rng.uniform(0.62, 0.72)
        y = rng.uniform(0.20, 0.80)
        goal = np.array(
            [rng.uniform(0.85, 0.95), rng.uniform(0.20, 0.80)],
            dtype=np.float32,
        )
        nuisance = sample_nuisance(rng)
        params = (x, y, goal, nuisance)
        for closed in (False, True):
            frames, goal_frames, label = make_pair(rng, closed, params)
            jf = jepa_frame_feats(jepa, frames)
            zp = jepa.encode(
                mx.array(frames[-3:][None].astype(np.float32) / 255.0)
            )
            zg = jepa.encode(
                mx.array(goal_frames[None].astype(np.float32) / 255.0)
            )
            p4 = state_model(
                mx.array(frames[-3:][None].astype(np.float32) / 255.0)
            )[1]
            p4g = state_model(
                mx.array(goal_frames[None].astype(np.float32) / 255.0)
            )[1]
            mx.eval(zp, zg, p4, p4g)
            state_static.append(
                np.concatenate([np.array(p4)[0], np.array(p4g)[0]])
            )
            jepa_static.append(
                np.concatenate([np.array(zp)[0], np.array(zg)[0]])
            )
            jepa_seq.append(jf)
            labels.append(label)
    return {
        "state_static": np.stack(state_static).astype(np.float32),
        "jepa_static": np.stack(jepa_static).astype(np.float32),
        "jepa_seq": np.stack(jepa_seq).astype(np.float32),
        "y": np.array(labels, dtype=np.float32),
    }


def train_static(Xtr, ytr, Xte, yte, seed, epochs=50, lr=1e-3):
    rng = np.random.default_rng(seed)
    mx.random.seed(seed)
    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    model = StaticClf(Xtr.shape[1])
    opt = optim.AdamW(learning_rate=lr, weight_decay=1e-3)

    def loss_fn(model, x, y):
        p = mx.sigmoid(model(x))[:, 0]
        return -mx.mean(y * mx.log(p + 1e-6) + (1 - y) * mx.log(1 - p + 1e-6))

    lg = nn.value_and_grad(model, loss_fn)
    for epoch in range(1, epochs + 1):
        opt.learning_rate = lr * 0.5 * (1 + math.cos(math.pi * (epoch - 1) / max(epochs - 1, 1)))
        p = rng.permutation(len(Xtr))
        for s in range(0, len(Xtr), 64):
            idx = p[s:s + 64]
            loss, grads = lg(model, mx.array(Xtr[idx]), mx.array(ytr[idx]))
            opt.update(model, grads)
            mx.eval(model.parameters(), opt.state)
    prob = np.array(mx.sigmoid(model(mx.array(Xte)))[:, 0])
    return auc(prob[yte > 0.5], prob[yte <= 0.5]), prob


def train_gru(Xtr, ytr, Xte, yte, seed, epochs=40, lr=1e-3):
    rng = np.random.default_rng(seed)
    mx.random.seed(seed)
    model = GRUClf(Xtr.shape[-1], 64)
    opt = optim.AdamW(learning_rate=lr, weight_decay=1e-3)

    def loss_fn(model, x, y):
        p = mx.sigmoid(model(x))[:, 0]
        return -mx.mean(y * mx.log(p + 1e-6) + (1 - y) * mx.log(1 - p + 1e-6))

    lg = nn.value_and_grad(model, loss_fn)
    for epoch in range(1, epochs + 1):
        opt.learning_rate = lr * 0.5 * (1 + math.cos(math.pi * (epoch - 1) / max(epochs - 1, 1)))
        p = rng.permutation(len(Xtr))
        for s in range(0, len(Xtr), 32):
            idx = p[s:s + 32]
            loss, grads = lg(model, mx.array(Xtr[idx]), mx.array(ytr[idx]))
            opt.update(model, grads)
            mx.eval(model.parameters(), opt.state)
    prob = np.array(mx.sigmoid(model(mx.array(Xte)))[:, 0])
    return auc(prob[yte > 0.5], prob[yte <= 0.5]), prob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default="runs/dual-stream-state/state_stream_s0.safetensors")
    ap.add_argument("--jepa", default="runs/dual-stream-jepa/models/s0-asym_clean.safetensors")
    ap.add_argument("--train-n", type=int, default=400)
    ap.add_argument("--test-n", type=int, default=200)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", default="runs/counterfactual-belief")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    state_model = OracleCNN()
    state_model.load_weights(args.state)
    jepa = load_jepa(args.jepa, image_size=96)
    print("building counterfactual data...", flush=True)
    tr = build(args.train_n, 0, state_model, jepa)
    te = build(args.test_n, 9000, state_model, jepa)
    print(
        f"train {len(tr['y'])} test {len(te['y'])}; "
        f"static last-3 diff "
        f"{np.abs(tr['state_static'][0::2] - tr['state_static'][1::2]).max():.2e}",
        flush=True,
    )

    rows = []
    for label, key in (
        ("state_static", "state_static"),
        ("jepa_static", "jepa_static"),
    ):
        for seed in args.seeds:
            a, _ = train_static(
                tr[key], tr["y"], te[key], te["y"], seed
            )
            rows.append({"model": label, "seed": seed, "auc": a})
            print(f"{label} seed {seed}: AUC {a:.3f}", flush=True)

    for label, key in (("jepa_gru", "jepa_seq"),):
        for seed in args.seeds:
            a, _ = train_gru(
                tr[key], tr["y"], te[key], te["y"], seed
            )
            rows.append({"model": label, "seed": seed, "auc": a})
            print(f"{label} seed {seed}: AUC {a:.3f}", flush=True)

    (out / "per_seed.json").write_text(json.dumps(rows, indent=2) + "\n")
    lines = [
        "# Same-side counterfactual belief test",
        "",
        "| model | seed | AUC |",
        "|---|---:|---:|",
    ]
    for r in rows:
        lines.append(f"| {r['model']} | {r['seed']} | {r['auc']:.3f} |")
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    print(f"wrote {out / 'report.md'}")


if __name__ == "__main__":
    main()
