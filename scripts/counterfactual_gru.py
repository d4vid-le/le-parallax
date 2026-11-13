#!/usr/bin/env python3
"""End-to-end JEPA + GRU on the same-side counterfactual.

The frozen JEPA frame features were at chance on the belief test. This script
fine-tunes the JEPA frame encoder with a GRU and a reachability head for 40
epochs on the counterfactual sequences.
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

from counterfactual_belief import make_pair
from leparallax.train import load_jepa
from leparallax.world import sample_nuisance


class JEPAGRU(nn.Module):
    def __init__(self, frame_encoder, hidden: int = 64):
        super().__init__()
        self.frame = frame_encoder
        self.gru = nn.GRU(64, hidden)
        self.head = nn.Linear(hidden, 1)

    def __call__(self, frames):  # (B, T, H, W, C)
        b, t, h, w, c = frames.shape
        f = self.frame(frames.reshape(b * t, h, w, c)).reshape(b, t, 64)
        out = self.gru(f)
        return self.head(out[:, -1])


def auc(pos, neg):
    x = np.concatenate([pos, neg])
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(x) + 1)
    n_pos, n_neg = len(pos), len(neg)
    return float((ranks[:n_pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def build_raw(n, seed):
    rng = np.random.default_rng(seed)
    frames, labels = [], []
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
            f, _, label = make_pair(rng, closed, params)
            frames.append(f)
            labels.append(label)
    return (
        np.stack(frames).astype(np.float32) / 255.0,
        np.array(labels, dtype=np.float32),
    )


def train_one(Xtr, ytr, Xte, yte, frame_encoder, seed, epochs=40,
              batch=8, lr=1e-3, save_path=None):
    rng = np.random.default_rng(seed)
    mx.random.seed(seed)
    model = JEPAGRU(frame_encoder)
    opt = optim.AdamW(learning_rate=lr, weight_decay=1e-4)

    def loss_fn(model, x, y):
        p = mx.sigmoid(model(x))[:, 0]
        return -mx.mean(y * mx.log(p + 1e-6) + (1 - y) * mx.log(1 - p + 1e-6))

    lg = nn.value_and_grad(model, loss_fn)
    for epoch in range(1, epochs + 1):
        lr_t = lr * 0.5 * (
            1.0 + math.cos(math.pi * (epoch - 1) / max(epochs - 1, 1))
        )
        opt.learning_rate = lr_t
        p = rng.permutation(len(Xtr))
        total, seen = 0.0, 0
        for s in range(0, len(Xtr), batch):
            idx = p[s:s + batch]
            loss, grads = lg(model, mx.array(Xtr[idx]), mx.array(ytr[idx]))
            opt.update(model, grads)
            mx.eval(model.parameters(), opt.state)
            total += float(loss.item()) * len(idx)
            seen += len(idx)
        if epoch % 10 == 0:
            print(
                f"    epoch {epoch}/{epochs} loss {total / max(seen, 1):.4f}",
                flush=True,
            )
    if save_path is not None:
        pathlib.Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        model.frame.save_weights(str(save_path))
    prob = np.array(mx.sigmoid(model(mx.array(Xte)))[:, 0])
    return auc(prob[yte > 0.5], prob[yte <= 0.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jepa", default="runs/dual-stream-jepa/models/s0-asym_clean.safetensors")
    ap.add_argument("--train-n", type=int, default=400)
    ap.add_argument("--test-n", type=int, default=200)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--out", default="runs/counterfactual-gru")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print("building raw counterfactual data...", flush=True)
    Xtr, ytr = build_raw(args.train_n, 0)
    Xte, yte = build_raw(args.test_n, 9000)
    print(f"train {Xtr.shape} test {Xte.shape}", flush=True)
    jepa = load_jepa(args.jepa, image_size=96)
    rows = []
    for seed in args.seeds:
        a = train_one(
            Xtr, ytr, Xte, yte, jepa.encoder.frame, seed,
            epochs=args.epochs,
            save_path=(
                out / f"frame_s{seed}.safetensors" if args.save else None
            ),
        )
        rows.append({"seed": seed, "auc": a})
        print(f"jepa_gru end2end seed {seed}: AUC {a:.3f}", flush=True)
    (out / "per_seed.json").write_text(json.dumps(rows, indent=2) + "\n")
    lines = ["# End-to-end JEPA+GRU counterfactual", "", "| seed | AUC |", "|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['seed']} | {r['auc']:.3f} |")
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    print(f"wrote {out / 'report.md'}")


if __name__ == "__main__":
    main()
