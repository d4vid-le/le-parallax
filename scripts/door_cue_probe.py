#!/usr/bin/env python3
"""Sanity check 2: does the frozen JEPA encode the visible door cue?

Controlled left-side histories, same puck/goal/nuisance, only door open/closed.
Probes a linear classifier on:
- the single-frame JEPA frame feature at x=0.45 (door visible),
- the single-frame feature at x=0.50 (occlusion band hides the door),
- the 3-frame JEPA latent at x=0.45 and x=0.50.

High AUC at x=0.45 and chance at x=0.50 is the mechanistic proof: the cue is
encoded when visible and lost at the crossing, so memory is required.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from leparallax.train import load_jepa
from leparallax.world import render_state, sample_nuisance


def auc(pos, neg):
    x = np.concatenate([pos, neg])
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(x) + 1)
    n_pos, n_neg = len(pos), len(neg)
    return float((ranks[:n_pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def ridge_auc(Xtr, ytr, Xte, yte, alpha=1.0):
    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    Xtr = np.concatenate([Xtr, np.ones((len(Xtr), 1))], 1)
    Xte = np.concatenate([Xte, np.ones((len(Xte), 1))], 1)
    W = np.linalg.solve(Xtr.T @ Xtr + alpha * np.eye(Xtr.shape[1]), Xtr.T @ ytr)
    score = Xte @ W
    return auc(score[yte > 0.5], score[yte <= 0.5])


def make_pair(rng, x):
    y = rng.uniform(0.20, 0.80)
    goal = np.array(
        [rng.uniform(0.10, 0.90), rng.uniform(0.20, 0.80)], dtype=np.float32
    )
    nuisance = sample_nuisance(rng)
    states = [
        np.array([x - 0.10, y, 0, 0], dtype=np.float32),
        np.array([x - 0.05, y, 0, 0], dtype=np.float32),
        np.array([x, y, 0, 0], dtype=np.float32),
    ]
    outs = []
    for door in (True, False):
        frames = np.stack([
            render_state(s, goal, nuisance, size=96, door_open=door,
                         occlude_door_right=True)
            for s in states
        ])
        outs.append(frames)
    return outs[0], outs[1]


def feat(model, frames, single=False):
    if single:
        z = model.encoder.frame(mx.array(frames[-1][None].astype(np.float32) / 255.0))
    else:
        z = model.encode(mx.array(frames[None].astype(np.float32) / 255.0))
    mx.eval(z)
    return np.array(z)[0]


def build(n, seed, model):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        for x in (0.45, 0.50):
            fo, fc = make_pair(rng, x)
            rows.append({
                "x": x,
                "single_open": feat(model, fo, True),
                "single_closed": feat(model, fc, True),
                "seq_open": feat(model, fo, False),
                "seq_closed": feat(model, fc, False),
            })
    return rows


def probe(rows, key_open, key_closed):
    Xtr, ytr, Xte, yte = [], [], [], []
    n = len(rows)
    split = int(0.7 * n)
    for i, r in enumerate(rows):
        X_tr = [r[key_open], r[key_closed]]
        y_tr = [1.0, 0.0]
        target = (Xtr, ytr) if i < split else (Xte, yte)
        target[0].extend(X_tr)
        target[1].extend(y_tr)
    return ridge_auc(
        np.stack(Xtr), np.array(ytr), np.stack(Xte), np.array(yte)
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jepa", default="runs/dual-stream-jepa/models/s0-asym_clean.safetensors")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--out", default="runs/door-cue-probe")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model = load_jepa(args.jepa, image_size=96)
    rows = build(args.n, 0, model)
    results = {}
    for x in (0.45, 0.50):
        sub = [r for r in rows if r["x"] == x]
        results[f"single_x{x:.2f}"] = probe(sub, "single_open", "single_closed")
        results[f"seq_x{x:.2f}"] = probe(sub, "seq_open", "seq_closed")
    for k, v in results.items():
        print(f"{k}: AUC {v:.3f}")
    (out / "report.json").write_text(json.dumps(results, indent=2) + "\n")
    lines = ["# Door-cue probe (frozen JEPA)", "", "| probe | AUC |", "|---|---:|"]
    for k, v in results.items():
        lines.append(f"| {k} | {v:.3f} |")
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {out / 'report.md'}")


if __name__ == "__main__":
    main()
