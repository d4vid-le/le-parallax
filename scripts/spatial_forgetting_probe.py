#!/usr/bin/env python3
"""Sanity check 1: did the reachability fine-tune collapse spatial/dynamics info?

Compares the frozen JEPA encoder with the end-to-end fine-tuned frame encoder on:
- a ridge state probe (position/velocity), and
- one-step JEPA dynamics MSE on held-out natural transitions.

The merge/predictor are frozen for both; only the frame encoder differs.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from leparallax.data import build_temporal_samples, collate_temporal
from leparallax.model import FrameEncoder
from leparallax.probes import _ridge_eval
from leparallax.train import load_jepa
from leparallax.world import generate_episodes, history_indices


def collect(episodes):
    obs, states = [], []
    for ep in episodes:
        for t in range(len(ep.states)):
            obs.append(ep.frames[history_indices(t)])
            states.append(ep.states[t])
    return np.stack(obs).astype(np.float32) / 255.0, np.stack(states).astype(np.float32)


def encode(model, obs, batch=64):
    out = []
    for s in range(0, len(obs), batch):
        idx = np.arange(s, min(s + batch, len(obs)))
        z = model.encode(mx.array(obs[idx]))
        mx.eval(z)
        out.append(np.array(z))
    return np.concatenate(out)


def dynamics_mse(model, samples, batch=64):
    total, n = 0.0, 0
    for s in range(0, len(samples), batch):
        idx = np.arange(s, min(s + batch, len(samples)))
        obs, act, nxt, _, _, _, _ = collate_temporal(samples, idx)
        z = model.encode(mx.array(obs))
        zn = model.encode(mx.array(nxt))
        zp = model.predict(z, mx.array(act))
        loss = mx.mean((zp - zn) ** 2)
        mx.eval(loss)
        total += float(loss) * len(idx)
        n += len(idx)
    return total / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frozen", default="runs/dual-stream-jepa/models/s0-asym_clean.safetensors")
    ap.add_argument("--finetuned-frame", default="runs/counterfactual-gru/frame_s0.safetensors")
    ap.add_argument("--out", default="runs/spatial-forgetting")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    train_eps = generate_episodes(
        n=40, steps=32, size=96, seed=0, goal_directed=0.5,
        nuisance_mode="normal", one_way_door=True, occlude_door_right=True,
    )
    test_eps = generate_episodes(
        n=20, steps=32, size=96, seed=9000, goal_directed=0.5,
        nuisance_mode="normal", one_way_door=True, occlude_door_right=True,
    )
    tr_obs, tr_state = collect(train_eps)
    te_obs, te_state = collect(test_eps)
    te_samples = build_temporal_samples(test_eps)

    frozen = load_jepa(args.frozen, image_size=96)
    fine_frame = FrameEncoder(64, image_size=96)
    fine_frame.load_weights(args.finetuned_frame)
    fine = load_jepa(args.frozen, image_size=96)
    fine.encoder.frame = fine_frame

    rows = {}
    for label, model in (("frozen", frozen), ("finetuned", fine)):
        Ztr = encode(model, tr_obs)
        Zte = encode(model, te_obs)
        r2 = _ridge_eval(Ztr, tr_state, Zte, te_state)
        row = {
            "state_r2": float(np.mean(r2)),
            "position_r2": float(np.mean(r2[:2])),
            "velocity_r2": float(np.mean(r2[2:])),
            "dynamics_mse": float(dynamics_mse(model, te_samples)),
        }
        rows[label] = row
        print(
            f"{label}: state {row['state_r2']:.3f} "
            f"(pos {row['position_r2']:.3f} vel {row['velocity_r2']:.3f}) "
            f"dynamics MSE {row['dynamics_mse']:.6f}",
            flush=True,
        )

    (out / "report.json").write_text(json.dumps(rows, indent=2) + "\n")
    lines = [
        "# Spatial/dynamics forgetting check",
        "",
        "| encoder | 4-D state R² | position R² | velocity R² | dynamics MSE |",
        "|---|---:|---:|---:|---:|",
    ]
    for label in ("frozen", "finetuned"):
        r = rows[label]
        lines.append(
            f"| {label} | {r['state_r2']:.3f} | {r['position_r2']:.3f} | "
            f"{r['velocity_r2']:.3f} | {r['dynamics_mse']:.6f} |"
        )
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {out / 'report.md'}")


if __name__ == "__main__":
    main()
