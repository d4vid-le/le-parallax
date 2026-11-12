#!/usr/bin/env python3
"""Nonlinear state probe: is the state information missing, or just not linear?

Compares a linear ridge probe with a random-Fourier-feature ridge probe on the
same frozen latents. A large nonlinear gain means the state is present but the
linear guardrail is the wrong readout; no gain means the encoder lost the state.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from leparallax.probes import _ridge_eval
from leparallax.train import load_jepa
from leparallax.world import generate_episodes, history_indices


def collect(episodes):
    obs, states = [], []
    for ep in episodes:
        for t in range(len(ep.states)):
            obs.append(ep.frames[history_indices(t)])
            states.append(ep.states[t])
    return np.stack(obs), np.stack(states)


def encode(model, obs, batch=64):
    out = []
    for start in range(0, len(obs), batch):
        idx = np.arange(start, min(start + batch, len(obs)))
        z = model.encode(mx.array(obs[idx].astype(np.float32) / 255.0))
        mx.eval(z)
        out.append(np.array(z))
    return np.concatenate(out)


def rff_ridge(Ztr, Ytr, Zte, Yte, features=512, alpha=10.0, seed=0):
    rng = np.random.default_rng(seed)
    mu = Ztr.mean(axis=0, keepdims=True)
    sd = Ztr.std(axis=0, keepdims=True) + 1e-6
    Ztr = (Ztr - mu) / sd
    Zte = (Zte - mu) / sd
    W = rng.normal(0.0, 1.0 / np.sqrt(Ztr.shape[1]), (Ztr.shape[1], features))
    b = rng.uniform(0.0, 2.0 * np.pi, size=features)
    Ptr = np.concatenate([np.cos(Ztr @ W + b), np.sin(Ztr @ W + b)], axis=1)
    Pte = np.concatenate([np.cos(Zte @ W + b), np.sin(Zte @ W + b)], axis=1)
    # Standardize features and add a bias.
    fmu = Ptr.mean(axis=0, keepdims=True)
    fsd = Ptr.std(axis=0, keepdims=True) + 1e-6
    Ptr = (Ptr - fmu) / fsd
    Pte = (Pte - fmu) / fsd
    Ptr = np.concatenate([Ptr, np.ones((len(Ptr), 1))], axis=1)
    Pte = np.concatenate([Pte, np.ones((len(Pte), 1))], axis=1)
    Wr = np.linalg.solve(Ptr.T @ Ptr + alpha * np.eye(Ptr.shape[1]), Ptr.T @ Ytr)
    pred = Pte @ Wr
    ss_res = ((Yte - pred) ** 2).sum(axis=0)
    ss_tot = ((Yte - Yte.mean(axis=0)) ** 2).sum(axis=0)
    return float(np.mean(1.0 - ss_res / (ss_tot + 1e-8)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/phase2-pilot")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--arms", nargs="+", default=[
        "temp_matched", "sym_head", "asym_clean", "asym_vol_bounded"
    ])
    ap.add_argument("--train-episodes", type=int, default=40)
    ap.add_argument("--test-episodes", type=int, default=20)
    ap.add_argument("--features", type=int, default=512)
    ap.add_argument("--frame-state-head", action="store_true")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    rows = []
    for seed in args.seeds:
        train_eps = generate_episodes(
            n=args.train_episodes, steps=32, size=96, seed=seed,
            goal_directed=0.5, nuisance_mode="normal", one_way_door=True,
            occlude_door_right=True,
        )
        test_eps = generate_episodes(
            n=args.test_episodes, steps=32, size=96, seed=9000 + seed,
            goal_directed=0.5, nuisance_mode="normal", one_way_door=True,
            occlude_door_right=True,
        )
        obs_tr, s_tr = collect(train_eps)
        obs_te, s_te = collect(test_eps)
        for arm in args.arms:
            path = out / "models" / f"s{seed}-{arm}.safetensors"
            if not path.exists():
                print("missing", path, flush=True)
                continue
            model = load_jepa(
                str(path), image_size=96,
                frame_state_head=args.frame_state_head,
            )
            Ztr = encode(model, obs_tr)
            Zte = encode(model, obs_te)
            lin = float(np.mean(_ridge_eval(Ztr, s_tr, Zte, s_te)))
            nonlin = rff_ridge(
                Ztr, s_tr, Zte, s_te,
                features=args.features, alpha=10.0, seed=seed,
            )
            row = {"seed": seed, "arm": arm, "linear": lin,
                   "nonlinear": nonlin, "gain": nonlin - lin}
            rows.append(row)
            print(
                f"seed {seed} {arm}: linear {lin:.3f} -> "
                f"nonlinear {nonlin:.3f} (gain {nonlin - lin:+.3f})",
                flush=True,
            )

    (out / "state_probe.json").write_text(json.dumps(rows, indent=2) + "\n")
    lines = [
        "# Nonlinear state probe",
        "",
        "| seed | arm | linear R² | RFF R² | gain |",
        "|---:|---|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['seed']} | {r['arm']} | {r['linear']:.3f} | "
            f"{r['nonlinear']:.3f} | {r['gain']:+.3f} |"
        )
    (out / "state_probe.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {out / 'state_probe.md'}")


if __name__ == "__main__":
    main()
