#!/usr/bin/env python3
"""Collision 1: counterfactual nuisance swap.

Train a state probe on normal latents, then test the same states with a
different state's nuisance. A causal state representation should survive; a
correlational one should break.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from leparallax.data import ProbeSample, build_probe_samples, history_obs
from leparallax.probes import _ridge_eval, collect_latents
from leparallax.train import load_jepa
from leparallax.world import generate_episodes, history_indices


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--per-episode", type=int, default=16)
    ap.add_argument("--out", default="runs/nuisance-collision")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in args.seeds:
        print(f"\n=== seed {seed} ===", flush=True)
        train_eps = generate_episodes(
            n=80, steps=32, size=96, seed=seed,
            goal_directed=0.5, nuisance_mode="normal",
        )
        test_eps = generate_episodes(
            n=10, steps=32, size=96, seed=11000 + seed,
            goal_directed=0.5, nuisance_mode="normal",
        )
        rng = np.random.default_rng(seed + 1)

        normal, swapped = [], []
        for ep in test_eps:
            n = min(args.per_episode, len(ep.states))
            for t in rng.choice(len(ep.states), size=n, replace=False):
                t = int(t)
                normal.append(ProbeSample(
                    obs=ep.frames[history_indices(t)],
                    state=ep.states[t],
                    nuisance=ep.nuisance,
                ))
                other = train_eps[int(rng.integers(0, len(train_eps)))]
                swapped.append(ProbeSample(
                    obs=history_obs(ep, t, other.nuisance),
                    state=ep.states[t],
                    nuisance=other.nuisance,
                ))

        train = build_probe_samples(
            train_eps, per_episode=args.per_episode, seed=seed
        )
        for arm in ("temporal", "good"):
            model = load_jepa(
                f"runs/h1-confirm/models/s{seed}-{arm}.safetensors",
                image_size=96,
            )
            Xtr, Ytr, _ = collect_latents(model, train)
            Xn, Yn, _ = collect_latents(model, normal)
            Xs, Ys, _ = collect_latents(model, swapped)
            r2_normal = _ridge_eval(Xtr, Ytr, Xn, Yn)
            r2_swap = _ridge_eval(Xtr, Ytr, Xs, Ys)
            row = {
                "seed": seed, "arm": arm,
                "normal_r2": float(np.mean(r2_normal)),
                "swap_r2": float(np.mean(r2_swap)),
                "drop": float(np.mean(r2_normal) - np.mean(r2_swap)),
            }
            rows.append(row)
            print(
                f"  {arm}: normal R2 {row['normal_r2']:.3f} "
                f"swap R2 {row['swap_r2']:.3f} drop {row['drop']:+.3f}",
                flush=True,
            )

    (out / "per_seed.json").write_text(json.dumps(rows, indent=2) + "\n")
    lines = [
        "# Counterfactual nuisance swap",
        "",
        "| seed | arm | normal R² | swapped R² | drop |",
        "|---:|---|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['seed']} | {r['arm']} | {r['normal_r2']:.3f} | "
            f"{r['swap_r2']:.3f} | {r['drop']:+.3f} |"
        )
    lines += ["", "## Means", ""]
    for arm in ("temporal", "good"):
        rs = [r for r in rows if r["arm"] == arm]
        lines.append(
            f"- **{arm}**: normal {np.mean([r['normal_r2'] for r in rs]):.3f}, "
            f"swapped {np.mean([r['swap_r2'] for r in rs]):.3f}, "
            f"drop {np.mean([r['drop'] for r in rs]):+.3f}"
        )
    lines += [
        "",
        "Bet: good-view drop ≥ 0.10 (invariance is correlational).",
        "Surprise: good-view drop ≤ 0.05 (invariance is causal).",
        "",
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[-7:]))
    print(f"wrote {out / 'report.md'}")


if __name__ == "__main__":
    main()
