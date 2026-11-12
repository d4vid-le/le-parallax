#!/usr/bin/env python3
"""Five-seed OOD diagnostic: temporal vs good-view H1 encoders.

No training. Reuses the held-out/shifted/extreme conditions from the collision
suite to quantify how the good-view state geometry behaves outside its training
nuisance distribution.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from collision_utils import bootstrap_ci, build_conditions, evaluate
from leparallax.data import build_probe_samples
from leparallax.train import load_jepa
from leparallax.world import generate_episodes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--train-episodes", type=int, default=80)
    ap.add_argument("--test-episodes", type=int, default=10)
    ap.add_argument("--per-episode", type=int, default=16)
    ap.add_argument("--out", default="runs/ood-diagnostic")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in args.seeds:
        print(f"\n=== seed {seed} ===", flush=True)
        a_train = generate_episodes(
            n=args.train_episodes, steps=32, size=96, seed=seed,
            goal_directed=0.5, nuisance_mode="normal",
        )
        a_test = generate_episodes(
            n=args.test_episodes, steps=32, size=96, seed=11000 + seed,
            goal_directed=0.5, nuisance_mode="normal",
        )
        probe_train = build_probe_samples(
            a_train, per_episode=args.per_episode, seed=seed
        )
        conditions = build_conditions(a_test, args.per_episode, seed)
        for arm in ("temporal", "good"):
            model = load_jepa(
                f"runs/h1-confirm/models/s{seed}-{arm}.safetensors",
                image_size=96,
            )
            metrics = evaluate(model, probe_train, conditions)
            row = {"seed": seed, "arm": arm, **{
                f"{k}_{m}": metrics[k][m]
                for k in conditions for m in ("state", "position", "velocity")
            }}
            rows.append(row)
            print(
                f"  {arm}: "
                + " | ".join(f"{k} {metrics[k]['state']:.3f}" for k in conditions),
                flush=True,
            )

    (out / "per_seed.json").write_text(json.dumps(rows, indent=2) + "\n")

    def vals(arm, cond):
        return np.array([
            r[f"{cond}_state"] for r in rows if r["arm"] == arm
        ])

    lines = [
        "# OOD nuisance diagnostic — temporal vs good",
        "",
        "| arm | normal | heldout swap | shifted | extreme |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm in ("temporal", "good"):
        lines.append(
            f"| {arm} | {vals(arm, 'normal').mean():.3f} | "
            f"{vals(arm, 'heldout_swap').mean():.3f} | "
            f"{vals(arm, 'shifted').mean():.3f} | "
            f"{vals(arm, 'extreme').mean():.3f} |"
        )
    d_shift = vals("good", "shifted") - vals("temporal", "shifted")
    d_extreme = vals("good", "extreme") - vals("temporal", "extreme")
    d_normal = vals("good", "normal") - vals("temporal", "normal")
    lines += [
        "",
        "## Paired differences (good − temporal)",
        "",
        f"- normal: **{d_normal.mean():+.3f}** "
        f"(95% CI {bootstrap_ci(d_normal, seed=0)[0]:+.3f} to "
        f"{bootstrap_ci(d_normal, seed=0)[1]:+.3f})",
        f"- shifted: **{d_shift.mean():+.3f}** "
        f"(95% CI {bootstrap_ci(d_shift, seed=1)[0]:+.3f} to "
        f"{bootstrap_ci(d_shift, seed=1)[1]:+.3f})",
        f"- extreme: **{d_extreme.mean():+.3f}** "
        f"(95% CI {bootstrap_ci(d_extreme, seed=2)[0]:+.3f} to "
        f"{bootstrap_ci(d_extreme, seed=2)[1]:+.3f})",
        "",
        "Positive shifted/extreme means would favour good views; negative means",
        "mean the good-view encoder is worse than temporal outside its nuisance",
        "distribution.",
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines[-12:]))
    print(f"wrote {out / 'report.md'}")


if __name__ == "__main__":
    main()
