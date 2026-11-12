#!/usr/bin/env python3
"""Collision 1c: Counterfactual State Alignment (CSA).

Same temporal support as the mix-ft control, plus cross-domain counterfactual
pairs whose state-head predictions must agree. Tests whether alignment adds
anything beyond simply training on shifted data.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from collision_utils import (
    bootstrap_ci,
    build_conditions,
    build_cross_domain_pairs,
    evaluate,
    train_csa,
)
from leparallax.data import build_probe_samples, build_temporal_samples
from leparallax.train import load_jepa, train_jepa
from leparallax.world import generate_episodes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--train-episodes", type=int, default=80)
    ap.add_argument("--test-episodes", type=int, default=10)
    ap.add_argument("--per-episode", type=int, default=16)
    ap.add_argument("--pair-per-episode", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lambda-latent", type=float, default=0.1)
    ap.add_argument("--lambda-state", type=float, default=0.5)
    ap.add_argument("--out", default="runs/csa")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    (out / "models").mkdir(parents=True, exist_ok=True)
    manifest = {
        "prereg": "docs/PREREG-CSA.md",
        "args": vars(args),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

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
        b_train = generate_episodes(
            n=args.train_episodes, steps=32, size=96, seed=2000 + seed,
            goal_directed=0.5, nuisance_mode="shift",
        )
        probe_train = build_probe_samples(
            a_train, per_episode=args.per_episode, seed=seed
        )
        conditions = build_conditions(a_test, args.per_episode, seed)
        half = args.train_episodes // 2
        mixed = a_train[:half] + b_train[:half]
        mixed_samples = build_temporal_samples(mixed)

        good_path = f"runs/h1-confirm/models/s{seed}-good.safetensors"
        model = load_jepa(good_path, image_size=96)
        metrics = evaluate(model, probe_train, conditions)
        rows.append({"seed": seed, "stage": "frozen", **{
            f"{k}_{m}": metrics[k][m]
            for k in conditions for m in ("state", "position", "velocity")
        }})
        print(
            "  frozen: "
            + " | ".join(f"{k} {metrics[k]['state']:.3f}" for k in conditions),
            flush=True,
        )

        model = load_jepa(good_path, image_size=96)
        model = train_jepa(
            mixed_samples, epochs=args.epochs, batch_size=args.batch_size,
            lr=args.lr, seed=seed, image_size=96, model=model,
            save_path=out / "models" / f"s{seed}-mix-ft.safetensors",
            verbose=False,
        )
        metrics = evaluate(model, probe_train, conditions)
        rows.append({"seed": seed, "stage": "mix-ft", **{
            f"{k}_{m}": metrics[k][m]
            for k in conditions for m in ("state", "position", "velocity")
        }})
        print(
            "  mix-ft: "
            + " | ".join(f"{k} {metrics[k]['state']:.3f}" for k in conditions),
            flush=True,
        )

        pairs = build_cross_domain_pairs(
            a_train[:half], b_train[:half], args.pair_per_episode, seed
        )
        model = load_jepa(good_path, image_size=96)
        model = train_csa(
            model, mixed_samples, pairs, epochs=args.epochs,
            batch_size=args.batch_size, lr=args.lr, seed=seed,
            out_path=out / "models" / f"s{seed}-csa.safetensors",
            lambda_latent=args.lambda_latent, lambda_state=args.lambda_state,
        )
        metrics = evaluate(model, probe_train, conditions)
        rows.append({"seed": seed, "stage": "csa", **{
            f"{k}_{m}": metrics[k][m]
            for k in conditions for m in ("state", "position", "velocity")
        }})
        print(
            "  csa: "
            + " | ".join(f"{k} {metrics[k]['state']:.3f}" for k in conditions),
            flush=True,
        )

    (out / "per_seed.json").write_text(json.dumps(rows, indent=2) + "\n")

    def vals(stage, cond):
        return np.array([
            r[f"{cond}_state"] for r in rows if r["stage"] == stage
        ])

    lines = [
        "# Counterfactual State Alignment (CSA)",
        "",
        "| stage | normal | heldout swap | shifted | extreme |",
        "|---|---:|---:|---:|---:|",
    ]
    for stage in ("frozen", "mix-ft", "csa"):
        lines.append(
            f"| {stage} | {vals(stage, 'normal').mean():.3f} | "
            f"{vals(stage, 'heldout_swap').mean():.3f} | "
            f"{vals(stage, 'shifted').mean():.3f} | "
            f"{vals(stage, 'extreme').mean():.3f} |"
        )

    d_shift = vals("csa", "shifted") - vals("mix-ft", "shifted")
    d_extreme = vals("csa", "extreme") - vals("mix-ft", "extreme")
    d_normal = vals("frozen", "normal") - vals("csa", "normal")
    ci_shift = bootstrap_ci(d_shift, seed=0)
    ci_extreme = bootstrap_ci(d_extreme, seed=1)
    ci_normal = bootstrap_ci(d_normal, seed=2)
    hold_shift = d_shift.mean() >= 0.03 and ci_shift[0] > 0
    hold_normal = d_normal.mean() <= 0.03
    hold_extreme = d_extreme.mean() >= 0.03 and ci_extreme[0] > 0
    verdict = hold_shift and hold_normal

    lines += [
        "",
        "## Paired differences (5 seeds)",
        "",
        f"- CSA − mix-ft shifted: **{d_shift.mean():+.3f}** "
        f"(95% CI {ci_shift[0]:+.3f} to {ci_shift[1]:+.3f}); "
        f"needed ≥0.03 with CI >0 → **{'holds' if hold_shift else 'fails'}**.",
        f"- frozen − CSA normal: **{d_normal.mean():+.3f}** "
        f"(95% CI {ci_normal[0]:+.3f} to {ci_normal[1]:+.3f}); "
        f"needed ≤0.03 → **{'holds' if hold_normal else 'fails'}**.",
        f"- CSA − mix-ft extreme: **{d_extreme.mean():+.3f}** "
        f"(95% CI {ci_extreme[0]:+.3f} to {ci_extreme[1]:+.3f}); "
        f"secondary → **{'holds' if hold_extreme else 'fails'}**.",
        "",
        f"**Endpoint verdict: {'CSA beats domain exposure' if verdict else 'CSA fails the endpoint'}.**",
        "",
        "Per-seed values are in `per_seed.json`.",
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines[-14:]))
    print(f"wrote {out / 'report.md'}")


if __name__ == "__main__":
    main()
