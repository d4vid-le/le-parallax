#!/usr/bin/env python3
"""Phase 0: train the temporal baseline, probe it, and plan with it."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from leparallax.data import build_temporal_samples
from leparallax.plan import evaluate_planner, evaluate_random
from leparallax.probes import probe_report, nuisance_sensitivity
from leparallax.train import train_jepa
from leparallax.world import generate_episodes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=48)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lambda-sig", type=float, default=0.09)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plan-episodes", type=int, default=10)
    ap.add_argument("--out", default="runs/phase0")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(
        f"generating {args.episodes} episodes at {args.size}px, "
        f"{args.steps} steps",
        flush=True,
    )
    episodes = generate_episodes(
        n=args.episodes,
        steps=args.steps,
        size=args.size,
        seed=args.seed,
        goal_directed=0.30,
    )
    n_train = int(0.8 * len(episodes))
    n_val = int(0.1 * len(episodes))
    train_eps = episodes[:n_train]
    val_eps = episodes[n_train : n_train + n_val]
    test_eps = episodes[n_train + n_val :]
    print(
        f"train {len(train_eps)} / val {len(val_eps)} / test {len(test_eps)}",
        flush=True,
    )

    samples = build_temporal_samples(train_eps)
    print(f"{len(samples)} temporal samples", flush=True)

    model = train_jepa(
        samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lambda_sig=args.lambda_sig,
        seed=args.seed,
        image_size=args.size,
        save_path=out / "model.safetensors",
    )

    print("probing held-out episodes...", flush=True)
    probe = probe_report(model, train_eps, test_eps, per_episode=16, seed=args.seed)
    sensitivity = nuisance_sensitivity(
        model, test_eps, per_episode=8, seed=args.seed
    )
    print(json.dumps({"probe": probe, "sensitivity": sensitivity}, indent=2), flush=True)

    print("planning...", flush=True)
    planner = evaluate_planner(
        model, test_eps, n_episodes=args.plan_episodes, horizon=5, seed=args.seed
    )
    random = evaluate_random(
        test_eps, n_episodes=args.plan_episodes, seed=args.seed
    )
    print(json.dumps({"planner": planner, "random": random}, indent=2), flush=True)

    report = {
        "args": vars(args),
        "probe": probe,
        "sensitivity": sensitivity,
        "planner": planner,
        "random": random,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {out / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
