#!/usr/bin/env python3
"""One-seed pilot: temporal vs bad-view vs good-view arms."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from leparallax.data import (
    build_bad_view_pairs,
    build_temporal_samples,
    build_view_pairs,
)
from leparallax.plan import evaluate_planner, evaluate_random
from leparallax.probes import nuisance_sensitivity, probe_report
from leparallax.train import train_jepa
from leparallax.world import generate_episodes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=48)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--per-episode", type=int, default=4)
    ap.add_argument("--lambda-view", type=float, default=0.5)
    ap.add_argument("--plan-episodes", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/pilot")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    episodes = generate_episodes(
        n=args.episodes, steps=args.steps, size=args.size,
        seed=args.seed, goal_directed=0.5,
    )
    n_train = int(0.8 * len(episodes))
    n_val = int(0.1 * len(episodes))
    train_eps = episodes[:n_train]
    test_eps = episodes[n_train + n_val :]
    print(f"train {len(train_eps)} / test {len(test_eps)}", flush=True)

    temporal = build_temporal_samples(train_eps)
    good = build_view_pairs(train_eps, per_episode=args.per_episode, seed=args.seed)
    bad = build_bad_view_pairs(train_eps, per_episode=args.per_episode, seed=args.seed)
    print(f"{len(temporal)} temporal / {len(good)} good / {len(bad)} bad", flush=True)

    arms = [
        ("temporal", None),
        ("bad", bad),
        ("good", good),
    ]
    report = {"args": vars(args), "arms": {}}
    random = evaluate_random(test_eps, n_episodes=args.plan_episodes, seed=args.seed)
    report["random"] = random

    for name, views in arms:
        print(f"\n=== {name} ===", flush=True)
        t0 = time.time()
        model = train_jepa(
            temporal,
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=args.seed,
            image_size=args.size,
            view_pairs=views,
            lambda_view=args.lambda_view if views is not None else 0.0,
            save_path=out / f"{name}.safetensors",
            verbose=False,
        )
        probe = probe_report(model, train_eps, test_eps, per_episode=16, seed=args.seed)
        sens = nuisance_sensitivity(model, test_eps, per_episode=8, seed=args.seed)
        planner = evaluate_planner(
            model, test_eps, n_episodes=args.plan_episodes,
            horizon=3, seed=args.seed,
        )
        report["arms"][name] = {
            "probe": probe,
            "sensitivity": sens,
            "planner": planner,
            "train_s": round(time.time() - t0, 2),
        }
        print(
            f"state {probe['state']:.3f}  pos {probe['position']:.3f}  "
            f"vel {probe['velocity']:.3f}  ratio {sens['ratio']:.3f}  "
            f"plan {planner['success']:.2f} ({planner['mean_dist']:.3f})",
            flush=True,
        )

    report["random"] = random
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("\nrandom", random, flush=True)
    print("wrote", out / "report.json", flush=True)


if __name__ == "__main__":
    main()
