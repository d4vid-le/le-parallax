#!/usr/bin/env python3
"""H2-v3: domain-randomized encoders, robust readout, state-feedback control."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from leparallax.data import build_temporal_samples, build_view_pairs
from leparallax.plan import evaluate_random
from leparallax.readout import (
    build_readout_data,
    fit_ridge,
    readout_error,
    state_feedback,
)
from leparallax.train import load_jepa, train_jepa
from leparallax.world import generate_episodes, sample_nuisance_mix

PREREG = pathlib.Path(__file__).resolve().parents[1] / "docs" / "PREREG-H2v3.md"


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--readout-episodes", type=int, default=80)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--per-episode", type=int, default=4)
    ap.add_argument("--states-per-episode", type=int, default=16)
    ap.add_argument("--lambda-view", type=float, default=2.0)
    ap.add_argument("--ood-episodes", type=int, default=10)
    ap.add_argument("--id-episodes", type=int, default=5)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--k", type=float, default=2.0)
    ap.add_argument("--damp", type=float, default=1.0)
    ap.add_argument("--out", default="runs/h2v3")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    (out / "models").mkdir(parents=True, exist_ok=True)
    manifest = {
        "prereg": str(PREREG),
        "prereg_sha256": sha256(PREREG),
        "args": vars(args),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    rows = []
    for seed in args.seeds:
        print(f"\n=== seed {seed} ===", flush=True)
        pool = generate_episodes(
            n=args.episodes, steps=args.steps, size=96, seed=seed,
            goal_directed=0.5, nuisance_mode="mix",
        )
        train_eps = pool[: args.readout_episodes]
        temporal = build_temporal_samples(train_eps)
        good = build_view_pairs(
            train_eps, per_episode=args.per_episode, seed=seed,
            nuisance_fn=sample_nuisance_mix,
        )
        ood = generate_episodes(
            n=args.ood_episodes, steps=args.steps, size=96,
            seed=7000 + seed, goal_directed=0.5, nuisance_mode="shift",
        )
        id_ = generate_episodes(
            n=args.id_episodes, steps=args.steps, size=96,
            seed=8000 + seed, goal_directed=0.5, nuisance_mode="normal",
        )
        random_ood = evaluate_random(
            ood, n_episodes=len(ood), steps=args.steps, seed=seed
        )
        random_id = evaluate_random(
            id_, n_episodes=len(id_), steps=args.steps, seed=seed
        )

        for arm, views, lam in (
            ("temporal", None, 0.0),
            ("good", good, args.lambda_view),
        ):
            path = out / "models" / f"s{seed}-{arm}.safetensors"
            if path.exists():
                model = load_jepa(path, image_size=96)
            else:
                model = train_jepa(
                    temporal, epochs=args.epochs, batch_size=args.batch_size,
                    seed=seed, image_size=96, view_pairs=views,
                    lambda_view=lam, save_path=path, verbose=False,
                )
            X, Y = build_readout_data(
                model, train_eps, n_states=args.states_per_episode, seed=seed
            )
            mu, sd, W = fit_ridge(X, Y)
            err = readout_error(model, mu, sd, W, ood, seed=seed)
            ood_ctrl = state_feedback(
                model, mu, sd, W, ood, k=args.k, damp=args.damp,
                steps=args.steps,
            )
            id_ctrl = state_feedback(
                model, mu, sd, W, id_, k=args.k, damp=args.damp,
                steps=args.steps,
            )
            row = {
                "seed": seed, "arm": arm,
                "ood_success": ood_ctrl["success"],
                "ood_dist": ood_ctrl["mean_dist"],
                "id_success": id_ctrl["success"],
                "id_dist": id_ctrl["mean_dist"],
                "ood_pos_r2": err["pos_r2"],
                "random_ood_success": random_ood["success"],
                "random_ood_dist": random_ood["mean_dist"],
                "random_id_success": random_id["success"],
                "random_id_dist": random_id["mean_dist"],
            }
            rows.append(row)
            print(
                f"  {arm}: OOD {row['ood_success']:.2f}/{row['ood_dist']:.3f} "
                f"ID {row['id_success']:.2f}/{row['id_dist']:.3f} "
                f"readout R2 {row['ood_pos_r2']:.3f}",
                flush=True,
            )

    (out / "per_seed.json").write_text(json.dumps(rows, indent=2) + "\n")
    by = {(r["seed"], r["arm"]): r for r in rows}
    per = []
    for s in sorted({r["seed"] for r in rows}):
        t = by.get((s, "temporal"))
        g = by.get((s, "good"))
        if not t or not g:
            continue
        per.append({
            "seed": s,
            "ood_temporal_dist": t["ood_dist"],
            "ood_good_dist": g["ood_dist"],
            "ood_diff": g["ood_dist"] - t["ood_dist"],
            "ood_temporal_success": t["ood_success"],
            "ood_good_success": g["ood_success"],
            "id_temporal_dist": t["id_dist"],
            "id_good_dist": g["id_dist"],
            "id_diff": g["id_dist"] - t["id_dist"],
            "ood_r2_temporal": t["ood_pos_r2"],
            "ood_r2_good": g["ood_pos_r2"],
            "random_ood_success": t["random_ood_success"],
            "random_ood_dist": t["random_ood_dist"],
        })

    if not per:
        print("no complete pairs")
        return
    diffs = np.array([p["ood_diff"] for p in per])
    rng = np.random.default_rng(0)
    boot = np.array([
        np.mean(rng.choice(diffs, size=len(diffs), replace=True))
        for _ in range(10_000)
    ])
    ci = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
    mean_diff = float(diffs.mean())
    count_lower = int((diffs < 0).sum())
    mean_id_diff = float(np.mean([p["id_diff"] for p in per]))
    mean_r2_t = float(np.mean([p["ood_r2_temporal"] for p in per]))
    mean_r2_g = float(np.mean([p["ood_r2_good"] for p in per]))
    complete = len(per) == 5
    passed = (
        complete and mean_diff <= -0.05 and count_lower >= 4
        and mean_id_diff <= 0.05
    )
    report = {
        "prereg_sha256": sha256(PREREG),
        "complete": complete,
        "seeds": per,
        "mean_ood_diff": mean_diff,
        "ood_diff_ci95": ci,
        "seeds_good_lower": count_lower,
        "mean_id_diff": mean_id_diff,
        "ood_r2_temporal": mean_r2_t,
        "ood_r2_good": mean_r2_g,
        "verdict": "PASS" if passed else "FAIL",
    }
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    lines = [
        "# H2-v3 domain-randomized encoders — result",
        "",
        f"- pre-registration: `{report['prereg_sha256']}`",
        f"- verdict: **{report['verdict']}**",
        f"- complete: {complete}",
        "",
        "| seed | OOD temp dist | OOD good dist | diff | OOD temp succ | OOD good succ | ID diff | readout R² temp | readout R² good |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for p in per:
        lines.append(
            f"| {p['seed']} | {p['ood_temporal_dist']:.3f} | "
            f"{p['ood_good_dist']:.3f} | {p['ood_diff']:+.3f} | "
            f"{p['ood_temporal_success']:.2f} | {p['ood_good_success']:.2f} | "
            f"{p['id_diff']:+.3f} | {p['ood_r2_temporal']:.3f} | "
            f"{p['ood_r2_good']:.3f} |"
        )
    lines += [
        "",
        f"- mean OOD diff: **{mean_diff:+.3f}** (95% CI {ci[0]:+.3f} to {ci[1]:+.3f})",
        f"- seeds good lower: **{count_lower}/{len(per)}**",
        f"- mean ID diff: **{mean_id_diff:+.3f}**",
        f"- OOD readout R²: temporal **{mean_r2_t:.3f}**, good **{mean_r2_g:.3f}**",
        "",
        "Endpoint: mean OOD diff ≤ −0.05, 4/5 seeds lower, ID diff ≤ +0.05.",
        "",
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[-7:]))
    print(f"wrote {out / 'report.md'}")


if __name__ == "__main__":
    main()
