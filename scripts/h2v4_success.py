#!/usr/bin/env python3
"""H2-v4: success-primary replication on fresh OOD episodes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from leparallax.readout import (
    build_readout_data,
    fit_ridge,
    predict_ridge,
    encode_batches,
)
from leparallax.train import load_jepa
from leparallax.world import generate_episodes, render_state, step_state

PREREG = pathlib.Path(__file__).resolve().parents[1] / "docs" / "PREREG-H2v4.md"


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def control_details(model, mu, sd, W, episodes, k=2.0, damp=1.0, steps=32):
    successes, distances = [], []
    for ep in episodes:
        s = ep.states[0].copy()
        frame = render_state(s, ep.goal, ep.nuisance)
        hist = [frame, frame, frame]
        for _ in range(steps):
            obs = np.stack(hist[-3:])
            X = encode_batches(model, obs[None])
            pred = predict_ridge(X, mu, sd, W)[0]
            a = np.clip(k * (ep.goal - pred[:2]) - damp * pred[2:], -1.0, 1.0)
            s = step_state(s, a)
            hist.append(render_state(s, ep.goal, ep.nuisance))
            if np.linalg.norm(s[:2] - ep.goal) < 0.08:
                break
        d = float(np.linalg.norm(s[:2] - ep.goal))
        distances.append(d)
        successes.append(d < 0.08)
    return successes, distances


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / (2.0**n)
    return min(1.0, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--readout-episodes", type=int, default=80)
    ap.add_argument("--states-per-episode", type=int, default=16)
    ap.add_argument("--ood-episodes", type=int, default=20)
    ap.add_argument("--id-episodes", type=int, default=5)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--k", type=float, default=2.0)
    ap.add_argument("--damp", type=float, default=1.0)
    ap.add_argument("--out", default="runs/h2v4")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "prereg": str(PREREG),
        "prereg_sha256": sha256(PREREG),
        "args": vars(args),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    per = []
    for seed in args.seeds:
        print(f"\n=== seed {seed} ===", flush=True)
        pool = generate_episodes(
            n=100, steps=args.steps, size=96, seed=seed,
            goal_directed=0.5, nuisance_mode="mix",
        )
        readout_eps = pool[: args.readout_episodes]
        ood = generate_episodes(
            n=args.ood_episodes, steps=args.steps, size=96,
            seed=9000 + seed, goal_directed=0.5, nuisance_mode="shift",
        )
        id_ = generate_episodes(
            n=args.id_episodes, steps=args.steps, size=96,
            seed=10000 + seed, goal_directed=0.5, nuisance_mode="normal",
        )
        results = {}
        for arm in ("temporal", "good"):
            model = load_jepa(
                f"runs/h2v3/models/s{seed}-{arm}.safetensors", image_size=96
            )
            X, Y = build_readout_data(
                model, readout_eps, n_states=args.states_per_episode, seed=seed
            )
            mu, sd, W = fit_ridge(X, Y)
            ood_s, ood_d = control_details(
                model, mu, sd, W, ood, k=args.k, damp=args.damp,
                steps=args.steps,
            )
            id_s, id_d = control_details(
                model, mu, sd, W, id_, k=args.k, damp=args.damp,
                steps=args.steps,
            )
            results[arm] = {
                "ood_success": ood_s,
                "ood_dist": ood_d,
                "id_success": id_s,
                "id_dist": id_d,
            }
            print(
                f"  {arm}: OOD succ {np.mean(ood_s):.2f} dist "
                f"{np.mean(ood_d):.3f} | ID succ {np.mean(id_s):.2f} dist "
                f"{np.mean(id_d):.3f}",
                flush=True,
            )

        t = results["temporal"]
        g = results["good"]
        b = int(sum(gs and not ts for gs, ts in zip(g["ood_success"], t["ood_success"])))
        c = int(sum(ts and not gs for gs, ts in zip(g["ood_success"], t["ood_success"])))
        row = {
            "seed": seed,
            "ood_temporal_success": float(np.mean(t["ood_success"])),
            "ood_good_success": float(np.mean(g["ood_success"])),
            "ood_success_delta": float(np.mean(g["ood_success"]) - np.mean(t["ood_success"])),
            "ood_temporal_dist": float(np.mean(t["ood_dist"])),
            "ood_good_dist": float(np.mean(g["ood_dist"])),
            "ood_dist_diff": float(np.mean(g["ood_dist"]) - np.mean(t["ood_dist"])),
            "id_temporal_success": float(np.mean(t["id_success"])),
            "id_good_success": float(np.mean(g["id_success"])),
            "id_temporal_dist": float(np.mean(t["id_dist"])),
            "id_good_dist": float(np.mean(g["id_dist"])),
            "id_dist_diff": float(np.mean(g["id_dist"]) - np.mean(t["id_dist"])),
            "good_only": b,
            "temporal_only": c,
            "mcnemar_p": mcnemar_exact(b, c),
        }
        per.append(row)
        print(
            f"  success delta {row['ood_success_delta']:+.2f}; "
            f"McNemar b={b} c={c} p={row['mcnemar_p']:.4f}",
            flush=True,
        )

    (out / "per_seed.json").write_text(json.dumps(per, indent=2) + "\n")
    deltas = np.array([p["ood_success_delta"] for p in per])
    rng = np.random.default_rng(0)
    boot = np.array([
        np.mean(rng.choice(deltas, size=len(deltas), replace=True))
        for _ in range(10_000)
    ])
    ci = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
    mean_delta = float(deltas.mean())
    count_pos = int((deltas > 0).sum())
    total_b = sum(p["good_only"] for p in per)
    total_c = sum(p["temporal_only"] for p in per)
    p_mcnemar = mcnemar_exact(total_b, total_c)
    mean_id_diff = float(np.mean([p["id_dist_diff"] for p in per]))
    mean_id_succ_delta = float(np.mean([
        p["id_good_success"] - p["id_temporal_success"] for p in per
    ]))
    mean_dist_diff = float(np.mean([p["ood_dist_diff"] for p in per]))
    complete = len(per) == 5
    passed = (
        complete and mean_delta >= 0.20 and count_pos >= 4
        and p_mcnemar < 0.05
        and mean_id_diff <= 0.05
        and mean_id_succ_delta >= -0.10
    )
    report = {
        "prereg_sha256": sha256(PREREG),
        "complete": complete,
        "seeds": per,
        "mean_success_delta": mean_delta,
        "success_delta_ci95": ci,
        "seeds_positive": count_pos,
        "good_only": total_b,
        "temporal_only": total_c,
        "mcnemar_p": p_mcnemar,
        "mean_ood_dist_diff": mean_dist_diff,
        "mean_id_dist_diff": mean_id_diff,
        "mean_id_success_delta": mean_id_succ_delta,
        "verdict": "PASS" if passed else "FAIL",
    }
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    lines = [
        "# H2-v4 success-primary — result",
        "",
        f"- pre-registration: `{report['prereg_sha256']}`",
        f"- verdict: **{report['verdict']}**",
        f"- complete: {complete}",
        "",
        "| seed | OOD temp succ | OOD good succ | delta | OOD temp dist | OOD good dist | McNemar b/c | p |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for p in per:
        lines.append(
            f"| {p['seed']} | {p['ood_temporal_success']:.2f} | "
            f"{p['ood_good_success']:.2f} | {p['ood_success_delta']:+.2f} | "
            f"{p['ood_temporal_dist']:.3f} | {p['ood_good_dist']:.3f} | "
            f"{p['good_only']}/{p['temporal_only']} | {p['mcnemar_p']:.4f} |"
        )
    lines += [
        "",
        f"- mean success delta: **{mean_delta:+.2f}** "
        f"(95% CI {ci[0]:+.2f} to {ci[1]:+.2f})",
        f"- seeds positive: **{count_pos}/{len(per)}**",
        f"- pooled McNemar: good-only **{total_b}**, temporal-only **{total_c}**, "
        f"**p = {p_mcnemar:.4f}**",
        f"- mean OOD distance diff: **{mean_dist_diff:+.3f}**",
        f"- mean ID distance diff: **{mean_id_diff:+.3f}**, "
        f"ID success delta **{mean_id_succ_delta:+.2f}**",
        "",
        "Endpoint: success delta ≥ +20 points, 4/5 seeds positive, McNemar p < 0.05, "
        "ID no regression.",
        "",
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[-8:]))
    print(f"wrote {out / 'report.md'}")


if __name__ == "__main__":
    main()
