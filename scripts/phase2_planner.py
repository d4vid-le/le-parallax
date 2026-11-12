#!/usr/bin/env python3
"""Phase 2 planner test: can a latent shooting planner identify unreachable goals?

Uses the frozen Phase 2 encoders after the one-way door has closed. The left goal
is reachable from the left side and unreachable from the right side. The planner
scores random action sequences plus a model-based controller by the predicted
state-head distance to the goal. A calibrated threshold decides when to abort.

This is a representation/planner diagnostic, not a closed-loop MPC in the true
environment.
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
from leparallax.world import generate_episodes, history_indices


def encode(model, obs: np.ndarray) -> mx.array:
    z = model.encode(mx.array(obs[None].astype(np.float32) / 255.0))
    mx.eval(z)
    return z


def planner_distance(model, start_obs, goal_xy, n_samples=64,
                     horizon=6, seed=0):
    """Best predicted final distance over random shots + model-based controller."""
    rng = np.random.default_rng(seed)
    z0 = encode(model, start_obs)
    z = mx.broadcast_to(z0, (n_samples, z0.shape[-1]))
    actions = mx.array(
        rng.uniform(-1.0, 1.0, size=(horizon, n_samples, 2)).astype(np.float32)
    )
    for t in range(horizon):
        z = model.predict(z, actions[t])
    s = model.state_head(z)
    mx.eval(s)
    pos = np.array(s)[:, :2]
    dists = np.linalg.norm(pos - goal_xy[None], axis=1)

    # Model-based controller candidate: uses the learned state readout only.
    zc = z0
    for _ in range(horizon):
        sc = model.state_head(zc)
        mx.eval(sc)
        sc = np.array(sc)[0]
        a = np.clip(2.0 * (goal_xy - sc[:2]) - 1.0 * sc[2:], -1.0, 1.0)
        zc = model.predict(zc, mx.array(a[None].astype(np.float32)))
    sc = model.state_head(zc)
    mx.eval(sc)
    ctrl_dist = float(np.linalg.norm(np.array(sc)[0, :2] - goal_xy))
    return float(dists.min()), ctrl_dist, float(min(dists.min(), ctrl_dist))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/phase2-pilot")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--arms", nargs="+", default=[
        "temp_matched", "sym_head", "asym_clean", "asym_vol_bounded"
    ])
    ap.add_argument("--test-episodes", type=int, default=20)
    ap.add_argument("--n-samples", type=int, default=64)
    ap.add_argument("--horizon", type=int, default=6)
    ap.add_argument("--frame-state-head", action="store_true")
    ap.add_argument("--latent-dim", type=int, default=64)
    ap.add_argument("--width", type=float, default=1.0)
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    rows = []
    for seed in args.seeds:
        test_eps = generate_episodes(
            n=args.test_episodes, steps=32, size=96, seed=9000 + seed,
            goal_directed=0.5, nuisance_mode="normal", one_way_door=True,
            occlude_door_right=True,
        )
        left_goal = [ep for ep in test_eps if ep.goal[0] < 0.5]
        print(f"seed {seed}: {len(left_goal)} left-goal episodes", flush=True)
        for arm in args.arms:
            path = out / "models" / f"s{seed}-{arm}.safetensors"
            if not path.exists():
                print("  missing", path, flush=True)
                continue
            model = load_jepa(
                str(path), image_size=96,
                frame_state_head=args.frame_state_head,
                latent_dim=args.latent_dim,
                width=args.width,
            )
            reachable, unreachable = [], []
            for ep in left_goal:
                sides = ep.states[:, 0] >= 0.5
                crossings = np.where((~sides[:-1]) & sides[1:])[0]
                if len(crossings) == 0:
                    continue
                tc = int(crossings[0])
                g_xy = ep.goal.astype(np.float64)
                for pre in (2, 4):
                    ta = max(0, tc - pre)
                    if not ep.door_open[ta] or ep.states[ta, 0] >= 0.5:
                        continue
                    obs = ep.frames[history_indices(ta)]
                    _, _, best = planner_distance(
                        model, obs, g_xy, n_samples=args.n_samples,
                        horizon=args.horizon, seed=seed + ta,
                    )
                    reachable.append(best)
                for post in (3, 5):
                    tb = min(len(ep.states) - 1, tc + post)
                    if ep.door_open[tb] or ep.states[tb, 0] < 0.5:
                        continue
                    obs = ep.frames[history_indices(tb)]
                    _, _, best = planner_distance(
                        model, obs, g_xy, n_samples=args.n_samples,
                        horizon=args.horizon, seed=seed + tb,
                    )
                    unreachable.append(best)
            if not reachable or not unreachable:
                print(f"  {arm}: not enough starts", flush=True)
                continue
            threshold = float(np.percentile(reachable, 90))
            abort_unreach = float(np.mean(np.array(unreachable) > threshold))
            false_abort = float(np.mean(np.array(reachable) > threshold))
            # AUC of the planner distance for unreachable vs reachable.
            pos = np.array(unreachable)
            neg = np.array(reachable)
            x = np.concatenate([pos, neg])
            order = np.argsort(x, kind="mergesort")
            ranks = np.empty_like(order, dtype=np.float64)
            ranks[order] = np.arange(1, len(x) + 1)
            n_pos, n_neg = len(pos), len(neg)
            auc = (ranks[:n_pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
            row = {
                "seed": seed, "arm": arm,
                "reachable_n": len(reachable),
                "unreachable_n": len(unreachable),
                "reachable_dist": float(np.mean(reachable)),
                "unreachable_dist": float(np.mean(unreachable)),
                "threshold": threshold,
                "abort_unreachable": abort_unreach,
                "false_abort_reachable": false_abort,
                "planner_auc": float(auc),
            }
            rows.append(row)
            print(
                f"  {arm}: reach {row['reachable_dist']:.3f} "
                f"unreach {row['unreachable_dist']:.3f}  "
                f"abort {abort_unreach:.2f}  false {false_abort:.2f}  "
                f"AUC {auc:.3f}",
                flush=True,
            )

    (out / "planner.json").write_text(json.dumps(rows, indent=2) + "\n")
    lines = [
        "# Phase 2 planner diagnostic",
        "",
        "| arm | reach dist | unreach dist | abort unreach | false abort | AUC |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['arm']} | {r['reachable_dist']:.3f} | "
            f"{r['unreachable_dist']:.3f} | {r['abort_unreachable']:.2f} | "
            f"{r['false_abort_reachable']:.2f} | {r['planner_auc']:.3f} |"
        )
    lines += ["", "Per-seed values are in `planner.json`."]
    (out / "planner.md").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    print(f"wrote {out / 'planner.md'}")


if __name__ == "__main__":
    main()
