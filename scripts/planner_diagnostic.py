#!/usr/bin/env python3
"""Planner diagnostic: oracle CEM with true dynamics + state-head OOD error.

Uses the exact H2 OOD episodes (seeds 1000 + seed, 10 episodes, shifted
nuisance) so the oracle is directly comparable to the saved H2 planner results.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from mlx.core import array as mx_array, eval as mx_eval

from leparallax.data import build_probe_samples, collate_probe
from leparallax.train import load_jepa
from leparallax.world import generate_episodes, step_state


def true_rollout(states: np.ndarray, actions: np.ndarray) -> np.ndarray:
    """Vectorized version of step_state. states (P,4), actions (P,H,2)."""
    x = states[:, 0].copy()
    y = states[:, 1].copy()
    vx = states[:, 2].copy()
    vy = states[:, 3].copy()
    for h in range(actions.shape[1]):
        ax = np.clip(actions[:, h, 0], -1.0, 1.0)
        ay = np.clip(actions[:, h, 1], -1.0, 1.0)
        vx = 0.85 * vx + 0.20 * ax
        vy = 0.85 * vy + 0.20 * ay
        x = x + vx
        y = y + vy
        hit_x = (x < 0.05) | (x > 0.95)
        hit_y = (y < 0.05) | (y > 0.95)
        x = np.clip(x, 0.05, 0.95)
        y = np.clip(y, 0.05, 0.95)
        vx = np.where(hit_x, 0.0, vx)
        vy = np.where(hit_y, 0.0, vy)
    return np.stack([x, y, vx, vy], axis=1)


def oracle_plan(
    state: np.ndarray,
    goal: np.ndarray,
    horizon: int = 3,
    pop: int = 128,
    elite: int = 16,
    iters: int = 5,
    seed: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mean = np.zeros((horizon, 2), dtype=np.float32)
    std = np.ones((horizon, 2), dtype=np.float32)
    for _ in range(iters):
        acts = mean + std * rng.normal(size=(pop, horizon, 2))
        acts = np.clip(acts, -1.0, 1.0).astype(np.float32)
        final = true_rollout(np.tile(state, (pop, 1)), acts)
        cost = ((final[:, :2] - goal) ** 2).sum(axis=1)
        order = np.argsort(cost)[:elite]
        mean = acts[order].mean(axis=0)
        std = acts[order].std(axis=0) + 0.05
    return mean


def evaluate_oracle(
    episodes, steps: int = 32, horizon: int = 3,
    pop: int = 128, elite: int = 16, iters: int = 5, seed: int = 0,
) -> dict:
    successes, distances = 0, []
    for i, ep in enumerate(episodes):
        s = ep.states[0].copy()
        for t in range(steps):
            plan = oracle_plan(
                s, ep.goal, horizon=horizon, pop=pop, elite=elite,
                iters=iters, seed=seed + t,
            )
            s = step_state(s, plan[0])
            if np.linalg.norm(s[:2] - ep.goal) < 0.08:
                break
        d = float(np.linalg.norm(s[:2] - ep.goal))
        distances.append(d)
        successes += d < 0.08
    return {
        "success": successes / max(len(episodes), 1),
        "mean_dist": float(np.mean(distances)),
    }


def state_head_error(model, episodes, per_episode: int = 16, seed: int = 0) -> dict:
    samples = build_probe_samples(episodes, per_episode=per_episode, seed=seed)
    obs, state, _ = collate_probe(samples, np.arange(len(samples)))
    z = model.encode(mx_array(obs))
    pred = model.state_head(z)
    mx_eval(pred)
    pred = np.array(pred)
    pos_mse = float(np.mean((pred[:, :2] - state[:, :2]) ** 2))
    vel_mse = float(np.mean((pred[:, 2:] - state[:, 2:]) ** 2))
    ss_res = ((state[:, :2] - pred[:, :2]) ** 2).sum(axis=0)
    ss_tot = ((state[:, :2] - state[:, :2].mean(axis=0)) ** 2).sum(axis=0)
    pos_r2 = float(np.mean(1.0 - ss_res / (ss_tot + 1e-8)))
    return {"pos_mse": pos_mse, "vel_mse": vel_mse, "pos_r2": pos_r2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--out", default="runs/planner-diagnostic")
    ap.add_argument("--h2-rows", default="runs/h2-plan/per_seed.jsonl")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    h2 = {}
    if pathlib.Path(args.h2_rows).exists():
        for line in pathlib.Path(args.h2_rows).read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                h2[(r["seed"], r["arm"], r["split"])] = r

    rows = []
    for seed in args.seeds:
        episodes = generate_episodes(
            n=args.episodes, steps=args.steps, size=96,
            seed=1000 + seed, goal_directed=0.5, nuisance_shift=True,
        )
        oracle = evaluate_oracle(
            episodes, steps=args.steps, horizon=3, pop=128, elite=16,
            iters=5, seed=seed,
        )
        oracle_strong = evaluate_oracle(
            episodes, steps=args.steps, horizon=5, pop=256, elite=32,
            iters=10, seed=seed,
        )
        temporal = load_jepa(
            f"runs/h1-confirm/models/s{seed}-temporal.safetensors", image_size=96
        )
        good = load_jepa(
            f"runs/h1-confirm/models/s{seed}-good.safetensors", image_size=96
        )
        t_err = state_head_error(temporal, episodes, per_episode=16, seed=seed)
        g_err = state_head_error(good, episodes, per_episode=16, seed=seed)
        row = {
            "seed": seed,
            "oracle_same": oracle,
            "oracle_strong": oracle_strong,
            "state_temporal": t_err,
            "state_good": g_err,
            "h2_temporal_ood": h2.get((seed, "temporal", "ood")),
            "h2_good_ood": h2.get((seed, "good", "ood")),
            "h2_random_ood": h2.get((seed, "random", "ood")),
        }
        rows.append(row)
        print(
            f"seed {seed}: oracle {oracle['success']:.2f}/{oracle['mean_dist']:.3f} "
            f"strong {oracle_strong['success']:.2f}/{oracle_strong['mean_dist']:.3f} "
            f"stateR2 t {t_err['pos_r2']:.3f} g {g_err['pos_r2']:.3f}",
            flush=True,
        )

    (out / "per_seed.json").write_text(json.dumps(rows, indent=2) + "\n")

    def mean(key, sub):
        vals = [r[key][sub] for r in rows if r.get(key)]
        return float(np.mean(vals)) if vals else None

    oracle_s = mean("oracle_same", "success")
    oracle_d = mean("oracle_same", "mean_dist")
    strong_s = mean("oracle_strong", "success")
    temporal_s = float(np.mean([
        r["h2_temporal_ood"]["success"] for r in rows if r["h2_temporal_ood"]
    ]))
    good_s = float(np.mean([
        r["h2_good_ood"]["success"] for r in rows if r["h2_good_ood"]
    ]))
    random_s = float(np.mean([
        r["h2_random_ood"]["success"] for r in rows if r["h2_random_ood"]
    ]))
    t_r2 = float(np.mean([r["state_temporal"]["pos_r2"] for r in rows]))
    g_r2 = float(np.mean([r["state_good"]["pos_r2"] for r in rows]))

    lines = [
        "# Planner diagnostic",
        "",
        "Oracle CEM uses the true dynamics on the exact H2 OOD episodes.",
        "State-head error is measured on the same episodes.",
        "",
        "| seed | oracle succ | oracle dist | strong succ | H2 temp succ | H2 good succ | H2 random succ | state R² temp | state R² good |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['seed']} | {r['oracle_same']['success']:.2f} | "
            f"{r['oracle_same']['mean_dist']:.3f} | "
            f"{r['oracle_strong']['success']:.2f} | "
            f"{r['h2_temporal_ood']['success']:.2f} | "
            f"{r['h2_good_ood']['success']:.2f} | "
            f"{r['h2_random_ood']['success']:.2f} | "
            f"{r['state_temporal']['pos_r2']:.3f} | "
            f"{r['state_good']['pos_r2']:.3f} |"
        )
    lines += [
        "",
        f"- Oracle (same planner budget): success **{oracle_s:.2f}**, distance **{oracle_d:.3f}**",
        f"- Oracle (strong: H=5, pop 256, 10 iters): success **{strong_s:.2f}**",
        f"- H2 learned: temporal **{temporal_s:.2f}**, good **{good_s:.2f}**, random **{random_s:.2f}**",
        f"- State-head OOD position R²: temporal **{t_r2:.3f}**, good **{g_r2:.3f}**",
        "",
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[-6:]))
    print(f"wrote {out / 'report.md'}")


if __name__ == "__main__":
    main()
