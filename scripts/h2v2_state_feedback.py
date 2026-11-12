#!/usr/bin/env python3
"""H2-v2: state-feedback control with a nuisance-robust ridge readout."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

import numpy as np
import mlx.core as mx

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from leparallax.data import (
    build_probe_samples,
    collate_probe,
    history_obs,
)
from leparallax.plan import evaluate_random
from leparallax.train import load_jepa
from leparallax.world import (
    generate_episodes,
    render_state,
    sample_nuisance,
    sample_nuisance_shift,
    step_state,
)

PREREG = pathlib.Path(__file__).resolve().parents[1] / "docs" / "PREREG-H2v2.md"


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def encode_batches(model, obs: np.ndarray, batch: int = 64) -> np.ndarray:
    out = []
    for start in range(0, len(obs), batch):
        x = obs[start : start + batch].astype(np.float32) / 255.0
        z = model.encode(mx.array(x))
        mx.eval(z)
        out.append(np.array(z))
    return np.concatenate(out)


def build_readout_data(model, episodes, n_states: int = 16, seed: int = 0):
    rng = np.random.default_rng(seed)
    obs, states = [], []
    for ep in episodes:
        n = min(n_states, len(ep.states))
        for t in rng.choice(len(ep.states), size=n, replace=False):
            t = int(t)
            views = [
                ep.nuisance,
                sample_nuisance(rng),
                sample_nuisance_shift(rng),
                sample_nuisance_shift(rng),
            ]
            for nu in views:
                obs.append(history_obs(ep, t, nu))
                states.append(ep.states[t])
    X = encode_batches(model, np.stack(obs))
    return X, np.stack(states)


def fit_ridge(X: np.ndarray, Y: np.ndarray, alpha: float = 1.0):
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True) + 1e-6
    Xs = (X - mu) / sd
    Xa = np.concatenate([Xs, np.ones((len(Xs), 1), dtype=Xs.dtype)], axis=1)
    d = Xa.shape[1]
    W = np.linalg.solve(Xa.T @ Xa + alpha * np.eye(d), Xa.T @ Y)
    return mu, sd, W


def predict_ridge(X, mu, sd, W):
    Xs = (X - mu) / sd
    Xa = np.concatenate([Xs, np.ones((len(Xs), 1), dtype=Xs.dtype)], axis=1)
    return Xa @ W


def readout_error(model, mu, sd, W, episodes, per_episode: int = 16, seed: int = 0):
    samples = build_probe_samples(episodes, per_episode=per_episode, seed=seed)
    obs, state, _ = collate_probe(samples, np.arange(len(samples)))
    X = encode_batches(model, obs)
    pred = predict_ridge(X, mu, sd, W)
    pos_mse = float(np.mean((pred[:, :2] - state[:, :2]) ** 2))
    vel_mse = float(np.mean((pred[:, 2:] - state[:, 2:]) ** 2))
    ss_res = ((state[:, :2] - pred[:, :2]) ** 2).sum(axis=0)
    ss_tot = ((state[:, :2] - state[:, :2].mean(axis=0)) ** 2).sum(axis=0)
    pos_r2 = float(np.mean(1.0 - ss_res / (ss_tot + 1e-8)))
    return {"pos_mse": pos_mse, "vel_mse": vel_mse, "pos_r2": pos_r2}


def state_feedback(
    model, mu, sd, W, episodes, k: float = 2.0, damp: float = 1.0,
    steps: int = 32,
):
    successes, distances = 0, []
    for ep in episodes:
        s = ep.states[0].copy()
        frame = render_state(s, ep.goal, ep.nuisance)
        hist = [frame, frame, frame]
        for t in range(steps):
            obs = np.stack(hist[-3:])
            X = encode_batches(model, obs[None])
            pred = predict_ridge(X, mu, sd, W)[0]
            a = np.clip(
                k * (ep.goal - pred[:2]) - damp * pred[2:], -1.0, 1.0
            )
            s = step_state(s, a)
            hist.append(render_state(s, ep.goal, ep.nuisance))
            if np.linalg.norm(s[:2] - ep.goal) < 0.08:
                break
        d = float(np.linalg.norm(s[:2] - ep.goal))
        distances.append(d)
        successes += d < 0.08
    return {
        "success": successes / max(len(episodes), 1),
        "mean_dist": float(np.mean(distances)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--readout-episodes", type=int, default=80)
    ap.add_argument("--states-per-episode", type=int, default=16)
    ap.add_argument("--ood-episodes", type=int, default=10)
    ap.add_argument("--id-episodes", type=int, default=5)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--k", type=float, default=2.0)
    ap.add_argument("--damp", type=float, default=1.0)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--out", default="runs/h2v2")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in args.seeds:
        print(f"\n=== seed {seed} ===", flush=True)
        pool = generate_episodes(
            n=100, steps=args.steps, size=96, seed=seed, goal_directed=0.5
        )
        readout_eps = pool[: args.readout_episodes]
        ood = generate_episodes(
            n=args.ood_episodes, steps=args.steps, size=96,
            seed=3000 + seed, goal_directed=0.5, nuisance_shift=True,
        )
        id_ = generate_episodes(
            n=args.id_episodes, steps=args.steps, size=96,
            seed=4000 + seed, goal_directed=0.5, nuisance_shift=False,
        )
        random_ood = evaluate_random(
            ood, n_episodes=len(ood), steps=args.steps, seed=seed
        )
        random_id = evaluate_random(
            id_, n_episodes=len(id_), steps=args.steps, seed=seed
        )
        for arm in ("temporal", "good"):
            model = load_jepa(
                f"runs/h1-confirm/models/s{seed}-{arm}.safetensors",
                image_size=96,
            )
            X, Y = build_readout_data(
                model, readout_eps, n_states=args.states_per_episode, seed=seed
            )
            mu, sd, W = fit_ridge(X, Y, alpha=args.alpha)
            ood_err = readout_error(
                model, mu, sd, W, ood, per_episode=16, seed=seed
            )
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
                "ood_pos_r2": ood_err["pos_r2"],
                "ood_pos_mse": ood_err["pos_mse"],
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
        "# H2-v2 state-feedback control — result",
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
