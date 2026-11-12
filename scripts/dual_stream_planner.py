#!/usr/bin/env python3
"""Dual-stream planner: state stream for physical reachability, JEPA stream for
irreversibility. Compares state-only, state+sym_head, and state+asym_clean.

The state stream is the frozen 47k oracle CNN. The JEPA streams are trained with
no state loss. The planner scores candidate action sequences with the state
dynamics and the JEPA latent distance to the goal, then aborts when the combined
score exceeds a threshold calibrated on reachable starts.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from oracle_state_cnn import OracleCNN
from leparallax.data import goal_obs
from leparallax.train import load_jepa
from leparallax.world import generate_episodes, history_indices


class StateDynamics(nn.Module):
    def __init__(self, hidden: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(6, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 4)

    def __call__(self, s: mx.array, a: mx.array) -> mx.array:
        x = mx.concatenate([s, a], axis=-1)
        x = nn.relu(self.fc1(x))
        x = nn.relu(self.fc2(x))
        return s + self.fc3(x)


def build_transitions(episodes):
    S, A, N = [], [], []
    for ep in episodes:
        for t in range(len(ep.actions)):
            S.append(ep.states[t])
            A.append(ep.actions[t])
            N.append(ep.states[t + 1])
    return (
        np.stack(S).astype(np.float32),
        np.stack(A).astype(np.float32),
        np.stack(N).astype(np.float32),
    )


def train_dynamics(episodes, seed: int = 0, epochs: int = 300,
                   batch: int = 64, lr: float = 1e-3):
    S, A, N = build_transitions(episodes)
    mx.random.seed(seed)
    model = StateDynamics()
    rng = np.random.default_rng(seed)

    def loss_fn(model, s, a, n):
        return mx.mean((model(s, a) - n) ** 2)

    lg = nn.value_and_grad(model, loss_fn)
    opt = optim.AdamW(learning_rate=lr, weight_decay=1e-4)
    for epoch in range(1, epochs + 1):
        lr_t = lr * 0.5 * (
            1.0 + math.cos(math.pi * (epoch - 1) / max(epochs - 1, 1))
        )
        opt.learning_rate = lr_t
        p = rng.permutation(len(S))
        for start in range(0, len(S), batch):
            idx = p[start:start + batch]
            loss, grads = lg(
                model, mx.array(S[idx]), mx.array(A[idx]), mx.array(N[idx])
            )
            opt.update(model, grads)
            mx.eval(model.parameters(), opt.state)
    return model


def encode_jepa(model, obs):
    z = model.encode(mx.array(obs[None].astype(np.float32) / 255.0))
    mx.eval(z)
    return z


def state_from_stream(state_model, obs):
    _, p4 = state_model(mx.array(obs[None].astype(np.float32) / 255.0))
    mx.eval(p4)
    return np.array(p4)[0]


def state_rollout_batch(dyn, s0, actions):
    # s0: (4,), actions: (N, H, 2) -> (N, 4)
    s = mx.broadcast_to(mx.array(s0[None]), (actions.shape[0], 4))
    for t in range(actions.shape[1]):
        s = dyn(s, mx.array(actions[:, t]))
    mx.eval(s)
    return np.array(s)


def jepa_rollout_batch(model, z0, actions):
    z = mx.broadcast_to(z0, (actions.shape[0], z0.shape[-1]))
    for t in range(actions.shape[1]):
        z = model.predict(z, mx.array(actions[:, t]))
    mx.eval(z)
    return np.array(z)


def candidates(dyn, s0, goal, n=64, horizon=6, seed=0):
    rng = np.random.default_rng(seed)
    acts = rng.uniform(-1.0, 1.0, size=(n - 1, horizon, 2)).astype(np.float32)
    # Model-based controller candidate.
    s = s0.copy()
    ctrl = []
    for _ in range(horizon):
        a = np.clip(2.0 * (goal - s[:2]) - 1.0 * s[2:], -1.0, 1.0)
        ctrl.append(a.astype(np.float32))
        sh = dyn(
            mx.array(s[None].astype(np.float32)),
            mx.array(a[None].astype(np.float32)),
        )
        mx.eval(sh)
        s = np.array(sh)[0]
    return np.concatenate([acts, np.stack(ctrl)[None]], axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default="runs/dual-stream-state/state_stream_s0.safetensors")
    ap.add_argument("--jepa-dir", default="runs/dual-stream-jepa")
    ap.add_argument("--arms", nargs="+", default=["sym_head", "asym_clean"])
    ap.add_argument("--test-episodes", type=int, default=20)
    ap.add_argument("--horizon", type=int, default=6)
    ap.add_argument("--n-samples", type=int, default=64)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--out", default="runs/dual-stream-planner")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    train_eps = generate_episodes(
        n=40, steps=32, size=96, seed=0, goal_directed=0.5,
        nuisance_mode="normal", one_way_door=True, occlude_door_right=True,
    )
    test_eps = generate_episodes(
        n=args.test_episodes, steps=32, size=96, seed=9000,
        goal_directed=0.5, nuisance_mode="normal", one_way_door=True,
        occlude_door_right=True,
    )
    state_model = OracleCNN()
    state_model.load_weights(args.state)
    dyn = train_dynamics(train_eps)
    left_goal = [ep for ep in test_eps if ep.goal[0] < 0.5]
    print(f"{len(left_goal)} left-goal episodes", flush=True)

    # Collect starts and per-arm scores.
    starts = []
    for ep in left_goal:
        sides = ep.states[:, 0] >= 0.5
        cr = np.where((~sides[:-1]) & sides[1:])[0]
        if len(cr) == 0:
            continue
        tc = int(cr[0])
        for pre in (2, 4):
            ta = max(0, tc - pre)
            if ep.door_open[ta] and ep.states[ta, 0] < 0.5:
                starts.append((ep, ta, "reachable"))
        # Keep the crossing in the 3-frame history so the door is identifiable.
        for post in (1, 2):
            tb = min(len(ep.states) - 1, tc + post)
            if not ep.door_open[tb] and ep.states[tb, 0] >= 0.5:
                starts.append((ep, tb, "unreachable"))
    print(f"{len(starts)} starts", flush=True)

    rows = []
    for arm in args.arms:
        jepa = load_jepa(
            str(pathlib.Path(args.jepa_dir) / "models" / f"s0-{arm}.safetensors"),
            image_size=96,
        )
        per_start = []
        for ep, t, kind in starts:
            obs = ep.frames[history_indices(t)]
            s0 = state_from_stream(state_model, obs)
            z0 = encode_jepa(jepa, obs)
            gz = encode_jepa(jepa, goal_obs(ep))
            acts = candidates(
                dyn, s0, ep.goal.astype(np.float64),
                n=args.n_samples, horizon=args.horizon, seed=t,
            )
            sf = state_rollout_batch(dyn, s0, acts)
            zf = jepa_rollout_batch(jepa, z0, acts)
            sd = np.linalg.norm(sf[:, :2] - ep.goal[None], axis=1)
            jd = np.mean((zf - np.array(gz)) ** 2, axis=1)
            per_start.append({
                "kind": kind,
                "state": float(sd.min()),
                "jepa": float(jd.min()),
            })
        # Normalize by reachable starts.
        rs = [p for p in per_start if p["kind"] == "reachable"]
        us = [p for p in per_start if p["kind"] == "unreachable"]
        if not rs or not us:
            print(f"{arm}: not enough starts", flush=True)
            continue
        smu, ssd = np.mean([p["state"] for p in rs]), np.std([p["state"] for p in rs]) + 1e-6
        jmu, jsd = np.mean([p["jepa"] for p in rs]), np.std([p["jepa"] for p in rs]) + 1e-6
        for p in per_start:
            p["state_z"] = (p["state"] - smu) / ssd
            p["jepa_z"] = (p["jepa"] - jmu) / jsd
            p["combined"] = p["state_z"] + args.alpha * p["jepa_z"]
        for label, key in (
            ("state_only", "state_z"),
            (f"state+{arm}", "combined"),
        ):
            rv = np.array([p[key] for p in per_start if p["kind"] == "reachable"])
            uv = np.array([p[key] for p in per_start if p["kind"] == "unreachable"])
            th = float(np.percentile(rv, 90))
            abort = float(np.mean(uv > th))
            false = float(np.mean(rv > th))
            x = np.concatenate([uv, rv])
            order = np.argsort(x, kind="mergesort")
            ranks = np.empty_like(order, dtype=np.float64)
            ranks[order] = np.arange(1, len(x) + 1)
            n_pos, n_neg = len(uv), len(rv)
            auc = (ranks[:n_pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
            rows.append({
                "arm": arm, "scorer": label,
                "reach_n": len(rv), "unreach_n": len(uv),
                "abort_unreachable": abort,
                "false_abort_reachable": false,
                "planner_auc": float(auc),
                "reach_mean": float(rv.mean()),
                "unreach_mean": float(uv.mean()),
            })
            print(
                f"{arm}/{label}: abort {abort:.2f} false {false:.2f} "
                f"AUC {auc:.3f}",
                flush=True,
            )

    (out / "planner.json").write_text(json.dumps(rows, indent=2) + "\n")
    lines = [
        "# Dual-stream planner",
        "",
        "| arm | scorer | abort unreach | false abort | AUC |",
        "|---|---|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['arm']} | {r['scorer']} | {r['abort_unreachable']:.2f} | "
            f"{r['false_abort_reachable']:.2f} | {r['planner_auc']:.3f} |"
        )
    (out / "planner.md").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    print(f"wrote {out / 'planner.md'}")


if __name__ == "__main__":
    main()
