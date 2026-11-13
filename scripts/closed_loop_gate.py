#!/usr/bin/env python3
"""Step 3: closed-loop dual-stream planner with abort gates.

Base planner: state stream + state-feedback controller, true hidden-alcove world.
Gates: none, state-only, frozen-JEPA, belief-stream. The gate never steers; it
only aborts. Calibrate (tau, k) on validation episodes under a false-abort
ceiling, then evaluate on held-out episodes.
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

from natural_belief import BeliefGRU, StaticClf, static_features
from oracle_state_cnn import OracleCNN
from leparallax.train import load_jepa
from leparallax.world import (
    generate_episodes, render_state, step_state,
)


def auc(pos, neg):
    x = np.concatenate([pos, neg])
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(x) + 1)
    n_pos, n_neg = len(pos), len(neg)
    return float((ranks[:n_pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def train_gate(X, y, seed, epochs=50):
    rng = np.random.default_rng(seed)
    mx.random.seed(seed)
    mu, sd = X.mean(0, keepdims=True), X.std(0, keepdims=True) + 1e-6
    Xs = (X - mu) / sd
    model = StaticClf(X.shape[1])
    opt = optim.AdamW(learning_rate=1e-3, weight_decay=1e-3)

    def loss_fn(model, x, y):
        p = mx.sigmoid(model(x))[:, 0]
        return -mx.mean(y * mx.log(p + 1e-6) + (1 - y) * mx.log(1 - p + 1e-6))

    lg = nn.value_and_grad(model, loss_fn)
    for epoch in range(1, epochs + 1):
        opt.learning_rate = 1e-3 * 0.5 * (
            1 + math.cos(math.pi * (epoch - 1) / max(epochs - 1, 1))
        )
        p = rng.permutation(len(Xs))
        for s in range(0, len(Xs), 64):
            idx = p[s:s + 64]
            loss, grads = lg(model, mx.array(Xs[idx]), mx.array(y[idx]))
            opt.update(model, grads)
            mx.eval(model.parameters(), opt.state)
    return model, mu, sd


def predict_gate(model, mu, sd, x):
    z = (x[None] - mu) / sd
    p = mx.sigmoid(model(mx.array(z)))[0, 0]
    mx.eval(p)
    return float(p)


def run_episode(ep, state_model, jepa, gates, belief, max_steps=32):
    state = ep.states[0].copy()
    goal = ep.goal.astype(np.float32)
    nuisance = ep.nuisance
    door = True
    gate = True
    frames = []
    ps, pj = [], []
    success = False
    steps = 0
    gate_close_t = None
    for t in range(max_steps):
        frame = render_state(
            state, goal, nuisance, size=96, door_open=door,
            occlude_door_right=True,
        )
        frames.append(frame)
        hist = frames[max(0, t - 2):t + 1]
        while len(hist) < 3:
            hist = [frames[0]] + hist
        obs = np.stack(hist).astype(np.float32) / 255.0
        p4 = np.array(state_model(
            mx.array(obs[None])
        )[1])[0]
        z = np.array(jepa.encode(mx.array(obs[None])))[0]
        ps.append(predict_gate(
            gates["state"][0], gates["state"][1], gates["state"][2],
            np.concatenate([p4, goal]),
        ))
        pj.append(predict_gate(
            gates["jepa"][0], gates["jepa"][1], gates["jepa"][2], z,
        ))
        a = np.clip(2.0 * (goal - p4[:2]) - 1.0 * p4[2:], -1.0, 1.0)
        old_x = state[0]
        state = step_state(
            state, a.astype(np.float32), door_open=door, gate_open=gate
        )
        if (
            door
            and old_x < 0.5 <= state[0]
            and 0.35 <= state[1] <= 0.65
        ):
            door = False
            gate = False
            gate_close_t = t + 1
        steps = t + 1
        if np.linalg.norm(state[:2] - goal) < 0.08 and gate:
            success = True
            break

    frames_arr = np.stack(frames).astype(np.float32) / 255.0
    pb = np.array(mx.sigmoid(belief(mx.array(frames_arr[None])))[0])
    return {
        "success": bool(success),
        "steps": int(steps),
        "state_p": np.array(ps),
        "jepa_p": np.array(pj),
        "belief_p": np.array(pb),
        "gate_end": bool(gate),
        "gate_close_t": gate_close_t,
        "final_x": float(state[0]),
    }


def gate_eval(runs, key, tau, k):
    reach_n = reach_success = false_abort = 0
    unreach_n = unreach_abort = 0
    times, delays = [], []
    clean_n = clean_false = fail_n = fail_false = 0
    for r in runs:
        p = r[key]
        aborted = False
        at = None
        for t in range(len(p) - k + 1):
            if np.all(p[t:t + k] < tau):
                aborted = True
                at = t + k - 1
                break
        if r["gate_end"]:
            reach_n += 1
            if r["success"]:
                clean_n += 1
                clean_false += int(aborted)
            else:
                fail_n += 1
                fail_false += int(aborted)
            if aborted:
                false_abort += 1
            else:
                reach_success += int(r["success"])
        else:
            unreach_n += 1
            if aborted:
                unreach_abort += 1
                times.append(at)
                if r["gate_close_t"] is not None:
                    delays.append(at - r["gate_close_t"])
    return {
        "reach_n": reach_n,
        "reach_success": reach_success / max(reach_n, 1),
        "false_abort": false_abort / max(reach_n, 1),
        "false_abort_clean": clean_false / max(clean_n, 1),
        "false_abort_failure": fail_false / max(fail_n, 1),
        "clean_n": clean_n,
        "fail_n": fail_n,
        "unreach_n": unreach_n,
        "unreach_abort": unreach_abort / max(unreach_n, 1),
        "time_to_abort": float(np.mean(times)) if times else float("nan"),
        "abort_delay_after_close": float(np.mean(delays)) if delays else float("nan"),
        "step_savings": 32.0 - float(np.mean(times)) if times else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jepa", default="runs/dual-stream-jepa/models/s0-asym_clean.safetensors")
    ap.add_argument("--state", default="runs/dual-stream-state/state_stream_s0.safetensors")
    ap.add_argument("--belief", default="runs/planner-belief-hard/belief_s0.safetensors")
    ap.add_argument("--cal-episodes", type=int, default=160)
    ap.add_argument("--test-episodes", type=int, default=240)
    ap.add_argument("--cal-seed", type=int, default=7000)
    ap.add_argument("--test-seed", type=int, default=12000)
    ap.add_argument("--out", default="runs/closed-loop-gate")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    state_model = OracleCNN()
    state_model.load_weights(args.state)
    jepa = load_jepa(args.jepa, image_size=96)

    train_eps = generate_episodes(
        n=80, steps=32, size=96, seed=0, goal_directed=0.5,
        nuisance_mode="normal", hidden_alcove=True, occlude_door_right=True,
    )
    sf, jf, ysf, side, goal = static_features(train_eps, state_model, jepa)
    gate_state = train_gate(sf, ysf, 0)
    gate_jepa = train_gate(jf, ysf, 0)
    gates = {"state": gate_state, "jepa": gate_jepa}

    belief_jepa = load_jepa(args.jepa, image_size=96)
    belief = BeliefGRU(belief_jepa.encoder.frame)
    belief.load_weights(args.belief)

    cal_eps = generate_episodes(
        n=args.cal_episodes, steps=32, size=96, seed=args.cal_seed,
        goal_directed=0.5, nuisance_mode="normal",
        hidden_alcove=True, occlude_door_right=True,
    )
    test_eps = generate_episodes(
        n=args.test_episodes, steps=32, size=96, seed=args.test_seed,
        goal_directed=0.5, nuisance_mode="normal",
        hidden_alcove=True, occlude_door_right=True,
    )
    print(f"cal {len(cal_eps)} test {len(test_eps)}", flush=True)
    cal_runs = [run_episode(ep, state_model, jepa, gates, belief) for ep in cal_eps]
    test_runs = [run_episode(ep, state_model, jepa, gates, belief) for ep in test_eps]
    cal_reach = [r for r in cal_runs if r["gate_end"]]
    cal_unreach = [r for r in cal_runs if not r["gate_end"]]
    print(
        f"cal reach {len(cal_reach)} success "
        f"{np.mean([r['success'] for r in cal_reach]):.2f}; "
        f"unreach {len(cal_unreach)}",
        flush=True,
    )

    taus = [0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.25]
    ks = [2, 3, 4, 5]
    report = {"base": {}, "calibration": {}, "test": {}, "selected": {}}
    for key in ("state", "jepa", "belief"):
        col = f"{key}_p"
        grid = []
        for tau in taus:
            for k in ks:
                r = gate_eval(cal_runs, col, tau, k)
                grid.append({"tau": tau, "k": k, **r})
        feas = [g for g in grid if g["false_abort"] <= 0.05]
        if feas:
            best = max(
                feas,
                key=lambda g: (
                    g["unreach_abort"],
                    -(g["time_to_abort"]
                      if not np.isnan(g["time_to_abort"]) else 1e9),
                ),
            )
        else:
            best = min(grid, key=lambda g: g["false_abort"])
        report["calibration"][key] = grid
        report["selected"][key] = best
        rt = gate_eval(test_runs, col, best["tau"], best["k"])
        report["test"][key] = rt
        print(
            f"{key}: tau {best['tau']:.2f} k {best['k']} | "
            f"cal false {best['false_abort']:.2f} abort {best['unreach_abort']:.2f} "
            f"t {best['time_to_abort']:.1f} | "
            f"test success {rt['reach_success']:.2f} false {rt['false_abort']:.2f} "
            f"(clean {rt['false_abort_clean']:.2f} fail {rt['false_abort_failure']:.2f}) "
            f"abort {rt['unreach_abort']:.2f} t {rt['time_to_abort']:.1f}",
            flush=True,
        )

    # Full test grid for the belief/state gates (the user's k/tau sweep).
    report["test_grid"] = {}
    for key in ("state", "jepa", "belief"):
        col = f"{key}_p"
        grid = []
        for tau in taus:
            for k in ks:
                r = gate_eval(test_runs, col, tau, k)
                grid.append({"tau": tau, "k": k, **r})
        report["test_grid"][key] = grid
        print(f"test grid {key}:", flush=True)
        for g in grid:
            if g["tau"] in (0.18, 0.20, 0.25):
                print(
                    f"  tau {g['tau']:.2f} k {g['k']}: "
                    f"false {g['false_abort']:.3f} "
                    f"(clean {g['false_abort_clean']:.3f} "
                    f"fail {g['false_abort_failure']:.3f}) "
                    f"abort {g['unreach_abort']:.2f} "
                    f"delay {g['abort_delay_after_close']:.2f}",
                    flush=True,
                )

    base = {
        "reach_success": float(np.mean([
            r["success"] for r in test_runs if r["gate_end"]
        ])),
        "reach_n": int(sum(r["gate_end"] for r in test_runs)),
        "unreach_n": int(sum(not r["gate_end"] for r in test_runs)),
        "unreach_abort": 0.0,
        "time_to_abort": float("nan"),
        "step_savings": 0.0,
    }
    report["base"] = base
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("base test", base)
    print(f"wrote {out / 'report.json'}")


if __name__ == "__main__":
    main()
