#!/usr/bin/env python3
"""Probe the base state-feedback controller's door-gap failures.

Compares the current goal-only controller with a two-phase controller that
targets the door gap while the puck is left of the one-way door. Reports
reachable success and the failure states. No gates or belief model involved.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from oracle_state_cnn import OracleCNN
from leparallax.world import generate_episodes, render_state, step_state


def run(ep, state_model, mode, max_steps=32):
    state = ep.states[0].copy()
    goal = ep.goal.astype(np.float32)
    nuisance = ep.nuisance
    door = True
    gate = True
    frames = []
    success = False
    final = state
    for t in range(max_steps):
        frame = render_state(
            state, goal, nuisance, size=96, door_open=door,
            occlude_door_right=True,
        )
        frames.append(frame)
        obs = np.stack(frames[-3:]).astype(np.float32) / 255.0
        while len(obs) < 3:
            obs = np.concatenate([obs[:1], obs], axis=0)
        p4 = np.array(state_model(mx.array(obs[None]))[1])[0]
        ctrl = state if mode.startswith("true_") else p4
        if mode in ("waypoint", "waypoint56", "waypoint60") and door and goal[0] > 0.5:
            if ctrl[0] < 0.5:
                tx = {"waypoint": 0.52, "waypoint56": 0.56, "waypoint60": 0.60}[mode]
                target = np.array([tx, 0.5], dtype=np.float32)
            else:
                target = goal
        elif mode in ("staged", "staged40", "staged_true") and door and goal[0] > 0.5 and ctrl[0] < 0.5:
            stage_x = 0.35 if mode == "staged" else 0.40
            if abs(ctrl[1] - 0.5) > 0.06:
                target = np.array([stage_x, 0.5], dtype=np.float32)
            else:
                target = np.array([0.62, 0.5], dtype=np.float32)
        else:
            target = goal
        a = np.clip(2.0 * (target - ctrl[:2]) - 1.0 * ctrl[2:], -1.0, 1.0)
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
        if np.linalg.norm(state[:2] - goal) < 0.08 and gate:
            success = True
            final = state
            break
        final = state
    return success, door, gate, final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default="runs/dual-stream-state/state_stream_s0.safetensors")
    ap.add_argument("--episodes", type=int, default=240)
    ap.add_argument("--seed", type=int, default=9000)
    ap.add_argument("--out", default="runs/controller-probe")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    state_model = OracleCNN()
    state_model.load_weights(args.state)
    eps = generate_episodes(
        n=args.episodes, steps=32, size=96, seed=args.seed,
        goal_directed=0.5, nuisance_mode="normal",
        hidden_alcove=True, occlude_door_right=True,
    )
    rows = {}
    for mode in ("goal", "staged", "true_goal"):
        results = [run(ep, state_model, mode) for ep in eps]
        reach = [r for r in results if r[1]]
        unreach = [r for r in results if not r[1]]
        fails = [
            {
                "start": eps[i].states[0].tolist(),
                "goal": eps[i].goal.tolist(),
                "final": r[3].tolist(),
                "gate_end": bool(r[1]),
            }
            for i, r in enumerate(results)
            if r[1] and not r[0]
        ]
        start_right = [ep.states[0, 0] >= 0.5 for ep in eps]
        right = [r for r, sr in zip(results, start_right) if sr]
        left = [r for r, sr in zip(results, start_right) if not sr]
        rows[mode] = {
            "reach_n": len(reach),
            "reach_success": float(np.mean([r[0] for r in reach])) if reach else float("nan"),
            "unreach_n": len(unreach),
            "right_start_n": len(right),
            "right_start_success": float(np.mean([r[0] for r in right])) if right else float("nan"),
            "left_start_n": len(left),
            "left_start_success": float(np.mean([r[0] for r in left])) if left else float("nan"),
            "left_gate_closed_n": int(sum(not r[1] for r in left)),
            "left_gate_open_n": int(sum(r[1] for r in left)),
            "failures": fails,
        }
        print(
            f"{mode}: reach {len(reach)} success {rows[mode]['reach_success']:.3f}; "
            f"unreach {len(unreach)}; fails {len(fails)}; "
            f"right {rows[mode]['right_start_n']} success "
            f"{rows[mode]['right_start_success']:.3f}; "
            f"left {rows[mode]['left_start_n']} success "
            f"{rows[mode]['left_start_success']:.3f} "
            f"(closed {rows[mode]['left_gate_closed_n']})",
            flush=True,
        )

    import json
    (out / "report.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(f"wrote {out / 'report.json'}")


if __name__ == "__main__":
    main()
