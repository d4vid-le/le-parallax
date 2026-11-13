#!/usr/bin/env python3
"""Retrain the belief stream on planner-driven closed-loop trajectories.

The natural training set did not contain enough open-gate left-start episodes
that the state-feedback controller fails to cross. The belief gate falsely
aborted those. This script rolls out the actual closed-loop controller, records
frames and the true per-step gate status, duplicates the left-start
non-crossing hard negatives, mixes in the counterfactual pairs, and retrains
the belief stream. The final recipe uses 480 rollouts, 2x hard-negative
duplication, 400 cf pairs, and 60 epochs.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from natural_belief import BeliefGRU, build_cf_raw, train_belief
from oracle_state_cnn import OracleCNN
from leparallax.train import load_jepa
from leparallax.world import generate_episodes, render_state, step_state


def history(frames, t):
    h = frames[max(0, t - 2):t + 1]
    while len(h) < 3:
        h = [frames[0]] + h
    return np.stack(h)


def rollout(ep, state_model, max_steps=32):
    state = ep.states[0].copy()
    goal = ep.goal.astype(np.float32)
    nuisance = ep.nuisance
    door = True
    gate = True
    frames, labels = [], []
    for t in range(max_steps):
        frame = render_state(
            state, goal, nuisance, size=96, door_open=door,
            occlude_door_right=True,
        )
        frames.append(frame)
        labels.append(float(gate))
        obs = history(frames, t).astype(np.float32) / 255.0
        p4 = np.array(state_model(mx.array(obs[None]))[1])[0]
        a = np.clip(2.0 * (goal - p4[:2]) - 1.0 * p4[2:], -1.0, 1.0)
        old_x = state[0]
        state = step_state(
            state, a.astype(np.float32), door_open=door, gate_open=gate
        )
        if door and old_x < 0.5 <= state[0] and 0.35 <= state[1] <= 0.65:
            door = False
            gate = False
    return (
        np.stack(frames),
        np.array(labels, dtype=np.float32),
        {
            "start_right": bool(ep.states[0, 0] >= 0.5),
            "gate_end": bool(gate),
        },
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jepa", default="runs/dual-stream-jepa/models/s0-asym_clean.safetensors")
    ap.add_argument("--state", default="runs/dual-stream-state/state_stream_s0.safetensors")
    ap.add_argument("--episodes", type=int, default=480)
    ap.add_argument("--cf-pairs", type=int, default=400)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--hard-repeat", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/planner-belief-hard")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    state_model = OracleCNN()
    state_model.load_weights(args.state)
    eps = generate_episodes(
        n=args.episodes, steps=32, size=96, seed=5000,
        goal_directed=0.5, nuisance_mode="normal",
        hidden_alcove=True, occlude_door_right=True,
    )
    frames, labels, meta = [], [], []
    for i, ep in enumerate(eps):
        f, y, m = rollout(ep, state_model)
        frames.append(f)
        labels.append(y)
        meta.append(m)
        if (i + 1) % 40 == 0:
            print(f"  rollouts {i+1}/{len(eps)}", flush=True)

    hard = [
        i for i, m in enumerate(meta)
        if not m["start_right"] and m["gate_end"]
    ]
    counts = {
        "episodes": len(eps),
        "right_start": int(sum(m["start_right"] for m in meta)),
        "left_start": int(sum(not m["start_right"] for m in meta)),
        "left_nocross": int(len(hard)),
        "left_cross": int(sum(
            not m["start_right"] and not m["gate_end"] for m in meta
        )),
        "hard_repeat": args.hard_repeat,
    }
    for i in hard:
        for _ in range(args.hard_repeat):
            frames.append(frames[i])
            labels.append(labels[i])
    frames = np.stack(frames).astype(np.float32) / 255.0
    labels = np.stack(labels)
    cf_frames, cf_labels = build_cf_raw(args.cf_pairs, 123)
    counts["train_sequences"] = int(len(frames))
    print(
        f"closed-loop train {frames.shape} labels open "
        f"{labels.mean():.3f}; cf {cf_frames.shape}; counts {counts}",
        flush=True,
    )
    (out / "train_counts.json").write_text(json.dumps(counts, indent=2) + "\n")
    jepa = load_jepa(args.jepa, image_size=96)
    model = train_belief(
        frames, labels, jepa.encoder.frame, args.seed, epochs=args.epochs,
        cf_frames=cf_frames, cf_labels=cf_labels,
    )
    model.save_weights(str(out / f"belief_s{args.seed}.safetensors"))
    print(f"saved {out / f'belief_s{args.seed}.safetensors'}")


if __name__ == "__main__":
    main()
