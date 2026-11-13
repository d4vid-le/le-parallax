#!/usr/bin/env python3
"""Matched counterfactual closed loop for the Step 3 abort gates.

Each pair has:
  * the same goal, nuisance, and branch state S_b (position and velocity),
  * the same last three frames at S_b (identical static input),
  * one branch whose history crossed the one-way door (gate closed),
  * one branch whose history started right of the door (gate open).

Both branches are then rolled forward with the same state-feedback controller,
under their respective hidden gate status. State-only and frozen-JEPA gates see
identical current inputs at the branch; only a history-dependent belief stream
can separate the two. This is the decisive shortcut test.
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

from closed_loop_gate import predict_gate, train_gate
from natural_belief import BeliefGRU, static_features
from oracle_state_cnn import OracleCNN
from leparallax.train import load_jepa
from leparallax.world import (
    generate_episodes, render_state, sample_nuisance, step_state,
)


def clip_action(target, state):
    a = 2.0 * (target - state[:2]) - 1.0 * state[2:]
    return np.clip(a, -1.0, 1.0).astype(np.float32)


def make_history(kind, branch, goal, nuisance, rng, steps=12):
    """Generate a plausibly controlled approach ending at the branch state.

    Closed history starts left of the door and crosses it, so the gate closes.
    Open history starts just right of the door and never crosses. Both end with
    three identical frames at the branch state.
    """
    target = branch[:2].copy()
    frames = []
    crossed = False
    if kind == "closed":
        state = np.array(
            [rng.uniform(0.25, 0.40), rng.uniform(0.38, 0.62), 0.0, 0.0],
            dtype=np.float32,
        )
        door, gate = True, True
        for _ in range(steps):
            frames.append(render_state(
                state, goal, nuisance, size=96, door_open=door,
                occlude_door_right=True,
            ))
            action = clip_action(target, state)
            old_x = state[0]
            state = step_state(
                state, action, door_open=door, gate_open=gate
            )
            if door and old_x < 0.5 <= state[0] and 0.35 <= state[1] <= 0.65:
                door, gate = False, False
                crossed = True
        for _ in range(3):
            frames.append(render_state(
                branch, goal, nuisance, size=96, door_open=False,
                occlude_door_right=True,
            ))
        return frames, crossed, False, False

    state = np.array(
        [rng.uniform(0.54, 0.60), branch[1] + rng.uniform(-0.03, 0.03), 0.0, 0.0],
        dtype=np.float32,
    )
    door, gate = True, True
    for _ in range(steps):
        frames.append(render_state(
            state, goal, nuisance, size=96, door_open=door,
            occlude_door_right=True,
        ))
        action = clip_action(target, state)
        state = step_state(state, action, door_open=door, gate_open=gate)
    for _ in range(3):
        frames.append(render_state(
            branch, goal, nuisance, size=96, door_open=True,
            occlude_door_right=True,
        ))
    return frames, False, True, True


def rollout(branch, goal, nuisance, door, gate, history, state_model, jepa,
            gates, belief, max_steps=10):
    state = branch.copy()
    frames = list(history)
    ps, pj = [], []
    for _ in range(max_steps):
        obs = np.stack(frames[-3:]).astype(np.float32) / 255.0
        p4 = np.array(state_model(mx.array(obs[None]))[1])[0]
        z = np.array(jepa.encode(mx.array(obs[None])))[0]
        ps.append(predict_gate(
            gates["state"][0], gates["state"][1], gates["state"][2],
            np.concatenate([p4, goal]),
        ))
        pj.append(predict_gate(
            gates["jepa"][0], gates["jepa"][1], gates["jepa"][2], z,
        ))
        action = clip_action(goal, state)
        state = step_state(
            state, action, door_open=door, gate_open=gate
        )
        frames.append(render_state(
            state, goal, nuisance, size=96, door_open=door,
            occlude_door_right=True,
        ))

    arr = np.stack(frames).astype(np.float32) / 255.0
    p = np.array(mx.sigmoid(belief(mx.array(arr[None])))[0])
    branch_idx = len(history) - 1
    # Three-frame ablation: only the matched branch frames.
    short = arr[branch_idx - 2:branch_idx + 1]
    p_short = float(mx.sigmoid(belief(mx.array(short[None])))[0, -1])
    return {
        "state_p": np.array(ps),
        "jepa_p": np.array(pj),
        "belief_p": p[branch_idx:branch_idx + max_steps],
        "belief_branch": float(p[branch_idx]),
        "belief_short_branch": p_short,
        "final_x": float(state[0]),
        "gate_end": bool(gate),
    }


def auc(pos, neg):
    x = np.concatenate([pos, neg])
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(x) + 1)
    n_pos, n_neg = len(pos), len(neg)
    return float((ranks[:n_pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def abort_at(p, tau, k):
    for t in range(len(p) - k + 1):
        if np.all(p[t:t + k] < tau):
            return t + k - 1
    return None


def summarize(pairs, key, tau, k):
    closed = [r for r in pairs if not r["open"]]
    open_ = [r for r in pairs if r["open"]]
    c_abort = [abort_at(r[key], tau, k) for r in closed]
    o_abort = [abort_at(r[key], tau, k) for r in open_]
    c_hit = [a is not None for a in c_abort]
    o_hit = [a is not None for a in o_abort]
    delays = [a for a in c_abort if a is not None]
    return {
        "n_closed": len(closed),
        "n_open": len(open_),
        "closed_abort": float(np.mean(c_hit)) if c_hit else float("nan"),
        "open_false_abort": float(np.mean(o_hit)) if o_hit else float("nan"),
        "abort_delay": float(np.mean(delays)) if delays else float("nan"),
        "closed_branch_low": float(np.mean([r[key][0] < tau for r in closed])),
        "open_branch_low": float(np.mean([r[key][0] < tau for r in open_])),
        "branch_auc": auc(
            [r["belief_branch"] for r in open_],
            [r["belief_branch"] for r in closed],
        ) if key == "belief_p" else float("nan"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jepa", default="runs/dual-stream-jepa/models/s0-asym_clean.safetensors")
    ap.add_argument("--state", default="runs/dual-stream-state/state_stream_s0.safetensors")
    ap.add_argument("--belief", default="runs/planner-belief/belief_s0.safetensors")
    ap.add_argument("--pairs", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=10)
    ap.add_argument("--tau-belief", type=float, default=0.18)
    ap.add_argument("--k-belief", type=int, default=4)
    ap.add_argument("--tau-state", type=float, default=0.25)
    ap.add_argument("--k-state", type=int, default=3)
    ap.add_argument("--tau-jepa", type=float, default=0.12)
    ap.add_argument("--k-jepa", type=int, default=4)
    ap.add_argument("--out", default="runs/matched-cf-closed-loop")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    mx.random.seed(args.seed)

    state_model = OracleCNN()
    state_model.load_weights(args.state)
    jepa = load_jepa(args.jepa, image_size=96)
    belief_jepa = load_jepa(args.jepa, image_size=96)
    belief = BeliefGRU(belief_jepa.encoder.frame)
    belief.load_weights(args.belief)

    train_eps = generate_episodes(
        n=80, steps=32, size=96, seed=0, goal_directed=0.5,
        nuisance_mode="normal", hidden_alcove=True, occlude_door_right=True,
    )
    sf, jf, ysf, _, _ = static_features(train_eps, state_model, jepa)
    gates = {
        "state": train_gate(sf, ysf, 0),
        "jepa": train_gate(jf, ysf, 0),
    }

    pairs = []
    tries = 0
    while len(pairs) < 2 * args.pairs and tries < args.pairs * 40:
        tries += 1
        goal = np.array(
            [rng.uniform(0.85, 0.95), rng.uniform(0.20, 0.80)],
            dtype=np.float32,
        )
        nuisance = sample_nuisance(rng)
        branch = np.array(
            [rng.uniform(0.62, 0.72), rng.uniform(0.30, 0.70), 0.0, 0.0],
            dtype=np.float32,
        )
        closed_hist, crossed, _, _ = make_history(
            "closed", branch, goal, nuisance, rng
        )
        if not crossed:
            continue
        open_hist, _, _, _ = make_history(
            "open", branch, goal, nuisance, rng
        )
        # Sanity: the matched branch frames must be byte-identical.
        if not np.array_equal(closed_hist[-1], open_hist[-1]):
            continue
        rc = rollout(
            branch, goal, nuisance, False, False, closed_hist,
            state_model, jepa, gates, belief, args.max_steps,
        )
        ro = rollout(
            branch, goal, nuisance, True, True, open_hist,
            state_model, jepa, gates, belief, args.max_steps,
        )
        pairs.append({
            "open": False,
            **{f"{k}": v for k, v in rc.items()},
            "state_p": rc["state_p"],
            "jepa_p": rc["jepa_p"],
            "belief_p": rc["belief_p"],
        })
        pairs.append({
            "open": True,
            **{f"{k}": v for k, v in ro.items()},
            "state_p": ro["state_p"],
            "jepa_p": ro["jepa_p"],
            "belief_p": ro["belief_p"],
        })

    if len(pairs) < 2 * args.pairs:
        raise RuntimeError(f"only built {len(pairs)//2} pairs")

    # Static gates must be exactly tied at the branch: same 3 frames + goal.
    closed = [r for r in pairs if not r["open"]]
    open_ = [r for r in pairs if r["open"]]
    pair_state_diffs = [
        abs(c["state_p"][0] - o["state_p"][0])
        for c, o in zip(closed, open_)
    ]
    pair_jepa_diffs = [
        abs(c["jepa_p"][0] - o["jepa_p"][0])
        for c, o in zip(closed, open_)
    ]

    belief_branch_auc = auc(
        [r["belief_branch"] for r in open_],
        [r["belief_branch"] for r in closed],
    )
    belief_short_auc = auc(
        [r["belief_short_branch"] for r in open_],
        [r["belief_short_branch"] for r in closed],
    )
    # Early post-branch window before the gate position can differ.
    early = 2
    belief_early_auc = auc(
        [float(np.mean(r["belief_p"][:early])) for r in open_],
        [float(np.mean(r["belief_p"][:early])) for r in closed],
    )
    state_early_auc = auc(
        [float(np.mean(r["state_p"][:early])) for r in open_],
        [float(np.mean(r["state_p"][:early])) for r in closed],
    )
    jepa_early_auc = auc(
        [float(np.mean(r["jepa_p"][:early])) for r in open_],
        [float(np.mean(r["jepa_p"][:early])) for r in closed],
    )

    arms = {
        "belief": summarize(pairs, "belief_p", args.tau_belief, args.k_belief),
        "state": summarize(pairs, "state_p", args.tau_state, args.k_state),
        "jepa": summarize(pairs, "jepa_p", args.tau_jepa, args.k_jepa),
    }
    arms["belief"]["branch_auc"] = belief_branch_auc
    arms["belief"]["short_history_auc"] = belief_short_auc
    arms["belief"]["early_auc"] = belief_early_auc
    arms["state"]["early_auc"] = state_early_auc
    arms["jepa"]["early_auc"] = jepa_early_auc

    report = {
        "pairs": len(pairs) // 2,
        "max_steps": args.max_steps,
        "thresholds": {
            "belief": [args.tau_belief, args.k_belief],
            "state": [args.tau_state, args.k_state],
            "jepa": [args.tau_jepa, args.k_jepa],
        },
        "match": {
            "state_branch_max_abs_diff": float(max(pair_state_diffs)),
            "jepa_branch_max_abs_diff": float(max(pair_jepa_diffs)),
        },
        "arms": arms,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    print(f"pairs {len(pairs)//2}")
    print(
        f"branch static diffs: state {max(pair_state_diffs):.2e} "
        f"jepa {max(pair_jepa_diffs):.2e}"
    )
    print(
        f"belief branch AUC {belief_branch_auc:.3f}; "
        f"short-history AUC {belief_short_auc:.3f}; "
        f"early-post AUC {belief_early_auc:.3f}"
    )
    print(
        f"early static AUC: state {state_early_auc:.3f} "
        f"jepa {jepa_early_auc:.3f}"
    )
    for name, row in arms.items():
        print(
            f"{name}: closed abort {row['closed_abort']:.2f} "
            f"open false {row['open_false_abort']:.2f} "
            f"delay {row['abort_delay']:.2f} "
            f"branch AUC {row['branch_auc']:.3f}"
        )
    print(f"wrote {out / 'report.json'}")


if __name__ == "__main__":
    main()
