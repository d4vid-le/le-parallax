#!/usr/bin/env python3
"""Generate the Step 3 figures from the actual world and saved reports.

Figure 1: world / hidden-gate schematic, rendered observations, matched
counterfactual belief curves, and branch-AUC controls.
Figure 2: final five-seed natural matrix, gate-closed recall/delay, matched
counterfactual summary, and the per-seed result-space scatter.

Run after the final gate and matched-CF reports exist in runs/.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
import mlx.core as mx
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from closed_loop_gate import static_features, train_gate
from matched_cf_closed_loop import make_history, rollout
from natural_belief import BeliefGRU
from oracle_state_cnn import OracleCNN
from leparallax.train import load_jepa
from leparallax.world import (
    generate_episodes, render_state, sample_nuisance,
)

BLUE = "#2E5EAA"
ORANGE = "#C26A00"
GREEN = "#2F7D4A"
RED = "#B42318"
GRAY = "#6B7280"
LIGHT = "#E5E7EB"


def frame_state(x, y, door_open, nuisance, goal):
    return render_state(
        np.array([x, y, 0.0, 0.0], dtype=np.float32),
        goal, nuisance, size=96, door_open=door_open,
        occlude_door_right=True,
    )


def draw_schematic(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(Rectangle((0.02, 0.02), 0.96, 0.96, fill=False,
                           lw=1.5, ec=GRAY))

    # One-way door, visible.
    ax.plot([0.5, 0.5], [0.02, 0.35], color=GRAY, lw=4, solid_capstyle="butt")
    ax.plot([0.5, 0.5], [0.65, 0.98], color=GRAY, lw=4, solid_capstyle="butt")
    ax.text(0.5, 0.09, "one-way\ndoor", ha="center", va="bottom",
            fontsize=8, color=GRAY)

    # Hidden gate, never rendered.
    ax.plot([0.78, 0.78], [0.08, 0.92], color=RED, lw=1.6, ls=(0, (4, 3)))
    ax.text(0.79, 0.86, "hidden alcove\ngate (never rendered)",
            ha="left", va="top", fontsize=8, color=RED)

    # Occlusion wall that hides the door when the puck is right.
    ax.add_patch(Rectangle((0.43, 0.02), 0.14, 0.96, color=LIGHT,
                           alpha=0.55, lw=0, zorder=0))
    ax.text(0.50, 0.72, "occlusion\nwall", ha="center", va="bottom",
            fontsize=7, color=GRAY)

    # Goal and starts.
    ax.add_patch(Circle((0.90, 0.50), 0.055, fill=False, ec=GREEN, lw=2.2))
    ax.text(0.90, 0.42, "goal", ha="center", va="top", fontsize=8,
            color=GREEN)
    ax.add_patch(Circle((0.22, 0.50), 0.035, color=ORANGE, zorder=5))
    ax.add_patch(Circle((0.62, 0.50), 0.035, color=BLUE, zorder=5))
    ax.text(0.22, 0.43, "left start", ha="center", va="top", fontsize=8,
            color=ORANGE)
    ax.text(0.62, 0.43, "right start", ha="center", va="top", fontsize=8,
            color=BLUE)

    # Left-start path: cross door, gate closes, blocked.
    ax.add_patch(FancyArrowPatch(
        (0.24, 0.56), (0.74, 0.56), arrowstyle="-|>", mutation_scale=12,
        color=ORANGE, lw=2.0, zorder=3,
    ))
    ax.plot([0.76], [0.56], marker="x", ms=9, mew=2.2, color=RED, zorder=6)
    ax.text(0.58, 0.63, "blocked at gate", fontsize=7.5, color=ORANGE,
            ha="left", va="bottom")

    # Right-start path: gate open, goal reachable.
    ax.add_patch(FancyArrowPatch(
        (0.64, 0.44), (0.86, 0.48), arrowstyle="-|>", mutation_scale=12,
        color=BLUE, lw=2.0, zorder=3,
    ))
    ax.text(0.66, 0.37, "reachable", fontsize=7.5, color=BLUE, ha="left",
            va="top")

    ax.text(0.02, 0.99, "A  world geometry", fontsize=11, fontweight="bold",
            va="top")


def draw_frames(fig, gs):
    nuisance = sample_nuisance(np.random.default_rng(7))
    goal = np.array([0.90, 0.50], dtype=np.float32)
    panels = [
        (0.30, "left start\ndoor open", True),
        (0.52, "crossed\ndoor closes", False),
        (0.62, "right start\ngate open", True),
        (0.78, "blocked at\nthe hidden gate", False),
    ]
    sub = gs.subgridspec(2, 2, wspace=0.08, hspace=0.18)
    for i, (x, title, door) in enumerate(panels):
        ax = fig.add_subplot(sub[i // 2, i % 2])
        ax.imshow(frame_state(x, 0.50, door, nuisance, goal))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=8, pad=3)
        for spine in ax.spines.values():
            spine.set_color(LIGHT)
    fig.text(0.64, 0.965, "B  what the model sees", fontsize=11,
             fontweight="bold", va="top", ha="center")


def load_models(args):
    state_model = OracleCNN()
    state_model.load_weights(args.state)
    jepa = load_jepa(args.jepa, image_size=96)
    belief = BeliefGRU(load_jepa(args.jepa, image_size=96).encoder.frame)
    belief.load_weights(args.belief)

    train_eps = generate_episodes(
        n=80, steps=32, size=96, seed=0, goal_directed=0.5,
        nuisance_mode="normal", hidden_alcove=True,
        occlude_door_right=True,
    )
    sf, jf, ysf, _, _ = static_features(train_eps, state_model, jepa)
    gates = {
        "state": train_gate(sf, ysf, 0),
        "jepa": train_gate(jf, ysf, 0),
    }
    return state_model, jepa, belief, gates


def collect_cf_curves(args, state_model, jepa, belief, gates, n_pairs):
    rng = np.random.default_rng(args.cf_seed)
    curves = {
        "belief_closed": [], "belief_open": [],
        "state_closed": [], "state_open": [],
        "jepa_closed": [], "jepa_open": [],
    }
    tries = 0
    while len(curves["belief_closed"]) < n_pairs and tries < n_pairs * 40:
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
        if not np.array_equal(closed_hist[-1], open_hist[-1]):
            continue
        rc = rollout(branch, goal, nuisance, False, False, closed_hist,
                     state_model, jepa, gates, belief, args.max_steps)
        ro = rollout(branch, goal, nuisance, True, True, open_hist,
                     state_model, jepa, gates, belief, args.max_steps)
        curves["belief_closed"].append(rc["belief_p"])
        curves["belief_open"].append(ro["belief_p"])
        curves["state_closed"].append(rc["state_p"])
        curves["state_open"].append(ro["state_p"])
        curves["jepa_closed"].append(rc["jepa_p"])
        curves["jepa_open"].append(ro["jepa_p"])
    return {k: np.mean(np.stack(v), axis=0) for k, v in curves.items()}


def draw_curves(ax, curves):
    steps = np.arange(len(curves["belief_closed"]))
    ax.plot(steps, curves["belief_closed"], color=ORANGE, lw=2.4,
            marker="o", ms=3, label="belief, gate closed")
    ax.plot(steps, curves["belief_open"], color=BLUE, lw=2.4,
            marker="o", ms=3, label="belief, gate open")
    ax.plot(steps, curves["state_closed"], color=ORANGE, lw=1.1, ls="--",
            alpha=0.7, label="state-only, closed")
    ax.plot(steps, curves["state_open"], color=BLUE, lw=1.1, ls="--",
            alpha=0.7, label="state-only, open")
    ax.axhline(0.5, color=GRAY, lw=0.8, ls=":")
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("post-branch step", fontsize=8)
    ax.set_ylabel("P(hidden gate open)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6.5, ncol=2, frameon=False, loc="center right")
    ax.set_title("C  identical current view, different history", fontsize=10,
                 loc="left", fontweight="bold")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def draw_auc_bars(ax, cf_report):
    b = cf_report["arms"]["belief"]
    vals = [
        b["branch_auc"],
        b["short_history_auc"],
        cf_report["arms"]["state"]["early_auc"],
        cf_report["arms"]["jepa"]["early_auc"],
    ]
    names = ["belief\nfull history", "short\nhistory", "state\nearly", "JEPA\nearly"]
    colors = [BLUE, GRAY, ORANGE, GREEN]
    ax.bar(range(len(vals)), vals, color=colors, width=0.68)
    ax.axhline(0.5, color=GRAY, lw=0.8, ls=":")
    ax.set_ylim(0.40, 1.06)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(names, fontsize=7)
    ax.set_ylabel("branch AUC", fontsize=8)
    ax.tick_params(labelsize=7)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.015, f"{v:.3f}", ha="center", fontsize=7)
    ax.set_title("D  who actually separates the pair?", fontsize=10,
                 loc="left", fontweight="bold")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def load_matrix(args):
    gates, cfs = [], []
    for seed in args.seeds:
        gate = json.loads(
            pathlib.Path(f"{args.gate_prefix}{seed}/report.json").read_text()
        )
        row = next(
            r for r in gate["test_grid"]["belief"]
            if r["tau"] == args.tau and r["k"] == args.k
        )
        gates.append({"seed": seed, **row})
        cf = json.loads(
            pathlib.Path(f"{args.cf_prefix}{seed}/report.json").read_text()
        )
        b = cf["arms"]["belief"]
        cfs.append({
            "seed": seed,
            "branch_auc": b["branch_auc"],
            "short_history_auc": b["short_history_auc"],
            "state_early": cf["arms"]["state"]["early_auc"],
            "jepa_early": cf["arms"]["jepa"]["early_auc"],
            "closed_abort": b["closed_abort"],
            "open_false": b["open_false_abort"],
            "delay": b["abort_delay"],
        })
    return gates, cfs


def draw_false_aborts(ax, gates):
    seeds = np.arange(len(gates))
    width = 0.26
    all_open = np.array([g["false_abort"] for g in gates]) * 100
    clean = np.array([g["false_abort_clean"] for g in gates]) * 100
    left = np.array([g["false_abort_failure"] for g in gates]) * 100
    ax.bar(seeds - width, all_open, width, color=BLUE, label="all open")
    ax.bar(seeds, left, width, color=ORANGE, label="left non-cross")
    ax.bar(seeds + width, clean, width, color=GREEN,
           label="clean/reachable (0/124)")
    ax.axhline(5, color=RED, lw=1.0, ls="--")
    ax.text(4.55, 5.25, "5% ceiling", ha="right", fontsize=7, color=RED)
    for s, ao, lf in zip(seeds, all_open, left):
        if ao > 0:
            ax.text(s - width, ao + 0.10, f"{ao:.1f}", ha="center",
                    fontsize=6.5, color=BLUE)
        if lf > 0:
            ax.text(s, lf + 0.10, f"{lf:.1f}", ha="center",
                    fontsize=6.5, color=ORANGE)
    ax.set_xticks(seeds)
    ax.set_xticklabels([f"seed {g['seed']}" for g in gates], fontsize=7)
    ax.set_ylabel("false abort (%)", fontsize=8)
    ax.set_ylim(0, 6.2)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6.3, frameon=False, ncol=1, loc="upper left")
    ax.set_title("A  held-out false aborts", fontsize=10, loc="left",
                 fontweight="bold")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def draw_recall(ax, gates):
    seeds = np.arange(len(gates))
    recall = np.array([g["unreach_abort"] for g in gates]) * 100
    delay = np.array([g["abort_delay_after_close"] for g in gates])
    ax.bar(seeds, recall, 0.55, color=BLUE, alpha=0.85,
           label="gate-closed abort")
    ax.set_ylim(0, 112)
    ax.set_ylabel("gate-closed abort (%)", fontsize=8)
    ax.set_xticks(seeds)
    ax.set_xticklabels([f"seed {g['seed']}" for g in gates], fontsize=7)
    ax.tick_params(labelsize=7)
    ax2 = ax.twinx()
    ax2.plot(seeds, delay, color=ORANGE, marker="o", ms=4, lw=1.5,
             label="delay after close")
    ax2.set_ylim(0, 6)
    ax2.set_ylabel("delay (steps)", fontsize=8, color=ORANGE)
    ax2.tick_params(labelsize=7, colors=ORANGE)
    ax.set_title("B  unreachable recall and delay", fontsize=10, loc="left",
                 fontweight="bold")
    for spine in ("top",):
        ax.spines[spine].set_visible(False)
        ax2.spines[spine].set_visible(False)


def draw_cf_summary(ax, cfs):
    seeds = np.arange(len(cfs))
    closed = np.array([c["closed_abort"] for c in cfs]) * 100
    open_false = np.array([c["open_false"] for c in cfs]) * 100
    width = 0.34
    ax.bar(seeds - width / 2, closed, width, color=BLUE,
           label="closed abort")
    ax.bar(seeds + width / 2, open_false, width, color=GREEN,
           label="open false abort (0/100)")
    for s, c in zip(seeds, closed):
        ax.text(s - width / 2, c + 2, f"{c:.0f}", ha="center", fontsize=6.5,
                color=BLUE)
    ax.set_ylim(0, 112)
    ax.set_xticks(seeds)
    ax.set_xticklabels([f"seed {c['seed']}" for c in cfs], fontsize=7)
    ax.set_ylabel("matched pairs (%)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6.5, frameon=False, loc="lower right")
    ax.set_title("C  fresh matched counterfactual", fontsize=10, loc="left",
                 fontweight="bold")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def draw_result_space(ax, gates, cfs):
    x = np.array([g["false_abort"] for g in gates]) * 100
    y = np.array([g["abort_delay_after_close"] for g in gates])
    ax.scatter(x, y, s=60, color=BLUE, zorder=3)
    offsets = {
        0: (6, 6), 1: (6, -11), 2: (6, 6),
        3: (6, -11), 4: (6, 0),
    }
    for g, xi, yi in zip(gates, x, y):
        dx, dy = offsets.get(g["seed"], (6, 4))
        ax.annotate(f"s{g['seed']}", (xi, yi), xytext=(dx, dy),
                    textcoords="offset points", fontsize=7)
    ax.axvline(5, color=RED, lw=1.0, ls="--")
    ax.text(4.85, 2.862, "5% ceiling", ha="right", fontsize=7, color=RED)
    ax.set_xlim(-0.2, 5.2)
    ax.set_ylim(2.85, 3.04)
    ax.set_xlabel("all-open false abort (%)", fontsize=8)
    ax.set_ylabel("delay after close (steps)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.text(0.12, 2.992, "lower is better", fontsize=7, color=GREEN)
    ax.set_title("D  false abort vs abort delay", fontsize=10, loc="left",
                 fontweight="bold")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jepa", default="runs/dual-stream-jepa/models/s0-asym_clean.safetensors")
    ap.add_argument("--state", default="runs/dual-stream-state/state_stream_s0.safetensors")
    ap.add_argument("--belief", default="runs/planner-belief-hard/belief_s0.safetensors")
    ap.add_argument("--gate-prefix", default="runs/final-gate-s")
    ap.add_argument("--cf-prefix", default="runs/final-cf-s")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--tau", type=float, default=0.18)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--pairs", type=int, default=16)
    ap.add_argument("--seed", type=int, default=12000)
    ap.add_argument("--cf-seed", type=int, default=12000)
    ap.add_argument("--max-steps", type=int, default=10)
    ap.add_argument("--out-dir", default="snapshots")
    args = ap.parse_args()

    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("loading models...", flush=True)
    state_model, jepa, belief, gates = load_models(args)
    print("collecting matched counterfactual curves...", flush=True)
    curves = collect_cf_curves(
        args, state_model, jepa, belief, gates, args.pairs
    )
    cf_report = json.loads(
        pathlib.Path(f"{args.cf_prefix}{args.seeds[0]}/report.json").read_text()
    )
    gate_rows, cf_rows = load_matrix(args)

    # Figure 1: world, observations, counterfactual curves, controls.
    fig = plt.figure(figsize=(13.0, 9.2))
    gs = GridSpec(2, 3, figure=fig, width_ratios=[1.15, 1.0, 1.0],
                  height_ratios=[1.0, 0.95], wspace=0.28, hspace=0.34)
    ax_world = fig.add_subplot(gs[0, 0])
    draw_schematic(ax_world)
    draw_frames(fig, gs[0, 1:])
    ax_curves = fig.add_subplot(gs[1, :2])
    draw_curves(ax_curves, curves)
    ax_bars = fig.add_subplot(gs[1, 2])
    draw_auc_bars(ax_bars, cf_report)
    fig.suptitle("Step 3: the gate uses history, not the current view",
                 fontsize=14, fontweight="bold", x=0.02, ha="left", y=0.985)
    fig.text(
        0.02, 0.015,
        "Left-start episodes cross the one-way door, which closes the hidden "
        "alcove gate; right-start episodes keep it open. In the matched pairs "
        "the last three frames and the 4-D state are identical, so only the "
        "belief stream can tell them apart.",
        fontsize=8, color=GRAY,
    )
    fig.savefig(out / "step3-world-and-counterfactual.png", dpi=200,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # Figure 2: final five-seed matrix.
    fig = plt.figure(figsize=(12.8, 8.6))
    gs = GridSpec(2, 2, figure=fig, wspace=0.28, hspace=0.34)
    ax1 = fig.add_subplot(gs[0, 0])
    draw_false_aborts(ax1, gate_rows)
    ax2 = fig.add_subplot(gs[0, 1])
    draw_recall(ax2, gate_rows)
    ax3 = fig.add_subplot(gs[1, 0])
    draw_cf_summary(ax3, cf_rows)
    ax4 = fig.add_subplot(gs[1, 1])
    draw_result_space(ax4, gate_rows, cf_rows)
    fig.suptitle(
        "Step 3 final matrix: five seeds, fresh held-out episode seed",
        fontsize=14, fontweight="bold", x=0.02, ha="left", y=0.985,
    )
    fig.text(
        0.02, 0.015,
        "Locked belief gate: tau=0.18, k=4. Natural test: 124 reachable "
        "right-start, 37 left-start non-crossings, 79 left-start gate-closed. "
        "Matched pairs: 100 per seed, fresh pair seed.",
        fontsize=8, color=GRAY,
    )
    fig.savefig(out / "step3-final-matrix.png", dpi=200,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out / 'step3-world-and-counterfactual.png'}")
    print(f"wrote {out / 'step3-final-matrix.png'}")


if __name__ == "__main__":
    main()
