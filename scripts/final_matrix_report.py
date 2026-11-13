#!/usr/bin/env python3
"""Collect the final five-seed gate matrix from the saved run reports.

Reads the held-out natural-test reports and the fresh matched-counterfactual
reports, selects the locked (tau, k), and prints the decomposed tables plus
mean +/- std. No training or inference.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np


def mean_std(values):
    a = np.asarray(values, dtype=np.float64)
    return float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate-prefix", default="runs/final-gate-s")
    ap.add_argument("--cf-prefix", default="runs/final-cf-s")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--tau", type=float, default=0.18)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--out", default="runs/final-matrix-report")
    args = ap.parse_args()

    gate_rows, cf_rows = [], []
    state_row = jepa_row = None
    for seed in args.seeds:
        gate = json.loads(
            pathlib.Path(f"{args.gate_prefix}{seed}/report.json").read_text()
        )
        row = next(
            r for r in gate["test_grid"]["belief"]
            if r["tau"] == args.tau and r["k"] == args.k
        )
        gate_rows.append({"seed": seed, **row})
        if state_row is None:
            state_row = next(
                r for r in gate["test_grid"]["state"]
                if r["tau"] == 0.25 and r["k"] == 3
            )
            jepa_row = next(
                r for r in gate["test_grid"]["jepa"]
                if r["tau"] == 0.12 and r["k"] == 4
            )

        cf = json.loads(
            pathlib.Path(f"{args.cf_prefix}{seed}/report.json").read_text()
        )
        b = cf["arms"]["belief"]
        cf_rows.append({
            "seed": seed,
            "branch_auc": b["branch_auc"],
            "short_history_auc": b["short_history_auc"],
            "static_state_early_auc": cf["arms"]["state"]["early_auc"],
            "static_jepa_early_auc": cf["arms"]["jepa"]["early_auc"],
            "closed_abort": b["closed_abort"],
            "open_false_abort": b["open_false_abort"],
            "delay": b["abort_delay"],
        })

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {
        "tau": args.tau,
        "k": args.k,
        "natural": gate_rows,
        "counterfactual": cf_rows,
        "state_only": state_row,
        "frozen_jepa": jepa_row,
    }
    (out / "report.json").write_text(json.dumps(summary, indent=2) + "\n")

    lines = [
        "# Final five-seed gate matrix",
        "",
        f"Locked belief gate: tau={args.tau}, k={args.k}.",
        "",
        "## Held-out natural test",
        "",
        "| seed | all-open false | clean false | left non-cross false | gate-closed abort | delay |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for r in gate_rows:
        lines.append(
            f"| {r['seed']} | {r['false_abort']*100:.2f}% "
            f"({round(r['false_abort']*r['reach_n'])}/{r['reach_n']}) | "
            f"{r['false_abort_clean']*100:.2f}% "
            f"({round(r['false_abort_clean']*r['clean_n'])}/{r['clean_n']}) | "
            f"{r['false_abort_failure']*100:.2f}% "
            f"({round(r['false_abort_failure']*r['fail_n'])}/{r['fail_n']}) | "
            f"{r['unreach_abort']*100:.2f}% "
            f"({round(r['unreach_abort']*r['unreach_n'])}/{r['unreach_n']}) | "
            f"{r['abort_delay_after_close']:.2f} |"
        )
    vals = {
        "all-open": [r["false_abort"] for r in gate_rows],
        "clean": [r["false_abort_clean"] for r in gate_rows],
        "left non-cross": [r["false_abort_failure"] for r in gate_rows],
        "gate-closed abort": [r["unreach_abort"] for r in gate_rows],
        "delay": [r["abort_delay_after_close"] for r in gate_rows],
    }
    lines.append("")
    lines.append("| metric | mean ± std |")
    lines.append("|---|---:|")
    for name, v in vals.items():
        m, s = mean_std(v)
        lines.append(f"| {name} | {m:.4f} ± {s:.4f} |")

    lines += [
        "",
        "## Fresh matched counterfactual",
        "",
        "| seed | belief branch AUC | short-history | static state early | static JEPA early | closed abort | open false | delay |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in cf_rows:
        lines.append(
            f"| {r['seed']} | {r['branch_auc']:.4f} | "
            f"{r['short_history_auc']:.4f} | "
            f"{r['static_state_early_auc']:.4f} | "
            f"{r['static_jepa_early_auc']:.4f} | "
            f"{r['closed_abort']:.2f} | {r['open_false_abort']:.2f} | "
            f"{r['delay']:.2f} |"
        )
    lines.append("")
    for name, key in (
        ("branch AUC", "branch_auc"),
        ("short-history", "short_history_auc"),
        ("closed abort", "closed_abort"),
        ("open false", "open_false_abort"),
        ("delay", "delay"),
    ):
        m, s = mean_std([r[key] for r in cf_rows])
        lines.append(f"- {name}: **{m:.4f} ± {s:.4f}**")

    lines += [
        "",
        "## Other natural gates",
        "",
        "| arm | all-open false | clean false | gate-closed abort | delay |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, row in (("state-only", state_row), ("frozen JEPA", jepa_row)):
        lines.append(
            f"| {name} | {row['false_abort']*100:.2f}% "
            f"({round(row['false_abort']*row['reach_n'])}/{row['reach_n']}) | "
            f"{row['false_abort_clean']*100:.2f}% "
            f"({round(row['false_abort_clean']*row['clean_n'])}/{row['clean_n']}) | "
            f"{row['unreach_abort']*100:.2f}% "
            f"({round(row['unreach_abort']*row['unreach_n'])}/{row['unreach_n']}) | "
            f"{row['abort_delay_after_close']:.2f} |"
        )

    text = "\n".join(lines) + "\n"
    (out / "report.md").write_text(text)
    print(text)
    print(f"wrote {out / 'report.md'}")


if __name__ == "__main__":
    main()
