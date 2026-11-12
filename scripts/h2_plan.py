#!/usr/bin/env python3
"""H2: planning under a nuisance shift with the frozen H1 checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from leparallax.plan import evaluate_planner, evaluate_random
from leparallax.train import load_jepa
from leparallax.world import generate_episodes

PREREG = pathlib.Path(__file__).resolve().parents[1] / "docs" / "PREREG-H2.md"


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_rows(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def append_row(path: pathlib.Path, row: dict):
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")
        f.flush()


def summarize(out: pathlib.Path):
    rows = load_rows(out / "per_seed.jsonl")
    by = {(r["seed"], r["arm"], r["split"]): r for r in rows}
    seeds = sorted({r["seed"] for r in rows})
    per = []
    for s in seeds:
        t_ood = by.get((s, "temporal", "ood"))
        g_ood = by.get((s, "good", "ood"))
        t_id = by.get((s, "temporal", "id"))
        g_id = by.get((s, "good", "id"))
        r_ood = by.get((s, "random", "ood"))
        if not (t_ood and g_ood and t_id and g_id):
            continue
        per.append(
            {
                "seed": s,
                "ood_temporal_dist": t_ood["mean_dist"],
                "ood_good_dist": g_ood["mean_dist"],
                "ood_diff": g_ood["mean_dist"] - t_ood["mean_dist"],
                "ood_temporal_success": t_ood["success"],
                "ood_good_success": g_ood["success"],
                "id_temporal_dist": t_id["mean_dist"],
                "id_good_dist": g_id["mean_dist"],
                "id_diff": g_id["mean_dist"] - t_id["mean_dist"],
                "random_ood_dist": r_ood["mean_dist"] if r_ood else None,
                "random_ood_success": r_ood["success"] if r_ood else None,
            }
        )

    if not per:
        print("no complete seed pairs yet")
        return

    diffs = np.array([p["ood_diff"] for p in per], dtype=np.float64)
    rng = np.random.default_rng(0)
    boot = np.array(
        [np.mean(rng.choice(diffs, size=len(diffs), replace=True)) for _ in range(10_000)]
    )
    ci = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
    mean_diff = float(diffs.mean())
    count_lower = int((diffs < 0).sum())
    mean_id_diff = float(np.mean([p["id_diff"] for p in per]))
    complete = len(per) == len(seeds) and len(per) == 5
    success_drop = [
        p["ood_good_success"] - p["ood_temporal_success"] for p in per
    ]
    passed = (
        complete
        and mean_diff <= -0.05
        and count_lower >= 4
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
        "mean_success_delta": float(np.mean(success_drop)),
        "verdict": "PASS" if passed else "FAIL",
    }
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    lines = [
        "# H2 planning under nuisance shift — result",
        "",
        f"- pre-registration: `{report['prereg_sha256']}`",
        f"- verdict: **{report['verdict']}**",
        f"- complete: {complete}",
        "",
        "| seed | OOD temp dist | OOD good dist | diff | OOD temp succ | OOD good succ | ID temp dist | ID good dist |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for p in per:
        lines.append(
            f"| {p['seed']} | {p['ood_temporal_dist']:.3f} | {p['ood_good_dist']:.3f} | "
            f"{p['ood_diff']:+.3f} | {p['ood_temporal_success']:.2f} | "
            f"{p['ood_good_success']:.2f} | {p['id_temporal_dist']:.3f} | "
            f"{p['id_good_dist']:.3f} |"
        )
    lines += [
        "",
        f"- mean OOD difference (good − temporal): **{mean_diff:+.3f}** "
        f"(95% CI {ci[0]:+.3f} to {ci[1]:+.3f})",
        f"- seeds with good lower OOD distance: **{count_lower}/{len(per)}**",
        f"- mean ID difference: **{mean_id_diff:+.3f}**",
        f"- mean OOD success delta: **{report['mean_success_delta']:+.2f}**",
        "",
        "Endpoint: mean OOD diff ≤ −0.05, at least 4/5 seeds lower, ID diff ≤ +0.05.",
        "",
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({k: report[k] for k in ("verdict", "mean_ood_diff", "ood_diff_ci95", "seeds_good_lower", "mean_id_diff", "mean_success_delta")}, indent=2))
    print(f"wrote {out / 'report.md'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="runs/h1-confirm/models")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--ood-episodes", type=int, default=10)
    ap.add_argument("--id-episodes", type=int, default=5)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--horizon", type=int, default=3)
    ap.add_argument("--out", default="runs/h2-plan")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    models = pathlib.Path(args.models)
    manifest = {
        "prereg": str(PREREG),
        "prereg_sha256": sha256(PREREG),
        "args": vars(args),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    try:
        manifest["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PREREG.parents[1], text=True
        ).strip()
    except Exception:
        manifest["git_commit"] = None
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    rows_path = out / "per_seed.jsonl"
    done = {(r["seed"], r["arm"], r["split"]) for r in load_rows(rows_path)}

    for seed in args.seeds:
        print(f"\n=== seed {seed} ===", flush=True)
        ood = generate_episodes(
            n=args.ood_episodes, steps=args.steps, size=96,
            seed=1000 + seed, goal_directed=0.5, nuisance_shift=True,
        )
        id_ = generate_episodes(
            n=args.id_episodes, steps=args.steps, size=96,
            seed=2000 + seed, goal_directed=0.5, nuisance_shift=False,
        )

        if (seed, "random", "ood") not in done:
            r = evaluate_random(ood, n_episodes=len(ood), steps=args.steps, seed=seed)
            append_row(rows_path, {"seed": seed, "arm": "random", "split": "ood", **r})
            done.add((seed, "random", "ood"))
        if (seed, "random", "id") not in done:
            r = evaluate_random(id_, n_episodes=len(id_), steps=args.steps, seed=seed)
            append_row(rows_path, {"seed": seed, "arm": "random", "split": "id", **r})
            done.add((seed, "random", "id"))

        for arm in ("temporal", "good"):
            path = models / f"s{seed}-{arm}.safetensors"
            if not path.exists():
                raise FileNotFoundError(path)
            model = load_jepa(path, image_size=96)
            for split, episodes in (("ood", ood), ("id", id_)):
                key = (seed, arm, split)
                if key in done:
                    print(f"  {arm}/{split}: already done", flush=True)
                    continue
                r = evaluate_planner(
                    model, episodes, n_episodes=len(episodes),
                    steps=args.steps, horizon=args.horizon, seed=seed,
                )
                append_row(rows_path, {
                    "seed": seed, "arm": arm, "split": split, **r,
                })
                done.add(key)
                print(
                    f"  {arm}/{split}: success {r['success']:.2f} "
                    f"dist {r['mean_dist']:.3f}",
                    flush=True,
                )

    summarize(out)


if __name__ == "__main__":
    main()
