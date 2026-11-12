#!/usr/bin/env python3
"""Confirmatory H1 runner (PREREG-v2).

Five seeds, 100 episodes, 96px, three arms: temporal, good view, bad view.
Writes one row per seed/arm to per_seed.jsonl, so an interrupted run resumes.
"""

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

from leparallax.data import (
    build_bad_view_pairs,
    build_temporal_samples,
    build_view_pairs,
)
from leparallax.probes import nuisance_sensitivity, probe_report
from leparallax.train import train_jepa
from leparallax.world import generate_episodes

PREREG = pathlib.Path(__file__).resolve().parents[1] / "docs" / "PREREG-v2.md"


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
    by = {(r["seed"], r["arm"]): r for r in rows}
    seeds = sorted({r["seed"] for r in rows})
    per = []
    for s in seeds:
        t = by.get((s, "temporal"))
        g = by.get((s, "good"))
        b = by.get((s, "bad"))
        if not t or not g:
            continue
        per.append(
            {
                "seed": s,
                "temporal_ratio": t["ratio"],
                "good_ratio": g["ratio"],
                "bad_ratio": b["ratio"] if b else None,
                "rel": g["ratio"] / t["ratio"],
                "temporal_state": t["state"],
                "good_state": g["state"],
                "bad_state": b["state"] if b else None,
            }
        )

    if not per:
        print("no complete seed/arm pairs yet")
        return

    rels = np.array([p["rel"] for p in per], dtype=np.float64)
    rng = np.random.default_rng(0)
    boot = np.array(
        [np.mean(rng.choice(rels, size=len(rels), replace=True)) for _ in range(10_000)]
    )
    ci = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
    mean_rel = float(rels.mean())
    count_le = int((rels <= 0.7).sum())
    state_values = [
        v for p in per for v in (p["temporal_state"], p["good_state"])
    ]
    min_state = float(min(state_values))
    bad_ratios = [p["bad_ratio"] for p in per if p["bad_ratio"] is not None]
    bad_mean = float(np.mean(bad_ratios)) if bad_ratios else None
    complete = len(per) == len(seeds) and all(p["bad_ratio"] is not None for p in per)

    passed = (
        complete
        and mean_rel <= 0.60
        and count_le >= max(4, len(per) - 1)
        and min_state >= 0.75
    )

    report = {
        "prereg_sha256": sha256(PREREG),
        "complete": complete,
        "seeds": per,
        "mean_rel": mean_rel,
        "rel_ci95": ci,
        "count_le_0_7": count_le,
        "min_state": min_state,
        "bad_mean_ratio": bad_mean,
        "verdict": "PASS" if passed else "FAIL",
    }
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    lines = [
        "# H1 confirmatory — result",
        "",
        f"- pre-registration: `{report['prereg_sha256']}`",
        f"- verdict: **{report['verdict']}**",
        f"- complete: {complete}",
        "",
        "| seed | temporal ratio | good ratio | bad ratio | rel | temporal R² | good R² |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for p in per:
        bad = f"{p['bad_ratio']:.3f}" if p['bad_ratio'] is not None else "—"
        lines.append(
            f"| {p['seed']} | {p['temporal_ratio']:.3f} | {p['good_ratio']:.3f} | "
            f"{bad} | {p['rel']:.3f} | {p['temporal_state']:.3f} | {p['good_state']:.3f} |"
        )
    lines += [
        "",
        f"- mean relative ratio: **{mean_rel:.3f}** (95% bootstrap CI {ci[0]:.3f}–{ci[1]:.3f})",
        f"- seeds with rel ≤ 0.7: **{count_le}/{len(per)}**",
        f"- minimum state R²: **{min_state:.3f}**",
        f"- bad-view mean ratio: {bad_mean if bad_mean is not None else 'not run'}",
        "",
        "Endpoint: mean rel ≤ 0.60, at least 4/5 seeds ≤ 0.70, and state R² ≥ 0.75.",
        "",
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({k: report[k] for k in ("verdict", "mean_rel", "rel_ci95", "count_le_0_7", "min_state", "bad_mean_ratio")}, indent=2))
    print(f"wrote {out / 'report.md'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--size", type=int, default=96)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--per-episode", type=int, default=4)
    ap.add_argument("--lambda-view", type=float, default=2.0)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--out", default="runs/h1-confirm")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "models").mkdir(exist_ok=True)

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
    done = {(r["seed"], r["arm"]) for r in load_rows(rows_path)}

    for seed in args.seeds:
        print(f"\n=== seed {seed} ===", flush=True)
        episodes = generate_episodes(
            n=args.episodes, steps=args.steps, size=args.size,
            seed=seed, goal_directed=0.5,
        )
        n_train = int(0.8 * len(episodes))
        n_val = int(0.1 * len(episodes))
        train_eps = episodes[:n_train]
        test_eps = episodes[n_train + n_val :]
        temporal = build_temporal_samples(train_eps)
        good = build_view_pairs(train_eps, per_episode=args.per_episode, seed=seed)
        bad = build_bad_view_pairs(train_eps, per_episode=args.per_episode, seed=seed)

        arms = [
            ("temporal", None, 0.0),
            ("good", good, args.lambda_view),
            ("bad", bad, args.lambda_view),
        ]
        for arm, views, lam in arms:
            if (seed, arm) in done:
                print(f"  {arm}: already done", flush=True)
                continue
            t0 = time.time()
            model = train_jepa(
                temporal,
                epochs=args.epochs,
                batch_size=args.batch_size,
                seed=seed,
                image_size=args.size,
                view_pairs=views,
                lambda_view=lam,
                save_path=out / "models" / f"s{seed}-{arm}.safetensors",
                verbose=False,
            )
            probe = probe_report(model, train_eps, test_eps, per_episode=16, seed=seed)
            sens = nuisance_sensitivity(model, test_eps, per_episode=8, seed=seed)
            row = {
                "seed": seed,
                "arm": arm,
                "state": probe["state"],
                "position": probe["position"],
                "velocity": probe["velocity"],
                "ratio": sens["ratio"],
                "mean_ratio": sens["mean_ratio"],
                "train_s": round(time.time() - t0, 2),
            }
            append_row(rows_path, row)
            done.add((seed, arm))
            print(
                f"  {arm}: state {row['state']:.3f} pos {row['position']:.3f} "
                f"vel {row['velocity']:.3f} ratio {row['ratio']:.3f} "
                f"({row['train_s']}s)",
                flush=True,
            )

    summarize(out)


if __name__ == "__main__":
    main()
