#!/usr/bin/env python3
"""H3: representation forgetting after adaptation to a nuisance shift."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from leparallax.data import build_temporal_samples
from leparallax.probes import nuisance_sensitivity, probe_report
from leparallax.train import load_jepa, train_jepa
from leparallax.world import generate_episodes

PREREG = pathlib.Path(__file__).resolve().parents[1] / "docs" / "PREREG-H3.md"


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--train-episodes", type=int, default=80)
    ap.add_argument("--test-episodes", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out", default="runs/h3")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    (out / "models").mkdir(parents=True, exist_ok=True)
    manifest = {
        "prereg": str(PREREG),
        "prereg_sha256": sha256(PREREG),
        "args": vars(args),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    rows = []
    for seed in args.seeds:
        print(f"\n=== seed {seed} ===", flush=True)
        a_train = generate_episodes(
            n=args.train_episodes, steps=32, size=96, seed=seed,
            goal_directed=0.5, nuisance_mode="normal",
        )
        a_test = generate_episodes(
            n=args.test_episodes, steps=32, size=96, seed=11000 + seed,
            goal_directed=0.5, nuisance_mode="normal",
        )
        b_train = generate_episodes(
            n=args.train_episodes, steps=32, size=96, seed=2000 + seed,
            goal_directed=0.5, nuisance_mode="shift",
        )
        b_test = generate_episodes(
            n=args.test_episodes, steps=32, size=96, seed=12000 + seed,
            goal_directed=0.5, nuisance_mode="shift",
        )
        b_samples = build_temporal_samples(b_train)

        for arm in ("temporal", "good"):
            model = load_jepa(
                f"runs/h1-confirm/models/s{seed}-{arm}.safetensors",
                image_size=96,
            )
            pre_a = probe_report(model, a_train, a_test, per_episode=16, seed=seed)
            pre_b = probe_report(model, b_train, b_test, per_episode=16, seed=seed)
            pre_sa = nuisance_sensitivity(model, a_test, per_episode=8, seed=seed)
            pre_sb = nuisance_sensitivity(model, b_test, per_episode=8, seed=seed)

            model = train_jepa(
                b_samples, epochs=args.epochs, batch_size=args.batch_size,
                lr=args.lr, seed=seed, image_size=96, model=model,
                save_path=out / "models" / f"s{seed}-{arm}.safetensors",
                verbose=False,
            )

            post_a = probe_report(model, a_train, a_test, per_episode=16, seed=seed)
            post_b = probe_report(model, b_train, b_test, per_episode=16, seed=seed)
            post_sa = nuisance_sensitivity(model, a_test, per_episode=8, seed=seed)
            post_sb = nuisance_sensitivity(model, b_test, per_episode=8, seed=seed)

            row = {
                "seed": seed, "arm": arm,
                "a_pre": pre_a["state"], "a_post": post_a["state"],
                "a_forget": pre_a["state"] - post_a["state"],
                "b_pre": pre_b["state"], "b_post": post_b["state"],
                "b_gain": post_b["state"] - pre_b["state"],
                "a_sens_pre": pre_sa["ratio"], "a_sens_post": post_sa["ratio"],
                "b_sens_pre": pre_sb["ratio"], "b_sens_post": post_sb["ratio"],
            }
            rows.append(row)
            print(
                f"  {arm}: A {pre_a['state']:.3f}->{post_a['state']:.3f} "
                f"(forget {row['a_forget']:+.3f}) | "
                f"B {pre_b['state']:.3f}->{post_b['state']:.3f}",
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
            "a_temporal": t["a_post"], "a_good": g["a_post"],
            "a_diff": g["a_post"] - t["a_post"],
            "a_forget_temporal": t["a_forget"],
            "a_forget_good": g["a_forget"],
            "b_temporal": t["b_post"], "b_good": g["b_post"],
            "b_diff": g["b_post"] - t["b_post"],
            "b_gain_temporal": t["b_gain"], "b_gain_good": g["b_gain"],
            "a_sens_temporal": t["a_sens_post"], "a_sens_good": g["a_sens_post"],
        })

    if not per:
        print("no complete pairs")
        return
    diffs = np.array([p["a_diff"] for p in per])
    rng = np.random.default_rng(0)
    boot = np.array([
        np.mean(rng.choice(diffs, size=len(diffs), replace=True))
        for _ in range(10_000)
    ])
    ci = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
    mean_diff = float(diffs.mean())
    count_higher = int((diffs > 0).sum())
    mean_b_diff = float(np.mean([p["b_diff"] for p in per]))
    mean_a_forget_t = float(np.mean([p["a_forget_temporal"] for p in per]))
    mean_a_forget_g = float(np.mean([p["a_forget_good"] for p in per]))
    complete = len(per) == 5
    passed = (
        complete and mean_diff >= 0.05 and count_higher >= 4
        and mean_b_diff >= -0.05
    )
    report = {
        "prereg_sha256": sha256(PREREG),
        "complete": complete,
        "seeds": per,
        "mean_a_diff": mean_diff,
        "a_diff_ci95": ci,
        "seeds_good_higher": count_higher,
        "mean_b_diff": mean_b_diff,
        "mean_forget_temporal": mean_a_forget_t,
        "mean_forget_good": mean_a_forget_g,
        "verdict": "PASS" if passed else "FAIL",
    }
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    lines = [
        "# H3 forgetting after nuisance shift — result",
        "",
        f"- pre-registration: `{report['prereg_sha256']}`",
        f"- verdict: **{report['verdict']}**",
        f"- complete: {complete}",
        "",
        "| seed | A temp post | A good post | diff | A forget temp | A forget good | B temp post | B good post | B diff |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for p in per:
        lines.append(
            f"| {p['seed']} | {p['a_temporal']:.3f} | {p['a_good']:.3f} | "
            f"{p['a_diff']:+.3f} | {p['a_forget_temporal']:+.3f} | "
            f"{p['a_forget_good']:+.3f} | {p['b_temporal']:.3f} | "
            f"{p['b_good']:.3f} | {p['b_diff']:+.3f} |"
        )
    lines += [
        "",
        f"- mean A post difference: **{mean_diff:+.3f}** "
        f"(95% CI {ci[0]:+.3f} to {ci[1]:+.3f})",
        f"- seeds good higher: **{count_higher}/{len(per)}**",
        f"- mean A forgetting: temporal **{mean_a_forget_t:+.3f}**, "
        f"good **{mean_a_forget_g:+.3f}**",
        f"- mean B post difference: **{mean_b_diff:+.3f}**",
        "",
        "Endpoint: A post diff ≥ +0.05, 4/5 seeds higher, B post diff ≥ −0.05.",
        "",
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[-7:]))
    print(f"wrote {out / 'report.md'}")


if __name__ == "__main__":
    main()
