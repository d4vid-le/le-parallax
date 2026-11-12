#!/usr/bin/env python3
"""H3b: does good-view pretraining protect against a dynamics shift?"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import mlx.core as mx

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from leparallax.data import build_temporal_samples, collate_temporal
from leparallax.train import load_jepa, train_jepa
from leparallax.world import generate_episodes

SHIFT_DYNAMICS = np.array([0.65, 0.32], dtype=np.float32)


def predictor_mse(model, samples, batch: int = 64) -> float:
    total, n = 0.0, 0
    for start in range(0, len(samples), batch):
        idx = np.arange(start, min(start + batch, len(samples)))
        obs, act, nxt, _, _, _, _ = collate_temporal(samples, idx)
        z = model.encode(mx.array(obs))
        zn = model.encode(mx.array(nxt))
        zp = model.predict(z, mx.array(act))
        loss = mx.mean((zp - zn) ** 2)
        mx.eval(loss)
        total += float(loss) * len(idx)
        n += len(idx)
    return total / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--train-episodes", type=int, default=80)
    ap.add_argument("--test-episodes", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out", default="runs/h3b")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    (out / "models").mkdir(parents=True, exist_ok=True)
    manifest = {
        "args": vars(args),
        "shift_dynamics": SHIFT_DYNAMICS.tolist(),
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
            n=args.test_episodes, steps=32, size=96, seed=13000 + seed,
            goal_directed=0.5, nuisance_mode="normal",
        )
        b_train = generate_episodes(
            n=args.train_episodes, steps=32, size=96, seed=3000 + seed,
            goal_directed=0.5, nuisance_mode="normal", dynamics=SHIFT_DYNAMICS,
        )
        b_test = generate_episodes(
            n=args.test_episodes, steps=32, size=96, seed=14000 + seed,
            goal_directed=0.5, nuisance_mode="normal", dynamics=SHIFT_DYNAMICS,
        )
        a_samples = build_temporal_samples(a_test)
        b_samples = build_temporal_samples(b_test)
        b_train_samples = build_temporal_samples(b_train)

        for arm in ("temporal", "good"):
            model = load_jepa(
                f"runs/h1-confirm/models/s{seed}-{arm}.safetensors",
                image_size=96,
            )
            a_pre = predictor_mse(model, a_samples)
            b_pre = predictor_mse(model, b_samples)
            model = train_jepa(
                b_train_samples, epochs=args.epochs,
                batch_size=args.batch_size, lr=args.lr, seed=seed,
                image_size=96, model=model,
                save_path=out / "models" / f"s{seed}-{arm}.safetensors",
                verbose=False,
            )
            a_post = predictor_mse(model, a_samples)
            b_post = predictor_mse(model, b_samples)
            row = {
                "seed": seed, "arm": arm,
                "a_pre": a_pre, "a_post": a_post, "a_forget": a_post - a_pre,
                "b_pre": b_pre, "b_post": b_post, "b_gain": b_pre - b_post,
            }
            rows.append(row)
            print(
                f"  {arm}: A {a_pre:.5f}->{a_post:.5f} "
                f"(forget {row['a_forget']:+.5f}) | "
                f"B {b_pre:.5f}->{b_post:.5f}",
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
            "b_gain_temporal": t["b_gain"], "b_gain_good": g["b_gain"],
        })

    if not per:
        print("no complete pairs")
        return
    diffs = np.array([p["a_diff"] for p in per])
    mean_diff = float(diffs.mean())
    mean_forget_t = float(np.mean([p["a_forget_temporal"] for p in per]))
    mean_forget_g = float(np.mean([p["a_forget_good"] for p in per]))
    mean_b_diff = float(np.mean([p["b_temporal"] - p["b_good"] for p in per]))
    mean_advantage = float(np.mean([
        p["a_temporal"] - p["a_good"] for p in per
    ]))
    mean_b_gain_t = float(np.mean([p["b_gain_temporal"] for p in per]))
    mean_b_gain_g = float(np.mean([p["b_gain_good"] for p in per]))
    boundary_holds = (
        len(per) == 5
        and mean_advantage <= 0.005
        and mean_b_gain_t > 0
        and mean_b_gain_g > 0
    )
    report = {
        "complete": len(per) == 5,
        "seeds": per,
        "mean_a_diff": mean_diff,
        "mean_good_advantage": mean_advantage,
        "mean_forget_temporal": mean_forget_t,
        "mean_forget_good": mean_forget_g,
        "mean_b_diff": mean_b_diff,
        "mean_b_gain_temporal": mean_b_gain_t,
        "mean_b_gain_good": mean_b_gain_g,
        "boundary_holds": boundary_holds,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    lines = [
        "# H3b dynamics shift — result",
        "",
        "| seed | A temp post | A good post | diff | A forget temp | A forget good | B temp post | B good post |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for p in per:
        lines.append(
            f"| {p['seed']} | {p['a_temporal']:.5f} | {p['a_good']:.5f} | "
            f"{p['a_diff']:+.5f} | {p['a_forget_temporal']:+.5f} | "
            f"{p['a_forget_good']:+.5f} | {p['b_temporal']:.5f} | "
            f"{p['b_good']:.5f} |"
        )
    lines += [
        "",
        f"- mean A post difference (good − temporal): **{mean_diff:+.5f}**",
        f"- mean good-view advantage (temporal − good): **{mean_advantage:+.5f}**",
        f"- mean A forgetting: temporal **{mean_forget_t:+.5f}**, "
        f"good **{mean_forget_g:+.5f}**",
        f"- mean B post difference (temporal − good): **{mean_b_diff:+.5f}**",
        f"- mean B gain: temporal **{mean_b_gain_t:+.5f}**, "
        f"good **{mean_b_gain_g:+.5f}**",
        f"- boundary holds: **{boundary_holds}**",
        "",
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[-6:]))
    print(f"wrote {out / 'report.md'}")


if __name__ == "__main__":
    main()
