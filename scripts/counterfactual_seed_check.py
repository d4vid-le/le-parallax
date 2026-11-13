#!/usr/bin/env python3
"""Check the counterfactual re-score: per-timestep and per-seed.

The earlier 0.766 used seed 0 and the last frame. This script reports every seed
and every timestep, so we know whether the score is a pre-crossing artifact or a
post-crossing/current-frame belief result.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from natural_belief import BeliefGRU, auc, build_cf_raw, predict
from leparallax.train import load_jepa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="runs/natural-belief")
    ap.add_argument("--jepa", default="runs/dual-stream-jepa/models/s0-asym_clean.safetensors")
    ap.add_argument("--pairs", type=int, default=200)
    ap.add_argument("--out", default="runs/counterfactual-seed-check")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cf_x, cf_y = build_cf_raw(args.pairs, 9000)
    rows = []
    for seed in (0, 1, 2):
        path = pathlib.Path(args.dir) / f"belief_s{seed}.safetensors"
        if not path.exists():
            print("missing", path, flush=True)
            continue
        jepa = load_jepa(args.jepa, image_size=96)
        model = BeliefGRU(jepa.encoder.frame)
        model.load_weights(str(path))
        p = predict(model, cf_x)
        per = []
        for t in range(p.shape[1]):
            per.append(auc(p[cf_y > 0.5, t], p[cf_y <= 0.5, t]))
        row = {"seed": seed, "per_timestep": per, "last": per[-1]}
        rows.append(row)
        print(
            f"seed {seed}: " + " ".join(f"t{t} {a:.3f}" for t, a in enumerate(per)),
            flush=True,
        )

    (out / "report.json").write_text(json.dumps(rows, indent=2) + "\n")
    lines = [
        "# Counterfactual re-score — per timestep / per seed",
        "",
        "| seed | t0 | t1 | t2 | t3 | t4 | t5 (last) |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['seed']} | " + " | ".join(f"{a:.3f}" for a in r["per_timestep"]) + " |"
        )
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    print(f"wrote {out / 'report.md'}")


if __name__ == "__main__":
    main()
