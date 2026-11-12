#!/usr/bin/env python3
"""Collision 2: capacity autopsy of Seed 4 vs Seed 1."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from leparallax.data import build_probe_samples
from leparallax.probes import collect_latents
from leparallax.train import load_jepa
from leparallax.world import generate_episodes


def spectrum_metrics(Z: np.ndarray) -> dict:
    Zc = Z - Z.mean(axis=0, keepdims=True)
    cov = Zc.T @ Zc / max(len(Z) - 1, 1)
    eig = np.linalg.eigvalsh(cov)[::-1]
    eig = np.clip(eig, 0, None)
    total = eig.sum() + 1e-12
    p = eig / total
    nz = p[p > 0]
    effective_rank = float(np.exp(-np.sum(nz * np.log(nz))))
    participation = float((total**2) / (np.sum(eig**2) + 1e-12))
    top1 = float(eig[0] / total)
    return {
        "effective_rank": effective_rank,
        "participation_ratio": participation,
        "top1_share": top1,
        "n": int(len(Z)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 4])
    ap.add_argument("--out", default="runs/capacity-autopsy")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in args.seeds:
        print(f"\n=== seed {seed} ===", flush=True)
        a = generate_episodes(
            n=10, steps=32, size=96, seed=11000 + seed,
            goal_directed=0.5, nuisance_mode="normal",
        )
        b = generate_episodes(
            n=10, steps=32, size=96, seed=12000 + seed,
            goal_directed=0.5, nuisance_mode="shift",
        )
        for arm in ("temporal", "good"):
            for stage, path in (
                ("pre", f"runs/h1-confirm/models/s{seed}-{arm}.safetensors"),
                ("post", f"runs/h3/models/s{seed}-{arm}.safetensors"),
            ):
                if not pathlib.Path(path).exists():
                    print("  missing", path, flush=True)
                    continue
                model = load_jepa(path, image_size=96)
                for domain, eps in (("A", a), ("B", b)):
                    samples = build_probe_samples(eps, per_episode=16, seed=seed)
                    Z, _, _ = collect_latents(model, samples)
                    m = spectrum_metrics(Z)
                    row = {"seed": seed, "arm": arm, "stage": stage,
                           "domain": domain, **m}
                    rows.append(row)
                    print(
                        f"  {arm}/{stage}/{domain}: eff_rank {m['effective_rank']:.2f} "
                        f"part {m['participation_ratio']:.2f} "
                        f"top1 {m['top1_share']:.3f}",
                        flush=True,
                    )

    (out / "per_run.json").write_text(json.dumps(rows, indent=2) + "\n")
    lines = [
        "# Capacity autopsy — Seed 1 vs Seed 4",
        "",
        "| seed | arm | stage | domain | effective rank | participation | top-1 share |",
        "|---:|---|---|---|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['seed']} | {r['arm']} | {r['stage']} | {r['domain']} | "
            f"{r['effective_rank']:.2f} | {r['participation_ratio']:.2f} | "
            f"{r['top1_share']:.3f} |"
        )
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {out / 'report.md'}")


if __name__ == "__main__":
    main()
