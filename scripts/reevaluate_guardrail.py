#!/usr/bin/env python3
"""Re-score the Phase 2 state guardrail against the oracle ceiling."""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]


def load(path):
    return json.loads((ROOT / path).read_text())


def main():
    oracle = load("runs/oracle-state/per_seed.json")
    o_xy = float(np.mean([r["r2_xy"] for r in oracle]))
    o_4d = float(np.mean([r["r2_state"] for r in oracle]))
    print(f"oracle x,y R² {o_xy:.3f}; 4-D state R² {o_4d:.3f}")
    print(f"adjusted 4-D guardrail (0.90×oracle): {0.90 * o_4d:.3f}")

    for label, path in (
        ("142k frame-head, 40ep", "runs/phase2-framestate/per_seed.json"),
        ("340k, 60ep cosine", "runs/phase2-capacity-w010/per_seed.json"),
    ):
        rows = load(path)
        by = {r["arm"]: r for r in rows}
        print(f"\n{label}")
        for arm in ("temp_matched", "sym_head", "asym_clean", "asym_vol_bounded"):
            r = by.get(arm)
            if r is None:
                print(f"  {arm}: missing")
                continue
            print(
                f"  {arm}: state R² {r['state_r2']:.3f} "
                f"(adjusted pass {r['state_r2'] >= 0.90 * o_4d}), "
                f"Δasym {r['delta_asym']:+.3f}, "
                f"std {r.get('latent_std', float('nan')):.3f}"
            )
        if "asym_clean" in by and "sym_head" in by:
            margin = by["asym_clean"]["delta_asym"] - by["sym_head"]["delta_asym"]
            print(f"  direction margin: {margin:+.3f} (need ≥ +0.100)")


if __name__ == "__main__":
    main()
