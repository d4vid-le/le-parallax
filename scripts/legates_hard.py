#!/usr/bin/env python3
"""Le-Gates v3: hard error gate on the dynamics boundary.

Train the residual adapter on B with the gate fixed at 1. At evaluation, apply
it only when the previous frozen-base error exceeds a threshold calibrated on
the A replay anchor. A is preserved on every non-routed transition.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from legates_dynamics import Adapter, base_errors, previous_error
from leparallax.data import build_temporal_samples, collate_temporal
from leparallax.train import load_jepa
from leparallax.world import generate_episodes

SHIFT_DYNAMICS = np.array([0.65, 0.32], dtype=np.float32)


class HardModel(nn.Module):
    def __init__(self, base, adapter: Adapter):
        super().__init__()
        self.base = base
        self.adapter = adapter

    def predict_full(self, z: mx.array, action: mx.array) -> mx.array:
        return self.base.predict(z, action) + 0.1 * self.adapter(z, action)

    def predict_hard(self, z: mx.array, action: mx.array, g: mx.array) -> mx.array:
        return self.base.predict(z, action) + 0.1 * g[:, None] * self.adapter(z, action)


def train_full_adapter(model, samples, epochs: int, batch_size: int, lr: float,
                       seed: int, out_path) -> None:
    def loss_fn(model, obs, act, nxt):
        z = model.base.encode(obs)
        zn = model.base.encode(nxt)
        zp = model.predict_full(z, act)
        return mx.mean((zp - zn) ** 2)

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    optimizer = optim.AdamW(learning_rate=lr, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    n = len(samples)
    for epoch in range(1, epochs + 1):
        perm = rng.permutation(n)
        total, seen = 0.0, 0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            obs, act, nxt, _, _, _, _ = collate_temporal(samples, idx)
            loss, grads = loss_and_grad(
                model, mx.array(obs), mx.array(act), mx.array(nxt)
            )
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
            total += float(loss.item()) * len(idx)
            seen += len(idx)
        print(
            f"    epoch {epoch}/{epochs}  loss {total / max(seen, 1):.5f}",
            flush=True,
        )
    if out_path is not None:
        pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        model.save_weights(str(out_path))


def eval_hard(model, samples, prev_err: np.ndarray, tau: float,
              batch: int = 64):
    total, n, routed = 0.0, 0, 0
    for start in range(0, len(samples), batch):
        idx = np.arange(start, min(start + batch, len(samples)))
        obs, act, nxt, _, _, _, _ = collate_temporal(samples, idx)
        z = model.base.encode(mx.array(obs))
        zn = model.base.encode(mx.array(nxt))
        g_np = (prev_err[idx] > tau).astype(np.float32)
        g = mx.array(g_np)
        zp = model.predict_hard(z, mx.array(act), g)
        loss = mx.mean((zp - zn) ** 2, axis=1)
        mx.eval(loss)
        total += float(np.sum(np.array(loss)))
        n += len(idx)
        routed += int(g_np.sum())
    return total / max(n, 1), routed / max(n, 1)


def base_mse(model, samples, batch: int = 64) -> float:
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
    ap.add_argument("--anchor-episodes", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--tau-percentile", type=float, default=95.0)
    ap.add_argument("--out", default="runs/legates-hard")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    (out / "models").mkdir(parents=True, exist_ok=True)
    manifest = {
        "prereg": "docs/PREREG-LEGATES-V3.md",
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

        base = load_jepa(
            f"runs/h1-confirm/models/s{seed}-good.safetensors", image_size=96
        )
        base.freeze()
        adapter = Adapter()
        model = HardModel(base, adapter)
        adapter.unfreeze()
        n_params = sum(p.size for _, p in tree_flatten(model.trainable_parameters()))
        print(f"  adapter: {n_params} trainable params", flush=True)
        train_full_adapter(
            model, b_train_samples, epochs=args.epochs,
            batch_size=args.batch_size, lr=args.lr, seed=seed,
            out_path=out / "models" / f"s{seed}-hard.safetensors",
        )

        a_err = base_errors(base, a_samples, len(a_test))
        b_err = base_errors(base, b_samples, len(b_test))
        a_prev = previous_error(a_err)
        b_prev = previous_error(b_err)

        anchor_samples = build_temporal_samples(a_train[: args.anchor_episodes])
        anchor_err = base_errors(base, anchor_samples, args.anchor_episodes)
        anchor_prev = previous_error(anchor_err)
        tau = float(np.percentile(anchor_prev, args.tau_percentile))

        a_pre = base_mse(base, a_samples)
        b_pre = base_mse(base, b_samples)
        a_post, frac_a = eval_hard(model, a_samples, a_prev, tau)
        b_post, frac_b = eval_hard(model, b_samples, b_prev, tau)
        row = {
            "seed": seed,
            "a_pre": a_pre, "b_pre": b_pre,
            "a_post": a_post, "b_post": b_post,
            "a_rel_forget": (a_post - a_pre) / a_pre,
            "b_rel_gain": (b_pre - b_post) / b_pre,
            "tau": tau,
            "frac_a": frac_a,
            "frac_b": frac_b,
            "trainable_params": n_params,
        }
        rows.append(row)
        print(
            f"  hard: A {a_post:.5f}  B {b_post:.5f}  "
            f"forget {row['a_rel_forget']:+.3f}  gain {row['b_rel_gain']:+.3f}  "
            f"frac A/B {frac_a:.3f}/{frac_b:.3f}  tau {tau:.5f}",
            flush=True,
        )

    (out / "per_seed.json").write_text(json.dumps(rows, indent=2) + "\n")

    passes = sum(
        1 for r in rows
        if r["a_rel_forget"] <= 0.05
        and r["b_rel_gain"] >= 0.20
        and r["frac_a"] <= 0.10
        and r["frac_b"] >= 0.70
    )
    lines = [
        "# Le-Gates v3: hard error gate",
        "",
        "| seed | A forget | B gain | frac A | frac B | tau |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['seed']} | {r['a_rel_forget']:+.3f} | "
            f"{r['b_rel_gain']:+.3f} | {r['frac_a']:.3f} | "
            f"{r['frac_b']:.3f} | {r['tau']:.5f} |"
        )
    lines += [
        "",
        f"- mean A forget **{np.mean([r['a_rel_forget'] for r in rows]):+.3f}**",
        f"- mean B gain **{np.mean([r['b_rel_gain'] for r in rows]):+.3f}**",
        f"- mean routing fraction A **{np.mean([r['frac_a'] for r in rows]):.3f}**, "
        f"B **{np.mean([r['frac_b'] for r in rows]):.3f}**",
        f"- pass count **{passes}/{len(rows)}** "
        f"(needed ≥4/5 for the pre-registered endpoint)",
        "",
        "Per-seed values are in `per_seed.json`.",
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines[-9:]))
    print(f"wrote {out / 'report.md'}")


if __name__ == "__main__":
    main()
