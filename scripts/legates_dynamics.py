#!/usr/bin/env python3
"""Collision 3: Le-Gates-style residual adapters on the dynamics boundary.

Freeze the H1 good-view encoder and base predictor, adapt only a small residual
adapter on shifted dynamics B, and compare three gates:

  global  scalar gate, same for every input
  input   gate from (z, a)
  error   gate from (z, a, previous base-predictor error)

The error signal is causal: it uses the previous transition's prediction error,
which an online agent has already observed.
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

from leparallax.data import build_temporal_samples, collate_temporal
from leparallax.train import load_jepa
from leparallax.world import generate_episodes

SHIFT_DYNAMICS = np.array([0.65, 0.32], dtype=np.float32)


class Adapter(nn.Module):
    def __init__(self, latent_dim: int = 64, action_dim: int = 2, hidden: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim + action_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, latent_dim)
        # Zero-init the residual: the adapted model starts identical to the base.
        self.fc3.weight = mx.zeros_like(self.fc3.weight)
        self.fc3.bias = mx.zeros_like(self.fc3.bias)

    def __call__(self, z: mx.array, action: mx.array) -> mx.array:
        x = mx.concatenate([z, action], axis=-1)
        x = nn.relu(self.fc1(x))
        x = nn.relu(self.fc2(x))
        return self.fc3(x)


class Gate(nn.Module):
    def __init__(self, mode: str, latent_dim: int = 64, action_dim: int = 2,
                 hidden: int = 32):
        super().__init__()
        self.mode = mode
        if mode == "global":
            self.logit = mx.array([0.0])
        else:
            in_dim = latent_dim + action_dim + (1 if mode == "error" else 0)
            self.fc1 = nn.Linear(in_dim, hidden)
            self.fc2 = nn.Linear(hidden, 1)

    def __call__(self, z: mx.array, action: mx.array, err: mx.array) -> mx.array:
        if self.mode == "global":
            return mx.sigmoid(self.logit)
        x = mx.concatenate([z, action], axis=-1)
        if self.mode == "error":
            x = mx.concatenate([x, err[:, None]], axis=-1)
        x = nn.relu(self.fc1(x))
        return mx.sigmoid(self.fc2(x))


class AdaptedModel(nn.Module):
    def __init__(self, base, adapter: Adapter, gate: Gate):
        super().__init__()
        self.base = base
        self.adapter = adapter
        self.gate = gate

    def encode(self, obs: mx.array) -> mx.array:
        return self.base.encode(obs)

    def predict(self, z: mx.array, action: mx.array, err: mx.array) -> mx.array:
        base = self.base.predict(z, action)
        g = self.gate(z, action, err)
        return base + 0.1 * g * self.adapter(z, action)


def base_errors(model, samples, n_episodes: int, steps: int = 32,
                batch: int = 64) -> np.ndarray:
    errs = []
    for start in range(0, len(samples), batch):
        idx = np.arange(start, min(start + batch, len(samples)))
        obs, act, nxt, _, _, _, _ = collate_temporal(samples, idx)
        z = model.encode(mx.array(obs))
        zn = model.encode(mx.array(nxt))
        zp = model.predict(z, mx.array(act))
        e = mx.mean((zp - zn) ** 2, axis=1)
        mx.eval(e)
        errs.append(np.array(e))
    e = np.concatenate(errs)
    return e.reshape(n_episodes, steps - 1)


def previous_error(e: np.ndarray) -> np.ndarray:
    prev = np.zeros_like(e)
    prev[:, 1:] = e[:, :-1]
    return prev.reshape(-1)


def predict_mse(model, samples, prev_err: np.ndarray, use_adapter: bool,
                batch: int = 64) -> float:
    total, n = 0.0, 0
    for start in range(0, len(samples), batch):
        idx = np.arange(start, min(start + batch, len(samples)))
        obs, act, nxt, _, _, _, _ = collate_temporal(samples, idx)
        z = model.encode(mx.array(obs))
        zn = model.encode(mx.array(nxt))
        if use_adapter:
            zp = model.predict(z, mx.array(act), mx.array(prev_err[idx]))
        else:
            zp = model.predict(z, mx.array(act))
        loss = mx.mean((zp - zn) ** 2)
        mx.eval(loss)
        total += float(loss) * len(idx)
        n += len(idx)
    return total / max(n, 1)


def gate_values(model, samples, prev_err: np.ndarray, batch: int = 64) -> np.ndarray:
    vals = []
    for start in range(0, len(samples), batch):
        idx = np.arange(start, min(start + batch, len(samples)))
        obs, act, _, _, _, _, _ = collate_temporal(samples, idx)
        z = model.base.encode(mx.array(obs))
        g = model.gate(z, mx.array(act), mx.array(prev_err[idx]))
        if g.size == 1:
            g = mx.broadcast_to(g.reshape(-1)[0], (len(idx),))
        else:
            g = g.reshape(-1)
        mx.eval(g)
        vals.append(np.array(g))
    return np.concatenate(vals)


def residual_norms(model, samples, prev_err: np.ndarray,
                   batch: int = 64) -> np.ndarray:
    """Effective adapter residual magnitude per sample."""
    vals = []
    for start in range(0, len(samples), batch):
        idx = np.arange(start, min(start + batch, len(samples)))
        obs, act, _, _, _, _, _ = collate_temporal(samples, idx)
        z = model.base.encode(mx.array(obs))
        g = model.gate(z, mx.array(act), mx.array(prev_err[idx]))
        if g.size == 1:
            g = mx.broadcast_to(g.reshape(-1)[0], (len(idx),))
        else:
            g = g.reshape(-1)
        r = model.adapter(z, mx.array(act))
        eff = 0.1 * g[:, None] * r
        norm = mx.sqrt(mx.mean(eff ** 2, axis=1))
        mx.eval(norm)
        vals.append(np.array(norm))
    return np.concatenate(vals)


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Rank-based AUC: P(gate_B > gate_A)."""
    x = np.concatenate([pos, neg])
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(x) + 1)
    # Average ties.
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    n_pos, n_neg = len(pos), len(neg)
    return float((ranks[:n_pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def train_adapter(model, samples, prev_err: np.ndarray, epochs: int,
                  batch_size: int, lr: float, seed: int, out_path,
                  anchor_samples=None, anchor_prev_err=None,
                  lambda_anchor: float = 0.0) -> None:
    def loss_fn(model, obs, act, nxt, err, aobs, aact, anxt, aerr):
        z = model.base.encode(obs)
        zn = model.base.encode(nxt)
        zp = model.predict(z, act, err)
        loss = mx.mean((zp - zn) ** 2)
        if lambda_anchor > 0.0:
            za = model.base.encode(aobs)
            zan = model.base.encode(anxt)
            base_a = model.base.predict(za, aact)
            adapted_a = model.predict(za, aact, aerr)
            # Distill the frozen base on A: the adapter should be identity there.
            loss = loss + lambda_anchor * mx.mean((adapted_a - base_a) ** 2)
        return loss

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
            if lambda_anchor > 0.0:
                ai = rng.integers(0, len(anchor_samples), size=len(idx))
                aobs, aact, anxt, _, _, _, _ = collate_temporal(anchor_samples, ai)
                loss, grads = loss_and_grad(
                    model, mx.array(obs), mx.array(act), mx.array(nxt),
                    mx.array(prev_err[idx]), mx.array(aobs), mx.array(aact),
                    mx.array(anxt), mx.array(anchor_prev_err[ai]),
                )
            else:
                loss, grads = loss_and_grad(
                    model, mx.array(obs), mx.array(act), mx.array(nxt),
                    mx.array(prev_err[idx]), mx.array(obs), mx.array(act),
                    mx.array(nxt), mx.array(prev_err[idx]),
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--train-episodes", type=int, default=80)
    ap.add_argument("--test-episodes", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--anchor-episodes", type=int, default=0)
    ap.add_argument("--lambda-anchor", type=float, default=1.0)
    ap.add_argument("--out", default="runs/legates-dynamics")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    (out / "models").mkdir(parents=True, exist_ok=True)
    manifest = {
        "prereg": "docs/PREREG-COLLISION-2.md",
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
        a_err = base_errors(base, a_samples, len(a_test))
        b_err = base_errors(base, b_samples, len(b_test))
        b_train_err = base_errors(base, b_train_samples, len(b_train))
        mean_b = float(b_train_err.mean())
        std_b = float(b_train_err.std() + 1e-8)
        norm = lambda e: np.clip((previous_error(e) - mean_b) / std_b, -5.0, 5.0)
        a_prev = norm(a_err)
        b_prev = norm(b_err)

        anchor_samples = None
        anchor_prev = None
        if args.anchor_episodes > 0:
            a_anchor = a_train[: args.anchor_episodes]
            anchor_samples = build_temporal_samples(a_anchor)
            anchor_err = base_errors(base, anchor_samples, len(a_anchor))
            anchor_prev = norm(anchor_err)

        a_pre = predict_mse(base, a_samples, a_prev, use_adapter=False)
        b_pre = predict_mse(base, b_samples, b_prev, use_adapter=False)
        print(f"  frozen: A {a_pre:.5f}  B {b_pre:.5f}", flush=True)

        full = load_jepa(
            f"runs/h3b/models/s{seed}-good.safetensors", image_size=96
        )
        a_full = predict_mse(full, a_samples, a_prev, use_adapter=False)
        b_full = predict_mse(full, b_samples, b_prev, use_adapter=False)
        rows.append({
            "seed": seed, "arm": "full-ft",
            "a_pre": a_pre, "b_pre": b_pre,
            "a_post": a_full, "b_post": b_full,
            "a_rel_forget": (a_full - a_pre) / a_pre,
            "b_rel_gain": (b_pre - b_full) / b_pre,
            "gate_mean_a": None, "gate_mean_b": None, "gate_auc": None,
        })
        print(f"  full-ft: A {a_full:.5f}  B {b_full:.5f}", flush=True)

        for mode in ("global", "input", "error"):
            base = load_jepa(
                f"runs/h1-confirm/models/s{seed}-good.safetensors", image_size=96
            )
            base.freeze()
            adapter = Adapter()
            gate = Gate(mode)
            model = AdaptedModel(base, adapter, gate)
            adapter.unfreeze()
            gate.unfreeze()
            n_params = sum(p.size for _, p in tree_flatten(model.trainable_parameters()))
            print(f"  {mode}: {n_params} trainable params", flush=True)
            train_adapter(
                model, b_train_samples, norm(b_train_err), epochs=args.epochs,
                batch_size=args.batch_size, lr=args.lr, seed=seed,
                out_path=out / "models" / f"s{seed}-{mode}.safetensors",
                anchor_samples=anchor_samples, anchor_prev_err=anchor_prev,
                lambda_anchor=args.lambda_anchor if anchor_samples is not None else 0.0,
            )
            a_post = predict_mse(model, a_samples, a_prev, use_adapter=True)
            b_post = predict_mse(model, b_samples, b_prev, use_adapter=True)
            gm_a = gate_values(model, a_samples, a_prev)
            gm_b = gate_values(model, b_samples, b_prev)
            g_auc = auc(gm_b, gm_a)
            ra = residual_norms(model, a_samples, a_prev)
            rb = residual_norms(model, b_samples, b_prev)
            r_auc = auc(rb, ra)
            row = {
                "seed": seed, "arm": mode,
                "a_pre": a_pre, "b_pre": b_pre,
                "a_post": a_post, "b_post": b_post,
                "a_rel_forget": (a_post - a_pre) / a_pre,
                "b_rel_gain": (b_pre - b_post) / b_pre,
                "gate_mean_a": float(gm_a.mean()),
                "gate_mean_b": float(gm_b.mean()),
                "gate_auc": g_auc,
                "resid_a": float(ra.mean()),
                "resid_b": float(rb.mean()),
                "resid_auc": r_auc,
                "trainable_params": n_params,
            }
            rows.append(row)
            print(
                f"  {mode}: A {a_post:.5f}  B {b_post:.5f}  "
                f"forget {row['a_rel_forget']:+.3f}  gain {row['b_rel_gain']:+.3f}  "
                f"gate A/B {row['gate_mean_a']:.3f}/{row['gate_mean_b']:.3f}  "
                f"AUC {g_auc:.3f}  resid A/B {row['resid_a']:.4f}/"
                f"{row['resid_b']:.4f} AUC {r_auc:.3f}",
                flush=True,
            )

    (out / "per_seed.json").write_text(json.dumps(rows, indent=2) + "\n")

    lines = [
        "# Le-Gates on the dynamics boundary",
        "",
        f"Anchor: {args.anchor_episodes} A episodes, lambda {args.lambda_anchor}.",
        "",
        "| arm | A rel forget (mean) | B rel gain (mean) | pass seeds | gate A | gate B | gate AUC | resid A | resid B | resid AUC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    def mean_or_nan(values):
        values = [v for v in values if v is not None]
        return float(np.mean(values)) if values else float("nan")

    summary = {}
    for arm in ("full-ft", "global", "input", "error"):
        rs = [r for r in rows if r["arm"] == arm]
        if not rs:
            continue
        passes = sum(
            1 for r in rs
            if r["b_rel_gain"] >= 0.20 and r["a_rel_forget"] <= 0.05
        )
        gm_a = mean_or_nan([r["gate_mean_a"] for r in rs])
        gm_b = mean_or_nan([r["gate_mean_b"] for r in rs])
        g_auc = mean_or_nan([r["gate_auc"] for r in rs])
        ra = mean_or_nan([r.get("resid_a") for r in rs])
        rb = mean_or_nan([r.get("resid_b") for r in rs])
        r_auc = mean_or_nan([r.get("resid_auc") for r in rs])
        summary[arm] = {
            "a_rel_forget": float(np.mean([r["a_rel_forget"] for r in rs])),
            "b_rel_gain": float(np.mean([r["b_rel_gain"] for r in rs])),
            "passes": int(passes),
            "gate_mean_a": float(gm_a),
            "gate_mean_b": float(gm_b),
            "gate_auc": float(g_auc),
            "resid_a": float(ra),
            "resid_b": float(rb),
            "resid_auc": float(r_auc),
        }
        lines.append(
            f"| {arm} | {summary[arm]['a_rel_forget']:+.3f} | "
            f"{summary[arm]['b_rel_gain']:+.3f} | {passes}/{len(rs)} | "
            f"{gm_a:.3f} | {gm_b:.3f} | {g_auc:.3f} | "
            f"{ra:.4f} | {rb:.4f} | {r_auc:.3f} |"
        )

    global_fail = len([r for r in rows if r["arm"] == "global"]) - summary["global"]["passes"]
    input_fail = len([r for r in rows if r["arm"] == "input"]) - summary["input"]["passes"]
    error_passes = summary["error"]["passes"]
    error_gap = summary["error"]["gate_mean_b"] - summary["error"]["gate_mean_a"]
    bet_3a = global_fail >= 3
    bet_3b = input_fail >= 3 and summary["input"]["gate_auc"] <= 0.65
    bet_3c = error_passes >= 4 and error_gap >= 0.20
    lines += [
        "",
        "## Bets",
        "",
        f"- **3a global fails:** {global_fail}/5 seeds fail "
        f"(needed ≥3). Verdict: **{'holds' if bet_3a else 'falsified'}**.",
        f"- **3b input fails:** {input_fail}/5 seeds fail, AUC "
        f"{summary['input']['gate_auc']:.3f} (needed fail ≥3 and AUC ≤0.65). "
        f"Verdict: **{'holds' if bet_3b else 'falsified'}**.",
        f"- **3c error succeeds:** {error_passes}/5 seeds pass, gate gap "
        f"{error_gap:+.3f} (needed ≥4 and gap ≥0.20). "
        f"Verdict: **{'holds' if bet_3c else 'falsified'}**.",
        "",
        "Per-seed values are in `per_seed.json`.",
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines[-14:]))
    print(f"wrote {out / 'report.md'}")


if __name__ == "__main__":
    main()
