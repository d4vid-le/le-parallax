#!/usr/bin/env python3
"""Collision 1b: adversarial nuisance stress test + counterfactual alignment.

Part A diagnoses whether good-view invariance survives held-out, shifted, and
extreme nuisance. Part B tests whether a counterfactual nuisance alignment
(CNA) pipeline buys OOD robustness beyond ordinary fine-tuning.
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

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from leparallax.data import (
    ProbeSample,
    build_probe_samples,
    build_temporal_samples,
    collate_probe,
    collate_temporal,
    history_obs,
)
from leparallax.model import SIGReg, covariance_term, make_loss, variance_term
from leparallax.probes import _ridge_eval, collect_latents
from leparallax.train import load_jepa, train_jepa
from leparallax.world import generate_episodes, history_indices, sample_nuisance_shift

CORNERS = [
    (1.10, 0.35, 0.20, +0.08),
    (0.80, 0.20, 0.80, -0.08),
    (1.10, 0.20, 0.80, +0.08),
    (0.80, 0.35, 0.20, -0.08),
]


def build_conditions(test_eps, per_episode: int, seed: int):
    rng = np.random.default_rng(seed + 700)
    conditions = {"normal": [], "heldout_swap": [], "shifted": [], "extreme": []}
    counter = 0
    for i, ep in enumerate(test_eps):
        n = min(per_episode, len(ep.states))
        for t in rng.choice(len(ep.states), size=n, replace=False):
            t = int(t)
            conditions["normal"].append(ProbeSample(
                obs=ep.frames[history_indices(t)],
                state=ep.states[t],
                nuisance=ep.nuisance,
            ))
            other = test_eps[(i + 1 + int(rng.integers(0, len(test_eps) - 1))) % len(test_eps)]
            if other is ep:
                other = test_eps[(i + 1) % len(test_eps)]
            conditions["heldout_swap"].append(ProbeSample(
                obs=history_obs(ep, t, other.nuisance),
                state=ep.states[t],
                nuisance=other.nuisance,
            ))
            shift = sample_nuisance_shift(rng)
            conditions["shifted"].append(ProbeSample(
                obs=history_obs(ep, t, shift),
                state=ep.states[t],
                nuisance=shift,
            ))
            strength, bg, puck, cam = CORNERS[counter % len(CORNERS)]
            counter += 1
            angle = rng.uniform(0.0, 2.0 * np.pi)
            extreme = np.concatenate([
                [np.sin(angle), np.cos(angle), strength],
                np.full(3, bg, dtype=np.float32),
                np.full(3, puck, dtype=np.float32),
                np.full(2, cam, dtype=np.float32),
            ]).astype(np.float32)
            conditions["extreme"].append(ProbeSample(
                obs=history_obs(ep, t, extreme),
                state=ep.states[t],
                nuisance=extreme,
            ))
    return conditions


def evaluate(model, train_samples, conditions):
    Xtr, Ytr, _ = collect_latents(model, train_samples)
    out = {}
    for name, samples in conditions.items():
        X, Y, _ = collect_latents(model, samples)
        r2 = _ridge_eval(Xtr, Ytr, X, Y)
        out[name] = {
            "state": float(np.mean(r2)),
            "position": float(np.mean(r2[:2])),
            "velocity": float(np.mean(r2[2:])),
        }
    return out


def build_counterfactual_triples(episodes, per_episode: int, seed: int):
    rng = np.random.default_rng(seed + 900)
    triples = []
    for i, ep in enumerate(episodes):
        n = min(per_episode, len(ep.states))
        for t in rng.choice(len(ep.states), size=n, replace=False):
            t = int(t)
            other = episodes[int(rng.integers(0, len(episodes)))]
            if other is ep:
                other = episodes[(i + 1) % len(episodes)]
            shift = sample_nuisance_shift(rng)
            triples.append((
                ep.frames[history_indices(t)],
                history_obs(ep, t, other.nuisance),
                history_obs(ep, t, shift),
                ep.states[t],
            ))
    return triples


def collate_triples(triples, indices):
    o1 = np.stack([triples[i][0] for i in indices]).astype(np.float32) / 255.0
    o2 = np.stack([triples[i][1] for i in indices]).astype(np.float32) / 255.0
    o3 = np.stack([triples[i][2] for i in indices]).astype(np.float32) / 255.0
    state = np.stack([triples[i][3] for i in indices]).astype(np.float32)
    return o1, o2, o3, state


def train_cna(
    model,
    temporal_samples,
    triples,
    epochs: int = 5,
    batch_size: int = 32,
    lr: float = 1e-4,
    seed: int = 0,
    out_path=None,
    lambda_align: float = 1.0,
):
    sigreg = SIGReg()
    loss_fn = make_loss(sigreg, 0.09)

    def full_loss(model, obs, act, nxt, state, act2, nxt2, state2,
                  o1, o2, o3, cf_state):
        base = loss_fn(model, obs, act, nxt, state, act2, nxt2, state2)
        z1 = model.encode(o1)
        z2 = model.encode(o2)
        z3 = model.encode(o3)
        align = (
            mx.mean((z1 - z2) ** 2)
            + mx.mean((z1 - z3) ** 2)
            + mx.mean((z2 - z3) ** 2)
        )
        state_cf = (
            mx.mean((model.state_head(z1) - cf_state) ** 2)
            + mx.mean((model.state_head(z2) - cf_state) ** 2)
            + mx.mean((model.state_head(z3) - cf_state) ** 2)
        ) / 3.0
        reg = (sigreg(z1) + sigreg(z2) + sigreg(z3)) / 3.0
        var = (variance_term(z1) + variance_term(z2) + variance_term(z3)) / 3.0
        cov = (covariance_term(z1) + covariance_term(z2) + covariance_term(z3)) / 3.0
        return (
            base
            + lambda_align * align
            + 0.5 * state_cf
            + 0.09 * reg
            + 0.10 * var
            + 0.01 * cov
        )

    loss_and_grad = nn.value_and_grad(model, full_loss)
    optimizer = optim.AdamW(learning_rate=lr, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    n = len(temporal_samples)
    for epoch in range(1, epochs + 1):
        perm = rng.permutation(n)
        total, seen = 0.0, 0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            obs, act, nxt, state, act2, nxt2, state2 = collate_temporal(
                temporal_samples, idx
            )
            ti = rng.integers(0, len(triples), size=len(idx))
            o1, o2, o3, cf_state = collate_triples(triples, ti)
            loss, grads = loss_and_grad(
                model, mx.array(obs), mx.array(act), mx.array(nxt),
                mx.array(state), mx.array(act2), mx.array(nxt2), mx.array(state2),
                mx.array(o1), mx.array(o2), mx.array(o3), mx.array(cf_state),
            )
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
            total += float(loss.item()) * len(idx)
            seen += len(idx)
        print(
            f"    cna epoch {epoch}/{epochs}  loss {total / max(seen, 1):.5f}",
            flush=True,
        )
    if out_path is not None:
        pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        model.save_weights(str(out_path))
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--train-episodes", type=int, default=80)
    ap.add_argument("--test-episodes", type=int, default=10)
    ap.add_argument("--per-episode", type=int, default=16)
    ap.add_argument("--cf-per-episode", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lambda-align", type=float, default=1.0)
    ap.add_argument("--out", default="runs/adversarial-nuisance")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    (out / "models").mkdir(parents=True, exist_ok=True)
    manifest = {
        "prereg": "docs/PREREG-COLLISION-2.md",
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
        probe_train = build_probe_samples(
            a_train, per_episode=args.per_episode, seed=seed
        )
        conditions = build_conditions(a_test, args.per_episode, seed)

        for arm in ("temporal", "good"):
            model = load_jepa(
                f"runs/h1-confirm/models/s{seed}-{arm}.safetensors",
                image_size=96,
            )
            metrics = evaluate(model, probe_train, conditions)
            rows.append({"seed": seed, "arm": arm, "stage": "frozen", **{
                f"{k}_{m}": metrics[k][m]
                for k in conditions for m in ("state", "position", "velocity")
            }})
            print(
                f"  {arm}/frozen: "
                + " | ".join(
                    f"{k} {metrics[k]['state']:.3f}" for k in conditions
                ),
                flush=True,
            )

        good_path = f"runs/h1-confirm/models/s{seed}-good.safetensors"
        temporal_samples = build_temporal_samples(a_train)

        # normal-ft: extra training on normal only.
        model = load_jepa(good_path, image_size=96)
        model = train_jepa(
            temporal_samples, epochs=args.epochs, batch_size=args.batch_size,
            lr=args.lr, seed=seed, image_size=96, model=model,
            save_path=out / "models" / f"s{seed}-normal-ft.safetensors",
            verbose=False,
        )
        metrics = evaluate(model, probe_train, conditions)
        rows.append({"seed": seed, "arm": "good", "stage": "normal-ft", **{
            f"{k}_{m}": metrics[k][m]
            for k in conditions for m in ("state", "position", "velocity")
        }})
        print(
            f"  good/normal-ft: "
            + " | ".join(f"{k} {metrics[k]['state']:.3f}" for k in conditions),
            flush=True,
        )

        # mix-ft: ordinary temporal fine-tuning on normal + shifted episodes.
        mixed = a_train[: args.train_episodes // 2] + b_train[: args.train_episodes // 2]
        model = load_jepa(good_path, image_size=96)
        model = train_jepa(
            build_temporal_samples(mixed), epochs=args.epochs,
            batch_size=args.batch_size, lr=args.lr, seed=seed,
            image_size=96, model=model,
            save_path=out / "models" / f"s{seed}-mix-ft.safetensors",
            verbose=False,
        )
        metrics = evaluate(model, probe_train, conditions)
        rows.append({"seed": seed, "arm": "good", "stage": "mix-ft", **{
            f"{k}_{m}": metrics[k][m]
            for k in conditions for m in ("state", "position", "velocity")
        }})
        print(
            f"  good/mix-ft: "
            + " | ".join(f"{k} {metrics[k]['state']:.3f}" for k in conditions),
            flush=True,
        )

        # CNA: counterfactual triples + alignment loss.
        triples = build_counterfactual_triples(
            a_train, args.cf_per_episode, seed
        )
        model = load_jepa(good_path, image_size=96)
        model = train_cna(
            model, temporal_samples, triples, epochs=args.epochs,
            batch_size=args.batch_size, lr=args.lr, seed=seed,
            out_path=out / "models" / f"s{seed}-cna.safetensors",
            lambda_align=args.lambda_align,
        )
        metrics = evaluate(model, probe_train, conditions)
        rows.append({"seed": seed, "arm": "good", "stage": "cna", **{
            f"{k}_{m}": metrics[k][m]
            for k in conditions for m in ("state", "position", "velocity")
        }})
        print(
            f"  good/cna: "
            + " | ".join(f"{k} {metrics[k]['state']:.3f}" for k in conditions),
            flush=True,
        )

    (out / "per_seed.json").write_text(json.dumps(rows, indent=2) + "\n")

    def mean_state(arm, stage, cond):
        vals = [
            r[f"{cond}_state"] for r in rows
            if r["arm"] == arm and r["stage"] == stage
        ]
        return float(np.mean(vals)) if vals else float("nan")

    def mean_arm(arm, stage, cond):
        vals = [
            r[f"{cond}_state"] for r in rows
            if r["arm"] == arm and r["stage"] == stage
        ]
        return np.array(vals) if vals else np.array([])

    lines = [
        "# Adversarial nuisance stress test",
        "",
        "## Diagnostic (frozen encoders)",
        "",
        "| arm | normal | heldout swap | shifted | extreme |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm in ("temporal", "good"):
        lines.append(
            f"| {arm} | {mean_state(arm, 'frozen', 'normal'):.3f} | "
            f"{mean_state(arm, 'frozen', 'heldout_swap'):.3f} | "
            f"{mean_state(arm, 'frozen', 'shifted'):.3f} | "
            f"{mean_state(arm, 'frozen', 'extreme'):.3f} |"
        )
    lines += [
        "",
        "## Intervention (good-view encoder)",
        "",
        "| stage | normal | heldout swap | shifted | extreme |",
        "|---|---:|---:|---:|---:|",
    ]
    for stage in ("frozen", "normal-ft", "mix-ft", "cna"):
        lines.append(
            f"| {stage} | {mean_state('good', stage, 'normal'):.3f} | "
            f"{mean_state('good', stage, 'heldout_swap'):.3f} | "
            f"{mean_state('good', stage, 'shifted'):.3f} | "
            f"{mean_state('good', stage, 'extreme'):.3f} |"
        )

    t_shift = mean_arm("temporal", "frozen", "shifted")
    g_shift = mean_arm("good", "frozen", "shifted")
    g_norm = mean_arm("good", "frozen", "normal")
    c_shift = mean_arm("good", "cna", "shifted")
    c_norm = mean_arm("good", "cna", "normal")
    m_shift = mean_arm("good", "mix-ft", "shifted")
    f_shift = mean_arm("good", "frozen", "shifted")
    f_norm = mean_arm("good", "frozen", "normal")

    diag_drop = float(np.mean(g_norm - g_shift))
    diag_gap = float(np.mean(g_shift - t_shift))
    diag_bet = diag_drop >= 0.10 and diag_gap <= 0.05
    cna_gain = float(np.mean(c_shift - f_shift))
    cna_forget = float(np.mean(f_norm - c_norm))
    cna_vs_mix = float(np.mean(c_shift - m_shift))
    cna_bet = cna_gain >= 0.10 and cna_forget <= 0.03 and cna_vs_mix >= 0.03

    lines += [
        "",
        "## Bets",
        "",
        f"- **1b-d good OOD drop (normal − shifted): {diag_drop:+.3f}** "
        f"(needed ≥0.10).",
        f"- good shifted − temporal shifted: **{diag_gap:+.3f}** "
        f"(needed ≤0.05).",
        f"- 1b-d verdict: **{'bet holds' if diag_bet else 'bet falsified'}**.",
        "",
        f"- **1b-i CNA shifted gain over frozen: {cna_gain:+.3f}** "
        f"(needed ≥0.10).",
        f"- CNA normal drop: **{cna_forget:+.3f}** (needed ≤0.03).",
        f"- CNA shifted − mix-ft shifted: **{cna_vs_mix:+.3f}** (needed ≥0.03).",
        f"- 1b-i verdict: **{'bet holds' if cna_bet else 'bet falsified'}**.",
        "",
        "Per-seed values are in `per_seed.json`.",
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines[-18:]))
    print(f"wrote {out / 'report.md'}")


if __name__ == "__main__":
    main()
