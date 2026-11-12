#!/usr/bin/env python3
"""Collision 4: Anti-Equivalence on an irreversible one-way door.

Tests whether asymmetric pair supervision produces a directed latent geometry:
A -> B is easier than B -> A, B has lower volume/rank, and the latent encodes
goal reachability after the door closes.

The proposed determinant penalty is implemented as a diagonal log-volume proxy
because MLX has no GPU determinant. The true effective rank is measured post hoc.
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
    build_temporal_samples,
    collate_temporal,
    goal_obs,
    history_obs,
)
from leparallax.model import JEPA, SIGReg, make_loss
from leparallax.probes import _ridge_eval
from leparallax.train import load_jepa
from leparallax.world import HISTORY, generate_episodes, history_indices, render_state

ARM_NAMES = [
    "temp_matched", "sym_head", "asym_clean", "asym_vol_bounded",
    "temporal", "symmetric", "asym", "asym_vol", "asym_wrong",
]
PHASE2_ARMS = ["temp_matched", "sym_head", "asym_clean", "asym_vol_bounded"]


class PairHeads(nn.Module):
    def __init__(self, latent_dim: int = 64, reverse_hidden: int = 32):
        super().__init__()
        self.g1 = nn.Linear(latent_dim, 128)
        self.g2 = nn.Linear(128, latent_dim)
        self.h1 = nn.Linear(latent_dim, reverse_hidden)
        self.h2 = nn.Linear(reverse_hidden, latent_dim)

    def forward(self, z: mx.array) -> mx.array:
        return self.g2(nn.relu(self.g1(z)))

    def reverse(self, z: mx.array) -> mx.array:
        return self.h2(nn.relu(self.h1(z)))


class ArmModel(nn.Module):
    def __init__(self, jepa: JEPA, heads: PairHeads | None = None):
        super().__init__()
        self.jepa = jepa
        self.heads = heads

    def encode(self, obs: mx.array) -> mx.array:
        return self.jepa.encode(obs)


def log_volume(z: mx.array, eps: float = 1e-3) -> mx.array:
    """Diagonal log-volume proxy for det(Cov(z)); GPU-friendly."""
    return mx.mean(mx.log(mx.var(z, axis=0) + eps))


def variance_floor(z: mx.array, gamma: float = 1.0) -> mx.array:
    """VICReg-style floor: every dimension must have std >= gamma."""
    std = mx.sqrt(mx.var(z, axis=0) + 1e-4)
    return mx.mean(mx.maximum(gamma - std, 0.0))


def build_pairs(episodes, irreversible: bool = True,
                post_offsets: tuple[int, ...] = (2, 4)):
    pairs = []
    for ep in episodes:
        sides = ep.states[:, 0] >= 0.5
        crossings = np.where((~sides[:-1]) & sides[1:])[0]
        if len(crossings) == 0:
            continue
        t = int(crossings[0])
        if irreversible:
            for pre in (2, 4):
                for post in post_offsets:
                    ta = max(0, t - pre)
                    tb = min(len(ep.states) - 1, t + post)
                    if ta >= tb:
                        continue
                    if not ep.door_open[ta] or ep.door_open[tb]:
                        continue
                    pairs.append((
                        ep.frames[history_indices(ta)],
                        ep.frames[history_indices(tb)],
                        ep.states[ta],
                        ep.states[tb],
                    ))
        else:
            # Matched reversible control: both states on the right, door closed,
            # with the same time gaps as the irreversible pairs (4/6/8).
            for gap in (4, 6, 8):
                for ta in range(t + 2, len(ep.states) - gap):
                    tb = ta + gap
                    if ep.door_open[ta] or ep.door_open[tb]:
                        continue
                    if ep.states[ta, 0] < 0.5 or ep.states[tb, 0] < 0.5:
                        continue
                    pairs.append((
                        ep.frames[history_indices(ta)],
                        ep.frames[history_indices(tb)],
                        ep.states[ta],
                        ep.states[tb],
                    ))
                    break
    return pairs


def collate_pairs(pairs, idx):
    oa = np.stack([pairs[i][0] for i in idx]).astype(np.float32) / 255.0
    ob = np.stack([pairs[i][1] for i in idx]).astype(np.float32) / 255.0
    sa = np.stack([pairs[i][2] for i in idx]).astype(np.float32)
    sb = np.stack([pairs[i][3] for i in idx]).astype(np.float32)
    return oa, ob, sa, sb


def build_probe(episodes):
    obs, goal_obs_list, states, door, side, reach = [], [], [], [], [], []
    for ep in episodes:
        g_obs = goal_obs(ep)
        for t in range(len(ep.states)):
            obs.append(ep.frames[history_indices(t)])
            door_t = bool(ep.door_open[t])
            goal_obs_list.append(g_obs)
            states.append(ep.states[t])
            door.append(door_t)
            s = bool(ep.states[t, 0] >= 0.5)
            g = bool(ep.goal[0] >= 0.5)
            side.append(s)
            reach.append(door_t or (s == g))
    return (
        np.stack(obs).astype(np.float32) / 255.0,
        np.stack(goal_obs_list).astype(np.float32) / 255.0,
        np.stack(states).astype(np.float32),
        np.array(door, dtype=np.float32),
        np.array(side, dtype=np.float32),
        np.array(reach, dtype=np.float32),
    )


def encode_all(model, obs, batch: int = 64) -> np.ndarray:
    out = []
    for start in range(0, len(obs), batch):
        idx = np.arange(start, min(start + batch, len(obs)))
        z = model.encode(mx.array(obs[idx]))
        mx.eval(z)
        out.append(np.array(z))
    return np.concatenate(out)


def effective_rank(Z: np.ndarray) -> float:
    Zc = Z - Z.mean(axis=0, keepdims=True)
    cov = Zc.T @ Zc / max(len(Z) - 1, 1)
    eig = np.clip(np.linalg.eigvalsh(cov)[::-1], 0, None)
    total = eig.sum() + 1e-12
    p = eig / total
    nz = p[p > 0]
    return float(np.exp(-np.sum(nz * np.log(nz))))


def residual_rank(Z: np.ndarray, S: np.ndarray, alpha: float = 1.0) -> float:
    """Effective rank after projecting out the state-predictive subspace."""
    mu = Z.mean(axis=0, keepdims=True)
    sd = Z.std(axis=0, keepdims=True) + 1e-6
    Zs = (Z - mu) / sd
    Zb = np.concatenate([Zs, np.ones((len(Zs), 1))], axis=1)
    W = np.linalg.solve(Zb.T @ Zb + alpha * np.eye(Zb.shape[1]), Zb.T @ S)
    W = W[: Z.shape[1]].T  # (4, 64)
    P = W.T @ np.linalg.solve(W @ W.T + 1e-3 * np.eye(W.shape[0]), W)
    Zres = Zs @ (np.eye(Z.shape[1]) - P)
    return effective_rank(Zres)


def ridge_fit(X: np.ndarray, Y: np.ndarray, alpha: float = 1.0):
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True) + 1e-6
    Xs = (X - mu) / sd
    Xb = np.concatenate([Xs, np.ones((len(Xs), 1))], axis=1)
    W = np.linalg.solve(Xb.T @ Xb + alpha * np.eye(Xb.shape[1]), Xb.T @ Y)
    return mu, sd, W


def ridge_predict(params, X):
    mu, sd, W = params
    Xs = (X - mu) / sd
    Xb = np.concatenate([Xs, np.ones((len(Xs), 1))], axis=1)
    return Xb @ W


def ridge_auc(Xtr, Ytr, Xte, Yte) -> float:
    params = ridge_fit(Xtr, Ytr[:, None])
    pred = ridge_predict(params, Xte)[:, 0]
    return auc(pred[Yte > 0.5], pred[Yte <= 0.5])


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    x = np.concatenate([pos, neg])
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(x) + 1)
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    n_pos, n_neg = len(pos), len(neg)
    return float((ranks[:n_pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def train_arm(model, temporal_samples, pairs, arm: str, epochs: int,
              batch_size: int, pair_batch: int, lr: float, seed: int,
              lambda_vol: float, lambda_var: float = 1.0,
              pair_weight: float = 0.25,
              state_weight: float = 0.25, warmup: int = 5,
              verbose: bool = True):
    base_loss = make_loss(SIGReg(), 0.09)
    rng = np.random.default_rng(seed)
    pw = [0.0]
    sw = [0.0]

    def loss_fn(model, obs, act, nxt, state, act2, nxt2, state2,
                oa, ob, sa, sb):
        base = base_loss(
            model.jepa, obs, act, nxt, state, act2, nxt2, state2
        )
        if arm in ("temporal", "temp_matched") or pw[0] == 0.0:
            return base
        za = model.jepa.encode(oa)
        zb = model.jepa.encode(ob)
        state_pair = 0.5 * (
            mx.mean((model.jepa.state_head(za) - sa) ** 2)
            + mx.mean((model.jepa.state_head(zb) - sb) ** 2)
        )
        if arm == "symmetric":
            pair = mx.mean((za - zb) ** 2)
            return base + pw[0] * pair + sw[0] * state_pair
        if arm == "sym_head":
            # Same heads, same capacity, but both directions for both heads.
            pair = 0.5 * (
                mx.mean((model.heads.forward(za) - zb) ** 2)
                + mx.mean((model.heads.forward(zb) - za) ** 2)
                + mx.mean((model.heads.reverse(zb) - za) ** 2)
                + mx.mean((model.heads.reverse(za) - zb) ** 2)
            )
            return base + pw[0] * pair + sw[0] * state_pair
        pair = (
            mx.mean((model.heads.forward(za) - zb) ** 2)
            + mx.mean((model.heads.reverse(zb) - za) ** 2)
        )
        if arm in ("asym", "asym_clean"):
            return base + pw[0] * pair + sw[0] * state_pair
        if arm == "asym_vol_bounded":
            return (
                base + pw[0] * pair + sw[0] * state_pair
                + lambda_vol * log_volume(zb)
                + lambda_var * variance_floor(zb)
            )
        return (
            base + pw[0] * pair + sw[0] * state_pair
            + lambda_vol * log_volume(zb)
        )

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    optimizer = optim.AdamW(learning_rate=lr, weight_decay=1e-4)
    n = len(temporal_samples)
    for epoch in range(1, epochs + 1):
        pw[0] = 0.0 if epoch <= warmup else pair_weight
        sw[0] = 0.0 if epoch <= warmup else state_weight
        perm = rng.permutation(n)
        total, seen = 0.0, 0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            obs, act, nxt, state, act2, nxt2, state2 = collate_temporal(
                temporal_samples, idx
            )
            pi = rng.integers(0, len(pairs), size=pair_batch)
            oa, ob, sa, sb = collate_pairs(pairs, pi)
            loss, grads = loss_and_grad(
                model, mx.array(obs), mx.array(act), mx.array(nxt),
                mx.array(state), mx.array(act2), mx.array(nxt2),
                mx.array(state2), mx.array(oa), mx.array(ob),
                mx.array(sa), mx.array(sb),
            )
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
            total += float(loss.item()) * len(idx)
            seen += len(idx)
        if verbose:
            print(
                f"    epoch {epoch}/{epochs}  loss {total / max(seen, 1):.5f}",
                flush=True,
            )


def pair_metrics(model, train_pairs, test_pairs):
    Xa_tr = encode_all(model, np.stack([p[0] for p in train_pairs]).astype(np.float32) / 255.0)
    Xb_tr = encode_all(model, np.stack([p[1] for p in train_pairs]).astype(np.float32) / 255.0)
    Xa_te = encode_all(model, np.stack([p[0] for p in test_pairs]).astype(np.float32) / 255.0)
    Xb_te = encode_all(model, np.stack([p[1] for p in test_pairs]).astype(np.float32) / 255.0)

    f_params = ridge_fit(Xa_tr, Xb_tr)
    r_params = ridge_fit(Xb_tr, Xa_tr)
    pred_b = ridge_predict(f_params, Xa_te)
    pred_a = ridge_predict(r_params, Xb_te)

    def r2(pred, target):
        ss_res = ((target - pred) ** 2).sum(axis=0)
        ss_tot = ((target - target.mean(axis=0)) ** 2).sum(axis=0)
        return float(np.mean(1.0 - ss_res / (ss_tot + 1e-8)))

    r2_f = r2(pred_b, Xb_te)
    r2_r = r2(pred_a, Xa_te)
    err_f = max(1.0 - r2_f, 1e-6)
    err_r = max(1.0 - r2_r, 1e-6)
    rank_a = effective_rank(Xa_te)
    rank_b = effective_rank(Xb_te)
    res_a = residual_rank(Xa_te, np.stack([p[2] for p in test_pairs]))
    res_b = residual_rank(Xb_te, np.stack([p[3] for p in test_pairs]))
    return {
        "r2_forward": r2_f, "r2_reverse": r2_r,
        "err_forward": err_f, "err_reverse": err_r,
        "asym_ratio": err_r / err_f,
        "rank_a": rank_a, "rank_b": rank_b,
        "residual_rank_a": res_a, "residual_rank_b": res_b,
        "norm_a": float(np.mean(np.linalg.norm(Xa_te, axis=1))),
        "norm_b": float(np.mean(np.linalg.norm(Xb_te, axis=1))),
        "std_a": float(np.mean(np.std(Xa_te, axis=0))),
        "std_b": float(np.mean(np.std(Xb_te, axis=0))),
    }


def evaluate(model, train_irr, test_irr, train_rev, test_rev,
             probe_train, probe_test):
    di = pair_metrics(model, train_irr, test_irr)
    dr = pair_metrics(model, train_rev, test_rev)

    obs_tr, goal_tr, s_tr, door_tr, side_tr, reach_tr = probe_train
    obs_te, goal_te, s_te, door_te, side_te, reach_te = probe_test
    Zstate_tr = encode_all(model, obs_tr)
    Zstate_te = encode_all(model, obs_te)
    Zgoal_tr = encode_all(model, goal_tr)
    Zgoal_te = encode_all(model, goal_te)
    Ztr = np.concatenate([Zstate_tr, Zgoal_tr], axis=1)
    Zte = np.concatenate([Zstate_te, Zgoal_te], axis=1)

    state_r2 = float(np.mean(_ridge_eval(
        Zstate_tr, s_tr, Zstate_te, s_te
    )))
    door_auc = ridge_auc(Ztr, door_tr, Zte, door_te)
    side_auc = ridge_auc(Ztr, side_tr, Zte, side_te)
    reach_auc = ridge_auc(Ztr, reach_tr, Zte, reach_te)

    n_te = np.linalg.norm(Zstate_te, axis=1)
    reachable = reach_te > 0.5
    norm_reach = float(n_te[reachable].mean()) if reachable.any() else float("nan")
    norm_unreach = float(n_te[~reachable].mean()) if (~reachable).any() else float("nan")
    norm_auc = auc(-n_te[~reachable], -n_te[reachable])
    rank_reach = effective_rank(Zstate_te[reachable]) if reachable.sum() > 32 else float("nan")
    rank_unreach = effective_rank(Zstate_te[~reachable]) if (~reachable).sum() > 32 else float("nan")
    rank_auc = auc(
        -np.linalg.norm(Zstate_te[~reachable], axis=1),
        -np.linalg.norm(Zstate_te[reachable], axis=1),
    )

    delta_rank_irr = di["rank_b"] - di["rank_a"]
    delta_rank_rev = dr["rank_b"] - dr["rank_a"]
    delta_res_irr = di["residual_rank_b"] - di["residual_rank_a"]
    delta_res_rev = dr["residual_rank_b"] - dr["residual_rank_a"]
    return {
        "asym_irr": di["asym_ratio"],
        "asym_rev": dr["asym_ratio"],
        "delta_asym": di["asym_ratio"] - dr["asym_ratio"],
        "r2_forward_irr": di["r2_forward"],
        "r2_reverse_irr": di["r2_reverse"],
        "rank_a_irr": di["rank_a"], "rank_b_irr": di["rank_b"],
        "rank_a_rev": dr["rank_a"], "rank_b_rev": dr["rank_b"],
        "delta_rank_irr": delta_rank_irr,
        "delta_rank_rev": delta_rank_rev,
        "delta_delta_rank": delta_rank_irr - delta_rank_rev,
        "residual_rank_a_irr": di["residual_rank_a"],
        "residual_rank_b_irr": di["residual_rank_b"],
        "residual_rank_a_rev": dr["residual_rank_a"],
        "residual_rank_b_rev": dr["residual_rank_b"],
        "delta_residual_irr": delta_res_irr,
        "delta_residual_rev": delta_res_rev,
        "delta_delta_residual": delta_res_irr - delta_res_rev,
        "norm_a_irr": di["norm_a"], "norm_b_irr": di["norm_b"],
        "std_a_irr": di["std_a"], "std_b_irr": di["std_b"],
        "latent_norm": float(np.mean(np.linalg.norm(Zstate_te, axis=1))),
        "latent_std": float(np.mean(np.std(Zstate_te, axis=0))),
        "state_r2": state_r2,
        "door_auc": door_auc, "side_auc": side_auc, "reach_auc": reach_auc,
        "norm_reach": norm_reach, "norm_unreach": norm_unreach,
        "norm_auc": norm_auc, "rank_reach": rank_reach,
        "rank_unreach": rank_unreach, "rank_auc": rank_auc,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--train-episodes", type=int, default=40)
    ap.add_argument("--test-episodes", type=int, default=20)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--pair-batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lambda-vol", type=float, default=0.05)
    ap.add_argument("--lambda-var", type=float, default=1.0)
    ap.add_argument("--pair-weight", type=float, default=0.25)
    ap.add_argument("--state-weight", type=float, default=0.25)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--post-offsets", type=int, nargs="+", default=[2, 4])
    ap.add_argument("--occlude-door-right", action="store_true")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--arms", nargs="+", default=PHASE2_ARMS)
    ap.add_argument("--out", default="runs/asymmetry")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    (out / "models").mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps({
        "prereg": "docs/PREREG-ASYMMETRY.md",
        "args": vars(args),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, indent=2) + "\n")

    rows = []
    for seed in args.seeds:
        print(f"\n=== seed {seed} ===", flush=True)
        train_eps = generate_episodes(
            n=args.train_episodes, steps=args.steps, size=96, seed=seed,
            goal_directed=0.5, nuisance_mode="normal", one_way_door=True,
            occlude_door_right=args.occlude_door_right,
        )
        test_eps = generate_episodes(
            n=args.test_episodes, steps=args.steps, size=96, seed=9000 + seed,
            goal_directed=0.5, nuisance_mode="normal", one_way_door=True,
            occlude_door_right=args.occlude_door_right,
        )
        post_offsets = tuple(args.post_offsets)
        train_irr = build_pairs(
            train_eps, irreversible=True, post_offsets=post_offsets
        )
        test_irr = build_pairs(
            test_eps, irreversible=True, post_offsets=post_offsets
        )
        train_rev = build_pairs(train_eps, irreversible=False)
        test_rev = build_pairs(test_eps, irreversible=False)
        print(
            f"  irr pairs train {len(train_irr)} test {len(test_irr)} | "
            f"rev pairs train {len(train_rev)} test {len(test_rev)}",
            flush=True,
        )
        if len(train_irr) < 8 or len(test_irr) < 4:
            raise RuntimeError("not enough irreversible pairs; increase episodes")
        if len(train_rev) < 8 or len(test_rev) < 4:
            raise RuntimeError("not enough reversible pairs; increase episodes")
        temporal = build_temporal_samples(train_eps)
        probe_train = build_probe(train_eps)
        probe_test = build_probe(test_eps)

        for arm in args.arms:
            if args.eval_only:
                model = ArmModel(
                    load_jepa(
                        str(out / "models" / f"s{seed}-{arm}.safetensors"),
                        image_size=96,
                    ),
                    None,
                )
            else:
                mx.random.seed(seed)
                jepa = JEPA(image_size=96)
                heads = (
                    None
                    if arm in ("temporal", "temp_matched", "symmetric")
                    else PairHeads()
                )
                model = ArmModel(jepa, heads)
                vol = 0.0
                if arm in ("asym_vol", "asym_vol_bounded"):
                    vol = args.lambda_vol
                elif arm == "asym_wrong":
                    vol = -args.lambda_vol
                print(f"  {arm}: training", flush=True)
                train_arm(
                    model, temporal, train_irr, arm, epochs=args.epochs,
                    batch_size=args.batch_size, pair_batch=args.pair_batch,
                    lr=args.lr, seed=seed, lambda_vol=vol,
                    lambda_var=args.lambda_var,
                    pair_weight=args.pair_weight,
                    state_weight=args.state_weight,
                    warmup=args.warmup,
                )
            metrics = evaluate(
                model, train_irr, test_irr, train_rev, test_rev,
                probe_train, probe_test,
            )
            row = {"seed": seed, "arm": arm, **metrics}
            rows.append(row)
            print(
                f"  {arm}: asym irr/rev {metrics['asym_irr']:.3f}/"
                f"{metrics['asym_rev']:.3f} Δ {metrics['delta_asym']:+.3f}  "
                f"ΔΔrank {metrics['delta_delta_rank']:+.3f}  "
                f"state {metrics['state_r2']:.3f}  "
                f"reach AUC {metrics['reach_auc']:.3f}",
                flush=True,
            )
            if not args.eval_only:
                model.jepa.save_weights(
                    str(out / "models" / f"s{seed}-{arm}.safetensors")
                )

    (out / "per_seed.json").write_text(json.dumps(rows, indent=2) + "\n")

    lines = [
        "# Anti-Equivalence collision — one-way door",
        "",
        "| arm | asym irr | asym rev | Δasym | ΔΔrank | ΔΔres | state R² | reach AUC | latent std | norm A/B |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in args.arms:
        rs = [r for r in rows if r["arm"] == arm]
        if not rs:
            continue
        m = lambda k: float(np.mean([r[k] for r in rs]))
        lines.append(
            f"| {arm} | {m('asym_irr'):.3f} | {m('asym_rev'):.3f} | "
            f"{m('delta_asym'):+.3f} | {m('delta_delta_rank'):+.3f} | "
            f"{m('delta_delta_residual'):+.3f} | {m('state_r2'):.3f} | "
            f"{m('reach_auc'):.3f} | {m('latent_std'):.3f} | "
            f"{m('norm_a_irr'):.2f}/{m('norm_b_irr'):.2f} |"
        )
    lines += ["", "Per-seed values are in `per_seed.json`."]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    print(f"wrote {out / 'report.md'}")


if __name__ == "__main__":
    main()
