#!/usr/bin/env python3
"""Step 2: natural-episode belief stream with per-step reachability.

Separate belief encoder + GRU, initialized from the frozen JEPA frame encoder but
trained end-to-end on natural hidden-alcove rollouts. Per-step labels, causal GRU
loss over full episodes. Static 3-frame JEPA/state baselines; hard same-side
subset; occlusion horizons; counterfactual re-check.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from counterfactual_belief import make_pair
from oracle_state_cnn import OracleCNN
from leparallax.train import load_jepa
from leparallax.world import generate_episodes, history_indices, sample_nuisance


class BeliefGRU(nn.Module):
    def __init__(self, frame_encoder, hidden: int = 64):
        super().__init__()
        self.frame = frame_encoder
        self.gru = nn.GRU(64, hidden)
        self.head = nn.Linear(hidden, 1)

    def __call__(self, frames):  # (B,T,H,W,C) -> (B,T)
        b, t, h, w, c = frames.shape
        f = self.frame(frames.reshape(b * t, h, w, c)).reshape(b, t, 64)
        out = self.gru(f)
        return self.head(out)[:, :, 0]


class StaticClf(nn.Module):
    def __init__(self, in_dim, hidden=128):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden // 2)
        self.fc3 = nn.Linear(hidden // 2, 1)

    def __call__(self, x):
        x = nn.relu(self.fc1(x))
        x = nn.relu(self.fc2(x))
        return self.fc3(x)


def auc(pos, neg):
    x = np.concatenate([pos, neg])
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(x) + 1)
    n_pos, n_neg = len(pos), len(neg)
    return float((ranks[:n_pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def build_cf_raw(n, seed):
    rng = np.random.default_rng(seed)
    frames, labels = [], []
    for _ in range(n):
        x = rng.uniform(0.62, 0.72)
        y = rng.uniform(0.20, 0.80)
        goal = np.array(
            [rng.uniform(0.85, 0.95), rng.uniform(0.20, 0.80)],
            dtype=np.float32,
        )
        nuisance = sample_nuisance(rng)
        params = (x, y, goal, nuisance)
        for closed in (False, True):
            f, _, lab = make_pair(rng, closed, params)
            frames.append(f)
            labels.append(lab)
    return (
        np.stack(frames).astype(np.float32) / 255.0,
        np.array(labels, dtype=np.float32),
    )


def episode_arrays(episodes):
    frames = np.stack([ep.frames for ep in episodes]).astype(np.float32) / 255.0
    labels = np.stack([
        ep.gate_open.astype(np.float32) if ep.gate_open is not None
        else np.ones(len(ep.states), dtype=np.float32)
        for ep in episodes
    ])
    states = np.stack([ep.states for ep in episodes])
    return frames, labels, states


def static_features(episodes, state_model, jepa):
    sf, jf, labels, sides, goals = [], [], [], [], []
    for ep in episodes:
        goal = ep.goal
        for t in range(len(ep.states)):
            obs = ep.frames[history_indices(t)]
            sf.append(np.concatenate([
                np.array(state_model(
                    mx.array(obs[None].astype(np.float32) / 255.0)
                )[1])[0], goal,
            ]))
            jf.append(np.array(jepa.encode(
                mx.array(obs[None].astype(np.float32) / 255.0)
            ))[0])
            labels.append(
                float(ep.gate_open[t]) if ep.gate_open is not None else 1.0
            )
            sides.append(float(ep.states[t, 0] >= 0.5))
            goals.append(float(goal[0] >= 0.5))
    return (
        np.stack(sf).astype(np.float32),
        np.stack(jf).astype(np.float32),
        np.array(labels, dtype=np.float32),
        np.array(sides, dtype=np.float32),
        np.array(goals, dtype=np.float32),
    )


def train_static(Xtr, ytr, Xte, yte, seed, epochs=50, lr=1e-3):
    rng = np.random.default_rng(seed)
    mx.random.seed(seed)
    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    model = StaticClf(Xtr.shape[1])
    opt = optim.AdamW(learning_rate=lr, weight_decay=1e-3)

    def loss_fn(model, x, y):
        p = mx.sigmoid(model(x))[:, 0]
        return -mx.mean(y * mx.log(p + 1e-6) + (1 - y) * mx.log(1 - p + 1e-6))

    lg = nn.value_and_grad(model, loss_fn)
    for epoch in range(1, epochs + 1):
        opt.learning_rate = lr * 0.5 * (
            1 + math.cos(math.pi * (epoch - 1) / max(epochs - 1, 1))
        )
        p = rng.permutation(len(Xtr))
        for s in range(0, len(Xtr), 64):
            idx = p[s:s + 64]
            loss, grads = lg(model, mx.array(Xtr[idx]), mx.array(ytr[idx]))
            opt.update(model, grads)
            mx.eval(model.parameters(), opt.state)
    return np.array(mx.sigmoid(model(mx.array(Xte)))[:, 0])


def train_belief(frames, labels, frame_encoder, seed, epochs=40, batch=8,
                 lr=1e-3, unreach_weight=3.0, cf_frames=None, cf_labels=None,
                 lambda_cf=1.0):
    rng = np.random.default_rng(seed)
    mx.random.seed(seed)
    model = BeliefGRU(frame_encoder)
    opt = optim.AdamW(learning_rate=lr, weight_decay=1e-4)
    use_cf = cf_frames is not None and lambda_cf > 0

    def loss_fn(model, x, y, cx, cy):
        p = mx.sigmoid(model(x))
        w = 1.0 + unreach_weight * (1.0 - y)
        loss = -mx.mean(
            w * (y * mx.log(p + 1e-6) + (1 - y) * mx.log(1 - p + 1e-6))
        )
        if use_cf:
            cp = mx.sigmoid(model(cx))
            cw = 1.0 + unreach_weight * (1.0 - cy[:, None])
            loss = loss - lambda_cf * mx.mean(
                cw * (
                    cy[:, None] * mx.log(cp + 1e-6)
                    + (1 - cy[:, None]) * mx.log(1 - cp + 1e-6)
                )
            )
        return loss

    lg = nn.value_and_grad(model, loss_fn)
    n = len(frames)
    for epoch in range(1, epochs + 1):
        opt.learning_rate = lr * 0.5 * (
            1 + math.cos(math.pi * (epoch - 1) / max(epochs - 1, 1))
        )
        p = rng.permutation(n)
        total, seen = 0.0, 0
        for s in range(0, n, batch):
            idx = p[s:s + batch]
            if use_cf:
                ci = rng.integers(0, len(cf_frames), size=len(idx))
                cx, cy = cf_frames[ci], cf_labels[ci]
            else:
                cx, cy = frames[idx], labels[idx][:, 0]
            loss, grads = lg(
                model, mx.array(frames[idx]), mx.array(labels[idx]),
                mx.array(cx), mx.array(cy),
            )
            opt.update(model, grads)
            mx.eval(model.parameters(), opt.state)
            total += float(loss.item()) * len(idx)
            seen += len(idx)
        if epoch % 10 == 0:
            print(
                f"    belief epoch {epoch}/{epochs} loss {total / max(seen,1):.4f}",
                flush=True,
            )
    return model


def predict(model, frames):
    p = mx.sigmoid(model(mx.array(frames)))
    mx.eval(p)
    return np.array(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jepa", default="runs/dual-stream-jepa/models/s0-asym_clean.safetensors")
    ap.add_argument("--state", default="runs/dual-stream-state/state_stream_s0.safetensors")
    ap.add_argument("--train-episodes", type=int, default=80)
    ap.add_argument("--test-episodes", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--cf-pairs", type=int, default=400)
    ap.add_argument("--lambda-cf", type=float, default=1.0)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", default="runs/natural-belief")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    train_eps = generate_episodes(
        n=args.train_episodes, steps=32, size=96, seed=0,
        goal_directed=0.5, nuisance_mode="normal",
        hidden_alcove=True, occlude_door_right=True,
    )
    test_eps = generate_episodes(
        n=args.test_episodes, steps=32, size=96, seed=9000,
        goal_directed=0.5, nuisance_mode="normal",
        hidden_alcove=True, occlude_door_right=True,
    )
    Xtr, ytr, _ = episode_arrays(train_eps)
    Xte, yte, str_states = episode_arrays(test_eps)
    cf_tr_frames, cf_tr_labels = build_cf_raw(args.cf_pairs, 123)
    cf_te_frames, cf_te_labels = build_cf_raw(200, 9000)
    print(
        f"train {Xtr.shape} test {Xte.shape} cf {cf_tr_frames.shape}",
        flush=True,
    )

    state_model = OracleCNN()
    state_model.load_weights(args.state)
    jepa_static = load_jepa(args.jepa, image_size=96)
    sf_tr, jf_tr, ysf_tr, side_tr, goal_tr = static_features(
        train_eps, state_model, jepa_static
    )
    sf_te, jf_te, ysf_te, side_te, goal_te = static_features(
        test_eps, state_model, jepa_static
    )
    hard_te = side_te >= 0.5
    print(f"hard test frames {int(hard_te.sum())}/{len(side_te)}", flush=True)

    rows = {}
    for label, Xtr_, Xte_ in (("state_static", sf_tr, sf_te), ("jepa_static", jf_tr, jf_te)):
        p = train_static(Xtr_, ysf_tr, Xte_, ysf_te, 0)
        rows[label] = {
            "overall": auc(p[ysf_te > 0.5], p[ysf_te <= 0.5]),
            "hard": auc(p[hard_te & (ysf_te > 0.5)], p[hard_te & (ysf_te <= 0.5)])
            if (hard_te & (ysf_te <= 0.5)).any() else float("nan"),
        }
        print(label, rows[label], flush=True)
    yte_flat = yte.reshape(-1)

    # Belief model: a fresh separate frame-encoder copy per seed.
    belief_models = []
    for seed in args.seeds:
        print(f"training belief seed {seed}", flush=True)
        belief_jepa = load_jepa(args.jepa, image_size=96)
        model = train_belief(
            Xtr, ytr, belief_jepa.encoder.frame, seed, epochs=args.epochs,
            cf_frames=cf_tr_frames, cf_labels=cf_tr_labels,
            lambda_cf=args.lambda_cf,
        )
        belief_models.append(model)
        p = predict(model, Xte).reshape(-1)
        rows[f"belief_s{seed}"] = {
            "overall": auc(p[yte_flat > 0.5], p[yte_flat <= 0.5]),
            "hard": auc(
                p[hard_te & (yte_flat > 0.5)],
                p[hard_te & (yte_flat <= 0.5)],
            ) if (hard_te & (yte_flat <= 0.5)).any() else float("nan"),
        }
        print(f"belief seed {seed}: {rows[f'belief_s{seed}']}", flush=True)
        model.frame.save_weights(str(out / f"belief_frame_s{seed}.safetensors"))
        model.save_weights(str(out / f"belief_s{seed}.safetensors"))

    # Horizon: left-start post-crossing vs right-start reachable at matched t.
    horizons = [1, 3, 5, 10]
    p_best = predict(belief_models[0], Xte)
    hrows = []
    for h in horizons:
        pos, neg = [], []
        for i, ep in enumerate(test_eps):
            gate = ep.gate_open
            if gate is None:
                continue
            sides = ep.states[:, 0] >= 0.5
            cr = np.where((~sides[:-1]) & sides[1:])[0]
            if len(cr) and ep.states[0, 0] < 0.5:
                tc = int(cr[0])
                if tc + h < len(ep.states):
                    neg.append(p_best[i, tc + h])
            elif ep.states[0, 0] >= 0.5:
                # Reachable right-start; use the same absolute index if valid.
                if h < len(ep.states):
                    pos.append(p_best[i, h])
        if pos and neg:
            hrows.append({
                "h": h,
                "reach_mean": float(np.mean(pos)),
                "unreach_mean": float(np.mean(neg)),
                "auc": auc(np.array(pos), np.array(neg)),
            })
            print(f"horizon +{h}: {hrows[-1]}", flush=True)

    # Counterfactual re-check on held-out raw sequences.
    cf_p = predict(belief_models[0], cf_te_frames)[:, -1]
    cf_auc = auc(
        cf_p[cf_te_labels > 0.5], cf_p[cf_te_labels <= 0.5]
    )
    print(f"counterfactual re-check AUC: {cf_auc:.3f}", flush=True)

    report = {
        "rows": rows,
        "horizons": hrows,
        "counterfactual_auc": cf_auc,
        "hard_frames": int(hard_te.sum()),
    }
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {out / 'report.json'}")


if __name__ == "__main__":
    main()
