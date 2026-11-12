#!/usr/bin/env python3
"""Oracle check: standalone ~50k CNN on the same 3-frame door observations.

This is the empirical ceiling for state regression with the same data and input
history as the JEPA, but no JEPA/pair objective. Reports R² for x,y and for the
full [x,y,vx,vy] state.
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

from leparallax.world import generate_episodes, history_indices


class OracleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 24, 4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(24, 40, 4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(40, 40, 3, stride=1, padding=1)
        self.feat_dim = 2 * 40
        self.fc = nn.Linear(3 * self.feat_dim, 64)
        self.head2 = nn.Linear(64, 2)
        self.head4 = nn.Linear(64, 4)

    def features(self, obs: mx.array) -> mx.array:
        b, t, h, w, c = obs.shape
        x = obs.reshape(b * t, h, w, c)
        x = nn.relu(self.conv1(x))
        x = nn.relu(self.conv2(x))
        x = nn.relu(self.conv3(x))
        _, hh, ww, cc = x.shape
        att = mx.softmax(x.reshape(b * t, hh * ww, cc), axis=1)
        att = att.reshape(b * t, hh, ww, cc)
        xs = mx.array(np.linspace(-1.0, 1.0, hh, dtype=np.float32))
        ys = mx.array(np.linspace(-1.0, 1.0, ww, dtype=np.float32))
        ex = mx.sum(att * xs[None, :, None, None], axis=1)
        ex = mx.sum(ex, axis=1)
        ey = mx.sum(att * ys[None, None, :, None], axis=2)
        ey = mx.sum(ey, axis=1)
        f = mx.concatenate([ex, ey], axis=-1).reshape(b, t * self.feat_dim)
        return nn.relu(self.fc(f))

    def __call__(self, obs: mx.array):
        f = self.features(obs)
        return self.head2(f), self.head4(f)


def build_samples(episodes):
    obs, states = [], []
    for ep in episodes:
        for t in range(len(ep.states)):
            obs.append(ep.frames[history_indices(t)])
            states.append(ep.states[t])
    return (
        np.stack(obs).astype(np.float32) / 255.0,
        np.stack(states).astype(np.float32),
    )


def r2(pred, target):
    ss_res = ((target - pred) ** 2).sum(axis=0)
    ss_tot = ((target - target.mean(axis=0)) ** 2).sum(axis=0)
    return 1.0 - ss_res / (ss_tot + 1e-8)


def train_one(train_obs, train_state, test_obs, test_state, seed: int,
              epochs: int = 100, batch: int = 32, lr: float = 1e-3,
              save_path=None):
    mx.random.seed(seed)
    model = OracleCNN()
    rng = np.random.default_rng(seed)

    def loss_fn(model, obs, state):
        p2, p4 = model(obs)
        return (
            mx.mean((p2 - state[:, :2]) ** 2)
            + mx.mean((p4 - state) ** 2)
        )

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    opt = optim.AdamW(learning_rate=lr, weight_decay=1e-4)
    n = len(train_obs)
    for epoch in range(1, epochs + 1):
        lr_t = lr * 0.5 * (
            1.0 + math.cos(math.pi * (epoch - 1) / max(epochs - 1, 1))
        )
        opt.learning_rate = lr_t
        perm = rng.permutation(n)
        total, seen = 0.0, 0
        for start in range(0, n, batch):
            idx = perm[start:start + batch]
            loss, grads = loss_and_grad(
                model, mx.array(train_obs[idx]), mx.array(train_state[idx])
            )
            opt.update(model, grads)
            mx.eval(model.parameters(), opt.state)
            total += float(loss.item()) * len(idx)
            seen += len(idx)
        if epoch % 20 == 0:
            print(
                f"    epoch {epoch}/{epochs} loss {total / max(seen, 1):.5f}",
                flush=True,
            )

    p2, p4 = model(mx.array(test_obs))
    mx.eval(p2, p4)
    p2 = np.array(p2)
    p4 = np.array(p4)
    r2_2 = r2(p2, test_state[:, :2])
    r2_4 = r2(p4, test_state)
    if save_path is not None:
        pathlib.Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        model.save_weights(str(save_path))
    n_params = sum(
        p.size for _, p in __import__("mlx.utils", fromlist=["tree_flatten"]).tree_flatten(
            model.parameters()
        )
    )
    return {
        "seed": seed,
        "params": int(n_params),
        "r2_xy": float(np.mean(r2_2)),
        "r2_xy_dims": r2_2.tolist(),
        "r2_state": float(np.mean(r2_4)),
        "r2_position": float(np.mean(r2_4[:2])),
        "r2_velocity": float(np.mean(r2_4[2:])),
        "r2_state_dims": r2_4.tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--train-episodes", type=int, default=40)
    ap.add_argument("--test-episodes", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--out", default="runs/oracle-state")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    train_eps = generate_episodes(
        n=args.train_episodes, steps=32, size=96, seed=0,
        goal_directed=0.5, nuisance_mode="normal", one_way_door=True,
        occlude_door_right=True,
    )
    test_eps = generate_episodes(
        n=args.test_episodes, steps=32, size=96, seed=9000,
        goal_directed=0.5, nuisance_mode="normal", one_way_door=True,
        occlude_door_right=True,
    )
    train_obs, train_state = build_samples(train_eps)
    test_obs, test_state = build_samples(test_eps)
    print(
        f"train {train_obs.shape} test {test_obs.shape} "
        f"state {train_state.shape}",
        flush=True,
    )

    rows = []
    for seed in args.seeds:
        print(f"=== seed {seed} ===", flush=True)
        row = train_one(
            train_obs, train_state, test_obs, test_state,
            seed=seed, epochs=args.epochs,
            save_path=(
                out / f"state_stream_s{seed}.safetensors"
                if args.save else None
            ),
        )
        rows.append(row)
        print(
            f"  oracle: params {row['params']}  "
            f"xy {row['r2_xy']:.3f}  state {row['r2_state']:.3f}  "
            f"pos {row['r2_position']:.3f}  vel {row['r2_velocity']:.3f}",
            flush=True,
        )

    (out / "per_seed.json").write_text(json.dumps(rows, indent=2) + "\n")
    lines = [
        "# Oracle state ceiling — standalone CNN",
        "",
        "| seed | params | x,y R² | 4D state R² | position R² | velocity R² |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['seed']} | {r['params']} | {r['r2_xy']:.3f} | "
            f"{r['r2_state']:.3f} | {r['r2_position']:.3f} | "
            f"{r['r2_velocity']:.3f} |"
        )
    lines += [
        "",
        f"- mean x,y R²: **{np.mean([r['r2_xy'] for r in rows]):.3f}**",
        f"- mean 4D state R²: **{np.mean([r['r2_state'] for r in rows]):.3f}**",
        f"- mean position R²: **{np.mean([r['r2_position'] for r in rows]):.3f}**",
        f"- mean velocity R²: **{np.mean([r['r2_velocity'] for r in rows]):.3f}**",
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    print(f"wrote {out / 'report.md'}")


if __name__ == "__main__":
    main()
