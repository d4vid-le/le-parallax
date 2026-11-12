"""Training loop for the small JEPA."""

from __future__ import annotations

import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from .data import TemporalSample, ViewPair, collate_temporal, collate_views
from .model import JEPA, SIGReg, make_loss


def train_jepa(
    samples: list[TemporalSample],
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 3e-4,
    lambda_sig: float = 0.09,
    seed: int = 0,
    verbose: bool = True,
    save_path: str | Path | None = None,
    image_size: int = 96,
    view_pairs: list[ViewPair] | None = None,
    lambda_view: float = 0.0,
    lambda_view_state: float = 0.25,
    model: JEPA | None = None,
) -> JEPA:
    if model is None:
        mx.random.seed(seed)
        model = JEPA(image_size=image_size)
    sigreg = SIGReg()
    loss_fn = make_loss(sigreg, lambda_sig)

    def full_loss(model, obs, act, nxt, state, act2, nxt2, state2,
                  v1, v2, vstate):
        base = loss_fn(model, obs, act, nxt, state, act2, nxt2, state2)
        if lambda_view <= 0.0:
            return base
        z1 = model.encode(v1)
        z2 = model.encode(v2)
        view = mx.mean((z1 - z2) ** 2)
        vstate_loss = 0.5 * (
            mx.mean((model.state_head(z1) - vstate) ** 2)
            + mx.mean((model.state_head(z2) - vstate) ** 2)
        )
        return base + lambda_view * view + lambda_view_state * vstate_loss

    loss_and_grad = nn.value_and_grad(model, full_loss)
    optimizer = optim.AdamW(learning_rate=lr, weight_decay=1e-4)
    rng = np.random.default_rng(seed)

    n = len(samples)
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        perm = rng.permutation(n)
        total = 0.0
        seen = 0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            obs, act, nxt, state, act2, nxt2, state2 = collate_temporal(samples, idx)
            if view_pairs is not None and lambda_view > 0.0:
                vi = rng.integers(0, len(view_pairs), size=len(idx))
                v1, v2, vstate = collate_views(view_pairs, vi)
            else:
                v1, v2, vstate = obs, obs, state
            loss, grads = loss_and_grad(
                model, mx.array(obs), mx.array(act), mx.array(nxt),
                mx.array(state), mx.array(act2), mx.array(nxt2), mx.array(state2),
                mx.array(v1), mx.array(v2), mx.array(vstate)
            )
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
            total += float(loss.item()) * len(idx)
            seen += len(idx)
        if verbose:
            print(
                f"epoch {epoch:3d}  loss {total / max(seen, 1):.5f}  "
                f"{time.time() - t0:.1f}s",
                flush=True,
            )

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        model.save_weights(str(save_path))
    return model


def load_jepa(path: str | Path, image_size: int = 96,
              frame_state_head: bool = False, latent_dim: int = 64,
              width: float = 1.0) -> JEPA:
    model = JEPA(
        image_size=image_size, frame_state_head=frame_state_head,
        latent_dim=latent_dim, width=width,
    )
    model.load_weights(str(path))
    return model
