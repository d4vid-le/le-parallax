"""Training loop for the small JEPA."""

from __future__ import annotations

import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from .data import TemporalSample, collate_temporal
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
) -> JEPA:
    model = JEPA(image_size=image_size)
    sigreg = SIGReg()
    loss_fn = make_loss(sigreg, lambda_sig)
    loss_and_grad = nn.value_and_grad(model, loss_fn)
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
            loss, grads = loss_and_grad(
                model, mx.array(obs), mx.array(act), mx.array(nxt),
                mx.array(state), mx.array(act2), mx.array(nxt2), mx.array(state2)
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


def load_jepa(path: str | Path, image_size: int = 96) -> JEPA:
    model = JEPA(image_size=image_size)
    model.load_weights(str(path))
    return model
