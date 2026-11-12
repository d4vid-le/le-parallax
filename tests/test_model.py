"""Model and training smoke tests."""

import numpy as np
import mlx.core as mx

from leparallax.data import build_temporal_samples, collate_temporal
from leparallax.model import JEPA, SIGReg, make_loss
from leparallax.train import train_jepa
from leparallax.world import generate_episodes


def test_forward_shapes():
    model = JEPA(image_size=32)
    obs = mx.zeros((2, 3, 32, 32, 3))
    z = model.encode(obs)
    assert z.shape == (2, 64)
    action = mx.zeros((2, 2))
    zp = model.predict(z, action)
    assert zp.shape == (2, 64)
    state = model.state_head(z)
    assert state.shape == (2, 4)


def test_loss_is_finite():
    model = JEPA(image_size=32)
    eps = generate_episodes(1, steps=4, size=32, seed=0)
    samples = build_temporal_samples(eps)
    obs, act, nxt, state, act2, nxt2, state2 = collate_temporal(
        samples, np.arange(len(samples))
    )
    loss_fn = make_loss(SIGReg())
    loss = loss_fn(
        model,
        mx.array(obs),
        mx.array(act),
        mx.array(nxt),
        mx.array(state),
        mx.array(act2),
        mx.array(nxt2),
        mx.array(state2),
    )
    mx.eval(loss)
    assert np.isfinite(float(loss))


def test_train_one_epoch():
    eps = generate_episodes(2, steps=4, size=32, seed=0)
    samples = build_temporal_samples(eps)
    model = train_jepa(
        samples, epochs=1, batch_size=2, image_size=32, verbose=False
    )
    assert model is not None
