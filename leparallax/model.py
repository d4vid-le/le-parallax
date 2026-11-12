"""Small action-conditioned JEPA in MLX.

Encoder: a shared CNN over the history frames, concatenated and projected to a
64-d latent. Predictor: an MLP over `[z, action]` that predicts a residual delta.
Regularizer: SIGReg (sketched isotropic Gaussian) over the latents.
"""

from __future__ import annotations

import numpy as np
import mlx.core as mx
import mlx.nn as nn

LATENT_DIM = 64
HISTORY = 3
IMAGE_SIZE = 96


class FrameEncoder(nn.Module):
    def __init__(self, latent_dim: int = LATENT_DIM, image_size: int = IMAGE_SIZE,
                 width: float = 1.0):
        super().__init__()
        c1 = max(1, int(32 * width))
        c2 = max(1, int(64 * width))
        c3 = max(1, int(64 * width))
        feat = 2 * c3
        self.feat_dim = feat
        self.conv1 = nn.Conv2d(3, c1, 4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(c1, c2, 4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(c2, c3, 3, stride=1, padding=1)
        # Three stride-2 convolutions: size -> size/2 -> size/4 -> size/8.
        grid = image_size // 4
        self.grid = grid
        # Spatial softmax: one expected (x, y) per conv channel.
        self.fc = nn.Linear(feat, feat)
        self.ln = nn.LayerNorm(feat)
        self.out = nn.Linear(feat, latent_dim)
        self.out_ln = nn.LayerNorm(latent_dim)

    def features(self, x: mx.array) -> mx.array:
        x = nn.relu(self.conv1(x))
        x = nn.relu(self.conv2(x))
        x = nn.relu(self.conv3(x))
        b, h, w, c = x.shape
        att = mx.softmax(x.reshape(b, h * w, c), axis=1).reshape(b, h, w, c)
        xs = mx.array(np.linspace(-1.0, 1.0, h, dtype=np.float32))
        ys = mx.array(np.linspace(-1.0, 1.0, w, dtype=np.float32))
        ex = mx.sum(att * xs[None, :, None, None], axis=1)
        ex = mx.sum(ex, axis=1)
        ey = mx.sum(att * ys[None, None, :, None], axis=2)
        ey = mx.sum(ey, axis=1)
        x = mx.concatenate([ex, ey], axis=-1)
        return nn.relu(self.ln(self.fc(x)))

    def __call__(self, x: mx.array) -> mx.array:
        return self.out_ln(self.out(self.features(x)))


class Encoder(nn.Module):
    def __init__(self, latent_dim: int = LATENT_DIM, history: int = HISTORY,
                 image_size: int = IMAGE_SIZE, width: float = 1.0):
        super().__init__()
        self.history = history
        self.frame = FrameEncoder(latent_dim, image_size, width=width)
        self.merge = nn.Linear(history * latent_dim, latent_dim)
        self.ln = nn.LayerNorm(latent_dim)

    def __call__(self, obs: mx.array) -> mx.array:
        # obs: (B, T, H, W, C)
        b, t, h, w, c = obs.shape
        x = obs.reshape(b * t, h, w, c)
        z = self.frame(x).reshape(b, -1)
        return self.ln(self.merge(z))

    def frame_features(self, obs: mx.array) -> mx.array:
        """Features of the current (last) frame, before history merge."""
        return self.frame.features(obs[:, -1])


class Predictor(nn.Module):
    def __init__(self, latent_dim: int = LATENT_DIM, action_dim: int = 2,
                 hidden: int | None = None):
        super().__init__()
        hidden = max(128, latent_dim * 2) if hidden is None else hidden
        self.fc1 = nn.Linear(latent_dim + action_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, latent_dim)

    def __call__(self, z: mx.array, action: mx.array) -> mx.array:
        x = mx.concatenate([z, action], axis=-1)
        x = nn.relu(self.fc1(x))
        x = nn.relu(self.fc2(x))
        delta = self.fc3(x)
        return z + 0.1 * delta


class JEPA(nn.Module):
    def __init__(self, latent_dim: int = LATENT_DIM, image_size: int = IMAGE_SIZE,
                 frame_state_head: bool = False, width: float = 1.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(
            latent_dim, image_size=image_size, width=width
        )
        self.predictor = Predictor(latent_dim)
        self.state_head = nn.Linear(latent_dim, 4)
        # Optional auxiliary head on the current-frame keypoint features, before
        # the history merge and before any asymmetric pair head.
        self.frame_state_head = (
            nn.Linear(self.encoder.frame.feat_dim, 4)
            if frame_state_head else None
        )

    def encode(self, obs: mx.array) -> mx.array:
        return self.encoder(obs)

    def encode_frame_features(self, obs: mx.array) -> mx.array:
        return self.encoder.frame_features(obs)

    def predict(self, z: mx.array, action: mx.array) -> mx.array:
        return self.predictor(z, action)


def _sigreg_constants(dim: int, num_proj: int = 64, knots: int = 17):
    rng = np.random.default_rng(0)
    u = rng.normal(size=(dim, num_proj)).astype(np.float32)
    u /= np.linalg.norm(u, axis=0, keepdims=True) + 1e-8
    t = np.linspace(0.2, 4.0, knots).astype(np.float32)
    w = np.exp(-0.5 * t * t).astype(np.float32)
    return mx.array(u), mx.array(t), mx.array(w)


class SIGReg:
    def __init__(self, dim: int = LATENT_DIM):
        self.u, self.t, self.w = _sigreg_constants(dim)

    def __call__(self, z: mx.array) -> mx.array:
        # z: (B, D); h: (B, M)
        h = z @ self.u
        # (B, K, M)
        ht = h[:, None, :] * self.t[None, :, None]
        cos_mean = mx.mean(mx.cos(ht), axis=0)
        sin_mean = mx.mean(mx.sin(ht), axis=0)
        phi0 = mx.exp(-0.5 * self.t * self.t)[:, None]
        stat = (cos_mean - phi0) ** 2 + sin_mean**2
        return mx.mean(stat * self.w[:, None])


def variance_term(z: mx.array, eps: float = 1e-4) -> mx.array:
    """Keep every latent dimension near unit variance (both directions)."""
    std = mx.sqrt(mx.var(z, axis=0) + eps)
    return mx.mean((std - 1.0) ** 2)


def covariance_term(z: mx.array) -> mx.array:
    """Decorate dimensions: penalize off-diagonal covariance."""
    zc = z - mx.mean(z, axis=0, keepdims=True)
    cov = (zc.T @ zc) / max(z.shape[0] - 1, 1)
    d = cov.shape[0]
    off = cov * (1.0 - mx.eye(d))
    return mx.mean(off ** 2)


def contrastive_term(z_pred: mx.array, z_next: mx.array,
                     temperature: float = 0.1) -> mx.array:
    """InfoNCE: the predicted latent must match its own next latent, not others."""
    eps = 1e-6
    zp = z_pred / (mx.sqrt(mx.sum(z_pred ** 2, axis=1, keepdims=True)) + eps)
    zn = z_next / (mx.sqrt(mx.sum(z_next ** 2, axis=1, keepdims=True)) + eps)
    logits = (zp @ zn.T) / temperature
    b = z_pred.shape[0]
    target = mx.sum(logits * mx.eye(b), axis=1)
    return mx.mean(mx.logsumexp(logits, axis=1) - target)


def make_loss(sigreg: SIGReg, lambda_sig: float = 0.09,
              lambda_var: float = 0.10, lambda_cov: float = 0.01,
              lambda_con: float = 0.10, lambda_state: float = 0.50,
              lambda_multi: float = 0.50, lambda_frame_state: float = 0.50):
    def loss_fn(model: JEPA, obs, action, next_obs, state,
                action2, next_next_obs, state2):
        z = model.encode(obs)
        z_next = model.encode(next_obs)
        z_pred = model.predict(z, action)
        pred = mx.mean((z_pred - z_next) ** 2)

        z2_pred = model.predict(z_pred, action2)
        z2_next = model.encode(next_next_obs)
        pred2 = mx.mean((z2_pred - z2_next) ** 2)

        reg = 0.5 * (sigreg(z) + sigreg(z_next))
        var = 0.5 * (variance_term(z) + variance_term(z_next))
        cov = 0.5 * (covariance_term(z) + covariance_term(z_next))
        con = contrastive_term(z_pred, z_next)
        state_loss = mx.mean((model.state_head(z) - state) ** 2)
        state2_loss = mx.mean((model.state_head(z2_pred) - state2) ** 2)
        frame_state_loss = 0.0
        if model.frame_state_head is not None:
            ff = model.encode_frame_features(obs)
            frame_state_loss = mx.mean(
                (model.frame_state_head(ff) - state) ** 2
            )
        return (pred + lambda_multi * pred2
                + lambda_sig * reg + lambda_var * var
                + lambda_cov * cov + lambda_con * con
                + lambda_state * (state_loss + 0.5 * state2_loss)
                + lambda_frame_state * frame_state_loss)

    return loss_fn
