"""Synthetic 2D world for LeParallax Phase 0.

A puck moves in a unit square under continuous forces. The observation is a
rendered RGB frame. The state is `[x, y, vx, vy]`; the goal is `[gx, gy]`.
Nuisance variables (lighting, background, puck color, camera offset) change the
image but not the task.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

STATE_DIM = 4
ACTION_DIM = 2
NUISANCE_DIM = 11
IMAGE_SIZE = 96
HISTORY = 3


@dataclass
class Episode:
    frames: np.ndarray      # (T + 1, H, W, 3) uint8
    states: np.ndarray      # (T + 1, 4) float32
    actions: np.ndarray     # (T, 2) float32
    goal: np.ndarray        # (2,) float32
    nuisance: np.ndarray    # (11,) float32


def sample_nuisance(rng: np.random.Generator) -> np.ndarray:
    angle = rng.uniform(0.0, 2.0 * np.pi)
    strength = rng.uniform(0.20, 0.80)
    bg = rng.uniform(0.03, 0.20, size=3)
    puck = rng.uniform(0.35, 1.00, size=3)
    cam = rng.uniform(-0.04, 0.04, size=2)
    return np.concatenate(
        [[np.sin(angle), np.cos(angle), strength], bg, puck, cam]
    ).astype(np.float32)


def sample_nuisance_shift(rng: np.random.Generator) -> np.ndarray:
    """Out-of-distribution nuisance: harsher light, brighter background,
    darker puck palette, larger camera offset."""
    angle = rng.uniform(0.0, 2.0 * np.pi)
    strength = rng.uniform(0.80, 1.10)
    bg = rng.uniform(0.20, 0.35, size=3)
    puck = rng.uniform(0.20, 0.80, size=3)
    cam = rng.uniform(-0.08, 0.08, size=2)
    return np.concatenate(
        [[np.sin(angle), np.cos(angle), strength], bg, puck, cam]
    ).astype(np.float32)


def _state_to_px(x: float, y: float, nuisance: np.ndarray, size: int):
    cam_x, cam_y = nuisance[-2], nuisance[-1]
    px = (x + cam_x) * (size - 1)
    py = (y + cam_y) * (size - 1)
    return float(px), float(py)


def render_state(
    state: np.ndarray,
    goal: np.ndarray,
    nuisance: np.ndarray,
    size: int = IMAGE_SIZE,
) -> np.ndarray:
    """Render one state. State is [x, y, vx, vy]; only position is visible."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) / max(size - 1, 1)

    light_sin, light_cos, strength = nuisance[0], nuisance[1], nuisance[2]
    bg = nuisance[3:6]
    puck_rgb = nuisance[6:9]

    # Directional light gradient over a colored background.
    light = 1.0 + strength * (
        light_cos * (xx - 0.5) + light_sin * (yy - 0.5)
    )
    img = np.clip(bg[None, None, :] * light[..., None], 0.0, 1.0)

    # Subtle grid so the scene has spatial structure.
    grid = ((xx * 8).astype(np.int32) + (yy * 8).astype(np.int32)) % 2
    img = np.clip(img + 0.025 * grid[..., None], 0.0, 1.0)

    img_u8 = (img * 255).astype(np.uint8)
    pil = Image.fromarray(img_u8, mode="RGB")
    draw = ImageDraw.Draw(pil)

    r = max(6, size // 12)
    gx, gy = _state_to_px(goal[0], goal[1], nuisance, size)
    draw.ellipse(
        [gx - r, gy - r, gx + r, gy + r],
        outline=(60, 230, 120),
        width=max(2, size // 40),
    )

    x, y = _state_to_px(state[0], state[1], nuisance, size)
    color = tuple(int(c * 255) for c in np.clip(puck_rgb, 0.0, 1.0))
    draw.ellipse([x - r, y - r, x + r, y + r], fill=color, outline=(20, 20, 20))

    return np.asarray(pil, dtype=np.uint8)


def step_state(state: np.ndarray, action: np.ndarray) -> np.ndarray:
    """One physics step. Walls absorb normal velocity."""
    x, y, vx, vy = state
    ax, ay = np.clip(action, -1.0, 1.0)
    vx = 0.85 * vx + 0.20 * ax
    vy = 0.85 * vy + 0.20 * ay
    x = x + vx
    y = y + vy
    if x < 0.05:
        x, vx = 0.05, 0.0
    elif x > 0.95:
        x, vx = 0.95, 0.0
    if y < 0.05:
        y, vy = 0.05, 0.0
    elif y > 0.95:
        y, vy = 0.95, 0.0
    return np.array([x, y, vx, vy], dtype=np.float32)


def _initial_state(rng: np.random.Generator):
    state = np.array(
        [rng.uniform(0.15, 0.85), rng.uniform(0.15, 0.85), 0.0, 0.0],
        dtype=np.float32,
    )
    for _ in range(100):
        goal = rng.uniform(0.10, 0.90, size=2).astype(np.float32)
        if np.linalg.norm(goal - state[:2]) > 0.35:
            return state, goal
    return state, goal


def generate_episode(
    rng: np.random.Generator,
    steps: int = 32,
    size: int = IMAGE_SIZE,
    goal_directed: float = 0.0,
    nuisance_shift: bool = False,
) -> Episode:
    state, goal = _initial_state(rng)
    nuisance = sample_nuisance_shift(rng) if nuisance_shift else sample_nuisance(rng)
    states = [state.copy()]
    actions = []
    for _ in range(steps):
        if rng.random() < goal_directed:
            direction = goal - state[:2]
            norm = np.linalg.norm(direction) + 1e-6
            action = (direction / norm).astype(np.float32)
        else:
            action = rng.uniform(-1.0, 1.0, size=2).astype(np.float32)
        actions.append(action)
        state = step_state(state, action)
        states.append(state.copy())

    states_arr = np.stack(states).astype(np.float32)
    frames = np.stack(
        [render_state(s, goal, nuisance, size=size) for s in states_arr]
    )
    return Episode(
        frames=frames,
        states=states_arr,
        actions=np.stack(actions).astype(np.float32),
        goal=goal.astype(np.float32),
        nuisance=nuisance,
    )


def generate_episodes(
    n: int,
    steps: int = 32,
    size: int = IMAGE_SIZE,
    seed: int = 0,
    goal_directed: float = 0.0,
    nuisance_shift: bool = False,
) -> list[Episode]:
    rng = np.random.default_rng(seed)
    return [
        generate_episode(
            rng, steps=steps, size=size, goal_directed=goal_directed,
            nuisance_shift=nuisance_shift,
        )
        for _ in range(n)
    ]


def history_indices(t: int, history: int = HISTORY) -> list[int]:
    return [max(0, t - (history - 1 - i)) for i in range(history)]
