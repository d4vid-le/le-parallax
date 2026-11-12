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
DEFAULT_DYNAMICS = np.array([0.85, 0.20], dtype=np.float32)


@dataclass
class Episode:
    frames: np.ndarray      # (T + 1, H, W, 3) uint8
    states: np.ndarray      # (T + 1, 4) float32
    actions: np.ndarray     # (T, 2) float32
    goal: np.ndarray        # (2,) float32
    nuisance: np.ndarray    # (11,) float32
    dynamics: np.ndarray    # (2,) [damping, action_gain]
    door_open: np.ndarray | None = None  # (T + 1,) bool, optional one-way door
    occlude_door_right: bool = False  # hide door status when the puck is right


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


def sample_nuisance_mix(rng: np.random.Generator) -> np.ndarray:
    """50/50 normal and shifted nuisance, for domain randomization."""
    return sample_nuisance_shift(rng) if rng.random() < 0.5 else sample_nuisance(rng)


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
    door_open: bool | None = None,
    occlude_door_right: bool = False,
) -> np.ndarray:
    """Render one state. State is [x, y, vx, vy]; only position is visible.

    If `door_open` is not None, draw the one-way door at x=0.5. An open door has
    a gap from y=0.35 to 0.65; a closed door is solid. If
    `occlude_door_right` is set and the puck is on the right, an occlusion wall
    hides the door status while keeping the puck visible.
    """
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

    if door_open is not None:
        door_x, _ = _state_to_px(0.5, 0.5, nuisance, size)
        _, y_top = _state_to_px(0.5, 0.0, nuisance, size)
        _, y_gap0 = _state_to_px(0.5, 0.35, nuisance, size)
        _, y_gap1 = _state_to_px(0.5, 0.65, nuisance, size)
        _, y_bot = _state_to_px(0.5, 1.0, nuisance, size)
        width = max(2, size // 32)
        color = (210, 210, 210)
        if door_open:
            draw.line([(door_x, y_top), (door_x, y_gap0)], fill=color, width=width)
            draw.line([(door_x, y_gap1), (door_x, y_bot)], fill=color, width=width)
        else:
            draw.line([(door_x, y_top), (door_x, y_bot)], fill=color, width=width)

        if occlude_door_right and state[0] >= 0.5:
            x0, _ = _state_to_px(0.42, 0.5, nuisance, size)
            x1, _ = _state_to_px(0.58, 0.5, nuisance, size)
            draw.rectangle(
                [x0, 0, x1, size - 1], fill=(38, 38, 44)
            )
            draw.line(
                [(x0, 0), (x0, size - 1)], fill=(120, 120, 130), width=1
            )

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


def step_state(
    state: np.ndarray,
    action: np.ndarray,
    dynamics: np.ndarray | None = None,
    door_open: bool | None = None,
    door_x: float = 0.5,
    door_gap: tuple[float, float] = (0.35, 0.65),
) -> np.ndarray:
    """One physics step. Walls absorb normal velocity.

    If `door_open` is not None, a one-way door at `door_x` blocks crossing unless
    it is open and the puck is inside `door_gap`.
    """
    damping, gain = DEFAULT_DYNAMICS if dynamics is None else dynamics
    x, y, vx, vy = state
    old_x = x
    ax, ay = np.clip(action, -1.0, 1.0)
    vx = damping * vx + gain * ax
    vy = damping * vy + gain * ay
    x = x + vx
    y = y + vy
    if door_open is not None:
        crossed = (old_x < door_x <= x) or (old_x > door_x >= x)
        if crossed:
            in_gap = door_gap[0] <= y <= door_gap[1]
            if not (door_open and in_gap):
                x = door_x - 1e-4 if old_x < door_x else door_x + 1e-4
                vx = 0.0
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
    nuisance_mode: str = "normal",
    dynamics: np.ndarray | None = None,
    one_way_door: bool = False,
    door_goal_right: float = 0.6,
    occlude_door_right: bool = False,
) -> Episode:
    dynamics = DEFAULT_DYNAMICS if dynamics is None else np.asarray(dynamics, dtype=np.float32)
    state, goal = _initial_state(rng)
    if one_way_door:
        state[0] = rng.uniform(0.15, 0.45)
        for _ in range(100):
            if rng.random() < door_goal_right:
                goal[0] = rng.uniform(0.55, 0.90)
            else:
                goal[0] = rng.uniform(0.10, 0.45)
            goal[1] = rng.uniform(0.10, 0.90)
            if np.linalg.norm(goal - state[:2]) > 0.25:
                break
    if nuisance_mode == "normal":
        nuisance = sample_nuisance(rng)
    elif nuisance_mode == "shift":
        nuisance = sample_nuisance_shift(rng)
    elif nuisance_mode == "mix":
        nuisance = sample_nuisance_mix(rng)
    else:
        raise ValueError(f"unknown nuisance_mode: {nuisance_mode}")
    states = [state.copy()]
    actions = []
    door_open = True if one_way_door else None
    door_flags = [door_open] if one_way_door else None
    for _ in range(steps):
        if rng.random() < goal_directed:
            direction = goal - state[:2]
            norm = np.linalg.norm(direction) + 1e-6
            action = (direction / norm).astype(np.float32)
        else:
            action = rng.uniform(-1.0, 1.0, size=2).astype(np.float32)
        actions.append(action)
        old_x = state[0]
        state = step_state(state, action, dynamics, door_open=door_open)
        if (
            one_way_door
            and door_open
            and old_x < 0.5 <= state[0]
            and 0.35 <= state[1] <= 0.65
        ):
            door_open = False
        states.append(state.copy())
        if door_flags is not None:
            door_flags.append(door_open)

    states_arr = np.stack(states).astype(np.float32)
    frames = np.stack(
        [
            render_state(
                s, goal, nuisance, size=size,
                door_open=bool(door_flags[i]) if door_flags is not None else None,
                occlude_door_right=occlude_door_right,
            )
            for i, s in enumerate(states_arr)
        ]
    )
    return Episode(
        frames=frames,
        states=states_arr,
        actions=np.stack(actions).astype(np.float32),
        goal=goal.astype(np.float32),
        nuisance=nuisance,
        dynamics=dynamics.astype(np.float32),
        door_open=np.array(door_flags, dtype=bool) if door_flags is not None else None,
        occlude_door_right=occlude_door_right,
    )


def generate_episodes(
    n: int,
    steps: int = 32,
    size: int = IMAGE_SIZE,
    seed: int = 0,
    goal_directed: float = 0.0,
    nuisance_shift: bool = False,
    nuisance_mode: str = "normal",
    dynamics: np.ndarray | None = None,
    one_way_door: bool = False,
    door_goal_right: float = 0.6,
    occlude_door_right: bool = False,
) -> list[Episode]:
    rng = np.random.default_rng(seed)
    return [
        generate_episode(
            rng, steps=steps, size=size, goal_directed=goal_directed,
            nuisance_shift=nuisance_shift, nuisance_mode=nuisance_mode,
            dynamics=dynamics, one_way_door=one_way_door,
            door_goal_right=door_goal_right,
            occlude_door_right=occlude_door_right,
        )
        for _ in range(n)
    ]


def history_indices(t: int, history: int = HISTORY) -> list[int]:
    return [max(0, t - (history - 1 - i)) for i in range(history)]
