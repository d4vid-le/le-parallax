#!/usr/bin/env python3
"""Generate static snapshots of the synthetic world for inspection."""

from __future__ import annotations

import pathlib
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from leparallax.data import history_obs
from leparallax.world import (
    generate_episodes,
    history_indices,
    render_state,
    sample_nuisance,
)

OUT = pathlib.Path(__file__).resolve().parents[1] / "snapshots"
SIZE = 96
SCALE = 2


def upscale(img: np.ndarray) -> Image.Image:
    im = Image.fromarray(img)
    return im.resize((im.width * SCALE, im.height * SCALE), Image.NEAREST)


def label(im: Image.Image, text: str) -> Image.Image:
    canvas = Image.new("RGB", (im.width, im.height + 18), (245, 245, 245))
    canvas.paste(im, (0, 18))
    ImageDraw.Draw(canvas).text((4, 3), text, fill=(20, 20, 20))
    return canvas


def crop_resize(img: np.ndarray, rng: np.random.Generator, scale: float = 0.72):
    h, w = img.shape[:2]
    side = int(min(h, w) * scale)
    x = int(rng.integers(0, w - side + 1))
    y = int(rng.integers(0, h - side + 1))
    crop = Image.fromarray(img[y : y + side, x : x + side])
    return np.asarray(crop.resize((w, h), Image.BILINEAR))


def hstack(items: list[Image.Image]) -> Image.Image:
    w = sum(i.width for i in items) + 4 * (len(items) - 1)
    h = max(i.height for i in items)
    out = Image.new("RGB", (w, h), (245, 245, 245))
    x = 0
    for item in items:
        out.paste(item, (x, 0))
        x += item.width + 4
    return out


def vstack(items: list[Image.Image]) -> Image.Image:
    w = max(i.width for i in items)
    h = sum(i.height for i in items) + 4 * (len(items) - 1)
    out = Image.new("RGB", (w, h), (245, 245, 245))
    y = 0
    for item in items:
        out.paste(item, (0, y))
        y += item.height + 4
    return out


def main():
    OUT.mkdir(exist_ok=True)
    episodes = generate_episodes(4, steps=24, size=SIZE, seed=7, goal_directed=0.4)
    rng = np.random.default_rng(0)

    # 1. Grid: four episodes, five moments.
    rows = []
    for i, ep in enumerate(episodes):
        frames = [ep.frames[t] for t in (0, 6, 12, 18, 24)]
        rows.append(hstack([label(upscale(f), f"ep{i} t{t}") for f, t in zip(frames, (0, 6, 12, 18, 24))]))
    vstack(rows).save(OUT / "episode_grid.png")

    # 2. Temporal sequence: same episode, four consecutive moments.
    ep = episodes[0]
    frames = [ep.frames[t] for t in (8, 9, 10, 11)]
    hstack([label(upscale(f), f"t={t}") for f, t in zip(frames, (8, 9, 10, 11))]).save(
        OUT / "temporal_sequence.png"
    )

    # 3. Good view: same state, independent nuisance.
    t = 12
    other = sample_nuisance(rng)
    v1 = history_obs(ep, t, ep.nuisance)
    v2 = history_obs(ep, t, other)
    row1 = hstack([label(upscale(f), f"view A {i}") for i, f in enumerate(v1)])
    row2 = hstack([label(upscale(f), f"view B {i}") for i, f in enumerate(v2)])
    vstack([row1, row2]).save(OUT / "good_view_pair.png")

    # 4. Bad view: two crops of the same observation.
    base = ep.frames[history_indices(t)[-1]]
    crop1 = crop_resize(base, rng)
    crop2 = crop_resize(base, rng)
    hstack(
        [
            label(upscale(base), "original"),
            label(upscale(crop1), "crop A"),
            label(upscale(crop2), "crop B"),
        ]
    ).save(OUT / "bad_view_pair.png")

    # 5. A 2x2 contact sheet for the README.
    sheet = hstack(
        [
            vstack(
                [
                    label(upscale(episodes[0].frames[12]), "state A / nuisance A"),
                    label(upscale(v1[-1]), "history t"),
                ]
            ),
            vstack(
                [
                    label(upscale(render_state(episodes[0].states[12], ep.goal, other, size=SIZE)), "state A / nuisance B"),
                    label(upscale(v2[-1]), "same state, new nuisance"),
                ]
            ),
        ]
    )
    sheet.save(OUT / "overview.png")

    (OUT / "README.md").write_text(
        "# Simulation snapshots\n\n"
        "Synthetic 2D puck world used in Phase 0. State is `[x, y, vx, vy]`; the\n"
        "goal is the green ring; nuisance is lighting, background, puck color, and\n"
        "camera offset.\n\n"
        "| Image | What it shows |\n|---|---|\n"
        "| `episode_grid.png` | Four episodes, five moments each, different nuisance styles |\n"
        "| `temporal_sequence.png` | Four consecutive states from one episode |\n"
        "| `good_view_pair.png` | Same state and history, two independent nuisance draws |\n"
        "| `bad_view_pair.png` | One observation and two crops of it (the bad-view control) |\n"
        "| `overview.png` | The same state under two nuisance styles |\n\n"
        "All images are synthetic. Regenerate with `python scripts/make_snapshots.py`.\n"
    )
    print("wrote", OUT)


if __name__ == "__main__":
    main()
