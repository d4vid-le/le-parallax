# Simulation snapshots

Synthetic 2D puck world used in Phase 0. State is `[x, y, vx, vy]`; the
goal is the green ring; nuisance is lighting, background, puck color, and
camera offset.

| Image | What it shows |
|---|---|
| `episode_grid.png` | Four episodes, five moments each, different nuisance styles |
| `temporal_sequence.png` | Four consecutive states from one episode |
| `good_view_pair.png` | Same state and history, two independent nuisance draws |
| `bad_view_pair.png` | One observation and two crops of it (the bad-view control) |
| `overview.png` | The same state under two nuisance styles |

All images are synthetic. Regenerate with `python scripts/make_snapshots.py`.
