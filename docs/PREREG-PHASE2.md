# Pre-registration — Phase 2: un-confounding asymmetry

**Status:** **FROZEN 2026-11-12** for the confirmatory run. The one-seed pilot is
exploratory and is reported separately.

## Goal

Give the Anti-Equivalence hypothesis a fair test by removing the two confounds
from Collision 4:

1. the door was visible after crossing, so the temporal model already had the
   information;
2. the symmetric baseline did not have parameter-matched heads.

Phase 2 uses a **partially observed** one-way door and a parameter-matched
symmetric bottleneck control, then tests a **shooting/MPC planner**, not a
linear probe.

## World

One-way door at `x=0.5`, gap `y∈[0.35,0.65]`, closing permanently on the first
left→right crossing. On the **right** side of the door, an occlusion wall hides
the door status: the rendered frame is identical whether the door is open or
closed. The puck's `x,y` remains visible. On the left, the door is visible.

This makes the transition `A→B` a lossy projection of future potential: `A`
knows the door is open; `B` does not know whether it closed behind the puck or
was locked manually. The left goal is reachable from `A`, unreachable from `B`.

## Data

- 80 train / 20 test episodes, 32 steps, `goal_directed=0.5`,
  `door_goal_right=0.6`, `occlude_door_right=True`.
- Irreversible pairs: `A=t_cross−{2,4}`, `B=t_cross+{3,5}`. The post offsets
  ensure the three-frame history at `B` is fully on the right and the door is
  occluded.
- Reversible control pairs: both states on the right, door closed, gaps 4/6/8.
- Goal observation for the planner: a three-frame render of the goal state with
  no door.

## Arms

| arm | pair loss | heads | volume |
|---|---|---|---|
| `temp_matched` | none | none | — |
| `sym_head` | symmetric swapped-target `g`/`h` | g 128, h 32 | — |
| `asym_clean` | directed `g(A)→B`, `h(B)→A` | g 128, h 32 | — |
| `asym_vol_bounded` | directed `g`/`h` | g 128, h 32 | `+0.05·log-vol(z_B) + 1.0·variance-floor(z_B)` |

`variance_floor(z) = mean(max(0, 1 − std(z)))`, VICReg-style. The floor prevents
the log-volume term from shrinking the latent globally; the log-volume can only
contract dimensions while others stay at unit variance.

Five seeds, 40 epochs, 5 warmup, `λ_pair=0.25`, `λ_state=0.25`.

## Metrics

- **Direction:** `Δasym = asym_ratio(irr) − asym_ratio(rev)` from ridge
  forward/reverse probes on held-out pairs.
- **Rank/volume:** effective rank, latent std, and residual rank.
- **State guardrail:** ridge state R². The trial is **inconclusive** if the
  temporal baseline is below **0.80**.
- **Planner:** latent shooting planner with 64 random action sequences plus a
  model-based controller candidate, horizon 6. Score = predicted state-head
  distance to the goal. Threshold = 90th percentile of reachable-start distances.
  Metrics: abort rate on unreachable starts, false-abort rate on reachable
  starts, and planner AUC for unreachable vs reachable.

## Bets

- **P2-A direction:** `asym_clean` `Δasym` is ≥ `sym_head` + 0.10 in ≥4/5
  seeds.
- **P2-B bounded volume:** `asym_vol_bounded` has latent std ≥0.90 and a rank
  contraction ≥10% relative to `sym_head`, with state R² drop ≤0.05.
- **P2-C planner:** `asym_clean` or `asym_vol_bounded` planner AUC is ≥
  `sym_head` + 0.05, and unreachable abort rate is ≥ reachable false-abort rate
  + 0.20.
- **Guardrail:** if the temporal state R² is <0.80, report the trial as
  inconclusive on the representation claim and do not force the asymmetry bet.

## Falsifiers

No direction gain over `sym_head`; the bounded volume arm still collapses or
fails the variance floor; the planner cannot separate reachable from unreachable
better than `sym_head`; or the state guardrail fails.

## What this does not establish

- A general partially observable world model or a recurrent belief state.
- Closed-loop control in the true environment; the planner is a latent shooting
  diagnostic.
- That the one-way door is the right irreversibility; it is a controlled proxy.
