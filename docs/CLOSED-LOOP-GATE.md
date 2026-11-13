# Closed-loop abort gates — Step 3

**Status:** matched counterfactual **passes**; clean-approach false abort is
**0%** for all three planner-belief seeds. The literal 5% *overall gate-status*
ceiling is not met by seed 1 (8.8%), so the five-seed matrix is **not started**.
2026-11-12.

**Code:** `scripts/closed_loop_gate.py`,
`scripts/planner_belief_train.py`,
`scripts/matched_cf_closed_loop.py`,
`scripts/controller_probe.py`,
`scripts/counterfactual_seed_check.py`.
**Artifacts:** `runs/closed-loop-gate-large*/`,
`runs/matched-cf-closed-loop*/`,
`runs/controller-probe*/` (git-ignored).

## 1. Setup

Base planner: frozen state stream + state-feedback controller on the natural
hidden-alcove world. The abort gate never steers; it only stops the episode.
The hidden right-side gate closes with the one-way door and is never rendered.

- Calibration: 160 episodes (99 gate-open / 61 gate-closed at the end).
- Test: 240 episodes (159 gate-open / 81 gate-closed at the end).
- Task-reachable episodes: **119 right-start**; left-start episodes are
  unreachable by construction (crossing the one-way door closes the alcove
  gate, so the goal behind it cannot be reached).
- Base controller success on right-start episodes: **119/119 = 1.000**.
  The quoted 0.748 is 119/159: the denominator also contains 40 left-start
  episodes where the controller never crossed and therefore never reached the
  goal. This is not a door-gap controller bug; see §5.
- Gate thresholds compared here: belief **τ=0.18, k=4**; state **τ=0.25,
  k=3**; frozen JEPA **τ=0.12, k=4**.
- Persistence `k` is the number of consecutive frames below `τ`; abort delay is
  measured after the hidden gate closes; step savings are unreachable-only.

## 2. Large test, three planner-belief seeds

| seed | false abort (all open) | clean false abort | failure false abort | gate-closed abort | delay after close |
|---:|---:|---:|---:|---:|---:|
| 0 | **3.1%** (5/159) | **0/119** | 5/40 (12.5%) | **81/81 (100%)** | **3.40** |
| 1 | **8.8%** (14/159) | **0/119** | 14/40 (35.0%) | 79/81 (97.5%) | **2.90** |
| 2 | **0.0%** (0/159) | **0/119** | 0/40 (0.0%) | 77/81 (95.1%) | **3.19** |

“Clean” means the base controller reached the goal; “failure” means it did not.
**Every false abort is on a left-start controller failure, and every clean
right-start episode is protected in all three seeds.** The gate-closed abort
denominator is the 81 left-start episodes that crossed the door. The other 40
left-start non-crossings are task-unreachable but keep the gate open, so they
are in the failure column, not the gate-closed column.

Persistence trade-off (seed 0, τ=0.18): `k=3` gives 5.0% overall / 0% clean /
100% abort / 2.40-step delay; `k=4` gives 3.1% / 0% / 100% / 3.40; `k=5` gives
1.3% / 0% / 98.8% / 4.30. Seed 2 is at 0% overall for `k=4` and `k=5`. Seed 1
has **no** `(τ,k)` on this grid below 5% overall (minimum ≈8.2%, at τ=0.16,
`k=4` and τ=0.18, `k=5`); its clean false abort is still 0%.

## 3. Why “gate open at the end” is not the same as “goal reachable”

The first pass labeled episodes by the final gate status. That made a
left-start episode whose controller never crossed look “reachable” because the
gate was still open. But a left-start puck cannot reach the right-alcove goal
without crossing the one-way door, and crossing closes the alcove gate. The
goal is therefore unreachable from the left regardless of whether the
controller crosses. The 40 “failure” episodes in the table above are exactly
these left-start non-crossings; they are task failures, not clean approaches.

This is why the overall gate-status false-abort rate and the clean/task-
reachable false-abort rate differ. The clean rate is the one that measures harm
to successful episodes. The overall rate is inflated by counting aborts on
episodes that were already task-unreachable.

## 4. Matched counterfactual closed loop (decisive test)

Each of 100 pairs has the same goal, nuisance, branch state `[x, y, vx, vy]`,
and **byte-identical last three frames** at the branch. One history crossed the
one-way door (gate closed); the other started right of the door (gate open).
Both branches then run the same state-feedback controller under their hidden
gate status. Static gates see identical inputs at the branch; only a
history-dependent belief can separate the pair.

| seed | belief branch AUC | short-history control | static state early AUC | static JEPA early AUC | closed abort | open false abort | delay |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | **1.000** | 0.495 | 0.501 | 0.496 | 0.97 | **0.00** | 3.00 |
| 1 | **0.998** | 0.495 | 0.501 | 0.496 | 0.98 | **0.00** | 3.00 |
| 2 | **1.000** | 0.495 | 0.501 | 0.496 | 0.93 | **0.00** | 3.15 |

The matched branch static differences are exactly **0.0** for both the state
and JEPA features. The short-history control (last three frames only) is at
chance, so the belief is using the earlier crossing history, not the current
observation. Over the full post-branch window, the state gate reaches 0.78
closed-abort / 0.13 open-false and the JEPA gate 0.22 / 0.25, but their early
AUCs are chance; they separate only after the closed branch is physically
stuck at the gate. That is the spatial shortcut the natural test could not
rule out.

## 5. Controller probe

The 0.748 base success is a denominator artifact, not a gap-tracking bug:

| controller | right-start success | left-start success | left-start gate closed | total successes |
|---|---:|---:|---:|---:|
| goal-only (current) | **119/119 (1.000)** | 0/121 | 81/121 | 119 |
| staged door waypoint | 118/119 (0.992) | 0/121 | 121/121 | 118 |
| goal-only with true state | 119/119 (1.000) | 0/121 | 82/121 | 119 |

The waypoint controller crosses the door in every left-start episode, but
those episodes are still unreachable because the gate closes; it does not
increase total success. The base controller is already perfect on the
reachable right-start set. Leave it as-is.

## 6. Verdict

**The decisive matched counterfactual passes for all three seeds.** The belief
gate protects every clean, task-reachable episode (0/119 false aborts) and
aborts 93–100% of matched closed-gate branches, while state-only and JEPA are
at chance at the branch and only separate after spatial divergence.

**The literal 5% overall gate-status ceiling does not pass for seed 1**
(8.8%), because it aborts 14/40 left-start non-crossing controller failures.
Those episodes are task-unreachable; they are not clean approaches. The
ensemble is not used to hide this. The five-seed matrix is **not started** until
the primary metric is fixed or seed 1 is improved on a fresh episode set.

## 7. Next

1. **Decide the primary false-abort set**: clean/task-reachable episodes
   (0% for all seeds) or all gate-open episodes (seed 1 fails at 8.8%).
2. If the overall ceiling is required, retrain the planner belief with more
   planner rollouts and explicit left-start non-crossing hard negatives; do not
   reseed until a seed passes.
3. Lock `(τ,k)` on a **fresh** validation seed, then run a fresh 240-episode
   single-seed test before the five-seed matrix. The current 240 episodes have
   already been used to pick τ=0.18, `k=4`.
4. Keep the ensemble as an optional deployment row only; report individual
   seeds in the matrix.

## Caveats

- The 240-episode set was used to inspect the grid, so τ=0.18, `k=4` is a
  validation choice, not a fully held-out confirmatory threshold.
- Seed 1 is the weak belief seed: counterfactual last-frame AUC 0.918 vs
  1.000 for seeds 0/2, and 8.8% overall gate-status false abort.
- Left-start episodes are task-unreachable by construction; the “open gate”
  label alone overstates reachability.
- The gate status is hidden; a manual lock with no history is unobservable to
  every gate.
- The closed-loop controller is a state-feedback heuristic, not a full MPC.
