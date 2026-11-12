# Hypothesis A — discriminative reachability probe

**Status:** complete, 2026-11-12. One dual-stream checkpoint, 3 classifier seeds.
**Code:** `scripts/reachability_classifier.py`.
**Artifacts:** `runs/reachability-classifier/` (git-ignored).

## 1. Protocol

Frozen dual-stream latents:

- state stream: 47k oracle CNN features (`z_state`, 64-d);
- JEPA stream: 142k `asym_clean` latent (`z_jepa`, 64-d).

Classifier input: `[z_current, z_goal]` (concatenated current + goal features).
Target: binary reachability from the ground-truth one-way-door rule
(`door_open or side == goal_side`). 2-layer MLP, 50 epochs, 3 seeds. The abort
threshold is the 17th percentile of validation reachable probabilities; the test
starts are the planner starts (reachable left, unreachable right after crossing).

Ablations: **state-only**, **JEPA-only**, **combined**. Without these, a high
combined AUC could be the state stream's side/goal information.

## 2. Results

| features | seed | AUC | abort unreach | false abort | τ |
|---|---:|---:|---:|---:|---:|
| state | 0 | 1.000 | 1.00 | 0.11 | 1.000 |
| state | 1 | 1.000 | 1.00 | 0.16 | 1.000 |
| state | 2 | 1.000 | 1.00 | 0.19 | 1.000 |
| JEPA | 0 | 0.285 | 0.00 | 0.10 | 1.000 |
| JEPA | 1 | 0.874 | 0.67 | 0.11 | 1.000 |
| JEPA | 2 | 0.290 | 0.00 | 0.06 | 1.000 |
| combined | 0 | 1.000 | 1.00 | 0.05 | 1.000 |
| combined | 1 | 1.000 | 1.00 | 0.05 | 1.000 |
| combined | 2 | 1.000 | 1.00 | 0.03 | 1.000 |

## 3. The pass is confounded

The combined classifier clears the numeric table:

- AUC **1.000** ≥0.700;
- abort unreachable **1.00** ≥0.67;
- false abort **0.03–0.05** ≤0.17.

But the **state-only** classifier also clears the decision-relevant metrics:
AUC 1.000, abort 1.00, false abort 0.11–0.19. The state stream's x,y features
plus the goal position are sufficient for the test distribution: the unreachable
starts are all right-side/left-goal. The task's reachability label is predictable
from **side + goal side**; the door status is not needed.

The **JEPA-only** classifier is unstable and mostly fails: AUC 0.285 / 0.874 /
0.290. It does not provide a reliable reachability predicate.

Therefore the combined pass is **not** evidence that the JEPA pair objective
encoded irreversibility. It is the state stream's side/goal shortcut. The user's
decision tree should treat this as a **fail on attribution**, not a pass for the
JEPA.

## 4. Calibration caveat

The classifier probabilities are saturated: reachable P≈1.0, unreachable P≈0.0
(or 0.33 for the combined seeds). The 17th-percentile threshold is τ=1.000, so
the abort gate is effectively “abort if P<1”. The AUC is the robust metric; the
abort-rate table is less informative under saturation.

## 5. Verdict

**Hypothesis A passes numerically, but fails attribution.**

- The state stream alone solves the reachability test.
- The JEPA stream alone does not.
- The combined classifier's advantage is the state stream; adding JEPA slightly
  reduces false aborts on these seeds but does not create the signal.

This is the same pattern as the dual-stream planner: state is solved, JEPA
irreversibility is not.

## 6. Next

The current test is too easy for a state-only shortcut. To test Hypothesis A
fairly, the task must break the side+goal rule:

- same side, door open vs closed (a manual lock or a counterfactual door), or
- fully occluded `t_cross+3/5` with a **belief state** that remembers the
  crossing.

Without that, the next hypothesis is **Hypothesis B: recurrent/belief-state
JEPA** for the partially observed, history-dependent case.

## Caveats

- One dual-stream checkpoint, 3 classifier seeds; not a five-seed matrix.
- Classifier probabilities are saturated; τ=1.000.
- The test starts keep the crossing in the 3-frame history; fully occluded
  `t_cross+3/5` is not identifiable from the current observation.
- The state stream is the oracle CNN, so the state-only ablation is an upper
  bound on the shortcut.
