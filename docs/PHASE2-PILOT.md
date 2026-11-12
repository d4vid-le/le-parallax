# Phase 2 pilot — partial observation, matched heads, bounded volume, planner

**Status:** pilot complete, 2026-11-12. One seed, 40 epochs. The confirmatory
five-seed run is **not** started because the state guardrail fails.
**Protocol:** [`PREREG-PHASE2.md`](PREREG-PHASE2.md), sha256
`eab4147cc77c2aa99b0ef5ad293e98503d21a9a6b61ceea67dcff08826412e46`.
**Artifacts:** `runs/phase2-pilot/` (git-ignored).

---

## 1. What Phase 2 fixed

Collision 4 had two confounds:

1. the door was visible after crossing, so the temporal model already had the
   information;
2. the symmetric baseline did not have parameter-matched heads.

Phase 2 removes both:

- **Partial observation.** On the right side of the one-way door, an occlusion
  wall hides the door status. The rendered frame is **identical** whether the
  door is open or closed; the puck's `x,y` is still visible. On the left, the
  door is visible. The transition `A→B` is a lossy projection of future
  potential.
- **Matched heads.** `sym_head` and `asym_clean` have the same forward/reverse
  head capacities (g 128, h 32). `sym_head` uses a symmetric swapped-target
  loss; `asym_clean` uses the directed `g(A)→B`, `h(B)→A` loss.
- **Bounded volume.** `asym_vol_bounded` adds a VICReg-style variance floor
  `mean(max(0, 1−std(z_B)))` to the log-volume term, so the latent cannot shrink
  globally.

## 2. Pilot results

One seed, 40 epochs, 40 train / 20 test episodes, post offsets 3/5 (fully
occluded right-side history).

| arm | Δasym | ΔΔrank | state R² | reach AUC | latent std | planner AUC | abort unreach | false abort |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| temp_matched | −0.072 | −0.365 | **0.574** | 0.290 | 0.915 | 0.667 | 0.67 | 0.17 |
| sym_head | −0.055 | −0.129 | **0.223** | 0.957 | 0.851 | **0.750** | 0.67 | 0.17 |
| asym_clean | **+0.146** | −0.306 | **0.308** | 0.912 | 0.858 | 0.722 | 0.67 | 0.17 |
| asym_vol_bounded | **+0.049** | −0.240 | **0.322** | 0.445 | **1.044** | 0.722 | **0.33** | 0.17 |

## 3. What worked

- **The partial-observation manipulation works.** The right-side frames are
  identical for door open/closed; the model cannot read the door status from
  `B`.
- **The direction signal appears.** For the first time, the asymmetric arm has
  a positive `Δasym` (**+0.146** for `asym_clean`, **+0.049** for
  `asym_vol_bounded`) while the matched symmetric head is negative
  (**−0.055**). `asym_clean` beats the pre-registered margin over `sym_head`
  (+0.201) on this seed. This is the first evidence consistent with
  Anti-Equivalence.
- **The variance floor works as specified.** `asym_vol_bounded` has latent std
  **1.044** (≥0.90) and does not collapse to a shrunken latent; the rank
  contraction is **−0.240**, not the catastrophic `−5.08` reverse-R² collapse
  from Collision 4.

## 4. What failed

- **The state guardrail fails.** The temporal baseline is **0.574**, far below
  the required 0.80; the pair arms are **0.223–0.322**. The pair loss is
  distorting the encoder on the occluded-door task. The trial is therefore
  **inconclusive on the representation claim**, by the pre-registered rule.
- **The planner does not improve.** `sym_head` has the best planner AUC
  (**0.750**); `asym_clean` and `asym_vol_bounded` are **0.722**, and the
  bounded volume arm's unreachable abort rate falls to **0.33** (worse than the
  symmetric 0.67). A positive latent direction did not produce a planning
  advantage.
- **The state cost is the bottleneck.** The direction signal comes with a
  roughly 50% drop in state R² relative to the temporal baseline. A latent that
  encodes irreversibility at the expense of the state is not a usable world
  model for control.

## 5. Verdict

**Inconclusive, with a suggestive positive direction signal.**

Phase 2 is the right manipulation: hiding the door changes the direction result
from negative (Collision 4) to positive. But the model cannot hold both the
state and the irreversibility at this scale and pair weight. The planner does
not benefit, and the bounded volume arm under-aborts unreachable goals. This is
not a confirmation of Anti-Equivalence; it is a reason to fix the capacity and
re-run.

## 6. What to change before the five-seed run

1. **Lower the pair weight** from 0.25 to 0.05–0.10, or warm up longer. The pair
   loss is dominating the state objective.
2. **Lower the variance-floor weight** from 1.0 to 0.1–0.25; it is strong enough
   to hold std at 1.04, but it may be contributing to the state cost.
3. **Scale the encoder or latent** (more channels or 128-d latent) so the model
   can represent both the state and the door/irreversibility.
4. **Add a keypoint-specific state head** or a stronger state auxiliary loss;
   the linear head on the shared latent is the weakest link.
5. **Recalibrate the planner** with a proper CEM and a held-out threshold; the
   random shooting planner's abort rates are coarse.

The pre-registered five-seed run should start only after the temporal state R²
is at least 0.80 on a pilot.

## 7. Caveats

- One seed, 40 epochs; not a confirmatory result.
- The state R² is low; all downstream comparisons are exploratory.
- The planner is a latent shooting diagnostic, not closed-loop MPC in the true
  environment.
- The variance floor is a diagonal proxy, not the full covariance determinant.
- The one-way door is a controlled proxy for irreversibility.
