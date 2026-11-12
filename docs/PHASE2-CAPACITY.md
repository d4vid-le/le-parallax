# Phase 2 capacity + schedule upgrade

**Status:** complete, 2026-11-12. Single-seed, 60 epochs. The triggers failed;
the five-seed matrix is **not** started.

## 1. Capacity

The 142k model was the state bottleneck: the pre-merge state head raised state
R² from 0.223–0.322 to 0.670–0.695, but the 0.80 guardrail was still missed.

The expanded model:

| component | 142k model | expanded |
|---|---|---|
| latent | 64 | **128** |
| conv width | 32/64/64 | **40/80/80** (`width=1.25`) |
| frame features | 128 | **160** |
| predictor hidden | 128 | **256** |
| **total params** | **142,564** | **340,784** |

The pre-merge frame state head remains: current-frame 160-d keypoint features →
4-d state, before the history merge and before the asymmetric pair heads.

## 2. Schedule

- 60 epochs, 5 warmup
- cosine-decay learning rate (3e-4 → 0)
- `λ_pair=0.10`
- variance-floor weight **0.10**
- pair batch 64, temporal batch 32
- occluded door, post offsets 3/5, 40 train / 20 test episodes

## 3. Guardrails for the five-seed trigger

1. State R² ≥ **0.80** for the temporal baseline and pair arms.
2. `Δasym(asym_clean) − Δasym(sym_head)` ≥ **+0.100**.
3. Bounded latent std ≥ **0.90**.

If all three clear on the single seed, run the five-seed matrix. If not, report
the remaining gap rather than launching.

## 4. Results

60 epochs, 40 train / 20 test, cosine, 340,784 params:

| arm | Δasym | state R² | frame probe R² | frame head R² | latent std | planner AUC |
|---|---:|---:|---:|---:|---:|---:|
| temp_matched | −0.090 | **0.610** | −0.060 | 0.243 | 0.915 | 0.654 |
| sym_head | +0.087 | **0.509** | −0.009 | 0.298 | 0.916 | 0.835 |
| asym_clean | +0.078 | **0.662** | −0.027 | 0.308 | 0.917 | 0.966 |
| asym_vol_bounded | +0.043 | **0.623** | −0.065 | 0.282 | **0.882** | 0.980 |

The three trigger checks:

1. **State R² ≥0.80:** best **0.662**. Fail.
2. **Direction margin ≥+0.100:** `asym_clean +0.078` minus `sym_head +0.087` =
   **−0.009**. Fail. The symmetric control is slightly higher.
3. **Bounded std ≥0.90:** `asym_vol_bounded` **0.882**. Fail.

## 5. What the failure says

- **Capacity is not the state bottleneck.** The 340k model is *worse* than the
  142k frame-head model on state R² (temporal 0.610 vs 0.638; sym_head 0.509 vs
  0.670). More width made optimization harder, not state readability better.
- **The pre-merge frame head is not the fix.** The frame-feature ridge R² is
  **negative**; the trained frame head is **0.24–0.31**. The state information
  is not linearly available in the current-frame keypoint features. The merged
  latent (0.51–0.66) is still the better state path.
- **The direction metric is not a stable signature.** At 142k/40 epochs,
  `asym_clean` was +0.080 vs `sym_head` −0.019. At 340k/60 epochs, it is +0.078
  vs +0.087. The ordering flips with scale/schedule. A five-seed run now would
  not resolve the state guardrail; it would average an unstable metric.
- **Do not run the 0.12 sweep.** Pair weight cannot move the temporal state R²,
  and the 0.10 point already fails the state and margin checks. The next move is
  a **separate state encoder with gradient isolation**, not another weight.

## 6. Next move

A dedicated state encoder (a small CNN trained only on the state loss, with no
gradient from the pair/temporal objective) is the clean test of the decoupling
idea. If that also fails to reach 0.80, the state guardrail is not reachable
with this observation/model and the Phase 2 confirmatory matrix should be
abandoned rather than forced.
