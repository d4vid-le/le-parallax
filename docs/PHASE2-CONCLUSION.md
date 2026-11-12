# Phase 2 conclusion — the guardrail did not clear

**Status:** closed, 2026-11-12. No five-seed confirmatory matrix was run.
**Records:** [`PHASE2-PILOT.md`](PHASE2-PILOT.md),
[`PHASE2-TUNING.md`](PHASE2-TUNING.md),
[`PHASE2-CAPACITY.md`](PHASE2-CAPACITY.md).
**Protocol:** [`PREREG-PHASE2.md`](PREREG-PHASE2.md).

## What Phase 2 was for

Collision 4 tested Anti-Equivalence on a visible one-way door. The temporal
model could read the door state, so the asymmetry test was confounded. Phase 2
removed the confounds:

- the door is **occluded on the right**; the frame is identical whether it is
  open or closed, while the puck remains visible;
- the symmetric control has **parameter-matched heads**;
- the volume penalty has a **VICReg-style variance floor**;
- the planner is a **latent shooting planner**, not a linear probe.

The pre-registered triggers were:

1. state R² ≥ **0.80** across the temporal baseline and pair arms;
2. `Δasym(asym_clean) − Δasym(sym_head)` ≥ **+0.100**;
3. bounded latent std ≥ **0.90**.

## The three iterations

| iteration | state R² | direction margin | bounded std | verdict |
|---|---:|---:|---:|---|
| pilot, 142k, 40 epochs | 0.308–0.322 | +0.201 (seed 0) | 1.044 | state guardrail fails |
| frame state head + `λ_pair=0.08`, 142k, 40 epochs | 0.670–0.695 | +0.099 | 0.968 | close, still fails |
| expanded 340k, 60 epochs cosine, `λ_pair=0.10` | 0.610–0.662 | **−0.009** | 0.882 | worse on all three |

The expanded model was **worse** than the 142k frame-head model, not better:
temporal state R² 0.610 vs 0.638, `sym_head` 0.509 vs 0.670. The pre-merge frame
probe was negative (R² −0.06 to −0.01) and the trained frame head only 0.24–0.31.
Capacity and the dedicated frame head are not the state fix.

## What survived

- **The partial-observation manipulation works.** The right-side frame no
  longer reveals the door state.
- **The direction sign can flip positive.** In the first Phase 2 pilot,
  `asym_clean` was +0.146 vs `sym_head` −0.055. The occluded-door environment is
  the right kind of test.
- **The frame state head improves state a lot.** At 142k/40 epochs it raised
  pair-arm state R² from 0.223–0.322 to 0.670–0.695.
- **The variance floor works.** It kept bounded latent std at 0.968–1.044
  without the Collision 4 collapse.

## What failed

- **The state guardrail.** Best 0.695 at 142k; 0.662 at 340k. The state
  representation is the bottleneck.
- **The direction margin.** The 142k tune reached +0.099, just under the +0.10
  bar; the 340k run flipped to −0.009. The metric is not stable across scale and
  schedule.
- **The planner.** `asym_clean` did not beat `sym_head` (0.694 vs 0.694 at 142k;
  0.966 vs 0.835 at 340k, but the state guardrail fails). The bounded arm’s
  abort rate did not improve.
- **Capacity scaling.** 340k parameters made optimization harder, not state
  readability better.
- **The pre-merge frame head as a state path.** The frame features are not
  linearly state-decodable; the head is 0.24–0.31.

## The oracle check

A standalone **46,830-parameter CNN** on the same 3-frame occluded observations
reaches:

- **x,y R² 0.978**
- **4-D state R² 0.935**
- position 0.984, velocity 0.887

The oracle-relative guardrail is **0.90 × 0.935 = 0.842** — higher than 0.80,
not lower. The JEPA's best state R² is **0.695**, and `asym_clean` is **0.671**.
The state gap is **0.171** below the oracle-relative guardrail.

The stop-gradient MLP probe does **not** recover more state: **0.564** vs
**0.671** for the linear probe (position 0.708 vs 0.707; velocity 0.420 vs
0.635). The latent's state information is linear and insufficient. The readout
is not the bottleneck.

## Verdict

**Phase 2 is not confirmatory, and the guardrail is real.**

The occluded-door manipulation is right and the direction signal appeared once,
but the state guardrail never cleared and the direction margin is unstable. The
oracle check rules out the "0.80 is impossible" explanation: the observation
supports **0.935** 4D state R². The JEPA is losing state information that a
47k CNN recovers. Running five seeds at state R² ~0.66–0.70 would produce five
inconclusive runs. We stopped before that.

## The next move

A **separate state encoder with gradient isolation** — a small CNN trained only
on the state loss, independent of the JEPA backbone and the pair heads. The
oracle proves that such a CNN can reach the state; the shared JEPA encoder is
the thing to fix. If the separate encoder also fails, the problem is the data or
the task, not the architecture.

The broader lesson: for this world, the bottleneck is the **state path inside
the JEPA**, not the pair weight, the parameter count, or the readout. More
capacity is not a substitute for a state path that actually sees the puck.
