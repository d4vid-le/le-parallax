# Pre-registration — Le-Gates v2: retention-anchored adapters

**Status:** **FROZEN 2026-11-12.**
**Motivation:** Le-Gates v1 failed on seed 0. All three gates forgot A
(+131% to +136% relative predictor MSE) because the adapter was trained only on
B. The error gate never learned to close on A: it had no A examples during
adaptation, so a constant ~0.6 gate was sufficient to fit B. The error signal
was available but unused.

v2 adds the missing retention signal, not a new threshold:

- a small **A replay anchor** (8 of the 80 A training episodes, ~10%),
- a **distillation loss** on A: the adapted prediction must match the frozen
  base prediction, so the gate learns to stay closed where the base is already
  correct.

The anchor is a standard continual-learning budget, not test data. It is the
smallest change that makes the gate's routing decision learnable.

---

## Setup

Same as Collision 3 / Le-Gates v1:

- frozen H1 good-view encoder and base predictor,
- residual adapter, zero-initialized, 10 epochs on B `[0.65, 0.32]`,
- lr 1e-3, batch 32,
- gates: global, input `(z, a)`, error `(z, a, e_prev)`,
- `e_prev` is the previous transition's frozen-base MSE, standardized with B
  training statistics.

v2 adds:

- `lambda_anchor = 1.0`,
- 8 A training episodes (248 transitions) sampled independently each batch,
- anchor loss `mean((adapted_A − base_A)²)`.

Five seeds, 0–4. Evaluate on held-out A and B transitions.

## Endpoint

- **3c-v2 error+anchor succeeds:** in ≥4/5 seeds, `B_rel_gain ≥ 0.20` and
  `A_rel_forget ≤ 0.05`, with mean gate on B at least **0.20 above** mean gate
  on A.
- **3a-v2 global+anchor fails:** in ≥3/5 seeds it misses one of the two
  criteria. A single scalar cannot be both open on B and closed on A.
- **3b-v2 input+anchor:** report pass count and A-vs-B gate AUC. The bet is
  that it still fails the same criterion in ≥3/5 seeds with AUC ≤0.65, because
  `(z, a)` does not identify the dynamics regime. If it succeeds with AUC >0.65,
  that is a surprise and a positive routing result.

## Surprise conditions

- Error+anchor fails: the gate cannot use the error signal even with a retention
  anchor.
- Global+anchor succeeds: a scalar gate is enough, which would contradict the
  routing story.
- Input+anchor succeeds: state/action context is sufficient to identify the
  dynamics regime.

Any surprise is reported against the bet. Per-seed values, not just means.

## What this does not establish

- A general continual-learning solution; the anchor is 10% replay.
- Control retention; this is predictor MSE.
- That the anchor is necessary for every architecture; it is necessary here
  because the gate is otherwise unidentifiable on A.
