# Pre-registration — Le-Gates v3: hard error gate

**Status:** **FROZEN 2026-11-12.**
**Motivation:** Le-Gates v2 (10% A replay + distillation) reduced forgetting but
did not stop it. On seeds 0–1 the soft error gate routed (AUC 0.72–0.75) but
still opened on A, leaving **+60% to +84%** relative A forgetting. The gate
learns the right direction but the soft sigmoid leaks.

v3 replaces the soft gate with a **hard 0/1 gate** at evaluation, calibrated on
the A replay anchor. This is the structural version of the same router:

- train the residual adapter on B with the gate fixed at 1 (full residual),
- calibrate `τ` as the **95th percentile** of the A-anchor previous-transition
  errors (so at most ~5% of A transitions can open),
- at evaluation, apply the adapter only when `e_prev > τ`.

A is preserved exactly on every non-routed transition; B uses the full adapter
where the base predictor is inaccurate. This matches the earlier Le-Gates
evidence that hard gates give zero non-routed forgetting by construction.

---

## Setup

- Frozen H1 good-view encoder and base predictor.
- Residual adapter zero-initialized, 10 epochs on B `[0.65, 0.32]`, lr 1e-3,
  batch 32, gate fixed at 1 during training.
- A anchor: 8 of the 80 A training episodes (~10%), used only to calibrate `τ`.
- `e_prev` is the previous transition's raw frozen-base MSE.
- Five seeds, 0–4. Held-out A and B transitions.

## Endpoint

In **≥4/5 seeds**:

1. `A_rel_forget ≤ 0.05`,
2. `B_rel_gain ≥ 0.20`,
3. routing fractions `frac_A ≤ 0.10` and `frac_B ≥ 0.70`.

If `frac_A` is high, the A/B error distributions overlap and a hard threshold
cannot isolate A. If `frac_B` is low, the threshold is too conservative and B
loses the adapter.

**Bet:** the hard gate passes all three criteria in ≥4/5 seeds, because the
error signal already routes with AUC ≈0.75 and A is preserved on non-routed
transitions by construction.

**Surprise conditions:** hard gate fails A retention (threshold overlap), or
fails B gain (too few B transitions routed), or both. Any failure is reported
against the bet.

## What this does not establish

- A general hard-routing solution; τ is calibrated with 10% A replay.
- Control retention.
- That the soft gate is useless; v2 shows it learns the routing direction but
  leaks. v3 tests whether hard isolation closes the leak.
