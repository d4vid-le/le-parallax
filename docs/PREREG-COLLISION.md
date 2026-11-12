# Pre-registration — Collision suite

**Status:** **FROZEN 2026-11-12.**
**Purpose:** force empirical collisions where the current story can lose.

The previous experiments were confirmatory and increasingly safe. This suite is
designed to falsify the nice story:

1. **Counterfactual nuisance swap.** If good-view invariance is causal, the
   state probe survives when a state is rendered with a *different state's*
   nuisance. If the invariance is only correlational, it breaks.
2. **Seed 4 capacity autopsy.** If Seed 4's high A-retention comes from latent
   rank collapse, its effective rank should be lower than Seed 1's.

---

## Collision 1 — counterfactual nuisance swap

**Prediction (the bet):** the good-view state probe **drops by ≥0.10 R²** when
the nuisance is swapped for another state's nuisance. The invariance learned
from independent nuisance pairs is correlational, not causal.

If the drop is ≤0.05, the bet is wrong: the good-view latent is causally
state-like, not just nuisance-independent.

**Design.**

- Five seeds.
- Probe trained on normal A training episodes (16 states each).
- Test on the same A states rendered with:
  - their own nuisance (normal), and
  - a nuisance drawn from a different training episode/state (swapped).
- Target is always the true state.
- Metric: state R² normal, state R² swapped, drop = normal − swapped.
- Compare temporal and good encoders.

**Falsifier of the bet:** good-view drop <0.05, or good-view drop is not larger
than temporal's.

## Collision 2 — Seed 4 capacity autopsy

**Prediction (the bet):** Seed 4's good-view latent has **lower effective rank**
than Seed 1's, consistent with a capacity/retention trade-off.

**Design.**

- Seeds 1 and 4; temporal and good encoders; pre-adaptation H1 and
  post-adaptation H3 checkpoints.
- Latents on A and B test episodes.
- Covariance eigenvalues; effective rank
  `exp(−Σ p log p)`, participation ratio `(Σλ)² / Σλ²`, top-1 variance share.
- Compare Seed 1 vs Seed 4, good vs temporal, pre vs post, A vs B.

**Falsifier of the bet:** Seed 4's effective rank is equal or higher than
Seed 1's on the relevant domain.

## Collision 3 — Le-Gates on the dynamics boundary (next)

- Freeze the good-view encoder.
- Replace the predictor with a residual adapter: global gate vs
  input-conditioned gate.
- Adapt on shifted dynamics B; measure A/B predictor MSE.
- **Bet:** the global gate cannot recover B without A drift; the
  input-conditioned gate recovers both only if the latent retains
  dynamics-relevant features.
- This is the prototype that turns H3b from a boundary into a mechanism test.

---

The point is not to collect another win. It is to find the first place where the
story breaks, and to say so before running it.
