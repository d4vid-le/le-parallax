# Collisions 1–2 — results

**Status:** complete, 2026-11-12.
**Protocol:** [`PREREG-COLLISION.md`](PREREG-COLLISION.md).
**Artifacts:** `runs/nuisance-collision/`, `runs/capacity-autopsy/` (git-ignored).

The point of this suite is to let the story lose. One bet was falsified, one was
confirmed but for a different reason than expected.

---

## Collision 1 — counterfactual nuisance swap

**Bet:** the good-view state probe drops by **≥0.10 R²** when a state is
re-rendered with another state's nuisance. If the drop is ≤0.05, the invariance
is causal rather than correlational.

Probe trained on normal A training latents; tested on the same held-out A states
with their own nuisance and with a nuisance drawn from a different **training**
episode.

| seed | temporal normal | temporal swap | temporal drop | good normal | good swap | good drop |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.850 | 0.866 | −0.016 | 0.919 | 0.918 | +0.001 |
| 1 | 0.817 | 0.873 | −0.056 | 0.938 | 0.925 | +0.012 |
| 2 | 0.725 | 0.873 | −0.147 | 0.958 | 0.958 | −0.000 |
| 3 | 0.816 | 0.853 | −0.037 | 0.921 | 0.928 | −0.007 |
| 4 | 0.818 | 0.874 | −0.056 | 0.942 | 0.944 | −0.002 |
| **mean** | **0.805** | **0.868** | **−0.063** | **0.936** | **0.935** | **+0.001** |

**Verdict: bet falsified.** The good-view probe does not drop at all
(+0.001 mean); the temporal probe actually *improves* under the swap (−0.063).
The state geometry is not tied to a particular nuisance identity.

**But the test is weaker than advertised.** The swapped nuisance was drawn from
a **training episode**, so its identity was already in the probe's training set.
Temporal's improvement is exactly what familiarity predicts: the probe has
learned a correction for those nuisance identities, and the swap moves the
latent toward a familiar regime. This is a **familiar-nuisance** test, not an
out-of-distribution stress test.

What it does support: good views do not memorize a single nuisance. What it does
**not** support: that the invariance generalizes to unseen or harsher nuisance.
H2-v4 already showed that support for the shifted nuisance was a prerequisite
for the control advantage. The next collision must use held-out and OOD nuisance
identities, not training identities.

---

## Collision 2 — Seed 4 capacity autopsy

**Bet:** Seed 4's good-view latent has **lower effective rank** than Seed 1's,
consistent with a capacity/retention trade-off.

Effective rank `exp(−Σ p log p)` of the 64-d latent covariance, 160 latents per
condition.

| seed | arm | stage | domain | effective rank | participation | top-1 share |
|---:|---|---|---|---:|---:|---:|
| 1 | temporal | pre | A | 7.70 | 5.83 | 0.320 |
| 1 | temporal | pre | B | 2.57 | 2.18 | 0.544 |
| 1 | temporal | post | A | 7.32 | 5.76 | 0.267 |
| 1 | temporal | post | B | 7.69 | 6.29 | 0.240 |
| 1 | good | pre | A | 6.89 | 5.54 | 0.305 |
| 1 | good | pre | B | 5.70 | 4.72 | 0.329 |
| 1 | good | post | A | 6.91 | 5.95 | 0.234 |
| 1 | good | post | B | 6.40 | 4.93 | 0.338 |
| 4 | temporal | pre | A | 7.55 | 6.17 | 0.246 |
| 4 | temporal | pre | B | 3.46 | 2.69 | 0.509 |
| 4 | temporal | post | A | 6.02 | 4.38 | 0.387 |
| 4 | temporal | post | B | 7.55 | 6.26 | 0.268 |
| 4 | good | pre | A | 6.41 | 5.42 | 0.281 |
| 4 | good | pre | B | 4.32 | 3.28 | 0.460 |
| 4 | good | post | A | 6.21 | 5.15 | 0.296 |
| 4 | good | post | B | 5.21 | 4.26 | 0.326 |

**Verdict: the directional bet is confirmed, but the simple mechanism is not.**
Seed 4's good latent has lower effective rank than Seed 1's in all four
good-view conditions (pre/post × A/B). But it is **not** a global bottleneck:

- Seed 4 good post-A rank **6.21** is *higher* than Seed 4 temporal post-A rank
  **6.02**, and Seed 4 good A R² is much better (0.925 vs 0.578). Rank is not
  tracking A performance.
- Seed 1 good post-B rank **6.40** is lower than Seed 1 temporal post-B **7.69**,
  yet Seed 1 good B R² is *better* (0.861 vs 0.765). Again, lower rank does not
  mean worse B performance.
- The informative gap is **within Seed 4, on B**: good rank **5.21** vs temporal
  **7.55**, and good B R² **0.626** vs temporal **0.789**.

The Seed 4 anomaly is better described as **domain-specific capacity
allocation**, not rank collapse: the good encoder keeps more of its geometry for
A and allocates fewer effective dimensions to B, which costs B accuracy. That is
a real stability–plasticity trade-off, but it is not a literal bottleneck, and
it does not generalize to Seed 1.

This kills the cleanest version of the capacity story. A causal test would need
to manipulate the rank directly (e.g., a rank penalty on B) and show that B
performance follows. The autopsy alone cannot establish causality, and n=1 seed
pair is not a law.

---

## What changes in the paper

1. The nuisance-swap result should be reported as a **familiar-nuisance
   counterfactual** that good views survive — not as evidence of causal
   invariance under arbitrary nuisance. The OOD stress test is still missing.
2. The Seed 4 trade-off should be described as **domain-specific allocation**,
   not rank collapse. The rank autopsy is evidence against the simplest
   bottleneck explanation.
3. Both results sharpen the next experiments: an adversarial held-out/OOD
   nuisance pipeline, and an adapter that can exploit a dynamics-error signal.
