# Pre-registration — Collision suite 2

**Status:** **FROZEN 2026-11-12.**
**Purpose:** two interventions that can fail loudly:

1. **Adversarial nuisance stress test + counterfactual nuisance alignment
   (CNA).** Does good-view invariance survive held-out and OOD nuisance, and
   can a new counterfactual data pipeline buy that robustness?
2. **Le-Gates on the dynamics boundary.** Can a frozen-encoder residual adapter
   recover shifted dynamics without forgetting the original dynamics, and does
   the gate need an error signal to route?

The previous suite falsified one bet (familiar-nuisance swap) and complicated
the other (Seed 4 rank). This suite is deliberately harder.

---

## Collision 1b — adversarial nuisance stress test

### Diagnostic

**Question.** Is the good-view invariance specific to the nuisance
distribution it was trained on?

**Conditions.** Probe trained on normal A training latents (80 episodes × 16
states). Tested on held-out A states (10 episodes × 16 states) re-rendered
under:

- **normal**: the episode's own nuisance;
- **heldout-swap**: a nuisance from another **test** episode (never in the
  probe's training set);
- **shifted**: `sample_nuisance_shift` (harsher light, brighter background,
  darker puck, larger camera offset — OOD);
- **extreme**: one of four corner combinations of the shifted ranges, with all
  nuisance dimensions at an edge simultaneously:
  - strength 1.10, bg 0.35, puck 0.20, cam +0.08;
  - strength 0.80, bg 0.20, puck 0.80, cam −0.08;
  - strength 1.10, bg 0.20, puck 0.80, cam +0.08;
  - strength 0.80, bg 0.35, puck 0.20, cam −0.08.

**Bet 1b-d.** The good-view OOD drop
`R²(normal) − R²(shifted)` is **≥0.10**, and good-view shifted R² is **not more
than 0.05 above** temporal shifted R². This says the invariance is
distribution-specific.

- If the drop is <0.10 and good shifted R² is ≥ temporal + 0.05, the bet is
  wrong: the invariance generalizes OOD.
- The extreme condition is exploratory; no confirmatory claim.

### Intervention — Counterfactual Nuisance Alignment (CNA)

**New data pipeline.** For each of 80 normal training episodes and 8 sampled
states, render three views of the **same state**:

1. the episode's original nuisance,
2. a nuisance drawn from a **different training episode**,
3. a nuisance drawn from `sample_nuisance_shift` (OOD).

**New alignment loss.** Fine-tune the H1 good encoder for 5 epochs with:

- the standard temporal JEPA loss on normal A transitions (keeps dynamics),
- pairwise latent MSE between the three counterfactual views,
- state-head loss on each view,
- SIGReg, variance, and covariance terms on each view.

**Controls.**

- `frozen`: the H1 good encoder, no adaptation.
- `normal-ft`: 5 epochs of the same standard temporal loss on normal A only
  (controls for extra training).
- `mix-ft`: 5 epochs of the same standard temporal loss on 40 normal + 40
  shifted episodes (controls for merely seeing shifted data).

**Bet 1b-i.** CNA improves shifted R² by **≥0.10** over `frozen`, loses **≤0.03**
normal R², and beats `mix-ft` on shifted R² by **≥0.03**. If CNA fails to beat
`mix-ft`, the counterfactual alignment adds nothing beyond domain exposure.

Five seeds, 0–4. Report per-seed R² for all arms and conditions.

---

## Collision 3 — Le-Gates on the dynamics boundary

**Setup.** Freeze the H1 good-view encoder and the H1 base predictor. Adapt only
a residual adapter on shifted dynamics B `[damping=0.65, gain=0.32]`, 10 epochs,
lr 1e-3, batch 32. The adapter is zero-initialized, so the adapted model starts
identical to the base. Adapted prediction:

```
z' = base(z, a) + 0.1 · g · adapter(z, a, e_prev)
```

**Gate arms.**

- **global**: scalar gate `g = sigmoid(θ)`, same for every input.
- **input**: scalar gate `g = sigmoid(MLP(z, a))`.
- **error**: scalar gate `g = sigmoid(MLP(z, a, e_prev))`, where `e_prev` is the
  frozen base predictor's MSE on the **previous transition** in the episode,
  standardized with B-training statistics. This is a causal, online dynamics-
  error signal, not the current target.

**Metric.** One-step predictor MSE on held-out A and B transitions, before and
after adaptation. Per seed:

- `A_rel_forget = (A_post − A_pre) / A_pre`
- `B_rel_gain = (B_pre − B_post) / B_pre`
- gate mean on A and B, and gate AUC for classifying A vs B transitions.

**Baselines.** `frozen` (base model) and `full-ft` (the H3b good-arm full
fine-tune checkpoints).

**Bets.**

- **3a — global fails.** In ≥3/5 seeds, the global gate fails
  `B_rel_gain ≥ 0.20` **or** `A_rel_forget ≤ 0.05`. A global gate that opens for
  B should also fire on A.
- **3b — input fails.** The input gate fails the same criterion in ≥3/5 seeds,
  and its A-vs-B gate AUC is ≤0.65. `(z, a)` does not identify the dynamics
  regime, so the gate cannot separate the domains.
- **3c — error succeeds.** The error gate passes both criteria in ≥4/5 seeds,
  with mean gate on B at least 0.20 above mean gate on A. The base predictor's
  error is the dynamics signal the gate needs.

If the error gate also fails, the adapter architecture is insufficient even
with an error signal; that is a negative result for this Le-Gates prototype and
will be reported as such.

**Surprise conditions.** Global succeeds, or input succeeds with AUC >0.65, or
error fails. Any of these changes the Le-Gates story and must be reported
against the bet.

---

## Reporting

Per-seed tables, means, and the exact count of seeds meeting each criterion.
No averaging away a failed seed. If a bet loses, the paper's claim is revised
rather than the threshold. A one-seed smoke test may be run to validate shapes
and optimization only; it is disclosed and is not part of the confirmatory
count.
