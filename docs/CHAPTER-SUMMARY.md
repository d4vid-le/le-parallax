# Chapter summary — LeParallax, first pass

**Date:** 2026-11-12.
**Question:** Does same-state good-view supervision produce a world-state
representation that is more nuisance-invariant, and does that advantage survive
into control and continual adaptation?

**Answer:** Yes, in a synthetic 96px world, with a small action-conditioned JEPA.
The advantage appears at the representation level, in out-of-distribution
control, and in retention after a nuisance shift. It is not absolute: residual
nuisance remains, forgetting is reduced not eliminated, and one seed shows a
stability–plasticity trade-off.

---

## 1. The three confirmed results

| Result | Experiment | Metric | Effect | Evidence |
|---|---|---|---|---|
| **Mechanism** | H1 confirmatory | same-state nuisance ratio | **0.114** of temporal (95% CI 0.089–0.143), **5/5** seeds; state probe 0.90–0.95 vs 0.69–0.87; bad views **6.71** | `docs/H1-CONFIRM.md` |
| **Control** | H2-v4 success replication | OOD success | **+23 points** (68% → 91%), 95% CI +10 to +37, **4/5** seeds, McNemar **p = 0.0001**; OOD distance −0.036 | `docs/H2V4-CONFIRM.md` |
| **Retention** | H3 forgetting | A-state R² after B adaptation | **+0.229** (95% CI 0.161–0.298), **5/5** seeds; forgetting **0.025 vs 0.132** (~81% less); B post +0.005 mean | `docs/H3-CONFIRM.md` |

The one-line story:

> Good views force the encoder to keep what survives independent nuisance. That
> cleaner state geometry makes the latent more readable, makes control more
> robust to a nuisance shift, and makes continual adaptation forget less.

---

## 2. What failed, and why it matters

| Failure | What happened | What it taught |
|---|---|---|
| **H1b** | Absolute state floor ≥0.75 failed on one temporal seed (0.687), while good was 0.898 and higher in 5/5. | Non-inferiority should be relative to the comparator, not an absolute floor on both arms. H1a/H1c pass; the full H1 verdict is a literal fail. |
| **H2** | CEM with true dynamics succeeded 28%; random 34%; a true-state P-controller 100%. | H2 was a planner test, not a representation test. |
| **H2-v2** | State-feedback + ridge readout tied at 0.20 success; readout R² negative OOD (−0.16 good vs −1.29 temporal). | The readout/encoder lacked support for the shifted nuisance. |
| **H2-v3** | Domain randomization unlocked control; distance −0.026 vs a −0.05 target (fail), but success 88% vs 60% (+28 points). | The distance endpoint was near-saturated; success was the control metric that mattered. |
| **H3 seed 4** | Good lost B performance (0.789 → 0.626) while gaining the most A (0.578 → 0.925). | No systematic plasticity penalty on average, but the stability–plasticity trade-off is real in at least one seed. |

These are not embarrassments. They are the reason the surviving claims are
credible.

---

## 3. What is actually established

- **Positive-pair geometry is a design variable.** Same-state views change what
  the latent encodes; this is not a scale effect.
- **The effect is specific.** Bad views (crops of one observation) make nuisance
  sensitivity worse (6.71 vs 2.54), so the gain is not generic augmentation.
- **It survives control.** With domain-randomized encoders, a fixed readout, and
  a valid state-feedback controller, good views improve OOD success by 23
  points.
- **It survives adaptation.** After fine-tuning on the shifted domain, the
  good-view latent retains the original state geometry far better (forgetting
  0.025 vs 0.132).
- **No free lunch.** Residual nuisance remains; forgetting is reduced, not
  eliminated; one seed pays a plasticity cost.

## 4. What is not established

- Real-world data, language, or cross-modal transfer.
- Control retention (H3 is representation-level).
- Dynamics-shift retention. Good views should not protect against a changed
  dynamics model; that needs adaptation.
- A general claim about all world models. This is one synthetic world, one small
  model, five seeds.
- Full H1 as pre-registered. H1b failed literally.
- Full H2-v3 distance. It failed; H2-v4 is the success-primary replication.

## 5. Why anyone should care

The Platonic JEPA theorem says positive pairs determine whether representations
are about the world or the modality. LeWM and other world models train on
temporal transitions, which the theorem does not cover. This chapter tests the
gap:

- **Mechanism:** good views reduce nuisance sensitivity by ~89% relative.
- **Control:** that advantage becomes +23 points of OOD success.
- **Continual learning:** it becomes ~81% less forgetting after a nuisance
  shift.

The practical lesson is simple and cheap: if you can collect two observations of
the same state with independently varying nuisance, add a same-state consistency
term. It is a representation regularizer that pays off in control and retention,
without changing the planner or the model size.

## 6. Next

1. **H3b — dynamics shift.** Show the boundary: good views should not protect
   against a changed dynamics model. That would complete the stability–plasticity
   story.
2. **H4 — cross-modal rotation.** Separately trained vision/proprio encoders
   align up to a rotation; test modularity.
3. **Write-up.** The three-part result is coherent enough for a research note.
   The failures belong in the paper, not the appendix.

The honest short version:

> Positive views, not scale alone, are the lever. In a small world model, good
> views produced a more world-like latent, better OOD control, and less
> forgetting after a nuisance shift — with residual nuisance and one visible
> plasticity trade-off.
