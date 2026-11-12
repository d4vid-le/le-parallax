# Chapter summary — LeParallax, first pass

**Date:** 2026-11-12.
**Question:** Does same-state good-view supervision produce a world-state
representation that is more nuisance-invariant, and does that advantage survive
into control and continual adaptation?

**Answer:** Yes in-distribution, conditionally out of distribution. In a
synthetic 96px world with a small action-conditioned JEPA, good views improve
representation, OOD control (with support), and retention after a nuisance
shift. The collision suite shows the invariance is **distribution-specific**:
without support, absolute OOD state decoding still fails, though good degrades
far less than temporal. Support plus counterfactual state alignment restores
usable OOD geometry. Le-Gates-style adapters need a retention signal and a
reliable router; soft gates leak, and a hard gate with 10% replay routes too much
of the original domain.

---

## 1. The core results and the boundary

| Result | Experiment | Metric | Effect | Evidence |
|---|---|---|---|---|
| **Mechanism** | H1 confirmatory | same-state nuisance ratio | **0.114** of temporal (95% CI 0.089–0.143), **5/5** seeds; state probe 0.90–0.95 vs 0.69–0.87; bad views **6.71** | `docs/H1-CONFIRM.md` |
| **Control** | H2-v4 success replication | OOD success | **+23 points** (68% → 91%), 95% CI +10 to +37, **4/5** seeds, McNemar **p = 0.0001**; OOD distance −0.036 | `docs/H2V4-CONFIRM.md` |
| **Retention** | H3 forgetting | A-state R² after B adaptation | **+0.229** (95% CI 0.161–0.298), **5/5** seeds; forgetting **0.025 vs 0.132** (~81% less); B post +0.005 mean | `docs/H3-CONFIRM.md` |
| **Boundary** | H3b dynamics shift | A predictor MSE after B adaptation | good **worse** by **0.00626** (5/5 seeds); both adapt to B (B post essentially tied) | `docs/H3B-CONFIRM.md` |

### Collision results

| Result | Experiment | Metric | Effect | Evidence |
|---|---|---|---|---|
| **OOD diagnostic** | adversarial nuisance, 5 seeds | state R² under shifted nuisance | good **−0.049** vs temporal **−0.840**; absolute failure, relative gain **+0.791** | `docs/COLLISION-1B.md` |
| **CSA** | support + state alignment | shifted state R² | **0.828** vs mix-ft **0.728**; **+0.101** (CI +0.050 to +0.180), 5/5 | `docs/COLLISION-1B.md` |
| **CNA-v1** | alignment without support | normal / shifted R² | normal **0.929 → 0.739**; shifted 0.282 vs mix-ft 0.781 (pilot) | `docs/COLLISION-1B.md` |
| **Capacity** | Seed 4 vs Seed 1 | effective rank | Seed 4 lower in **4/4** good conditions, but **not** global collapse; domain-specific allocation | `docs/COLLISION-1-2.md` |
| **Le-Gates v1** | soft gate, no anchor | A predictor MSE | all gates forget A **+146% to +155%** | `docs/COLLISION-3.md` |
| **Le-Gates v2** | soft gate + 10% anchor | A predictor MSE | error gate forgets **+104%**, routing AUC 0.667; still leaks | `docs/COLLISION-3.md` |
| **Le-Gates v3** | hard gate + 10% anchor | A predictor MSE | forgets **+26%**, routes **18.5%** of A; router is the bottleneck | `docs/COLLISION-3.md` |

The one-line story:

> Good views force the encoder to keep what survives independent nuisance. That
> cleaner state geometry makes the latent more readable, makes control more
> robust to a nuisance shift, and makes continual adaptation forget less — but
> it does not protect against a dynamics shift. The mechanism is specific, and
> the invariance is distribution-specific: without support, absolute OOD state
> decoding fails; support plus state alignment restores it. Gating is not
> retention; the router is the bottleneck.

---

## 2. What failed, and why it matters

| Failure | What happened | What it taught |
|---|---|---|
| **H1b** | Absolute state floor ≥0.75 failed on one temporal seed (0.687), while good was 0.898 and higher in 5/5. | Non-inferiority should be relative to the comparator, not an absolute floor on both arms. H1a/H1c pass; the full H1 verdict is a literal fail. |
| **H2** | CEM with true dynamics succeeded 28%; random 34%; a true-state P-controller 100%. | H2 was a planner test, not a representation test. |
| **H2-v2** | State-feedback + ridge readout tied at 0.20 success; readout R² negative OOD (−0.16 good vs −1.29 temporal). | The readout/encoder lacked support for the shifted nuisance. |
| **H2-v3** | Domain randomization unlocked control; distance −0.026 vs a −0.05 target (fail), but success 88% vs 60% (+28 points). | The distance endpoint was near-saturated; success was the control metric that mattered. |
| **H3 seed 4** | Good lost B performance (0.789 → 0.626) while gaining the most A (0.578 → 0.925). | No systematic plasticity penalty on average, but the stability–plasticity trade-off is real in at least one seed. |
| **Collision 1** | Good-view swap drop +0.001, not ≥0.10; temporal improved −0.063. | The swap used training nuisance identities. It is a familiar-nuisance test, not OOD. |
| **Collision 1b pilot** | Seed 0 made good look worse than temporal OOD (good −0.877 vs temporal −0.592). | Five seeds reversed it (good −0.049 vs temporal −0.840). One-seed collisions can point the wrong way. |
| **Collision 1b full** | Absolute good-view OOD state R² is −0.049. | The invariance does not transfer as a usable latent; support is a prerequisite. |
| **CNA-v1** | Normal R² 0.929 → 0.739; shifted 0.282 vs mix-ft 0.781. | Latent-MSE alignment without shifted support collapses state-relevant dimensions. |
| **Collision 2** | Seed 4 rank bet confirmed, but Seed 4 good A rank is higher than temporal and Seed 1 good has lower B rank with better B R². | The anomaly is domain-specific capacity allocation, not global rank collapse. |
| **Le-Gates v1** | All gates forget A +146% to +155%. | A gate trained only on B has no signal to close on A. |
| **Le-Gates v2** | Soft error gate routes (AUC 0.667) but forgets +104%. | Routing is not retention; the sigmoid leaks. |
| **Le-Gates v3** | Hard gate forgets +26%, routes 18.5% of A. | The router, not the gate scalar, is the bottleneck. |
| **Anti-Equivalence** | One-way door; volume arm Δasym +7.32 but forward R² 0.184 / reverse R² −5.08 and latent std 0.599; asym Δasym −0.115 vs symmetric −0.013, lower on 4/5 seeds; wrong-sign reach AUC 0.670. | The determinant penalty collapses; the matched reversible control removes the direction effect; the wrong-sign control gets the same reachability gain. Direction is not volume. |
| **Phase 2 pilot** | Occluded door + matched heads; `asym_clean` Δasym +0.146 vs `sym_head` −0.055, bounded-volume std 1.044, but state R² 0.308/0.322 vs the 0.8 guardrail and planner AUC 0.722/0.750. | The partial-observation manipulation is right and the direction signal flips positive, but the pair loss/capacity is the bottleneck. Inconclusive, not confirmatory. |
| **Phase 2 tuning** | Pre-merge frame state head + `λ_pair=0.08`: state R² 0.670/0.671/0.695, direction +0.080 vs −0.019 (margin +0.099), planner 0.694/0.694/0.750. | The decoupling fix nearly balances state and direction, but the 0.8 state guardrail and the +0.10 direction margin are still missed. Not confirmatory. |
| **Phase 2 capacity** | 340k params, 60 epochs, cosine, `λ_pair=0.10`, `λ_var=0.10`: state R² 0.610/0.509/0.662/0.623, direction margin −0.009, bounded std 0.882; frame probe R² negative, frame head 0.24–0.31. | More capacity did not fix state; the merged latent/observation is the bottleneck. The 5-seed matrix is not justified. |

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
- **The mechanism is specific.** It does not protect against a dynamics shift;
  good was slightly worse on A in H3b, while both arms adapted to B.
- **The invariance is distribution-specific.** Under shifted nuisance, good
  state R² is −0.049 in absolute terms; it is +0.791 better than temporal, but
  neither latent is usable without support.
- **Support + alignment works.** CSA (mixed support + state-head alignment +
  weak latent term) reaches 0.828 shifted R², beating mix-ft by +0.101 with no
  normal loss.
- **Gating is not retention.** Le-Gates v1/v2/v3 show the progression: no
  anchor → +146%; soft + anchor → +104%; hard + anchor → +26% but 18.5% of A
  routed. The router is the bottleneck.
- **The capacity anomaly is allocation, not collapse.** Seed 4's good latent has
  lower rank in all good conditions, but A rank is higher than temporal and
  Seed 1 has lower B rank with better B R². Rank alone does not predict R².
- **No free lunch.** Residual nuisance remains; forgetting is reduced, not
  eliminated; one seed pays a plasticity cost; the OOD fix needs support and
  alignment.

## 4. What is not established

- Real-world data, language, or cross-modal transfer.
- Control retention (H3 and CSA are representation-level).
- Zero-shot OOD usability. Good views degrade gracefully but absolute shifted
  state R² is −0.049; support is required.
- Dynamics-shift retention. Good views do not protect against a changed dynamics
  model; the Le-Gates adapter needs a reliable router and does not reach zero
  forgetting at 10% replay.
- A causal capacity story for Seed 4. The autopsy is one seed pair; it rejects a
  simple global rank-collapse explanation, not every allocation explanation.
- A directed semi-lattice from asymmetric pair loss. The five-seed one-way-door
  test falsified the operationalized version; the determinant/log-volume penalty
  collapsed the latent, and the matched reversible control removed the apparent
  direction effect.
- A definitive confirmation from Phase 2. The occluded-door pilot flips the
  direction sign positive but fails the state R² guardrail (0.308/0.322 vs 0.8)
  and does not improve the planner. The frame-head tuning gets to 0.670–0.695,
  and the expanded 340k model is worse (0.610–0.662, margin −0.009). It is
  suggestive, not confirmatory.
- A general claim about all world models. This is one synthetic world, one small
  model, five seeds.
- Full H1 as pre-registered. H1b failed literally.
- Full H2-v3 distance. It failed; H2-v4 is the success-primary replication.
- CNA-v1 as a five-seed result. It was a seed-0 pilot; the design was superseded
  by CSA before the confirmatory budget was spent.

## 5. Why anyone should care

The Platonic JEPA theorem says positive pairs determine whether representations
are about the world or the modality. LeWM and other world models train on
temporal transitions, which the theorem does not cover. This chapter tests the
gap:

- **Mechanism:** good views reduce nuisance sensitivity by ~89% relative.
- **Control:** that advantage becomes +23 points of OOD success when the encoder
  has support.
- **Continual learning:** it becomes ~81% less forgetting after a nuisance
  shift.
- **Collision:** absolute OOD state decoding still fails without support; CSA
  (support + state alignment) restores it and beats domain exposure.
- **Adaptation:** a frozen-encoder Le-Gates adapter needs a retention signal and
  a reliable router; a soft gate leaks, and a hard gate with 10% replay routes
  too much A.
- **Direction is not volume:** a determinant/log-volume penalty can manufacture
  a huge A→B asymmetry by collapsing the latent; the reverse R² goes negative,
  the latent std shrinks, and a matched reversible control removes the apparent
  direction effect.

The practical lesson is conditional: **support first, then alignment.** If you
can collect two observations of the same state with independently varying
nuisance, add a same-state consistency term — but only after the encoder has
seen the nuisance distribution you care about. Alignment without support
(CNA-v1) collapses normal state readability. Alignment with support and
state-level supervision (CSA) beats ordinary domain exposure. On the dynamics
side, gating is not retention: the router must be reliable before a hard gate
can guarantee isolation.

## 6. Next

1. **A better router.** The Le-Gates result says the gate is not the bottleneck;
   the detector is. Test more A replay (25–50%), a learned detector with a
   false-open penalty, or per-layer masks/frozen slots.
2. **Phase 2 scale-up.** The occluded-door pilot flips the direction sign
   positive but fails the state guardrail (0.308/0.322 vs 0.8) and does not
   improve the planner. Lower the pair weight to 0.05–0.10, lower the variance
   floor to 0.1–0.25, scale the encoder/latent, and add a keypoint-specific
   state head. Re-run five seeds only after the temporal state R² is ≥0.8.
3. **Control retention.** CSA and the Le-Gates prototype are predictor-level.
   The next test is whether the OOD geometry and the router preserve control,
   not just state R²/MSE.
4. **H4 — cross-modal rotation.** Separately trained vision/proprio encoders
   align up to a rotation; test modularity.
5. **Write-up.** The conditional result — mechanism, control, retention, OOD
   boundary, CSA fix, router boundary, and the failed directed-manifold test —
   is coherent enough for a research note. The failures belong in the paper,
   not the appendix.

The honest short version:

> Positive views, not scale alone, are the lever — but the lever is conditional.
> In a small world model, good views produced a more world-like latent, better
> OOD control with support, and less forgetting after a nuisance shift. Absolute
> OOD state decoding still fails without support; support plus state alignment
> fixes it. Gating is not retention; the router is the bottleneck.
