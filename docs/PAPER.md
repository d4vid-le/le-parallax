# LeParallax: What Good Views Buy, and Where They Break

### Positive-pair geometry in an action-conditioned world model

**Research note / preprint draft.** Not peer reviewed.
Code, pre-registrations, and reproduction scripts:
`github.com/d4vid-le/le-parallax`.

---

## Abstract

The Platonic Representation Hypothesis conjectures that models trained on
different modalities converge to a shared representation of the world. A recent
theorem makes this precise for JEPAs: when positive pairs are two observations of
the same world state with independent nuisance — *good views* — separately trained
encoders agree up to one rotation. The theorem assumes an identity predictor.
Action-conditioned world models, which learn from temporal transitions, are
outside its scope.

We test whether the good-view recipe transfers to a small action-conditioned
world model: a **142,564-parameter** CNN/predictor trained on a synthetic 2D world
with controllable state, nuisance, and dynamics. Across five seeds:

1. **Mechanism.** Good views reduce the same-state nuisance ratio to **0.114** of
   a temporal baseline (95% CI 0.089–0.143, 5/5 seeds) while raising the state
   probe from 0.69–0.87 to 0.90–0.95. Bad views (crops of one observation) are
   **6.71**, worse than temporal.
2. **Control.** With domain-randomized encoders and a fixed state-feedback
   controller, good views improve out-of-distribution success from **68% to
   91%** (**+23 points**, 95% CI +10 to +37, 4/5 seeds, pooled McNemar
   **p = 0.0001**).
3. **Retention.** After adapting to a nuisance shift, good views retain the
   original domain's state probe **+0.229** better (95% CI 0.161–0.298, 5/5) and
   forget **0.025 vs 0.132** (~81% less).
4. **Boundary.** After a **dynamics** shift, good views have no retention
   advantage; they are slightly worse on the original dynamics while both arms
   adapt.

The effect is conditional, and the collision suite says where. Under a held-out
nuisance shift with no support, the good-view latent is still not linearly
state-decodable — mean shifted R² **−0.05** — though it degrades far less than
temporal (**−0.84**). A counterfactual state alignment (CSA) pipeline with
mixed-domain support restores shifted R² to **0.83**, beating a domain-exposure
control by **+0.10** with no normal loss. On a dynamics shift, a frozen-encoder
Le-Gates adapter forgets the original dynamics: soft gates forget **+104% to
+146%**, and a hard error gate cuts that to **+26%** but still routes **18.5%**
of the original domain. The gate is the policy; isolation is the guarantee, and
the router is the bottleneck. An anti-equivalence test on a one-way door also
failed across five seeds: the determinant-style volume penalty collapsed the
latent, and a matched reversible control removed the apparent direction effect.

The failures are part of the result: one pre-registered state-floor endpoint
failed on a baseline seed, CEM was a broken planner before the control test, a
one-seed OOD pilot reversed under five seeds, and the Seed 4 capacity anomaly is
domain-specific allocation rather than rank collapse. We report all of them.

---

## 1. Introduction

Joint-Embedding Predictive Architectures (JEPAs) learn by predicting in a compact
latent space rather than reconstructing pixels. The Platonic Representation
Hypothesis (PRH) observes that separately trained vision and language models
become more similar as they scale. A recent theorem turns that observation into a
statement about positive pairs: if two modalities' positives are observations of
the same world state with independent nuisance, their encoders agree up to one
rotation; if the positives are corruptions of one observation, the encoder spends
dimensions on modality nuisance.

The theorem's predictor is the identity. World models such as LeWM, Dreamer, and
V-JEPA-style planners use an action-conditioned predictor over temporal
transitions. That regime is outside the proof:
`(o_t, a_t, o_{t+1})` pairs are two *different* states, not two views of one state.
It is therefore an open empirical question whether good views help a world model,
hurt it, or do nothing.

This note answers that question in a controlled setting. We keep the model small
(142,564 parameters), the world synthetic (a 2D puck with controllable nuisance
and dynamics), and the protocol pre-registered. The contribution is not a new
architecture. It is a set of measurements that separate five claims:

- **Mechanism:** do good views make the latent more state-like?
- **Control:** does that advantage survive into planning/control?
- **Retention:** does it reduce forgetting after a domain shift?
- **Boundary:** is the advantage specific to nuisance, or does it also cover
  dynamics?
- **Collision:** does the invariance survive held-out and out-of-distribution
  nuisance, and can an adapter handle the dynamics boundary?

The answer is yes, yes, yes, no, and only conditionally. The boundary and the
collisions are what make the first three credible. A good-view latent is not a
generic invariance fix: it is an in-distribution state geometry that degrades
gracefully and needs support plus alignment before it is usable out of
distribution.

---

## 2. Related work

**Platonic representation and JEPAs.** Huh et al. observed cross-modal
representation convergence with scale. Balestriero and LeCun prove a Platonic
Representation Theorem for JEPAs with shared positives and bound the
misalignment when positives differ; next-token prediction is analyzed as a
bad-view JEPA. LeJEPA introduces SIGReg, the Gaussianity regularizer used here.
LeWM applies a two-term JEPA objective to pixels with an action-conditioned
predictor. Our work uses the same substrate and tests the theorem's uncovered
regime.

**Action-conditioned world models.** Dreamer, TD-MPC, and LeWM learn latent
dynamics for control. Their training signal is transition prediction, not
same-state view invariance. The question of what the latent represents is
usually evaluated post hoc with probes or planning success. We make the
positive-pair definition the independent variable.

**Continual learning and gated adaptation.** Le-Gates, the author's gated
residual adapter family, freezes an old path and learns a scalar-gated residual
for a new domain. In earlier experiments the soft single gate reduced
old-domain drift from +1015% (naive fine-tuning) to +169%, not zero; masked
growable slots reached 0.0025 → 0.0026; hard 0/1 gates with a frozen base gave
zero non-routed forgetting by construction, with a Σ≤1 composition constraint.
Le-Gates is a dynamics-adaptation mechanism. We evaluate a minimal version of it
on the H3b boundary (§4.7): a frozen-encoder residual adapter with global,
input-conditioned, and error-conditioned gates. The result is consistent with
the earlier evidence — soft gates leak, hard gates isolate only where the router
is correct, and the router is the bottleneck. H3b remains the evidence that good
views and dynamics adaptation are different problems.

---

## 3. Method

### 3.1 Synthetic world

A puck moves in a unit square under continuous forces. State
`s = [x, y, vx, vy]`; action `a = [fx, fy]` in `[-1, 1]`; dynamics

```
v' = damping · v + gain · a
x' = clip(x + v', 0.05, 0.95)
```

with walls absorbing normal velocity. The default dynamics are
`[damping=0.85, gain=0.20]`; the dynamics-shift experiment uses `[0.65, 0.32]`.

Observations are 96×96 RGB renders of the puck and a goal ring. Nuisance is
lighting direction/strength, background color, puck color, and camera offset.
Nuisance is sampled independently of state. A **good view** is two renders of the
same state with independent nuisance; a **bad view** is two crops of one render;
a **temporal pair** is `(o_t, a_t, o_{t+1})`.

![Same state, two nuisance styles](../snapshots/overview.png)

### 3.2 Model

A three-frame history is encoded by a shared CNN with a spatial-softmax keypoint
head (one expected `(x, y)` per channel), merged into a 64-d latent. A residual
predictor maps `[z, a]` to `z + 0.1·f(z, a)`. The training objective combines:

- next-latent MSE,
- SIGReg,
- variance and covariance terms,
- a contrastive next-latent term,
- a linear state readout,
- a two-step rollout loss.

The full model is **142,564 parameters**. The state readout and two-step loss are
auxiliary objectives shared by every arm; the positive-pair definition is the
only experimental variable.

### 3.3 Training

100 episodes per seed (80 train, 10 validation, 10 test), 32 steps, 96px, 60
epochs, batch 32, AdamW, learning rate 3e-4, five seeds. Good-view arms add a
same-state consistency loss with weight 2.0. Bad-view arms use crop pairs with
the same weight. Domain-randomized encoders use a 50/50 mixture of normal and
shifted nuisance for both arms.

### 3.4 Readout and controller

For control, a ridge readout is fit on augmented latents (normal + shifted
renderings of the same states) and a state-feedback controller is used:

```
a = clip(2 · (goal − predicted_position) − 1.0 · predicted_velocity, −1, 1)
```

for 32 steps, success threshold 0.08. A true-state version of this controller
succeeds 100%; it is the oracle that exposed CEM as the H2 bottleneck.

### 3.5 Protocol

Every confirmatory experiment is pre-registered, hashed, and run on five seeds.
We report per-seed values, paired effect sizes, bootstrap confidence intervals,
and exact McNemar tests where the outcome is binary. Test episodes are fresh and
never used for tuning. The failed endpoints are reported alongside the passes.

---

## 4. Results

### 4.1 Mechanism: good views disentangle state from nuisance

Five seeds, 100 episodes, 96px. Same-state nuisance ratio is the mean distance
between latents of the same state under different nuisance, divided by the mean
distance between different states.

| Arm | Ratio per seed | Mean | State R² per seed | Mean |
|---|---|---:|---|---:|
| Temporal | 2.244, 1.961, 2.804, 3.061, 2.627 | **2.539** | 0.870, 0.687, 0.793, 0.843, 0.859 | **0.810** |
| Good view | 0.287, 0.323, 0.303, 0.282, 0.199 | **0.279** | 0.932, 0.898, 0.945, 0.917, 0.939 | **0.926** |
| Bad view | — | **6.708** | — | 0.606–0.729 |

The good/temporal relative ratio is **0.114** (95% CI 0.089–0.143); 5/5 seeds
are below 0.7. Bad views are **2.6× worse than temporal**, so the effect is not
generic augmentation.

The pre-registered H1b clause required state R² ≥0.75 for both arms in every
seed. Seed 1's temporal baseline is **0.687**, so the literal H1 verdict is
**FAIL** even though the good arm is 0.898 there and higher than temporal in 5/5.
This is a protocol design error: an absolute floor on both arms conflates
baseline variance with treatment harm. H1a and H1c pass; the full H1 claim is not
made.

### 4.2 Control: the advantage survives into behavior

The first control attempt failed because CEM was broken: with true dynamics it
succeeded 28%, random 34%, while a true-state proportional controller succeeded
100%. A second attempt with state-feedback and a linear readout tied at 20%
because the readout had no support for the shifted nuisance. Domain randomization
fixed the support.

H2-v4 uses the frozen domain-randomized encoders, the same readout and
controller, and 20 fresh OOD episodes per seed.

| seed | OOD temporal success | OOD good success | delta |
|---:|---:|---:|---:|
| 0 | 0.60 | 0.95 | +0.35 |
| 1 | 0.50 | 0.95 | +0.45 |
| 2 | 0.75 | 0.95 | +0.20 |
| 3 | 0.90 | 0.90 | +0.00 |
| 4 | 0.65 | 0.80 | +0.15 |

- Mean success delta **+0.23** (95% CI +0.10 to +0.37).
- 4/5 seeds positive.
- Pooled McNemar: good-only 28, temporal-only 5, **p = 0.0001**.
- OOD distance **−0.036**; ID distance **−0.055**; ID success **+0.36**.

The good-view representation-to-control bridge holds under domain randomization.

### 4.3 Retention: less forgetting after a nuisance shift

Frozen H1 encoders adapted to the shifted nuisance with the same 10-epoch
temporal fine-tuning for both arms. A-state R² after adaptation:

| seed | Temporal | Good | Diff | Temporal forget | Good forget |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.783 | 0.897 | +0.114 | +0.074 | +0.028 |
| 1 | 0.728 | 0.909 | +0.180 | +0.085 | +0.023 |
| 2 | 0.654 | 0.921 | +0.267 | +0.071 | +0.037 |
| 3 | 0.661 | 0.898 | +0.237 | +0.173 | +0.022 |
| 4 | 0.578 | 0.925 | +0.347 | +0.255 | +0.015 |

- Mean A post difference **+0.229** (95% CI 0.161–0.298), 5/5.
- Mean forgetting: temporal **0.132**, good **0.025** (~81% less).
- Mean B post difference **+0.005**; seed 4 pays a B-side cost (−0.162) while
  gaining the most on A (+0.347).

This is representation-level retention, not control retention. Forgetting is
reduced, not eliminated. It is also an **in-distribution** result: §4.5 shows
that the good-view latent still fails absolute state decoding under an unseen
shift. H3's advantage is about adapting to the shift, not zero-shot
generalization.

### 4.4 Boundary: no dynamics protection

Frozen H1 encoders adapted to shifted dynamics `[0.65, 0.32]` with the same
10-epoch objective. Metric: one-step predictor MSE on A/B transitions.

| seed | A temporal post | A good post | diff | A forget temporal | A forget good |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.00461 | 0.00980 | +0.00519 | −0.00098 | +0.00445 |
| 1 | 0.00577 | 0.01049 | +0.00471 | −0.00058 | +0.00481 |
| 2 | 0.00555 | 0.01226 | +0.00672 | +0.00024 | +0.00539 |
| 3 | 0.00490 | 0.01195 | +0.00705 | −0.00087 | +0.00562 |
| 4 | 0.00525 | 0.01289 | +0.00764 | −0.00041 | +0.00635 |

- Mean A post difference (good − temporal) **+0.00626**; good is worse in 5/5.
- Mean A forgetting: temporal **−0.00052**, good **+0.00532**.
- B post difference **−0.00042** (essentially tied); both arms adapt.

Good views buy nuisance invariance, not dynamics immunity. The mechanism is
specific. §4.7 tests whether a Le-Gates adapter can handle the dynamics
boundary; it can adapt B, but retaining A requires a reliable router.

### 4.5 Adversarial nuisance: the invariance is distribution-specific

H1 trained on normal nuisance. We re-rendered held-out states under a held-out
nuisance, a shifted nuisance distribution, and four corner combinations of the
shifted ranges. Probe trained on normal latents; no adaptation.

| arm | normal | held-out swap | shifted | extreme |
|---|---:|---:|---:|---:|
| temporal | 0.815 | 0.806 | **−0.840** | **−0.888** |
| good | 0.932 | 0.933 | **−0.049** | **−0.053** |

Paired differences (good − temporal), five seeds:

- normal **+0.117** (95% CI +0.078 to +0.170)
- shifted **+0.791** (95% CI +0.224 to +1.165)
- extreme **+0.836** (95% CI +0.509 to +1.029)

Two things are true at once. **Absolute OOD state decoding fails**: the
good-view mean shifted R² is −0.049, so the invariance does not transfer as a
usable latent. **Relative degradation favors good**: it is +0.79 R² better than
temporal under the shift and +0.84 under the extreme corners.

This also corrects a one-seed pilot. Seed 0 made good look *worse* than temporal
out of distribution (good −0.877 vs temporal −0.592). The five-seed diagnostic
reverses that. The pilot is reported in
[`COLLISION-1B.md`](COLLISION-1B.md); it is exactly why the collision suite
uses five seeds.

The pre-registered 1b-d bet was partially falsified: the absolute drop is
≥0.10 (0.981), but good is **+0.791** above temporal, not within 0.05. The
invariance is distribution-specific in absolute terms, and relatively robust.

### 4.6 Counterfactual State Alignment: support first, then alignment

The first pipeline, CNA-v1, added counterfactual views but no shifted temporal
support and used a strong latent-MSE alignment. It failed on seed 0: normal R²
fell from 0.929 to **0.739**, and shifted R² was 0.282 vs **0.781** for an
ordinary mixed-domain control. Latent alignment without support collapses
state-relevant dimensions.

CSA changes both ingredients: the same 40 normal + 40 shifted episodes as the
`mix-ft` control, plus cross-domain counterfactual pairs whose **state-head**
predictions must agree (weight 0.5), with only a weak latent term (weight 0.1).

| stage | normal | held-out swap | shifted | extreme |
|---|---:|---:|---:|---:|
| frozen good | 0.932 | 0.933 | −0.049 | −0.053 |
| mix-ft | 0.924 | 0.926 | 0.728 | 0.328 |
| **CSA** | **0.927** | **0.927** | **0.828** | **0.471** |

- **CSA − mix-ft shifted: +0.101** (95% CI +0.050 to +0.180), 5/5 seeds.
- **frozen − CSA normal: +0.005** (95% CI −0.000 to +0.011) — normal retained.
- **CSA − mix-ft extreme: +0.143** (95% CI +0.077 to +0.209).

Per-seed shifted R²: 0.861, 0.865, 0.871, 0.850, 0.694 vs mix-ft 0.781, 0.830,
0.813, 0.777, 0.438. CSA wins on every seed; the largest gain is Seed 4, the
seed with the worst domain-exposure control.

**Support is necessary; alignment is not sufficient without it.** Once support
is present, state-level alignment adds a real OOD gain without a normal cost.

### 4.7 Le-Gates on the dynamics boundary: the router is the bottleneck

H3b showed good views do not protect dynamics. We froze the good-view encoder
and base predictor and trained only a 33k–35k residual adapter on B. Three
routers, five seeds each:

| version | A rel forget | B rel gain | pass seeds | error-gate AUC |
|---|---:|---:|---:|---:|
| full fine-tune | +0.865 | +0.723 | 0/5 | — |
| soft, no anchor | +1.459 | +0.544 | 0/5 | 0.612 |
| soft + 10% A anchor | +1.041 | +0.506 | 0/5 | 0.667 |
| hard + 10% A anchor | **+0.259** | +0.415 | 0/5 | — |

- **No anchor, no closure.** Trained only on B, every gate forgets A (+146% to
  +155%). The gate never sees an A example, so it has no signal to close.
- **The soft gate leaks even when it routes.** A 10% replay anchor plus
  distillation cuts error-gate forgetting to +104% and raises routing AUC to
  0.667, but the sigmoid still applies a nonzero residual on A.
- **The hard gate isolates exactly where it routes — and routes too much A.**
  With `τ` at the 95th percentile of A-anchor errors, A forgetting falls to
  **+26%**, but **18.5%** of A transitions open and only **65.5%** of B
  transitions route. The A/B error distributions overlap; a scalar threshold
  cannot separate them cleanly with 10% replay.

The ordering is the result: **the gate is the policy; isolation is the
guarantee; the router is the bottleneck.** A soft scalar gate is bounded
adaptation, not a no-forgetting mechanism. Hard isolation works only where the
router is correct, and with 10% A replay it is not correct enough. This is
consistent with the earlier Le-Gates evidence that masks, not gate scalars, do
the isolation.

---

## 5. What failed, and what it taught

| Experiment | Failure | Lesson |
|---|---|---|
| H1b | Absolute state floor failed on one temporal seed (0.687) | Use relative non-inferiority, not an absolute floor on both arms |
| H2 | CEM with true dynamics 28%, random 34%, P-controller 100% | The planner was the bottleneck, not the representation |
| H2-v2 | State-feedback + linear readout tied at 20%; OOD R² negative | The readout lacked support for the shift |
| H2-v3 | Distance −0.026 vs −0.05 target (fail), success +28 points | The distance endpoint was near-saturated; success was the right control metric |
| H3 seed 4 | Good B post 0.626 vs temporal 0.789 while A gained +0.347 | A real stability–plasticity trade-off, not a free lunch |
| H3b | Good A predictor MSE worse by 0.00626 in 5/5 | Nuisance invariance is not dynamics adaptation |
| Collision 1 | Good-view swap drop +0.001, not ≥0.10 | The swap used training nuisance identities; familiar-nuisance test, not OOD |
| Collision 1b pilot | Seed 0 made good look worse than temporal OOD | A one-seed collision reversed under five seeds; report the reversal |
| Collision 1b full | Good shifted R² −0.049 (absolute OOD failure) | Invariance is distribution-specific; relative degradation still favors good |
| CNA-v1 | Normal R² 0.929 → 0.739; lost to mix-ft by 0.498 shifted | Latent alignment without support collapses state-relevant dimensions |
| Collision 2 | Seed 4 rank bet confirmed, but no global collapse | Domain-specific capacity allocation, not a bottleneck; rank alone does not predict R² |
| Le-Gates v1 | All gates forget A (+146% to +155%) | A gate trained only on B has no signal to close on A |
| Le-Gates v2 | Soft error gate routes (AUC 0.667) but forgets +104% | Routing is not retention; the soft sigmoid leaks |
| Le-Gates v3 | Hard gate forgets +26%, routes 18.5% of A | The router, not the gate scalar, is the bottleneck |
| Anti-Equivalence | Volume arm Δasym +7.32 but forward R² 0.184 / reverse R² −5.08, latent std 0.599; asym Δasym −0.115 vs symmetric −0.013, lower 4/5; wrong-sign reach AUC 0.670 | The determinant penalty collapses; the matched control removes the direction effect; the wrong-sign control gets the same reachability gain |
| Phase 2 pilot | Occluded door flips the direction positive (+0.146 vs −0.055), but state R² is 0.308 vs the 0.8 guardrail; planner AUC 0.722 vs 0.750 | Partial observation is the right manipulation, but the pair loss/capacity is the bottleneck; not confirmatory |
| Phase 2 tuning | Pre-merge state head + `λ_pair=0.08`: state R² 0.670–0.695, direction +0.080 vs −0.019 (margin +0.099); still <0.8, planner flat | The decoupling fix works; the model is at its capacity ceiling; not confirmatory |
| Phase 2 capacity | 340k params, 60 epochs: state R² 0.610–0.662, direction margin −0.009, bounded std 0.882; frame probe R² negative, frame head 0.24–0.31 | More capacity did not fix state; the merged latent/observation is the bottleneck; the 5-seed matrix is not justified |
| Dual-stream | State stream **0.936** passes; JEPA no-state-loss direction margin **+0.018**; planner **0.500** vs state-only **0.611** | State decoupling works; the asymmetry/planner value does not; the next test is a JEPA reachability classifier, not another matrix |
| Hypothesis A | Combined classifier AUC **1.000**, abort **1.00**; state-only same; JEPA-only **0.285 / 0.874 / 0.290** | The pass is the state stream's side+goal shortcut, not JEPA irreversibility; the test is confounded, next is a belief-state JEPA |
| Hypothesis B | Static controls **0.497 / 0.391–0.497**; frozen JEPA+GRU **0.515**; end-to-end JEPA+GRU **0.999–1.000** | The belief state solves the controlled counterfactual, but only with end-to-end training; this is a memory result, not control |
| Sanity checks | Reachability fine-tune: state **0.157→0.442** but dynamics **0.0179→0.1097**; frozen JEPA door cue at x=0.45 **0.524/0.529** | The fine-tune breaks dynamics and the frozen JEPA lacks the door cue; use a separate belief encoder + GRU (Option A) |
| Step 2 natural belief | Static JEPA **0.431/0.432**; belief hard **0.948–0.997**; +3/+5/+10 **0.960/0.980/0.980**; counterfactual **0.766**; pure natural hard ~1.0 but counterfactual **0.407** | The natural+counterfactual mixture fixes the lazy shortcut; all three gates pass; Step 3 is justified |
| Step 3 closed-loop | Base reach success 0.75; state gate 0% false/100% abort; JEPA 0%/25%; natural belief 18% false; planner belief **7.1% false/100% abort/0.83 after close**; calibration 1% | Not clean: 7.1% >5%, 28 reachable test too small, state gate likely shortcut, matched counterfactual missing |

These failures are part of the contribution. They define the boundary of the
claim and make the surviving results reproducible rather than rhetorical.

---

## 6. Discussion

### 6.1 Positive pairs are a design variable

The practical recipe is cheap: if two observations of the same state with
independently varying nuisance are available, add a same-state consistency term.
It is a representation regularizer, not a new planner. In this world it reduced
nuisance sensitivity by ~89% relative, improved OOD control by 23 points, and
reduced nuisance-shift forgetting by ~81%.

The recipe has a precondition: **support**. Without shifted-domain examples, the
good-view latent is still not linearly state-decodable under a held-out
nuisance (mean shifted R² −0.05), even though it degrades less than temporal
(−0.84). Support alone (mix-ft) recovers 0.73; adding counterfactual state
alignment recovers 0.83 and beats the support-only control by +0.10. The
practical order is: support first, then align states, and only then expect the
representation to be useful out of distribution.

### 6.2 Good views and Le-Gates are complementary — and gating is not retention

Good views address **nuisance**: what should be invariant. Le-Gates addresses
**dynamics**: what should change and be learned. H3b is the empirical case for
the division of labor: good views do not protect against a dynamics shift, and
adaptation is still required.

The dynamics-boundary prototype sharpens the Le-Gates claim. A frozen-encoder
adapter trained only on B forgets A (+146% to +155%), because the gate never
sees an A example and has no signal to close. A 10% A replay anchor plus
distillation reduces that to +104%, and the soft error gate learns to route
(AUC 0.667), but the sigmoid still leaks. A hard gate calibrated on the A
anchor cuts forgetting to +26%, but routes 18.5% of A and only 65.5% of B: the
A/B error distributions overlap, and the router cannot separate them cleanly at
this replay budget.

This is the same lesson as the earlier Le-Gates work, now on the dynamics
boundary: the soft single gate reduced drift from +1015% to +169% (not zero);
masked growable slots reached 0.0025 → 0.0026; hard gates with a frozen base
gave zero **non-routed** forgetting by construction. Effective as bounded
adaptation/isolation, not as a no-forgetting guarantee. **The gate is the
policy; isolation is the guarantee; the router is the bottleneck.** A reliable
router (more replay, a learned detector with a false-open penalty, or structural
masks) is the next requirement, not a better gate scalar.

### 6.3 Anti-Equivalence: direction is not volume

We also tested the stronger claim that an irreversible transition should produce
a directed latent semi-lattice: A→B easy, B→A hard, `z_B` lower-rank. A one-way
door gives a real irreversible event. The five-seed test falsified the operationalized
version. An asymmetric forward/bottlenecked-reverse objective did not make A→B
easier; its five-seed mean `Δasym` was **−0.115** vs **−0.013** for a symmetric
pair loss. It was lower on **4/5 seeds** and higher on **1/5** (paired differences
−0.059, −0.387, −0.039, −0.166, **+0.141**), so there is no consistent directed
advantage. The corrected log-volume penalty produced a huge apparent asymmetry
(**+7.32**) only by shrinking the latent: forward R² **0.184**, reverse R²
**−5.08**, latent std **0.599** vs 0.911. The wrong-sign control had the highest
mean reachability AUC (**0.670**), so the reachability gain is not caused by the
directed geometry. A matched reversible control with the same time gaps removed
the apparent effect.

The lesson: **direction is not volume.** A determinant penalty is a collapse
knob unless it is paired with a sufficiency constraint. On a visible
irreversible event the temporal model already has the information; a fair test
needs an occlusion or a same-state counterfactual where the information is
actually lost. See [`COLLISION-4.md`](COLLISION-4.md).

**Phase 2** built that fairer test: the door is occluded on the right, the
symmetric control has parameter-matched heads, and the volume penalty has a
VICReg-style variance floor. The direction sign flips positive — `asym_clean`
`Δasym` **+0.146** vs `sym_head` **−0.055** — and the bounded volume arm keeps
latent std **1.044**. But state R² falls to **0.308 / 0.322**, far below the
pre-registered **0.80** guardrail, and the planner AUC is **0.722** vs
**0.750** for the symmetric head. Phase 2 is therefore **inconclusive on the
representation claim**: the partial-observation manipulation works, but the
142k model cannot hold both the state and the irreversibility at this pair
weight. A follow-up tuning loop — a pre-merge frame state head and
`λ_pair=0.08` — raises pair-arm state R² to **0.670 / 0.671 / 0.695** and keeps
the direction positive (`asym_clean` **+0.080** vs `sym_head` **−0.019**, a
+0.099 margin), but the state guardrail still fails and the planner does not
improve. An expanded **340,784-parameter** model at 60 epochs with cosine decay
was worse, not better: state R² **0.610 / 0.509 / 0.662 / 0.623**, direction
margin **−0.009**, bounded std **0.882**. Capacity is not the state bottleneck;
the five-seed run is not started. An oracle check closes the question: a
**46,830-parameter** CNN on the same three-frame observations reaches **0.935**
4-D state R² (**0.978** x,y), so the oracle-relative guardrail is **0.842** and
the JEPA's **0.671** is a genuine representation gap. A stop-gradient MLP probe
does not recover more state (**0.564** vs 0.671 linear); the readout is not the
bottleneck. The next move is a separate state encoder. See
[`PHASE2-PILOT.md`](PHASE2-PILOT.md),
[`PHASE2-TUNING.md`](PHASE2-TUNING.md),
[`PHASE2-CAPACITY.md`](PHASE2-CAPACITY.md), and
[`PHASE2-ORACLE.md`](PHASE2-ORACLE.md).

**Dual stream** separates the state path from the world-model path. The frozen
47k state stream passes the oracle-relative guardrail at **0.936**, while the
JEPA streams trained with **no state loss** do not form a stronger directional
latent: `asym_clean` Δasym **+0.014** vs `sym_head` **−0.003**, a margin of
**+0.018**. The combined planner is worse than state-only (**AUC 0.500 vs
0.611**). The state bottleneck is solved; the irreversibility signal is not yet
usable by this planner. See
[`DUAL-STREAM-PILOT.md`](DUAL-STREAM-PILOT.md). A discriminative reachability
classifier on the combined latent passes numerically (AUC **1.000**, abort
**1.00**, false abort **0.03–0.05**), but the **state-only** ablation gets the
same AUC and abort, and the **JEPA-only** probe is unstable (AUC **0.285 /
0.874 / 0.290**). The pass is the state stream's side+goal shortcut, not a JEPA
irreversibility signal. Hypothesis A is **confounded**; the next test is a
belief-state JEPA or a task that breaks the side+goal shortcut. See
[`HYPOTHESIS-A.md`](HYPOTHESIS-A.md).

**Hypothesis B** builds that task. A hidden right-side alcove gate makes a
right/right goal reachable only if the door stayed open; paired histories share
the **same last three frames**, so static encoders are at chance. The state and
static JEPA baselines score **0.497** and **0.391–0.497**; a GRU over frozen
JEPA features is also at chance (**0.515**). Fine-tuning the JEPA frame encoder
with a GRU end-to-end for 40 epochs reaches **AUC 0.999 / 1.000 / 1.000**,
clearing the 0.750 trigger. The belief state solves the controlled memory test;
this is not yet a natural-episode or closed-loop control result. See
[`HYPOTHESIS-B.md`](HYPOTHESIS-B.md).

Two sanity checks decide the architecture. The reachability fine-tune **improved**
state R² (0.157 → 0.442) but made one-step JEPA dynamics **6.1× worse**
(0.0179 → 0.1097): it is not safe as the planner's JEPA encoder. And at x=0.45
with the door fully visible, a linear probe on the frozen JEPA is **AUC 0.524 /
0.529** — the pretrained features never encoded the door cue. The frozen GRU's
0.515 is explained. The architecture choice is **Option A**: keep the planner
JEPA frozen and train a **separate belief encoder + GRU**. See
[`SANITY-CHECKS.md`](SANITY-CHECKS.md).

**Step 2** tests that separate stream on natural hidden-alcove episodes with
per-step reachability labels. The static JEPA is **0.431 overall / 0.432 hard**;
the state baseline is 0.860 / 0.832. The belief stream, trained with the natural
episodes plus held-out counterfactual pairs, reaches **0.948–0.997 hard** and
holds **AUC 0.960 / 0.980 / 0.980 at t_cross+3/+5/+10**. The held-out
counterfactual re-check is **0.766**, clearing the 0.750 gate. The pure natural
model was a shortcut (hard ~1.0, counterfactual **0.407**); the mixture fixes
it. Step 3 planner integration is justified. See
[`NATURAL-BELIEF.md`](NATURAL-BELIEF.md).

**Step 3 first pass** calibrates the abort gates in closed loop. The base
reachable success is **0.75** (the controller fails some open-gate door gaps);
the gate arms do not reduce it. On the held-out test (28 reachable / 12
unreachable), the state gate gets **0% false / 100% abort** but is likely a
spatial shortcut; the frozen-JEPA gate gets 0% false / **25%** abort. The
natural-trained belief gate has **18%** false aborts. Retraining the belief
stream on planner-driven closed-loop trajectories cuts false aborts to
**7.1% (2/28)** and aborts **100%** of unreachable episodes **0.83 steps after
the gate closes** (calibration false abort 1.0%). The 7.1% test is one episode
over the 5% ceiling, the reachable test is small, and the matched
counterfactual closed-loop is missing. Step 3 is **not cleanly passed**; no
five-seed matrix. See [`CLOSED-LOOP-GATE.md`](CLOSED-LOOP-GATE.md).

### 6.4 Why the effect appears in control only after domain randomization

The first control tests failed because the encoder had never seen the shifted
nuisance; no readout could fix that. Once both arms had support, the good-view
advantage appeared. The adversarial diagnostic makes the point quantitative:
without support, good-view shifted R² is −0.049 and temporal is −0.840. Good is
relatively robust, but neither latent is usable. Support is a prerequisite, not
a detail. CSA shows that support plus state-level alignment is enough to make
the representation usable (0.828 shifted), and that alignment without support is
not (CNA-v1: normal R² 0.739, shifted 0.282).

### 6.5 Limitations

- Synthetic 96px world; one model size (142k); five seeds.
- State readout and two-step loss are auxiliary and shared; this is not a pure
  JEPA ablation.
- H3 is representation-level, not control retention.
- H3b and the Le-Gates prototype use predictor MSE, not control.
- The adversarial diagnostic uses one shifted nuisance distribution and four
  corner combinations; it is not a general OOD nuisance law. The absolute
  good-view shifted R² is still near zero; “relatively robust” is not “usable.”
- CSA is representation-level, not control retention; it uses a 40/40 mixed
  budget and a state-head alignment term that was not ablated separately.
- The Le-Gates prototype uses one dynamics shift and a 10% A replay anchor; the
  hard gate's threshold is calibrated, not learned. More replay or a learned
  detector may change the router result.
- The capacity autopsy is a single seed pair (1 vs 4); it rules out a simple
  global rank-collapse story but does not establish the allocation mechanism
  causally.
- The familiar-nuisance swap used training nuisance identities; it is not the
  OOD test.
- The Anti-Equivalence test is five seeds / 20 epochs. The door is visible, so
  the irreversibility is observable; the volume penalty is a diagonal proxy,
  not the full determinant; reachability is a linear probe, not a closed-loop
  planner. The state probe is unstable on one seed (symmetric R² 0.019), which
  is reported rather than dropped.
- No real-world data, language, or cross-modal transfer.
- H1b and H2-v3 are literal failures and remain so.

---

## 7. Conclusion

Same-state good views produce a more world-like latent in an action-conditioned
world model. That latent is more robust to nuisance in control and forgets less
after a nuisance shift, while providing no protection against a dynamics shift.
The effect is conditional: without support, absolute OOD state decoding still
fails, even though good views degrade far less than temporal. Support plus
counterfactual state alignment restores usable OOD geometry and beats a
domain-exposure control. On the dynamics boundary, a frozen-encoder adapter
needs a retention signal and a reliable router; a soft gate leaks, and a hard
gate with 10% replay routes too much of the original domain. The occluded-door
Anti-Equivalence test flips the direction sign but never clears the state
guardrail; scaling the model from 142k to 340k parameters made state readback
worse. The state path, not the pair weight, is the next frontier. A recurrent
belief state solves the controlled same-side counterfactual (**AUC 0.999–1.000**)
but only with end-to-end training; frozen JEPA features are at chance (0.515).
On natural hidden-alcove episodes, a separate belief stream reaches **0.948–0.997
hard AUC**, holds **0.960–0.980 through t_cross+10**, and passes the held-out
counterfactual re-check (**0.766**). The pure natural model was a shortcut
(counterfactual **0.407**); the natural+counterfactual mixture is necessary.
In closed loop, retraining the belief stream on planner-driven trajectories cuts
false aborts from **18% to 7.1% (2/28)** and aborts **100%** of unreachable
episodes **0.83 steps after the gate closes** (calibration 1.0%). The state gate
gets 0% false / 100% abort but is likely a spatial shortcut. The 7.1% test is
one episode over the 5% ceiling, and the matched counterfactual closed-loop is
still missing. The memory mechanism works; the closed-loop gate is close but not
clean.

Positive-pair geometry is a conditional lever for continual world models:
**support first, then alignment; gating is not retention.** It is not a universal
invariance fix, and the collisions that break it are part of the result.

---

## Appendix A — Reproducibility

| Result | Script | Pre-registration |
|---|---|---|
| H1 mechanism | `scripts/h1_confirm.py` | `docs/PREREG-v2.md` |
| H2-v4 control | `scripts/h2v4_success.py` | `docs/PREREG-H2v4.md` |
| H3 retention | `scripts/h3_forgetting.py` | `docs/PREREG-H3.md` |
| H3b boundary | `scripts/h3b_dynamics.py` | `docs/PREREG-H3b.md` |
| Collision 1 familiar swap | `scripts/nuisance_collision.py` | `docs/PREREG-COLLISION.md` |
| Collision 2 capacity | `scripts/capacity_autopsy.py` | `docs/PREREG-COLLISION.md` |
| OOD diagnostic | `scripts/ood_diagnostic.py` | `docs/PREREG-COLLISION-2.md` |
| CNA-v1 | `scripts/adversarial_nuisance.py` | `docs/PREREG-COLLISION-2.md` |
| CSA | `scripts/csa_nuisance.py` | `docs/PREREG-CSA.md` |
| Le-Gates v1/v2 | `scripts/legates_dynamics.py` | `docs/PREREG-COLLISION-2.md`, `docs/PREREG-LEGATES-V2.md` |
| Le-Gates v3 hard | `scripts/legates_hard.py` | `docs/PREREG-LEGATES-V3.md` |
| Anti-Equivalence | `scripts/asymmetry_collision.py` | `docs/PREREG-ASYMMETRY.md` |
| Phase 2 pilot | `scripts/asymmetry_collision.py`, `scripts/phase2_planner.py` | `docs/PREREG-PHASE2.md`, `docs/PHASE2-CONCLUSION.md` |
| Phase 2 tuning | `scripts/asymmetry_collision.py`, `scripts/state_probe_nonlinear.py` | `docs/PHASE2-TUNING.md` |
| Phase 2 capacity | `scripts/asymmetry_collision.py` | `docs/PHASE2-CAPACITY.md` |
| Phase 2 oracle | `scripts/oracle_state_cnn.py`, `scripts/state_probe_mlp.py` | `docs/PHASE2-ORACLE.md` |
| Dual-stream pilot | `scripts/dual_stream_planner.py`, `scripts/asymmetry_collision.py` | `docs/DUAL-STREAM-PILOT.md` |
| Hypothesis A | `scripts/reachability_classifier.py` | `docs/HYPOTHESIS-A.md` |
| Hypothesis B | `scripts/counterfactual_belief.py`, `scripts/counterfactual_gru.py` | `docs/HYPOTHESIS-B.md` |
| Sanity checks | `scripts/spatial_forgetting_probe.py`, `scripts/door_cue_probe.py` | `docs/SANITY-CHECKS.md` |
| Step 2 belief | `scripts/natural_belief.py` | `docs/NATURAL-BELIEF.md` |
| Step 3 gate | `scripts/closed_loop_gate.py`, `scripts/planner_belief_train.py` | `docs/CLOSED-LOOP-GATE.md` |
| Snapshots | `scripts/make_snapshots.py` | — |

Environment: Apple M5 Pro, 64 GB, MLX 0.32.4, Python 3.12. Model: 142,564
parameters. Training is minutes per model at 96px; the full set of reported runs
fits in a few GPU-hours on one machine.

Pre-registration hashes:

```
H1        d08d0d69d24cadd8a2b46a0381056b222a34a7a69fa8bec68dc2ea466c95e411
H2v4      ba4e3c37adcf4a5190394683684e942f382336838628e1f934c4623dac6daf57
H3        c4660371980ef02518b99f9d41c817cc907aa7d67e7a450146f2602e2624f1ec
H3b       eb2a6086b28bca9a44bd430d5d3338455cffc6038afc426b1394301ff30cea1e
Collision 48168a0df7e29371c3b568b4573f79332777471fc47cd186c476570a8b603c85
CSA       ebe06189f15c3e9c42788e709b93d9f02c9ff491ce5682e8e99a7f259b9885c7
LG-v2     d4dc847f6b2df2dd431292e9c91e98ffd9920ee68312fbf1defe458be3caea2c
LG-v3     6d76fc43a881ba1210d16cacc92b55d1e98c2cc15798abfe2893fb8956fda6a5
Asym      e1fc79ff68fb92d9a2a5552158ea1c1a17e89de90178d97e15a43302b275eea1
Phase2    eab4147cc77c2aa99b0ef5ad293e98503d21a9a6b61ceea67dcff08826412e46
```

## Appendix B — Hyperparameters

| Parameter | Value |
|---|---|
| Image size | 96 |
| History | 3 frames |
| Latent | 64 |
| Batch | 32 |
| Optimizer | AdamW |
| Learning rate | 3e-4 (pretrain), 1e-4 (fine-tune) |
| Epochs | 60 (pretrain), 10 (adapt) |
| SIGReg weight | 0.09 |
| Good-view weight | 2.0 |
| Two-step loss weight | 0.5 |
| State-head weight | 0.5 |
| Controller | `k=2`, `damp=1.0`, 32 steps, success 0.08 |
| Readout | ridge, `alpha=1.0` |
| Le-Gates adapter | 2×128 hidden, zero-init residual, 33k–35k params |
| Le-Gates adaptation | 10 epochs, lr 1e-3, batch 32, weight decay 1e-4 |
| Le-Gates anchor | 8 A episodes, distillation weight 1.0 |
| Hard gate threshold | 95th percentile of A-anchor previous errors |
| One-way door | `x=0.5`, gap `y∈[0.35,0.65]`, closes on first crossing |
| Asym pair offsets | pre 2/4, post 2/4; reversible gaps 4/6/8 |
| Asym training | 20 epochs, 5 warmup, `λ_pair=0.25`, `λ_state=0.25`, `λ_vol=0.05` |
| Phase 2 occlusion | door hidden on the right; post offsets 3/5 |
| Phase 2 training | 40 epochs, 5 warmup, `λ_pair=0.25`, `λ_state=0.25` |
| Phase 2 variance floor | `gamma=1.0`, weight 1.0 |
| Phase 2 planner | 64 random shots + controller, horizon 6, threshold p90 reachable |
| Phase 2 tuning | pre-merge 128→4 state head; `λ_pair=0.08`, `λ_var=0.2`, 40 epochs |
| Phase 2 capacity | 340,784 params, latent 128, width 1.25; 60 epochs cosine; `λ_pair=0.10`, `λ_var=0.10` |
| Oracle CNN | 46,830 params; conv 24/40/40; 3-frame history; 100 epochs, 3 seeds |
| SG MLP probe | 64→256→256→4; frozen encoder; 3 probe seeds, early stopping |
| Dual-stream state | 46,830-param oracle CNN, frozen for the planner |
| Dual-stream JEPA | 142k, no state loss, 40 epochs, `λ_pair=0.10`, cosine |
| Dual-stream planner | 64 shots, horizon 6, state + α·JEPA score, α=1.0 |
| Reachability classifier | 2-layer MLP, 50 epochs, 3 seeds, τ = p17 validation reachable |
| Belief counterfactual | paired open/closed, identical last 3 frames, 800/400 sequences |
| Belief GRU | JEPA frame encoder + GRU 64, 40 epochs, batch 8, 3 seeds |
| Spatial probe | ridge on 3-frame latent; one-step dynamics MSE |
| Door-cue probe | linear ridge at x=0.45/0.50, single vs 3-frame |
| Natural belief | hidden alcove, 80/20, per-step labels; separate frame enc + GRU 64; 400 cf pairs; 40 epochs; unreachable weight 3× |
| Closed-loop gate | 160 cal / 40 test; τ 0.10–0.25; k 2–5; false ceiling 5%; time after gate close |
| Planner belief | 160 closed-loop rollouts + 400 cf pairs; 40 epochs; one seed |

## References

- Huh, Cheung, Wang, Isola. *The Platonic Representation Hypothesis.* 2024.
- Balestriero, LeCun. *Proving and Improving the Platonic Representation
  Hypothesis with JEPAs.* arXiv:2610.39302, 2026.
- Balestriero, LeCun. *LeJEPA: Provable and Scalable Self-Supervised Learning
  Without the Heuristics.* arXiv:2511.08544, 2025.
- Maes, Le Lidec, Scieur, LeCun, Balestriero. *LeWorldModel: Stable End-to-End
  Joint-Embedding Predictive Architecture from Pixels.* arXiv:2603.19312, 2026.
- Assran et al. *I-JEPA: Self-Supervised Learning from Images with a Joint-
  Embedding Predictive Architecture.* 2023.
- Bardes et al. *V-JEPA: Latent Video Prediction for Visual Representation
  Learning.* 2023.
- Hafner, Pasukonis, Ba, Lillicrap. *Mastering Diverse Domains through World
  Models.* 2023.
- LeCun. *A Path Towards Autonomous Machine Intelligence.* 2022.
