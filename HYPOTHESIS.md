# LeParallax — hypotheses

This document states the claims we intend to test, the predictions that follow
from them, and the observations that would prove them wrong. It is written before
the code, so the protocol can be frozen before the first result.

---

## 1. The problem

A small action-conditioned world model learns from transitions:

```
(o_t, a_t) -> o_{t+1}
```

The encoder is shaped by the prediction loss. Nothing in that objective says
"encode the world state and ignore the camera." If nuisance is predictive of the
next frame — lighting that changes with time, a texture correlated with an
object, a camera that moves with the agent — the encoder can keep it.

The Platonic JEPA theorem says same-state views fix this: if two observations
differ only in nuisance, the encoder must keep what they share. But the theorem
assumes an identity predictor, not an action-conditioned one. Action-conditioned
world models are outside its scope.

**The problem LeParallax addresses:** does good-view supervision make an
action-conditioned world model more state-like and more reliable, or does the
invariance it imposes remove information the planner needs?

This is a question about the interaction between representation learning and
control, not about scale.

---

## 2. Definitions

| Term | Meaning in this project |
|---|---|
| **State** `s` | The physical variables that determine the task: position, velocity, pose, contact, goal. |
| **Nuisance** `n` | Variables that change the observation but not the task: lighting, texture, background, camera offset, render style. |
| **Observation** `o` | A rendering of `(s, n)`. |
| **Good view** | Two observations of the **same state** with **independent nuisance**: `(render(s, n1), render(s, n2))`. |
| **Bad view** | Two corruptions of **one observation**: crops, color jitter, masks. They share the nuisance. |
| **Temporal pair** | `(o_t, a_t, o_{t+1})`: two **different states** linked by an action. |
| **State probe** | A linear readout from the latent to the physical state. |
| **Nuisance probe** | A linear readout from the latent to the nuisance variables. Lower is better. |
| **Forgetting** | The drop in old-domain planning success after adaptation to a new domain. |

The distinction between a good view and a temporal pair is the crux. A video
frame at `t` and a frame at `t+1` are not two views of the same state; they are
two states. Treating them as positives teaches scene identity, not state
invariance, and can blur the dynamics.

---

## 3. The claim ladder

We make four claims, in order. Each is a separate experiment and a separate
falsifier. We do not skip a rung.

```
H1  mechanism   the latent becomes more state-like, less nuisance-like
H2  behavior    planning becomes more robust and more data-efficient
H3  retention   adaptation forgets less under nuisance shift
H4  transfer    separately trained modalities align up to a rotation
```

H1 is about representation. H2 is about control. H3 is about continual learning.
H4 is about modularity. A failure at H1 stops the ladder; a failure at H2 does
not invalidate H1, but it changes the story from "better representation, better
control" to "better representation, wrong information."

---

## 4. H1 — the mechanism claim

**Claim.** Same-state good-view supervision reduces the latent's sensitivity to
nuisance while preserving its state readout, relative to temporal-only and
bad-view supervision.

**Baseline choice.** All arms share a linear state readout and a two-step rollout
loss. Phase 0 showed that a temporal-only JEPA learns a predictable but
state-free summary; the state readout makes the baseline a usable planner. H1
therefore tests nuisance invariance, not whether state is encoded at all. The
pure-JEPA ablation without the state readout is a secondary check.

**Intervention.** Add a good-view consistency term to the encoder:

```
L_good = || f(o_i) - f(o_j) ||^2,   o_i, o_j = render(s, n_i), render(s, n_j)
```

with independent nuisance `n_i, n_j` and the same state `s`. The term is added to
the standard prediction loss and the shared auxiliary objectives.

**Prediction.**

- State probe R² stays at least **0.75** for every arm (manipulation check).
- The same-state nuisance distance ratio is below **0.5** for good views, versus
  above **1.0** for the temporal baseline. (Phase 0 baseline: **1.25**.)
- Bad views do not improve the ratio; temporal + good views does.
- The effect is ordered: good view < temporal + good < temporal ≈ bad view.

**Falsifier.** The ratio does not fall below 0.7, or the state probe drops below
0.6, or bad views improve the ratio as much as good views. In that case the
theorem does not transfer to this regime, and we report it.

**Pilot update (exploratory).** A three-seed pilot found good views reduce the
ratio in every seed: 64px mean **1.86 → 0.93**, 96px mean **2.22 → 0.81**, with
state probe **0.62 → 0.86** at 96px. The absolute `<0.5` threshold was not met
on average; one 96px run reached 0.392 but did not replicate. The confirmatory
endpoint is the paired relative reduction in
[`docs/PREREG-v2.md`](docs/PREREG-v2.md), frozen before the confirmatory run. The
original threshold remains here for the record.

**Confirmatory result (2026-11-12).** H1a and H1c pass; H1b fails literally;
overall **FAIL**. Five seeds: mean relative ratio **0.114** (CI 0.089–0.143),
5/5 seeds ≤0.7; good state R² 0.898–0.945 vs temporal 0.687–0.870; bad mean
ratio 6.708. The failure is the absolute 0.75 state floor on seed 1's temporal
baseline; the good arm is 0.898 there and higher than temporal in every seed. See
[`docs/H1-CONFIRM.md`](docs/H1-CONFIRM.md).

**Why it matters.** This is the manipulation check. If the good-view term does
not move the latent toward nuisance invariance, any control gain would be a
coincidence.

## 5. H2 — the behavior claim

**Claim.** Given the shared state-aware baseline, good-view supervision improves
out-of-distribution planning under a nuisance shift and improves data efficiency,
without a meaningful in-distribution regression.

**Prediction.**

- At **50% of the training data**, good-view pretraining improves OOD planning
  success by at least **10 percentage points** over the temporal baseline.
- In-distribution planning success drops by no more than **2 percentage points**.
- At 100% data, the OOD gain is at least **5 points**.
- The gain survives three seeds; the confidence interval excludes zero.

**Falsifier.** OOD gain below 5 points, ID regression above 5 points, or the gain
disappears when the baseline is given the same total compute. Any of these is a
negative result worth reporting.

**Confirmatory result (2026-11-12).** FAIL as pre-registered: mean OOD
difference **−0.041** against a **−0.05** target, 3/5 seeds lower, CI
[−0.113, +0.036]. OOD success is **+0.10** (0.18 vs 0.08), but the random policy
(**0.34**) beats both planners. Mean ID difference is −0.011, but seed 3
regresses by +0.110. The planner is the bottleneck; the H1 mechanism stands. See
[`docs/H2-CONFIRM.md`](docs/H2-CONFIRM.md).

A follow-up diagnostic shows H2 is confounded: a true-state proportional
controller solves the same OOD episodes **100%**, while CEM succeeds **28%** and
random **34%**; the learned state head is negative-R² OOD. The planner and
readout, not the representation, are the bottleneck. See
[`docs/PLANNER-DIAGNOSTIC.md`](docs/PLANNER-DIAGNOSTIC.md).

H2-v2 fixed the planner (state-feedback) and the readout (ridge on augmented
latents) and still failed: mean OOD difference **0.000**, 3/5 seeds lower, control
tie at 0.20. The good-view OOD readout R² is **−0.163** vs temporal **−1.293** —
better, but both negative. The shift is outside the encoder's training support;
this is a readout-limited negative, not a clean dissociation. See
[`docs/H2V2-CONFIRM.md`](docs/H2V2-CONFIRM.md).

The readout diagnostic (linear vs MLP) found both readouts negative OOD: good
R² −0.074 vs temporal −1.290, MLP no better. The encoder's OOD support is the
limit. H2-v3 domain-randomizes encoder training and tests a held-out shift inside
the support. See [`docs/READOUT-DIAGNOSTIC.md`](docs/READOUT-DIAGNOSTIC.md).

H2-v3 domain-randomized both arms and failed the distance endpoint (−0.026 vs
−0.05), but 4/5 seeds were lower with a CI excluding zero, and OOD success rose
from **60% to 88%** (+28 points, all five seeds positive). The distance endpoint
was near-saturated. H2-v4 is a success-primary replication on fresh episodes. See
[`docs/H2V3-CONFIRM.md`](docs/H2V3-CONFIRM.md).

H2-v4 is the success-primary replication on fresh episodes: good-view OOD
success **88% vs 65%** (**+23 points**, 95% CI +10 to +37), 4/5 seeds positive,
pooled McNemar **p = 0.0001**; OOD distance −0.036; ID improves. **PASS.** See
[`docs/H2V4-CONFIRM.md`](docs/H2V4-CONFIRM.md).

**Why it matters.** This is the translation from representation to control. The
paper's 42× sample-efficiency claim is about alignment, not planning; H2 tests
whether it survives the translation.

---

## 6. H3 — the retention claim

**Claim.** Good-view pretraining reduces forgetting under a **nuisance shift**,
but not necessarily under a **dynamics shift**.

**Setup.** Train on domain A, adapt to domain B, measure old-domain retention.

- Nuisance shift: new lighting, texture, or camera; same dynamics.
- Dynamics shift: new friction, mass, or action effect; same nuisance.

**Prediction.**

- Under nuisance shift, old-domain success drops by **≤5 points** for good-view
  pretraining versus **≥15 points** for the temporal baseline.
- Under dynamics shift, the advantage shrinks or disappears; both models need
  adaptation.
- The dissociation is the result: good views buy invariance to nuisance, not
  immunity to dynamics change.

**Falsifier.** No forgetting reduction under nuisance shift, or good views hurt
adaptation to the new domain. Either outcome is informative.

**Operationalization used.** The confirmatory experiment measured A/B **state
probe R²**, not control success. The control pipeline became valid only in
H2-v4; adding control retention would introduce another readout confound. This is
the representation-level version of H3. Control retention is future work.

**Confirmatory result (2026-11-12).** **PASS.** A-state R² after adaptation:
good **+0.229** higher (95% CI **+0.161 to +0.298**), **5/5** seeds. Forgetting:
temporal **0.132**, good **0.025** (~81% less). B post difference **+0.005** on
average, but seed 4 pays a B-side cost (−0.162) while gaining the most on A
(+0.347). See [`docs/H3-CONFIRM.md`](docs/H3-CONFIRM.md).

**Why it matters.** This is the continual-learning link. If nuisance invariance
makes nuisance shifts free, good views are a representation-level regularizer
that composes with an adapter like Le-Gates for the dynamics. If not, the
continual-learning problem is not a representation problem.

---

## 6b. H3b — the dynamics boundary

**Claim.** The good-view retention advantage is specific to nuisance shift. Under
a dynamics shift, good views should not protect the original dynamics; both arms
need adaptation.

**Confirmatory result (2026-11-12).** **Boundary holds.** After adapting to
`[damping=0.65, gain=0.32]`, the good-view A predictor MSE is **0.00626 worse**
than temporal in 5/5 seeds; mean A forgetting is **+0.00532** for good vs
**−0.00052** for temporal. Both arms adapt to B and end essentially tied
(B post difference −0.00042). Good views do not transfer to dynamics. See
[`docs/H3B-CONFIRM.md`](docs/H3B-CONFIRM.md).

## 7. H4 — the transfer claim

**Claim.** Vision and proprioception encoders trained **separately** with good
views align up to a rotation, and planning with the rotated latent is competitive
with joint training.

**Setup.** Train a vision JEPA and a small proprio JEPA on the same trajectories.
Neither sees the other modality. Fit one orthogonal map on `k` paired timesteps
(Procrustes). Use the aligned proprio latent as an auxiliary planner input.

**Prediction.**

- Rotation retrieval is above chance and improves with `k`.
- Planning with the aligned latent is within **5 points** of joint training.
- The rotation is stable across seeds.

**Falsifier.** Alignment at chance, planning degrades by more than 10 points, or
the rotation is unstable. The theorem's assumptions would not survive
action-conditioned dynamics.

**Why it matters.** If it works, modalities can be added without retraining the
whole world model: train the encoder separately, align with a rotation, plug it
in. That is a modular path to reliable small models.

---

## 8. The alternative hypothesis

**Good views over-invariantize the latent.**

Nuisance is not always noise. Color can identify a block; texture can signal
friction; a camera offset can be task-relevant. If good views remove information
the planner needs, then:

- State probes improve,
- Nuisance probes drop,
- But in-distribution planning gets worse, or dynamics prediction degrades.

This is a real possibility and is pre-registered as an alternative. The result
would be a **trade-off curve**, not a free lunch: invariance buys robustness at
the cost of task-relevant detail. That is still a useful result, and it would
change the recommendation from "add good views" to "add good views only for the
nuisance dimensions you can afford to lose."

---

## 9. Scope and assumptions

- **One model family.** A small action-conditioned JEPA, 1–15M parameters, 96px,
  3-frame history, SIGReg.
- **Shared auxiliary objectives.** A linear state readout and a two-step rollout
  loss are used in every arm. The pure-JEPA ablation without them is secondary.
- **Simulated worlds.** A controlled 2D world first; PushT/Two-Room second. No
  real-robot data.
- **Three seeds.** Enough to see a large effect, not enough for a small one. We
  pre-register the effect size we care about.
- **Nuisance is defined by the simulator.** We can render the same state with
  different nuisance because we control the renderer. This is the cleanest
  possible test; real data will be messier.
- **The theorem is a guide, not a guarantee.** The action-conditioned predictor
  is outside its assumptions. A positive result is an extension; a negative
  result is a boundary.

---

## 10. What would change our mind

- H1 shows no probe difference: the mechanism does not transfer; stop.
- H2 shows a gain only at equal compute but not equal data: it is a compute
  effect, not a representation effect.
- H3 shows forgetting is unchanged: representation invariance is not the lever
  for continual learning; adaptation is.
- H4 shows alignment at chance: the rotation result needs the identity predictor,
  not just good views.
- The alternative hypothesis wins: good views hurt control. Then the project's
  contribution is the trade-off, not the recipe.

A clean negative at any rung is a result. The protocol is built to make it
publishable.

---

## 11. Why this is worth doing

- It tests a theorem **outside its proven scope**, in the regime where world
  models actually live.
- It connects **representation learning** (good views) to **continual learning**
  (forgetting), which are usually studied separately.
- It is cheap: a small model, a simulated world, one M5. The result does not
  depend on scale.
- It has a clear failure mode. If good views over-invariantize, we learn
  something about the limits of the Platonic recipe for control.

The north star is reliable small models that learn continually. This chapter asks
whether one representational lever — same-state views — is part of the answer.
