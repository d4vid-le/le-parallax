# Related work and the gap

LeParallax sits between three lines of work: the Platonic Representation
Hypothesis, JEPA representation learning, and action-conditioned world models.
This note records what we borrow, what we test, and what we do not claim.

---

## 1. The Platonic Representation Hypothesis

Huh, Cheung, Wang, and Isola (2024) observed that vision and language models
become more similar as they scale, and hypothesized that they converge to a
shared statistical model of reality. The hypothesis was supported by
representation-similarity measurements, not a proof.

The practical question for this project: if the convergence depends on the
**positive pairs** a model uses, then scale is not the only lever. The pair
definition is.

---

## 2. The Platonic JEPA theorem

Balestriero and LeCun (2026, arXiv:2610.39302) turn the hypothesis into a theorem
for JEPAs:

- A JEPA with second-order anti-collapse learns the spectral embedding of the
  graph its positive pairs draw.
- If two modalities' positives are **two observations of the same world state**
  with independent nuisance — "good views" — their encoders agree up to one
  rotation.
- If the positives are **corruptions of one observation** — crops, masks — the
  encoder spends dimensions on modality nuisance, and alignment is bounded by
  the kernel mismatch.
- Next-token prediction is a bad-view JEPA: a prefix and its continuation share
  the document's nuisance. This explains why language models align with vision
  mostly through scale.
- Good views need roughly 42× fewer samples for the same alignment on their
  benchmark; a 109M text JEPA aligns better than LLMs up to 72B.

The theorem is the reason this project exists. It gives a clean intervention:
change the positives, hold everything else fixed, and see whether the
representation moves.

---

## 3. The hole: action-conditioned predictors

The theorem's predictor is the identity. The paper states the scope directly:

> "the predictor is the identity ... whereas JEPAs used as world models predict
> one embedding from the other through a learned predictor, often conditioned on
> an action, which our analysis does not cover."

Every action-conditioned world model is outside the proof:

- **LeWM** (Maes et al., 2026, arXiv:2603.19312) trains an encoder and a
  predictor end-to-end on `(o_t, a_t, o_{t+1})` with MSE + SIGReg. The positives
  are temporal transitions, not same-state views.
- **V-JEPA** predicts masked video features; the task is representation learning,
  not control.
- **Dreamer** learns a latent dynamics model from rewards and pixels; the
  objective is reconstruction/prediction, not good-view alignment.

The open question is not whether the theorem is true. It is whether its recipe
helps when the predictor is action-conditioned and the downstream metric is
control.

---

## 4. What we borrow

- **The good-view definition.** Two observations of one state with independent
  nuisance. This is the intervention.
- **The bad-view control.** Two crops of one observation. This is the negative
  control.
- **SIGReg.** The anti-collapse regularizer from LeJEPA, used by LeWM. It is
  rotation-invariant and cheap.
- **The small-world-model recipe.** A ViT-tiny/CNN encoder, a compact predictor,
  96px, one machine.
- **The probe methodology.** Linear readouts of physical state, and the idea of
  measuring what the latent explains.
- **The instinct that positive pairs are the lever.** This is the hypothesis we
  test, not an assumption we import.

---

## 5. What we test

| Rung | Question | Why it is not covered by the theorem |
|---|---|---|
| H1 | Does good-view supervision make the latent more state-like? | The predictor is action-conditioned; the theorem assumes identity. |
| H2 | Does that improve planning under nuisance shift? | The theorem measures alignment, not control. |
| H3 | Does it reduce forgetting under domain shift? | The theorem has no continual-learning setting. |
| H4 | Do separately trained modalities align up to a rotation? | The theorem's rotation result assumes identity predictors. |

H1 is a manipulation check. H2 is the behavioral claim. H3 is the continual-
learning link. H4 is the modularity claim.

---

## 6. What we do not claim

- That the Platonic theorem is wrong. It is a theorem under its assumptions.
- That good views are sufficient for control. They may remove task-relevant
  information.
- That this solves continual learning. It tests one representational lever.
- That a 1–5M model is a path to AGI. It is a controlled testbed for a mechanism.

---

## 7. The contribution if it works

A small, reproducible result:

> Same-state good views improve the state-likeness of an action-conditioned world
> model's latent, and that improvement survives the translation to planning and
> retention under nuisance shift — without a meaningful in-distribution cost.

If it fails, the contribution is the boundary:

> Good views improve the representation but not the controller, or they improve
> robustness at the cost of task-relevant detail. The Platonic recipe is a
> representation result, not a control result.

Either way, the result is about the gap between representation and control, which
is the part of the world-model story that is still open.

---

## 8. References

- Huh, Cheung, Wang, Isola. *The Platonic Representation Hypothesis.* 2024.
- Balestriero, LeCun. *Proving and Improving the Platonic Representation
  Hypothesis with JEPAs.* arXiv:2610.39302, 2026.
- Balestriero, LeCun. *LeJEPA: Provable and Scalable Self-Supervised Learning
  Without the Heuristics.* arXiv:2511.08544, 2025.
- Maes, Le Lidec, Scieur, LeCun, Balestriero. *LeWorldModel: Stable End-to-End
  Joint-Embedding Predictive Architecture from Pixels.* arXiv:2603.19312, 2026.
- Bardes et al. *V-JEPA: Latent Video Prediction for Visual Representation
  Learning.* 2023.
- Hafner, Pasukonis, Ba, Lillicrap. *Mastering Diverse Domains through World
  Models.* 2023.
- Assran et al. *I-JEPA: Self-Supervised Learning from Images with a Joint-
  Embedding Predictive Architecture.* 2023.
