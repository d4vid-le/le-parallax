# LeParallax

**Aspect-seeing in action-conditioned world models.**

LeParallax is a small, simulated research project about one question:

> When a world model looks at the world, does it learn the world or the camera?

The name joins two ideas. **Parallax** is the apparent shift in a scene when it is
viewed from two positions: the difference between the views is nuisance, the
invariant is the state. Wittgenstein's **aspect-seeing** is the same image seen
as two different things. The project asks which of the two a learned latent
captures.

This is not a product. It is a controlled experiment with a small model, a
simulated world, and a pre-registered claim. The north star is reliable small
models that learn continually; this chapter is one wedge into that problem.

---

## The gap

The Platonic Representation Hypothesis says models trained on different
modalities converge to a shared representation of the world. A recent theorem
(Balestriero & LeCun, 2026) makes this precise for JEPAs: if the positive pairs
are **two observations of the same world state** with independent nuisance —
"good views" — then separately trained encoders learn the same representation up
to one rotation. If the positives are corruptions of one observation — crops,
masks, a prefix and its continuation — the encoder spends dimensions on
modality-specific nuisance and aligns only partly.

The theorem has a deliberate hole:

> "the predictor is the identity ... whereas JEPAs used as world models predict
> one embedding from the other through a learned predictor, often conditioned on
> an action, which our analysis does not cover."

That is exactly the regime of LeWM, Dreamer, and every action-conditioned world
model. Their positive pairs are temporal transitions, not same-state views. We
do not know whether good views help a world model, hurt it, or do nothing.

LeParallax tests that hole.

---

## The hypothesis

**Same-state good views make a small action-conditioned world model's latent more
state-like and less nuisance-like, which improves out-of-distribution planning
and reduces forgetting under nuisance shift — without a meaningful
in-distribution regression.**

The mechanism is representational. A good view changes the nuisance (lighting,
texture, camera) while holding the state fixed. A model trained to align good
views must encode what survives the change: the state. A model trained only on
temporal transitions can encode whatever helps predict the next frame, including
nuisance that happens to correlate with the dynamics.

The risk is equally clear. Nuisance is not always noise. Color can identify a
block; texture can signal friction. If good views remove task-relevant
information, the planner will suffer. The experiment is designed to detect both
outcomes.

---

## The experiment at a glance

A small world model is trained on a simulated 2D world with controllable state
and nuisance. Four arms, same compute:

| Arm | Positive pairs | What it should learn |
|---|---|---|
| **Temporal** | `(o_t, a_t, o_{t+1})` | dynamics |
| **Good view** | two independent renders of the same state | state, not nuisance |
| **Bad view** | two crops of one render | appearance, not state |
| **Temporal + good view** | both | dynamics plus a state-like latent |

The primary readouts are planning success in-distribution and under a nuisance
shift, a linear probe of physical state, and a linear probe of nuisance. The
secondary readout is forgetting after a domain shift.

A full protocol, including arms, metrics, thresholds, seeds, and stop rules, is
in [`PLAN.md`](PLAN.md). The formal hypotheses and their falsifiers are in
[`HYPOTHESIS.md`](HYPOTHESIS.md).

---

## What this is not

- **Not AGI.** The model is small, the world is simulated, and the claims are
  narrow. AGI is the direction, not the result.
- **Not a language model.** There is no text encoder in the first experiment.
  Language may enter later as a second modality; it is not the initial variable.
- **Not a full continual-learning algorithm.** Good views are one lever on
  representation stability. Adaptation still needs a mechanism; if the effect is
  real, it composes with Le-Gates or another adapter.
- **Not a benchmark paper.** One model family, a few controlled shifts, three
  seeds. The goal is a clean mechanism result, not a leaderboard.

---

## Status

Planning. No code yet.

The first milestone is a synthetic 2D world and a baseline that reproduces the
known LeWM behaviour at small scale. The first claim is H1, the representation
claim; we do not run the control experiment until H1 is visible in the probes.

---

## Layout

```
README.md              this document
HYPOTHESIS.md          formal claims, predictions, falsifiers
PLAN.md                phases, arms, metrics, compute, gates
docs/PREREG.md         pre-registration template for the first experiment
docs/RELATED-WORK.md   the papers this project sits between
```

Code will be added after the first pre-registration. The stack is deliberately
undecided until Phase 0; the likely path is MLX for the model and training, with
a small NumPy simulator for the controlled world, and PyTorch only if a
stable-worldmodel environment is needed.

---

## References

- Huh, Cheung, Wang, Isola. *The Platonic Representation Hypothesis.* 2024.
- Balestriero, LeCun. *Proving and Improving the Platonic Representation
  Hypothesis with JEPAs.* arXiv:2610.39302, 2026.
- Balestriero, LeCun. *LeJEPA: Provable and Scalable Self-Supervised Learning
  Without the Heuristics.* arXiv:2511.08544, 2025.
- Maes, Le Lidec, Scieur, LeCun, Balestriero. *LeWorldModel: Stable End-to-End
  Joint-Embedding Predictive Architecture from Pixels.* arXiv:2603.19312, 2026.
- Bardes, Garrido, Ponce, Chen, Rabbat, LeCun, Assran, Ballas. *V-JEPA: Latent
  Video Prediction for Visual Representation Learning.* 2023.
- Hafner, Pasukonis, Ba, Lillicrap. *Mastering Diverse Domains through World
  Models.* 2023.
