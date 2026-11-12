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
shift, a linear probe of physical state, and the **same-state nuisance
distance**: how far the latent moves when only the nuisance changes, divided by
how far it moves when the state changes. The secondary readout is forgetting
after a domain shift.

All arms share the same architecture and auxiliary objectives: a linear state
readout and a two-step rollout loss. The positive-pair definition is the only
difference. The baseline run is in [`docs/PHASE0.md`](docs/PHASE0.md).

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

Phase 0 baseline is running. On 48 episodes at 64px, the temporal baseline
reaches a state probe R² of **~0.84** and plans **2/6** vs random **1/6** on
held-out episodes. Its same-state nuisance distance is **1.25×** its
between-state distance: the latent is more sensitive to nuisance than to state.
That is the target for the good-view arms.

The good-view and bad-view modes are implemented. The exploratory pilot
([`docs/PILOT-VIEWS.md`](docs/PILOT-VIEWS.md)) found that good views reduce the
same-state nuisance ratio in every seed — 96px means **2.22 → 0.81**, with state
probe **0.62 → 0.86** — while bad views make it worse (**3.69** at 64px). The
original absolute `<0.5` threshold was not met; the confirmatory protocol uses a
paired relative endpoint ([`docs/PREREG-v2.md`](docs/PREREG-v2.md)). No
confirmatory result yet.

![Same state under two nuisance styles](snapshots/overview.png)

## Running Phase 0

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

python scripts/phase0.py --episodes 48 --steps 32 --size 64 --epochs 60
python scripts/pilot_views.py --episodes 48 --steps 32 --size 96 --epochs 60
pytest -q
```

The script writes `runs/phase0/report.json` and a checkpoint. `runs/` is
git-ignored.

---

## Layout

```
README.md              this document
HYPOTHESIS.md          formal claims, predictions, falsifiers
PLAN.md                phases, arms, metrics, compute, gates
leparallax/            world, data, model, training, probes, planner
scripts/phase0.py      end-to-end baseline
snapshots/              simulation snapshots (PNG)
docs/PHASE0.md         baseline report and failure log
docs/PILOT-VIEWS.md    exploratory good/bad-view pilot
docs/PREREG.md         pre-registration template for the first experiment
docs/PREREG-v2.md      confirmatory H1 protocol
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
