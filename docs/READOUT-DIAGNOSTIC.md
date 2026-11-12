# Readout diagnostic — linear vs MLP

**Status:** exploratory, 2026-11-12.
**Models:** frozen H1 encoders.
**Test:** fresh OOD episodes, seeds `5000 + seed`; not the H2 or H2-v2 sets.
**Code:** `scripts/readout_diagnostic.py`.

---

## 1. Result

Means over five seeds:

| Arm | Linear readout R² | MLP readout R² | Linear success | MLP success |
|---|---:|---:|---:|---:|
| Temporal | **−1.290** | **−1.180** | 0.24 | 0.28 |
| Good view | **−0.074** | **−0.113** | 0.32 | 0.22 |

Per-seed readouts are in `runs/readout-diagnostic/report.md`.

The good latent is much easier to decode OOD than the temporal latent — linear
R² ≈ −0.07 vs −1.29 — but **both are still negative**. The MLP does not improve
on the linear readout; if anything it overfits slightly.

## 2. Interpretation

The readout class is not the limit. A nonlinear readout cannot recover the state
from either encoder under the shifted nuisance.

The encoder's **OOD support** is the limit. The H1 encoders were trained on the
normal nuisance distribution; the shifted renders are outside it. The good
encoder degrades less, which is consistent with H1, but neither is good enough
to drive the controller.

This is the same conclusion as the planner diagnostic, one layer deeper:

> The representation is better, but the shift is outside the encoder's training
> distribution. No readout can fix that. The next experiment is domain
> randomization during encoder training.

## 3. Next

**H2-v3 — domain randomization.**

- Train both arms on a mixture of normal and shifted nuisances, so the encoder
  sees the test range.
- Keep the same good-view intervention and the same temporal baseline.
- Use the same ridge readout and state-feedback controller as H2-v2.
- Test on a held-out shift distribution inside the training support, with fresh
  episodes.
- Primary endpoint remains OOD distance, paired by seed.

If H2-v3 passes, the representation-to-control bridge holds once the encoder has
support. If it fails, the good-view advantage does not survive control even with
support, which is a real dissociation.

The H1 mechanism result is unchanged: good views make the latent more state-like
and less nuisance-like, by a large replicated margin.
