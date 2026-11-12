# Pre-registration H2-v3 — domain-randomized encoders

**Status:** **FROZEN 2026-11-12.** Do not change after this commit.
**Models:** trained fresh with mixed nuisance.
**Code:** `scripts/h2v3_domain_rand.py`.

---

## 1. Question

When both encoders are trained on a **mixture of normal and shifted nuisances**,
does the good-view encoder produce better state-feedback control on a held-out
shift inside that support than the temporal encoder?

The previous tests failed because the shifted renders were outside the encoders'
training distribution. H2-v3 gives both arms support, then tests control.

---

## 2. Design

### Encoders

- Five seeds, `0–4`.
- Two arms: temporal and good view.
- 100 episodes per seed, 80 training episodes.
- Nuisance sampled 50/50 from the normal and shifted distributions
  (`nuisance_mode="mix"`).
- Good-view pairs use independent nuisances from the same mixture.
- 96px, 60 epochs, batch 32, lambda_view 2.0.
- Encoder training is the only difference between arms.

### Readout

- Ridge regression on augmented latents: 80 training episodes, 16 states each,
  4 nuisance renders per state (2 normal, 2 shifted).
- `alpha=1.0`; same for both arms.

### Controller

```
action = clip(2 * (goal - predicted_position) - 1.0 * predicted_velocity, -1, 1)
```

32 steps, success threshold 0.08.

### Test splits

| Split | Episodes per seed | Episode seed | Nuisance |
|---|---:|---:|---|
| OOD | 10 | `7000 + seed` | shifted |
| ID | 5 | `8000 + seed` | normal |

The OOD shift is inside the encoder training mixture's support. This is an
interpolation/robustness test, not an extrapolation test.

---

## 3. Endpoint

**H2v3a (primary).** Good-view OOD mean final distance is at least **0.05
lower** than temporal, with **4/5 seeds** lower and a paired bootstrap 95% CI
that excludes zero.

**H2v3b (no regression).** Good-view ID mean distance is no more than **0.05
worse** than temporal.

**H2v3c (readout).** Good-view OOD readout R² is higher than temporal. Reported;
not gating.

**Falsifiers.**

- No OOD distance improvement.
- ID regression above 0.05.
- Good readout not better, and control not better.
- Both arms fail despite domain randomization; then the controller/readout is
  still the bottleneck and the experiment is inconclusive.

---

## 4. Analysis

- Paired by seed: one temporal and one good model per seed, same episodes.
- Per-seed OOD distance, OOD success, ID distance, OOD readout R².
- 95% bootstrap CI over five paired differences.
- No tuning on test episodes. `k=2`, `damp=1.0`, `alpha=1.0`, `lambda_view=2.0`
  are frozen from prior diagnostics.

---

## 5. What this does not establish

- H3 continual retention.
- H4 cross-modal rotation.
- Real-world data.
- Extrapolation beyond the randomized nuisance range.

If H2-v3 passes, the representation-to-control bridge holds once the encoder has
support. If it fails, the good-view advantage does not survive control even with
support — a real dissociation.
