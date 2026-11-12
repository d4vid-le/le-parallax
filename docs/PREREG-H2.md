# Pre-registration H2 — planning under a nuisance shift

**Status:** **FROZEN 2026-11-12.** Do not change after this commit.
**Models:** the frozen H1 confirmatory checkpoints.
**Code:** `scripts/h2_plan.py`.

---

## 1. Question

Does good-view training improve **planning** under a nuisance shift, relative to
temporal-only training, using the same frozen checkpoints from H1?

This is the representation-to-control bridge. H1 established the mechanism at the
latent level; H2 asks whether the planner cashes it in.

---

## 2. Design

### Models

- Five seeds, `0–4`.
- Two arms: `temporal` and `good`, loaded from
  `runs/h1-confirm/models/s{seed}-{arm}.safetensors`.
- No retraining. The checkpoints are frozen.

### Test splits

| Split | Episodes per seed | Episode seed | Nuisance |
|---|---:|---:|---|
| OOD | 10 | `1000 + seed` | shifted: harsher light, brighter background, darker puck palette, camera ±0.08 |
| ID | 5 | `2000 + seed` | in-distribution |

The OOD shift is defined by `sample_nuisance_shift` in `leparallax/world.py`.

### Planner

- CEM, horizon 3, population 128, 16 elites, 5 iterations.
- State-readout terminal cost.
- 32-step episodes.
- Identical planner for both arms.

### Metrics

- **Primary:** mean final distance to goal on OOD episodes.
- **Secondary:** OOD success rate; ID mean final distance (no-regression).
- Random-policy baseline on the same episodes.

---

## 3. Endpoint

**H2a (primary).** Good-view OOD mean distance is at least **0.05 lower** than
temporal, and at least **4/5 seeds** are lower, with a paired bootstrap 95% CI
that excludes zero.

**H2b (no regression).** Good-view ID mean distance is no more than **0.05
worse** than temporal.

**H2c (success, secondary).** Good-view OOD success is not worse than temporal by
more than 5 percentage points. Reported, not gating.

**Falsifiers.**

- Good is not lower on OOD distance by 0.05.
- Good is worse on ID distance by more than 0.05.
- Success collapses while distance improves (the planner reaches the
  neighborhood but not the goal).

---

## 4. Analysis

- Paired by seed: one temporal and one good model per seed, evaluated on the same
  OOD episodes.
- Report per-seed OOD distance, OOD success, ID distance, and the paired
  difference.
- 95% bootstrap CI over the five paired differences (10,000 resamples).
- No tuning on the test episodes.
- No checkpoint selection; the H1 checkpoints are used as saved.

---

## 5. What this does not establish

- Continual retention (H3).
- Cross-modal rotation (H4).
- Real-world data.
- That good views help planning in general; only in this synthetic nuisance
  shift.

If H2a passes, the chapter has a representation-to-control result. If it fails
while H1a holds, the result is a dissociation: the latent becomes more world-like
but the planner does not benefit. Both are informative.
