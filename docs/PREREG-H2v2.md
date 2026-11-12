# Pre-registration H2-v2 — state-feedback control with a robust readout

**Status:** **FROZEN 2026-11-12.** Do not change after this commit.
**Models:** the frozen H1 encoders.
**Code:** `scripts/h2v2_state_feedback.py`.

---

## 1. Question

With the planner confound removed, does the good-view encoder produce better
**state-feedback control** under a nuisance shift than the temporal encoder?

The diagnostic showed:

- a true-state proportional controller solves the task **100%**;
- CEM with true dynamics succeeds only **28%**;
- the learned state head is negative-R² OOD.

H2-v2 replaces CEM with the state-feedback controller and replaces the jointly
trained state head with a readout fitted on augmented latents. Both arms get the
same readout protocol. The encoder is the only difference.

---

## 2. Design

### Models

- Five seeds, `0–4`.
- Two frozen encoders: `s{seed}-temporal` and `s{seed}-good` from the H1
  confirmatory run.
- No encoder training.

### Readout

For each encoder and seed:

- 80 training episodes, 16 states per episode.
- Four nuisance renders per state: two from the normal distribution, two from
  the shifted distribution.
- Encode the 3-frame history for each render; fit a **ridge regression** from the
  64-d latent to `[x, y, vx, vy]`.
- `alpha = 1.0`; standardize latents on the readout training set.
- Same readout class and data budget for both arms.

The readout sees the shifted nuisance distribution; the encoder never does. This
isolates the encoder’s invariance from the readout’s training distribution.

### Controller

```
action = clip(2 * (goal - predicted_position) - 1.0 * predicted_velocity, -1, 1)
```

32 steps, success threshold 0.08. This is the controller that succeeds 100% with
the true state.

### Test splits

| Split | Episodes per seed | Episode seed | Nuisance |
|---|---:|---:|---|
| OOD | 10 | `3000 + seed` | shifted |
| ID | 5 | `4000 + seed` | in-distribution |

Fresh episodes; not the H2 or diagnostic episodes.

---

## 3. Endpoint

**H2v2a (primary).** Good-view OOD mean final distance is at least **0.05
lower** than temporal, with **4/5 seeds** lower and a paired bootstrap 95% CI
that excludes zero.

**H2v2b (no regression).** Good-view ID mean distance is no more than **0.05
worse** than temporal.

**H2v2c (readout).** The good-view readout has higher OOD position R² than the
temporal readout. Reported as a mechanism check, not a gating endpoint.

**Falsifiers.**

- No OOD distance improvement.
- ID regression above 0.05.
- Good readout is not better OOD, while control is also not better.
- The state-feedback controller fails for both arms despite a high oracle; then
  the readout is still the bottleneck and the experiment is inconclusive.

---

## 4. Analysis

- Paired by seed: one temporal and one good encoder/readout per seed.
- Per-seed OOD distance, OOD success, ID distance, and OOD readout R².
- 95% bootstrap CI over the five paired OOD differences.
- No tuning on the test episodes. `k=2`, `damp=1.0`, `alpha=1.0` are frozen from
  the true-state oracle and pilot.

---

## 5. What this does not establish

- H3 continual retention.
- H4 cross-modal rotation.
- Real-world data.
- That CEM is fixed; it is not used here.

If H2-v2 passes, the representation-to-control bridge holds under a fixed
readout and a valid controller. If it fails while the readout R² is higher for
good views, the failure is in the controller; if the readout R² is also not
higher, the failure is in the representation under this shift.
