# View pilot — exploratory report

**Status:** exploratory, not confirmatory.
**Date:** 2026-11-12.
**Machine:** M5 Pro, 64 GB, MLX.
**Code:** `leparallax/` + `scripts/pilot_views.py`.

This is the first A/D comparison: temporal-only vs temporal + good views, plus a
bad-view control. It is a pilot. It is not the pre-registered confirmatory run.

---

## 1. Setup

- 48 episodes, 32 steps, 60 epochs, batch 32.
- Two resolutions: 64px and 96px.
- Three seeds for the A/D comparison: 0, 1, 2.
- Good-view weight: `lambda_view=2.0` for the 3-seed runs.
- Bad-view control: one seed at 64px, `lambda_view=2.0`.
- Model and auxiliary objectives are identical across arms: state readout,
  two-step rollout loss, SIGReg + variance/covariance, contrastive next-latent.

---

## 2. Results

### 64px — three seeds

| Arm | Ratio per seed | Mean ratio | State R² per seed | Mean state R² |
|---|---|---:|---|---:|
| Temporal | 1.63, 1.25, 2.70 | **1.86** | 0.80, 0.76, 0.69 | **0.75** |
| Good view | 0.67, 0.85, 1.25 | **0.93** | 0.87, 0.82, 0.89 | **0.86** |

Good views reduce the ratio in every seed. The mean reduction is **50%**, but
the absolute ratio is **0.93**, not below the original 0.5 threshold.

### 96px — three seeds

| Arm | Ratio per seed | Mean ratio | State R² per seed | Mean state R² |
|---|---|---:|---|---:|
| Temporal | 1.54, 1.93, 3.21 | **2.22** | 0.63, 0.75, 0.46 | **0.62** |
| Good view | 0.69, 0.74, 1.00 | **0.81** | 0.84, 0.84, 0.89 | **0.86** |

The direction is consistent: good views reduce the ratio in every seed and raise
the state probe in every seed. The mean reduction is **64%**. The absolute mean
is **0.81**, still not below 0.5.

One earlier 96px seed-0 run reached **0.392** with state R² **0.925**, but it did
not replicate under the 3-seed protocol. Treat it as a lucky initialization, not
a result.

### Bad-view control (64px, one seed)

| Arm | Ratio | State R² | Planner success | Mean distance |
|---|---:|---:|---:|---:|
| Temporal | 2.18 | 0.71 | 0.20 | 0.240 |
| Bad view | **3.69** | 0.67 | 0.40 | 0.184 |
| Good view | **0.90** | 0.84 | 0.60 | 0.140 |
| Random | — | — | 0.40 | 0.371 |

Bad views make the latent **more** nuisance-sensitive than the temporal
baseline, as the theory predicts. The planning column is five episodes and is
not confirmatory, but the ordering is consistent with the representation result.

### Good-view weight sweep (64px, one seed)

| `lambda_view` | State R² | Ratio |
|---:|---:|---:|
| 0.5 | 0.84 | 0.90 |
| 1.0 | 0.82 | 0.73 |
| 2.0 | 0.90 | 0.49–0.57 (seed-sensitive) |
| 4.0 | 0.79 | 1.08 |

Invariance is non-monotonic. Too little weight leaves the model nuisance-
sensitive; too much weight over-invariantizes and starts losing state. The sweet
spot is around 2.0 at this scale.

---

## 3. What this establishes

- **The direction is real.** Good views reduce nuisance sensitivity in every
  seed, at both resolutions. Bad views increase it.
- **State is not sacrificed.** The state probe is higher for good views in every
  seed; the state readout keeps the latent usable.
- **Resolution matters.** The effect is stronger at 96px than 64px, which is
  consistent with a small puck being harder to localize at low resolution.
- **Invariance has a cost at the margin.** `lambda_view=4.0` over-invariantizes
  and loses state.
- **The bad-view control behaves as predicted.** It is worse than the temporal
  baseline on nuisance sensitivity.

## 4. What this does not establish

- **The original H1 threshold is not met.** The mean ratio is 0.81 at 96px, not
  <0.5. Do not claim H1.
- **The planning result is underpowered.** Five episodes, one seed. The ordering
  is encouraging, not confirmatory.
- **Three seeds, 48 episodes.** The baseline ratio varies 1.54–3.21 across seeds;
  the effect size needs a larger sample.
- **No private holdout.** The synthetic world has no equivalent; the confirmatory
  run uses a frozen seed split.
- **The state readout is supervised.** This is a state-aware baseline, not a pure
  JEPA. The pure-JEPA ablation is secondary.

## 5. Decision

The pilot confirms the mechanism direction and rules out the absolute `<0.5`
endpoint as a realistic target at this scale. The confirmatory experiment should
use a **paired relative endpoint**:

> The per-seed ratio `good / temporal` is at most **0.6**, with state probe R²
> at least **0.75** in every arm.

The pilot's per-seed relative ratios are 0.41, 0.68, 0.46 at 64px and 0.45, 0.38,
0.31 at 96px. A 0.6 threshold is a real but achievable bar. The exact protocol is
in [`PREREG-v2.md`](PREREG-v2.md).

The original absolute threshold remains in `HYPOTHESIS.md` for the record. It was
not met; the confirmatory run is a new pre-registration, not a retrofit.

## 6. Next

1. Run `docs/PREREG-v2.md` with five seeds at 96px and 100 episodes.
2. Keep the bad-view control and the random planner baseline.
3. Report per-seed ratios, the paired effect, state probe, and planning with
   confidence intervals.
4. If the relative endpoint holds, write H1 up as a mechanism result; if not,
   report the negative and move to H3 (retention).
