# Phase 0 — baseline report

**Date:** 2026-11-12
**Machine:** Apple M5 Pro, 64 GB, MLX 0.32.4
**Status:** baseline runs; good-view arms not implemented yet

---

## 1. What was built

- A synthetic 2D puck world: state `[x, y, vx, vy]`, continuous actions, rendered
  64×64 RGB observations, goal marker, and controllable nuisance (lighting,
  background color, puck color, camera offset).
- A small action-conditioned JEPA: CNN + spatial-softmax keypoints, 3-frame
  history, 64-d latent, residual predictor, SIGReg + variance/covariance terms,
  a contrastive next-latent term, a linear state readout, and a two-step rollout
  loss.
- Linear probes for state and nuisance, plus a direct same-state nuisance
  sensitivity metric.
- A CEM planner that uses the state readout for its terminal cost.
- `scripts/phase0.py`: end-to-end generation → training → probing → planning.

The state readout and the two-step loss are **auxiliary objectives shared by all
arms**. The independent variable in the experiment remains the positive-pair
definition. The pure-JEPA ablation without the state readout is a secondary
check, not the main baseline.

---

## 2. Baseline result

48 episodes, 32 steps, 64px, 60 epochs.

| Metric | Value |
|---|---:|
| State probe R² (mean) | **0.84** |
| Position probe R² | **0.89** |
| Velocity probe R² | **0.80** |
| Planner success (6 held-out episodes) | **2/6 (0.33)** |
| Random success (same episodes) | **1/6 (0.17)** |
| Planner mean final distance | **0.23** |
| Random mean final distance | **0.62** |
| Same-state nuisance distance | **5.34** |
| Between-state distance | **4.27** |
| Nuisance / state distance ratio | **1.25** |

The baseline is a usable world model: it encodes state, and the planner beats
random on distance and success. But the last row is the target for the next
chapter:

> The temporal model is **more sensitive to nuisance than to state**. Changing
> the lighting, background, puck color, and camera moves the latent 1.25× as far
> as moving the puck from one state to another.

That is the failure good views are meant to fix.

---

## 3. What failed before it worked

This is the useful part.

1. **The first encoder had near-zero variance.** The latent std at init was
   ~0.0004; the regularizer had no signal. Layer normalization on the frame and
   merged latents fixed the scale.
2. **A temporal-only latent was predictable but state-free.** The predictor could
   forecast a global scene summary without encoding the puck. The probe R² for
   position was ~0.1. This is the core failure the project is about.
3. **Contrastive next-latent prediction helped, but not enough.** InfoNCE on the
   predicted next latent raised within-train position R² to ~0.4, but the planner
   still failed. The latent was still not reliably state-like.
4. **A linear state readout made the baseline usable.** With a state head, the
   probe R² rose to ~0.84 and CEM began to move toward goals. This is an
   auxiliary objective, held constant across arms.
5. **Multi-step rollouts drifted.** Horizon 1 was the only safe planner; longer
   horizons got worse. A two-step rollout loss (predict `z_{t+2}` through two
   applications of the predictor) fixed the drift enough for CEM at horizons
   1–5. The planner reached 2/6 vs random 1/6.
6. **The nuisance probe was uninformative.** A linear readout of nuisance had
   negative R²; the latent does not expose nuisance linearly. The direct
   same-state distance metric is the better measurement and is now part of the
   protocol.

---

## 4. What this establishes

- The synthetic world and the training/eval loop work on the M5.
- The temporal baseline is a real world model, not a degenerate one: state probe
  ~0.84, planner above random.
- The baseline has the exact pathology the Platonic recipe predicts: it is more
  sensitive to nuisance than to state.
- A good-view arm can now be compared against a baseline that can actually plan.
  The comparison will not be “good views versus a broken model.”

## 5. What it does not establish

- No good-view arm has been run yet.
- Six test episodes is a smoke test, not a result. The planner numbers are noisy.
- 64px and 48 training episodes are small. The paper's scale is 96px and more
  data.
- The state readout is a supervised auxiliary; the pure-JEPA version without it
  is weaker. The main experiment tests nuisance invariance given a state-aware
  baseline, not whether good views create state from nothing.
- The nuisance probe is not usable as a linear readout. The same-state distance
  ratio is the primary nuisance metric.

## 6. Next

1. Implement the good-view and bad-view data modes.
   - Good view: two renders of the same state with independent nuisance.
   - Bad view: two crops of one render.
2. Train four arms with the same state head and two-step loss:
   temporal, good view, bad view, temporal + good view.
3. Run a one-seed pilot at 64px. Check the same-state nuisance ratio and the
   state probe before scaling.
4. Freeze the H1/H2 pre-registration.
5. Run three seeds at 96px.
