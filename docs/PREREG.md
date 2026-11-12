# Pre-registration template

Copy this file to `docs/PREREG-<experiment>.md`, fill it in, commit it, and record
the commit hash before the confirmatory run. The pilot may run before the freeze;
label it exploratory and do not report it as confirmation.

---

## 1. Identification

- **Experiment ID:** H1-H2-synthetic-v1
- **Date frozen:**
- **Commit hash:**
- **Author:**
- **Status:** draft / frozen / confirmed / abandoned

---

## 2. Question

Does same-state good-view supervision change the latent of a small
action-conditioned world model in the predicted direction, and does that change
improve out-of-distribution planning without a meaningful in-distribution
regression?

---

## 3. Hypotheses

**H1 (mechanism).** Good-view supervision reduces the same-state nuisance
distance ratio below **0.5**, versus above **1.0** for the temporal baseline,
while keeping state probe R² at least **0.75**. The Phase 0 baseline ratio is
**1.25** with state R² **0.84**.

**H2 (behavior).** At 50% of the training data, good-view pretraining improves
OOD planning success by at least 10 percentage points over the temporal baseline,
with no more than 2 points of in-distribution regression.

**Alternative.** Good views over-invariantize the latent: state probes improve,
nuisance probes drop, but in-distribution planning or dynamics prediction
degrades. This is a pre-registered possible outcome, not a failure of the
protocol.

---

## 4. Design

### Arms

| Arm | Positive pairs | Notes |
|---|---|---|
| A. Temporal | `(o_t, a_t, o_{t+1})` | baseline |
| B. Good view | same state, independent nuisance | intervention |
| C. Bad view | two crops of one render | negative control |
| D. Temporal + good | both | practical candidate |

### Data fractions

25%, 50%, 100% of the training trajectories.

### Seeds

Three seeds per arm and fraction: `0, 1, 2`.

### Model

- Encoder: CNN or ViT-tiny, 96×96 input.
- Predictor: action-conditioned MLP or small transformer.
- Latent: 64–192 dimensions.
- Regularizer: SIGReg plus variance/covariance terms.
- Auxiliary objectives: a linear state readout and a two-step rollout loss,
  shared by every arm.
- Parameters: 1–5M.
- Training: AdamW, fixed schedule, gradient clipping.
- Batch: 32–64.
- Good-view loss weight: frozen after the pilot.

### Compute

One M5 Mac mini, 64 GB. One run under one hour. Total confirmatory budget:
4 arms × 3 seeds × 3 fractions ≈ 36 runs.

---

## 5. Splits

| Split | Definition | Use |
|---|---|---|
| ID | Nuisance distribution A, dynamics A | Primary in-distribution metric |
| OOD-nuisance | Nuisance distribution B, dynamics A | Primary OOD metric |
| OOD-dynamics | Nuisance A, dynamics B | Secondary; H3 only |
| Held-out seeds | New random seeds for the simulator | No tuning |

The OOD-nuisance split is fixed before training. It is never used to choose the
loss weight or the checkpoint.

---

## 6. Metrics

### Primary

- **State probe R²** — linear readout of position and velocity from the latent.
- **Same-state nuisance distance ratio** — distance between latents of the same
  state under different nuisance, divided by the distance between different
  states. Lower is better. This is the primary nuisance metric.
- **Nuisance probe R²** — linear readout of lighting/texture/camera. Secondary;
  Phase 0 showed it can be uninformative.
- **OOD planning success** — goal reached under nuisance shift.
- **ID planning success** — goal reached in-distribution.

### Secondary

- Mutual 10-NN alignment with a state encoder.
- Success variance across seeds.
- Cost-to-go.
- Dynamics prediction error.

### Reporting

- Per-seed values, mean, and 95% bootstrap CI.
- Per-task diff for planning.
- Paired comparison against the temporal baseline.
- Effect size, not just p-value.

---

## 7. Thresholds

| Claim | Threshold | Falsifier |
|---|---|---|
| H1 state | ≥0.75 R² for every arm | <0.60 |
| H1 nuisance ratio | <0.50 for good views | >0.70 |
| H1 ordering | good < temporal + good < temporal ≈ bad | bad ≈ good |
| H2 OOD | +10 points at 50% data | < +5 |
| H2 ID | ≤2 points regression | >5 points regression |
| H2 seeds | CI excludes zero | CI includes zero |

The thresholds are fixed before the confirmatory run. A result below threshold is
a negative result, not a reason to change the threshold.

---

## 8. Analysis

- Paired bootstrap over seeds and tasks for the planning metrics.
- Linear probe with a fixed train/test split; report R² on the held-out split.
- Check the nuisance–state correlation matrix; if any nuisance variable is
  correlated with state above 0.2, flag it and re-sample.
- Report the alternative hypothesis explicitly if ID planning regresses.

---

## 9. Stop rules

- If Gate 0 fails (baseline does not plan above random, or probes are at chance),
  fix the pipeline. Do not run the arms.
- If H1 fails, stop and write the negative. Do not run H2.
- If H1 passes but H2 fails, write the invariance–control trade-off. Do not run
  H3.
- If H2 passes, proceed to H3 with a new pre-registration.

---

## 10. Deviations log

Record every change after the freeze:

| Date | Change | Reason | Impact |
|---|---|---|---|
| | | | |

A deviation does not invalidate the experiment, but it must be reported. If the
change affects the primary endpoint, the run becomes exploratory.

---

## 11. Artifacts

- `runs/<experiment>/manifest.json` — model, seeds, commit, hashes.
- `runs/<experiment>/metrics.jsonl` — one row per run and metric.
- `runs/<experiment>/probes/` — probe weights and R².
- `runs/<experiment>/planning/` — per-task success.
- `runs/<experiment>/report.md` — the pre-registered endpoint vs the actual.
- No aggregate-only reporting.
