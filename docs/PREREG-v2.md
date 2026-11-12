# Pre-registration v2 — confirmatory H1

**Status:** **FROZEN 2026-11-12.** Do not change after this commit.
**Supersedes:** the H1 endpoint in `docs/PREREG.md`, which was written before the
pilot. The original endpoint is retained in `HYPOTHESIS.md` for the record.

---

## 1. Question

Does same-state good-view supervision reduce the latent's sensitivity to
nuisance while preserving its state readout, relative to a temporal-only
baseline?

This is the H1 mechanism claim. H2 (planning under nuisance shift) is a separate
pre-registration after H1.

---

## 2. Why v2

The exploratory pilot (`docs/PILOT-VIEWS.md`) found a consistent direction across
three seeds at 64px and 96px:

- 96px mean ratio: temporal **2.22**, good **0.81**.
- Good views reduced the ratio in every seed.
- The absolute `<0.5` endpoint was not met on average.
- One 96px run reached 0.392, but did not replicate.

The pilot is exploratory. v2 changes the endpoint to a **paired relative
reduction**, which is less sensitive to the baseline's seed variance, and
increases the sample to five seeds. The mechanism claim is the same; the
measurement is better matched to the observed variance.

---

## 3. Hypotheses and endpoints

**H1a (primary).** For each seed, the same-state nuisance ratio under good views
is at most **0.6×** the temporal baseline's ratio:

```
ratio_good / ratio_temporal <= 0.6
```

The five per-seed ratios are reported individually and as a mean with a 95%
bootstrap CI. The claim holds if the mean is ≤0.6 and at least 4 of 5 seeds are
≤0.7.

**H1b (non-inferiority).** State probe R² is at least **0.75** for both arms in
every seed. Good views must not buy invariance by losing state.

**H1c (specificity).** The bad-view control does not reduce the ratio relative to
the temporal baseline; it should be equal or worse.

**Falsifiers.**

- Mean relative ratio >0.8, or fewer than 3/5 seeds ≤0.7.
- State probe R² <0.60 in any arm.
- Bad views reduce the ratio as much as good views.
- The effect disappears at 96px.

---

## 4. Design

### Arms

| Arm | Positive pairs | `lambda_view` |
|---|---|---:|
| A. Temporal | `(o_t, a_t, o_{t+1})` | 0.0 |
| B. Good view | same state, independent nuisance | 2.0 |
| C. Bad view | two crops of one render | 2.0 |

All arms share the architecture, state readout, two-step rollout loss, SIGReg +
variance/covariance, and contrastive next-latent term.

### Data

- 100 episodes, 32 steps, 96px.
- 80 train / 10 validation / 10 test episodes.
- Five seeds: 0, 1, 2, 3, 4.
- Good views: 4 pairs per training episode, independent nuisance.
- Bad views: 4 pairs per training episode, two crops of one observation.

### Training

- 60 epochs, batch 32, AdamW, lr 3e-4, gradient clipping off (no clipping in the
  current loop; record if it changes).
- Good-view weight 2.0, frozen from the pilot.
- The state readout and two-step loss are held constant.

### Evaluation

- State probe R² on held-out episodes, per seed.
- Same-state nuisance ratio: mean same-state distance divided by mean
  between-state distance, per seed.
- Relative ratio: `ratio_good / ratio_temporal` per seed.
- Bad-view ratio for the specificity check.
- Per-seed values, mean, 95% bootstrap CI over seeds.

---

## 5. Analysis

- Paired by seed: each seed contributes one temporal and one good-view model.
- Primary statistic: mean of the five relative ratios.
- 95% bootstrap CI over the five ratios (10,000 resamples).
- Report the per-seed table, not just the mean.
- No tuning on the test episodes. The `lambda_view=2.0` choice is frozen from the
  pilot; if a different weight is used, this becomes a new pre-registration.

---

## 6. Stop rules

- If the paired mean ratio is >0.8, stop and report the negative.
- If state probe R² drops below 0.6 in any arm, stop and report the trade-off.
- If the bad-view control matches good views, stop and report that the effect is
  not specific to good views.
- If the sign flips across seeds, stop and report the instability.

---

## 7. Artifacts

```
runs/h1-confirm/
  manifest.json          # model, seeds, commit, hashes
  per_seed.json          # one row per seed and arm
  report.md              # pre-registered endpoint vs actual
  models/                # checkpoints, git-ignored
```

No aggregate-only reporting. The per-seed table is the result.

---

## 8. What this does not establish

- H2: planning under a nuisance shift. That is the next pre-registration.
- Continual retention (H3).
- Cross-modal rotation (H4).
- Anything about AGI, language, or real-world data.

A clean H1 result is a mechanism result: good views move the latent toward
nuisance invariance without losing state. It is not a world-model capability
claim until H2.
