# Pre-registration H2-v4 — success-primary replication

**Status:** **FROZEN 2026-11-12.** Do not change after this commit.
**Models:** the frozen H2-v3 encoders.
**Code:** `scripts/h2v4_success.py`.

---

## 1. Why this experiment

H2-v3 failed its pre-registered distance endpoint (−0.026 vs −0.05), but showed a
strong secondary success result: **88% vs 60%** OOD, **+28 points**, with all five
seeds positive.

The distance endpoint was near-saturated: the temporal baseline was already at
0.088, so a 0.05 absolute improvement was a very high bar. Success is the control
metric that matters. H2-v4 pre-registers success as the primary endpoint on
**fresh** OOD episodes, using the same frozen encoders and the same readout and
controller.

This is a new experiment, not a reinterpretation of H2-v3.

---

## 2. Design

### Models

- Five seeds, `0–4`.
- Frozen H2-v3 encoders (`runs/h2v3/models/s{seed}-{arm}.safetensors`).
- Two arms: temporal and good view.

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
| OOD | 20 | `9000 + seed` | shifted |
| ID | 5 | `10000 + seed` | normal |

Fresh episodes; not used in H2-v3 or any diagnostic.

---

## 3. Endpoint

**H2v4a (primary).** Good-view OOD success is at least **20 percentage points**
higher than temporal, with **4/5 seeds** positive, and a pooled paired McNemar
test over the 100 episode pairs at **p < 0.05**.

**H2v4b (secondary).** Good-view OOD mean distance is lower than temporal.
Reported with a paired bootstrap CI over seeds.

**H2v4c (no regression).** Good-view ID mean distance is no more than **0.05
worse** than temporal, and ID success is not worse by more than 10 points.

**Falsifiers.**

- Success delta below 20 points.
- Fewer than 4/5 seeds positive.
- McNemar p ≥ 0.05.
- ID regression beyond the allowance.

---

## 4. Analysis

- Per-seed OOD success and distance.
- Pooled 2×2 table over 100 paired episodes: good-win, temporal-win, both-win,
  both-lose.
- Exact two-sided McNemar (binomial on discordant pairs).
- 95% bootstrap CI over the five per-seed success deltas.
- No tuning on test episodes. Frozen controller/readout.

---

## 5. What this does not establish

- H3 continual retention.
- H4 cross-modal rotation.
- Real-world data.
- That the success gain is large outside this simulated, domain-randomized
  setting.

If H2-v4 passes, the chapter has a representation-to-control result on success
under domain randomization. If it fails, the H2-v3 +28-point signal was
underpowered or noise, and the honest result is a distance near-miss with no
confirmed control benefit.
