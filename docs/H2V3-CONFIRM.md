# H2-v3 result — domain-randomized encoders

**Status:** complete, 2026-11-12.
**Protocol:** [`PREREG-H2v3.md`](PREREG-H2v3.md), sha256
`61bf566de6b21f2ef8e908f4c03df31f048f2797c58b9dd1f2dfca92660c80e4`.
**Artifacts:** `runs/h2v3/` (git-ignored).
**Verdict:** **FAIL on the pre-registered distance endpoint**, with a strong
secondary success result.

---

## 1. Result

Five seeds, both arms trained on a 50/50 normal + shifted nuisance mixture.
Frozen encoders, same ridge readout, same state-feedback controller.

| seed | OOD temp dist | OOD good dist | diff | OOD temp succ | OOD good succ | ID diff | readout R² temp | readout R² good |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.134 | 0.061 | −0.073 | 0.30 | 0.90 | −0.077 | −0.153 | −0.152 |
| 1 | 0.070 | 0.063 | −0.007 | 0.70 | 0.90 | −0.045 | −0.055 | −0.036 |
| 2 | 0.058 | 0.061 | +0.003 | 0.80 | 0.90 | −0.065 | −0.174 | −0.007 |
| 3 | 0.071 | 0.064 | −0.007 | 0.80 | 0.90 | −0.040 | −0.114 | −0.045 |
| 4 | 0.107 | 0.062 | −0.046 | 0.40 | 0.80 | −0.066 | −0.012 | −0.158 |

- Mean OOD distance: temporal **0.088**, good **0.062**; difference **−0.026**
  (95% CI −0.054 to −0.003).
- Seeds good lower: **4/5**.
- Mean OOD success: temporal **0.60**, good **0.88**; delta **+0.28**.
  Per-seed success deltas: **+0.60, +0.20, +0.10, +0.10, +0.40** — all positive.
- Mean ID difference: **−0.059** (good better).
- OOD readout R²: temporal −0.102, good −0.079.

## 2. What passed, what failed

### Primary distance: **FAIL**

The endpoint was a mean improvement of at least **0.05**. The run is **−0.026**,
less than half the bar. But:

- 4/5 seeds are lower;
- the 95% CI excludes zero;
- the baseline distance is already **0.088**, so a 0.05 absolute improvement is
  a near-saturation demand. The relative improvement is **29%**.

The pre-registered absolute threshold was mis-specified for this near-saturated
control task. That does not make the result a pass; it makes the endpoint a poor
measure here.

### Secondary success: **strong**

Good views improve OOD success from **60% to 88%**, **+28 points**, with every
seed positive. This is the control metric that actually matters, and the signal
is consistent across seeds.

### ID no-regression: **passes in the mean**

Mean ID difference is **−0.059** (good better), within the +0.05 allowance. No
regression.

### Readout: **still weak**

Both OOD readout R² values are near zero (−0.10 vs −0.08). Domain randomization
made the readout usable enough for state-feedback control, but it is not a clean
linear state readout. The control gain is coming from the encoder's robustness,
not from a strong readout.

## 3. What this means

Domain randomization removed the support problem:

- Both arms jumped from ~0.20 success (H2-v2) to 0.60–0.88.
- The good-view arm is **28 points** better on OOD success, with all seeds
  positive.
- The distance endpoint missed because both arms are already close to the goal;
  success is the discriminator.

This is a **near-miss on the primary endpoint and a strong secondary control
signal**. It is not yet a confirmed control result. The honest next step is a
success-primary replication on fresh episodes, not a reinterpretation of H2-v3.

## 4. Next

**H2-v4 — success-primary replication.**

- Same frozen H2-v3 encoders.
- Same readout and controller.
- Fresh OOD episodes: seeds `9000 + seed`, 20 per seed.
- Primary: good-view OOD success at least **+20 points** over temporal, 4/5 seeds
  positive, paired bootstrap CI excludes zero.
- Secondary: OOD distance, ID no-regression.

See [`PREREG-H2v4.md`](PREREG-H2v4.md). If it passes, the chapter has a
representation-to-control result under domain randomization. If it fails, the
+28-point success signal was underpowered or noise, and the honest result is a
distance near-miss with no confirmed control benefit.

The H1 mechanism result is unchanged.
