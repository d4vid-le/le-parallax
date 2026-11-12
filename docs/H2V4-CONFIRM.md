# H2-v4 result — success-primary replication

**Status:** complete, 2026-11-12.
**Protocol:** [`PREREG-H2v4.md`](PREREG-H2v4.md), sha256
`ba4e3c37adcf4a5190394683684e942f382336838628e1f934c4623dac6daf57`.
**Artifacts:** `runs/h2v4/` (git-ignored).
**Verdict:** **PASS.**

---

## 1. Result

Five seeds, frozen H2-v3 encoders, same ridge readout, same state-feedback
controller, 20 fresh OOD episodes per seed (seeds `9000 + seed`).

| seed | OOD temp succ | OOD good succ | delta | OOD temp dist | OOD good dist | McNemar b/c | p |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.60 | 0.95 | +0.35 | 0.112 | 0.052 | 8/1 | 0.039 |
| 1 | 0.50 | 0.95 | +0.45 | 0.108 | 0.051 | 9/0 | 0.004 |
| 2 | 0.75 | 0.95 | +0.20 | 0.094 | 0.056 | 4/0 | 0.125 |
| 3 | 0.90 | 0.90 | +0.00 | 0.069 | 0.062 | 1/1 | 1.000 |
| 4 | 0.65 | 0.80 | +0.15 | 0.096 | 0.075 | 6/3 | 0.508 |

- Mean success delta: **+0.23** (95% CI **+0.10 to +0.37**).
- Seeds positive: **4/5**.
- Pooled McNemar: good-only **28**, temporal-only **5**, **p = 0.0001**.
- Mean OOD distance difference: **−0.036** (good better).
- Mean ID distance difference: **−0.055** (good better).
- ID success delta: **+0.36** (good better).

## 2. Endpoint

The pre-registered endpoint was:

- success delta ≥ **+20 points**,
- **4/5 seeds** positive,
- pooled McNemar **p < 0.05**,
- ID no regression.

All four hold. The verdict is **PASS**.

## 3. What this establishes

Under domain randomization, with the planner and readout fixed:

> Good-view training improves out-of-distribution control success by **23
> percentage points** over the temporal baseline (95% CI +10 to +37), replicated
> on fresh episodes with 4/5 seeds positive and a pooled McNemar p = 0.0001.

The distance also improves (−0.036), and ID performance improves rather than
regresses. The representation-to-control bridge holds when the encoder has
support for the shift.

The H1 mechanism result is the explanation:

- nuisance sensitivity ratio **0.114** (5/5 seeds, 95% CI 0.089–0.143);
- state probe **0.90–0.95** vs **0.69–0.87**;
- bad views **6.7×** worse.

Good views make the latent more world-like; once the encoder has support and the
controller is valid, that advantage shows up in control.

## 4. What this does not establish

- **Not H2-v3 distance.** H2-v3 failed its pre-registered distance endpoint. H2-v4
  is a new pre-registered success-primary experiment on fresh episodes; it does
  not retroactively fix H2-v3.
- **Not H1b.** The absolute state floor failed on one temporal baseline seed. H1a
  and H1c are confirmed; the full H1 claim is not.
- **Not continual learning.** H3 has not been run.
- **Not real-world data.** Synthetic world, domain randomization, 96px.
- **Not AGI.** A 15M-ish model in a toy world.

## 5. Next

**H3 — continual retention.**

- Does good-view pretraining reduce forgetting under a domain shift?
- Separate nuisance shift from dynamics shift.
- Measure old-domain retention, new-domain success, and forgetting.
- Use the H2-v3/v4 models and controller where possible; otherwise retrain with
  the same protocol.

If H3 passes under nuisance shift but not dynamics shift, the chapter has a clean
story: good views buy nuisance invariance, and adaptation still handles dynamics.
If H3 fails, the control benefit does not survive continual adaptation.

The chapter's confirmed results so far:

```
H1a/H1c  good views reduce nuisance sensitivity without losing state
H2-v4    good views improve OOD control success under domain randomization
H1b      failed literally on an absolute state floor (protocol error)
H2       failed on CEM; H2-v2 failed on readout/OOD support
H2-v3    failed on distance, strong secondary success
```
