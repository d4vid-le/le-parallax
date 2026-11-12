# H1 confirmatory result

**Status:** complete, 2026-11-12.
**Protocol:** [`PREREG-v2.md`](PREREG-v2.md), sha256
`d08d0d69d24cadd8a2b46a0381056b222a34a7a69fa8bec68dc2ea466c95e411`.
**Artifacts:** `runs/h1-confirm/` (git-ignored).
**Verdict:** **FAIL as pre-registered**, on H1b only. H1a and H1c pass
strongly.

---

## 1. Result

Five seeds, 100 episodes, 96px, 60 epochs, three arms.

| seed | temporal ratio | good ratio | rel | bad ratio | temporal R² | good R² |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2.244 | 0.287 | 0.128 | 6.182 | 0.870 | 0.932 |
| 1 | 1.961 | 0.323 | 0.165 | 7.113 | **0.687** | 0.898 |
| 2 | 2.804 | 0.303 | 0.108 | 6.408 | 0.793 | 0.945 |
| 3 | 3.061 | 0.282 | 0.092 | 7.382 | 0.843 | 0.917 |
| 4 | 2.627 | 0.199 | 0.076 | 6.457 | 0.859 | 0.939 |

- Mean relative ratio: **0.114** (95% bootstrap CI **0.089–0.143**).
- Seeds with rel ≤0.7: **5/5**.
- Good-view mean ratio: **0.279**; temporal mean: **2.539**.
- Bad-view mean ratio: **6.708**.
- Minimum state R²: **0.687** (temporal, seed 1).

---

## 2. What passed, what failed

### H1a — primary relative effect: **PASS**

Good views cut the same-state nuisance ratio to **11.4%** of the temporal
baseline (CI 8.9–14.3%). The pre-registered bar was ≤60% mean and 4/5 seeds
≤70%; the run is 5/5 and roughly five times better than the bar.

### H1b — absolute state floor: **FAIL**

The protocol required state probe R² ≥0.75 for **both** arms in **every** seed.
Seed 1's temporal baseline came in at **0.687**. The good arm in that seed is
**0.898**, and the good arm is higher than temporal in all five seeds:

| seed | temporal R² | good R² | margin |
|---:|---:|---:|---:|
| 0 | 0.870 | 0.932 | +0.062 |
| 1 | 0.687 | 0.898 | +0.211 |
| 2 | 0.793 | 0.945 | +0.152 |
| 3 | 0.843 | 0.917 | +0.074 |
| 4 | 0.859 | 0.939 | +0.079 |

The *spirit* of H1b — good views do not sacrifice state — holds in 5/5 seeds.
The *letter* fails because the baseline itself dipped below an absolute floor.
This is a protocol design error: an absolute floor on both arms conflates
baseline variance with treatment harm. The correct test is relative
non-inferiority: `good_state ≥ temporal_state`, plus a floor on the treatment
arm. H1b as written is failed and will not be retroactively changed.

### H1c — bad-view specificity: **PASS**

Bad views are **6.708** on average, worse than temporal (**2.539**) and far worse
than good (**0.279**). The effect is specific to same-state good views, not to
augmentation in general.

---

## 3. What this means

The mechanism result is real and large:

> Same-state good-view supervision moves an action-conditioned world model's
> latent toward the world and away from the camera. It cuts nuisance sensitivity
> by ~89% relative to a temporal baseline, while making physical state *more*
> readable, not less. Corruptions of one observation do the opposite.

This is the first confirmatory result of the chapter. It extends the Platonic
JEPA recipe — which was proven for identity predictors — to an action-conditioned
world model in a controlled simulation.

It is **not** a capability result yet:

- H2 (planning under a nuisance shift) has not been run at confirmatory scale.
- H3 (continual retention) has not been run.
- The world is synthetic; the state head and two-step loss are auxiliary and
  shared by all arms.
- The overall H1 verdict is FAIL, not PASS, because of H1b. Do not cite H1 as a
  clean confirmatory pass.

## 4. What to do next

1. **Record H1a/H1c as confirmed, H1b as a protocol failure.** The mechanism
   claim is supported; the full H1 claim is not.
2. **Write PREREG-v3** with relative non-inferiority for H1b:
   `good_state ≥ temporal_state` and `good_state ≥ 0.75`, evaluated on fresh
   seeds 5–9 if a clean confirmation is required.
3. **Run H2** on the saved models: planning under a nuisance shift. This is the
   bridge from representation to control. The models are in
   `runs/h1-confirm/models/`.
4. If H2 passes, the chapter has a representation-to-control result. If H2 fails
   while H1a holds, the result is a dissociation: the latent becomes more
   world-like, but the planner does not benefit. That is still informative.

The honest one-liner:

> Good views did what the theory predicted, by a large margin. The confirmatory
> protocol failed on a mis-specified non-inferiority clause, not on the effect.
