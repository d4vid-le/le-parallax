# H2 confirmatory result — planning under nuisance shift

**Status:** complete, 2026-11-12.
**Protocol:** [`PREREG-H2.md`](PREREG-H2.md), sha256
`2a9ecdff735adb27dd0d81de8ce4898e806eb0101be57f3376f096684385e4fd`.
**Artifacts:** `runs/h2-plan/` (git-ignored).
**Verdict:** **FAIL as pre-registered.**

---

## 1. Result

Five seeds, frozen H1 checkpoints, 10 OOD episodes and 5 ID episodes per seed.

| seed | OOD temporal dist | OOD good dist | diff | OOD temp succ | OOD good succ | ID temp dist | ID good dist |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.298 | 0.385 | +0.087 | 0.20 | 0.10 | 0.190 | 0.076 |
| 1 | 0.349 | 0.242 | −0.107 | 0.00 | 0.30 | 0.185 | 0.144 |
| 2 | 0.257 | 0.284 | +0.027 | 0.10 | 0.10 | 0.217 | 0.207 |
| 3 | 0.340 | 0.268 | −0.072 | 0.10 | 0.10 | 0.109 | 0.219 |
| 4 | 0.436 | 0.296 | −0.140 | 0.00 | 0.30 | 0.170 | 0.171 |

- Mean OOD difference (good − temporal): **−0.041** (95% CI **−0.113 to +0.036**).
- Seeds with good lower OOD distance: **3/5**.
- Mean ID difference: **−0.011**; seed 3 is +0.110.
- Mean OOD success delta: **+0.10** (good 0.18 vs temporal 0.08).
- Random OOD success: **0.34**; random OOD distance: **0.41**.

## 2. What passed, what failed

### H2a — OOD distance: **FAIL**

The endpoint was a mean improvement of at least **0.05** with **4/5** seeds
lower. The run is **−0.041** with **3/5** seeds lower, and the CI includes zero.
It is directionally positive but below the pre-registered bar. This is a
near-miss, not a confirmed effect.

### H2b — ID no-regression: **mean passes, variance does not**

The mean ID difference is **−0.011**, within the +0.05 allowance. But seed 3
regresses by **+0.110** (0.109 → 0.219). The mean hides a real per-seed
regression. Report it; do not claim no-regression as a blanket statement.

### H2c — OOD success: **directionally positive, not a clean win**

Good is **+0.10** on success (0.18 vs 0.08), with seeds 1 and 4 going from 0.0 to
0.3. But seed 0 goes the other way (0.10 vs 0.20), and the sample is 10 episodes
per seed.

## 3. The red flag

**The random policy beats both planners on success** — 0.34 vs 0.08 (temporal)
and 0.18 (good) — while both planners have lower mean distance.

That pattern means the planner reaches the neighborhood but does not finish.
Random occasionally stumbles into the goal; CEM gets close and stalls. The
success metric is not measuring the model's representation cleanly; it is
measuring the planner's ability to close the last 0.08.

This is a planner problem, not necessarily a representation problem. H2 is
underpowered and confounded by a weak planner. It is not a clean dissociation.

## 4. What this means

The H1 result stands:

> Good views make the latent more state-like and less nuisance-like, by a large
> replicated margin.

The H2 result is:

> That better latent does not reliably translate into better planning under a
> nuisance shift at this planner/sample size. The direction is positive on
> distance and success, but below the pre-registered threshold, and the planner
> is worse than random on success.

This is a near-miss with a confound, not a refutation. The correct next move is
to fix the planner before re-testing the control claim.

## 5. What to do next

1. **Diagnose the planner.**
   - Measure state-head error on the OOD episodes for temporal and good models.
   - Run an **oracle planner** with the true state to establish the achievable
     success rate.
   - If the oracle also stalls, the task/planner is the bottleneck, not the
     model.
2. **Improve the planner, then pre-register H2-v2.**
   - More CEM samples/iterations, better terminal cost (position + velocity),
     or a short action sequence with a success bonus.
   - More OOD episodes per seed, and fresh episode seeds.
   - Do not tune on the H2 test episodes.
3. **Do not claim H2.** The control benefit is suggestive, not confirmed.
4. Keep the H1a/H1c result as the chapter's confirmed mechanism result.

The honest one-liner:

> Good views made the latent more world-like. The planner did not cash it in, and
> the planner itself is now the bottleneck.
