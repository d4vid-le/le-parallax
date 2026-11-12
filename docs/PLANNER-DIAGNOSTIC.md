# Planner diagnostic — why H2 is confounded

**Status:** exploratory diagnostic, 2026-11-12.
**Episodes:** the exact H2 OOD episodes (seeds `1000 + seed`, 10 each, shifted
nuisance).
**Code:** `scripts/planner_diagnostic.py`.

---

## 1. The question

H2 failed. Was the representation bad, or was the planner bad?

## 2. Result

| seed | oracle CEM succ | oracle CEM dist | strong oracle succ | H2 temporal succ | H2 good succ | H2 random succ | state-head R² temporal | state-head R² good |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.40 | 0.156 | 0.10 | 0.20 | 0.10 | 0.30 | −2.887 | −2.828 |
| 1 | 0.30 | 0.188 | 0.00 | 0.00 | 0.30 | 0.20 | −3.066 | −0.209 |
| 2 | 0.20 | 0.203 | 0.20 | 0.10 | 0.10 | 0.50 | −0.930 | +0.012 |
| 3 | 0.30 | 0.204 | 0.00 | 0.10 | 0.10 | 0.40 | −3.305 | −0.944 |
| 4 | 0.20 | 0.167 | 0.20 | 0.00 | 0.30 | 0.30 | −1.659 | −0.068 |

- Oracle CEM, same budget as H2: **0.28** success, **0.184** mean distance.
- Oracle CEM, stronger budget (H=5, pop 256, 10 iters): **0.10** success.
- H2 learned: temporal **0.08**, good **0.18**, random **0.34**.
- State-head OOD position R²: temporal **−2.369**, good **−0.807**.

Then a simple **true-state proportional controller**:

```
action = clip(2 * (goal - position) - 1.0 * velocity, -1, 1)
```

succeeds **100%** on all 50 OOD episodes, mean distance **0.01–0.06**.

A learned state-feedback controller using the saved state head:

| seed | temporal | good |
|---:|---:|---:|
| 0 | 0.10 / 0.467 | 0.10 / 0.505 |
| 1 | 0.00 / 0.495 | 0.00 / 0.300 |

(success / mean distance)

## 3. What this means

Three separate bottlenecks, in order of severity:

1. **CEM is broken for this task.** Even with true dynamics it succeeds only
   28%, below random (34%). A trivial proportional controller succeeds 100%.
   H2 is therefore not measuring the representation; it is measuring CEM's
   ability to optimize a short-horizon terminal cost.
2. **The state head does not generalize OOD.** Position R² is negative for both
   arms. The readout was trained on the normal nuisance distribution; the OOD
   shift is outside its support. The good latent is less bad (−0.807 vs −2.369),
   which is consistent with H1, but neither readout is usable.
3. **The representation result still stands.** H1a/H1c were measured with a
   linear probe on the latent, not through CEM or the state head. Good views
   reduce nuisance sensitivity by ~89% relative, 5/5 seeds, while state probe is
   higher. That result is unaffected by the planner failure.

H2 is **not** a clean dissociation. It is a confounded negative: the controller
and readout are not good enough to test the representation.

## 4. What to do next

1. **Replace CEM for this task.** Use a state-feedback controller:
   `a = clip(k(goal − ŝ_pos) − d·ŝ_vel, −1, 1)`, with `ŝ` from the learned state
   head. The true-state version is a 100% oracle; the learned version isolates
   the readout.
2. **Fix the readout for OOD.** Train the state head (or a ridge probe) with
   nuisance augmentation covering the test range, or train both arms with a
   wider nuisance distribution and test on a held-out range within that support.
   The current OOD shift is too far outside the training support for any readout
   to generalize; that is a distribution-shift problem, not a good-view problem.
3. **Pre-register H2-v2** with the new planner/readout and fresh episodes. The
   primary endpoint stays OOD distance; success is secondary.
4. **Do not claim H2.** The control link is unconfirmed; the mechanism result is
   confirmed.

The honest one-liner:

> The planner failed before the representation could be tested. A true-state
> proportional controller solves the task 100%; CEM solves it 28%; the learned
> readout is negative-R² OOD. H2 is confounded, not a verdict on good views.
