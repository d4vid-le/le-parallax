# Closed-loop abort gates — Step 3 first pass

**Status:** first pass complete, 2026-11-12. **Not a clean pass**; five-seed
matrix not started.
**Code:** `scripts/closed_loop_gate.py`,
`scripts/planner_belief_train.py`.
**Artifacts:** `runs/closed-loop-gate/`,
`runs/closed-loop-gate-planner-belief/` (git-ignored).

## 1. Setup

Base planner: frozen state stream + state-feedback controller on the natural
hidden-alcove world. The gate never steers; it only aborts. Labels are the
**actual gate status at the end**, not the start side.

- Calibration: 160 episodes → 99 gate-open (reachable), 61 gate-closed.
- Test: 40 episodes → 28 gate-open, 12 gate-closed.
- Base reachable success: **0.75** (some open-gate episodes fail because the
  controller cannot thread the door gap; this is a planner-path issue, not a
  gate false abort).
- Gates: state-only, frozen-JEPA, belief-stream.
- Threshold selection: false-abort ceiling ≤5% on calibration, then maximize
  unreachable abort, then minimize time-to-abort.
- Time-to-abort is measured **after the gate closes**; step savings are
  unreachable-only.

## 2. Test results

| arm | reach success | false abort | unreach abort | abort delay after close | step savings |
|---|---:|---:|---:|---:|---:|
| base (no gate) | 0.75 | — | 0.00 | — | 0 |
| state-only | 0.75 | **0.00** | **1.00** | 3.17 | 27.3 |
| frozen JEPA | 0.75 | 0.00 | 0.25 | 3.67 | 26.7 |
| belief, natural-trained | 0.75 | **0.18** | 1.00 | 1.25 | 29.3 |
| **belief, planner-trained** | 0.75 | **0.071** | **1.00** | **0.83** | **29.7** |

The planner-trained belief stream cuts false aborts from 18% to **7.1% (2/28)**
and aborts every unreachable episode **0.83 steps after the gate closes**. The
calibration false abort was **1.0% (1/99)**.

## 3. The label bug

The first closed-loop pass labeled “unreachable” by **start side**. Only 23% of
left-start rollouts actually close the gate; the rest stay open and are
reachable. That made the belief gate look falsely safe. Re-labeling by the
**actual final gate status** exposed the 18% false-abort rate on open-gate
reachable episodes. The planner-distribution retrain is the fix.

## 4. The remaining problem

- **7.1% is above the 5% ceiling.** The test has only 28 reachable episodes;
  2/28 is one episode over the ceiling. The calibration (1/99) is under. The
  test is too small to resolve this cleanly.
- **The state gate looks better on natural** (0/28 false, 100% abort), but it is
  probably a spatial shortcut: it sees puck position/velocity and the gate-open
  vs gate-closed trajectories differ. The natural closed-loop does **not** match
  the current state across conditions.
- **The matched counterfactual closed-loop is missing.** We need episodes where
  the current puck/goal frames are identical and only the gate history differs.
  The state gate should collapse there; the belief gate should hold. That is the
  test that decides Step 3.
- **Base reachable success is 0.75**, not 1.0. The state-feedback controller
  fails some open-gate episodes at the door gap. The gate arms do not reduce the
  base success, but the planner itself needs work.

## 5. Verdict

**Step 3 is not cleanly passed.** The belief gate is the best
unreachable-abort/time arm and the planner-distribution retrain is the right
fix, but the held-out false-abort rate is 7.1% against a 5% ceiling, and the
state gate’s natural result is confounded by spatial shortcuts. Do not run the
five-seed matrix yet.

## 6. Next

1. Run the **matched counterfactual closed-loop**: identical current frames,
   gate open vs closed, history differs. Confirm the state gate collapses and
   the belief gate holds.
2. Enlarge the held-out reachable test to **100+ episodes** (or rotate natural
   dataset seeds); 2/28 is one episode from the ceiling.
3. Ensemble the 3 planner-trained belief seeds and re-calibrate; the current
   arm uses one seed.
4. Fix or bound the base controller’s 0.75 reachable success before claiming
   closed-loop efficiency.
5. Only then run the five-seed final matrix.

## Caveats

- One planner-trained belief seed; the natural-trained model is the previous
  seed-1 model.
- The threshold is calibrated on 99 reachable episodes; the test has 28.
- The gate status is hidden; a manual lock with no history is unobservable to
  every gate.
- The closed-loop controller is a state-feedback heuristic, not a full MPC.
