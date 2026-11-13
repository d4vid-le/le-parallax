# Hypothesis B — recurrent belief-state JEPA

**Status:** complete, 2026-11-12. Controlled counterfactual; 3 seeds.
**Code:** `scripts/counterfactual_belief.py`,
`scripts/counterfactual_gru.py`.
**Artifacts:** `runs/counterfactual-belief/`,
`runs/counterfactual-gru/` (git-ignored).

## 1. The counterfactual

The current one-way door cannot make a right/right goal unreachable. We added a
hidden right-side **alcove gate**: the goal sits in the right alcove. If the door
stayed open, the gate is open (reachable); if the puck crossed a closing door,
the gate is locked (unreachable). The gate is not rendered.

Each pair uses the **same puck, goal, nuisance, and last three frames**. Only the
earlier history differs:

- **Open:** right-side approach, door open, alcove open.
- **Closed:** left-side frame, crossing, door closes; same three right-side
  current frames, alcove locked.

The last-three-frame difference is **0.00**. Static 3-frame encoders see
identical inputs and must be at chance; only a recurrent state can remember the
crossing.

## 2. Static control check

| model | history | seed 0 | seed 1 | seed 2 |
|---|---|---:|---:|---:|
| state baseline (47k) | static 3-frame | 0.497 | 0.497 | 0.497 |
| JEPA static (142k) | static 3-frame | 0.475 | 0.497 | 0.391 |

Both static baselines are at chance. The side+goal shortcut is dead.

## 3. Frozen-feature GRU

A GRU over the **frozen** JEPA frame features:

| seed | AUC |
|---:|---:|
| 0 | 0.515 |
| 1 | 0.516 |
| 2 | 0.515 |

At chance. The frozen JEPA frame features do not carry the crossing/door history
in a form the GRU can use.

## 4. End-to-end JEPA + GRU

Fine-tune the JEPA frame encoder + GRU + reachability head for 40 epochs on the
counterfactual sequences:

| seed | AUC |
|---:|---:|
| 0 | **0.999** |
| 1 | **1.000** |
| 2 | **1.000** |

Pass criterion: AUC ≥0.750. **Pass.**

## 5. Verdict

**Hypothesis B is supported in the controlled counterfactual.**

- Static state/JEPA: chance, as designed.
- Frozen JEPA features + GRU: chance; the existing backbone does not carry the
  history.
- End-to-end JEPA + GRU: **0.999–1.000**; the belief state remembers the
  crossing and predicts reachability.

The result is a **memory** result, not a general irreversibility/control result.
The counterfactual label is a deterministic history feature (a crossing
followed by the matched current state); the GRU can learn it. The static
baselines confirm the test is honest.

## 5b. Sanity checks and architecture decision

See [`SANITY-CHECKS.md`](SANITY-CHECKS.md).

- **Spatial/dynamics:** the reachability fine-tune improved state R²
  (0.157 → 0.442) but made one-step JEPA dynamics **6.1× worse**
  (0.0179 → 0.1097). It is not safe as the planner's JEPA encoder.
- **Door cue:** at x=0.45 with the door fully visible, the frozen JEPA linear
  probe is **0.524** (single frame) / **0.529** (3-frame). The frozen features
  do not carry the cue; the frozen GRU's 0.515 is explained.
- **Decision:** keep the planner JEPA frozen and train a **separate belief
  encoder + GRU** (Option A). Option B (joint multi-task JEPA dynamics +
  reachability) is the fallback. The natural-episode test must use the separate
  belief stream, not the fine-tuned JEPA.

## 6. What this does not establish

- Natural-episode reachability, where the history is noisy and the crossing time
  varies.
- Closed-loop control or planner improvement.
- A general belief-state JEPA trained on the real task distribution; this is a
  controlled synthetic counterfactual.
- That the frozen JEPA backbone is useless; it is useful with end-to-end
  fine-tuning, but its frozen frame features are not sufficient.
- A five-seed confirmatory matrix; this is a 3-seed pilot.

## 7. Next

1. Train the GRU on **natural episodes** with the same reachability labels and
   evaluate on held-out natural episodes, not synthetic pairs.
2. Integrate the belief state into the dual-stream planner and test
   unreachable-goal aborts.
3. Test the fully occluded `t_cross+3/5` case, where the crossing is outside the
   current 3-frame history and the GRU must carry it longer.
4. Ablate the GRU hidden size and sequence length to see how long the belief
   persists.
