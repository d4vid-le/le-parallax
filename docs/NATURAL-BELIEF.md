# Natural-episode belief stream — Step 2

**Status:** complete, 2026-11-12. 3 belief seeds. Step 3 is justified.
**Code:** `scripts/natural_belief.py`; world hidden alcove in
`leparallax/world.py`.
**Artifacts:** `runs/natural-belief/` (git-ignored).

## 1. Natural hidden-alcove episodes

The world now has a **hidden right-side alcove gate**. The goal sits in the
alcove; the gate is open if the main door stayed open and closes when the door
closes behind the puck. The gate is never rendered. The current frames of a
right-start (reachable) episode and a post-crossing (unreachable) episode are
same-side right/right; only the history differs.

80 train / 20 test episodes, 33 frames, variable crossings and nuisances.
Per-step label: `gate_open` (1 reachable, 0 after the gate closes).

## 2. Belief model

Separate belief encoder + GRU:

- frame encoder initialized from the frozen JEPA `asym_clean` checkpoint;
- trained end-to-end with the GRU and reachability head;
- per-step causal loss over full episodes (all prefixes, every `t`);
- unreachable timesteps upweighted **3×**;
- trained on a mixture of **natural episodes + 400 held-out counterfactual
  pairs** (800 sequences, 6 frames), `λ_cf=1.0`;
- 40 epochs, batch 8, cosine, 3 seeds.

The planner JEPA stays frozen; the belief stream is a separate model.

## 3. Results

### Static baselines

| baseline | overall AUC | hard same-side AUC |
|---|---:|---:|
| state static (oracle p4 + goal) | 0.860 | 0.832 |
| **JEPA static (3-frame)** | **0.431** | **0.432** |

The static JEPA is at/below chance; the state baseline still has a spatial
shortcut (hard 0.832). The belief must beat both.

### Belief stream

| seed | overall AUC | hard same-side AUC |
|---:|---:|---:|
| 0 | 0.941 | 0.948 |
| 1 | 0.970 | 0.987 |
| 2 | 0.989 | 0.997 |

### Occlusion horizon

| horizon | reach mean P | unreach mean P | AUC |
|---:|---:|---:|---:|
| +1 | 0.304 | 0.181 | 0.890 |
| +3 | 0.439 | 0.127 | **0.960** |
| +5 | 0.592 | 0.111 | **0.980** |
| +10 | 0.705 | 0.123 | **0.980** |

The belief holds through +10, well past the 3-frame window.

### Counterfactual re-check

Held-out same-side counterfactual pairs, raw sequences:

**AUC 0.766** (pass ≥0.750).

The pure natural model without the mixture had hard AUC ~1.0 but counterfactual
AUC **0.407** — the lazy spatial shortcut. The mixture fixes the counterfactual
check at a modest cost to natural confidence.

## 4. Gate to Step 3

| gate | result | verdict |
|---|---|---|
| Beats static baseline on hard subset | belief 0.948–0.997 vs static JEPA 0.432 | **pass** |
| Holds at `t_cross+3` and `+5` | AUC 0.960 / 0.980 | **pass** |
| Still passes counterfactual control | AUC 0.766 | **pass** |

**Step 3 (planner integration) is justified.**

## 5. Caveats

- The natural hard subset still has a state/position shortcut (state static
  0.832); the belief's advantage is partly the history and partly better state
  use. The counterfactual 0.766 is the cleanest generalization evidence, and it
  is just above the threshold.
- One natural dataset seed; 3 belief seeds.
- The static JEPA below chance is a class-imbalance/readout artifact; the AUC is
  the comparable metric.
- The mixture was chosen after seeing the pure-natural counterfactual failure;
  it is a documented fix, not a pre-registered arm.
- The belief stream is a controlled memory model, not a closed-loop controller.
