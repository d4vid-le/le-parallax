# Dual-stream pilot — state solved, asymmetry not

**Status:** complete, 2026-11-12. One seed, 40 epochs. No five-seed matrix.
**Code:** `scripts/oracle_state_cnn.py`, `scripts/asymmetry_collision.py`,
`scripts/dual_stream_planner.py`.
**Artifacts:** `runs/dual-stream-state/`, `runs/dual-stream-jepa/`,
`runs/dual-stream-planner/` (git-ignored).

## 1. Design

Two independent streams:

- **State stream:** the frozen 47k oracle CNN, trained only on state regression
  with zero gradient from the JEPA/pair objectives.
- **JEPA stream:** the 142k JEPA backbone trained on temporal + pair loss with
  **no state loss** and no frame state head. The pair arms are `sym_head` and
  `asym_clean`, 40 epochs, `λ_pair=0.10`, cosine, 40/20 episodes.
- **Planner:** candidate actions are rolled out through a state dynamics MLP and
  the JEPA predictor. The score combines the state goal distance and the JEPA
  latent goal distance, normalized on reachable starts; abort threshold = p90 of
  reachable scores. Starts with the crossing in the 3-frame history so the door
  is identifiable.

## 2. Results

### State guardrail

| stream | 4-D state R² | guardrail 0.90×oracle |
|---|---:|---:|
| state stream (47k CNN) | **0.936** | **0.842** — pass |

The state stream does exactly what the dual-stream design promised.

### JEPA direction

| arm | Δasym | state R² | reach AUC | door AUC |
|---|---:|---:|---:|---:|
| `sym_head` | −0.003 | 0.148 | 0.683 | 0.771 |
| `asym_clean` | +0.014 | 0.157 | 0.728 | 0.772 |

Direction margin: **+0.018**, far below the **+0.100** trigger. Freeing the JEPA
from state-retention gradients did **not** unlock the predicted direction.

### Planner

| scorer | abort unreach | false abort | AUC |
|---|---:|---:|---:|
| state-only | 0.33 | 0.17 | **0.611** |
| state + `sym_head` | 0.33 | 0.17 | 0.500 |
| state + `asym_clean` | 0.33 | 0.17 | 0.500 |

The combined planner is **worse** than state-only. The JEPA latent goal distance
adds noise, not irreversibility. The JEPA latent does carry some door/reach
signal (AUC 0.77/0.73), but it does not translate into a better abort decision.

## 3. Verdict

**The dual stream solves the state bottleneck, not the asymmetry bottleneck.**

- State stream: **pass** (0.936 vs 0.842). The state path is decoupled and
  healthy.
- JEPA direction: **fail** (+0.018 vs +0.100). The asymmetric pair objective did
  not form a stronger directed latent once the state loss was removed.
- Planner: **fail** (0.500 vs state-only 0.611). The JEPA distance is not a
  usable irreversibility signal in this planner.

The five-seed matrix is not justified. The next question is not state capacity;
it is whether the pair objective can produce a planner-usable irreversibility
signal at all. A reachability classifier on the JEPA latent is a cheaper next
test than another end-to-end matrix.

## 4. Caveats

- One seed, 40 epochs, 142k JEPA; not confirmatory.
- The planner starts keep the crossing in the 3-frame history so the door is
  identifiable. A fully occluded B at `t_cross+3/5` is not identifiable from the
  current history; that is a separate belief-state problem.
- The state dynamics MLP is trained on true states and does not model the door.
  The state-only planner is therefore not expected to abort; the JEPA stream is
  supposed to supply that signal.
- The planner is a latent shooting diagnostic, not closed-loop MPC in the true
  environment.
