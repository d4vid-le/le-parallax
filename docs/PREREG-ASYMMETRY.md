# Pre-registration — Anti-Equivalence collision

**Status:** **FROZEN 2026-11-12.** One-seed pilot may be run first; it is
disclosed and excluded from the confirmatory count if the code changes.

## Hypothesis

Standard pair losses impose an equivalence relation: if `A` and `B` are the same
state, `z_A ≈ z_B` in both directions. On an **irreversible** transition, that
throws away causal direction: `A` has futures that `B` no longer has. The
Anti-Equivalence hypothesis says the latent should instead be a directed map:

```
z_B = g(z_A)     but     z_A ≠ h(z_B)
```

with `z_B` lower-volume than `z_A`, because `B` is a constrained slice of `A`'s
causal potential.

## Two corrections to the proposed loss

1. **Sign.** `−λ·det(Cov(z_B))` *maximizes* the determinant. Contraction needs
   `+λ·det(...)`. We test the corrected sign and a wrong-sign control.
2. **Collapse.** A determinant penalty can “win” by collapsing `z_B` and
   destroying state information. We require the state probe to survive and
   measure the rank of the **non-state residual**, not just the raw latent.

MLX has no GPU determinant/eigendecomposition, so the training penalty is a
diagonal log-volume proxy:

```
L_vol = mean(log(var(z_B) + ε))
```

The true effective rank and residual rank are computed post hoc in NumPy.

## World

One-way door at `x = 0.5`, gap `y ∈ [0.35, 0.65]`. The puck starts on the left;
the first left→right crossing closes the door permanently. The goal can be on
either side. If the goal is on the left after crossing, it is unreachable.

State is still `[x, y, vx, vy]`; observation is a 96px render showing the door.
Labels per timestep: `door_open`, `side`, `goal_reachable`.

## Data

- 80 train / 20 test door episodes, 32 steps, `goal_directed=0.5`,
  `door_goal_right=0.6`.
- **Irreversible pairs** at the first crossing: `A = t_cross − {2,4}`,
  `B = t_cross + {2,4}`, with `door_open[A]=True`, `door_open[B]=False`.
- **Reversible control pairs** after the door closes: `(t, t+2)` with both
  states on the right and `door_open=False`. These match the B-side state
  distribution and the time gap, but the transition is not irreversible.
- Standard temporal samples for the JEPA loss.
- Temporal warmup: the pair loss is off for the first 5 epochs; then
  `λ_pair=0.25`, `λ_state=0.25`. This prevents the pair term from distorting the
  latent before the temporal model has learned the state.

## Arms

Five seeds for confirmation; one-seed pilot first.

| arm | pair loss | reverse path | volume penalty |
|---|---|---|---|
| `temporal` | none | — | — |
| `symmetric` | `‖z_A − z_B‖²` | symmetric by construction | — |
| `asym` | forward `g` + reverse `h` | **bottleneck 32** | 0 |
| `asym_vol` | forward `g` + reverse `h` | **bottleneck 32** | `+λ·L_vol(z_B)` |
| `asym_wrong` | forward `g` + reverse `h` | bottleneck 32 | `−λ·L_vol(z_B)` |

`g` is 64→128→64. `h` is 64→32→64 for the asymmetric arms. `λ_vol = 0.05`.
The bottleneck is the primary asymmetric mechanism; the volume term is the
secondary test of the rank/volume claim.

## Metrics

- **Direction:** ridge forward `z_A→z_B` and reverse `z_B→z_A` R² on held-out
  pairs; asymmetry ratio `reverse_error / forward_error`, computed separately
  for irreversible and reversible pairs.
- **Difference-in-differences:** `Δasym = asym_ratio(irr) − asym_ratio(rev)`.
  This is the primary direction metric. It controls for ordinary temporal
  asymmetry and the B-side state distribution.
- **Rank:** effective rank of `z_A` and `z_B`; effective rank of the residual
  after projecting out the state-predictive subspace. Contract as
  `Δrank = (rank_B − rank_A)_irr − (rank_B − rank_A)_rev`.
- **State:** ridge state R² on `A`, `B`, and held-out episodes.
- **Irreversibility:** ridge AUC for `door_open`, `side`, and `goal_reachable`
  from `z`.
- **Directional geometry:** mean norm and effective rank of reachable vs
  unreachable states; AUC of `−norm` and `−rank` for unreachable.
- **Planner proxy:** `goal_reachable` AUC is the planner’s feasibility signal.
  A latent-distance planner is optional and exploratory.

## Bets

- **A1 direction:** `Δasym` for `asym` is ≥0.10 and at least 0.10 above
  `symmetric`, in ≥4/5 seeds.
- **A2 rank/volume:** `asym_vol` `Δrank` is ≤−0.10·mean_rank and at least 0.05
  more negative than `symmetric`; state R² drops ≤0.05.
- **A3 reachability:** `asym` or `asym_vol` `goal_reachable` AUC is ≥
  `symmetric` + 0.05.
- **A4 directional geometry:** in the asymmetric arms, unreachable states have
  lower mean norm or rank than reachable states, with AUC ≥0.65.

**Falsifiers.** State collapse (R² drop >0.05), no `Δasym` gain over `symmetric`,
no reachability gain, or the wrong-sign control behaving the same as the
corrected sign. Any of these is a negative result and is reported.

## What this does not establish

- Real-world irreversibility, a general semi-lattice, or control.
- That the determinant/volume term is necessary; the bottleneck may do all the
  work.
- That a planner will avoid irreversible actions without reward. The
  reachability probe is a feasibility signal, not a closed-loop planner.
