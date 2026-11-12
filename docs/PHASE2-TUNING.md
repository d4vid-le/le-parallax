# Phase 2 tuning loop — lower pair pressure + pre-merge state head

**Status:** complete, 2026-11-12. The 10-epoch sweep and the 40-epoch
frame-state pilot are done. The five-seed confirmatory run is **not** started.

## 1. What was tried

The pre-confirmatory loop:

- `λ_pair`: 0.25 → **0.08**
- variance-floor weight: 1.0 → **0.2**
- a dedicated **pre-merge frame state head**: 128-d current-frame keypoint
  features → 4-d state, with its own auxiliary loss, before the history merge
  and before the asymmetric pair heads.

## 2. 10-epoch sweep (seed 0)

| arm | Δasym | state R² | latent std | reach AUC |
|---|---:|---:|---:|---:|
| temp_matched | −0.015 | **0.284** | 0.894 | 0.642 |
| sym_head | **+0.143** | 0.380 | 0.893 | 0.737 |
| asym_clean | **+0.064** | 0.342 | 0.892 | 0.805 |
| asym_vol_bounded | **+0.015** | 0.371 | 0.936 | 0.568 |

What this says:

1. **Lower pair pressure helps state retention.** `sym_head` 0.380 vs 0.223 at
   40 epochs with λ_pair=0.25; `asym_clean` 0.342 vs 0.308. The direction is
   right, but 10 epochs is far too short for the 0.80 guardrail: the temporal
   baseline is **0.284**.
2. **The Δasym metric is not stable.** At the lower weight, `sym_head` has the
   *larger* direction signal (**+0.143**) than `asym_clean` (**+0.064**). The
   previous 40-epoch run had `asym_clean` +0.146 vs `sym_head` −0.055. The
   sign/ordering depends on the training budget and pair weight, so a single
   pilot cannot establish Anti-Equivalence.
3. **The variance floor works.** `asym_vol_bounded` keeps std **0.936**, close
   to the 0.90 target, but its direction signal collapses to +0.015.

The requested 10-epoch target — temporal state R² ≥0.80 — is not reachable at
this model scale. The temporal arm has no pair loss or volume term, so lowering
those weights cannot move it. At 40 epochs it was 0.574. The guardrail is a
**baseline capacity/readout** problem, not a pair-pressure problem.

## 3. 40-epoch frame-state pilot

The pilot adds the pre-merge state head and uses λ_pair=0.08, variance weight
0.2.

| arm | Δasym | state R² | latent std | reach AUC | planner AUC |
|---|---:|---:|---:|---:|---:|
| temp_matched | −0.115 | **0.638** | 0.911 | 0.836 | 0.722 |
| sym_head | −0.019 | **0.670** | 0.903 | 0.704 | 0.694 |
| asym_clean | **+0.080** | **0.671** | 0.901 | 0.521 | 0.694 |
| asym_vol_bounded | **+0.009** | **0.695** | **0.968** | 0.933 | **0.750** |

What this says:

1. **The frame state head is the real fix.** Pair-arm state R² rises from
   **0.223 / 0.308** (40 epochs, no frame head, λ_pair=0.25) to **0.670 /
   0.671 / 0.695**. The state cost is largely gone; the dedicated pre-merge
   state path does what the shared latent could not.
2. **The direction signal survives, barely.** `asym_clean` is **+0.080** vs
   `sym_head` **−0.019**, a margin of **+0.099** — just under the pre-registered
   +0.10 bar. This is the first balanced result: state retained and direction
   positive.
3. **The bounded-volume arm trades direction for safety.** It has the best
   state R² (**0.695**), std **0.968**, and planner AUC (**0.750**), but its
   direction signal collapses to **+0.009**. The variance floor is strong
   enough to keep the latent healthy; the volume term is no longer shaping the
   direction.
4. **The planner is still weak.** `asym_clean` does not beat `sym_head`
   (0.694 vs 0.694). `asym_vol_bounded` gains +0.056 AUC but the abort rate is
   unchanged (0.67 vs 0.17 false), so the planner threshold is coarse.
5. **The state guardrail still fails.** Best state R² is **0.695**, below 0.80.
   The five-seed confirmatory run is not justified yet.

## 4. What counts as a pass

- temporal state R² ≥0.80 (or a documented capacity/readout fix that reaches
  it), and
- `asym_clean` Δasym >0 and ≥ `sym_head` +0.10 across seeds, and
- the bounded-volume arm std ≥0.90 with no state collapse, and
- a planner AUC gain over `sym_head`.

Current result: **not a pass.** State R² 0.695 <0.80; direction margin +0.099 is
just under +0.10; planner is flat. The tuning loop is close, but this is not the
confirmatory signal.

## 5. Next tuning move

1. Train 60–80 epochs; the state R² is still climbing at 40.
2. Increase the latent to 128-d or the conv channels; the 142k model is the
   likely ceiling.
3. Try `λ_pair=0.10–0.12` to push the direction margin while keeping the frame
   state head.
4. Lower the variance-floor weight slightly (0.1) if the bounded arm's direction
   stays collapsed.
5. Re-run the 1-seed pilot; start the five-seed matrix only when state R² ≥0.80
   and the direction margin clears +0.10.
