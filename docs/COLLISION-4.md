# Collision 4 — Anti-Equivalence on a one-way door

**Status:** complete, 2026-11-12. Five seeds.
**Protocol:** [`PREREG-ASYMMETRY.md`](PREREG-ASYMMETRY.md), sha256
`e1fc79ff68fb92d9a2a5552158ea1c1a17e89de90178d97e15a43302b275eea1`.
**Artifacts:** `runs/asymmetry-pilot/`, `runs/asymmetry-confirm/`,
`runs/asymmetry-confirm2/` (git-ignored).

The Anti-Equivalence hypothesis says symmetric pair losses destroy causal
direction on irreversible transitions. The latent should be a directed map:
`A→B` easy, `B→A` hard, with `z_B` lower-volume than `z_A`. We built a genuine
irreversible event — a one-way door — and tested it.

The five-seed result **falsifies the hypothesis as operationalized here**, and
it exposes a collapse trap in the proposed determinant penalty.

---

## 1. Two corrections before running

1. **The sign in the proposed loss is backwards.**
   `−λ·det(Cov(z_B))` *maximizes* the determinant. Contraction needs
   `+λ·det(...)`. We test the corrected sign and a wrong-sign control.
2. **A determinant penalty can win by collapsing `z_B`.** MLX has no GPU
   determinant, so the training penalty is a diagonal log-volume proxy
   `mean(log(var(z_B)+ε))`; the true effective rank is measured post hoc. A
   state-probe floor and a wrong-sign control are mandatory.

Two evaluation controls were also necessary:

- **Matched reversible control.** Irreversible pairs straddle the crossing;
  reversible pairs are both on the right, door closed, with the same time gaps
  (4/6/8). Without this, the time gap and state distribution confound the
  forward/reverse asymmetry.
- **Goal-conditioned reachability probe.** `goal_reachable` depends on the goal
  side, which is not in the puck latent. The probe uses the joint state+goal
  latent, not the state latent alone.

## 2. World and data

A one-way door at `x=0.5`, gap `y∈[0.35,0.65]`. The puck starts left; the first
left→right crossing closes the door permanently. The goal can be on either side;
a left goal becomes unreachable after crossing. Observations are 96px and show
the door.

40 train / 20 test door episodes, 32 steps, goal-directed 0.5. Irreversible
pairs: `A=t_cross−{2,4}`, `B=t_cross+{2,4}`, door open at A and closed at B.
Reversible pairs: both states on the right, door closed, gaps 4/6/8.

Arms (20 epochs, 5 warmup, `λ_pair=0.25`, `λ_state=0.25`):

| arm | pair loss | reverse path | volume |
|---|---|---|---|
| temporal | none | — | — |
| symmetric | `‖z_A−z_B‖²` | symmetric | — |
| asym | forward `g` + reverse `h` | bottleneck 32 | 0 |
| asym_vol | forward `g` + reverse `h` | bottleneck 32 | `+0.05·log-volume(z_B)` |
| asym_wrong | forward `g` + reverse `h` | bottleneck 32 | `−0.05·log-volume(z_B)` |

## 3. Seed-0 pilot

| arm | Δasym | ΔΔrank | state R² | reach AUC | latent std | norm A/B | forward R² | reverse R² |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| temporal | −0.033 | −0.157 | 0.332 | 0.217 | 0.901 | 7.64/7.63 | 0.960 | 0.961 |
| symmetric | −0.067 | −0.122 | 0.444 | 0.565 | 0.915 | 7.63/7.63 | 0.970 | 0.972 |
| asym | −0.126 | −0.121 | 0.507 | 0.770 | 0.885 | 7.34/7.34 | 0.971 | 0.975 |
| asym_vol | **+12.409** | **−0.418** | 0.551 | 0.652 | **0.604** | **5.35/5.10** | **0.204** | **−8.896** |
| asym_wrong | −0.122 | −0.090 | 0.533 | **0.848** | 1.000 | 8.31/8.32 | 0.978 | 0.981 |

`Δasym = asym_ratio(irr) − asym_ratio(rev)`. A positive value is the
Anti-Equivalence prediction.

### Five-seed aggregate

| arm | Δasym | ΔΔrank | state R² | reach AUC | latent std | forward R² | reverse R² |
|---|---:|---:|---:|---:|---:|---:|---:|
| symmetric | **−0.013** | −0.047 | 0.327 | 0.661 | 0.911 | 0.934 | 0.936 |
| asym | **−0.115** | −0.028 | 0.485 | 0.506 | 0.882 | 0.911 | 0.921 |
| asym_vol | **+7.324** | −0.236 | 0.437 | 0.492 | **0.599** | **0.184** | **−5.078** |
| asym_wrong | −0.058 | −0.014 | 0.465 | **0.670** | 1.005 | 0.923 | 0.928 |

Paired direction difference `asym − symmetric` on `Δasym`:

| seed | symmetric | asym | asym − symmetric |
|---:|---:|---:|---:|
| 0 | −0.067 | −0.126 | **−0.059** |
| 1 | −0.006 | −0.393 | **−0.387** |
| 2 | +0.132 | +0.093 | **−0.039** |
| 3 | −0.006 | −0.172 | **−0.166** |
| 4 | −0.118 | +0.023 | **+0.141** |

The asymmetric arm is lower on **4/5 seeds** and higher on **1/5**; mean paired
difference **−0.102**. There is no consistent directed advantage. The volume
collapse is 5/5: forward R² 0.087–0.311, reverse R² −0.71 to −10.84, latent std
0.517–0.714 vs symmetric 0.890–0.932. The state probe survives on most seeds
(0.121–0.556), so this is a collapsed **pair geometry**, not total state
collapse. Seed 3's symmetric arm is an outlier (state R² 0.019); it is reported,
not dropped.

## 4. What the result shows

1. **The direction bet fails, but not as a simple law.** The asymmetric arm's
   mean `Δasym` is **−0.115** vs symmetric **−0.013**; it is lower on **4/5
   seeds** and higher on **1/5** (paired differences −0.059, −0.387, −0.039,
   −0.166, **+0.141**). There is no consistent directed advantage and no
   positive effect of the predicted size.
2. **The volume arm's huge asymmetry is collapse.** `asym_vol` has mean
   `Δasym=+7.32`, but forward R² is **0.184** and reverse R² is **−5.078**.
   Both directions are bad; the reverse is catastrophic. The latent std falls
   from 0.911 to **0.599**. This is global shrinkage, not a causal slice. The
   state probe survives on most seeds (mean 0.437, range 0.121–0.556), so it is
   not total state collapse — but the pair geometry is degenerate.
3. **The wrong-sign control kills the reachability story.** `asym_wrong` has the
   **highest mean reach AUC (0.670)**, the largest latent (std 1.005), and no
   rank contraction. The volume arm is 0.492 and the asym arm 0.506. The
   reachability gain is not caused by the corrected volume sign or the directed
   geometry; it is unstable and appears in the wrong-sign control.
4. **The matched control reverses the raw result.** With the old gap-2 reversible
   pairs, the temporal arm had `Δasym=+0.129`; with gaps matched to 4/6/8 it is
   **−0.033**. The apparent direction effect was a time-gap confound.
5. **The door is observable.** The temporal model can encode the irreversible
   door state from the image. An asymmetric loss is not necessary to represent
   the irreversibility; the latent already has the information. The hypothesis
   needs an observation where the information is actually lost or a
   counterfactual where only the future potential changes.

## 5. Verdict

**Not supported across five seeds.** In this observable one-way-door world:

- asymmetric prediction does not produce the predicted `A→B` advantage;
- the determinant/log-volume penalty contracts by shrinking the latent, not by
  learning a causal slice;
- the reachability improvement is not specific to the asymmetric mechanism;
- the matched control removes the apparent direction effect.

The honest summary is the one the user asked for: the Anti-Equivalence idea is
interesting, but **the proposed loss is a collapse knob**, and on a visible
irreversible event the temporal model already has the information. A fair next
test needs an occlusion or a same-state counterfactual door open/closed pair,
where the lost information is real, plus a **sufficiency constraint** that
prevents the volume term from shrinking the state away.

## 6. Caveats

- Five seeds, 20 epochs, 40/20 episodes. State R² is low (0.33 mean symmetric,
  0.49 asym) relative to H1 (0.8+); the door is a harder observation and 20
  epochs is short. Seed 3's symmetric arm collapsed (state R² 0.019); it is
  reported, not dropped.
- The volume penalty is a diagonal proxy, not the full determinant.
- Reachability is a linear probe, not a closed-loop planner.
- The one-way door is visible, so the irreversibility is observable; this is a
  test of future-potential asymmetry, not information loss.
- The asymmetric arms have extra pair heads; the symmetric arm does not. The
  reachability comparison is therefore not a pure loss ablation. The wrong-sign
  control shares the heads and gets the same reachability AUC, which suggests
  the heads / pair state loss, not the direction, drive that gain. A full-rank
  `symmetric_heads` arm is the missing control.
