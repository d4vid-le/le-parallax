# Pre-registration — Collision 1c: Counterfactual State Alignment (CSA)

**Status:** **FROZEN 2026-11-12.**
**Motivation:** CNA-v1 (`PREREG-COLLISION-2.md`) failed. It improved shifted R²
from −0.877 to +0.282, but normal R² fell from 0.929 to 0.739, and it lost badly
to the ordinary mixed-domain fine-tune control (shifted 0.781). Two design
errors are visible:

1. CNA-v1 had **no shifted temporal support**. Its counterfactual views taught
   nuisance invariance but not how to encode state under the shifted nuisance.
2. Its **latent MSE alignment** was too strong. Pulling latents together
   collapses state-relevant dimensions; the normal probe degrades.

CSA changes the pipeline, not just the weight:

- train on **normal + shifted temporal transitions** (support), and
- align **state-head predictions** across counterfactual views, with only a weak
  latent term to preserve geometry.

---

## Data pipeline

For 40 normal and 40 shifted training episodes:

- standard temporal samples for the JEPA loss (same budget as `mix-ft`);
- for 8 states per episode, a **cross-domain counterfactual pair**:
  - a normal-domain state rendered with a shifted nuisance, and
  - a shifted-domain state rendered with a normal nuisance.
  The state label is fixed; only nuisance changes.

## Alignment loss

Fine-tune the H1 good encoder for 5 epochs with:

- the standard temporal JEPA loss on the mixed transitions,
- state-head MSE on both counterfactual views (weight 0.5),
- weak latent MSE between the two views (weight 0.1),
- SIGReg, variance, and covariance on both views.

## Arms

- `frozen`: H1 good encoder, no adaptation.
- `mix-ft`: 5 epochs of the standard temporal loss on the same 40 normal + 40
  shifted episodes, no counterfactual pairs. This is the strong control from
  CNA-v1 and the arm CSA must beat.
- `csa`: the pipeline above.

Five seeds, 0–4. Same evaluation conditions as Collision 1b: normal,
heldout-swap, shifted, extreme.

## Endpoint

CSA must satisfy all three:

1. **Shifted gain:** paired mean `CSA_shifted − mix_shifted` is **≥0.03**, with a
   bootstrap 95% CI lower bound **>0**.
2. **Normal retention:** paired mean `frozen_normal − CSA_normal` is **≤0.03**.
3. **Extreme gain (secondary):** paired mean `CSA_extreme − mix_extreme` is
   **≥0.03**; report but do not gate the primary verdict on it.

If CSA fails condition 1 or 2, the counterfactual alignment adds nothing beyond
domain exposure, and the paper says so. A one-seed pilot may be run for
optimization only; it is disclosed and excluded from the confirmatory count if
the code changes.

## What this does not establish

- That CSA generalizes to unseen nuisance families.
- That state-head alignment is the causal ingredient (no ablation of the state
  term alone).
- Anything about control or continual retention; this is a representation test.
