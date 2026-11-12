# Collision 1b — adversarial nuisance, and the pipeline that fixes it

**Status:** complete, 2026-11-12.
**Protocols:** [`PREREG-COLLISION-2.md`](PREREG-COLLISION-2.md) (diagnostic and
CNA), [`PREREG-CSA.md`](PREREG-CSA.md) (CSA).
**Artifacts:** `runs/ood-diagnostic/`, `runs/adversarial-pilot/`, `runs/csa/`
(git-ignored). Five seeds unless stated.

---

## 1. OOD diagnostic — the seed-0 pilot was misleading

The first adversarial pilot (seed 0) suggested the good-view encoder was
**worse** than temporal outside its nuisance distribution:

| seed 0 pilot | normal | heldout swap | shifted | extreme |
|---|---:|---:|---:|---:|
| temporal | 0.869 | 0.863 | −0.592 | −0.641 |
| good | 0.929 | 0.932 | **−0.877** | −0.426 |

That result would have killed the invariance claim. It does not replicate.

Five-seed diagnostic, same conditions, no training:

| arm | normal | heldout swap | shifted | extreme |
|---|---:|---:|---:|---:|
| temporal | 0.815 | 0.806 | **−0.840** | **−0.888** |
| good | 0.932 | 0.933 | **−0.049** | **−0.053** |

Paired differences (good − temporal), 5 seeds:

- normal: **+0.117** (95% CI +0.078 to +0.170)
- shifted: **+0.791** (95% CI +0.224 to +1.165)
- extreme: **+0.836** (95% CI +0.509 to +1.029)

The correct reading is two-sided:

1. **Absolute OOD failure is real.** The good-view latent is not linearly
   state-decodable under shifted nuisance: mean shifted R² **−0.049**. The
   invariance learned from normal nuisance does not transfer as a usable latent.
   This is why H2-v4 needed domain randomization; support is a prerequisite.
2. **Relative OOD degradation favors good, not temporal.** Good is **+0.79 R²**
   better than temporal on shifted nuisance, and **+0.84** better on the extreme
   corners. The seed-0 pilot was an outlier and pointed the other way.

**Bet 1b-d verdict: partially falsified.** The absolute drop is ≥0.10 (0.981),
but the second clause — good shifted R² not more than 0.05 above temporal — is
false; good is +0.791 above temporal. The pilot's "good is worse OOD" story is
rejected by the five-seed run.

The methodological lesson is the important one: a one-seed collision almost
produced the wrong conclusion. The five-seed paired diagnostic reversed it.

---

## 2. CNA-v1 — the first pipeline failed

Counterfactual Nuisance Alignment (CNA-v1) added counterfactual views but **no
shifted temporal support**, and used a strong latent-MSE alignment. Seed-0 pilot:

| stage | normal | heldout swap | shifted | extreme |
|---|---:|---:|---:|---:|
| frozen good | 0.929 | 0.932 | −0.877 | −0.426 |
| normal-ft | 0.931 | 0.934 | −0.679 | −0.295 |
| mix-ft | 0.908 | 0.915 | **0.781** | **0.433** |
| CNA-v1 | **0.739** | 0.706 | 0.282 | 0.102 |

CNA-v1 improved shifted R² over frozen by +1.159, but:

- normal R² fell by **0.190**, far beyond the 0.03 budget;
- it lost to the ordinary mixed-domain control by **−0.498** shifted and
  **−0.331** extreme.

The latent-MSE alignment collapsed state-relevant dimensions, and the pipeline
never saw shifted transitions, so it could not encode state under the shift.
This was a pilot, not a five-seed run: the design was superseded by CSA before
the confirmatory budget was spent. The failure is large enough to be diagnostic,
not noise.

---

## 3. CSA — support first, then alignment

Counterfactual State Alignment (CSA) changes both ingredients:

- **support:** the same 40 normal + 40 shifted episodes as `mix-ft`;
- **alignment:** cross-domain counterfactual pairs (same state, normal vs
  shifted nuisance) with a **state-head** consistency loss (weight 0.5) and only
  a weak latent term (weight 0.1).

Five seeds:

| stage | normal | heldout swap | shifted | extreme |
|---|---:|---:|---:|---:|
| frozen good | 0.932 | 0.933 | −0.049 | −0.053 |
| mix-ft | 0.924 | 0.926 | 0.728 | 0.328 |
| **CSA** | **0.927** | **0.927** | **0.828** | **0.471** |

Paired differences:

- **CSA − mix-ft shifted: +0.101** (95% CI +0.050 to +0.180) — endpoint holds.
- **frozen − CSA normal: +0.005** (95% CI −0.000 to +0.011) — normal retained.
- **CSA − mix-ft extreme: +0.143** (95% CI +0.077 to +0.209) — secondary
  endpoint holds.

Per-seed shifted R²:

| seed | mix-ft | CSA | diff |
|---:|---:|---:|---:|
| 0 | 0.781 | 0.861 | +0.080 |
| 1 | 0.830 | 0.865 | +0.035 |
| 2 | 0.813 | 0.871 | +0.058 |
| 3 | 0.777 | 0.850 | +0.073 |
| 4 | 0.438 | 0.694 | +0.256 |

CSA wins on all five seeds. The seed-4 gain is the largest, which is notable:
the seed with the worst mixed-domain control benefits most from alignment.

**Verdict: CSA beats domain exposure.** The new pipeline is not a hyperparameter
tweak; it changes what the model is asked to align (state predictions, not raw
latents) and gives it the support it needs.

---

## 4. What this changes

- The good-view mechanism is real in-distribution and degrades more gracefully
  than temporal OOD, but it is **not** a zero-shot OOD invariance. Absolute OOD
  state readability is near zero without adaptation.
- **Support is necessary; alignment is not sufficient without it.** CNA-v1
  failed because it aligned without support. CSA succeeded because it did both.
- The seed-0 pilot's "good is worse OOD" claim is rejected. It is reported here
  as a collision that corrected itself under five seeds.
- The paper's claim should be: good views buy an in-distribution state geometry
  that degrades gracefully; turning that into usable OOD behavior requires
  support plus state-level alignment.
