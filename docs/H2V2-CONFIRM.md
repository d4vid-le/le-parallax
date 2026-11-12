# H2-v2 result — state-feedback control with a robust readout

**Status:** complete, 2026-11-12.
**Protocol:** [`PREREG-H2v2.md`](PREREG-H2v2.md), sha256
`36bb6fb9cecb76039c1af73ced0d06d79500ef859b490bfa6b2aedd834e5e6fe`.
**Artifacts:** `runs/h2v2/` (git-ignored).
**Verdict:** **FAIL as pre-registered**, readout-limited.

---

## 1. Result

Five seeds, frozen H1 encoders, ridge readout trained on 5,120 augmented
latents per arm, state-feedback controller.

| seed | OOD temp dist | OOD good dist | diff | OOD temp succ | OOD good succ | ID diff | readout R² temp | readout R² good |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.223 | 0.271 | +0.048 | 0.30 | 0.20 | +0.003 | −0.691 | −0.164 |
| 1 | 0.227 | 0.148 | −0.079 | 0.20 | 0.30 | −0.017 | −0.460 | −0.242 |
| 2 | 0.259 | 0.382 | +0.123 | 0.30 | 0.20 | −0.124 | −0.357 | −0.083 |
| 3 | 0.297 | 0.242 | −0.055 | 0.10 | 0.10 | −0.131 | −1.969 | −0.291 |
| 4 | 0.275 | 0.238 | −0.037 | 0.10 | 0.20 | +0.013 | −2.987 | −0.036 |

- Mean OOD difference: **−0.000** (95% CI −0.061 to +0.073).
- Seeds good lower: **3/5**.
- Mean ID difference: **−0.051** (at the +0.05 boundary).
- OOD readout R²: temporal **−1.293**, good **−0.163**.
- OOD success: temporal **0.20**, good **0.20**.

## 2. What passed, what failed

### H2v2a — OOD distance: **FAIL**

The mean difference is zero, 3/5 seeds lower, CI includes zero. No control
benefit.

### H2v2b — ID no-regression: **borderline**

Mean ID difference is **−0.051**, right at the allowed +0.05. Seed 2 and seed 3
improve by −0.124 and −0.131; seed 0 and 4 are flat. Not a clean pass, not a
clean regression.

### H2v2c — readout: **good is much better, both are bad**

The good-view readout is **−0.163** vs temporal **−1.293** on OOD position R².
That is a large, consistent representation-level advantage. But both are
negative: the readout cannot linearly decode the state under the shifted
nuisance for either arm.

## 3. The bottleneck is now the readout/encoder OOD support

The planner is fixed:

- true-state proportional controller: **100%**;
- learned state-feedback: **20%** for both arms;
- readout R² negative for both.

The good latent carries more state information OOD, but not enough for a linear
readout to recover it. The shifted nuisance is outside the encoders' training
support; the readout was trained on shifted renders, but the encoder maps those
renders poorly.

This is **not** a clean dissociation. It is a readout-limited negative:

> The representation improved in the predicted direction, but the OOD shift is
> too far outside the encoder's support for the readout to turn that improvement
> into control.

## 4. What to do next

1. **Domain-randomize the encoder training.** Train both arms on a wider
   nuisance distribution (normal + shifted range), then test on a held-out shift
   **inside** that support. This is the fair H2-v3. The current shift was never
   in the encoder's training distribution; no readout can be expected to fix
   that.
2. **Try a nonlinear readout as a diagnostic.** An MLP readout on the same
   latents may recover more state OOD. If it does, the linear readout was the
   limit; if not, the encoder is.
3. **Do not claim H2.** The control link is unconfirmed. H1a/H1c remain the
   chapter's confirmed mechanism result.

The honest one-liner:

> The good latent is better OOD, but both readouts are negative and control is a
> tie. The shift is outside the encoder's training support. Domain
> randomization is the next experiment, not another planner tweak.
