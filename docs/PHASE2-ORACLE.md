# Phase 2 oracle ceiling + stop-gradient probe

**Status:** complete, 2026-11-12. Three oracle seeds and three probe seeds.

## 1. The question

The Phase 2 guardrail asked for state R² ≥0.80. Before spending more compute on
the JEPA backbone, we need the **empirical ceiling** of the observation → state
mapping with the same 3-frame history and the same 40/20 episode split. If the
oracle cannot reach 0.80, the guardrail is a ghost hunt.

## 2. Oracle

A standalone **~47k-parameter CNN**:

- conv 3→24→40→40, spatial softmax, 80 keypoint features per frame;
- 3 frames concatenated (240) → 64-d feature → 2-d (`x,y`) and 4-d state heads;
- trained only on state MSE, no JEPA, no pair loss;
- 100 epochs, cosine, AdamW, 40 train / 20 test episodes, 3 seeds.

The oracle uses the same occluded-door observations as Phase 2.

## 3. Stop-gradient auxiliary head

A 2-layer MLP probe (64→256→256→4) trained on the **frozen** JEPA latent. The
encoder is not updated; this is the `sg(z)` auxiliary head, evaluated as a pure
readout. 3 probe seeds, 300 epochs, early stopping on a 10% validation split.
Compared with a linear ridge probe on the same latents.

## 4. Re-evaluated trigger

For each target (x,y and 4-D state):

```
R²_adjusted = 0.90 × R²_oracle
```

The state guardrail passes if the JEPA readout is ≥ `R²_adjusted`. The direction
margin is re-checked with the arms that pass the adjusted state guardrail.

## 5. Stop-gradient MLP result (142k `asym_clean`)

| readout | 4-D state R² | position R² | velocity R² |
|---|---:|---:|---:|
| linear ridge | **0.671** | 0.707 | **0.635** |
| MLP on `sg(z)` | 0.564 | 0.708 | 0.420 |

The MLP does **not** recover additional state. Position is unchanged; velocity is
worse. The latent's state information is essentially linear, so the low state R²
is not a linear-readout limitation. A stop-gradient auxiliary head would not
help.

## 6. Oracle result

Standalone CNN, **46,830 params**, same 3-frame occluded observations, 3 seeds:

| seed | x,y R² | 4-D state R² | position R² | velocity R² |
|---:|---:|---:|---:|---:|
| 0 | 0.986 | 0.936 | 0.985 | 0.886 |
| 1 | 0.988 | 0.936 | 0.987 | 0.886 |
| 2 | 0.960 | 0.933 | 0.978 | 0.888 |
| **mean** | **0.978** | **0.935** | **0.984** | **0.887** |

The observation contains the state. The JEPA's **0.671** is far below the
oracle. The 0.80 guardrail is **not** a ghost hunt.

## 7. Re-evaluated trigger

```
0.90 × 0.935 = 0.842   (4-D state)
0.90 × 0.978 = 0.880   (x,y)
```

The adjusted state guardrail is **higher** than 0.80, not lower.

| iteration / arm | state R² | adjusted pass? | direction margin |
|---|---:|---|---:|
| 142k frame-head, `asym_clean` | 0.671 | **no** (0.842) | +0.099 |
| 142k frame-head, `asym_vol_bounded` | 0.695 | **no** | — |
| 340k, `asym_clean` | 0.662 | **no** | −0.010 |
| 340k, `asym_vol_bounded` | 0.623 | **no** | — |

Both the state and direction triggers fail. The state gap is 0.171 below the
oracle-relative guardrail.

## 8. Verdict

**The guardrail is real, not a ceiling artifact.**

- The raw observation supports **0.935** 4D state R² with a 47k CNN.
- The JEPA reaches **0.671**; the shared encoder/merge/pair objective is losing
  state information that the CNN recovers.
- The stop-gradient MLP does **not** recover more state (0.564 vs 0.671 linear);
  the latent's state information is linear and insufficient. The readout is not
  the bottleneck.
- The next move is a **separate state encoder** with gradient isolation, not a
  lower guardrail. The oracle proves that a small dedicated CNN can reach the
  state; the JEPA backbone is the thing to fix.
