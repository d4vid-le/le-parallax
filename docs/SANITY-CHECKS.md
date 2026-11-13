# Sanity checks — spatial damage and the missing door cue

**Status:** complete, 2026-11-12.
**Code:** `scripts/spatial_forgetting_probe.py`,
`scripts/door_cue_probe.py`.
**Artifacts:** `runs/spatial-forgetting/`, `runs/door-cue-probe/`
(git-ignored).

## 1. Did the reachability fine-tune damage spatial/dynamics information?

The end-to-end JEPA+GRU fine-tuned the frame encoder for 40 epochs. We compare
the frozen and fine-tuned frame encoders with the same frozen merge/predictor:

| encoder | 4-D state R² | position R² | velocity R² | one-step dynamics MSE |
|---|---:|---:|---:|---:|
| frozen JEPA | 0.157 | 0.252 | 0.062 | **0.0179** |
| fine-tuned (reachability) | **0.442** | **0.627** | **0.257** | **0.1097** |

The fine-tune did **not** collapse spatial information. It improved state
readability substantially. But it **damaged JEPA dynamics**: one-step MSE is
**6.1× worse** (0.0179 → 0.1097) because the frozen predictor/merge now sees a
shifted latent.

**Decision:** the reachability fine-tune is not safe as the planner's JEPA
encoder. Keep the planner's JEPA frozen. Give the belief state its own separate
encoder + GRU (Option A). Option B — a joint multi-task loss with the original
JEPA dynamics objective — is the fallback if a shared encoder is required.

## 2. Does the frozen JEPA encode the visible door cue?

Controlled left-side histories, same puck/goal/nuisance, only door open/closed.
Linear probe on the frozen JEPA:

| probe | AUC |
|---|---:|
| single frame, x=0.45 (door visible) | **0.524** |
| 3-frame history, x=0.45 (door visible) | **0.529** |
| single frame, x=0.50 (occlusion hides door) | 0.496 |
| 3-frame history, x=0.50 | 0.522 |

At **x=0.45, with the door fully visible**, the frozen JEPA probe is at chance.
The pretrained JEPA discarded the transient door cue. The frozen-feature GRU's
**0.515** is therefore not a GRU optimization quirk: the features do not carry
the cue. End-to-end training is required to learn it.

## 3. Architecture decision

**Option A — separate belief encoder + GRU.**

- Planner JEPA stays frozen; its dynamics are preserved.
- The belief stream gets its own encoder + GRU trained on the reachability task.
- The belief stream cannot corrupt the planner's spatial/dynamics path.
- This matches the dual-stream design: state stream for position, JEPA for
  dynamics, belief stream for history-dependent reachability.

**Option B — joint multi-task fine-tune.** Add the original JEPA dynamics loss to
the reachability fine-tune. Viable, but it must learn the door cue and preserve
dynamics at the same time; the current fine-tune shows the trade-off is real.
Option A is the safer next experiment.

## 4. Consequence for the next step

The frozen JEPA cannot be the belief encoder. The natural-episode / horizon test
must use the **separate belief encoder + GRU** (Option A), not the fine-tuned
JEPA. The planner comparison is:

- state-only,
- state + frozen JEPA (dynamics/irreversibility),
- state + separate belief stream (reachability),
- state + frozen JEPA + belief stream.

The frozen JEPA's door-cue failure also means the earlier `door_auc` 0.77 on
natural states was not reading the door; it was reading the puck/side geometry.

## Caveats

- One fine-tuned seed; the spatial probe is a 3-frame latent ridge readout.
- The door-cue probe is a linear readout; a nonlinear probe could in principle
  recover more, but the MLP on `sg(z)` in Phase 2 did not.
- The counterfactual door cue is rendered with the same world; the probe tests
  the frozen encoder, not the fine-tuned one.
