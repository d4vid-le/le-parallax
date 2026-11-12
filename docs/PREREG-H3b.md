# Pre-registration H3b — dynamics-shift boundary

**Status:** **FROZEN 2026-11-12.** Do not change after this commit.
**Models:** the frozen H1 temporal and good encoders.
**Code:** `scripts/h3b_dynamics.py`.

---

## 1. Question

Does good-view pretraining protect against a **dynamics shift**, the way it
protects against a nuisance shift?

**Prediction:** no. Good views buy nuisance invariance, not dynamics immunity.
The good-view advantage should shrink or disappear; both arms should need
adaptation.

---

## 2. Design

- Domain A: normal dynamics `[damping=0.85, gain=0.20]`, normal nuisance.
- Domain B: shifted dynamics `[0.65, 0.32]`, normal nuisance.
- Five seeds, `0–4`.
- Frozen H1 temporal and good encoders.
- Both arms fine-tuned on B with the same 10-epoch temporal objective, lr 1e-4.
- Metric: one-step predictor MSE on held-out A and B transitions.
- A train 80 / A test 10 (seeds `13000 + seed`).
- B train 80 / B test 10 (seeds `14000 + seed`).

The pilot seed (seed 0, full budget) showed good **worse** on A by 0.005 MSE
while both adapted to B. The threshold below is informed by that pilot.

---

## 3. Endpoint

**H3b (boundary).** The mean good-view advantage on A post-adaptation
`temporal_mse − good_mse` is **≤ 0.005**, and both arms improve on B
(`B_gain > 0`).

- If the advantage is ≤0.005, the boundary holds: no meaningful dynamics
  retention benefit.
- If the advantage is >0.005, good views transfer to dynamics, which would be a
  surprising positive result.

Report per-seed A/B MSE, A forgetting, B gain, and the advantage. No control
metric; this is a predictor-level test.

---

## 4. What this does not establish

- That dynamics adaptation is solved.
- Control retention.
- Real-world dynamics.
- Anything beyond this shift size and the synthetic world.

The expected result is a **specificity boundary**: good views are for nuisance,
not for dynamics.
