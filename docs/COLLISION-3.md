# Collision 3 — Le-Gates on the dynamics boundary

**Status:** complete, 2026-11-12.
**Protocols:** [`PREREG-COLLISION-2.md`](PREREG-COLLISION-2.md) (v1),
[`PREREG-LEGATES-V2.md`](PREREG-LEGATES-V2.md) (soft + anchor),
[`PREREG-LEGATES-V3.md`](PREREG-LEGATES-V3.md) (hard gate).
**Artifacts:** `runs/legates-v1/`, `runs/legates-v2/`, `runs/legates-hard/`
(git-ignored).

H3b showed that good views do not protect against a dynamics shift. This
collision asks the next question: can a Le-Gates-style residual adapter recover
the shifted dynamics without forgetting the original one?

Three versions, five seeds each. Frozen H1 good-view encoder and base
predictor; only a 33k–35k-parameter adapter is trained on B
`[damping=0.65, gain=0.32]` for 10 epochs.

---

## 1. v1 — soft gates, no retention signal

Adapter trained on B only. Gate is global, input-conditioned `(z, a)`, or
error-conditioned `(z, a, e_prev)`.

| arm | A rel forget | B rel gain | pass seeds | gate A/B | gate AUC |
|---|---:|---:|---:|---:|---:|
| full-ft | +0.865 | +0.723 | 0/5 | — | — |
| global | +1.551 | +0.540 | 0/5 | 0.526 / 0.526 | 0.500 |
| input | +1.536 | +0.542 | 0/5 | 0.649 / 0.662 | 0.542 |
| error | +1.459 | +0.544 | 0/5 | 0.666 / 0.708 | 0.612 |

**Result: all gates fail.** A forgetting is **+146% to +155%**, worse than full
fine-tuning (+87%). The error gate learns a little routing (AUC 0.612; seed 1
0.796) but still opens on A. The adapter also underperforms full fine-tuning on
B (+54% vs +72% gain), because the soft gate is stuck near 0.53 and only half
the residual is applied.

**Why:** the gate is trained only on B. It never sees an A example, so it has no
signal to close on A. This is a credit-assignment failure, not a threshold
problem.

---

## 2. v2 — soft gate + 10% A replay anchor

Add 8 of the 80 A training episodes (248 transitions) and a distillation loss:
the adapted A prediction must match the frozen base prediction.

| arm | A rel forget | B rel gain | pass seeds | gate A/B | gate AUC | resid AUC |
|---|---:|---:|---:|---:|---:|---:|
| full-ft | +0.865 | +0.723 | 0/5 | — | — | — |
| global | +1.364 | +0.465 | 0/5 | 0.545 / 0.545 | 0.500 | 0.540 |
| input | +1.294 | +0.484 | 0/5 | 0.755 / 0.747 | 0.475 | 0.535 |
| error | +1.041 | +0.506 | 0/5 | 0.667 / 0.815 | **0.667** | 0.611 |

**Result: partial improvement, still fails.** The anchor cuts error-gate
forgetting from +146% to +104% and raises routing AUC from 0.61 to 0.67, but the
gate still opens on A (mean 0.667) and A forgetting is 20× the 5% endpoint. The
soft sigmoid leaks.

The error signal is real — the gate opens more on B than A — but **routing is
not retention.** The gate can be directionally correct and still leak enough to
destroy A.

---

## 3. v3 — hard error gate

Train the adapter on B with the gate fixed at 1. At evaluation, apply it only
when `e_prev > τ`, with `τ` = 95th percentile of the A-anchor previous errors.

| seed | A forget | B gain | frac A routed | frac B routed | τ |
|---:|---:|---:|---:|---:|---:|
| 0 | +0.168 | +0.357 | 0.181 | 0.577 | 0.00813 |
| 1 | +0.234 | +0.344 | 0.184 | 0.610 | 0.00770 |
| 2 | +0.287 | +0.393 | 0.229 | 0.684 | 0.00861 |
| 3 | +0.317 | +0.496 | 0.200 | 0.735 | 0.00819 |
| 4 | +0.290 | +0.485 | 0.132 | 0.668 | 0.01199 |
| **mean** | **+0.259** | **+0.415** | **0.185** | **0.655** | — |

**Result: 0/5 pass.** The hard gate cuts forgetting from +146% (v1) and +104%
(v2) to **+26%**, and B gain stays at +42%. But:

- **18.5% of A transitions are routed**, almost double the 10% budget. The A
  anchor's 95th percentile does not cover the A test tail.
- Only 65.5% of B transitions are routed; some B transitions are indistinguishable
  from A by the error signal.
- A forgetting is still **+26%**, five times the 5% endpoint.

**Why:** the A and B error distributions overlap. A single scalar threshold on
`e_prev` cannot separate them cleanly with 10% A replay. The hard gate is
structurally correct — non-routed transitions are exactly preserved — but the
**router** sends too many A transitions through the adapter.

---

## 4. What this establishes

| version | A forget | B gain | verdict |
|---|---:|---:|---|
| full fine-tune | +87% | +72% | plasticity without stability |
| soft, no anchor | +146% | +54% | gate cannot close on A |
| soft + 10% anchor | +104% | +51% | gate routes, but leaks |
| hard + 10% anchor | **+26%** | +42% | isolation works only where routed |

The ordering is the result:

1. **A gate trained only on B cannot preserve A.** It has no signal to close.
2. **A soft gate leaks even when it routes correctly.** The sigmoid applies a
   nonzero residual on A. This matches the earlier Le-Gates evidence: soft
   single gate +169% drift vs +1015% naive, not zero.
3. **A hard gate isolates exactly, but only where the router says so.** With 10%
   A replay, the error distributions overlap; 18.5% of A is routed and the
   retention endpoint fails. The router/detector is the bottleneck, not the gate
   scalar.
4. **The adapter is not a free lunch on B either.** It gains +42% vs full
   fine-tuning's +72%. A 33k adapter on a frozen encoder is cheaper, not better.

This is consistent with the user's Le-Gates framing: **the gate is the policy;
isolation is the guarantee.** A soft scalar gate is bounded adaptation, not a
no-forgetting mechanism. Zero non-routed forgetting requires a reliable router
plus structural isolation (hard masks, frozen slots, or a much stronger
detector). The 10% replay budget used here is not enough to make the router
reliable.

## 5. What would be tested next

- More A replay (25–50%) or a learned detector with an explicit safety margin.
- Per-layer hard masks / growable slots, which gave zero non-routed forgetting
  in the earlier Le-Gates work, instead of a scalar gate.
- A router trained with a **false-open penalty** on A, not just distillation.
- Control retention, not only predictor MSE.

None of these were run here. The current result is a boundary, not a solution.
