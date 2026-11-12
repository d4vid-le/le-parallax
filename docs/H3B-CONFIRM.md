# H3b result — dynamics-shift boundary

**Status:** complete, 2026-11-12.
**Protocol:** [`PREREG-H3b.md`](PREREG-H3b.md), sha256
`eb2a6086b28bca9a44bd430d5d3338455cffc6038afc426b1394301ff30cea1e`.
**Artifacts:** `runs/h3b/` (git-ignored).
**Verdict:** **boundary holds** — no good-view dynamics retention advantage.

---

## 1. Result

Five seeds. Frozen H1 temporal/good encoders. Both arms adapted to shifted
dynamics `[damping=0.65, gain=0.32]` with the same 10-epoch temporal objective.
Metric: one-step predictor MSE on held-out A (normal dynamics) and B (shifted
dynamics) transitions.

| seed | A temp post | A good post | diff | A forget temp | A forget good | B temp post | B good post |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.00461 | 0.00980 | +0.00519 | −0.00098 | +0.00445 | 0.00522 | 0.00439 |
| 1 | 0.00577 | 0.01049 | +0.00471 | −0.00058 | +0.00481 | 0.00385 | 0.00580 |
| 2 | 0.00555 | 0.01226 | +0.00672 | +0.00024 | +0.00539 | 0.00496 | 0.00645 |
| 3 | 0.00490 | 0.01195 | +0.00705 | −0.00087 | +0.00562 | 0.00360 | 0.00475 |
| 4 | 0.00525 | 0.01289 | +0.00764 | −0.00041 | +0.00635 | 0.00677 | 0.00513 |

- Mean A post difference (good − temporal): **+0.00626**.
- Mean good-view advantage (temporal − good): **−0.00626**.
- Good was worse on A in **5/5** seeds.
- Mean A forgetting: temporal **−0.00052** (no forgetting), good **+0.00532**.
- Mean B post difference (temporal − good): **−0.00042** (essentially tied).
- Mean B gain: temporal **+0.00385**, good **+0.01482** (good started worse and
  improved more, ending in the same place).
- Boundary holds: **True**.

## 2. What this establishes

Good-view pretraining does **not** protect against a dynamics shift. If anything,
it costs a small amount of A retention in this setup while both arms adapt to the
new dynamics equally well at the end.

This is the specificity result the chapter needed:

> Good views buy nuisance invariance, control robustness under nuisance shift,
> and retention under nuisance shift — **not** dynamics immunity. Dynamics
> changes still require adaptation.

The mechanism is not “more invariance is always better.” It is specifically about
separating world state from nuisance. When the world’s dynamics change, the
problem is different, and the nuisance-invariant encoder has no special
advantage.

## 3. Caveats

- Predictor MSE only; no control retention.
- One dynamics shift size; not a general dynamics-shift law.
- Temporal A slightly improved rather than forgot; the comparison is “good is
  worse than temporal,” not “both forget.”
- Synthetic world, 96px, five seeds.

## 4. What it means for the chapter

The four-part story is now complete:

```
H1a/H1c   mechanism:   good views disentangle state from nuisance
H2-v4     control:     +23 points OOD success under nuisance shift
H3        retention:   ~81% less forgetting after a nuisance shift
H3b       boundary:    no dynamics protection; adaptation still required
```

H3b is what makes the first three credible. The good-view effect is specific, not
a generic “more invariance” claim.
