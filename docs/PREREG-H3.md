# Pre-registration H3 — forgetting after a nuisance shift

**Status:** **FROZEN 2026-11-12.** Do not change after this commit.
**Models:** the frozen H1 temporal and good encoders.
**Code:** `scripts/h3_forgetting.py`.

---

## 1. Question

After adapting to a shifted nuisance domain, does good-view pretraining retain
the original domain better than temporal pretraining?

This is a representation-level continual-learning test. It avoids the planner
and readout confounds from H2; the question is whether the latent still encodes
the old domain's state after adaptation.

---

## 2. Design

### Domains

- **A:** normal nuisance (the H1 training distribution).
- **B:** shifted nuisance (harsher light, brighter background, darker puck
  palette, larger camera offset).

### Models

- Five seeds, `0–4`.
- Frozen H1 temporal and good encoders.
- Both arms adapted with the **same** temporal fine-tuning on B: 10 epochs,
  lr 1e-4, batch 32, no good-view loss. This isolates the effect of the
  pretrained representation.

### Data

| Split | Episodes | Seed | Nuisance |
|---|---:|---:|---|
| A train | 80 | `seed` | normal |
| A test | 10 | `11000 + seed` | normal |
| B train | 80 | `2000 + seed` | shifted |
| B test | 10 | `12000 + seed` | shifted |

### Probes

- Linear state probe fit on each domain's training episodes and evaluated on its
  test episodes.
- Report state R² (position + velocity), position R², and nuisance sensitivity
  ratio.
- Probes are refit after adaptation; the comparison is state information in the
  new latent, not probe transfer.

---

## 3. Endpoint

**H3a (primary).** After adaptation, A-state R² for good views is at least
**0.05 higher** than temporal, with **4/5 seeds** higher and a paired bootstrap
95% CI excluding zero.

**H3b (adaptation).** B-state R² after adaptation is not worse than temporal by
more than 0.05; good views should adapt to B at least as well.

**H3c (nuisance).** Good-view nuisance sensitivity on A/B remains lower than
temporal. Reported; not gating.

**Falsifiers.**

- No A retention advantage.
- B adaptation worse by more than 0.05.
- Good views forget A as much as temporal.

---

## 4. Analysis

- Paired by seed.
- Per-seed A pre/post R², B pre/post R², forgetting (A pre − A post), B gain.
- 95% bootstrap CI over the five paired A post differences.
- No tuning on test episodes; adaptation budget and lr are fixed.

---

## 5. What this does not establish

- Control retention; this is representation-level.
- Dynamics-shift forgetting; that is a separate H3b experiment.
- Real-world data.
- That the H1 mechanism guarantees continual learning; it is one lever.

If H3a passes, good-view pretraining is a continual-learning regularizer under
nuisance shift. If it fails, the control benefit does not translate into
retention, which is a clean dissociation.
