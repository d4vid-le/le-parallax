# H3 result — forgetting after a nuisance shift

**Status:** complete, 2026-11-12.
**Protocol:** [`PREREG-H3.md`](PREREG-H3.md), sha256
`c4660371980ef02518b99f9d41c817cc907aa7d67e7a450146f2602e2624f1ec`.
**Artifacts:** `runs/h3/` (git-ignored).
**Verdict:** **PASS.**

---

## 1. Result

Five seeds. Frozen H1 temporal/good encoders. Both arms adapted to the shifted
nuisance with the same 10-epoch temporal fine-tuning. State probes refit after
adaptation.

| seed | A temp post | A good post | diff | A forget temp | A forget good | B temp post | B good post | B diff |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.783 | 0.897 | +0.114 | +0.074 | +0.028 | 0.734 | 0.779 | +0.045 |
| 1 | 0.728 | 0.909 | +0.180 | +0.085 | +0.023 | 0.765 | 0.861 | +0.096 |
| 2 | 0.654 | 0.921 | +0.267 | +0.071 | +0.037 | 0.787 | 0.761 | −0.026 |
| 3 | 0.661 | 0.898 | +0.237 | +0.173 | +0.022 | 0.720 | 0.791 | +0.072 |
| 4 | 0.578 | 0.925 | +0.347 | +0.255 | +0.015 | 0.789 | 0.626 | −0.162 |

- Mean A post difference: **+0.229** (95% CI **+0.161 to +0.298**).
- Seeds good higher: **5/5**.
- Mean A forgetting: temporal **+0.132**, good **+0.025**.
- Mean B post difference: **+0.005**.

## 2. What this establishes

After adapting to a shifted nuisance, the good-view encoder retains the original
domain far better:

- A-state R² is **+0.229** higher on average, 5/5 seeds, CI excludes zero.
- Forgetting is **0.025 vs 0.132** — roughly **81% less**.
- B adaptation is on average equal (+0.005), not a systematic plasticity
  penalty.

This is the continual-learning result the chapter was aiming for at the
representation level:

> Good-view pretraining acts as a retention regularizer under nuisance shift.
> The encoder keeps the old world's state geometry because it was trained to
> discard independently varying nuisance.

## 3. Caveats

- **Forgetting is reduced, not eliminated.** The good encoder still loses
  0.015–0.037 A R²; the temporal baseline loses 0.071–0.255.
- **B adaptation has one real trade-off.** Seed 4 regresses on B (0.789 → 0.626,
  −0.162), while gaining the most on A (0.578 → 0.925). That is the
  stability–plasticity trade-off, visible in one seed even though the mean is
  flat.
- **Representation-level only.** This is a state-probe result, not control
  retention. H3b would need a dynamics-shift/control experiment.
- **Synthetic.** 96px, toy world, domain randomization, five seeds.
- **H1b/H2-v3 misses remain.** H1b failed its absolute state floor; H2-v3 failed
  its distance endpoint. H3 does not retroactively fix those.

## 4. The chapter so far

```
H1a/H1c   good views remove nuisance and preserve state (mechanism)
H2-v4     good views improve OOD control success by +23 points (control)
H3        good views forget ~81% less after a nuisance shift (retention)
```

The failures are part of the record:

```
H1b       absolute state floor failed on one temporal baseline seed
H2        CEM was broken
H2-v2     readout/OOD support was insufficient
H2-v3     distance endpoint missed; success signal led to H2-v4
```

## 5. Next

Two honest options:

1. **H3b — dynamics shift.** Show the boundary: good views should not protect
   against a changed dynamics model; adaptation is still needed. This would
   complete the stability–plasticity story.
2. **Write-up.** The three-part result is coherent enough for a research note:
   mechanism, control, retention. H3b can be future work.

The short answer to the original question is yes: good views retain the
underlying domain better. The precise answer is: **much better, not perfectly;
and one seed shows the plasticity cost.**
