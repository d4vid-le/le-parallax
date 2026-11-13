# LeParallax — game plan

This is the operating plan for the first chapter: a small, simulated test of
whether same-state good views make an action-conditioned world model more
reliable and less forgetful. It is written to be executed by one person and one
M5 Mac mini with 64 GB of unified memory.

---

## 1. Principles

1. **Small.** One model family, 1–15M parameters, 96px, one machine.
2. **Simulated.** Control the state and the nuisance; do not start with real data.
3. **One variable at a time.** The positive-pair definition is the intervention.
   Everything else is held fixed.
4. **Pre-registered.** Freeze the protocol before the confirmatory run. Pilot
   runs are labelled exploratory.
5. **Negative results are results.** A clean failure at any rung is publishable
   and stops the ladder.
6. **No AGI claims.** AGI is the direction; the claims are narrow.

---

## 2. What we are solving

**The narrow problem.** Action-conditioned world models are outside the
good-view theorem. We do not know whether same-state views improve their latent,
their planning, or their retention — or whether the invariance they impose
removes information the planner needs.

**The broader problem.** Reliable small models that learn continually. Good
views are a candidate lever on representation stability, which is a prerequisite
for continual learning. This chapter tests that lever, not the whole problem.

**What we are not solving.**

- AGI.
- Language modelling.
- A complete continual-learning algorithm. Adaptation still needs a mechanism;
  good views are a regularizer on the representation.
- Large-scale pretraining or real-robot deployment.

---

## 3. How we make claims

A claim is not a hope. It has five parts:

```
direction + metric + population + threshold + falsifier
```

Example:

> At 50% of the training data, good-view pretraining improves OOD planning
> success by ≥10 percentage points over the temporal baseline (five seeds,
> paired bootstrap CI excludes zero), with ≤2 points in-distribution regression.

That is a claim. "Good views will make world models better" is not.

The workflow:

1. **Pilot** — one seed, small budget, exploratory. Debug the pipeline, find the
   effect size, choose the loss weight. Do not report this as confirmation.
2. **Freeze** — write the pre-registration: arms, splits, metrics, thresholds,
   seeds, stop rules. Commit the hash.
3. **Confirm** — run the frozen protocol with five seeds. No changes.
4. **Report** — effect size, confidence interval, per-task diff, negative
   controls, and what was not run.

This is how a lot of careful ML research works, even when the paper does not say
so. The difference here is that we write it down before the run.

---

## 4. Phase 0 — foundations (week 1)

**Goal:** a baseline that trains and plans on the synthetic world, with probes
that can see state and nuisance.

Tasks:

- Build a 2D simulator: a puck or block with position, velocity, and a goal.
  Actions are forces. Rendering is 96×96.
- Define nuisance: lighting direction, background texture, object color, camera
  offset, render style. Each is sampled independently of the state.
- Implement the four data modes:
  - temporal: `(o_t, a_t, o_{t+1})`
  - good view: `(render(s, n1), render(s, n2))`
  - bad view: `(crop(o), crop(o))`
  - temporal + good view
- Implement a small action-conditioned JEPA: CNN encoder with spatial-softmax
  keypoints, residual predictor, SIGReg. Target 1–5M parameters.
- Add the shared auxiliary objectives: a linear state readout and a two-step
  rollout loss. Every arm uses them.
- Implement the probes: linear R² for state and nuisance, plus the direct
  same-state nuisance distance ratio.
- Implement the planner: CEM in latent space with the state readout as the
  terminal cost.
- Reproduce a baseline run end-to-end. It does not need to be good; it needs to
  be measurable.

**Gate 0 (passed, 2026-11-12):** state probe R² ~0.84; planner 2/6 vs random
1/6; mean final distance 0.23 vs 0.62. The baseline is more sensitive to
nuisance than to state (ratio 1.25), which is the target for the good-view arms.
See [`docs/PHASE0.md`](docs/PHASE0.md).

---

## 5. Phase 1 — the controlled experiment (weeks 2–3)

**Goal:** H1 (mechanism) and H2 (behavior) on the synthetic world.

Arms, identical compute, architecture, state readout, and two-step rollout loss:

| Arm | Positive pairs | Purpose |
|---|---|---|
| A. Temporal | `(o_t, a_t, o_{t+1})` | baseline |
| B. Good view | same state, independent nuisance | the intervention |
| C. Bad view | two crops of one render | the paper's contrast |
| D. Temporal + good | both | the practical candidate |

Data fractions: 25%, 50%, 100% of trajectories.

Metrics:

- State probe R² (position, velocity).
- Same-state nuisance distance ratio: nuisance change vs state change. Lower is
  better. This is the primary nuisance metric; the linear nuisance probe is a
  secondary check because Phase 0 showed it can be uninformative.
- Planning success in-distribution.
- Planning success under nuisance shift.
- Success variance across seeds.
- Latent alignment: mutual 10-NN with a state encoder.

Pilot first: one seed, 25% data, all four arms. Use it to choose the good-view
loss weight (sweep `0.01, 0.1, 1.0`) and to check that the probes move. Freeze
the weight before the confirmatory run.

Confirmatory: five seeds, all arms, all data fractions. Pre-register the H1/H2
thresholds.

**Pilot result (exploratory).** Good views reduced the ratio in every seed:
64px mean 1.86 → 0.93; 96px mean 2.22 → 0.81; state probe 0.62 → 0.86 at 96px.
Bad views made it worse (3.69 at 64px). The absolute `<0.5` threshold was not
met, so the confirmatory protocol uses a paired relative endpoint
([`docs/PREREG-v2.md`](docs/PREREG-v2.md)).

**Gate 1 status:** H1a (relative ratio **0.114**, 5/5) and H1c (bad view
**6.71**) confirmed; H1b (absolute state floor) failed on one temporal baseline
seed; overall **FAIL** as pre-registered. Mechanism supported; full H1 claim not
made. See [`docs/H1-CONFIRM.md`](docs/H1-CONFIRM.md). Next: H2 planning under
nuisance shift.

**Gate 2:** H2 shows an OOD gain with no meaningful ID regression. If H2 fails
but H1 passes, the result is the invariance–control trade-off; write that.

**Gate 2 status (2026-11-12):** FAIL as pre-registered. Mean OOD difference
−0.041 (target −0.05), 3/5 seeds lower, CI includes zero; OOD success +0.10, but
random 0.34 beats both planners. The follow-up diagnostic shows a true-state
proportional controller solves the same episodes **100%**, while CEM succeeds
**28%** and the learned state head is negative-R² OOD. H2 is confounded by the
planner/readout, not a representation verdict. Replace the planner and readout
before H2-v2; do not claim control. See
[`docs/PLANNER-DIAGNOSTIC.md`](docs/PLANNER-DIAGNOSTIC.md).

**H2-v2 status (2026-11-12):** FAIL as pre-registered. State-feedback planner
and augmented ridge readout; mean OOD difference 0.000, 3/5 seeds, control tie
0.20. Readout R²: good −0.163 vs temporal −1.293, both negative. The shift is
outside the encoder's training support. Next: domain-randomize encoder training
and test on a held-out shift inside the support. See
[`docs/H2V2-CONFIRM.md`](docs/H2V2-CONFIRM.md).

**Readout diagnostic (2026-11-12):** linear and MLP readouts both negative OOD
(good −0.074 vs temporal −1.290); nonlinearity does not help. The encoder's OOD
support is the wall. H2-v3 trains both arms with domain randomization and tests a
held-out shift inside the support. See
[`docs/READOUT-DIAGNOSTIC.md`](docs/READOUT-DIAGNOSTIC.md).

**H2-v3 status (2026-11-12):** FAIL on distance (−0.026 vs −0.05), but 4/5 seeds
lower, CI excludes zero; OOD success 88% vs 60% (+28 points, all seeds positive).
H2-v4 success-primary replication is running. See
[`docs/H2V3-CONFIRM.md`](docs/H2V3-CONFIRM.md).

**H2-v4 status (2026-11-12):** **PASS.** Success-primary replication on fresh
episodes: +23 points (95% CI +10 to +37), 4/5 seeds positive, pooled McNemar
p = 0.0001; OOD distance −0.036; ID improves. The representation-to-control
bridge holds under domain randomization. Next: H3 continual retention. See
[`docs/H2V4-CONFIRM.md`](docs/H2V4-CONFIRM.md).

---

## 6. Phase 2 — the simulated benchmark (weeks 4–5)

**Goal:** external validity on a known environment.

- Port the same four arms to PushT (2D manipulation) or Two-Room (navigation),
  using the LeWM/MLX stack or stable-worldmodel if needed.
- Use the environment's factors of variation for nuisance: lighting, texture,
  camera, color.
- Keep the model small: 96px, ViT-tiny, 3-frame history, SIGReg.
- Re-run H1 and H2 at 50% and 100% data.
- Compare the synthetic and PushT orderings. If they disagree, the synthetic
  world was too easy or the nuisance definition was wrong.

**Gate 3:** the direction of the effect replicates. If it does not, report the
discrepancy; the synthetic result is not enough.

---

## 7. Phase 3 — continual learning (weeks 6–7)

**Goal:** H3 — does good-view pretraining reduce forgetting under nuisance shift?

Setup:

- Pretrain on domain A (default nuisance).
- Adapt to domain B:
  - **Nuisance shift:** new lighting/texture/camera, same dynamics.
  - **Dynamics shift:** new mass/friction/action effect, same nuisance.
- Adaptation mechanism: full fine-tuning first; then a gated adapter (Le-Gates)
  if the full fine-tuning result is interesting.

Metrics:

- Old-domain success before and after adaptation.
- New-domain success after adaptation.
- Forgetting = old-before − old-after.
- Backward transfer.
- Seed variance.

Prediction: good views reduce forgetting under nuisance shift, not under
dynamics shift. The dissociation is the result.

**Gate 4:** if forgetting is unchanged, representation invariance is not the
lever; the next chapter should focus on adaptation. Write the negative.

**H3 status (2026-11-12): PASS at the representation level.** A-state R² after
adaptation is +0.229 higher (95% CI 0.161–0.298), 5/5 seeds; forgetting 0.025 vs
0.132; B post +0.005 mean, with a seed-4 trade-off. Control retention (H3b) is
still open. See [`docs/H3-CONFIRM.md`](docs/H3-CONFIRM.md).

**H3b status (2026-11-12): boundary holds.** Dynamics shift `[0.65, 0.32]`; good
A predictor MSE is 0.00626 worse than temporal (5/5), both adapt to B. Good views
are nuisance-specific, not dynamics-immune. See
[`docs/H3B-CONFIRM.md`](docs/H3B-CONFIRM.md).

**Collision status (2026-11-12): the claim is conditional.** Five seeds:

- **OOD:** good shifted R² **−0.049** vs temporal **−0.840**. Absolute failure,
  relative gain **+0.791**. A seed-0 pilot reversed under five seeds.
- **CSA:** shifted R² **0.828** vs mix-ft **0.728** (**+0.101**, 95% CI +0.050
  to +0.180), normal retained. CNA-v1 (no support) failed.
- **Le-Gates:** no anchor **+146%** A forgetting; soft + 10% anchor **+104%**;
  hard + 10% anchor **+26%**, routing **18.5%** of A. The router is the
  bottleneck.
- **Capacity:** Seed 4 is domain-specific allocation, not global rank collapse.
- **Anti-Equivalence:** one-way door; the corrected volume penalty collapsed the
  latent (reverse R² −5.08, std 0.599), the asymmetric arm's Δasym was −0.115 vs
  symmetric −0.013 (lower 4/5), and the wrong-sign control had the highest mean
  reachability AUC (0.670). The hypothesis is not supported in this
  observable-door setting.
- **Phase 2 pilot:** the occluded door + matched heads flips the direction sign
  positive (`asym_clean` +0.146 vs `sym_head` −0.055) and the variance floor
  holds std 1.044, but state R² is 0.308/0.322 vs the 0.8 guardrail and the
  planner AUC is 0.722 vs 0.750. A pre-merge frame state head + `λ_pair=0.08`
  raises state R² to 0.670–0.695 and keeps the direction positive (+0.080 vs
  −0.019, margin +0.099). An expanded 340k model at 60 epochs is worse: state
  R² 0.610–0.662, direction margin −0.009, bounded std 0.882. The 0.80
  guardrail and the +0.10 margin are still missed; the 5-seed matrix is not
  started. The oracle CNN reaches **0.935** 4D state R² on the same observations;
  the oracle-relative guardrail is **0.842**, and the JEPA's 0.671 is a real
  representation gap. The stop-gradient MLP head is worse than linear (0.564 vs
  0.671). A dual-stream pilot solves the state path (frozen 47k stream
  **0.936**) but the JEPA no-state-loss direction margin is **+0.018**, and the
  combined planner is **0.500** vs state-only **0.611**. A discriminative
  reachability classifier on the combined latents passes numerically (AUC
  1.000, abort 1.00), but state-only gets the same and JEPA-only is unstable
  (0.285/0.874/0.290); the pass is the side+goal shortcut, not JEPA
  irreversibility. A same-side alcove counterfactual kills the shortcut:
  static state/JEPA are at chance (0.497/0.391–0.497), the frozen JEPA+GRU is
  at chance (0.515), and an end-to-end JEPA+GRU reaches **0.999–1.000**. The
  belief state solves the controlled memory test; natural-episode and
  closed-loop control remain open. Sanity checks: the reachability fine-tune
  improves state R² (0.157→0.442) but makes JEPA dynamics **6.1× worse**
  (0.0179→0.1097), and the frozen JEPA door cue is at chance (**0.524/0.529**)
  even when visible. Use **Option A**: keep the planner JEPA frozen and train a
  separate belief encoder + GRU. Step 2 passes: on natural hidden-alcove
  episodes the static JEPA is 0.431/0.432, the belief stream reaches
  **0.948–0.997 hard AUC**, holds **0.960/0.980/0.980** at `t_cross+3/+5/+10`,
  and passes the counterfactual re-check (**0.766**); the pure natural model was
  a shortcut (0.407). The natural+counterfactual mixture is necessary. Step 3
  first pass: base reachable success 0.75; state gate 0% false/100% abort but
  likely a spatial shortcut; JEPA 0%/25%; natural belief 18% false; planner
  belief **7.1% false/100% abort/0.83 steps after close**, calibration 1%. The
  7.1% test is one episode over the 5% ceiling, the reachable test is small, and
  the matched counterfactual closed-loop is missing. Next: matched
  counterfactual, 100+ reachable test, ensemble of 3 planner-trained belief
  seeds, and fix the base 0.75 reachable success; not another five-seed matrix.
  The state path is solved; the irreversibility/planner value is not yet.
  Inconclusive, not confirmatory.

See [`docs/COLLISION-1B.md`](docs/COLLISION-1B.md),
[`docs/COLLISION-3.md`](docs/COLLISION-3.md),
[`docs/COLLISION-1-2.md`](docs/COLLISION-1-2.md), and
[`docs/COLLISION-4.md`](docs/COLLISION-4.md).

---

## 8. Phase 4 — modular alignment (week 8, optional)

**Goal:** H4 — separately trained vision and proprio encoders align up to a
rotation.

- Train a vision JEPA and a proprio JEPA separately on the same trajectories.
- Fit one orthogonal map on `k` paired timesteps (Procrustes).
- Evaluate retrieval, probe transfer, and planning with the aligned latent.
- Compare to joint training.

This is the most ambitious rung. Run it only if H1–H3 are clean.

---

## 9. Phase 5 — write-up (week 9)

- Results with effect sizes and confidence intervals.
- Per-task diff for the planning and forgetting experiments.
- The negative controls: bad view, temporal-only, equal-compute baseline.
- What was not run.
- A short related-work section: PRH, LeJEPA, LeWM, V-JEPA, Dreamer.
- Code and seeds.

---

## 10. Compute budget

Target machine: M5 Mac mini, 64 GB unified memory.

| Item | Estimate |
|---|---|
| Model | 1–5M parameters, 96px, 3-frame history |
| Batch | 32–64 |
| One training run | < 1 hour at 96px |
| Phase 1 | 4 arms × 3 seeds × 3 data fractions ≈ 36 runs, ~36 h |
| Pilot + ablations | ~10 h |
| Phase 2 | ~20 h |
| Phase 3 | ~20 h |
| Total | ~80–100 GPU-hours, spread over 2–3 weeks |

The model is small enough that the Mac is not the bottleneck. The simulator and
the planner are.

---

## 11. Data plan

**Phase 1: no curation.** The synthetic world generates state, nuisance, and
views on the fly. Nothing is stored except the protocol and the seeds.

**Phase 2: small curated set.** PushT mini or the LeWM PushT dataset. Cache once.
Do not download the 46 GB Cube dataset.

**Phase 3: curate only if the effect holds.** If good views help, build a
multi-view/nuisance dataset with a fixed schema:

```
episode_id, timestep, state, observation, nuisance, action, next_state, split
```

Good-view pairs are two rows with the same `state` and different `nuisance`.
Bad-view pairs are two crops of one `observation`. Temporal pairs are
`(o_t, a_t, o_{t+1})`.

**Phase 4: real data only if the simulated effect is real.** The first chapter
does not need it.

---

## 12. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Nuisance correlates with state | Randomize nuisance independently; check the correlation matrix. |
| Over-invariance removes task-relevant information | Pre-register the alternative; measure ID regression, not just OOD gain. |
| Metric gaming on easy goals | Use hard goals, OOD splits, and per-task diffs. |
| Small n | Three seeds, paired comparisons, bootstrap CIs; pre-register the effect size. |
| Platform friction (MLX/MPS) | Start with MLX; fall back to PyTorch MPS only if necessary. |
| Data contamination | Freeze splits; never tune on the test set. |
| Scope creep | One model, one simulator, four arms. Language and real data are later chapters. |

---

## 13. Decision gates

```
Gate 0  baseline trains and plans; probes recover state
Gate 1  H1: good views change the latent in the predicted direction
Gate 2  H2: OOD planning improves without ID regression
Gate 3  H2 replicates on PushT/Two-Room
Gate 4  H3: forgetting drops under nuisance shift
Gate 5  H4: separate modalities align up to a rotation (optional)
```

A failed gate stops the ladder. It does not stop the project; it changes the
next question.

---

## 14. Timeline

| Week | Work |
|---|---|
| 1 | Simulator, model, probes, baseline; Gate 0 |
| 2 | Pilot; freeze H1/H2 pre-registration |
| 3 | Confirmatory H1/H2; analysis; Gate 1–2 |
| 4 | Port to PushT/Two-Room; Gate 3 |
| 5 | Finish external validity; analysis |
| 6 | Continual setup; domain A/B |
| 7 | H3 confirmatory; Gate 4 |
| 8 | H4 optional; or deepen H3 |
| 9 | Write-up, code, release |

The timeline is a target, not a contract. Gates matter more than dates.

---

## 15. Immediate next steps

1. Build the synthetic world and the four data modes. The temporal mode and the
   baseline are done; good view and bad view are next.
2. Run the one-seed pilot at 64px. Choose the good-view weight and check the
   same-state nuisance ratio.
3. Freeze `docs/PREREG.md` for H1/H2. Do this before the first confirmatory run;
   the pilot can proceed on a separate branch.
4. Run five seeds at 96px.
5. Decide Gate 1. Write the result either way.

The first result is not "does it work." It is "did the manipulation move the
latent?" Everything else follows from that. Phase 0 says the baseline is usable;
the next question is whether good views move the nuisance ratio.
