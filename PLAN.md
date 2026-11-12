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
> success by ≥10 percentage points over the temporal baseline (three seeds,
> paired bootstrap CI excludes zero), with ≤2 points in-distribution regression.

That is a claim. "Good views will make world models better" is not.

The workflow:

1. **Pilot** — one seed, small budget, exploratory. Debug the pipeline, find the
   effect size, choose the loss weight. Do not report this as confirmation.
2. **Freeze** — write the pre-registration: arms, splits, metrics, thresholds,
   seeds, stop rules. Commit the hash.
3. **Confirm** — run the frozen protocol with three seeds. No changes.
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

Confirmatory: three seeds, all arms, all data fractions. Pre-register the H1/H2
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
random 0.34 beats both planners. The planner is the bottleneck. Diagnose the
planner before H2-v2; do not claim control.

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
4. Run three seeds at 96px.
5. Decide Gate 1. Write the result either way.

The first result is not "does it work." It is "did the manipulation move the
latent?" Everything else follows from that. Phase 0 says the baseline is usable;
the next question is whether good views move the nuisance ratio.
