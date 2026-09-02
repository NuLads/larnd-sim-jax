# Differentiable LArTPC calibration — status and handoff

**Repo:** `larnd-sim-jax`, branch `dedx_soft_barrier` (many commits ahead of `origin`, **not pushed**)
**Full detail:** `NOISE_CAMPAIGN_REPORT.md` (long) and `NOISE_CAMPAIGN_DIGEST.md` (figure-by-figure,
67 figures). Both are also published to the Obsidian vault under `06_Analyses/`.
**As of:** 1 September 2026. Nothing is running; the queue is empty. `TS2` completed 1 Sep — see finding 3 and §5.

---

## 1. What we are doing

Fitting 5 LArTPC calibration parameters — recombination `A_b`, `E field`, transverse and
longitudinal diffusion, and electron `lifetime` — by differentiating the simulation and minimising a
marked Poisson point-process negative log-likelihood against target waveforms.

The fit has **three blocks**, and almost everything below is about how they interact:

| block | what | size |
|---|---|---|
| **C** calibration | the 5 physics parameters | 5 |
| **D** dE/dx | per-segment energy deposition (nuisance) | ~4 000 |
| **G** geometry | per-track spline track shape (nuisance) | ~5 000 |

Setup: 400 cm of track per batch, 100 batches, noise ON, targets drawn per seed (`--seed` draws both
the target parameters and the data, so **always compare seed-matched and paired**).

---

## 2. Where we are — the numbers

Best **calibration** is still `BT2` (n = 6 seeds); best **fit** is `TS1` (n = 3). `TS2` improves transverse diffusion to 1.20% but no other parameter — see finding 2b.

| parameter | `BT2` mean \|error\| (n = 6) | per-seed range | comment |
|---|---|---|---|
| `A_b` | **0.27%** | +0.13 … +0.47 | solved |
| `E field` | **0.03%** | −0.05 … −0.00 | solved |
| transverse diffusion | 1.46% | +0.54 … +2.75 | `G20K` reaches 0.61% |
| longitudinal diffusion | 7.41% | −1.01 … +13.03 | worst parameter, unsolved |
| lifetime | 1.65% | −3.04 … +3.69 | the historically hard one |
| position residual | 165 µm | 154 … 176 | `TS1` reaches **105 µm** |

⚠️ **The statistic matters and must always be stated.** These are the **mean of \|error\|** over 6
seeds. The *median* of the same data reads 0.24 / 0.03 / 0.99 / 8.54 / **0.97** — lifetime looks
40% better and longitudinal diffusion 15% worse purely from the choice. The per-seed spread is
larger than most effects we chase, which is why every comparison in the report is **paired** on
seed. See trap T3.

![TS1 — the current best fit](plots/noise_report/fig67_ts1.png)

*Fig 67 — the efficiency frontier (b) is the panel to look at: `TS1` reaches a lower loss than the
26 h `G20K` run in 13.5 h. Panel (f) is the caveat — `TS1`'s calibration is `BT2`'s, frozen, so it
is a better fit and not yet a better calibration.*

**Current best configuration** (this is `BT2`, the production baseline):

```bash
SCIMODE=full SCINBATCH=100 SCILEN=400 SCIBASIS=spline
SCIDEDXPRIOR=1.0088 SCIMEANW=2500.0                       # dE/dx prior at beta=1, mean weight
SCIDXNUL=21.760 SCIDXNUR=4.720 SCIDXLOC=1.8786 \
SCIDXSL=0.0782 SCIDXSR=0.0801                             # the REFIT dE/dx prior shape
SCICLR=1e-2 SCICHDECAY=0.9997 SCILRDECAY=0.91             # schedules, tuned for 10 000 iterations
```
Launcher: `optimize/scripts/start_sci_case.sh`. Every study script in
`optimize/scripts/run_*.sh` is self-documenting — read the header comment before running one.

---

## 3. Key findings, in order of importance

**1. Geometry was optimiser-limited, and fixing it is the largest single win.**
The objective wants geometry essentially exactly at truth (α\* = 0.988) and geometry carries ~5× the
information of any other question. Running 20 000 iterations with **both schedules stretched ×2 in
the exponent** (`G20K`) gives position **−57 ± 0.3 µm on 3/3 seeds (−34.7%)** and the **loss −8.7%**.
*(fig 61)*

![G20K — geometry was optimiser-limited](plots/noise_report/fig61_g20k.png)

*Fig 61 — read the position panel against the DASHED curve, not against `BT2`. Because both
schedules are stretched ×2, `G20K` at iteration 2N has the learning rates `BT2` had at N; the dashed
line is the null "extra iterations buy nothing". It is rejected by 42.7%.*

**2. Freezing calibration lets the nuisance blocks descend further — and it is cheap.**
`TS1`: freeze C at `BT2`'s values, warm-start D and G, re-anneal. Beats `G20K` on the objective on
**3/3 seeds at 51% of the compute** (13.5 h vs 26.4 h). It also converges by iteration ~3 500, so
stage 1 could run at 4 000 iterations (~5.4 h). **`TS1` is the best value in the campaign.**
*(figs 63, 65, 67)*

**2b. But releasing calibration afterwards (`TS2`) does NOT settle the recipe — it trades.**
Stage 2 buys **transverse diffusion (−0.93, 3/3)** and gives back **position (+2.7 µm, 0/3)**;
everything else is a null. Against `G20K` at equal compute (27.0 h vs 26.4 h) it **wins the
objective (−131 NLL, 3/3)** and **loses transverse diffusion (+0.49, 0/3)**. Neither dominates.
Against `BT2` it *is* a clear win (loss −799, position −55.7 µm, tran. diff. −0.93, all 3/3), so the
recipe is not a failure — it is just not better than the cheaper alternatives at what it was built
to fix. **Treat stage 2 as optional.** *(fig 68)*

![TS2 — the calibration release](plots/noise_report/fig68_ts2.png)

*Fig 68 — all paired per-seed differences, because the arms sit within ~3 µm and ~130 NLL of each
other, which is smaller than the seed spread: the mean and median of the absolute levels disagree
about whether `TS1` or `TS2` has the lower loss. Grey = seeds straddle zero = null.*

![The block matrix as a factorial](plots/noise_report/fig65_block_factorial.png)

*Fig 65 — the seven freeze/fit arms are a factorial: for each fitted block the other two form a 2×2,
so each outcome reduces to two main effects. Freezing calibration is the one large lever and helps
both nuisance blocks on both seeds (position −9.3 µm, dE/dx MAE −13.9 × 10⁻⁴).*

**3. The objective PREFERS truth — the remaining error is reachability, not bias.**
With C **and** D **and** G all set to truth, the loss beats the fit on **7/7 arm-seeds by 5.6%**. And
the gap is closing: 9.4% (`BT2`) → 6.6% (`BM0`) → **2.5%** (`G20K`). *(fig 66)*

![The objective prefers truth](plots/noise_report/fig66_objective_bias.png)

*Fig 66 — panel (a) is the test that settles it: with everything at truth the loss beats the fit on
7/7. Panel (c) is the mechanism below. **This figure corrects an earlier version of itself** that
concluded the opposite from a 1-D conditional scan — see trap T2.*

**4. But blocks must move TOGETHER.** Setting one block to truth while the others stay fitted:

| moved to truth | Δ loss | better on |
|---|---|---|
| geometry alone | **−7.00%** | 7/7 |
| calibration alone | **+0.18%** | 0/7 |
| dE/dx alone | **+2.79%** | 0/7 |
| **all three** | **−5.60%** | **7/7** |

Setting calibration to its *true* values while the nuisances are wrong makes the fit **worse**. This
is the single most important fact for interpreting any 1-D scan. *(fig 66)*

**5. The dE/dx prior is a genuine objective bias.** Putting per-segment dE/dx at **truth** makes the
loss *worse* by 2.79% on 7/7. The prior displaces its own block's minimum. Refitting the prior shape
to the segmentation actually in use (the `SCIDX*` numbers above) was a large win — `A_b` 0.77% →
0.27% — but the block remains prior-limited. *(figs 46–48, 66)*

**6. A_b and E field are decoupled from both nuisance blocks** (±0.1% shift when either goes to
truth), which is exactly why they are the two parameters we recover well. Everything else is
geometry-coupled. *(fig 58)*

![The degeneracy matrix](plots/noise_report/fig58_degeneracy_matrix.png)

*Fig 58 — which calibration parameter is entangled with which nuisance block, measured by scanning
the objective (no optimiser involved). `A_b` and `E field` are decoupled from both, at ±0.1%.
Panel (e) is the statistical-floor comparison behind finding 7.*

**7. Every parameter sits 4–16× from its statistical floor.** We are not statistics-limited
anywhere. *(fig 58)*

---

## 4. What is open

**(a) The calibration profile — the real open question.** Fig 66 shows the objective prefers truth
jointly, but every scan we have holds the *other* calibration parameters fixed while one moves. With
geometry and dE/dx both at truth the lifetime slice minimum is still 2.26% from truth. Whether that
survives a **profile over the full calibration block** is unknown — no run has done it. This is the
question that decides whether lifetime has a residual bias or not.

**(b) Longitudinal diffusion, 7.41%,** is the worst parameter by far and nothing has moved it. Its
optimum runs off a ±20% scan grid whenever geometry goes to truth.

**(c) Stage 2 of the two-stage recipe has not been run** — see below.

---

## 5. What to run next — concrete

*(`TS2` was Priority 1 and has now been run — it did not settle the recipe. See finding 2b.)*

**Priority 1 — the calibration profile (4(a)). Needs new code; this is now the top item.**
Every scan we have fixes the other four calibration parameters while one moves, so we cannot yet
say whether the residual lifetime displacement is real. Build a scan that **profiles** over the full
calibration block and re-run it at `BT2`'s, `TS1`'s and `G20K`'s checkpoints. It is evaluation-only
and cheap to run once written (~3 h/point set). **This is the question that decides whether lifetime
has a residual bias, and every optimisation avenue is now exhausted, so it is what is left.**

**Priority 2 — confirm `TS1` at n = 6 and adopt it.** ~6 jobs × 13.5 h. `TS1` is the best value we
have (best fit, half the compute of `G20K`) and everything is at n = 2–3, which only resolves large
effects. If it holds, `TS1` — not the full two-stage — becomes the production recipe. Cheaper still:
`TS1` converges by ~3 500 iterations, so test a 4 000-iteration variant (~5.4 h) at the same time.

**Priority 3 — longitudinal diffusion (4(b)),** at 7.4% the worst parameter, untouched by anything
we have tried.

**Do not** spend more effort on longer runs, fancier schedules or new block orderings. Between
`G20K`, the block matrix and the two stages, that avenue is done: the arms now differ by less than
the seed spread, and every further gain in the loss has come with a loss somewhere in the physics.

---

## 6. Traps — please read this section

Several published "results" in this campaign were wrong for these reasons. They are cheap to repeat.

**T1. Equal iteration count is NOT equal schedule phase.** The calibration LR decays **per epoch**,
so a 3 000-iteration run reads out at 0.059 of peak while a 10 000-iteration run reads out at
8 × 10⁻⁵ — a factor **700**. Comparing a short arm to a converged one at equal iteration count
compares a still-moving number to a frozen one. *This invalidated a published ×2.3 improvement.*
Either converge the short arm, or compress its decay to land on the same terminal LR
(`SCILRDECAY` 0.91 → 0.73 for 30 epochs), or quote its oscillation amplitude alongside its value.

**T2. A conditional minimum is not a joint minimum.** Scanning one parameter with the nuisances
pinned at their *fitted* values will show the fit sitting at "the optimum" **by construction**
whenever the blocks are coupled. It says nothing about where the joint minimum is. Use the
constructed truth points (`P0_fit` … `P4_geom_true`) for that. *This one also produced a published
conclusion that was exactly backwards.*

**T3. Quote |error|, not the signed mean,** for any parameter whose per-seed errors mix sign — and
never a ratio of medians at small n. With n = 3, "+73%" and "+88%" headline regressions turned out
to be nulls on the paired difference.

**T4. A median over 2 seeds is the mean of 2.** n = 2 finds large effects only.

**T5. Position residual is a per-batch quantity** and swings 120–1400 µm within one converged run.
Always a per-seed tail median. With geometry frozen it is recorded for batch 0 only, so it is *not*
comparable to a normal run's median.

**T6. Check jobs actually ran.** `sdfampere038` has twice reported a healthy GPU to the host while
JAX saw none inside the container; the task died in 9 s and SLURM recorded `COMPLETED 0:0`. There is
now a GPU preflight *and* an exit-status propagation fix in `start_sci_case.sh`, but always confirm
iteration counts, not just `sacct` states. Consider `--exclude=sdfampere038`.

**T7. Never compare runs across days on a dirty tree.** One "finding" was pure code drift between
two `git_dirty` runs.

**Practical:** prefer the `ampere` (A100) partition — `turing` is 3–4× slower for JAX. Multi-account
submission across `neutrino:cider-nu`, `neutrino:dune-ml`, `mli:cider-ml`, `mli:nu-ml-dev` avoids
per-account node limits. `sbatch --export` splits on commas, so colon-separate list arguments.

---

## 7. Where things are

| what | where |
|---|---|
| Full report | `NOISE_CAMPAIGN_REPORT.md` |
| Figure-by-figure digest (67 figs) | `NOISE_CAMPAIGN_DIGEST.md` |
| Figures | `plots/noise_report/` |
| Figure scripts | `make_*_fig.py` in the repo root |
| Study launchers | `optimize/scripts/run_*.sh` (each has a header explaining the design) |
| Main launcher | `optimize/scripts/start_sci_case.sh` |
| Fitter | `optimize/fit_params.py` |
| Results | `fit_result/sci_full_<TAG>/history_iter<N>_*seed<S>.pkl` |

Useful env knobs added during this campaign: `SCICHAININIT` / `SCICHFREEZE` (geometry warm start /
freeze), `SCIDEDXINIT` / `SCIDEDXFREEZE` (dE/dx), `SCICALIBFREEZE` (calibration),
`SCITRUTHPOINT` + `SCITPCALIB` (evaluation-only scans, no fitting — ~3 h for 145 points).
