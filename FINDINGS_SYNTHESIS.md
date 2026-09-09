# Where we stand — differentiable LArTPC calibration, September 2026

**This replaces the 5 September synthesis.** Everything below is measured, each claim tied to a
figure; the corrections table at the end tracks what changed. Full detail:
`NOISE_CAMPAIGN_REPORT.md` §6w (figs 61–72).

---

## The story in one paragraph

The campaign has moved through three phases and each closed cleanly. **Optimisation** (figs 61–68):
solved — `TS1` (freeze calibration, re-anneal nuisances from `BT2`, 13.5 h) is confirmed at
**n = 6/6** as the best fit, and the path from any fit to truth is measured to be open, monotone,
and blocked only by conditioning. **Understanding the objective** (figs 66, 69): the objective
prefers truth jointly; single blocks moved alone make it worse; the remaining gap is reachability.
**Uncertainties** (figs 70–72): profiling by re-fitting is impossible here (×2 105 too noisy), but
curvature works — and the resulting error-bar ladder delivers the campaign's biggest reframe:
**lifetime's achieved 1.65% error is only ×1.5–1.8 from its statistical floor once the dE/dx
nuisances are marginalised, with geometry still to come.** The fit is not underperforming on
lifetime; the *measurement* is nearly exhausted, because recombination and lifetime are up to 83%
anticorrelated. The frontier is no longer the optimiser — it is the degeneracy.

---

![The uncertainty ladder](plots/noise_report/fig72_uncertainty_ladder.png)

*Fig 72 — the figure behind the scoreboard. Panel (a): each marginalisation raises the floor, and
lifetime's achieved error (dotted) is nearly on top of the final rung. Panel (b): the A_b↔lifetime
degeneracy deepening. Panel (c): longitudinal diffusion's displaced, entangled axis.*

## The scoreboard

| parameter | achieved (`BT2`, mean \|err\|, n = 6) | 1σ floor (dE/dx-marginalised) | distance |
|---|---|---|---|
| A_b | 0.27% | ~0.05% | ~×6 |
| E field | 0.030% | ~0.0005% | **~×60 — the real headroom** |
| transverse diffusion | 1.46% | ~0.4–0.5% (unstable rows: indicative) | ~×3 |
| longitudinal diffusion | 7.41% | ~0.9% (indicative) | ~×8, but see below |
| **lifetime** | **1.65%** | **0.9–1.1%** | **×1.5–1.8 — nearly at the limit** |
| position | 165 → **105 µm** (`TS1`) | repr. floor ~80 µm | — |

*Floors are still conditional on geometry; marginalising it (needs remat through the simulation)
can only raise them further — every "distance" above is an upper bound.*

---

## The five load-bearing results

**1. The production recipe (figs 67–68, 72).** `TS1` — warm-start everything from `BT2`, freeze
calibration, re-anneal geometry+dE/dx 10 000 it — beats the 2×-cost `G20K` on the objective and is
confirmed **6/6 seeds**: position −55.1 ± 2.1 µm, loss −756 ± 55. Stage 2 (release calibration)
trades rather than wins → optional. The 4 000-it shortcut costs +10 µm / +378 loss (3/3): schedule
compression cuts total optimiser travel. **Adopt `TS1` at 10 000 iterations.**

![TS1 — the production recipe](plots/noise_report/fig67_ts1.png)

*Fig 67 — the recipe. Panel (b) is the efficiency frontier: `TS1` reaches a lower loss than the
26 h `G20K` at 13.5 h. Panel (f): its calibration is `BT2`'s, frozen — the fit gain, not a
calibration gain. Since confirmed at n = 6/6.*

**2. No barrier to truth (fig 69).** The joint fit→truth line is monotone downhill in every arm and
seed, minimum *at* truth. Fits stop because the annealed LR runs out of step size in an
ill-conditioned valley. `TS1` closes 86–87% of the baseline's gap.

![No barrier](plots/noise_report/fig69_fullpath.png)

*Fig 69 — the fit→truth path with all three blocks moving together: monotone downhill everywhere,
minimum AT truth (dashed red). The fits stop because the annealed learning rate runs out of step
size, not because the gradient vanishes.*

**3. Uncertainties come from curvature, not re-fitting (figs 70–71).** Re-fit profiling fails by
×2 105 (optimiser landing noise ≫ signal). The FD-gradient Hessian is validated against an
independent scan to 1–2%, and exposed that the old scan-based floors for A_b and E field were
measured outside the quadratic region (E field's was 7× too large).

![Why curvature and not re-fitting](plots/noise_report/fig70_profile.png)

*Fig 70 — the method boundary. Re-fit profiling is ×2 105 too noisy and cannot be averaged away;
the noise is the optimiser's landing point, not the data.*

![The validated Hessian](plots/noise_report/fig71_hessian.png)

*Fig 71 — the instrument that works. Panel (a): the Hessian reproduces an independent scan to 1–2%
where that scan is trustworthy. Panel (b): the A_b↔lifetime correlation, −0.71 conditional.*

**4. The ladder (fig 72).** Marginalising dE/dx exactly (banded+rank-1, direct solve — after three
documented failures: 129 GB, 8.3 GB-per-tangent, FD-CG divergence) raises the lifetime floor to
0.9–1.1%. **A_b↔lifetime = −0.71 conditional, −0.83 marginalised**: the trade partly routes through
the dE/dx field. This is the campaign's central degeneracy, quantified.

**5. Longitudinal diffusion is not a standalone parameter (fig 72c).** Its conditional optimum sits
10–44% *below* truth and moves further away as nuisances improve; since the joint minimum is at
truth, other calibration parameters compensate it. Its 7.4% error is a symptom of entanglement, not
a parameter-specific failure.

---

## What would move the needle now (ranked)

**1. Break the A_b↔lifetime degeneracy in the forward model.** The only route to a better lifetime
(external A_b priors are shown ineffective — even a 0.05% prior moves lifetime 0.43 → 0.41%).
`schDP` — the drift-profile arm's curvature, running now — is the first concrete test: does
dpw = 1e6 reduce the −0.83? Fit-quality cost is already measured at zero.

![The degeneracy matrix](plots/noise_report/fig58_degeneracy_matrix.png)

*Fig 58 — for orientation on move 1: which calibration parameter couples to which nuisance block.
A_b and E field are decoupled from both nuisances; the A_b↔lifetime coupling is internal to the
calibration block, which is why it survives every nuisance improvement.*

**2. Go after E field's ~×60 headroom.** The one parameter with a large, *stable* measured gap
between achieved and floor. Nothing has ever been tuned for it because 0.03% looked "solved".

**3. Remat through `simulate_signals`.** One engineering task unlocks both remaining builds: the
geometry marginalisation (completing the error bars) and the preconditioner (the measured remedy
for the ill-conditioned valley). Until then, geometry-conditional floors are lower bounds.

**4. Real-data validation.** The simulation-on-simulation programme is converged enough to freeze:
recipe (`TS1`), error bars (ladder), and known caveats (long. diffusion entanglement). The next
qualitatively new information is real detector hits.

**Not worth further effort:** longer runs, schedule variants, block orderings, re-fit profiling,
external priors on A_b — each now has a measured null or a measured wall behind it.

---

## Corrections ledger (so stale numbers don't circulate)

| claim | corrected to | when |
|---|---|---|
| "4–16× from the statistical floor" | 3.3–71× (conditional) → and for lifetime **×1.5–1.8** after marginalisation | figs 71→72 |
| lifetime 1σ ≈ 0.26% | 0.43% (calib corr) → **0.9–1.1%** (dE/dx marg) | figs 71→72 |
| E field floor 0.003% | 0.0004% — old scan outside the quadratic region | fig 71 |
| "lifetime objective-limited at ~2%" | retracted; objective prefers truth, error is reachability | fig 66 corr. |
| "TS1 converges by ~3 500 → 5.4 h variant free" | falsified: 4 000-it run is +10 µm / +378 loss | fig 72 |
| two-stage recipe as production | stage 1 only; stage 2 optional | fig 68 |

| A_b intrinsic bias +0.27% (6/6 seeds) | **the sample** — null (+0.06 ± 0.04) on 8 data replicas | fig 73 |
| "lifetime nearly statistics-limited" | per-sample only: on random samples the procedure is biased **+8.6 ± 1.6%** at the seed-0 truth, and sample-composition variation (spread 4.5%) dominates the budget | fig 73 |
| drift-profile term as a degeneracy breaker | null — conditional correlation untouched on both seeds | `schDP` |

**⚠️ 9 September addendum (fig 73):** `data_seed` was a silent no-op campaign-wide (IO pre-trim ran
before event selection), so **all previous fits share one track sample** — and that sample turns out
to sit at the 2.6σ *favourable* tail for lifetime. On 8 independent samples at pinned truth the
procedure shows a **+8.6 ± 1.6% lifetime bias**. The frontier is unchanged in kind (the degeneracy)
but the achieved-lifetime numbers quoted above inherit a sample-luck caveat until Arm B (a second
truth point) reports.
