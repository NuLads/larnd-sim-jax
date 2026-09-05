# What the last round of studies established, and what to do next

**Scope:** the work since the block matrix — `TS2` (fig 68), the fit→truth path (fig 69), the failed
profile (fig 70), and the calibration covariance (fig 71). One narrative, ranked by relevance, each
claim tied to its figure. Full detail: `NOISE_CAMPAIGN_REPORT.md` §6w.

---

## The one-paragraph summary

**The optimisation phase of this campaign is over, and it ended with a measurement, not a hunch.**
The path from our best fit to the truth is open — monotone downhill, no barrier — but the fits stop
short because the valley is ill-conditioned and the annealed learning rate runs out of step size
(fig 69). The remaining errors now come with real, independently-validated error bars (fig 71),
which expose the campaign's central degeneracy in one number: **recombination and lifetime are 71%
anticorrelated**. Everything actionable now points at one build — **second-order machinery** — which
simultaneously closes the measured-reachable gap (as a preconditioner) and delivers full error bars
(as a Schur complement). Meanwhile `TS1` — freeze calibration, re-anneal the nuisances, 13.5 h — is
the best-value configuration we have and should be adopted.

---

## Finding 1 — the gap to truth is open, and the optimiser is what stops short

![Full path](plots/noise_report/fig69_fullpath.png)

We evaluated the loss along the straight line from each converged fit to the truth, with
calibration, dE/dx and geometry all moving **together** (single blocks moved alone can make the loss
*worse*, so only the joint line answers this). The result, in every arm and every seed:

- **monotone downhill from the fit to truth** — no barrier, no shelf;
- **the minimum is AT truth**, not beyond — past α = 1 the loss climbs steeply. Along this
  direction the objective's optimum *is* the truth;
- the fits stop anyway, because the schedule anneals the learning rate to ~8 × 10⁻⁵ of peak: they
  **run out of step size while a downhill gradient still exists**. Shape (shallow near the fit,
  steepening toward truth) = an **ill-conditioned valley**.

**Relevance:** this converts "why is the fit stuck?" from speculation into a measurement. More
first-order iterations are *not* the answer — `G20K` already showed doubling them plateaus. A
**preconditioner** is.

Bonus: `TS1` closes **86–87%** of the baseline's gap to truth vs `G20K`'s 77–79% at half the
compute — independent confirmation on a measure neither was tuned against.

---

## Finding 2 — we now have real error bars, and they are validated

![Hessian](plots/noise_report/fig71_hessian.png)

The 5×5 calibration Hessian at the fitted point (finite differences of the gradient; two step sizes
agreeing to 7–11%). **Validation first:** where the old fig 58 loss-scan is trustworthy, the Hessian
reproduces it to **1–2%** (lifetime ratio 1.00). Where they disagree — E field, A_b — the *scan* is
at fault: those parameters are so tight that its ±5% grid sits 170–1 700 σ from the minimum, outside
the quadratic region. **The E field floor we had been quoting was 7× too large.**

The corrected picture (1σ floors with calibration correlations included, vs achieved at `BT2`):

| | 1σ floor | achieved | distance |
|---|---|---|---|
| A_b | 0.027% | 0.269% | ×10 |
| **E field** | **0.0004%** | 0.030% | **×71** |
| transverse diffusion | 0.44% | 1.46% | ×3.3 |
| longitudinal diffusion | 0.80% | 7.41% | ×9.3 |
| **lifetime** | **0.43%** | 1.65% | ×3.8 |

**Relevance:** (i) this supersedes the "4–16× from the floor" claim in the handoff — the true range
is **3.3–71×**, and E field, which we call "solved", is actually the furthest from what the data
supports. (ii) The floors are what a publication's error bars must be measured against. Caveat,
stated plainly: these are still **conditional** on geometry and dE/dx (lower bounds); the full
marginalisation is the Schur-complement build below.

---

## Finding 3 — the central degeneracy, finally quantified

Panel (b) of fig 71: the calibration correlation matrix, never measured before in this campaign.
**A_b ↔ lifetime = −0.71.** Both control how much charge reaches the pixels — A_b at production,
lifetime during drift — so the fit trades one against the other nearly freely. This single number
explains a season of observations: why lifetime co-adapts, why its errors are seed-dependent, why
every loss improvement seemed to come at its expense.

It costs exactly what it should: marginalising over the other calibration parameters inflates the
lifetime and A_b error bars by **×1.67** each — the two parameters joined by the −0.71 — so the
honest lifetime 1σ is **0.43%, not 0.26%**.

**Can the correlation be exploited?** We checked, honestly: knowing A_b *perfectly* would tighten
lifetime by 30% (0.43 → 0.30%), but a realistic external A_b constraint does almost nothing — even a
0.05%-wide prior only moves lifetime to 0.41%, because our own fit already determines A_b to 0.027%
conditionally. **The lever is not an external prior; it is breaking the degeneracy inside the
forward model** (e.g. anything that separates production-charge effects from drift-attenuation
effects, such as the drift-profile information already explored earlier in the campaign).

---

## Finding 4 — how *not* to measure uncertainties (kept because it is instructive)

![Profile](plots/noise_report/fig70_profile.png)

The textbook route — profile likelihood by re-fitting at pinned parameter values — **fails by a
factor of 2 105** here, and the noise budget shows it can never work in this setting: each profile
point is an independent stochastic re-optimisation of ~9 000 nuisance parameters, and where it
*lands* scatters more in loss (41 nats/batch between points) than the entire lifetime signal across
a ±10% grid (7.5 nats/batch). Averaging it away would need ~4 million times more sampling.

**Relevance:** this is why fig 71 uses curvature, and it is a general lesson for any differentiable-
simulation fit with a large nuisance space: **uncertainties must come from derivatives at the fitted
point, not from re-fitting.**

---

## Finding 5 — the production recipe question is settled (with one surprise)

![TS2 slopegraph](plots/noise_report/fig68_ts2.png)

The two-stage recipe (freeze calibration → re-anneal nuisances → release calibration):

- **Stage 1 (`TS1`) does essentially all the work**: loss 9 009 → 8 072, position 165 → 105 µm, at
  13.5 h — beating the 26.4 h `G20K` on the objective on 3/3 seeds. It also converges by iteration
  ~3 500, so a ~5.4 h variant likely exists.
- **Stage 2 (`TS2`) trades rather than wins**: it buys transverse diffusion (−0.93, 3/3) and gives
  back position (+2.7 µm, 0/3); against `G20K` at equal compute it wins the loss and loses
  transverse diffusion. **Treat stage 2 as optional.**

**Relevance:** `TS1` is the configuration to adopt and confirm at n = 6. Note the arms now differ by
*less than the seed-to-seed spread* — visible in fig 68's fanned lifetime panel — which is itself
evidence that squeezing the optimiser further is done.

---

## What was corrected along the way (so stale numbers don't circulate)

| old claim | corrected to | where |
|---|---|---|
| "every parameter 4–16× from its floor" | **3.3–71×**; E field floor was 7× too large | fig 71 |
| lifetime 1σ ≈ 0.26% (conditional scan) | **0.43%** once calibration correlations are included | fig 71 |
| "lifetime is objective-limited at ~2%" | retracted earlier: the objective **prefers truth**; the error is reachability | fig 66 (corrected) → fig 69 |
| two-stage recipe as production candidate | stage 1 only; stage 2 optional | fig 68 |

---

## The path forward, ranked

**1. Build the second-order machinery.** Everything above converges on it:
   - **Schur complement** of the full Hessian over the ~9 000 nuisances → full (not conditional)
     error bars. Reuses the CG/HVP machinery already written for the geometry block.
   - The same objects, used as a **preconditioner**, are the measured remedy for the
     ill-conditioned valley — the only route left to the 1.3–2.3% of loss still on the table.

**2. Adopt `TS1`; confirm at n = 6 and test the 4 000-iteration (~5.4 h) variant.** ~9 cheap jobs.

**3. Attack the A_b↔lifetime degeneracy in the forward model**, not with external priors (shown
   ineffective). The drift-profile direction from earlier in the campaign is the natural candidate,
   now with a quantified target: reduce |ρ| from 0.71 and the lifetime error bar shrinks with it.

**4. Longitudinal diffusion** remains the worst-recovered parameter (7.4%, floor 0.80%, ×9.3) and
   has never had a dedicated diagnosis.

*Not worth more effort:* longer runs, schedule variants, block orderings — the arms now differ by
less than seed scatter, and fig 69 shows why: the limit is conditioning, not effort.
