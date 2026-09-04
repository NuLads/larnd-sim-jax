# Differentiable LArTPC calibration — figure digest
### Every figure, ordered by argument rather than by date, with what each one tests and what it settles

**Companion to** `NOISE_CAMPAIGN_REPORT.md`, which carries the full discussion, the caveats and the
retractions. This version keeps all 36 figures and compresses the prose to two bullets each:
**Tests** — the question the figure was made to answer; **Concludes** — what it actually showed.

Ordering is logical, not chronological: the problem, why lifetime is intrinsically the hard
parameter, the fix, the checks, the ablations, the input-quality studies, and finally the external
cross-check that reframed the whole thing.

---

## Executive summary

- **Calibration is solved when the geometry is known.** With true positions and fitted per-segment
  dE/dx, all five parameters land within ~2% at the Hessian-predicted statistical floor.
- **The hard case is fitting geometry and calibration jointly.** For weeks this gave lifetime
  errors of +40 to +130%.
- **THE MINIMUM IS NOT MISPLACED, THE PRIOR IS (figs 46–48).** The likelihood prefers truth in 86%
  of batches; the priors prefer the fit in 100%. Geometry is unreached (optimiser-limited, 1.86×
  headroom left in the current basis); dE/dx is displaced (prior-limited). The dE/dx prior turns
  out to be **correct for the wrong segmentation**, and its weight is a tempering exponent whose
  principled value is w = 1/dx ≈ 1.01, not 5.
- **FREEZING CALIBRATION LETS THE NUISANCES DESCEND (fig 63).** BMC — every block warm-started from
  BT2, calibration frozen — takes the best loss (−3.25% vs BT2, −2.06% vs the re-anneal control),
  position (115.7 vs 165.3 µm) and dE/dx of any arm. Freezing GEOMETRY instead erases the descent:
  those arms sit within 0.1% of BT2.
- **A LOWER LOSS CAN MEAN A WORSE LIFETIME (fig 63).** A plain re-anneal beats BT2's loss by 1.21%
  while moving lifetime +3.6 further from truth, on both seeds. Selecting on the loss selects
  AGAINST lifetime accuracy.
- **AND THE OBJECTIVE STILL PREFERS TRUTH — SO IT IS REACHABILITY (fig 66).** With calibration,
  dE/dx and geometry ALL at truth the loss beats the fit on **7/7 arm-seeds by 5.6%**; the gap
  closes 9.4% (BT2) → 6.6% (BM0) → 2.5% (G20K). **Blocks must move together**: calibration alone at
  truth is WORSE (+0.18%), dE/dx alone worse still (+2.79%), all three together −5.6%. 1-D scans
  showing the fit at "the optimum" measure COUPLING, not bias.
- **BLOCK ALTERNATION IS UNTESTED — the fig 59 result was RETRACTED by fig 62.** The
  frozen-geometry pass runs only 3 000 iterations, so its calibration LR at readout is 0.059 of peak
  against BT2's 8.0e−5 (×700) and A_b still wanders ±0.13 vs BT2's ±0.001: the "×2.3" compared a
  frozen endpoint against a mid-oscillation sample, and all three arms sit within 0.4% on the
  objective. Fig 58's matrix is unaffected; acting on it is now open, not settled.
- **WE ARE NOT STATISTICS-LIMITED ANYWHERE (fig 58).** Every parameter sits 4–16× from its
  statistical floor. None of the remaining error is irreducible.
- **GEOMETRY IS OPTIMISER-LIMITED AND HOLDS MOST OF THE INFORMATION (figs 56, 58).** Its optimum is
  α* = 0.988 with no shrinkage, it is worth 5.1× the dE/dx question, and it is the dominant
  degeneracy — while A_b and E field are decoupled from both nuisance blocks.
- **AND ACTING ON IT IS THE LARGEST SINGLE WIN HERE (fig 61).** 20 000 iterations with both
  schedules stretched ×2: position **−57.2 ± 0.3 µm on 3/3 seeds (−34.7%)** and the **loss −8.7%**,
  plus transverse diffusion 2.61 → 0.61%. Same position as BT2 in 26% fewer iterations.
- **NEW BEST CONFIGURATION: BT2 (figs 54–55).** Production with the dE/dx prior REFIT to the
  segmentation in use and weighted at β = 1. A_b 0.77 → 0.27% (6/6 seeds, 5.6σ) and lifetime
  |error| 5.97 → 1.62%, better in both lever-arm strata, for ~13 µm of position. Correcting the
  weight ALONE is a regression, so the refit cannot be replaced by a knob.
- **THE dE/dx MEAN TARGET IS AN A_b SYSTEMATIC, NOT A LIFETIME LEVER (fig 53).** A_b moves
  −0.84 points per % of target error at weight 1e5 (47σ); lifetime does not move at all. The target
  must be known to 0.12% to keep A_b bias under 0.1 points.
- **THE DATA WANTS 66% OF THE TRUE dE/dx SPREAD (fig 52).** The likelihood's own optimum is α = 0.50,
  not truth — some shrinkage is correct. But the fit sits at 0 and no weight of the current prior
  reaches 0.50, because it is narrower than the data wants. Lower the weight (partial fix), don't
  remove it, refit the shape and use β = 1; keep the mean term at ~2500.
- **THE LIFETIME SEED SCATTER IS LEVER ARM, NOT NOISE (figs 49–51).** Charge swing runs 4.2–15.2%
  across seeds; split on it, annealing the dE/dx LR takes lifetime from 3.39% to **0.64%** where
  there is signal. Post-hoc split — a hypothesis for fresh seeds.
- **WHERE WE STAND (fig 45).** Best setup = S4 with an annealed calibration LR. Over 8 distinct
  seeds: A_b +0.73 ± 0.05%, E field −0.02 ± 0.01%, tran. diff. −0.72 ± 0.42%, long. diff.
  −7.86 ± 1.95%, lifetime −4.21 ± 1.53%, position 159 ± 4 µm. Three levers left (dE/dx prior
  5 → 0.5; more data with iterations scaled; ≥ 6 seeds), two hard limits (0.087-lifetime lever arm;
  ~45% dE/dx recovery ceiling).
- **The dominant cause was an optimizer-schedule artefact, not physics**: the calibration learning
  rate never annealed. Annealing it moves lifetime from +40 ± 25% to roughly −2 to −5%.
- **A_b and E field are done** — immune to every degradation tested. **Lifetime is the weakest
  parameter**, and long. diffusion the next weakest.
- **The objective is 91% joint hit log-intensity**, which supplies ~95% of the curvature pinning
  both lifetime and A_b. That term carries **both** the tick and the per-hit charge residual
  (σ_Q = 500 e⁻). *Correction: an earlier version of this bullet said the loss had no charge term —
  it does; the zero `log_likelihood_charge` field is a reporting stub. Lifetime has ONE structural
  handicap (lever arm), not two.*
- **Lifetime is intrinsically weak because our detector is only 0.087 lifetimes deep.** The
  anode-to-cathode charge swing is 8.3%. Any estimator leaning on charge *normalisation* rather
  than *shape* is ~25× levered on charge-scale errors.
- **Geometry error is not a bias channel** — not at 880 µm of random displacement, not at 1011 µm
  of systematic chord-cutting. It costs *variance*, not bias. Our 168 µm geometry is fine.
- **dE/dx error is a bias channel on 1-D slices**, monotonically. But the leverage does **not**
  transfer to the fits: our fitted dE/dx mean is accurate to ~0.1% and mispredicts fit lifetime in
  sign for five of nine seeds.
- **The dE/dx↔lifetime degeneracy is real but is not being exercised.** The fitted per-segment
  dE/dx profile versus depth is flat and matches truth in all 18 seed-points measured, despite a
  sloped input — the block corrects its input rather than faking attenuation. The correction is
  nonetheless worth ~10% of lifetime, so the two do trade.
- **The slice-vs-profile debt is now PAID for the key conditions (§6).** Profiling over A_b removes
  the apparent lifetime bias wherever the defect is a charge-normalisation error — the guess file
  goes **+13.9% → +0.4%** — and changes nothing for dE/dx spread or geometry. The ladder's rankings
  stand; the guess file's absolute displacement does not.
- **The production dE/dx prior weight is mis-set (§5).** `SCIDEDXPRIOR=5` vs the 0.5 default costs
  2.5× on dE/dx recovery, 4× on A_b, and moves long. diffusion from **−9.95% to +1.43%**. Under
  test in the production configuration.
- **The mean-constraint weight should stay at 1e5.** At ceiling, prior-corrected, 6 seeds, lowering
  it to 3000 makes A_b worse and inflates lifetime scatter 7×. The toy's 4× gain does not reproduce.
- **Readout noise costs ~20 points of dE/dx recovery but IMPROVES the calibration** (long. diffusion
  +1.4% with noise vs −13.9% without). Noise-ON is the correct configuration.
- **Batching order does not matter**; more data buys precision but starves geometry unless
  iterations scale with it. With iterations scaled, 2× data tightens every error bar and preserves
  geometry.
- **n = 3 cannot resolve ~4 points, and ~half of that is irreducible.** Run-to-run scatter at a
  FIXED seed and target is 2.22 points s.d. on lifetime (one seed spanned 6.35 points across three
  identical runs), so more seeds cannot get below it. Comparisons must be run at ceiling, and/or
  with repeats at fixed seed, and/or with `--non_deterministic` off.
- **Consolidated production performance (§8):** A_b **+0.73 ± 0.05%**, E field **−0.02 ± 0.01%**,
  tran. diff. −0.72 ± 0.42%, long. diff. −7.86 ± 1.95%, lifetime **−4.21 ± 1.53%**, position
  159 ± 4 µm — over 8 distinct seeds, repeats averaged within seed first.
- **The dE/dx minimum-length cut is a null** where it fires, and a no-op in production. And the
  dE/dx block **improves against truth** while appearing to degrade against its own starting point
  — the old "MAE gets worse" puzzle was the metric's reference.
- **The dE/dx block is objective-limited, not geometry-limited (§5).** Even with *perfect*
  geometry and noise ON it recovers only **49%** of the prior→truth gap (64% noiseless). Our recent
  arms reach 17.6% because they inherited a **10× stiffer dE/dx prior** (`SCIDEDXPRIOR=5` vs the
  0.5 default) from a fitted-geometry configuration — a configuration error, under test.
- **The standard exponential method is exact on truth (−1.2%) and collapses on real hits**
  (−14.5% to +123.5%), because per-hit charge conflates attenuation with diffusion sharing and
  threshold selection.

---

## §1 The problem: where the failure lives

![Stage ladder](plots/noise_report/fig1_stage_ladder.png)

**Tests** — a controlled ladder S1→S4 that adds one difficulty at a time: true geometry, then
fitted dE/dx, then wrong frozen geometry, then jointly fitted geometry.
**Concludes** — the failure is localised precisely: S1/S2 are at the statistical floor (~1–2%),
S3 (wrong frozen geometry) explodes to lifetime +109%, S4 (fitted geometry) to +68%. Everything
downstream is about the S3/S4 step, not about the simulator or the loss.

![S4 variants](plots/noise_report/fig2_s4_variants.png)

**Tests** — every S4 configuration tried across the campaign, on one axis.
**Concludes** — no combination of priors, weights or basis choices rescued S4; the spread across
variants was large and unsystematic, which is what eventually pointed at the schedule rather than
at any single physics term.

![Trajectories](plots/noise_report/fig3_trajectories.png)

**Tests** — what the parameter trajectories actually do, rather than where they end.
**Concludes** — the baseline and the former "winner" **do not oscillate around truth; they perform
a biased random walk that occasionally crosses it**. Reading such a run at its endpoint is
meaningless — this is the observation that forced the robust (median-of-tail) estimator.

---

## §2 Why lifetime is the hard parameter

![Lever arm and the standard method](plots/noise_report/fig34_lifetime_standard_method.png)

**Tests** — the physics objection that lifetime is a pure charge-vs-drift-*time* effect, so a
global dE/dx scale error should move the intercept and leave the slope untouched.
**Concludes** — correct, and it explains everything. **The detector is 190.8 µs deep against a
2200 µs lifetime: 0.087 lifetimes, an 8.3% charge swing.** Charge-weighted ⟨t/τ⟩ = 0.0391, so a
normalisation-only estimator carries a **25.6× leverage** on charge-scale errors — matching the
24.3× measured on the scans. A shape-based fit gives τ = 2173 ± 4 µs and is unchanged to four
significant figures under a ±2% rescale. **The leverage is a property of our lever arm, not of
dE/dx.**

![Hessian spectrum](plots/noise_report/fig5_hessian.png)

**Tests** — the curvature structure of the objective at the solution.
**Concludes** — a clean stiff/soft split. A_b and E field sit in stiff directions and are
well-determined; lifetime and long. diffusion live in soft directions. This is the structural
reason the same fit nails two parameters and struggles with two others.

![Mode decomposition](plots/noise_report/fig11_mode_decomposition.png)

**Tests** — which parameter combinations the soft directions actually correspond to.
**Concludes** — **the two soft modes are mixed**, not aligned with single parameters. Lifetime is
not individually soft; a *combination* involving lifetime and long. diffusion is.

![Soft plane](plots/noise_report/fig12_soft_plane.png)

**Tests** — the joint uncertainty in the (lifetime, long. diffusion) plane.
**Concludes** — the ellipse is strongly elongated along the soft diagonal: the fit can slide a long
way along that direction at almost no cost in loss. Any residual in one parameter is expected to
be correlated with the other, which is why they must be read together.

![Dynamic modes](plots/noise_report/fig13_dynamic_modes.png)

**Tests** — whether the *static* Hessian modes describe where the optimizer actually moves.
**Concludes** — they do not, fully. The realised trajectory has substantial overlap with the static
soft plane but is not confined to it, so a static-curvature argument alone cannot predict the
failure mode.

![Drift decomposition](plots/noise_report/fig14_drift_decomposition.png)

**Tests** — decomposing the fitted geometry displacement along the drift axis versus in the wire
plane.
**Concludes** — the fitted displacement is **231 µm along drift, 354 µm in the wire plane, with the
drift-projected part a median 58% of the total**. Since drift position and charge attenuation are
degenerate, this is the concrete mechanism by which geometry error can masquerade as lifetime.

---

## §3 The fix: anneal the calibration learning rate

![LR schedules](plots/noise_report/fig16_lr_schedules.png)

**Tests** — what the calibration learning rate actually did over a run.
**Concludes** — it never annealed. Physics parameters and the ~4000 dE/dx nuisances were updated at
the wrong relative "temperature" for the entire fit. This is the single most important finding in
the campaign.

![ANNEAL recovery](plots/noise_report/fig7_anneal_recovery.png)

**Tests** — the effect of annealing the calibration LR, all else equal.
**Concludes** — lifetime moves from +40 ± 25% to roughly −2 to −5%; the other four parameters
tighten as well. The largest single improvement obtained.

![ANNEAL parameter traces](plots/noise_report/fig8_anneal_param_traces.png)

**Tests** — whether the annealed run settles or is still walking.
**Concludes** — four of five parameters flatten into a genuine plateau; lifetime continues to
drift, which is the seed of the "is it converged or merely quenched?" question in §4.

![Position and dE/dx vs iteration](plots/noise_report/fig9_anneal_pos_dedx.png)

**Tests** — the nuisance blocks' behaviour during the annealed fit.
**Concludes** — geometry improves steadily to ~168 µm, but the per-segment dE/dx MAE **ends worse
than it starts**, absorbing signal during the phase when calibration moves fastest.

![Gap closed](plots/noise_report/fig10_gap_closed.png)

**Tests** — a per-seed accounting of how much of each seed's initial offset was actually removed,
rather than the raw final error.
**Concludes** — the fairer metric, because targets are drawn per seed and some start close to
truth. Five of six seeds close ≥64% of their initial lifetime offset, so the claim does not rest
on one lucky seed.

![Reproducibility at n = 6](plots/noise_report/fig15_annealmore_n6.png)

**Tests** — three fresh seeds on top of the original three.
**Concludes** — the annealed result reproduces at n = 6. A_b, E field and transverse diffusion are
convincingly recovered on all seeds; long. diffusion remains the weakest claim.

---

## §4 Is the annealed answer converged, or just quenched?

![ANNEALLONG traces](plots/noise_report/fig17_anneallong_traces.png)

**Tests** — extending the horizon from 5 000 to 10 000 iterations.
**Concludes** — four of five parameters are flat and stable; lifetime is still moving at read-out.

![ANNEALLONG convergence](plots/noise_report/fig18_anneallong_convergence.png)

**Tests** — a quantitative convergence criterion (tail drift versus the run-to-run noise floor).
**Concludes** — A_b, E field and both diffusions are *settled*; lifetime is *moving*. Its quoted
value is a lower bound on the bias, not a converged estimate.

![ANNEALLONG position / dE/dx / loss](plots/noise_report/fig19_anneallong_aux.png)

**Tests** — the auxiliary quantities that had not converged at 5 000 iterations.
**Concludes** — position and loss do settle by 10 000; the dE/dx MAE degradation persists and is
still unexplained.
**SUPERSEDED on the position half — see fig 61.** Position "settles" only *because the schedule
cools it*. G20K stretches the schedule ×2 and position keeps descending well past 10 000, reaching
110 µm against 166–177 µm here, still falling at 15 000. The flattening was the learning rate
running out, not the optimiser arriving. The loss half stands and in fact strengthens it: the NLL
also drops a further 7.2%.

![Schedule invariance](plots/noise_report/fig20_schedule_invariance.png)

**Tests** — the decisive question: is the answer a property of the objective, or of how hard the
optimizer worked? Compares runs at matched total optimizer travel ΣLR.
**Concludes** — **lifetime tracks ΣLR and is therefore schedule-dependent** (−2.01% → −2.70% →
−5.02% as travel increases). The other four parameters are schedule-invariant. This is why lifetime
alone is quoted as a bound.

![SLOWANNEAL traces](plots/noise_report/fig21_slowanneal_traces.png)
![SLOWANNEAL convergence](plots/noise_report/fig22_slowanneal_convergence.png)

**Tests** — a slower anneal (more total travel) as the extreme point of the ΣLR series.
**Concludes** — confirms the trend. **The honest summary is lifetime −2% to −5%, not converged**,
with the other four converged and schedule-invariant.

---

## §5 What each block actually contributes

![Global comparison of the annealed family](plots/noise_report/fig23_ablation_global.png)

**Tests** — two ablations against the annealed baseline: a probabilistically-drawn target
(PROBTGT) and dE/dx frozen at the mean (CONSTDEDX).
**Concludes** — **PROBTGT changes nothing measurable**: the deterministic target is not a bias
source, retiring that open question. **CONSTDEDX degrades long. diffusion ~4× and lifetime ~2.7×**:
the dE/dx block is doing real work, not merely adding freedom.

![dE/dx ablation trajectories](plots/noise_report/fig24_ablation_traces.png)

**Tests** — *when* during the fit the dE/dx block earns its keep.
**Concludes** — all arms track each other for ~3000 iterations, then CONSTDEDX separates downward
onto a distinctly worse plateau. The block matters in the late, fine-tuning phase.

![What the ablation costs elsewhere](plots/noise_report/fig25_ablation_cost.png)

**Tests** — whether freezing dE/dx also damages the geometry.
**Concludes** — **it does not**. The position residual is statistically unchanged across all four
arms (158 ± 9 to 173 ± 6 µm). The dE/dx block's effect is confined to the calibration parameters.

![Three single-knob ceiling comparisons](plots/noise_report/fig41_ceiling_knobs.png)

**Tests** — three controlled ceiling comparisons, each differing from its control in exactly one
knob: the dE/dx **prior** weight (5 → 0.5), the dE/dx **mean-constraint** weight (1e5 → 3000, at
the corrected prior), and **readout noise** (ON → OFF). Ceiling removes the geometry variance that
otherwise swamps single-knob differences.
**Concludes (a)** — **the production dE/dx prior weight is badly mis-set.** `SCIDEDXPRIOR=5`
(ANNEALLONG's value) versus the 0.5 default costs: A_b +0.77 → **+0.19**, lifetime −4.31 → **−2.32**
and 3× tighter, and long. diffusion **−9.95% → +1.43%** — the largest single gain in that parameter
anywhere in the campaign. dE/dx recovery goes 17.6% → **44.5%**, back into the historical band.
**Concludes (b)** — **lowering the mean-constraint weight does not help.** At the corrected prior
with 6 seeds, `w = 3000` makes A_b *worse* (+0.19 → +0.71) and inflates lifetime scatter 7×
(±0.69 → ±5.21). The second-order study's top recommendation does not reproduce. **Keep 1e5.**
**Concludes (c)** — **noise costs the dE/dx block but helps the calibration.** Recovery falls
64.3% → 44.5% with noise on, yet long. diffusion is **+1.43% with noise vs −13.94% without**. A fit
that never sees noise mis-assigns the smearing to diffusion. **Noise-ON is correct**, and the
noiseless historical numbers are not an achievable target.

![What the dE/dx block recovers with perfect geometry](plots/noise_report/fig39_ceiling_dedx_recovery.png)

**Tests** — the ceiling question: with **true geometry**, how much of the per-segment dE/dx can the
block actually recover? Every ceiling run is readable because there the sim input *is* the target,
so the old `dedx_mae_iter` coincides with a truth error. And all runs start from the same place —
the parameters are initialised at the prior centre, using no truth information — so "gap closed" is
well defined.
**Concludes** — **even with perfect geometry the block recovers only 49% of the prior→truth gap
with noise ON** (64% with noise off). So it is **not geometry-limited** — the remaining error is
intrinsic to the objective, the same verdict the second-order study reached from curvature.
**Two effects, not one** — the 17–64% spread splits cleanly: **noise costs ~14 points** (noiseless
53–64% vs noise-ON 40–49%, all at `prior_w = 0.5`), and the **dE/dx prior weight costs ~25 more**.
Against the matched noise-ON subset the historical band is 40–49% while our recent arms reach
17.6%, because they inherited `SCIDEDXPRIOR=5` from the ANNEALLONG *fitted-geometry* config against
the 0.5 default. `ceil_p05` is the clean single-variable test.
**Correction** — an earlier version claimed the historical prior weight was unrecoverable (those
runs predate the `provenance` block) and attributed the whole spread to the prior. Both wrong: the
full argparse `Namespace` is in every checkpoint under `config`, and most of the spread is noise.

![Does the fitted dE/dx develop a drift profile?](plots/noise_report/fig37_dedx_drift_profile.png)

**Tests** — the sharpest available check on the **dE/dx↔lifetime degeneracy**. Every segment sits
at a definite drift coordinate, so a drift-correlated pattern in the fitted per-segment dE/dx is
observationally almost identical to a lifetime change, and the ~4000 nuisances are free to produce
one. Reconstructed offline from `chain_contexts` + `batch_parent_ids` + `dedx_cache`, so it reads
archived runs without re-running anything. The statistic is the one
`--dedx_drift_profile_weight` minimises: `cov_w(|z|, log dEdx)/√var_w(|z|)`.
**Concludes** — **the degeneracy is not being exercised.** The fitted profile tracks truth in all
**18 seed-points across six configurations**, even though every run starts from an input carrying
**+0.0032**. Panel (a) shows it directly: the straight-line guess has a ~1.1% swing in log dE/dx
across the drift, the fit removes it, and the result follows truth. The block is correcting a real
defect in its input, not manufacturing a fake one. On the one run where the subset's own truth is
measurable, fitted **+0.0011** against subset-truth **+0.0012** — essentially exact.
**Why the guess is sloped at all** — the trend is **entirely between tracks (+0.000972/cm) and
exactly zero within a track**. So it is not residual uncorrected attenuation, which would act
per-segment. It is a per-track normalisation: dE/dx is nearly constant along each track (spread
0.043 vs truth's 0.107) and tracks at greater mean depth get a uniformly lower value — what you get
from dividing a track's total collected charge by its length without a lifetime correction.
**Reference caveat** — the whole true file gives +0.0002, but production uses 0.4% of it and that
subset's own trend is **+0.0012**, 6× higher. Only `MAECHECK` has the matched truth needed to check
this properly; for the archived runs the whole-file value is a proxy of unknown quality, so their
agreement is suggestive rather than established.
**But it does not exonerate the mechanism** — that correction is worth **~−10% of lifetime** in
exactly the degenerate direction, so dE/dx and lifetime *are* trading at a magnitude comparable to
the residual lifetime bias; here the trade happens to move toward the right answer. It also argues
**against** switching on `dpw=1e6` in this configuration: the penalty drives the trend toward zero
*value*, which these runs already have, so it would only fight the legitimate correction. That
likely explains why the earlier dpw success came from the 400 cm *ceiling* config, whose input is
the true file and already flat.
**Not readable** — the `ceiling_400_dpw1e6/1e7` arms, where the degeneracy was originally
demonstrated (lifetime +18.5 ± 18.9% with fitted dE/dx vs −1.5 ± 1.9% with true), freeze positions
and therefore store no geometry; CONSTDEDX freezes dE/dx and stores no `dedx_cache`. Both are
structurally unreadable, so the one configuration known to exercise the degeneracy is the one that
cannot be retro-diagnosed.

---

## §6 Where is the loss minimum? (objective versus optimizer)

![Likelihood scans](plots/noise_report/fig26_likelihood_scans.png)
![Scan zoom on lifetime and long. diffusion](plots/noise_report/fig27_scan_zoom.png)

**Tests** — evaluating the loss on a grid instead of fitting, to separate "the optimizer has not
arrived" from "the minimum is genuinely displaced". No optimizer is involved.
**Concludes** — with true geometry the minimum sits close to truth; with the straight-line guess it
is badly displaced (lifetime +15.2%, long. diffusion −34.1%) while A_b and E field barely move.
This is the §2 stiff/soft asymmetry measured directly on the objective rather than inferred from
curvature, and it explains the S3 stage mechanistically.
**Caveat that matters** — these are 1-D *slices* with the other four parameters frozen, not profile
likelihoods. §2 and §11 explain why that distinction turned out to be decisive.

![2-D scans: slice vs profile](plots/noise_report/fig40_scan2d_profile.png)

**Tests** — the fix for exactly that caveat. `LARND_SCAN_2D` scans (lifetime, A_b) jointly, so
minimising over A_b at each lifetime *is* the profile likelihood — no minimiser needed. Five
conditions, 21×21 grids over 21 batches.
**Concludes** — **profiling matters only where the defect is a charge-NORMALISATION error.**
The straight-line guess collapses from **+13.87% to +0.43%** (a factor 32) and the dE/dx mean +2%
rung from −15.00% to +4.23%. Where the defect is a dE/dx *spread* change (f=0.40: +4.96%) or a
*geometry* error (pos 880 µm: −4.78%), A_b sits at nominal and slice = profile exactly. A_b absorbs
normalisation and nothing else.
**Consequence** — the ladder's **rankings survive** (spread and geometry are unaffected), but the
guess file's absolute displacement does not. **Open question 11 dissolves: there was no unexplained
+9 points, because there was no +15.2% to explain.** It agrees with the fit-side test, where a 4%
dE/dx mean swing moved lifetime 0.8 ± 4.7 points against a 97-point slice prediction — objective
and optimiser now say the same thing.

---

## §7 How good must the inputs be? The quality ladder

![Scan wells along both ladders](plots/noise_report/fig28_ladder_scans.png)

**Tests** — controlled degradations of the *true* file along two axes separately: rigid position
error (50/170/400/880 µm) and dE/dx spread blended toward the mean (f = 0.75/0.5/0.25/0).
**Concludes** — the two axes behave completely differently. The dE/dx wells *slide* in an ordered
fan; the position wells *flatten* without moving.

![The calibration curve](plots/noise_report/fig29_ladder_calibration.png)

**Tests** — minimum location versus input quality, for all five parameters at once.
**Concludes** — dE/dx degradation moves lifetime and long. diffusion monotonically; position
degradation moves nothing significantly. A_b and E field are immune on both axes.

![Bias versus variance](plots/noise_report/fig30_bias_vs_variance.png)

**Tests** — the decomposition: does a degradation shift the centre or widen the uncertainty?
**Concludes** — **dE/dx error is a bias problem; position error is a variance problem.** Position
inflates long. diffusion's uncertainty 4.1 → 40.9 points across the ladder with no significant
centre shift. Our 168 µm geometry sits comfortably inside the useful range.

![The dE/dx mean is the dominant slice-level bias channel](plots/noise_report/fig31_dedx_mean_leverage.png)

**Tests** — the axis the ladder had missed: a dE/dx *mean* error (whole distribution scaled) rather
than a *spread* error.
**Concludes** — on 1-D slices this dominates everything: **−24.3 percentage points of lifetime per
1% of mean**, against +5.9 points for destroying 60% of the spread and nothing for geometry. It
also **closes the guess-file mystery**: that file's length-weighted dE/dx mean is −0.440% low,
predicting +12.9% against +15.2 ± 4.9% observed.
**Correction** — the −24.3 is the reciprocal of the drift lever arm (§2), not a property of dE/dx,
and it does **not** transfer to the fits. Our fitted mean is accurate to ~0.1% and mispredicts fit
lifetime in sign for five of nine seeds.

![The completed quality ladder](plots/noise_report/fig32_ladder_complete.png)

**Tests** — every degradation axis on one footing, including re-segmentation, systematic
chord-cutting, and a combined chord+dE/dx file.
**Concludes** — **segmentation is null** (a re-segmented true file is statistically identical to
truth); **systematic chord-cutting is null** (fully straightened tracks, 1011 µm mean deviation,
give lifetime −0.99 ± 3.29%); **there is no interaction** between the geometry and dE/dx axes. Only
dE/dx moves the slice minimum, and geometry survived a test built to break it.

---

## §8 Fit-side arms: batching order and data volume

![Fit-side arms](plots/noise_report/fig33_fit_arms.png)

**Tests** — two questions on the fits themselves: does batching order bias the result
(`--shuffle_bt` versus sequential, submitted together on one tree), and does 4× the data help?
**Concludes** — **order does not matter**: every parameter agrees within errors, largest gap 1.4σ,
position 492 vs 475 µm. **More data buys precision and starves geometry**: long. diffusion's seed
spread tightens 23.2 → 13.7 while position degrades 492 → 774 µm, because the geometry block gets
4× fewer updates per track at fixed iteration count.
**Read the caption** — only three of the four arms share `chain_decay_rate = 0.999`; ANNEALLONG
used 0.9997, which leaves the chain LR 1100× higher at iteration 10 000, so its geometry is not
comparable to the others.

![PPP loss decomposed](plots/noise_report/fig44_loss_components.png)

**Tests** — what the objective is actually made of. Every checkpoint stores a per-iteration
`aux_iter` record; the decomposition closes to 4.5e-7 relative, so it is exact. Panels (b,c) apply
it to the 1-D scans to ask which term supplies the curvature that pins each parameter.
**Concludes — the loss is 91% joint hit log-intensity**, plus dE/dx prior (−22%), the PPP count
integral (+17%) and the unmatched-target penalty (+14%). MCS and the dE/dx mean constraint are
~0.1% each.
**Concludes — that joint term contains BOTH tick and charge.** `ProbabilisticLossStrategy` adds a
Gaussian per-hit charge residual (σ_Q = 500 e⁻) to the tick log-probability before the `logsumexp`,
and stores the result under the mislabelled key `log_likelihood_tick`. The separate
`log_likelihood_charge = 0.0` field is a **reporting stub**, not a disabled term. Verified by
rerunning an identical scan at σ_Q = 50 e⁻: the loss moves **99.3%**, entirely inside that key,
while the charge-independent `expected_total_hits` moves 1e-4.
*Correction (2026-08-06): an earlier version of this entry concluded there was no charge term. Wrong
— withdrawn.*
**Concludes — the joint hit term supplies ~95% of the curvature** for both lifetime (73 887 vs
3 719) and A_b (30 604 vs 2 089); for lifetime the count integral is nearly flat near the minimum.
**So lifetime has ONE structural handicap** — the 0.087-lifetime lever arm (§2). The open question
is the *value* of σ_Q: 500 e⁻ is 4.7% of a median hit, comparable to the whole 8.3% lifetime
signal, and it has never been derived or scanned.

![Production convergence traces](plots/noise_report/fig43_convergence_traces.png)

**Tests** — what the production fit is doing at every iteration, rather than where it ends: the
five calibration parameters, the dE/dx MAE against *both* references, the position residual and the
loss. All 8 completed runs, 200-step rolling median.
**Concludes** — **A_b and E field settle by ~2000 iterations** and stay flat; transverse diffusion
converges cleanly. **Longitudinal diffusion overshoots to +40%** around iteration 800 before
returning through zero to ~−7% — the dE/dx co-adaptation, and why early read-outs of it were
meaningless. **Lifetime never flattens**: it crosses zero near iteration 4000 and drifts to the end,
which is the schedule-dependence of §4 seen directly and the reason it is quoted as a bound.
**And the metric bug in one picture** — the dE/dx MAE *rises* 0.0315 → 0.0353 against the sim input
while *falling* 0.0713 → 0.0584 against truth. Same fit, opposite conclusions.

![Every completed production run](plots/noise_report/fig42_production_runs.png)

**Tests** — the consolidated production number, and how it should be pooled. Three directories hold
the *same* configuration, giving 14 completed runs across only **8 distinct seeds** (0–2 appear
three times). `--seed` draws the target, so those repeats re-solve an identical problem and are not
independent samples.
**Concludes — the production result**, averaging repeats within a seed first, then over 8 seeds:
**A_b +0.73 ± 0.05%**, **E field −0.02 ± 0.01%**, tran. diff. −0.72 ± 0.42%, long. diff.
−7.86 ± 1.95%, **lifetime −4.21 ± 1.53%**, position 159 ± 4 µm.
**Concludes — half the scatter is bit-level irreproducibility.** Seed 0 gave −4.27 / −8.33 / −1.99%
on three runs of an identical config against an identical target — **6.35 points**, from GPU
non-determinism alone. Within-seed s.d. is 2.22 points against a between-seed 4.33, so **adding
seeds cannot get lifetime below ~2.2 points**. Comparisons finer than that need repeats at fixed
seed, or `--non_deterministic` off.
**And it is not target difficulty** — targets span 984–4901 µs with no trend (worst run at
τ = 3640; τ = 4901 gives +0.5%).

![Four arrays: the cut, the power floor, the variance, the metric](plots/noise_report/fig38_aug5_arrays.png)

**(a) Tests** — the dE/dx minimum-length cut (`LARND_DEDX_MIN_DX = 0.15`), in ceiling mode, the
only configuration where it fires at all. In production the guess file is uniformly re-segmented at
~1 cm (shortest segment 0.909 cm), so the cut is a measured **no-op** there.
**Concludes** — **a clean null.** Every parameter matches between `ceil_base` and `ceil_mdx`
(A_b +0.77 ± 0.01 vs +0.77 ± 0.03; lifetime −4.31 ± 1.94 vs −4.19 ± 1.91). Paired seeds and no
geometry variance make the error bars tight, so this is a well-powered negative, not an
inconclusive one. **The previously reported −0.97 ± 0.11 calibration gain does not reproduce.**

**(b) Tests** — reproducibility, by running the *same* configuration three times: ANNEALLONG, its
rerun `annl2`, and `mdx_full` (a no-op variant).
**Concludes** — lifetime means of **−2.70 ± 1.89, −5.07 ± 2.44, −1.44 ± 1.63** — a **3.6-point
spread between identical configurations**. That is the resolution floor of any n = 3 comparison,
measured rather than argued, and it is larger than most effects the campaign has chased. It also
confirms the current tree reproduces ANNEALLONG on every parameter.

**(c) Tests** — where the seed-to-seed variance actually lives.
**Concludes** — geometry owns it. Freezing positions at truth shrinks the seed sd by **11× on A_b**
(0.14 → 0.013) and **70× on E field**. Doubling the data with iterations scaled alongside helps too
(A_b 0.14 → 0.06, tran. diff. 0.67 → 0.17) **without** the position degradation seen when batches
were raised at fixed iteration count — 172 µm against 162 µm, versus NB400FIX's 774 µm.

**(d) Tests** — the per-segment dE/dx MAE against its two possible references, at production length.
**Concludes** — the block **improves 17% against truth** (0.0714 → 0.0590) while appearing to
**degrade 10% against its own starting point** (0.0316 → 0.0349). The long-standing "the MAE ends
worse than it starts" puzzle was the reference, not the fit. Free validation: in ceiling mode the
sim input *is* the target, so the two curves must coincide — and they do to four decimals, which
independently checks the arc-length matcher.

---

## §9 External cross-check: the standard method on real hits

![Standard fit on simulated hits](plots/noise_report/fig35_lifetime_hits.png)

**Tests** — the same shape-based exponential fit that is exact on truth (§2), applied to the hits
the simulation actually produces: ADC counts and tick numbers, with threshold, ADC response and
readout noise.
**Concludes** — **it collapses**. τ spans **−14.5% to +123.5% depending only on where the fit is
started**, with errors 20–150× larger than the truth-level ±4 µs. The noise-OFF control separates
the causes: the anode-edge artefact is **purely geometric** (−14.3% vs −14.5%, identical), the
threshold alone costs **+15 to +29%**, and **noise adds a further +11 to +15 points** by raising
the median hit charge 7.9% (9.92 → 10.70 ke) — the Eddington signature, large because the cut sits
only ~2× below the median rather than out in a tail.

![The two front-end mechanisms](plots/noise_report/fig36_hit_mechanisms.png)

**Tests** — the mechanisms directly, on the same hits.
**Concludes** — (a) the spectrum slides down into a fixed 5 ke cut, so survivors at long drift are
biased high; (b) **charge sharing**: hits per pixel climb 1.70 → 2.20 while mean Q per hit falls
14.1 → 12.5 ke over the same 25 µs — the charge is divided, not lost; (c) the sample carries **3.6×
fewer hits at the cathode**, so the fit is weighted away from where the signal lives; (d) summing
charge **per pixel** undoes the sharing and cuts the bias from +43% to −12.5%, while a naive charge
floor makes it *worse* (+43% → +55%).
**The lesson** — per-hit charge is the wrong observable, and both systematics exceed the entire
8.3% attenuation signal. This is an argument **for** the forward-model fit, which has the threshold
and the diffusion in the model, and against hand-rolled estimators that inherit them uncontrolled.

---

## §10 Methodology: the traps that produced false results

![Traps](plots/noise_report/fig6_traps.png)

**Tests** — a systematic audit of all 105 campaign runs with an automated plateau detector.
**Concludes** — **80% were still moving when they were read out.** Reading endpoints of
unconverged fits produced multiple false positives. This is why every number in these documents
uses a median over the tail plus an explicit drift diagnostic.

![Plateau audit](plots/noise_report/fig4_plateau_audit.png)

**Tests** — among runs that *did* plateau, how much does the choice of estimator matter?
**Concludes** — up to **70 percentage points** depending on endpoint versus plateau. Every
cross-variant comparison in the campaign had to be recomputed on a consistent estimator.

**Traps found since, not shown as figures but equally load-bearing:**

- **The LR schedule is indexed in epochs, not iterations.** Raising `max_nbatch` at fixed
  iterations silently un-anneals the calibration LR (1250× at 4× batches) — it reverts the
  campaign's most important fix while looking like a clean scale-up.
- **`git_dirty=True` means the recorded SHA does not pin the code.** A cross-day comparison between
  two dirty trees produced a confident, wholly wrong conclusion that batch shuffling destroys the
  geometry fit.
- **The geometry update was gated on a global iteration counter.** At `chain_update_freq > 1` a
  fixed subset of batches received *every* update and the rest none, ever (50/100 batches at
  freq = 2, 90/100 at freq = 10). Fixed; inert at the default of 1.
- **Per-parameter scan history files are snapshots of one growing history**, not interchangeable
  copies — reading the wrong one silently drops a batch.
- **Scan "seeds" are not independent** under `--probabilistic_sim`; error bars built from seed
  scatter are meaningless.
- **Read the stored config before reasoning about what a run used.** Every history pickle carries
  the full argparse `Namespace` under a `config` key — including runs that predate the
  `provenance` block. When a discrepancy surfaced between recent and historical ceiling runs, the
  old ones were declared unreadable and an attribution was built from script defaults plus a log
  survey instead; one lookup would have settled it, and the inference was partly wrong — most of
  the spread turned out to be a *noise* effect, not the prior.
- **A config knob inherited from a different regime silently changed the answer.** Every recent
  ceiling arm ran at `SCIDEDXPRIOR=5`, copied from the ANNEALLONG *fitted-geometry* config against
  the **0.5** default. Against matched noise-ON runs it cut dE/dx recovery from 40–49% to 17.6%,
  and **entangled** the `ceil_w3k` mean-constraint test. *When reusing a configuration across
  regimes, diff it against the script defaults, not only against the previous run.*
- **A cut can be a silent no-op on one input file and active on another.** `LARND_DEDX_MIN_DX=0.15`
  freezes 4% of segments on the true file and **nothing at all** on the straight-line guess, whose
  shortest segment is 0.909 cm. A recommendation measured in one regime need not even *execute* in
  another.
- **`n = 3` is below the resolution floor.** Three runs of an identical configuration span 3.6
  points on lifetime. Any A/B smaller than that is unmeasurable at this sample size.

---


![The state of play](plots/noise_report/fig45_state_of_play.png)

**Tests** — the campaign's bottom line, assembled from arms already documented rather than newly
fitted, so the conclusion can be checked instead of trusted. (a) production vs ceiling per
parameter; (b–d) the three levers that measurably improve it; (e) everything tested that does not;
(f) the two hard limits.
**Concludes — the best setup is S4 with an annealed calibration LR**: 100 batches × 400 cm,
10 000 iterations, spline geometry, dE/dx prior 5, mean constraint 1e5, noise ON. Over **8 distinct
seeds** (14 runs, repeats averaged within seed): A_b **+0.73 ± 0.05%**, E field **−0.02 ± 0.01%**,
tran. diff. −0.72 ± 0.42%, long. diff. −7.86 ± 1.95%, lifetime **−4.21 ± 1.53%**, position
**159 ± 4 µm**. A_b and E field are solved; lifetime remains a bound, not a measurement.
**Concludes — three levers, in order of gain.** (1) dE/dx prior **5 → 0.5**: at ceiling it moves
long. diffusion −9.95% → +1.43% and 2.5× the dE/dx recovery — production is mis-set and the fix is
free. (2) **More data with iterations scaled alongside**: seed-matched, every central value moves
toward truth and position holds (159 → 165 µm); more batches at *fixed* iterations does the
opposite (774 µm) because the geometry block starves. (3) **≥ 6 seeds, knobs settled at ceiling**.
**Concludes — four things not to do**: chase geometry accuracy (a variance lever, not a bias
lever), the dE/dx min-length cut (a null), mean-constraint weight 3000 (SUPERSEDED — see fig 53; the 7.5× the
lifetime scatter), noise OFF (+5.4 points of bias).
**Concludes — two hard limits.** The detector is **0.087 lifetimes deep** — an 8.3% anode-to-cathode
swing is the entire lifetime signal — and the dE/dx block recovers at most **~45%** of the
prior→truth gap even with perfect geometry. Neither is a tuning problem.
**Read the caption on (c) and (e).** (c) is seed-matched because `nb200` has only 2 completed seeds;
(e) plots bias *and* scatter because the mean-constraint arm looks better on central values alone
purely through a lifetime value with a ±2.1 error bar.


![Truth point vs fit point](plots/noise_report/fig46_truthpoint.png)

**Tests** — study 2: is the loss minimum misplaced, or merely unreachable? 6 seeds × 100 batches,
no fitting: the objective is evaluated at the fit point, the truth point, and three one-block
hybrids.
**Concludes — the total objective gives the WRONG answer on its own.** Truth loses on the total
(45% of batches, +4.09%) but wins on the LIKELIHOOD in **86%**, while the priors prefer the fit in
**100%**.
**Concludes — the two blocks fail for OPPOSITE reasons.** Geometry → truth: −213 likelihood, −1
prior (unreached, an optimiser problem). dE/dx → truth: −10 likelihood, **+565 prior** (displaced,
an objective problem). Calibration is already at its likelihood optimum (+18).
**Concludes — the spline basis is not the geometry limit**: the truth projection has a median
residual of **85 µm** against the 159 µm production achieves — 1.86× of headroom inside the current
parameterisation.

![Loss components in detail](plots/noise_report/fig47_truthpoint_components.png)

**Tests** — the same evaluation with all six non-zero terms kept separate.
**Concludes — `no_match_penalty` carries a third of the geometry signal** (239.5 → 110.5, better in
79% of batches) while the hit log-intensity's geometry gain is concentrated in a minority. Better
geometry mostly buys hits *existing* where the target says, not better-placed hits.
**Concludes — the dE/dx mean penalty rises at truth because of SPREAD, not offset**: the fit's
batch mean is pinned in a 0.0033 IQR, the true one varies over 0.021.
**Read the caption** — per-term medians do NOT sum to the median total; the waterfall is drawn on
the single batch at the median total Δ, where it is exact.

![The dE/dx prior](plots/noise_report/fig48_dedx_prior_shrinkage.png)

**Tests** — the prior's own parameters against the fitted and true fields, with no loss evaluations
at all. An independent line of evidence.
**Concludes — the prior is CORRECT FOR THE WRONG SEGMENTATION.** Its width (0.1359) is the MLE
optimum on the target file's grid (true spread 0.1354, refit gains 0.0002 nat); it is applied on
the guess file's grid where the true spread is 0.0843 and refitting gains 0.1532 nat/segment.
**Concludes — the weight is a TEMPERING EXPONENT**, β = w·dx, not a nuisance knob. The Student-t
family is closed under tempering exactly (ν_eff = β(ν+1)−1, s_eff = s√(ν/ν_eff)). β = 1 means
w = 1/dx ≈ 1.01.
**Concludes — TWO harms, both from the weight**: the field collapses to 32% of the true spread
(100% of 900 batches, Q-Q slope 0.36) and its mean slides 1.08% low, because powering an asymmetric
density moves its mean.
**Concludes — refit, don't re-weight.** Tempering has one knob against four parameters; matching
the width breaks the mean. The MLE refit matches both at β = 1. Confirmed on fits: prior 5 → 0.5
moves the mean error 0.433% → 0.061% and the recovered spread 0.20 → 0.77.
**Closure check** — the prior alone predicts +568/batch against the +562 measured in the loss.

![Study 1: schedules](plots/noise_report/fig49_study1_schedules.png)

**Tests** — study 1: five ceiling arms × 6 seeds differing only in a learning-rate schedule. All
comparisons PAIRED (same seeds = same targets).
**Concludes — the lifetime seed scatter is LEVER ARM, not noise.** Each seed draws its own target τ
against a fixed detector depth, so the charge swing runs 4.2–15.2%; |lifetime error| tracks it at
r = −0.56 and seeds below the median are 3.9× worse. This is why ceiling reduces the seed s.d. for
every parameter except lifetime.
**Concludes — lifetime has asymptoted in optimiser travel** over a 4.5× range of ΣLR (75→136→340).
**Concludes — dE/dx RECOVERY is schedule-independent** (45.9–47.4% across all arms), exactly as
study 2 predicts for a prior-limited block.

![Study 1 per seed](plots/noise_report/fig50_study1_per_seed.png)

**Tests** — the same five arms, one panel per seed, ordered by lever arm.
**Concludes — the schedules DO move lifetime, but only where there is signal.** dE/dx → 0.01×:
**3.39% → 0.64 ± 0.40%** on the four high-lever-arm seeds (5.4σ, 4/4 improve), sub-percent on three
of them; and worse on the two starved seeds. *Corrects the first readout of this study, which
reported a clean null from the pooled mean — withdrawn.*
**Concludes — the mechanism is co-adaptation.** The annealed arm is the LAST to settle: it sits
above the others for 4000–6000 iterations then descends late, while the baseline drops fast and
flattens several percent low. Cooling the nuisance block stops it chasing calibration; it does not
make the block itself better.
**Read the caption** — the split is post-hoc (n=4 and n=2). A hypothesis for fresh seeds.

![Study 1 per seed, other parameters](plots/noise_report/fig51_study1_per_seed_allpars.png)

**Tests** — the control: A_b and the diffusions, same layout and seed ordering.
**Concludes — the effect is lifetime-specific.** Spread across the five arms: A_b 0.15–0.48 points,
transverse diffusion 0.20–1.00, longitudinal diffusion 1.40–2.25, lifetime 3.2–19.8. And the
low-lever-arm seeds are not specially bad for A_b or transverse diffusion — so it is not that those
targets are hard in general, they are specifically starved of lifetime signal.


![The alpha scan](plots/noise_report/fig52_alpha_scan.png)

**Tests** — study 2's α-scan: the per-segment dE/dx field is walked along the straight line from the
fitted values to the true ones in LOG space, calibration and geometry held at the fit. 2 seeds × 100
batches × 9 α, no fitting. This is the direct measurement figs 46–48 could only infer.
**Concludes — the DATA does not want truth.** Its own optimum is **α = 0.50**, i.e. 66% of the true
spread; α = 0 (the fit) and α = 1 (truth) are almost equally disfavoured (ΔNLL 6964 vs 6810). With
finite hits per segment the maximum-likelihood field is itself shrunk, so *some* shrinkage is
correct. Per batch: median 0.535, 97% want more dispersion than the fit, 100% less than truth.
**Concludes — no weight of the current prior reaches it**: 0.00 at w=5, 0.23 at β=1, 0.32 at w=0.5,
0.46 even at w=0. Structural, not a tuning failure — the prior as applied is 0.046 wide against a
data optimum of 0.056, so it is narrower than the data wants and can only pull α DOWN. A
correctly-calibrated prior would pull α UP.
**Concludes, operationally — lower the weight (partial), do NOT remove the prior, refit the shape and
use β = 1 (the real fix), keep the mean constraint at ~2500.** The moment that couples to A_b and
lifetime is the MEAN, not the spread.
**Read the caption** — α = 0.50 is the likelihood's optimum, not a target: chasing it would be
fitting the noise. And a different prior SHAPE cannot be re-scored offline, so the refit prior's α*
needs its own scan.


![Mis-setting the mean target](plots/noise_report/fig53_study3_meantarget.png)

**Tests** — study 3: the dE/dx mean-constraint TARGET scanned ±2% and CROSSED with the weight
(1e5 / 3000), 7 ceiling arms, slopes fitted per seed and averaged. Distinct from FITDM0/P2/M2, which
varied the sim input's mean rather than the penalty target.
**Concludes — A_b absorbs it linearly, lifetime does not respond at all.**
dA_b/d(target %) = **−0.837 ± 0.018** at w=1e5 (47σ) and −0.264 ± 0.022 at w=3000; lifetime
−0.16 ± 0.48 and −0.25 ± 0.44, i.e. consistent with zero at both. The 2-D scans predicted this: A_b
is the cheap absorber of a charge-normalisation error. **The target is an A_b systematic to control,
not a lifetime lever to exploit.**
**Concludes — the stiffness is the mechanism**: the fitted mean follows 93% of any target error at
1e5, only 30% at 3000, where all three arms land near 1.86–1.88 whatever the target. The two dE/dx
priors are in a tug-of-war and the mean weight decides it.
**Concludes — the budget**: the target is off by 0.023%, giving 0.02 points of A_b bias. To keep it
under 0.1 points the target must be known to **0.12%**.
**CORRECTS fig 45(e)** — the "weight 3000 costs 7.5× the lifetime scatter" claim was a SEED-SET
confound (CEILP05 n=3 never saw the low-lever-arm seeds). On common seeds it is 2.6×, and stratified
by lever arm weight 3000 is mildly *better*. Same confound as figs 49–51, biting twice.


![The BT arms](plots/noise_report/fig54_bt_arms.png)

**Tests** — the study-2 corrections applied to production fits. Three arms x 6 seeds, paired,
differing only in the dE/dx prior: BT0 current shape/w=5/mean 1e5 (the control, run on this tree),
BT1 current shape at w=1/dx with mean 2500 (correct WEIGHT only), BT2 the REFIT shape at w=1/dx
with mean 2500.
**Concludes — BT2 is the new best configuration.** A_b **0.77 → 0.27%**, paired −0.50 ± 0.09,
better on **6/6 seeds** (5.6σ). Lifetime **|error| 5.97 → 1.62%**, better in BOTH lever-arm strata
(4.10 → 1.36 high, **9.71 → 2.14** low). dE/dx mean error −0.454 → −0.199%, recovered spread
0.32 → 0.70. Cost: ~13 µm of position (157 → 170). Nothing regresses.
**Concludes — correcting the WEIGHT alone is a REGRESSION, not a partial win.** BT1's longitudinal
diffusion |error| is +11.05 ± 2.03 worse, on **0/6 seeds**. It would have been invisible had we run
only BT0 and BT2 — which is what that arm was for. The refit is doing the work and cannot be
replaced by a knob.
**Concludes — BT0 also validates the tree**: it reproduces §6r's historical production numbers
(A_b +0.73 → +0.77, lifetime −4.21 → −5.44, position 159 → 157 µm), so a campaign's worth of new
tooling did not move the answer.
**WITHDRAWAL, and the third instance of one trap.** The n = 4 read of these arms reported a
longitudinal-diffusion regression (−0.60% → +7.32%). That was a SIGNED mean of a parameter whose
per-seed errors change sign; in |error| the paired difference is −0.50 ± 3.65, i.e. nothing. Same
failure as fig 45(e) and study 1's first readout. **For any mixed-sign parameter, quote |error|.**

![BT convergence traces](plots/noise_report/fig55_bt_traces.png)

**Tests** — what the three arms are doing at every iteration rather than where they end: the five
calibration parameters, the dE/dx MAE against truth, the length-weighted ⟨dE/dx⟩, position and loss.
**Concludes — the ⟨dE/dx⟩ panel shows the §6s mechanism happening.** BT1 drops away from the true
mean inside ~500 iterations and never recovers (1.8678 vs a truth of 1.8869); BT2 climbs to 1.8829
and holds; BT0 sits between. **The A_b panel reproduces that ordering exactly**, which is the
A_b↔charge-normalisation coupling of §6u seen as a trajectory rather than a slope.
**Concludes — an excursion the endpoints hide**: BT1's dE/dx MAE gets WORSE than its own starting
point for ~2500 iterations (0.070 → 0.081) before recovering, while BT0 and BT2 descend
monotonically. A too-wide prior at β = 1 lets the field wander before the data reins it in.
**Concludes — position is still descending in all three arms** at −6 to −7 µm per 20%, which is the
direct motivation for the 20 000-iteration runs; and longitudinal diffusion is the only parameter
still drifting everywhere, so its values are softer than their error bars suggest.
**Read the caption** — the loss panel is NOT comparable across arms: the prior term differs by
construction, so BT0 sits ~1700 lower purely because its dE/dx prior is 5× weighted.


![Diagnostic scans](plots/noise_report/fig56_diag_scans.png)

**Tests** — three ~1 h EVALUATION jobs on the BT2 fit (60 constructed points × 100 batches × 2
seeds, no fitting): a geometry α-scan, a joint (geometry × dE/dx) grid, a long-diffusion profile at
two geometries, and the dE/dx α-scan with the refit prior.
**Concludes — GEOMETRY IS OPTIMISER-LIMITED.** Its optimum is **α\* = 0.988**: the objective wants
geometry essentially exactly at truth, with no shrinkage, unlike dE/dx which stops at 0.50. Priors
barely participate (408 nats against the likelihood's 192 637).
**Concludes — and it is where the information is.** Likelihood range: geometry 192 637 nats, dE/dx
dispersion 37 475, long. diffusion at ±20% just 845. Geometry is worth **5.1×** the dE/dx question
and **228×** long. diffusion.
**Concludes — the two nuisance blocks are DEGENERATE.** The dE/dx optimum slides −0.178 → +0.419 as
geometry moves from the fit to truth, i.e. the preferred spread goes 63% → 82% of truth. Part of why
the data appeared to want a flat dE/dx field is that the GEOMETRY was wrong — and it is why the
ceiling arms did not predict production.
**Concludes — the refit is validated**: total and likelihood optima now differ by 0.17 in α against
0.50 with the old prior. Cross-check: fig 52 measured the data's preferred spread as 66% of truth on
the BT0 fit with the OLD prior; this measures 63% on the BT2 fit with the REFIT prior — the same
answer from two different fits, which is what a property of the DATA should look like.
*Correction: an earlier version said geometry carries 28× the dE/dx question. That compared this
figure's geometry range against fig 52's dE/dx range — two different fits. On one fit it is 5.1×.*

![The degeneracy matrix](plots/noise_report/fig58_degeneracy_matrix.png)

**Tests** — every calibration parameter scanned over ±20% of its BT2-fitted value at all four
combinations of (geometry, dE/dx) ∈ {fit, truth}. 145 points × 100 batches × 2 seeds. The SHIFT of
an optimum when a nuisance moves is its degeneracy; the CURVATURE is how tightly the data pins it.
**Concludes — A_b and E FIELD ARE DECOUPLED from both nuisance blocks** (±0.1%). That is why they
are the best-determined parameters, and it means their residual error cannot be blamed on the
nuisances. Geometry dominates every other coupling and acts almost entirely on longitudinal
diffusion (≥23.4%, the optimum ran off the grid); dE/dx coupling is ≤2.4% everywhere.
**Concludes — the degeneracies are ADDITIVE** (largest interaction 3.3 points), so block-wise
reasoning is valid; and BT2 sits at the objective's optimum for every parameter (≤1.3%).
**Concludes — WE ARE NOT STATISTICS-LIMITED ANYWHERE.** Against the curvature floor: A_b ×16,
E field ×11, tran. diff. ×4, long. diff. ×10, lifetime ×6. None of the remaining error is
irreducible — which qualifies "lifetime is lever-arm limited": that is about the available SIGNAL,
but even given it we are 6× short.
**Read the caption** — conditional curvature, nuisances held fixed. Exact for A_b and E field
(their optima do not move when the nuisances do); a LOWER BOUND for the diffusions.

![Geometry schedule screen](plots/noise_report/fig57_geom_screen.png)

**Tests** — a 3 000-iteration screen of geometry schedules, with BT2's own trace as the matched
control (equal iteration count ⇒ identical schedule indexing, so the baseline costs no job).
**Concludes — "more geometry learning rate" splits into two OPPOSITE things.** Raising the INITIAL
rate is worse, monotonically: 3× gives +6.8%, 10× diverges (1 833 µm at iteration 1 000 against 581).
Sustaining the rate LONGER via a slower decay is better, −8.1%, and the gain WIDENS with iteration
(−3.7% at 1 500) rather than being a fixed offset. The sustained arm also wins on both seeds while
the 3× arm is inconsistent (426/331 vs 351/357).
**Concludes — geometry needs small steps early and sustained steps late.** A schedule SHAPE, not a
magnitude — which independently validates the G20K design (chain_decay stretched 0.9997 → 0.99985).
**Read the caption** — 3 000 of 10 000 iterations: a screen for what deserves a full run, not a
verdict. Calibration is nowhere near converged at this point and nothing is claimed about it.


![G20K](plots/noise_report/fig61_g20k.png)

**Tests** — the long geometry run the screen above justified: 20 000 iterations with BOTH schedules
stretched ×2 in the exponent (chain_decay 0.9997 → 0.99985 = 0.9997^½, calib decay 0.91 → 0.9539).
Seeds 0–2, complete.
**Read the DASHED curve, not BT2.** The stretch means G20K at iteration 2N carries the LRs BT2 had
at N, so equal-iteration comparison pits a hotter optimiser against a cooler one. The dashed curve
is BT2 on a doubled axis = the null "the schedule is all that matters", ending at 20 000 where BT2
ended at 10 000.
**The ENDPOINTS are comparable, and that is what the ½ exponent buys** — terminal LR 0.9539²⁰⁰ =
7.95e−5 vs BT2's 0.91¹⁰⁰ = 8.02e−5, the same number, with end-of-run spreads 0.0001–0.016 in both.
This run satisfies trap 1b by construction, unlike the frozen-geometry arms.
**Concludes — the null is rejected and the OBJECTIVE agrees.** Position **−57.2 ± 0.3 µm on 3/3
seeds** (165→108, 165→108, 176→119; −34.7%), and −42.7% against the null; loss (PPP NLL, the
quantity actually minimised) **−667 ± 92 = −8.7%**; dE/dx MAE −0.0022 ± 0.0002. A better optimum,
not a metric that happens to look better. G20K also reaches BT2's FINAL position at iteration
**7 422** — 26% fewer iterations.
**Concludes — geometry was optimiser-limited, as fig 56 said (α\* = 0.988).** The shape confirms it:
G20K is WORSE early (386 vs 351 µm at 3 000 — hotter is noisier), crosses near 6 500, then keeps
descending where BT2 flattens. It is converged at the end (−0.4/+0.1/−1.0% over the last 2 000);
BT2 was still drifting when it stopped.
**Concludes — calibration: ONE improvement, four nulls.** Transverse diffusion 2.610 → **0.605%**,
paired −1.42 ± 0.40 on 3/3 seeds — the parameter fig 58 calls most geometry-coupled, so this is
predicted, not surprising (t = 3.6 on 2 dof, p ≈ 0.07: confirm at n = 6). Long. diffusion
(−1.23 ± 4.02), E field (−0.000 ± 0.012), A_b (+0.03 ± 0.09) and lifetime (+0.72 ± 0.89) are all
consistent with zero.
**Read the caption — do NOT read the ratio of medians.** A_b reads "+73%" and lifetime "+88%" as
median ratios and both are nulls on the paired difference; with n = 3 the median is essentially the
middle sample. Lifetime is worse on 2 of 3 seeds though — not established, but the sign is
consistently unhelpful and is the one place more geometry iterations might cost something.
**Cost** — 26.4 h vs 13.4 h: ~2× wall clock for 34% better position, or the SAME position in 26%
fewer iterations.


![Alternation](plots/noise_report/fig59_alternation.png)

**⚠️ RETRACTED by fig 62 — read the two together.** The headline below compared BT2's FROZEN
endpoint against a MID-OSCILLATION sample of a run whose LR was still 700× larger. Kept as written
because the retraction is only legible against the numbers it reported.
**Tests** — does alternating the blocks help? ALT takes BT2's converged geometry, FREEZES it, and
re-fits calibration and dE/dx for 3 000 iterations. Everything except geometry restarts from the
same nominal point BT2 used (A_b 0.800, lifetime 2200, dE/dx at the prior centre), so this is an
independent convergence, not a polish.
**Concludes — a frozen-geometry second pass BEATS the joint fit on the decoupled parameters.**
A_b 0.340 → **0.147** (×2.3, better on both seeds), transverse diffusion 1.883 → 0.875 (×2.2, but
that one rests on a single seed). E field, longitudinal diffusion and lifetime are unchanged to
within 2%. It also runs 24% cheaper per iteration with the geometry block off.
**Concludes — and the pattern is exactly what fig 58 predicts.** The parameters that improve are
those whose optima barely move when geometry moves (+0.1%, −3.4%); the ones that do not are those
that NEED geometry to move (−23.4% for long. diffusion). Two independent methods — scanning the
objective, running a fit — single out the same set.
**Concludes — the co-adaptation is visible in the trace**: ALT settles by ~1 500 iterations and
holds, while BT2 oscillates through zero near iteration 4 000 and then drifts back up to +0.4%.
**Read the caption** — ALT inherits a geometry BT2 spent 10 000 iterations earning, so it is a
two-pass scheme (~22 h vs 19 h), NOT a cheaper standalone recipe. n = 2. And the obvious freeze
check FAILS misleadingly: ALT's position reads 1104 µm against BT2's 170 µm median, but with
geometry frozen the residual is recorded once for batch 0 only, where BT2's own value is 1027 µm —
860 of 864 track coefficients match exactly.


![Warm-start control](plots/noise_report/fig62_altwarm.png)

**Tests** — the control fig 59 called for: the SAME frozen-geometry objective approached from the
other direction, calibration initialised at BT2's own fitted values. Converging back to ~0.15 would
have established 0.15 as the real optimum, reached from two directions.
**Verified first** — ALTW's iteration-0 calibration equals BT2's final to all printed digits on all
5 params and both seeds, and both arms inherited byte-identical geometry (860/864 match BT2 AND each
other, same 4 mismatches). The arms differ ONLY in the calibration init.
**Concludes — the premise is invalid, twice over, and fig 59 does not survive.**
(1) *The warm start is erased in ~200 iterations.* Only calibration is warm-started; the dE/dx block
has no init-from, restarts near the prior centre in BOTH arms, and drags A_b with it (the §6u
charge-normalisation coupling). ALTW leaves its init at +0.48 and is at **+1.27 by iteration 300**
while ALT arrives from −4.18 — by ~300 both sit in the same region. Not a two-directions test; two
runs of the same cold-dE/dx transient.
(2) *Neither arm is converged.* The calib LR decays 0.91 per EPOCH, so at 3 000 iterations it is
**0.059 of peak vs BT2's 8.0e−5 — a factor 700**. A_b's end-of-run spread: BT2 **0.001**, ALT 0.127,
ALTW 0.117. ALT's striking +0.008 on seed 1 carries a ±0.065 wander — a SAMPLE, not a value, and
with n = 2 that one sample set the headline.
**Concludes — the objective corroborates it.** All three arms within **0.4%** on the PPP NLL (9 108
/ 9 147 / 9 136); G20K moved the same quantity 7.2%. They are wandering across a flat region, so the
calibration differences are optimiser path, not objective.
**What survives** — the degeneracy matrix (fig 58) is untouched: evaluation-only, no optimiser in
it. Only the claim that ACTING on it via alternation buys anything falls. Alternation is now
untested, not disproven.
**The fix** — compress the calib schedule so 3 000 iterations reach BT2's terminal LR: SCILRDECAY
0.91 → **0.73** (0.73³⁰ = 7.9e−5 ≈ 0.91¹⁰⁰). 4 jobs, ~5 h.
**The lesson** — two arms compared at equal ITERATION COUNT when what matters is equal SCHEDULE
PHASE: the same error the K2 screen made, the inverse of the one G20K was built around. **Any
comparison against a converged run must be converged itself or quote its own oscillation amplitude.**


![Block matrix](plots/noise_report/fig63_block_matrix.png)

**Tests** — all seven subsets of the three blocks: calibration C, per-segment dE/dx D, spline
geometry G. Every block warm-started from BT2 (nothing restarts from scratch), then 1 or 2 frozen
and the rest re-annealed for 3 000 iterations. Required NEW machinery: LARND_DEDX_INIT_FROM (the
missing dE/dx warm start that invalidated ALTW) and SCICALIBFREEZE, both validated on a
30-iteration job first — calibration movement exactly 0, dE/dx max deviation 0.0100 = one Adam step
vs 0.2530 for a cold start.
**First study built with trap 1b designed in** — schedules COMPRESSED so 3 000 iterations land on
BT2's terminal LR (SCILRDECAY 0.91→0.73, SCICHDECAY 0.9997→0.999), so all arms share a schedule
phase. Panel (a) verifies every freeze against the checkpoint and the figure ASSERTS it.
**BM0 (nothing frozen) is the control and is not optional** — a re-anneal ALONE moves position
165 → 127 µm. Read arms against BM0, not BT2, or the re-anneal is credited to the freeze.
**Concludes — freezing CALIBRATION is the clear win.** BMC (fits D+G) takes the best loss
(**8 812, −3.25% vs BT2, −2.06% vs BM0**), the best position (**115.7 µm** vs 127.1 control, 165.3
BT2) and the best dE/dx MAE (0.0533 vs 0.0554, 0.0541). Same ordering on all three. The nuisances
WERE held back by co-adaptation with calibration: −38 µm from the re-anneal, a further −11 µm from
the freeze.
**Concludes — freezing GEOMETRY erases the descent.** BMG / BMCG / BMDG all sit within 0.1% of BT2's
loss whatever else is fitted. Geometry is where the objective lives (the α-scan's 5.1×), and this is
the independent confirmation that fig 59's frozen-geometry arms could not have been descending.
**Concludes — calibration is mixed, consistently on both seeds.** Every calibration-fitting arm
improves long. diffusion (BM0 8.54 → 2.85%) and degrades LIFETIME (BM0 1.85 → 5.44%). A_b improves
under the plain re-anneal only (0.34 → 0.10%); tran. diff. improves only with geometry frozen
(BMG 1.89 → 0.76%).
**THE UNCOMFORTABLE ONE (panel f)** — BM0 finds a **LOWER loss (−1.21%)** while moving lifetime
**+3.6 further from truth**, on both seeds independently. Better fit, worse physics: for lifetime the
objective's optimum is DISPLACED from truth, and any procedure that selects on the loss selects
AGAINST lifetime accuracy.
**Next** — the two-stage recipe (freeze calibration, descend, then release) is untested; no arm does
it. And n = 2 is a screen: sized for the −3.25% / −50 µm effects it found, not for small ones.


![Block traces](plots/noise_report/fig64_block_traces.png)

**Tests** — the block matrix as TRAJECTORIES (fig 63 gives only endpoints). BT2's last 3 000
iterations are drawn left of zero, since every arm starts where BT2 finished; dotted = that block is
frozen, so flat at BT2 by construction.
**Concludes — the re-anneal KICKS every fitted block away before it re-descends.** Not noise: it is
why a warm start does not sit still, and exactly what erased ALTW's warm start. Every arm pays it.
**Concludes — position descends monotonically and is STILL FALLING at 3 000** in every
geometry-fitting arm, so every position number here is an upper bound; the ranking is settled early.
**Concludes — lifetime does the opposite: kicked away, never returns.** BM0 leaves BT2's −1.2% and
settles at +5.4%; the other calibration-fitting arms leave the other way and settle near −3.2%. The
loss is content at the new point.
**Concludes — the loss panel splits into exactly two families**, geometry fitted (descending) and
geometry frozen (flat from iteration 0): the 5.1× information asymmetry made visual.


![Block factorial](plots/noise_report/fig65_block_factorial.png)

**Tests** — the same seven arms read as the FACTORIAL they are. For each fitted block the other two
are each frozen or fitted = a complete 2×2, so each outcome reduces to two MAIN EFFECTS and one
interaction: three numbers instead of four, comparable across blocks. The degeneracy matrix asks
this of the OBJECTIVE; this asks it of the FIT.
**Concludes — freezing CALIBRATION is the one large consistent lever** and it helps both nuisance
blocks: position **−9.3 µm** (−6.7/−11.9 per seed), dE/dx MAE **−13.9e−4** (−13.2/−14.5). Both seeds
agree in sign on both.
**Concludes — freezing a nuisance does little for the other nuisance.** Freeze dE/dx → no effect on
position (+0.1 µm, seeds +3.9/−3.7 = null). Freeze geometry → helps dE/dx (−4.6e−4, very
reproducible) but only ~⅓ as much as freezing calibration.
**Concludes — for CALIBRATION, freezing geometry is the only lever and it is a TRADE:** tran. diff.
−0.72 (better, both seeds) bought against long. diff. +4.50 and A_b +0.19 (both worse, both seeds).
Freezing dE/dx does essentially nothing — every entry but A_b straddles zero.
**Panel (f) is the one to keep** — differenced against each arm's OWN seed's BT2 (the seeds sit
~200 NLL apart, enough to fake a trend). **7 of 8 arm×seed points lie above zero**, and the three
largest loss gains (−149, −137, −72 NLL) are the three worst lifetime degradations. Better fit,
worse physics, in a controlled pair.


![TS1](plots/noise_report/fig67_ts1.png)

**Tests** — stage 1 of the two-stage recipe: calibration FROZEN at BT2's values, geometry and dE/dx
warm-started from BT2 and re-annealed for 10 000 iterations. BMC run to the floor it was still
descending toward at 3 000. 3 seeds.
**Verified** — calibration movement exactly 0 on all 5 params and 3 seeds (init = BT2's tail median
to 0.0000%); dE/dx moved 0.165–0.179 log units, geometry moved on 98.7–100% of tracks.
**Concludes — freezing the block that does not need to move BEATS doubling the iteration count.**

| arm | what | h | position | loss | dE/dx MAE |
|---|---|---|---|---|---|
| BT2 | all fitted, 10k | 13.4 | 165.4 | 9 009 | 0.05455 |
| BMC | calib frozen, 3k | 4.3 | 115.7 | 8 812 | 0.05333 |
| **TS1** | **calib frozen, 10k** | **13.5** | **105.0** | **8 072** | **0.05206** |
| G20K | all fitted, 20k | 26.4 | 108.0 | 8 229 | 0.05204 |

Against G20K: objective better on **3/3 seeds (−88 ± 34 NLL)**, position and dE/dx a tie, at **51%
of the compute**. Since the objective PREFERS truth (fig 66), lower loss = closer to the joint
minimum, so this is real.
**Concludes — and it converges EARLY.** Past G20K's final position by iteration **3 244** and its
final loss by **3 926**; within 2% of its own endpoint by ~3 500. Stage 1 could run at 4 000
iterations (~5.4 h), taking the full two-stage recipe from ~27 h to **~19 h** — cheaper than BT2.
**Read the caption — TS1's CALIBRATION IS BT2's, frozen.** Tran. diff. 2.64% vs G20K's 0.61%. A
better FIT, not a better CALIBRATION. Consistent with "blocks must move together": TS1 moves only
two of three, so it cannot be the endpoint. **Stage 2 (TS2) has NOT been run** — one command:
`STAGE=2 bash optimize/scripts/run_twostage.sh`.


![TS2](plots/noise_report/fig68_ts2.png)

**Tests** — stage 2 of the two-stage recipe: everything warm-started from TS1, calibration RELEASED,
fresh 10 000-iteration anneal. TS1 had the best FIT but its calibration was BT2's, frozen; this was
meant to convert that into a better CALIBRATION. 3 seeds.
**Verified** — calibration genuinely moved (ptp 755–1434 physical) from an init matching TS1's to
0.0000%; dE/dx moved 0.088–0.107; geometry moved on 98.7–100% of tracks.
**⚠️ PAIRED DIFFERENCES ONLY** — the arms sit within ~3 µm and ~130 NLL, *smaller than the
seed-to-seed spread*, so the MEAN and MEDIAN of the absolute levels disagree about whether TS1 or
TS2 has the lower loss. Absolute levels are meaningless at this precision. Trap T3 at its sharpest.
**Concludes — the release helped PARTLY, and cost something.** vs TS1: transverse diffusion
**−0.93 (3/3)**; position **+2.7 µm (0/3, worse)**; E field +0.023 (0/3, worse but negligible
absolutely); loss, A_b, long. diff., lifetime all nulls.
**Concludes — and it does NOT beat G20K at equal compute** (27.0 h vs 26.4 h). It TRADES: loss
**−131 (3/3, TS2 wins)** against transverse diffusion **+0.49 (0/3, G20K wins)**; everything else a
tie. Lower loss, worse physics — the same pattern as figs 63/65, explained by fig 66: the objective
prefers truth only when all three blocks move TOGETHER, and stage 2 moves calibration against
nuisances no longer free to follow.
**Against BT2 it IS a clear win** — loss −799, position −55.7 µm, dE/dx −0.0024, tran. diff. −0.93,
all 3/3. The recipe is not a failure; it is just not better than cheaper alternatives at the thing
it was built to fix.
**PRACTICAL — TS1 alone (13.5 h) remains the best value in the campaign.** Stage 2 doubles the cost
to buy one parameter and give back position. **Treat stage 2 as optional.**


![Full path](plots/noise_report/fig69_fullpath.png)

**Tests** — is there a BARRIER between the fit and truth, or is the path open? Evaluates the loss
along the joint line X(α) = X_fit + α(X_truth − X_fit) with calibration, dE/dx AND geometry moving
together. One-block scans cannot answer this: moving any single block to truth alone can make the
loss worse.
**Verified** — α=0 reproduces P0_fit and α=1 reproduces P1_truth to 1.6e−7 (float32 round-off), and
F1 is IDENTICAL across all three arms per seed (1 part in 1e8), as truth must be.
**Concludes — NO BARRIER.** The path is monotone downhill in every arm and every seed, and the
minimum sits AT α = 1, not beyond: past truth the loss climbs steeply, exceeding the fit's own value
by α = 1.2. Along this direction the objective's optimum IS truth — stronger than any conditional
scan supports.
**Concludes — the fits stop because they RUN OUT OF STEP SIZE, not because the gradient vanishes.**
The schedule anneals the LR to ~8e−5 of peak while a gradient toward truth is demonstrably still
there. With the shape (shallow near the fit, steepening toward truth) this is the signature of an
ILL-CONDITIONED valley: steepest descent nearly orthogonal to the useful direction.
**ACTIONABLE — build a preconditioner.** The gap is reachable, and more first-order iterations are
not the route (G20K plateaued at 2× the iterations). The GGN/Newton machinery from the dE/dx block
is the obvious start.
**Secondary** — TS1 closes **86–87%** of BT2's gap to truth vs G20K's 77–79%, at half the compute:
independent confirmation, on a measure neither was tuned against, that the cheap recipe wins.


![Profile](plots/noise_report/fig70_profile.png)

**Tests** — a genuine PROFILE over the calibration block: pin lifetime at 7 multipliers and
re-optimise everything else (the other 4 calibration params AND both nuisance blocks) from TS1.
The pin was validated first and held at every point (drift 8e−9 to 1.4e−7 vs free params at 1e−3).
**Concludes — IT DOES NOT WORK, and the noise budget says why.** The curve is jagged, not convex:
minimum at −5.3% while the far end of the grid is nearly as low. In nats per batch:
expected signal across the WHOLE grid **7.5** (an upper bound — it is the CONDITIONAL curvature and
profiling only flattens it); error bar on one point **10.5**; scatter between the 7 re-fits **41.4**.
The signal is smaller than the noise on a single point.
**Concludes — and it cannot be averaged away.** A 1σ interval needs Δ = 0.5 summed = 0.005
nats/batch against 10.5 achieved: **2 105× too noisy**, i.e. ~4 million times more sampling. Wrong
instrument, not insufficient resolution.
**Why** — each point is an independent stochastic re-optimisation of ~9 000 nuisance parameters, and
where it LANDS varies more in loss than the lifetime dependence. The noise is in the optimiser's
landing point, not in the data (every point already averages over full epochs).
**THE REDIRECTION** — uncertainties must come from the CURVATURE at the fitted point (the profile
covariance is the Schur complement of the full Hessian), which is deterministic. **This is the same
machinery fig 69 asks for**: both studies converge on building second-order information. The
GGN/Newton code from the dE/dx block is the start.
**Seed 1 not submitted** — the pilot answered it; a second seed buys a second unresolvable curve
for 30 GPU-hours.


![Objective bias](plots/noise_report/fig66_objective_bias.png)

**Tests** — is the calibration error the OPTIMISER's or the OBJECTIVE's? Evaluation only, at BT2's,
BM0's and G20K's converged checkpoints. The decisive quantity is P1_truth: the loss with
calibration AND dE/dx AND geometry all set to truth.
**Concludes — THE OBJECTIVE PREFERS TRUTH.** P1_truth beats P0_fit on **7/7 arm-seeds by 5.6%**, so
the fit is NOT at the joint minimum and the error is REACHABILITY, not objective bias.
**Concludes — and the gap is closing:** truth beats the fit by 9.4% (BT2) → 6.6% (BM0) → **2.5%**
(G20K). G20K genuinely reached a better region rather than trading errors.
**Concludes — BLOCKS MUST MOVE TOGETHER, and this is the mechanism.** Moving one block to truth
while the others stay fitted: geometry alone **−7.00%** (7/7), calibration alone **+0.18%** (0/7),
dE/dx alone **+2.79%** (0/7), all three together **−5.60%** (7/7). Setting calibration to its TRUE
values while the nuisances are wrong makes the fit WORSE.
**Concludes — the 1-D scans measure COUPLING, not bias.** With geometry and dE/dx pinned at their
FITTED values the fit sits on the scan's minimum every time — because wrong nuisances drag the
preferred value onto the fit. A conditional minimum cannot locate a joint one.
**One genuine objective bias survives, and it is not lifetime's** — P3_dedx_true is WORSE than the
fit on 7/7 by 2.79%: the dE/dx prior displaces its own block's minimum, the standing §6s result,
reconfirmed at three converged points.
**The residual, honestly** — with BOTH nuisances at truth the lifetime slice minimum is still 2.26%
off (from 2.83%), so the nuisance state explains only ~⅕. But that scan still fixes the other four
calibration parameters, so it is also conditional. **Settling it needs a profile over the full
calibration block — no run has done this. That is the open question.**
**A hypothesis killed (stands)** — lifetime ↔ long. diffusion trade: correlation +0.03 (n=7). None.
**⚠️ CORRECTION** — the first version of this entry concluded the OPPOSITE (lifetime
objective-limited, unfixable by optimisation). That read a CONDITIONAL minimum as a joint one; a
conditional minimum sits on the fit by construction when blocks are coupled. The test that
distinguishes them, P1_truth vs P0_fit, was in the same files all along.


![K2 spline basis](plots/noise_report/fig60_k2_basis.png)

**Tests** — is the third spline mode worth fitting? Measured against the truth projection it
recovers **−61.7%**: its error (0.0209 cm rms) EXCEEDS its true amplitude (0.0129 cm), so fitting it
injects noise. K2 drops it (`LARND_SPLINE_KMIN` 3 → 2). It took effect on **839 of 864 tracks (97%)**
— only the 25 long enough to demand three modes keep K = 3.
**Concludes — a NULL on position.** 354.0 → 345.1 µm (−2.5%), and that −2.5% is one seed of two
(351 → 333, 357 → 357). The one real gain is compute: **10% cheaper per iteration** (4929 vs 5485 ms).
**Concludes — and panel (c) says a null was predictable.** Mode 1 carries 4× the amplitude of the
others and is only 51.6% recovered, so removing mode 3 cuts the coefficient-space error just 5.5%
(0.0639 → 0.0604 cm) while perfect mode-1 recovery would cut it **54%**. The −61.7% correctly
diagnosed a noisy mode; that mode was never where the residual lived. **Basis truncation is not the
lever — mode 1 is.**
**Read the caption — this is the concrete instance of the screen-length trap.** At iteration 2 900
K2's calibration looks 2× better (lifetime |err| 2.81 vs 6.00, tran. diff. 3.36 vs 8.64) and NONE of
it survives: BT2 is merely still in its transient, running 6.00 (2 900) → **1.56 (5 000)** → 1.85
(9 800) and passing K2. A 3 000-iteration screen ranks POSITION, which descends monotonically; it
cannot rank calibration.
**Verdict** — adopt for the compute, not the accuracy.

## §11 What is settled, what is open, what is next

**Settled**
- A_b and E field are recovered at the statistical floor and are immune to every degradation tested.
- Geometry accuracy is a variance lever, not a bias lever, for random *and* systematic error.
- Batching order is irrelevant; the dE/dx block earns its keep; the target is not biased.
- The batch-size ceiling is **600 cm/batch** (700 OOMs); batch *count* is free in memory.
- **The dE/dx minimum-length cut is a null** (§8): no effect on any parameter where it fires, and
  a measured no-op in production. Do not enable it.
- **The dE/dx block improves 17% against truth** at production length (§8), and **40–64% with
  true geometry** (§5). The "MAE gets worse" puzzle was the metric's reference, not the fit.
- **The dE/dx block is objective-limited**: perfect geometry does not let it close the gap.
- **More data helps when compute scales with it** (§8): 2× data tightens every error bar and
  preserves geometry (172 vs 162 µm), unlike raising batches at fixed iteration count.
- **n = 3 cannot resolve ~4 points** (§8). Three runs of the *same* configuration give lifetime
  means spanning 3.6 points. Every A/B comparison must clear that floor first.

**Open**
1. **Redo every scan as a profile, not a slice.** ***Done for the five key conditions — §6.***
   Profiling removes the apparent bias wherever the defect is a charge-normalisation error and
   changes nothing otherwise. The remaining ladder rungs have not been re-scanned, but the pattern
   is now understood well enough to predict them: only rungs that shift the dE/dx *mean* will move.
2. **What actually drives the residual lifetime error?** Not the dE/dx mean, not geometry. Lifetime
   remains schedule-dependent and quoted as a bound.
7. **Is the recent campaign's dE/dx prior weight wrong?** *(new, §5)* Every recent arm ran at
   `SCIDEDXPRIOR=5`, inherited from ANNEALLONG, against a 0.5 default — and recovers 3× less of the
   dE/dx gap than historical ceiling runs. `ceil_p05` (3 seeds, ceiling, w = 0.5) is running. If it
   confirms, several recent conclusions were drawn with the nuisance block over-constrained,
   including the `ceil_w3k` mean-constraint test, whose two weights are then entangled.
3. **Why does the per-segment dE/dx MAE end worse than it starts?** ***Answered — §8.*** It was
   the reference: the MAE was computed against `--input_file_sim`, the guess file, so it measured
   distance from a bad starting point. Against an arc-length-matched truth reference the block
   **improves 17%** (0.0714 → 0.0590) at production length while appearing to degrade 10%.
5. **Does the drift-profile penalty help in the current config?** *(new)* Probably not, and
   possibly harmful: §5 shows the fitted profile is already flat, so `dpw` would have nothing to
   correct and would fight the fit's removal of the input's +0.0032 trend. Needs a matched pair to
   settle, ideally together with a looser `dedx_mean_constraint_weight` — the two constraints
   currently pin the *global normalisation* (which A_b could absorb harmlessly) while leaving the
   *drift-correlated* component (which is genuinely degenerate with the physics) unconstrained.
4. **Does more data help when compute scales with it?** ***Answered — §8.*** Yes: 2× data with
   epochs and per-batch visits preserved tightens every error bar (A_b s.d. 0.14 → 0.06,
   tran. diff. 0.67 → 0.17) and leaves geometry intact (172 vs 162 µm). Scaling batches *without*
   scaling iterations does not — that starves the geometry block (774 µm).
6. **Everything now needs a power gate.** *(new, §8)* With a 3.6-point n=3 floor, the outstanding
   questions — the mean-constraint weight, the drift-profile penalty, the profiled scans — must be
   run at ceiling and/or with ≥6 seeds, or they will return the same unresolvable answer a fourth
   time.

**Not worth further investment** — geometry-side optimizer work. Three independent measurements
agree it buys variance we can obtain more cheaply from data volume.
