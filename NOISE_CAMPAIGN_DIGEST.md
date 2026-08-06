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
lever), the dE/dx min-length cut (a null), mean-constraint weight 3000 (no bias gain, 7.5× the
lifetime scatter), noise OFF (+5.4 points of bias).
**Concludes — two hard limits.** The detector is **0.087 lifetimes deep** — an 8.3% anode-to-cathode
swing is the entire lifetime signal — and the dE/dx block recovers at most **~45%** of the
prior→truth gap even with perfect geometry. Neither is a tuning problem.
**Read the caption on (c) and (e).** (c) is seed-matched because `nb200` has only 2 completed seeds;
(e) plots bias *and* scatter because the mean-constraint arm looks better on central values alone
purely through a lifetime value with a ±2.1 error bar.

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
