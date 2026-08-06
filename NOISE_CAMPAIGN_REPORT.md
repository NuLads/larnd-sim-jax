# Differentiable LArTPC calibration — status report
### From the noise-ON reprocessing (S1 re-run) to the present

**Date:** 28 July 2026 · **Code:** `larnd-sim-jax`, branch `dedx_soft_barrier`
**Scope:** everything since we discovered that the whole preceding campaign had a
guess/target noise mismatch and re-ran the stage ladder with electronics noise ON.

---

## 1. TL;DR for the impatient

> **START HERE: §6r "Where we stand"** — the best setup, its measured performance, the three
> levers that improve it (in order of gain), everything tested that does not, and the two hard
> limits, in one section with one figure (fig 45). Everything below it is the evidence.

- **Calibration is solved when the track geometry is known.** With true positions and
  fitted per-segment dE/dx (stage S2), all five calibration parameters land within ~2% and
  sit at the statistical floor predicted by the Hessian.
- **The hard case was fitting geometry and calibration jointly (S4).** For weeks this gave
  lifetime errors of +40 to +130% with large seed-to-seed scatter.
- **We now believe the dominant cause was an optimizer-schedule artefact, not physics.**
  The calibration learning rate never annealed: nuisance parameters (≈4000 dE/dx values) and
  the physics parameters were updated with the wrong relative "temperature" for the whole run.
- **Annealing the calibration LR moves S4 from lifetime +40 ± 25% to roughly −2% to −5%**
  (n = 6 seeds at 5 000 iterations; see §6c — lifetime is schedule-dependent and its
  reported value is a LOWER BOUND on the bias). Geometry reaches **162–168 µm** at
  10 000 iterations. A_b, E field and both diffusion constants are converged and
  schedule-invariant; lifetime is not.
- **This is not yet final, and the evidence is uneven across parameters.** Nothing is actually
  converged at 5000 iterations — position, loss and long. diffusion are all still moving. And
  because each seed draws its own target, some seeds start close to truth and therefore do not
  constitute a real test: at n = 6 seeds five of six close ≥64% of their initial lifetime
  offset, so the claim no longer rests on a single seed (it did at n = 3). E field and transverse diffusion are
  convincingly recovered on all seeds; **long. diffusion is the weakest claim here**. See §6 for
  the per-parameter breakdown. Longer runs and more seeds are in flight; see §8.
- **Two ablations have since completed and both are informative (§6d).** Drawing the target
  from the probabilistic distribution changes nothing measurable — the deterministic target is
  **not** a source of bias. Freezing dE/dx at the mean instead of fitting it **degrades long.
  diffusion ~4× and lifetime ~2.7×**, which is direct evidence that the ≈4000 dE/dx nuisances
  are absorbing real per-segment structure rather than merely adding freedom.
- **A 24-scan quality ladder now separates the two input defects (§6f), and they act
  differently.** Degrading dE/dx *displaces* the likelihood minimum — monotonically, across four
  rungs, up to lifetime +7.0% and long. diffusion −23.8% — while degrading position only
  *widens* it, inflating long. diffusion's uncertainty 4.1 → 40.9 points at 880 µm with no
  significant centre shift. **dE/dx quality is a bias problem; position quality is a variance
  problem.** Our current 168 µm geometry sits comfortably inside the useful range.
- **The grid has since been refined, and with perfect geometry the objective is nearly
  unbiased (§6g):** lifetime +1.14 ± 0.57%, long. diffusion −1.67 ± 1.41%, against +2.18 and
  −5.27 on the coarse grid. The dE/dx ladder is unchanged by refinement, so §6f's comparisons
  stand — but §6e's claim that long. diffusion's fitted −5.40% "agrees with the objective
  minimum" was largely a coarse-grid artifact and no longer holds tightly.
- **THE HEADLINE (§6k): our drift lever arm is 0.087 lifetimes, and that governs everything.**
  The detector is 190.8 µs deep against a 2200 µs lifetime, so the anode-to-cathode charge swing
  is only **8.3%**. Lifetime is therefore intrinsically the weakest of the five parameters, and any
  estimator leaning on charge *normalisation* rather than *shape* is ~25× levered on charge-scale
  errors. Measured charge-weighted ⟨t/τ⟩ = 0.0391 → leverage 25.6×, against the 24.3× seen on the
  scan: exact agreement.
- **A standard shape-based fit recovers the lifetime and is immune to the dE/dx scale — but only
  on truth (§6k).** Binning dQ/dx by drift time gives **τ = 2173 ± 4 µs (−1.2%)**, unchanged to
  four significant figures under a ±2% charge rescale, as the physics requires.
- **On real hits that same fit collapses (§6l): −14.5% to +123.5% depending only on where the fit
  is started**, with errors 20–150× larger. Two front-end effects, both comparable to the entire
  8.3% attenuation signal: an anode-edge charge-*sharing* spike in the first ~20 µs (hits per pixel
  1.85 → 2.2 as diffusion sets in), and threshold selection (the 5 ke cut sits only ~2× below the
  10.7 ke median hit charge) which preferentially removes the most-attenuated hits and flattens the
  slope. **Per-hit charge is the wrong observable** — it measures attenuation and diffusion sharing
  together. This is an argument *for* the forward-model fit, which has both effects in the model.
- **CORRECTION to §6h.** The claim that the dE/dx *mean* is "the dominant bias channel" and its
  constraint weight "the highest-leverage parameter" was drawn from 1-D slices that freeze the
  other four parameters, leaving lifetime as the only knob able to change total charge. It is
  **withdrawn as a statement about our fits**: our fitted dE/dx mean is accurate to ~0.1%, and the
  §6h leverage mispredicts our fits' lifetime in magnitude everywhere and in *sign* for five of
  nine seeds. In a real fit A_b absorbs the normalisation (its own slice moves nearly 1:1 with a
  mean shift). The ranking among degradations, and the guess-file explanation, both survive —
  they were measured with the same estimator on both sides.
- **The guess file's +15.2% is explained (§6h), closing open question 11.** Its length-weighted
  dE/dx mean is 1.87704 against truth's 1.88533 — **−0.440%** — which at the measured slope
  predicts +12.9% against +15.2 ± 4.9% observed. The ~9 points that segmentation, spread, random
  displacement and chord-cutting all failed to account for were a sub-percent mean error.
- **Geometry error is not a bias channel, however it is shaped (§6h).** Fully straightened tracks
  (1011 µm mean deviation, systematic and correlated along track) give lifetime −0.99 ± 3.29% —
  indistinguishable from zero and from random displacement of the same size. Combining it with a
  dE/dx defect shows no interaction either. The position axis survived a test built to break it.
- **Open question 5 answered (§6i): batching order does not matter.** A paired shuffle-on/off run
  submitted together on one tree agrees within errors on every parameter.
- **More data buys precision and starves geometry (§6i).** 4× the data tightens long. diffusion's
  seed spread 23.2 → 13.7 as the bias/variance split predicts, but degrades position 492 → 774 µm
  because the geometry block gets 4× fewer updates per track.
- **The 4× data run failed, but the test was invalid (§6g).** Raising `max_nbatch` silently
  un-anneals the LR schedule, because the decay is indexed in epochs, not iterations. Redone
  correctly in §6n: with iterations scaled alongside batches, **2× data tightens every error bar
  and leaves geometry intact**.
- **WHAT THE OBJECTIVE IS MADE OF (§6q).** The loss is **91% joint hit log-intensity**, and that
  term supplies **~95% of the curvature** pinning both lifetime and A_b. The joint term contains
  **both** the tick residual and a Gaussian per-hit charge residual (σ_Q = 500 e⁻), marginalised
  together by `logsumexp`. **CORRECTION (2026-08-06):** an earlier version of this bullet claimed
  the objective had *no* per-hit charge term, reading `aux["log_likelihood_charge"] = 0.0` as a
  disabled term. It is a **reporting stub**: the charge residual is folded into the key stored as
  `log_likelihood_tick`. Proven by rerunning an identical scan at `--loss_sigma_charge 50` instead
  of 500 — the loss changes by **99.3%**, entirely inside that key. Lifetime therefore has **one**
  structural handicap (the 0.087-lifetime lever arm), not two. The open question is not *whether*
  to enable charge but *what σ_Q should be*: 500 e⁻ is 4.7% of a median hit, comparable to the
  entire 8.3% anode-to-cathode lifetime signal.
- **CONSOLIDATED PRODUCTION PERFORMANCE (§6q), 8 distinct seeds, repeats averaged within seed:**
  A_b **+0.73 ± 0.05%**, E field **−0.02 ± 0.01%**, transverse diffusion −0.72 ± 0.42%,
  longitudinal diffusion −7.86 ± 1.95%, lifetime **−4.21 ± 1.53%**, position 159 ± 4 µm.
- **THE THREE LEVERS, IN ORDER OF GAIN (§6r):** (1) drop the dE/dx prior weight **5 → 0.5** —
  largest available win, free, and production is mis-set; (2) **more data with iterations scaled
  alongside** — every central value moves toward truth and position holds at 165 µm, whereas more
  batches at *fixed* iterations starves the geometry block (774 µm); (3) **run at ≥ 6 seeds and
  settle knobs at ceiling**. The two hard limits are the **0.087-lifetime lever arm** and the
  **~45% dE/dx recovery ceiling** — neither is a tuning problem.
- **Half the scatter is bit-level irreproducibility (§6q).** Three runs of an identical config
  against an identical target spanned **6.35 points** on lifetime; within-seed s.d. is 2.22 against
  a between-seed 4.33. **More seeds cannot get below ~2.2 points** — that needs repeats at fixed
  seed or `--non_deterministic` off.
- **THE POWER FLOOR (§6n): n = 3 cannot resolve ~4 points.** Three runs of an *identical*
  configuration give lifetime means spanning **3.6 points** — larger than most effects this
  campaign has chased, including the 3.4-point difference on which the mean-constraint weight
  question previously died. Every remaining A/B comparison must be run at ceiling (which shrinks
  the seed s.d. 11–70×) and/or with ≥6 seeds.
- **The dE/dx minimum-length cut is retired (§6n).** A measured no-op in production (the guess
  file's shortest segment is 0.909 cm against a 0.15 cm cut) and a clean null where it does fire.
- **The dE/dx MAE puzzle is closed (§6n).** The block **improves 17% against truth** at production
  length while appearing to degrade 10% against its own starting point. It was the metric's
  reference all along.
- **The dE/dx block is objective-limited, not geometry-limited (§6o).** With *perfect* geometry the
  best run still recovers only **64%** of the prior→truth gap — a third of the per-segment
  structure is beyond reach whatever the geometry. Independent confirmation of the second-order
  study's verdict, measured on fits rather than inferred from curvature.
- **CONFIGURATION ERROR in every recent arm (§6o).** They were launched with `SCIDEDXPRIOR=5`,
  inherited from the fitted-geometry ANNEALLONG config, against the **0.5** default — a 10× stiffer
  dE/dx prior. They recover 17–19% of the gap where historical runs reach 40–64%. This entangles
  the `ceil_w3k` mean-constraint test, which varies one dE/dx weight while another sits 10× off
  default. `ceil_p05` is running to confirm.
- We also found and fixed **two measurement traps** that had produced two earlier false
  positives. Those are documented in §7 because they matter more than any single number.

---

## 2. Why everything was re-run: the noise mismatch

Every fit of the earlier campaigns passed `--no-noise`, which turns out to be **asymmetric**:

- the *target* had its noise stripped (`remove_noise_from_params`) → noiseless hits,
- but the *loss* used `ref_params`, whose noise fields were never stripped, so the
  probabilistic forward model kept analytically smearing with σ ≈ 900 e⁻.

So we were fitting **noiseless data with a noise-broadened model**, everywhere. This is
degenerate with longitudinal diffusion, which explains a long-standing `long_diff` deficit:
re-running one configuration noise-consistently moved `long_diff` from −19.2% to +12.6%.

The consistent, data-like mode — a stochastic target with FEE noise throws, fitted with a
probabilistic guess that marginalizes that noise — is now the default. **All results in this
report are noise-ON.** This invalidated the earlier campaign's numbers and is why the stage
ladder was re-run from S1.

---

## 3. The stage ladder: localising the failure

Four stages at 400 cm batch length, 3 target seeds each, isolating one difficulty at a time:

| stage | dE/dx | geometry |
|---|---|---|
| **S1** | true | true |
| **S2** | fitted (~4000 params) | true |
| **S3** | fitted | **wrong**, frozen at a straight-line guess |
| **S4** | fitted | **fitted** (spline basis) |

![Stage ladder](plots/noise_report/fig1_stage_ladder.png)

| stage | A_b | E field | lifetime | tran. diff. | long. diff. | position |
|---|---|---|---|---|---|---|
| S1 | −0.1 ± 0.1 | +0.1 ± 0.1 | +2.9 ± 2.5 | +0.5 ± 1.5 | +0.8 ± 0.8 | (true) |
| **S2** | **+0.3 ± 0.3** | **+0.2 ± 0.0** | **+0.6 ± 1.9** | **+1.9 ± 1.9** | **+0.8 ± 0.9** | (true) |
| S3 | +0.1 ± 0.6 | +0.0 ± 0.3 | +137.7 ± 47.4 | +16.6 ± 18.1 | +71.0 ± 49.2 | ~880 µm (frozen) |
| S4 | −1.1 ± 0.1 | +0.2 ± 0.2 | +62.7 ± 49.5 | +13.1 ± 4.7 | +65.2 ± 6.9 | 519 µm |

> **Corrected 29 Jul.** An earlier version of this table had wrong S1 and S3 rows. The S1 and
> S3 result directories contain **both** the 400 cm and 50 cm runs, and the analysis loader
> keyed only on the seed number, so it silently mixed a 50 cm seed into each of those two
> rows. The numbers above are 400 cm only. S2 and S4 were unaffected.

Readings (all %, mean ± s.d. over 3 seeds):

- **S2 is the reference success.** Fitting ~4000 nuisance dE/dx parameters jointly with the
  five calibration parameters costs essentially nothing when geometry is known.
- **Both stages that touch geometry break lifetime and long. diffusion**, while A_b and
  E field survive everywhere. That asymmetry is explained in §5.
- **A previously reported "S1 anomaly" was an analysis bug, not physics.** An earlier
  version of this report flagged S1 (+13 ± 12%) as inexplicably worse than S2 and listed it
  as an open question. It was the batch-length contamination described above. Corrected, S1
  is **+2.9 ± 2.5%** — consistent with S2, as it should be.

---

## 4. S4: everything we tried

S4 — joint calibration + per-segment dE/dx + per-track geometry — is the configuration that
matters for real data, where neither the track shape nor the energy deposition is known.
Roughly a dozen strategies were tested, 3–6 seeds each:

![S4 variants](plots/noise_report/fig2_s4_variants.png)

Every blue bar is a strategy that did **not** work: longer runs (10k iterations), a
drift-profile penalty on the dE/dx trend, a 10× stronger dE/dx prior, slowing the geometry
LR decay, freezing geometry after convergence, a two-pass scheme fitting calibration against
frozen already-fitted geometry, and several combinations.

The single most important methodological point of this campaign is visible in the
trajectories rather than the endpoints:

![Trajectories](plots/noise_report/fig3_trajectories.png)

The baseline and the former "winner" configuration do not oscillate around the true value —
they perform a **biased random walk that occasionally crosses truth**. Reading such a run at
a fixed iteration count can return any value you like. This produced **two separate false
positives** during the campaign (a configuration that read −3.8%/−1.6% at 5000 iterations
turned out to be +38 ± 26% when run to 8000 with more seeds). The annealed run in the right
panel behaves qualitatively differently: it converges into the ±5% band and stays there.

---

## 5. Why geometry poisons lifetime but not A_b or E field

We computed the exact Hessian of the loss at the true parameter values (S2 @400 cm,
noise-ON, 20 batches, sigmoid-normalised parameters):

![Hessian spectrum](plots/noise_report/fig5_hessian.png)

- The two softest modes are a **mixed (lifetime, long. diffusion) plane**, ~2400× flatter
  than the A_b direction and ~4×10⁶ flatter than E field. The corresponding statistical
  floors are **σ(lifetime) = 1.6%** and **σ(long_diff) = 1.4%** — and S2 sits at that floor,
  i.e. S2 is as good as this dataset allows.
- An extended 8×8 study adding three *rigid* geometry modes (offset, drift-scale, transverse
  offset) shows those modes are pinned by the data to **microns** (σ ≈ 0.5–1.3 µm), and that
  marginalising over them barely affects lifetime (3.75% → 3.79%) but inflates E field by
  **119×** — because a global drift-scale change mimics a drift-velocity change.
- Therefore **lifetime is immune to rigid geometry error**; its S4 degradation must come from
  *local, per-track* geometry noise being continuously injected while the geometry adjusts.

> ### UPDATE (29 Jul): the causal story in this section is contradicted by direct test.
> S3 has **frozen** geometry, so no per-track geometry noise can be injected at all — yet it
> fails at +137.7 ± 47.4%. Re-running S3 with the annealed LR schedule and the *same*
> ~880 µm-wrong frozen geometry recovers lifetime to **−0.9 ± 3.2%**.
>
> So the lifetime failure is an **optimizer-schedule pathology**; geometry error displaces the
> walk rather than causing the bias. Note the effect is parameter-specific: with wrong frozen
> geometry the diffusion constants remain poor (long. diff. +16 ± 25%, tran. diff. −4 ± 6%)
> versus −4 ± 10% and −0.1 ± 1.7% when geometry is fitted. **Lifetime becomes insensitive to
> geometry once annealed; the diffusion parameters do not.**
>
> Everything measured in this section (the eigen-decomposition, the soft plane, the drift
> projection) stands. What does not stand is the inference that local geometry noise is the
> mechanism behind the lifetime bias.

### 5.1 What a "mode" is, and why we decompose into them

Quoting a per-parameter uncertainty is misleading for this problem, because the parameters
are not independently constrained — *combinations* of them are. The natural language is the
eigen-decomposition of the loss curvature (the Hessian **H**) at a point:

- an **eigenvector** is a direction in the 5-dimensional parameter space, i.e. a specific
  *recipe* mixing A_b, E field, lifetime and the two diffusion constants;
- its **eigenvalue λ** is the stiffness of the loss along that direction;
- the uncertainty along a mode goes as **σ ∝ 1/√λ**, so small λ = a flat, poorly determined
  combination — a near-degeneracy.

Reading Fig 11 row by row: each row is one mode, each column a parameter, and the number in
the cell is that parameter's weight in the mode.

![Mode decomposition](plots/noise_report/fig11_mode_decomposition.png)

The important structural fact is that **the two soft modes (red boxes) are mixed**:

```
λ = 3.3e3   →   0.83 · lifetime  +  0.56 · long_diff      (softest)
λ = 6.1e3   →   0.56 · lifetime  −  0.83 · long_diff
```

Neither lifetime nor longitudinal diffusion is individually unconstrained. What is poorly
determined is the *pair*: you can raise lifetime and lower long. diffusion together and the
data barely notices. That is why these two parameters fail together in every configuration in
this report, and why fixing one without the other has never worked.

By contrast the stiff modes are almost pure single parameters — E field is its own mode at
λ = 1.3×10¹⁰, roughly **4 million times** stiffer than the softest direction, which is exactly
why E field survives every stage of the ladder.

The right-hand panel adds three *rigid* geometry modes. They are stiff (λ = 10⁷–10¹⁰, i.e.
pinned to microns), and they barely mix with lifetime — but note the λ = 3.0×10³ mode, where
E field and transverse diffusion mix with geometry. That mixing is the origin of the 119×
inflation of the E-field uncertainty when geometry is marginalised over.

### 5.2 What the soft mode means in practice

![Soft plane](plots/noise_report/fig12_soft_plane.png)

Left: the joint uncertainty ellipse in the (lifetime, long. diffusion) plane. It is strongly
elongated along the soft direction — the fit can slide a long way along that diagonal at
almost no cost in loss.

Right: a directly measured 2-D scan of the negative log-likelihood. The minimum sits
**exactly on truth**. This is worth emphasising for collaborators: **the loss function is not
biased.** The optimum is in the right place; the problem is purely that the valley floor is
nearly flat, so an optimizer can wander along it for thousands of iterations without the loss
objecting. Every failure mode in this report is a *navigation* problem, not a modelling error.

### 5.3 Static modes are not enough: the dynamic decomposition

The Hessian above is computed **at truth** and with only the 5 calibration parameters. The
real fit also carries ~4000 dE/dx nuisances and the per-track geometry, and it spends its
time away from truth. So we measured the modes the optimizer *actually travels in*: take the
parameter increments from the late half of each run, do a PCA, and ask how much of the leading
increment direction lies inside the soft plane predicted by the static Hessian.

![Dynamic modes](plots/noise_report/fig13_dynamic_modes.png)

| configuration | overlap with static soft plane (per seed) | mean |
|---|---|---|
| S2 — geometry known | 0.90, 0.94, 0.99, 0.88, 0.99, 0.85 | **0.93** |
| S4 baseline — geometry fitted | 0.97, 0.97, 0.34, 0.61, 0.76, 0.78 | **0.74** |
| S4 + LR anneal | 0.93, 0.90, 0.94 | **0.93** |

- With geometry known, the optimizer moves **inside the predicted soft plane** (0.93). The
  static picture describes the dynamics well.
- With geometry fitted, the overlap drops and becomes erratic (0.34–0.97). The fit is escaping
  into a direction the 5-parameter Hessian does not contain — a degeneracy created by the
  nuisance and geometry blocks, not visible in the calibration-only curvature.
- **Annealing restores the S2-like behaviour (0.93).** This is independent, mechanistic
  support for the LR-schedule diagnosis in §6: the fix does not merely improve the final
  numbers, it puts the optimizer back into the subspace the static analysis predicts.

The right panel shows what the travelled direction is made of: in all three cases it is
dominated by lifetime, with transverse and longitudinal diffusion next — consistent with the
soft plane being where the trouble lives.

### 5.4 Decomposing the geometry itself: the drift axis

The same logic applies to the *geometry* basis. Track shape is fitted as a smooth transverse
displacement expanded in sine modes about a straight axis, using two orthonormal vectors
(e₁, e₂) spanning the plane perpendicular to the track. That frame was chosen arbitrarily —
it has **no knowledge of which way the drift field points**.

That matters because displacement along the drift axis changes how much attenuation the
forward model applies to a segment, so it trades off directly against lifetime. Decomposing
the fitted displacement of all 864 tracks of an annealed run into drift-projected and
wire-plane components:

![Drift decomposition](plots/noise_report/fig14_drift_decomposition.png)

- median RMS displacement **along drift: 231 µm**, in the wire plane: 354 µm;
- the drift-projected part is a median **58% of the total displacement magnitude**;
- the second basis vector carries a median **|e₂·ẑ| = 0.94** — it is almost entirely aligned
  with the drift axis, purely by accident of the frame construction.

So a majority of the geometric freedom the fit is given points along precisely the direction
that is degenerate with the parameter we are trying to measure. The fix — build the frame so
that e₁ lies exactly in the wire plane and all drift sensitivity is isolated in e₂, then
optionally penalise that component — is implemented and under test; see §8.

**Caveat:** displacement along drift is not *impossible* in principle — real tracks are
three-dimensional. The objection is that the basis provides no separate control over the one
direction that is degenerate with attenuation, so the fit is free to use it to absorb
calibration error.

> ### UPDATE (29 Jul): this proposed fix has been TESTED and REFUTED.
> Four arms, 12 seeds, identical configuration apart from the penalty weight:
>
> | penalty | drift disp. | drift fraction | position | lifetime | long. diff. |
> |---|---|---|---|---|---|
> | **w = 0** (control) | 225 µm | 58% | **268 µm** | **+0.3 ± 2.7** | **−7.3 ± 5.3** |
> | w = 1e6 | 18 µm | 7% | 648 µm | +8.8 ± 6.0 | +59.0 ± 26.8 |
> | w = 1e7 | 2 µm | 1% | 686 µm | +8.1 ± 4.5 | +53.8 ± 17.3 |
>
> The penalty does exactly what it was designed to do — it drives the drift-projected
> component to ~zero — and **every physics number gets worse**, with position degrading 2.5×.
> The effect saturates rather than having a sweet spot: no weight helps.
>
> A separate control (drift-aligned frame, penalty **off**) reproduces the baseline exactly
> (drift fraction 58% vs 57%, identical A_b and E field), proving the frame rotation is
> inert and the penalty alone causes the damage.
>
> **Conclusion:** the drift-axis displacement is largely *legitimate* track structure, not
> absorbed calibration error. Clamping it removes a degree of freedom the geometry genuinely
> needs. The measurement in this section is sound; the causal interpretation was not.

---

## 6. The current best result: anneal the calibration learning rate

Diagnosing the optimizer schedule showed the calibration LR effectively **never annealed**
over a run (dE/dx stayed at ~100% of its initial LR, calibration ~93%, geometry 0.03%). So
the nuisance parameters and the parameters of interest were being updated at the wrong
relative rate for the entire fit — a classic recipe for the nuisances absorbing signal that
belongs to the physics parameters.

What "annealing" concretely means here: the schedule is
`optax.warmup_exponential_decay_schedule` with `transition_steps = epoch_size` and
`staircase=True`, so the decay is applied **once per epoch** (100 iterations), not per step.
Over 5000 iterations = 50 epochs, a decay of 0.999 leaves **95.7% of the peak LR at the end of
the run** — i.e. essentially no annealing at all — while 0.91 leaves 1.6%.

![LR schedules](plots/noise_report/fig16_lr_schedules.png)

The other parameter blocks were never changed: the ~4000 dE/dx nuisances run on a bare
`optax.adam(1e-2)` with **no schedule whatsoever** (100% of initial for the whole run), and
geometry decays to 22% by iteration 5000. So under the old setting the five physics parameters
stayed as "hot" as the nuisances for the entire fit. Annealing quenches the parameters of
interest while the nuisances keep adapting, so late in the run the nuisances absorb residuals
*around* nearly-fixed calibration values instead of calibration chasing nuisance fluctuations.
It also cuts the total distance Adam can travel (Σ LR) by 3.5×.

Changing the calibration LR schedule decay from 0.999 to **0.91** gives:

![ANNEAL recovery](plots/noise_report/fig7_anneal_recovery.png)

| | A_b | E field | lifetime | tran. diff. | long. diff. | position |
|---|---|---|---|---|---|---|
| S4 baseline | −1.1 ± 0.1 | +0.2 ± 0.2 | +62.7 ± 49.5 | +13.1 ± 4.7 | +65.2 ± 6.9 | 519 µm |
| S4 previous best (8k, 6 seeds) | +0.1 ± 0.4 | +0.2 ± 0.7 | +40.5 ± 24.7 | −1.1 ± 5.0 | +37.6 ± 15.3 | 327 µm |
| **S4 + LR anneal** | **+0.6 ± 0.1** | **−0.1 ± 0.0** | **−2.6 ± 2.0** | **−0.7 ± 1.2** | **−5.1 ± 7.3** | **255 µm** |
| S2 control + anneal | +0.2 | −0.0 | −2.8 | −0.1 | +3.1 | (true) |

The S2 control confirms annealing does not damage the already-solved stage.

### Iteration-level diagnostics

The summary table hides a lot. Here is every calibration parameter against iteration for the
three annealed runs:

![ANNEAL parameter traces](plots/noise_report/fig8_anneal_param_traces.png)

- **A_b and E field** converge quickly and stay put — E field within a few hundred iterations.
- **tran. diffusion** is a genuine recovery: seed 1 starts **+120%** off and converges to ~0.
- **long. diffusion is the problem child.** Seeds 0 and 1 start *below* truth, overshoot to
  **+50% and +100%**, and are still drifting downward through zero at iteration 5000. Seed 2
  never recovers, sitting near −15%. This parameter has clearly not settled.
- Note the legend: each seed's **starting** offset differs a lot, because `--seed` draws a
  different target. The seeds are not facing equally hard problems.

Position precision, the dE/dx nuisance accuracy, and the loss:

![Position and dEdx vs iteration](plots/noise_report/fig9_anneal_pos_dedx.png)

Three things stand out, and all of them argue that 5000 iterations is not enough:

1. **Position is still improving.** The residual falls roughly log-linearly from ~900 µm to
   ~230 µm and is *still descending* at the last iteration — no plateau. The "255 µm" quoted
   above is therefore a snapshot, not a converged value.
2. **The dE/dx nuisance gets worse, not better.** Per-segment MAE rises sharply from
   0.030 to 0.044 MeV/cm over the first ~1000 iterations, then only partially recovers to
   ~0.039 — ending roughly **30% worse than it started**. The ~4000 free dE/dx parameters are
   absorbing something during the phase when the calibration parameters are moving fastest.
   This is consistent with the nuisance/parameter timescale-mismatch diagnosis in §6.
3. **The loss is still decreasing** monotonically at 5000 iterations.

### How much work did each fit actually do?

A small final error is *not* by itself proof of recovery: if a seed's target happens to sit
close to the initial guess, a fit that barely moves still scores well. The honest metric is
the fraction of the initial init→target offset that the fit removed:

![Gap closed](plots/noise_report/fig10_gap_closed.png)

| parameter | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| A_b | −4.2 → +0.6 (**85%**) | −2.6 → +0.5 (**80%**) | −2.9 → +0.7 (**75%**) |
| E field | −4.1 → −0.1 (**98%**) | −4.2 → −0.0 (**99%**) | +10.5 → −0.1 (**99%**) |
| lifetime | −8.6 → −1.6 (**81%**) | +89.6 → −0.8 (**99%**) | −8.0 → −5.5 (**32%**) |
| tran. diff. | −12.2 → −0.4 (**96%**) | +119.9 → −2.2 (**98%**) | −7.3 → +0.7 (**91%**) |
| long. diff. | −31.2 → −3.0 (**90%**) | −2.8 → +2.6 (**9%**) | −20.7 → −15.0 (**28%**) |

This materially qualifies the headline:

- **E field and tran. diffusion are convincingly recovered** on all three seeds, including
  seed 1's +120% starting offset for tran. diffusion.
- **Lifetime rests mainly on seed 1**, which started **+89.6%** away (2200 → target 1160,
  nearly a factor of two) and closed **99%** of that gap. That is a real, hard recovery.
  Seed 0 closed 81%. Seed 2 closed only **32%** — it started 8.0% away and ended 5.5% away,
  so its "−5.5%" is largely *un-moved*, not converged. It is also the seed that failed the
  convergence criterion below.
- **long. diffusion is the weakest claim in this report.** Two of three seeds started close
  to truth and closed almost none of their gap (9% and 28%).

So "lifetime −2.6 ± 2.0%" is **not** three independent confirmations of a few-percent
recovery: one seed did the hard work, one did most of it, and one barely moved. Future seed
sets should be chosen with deliberately distant lifetime and long-diffusion targets so that
every seed constitutes a real test.

### Reproducibility: three fresh seeds (n = 6)

Three additional seeds (3–5) were run at the identical configuration. They reproduce the
recovery, and they substantially answer the "how much work did the fit do?" objection above.

![Reproducibility at n=6](plots/noise_report/fig15_annealmore_n6.png)

| parameter | pooled (n = 6) | per-seed gap-closed |
|---|---|---|
| A_b | **+0.63 ± 0.09%** | 80, 75, 85, −83, 82, 92 |
| E field | −0.06 ± 0.02% | 99, 99, 98, 99, 99, 93 |
| tran. diff. | −0.82 ± 1.51% | 98, 91, 96, 97, 90, 95 |
| **lifetime** | **−2.10 ± 4.49%** | 99, 32, 81, 64, 85, 91 |
| long. diff. | −5.02 ± 8.43% | 9, 28, 90, 87, 43, 87 |
| position | **255 ± 13 µm** | — |

- **The lifetime claim no longer rests on one seed.** Five of six seeds close **≥64%** of their
  initial offset (the new seeds start 85%, 91% and 64% away and converge). At n = 3 only one
  seed had done substantial work; that was the main weakness of the earlier reading.
- **`long_diff` also firms up** — three of the new seeds close 90/87/87%, versus one of three
  before. It remains the loosest parameter.
- **The error bar grew, correctly.** Lifetime moves from −2.6 ± 2.0% (n = 3) to
  **−2.10 ± 4.49%** (n = 6). The earlier ±2.0% was optimistic; ±4.5% is the honest spread.
- **Position is very stable**: 255 ± 13 µm across six independent seeds.

**New observation — a small systematic in A_b.** Across all six seeds A_b lands at
**+0.63 ± 0.09%**, and no seed is negative. That is a ~7σ offset from zero: a genuine
reproducible bias rather than scatter (E field shows a smaller, similarly consistent
−0.06 ± 0.02%). Both are far inside the 5% target and do not threaten the result, but "small
and systematic" is a different statement from "small and noisy", and it may indicate a mild
forward-model or prior mismatch. It should not be averaged away.

### Honest caveats

- **n = 6 seeds** at 5000 iterations for the headline configuration.
- **One seed had not converged.** We set an acceptance rule *before* looking at the results:
  (a) |lifetime| < 10% on all seeds, and (b) no monotone drift > 5 percentage points over the
  last 30% of iterations. Criterion (a) passes (−1.6 / −0.8 / −5.5%); **criterion (b) fails on
  seed 2**, which moved −8.0 points. A 10 000-iteration run is in flight to settle this.
- **Nothing in these runs is actually converged at 5000 iterations**: position is still
  descending, the loss is still falling, and long. diffusion is still drifting through zero
  (see the diagnostics above). The 5000-iteration numbers should be read as a strong interim
  result, not a final measurement.
- `long_diff` remains the loosest parameter (−5.1 ± 7.3%), is not fully understood, and — per
  the gap-closed analysis — is the least well evidenced number in this report.
- **The dE/dx nuisance ends ~30% worse than it started**, which is unexplained and may
  indicate the nuisance block is still absorbing signal even under the annealed schedule.
- Full configuration: `--lr_scheduler warmup_exponential_decay_schedule`,
  `--lr_kw '{"decay_rate": 0.91, "init_value": 0, "warmup_steps": 500}'`, `--lr 1e-1`,
  `--chain_lr 1e-2`, `--chain_decay_rate 0.9997`, `--chain_basis spline`,
  `--chain_spline_knot_cm 40`, `--dedx_prior_weight 5.0`, `--max_batch_len 400`,
  `--max_nbatch 100`, noise ON, 5000 iterations.

---

## 6b. Convergence of the 10 000-iteration runs (ANNEALLONG)

The 5000-iteration runs above were **not converged** — position, loss and long. diffusion were
all still moving. A 3-seed run at 10 000 iterations with the identical schedule settles that.

![ANNEALLONG traces](plots/noise_report/fig17_anneallong_traces.png)

Convergence is strongly parameter-dependent: **E field** is done by ~500 iterations, **A_b** by
~3000, **tran. diffusion** enters the ±5% band around 4000, and **lifetime** and **long.
diffusion** — the two soft ones — only flatten at 5000–6000, which is where the calibration LR
reaches ~1% of peak.

![ANNEALLONG convergence](plots/noise_report/fig18_anneallong_convergence.png)

Panel (a) is the decision-relevant one: it plots the value you *would report* had you stopped
at each iteration, using the trailing-20%-median estimator. The curves swing by ±20% out to
~3000 and go flat from ~6000. Panel (b) shows the residual drift per 500-iteration window
crossing below the 1.25-point run-to-run noise floor at **iteration ~6000–6500**. Panel (c)
shows the step size falling four orders of magnitude — the quench itself.

**Important caveat.** The flatness beyond 6000 is *enforced* by the learning rate going to
zero, not independently earned. These plots demonstrate that the answer is **stable**, not
that it is **right**. A run with a slower decay over a longer horizon (same final LR, 1.78×
more total travel) is in progress to separate those.

![ANNEALLONG position/dEdx/loss](plots/noise_report/fig19_anneallong_aux.png)

| parameter | seed 0 | seed 1 | seed 2 | mean ± s.d. | tail drift |
|---|---|---|---|---|---|
| A_b | +0.67 | +0.94 | +0.64 | **+0.75 ± 0.13** | 0.00 |
| E field | −0.03 | −0.05 | −0.02 | **−0.03 ± 0.01** | 0.00 |
| tran. diff. | −1.26 | −0.25 | −1.35 | −0.96 ± 0.50 | 0.03 |
| long. diff. | −8.72 | −3.44 | −4.05 | −5.40 ± 2.36 | 0.13 |
| lifetime | −4.27 | −3.79 | −0.05 | −2.70 ± 1.89 | 0.12 |
| position | 157 | 162 | 183 | **168 ± 11 µm** | — |

All tail drifts are two orders of magnitude below the noise floor. Compared with the same
seeds at 5000 iterations, **calibration is unchanged within noise** but **position improves
255 → 168 µm** and the seed spreads tighten (long. diff. ±9.7 → ±2.4). Geometry has its own,
much slower decay, so it keeps improving long after calibration has been quenched.

**10 000 iterations is therefore strictly better than 5000** for the same configuration: the
same calibration, substantially better geometry, tighter seed spread, at 2× the compute.

---

## 6c. Schedule invariance — is the answer converged or just quenched?

§6b showed the annealed fit is **stable**, but stability there is *enforced* by the learning
rate going to zero. To separate "converged" from "quenched", a third schedule was run:
**decay 0.9539 over 10 000 iterations**, chosen so the FINAL learning rate matches ANNEAL's
(1.19% vs 1.58% of peak) while the **total optimizer travel** (ΣLR) is 1.78x larger.

![Schedule invariance](plots/noise_report/fig20_schedule_invariance.png)

| run | decay | iters | total travel ΣLR | lifetime | position |
|---|---|---|---|---|---|
| ANNEAL | 0.91 | 5 000 | 134.5 | −2.01 ± 2.17 | 255 µm |
| ANNEALLONG | 0.91 | 10 000 | **136.0** | −2.70 ± 1.89 | 168 µm |
| SLOWANNEAL | 0.9539 | 10 000 | 239.4 | **−5.02 ± 2.56** | 162 µm |

**Most parameters are schedule-invariant.** A_b (+0.58 → +0.75 → +0.82), E field
(−0.02 → −0.03 → −0.07), transverse diffusion (−0.11 → −0.96 → −0.64) and longitudinal
diffusion (−4.04 → −5.40 → −4.81) all agree within the 1.25-point run-to-run noise floor
across a 1.78x change in total travel. That is the signature of a genuine attractor rather
than an arbitrary freezing point.

**Lifetime is not.** It moves −2.01 → −2.70 → −5.02 %, monotonically with travel, and the
2.3-point shift from ANNEAL to SLOWANNEAL is about 2x the noise floor.

**An important correction to §6b's framing.** The travel column shows ANNEALLONG added only
**1%** more travel than ANNEAL despite doubling the iteration count — because with decay 0.91
the learning rate is effectively zero beyond ~6 000 iterations. So ANNEALLONG was never an
independent check of the calibration numbers; it was the same amount of optimisation spread
over twice as many steps. (Its position improvement, 255 → 168 µm, is real: geometry has its
own much slower decay and kept moving.) Only SLOWANNEAL genuinely varies the travel.

**Consequence: the reported lifetime error is a lower bound on the bias, not a measurement of
it.** More optimisation moves it further negative, so the true value lies beyond −5%. This is
consistent with the gradient audit (§7), where lifetime carried the largest tail-gradient
t-statistics.

![SLOWANNEAL traces](plots/noise_report/fig21_slowanneal_traces.png)

![SLOWANNEAL convergence](plots/noise_report/fig22_slowanneal_convergence.png)

The honest summary for lifetime is therefore **−2% to −5%, not converged**, while the other
four parameters are converged and schedule-invariant.

---

## 6d. Two ablations: does the dE/dx block earn its keep, and is the target biased?

Two arms were run at exactly the ANNEALLONG configuration (10 000 iterations, len400, b100,
LR decay 0.91, chain_lr 1e-2, 3 seeds each) changing **one thing each**:

| arm | change | question |
|---|---|---|
| `CONSTDEDX` | dE/dx **frozen at the mean**, not fitted (≈4000 nuisances removed) | is the dE/dx block absorbing real structure, or just adding freedom? |
| `PROBTGT` | target drawn from the **probabilistic distribution** (`--probabilistic_sampling_target`) | does our deterministic target bias the recovered calibration? |

Both arrays are complete (3/3 seeds, exit 0) and **converged** — the residual-drift diagnostic
of §6b is below 1% of target for every parameter, so these are read-out values, not snapshots
of a moving run.

| parameter | ANNEALLONG (baseline) | CONSTDEDX | PROBTGT |
|---|---|---|---|
| A_b | +0.75 ± 0.16% | +1.03 ± 0.15% | +0.72 ± 0.12% |
| E field | −0.03 ± 0.02% | −0.03 ± 0.01% | −0.03 ± 0.00% |
| tran. diffusion | −0.96 ± 0.61% | −1.28 ± 0.96% | +0.61 ± 1.37% |
| **long. diffusion** | −5.40 ± 2.89% | **−21.08 ± 4.73%** | −7.23 ± 4.92% |
| **lifetime** | −2.70 ± 2.31% | **−7.22 ± 2.88%** | −3.23 ± 1.93% |
| dE/dx MAE | 0.03456 | *(not fitted)* | 0.03368 |
| position residual | 168 ± 14 µm | 158 ± 9 µm | 173 ± 6 µm |

![Global comparison of the annealed family](plots/noise_report/fig23_ablation_global.png)

### PROBTGT: the bias check passes

**No detectable bias.** Every parameter agrees with ANNEALLONG within uncertainties — A_b
+0.72 vs +0.75, E field identical to two decimals, lifetime −3.23 vs −2.70, long. diffusion
−7.23 vs −5.40. dE/dx MAE is marginally *better* (0.0337 vs 0.0346) and the position residual
is consistent. Drawing the target from the probabilistic distribution does not shift the
recovered calibration, so the deterministic target used throughout the campaign is not a
source of bias. This closes one of the open questions in §9.

### CONSTDEDX: the dE/dx block is doing real work

Freezing dE/dx at the mean **degrades long. diffusion ~4×** (−5.4% → −21.1%) and **lifetime
~2.7×** (−2.7% → −7.2%). Both shifts are ≈3σ against the seed-to-seed scatter, so this is not
noise. A_b also worsens slightly (+0.75 → +1.03%); E field and transverse diffusion are
untouched.

![dE/dx ablation trajectories](plots/noise_report/fig24_ablation_traces.png)

The trajectories show *when* it happens: all three arms track each other for the first ~3000
iterations, then CONSTDEDX separates downwards and settles onto a distinctly worse plateau,
flat through the read-out window.

**Interpretation.** The ≈4000 per-segment dE/dx nuisances absorb genuine per-segment
energy-deposition variation. Remove them and that variation does not disappear — it is
re-absorbed by the two parameters able to mimic a drift-dependent charge profile, which are
exactly long. diffusion and lifetime. This is direct evidence that the dE/dx block is not
merely soaking up freedom; it is separating a real effect from the calibration parameters.

It also sharpens §5: long. diffusion and lifetime are the *soft* directions, and this ablation
shows they are soft specifically with respect to per-segment charge structure.

![What the ablation costs elsewhere](plots/noise_report/fig25_ablation_cost.png)

The position residual is **statistically unchanged across all four arms** (158 ± 9 to
173 ± 6 µm; the CONSTDEDX–ANNEALLONG difference of 158 ± 9 vs 168 ± 14 is **not** significant
at n = 3). So removing the dE/dx block costs calibration accuracy without buying geometry
accuracy — the trade is one-sided. This is also consistent with the position residual being a
data/target floor rather than something the optimiser is competing for.

Panel (b) is worth reading with the per-batch caveat of §7 in mind: `dedx_mae_iter` is a
*per-batch* metric that swings between 0.02 and 0.06 within a single epoch, so it is smoothed
over one full epoch (100 batches). Smoothed, PROBTGT and ANNEALLONG are indistinguishable —
another way of seeing the null result — and both still end **above** where they started, which
is open question 7 and remains unexplained.

**Caveats.** n = 3 seeds per arm. The CONSTDEDX arm removes the nuisances *and* replaces them
with the mean, so it does not separate "no per-segment freedom" from "wrong per-segment
values" — an arm with dE/dx frozen at the *true* per-segment values would, and has not been
run. The PROBTGT null result is a null at this precision (≈±2–5% on the soft parameters), not
a proof of zero bias.

---

## 6e. Where is the loss minimum? A direct likelihood scan

Lifetime moves monotonically *further* from truth the more the optimizer works (ΣLR 134.5 →
−2.01%, 136.0 → −2.70%, 239.4 → −5.02%). Two explanations fit that equally well: either the
minimum is at truth and annealing has not arrived, or the minimum is genuinely displaced and
better optimization converges to a wrong answer. **No fit can separate them** — so we stopped
fitting and evaluated the loss on a grid (`--fit_type scan`, `LikelihoodProfiler`).

Each of the five parameters is scanned across its `ranges.py` interval on a 21-point grid with
the other four pinned at truth, i.e. a 1-D slice through the true point, for ~21 batches × 3
seeds (63 independent curves per parameter). Run twice: sim geometry = target geometry (**true**),
and sim geometry = the straight-line guess (**guess**).

| parameter | grid step | **true geometry** | **guess geometry** | well depth (true) |
|---|---|---|---|---|
| A_b | 0.63% | +0.07 ± 0.02% | +0.54 ± 0.11% | 19.7% |
| E field | 1.00% | +0.05 ± 0.00% | +0.07 ± 0.01% | 266% |
| tran. diffusion | 5.68% | +0.81 ± 0.63% | −8.38 ± 3.49% | 2.70% |
| long. diffusion | 8.75% | **−5.26 ± 1.78%** | **−34.10 ± 9.64%** | **0.95%** |
| lifetime | 10.23% | **+2.18 ± 0.50%** | **+15.22 ± 3.44%** | 43.2% |

> **Superseded error bars — see §6f.** The errors in this table divide the per-batch scatter by
> √63, treating 21 batches × 3 seeds as 63 independent curves. They are not: with
> `--probabilistic_sim` the forward model is analytic and the three seeds agree to 3 decimal
> places, so the effective n is 21 and these errors are **too small by √3** (lifetime ±0.50 →
> ±0.80, long. diffusion ±1.78 → ±4.11). Separately, the coarse grid adds a **±2.4-point**
> systematic on the lifetime minimum. Central values are unchanged; §6f carries the corrected
> uncertainties.

Quoted errors are the per-batch scatter divided by √63 (per-batch sd is 4.0% for lifetime,
14.2% for long. diffusion). **The grid is coarse** — the lifetime step alone is 10.2% of truth —
so the sub-grid parabolic minimum is only meaningful because it is averaged over 63 curves; the
grid spacing is drawn on every panel of the figure for exactly this reason.

![Likelihood scans](plots/noise_report/fig26_likelihood_scans.png)

![Scan zoom on lifetime and long. diffusion](plots/noise_report/fig27_scan_zoom.png)

### Two readings, and they point in opposite directions

**The objective is mildly displaced even with perfect geometry.** Lifetime's minimum sits at
+2.18% and long. diffusion's at −5.26%. So it is *not* true that a perfect optimizer would land
on truth. **How significant that is has since weakened** (§6f): with the corrected error bars
and the grid systematic these read +2.18 ± 0.80 (stat) ± 2.4 (grid) and
−5.27 ± 4.11 (stat) ± 1.2 (grid), so neither is more than ~1σ from zero on its own. The
*relative* statements below, and the whole §6f ladder, are unaffected because the grid
systematic is common-mode; the absolute displacement needs a finer grid to establish.

**Comparing the minimum against where our fits actually land splits the problem cleanly:**

| parameter | loss minimum | ANNEALLONG fit | reading |
|---|---|---|---|
| long. diffusion | −5.26 ± 1.78% | −5.40 ± 2.89% | **agree** — the optimizer *is* finding the minimum; the −5.4% is an **objective** displacement, not an optimizer failure |
| lifetime | +2.18 ± 0.50% | −2.70 ± 2.31% | **~5 points apart** — the fit undershoots the minimum, so this part *is* optimizer-side |

That retires open question 2 in its old form: long. diffusion is not "unexplained loose", it is
sitting where the objective's minimum is. And it keeps the lifetime work alive — the ΣLR drift is
a real optimizer deficit against a nearly-correct target.

**Geometry error moves the minima, and asymmetrically.** With the straight-line guess, lifetime
+15.2%, long. diffusion −34.1%, transverse diffusion −8.4%, while A_b (+0.54%) and E field
(+0.07%) barely move. This is the §5 asymmetry measured directly rather than inferred from
Hessian modes, and it explains the S3 stage mechanistically: the optimizer was correctly
converging to a minimum that geometry error had displaced.

The **well depths** explain the error ordering independently: across its full scan range the loss
varies by 266% for E field but only **0.95% for long. diffusion**, which is why that parameter
carries the largest uncertainty everywhere and is the most sensitive to geometry.

**Caveats.** These are 1-D *slices*, not profile likelihoods — the other four parameters are held
at truth rather than re-minimised, so the (lifetime, long. diffusion) correlation identified in §5
is not accounted for. A displaced minimum here should be re-checked with the others profiled
before being called a bias. All three seeds share one target (`--scan_tgt_nom`), so the seed
spread measures noise realisations, not target draws.

> **Analysis correction.** The first version of this scan selected each parameter's block by
> masking on "all other parameters at nominal". During the *other* parameters' blocks the scanned
> parameter sits at exactly its nominal value, so those rows leaked in — the nominal grid point
> acquired n = 42 curves instead of 20 and an anomalously high mean loss, corrupting the argmin
> and briefly producing the wrong conclusion that every minimum sat on truth. The blocks are now
> segmented by index arithmetic from the loop structure (`for batch: for param: for step`), which
> cannot leak. The numbers above are from the corrected extraction.

---

## 6f. The quality ladder: how good must the inputs be? (24 new scans)

§6e established the two endpoints — perfect geometry displaces lifetime by +2.2%, the
straight-line guess by +15.2% — but not the shape in between, and it could not say *which*
defect of the guess file was responsible. So we interpolated deliberately, along two axes
separately, using `optimize/scripts/make_quality_ladder.py` to perturb the **true** file (the
guess file cannot be interpolated toward: it has a different segmentation, 10.08M vs 3.03M
segments, so there is no row correspondence):

- **position ladder** — each trajectory displaced by one rigid random 3-D offset of
  **50 / 170 / 400 / 880 µm** RMS. Rigid rather than per-segment jitter because that is what
  the spline geometry basis represents and what the fitted position residual (~168 µm) measures.
  50 µm is our measured accuracy floor, 170 µm our actual fit quality, 880 µm the guess file's error.
- **dE/dx ladder** — dE/dx blended toward its global mean, `dEdx' = mean + f·(dEdx − mean)`,
  at **f = 0.75 / 0.5 / 0.25 / 0**, with `dE` and `n_electrons` rescaled consistently. f = 1 is
  truth, f = 0 is the CONSTDEDX condition of §6d.

24 scans (2 ladders × 4 rungs × 3 seeds), same grid and batching as §6e, all COMPLETE.

| condition | A_b | E field | tran. diff. | long. diff. | lifetime |
|---|---|---|---|---|---|
| **true** | +0.07 ± 0.03 | +0.05 ± 0.00 | +0.81 ± 1.55 | −5.27 ± 4.11 | +2.18 ± 0.80 |
| pos 50 µm | +0.06 ± 0.03 | +0.05 ± 0.00 | +0.02 ± 1.38 | −0.16 ± 2.58 | +2.12 ± 0.81 |
| pos 170 µm | +0.12 ± 0.04 | +0.05 ± 0.00 | −0.18 ± 2.40 | +0.93 ± 9.10 | +3.17 ± 0.83 |
| pos 400 µm | +0.01 ± 0.10 | +0.05 ± 0.01 | +0.83 ± 3.43 | +0.52 ± 17.2 | +1.19 ± 2.06 |
| pos 880 µm | −0.21 ± 0.34 | +0.10 ± 0.04 | −1.30 ± 10.3 | +41.3 ± 40.9 | −3.82 ± 6.34 |
| dE/dx f = 0.75 | +0.15 ± 0.05 | +0.05 ± 0.00 | −0.68 ± 1.40 | −9.70 ± 4.29 | +3.76 ± 1.18 |
| dE/dx f = 0.5 | +0.24 ± 0.07 | +0.05 ± 0.00 | −1.82 ± 1.39 | −15.95 ± 4.75 | +5.37 ± 1.85 |
| dE/dx f = 0.25 | +0.32 ± 0.10 | +0.05 ± 0.00 | −2.84 ± 1.63 | −21.24 ± 4.34 | +6.78 ± 2.71 |
| dE/dx f = 0 | +0.34 ± 0.14 | +0.05 ± 0.00 | −3.58 ± 1.87 | −23.77 ± 4.53 | +7.01 ± 3.49 |
| **guess** | +0.54 ± 0.17 | +0.07 ± 0.02 | −8.38 ± 4.78 | −34.11 ± 19.7 | +15.22 ± 4.87 |

![Scan wells along both ladders](plots/noise_report/fig28_ladder_scans.png)

![The calibration curve](plots/noise_report/fig29_ladder_calibration.png)

### The two axes do completely different things

![Bias versus variance](plots/noise_report/fig30_bias_vs_variance.png)

**dE/dx fidelity displaces the minimum — this is a bias.** Every vulnerable parameter moves
monotonically across all four rungs: long. diffusion −5.3 → −23.8%, lifetime +2.2 → +7.0%,
transverse diffusion +0.8 → −3.6%, A_b +0.07 → +0.34%. The error bars stay roughly constant
(fig 30b), so the well is *sliding*, not blurring — visible directly as the ordered fan of
minima in fig 28, lower right. Monotonicity across five ordered points is far stronger evidence
than any single rung's error bar.

**Position error inflates the uncertainty — this is variance, not bias.** No position rung
shows a centre shift significant against its own error (the largest, pos 170 µm vs true on
lifetime, is +0.99 ± 1.11). What changes is the determinacy: long. diffusion's bootstrap error
runs 4.1 → 2.6 → 9.1 → 17.2 → 40.9 points, and at 880 µm the scan is simply uninformative
(+41 ± 41%). Fig 28, lower left shows the mechanism — the 880 µm well is nearly flat from 0 to
+60%, so the argmin wanders. **The apparent long. diffusion "improvement" at 50–400 µm is not
significant and we do not believe it is real.**

**A_b and E field are immune on both axes.** E field sits at +0.05% in nine of the ten
conditions. This is the §5 stiff/soft split measured directly on the objective.

**Practical calibration curve.** Interpolating the dE/dx ladder, holding lifetime bias below
1 point of extra displacement needs f ≳ 0.85; below 3 points needs f ≳ 0.45. On the position
axis, bias is flat but the *statistical* usefulness of the fit collapses beyond ~400 µm — which
is the more useful threshold, and our current 168 µm sits comfortably inside it.

**The guess file is worse than having no dE/dx information at all** (+15.2% vs +7.0% on
lifetime; −34.1% vs −23.8% on long. diffusion), and worse than 880 µm of position error. So its
damage is *not* explained by either axis alone, nor by their sum. The remaining suspect is its
different segmentation — 3.3× more segments, so a per-segment dE/dx prior tuned on the true
segmentation is being applied to a different `dx` distribution. That is a concrete, testable
hypothesis and it is the natural next scan.

### Two analysis defects found and fixed

Both would have silently corrupted the numbers, and one invalidates an error bar quoted in §6e.

> **The per-parameter history files are snapshots, not copies.**
> `history_<param>_batch20_<label>.pkl` is the *same* growing history dumped when the scan
> reaches (batch 20, that parameter). The `Ab` file therefore stops 4 blocks short of the
> `lifetime` file, and reading it silently drops the last batch for 4 of the 5 parameters.
> Always read the last parameter in the loop order. `analyze_quality_ladder.py` does.

> **The three seeds are not independent, so §6e's error bars were too small.** With
> `--probabilistic_sim` the forward model is analytic, and seeds 0/1/2 place the lifetime
> minimum at +2.183% / +2.183% / +2.183%. §6e quoted "per-batch scatter / √63", treating
> 21 batches × 3 seeds as 63 independent curves; the effective n is 21, so those errors were
> **too small by √3**. Corrected here by bootstrapping over batches — which lands at ±0.80 for
> lifetime against §6e's ±0.50, exactly the expected √3. The §6e central values are unaffected.

### The grid is coarse, and here is what that costs

The lifetime scan step is 10.2% of truth while the Fisher width of the likelihood is **0.88%**,
so the argmin's own neighbours already sit 6–10σ away and the parabola vertex is an
extrapolation, not an interpolation. Fitting 5 points instead of 3 moves the lifetime vertex by
**+2.4 points** and long. diffusion by ~1 point.

This is a real systematic on the **absolute** minimum positions in this section and in §6e — the
"+2.18%" is `+2.18 ± 0.80 (stat) ± 2.4 (grid)`, i.e. its 2.5σ significance over zero does *not*
survive. But on lifetime the shift is almost perfectly **common-mode** across conditions
(+2.61, +2.42, +2.24, +2.04, +2.02, +2.04 going down the dE/dx ladder), so it **cancels in the
condition-to-condition differences that this section is built from**. The ladder trends are
robust; the absolute offsets are not. A finer grid around the minimum would settle both and is
cheap — it is now the top scan priority.

For reference, the statistical resolution of this dataset (from the NLL curvature at the
undegraded point) is A_b 0.037%, E field 0.002%, transverse diffusion 0.89%,
long. diffusion 2.07%, lifetime 0.88%.

Readout: `analyze_quality_ladder.py` (both traps documented in the docstring),
figures `make_ladder_plots.py`, summary `quality_ladder_summary.json`.

---

## 6g. Resolving the grid, and what the guess file's damage actually is

§6f left three things open: the grid was too coarse to trust absolute minima, the guess file's
+15.2% did not decompose, and the "4× data" run was still in flight. All three now have answers,
and two of them overturn something written above.

### The fine grid: the objective is close to unbiased with perfect geometry

`LARND_SCAN_WINDOW` (new, default unset = previous behaviour exactly) narrows the scan to
`nom·(1 ± w)` instead of the full `ranges.py` interval. Cost is steps × batches and does not
depend on the span, so resolution is free. At 41 steps × 41 batches with w = 0.15 the lifetime
step falls from 10.2% to **0.75%** — below the 0.88% Fisher width — and the batch count doubles.

| condition | grid | lifetime | long. diffusion |
|---|---|---|---|
| true | coarse (10.2%) | +2.18 ± 0.80 | −5.27 ± 4.11 |
| **true** | **fine (0.75%)** | **+1.14 ± 0.57** | **−1.67 ± 1.41** |
| dedx f = 0.5 | coarse | +5.37 ± 1.85 | −15.95 ± 4.75 |
| **dedx f = 0.5** | **fine (1.5%)** | **+5.35 ± 1.46** | **−16.94 ± 2.84** |

**With perfect geometry the objective is displaced by at most ~1%.** Lifetime is +1.14 ± 0.57
(2σ) and long. diffusion −1.67 ± 1.41 (1.2σ). The coarse grid was inflating both.

> **This weakens §6e's decomposition.** §6e argued that long. diffusion's fitted −5.40 ± 2.89%
> "agrees with" a loss minimum at −5.26% and is therefore an *objective* displacement rather than
> an optimizer failure. The minimum is actually at −1.67 ± 1.41%, so that agreement was largely an
> artifact of the coarse grid. The fit and the minimum still overlap (difference 3.7 ± 3.2, ~1.2σ),
> so the reading is not refuted — but it is no longer the tight match §6e presented, and long.
> diffusion cannot be written off as "sitting where the objective's minimum is".

> **Correction to §6f's own reasoning.** §6f argued the ladder would survive a finer grid because
> the grid systematic was *common-mode*, inferred from the 3-point vs 5-point vertex shift. That
> inference was wrong: measured directly, the fine grid moved `true` by ~1 point and `dedx f=0.5`
> by ~0. The conclusion survives — in fact the true → f=0.5 lifetime gap *grew* from 3.2 to 4.2
> points, so the dE/dx effect is larger than §6f reported — but it survives despite the argument,
> not because of it. The 3pt/5pt comparison is a usable warning flag, not a reliable correction.

### The guess file's damage: two suspects eliminated, one promoted

| condition | lifetime | long. diffusion | reading |
|---|---|---|---|
| true | +2.18 ± 0.80 | −5.27 ± 4.11 | reference (coarse grid, for comparability) |
| **re-segmented to ~1 cm** | **+2.11 ± 0.82** | −3.85 ± 5.15 | **identical to true** |
| dedx f = 0.40 (18/21 batches) | +5.66 ± 2.37 | −19.20 ± 4.99 | matches the guess file's real dE/dx spread |
| pos 880 µm (random) | −3.82 ± 6.34 | +41.3 ± 40.9 | random offsets: variance only |
| **guess** | **+15.22 ± 4.87** | −34.11 ± 19.7 | still unexplained |

**Segmentation alone does nothing.** `optimize/scripts/make_resegmented.py` splits every true
segment into collinear ~1 cm pieces (10.41M vs the guess file's 10.08M), conserving total dE and
dx exactly, leaving positions on the original track and dE/dx untouched — verified by the
length-weighted dE/dx spread being *identical* to truth (0.10718 both; the count-weighted sd falls
0.276 → 0.166 purely because long segments get replicated). Its minimum sits on top of `true`.
**Open question 11's leading hypothesis is dead.**

**The guess file's dE/dx is far smoother than we realised, but it only explains a third.**
Measuring the three files directly:

| file | N | dx mean | dE/dx sd | length-weighted dE/dx sd |
|---|---|---|---|---|
| true | 3.03M | 3.28 cm | 0.276 | **0.107** |
| guess | 10.08M | 0.99 cm | 0.043 | **0.043** |
| re-segmented | 10.41M | 0.95 cm | 0.166 | **0.107** |

The guess file retains only ~**40%** of the true length-weighted dE/dx structure. The matching
ladder rung (f = 0.40) gives lifetime **+5.66%** against the guess file's +15.22%. So dE/dx
smoothing accounts for roughly a third of the gap; segmentation for none; random position error
for none. **~9 points remain unexplained.**

**Promoted hypothesis: the error *structure*, not its magnitude.** The `pos` ladder displaces each
trajectory by a *random rigid* offset — uncorrelated between tracks, and preserving each track's
shape exactly. The guess file's error is nothing like that: it is a straight-line approximation, so
the error is *systematic and correlated along the track*, concentrated where the track curves. This
is the chord-cutting effect identified earlier in the campaign. Measured on the true file, the mean
transverse deviation from each trajectory's own chord is **1011 µm** (95th percentile 3551 µm) —
the same magnitude as the `pos880` rung, but structured rather than random. A new
`--mode chord` ladder (positions pulled toward each track's chord by fraction c, dE/dx held fixed
and dx/dE/n_electrons rescaled so the dE/dx axis is untouched) tests exactly this at c = 0.5 and
c = 1.0; both are running. If structure is the answer, c = 1.0 should reproduce a large part of the
missing 9 points at a position-error magnitude that costs nothing when applied randomly.

### The 4× data run failed, and the test was invalid

`sci_nb400` (400 batches, 1.6 km of track, 3 seeds, all complete) gives lifetime
**+51.6 ± 21.3%**, long. diffusion **+92.6 ± 35.4%**, position **3545 µm** against 162–255 µm for
the 100-batch runs, with lifetime still drifting 8.7 points at read-out. That is the *original* S4
failure mode, reproduced.

**It is not a statistics result — the configuration silently changed the LR schedule.**
`example_run.py:161` passes `epoch_size=len(tracks_dataloader_sim)` as the scheduler's
`transition_steps`, so the decay is indexed in **epochs, not iterations**. At 400 batches, 10 000
iterations is 25 epochs instead of 100, and the calibration LR ends at `0.91²⁵` = 9.5% of peak
instead of `0.91¹⁰⁰` = 0.008% — **1250× higher**. NB400 is therefore an approximately un-annealed
run, which is precisely the root cause §6 identifies for the whole S4 failure.

So the prediction that motivated it — *if position error acts through variance, 4× the data should
tighten long. diffusion ~2× without moving the centres* — is **untested, not falsified**. It is
being re-run as `NB400FIX` with `decay_rate = 0.91⁴ = 0.686`, which reproduces the same total
annealing over 25 epochs.

> **Trap — raising the batch count silently un-anneals the LR schedule.** Any comparison that
> changes `max_nbatch` while holding `iterations` and `decay_rate` fixed is confounded: it changes
> the number of epochs, and therefore the total annealing, by the same factor. To vary data volume
> alone, either scale `iterations` with the batch count or raise `decay_rate` to the
> (ratio)-th root. This is the fourth and worst member of the "requested batch count is not what
> you get" family in §7 — worst because it silently reverts the single most important fix in the
> campaign, and it does so while *looking* like a well-controlled scale-up.

---

## 6h. The dE/dx mean: the dominant bias channel, and the end of the guess-file mystery

§6f and §6g left one thing unexplained: the straight-line guess file displaces lifetime by
+15.2%, and *nothing we degraded reproduced it*. Segmentation was null, random position error was
null, and dE/dx spread accounted for only about a third. §6g promoted the error **structure** as
the suspect — systematic chord-cutting rather than random displacement. That has now been tested
and **refuted**, and the real answer turned out to be an axis the ladder never had.

### The chord axis is null: geometry error is not a bias channel, however it is shaped

`make_quality_ladder.py --mode chord` pulls each trajectory toward the straight line joining its
endpoints by a fraction c, holding dE/dx fixed and rescaling `dx`/`dE`/`n_electrons` so the dE/dx
axis cannot leak. On the true file the mean transverse deviation from each track's own chord is
**1011 µm** (95th pct 3551 µm) — deliberately matched in magnitude to the `pos880` rung, so the
*only* difference is that the error is correlated along the track instead of random.

| condition | lifetime | long. diffusion |
|---|---|---|
| true | +2.18 ± 0.80 | −5.27 ± 4.11 |
| chord c = 0.5 | −1.22 ± 2.57 | −1.44 ± 16.77 |
| chord c = 1.0 (fully straight) | **−0.99 ± 3.29** | −12.59 ± 23.71 |
| pos 880 µm (random) | −3.82 ± 6.34 | +41.28 ± 40.85 |
| chord 1.0 **+** dE/dx f = 0.40 | +4.73 ± 3.97 | −22.48 ± 21.70 |
| dE/dx f = 0.40 alone | +5.91 ± 2.14 | −18.63 ± 4.63 |

Fully straightened tracks give lifetime −0.99 ± 3.29%: indistinguishable from zero and from the
random rung. **Error structure matters no more than error magnitude did** — geometry simply is not
a bias channel. The combined file settles the last alternative: `chord1.0 + f=0.40` lands at
+4.73 ± 3.97, statistically identical to the dE/dx rung alone, so there is **no interaction** to
find either. This strengthens §6f's conclusion rather than weakening it: the position axis has now
survived an adversarial test it was designed to fail.

### The missing axis: a dE/dx MEAN error

Every dE/dx rung so far shrank the *spread* (`dEdx' = mean + f·(dEdx − mean)`) and left the mean
untouched. The orthogonal defect — the whole distribution scaled, shape preserved — had never been
measured. It is the error an over-stiff dE/dx mean constraint imposes: at `w = 1e5` the fitted mean
is **prior-determined rather than data-determined**, so any error in the pinned target becomes a
common multiplicative offset on every segment.

| dE/dx mean error | lifetime | long. diffusion | tran. diffusion | A_b |
|---|---|---|---|---|
| **−2%** | **+50.75 ± 2.15** | −39.57 ± 4.75 | −14.38 ± 1.13 | +1.87 ± 0.03 |
| 0 (true) | +2.18 ± 0.80 | −5.27 ± 4.11 | +0.81 ± 1.55 | +0.07 ± 0.03 |
| **+2%** | **−22.30 ± 0.30** | +48.82 ± 4.70 | +18.38 ± 1.32 | −1.66 ± 0.03 |

![The dE/dx mean is the dominant bias channel](plots/noise_report/fig31_dedx_mean_leverage.png)

**A 2% error in the dE/dx mean costs 22–51% on lifetime** — a local slope of **−24.3 percentage
points of lifetime per 1% of mean**. For comparison, destroying 60% of the dE/dx *spread* costs
+5.9 points and no geometry error of any kind costs anything measurable. The error bars are also
tiny (±0.30, ±2.15) against the geometry axis's ±6 to ±41, so this is a sharply determined
displacement, not a smeared one. The sign is physically sensible: less deposited charge means less
attenuation is needed to match the observed charge, so the fitted lifetime rises.

The response is strongly **non-linear and asymmetric** (−2% → +50.75%, +2% → −22.30%), so the
quoted slope is local to the negative arm — which is the side that matters, see below.

### This closes open question 11

Measuring the two files directly (length-weighted, so segmentation cannot confound it):

| file | length-weighted dE/dx mean |
|---|---|
| true | 1.88533 |
| guess | 1.87704 |
| | **−0.440%** |

At the measured slope, a −0.440% mean deficit predicts **+12.9%** on lifetime. The guess file's
observed displacement is **+15.2 ± 4.9%**. The ~9 points that segmentation, spread, random
displacement and chord-cutting all failed to explain were a **sub-percent error in the dE/dx
mean**. Open question 11 is answered.

![The completed quality ladder](plots/noise_report/fig32_ladder_complete.png)

### Why this is the most important result in the section

The mean constraint is a knob **we** set. §6d showed the dE/dx block earns its keep; the parallel
mean-constraint work found the production weight `w = 1e5` is ~6× too stiff, making the mean
prior-determined. Combined with the leverage measured here, **a sub-percent error in that pinned
mean is worth tens of percent on lifetime** — which makes the constraint weight the single
highest-leverage parameter in the whole configuration, far above anything on the geometry side.

## 6i. Fit-side arms: batching order and data volume

![Fit-side arms](plots/noise_report/fig33_fit_arms.png)

| arm | A_b | E field | tran. diff. | long. diff. | lifetime | position |
|---|---|---|---|---|---|---|
| ANNEALLONG (ref) | +0.75 ± 0.13 | −0.03 ± 0.01 | −0.96 ± 0.50 | −5.40 ± 2.36 | −2.70 ± 1.89 | 168 µm |
| SHUFOFF2 (sequential) | +0.71 ± 0.13 | +0.02 ± 0.03 | +1.28 ± 0.93 | +16.97 ± 23.17 | +0.45 ± 0.96 | 492 µm |
| SHUFON2 (`--shuffle_bt`) | +0.83 ± 0.08 | +0.00 ± 0.03 | +0.71 ± 0.94 | +8.63 ± 17.21 | −4.83 ± 3.81 | 475 µm |
| NB400FIX (4× data) | +0.44 ± 0.05 | +0.03 ± 0.01 | +3.18 ± 2.74 | +4.93 ± 13.65 | +3.82 ± 2.17 | 774 µm |

**Open question 5 is answered: batching order does not matter.** SHUFOFF2 and SHUFON2 were
submitted together on one tree and differ *only* in `--shuffle_bt`. Every parameter agrees within
errors; the largest gap (lifetime) is 1.4σ, and position is identical (492 vs 475 µm).

**More data buys calibration precision and starves geometry.** NB400FIX (400 batches, decay
corrected to 0.686 per §6g) tightens long. diffusion's seed spread **23.17 → 13.65** (1.7×,
against a predicted ~2×) with the centre stable — the variance reduction §6f's bias/variance split
predicts, since position error acts through variance. But position degrades **492 → 774 µm**,
consistent with the geometry block receiving 4× fewer updates per track at fixed iteration count.
The two effects are separable and both are understood.

> **Comparability warning.** Only the last three rows share `chain_decay_rate = 0.999`.
> ANNEALLONG used **0.9997**, which leaves the chain learning rate **1100× higher** at iteration
> 10 000 (0.0498 vs 4.52e-05). Its geometry is therefore *not* comparable to the other three, and
> the 168 µm vs ~490 µm gap is that flag, not a regression. The shuffle comparison and the
> NB400FIX comparison are each internally controlled and unaffected.

## 6j. A false alarm, and what it cost

An earlier reading of an archived `--shuffle_bt` run appeared to show that shuffling **destroyed**
the geometry fit: position 4561 µm vs 168 µm, final loss 18 433 vs 7 648, and fitted spline
coefficients 37× larger (2.16 cm vs 0.058 cm). That finding was **wrong**, and the way it was wrong
is worth recording.

Paired runs on one tree, submitted together, show no effect whatsoever:

| | epoch-0 median | vs recomputed nominal | vs recomputed final |
|---|---|---|---|
| shuffle OFF | 875 µm | 25.7 µm | **0.0 µm** |
| shuffle ON | 862 µm | 30.2 µm | **0.0 µm** |

**Root cause: uncontrolled code drift.** Both archived runs recorded `git_sha=1aaf114f` **and
`git_dirty=True`**, three days apart. A recorded SHA does not pin the code when the tree is dirty.
Current code reproduces the sequential run's epoch-0 value (875 vs 881 µm) but not the shuffled
one's (862 vs 3041 µm) — the intervening working tree was broken in a way neither neighbour was.

What the investigation did establish, all of it worth keeping:

- **The `pos_residual` metric is correct**, verified rather than assumed: recomputing it offline
  from each checkpoint's own `chain_contexts` / `true_positions` / `chain_cache` reproduces the
  fitter's number **exactly (0.0 µm)** in both orderings. This requires the real
  `_transverse_frame` — the default drift frame is `e1 = ẑ × u₀`, `e2 = u₀ × e1`; guessing
  `u₀ × ẑ` is a sign flip that silently inflates recomputed residuals ~7×.
- **Target, fitted dE/dx and fitted positions are correctly associated under shuffling.** Every
  per-batch lookup is keyed by the batch index (`batch{i}_target.npz` with `rngkey=i+1`;
  `_dedx_cache[i]`, `_batch_parent_ids[i]`; `_chain_cache[i]`, `_batch_chain_contexts[i]`,
  `_batch_true_positions[i]`), and the loop counter `i_bt` touches only `sz_mini_bt` windowing.
  Confirmed empirically by the new `pos_residual_batch` label.
- **A real bug, now fixed.** `_chain_active` gated the geometry update on
  `total_iter % chain_update_freq == 0` — a *global* iteration counter. Under sequential batching
  batch *b* is always visited at `total_iter ≡ b (mod nbatch)`, so the gate is a fixed function of
  the batch index. At `nbatch = 100`: `freq=2` → **50 of 100 batches never update**; `freq=4` →
  75; `freq=5` → 80; `freq=10` → 90. Shuffling accidentally repairs it, which is what made the two
  orderings incomparable in principle. Replaced with `_freq_gate()`, a per-batch visit counter;
  verified identical at `freq=1` (the default, so no existing run changes) and order-independent
  at `freq=3`. `_pos_residual_freq` had the same defect.
- **Two latent shape-keyed caches** (`_chain_geom_vg_cache`, and the sibling at ~2976) key on
  padded shapes rather than batch index while closing over one batch's `meta`. Only reachable via
  `--geom_optimizer gn/ggn/lbfgs`, not the default Adam path, but they will mis-associate track
  structure between batches when they are.

> **A second, self-inflicted error in the same area.** The `SHUFOFF2` arm was described as a
> reproducibility check against ANNEALLONG, and its disagreement was attributed to code drift.
> Diffing the two runs' *stored argv* (50 flags each) showed exactly one substantive difference:
> `chain_decay_rate` 0.9997 vs 0.999 — carried over by mistake from the NB400 configuration. The
> code-drift claim was withdrawn. **Diff the stored argv against the reference before submitting a
> comparison, not after it disagrees**, and exhaust configuration differences before reaching for
> anything more exotic. Having been genuinely burned by provenance once, the reflex to blame it
> again produced a confidently-stated wrong conclusion the second time.

---

## 6k. The standard method, and a major correction to §6h

§6h concluded that the dE/dx **mean** is "the dominant bias channel" and that the mean-constraint
weight is therefore "the highest-leverage parameter in the whole configuration". **That conclusion
was drawn from an estimator that cannot support it, and it is withdrawn in the form stated.**

The objection is a physics one: electron lifetime is a *pure charge-versus-drift-time* effect. It
is the **slope** of ln(dQ/dx) against drift time, so a global dE/dx scale error moves the
**intercept** and must leave the slope — and hence the lifetime — untouched. A −24.3 points-per-1%
sensitivity is incompatible with that, unless the two numbers refer to different estimators.

They do.

![Standard method vs the 1-D slice](plots/noise_report/fig34_lifetime_standard_method.png)

### Our lever arm is 0.087 lifetimes

Module0 drifts along x with the cathode at x = 0 and anodes at |x| = 30.27 cm, at
0.1587 cm/µs — so the **entire detector is 190.8 µs deep against a 2200 µs lifetime**. The charge
swing from anode to cathode is only **8.3%**. Measured on the real sample, 2.95 M segments span
0–189.7 µs with a mean of 87.6 µs and a charge-weighted ⟨t/τ⟩ of **0.0391**.

That single number is the whole story. A likelihood slice that must absorb a global charge error
using lifetime *alone* has to move ⟨exp(−t/τ)⟩ by the full size of that error, and since lifetime
only controls 8.3% of the charge across the whole detector, it takes an enormous lifetime change:

```
leverage = 1 / <t/tau> = 25.6x      (measured on the S6h scan: 24.3x)
```

The agreement is essentially exact. **§6h's −24.3 points-per-1% is not a property of the dE/dx
mean; it is the reciprocal of our drift lever arm**, exposed by a 1-D slice that freezes every
other parameter and so leaves lifetime as the only knob that can change total charge.

### The standard method is immune, exactly as the physics requires

Binning dQ/dx by drift time and fitting ln(dQ/dx) = ln A − t/τ on the same 2.95 M segments:

| charge scale applied | fitted τ |
|---|---|
| nominal | **2173.3 ± 3.6 µs** (−1.21% vs the 2200 µs truth) |
| dE/dx **+2%** | **2173.3 ± 3.6 µs** |
| dE/dx **−2%** | **2173.3 ± 3.6 µs** |

Identical to four significant figures. Fig 34(b) shows why: three parallel lines, the intercept
moving and the slope untouched. The residual −1.21% is a binning/median-estimator effect, not a
scale effect. So a shape-based estimator recovers the lifetime to ~1% and **does not care about
the dE/dx scale at all** — precisely as expected.

### What our fits actually do

The decisive test is our own fitted dE/dx mean, run through §6h's leverage and compared with the
lifetime those same fits report:

| run | seed | fitted mean | true mean | mean error | §6h prediction | **observed lifetime** |
|---|---|---|---|---|---|---|
| ANNEALLONG | 0 | 1.87848 | 1.88029 | −0.096% | +2.33% | **−4.27%** |
| ANNEALLONG | 1 | 1.87834 | 1.88029 | −0.104% | +2.52% | **−3.79%** |
| ANNEALLONG | 2 | 1.87832 | 1.88029 | −0.104% | +2.53% | **−0.05%** |
| SHUFOFF2 | 0 | 1.87753 | 1.88029 | −0.147% | +3.57% | **+1.22%** |
| SHUFOFF2 | 1 | 1.87808 | 1.88029 | −0.117% | +2.85% | **+1.03%** |
| SHUFOFF2 | 2 | 1.87812 | 1.88029 | −0.115% | +2.80% | **−0.90%** |
| NB400FIX | 0 | 1.87765 | 1.87725 | +0.021% | −0.52% | **+5.72%** |
| NB400FIX | 1 | 1.87747 | 1.87725 | +0.012% | −0.29% | **+4.96%** |
| NB400FIX | 2 | 1.87788 | 1.87725 | +0.034% | −0.81% | **+0.78%** |

Two things follow, and both matter.

**Our fitted dE/dx mean is accurate to ~0.1%** — a factor 4 better than the guess file's −0.44%
and far better than we assumed. On its own this is good news: the mean constraint is not leaving a
large mean error behind.

**The §6h leverage does not predict our fits.** The prediction misses in magnitude everywhere and
in *sign* for five of nine seeds. The fit does not behave like a 1-D slice, because in the fit A_b
is free and absorbs the normalisation — and indeed §6h's own A_b slice moves by ∓1.87% for a ±2%
mean shift, i.e. **nearly 1:1**, making A_b an almost perfect proxy for a global charge scale.
Whatever drives the residual lifetime error in our fits, **it is not the dE/dx mean.**

### What survives from §6h

- The **ranking among degradations measured the same way** stands: on 1-D slices, dE/dx defects
  displace the minimum and geometry defects do not. The chord-null and no-interaction results are
  unaffected.
- The **guess-file explanation stands**, because both sides of it were measured with the same
  estimator: its −0.440% mean deficit predicts +12.9% on the slice and +15.2 ± 4.9% is observed.
  Open question 11 remains answered — for the slice.
- What does **not** survive is the extrapolation from the slice to the fit: "the dE/dx mean is the
  dominant bias channel *of our calibration*" and "the mean-constraint weight is the
  highest-leverage parameter" were both unsupported. §6e already warned that these are 1-D slices
  and not profile likelihoods, and that a displaced minimum "must be re-checked with the others
  profiled before being called a bias". That warning applied to §6h and was not heeded.

### The real lesson, and what it demands next

The lever arm is the binding constraint on this measurement: **8.3% of charge swing is all the
signal there is**, so lifetime is intrinsically the weakest of the five parameters, and any
estimator that leans on normalisation rather than shape will be ~25× levered on charge-scale
errors. Two consequences:

1. **Every scan in §6e–§6h should be redone as a profile**, minimising over the other four
   parameters at each grid point, before any of their displacements is called a bias. This is the
   single most important methodological fix outstanding.
2. **A shape-based lifetime estimator is worth having**, either as a cross-check or as an explicit
   term. It recovers τ to ~1% here and is structurally immune to the charge-scale degeneracy that
   dominates the slice.

---

## 6l. The same standard fit on real hits: the front end destroys it

§6k measured the lifetime on **truth segments** and got τ = 2173 ± 4 µs (−1.2%), immune to the
dE/dx scale to four significant figures. But that bypasses the entire front end. The question that
matters is what survives once the measurement is made on the hits the detector actually produces —
ADC counts and tick numbers, with the discrimination threshold, the ADC response and readout noise
all in play.

Hit lists were dumped straight from the simulation (`optimize/scripts/start_hitdump.sh`, using the
fitter's own cached target generation with `LARND_KEEP_TARGETS=1`), 60 batches, ~102 k hits.
Charge from `adc2charge`, drift time from `tick × 0.1 µs`. Two validations first: hit ticks span
1.8–188.0 µs against the 190.8 µs geometric drift window, so the tick→time conversion carries no
t₀ contamination; and ADC = 81 maps to 5.05 ke, reproducing the 5000 e⁻ threshold exactly.

![Standard fit on simulated hits](plots/noise_report/fig35_lifetime_hits.png)

### The answer is not stable

| fit range | τ (µs) | bias vs 2200 µs |
|---|---|---|
| all hits (t ≥ 0) | 1882 ± 88 | **−14.5%** |
| t ≥ 5 µs | 4917 ± 625 | **+123.5%** |
| t ≥ 10 µs | 3872 ± 421 | +76.0% |
| t ≥ 20 µs | 3141 ± 315 | +42.8% |
| t ≥ 40 µs | 2864 ± 330 | +30.2% |
| t ≥ 50 µs | 2763 ± 350 | +25.6% |
| **truth segments (§6k)** | **2173 ± 4** | **−1.2%** |

**The same estimator that was exact on truth spans −15% to +124% on hits, depending only on where
the fit is started.** Nothing else changed. The quoted statistical errors (±88 to ±625 µs) are
themselves 20–150× larger than the truth-level ±4 µs, so even the *precision* collapses.

### Two front-end effects, pulling opposite ways

Fig 35(b) shows the mean hit charge versus drift time is **not a single exponential**, which is why
no single fit range is right:

- **An anode-edge spike (t ≲ 20 µs).** The first bin sits at 14.15 ke against ~12.7 ke just beyond
  it. Near the anode there has been almost no diffusion, so a segment's charge is concentrated into
  fewer, larger hits — measured, hits-per-pixel rises 1.845 → ~2.2 across that boundary. This is a
  charge-*sharing* effect, not an attenuation effect, and it manufactures a steep artificial drop
  in the first 20 µs. Including it drags the fit to −14.5%; excluding it flips the answer positive.
- **Threshold selection (all t).** Fig 35(a) shows the hit-charge spectrum truncated hard at 5 ke
  while peaking at ~8 ke and with a median of 10.7 ke — **the threshold sits only a factor 2 below
  the bulk**, not far out in a tail. Hits that drift furthest lose the most charge and are
  preferentially cut, so the survivors at large t are biased high, the slope flattens, and the
  lifetime is **over**estimated. Beyond the edge region the measured points sit visibly above the
  truth slope and much flatter, giving the ~+30% plateau.

The full-range fit is not a compromise between the two — it is dominated by the edge artefact,
which is why it lands on the opposite side of truth from every other range.

![The two front-end mechanisms](plots/noise_report/fig36_hit_mechanisms.png)

Fig 36 shows both mechanisms directly on the same hits, plus a third complication:

- **(a)** the charge spectrum in three drift slices, truncated at the fixed 5 ke cut while the
  distribution itself slides down — so the survivors at long drift are a biased-high subset;
- **(b)** the sharing effect as a clean anti-correlation: hits per pixel climbs **1.70 → 2.20**
  over the first ~25 µs while mean Q per hit falls **14.1 → 12.5 ke** over exactly the same range,
  then both plateau. The charge is not disappearing, it is being divided;
- **(c)** the sample carries **3.6× fewer hits at the cathode than at the anode**. This is *not*
  threshold loss — an 8.3% total attenuation cannot remove two thirds of the hits — it is the
  sample's own drift-time exposure. It matters because it weights the fit very unevenly, and the
  sparsest region is the long-drift end where the lifetime signal actually lives;
- **(d)** three candidate observables against the true exponential. Per-hit Q sits above the truth
  slope and flatter; summing charge **per pixel** undoes the sharing and tracks truth far better.

Summing per pixel is a real improvement but not a fix:

| observable (fit from t ≥ 20 µs) | τ (µs) | bias |
|---|---|---|
| per-hit Q | 3141 ± 315 | +42.8% |
| per-hit Q, Q > 7 ke | 3417 ± 362 | +55.3% |
| **charge summed per pixel** | **1926 ± 198** | **−12.5%** |

Undoing the sharing cuts the bias from +43% to −12.5% and flips its sign, which confirms sharing
as the dominant term — but the residual is the threshold selection, which per-pixel summing does
nothing about. Note also that a naive charge floor (Q > 7 ke) makes things **worse**, not better:
cutting on the observable that is itself being selected deepens the selection bias.

### What this means

**The 8.3% charge swing (§6k) is the whole signal**, and both of these effects are comparable to
or larger than it. A 1.5 ke shift in mean hit charge between the first bin and the rest is ~12% of
the charge scale, against an 8.3% total attenuation — the systematics are bigger than the physics
being measured.

Three consequences:

1. **The truth-level agreement in §6k was not evidence that the method works.** It only showed the
   estimator is unbiased *given perfect charge measurement*. Every part of the difficulty lives in
   the front end, and §6k should not be read as a validation of the standard approach.
2. **A per-hit charge is the wrong observable.** Diffusion redistributes a segment's charge across
   pixels and ticks in a drift-dependent way, so per-hit Q measures attenuation *and* sharing
   together. The standard method needs dQ/dx integrated over the full cluster — which is precisely
   what a forward-model fit does natively, and a hand-rolled estimator does not.
3. **This is an argument for the forward-model fit, not against it.** The likelihood fit sees the
   threshold and the diffusion because they are in the model; a binned exponential on hit charges
   does not, and inherits both as uncontrolled systematics an order of magnitude larger than the
   effect. The lifetime is hard here because the lever arm is short (§6k) *and* because the
   observable is contaminated — not because the fitter is doing something wrong.

### Separating the threshold from the noise

The noise-OFF control has since run (103 k hits against 102 k, same 60 batches), which splits the
bias into its two sources:

| fit range | noise OFF | noise ON | noise contributes |
|---|---|---|---|
| all hits | **−14.3%** | **−14.5%** | **+0.2** |
| t ≥ 10 µs | +74.7% | +76.0% | +1.3 |
| t ≥ 20 µs | +29.2% | +42.8% | +13.6 |
| t ≥ 30 µs | +22.9% | +37.1% | +14.2 |
| t ≥ 40 µs | +15.4% | +30.2% | +14.8 |
| t ≥ 50 µs | +14.7% | +25.6% | +10.9 |

Three things follow, and the prediction made above is only half right.

**The anode-edge artefact is purely geometric, as predicted.** With and without noise the
full-range fit gives −14.3% and −14.5% — identical. It is charge sharing, and noise has nothing to
do with it.

**Most of the plateau bias is the threshold, not the noise.** Noise-OFF already carries **+15 to
+29%**, so threshold selection and sharing alone account for the bulk of it. Turning noise on adds
a further **+11 to +15 percentage points**, roughly constant across the plateau.

**Noise raises the median hit charge by 7.9%** (9.92 → 10.70 ke) while slightly *reducing* the hit
count (103 068 → 101 853). That is the Eddington/Malmquist signature stated up front: near a
threshold, upward fluctuations are promoted into the sample and downward ones are lost, so the
survivors are biased high — and because the cut sits only ~2× below the median, the effect is
large. Fig 35(a) shows it directly: the noise-OFF spectrum has a sharp tall peak at ~9.5 ke, the
noise-ON one is broader and displaced upward.

So the ordering of causes is: **charge sharing (largest, geometric) > threshold selection > noise
(~+13 points)** — with noise the smallest of the three, not the dominant term the section title
might suggest.

---

## 6m. Is the dE/dx↔lifetime degeneracy actually being exercised?

Every segment sits at a definite drift coordinate, so a drift-correlated pattern in the fitted
per-segment dE/dx is observationally almost identical to a lifetime change. With ~4000 free
nuisances and nothing in the base likelihood forbidding it, this is the most plausible remaining
structural explanation for lifetime's weakness — and it has been demonstrated before: at 400 cm
with true positions, lifetime was **−1.5 ± 1.9% with TRUE dE/dx and +18.5 ± 18.9% with FITTED
dE/dx**, restored to −1.69 ± 0.68% by `--dedx_drift_profile_weight = 1e6`. That penalty is **0.0**
in the production config.

This is diagnosable on archived runs, because the drift coordinate can be reconstructed from
`chain_contexts` (nominal track path) plus `batch_parent_ids` (sub-step → parent segment, giving
each parent's fractional span), with values from `dedx_cache`. No global event ids are needed, so
unlike the truth-matched MAE it works offline. The statistic is the one the penalty minimises,
`cov_w(|z|, log dEdx)/√var_w(|z|)`.

![Fitted dE/dx profile versus depth](plots/noise_report/fig37_dedx_drift_profile.png)

| source | drift trend of log dE/dx |
|---|---|
| **TRUE file** (physics reference — should be flat) | **+0.0002** |
| **GUESS file** (what every S4 run starts from) | **+0.0035** |
| ANNEALLONG fitted (3 seeds) | +0.0003 / −0.0001 / +0.0006 |
| ANNEALMORE fitted (3 seeds) | +0.0001 / +0.0003 / +0.0003 |
| SHUFOFF2 / SHUFON2 fitted (6 seeds) | −0.0003 … +0.0003 |
| NB400FIX fitted (3 seeds) | +0.0000 / −0.0001 / +0.0002 |
| DRIFTW6b / DRIFTW7b fitted (6 seeds) | −0.0006 … +0.0001 |

**The degeneracy is not being exercised.** All 18 seed-points land on the true value, none near
their own input. Fig 37(a) shows it directly: the straight-line guess carries a ~1.1% swing in
log dE/dx across the drift range, the fit removes it, and the result tracks the true file. The
nuisance block is **correcting a real defect in its input**, not manufacturing a fake attenuation.

**This does not exonerate the mechanism.** The correction is worth **~−10% of lifetime** in exactly
the degenerate direction (Δtrend ≈ −0.0031, slope −0.00032/cm, via
`1/τ_fit = 1/τ_true − v_drift·m`). So dE/dx and lifetime *are* trading at a magnitude comparable to
the residual lifetime bias; here the trade happens to run toward the right answer.

**It also argues against switching the penalty on here.** `dpw` drives the trend toward zero
*value*, and these runs are already there — so it would have nothing to correct while fighting the
fit's legitimate removal of the input's +0.0032. That plausibly explains why the dpw=1e6 success
came from the 400 cm *ceiling* configuration, whose input is the true file and hence already flat.

### Why the straight-line guess has a drift-dependent dE/dx at all

Decomposing the guess file's trend into a component **within** each track and a component
**between** tracks:

| component | slope of log dE/dx per cm |
|---|---|
| **within** a track (z − ⟨z⟩ of that track) | **+0.000000** |
| **between** tracks (⟨z⟩ of each track) | **+0.000972** |
| *(full attenuation for τ = 2200 µs, for scale)* | *0.002864* |

**It is entirely a between-track effect — exactly zero within a track.** That rules out the obvious
first guess, residual uncorrected attenuation: uncorrected attenuation acts on each segment
according to its own drift distance and would show up *within* tracks.

What it looks like instead is a **per-track normalisation**: the guess file's dE/dx is nearly
constant along each track (its length-weighted spread is 0.043 against truth's 0.107), and tracks
sitting at greater mean drift depth are assigned a systematically lower value. That is what you get
if each track's dE/dx is derived from its own total collected charge without a lifetime correction
— the whole track is scaled down together, uniformly. This is consistent with, but not proof of,
how that file was produced; we have not inspected its generation.

> **Correction — the reference used above is the wrong one.** The `+0.0002` figure is the trend of
> the **whole** true file. Production fits use **0.4% of it**, and that subset carries its own drift
> correlation. Measured on the exact fitted segments (via the in-fitter matched truth, which exists
> only for the `MAECHECK` run):
>
> | | trend | slope /cm |
> |---|---|---|
> | TRUE, on the fitted subset | **+0.0012** | +0.000134 |
> | FITTED | **+0.0011** | +0.000119 |
> | INPUT (guess), same subset | +0.0045 | +0.000498 |
> | *TRUE, whole file* | *+0.0002* | *+0.000022* |
>
> The subset's true trend is **6× the whole-file value**, so the whole file is not a valid
> reference. The conclusion nonetheless **strengthens**: fitted (+0.0011) matches subset-truth
> (+0.0012) essentially exactly, rather than approximately. But this direct verification exists
> only for `MAECHECK`; for ANNEALLONG and the other archived runs the subset truth is unmeasurable
> (their checkpoints predate the matcher), so their agreement with `+0.0002` is against a proxy of
> unknown quality and should be read as suggestive rather than established.

> **What cannot be read, and why.** The `ceiling_400_dpw1e6/1e7` arms — the ones where the
> degeneracy was demonstrated — freeze positions, so no `chain_contexts` is stored and there is no
> way to recover a drift coordinate. CONSTDEDX freezes dE/dx, so there is no `dedx_cache`. Neither
> is a bug; both are structural. The single configuration known to exercise the degeneracy is
> therefore the one that cannot be retro-diagnosed, and settling it needs a re-run.

**Caveat.** The slope→lifetime conversion is a first-order argument and has not been independently
validated, so −10% is order-of-magnitude. A secondary axis mapping trend to lifetime bias was
deliberately left off fig 37(b): the mapping is defined only for a *change* in trend, not an
absolute value.

Tools: `analyze_dedx_drift_profile.py`, `make_drift_profile_fig.py`.

---

## 6n. Four arrays: the minimum-length cut, the power floor, and the dE/dx metric resolved

Four 3-seed arrays completed together on one tree, which makes them mutually comparable.

![Four arrays](plots/noise_report/fig38_aug5_arrays.png)

| arm | A_b | E field | tran. diff. | long. diff. | lifetime | position |
|---|---|---|---|---|---|---|
| ANNEALLONG (original) | +0.75 ± 0.13 | −0.03 ± 0.01 | −0.96 ± 0.50 | −5.40 ± 2.36 | −2.70 ± 1.89 | 168 µm |
| **annl2** (rerun, current tree) | +0.82 ± 0.14 | −0.02 ± 0.02 | −1.19 ± 0.67 | −3.21 ± 3.70 | −5.07 ± 2.44 | 162 µm |
| **mdx_full** (= annl2, cut is a no-op) | +0.71 ± 0.18 | −0.02 ± 0.02 | −0.79 ± 1.04 | −4.78 ± 3.72 | −1.44 ± 1.63 | 163 µm |
| **nb200** (2× data) | +0.77 ± 0.06 | −0.02 ± 0.02 | **+0.26 ± 0.17** | −2.06 ± 4.19 | −3.46 ± 1.59 | 172 µm |
| **ceil_base** (ceiling, w = 1e5) | +0.77 ± 0.01 | −0.00 ± 0.00 | −0.31 ± 0.67 | −9.95 ± 1.02 | −4.31 ± 1.94 | frozen |
| **ceil_mdx** (ceiling, min_dx = 0.15) | +0.77 ± 0.03 | −0.00 ± 0.00 | −0.36 ± 0.69 | −10.17 ± 0.79 | −4.19 ± 1.91 | frozen |

### The minimum-length cut is a null

`LARND_DEDX_MIN_DX = 0.15` was the top-ranked recommendation carried over from the second-order
report (measured there as MAE −62%, worst-segment error −84% → −11%, calibration −0.97 ± 0.11).
Two independent findings retire it:

1. **It is a no-op in production.** The straight-line guess file is uniformly re-segmented at
   ~1 cm — its shortest segment is **0.909 cm**, six times the cut — so nothing is ever frozen.
   `mdx_full` logged zero `[DEDX-MINDX]` lines and is configuration-identical to `annl2`. The
   diagnostic example that motivated the cut, a 0.0355 cm segment, cannot exist in that file.
2. **Where it does fire, it changes nothing.** In ceiling mode (true-geometry input, 4.05% of
   segments below the cut) every parameter matches between `ceil_base` and `ceil_mdx`. This is a
   paired comparison with the geometry variance removed, so the errors are tight (±0.01 on A_b) —
   a well-powered negative rather than an inconclusive one. **The claimed calibration gain does
   not reproduce.**

### n = 3 cannot resolve ~4 points — measured, not argued

ANNEALLONG, `annl2` and `mdx_full` are the **same configuration**. Their 3-seed lifetime means:

```
ANNEALLONG   -2.70 ± 1.89
annl2        -5.07 ± 2.44
mdx_full     -1.44 ± 1.63     ->  3.6 points of spread
```

That is the resolution floor of any n = 3 comparison on this configuration, and it is **larger than
most effects the campaign has chased** — including the 3.4-point difference on which the
mean-constraint weight question previously died. Every remaining A/B must clear it first.

It also **closes the reproducibility question**: `annl2` matches ANNEALLONG on all five parameters
and on position (162 vs 168 µm), so the ~1240 lines added to `fit_params.py` since have not
disturbed the production path. The code-drift claim made earlier in §6j is now positively refuted
rather than merely withdrawn.

### Where the variance lives, and how to reduce it

Freezing positions at truth shrinks the seed s.d. by **11× on A_b** (0.14 → 0.013) and **70× on
E field**. That is the same conclusion as §6f's bias/variance split, now visible directly in the
fit arms, and it is why the ceiling configuration is the right venue for any remaining comparison.

**2× data with compute scaled alongside works.** `nb200` (200 batches × 20 000 iterations, so
epochs and per-batch visits are both preserved) tightens every error bar — A_b s.d. 0.14 → 0.06,
transverse diffusion 0.67 → 0.17 — **and leaves geometry intact at 172 µm against 162 µm**. Compare
`NB400FIX`, which raised batches at fixed iteration count and degraded position to 774 µm. The
distinction is not the data volume but whether the geometry block still gets its updates.

*(Caveat: `nb200` seed 2 hit the wall clock at ~19 800 of 20 000 iterations. The robust estimator
reads the median over the final 20%, so it is unaffected.)*

### The dE/dx MAE puzzle, resolved at production length

| arm | vs SIM input | vs TRUTH (arc-length matched) |
|---|---|---|
| annl2 | 0.0316 → 0.0349 (**+10.4%**) | 0.0714 → 0.0590 (**−17.4%**) |
| mdx_full | 0.0316 → 0.0348 (+10.2%) | 0.0714 → 0.0589 (−17.6%) |
| ceil_base | 0.1294 → 0.1067 (−17.6%) | 0.1294 → 0.1067 (−17.6%) |
| ceil_mdx | 0.1293 → 0.1077 (−16.7%) | 0.1293 → 0.1077 (−16.7%) |

**The block improves 17% against truth while appearing to degrade 10% against its own starting
point.** The long-standing open question was the metric's reference, not the fit — `dedx_mae_iter`
is computed against `--input_file_sim`, which in every S4 run is the guess file.

The ceiling rows provide an unplanned **validation of the matcher**: there the sim input *is* the
target, so the two references must coincide — and they do to four decimal places.

---

## 6o. What the dE/dx block can recover with perfect geometry — and a configuration error

§6n resolved *what* the dE/dx MAE was measuring. This asks the harder question: with geometry
removed as a variable entirely, how much of the per-segment dE/dx can the block recover at all?

Two properties make this cleanly answerable on every historical run:

- **In ceiling mode the sim input IS the target**, so `dedx_mae_iter` — which is computed against
  `--input_file_sim` and is therefore not a truth error in any S3/S4 run — happens to coincide
  with the truth error. No matcher needed.
- **All runs start from the same place.** `_get_or_init_dedx_state` initialises every segment at
  the prior centre `dedx_student_loc`, and states explicitly that "no true dE/dx information is
  used". So the initial MAE (~0.129 everywhere) is the prior→truth distance, and *fraction of the
  gap closed* is well defined.

![dE/dx recovery with true geometry](plots/noise_report/fig39_ceiling_dedx_recovery.png)

| ceiling run (true geometry + fitted dE/dx + fitted calibration) | n | prior w | noise | gap closed |
|---|---|---|---|---|
| `sci_ceiling_400_dpw1e6` | 2 | 0.5 | OFF | 64.4% |
| `sci_ceiling_400` | 5 | 0.5 | OFF | 62.7% |
| `sci_ceiling_repro` | 3 | 0.5 | OFF | 60.6% |
| `sci_ceiling_gnref` | 3 | 0.5 | OFF | 59.2% |
| `sci_ceiling` | 5 | 0.5 | OFF | 53.3% |
| `sci_ceiling_noise_s2_anneal` | 1 | 0.5 | **ON** | 49.3% |
| `sci_ceiling_noise_s2_thr2500` | 2 | 0.5 | **ON** | 49.0% |
| `sci_ceiling_noise_s2dpw` | 3 | 0.5 | **ON** | 44.3% |
| `sci_ceiling_noiseon` | 1 | 0.5 | **ON** | 44.0% |
| `sci_ceiling_S2ANNEALX` | 2 | 0.5 | **ON** | 43.4% |
| `sci_ceiling_noise_s2` | 6 | 0.5 | **ON** | 39.9% |
| **`sci_ceiling_CEILW`** | 6 | **5.0** | **ON** | **18.6%** |
| **`sci_ceiling_CEILBASE`** | 3 | **5.0** | **ON** | **17.6%** |
| **`sci_ceiling_CEILMDX`** | 3 | **5.0** | **ON** | **16.7%** |

All configurations above are **read from the checkpoints**: every history pickle carries the full
argparse `Namespace` under a `config` key, old runs included.

### The block is objective-limited, not geometry-limited

**Even with perfect geometry and noise ON the best run recovers only 49% of the prior→truth gap**
(64% with noise off). At least half the per-segment dE/dx structure is not recoverable no matter
how good the geometry is.
That is the same verdict the second-order study reached from curvature — the block is
objective-limited — now measured directly on the fits rather than inferred.

It also reframes §6n's production result: the +17% improvement there is not evidence of a weak
fit so much as of a block that is intrinsically limited, and (see below) additionally
over-constrained.

### Two effects, and a configuration error in every recent arm

The 17–64% spread has **two** causes and they must not be conflated:

- **Noise costs ~14 points.** Noiseless runs reach 53–64%, noise-ON runs 40–49%. Every historical
  run used `prior_w = 0.5`, so the prior explains none of this part.
- **The dE/dx prior weight costs ~25 more.** Against the matched comparison — noise ON,
  `prior_w = 0.5` — the historical band is **40–49%** and the recent arms reach **17.6%**. Those
  arms were launched with **`SCIDEDXPRIOR=5`**, copied from the ANNEALLONG *fitted-geometry*
  configuration against the script default of **0.5** (`start_sci_case.sh:38`). It is the only
  dE/dx knob that differs across those six historical runs and ours. Fig 39(b) shows the
  signature: the stiff-prior arms plateau at ~0.103 within ~1000 iterations while the matched
  noise-ON run keeps descending to ~0.066.

ANNEALLONG is a *fitted-geometry* run; carrying its prior weight into a true-geometry study was my
error, and it affects more than the MAE:

- the `ceil_base` vs `ceil_mdx` minimum-length-cut comparison remains internally valid (both arms
  share the prior) but was made with the block heavily constrained — not the regime in which the
  cut was originally claimed to help;
- the **`ceil_w3k` mean-constraint test is entangled**: it varies the *mean-constraint* weight
  while the *prior* weight sits 10× above default, so the two cannot be cleanly separated.

`ceil_p05` (3 seeds, ceiling, `SCIDEDXPRIOR=0.5`, otherwise identical to `ceil_base`) is running as
the clean single-variable test. The two remaining unmatched variables against the noise-ON
historical band are `iterations` (10 000 vs 5 000) and the calibration `decay_rate` (0.91 vs
0.999) — but more iterations should *help* recovery, so neither explains a drop.

> **Correction.** An earlier version of this section stated that the historical prior weight
> "cannot be read back" because those runs predate the `provenance` block, and built the
> attribution from script defaults plus a launch-log survey. That was wrong on both counts: the
> `config` Namespace has always been in every checkpoint, and the measured values show the
> historical spread is mostly a **noise** effect, not a prior effect. The prior attribution
> survives only against the matched noise-ON subset.

---

## 6p. Profiling over A_b, and three configuration results

### The 2-D scans: the guess file's lifetime bias was never real

Every scan in §6e–§6h is a 1-D slice with the other four parameters frozen. `LARND_SCAN_2D` now
scans (lifetime, A_b) jointly, so minimising over A_b at each lifetime gives the **profile
likelihood** — the quantity a displaced minimum has to be measured on. Five conditions, 21×21 grids
over 21 batches:

![2-D scans: slice vs profile](plots/noise_report/fig40_scan2d_profile.png)

| condition | 1-D SLICE | PROFILE | A_b at joint min |
|---|---|---|---|
| true geometry | +1.41% | +1.41% | +0.00% |
| dE/dx f = 0.40 (spread) | +4.96% | +4.96% | +0.00% |
| pos 880 µm (geometry) | −4.78% | −4.78% | +0.00% |
| dE/dx **mean** +2% | −15.00% | **+4.23%** | −1.88% |
| **straight-line guess** | **+13.87%** | **+0.43%** | +0.63% |

**The guess file's lifetime displacement collapses from +13.9% to +0.4% — a factor 32.** The single
number that §6f–§6h spent the most effort trying to decompose was almost entirely an artefact of
holding A_b fixed. Profiled, the guess file's objective is essentially unbiased in lifetime.

**And the pattern is clean.** Profiling changes the answer *only* where the defect is a
charge-**normalisation** error — the guess file (whose dE/dx mean is −0.44% low) and the explicit
mean shift. Where the defect is a dE/dx **spread** change or a **geometry** error, A_b sits exactly
at nominal and slice = profile to two decimal places. A_b absorbs normalisation and nothing else.

This closes the loop with the fit-side test, which found a 4% dE/dx mean swing moves lifetime by
0.8 ± 4.7 points against a 97-point slice prediction. Objective and optimiser now agree.

**What this means for the ladder.** The *rankings* survive — the dE/dx spread effect (+4.96%) and
the geometry null are unchanged by profiling, because neither moves A_b. What does not survive is
the guess file's absolute displacement, and with it open question 11: **there was no unexplained
+9 points, because there was no +15.2% to explain.**

![Three single-knob ceiling comparisons](plots/noise_report/fig41_ceiling_knobs.png)

All four arms below are ceiling (true geometry, frozen), 10 000 iterations, 100 batches of 400 cm,
and differ from their control in exactly **one** knob. Ceiling is the right venue: it removes the
geometry variance that dominates production and shrinks the seed s.d. 11–70×, which is what makes a
single-knob difference resolvable at all — three runs of an identical *production* config span 3.6
points on lifetime.

### The dE/dx prior weight: production is mis-set

`ceil_p05` confirms the §6o attribution, and the effect is much larger than the MAE alone suggested:

| ceiling arm | A_b | tran. diff. | long. diff. | lifetime | dE/dx gap closed |
|---|---|---|---|---|---|
| `ceil_base` prior **5** | +0.77 ± 0.01 | −0.31 ± 0.67 | **−9.95 ± 1.02** | −4.31 ± 1.94 | 17.6% |
| `ceil_p05` prior **0.5** | **+0.19 ± 0.02** | +0.50 ± 0.41 | **+1.43 ± 1.76** | **−2.32 ± 0.69** | **44.5%** |

Dropping the prior weight from 5 to the script default of 0.5 improves **everything**: dE/dx
recovery 2.5× (17.6% → 44.5%, landing squarely in the historical noise-ON band of 40–49%), A_b 4×,
lifetime both closer to truth and 3× tighter — and long. diffusion moves from **−9.95% to +1.43%**,
which is the largest single improvement in that parameter anywhere in the campaign.

**ANNEALLONG — the production configuration — uses `SCIDEDXPRIOR=5`.** `full_p05` (6 seeds,
production config at prior 0.5) is running to test whether the same gain transfers off the ceiling.

### The mean-constraint weight: 1e5 beats 3000

With the prior now correct and 6 seeds at ceiling, the second-order study's top recommendation
does **not** reproduce:

| ceiling arm (prior 0.5) | A_b | long. diff. | lifetime | dE/dx gap |
|---|---|---|---|---|
| `ceil_p05` mean w = **1e5** | **+0.19 ± 0.02** | +1.43 ± 1.76 | −2.32 ± 0.69 | 44.5% |
| `w3k_p05` mean w = **3000** | +0.71 ± 0.14 | +1.95 ± 1.67 | +0.06 ± **5.21** | 45.7% |

Lowering the weight makes A_b **worse** (0.19 → 0.71), leaves dE/dx recovery unchanged, and
inflates lifetime's seed scatter 7× (0.69 → 5.21). The toy's reported 4× calibration gain and
+108% lifetime-bias removal do not survive at production scale — this is now the properly powered,
un-entangled version of a test that had failed twice before. **Keep `w = 1e5`.**

### What readout noise costs

`ceil_nonoise` (verified from the stored config: `no_noise=True`, prior 0.5):

| | A_b | long. diff. | lifetime | dE/dx gap |
|---|---|---|---|---|
| noise **ON** (`ceil_p05`) | +0.19 ± 0.02 | **+1.43 ± 1.76** | **−2.32 ± 0.69** | 44.5% |
| noise **OFF** | +0.10 ± 0.07 | **−13.94 ± 5.22** | −5.05 ± 1.34 | **64.3%** |

Noise costs **~20 points of dE/dx recovery** (64.3% → 44.5%), confirming fig 39's ~14-point
estimate on a properly matched configuration. But it *helps* the calibration: long. diffusion is
+1.4% with noise and −13.9% without. That is consistent with the campaign's founding observation —
diffusion and readout noise both smear the signal, and a fit that has never seen noise mis-assigns
the smearing to diffusion. **Noise-ON is the correct configuration**, and the noiseless historical
numbers should not be read as an achievable target.

---

## 6q. Consolidated production performance, and how to pool it

### What the objective is made of

![PPP loss decomposed](plots/noise_report/fig44_loss_components.png)

Every checkpoint stores a full per-iteration `aux_iter` record, and the decomposition is **exact**:

```
total = log_likelihood_tick + no_match_penalty + expected_total_hits
      + dedx_prior + dedx_mean_penalty + dedx_barrier + mcs_prior
      + chain_drift_penalty + dedx_drift_penalty + spatial_moment
```

verified on a production run to **4.4e-3 absolute, 4.5e-7 relative** — float32 noise.

| term | mean contribution | share of \|total\| |
|---|---|---|
| joint hit log-intensity (tick **and** charge) | +8925 | **91.1%** |
| dE/dx student-t prior | −2178 | 22.2% |
| expected hit count (PPP integral) | +1692 | 17.3% |
| unmatched-target penalty | +1343 | 13.7% |
| MCS prior | +9.5 | 0.1% |
| dE/dx mean constraint | +8.9 | 0.1% |
| barrier / drift penalties / spatial moment | 0 | 0 |

> **CORRECTION (2026-08-06).** An earlier version of this subsection claimed the production
> objective had **no** per-hit charge term, on the strength of `aux["log_likelihood_charge"] = 0.0`
> (`optimize/strategies.py:811`) and a commented-out `#+ log_likelihood_charge` line in a
> `.ipynb_checkpoints` copy. **That conclusion was wrong.** The zero field is a *reporting stub*;
> the charge residual is inside the term stored under the key `log_likelihood_tick`, which is
> simply mislabelled.

**The key `log_likelihood_tick` holds the JOINT tick-and-charge likelihood.**
`ProbabilisticLossStrategy` builds, for every (hit, candidate, tick-offset) triple,

```python
window_log_charge_intensity = (-0.5 * (charge_diffs / (sigma_charge/1000))**2
                               - 0.5 * jnp.log(2*jnp.pi * (sigma_charge/1000)**2))
joint_window_log_probs = (window_log_probs + log_time_weights
                          + window_log_charge_intensity)     # <-- charge IS here
joint_hit_log_probs = jax.nn.logsumexp(joint_window_log_probs, axis=(1, 2))
```

and `aux["log_likelihood_tick"] = -sum(joint_hit_log_probs)`. All three constructions of
`joint_hit_log_probs` in that class include the charge term; there is no reachable code path that
omits it. `sigma_charge` defaults to **500 e⁻** and is exposed as `--loss_sigma_charge`.

**Proven empirically, not just read off the source.** Two 3-point scans on identical data,
differing only in `--loss_sigma_charge`:

| grid point | σ_Q = 500 e⁻ | σ_Q = 50 e⁻ | Δ |
|---|---|---|---|
| 1 | 3408.63 | 7327.46 | +115% |
| 2 | 3457.82 | 7344.28 | +112% |
| 3 | 4042.66 | 8653.50 | +114% |
| 4 | 12180.65 | 12748.09 | +4.7% |

Maximum relative change **99.3%**. The entire change lands in `log_likelihood_tick`
(max |Δ| = 4610.84, identical to the total); the charge-independent `expected_total_hits` moves by
**1e-4**; and `log_likelihood_charge` is all-zero in *both* runs — confirming it is a stub, not a
switch. A 10× tighter charge width cannot change the loss by 100% if charge is absent from it.

Charge therefore enters the objective **three** ways: the per-hit Gaussian residual inside the
joint term, the hit intensity (whether a pixel fires and when), and the expected-hit-count
integral.

### Which term pins each parameter

Panels (b, c) apply the same decomposition to the 1-D likelihood scans, which have the same `aux`
record. Range of each term across the scan grid:

| scan | hit log-intensity | expected hit count |
|---|---|---|
| **lifetime** | **73 887** | 3 719 |
| **A_b** | **30 604** | 2 089 |

**The hit log-intensity supplies ~95% of the curvature for both.** The PPP count integral
contributes ~5% and, for lifetime, is nearly flat near the minimum — it saturates below −60% and
adds essentially no constraint where it matters.

So lifetime is pinned almost entirely by the **joint** per-hit term — the combination of *when a
pixel fires* and *how much charge it collected* — rather than by the count integral. **Lifetime has
one structural handicap, not two**: the 0.087-lifetime lever arm of §6k. The earlier "second
handicap" (a charge-free objective) does not exist and is withdrawn. The 2-D behaviour of §6p is
unchanged by this: A_b absorbs charge normalisation readily because it moves the joint per-hit
term strongly, which is exactly what the loss measures.

**What is now open is σ_Q, not whether charge is in.** At 500 e⁻ against a median hit charge of
~10.7 ke, the loss assumes a 4.7% per-hit charge resolution — comparable to the *entire* 8.3%
anode-to-cathode lifetime signal (§6k). That weighting has never been derived or scanned. It is a
cheap, well-posed test (`LOSSSIGQ=... start_loss_profile.sh`, already wired) and it is the natural
successor to this subsection.

### What the fit is actually doing, iteration by iteration

![Production convergence traces](plots/noise_report/fig43_convergence_traces.png)

Eight quantities on one axis, all 8 completed production runs (thin = seed, thick = across-seed
median, 200-step rolling median because a single iteration sees one batch and batches differ
enormously). Five things are visible that no endpoint table shows:

- **A_b and E field settle by ~2000 iterations** and then sit flat. They are done long before the
  run ends, which is why they are the two parameters we trust.
- **Transverse diffusion converges cleanly** from a wild ±100% start.
- **Longitudinal diffusion overshoots hard** — the median swings to **+40%** around iteration 800
  before coming back down through zero and settling near −7%. That excursion is the co-adaptation
  with the dE/dx block, and it is why early read-outs of this parameter were meaningless.
- **Lifetime never flattens.** It crosses zero around iteration 4000 and keeps drifting downward to
  the end of the run. This is the schedule-dependence of §6c seen directly, and the reason lifetime
  is quoted as a bound.
- **The dE/dx panel shows the metric bug in one picture.** The orange curve (vs the SIM input, the
  guess file) *rises* from 0.0315 to 0.0353 while the green curve (vs arc-length-matched truth)
  *falls* from 0.0713 to 0.0584. Same fit, same iterations, opposite conclusions — the whole of the
  "MAE gets worse" puzzle.

Position and loss both descend monotonically and are still improving at 10 000 iterations, which is
consistent with §6b: geometry and loss have converged in the sense that matters, but nothing here
has plateaued so hard that a longer horizon is pointless.

![Every completed production run](plots/noise_report/fig42_production_runs.png)

Three directories hold the **same** production configuration — `sci_full_ANNEALLONG`,
`sci_full_ANNEALLONG2`, and `sci_full_MDX` (the last a no-op variant, since its minimum-length cut
never fires on the guess file). That gives **14 completed runs across only 8 distinct seeds**:
seeds 0–2 appear three times each.

**Those repeats are not independent samples.** `--seed` draws the *target*, so a repeat re-solves an
identical problem — verified, seed 0 carries `lifetime_target = 2406.4466` in all three sets.
Pooling all 14 and dividing by √14 overstates the precision. The correct procedure is to average
repeats within a seed first, then take the mean and s.e.m. over distinct seeds.

| parameter | error vs truth (n = 8 seeds) | between-seed s.d. |
|---|---|---|
| **A_b** | **+0.73 ± 0.05%** | 0.15 |
| **E field** | **−0.02 ± 0.01%** | 0.02 |
| transverse diffusion | −0.72 ± 0.42% | 1.18 |
| longitudinal diffusion | −7.86 ± 1.95% | 5.52 |
| **lifetime** | **−4.21 ± 1.53%** | 4.33 |
| position residual | 159 ± 4 µm | 12.7 |

### Half the scatter is bit-level irreproducibility

Panel (c) separates the two sources. Averaged over the repeated seeds:

| parameter | WITHIN seed (same target) | BETWEEN seeds |
|---|---|---|
| lifetime | **2.22** | 4.33 |
| long. diffusion | 3.6 | 5.5 |
| A_b | 0.085 | 0.15 |

Seed 0 returned **−4.27%, −8.33%, −1.99%** on three runs of an identical configuration against an
identical target — **6.35 points apart**, from GPU non-determinism alone. `--non_deterministic` is
set in every production run.

**Consequence: adding seeds cannot reduce the lifetime uncertainty below ~2.2 points.** That is a
floor no amount of target sampling removes, and it reframes the "3.6-point n = 3 resolution floor"
of §6n — a large part of that was never seed scatter at all. Any comparison finer than ~2 points
needs repeats at fixed seed, or `--non_deterministic` turned off, and it is worth measuring what
determinism costs in wall-clock before accepting the current default for final numbers.

### It is not the extreme targets

Panel (d): targets span **984–4901 µs**, a factor 5, so seeds are genuinely different problems.
But there is no trend — the worst run (−13.4%) sits at τ = 3640 µs while τ = 4901 gives +0.5% and
τ = 984 gives −2.2%. Target difficulty does not explain the spread, which leaves non-determinism
plus per-target variation.

---

## 6r. Where we stand: the best setup, its performance, and what would improve it

![The state of play](plots/noise_report/fig45_state_of_play.png)

This section is the campaign's bottom line. Nothing in it is a new fit — every number is a re-read
of arms already documented above, assembled in one place so the conclusion can be checked rather
than trusted.

### The best setup we have

**Production (S4).** Straight-line guess geometry; jointly fitting per-track positions, ~4000
per-segment `log_dedx` nuisances, and the five calibration parameters. Configuration:

| knob | value |
|---|---|
| data | 100 batches × 400 cm |
| iterations | 10 000 |
| calibration LR | **annealed**, `decay_rate` 0.91 per *epoch* |
| geometry basis | spline |
| chain LR | 1e-2, `chain_decay` 0.9997 |
| dE/dx prior weight | 5 |
| dE/dx mean-constraint weight | 1e5 |
| readout noise | **ON** |

Pooled over **14 completed runs spanning 8 distinct seeds**, repeats averaged within a seed first
(§6q — seeds 0–2 were each run three times, and `--seed` draws the *target*, so those are repeats
of one experiment):

| parameter | error vs truth | between-seed s.d. |
|---|---|---|
| **A_b** | **+0.73 ± 0.05%** | 0.15 |
| **E field** | **−0.02 ± 0.01%** | 0.02 |
| transverse diffusion | −0.72 ± 0.42% | 1.18 |
| longitudinal diffusion | −7.86 ± 1.95% | 5.52 |
| **lifetime** | **−4.21 ± 1.53%** | 4.33 |
| position residual | **159 ± 4 µm** | 12.7 |

**Ceiling (true geometry frozen), best configuration** — dE/dx prior 0.5, n = 3:

| A_b | E field | tran. diff. | long. diff. | lifetime |
|---|---|---|---|---|
| +0.19 ± 0.02% | −0.00 ± 0.00% | +0.50 ± 0.29% | +1.43 ± 1.24% | −2.32 ± 0.49% |

So: **A_b and E field are solved** — sub-percent and sub-0.1% respectively, and immune to every
degradation tested. The two diffusions and lifetime carry a few percent. **Lifetime is still a
bound, not a measurement**: it tracks total optimiser travel (§6c) and has not been shown to
asymptote.

> **Two numbers in this table have moved since the first version of this summary**, and both moved
> the wrong way. That version pooled **n = 11 runs as if independent** and quoted lifetime
> −3.09 ± 0.87% and long. diffusion −5.37 ± 1.18%. Two things were wrong with it: repeats of the
> same seed were counted as independent samples (§6q), and four `annl2_more` seeds were still
> running at 6600–9000 iterations, so partial checkpoints were being read. With those fixed and the
> seeds now complete, lifetime is **−4.21 ± 1.53%** and long. diffusion **−7.86 ± 1.95%**. The
> honest reading is that the extra seeds were *harder*, not that anything regressed.

### What would improve it, in order of expected gain

**1. Drop the dE/dx prior weight 5 → 0.5** — panel (b). The single largest win available, and it
costs nothing. At ceiling it moves long. diffusion **−9.95% → +1.43%**, tightens A_b 4× and
lifetime ~2×, and 2.5× the dE/dx recovery (§6p). Production uses 5, inherited without
re-derivation; 0.5 is the *script default*. `full_p05` (6 seeds) is running to confirm it transfers
to production, where the geometry block is also live.

**2. More data, with iterations scaled alongside** — panel (c). `nb200` doubles the data (200
batches) *and* the iterations (20 000), so epochs and per-batch visit counts are preserved. Only
seeds 0–1 have finished, so panel (c) is **seed-matched**: each arm is compared with production
restricted to that arm's own seeds, because a 2-seed s.d. against an 8-seed s.d. is not a
comparison.

| seed-matched, seeds {0, 1} | production | 2× data + 2× iters |
|---|---|---|
| A_b | +0.85% | **+0.73%** |
| E field | −0.04% | −0.03% |
| transverse diffusion | −1.10% | **+0.30%** |
| longitudinal diffusion | −3.50% | **−1.28%** |
| lifetime | −3.99% | **−3.02%** |
| position | 159 µm | 165 µm |

**Every central value moves toward truth, and position holds.** That is the result. What does *not*
hold is the earlier claim that it "tightened every error bar" — seed-matched, the A_b and
transverse-diffusion spreads do tighten (0.15 → 0.02 and 1.55 → 0.39) but longitudinal diffusion
and lifetime get **worse** (0.77 → 9.90 and 1.74 → 3.59). With n = 2 — one degree of freedom —
none of those spreads is a measurement, and that claim is withdrawn until the remaining seeds land.

Raising batches at *fixed* iterations does the opposite: `nb400` gives **774 µm** of position
residual against 164 µm on the same seeds, and lifetime, both diffusions and their scatter all
degrade. The geometry block simply gets 4× fewer updates per track and starves. **Scaling
iterations alongside batches is not optional.**

**3. Run everything at ≥ 6 seeds, and settle knobs at ceiling** — panel (d). Three runs of an
identical configuration span 3.6 points on lifetime (§6n), and the expected error bar only drops
from 2.2 to 1.5 points between n = 3 and n = 8. Ceiling shrinks the seed s.d. 11–70×, which is what
makes single-knob differences resolvable at all. **Two campaign conclusions died purely from
insufficient power.**

### What the studies say *not* to do — panel (e)

| tested | verdict |
|---|---|
| improve geometry accuracy further | a **variance** lever, not a bias lever — against random *and* systematic (chord-cutting) error (§6h). 159 µm is fine. |
| the dE/dx minimum-length cut | a no-op in production (shortest guess segment 0.909 cm vs a 0.15 cm cut) and a clean null where it fires: Δbias +0.07, Δscatter −0.04 at ceiling (§6n). |
| lower the mean-constraint weight 1e5 → 3000 | buys **no** bias improvement that survives its own scatter, and costs **7.5× the lifetime seed s.d.** (0.69 → 5.21) and 3.7× worse A_b (§6p). |
| turn readout noise off | Δbias **+5.4 points** — long. diffusion −13.94%. It costs dE/dx recovery (44.5% → 64.3% of the gap closed) but noise-ON is what makes long. diffusion come out right (§6p). |
| second-order optimisers on any block | all three verdicts are in (GN report; calibration block, dE/dx block, geometry block). |

**A caveat on how panel (e) is read.** It plots two numbers per arm, because one is not enough.
The mean-constraint arm has a *lower* RMS central error than its control (0.94 vs 1.24) purely
because its lifetime central value happens to land at +0.06 — with a seed s.d. of 5.21, i.e.
±2.13 on the mean. That is not a measurement of anything. Judging it on central values alone would
reverse the §6p conclusion; judging it on both bias and scatter does not.

### The two hard limits — panel (f)

**Lifetime is lever-arm limited.** The detector is **0.087 lifetimes deep** — an **8.3%**
anode-to-cathode charge swing (§6k). That is the entire signal, and no optimiser, prior, or loss
term creates more of it. It is why lifetime is structurally the weakest parameter. Panel (f) also
overlays what the objective *assumes* about per-hit charge: σ_Q = 500 e⁻ ≈ 4.7% of a median
(10.7 ke) hit — i.e. the assumed per-hit resolution is **comparable to the whole signal**. That
weighting has never been derived, and scanning it is the cheapest open test we have (§6q).

**The dE/dx block is objective-limited.** With *perfect* geometry and realistic noise it recovers
at most ~**45%** of the prior→truth gap (64% with noise off). Over half the per-segment structure
is unreachable regardless of geometry quality (§6o) — an independent confirmation of the
second-order study's verdict, measured on fits rather than inferred from curvature.

### The one result that would change the picture

**`annl_2xtravel`** — double the total optimiser travel at the same final learning rate (running,
at 11 800–13 000 of 20 000 iterations). Lifetime has tracked travel monotonically across the
campaign: **−2.01 → −2.70 → −5.02%** as ΣLR grows. If it **asymptotes**, lifetime becomes a
measurement. If it **keeps drifting**, we should stop quoting a central value for it entirely and
report a bound. That result, plus `full_p05`, are the two worth waiting for before anything here is
finalised.

## 7. Methodology: the traps that cost us real time

These generalise beyond this project and are the reason several earlier "results" were wrong.

![Traps](plots/noise_report/fig6_traps.png)

**Trap 1 — reading endpoints of unconverged fits.** A systematic audit of all **105 campaign
runs** with an automated plateau detector found that **80% were still moving when they were
read out**:

![Plateau audit](plots/noise_report/fig4_plateau_audit.png)

Among the runs that *did* plateau, the reported value shifts by up to **70 percentage
points** depending on whether you read the endpoint or the plateau. Every cross-variant
comparison in this campaign was therefore made between seeds that were not equally
converged. One consequence: a threshold test previously called "inconclusive" reads +1.1% at
its endpoint and +11.9% at its plateau.

**Trap 2 — per-batch metrics.** The position residual is a *per-batch* quantity and swings
between 120 µm and 1400 µm within a single converged run. Comparing final values across runs
is meaningless; we now use a per-seed tail median.

**Trap 3 — configurations rebuilt from truncated logs.** A five-arm, 15-job scan was
invalidated because the configuration was reconstructed from a command-line dump that had
been truncated mid-flag, silently dropping `--chain_lr 1e-2` and letting a 100× smaller
default apply. Geometry never moved. The tell was that three *different* random seeds
returned position residuals agreeing to 0.1% (878.8 / 879.6 / 880.5 µm) — impossible unless
the quantity is frozen. We now diff a new run's full argv against the reference run's stored
provenance before trusting any comparison.

**Trap 4 — a knob inherited from a different regime, and not checking the record that would have
caught it.** Every recent ceiling arm was launched with `SCIDEDXPRIOR=5`, copied wholesale from the
ANNEALLONG *fitted-geometry* configuration, against the script default of **0.5**
(`start_sci_case.sh:38`). Against matched noise-ON runs the 10× stiffer dE/dx prior cut the block's
recovery from **40–49%** to **17.6%** (§6o), and it also **entangled** the `ceil_w3k`
mean-constraint test, which varies one dE/dx weight while another sits 10× off default.

The compounding failure is worse than the original slip. When the discrepancy surfaced, the
historical runs were declared unreadable because they predate the `provenance` block — and the
attribution was built from script defaults plus a launch-log survey instead. **Every history pickle
has always carried the full argparse `Namespace` under a `config` key.** One lookup would have
given the answer directly; the inference that replaced it was also partly wrong, since the measured
values show most of the historical spread is a *noise* effect, not a prior effect.

Two rules, in order of importance: **read the stored config before reasoning about what a run
used** — checkpoints carry more provenance than the obvious key — and, when reusing a
configuration across regimes, diff it against the script defaults, not only against the previous
run (which was Trap 3's lesson and is not sufficient).

**Trap 5 — a knob that is a no-op on one input file and active on another.**
`LARND_DEDX_MIN_DX = 0.15` freezes ~4% of segments on the true file and **nothing at all** on the
straight-line guess, whose shortest segment is 0.909 cm — six times the cut. The production arm
`mdx_full` was therefore configuration-identical to its control and tested nothing, while the
label still read `mdx0.15`. A recommendation measured in one regime need not even *execute* in
another; check that a knob fires before interpreting its effect.

**Trap 6 — n = 3 is below the resolution floor.** Three runs of an *identical* configuration give
lifetime means spanning **3.6 points** (§6n). Several campaign comparisons — the mean-constraint
weight at 3.4 points among them — were smaller than that and were therefore never measurable at
this sample size. Comparisons must now be run at ceiling (which shrinks the seed s.d. 11–70×)
and/or with ≥ 6 seeds.

---

## 8. What is running now

| arm | what it tests | status |
|---|---|---|
| `ANNEALLONG` (3 seeds, 10 000 it) | does the −2.6% result survive a longer horizon? settles the failed convergence criterion | **COMPLETE** — see §6b |
| `CONSTDEDX` (3 seeds, 10 000 it) | dE/dx frozen at the mean: is the nuisance block earning its keep? | **COMPLETE** — no; removing it costs long. diffusion 4×, §6d |
| `PROBTGT` (3 seeds, 10 000 it) | target drawn from the probabilistic distribution: is our target biased? | **COMPLETE** — no detectable bias, §6d |
| `ANNEALMORE` (seeds 3–5, 5000 it) | tightens n = 3 → 6 | **COMPLETE** — reproduces; see §6 |
| `FRAMEONLY` / `DRIFTW6` / `DRIFTW7` (9 jobs) | the drift-axis basis fix: rotation-only control, plus two penalty weights | `DRIFTW6` (w=1e6) **COMPLETE — see below**; other two running |
| `CHSTART1000` / `CHSTART2500` (6 jobs) | release geometry only after calibration settles | running |
| `S3ANNEAL` (3 seeds) | does annealing fix lifetime even with *wrong frozen* geometry? | queued |
| `S2ANNEALX` (seeds 1–2) | completes the S2 anneal control | queued |
| `qpos50/170/400/880` (12 scans) | position quality ladder | **COMPLETE** — variance, not bias; §6f |
| `qdedx0.75/0.5/0.25/0.0` (12 scans) | dE/dx quality ladder | **COMPLETE** — monotone bias; §6f |
| `sci_nb400` (3 seeds, 10 000 it) | 4× the data (400 batches, 1.6 km of track) | **COMPLETE but INVALID** — silently un-annealed the LR schedule; §6g |
| `sci_shuffle` (3 seeds, 10 000 it) | `--shuffle_bt`: does reshuffling batches remove the per-batch drift lever-arm effect (open question 5)? | running, 2/3 done |
| `prof_true_fine`, `prof_dedx0.5_fine` | 0.75%/1.5% scan step instead of 10.2% | **COMPLETE** — objective nearly unbiased; §6g |
| `qreseg` | segmentation alone | **COMPLETE** — no effect; §6g |
| `qdedx0.40` | dE/dx smoothed to the guess file's real spread | **COMPLETE** (18/21 batches) — explains ⅓ of the gap; §6g. Clean rerun queued |
| batch-size ceiling sweep (`ceil400`–`ceil1600`) | how large can `max_batch_len` go on the production config? | **COMPLETE** — **600 cm works, 700 OOMs**; 400 cm peaks at 23.0 GiB. Batch *count* stays free |
| `nb200` (3 seeds) | 2× the data with epochs, per-batch visits and final chain LR all preserved (200 batches × 20 000 iters, `chain_decay` 0.9997→0.99985) | running |
| `qchord0.5` / `qchord1.0` | *systematic* chord-cutting geometry error instead of random offsets | **COMPLETE** — null; geometry is not a bias channel, §6h |
| `qcombo` | chord 1.0 **+** dE/dx f=0.40 together: is there an interaction? | **COMPLETE** — no interaction, §6h |
| `qmeanup` / `qmeandown` | dE/dx **mean** shifted ±2% (new axis) | **COMPLETE** — the dominant bias channel; explains the guess file, §6h |
| `NB400FIX` (3 seeds) | the 4× data test, with `decay_rate` corrected to 0.686 | **COMPLETE** — variance ↓, geometry ↓, §6i |
| `SHUFOFF2` / `SHUFON2` (3 seeds each) | paired shuffle test on one tree | **COMPLETE** — no effect; open question 5 answered, §6i |

**First result on the drift-axis fix (§5.4), and it is negative.** The w = 1e6 arm completed
(3/3 seeds). The penalty does exactly what it was designed to do — drift-projected
displacement falls from 222 µm to **18 µm** and the drift fraction from 57% to **7%** — but
every physics number degrades: lifetime −2.6% → **+8.8 ± 6.0%**, long. diffusion −5.1% →
**+59 ± 27%**, and the position residual worsens 255 µm → **648 µm**. The most likely reading
is that the drift-axis displacement is largely *legitimate* track structure rather than
absorbed calibration error, so clamping it removes a degree of freedom the geometry genuinely
needs. **Caveat:** this arm changes both the frame and the penalty; the `FRAMEONLY` control
(rotation only, no penalty) is still running and is required to separate them. Until it
lands, treat §5.4's proposed fix as *tested and probably refuted*, not established.

The `S3ANNEAL` arm is worth flagging: an accidental by-product of the invalidated scan (§7,
Trap 3) was a set of runs with geometry effectively frozen at an ~880 µm-wrong guess *and*
the annealed LR schedule — and they recovered lifetime to −1.9 ± 1.2%, versus +88 ± 36% for
the original S3. If that reproduces deliberately, it would mean the **LR schedule, not
geometry accuracy, dominates the lifetime bias**, which would substantially reorder our
priorities. It came from a broken run, so it is being re-tested properly before we believe it.

---

## 9. Open questions

1. **Is the annealed result converged?** Resolved for four of five parameters at 10 000
   iterations (§6b/§6c): A_b, E field and both diffusions are converged and schedule-invariant;
   **lifetime is not** — it tracks total optimizer travel and its quoted value is a lower bound
   on the bias.
2. **`long_diff`** is the remaining loose parameter (−5.4 ± 2.9%). §6d now gives a partial
   explanation: it is the parameter most exposed to per-segment charge structure. Freezing the
   dE/dx nuisances sends it to −21%, so a sizeable part of its residual is competition with
   that block rather than an independent defect.
3. **The S1 anomaly** (§3): S1 should not be worse than S2.
4. **Unverified FEE claim.** A suspicion that the stochastic-vs-analytic front-end model
   introduces a charge bias has been repeated in internal discussion but **has never been
   measured** — a dedicated probe returned "could not test". It should not be cited until
   somebody runs `get_adc_values` against `get_adc_values_average_noise_chunked` directly.
5. **Data volume and batching order.** *Answered — §6i.* **Order does not matter**: a paired
   shuffle-on/off run submitted together on one tree agrees within errors on every parameter
   (largest gap 1.4σ; position 492 vs 475 µm). **Volume helps precision but costs geometry**:
   4× the data tightens long. diffusion's seed spread 23.2 → 13.7 while degrading position
   492 → 774 µm, because the geometry block receives 4× fewer updates per track at fixed
   iteration count. To scale data without that cost, iterations must scale with the batch count.
   Production still uses **864 of 216 091 tracks (0.4%)** of the input file.
6. **Gauss–Newton** was evaluated and rejected as a from-scratch optimizer (the landscape is
   a curved valley; it rides saddles and pins parameters at bounds). It *is* validated as a
   final polisher: Adam@1000 + 2–3 damped LM–GN steps reaches Adam@5000 quality and yields a
   5×5 covariance for free. Note the Gauss–Newton/Fisher approximation differs from the exact
   Hessian by 1.75–4.27× here and the gap *grows* with more batches, so it must not be used
   as a curvature source for preconditioning.
7. **Why does the per-segment dE/dx MAE end ~30% worse than it started?** The nuisance block
   appears to absorb signal during the phase when calibration moves fastest.
8. **Seed selection.** Targets are drawn per seed and sometimes land close to the initial
   guess, so some seeds are not meaningful tests. Future seed sets should force distant targets.
9. **Is the target biased?** *Answered — no.* The `PROBTGT` arm (§6d) reproduces ANNEALLONG on
   all five parameters, so the deterministic target is not a bias source at this precision.
10. **What exactly does the dE/dx block absorb?** §6d shows removing it costs long. diffusion
   4× and lifetime 2.7×, but the arm confounds "no per-segment freedom" with "wrong per-segment
   values". An arm with dE/dx frozen at the *true* per-segment values would separate the two and
   has not been run. §6f adds the dose-response curve for the second of those: dE/dx error
   displaces the minimum monotonically, so at least part of the §6d effect is genuine bias.
11. **Why is the guess file worse than *both* endpoints of the ladder?** ***Answered — §6h.***
   Its length-weighted dE/dx **mean** is 1.87704 vs truth's 1.88533 — **−0.440%** — and lifetime
   carries a −24.3 points-per-1% slope on that quantity, predicting +12.9% against +15.2 ± 4.9%
   observed. Segmentation, random position error and chord-cutting are all null; dE/dx *spread*
   contributes a further ~+3.7. The suspect promoted in §6g (error *structure*) was refuted.
13. **What is our fitted dE/dx mean error, and what does it cost?** *Answered — §6k.* It is
   **~0.1%** (−0.096% to −0.147% on the 100-batch arms, +0.012% to +0.034% on NB400FIX) — four
   times better than the guess file. And it costs nothing measurable: the §6h leverage
   mispredicts our fits' lifetime in magnitude everywhere and in sign for five of nine seeds.
14. **Redo every scan as a PROFILE, not a slice.** *(new, §6k — now the top methodological item.)*
   §6e–§6h all hold the other four parameters frozen, which forces lifetime to absorb any charge
   normalisation error and manufactures a ~25× sensitivity that the fit does not have. No
   displacement measured on a slice should be called a bias until it survives minimising over the
   others. §6e flagged this at the outset; §6h ignored it.
15. **Then what does drive the residual lifetime error?** *(new, §6k)* Not the dE/dx mean, and not
   geometry (§6h). With the lever arm at 0.087 lifetimes, lifetime is the weakest parameter we
   fit; a shape-based estimator (which recovers τ to ~1% and is scale-immune) is worth adding as a
   cross-check or an explicit loss term.
12. **How much of §6e/§6f's absolute offsets was grid coarseness?** *Answered — §6g.* About one
   point on lifetime and 3.6 on long. diffusion for the undegraded condition, and ~0 for the
   dE/dx rungs. It was **not** common-mode as §6f assumed. With perfect geometry the objective
   is displaced by at most ~1%, which weakens §6e's reading of long. diffusion.
13. **Does more data help?** *(new, §6g)* Unanswered — the first attempt confounded data volume
   with the LR schedule, because the decay is indexed in epochs. Re-running as `NB400FIX`.

---

## 10. Reproducing

```bash
# Current best S4 configuration (3 seeds)
SCIMODE=full SCIITER=5000 SCILEN=400 SCILRDECAY=0.91 SCIDEDXPRIOR=5 \
SCIBASIS=spline SCICLR=1e-2 SCICHDECAY=0.9997 \
  sbatch --array=0-2%3 optimize/scripts/start_sci_case.sh
```

Figures regenerate with `python3 make_noise_report_plots.py` (writes `plots/noise_report/`).
Every fit checkpoint stores its full `argv`, git SHA, dirty flag, hostname and SLURM job IDs
under `provenance`, plus all `LARND_*` environment overrides — use these rather than
reconstructing configurations by hand.
