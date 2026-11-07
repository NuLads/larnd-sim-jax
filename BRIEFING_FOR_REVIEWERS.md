# Briefing: differentiable LArTPC calibration — state as of 29 Jul 2026

You are reviewing an ongoing physics-analysis effort. Everything below is measured. Where an
interpretation is offered it is flagged as such, including three that have already been
refuted by their own tests. **Challenge anything, including the framing.**

## 1. The problem

`larnd-sim-jax` is a differentiable simulation of a liquid-argon TPC. We fit detector
calibration constants by gradient descent on a marked-Poisson-point-process negative
log-likelihood comparing simulated to target waveforms.

Five calibration parameters of interest:
`Ab` (recombination), `eField`, `lifetime` (electron attenuation), `tran_diff`, `long_diff`.

Simultaneously fitted nuisances:
- **~4000 per-segment dE/dx values** (one per track segment), with a student-t prior,
  soft barrier and mean constraint.
- **Per-track geometry**: track shape as a smooth transverse displacement in a sine basis
  (~3-12 modes/track, 864 tracks), starting from a straight-line guess.

Target data has FEE electronics noise thrown stochastically; the fitted model is a
probabilistic forward model that marginalises that noise analytically. Batches are 400 cm of
track length, 100 batches per epoch, 5000-10000 iterations, Adam.

Difficulty ladder (all noise-ON, 400 cm, 3 seeds, robust estimator = trailing-20% median):

| stage | dE/dx | geometry | Ab | eField | lifetime | tran_diff | long_diff |
|---|---|---|---|---|---|---|---|
| S1 | true | true | -0.1+-0.1 | +0.1+-0.1 | +2.9+-2.5 | +0.5+-1.5 | +0.8+-0.8 |
| S2 | fitted | true | +0.3+-0.3 | +0.2+-0.0 | +0.6+-1.9 | +1.9+-1.9 | +0.8+-0.9 |
| S3 | fitted | WRONG, frozen | +0.1+-0.6 | +0.0+-0.3 | +137.7+-47.4 | +16.6+-18.1 | +71.0+-49.2 |
| S4 | fitted | fitted | -1.1+-0.1 | +0.2+-0.2 | +62.7+-49.5 | +13.1+-4.7 | +65.2+-6.9 |

S4 is the configuration that matters for real data. It was broken for weeks.

## 2. The intervention that worked

The calibration learning-rate schedule (`optax.warmup_exponential_decay_schedule`) applies its
decay **once per epoch** (`transition_steps = epoch_size = 100`, `staircase=True`). With the
old `decay_rate=0.999`, over 50 epochs this leaves **95.7% of the peak LR at the end of the
run** -- effectively no annealing. Meanwhile the ~4000 dE/dx nuisances run on a bare
`optax.adam(1e-2)` with **no schedule at all**, and geometry decays to 22%.

Changing `decay_rate` to 0.91 (leaving 1.6% of peak) is the entire fix.

S4 @ 5000 iterations, n=6 seeds, robust estimator:
`Ab +0.60+-0.07 | eField -0.03+-0.02 | tran_diff -0.57+-1.54 | long_diff -4.35+-9.63 |
lifetime -1.08+-5.11 | position 255+-13 um`

S4 @ 10000 iterations, n=3 (same seeds): calibration unchanged within noise, but
**position 255 -> 168 um** and seed spreads tighten (long_diff +-9.7 -> +-2.4).
Residual drift falls below the noise floor at ~iteration 6000.

## 3. Everything else that was tried, and failed

- slow chain (geometry) LR decay; dE/dx prior x10; a drift-profile penalty on the dE/dx trend;
  freezing geometry after convergence; a two-pass scheme; 10k iterations without annealing;
  Polyak/tail averaging. All refuted.
- **Geometry release timing**: releasing geometry at iteration 1000 or 2500 instead of 0 --
  null result, statistically identical to releasing at 0.
- **Gauss-Newton** as a from-scratch optimizer: unsuitable (curved valley, rides saddles, pins
  parameters at bounds). Validated only as a final polisher. Note GN/Fisher differs from the
  exact Hessian by 1.75-4.27x here and the gap GROWS with more batches -- so it must not be
  used as a curvature source for preconditioning.

## 4. Curvature structure (measured, exact Hessian at truth, S2 @400, 20 batches)

Eigen-decomposition of the 5x5 calibration Hessian:
```
lambda=3.3e3   0.83*lifetime + 0.56*long_diff     <- softest
lambda=6.1e3   0.56*lifetime - 0.83*long_diff
lambda=3.9e5   tran_diff
lambda=8.0e6   Ab
lambda=1.3e10  eField
```
Statistical floors from H^-1: sigma(lifetime)=1.61%, sigma(long_diff)=1.44%.
A measured 2-D NLL scan (lifetime x tran_diff, others at truth, 13x13 grid, 0.1 in log units)
puts the minimum exactly on truth -- the loss is unbiased at ~10% grid resolution; the
difficulty is flatness, not a displaced optimum.

An 8x8 study adding 3 RIGID geometry modes: those modes are pinned by data to microns
(sigma 0.5-1.3 um); marginalising them barely affects lifetime (3.75->3.79%) but inflates
eField 119x (a global drift-scale change mimics a drift-velocity change).

## 5. THREE INTERPRETATIONS THAT WERE PROPOSED AND THEN REFUTED

Read these carefully -- the pattern of failure may itself be informative.

**(a) "Local per-track geometry noise poisons lifetime."** Motivated the whole geometry
work programme. REFUTED: S3 has *frozen* geometry (no noise injection possible) and still
fails at +137.7%; re-running S3 with the annealed schedule and the SAME ~880 um-wrong frozen
geometry recovers lifetime to **-0.9 +- 3.2%**. Geometry error displaces the walk; it does not
cause the bias. Caveat: with wrong frozen geometry the *diffusion* constants stay poor
(long_diff +16+-25%), so geometry fitting still earns its keep for those.

**(b) "The geometry basis is blind to the drift axis, letting it absorb attenuation."**
Measured: 58% of fitted displacement lies along the drift axis; the 2nd basis vector has
median |e2.zhat|=0.94. A penalty was built to suppress it. REFUTED by dose-response over 12
seeds: drift displacement 225 -> 18 -> 2 um as designed, while position degrades
268 -> 648 -> 686 um and long_diff blows up to ~+55%. A frame-only control (rotation, no
penalty) reproduces baseline exactly, so the penalty alone causes the damage. Conclusion: the
drift-axis displacement is largely legitimate track structure.

**(c) "The residual error is a slide along the degenerate (lifetime, long_diff) valley."**
Variance budget for lifetime across 6 targets: observed sigma 5.60%, run-to-run (paired,
same target) 1.25%, Hessian floor 1.61% -> 5.21% unexplained. The per-target residuals lie
96.8% along ONE direction with |cos|=0.906 to the Hessian's softest eigenvector, and
r(lifetime_err, long_diff_err)=+0.92. An init sweep showed the init-induced shift is 0.998
aligned with that same direction. REFUTED by its own test: pinning `long_diff` at truth (fit
the other 4) should have collapsed the lifetime error below 2%. It did not --
target S3: -7.27% -> -5.81%; target S5: **+8.95% -> -7.23% (sign flip)**. Both pinned runs
land near -6.5%, suggesting a real lifetime bias that long_diff freedom had been masking and
scattering. n=1 per target; unconfirmed.

## 6. Other measured facts

- **Run-to-run reproducibility at FIXED target** (paired replicates, robust estimator):
  lifetime 1.25 pts, long_diff 3.48, tran_diff 1.97, Ab 0.06. Differences smaller than these
  between configurations are not resolvable.
- **Initialization dependence** (4 starting points, same target, lifetime init 700..5700 vs
  target 2406): converges from +-137% away, so it is NOT merely quenching near the start. But
  final lifetime spans -9.16 to -1.74 (spread 2.77, above the 1.25 noise floor). Ab (+-0.19)
  and eField (+-0.02) are init-independent; the two soft parameters are not.
- **Target-dependence is strongly asymmetric.** Forcing every parameter far from init:
  low-lifetime target (600, i.e. +267% from init) recovers to -2.28% and -0.91%;
  high-lifetime target (5000, -56% from init) recovers to **-12.93% and -23.35%**.
  Long lifetimes = less attenuation = less signal.
- **Ab carries a reproducible +0.60 +- 0.07% bias** across all seeds, none negative
  (~7 sigma), and reproduces to 0.0 points at fixed target. Small but systematic.
- **The dE/dx nuisance gets WORSE during the fit**: per-segment MAE rises 0.030 -> 0.044
  MeV/cm over the first ~1000 iterations, recovering only to ~0.035-0.039. Unexplained.
- **Data volume**: production fits use **864 of 216,091 tracks (0.4%)** of the input file.
  Batches are built once by sequential file order and **never reshuffled**; per-batch drift
  lever-arm varies 6x (200cm) to 78x (50cm) between batches.
- The truth file's drift coordinate spans only **+-31 cm = 16% of the 385 cm attenuation
  length**. This is baked into the input file, not a code cut.
- **Unverified**: a claim that the stochastic-vs-analytic FEE/ADC model introduces a charge
  bias has circulated but has NEVER been measured. Do not treat it as established.
- Position residual is a PER-BATCH quantity that swings 120-1400 um within one converged run;
  always use a tail median. 80% of all 105 campaign runs were still moving when read out.

## 7. Where things live

- Report: `NOISE_CAMPAIGN_REPORT.md`, figures in `plots/noise_report/`
- Fitter: `optimize/fit_params.py`; loss: `optimize/strategies.py`; data: `optimize/dataio.py`
- Launch script: `optimize/scripts/start_sci_case.sh` (env-driven: SCILRDECAY, SCICLR,
  SCICHDECAY, SCIINIT, SCITGT, SCIPARAMS, SCILEN, SCIITER...)
- Results: `fit_result/sci_*/history_iter*.pkl` -- each stores full `argv`, git SHA and env
  overrides under `provenance`. **Directories may contain BOTH 400 cm and 50 cm runs; filter
  on `len400` or you will silently mix them.**
- Robust estimator + convergence table: `robust_convergence.py`

## 8. Your task

Review this critically and propose **the best ways to move forward**. Be concrete: what
should be measured or built next, in what order, and what would falsify each proposal.
Prioritise by (expected information gained) / (cost). Flag anything above you believe is
wrong, over-claimed, or resting on too little evidence. If you think the effort is optimising
the wrong thing entirely, say so.

Cheap CPU-only analyses of existing checkpoints are especially valuable; a GPU run costs
~7-14 hours for 3 seeds.
