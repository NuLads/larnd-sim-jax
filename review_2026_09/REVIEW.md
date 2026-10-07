# Can we calibrate a detector on cosmic muons with this fit? A review of the campaign

*23 September 2026. An independent re-check of the June–September campaign. Every number in the
"verified" columns was recomputed from the raw checkpoints (`fit_result/sci_full_*/history_*.pkl`)
or measured fresh for this review; nothing is copied from the project docs unchecked.*

---

## 0. Short answer

**Not yet — and not for the reason the recent docs focus on.**

1. **What works (in simulation closure):** the machinery runs end-to-end at production scale
   (100 batches × 400 cm, noise ON, ~4000 dE/dx nuisances + spline geometry). On 16 genuinely
   independent track samples it recovers **A_b to 0.11% and E field to 0.03%** (mean |error|), and
   **transverse diffusion to ~1.6%**.
2. **What doesn't:** **lifetime is biased by +8.85 ± 1.18%** (n = 16, 15/16 high), and a single fit
   scatters by ±4.7%. **Longitudinal diffusion is off by +14%** (mean |err| 14.6%). Neither is fixed
   by any of the ~40 interventions tried. The two "deliverables" from September (the ×0.40 prior
   scale and the calibration curve) do not hold up out of sample.
3. **New: the forward model itself is sound.** With everything at truth, the model the fit uses
   matches the noisy target to within **+0.34% in total charge**, with almost no drift dependence. The
   likelihood minimum sits at **lifetime +0.5 ± 0.3%** (A_b profiled). So the bias is **not** a
   forward-model defect. It comes from how the **nuisance blocks are estimated**.
4. **The mechanism, measured:** lifetime is **~20:1 degenerate with the overall charge scale**. A 0.1%
   shift in A_b is worth ~2–3% in lifetime, because at τ ≈ 2.2 ms only 8% of the charge is lost over
   the 30 cm drift. Across the 16 replicas the A_b and long_diff errors explain **79% of the lifetime
   scatter** (r(A_b, τ) = −0.81). The joint MAP fit over thousands of nuisances slides along this
   soft plane.

   **Update 24 Sep (Laplace test, §3.3b):** a clean test does **not** support the attribution I first
   made here, namely that point-fitting the dE/dx nuisances causes the offset. It used true
   geometry, a correct population prior and no mean constraint, on all 100 production batches.
   - Fitting per-segment dE/dx moves the lifetime by only −2.1 ± 2.2 points (not significant).
   - Integrating dE/dx out (Laplace) changes nothing further (−0.2 ± 0.3).
   - What free dE/dx *does* do is **widen the lifetime uncertainty 2.5×** (0.75% → 1.85% on 400 m
     of track).

   The production +8.85% must therefore come from production-specific ingredients: the dE/dx prior
   and mean-constraint configuration, the resegmented guess tracks, or their interplay with the
   geometry fit.

   **Update 25–27 Sep (§3.3c, §3.3d):**
   - **Production dE/dx configuration:** with exact geometry, the full configuration (1 cm pieces,
     production prior, mean constraint) is a null: −0.8 ± 1.1 points against the clean setup.
   - **The 2×2 in the production fitter:**
     - fitting dE/dx alone is harmless (+2.3 ± 1.9);
     - fitting geometry alone costs ~+7;
     - fitting **both** adds **+8.1 ± 2.2 beyond the sum** (3/3 samples).
   - **The bias is a geometry × dE/dx co-adaptation.**
   - **Drift coordinate (28–29 Sep, §3.3d, n = 6):** fixing the drift coordinate at (the spline
     projection of) truth shifts lifetime by a remarkably constant **−14.5 ± 1.3 points (6/6)**, and
     long_diff by −24. Over six samples lifetime goes from +9.4 ± 2.0% to **−5.0 ± 1.5%**. So
     the drift coordinate carries most of the bias, but not all: a ~5% low residual remains
     (my n = 3 "within 2% of truth" did not replicate). Whether that residual is the coarse
     3-mode drift representation is being tested.
5. **Why this matters most for real data:** every closure test so far gets **truth for free** in
   places a cosmic analysis never will:
   - geometry initialised from true endpoints;
   - track identity taken from MC labels;
   - selection cuts applied on true drift coordinates;
   - **t0 known (= 0)**;
   - dE/dx prior shape and mean target fitted to the true dE/dx of the file being fitted;
   - kb, v_drift, gain and W-value fixed at their true values.

   There is **no real-data code path**. The fit is also blind to any detector effect that isn't in
   both the target and the model: per-pixel gain and threshold spread, space charge, dead channels,
   the recombination model's functional form, pile-up.

   **Given (4), any unmodelled drift-correlated charge effect of 0.4% becomes a ~5% lifetime error.**
   That is the real obstacle for cosmic data.

**Recommendation:**
- Treat **A_b/E field/tran_diff** as the near-term deliverable.
- Keep the joint dE/dx fit. On its own it is harmless, and the mean constraint recovers most of
  its precision cost. But do not let dE/dx and geometry co-adapt freely, because that interaction
  is what biases lifetime (§3.3d). **Take the drift coordinate from hit timing (with a real t0),
  not from the nuisance fit**. In simulation that removes a constant ~14-point component (n = 6); a
  ~5% low residual remains under investigation. Marginalising dE/dx (Laplace) is *not* the fix: it only halves the A_b shift.
- Cross-check lifetime with a charge-scale-immune shape estimator.
- Before any real data, make the closure test **honest**: reconstructed inputs, t0 handling, a
  prior not fitted to truth, and a deliberate model-mismatch stress test.

Details, evidence and the ordered plan follow.

---

## 0c. CHECKPOINT 7 October — a truth-free recipe for all five parameters (simulation closure)

**Recipe R2 (validated, replicated on fresh samples).**
- **Configuration:** production joint fit (5 calibration parameters, per-segment dE/dx with the
  production priors, spline geometry, K = 3), plus two truth-free geometry steps:
  1. **Hit-seeded start** (`LARND_DRIFT_FROM_HITS=1 LARND_HITS_SEED_E1=1`). Each track's starting
    shape is fitted to its own target hits: one point per pixel, gated in 3-D within 1 cm of the
    nominal line, robust LS on [1, α, sine modes].
    - Drift modes are used only if significant against "straight" (χ² at 99.9%); pixel-plane modes
      only if the shape exceeds 500 µm.
    - Guards (added 7 Oct): skip if the pixels span < half the track; reject a seed implying > 2 cm
      displacement. The guards change 1 track in ~13 000 (the single runaway, below).
  2. **Freeze the drift shape of straight-looking tracks** (`LARND_CHAIN_FREEZE_DRIFT=1
    LARND_FREEZE_STRAIGHT_CL=0.5`). Tracks whose hit-derived drift modes are *not* significant at
    CL 0.5 (~15% of tracks) keep their drift coordinate fixed at the straight line; all other
    geometry stays free.
- **Why it works:** (§0b, §3.3)
  - Free drift geometry opens a nearly flat direction: charge moved later (geometry + eField)
    against threshold crossing moved earlier (long_diff, lifetime). It moves lifetime by ~10 points
    for ~10 loss units.
  - Frozen drift closes it. A small, accurately straight subset is enough to pin the global
    parameters, while the hit seed fixes the optimization problem (stuck curved tracks) for the rest.

| (% error vs truth) | A_b | E field | tran_diff | long_diff | lifetime |
|---|---|---|---|---|---|
| **R2, pooled n = 6** (samples 1,2,4,5,6,9; τ = 2 406 µs) | +0.17 ± 0.05 | +0.018 ± 0.003 | +1.1 ± 0.6 | **−0.8 ± 1.1** | **+0.9 ± 1.1** |
| per sample, lifetime | | | | | −3.5, +2.2, −0.4, +0.6, +2.1, +4.1 |
| per sample, long_diff | | | | +2.4, −2.9, −3.4, +1.6, −3.7, +1.1 | |
| production on the same 6 samples | +0.10 ± 0.04 | −0.034 ± 0.006 | +1.1 ± 0.4 | +11.9 ± 2.5 | **+6.8 ± 1.1** (+6.5, +5.2, +4.8, +5.1, +6.9, +12.2) |
| R2 at τ = 1 200 µs (samples 1/3/7) | +0.05…+0.27 | ≤ +0.01 | −0.3…+3.9 | −6.4, −0.1, +9.0 | −0.6, −0.7, +4.4 (production +6.4, +3.8, +6.5) |

- **The replication passes every rule stated before the results.**
  - Fresh samples 5/6/9: |lifetime| ≤ 5% and |long_diff| ≤ 8% on 3/3.
  - Pooled lifetime mean within ±3%.
  - τ = 1 200: |lifetime| ≤ 5% on 3/3.
- **Provenance and sequence:**
  - STRFRZ50 was one of two arms (CL 0.9 / 0.5) chosen *before* the first test. The designated arm
    (CL 0.9) failed its rule (over-correction: lifetime −3.7, offset −16…−33 µm).
  - CL 0.5 was then replicated on fresh samples with a new rule written beforehand.
  - Per-sample lifetime SD is 2.6% (frozen-truth: 0.3%). That is the sample-to-sample precision of
    one 40 m track sample.
- Figure: `figs/fig20_recipe_summary.png`.

**One defect found and fixed (7 Oct).** In STRFRZ50T1200D7, one 103-cm track, nearly parallel to
the anode, got a hit seed 1.6 m off: its pixels covered too little of the track and the fit
extrapolated. It never recovered: its charge was simulated outside the detector, so its hits were
simply unexplained. The new guards remove exactly this track and change nothing else in that sample
(skip count 7 → 8, straight calls unchanged).
- The 11–19 tracks per run with 0.5–1.4 cm coefficients are **genuine**: long, strongly curved
  tracks. The seed matches their true shape to 0.1–0.2 mm. Production never captured them (its
  largest coefficient is 0.33 cm), which is exactly the stuck-track problem the seed fixes.

**Not covered yet (unchanged truth leaks; all closure tests share them).**
- Nominal track lines run through the **true endpoints**.
- Hit-to-track association uses the MC event id.
- t0 = 0.
- Selection on true positions.
- The dE/dx prior is fitted to the target file.
- The robustness of R2 to these remains to be measured; the endpoints are the most important, since
  the straight-line freeze is defined relative to them.

## 0b. Working recipe as of 2 October (simulation closure)

**Recipe R1 — drift coordinate from timing, everything else fitted.**
- **Configuration:** the production fit (spline geometry with 6 modes, per-segment dE/dx with the
  production prior and mean constraint w = 2500), except that each track's **drift-direction
  shape is fixed from external information** (on data: hit timing with a known t0; here: the true
  trajectory) and frozen. Pixel-plane geometry, dE/dx and all five calibration parameters fit
  freely.
- **Knobs:** `SCIGEOMTRUTHINIT=drift LARND_CHAIN_FREEZE_DRIFT=1 SCIKMIN=6`.

| parameter (error %, mean ± SE, n = 3) | τ_true = 2 406 µs (samples 1/2/4) | τ_true = 1 200 µs (samples 1/3/7) | production, for comparison |
|---|---|---|---|
| A_b | +0.40 ± 0.05 | +0.28 ± 0.08 | +0.03 (n = 16) |
| E field | 0.00 | 0.00 | −0.03 |
| tran_diff | +1.5 ± 0.3 | +2.0 ± 0.5 | +1.3 |
| long_diff | **−2.0 ± 0.6** | **−1.8 ± 0.6** | +13.9 |
| lifetime | **−5.0 ± 0.2** | **−0.7 ± 1.0** | +8.85 (n = 16) |

(See the 4-point curve below: these two columns differ mainly through the samples, not through τ.)

- **Tolerance:** perturbing the drift shape by 50 µm RMS changes nothing (lifetime +0.35 ± 0.36,
  3/3 within ±1.5).
- **Correction (3 Oct): simple hit timing does NOT reach 50 µm.** Offline, against real targets
  (`scripts/curvek6/offline_test.py`, `diag3.py`, `diag4.py`):
  - a single hit's drift coordinate scatters ~3 mm;
  - the per-pixel charge-weighted mean scatters ~0.9 mm, and the scatter is white (lag-1
    autocorrelation 0.05);
  - with 40–150 pixels per track, a hit-derived drift shape is off by 300–700 µm, worse than the
    straight line (~250 µm) at every mode cap tried.
  - So "drift from timing" on data must come from the full forward fit, not a timing pre-fit.
- **Where the free fit loses (3 Oct, `shape_err.py`, `tail.py`):** production drift shapes are
  *better* than NZ50 for the typical track (median 30–73 µm), but the worst 15% of tracks hold 98–99% of the
  squared error (p95 400–650 µm, up to 8.6 mm). Those tracks are flatter and nearer the anode, and
  carry larger true curvature. Fits with lifetime/long_diff pinned wrong (PINSTRESS) give drift
  shapes of the same quality, so the shape is set by the hits, not by the lifetime value.
- **Lifetime vs true τ (4 points, fig 19).** K6 errors per sample:

| sample | 1 200 µs | 1 800 µs | 2 406 µs | 3 200 µs | Δ(1/τ) [10⁻⁵/µs] |
|---|---|---|---|---|---|
| 1 | −2.3 | −4.1 | −5.2 | −6.6 | +2.0 / +2.4 / +2.3 / +2.2 |
| 2 | | | −5.1 | | +2.2 |
| 3 | −0.8 | −2.2 | | −5.3 | +0.7 / +1.2 / +1.7 |
| 4 | | | −4.6 | | +2.0 |
| 7 | +1.2 | +1.4 | | +5.0 | −1.0 / −0.8 / −1.5 |
| production, mean | +5.6 | +7.9 | +5.5 (n = 3) / +8.85 (n = 16) | +18.6 | ≈ −4 |

  - **The K6 error is a per-sample constant offset in the attenuation rate 1/τ.** A constant
    Δ(1/τ) per sample leaves 21% of the variance (within-sample rms 0.30 vs 1.41 ×10⁻⁵/µs); a
    constant relative error leaves 50%.
  - **This corrects my 2 October reading** ("bias grows with τ: 0 at 1.2 ms, −5% at 2.4 ms"). That
    came from sample composition: the 2.4 ms points used samples 1/2/4, which all sit at about +2.2.
  - Pooled over 5 samples: Δ(1/τ) = +1.3 ± 0.6 ×10⁻⁵/µs, spread 1.4 ×10⁻⁵/µs between samples
    (production −4.5, constant). In lifetime terms the bias is −3% ± 3.4% per sample at 2.4 ms and
    scales ∝ τ.
  - The between-sample spread is about 2× the Laplace statistical width (~0.8 ×10⁻⁵/µs), so it is
    mostly sample noise plus the representation floor (§3.3g). No τ-dependent correction is needed
    beyond quoting the result in 1/τ.
- **Running (3 Oct 21:50):**
  - DRIFTNZ150 1/2/4: tolerance at 150 µm of drift noise.
  - K6FREE 1/2/4: production with 6 modes, **no truth drift**. This is the first truth-free
    candidate aimed at the badly fitted tail.
  - **Readout rules, stated 4 Oct 05:00 before any result.**
    - Reference values: production DSREP 1/2/4 lifetime +6.5 / +5.2 / +4.8; truth-drift K6
      −5.2 / −5.1 / −4.6.
    - K6FREE **success** if lifetime is within ±3 points of K6 on 3/3 and long_diff within ±5%.
      **Partial** if it moves ≥ 4 points from DSREP toward K6 on 3/3. **Null** otherwise.
    - NZ150 **pass** if lifetime is within ±1.5 points of noise-free K6 on 3/3 (the NZ50 rule).

![K6 curve](figs/fig19_k6_curve.png)

**Results 4 Oct (read at iteration 10 000, configs verified by argv/env diff):**

| arm (samples 1/2/4) | lifetime | long_diff | A_b | loss vs production |
|---|---|---|---|---|
| production DSREP | +5.5 ± 0.5 | +10.2 | +0.15 | — |
| K6FREE (6 modes, no truth) | **+13.4 ± 3.3** | +22.5 | −0.06 | −75 ± 25 |
| truth drift K6 (frozen) | −5.0 ± 0.2 | −2.0 | +0.40 | ≈ −1 000 |
| NZ150 (truth drift + 150 µm noise) | −3.9 ± 1.4 | −1.0 | +0.30 | ≈ −450 |

- **K6FREE: null, in fact worse** (paired +7.8 ± 3.6 vs production; long_diff +12.3). More free
  modes give the co-adaptation more room.
- **NZ150: formally FAILS the ±1.5 rule** (paired +1.1 ± 1.3: −1.5 / +2.5 / +2.3). The level is
  no worse than noise-free, so random drift-shape noise up to ~150 µm costs ≤ ~2.5 points.

**The losses separate two problems:**
1. **Optimization.** Production and K6FREE end ~900–1 250 loss units *worse* than the
   truth-drift fits on the same data (3/3 samples). The objective strongly prefers the truth-like
   geometry; the optimizer does not reach it from the straight-line start.
   - Sample 8 shows this directly. **GTRU8** (geometry started at truth, then fitted freely) keeps a
     clean drift shape: mean error 46 µm, no tail (production on the same sample: 168 µm).
   - Its loss equals the frozen-truth fit (8 550 vs 8 563) and is 1 245 below production (9 795).
2. **A flat direction inside the true basin.** GTRU8 and GEOFRZ8 differ by only 13 loss units,
   but by 10.7 points of lifetime (+5.8 vs −4.9) and 12.8 of long_diff. Fixing initialization is
   expected to recover about half of the bias; the rest needs something that pins the
   lifetime–long_diff–geometry direction.

**Resume bug found and fixed (4 Oct).** Nothing restored the fitted geometry on `--resume_from`.
- Every resumed run silently restarted its geometry from the default init.
- Affected: DRIFTTRU1/2/4, whose pixel-plane geometry was reset at iteration 2 200. Their shift
  still replicated 3/7/8, so the effect looks minor. GEOFRZ resumes re-created the identical frozen
  truth geometry.
- Resumes now restore geometry through the track_id-keyed `LARND_CHAIN_INIT_FROM` path.

**What the flat direction looks like (4 Oct, `valley_geom.py`, paired on the same 872 tracks).**
- Freeing the geometry from truth (GTRU8 − GEOFRZ8) moves every track's charge coherently farther
  from the anode: **+28 ± 1 µm** on average, +45 µm at 0–4 cm drift falling to +11 µm at 24–28 cm,
  with no change of track slope (−0.01 mrad).
- That ≈ 0.2-tick coherent shift accompanies lifetime +10.7, long_diff +12.8, A_b −0.35 and
  tran_diff −3.3 points.
- Production (DSREP8 − GEOFRZ8) shows the same pattern, larger: +63 ± 10 µm.
- Consistent with the noise tests: random per-track drift noise (50–150 µm) is harmless, while
  coherent shifts of tens of µm move lifetime by ~10 points.

**HITSEED / GTRU results (5 Oct, samples 1/2/4, configs verified: only the intended knobs differ).**

| arm | lifetime | long_diff | A_b | loss vs DSREP | drift-shape error (mean) | coherent drift offset |
|---|---|---|---|---|---|---|
| DSREP (production) | +5.5 ± 0.5 | +10.2 | +0.15 | — | 131–161 µm | +11 to +42 µm |
| **HITSEED** (truth-free) | +4.2 ± 3.4 | +3.1 | +0.08 | **−1 112 ± 115** | **42–46 µm** | **−6, +1, −3 µm** |
| GTRU (truth start) | +7.0 ± 2.5 | +5.0 | +0.11 | −1 238 ± 108 | 58–64 µm | +40, +42, +41 µm |

- **The optimization problem is solved without truth.** HITSEED recovers ~90% of the loss gap to
  the truth-started fit and removes the tail of stuck tracks. Its drift shapes are better than
  GTRU's.
- **Lifetime is not fixed by a good start.**
  - GTRU vs DSREP: +1.4 ± 2.3. The "initialization matters" rule FAILS; sample 8's 10-point gain
    was specific to that sample (DSREP8 is the outlier at +16.4).
  - HITSEED vs DSREP: −1.4 ± 3.3 (−2.5 / +4.9 / −6.4). This meets neither its success nor its null
    rule: no consistent lifetime effect.
- **The free fit's optimum is displaced from truth.** Across 25 free-geometry fits, every one started
  from the straight line or from truth (including GTRU, which starts *exactly* at truth) ends with
  the charge +6 to +59 µm farther from the anode. HITSEED is the only exception.
  - Lifetime correlates with the offset only moderately (r = +0.4 to +0.5;
    `coherent_vs_lifetime.py`).
  - A long_diff pin does not remove the shift: LDPIN offsets +19, +19, +39 µm.
- **Open hypothesis being tested now** (`E_truthdiag/timing/truthdiag_timing.py`, job 39911677):
  - The probabilistic fit model and the stochastic target generator may disagree on hit timing
    by ~0.1–0.3 tick at truth (1 tick = 159 µm of drift).
  - The test compares per pixel and per hit index, binned by drift distance. A coherent mismatch
    would explain an optimum displaced by tens of µm, and fixing it would be a model fix, not a
    fit trick.
- **Timing closure at truth (5 Oct, `truthdiag_timing.py`, 20 batches × 4 noise realisations, 136k hits).**
  For every observed hit, the probabilistic model's CDF of that pixel's k-th hit tick is evaluated at
  the observed tick (mid-PIT; calibrated means uniform, mean 0.5).
  - Overall well calibrated: first hits 0.4986 ± 0.0011, all deciles 0.10 ± 0.002.
  - **A small, significant drift-dependent trend.** PIT is 0.493 at 0–4 cm (2.6σ) and 0.515 at
    28–32 cm (3.2σ): the target's hits come slightly early near the anode and late near the cathode.
    That is ≈ −0.1 → +0.3 tick (−20 → +45 µm), the size of the free fits' coherent shifts.
  - Free fits also lower eField by 0.04–0.06% (all of them), while frozen-truth fits keep it at 0.00%.
    That drift-speed change would add ~+0.4 tick at the cathode, absorbing the stretch.
  - (Comparing raw means or medians was misleading: the per-pixel hit-time distributions are broad,
    SD ~8 ticks, and skewed. The PIT is the right test.)
- **Launched 5 Oct 04:05 — PSTGTRU / PSTDTRU 1/2/4 (jobs 39911902–07).** Both use *self-consistent*
  targets (`SCIPROBTGT=1`: drawn from the probabilistic fit model, so no timing mismatch by
  construction). PSTGTRU starts from truth with free geometry; PSTDTRU freezes the drift at truth.
  - **Rule:** the mismatch drives the displaced optimum if the free-minus-frozen lifetime gap is
    within ±3 points on 3/3 (≈ +15 with stochastic targets) **and** PSTGTRU's coherent offset is
    < 10 µm. Otherwise the displacement is intrinsic to free geometry.
- **PST result (5 Oct 20:10, self-consistent targets; configs verified): the rule FAILS on both criteria.**
  - Lifetime gap, free minus frozen: **+7.5 ± 3.5** (+2.8 / +14.4 / +5.1), vs +14.7 ± 2.5 with
    stochastic targets (GTRU − DRIFTTRU).
  - long_diff gap: +6.6 vs +18.4.
  - Free-fit coherent offset: **+31 / +30 / +24 µm** (stochastic: +40 / +42 / +41), with the same
    profile (largest at 4–8 cm, ~0 at the cathode).
  - eField still drops (−0.038% vs −0.054%).
  - ⇒ The FEE timing mismatch is a **minor contributor** (about ¼ of the shift, ½ of the lifetime
    gap). The displacement persists with a perfectly consistent electronics model.
- **New suspect: the loss itself.** Each observed hit is scored by a log-sum over ticks within ±2 of
  the observed tick of [log P(t′) + log K(t_obs − t′) + Gaussian charge term at q_pred(t′)], with
  K Gaussian, σ = 1 tick.
  - That is a misspecified likelihood. For skewed per-pixel hit-time distributions (mean and median
    differ by ~1 tick), its optimum can sit a fraction of a tick from truth, and it does so with
    any targets.
- **Launched 5 Oct 20:20 — PSTGTRUTW0 / GTRUTW0 1/2/4 (jobs 39975174–79).** Truth start, free
  geometry, **exact point-wise likelihood** (`--loss_time_window 0`); self-consistent and stochastic
  targets respectively.
  - **Rule:** the smearing is the main cause if PSTGTRUTW0's coherent offset is < 10 µm on 3/3.
  - It is a production-relevant fix if GTRUTW0's offset is < 15 µm on 3/3 **and** lifetime is
    within ±5% of truth on 3/3.
- **Likelihood shift scan (5 Oct 20:15, `truthdiag_shift.py`): the smearing hypothesis is REFUTED.**
  On true tracks at true parameters, δ* is the global timing shift maximising each likelihood form
  (ticks; 1 tick = 159 µm):

  | hits | exact (window 0) | production (±2 smeared) |
  |---|---|---|
  | self-consistent | +0.002 | −0.009 |
  | stochastic | +0.082 | +0.007 |

  - The production loss is timing-unbiased at truth to ≤ 0.02 tick (≤ 4 µm), far from the
    ~0.2 tick the free fits shift.
  - The smearing even absorbs the FEE mismatch that the exact form picks up.
  - The TW0 fits were therefore **cancelled** (jobs 39975174–79, ~50 min in).
  - So on true tracks the likelihood has no timing pull. The displacement comes from the fit's track
    representation or priors (the spline warp *replaces* segment positions by nominal line + K
    modes at 2-cm nodes).
- **Mechanism (consistent with all numbers so far).** The data pin a global timing offset very
  precisely (±0.01 tick from 70k hits), so the free fits' +0.2-tick geometry shift must be traded
  against parameters that move threshold-crossing times:
  - "charge arrives later": geometry shift largest near the anode, plus eField −0.05% acting at long
    drift;
  - "crossing comes earlier": long_diff ↑ and lifetime ↑.
  - Frozen drift geometry breaks this trade, which is why frozen fits have tiny lifetime scatter
    (±0.3%).
- **Launched 5 Oct ~21:00 — STRFRZ90 / STRFRZ50 1/2/4 (jobs 39976482–87): a truth-free attempt to
  pin the valley.**
  - HITSEED, plus the drift (e2) frozen at the straight line **only for tracks the hits call
    straight** (`LARND_FREEZE_STRAIGHT_CL` = 0.9 / 0.5; ~35% / ~15% of drift-frame tracks).
  - Offline, the frozen tracks' straight-line error is mean 148 / 142 µm (p95 ~440), i.e. NZ150-like.
  - Curved tracks stay free from the hit seed. Paired with HITSEED 1/2/4.
  - **Rule:** valley pinned if STRFRZ90 lifetime SD over the 3 samples ≤ 2 points (HITSEED: 5.8)
    **and** coherent offset < 15 µm on 3/3 **and** |lifetime| ≤ 8% on 3/3.
- **STRFRZ result (6 Oct 11:15; configs verified: only the freeze knobs differ from HITSEED).**

  | arm (1/2/4) | lifetime | long_diff | lifetime SD | coherent offset | A_b | tran_diff | loss vs HITSEED |
  |---|---|---|---|---|---|---|---|
  | HITSEED | +4.2 ± 3.4 (+4.0/+10.0/−1.6) | +3.1 | 5.8 | −6/+1/−3 µm | +0.08 | +1.3 | — |
  | STRFRZ90 | −3.7 ± 1.9 (−5.5/−5.7/0.0) | −8.3 | 3.2 | −33/−16/−18 µm | +0.25 | +1.8 | +326 |
  | **STRFRZ50** | **−0.6 ± 1.6** (−3.5/+2.2/−0.4) | **−1.3 ± 1.9** | 2.8 | −7/−9/−9 µm | +0.22 | +1.3 | +83 |

  - **STRFRZ90 FAILS the stated rule** (SD 3.2 > 2; offsets beyond 15 µm on 3/3); |lifetime| ≤ 8%
    holds. Freezing a third of the tracks at the straight line over-corrects (charge pulled toward
    the anode).
  - **STRFRZ50 (not the designated arm, n = 3)** is the best truth-free configuration so far: all
    five parameters within ~3.5% on every sample. Its SD (2.8) is still above 2, so it needs
    replication.
- **Launched 6 Oct 11:30 — STRFRZ50 replication (jobs 40031258–63).** Fresh samples 5/6/9 at
  τ = 2 406 µs, plus τ = 1 200 µs on samples 1/3/7.
  - **Rule:** replicates if on 5/6/9 |lifetime| ≤ 5% and |long_diff| ≤ 8% on 3/3, and the pooled
    n = 6 lifetime mean is within ±3%.
  - Robustness: at τ = 1 200, |lifetime| ≤ 5% on 3/3.
- The long_diff-pin follow-up (`launch_next_ldpin.sh`) is **on hold**: its stated launch condition
  (HITSEED passes on lifetime) was not met.

**A classic hit-level lifetime estimate is not a shortcut (4 Oct, `tau_std.py`).** dQ/dx of 2 cm
track pieces vs drift time, truncated mean, exponential slope; no forward model.
- Result on samples 8/3/7 (truth 2 406 µs): **−8.1 ± 3.5, −24.2 ± 2.2, −14.7 ± 3.0%**.
- The binned charge falls ~6% below true attenuation at the cathode end. That is threshold loss
  growing with diffusion, and it is as large as the whole lifetime signal (8% anode→cathode).
- So any lifetime measurement here needs the diffusion/threshold model at the ~1% charge level,
  which is the same entanglement the joint fit faces.

**Launched 4 Oct 12:50 — HITSEED 1/2/4 and GTRU 1/2/4.**
- **HITSEED** is production (3 modes, free geometry) started from a hit-derived geometry, gated per
  track. e2 is used if a χ² test against "straight" passes at 99.9%; e1 is used if the shape RMS
  exceeds 500 µm. It is truth-free.
  - Offline on sample 8: curved tracks start at 451 µm (e2) / 241 µm (e1) instead of 707 / 732;
    near-straight tracks are unchanged.
- **GTRU** is production started from the true geometry: the ceiling for any initialization fix.
- **Readout rules (stated before results):**
  - GTRU: "initialization matters" if lifetime drops ≥ 4 points vs DSREP on 3/3.
  - HITSEED **success**: recovers ≥ half of the GTRU − DSREP lifetime gain on 3/3 and lowers the
    loss vs DSREP.
  - HITSEED **partial**: drops ≥ 4 points on ≥ 2/3. **Null**: within ±2 of DSREP on 3/3.

**Recipe R0 — fallback with no lifetime from the fit.** Fit A_b, E field and tran_diff with
lifetime and long_diff fixed externally. Stress test with them deliberately wrong (+5%, +10%):
- A_b −0.01 / +0.12 / +0.04, E field −0.02 / −0.02 / −0.05: **robust**;
- tran_diff +3.9 / +0.7 / +2.6: within ~±4%.

**Not yet covered by either recipe:**
- truth leaks in the inputs: geometry initialised from true endpoints, MC track identity,
  selection on true positions, dE/dx prior fitted to the target file (§4);
- unmodelled detector effects;
- a real t0 source (anode–cathode crossers or light).

R1 needs, on data, the drift coordinate from timing. That is the single most important
requirement this review has established.

## 1. What was checked, and how

| Workstream | What | Method |
|---|---|---|
| **A. Setup audit** | truth leakage, model-vs-generator approximations, real-data gaps | code reading with file:line citations |
| **B. Ensemble re-analysis** | DSREP1–16, Arm B (TRU1–3), BOTHTRU / DXTRU / DXFRZ | 40 checkpoints recomputed from raw; independence, provenance and convergence checked |
| **C. Curve & prior scan** | CURVE (24 runs), NARR/WIDE ladder incl. fresh seeds, LDPIN | 66 checkpoints recomputed; leave-one-sample-out validation; config diffs |
| **D. Claims audit** | 28 bugs × 30 standing claims, prior session transcript | git log, docs, provenance blocks, transcript mining → `claims_audit.md` |
| **E. NEW: model at truth** | charge/hit closure vs drift; NLL profiles in lifetime, long_diff and A_b×lifetime | **fresh GPU runs**: 100 batches × 400 cm, production selection, 10 target-noise seeds, noise ON/OFF |
| **Own checks** | re-derived the DSREP numbers; lifetime vs charge-scale regression; lever-arm arithmetic | directly from pickles |

**Every checkpoint in the campaign has `git_dirty: True`.** No run was made from a clean tree, so
exact code state is unrecoverable. For the key comparisons (DSREP1–8 vs 9–16 on different commits:
+8.57 vs +9.13%, p = 0.82) the drift demonstrably does not matter, but it cannot be excluded in
general.

---

## 2. What works

### 2.1 The software and the scale
A 5-parameter + ~9000-nuisance joint fit at 400 m of track per run converges reproducibly in
~13 h on one A100. A_b and E field settle to peak-to-peak ≤0.05% in every one of 39 fitted arms.
The engineering is solid: the GPU preflight, exit-status propagation and the throttled submitter are
all in place.

### 2.2 A_b, E field, transverse diffusion (verified, n = 16 independent samples)

![per-parameter errors](figs/fig5_dsrep_params.png)

| parameter | mean error | mean \|error\| | verdict |
|---|---|---|---|
| A_b | +0.03 ± 0.03% | 0.11% | recovered |
| E field | −0.033 ± 0.004% | 0.03% | recovered — tiny, but **significant** (8σ) bias |
| tran_diff | +1.29 ± 0.37% | 1.6% | recovered at the few-% level |
| long_diff | **+13.9 ± 2.4%** | 14.6% | **fails** |
| lifetime | **+8.85 ± 1.18%** | 8.9% | **fails** |

**Real-data caveat on "A_b solved":** in simulation A_b is the *only* free charge-scale knob.
Gain, W-value, kb and MeV→electrons are fixed at truth. On data, A_b absorbs every
charge-scale error (ADC gain, pedestal, the recombination model's form). What is really
demonstrated is "the overall charge scale is recovered to 0.1% when everything else about the charge
chain is known exactly".

### 2.3 Geometry
Position residuals are ~185 µm for production and ~150 µm with dE/dx frozen mid-fit. The 51 µm
"floor" is a separate, noiseless, position-only configuration and must not be compared to these.
The one robust September positive holds: freezing the dE/dx block mid-fit costs no truth and gives
−31 µm on 3/3 seeds. **Caveat:** all geometry initialisation comes from truth (§4), so these numbers
say nothing yet about starting from a reconstruction.

---

## 3. What does not work: lifetime and longitudinal diffusion

### 3.1 The lifetime bias is real (CONFIRMED)

![lifetime by arm](figs/fig3_lifetime_by_arm.png)

- **DSREP1–16:** +8.85 ± 1.18% (median +9.0%, 15/16 positive, t = 7.5). I re-derived this
  independently: all 16 postdate the `data_seed` fix, and their true-position hashes are all
  distinct, so these really are 16 different samples.
- **Arm B (other truth points):**

  | τ_true | bias |
  |---|---|
  | 1160 µs | +2.9 ± 1.4% |
  | 2392 µs | +11.6 ± 2.5% |
  | 4518 µs | +6.0 ± 3.3% |

  Universal in sign; the magnitude depends on the truth point.
- **Convergence:** not everything is "≤0.5% settled" as claimed. Lifetime, tran_diff and long_diff
  exceed 0.5% peak-to-peak in ~8/39 arms (worst: TRU1D7 long_diff 1.0%). The differences are small
  compared with the bias, so the bias is not a convergence artefact.

![DSREP traces](figs/fig4_dsrep_traces.png)

### 3.2 NEW — the forward model at truth is essentially unbiased

A fresh GPU study (workstream E):
- **Setup:** the production sample; the guess simulated from the *same* true segments at true
  parameters; 10 independent target-noise realisations.
- **Self-check:** the drift-slope estimator recovers injected lifetime changes to ~4% of the
  analytic value.

![closure vs drift](figs/fig9_truth_closure_vs_drift.png)

| comparison at truth | total obs/exp | drift slope | lifetime equivalent |
|---|---|---|---|
| charge, noise-ON target vs production model | 1.00337 ± 0.00014 | −1.4 ± 0.3 ×10⁻⁵/µs | +3.5 ± 0.8% |
| hits, same | 1.0044 ± 0.0002 | −1.2 ± 0.4 ×10⁻⁵/µs | +3.1% |
| charge, noise-OFF target vs ~noiseless model | 1.0009 | +0.3 ± 0.1 ×10⁻⁵/µs | ≈ 0 |

- The small noise-ON excess is **noise-induced extra hits** that the model under-predicts. The model
  uses only reset noise in its threshold erf (`fee_jax.py:95`). **Per-hit charge is unbiased**
  (+0.03%).
- The excess is not exponential. It rises over the first ~50 µs, then stays flat, so it is not an
  attenuation-rate error.
- **What the fit actually sees is the likelihood minimum:**
  - lifetime with A_b at truth: **+1.47 ± 0.15%**;
  - **lifetime profiled over A_b: +0.51 ± 0.27%** (fig 10);
  - long_diff: −1.0 ± 0.2%.
- Seed-to-seed scatter from electronics noise alone is ±0.9% in lifetime.

![A_b-lifetime valley at truth](figs/fig10_truth_ab_lifetime_valley.png)

**Conclusion:** the model/target mismatch explains at most ~1.5 points of the +8.85, and ~0.5 once
A_b is free. The September hypothesis that "the forward model loses 0.8% too little charge over the
drift" is **refuted**. The measured slope is 3× smaller and not exponential. The long_diff-mechanism
idea is also not supported: at truth, the model does not want a larger long_diff.

### 3.3 Where the bias actually comes from: the charge-scale soft plane + nuisance estimation

![bias anatomy](figs/fig2_bias_anatomy.png)

- **At truth**, the NLL valley has dτ/τ = **−20.3 × dA_b/A_b** (correlation −0.84).
- **Across the 16 production fits**, lifetime error vs A_b error has r = **−0.81**, slope ≈ −23 to
  −28. A least-squares fit gives:

  > lifetime_err = 6.7(±1.2) − 22.6(±4.9)·A_b_err + 0.20(±0.07)·long_diff_err &nbsp;&nbsp;(R² = 0.79, residual rms 2.3%)

  The replica-to-replica lifetime scatter is almost entirely motion along the
  (A_b, lifetime, long_diff) soft plane, as predicted by the truth-point valley. A **+0.1% A_b wobble
  is a ~2.5% lifetime error**.
- **The offset (+6.7% at zero A_b and long_diff error) is produced by the nuisance blocks:**

  | configuration | lifetime |
  |---|---|
  | model at truth | +0.5 … +1.5% |
  | fit, both nuisances pinned at truth | **−3.7 ± 0.4%** |
  | fit, dE/dx pinned at truth | +3.0 ± 2.6% |
  | fit, dE/dx frozen at its own fit | +10.6 ± 1.1% |
  | production | +8.85 ± 1.18% |

  - The −3.7% with nuisances *at truth* differs from the +0.5% of the truth-segmentation model. The
    only difference is the **fit's own parameterisation**: the linear-guess resegmentation (dx ≈ 1 cm)
    and the spline geometry, whose best representation of truth still sits 173 µm off. That is a
    **representation bias** of about −4 to −5 points.
  - Freeing geometry on top of true dE/dx moves lifetime from −3.7 to +3.0 (**≈ +6.7 points**).
    Freeing dE/dx *in the production configuration* adds another ≈ +5.9 (+3.0 → +8.85). ~~I first
    attributed the latter to generic Neyman–Scott bias from point-fitting the dE/dx nuisances.~~ The
    clean test in §3.3b refutes that generic explanation. The production prior-scale ladder
    (+20% → −2% as the prior is tightened) shows the production *prior configuration* matters, not
    nuisance fitting per se.

### 3.3b NEW (24 Sep) — does integrating out dE/dx fix the lifetime? The Laplace test

![Laplace test](figs/fig12_laplace_test.png)

**Setup** (`scripts/laplace_test/`):
- All 100 production batches (95 used; 5 excluded where the inner fit had not reached a clean
  minimum). Seed-0 truth, the production noise-ON target.
- Everything at truth except per-segment dE/dx (~110 target-file segments per batch).
- Prior: a split-t fitted in segment-length bins over the whole 3M-segment file, i.e. a correct
  population prior, one count per segment, curvature smoothed at the mode. No mean constraint.
- For every (lifetime, A_b) grid point and batch, dE/dx is Newton-fitted to convergence
  (median decrement 4×10⁻⁵ nats).
- The dE/dx Hessian is recovered exactly from ~15 graph-coloured Hessian-vector products; it
  matches the dense Hessian to 6×10⁻⁶.
- Three objectives, all with A_b profiled:
  - **reference:** true dE/dx;
  - **MAP:** dE/dx fitted to its best point, as in production;
  - **Laplace:** MAP + ½ log det H, i.e. dE/dx integrated out.

| objective | lifetime minimum | 1σ statistical width | A_b at minimum |
|---|---|---|---|
| reference | −0.5 ± 0.7% | 0.75% | +0.10% |
| MAP | −2.6 ± 2.2% | 1.85% | +0.47% |
| Laplace | −2.9 ± 2.2% | 1.87% | +0.25% |

Paired differences (same data, jackknife): MAP − reference **−2.1 ± 2.2**; Laplace − MAP
**−0.2 ± 0.3**.

**What it shows:**
1. **Fitting dE/dx per segment is not, by itself, what biases lifetime.** Given a correct prior and
   true geometry, the shift is −2 ± 2 points, not significant and of the opposite sign to production.
2. **Integrating dE/dx out doesn't move lifetime.** It does halve the charge-scale bias from mode
   shrinkage: A_b goes +0.47% → +0.25%, against a reference of +0.10%.
3. **The real cost of free dE/dx is precision.** The lifetime profile is 2.5× wider, because
   per-segment dE/dx absorbs most of the drift-dependent charge information. This is the price of
   keeping the joint dE/dx fit.

**Caveats:**
- One truth point, one track sample, one noise realisation.
- Target-file segmentation (~3.3 cm), not the production 1 cm guess grid.
- The prior is correct by construction, which the production one is not quite.
- The smoothing width at the prior mode (0.3× the narrower scale) is a choice.
- Only A_b is profiled; E field, diffusions and geometry stay at truth.

### 3.3c NEW (25 Sep) — the production dE/dx configuration is not the cause

![production dE/dx config](figs/fig13_production_dedx_config.png)

**Setup** (`scripts/ladder_test/`): the same harness and data as §3.3b, but the nuisance is now
production-like:
- ~1 cm pieces laid along the *true* trajectory, like the production guess grid but with exact
  geometry;
- the exact production prior: refit split-t × length × 1.0088, no Jacobian, 8.5 MeV/cm barrier;
- the production mean constraint (w = 2500, target 1.887).

It was compared with rung 0 on the **same 92 batches and the same 21 grid points**, with A_b
profiled.

| arm | lifetime at minimum | 1σ width |
|---|---|---|
| true dE/dx (target segments) | −0.5 ± 0.8% | 0.76% |
| true dE/dx (1 cm pieces) | −0.5 ± 0.8% | 0.76% |
| rung 0: fitted dE/dx, clean setup | −1.6 ± 2.6% | 2.09% |
| **L3: fitted dE/dx, full production configuration** | **−2.4 ± 2.1%** | **1.06%** |

- **Production configuration − clean setup: −0.8 ± 1.1 points (paired).** With exact geometry,
  the complete production dE/dx treatment gives lifetime within ~2% of truth; it cannot produce
  the +8.85%.
- The 1 cm averaging of true dE/dx costs nothing (−0.01 ± 0.19 points).
- The mean constraint **halves the lifetime width** (2.09% → 1.06%): pinning the dE/dx scale
  recovers most of the information that free per-segment dE/dx costs.
- Fitted dE/dx scale is unbiased in every rung (within ±0.1%).
- **Consequence:** the production bias arises on the geometry side. Either fitting geometry
  itself does it, or its interaction with free dE/dx. The existing production arms agree: DXTRU
  (geometry fitted, dE/dx at truth) +3.0%; BOTHTRU (both pinned) −3.7%; DSREP (both fitted)
  +8.85%.
- A 17-batch interim read of this ladder (+3.8 ± 3.1) was too noisy to interpret, and I didn't
  use it.

### 3.3d NEW (27 Sep) — the bias is a geometry × dE/dx interaction

![production 2x2](figs/fig14_production_2x2.png)

**The run:** GEOFRZ3/7/8 are production fits with geometry frozen at its least-squares spline
representation of truth and dE/dx fitted in production config.
- **Config check:** the argv/env diff against BOTHTRU3 shows only the dE/dx freeze/init.
- **Resume:** the fits were resumed after a cache mis-fix, with histories verified continuous
  across the resume.
- **Readout:** all arms are read at iteration 10 000; every arm settled to peak-to-peak ≤ 0.6%.

Together with the existing arms this completes a paired 2×2 on samples 3/7/8.

Lifetime error (%), mean of 3 samples:

| | dE/dx at truth | dE/dx fitted |
|---|---|---|
| **geometry at truth** | −3.7 (BOTHTRU) | −1.5 (GEOFRZ) |
| **geometry fitted** | +3.0 (DXTRU) | **+13.4** (DSREP) |

Paired effects, n = 3:

| effect | lifetime | long_diff |
|---|---|---|
| fitting dE/dx, geometry at truth | +2.3 ± 1.9 (n.s.) | +4.3 ± 1.6 |
| fitting dE/dx, geometry fitted | **+10.4 ± 3.3** (3/3) | **+14.9 ± 1.2** (3/3) |
| **interaction** (difference of the two) | **+8.1 ± 2.2** (3/3: +3.7, +10.5, +10.1) | **+10.6 ± 1.6** (3/3) |
| fitting geometry, dE/dx at truth | +6.7 | +15.9 |

**Reading** (the pre-stated readout rule applies):
- Fitting dE/dx is harmless when geometry is right. This agrees with the independent harness
  result (§3.3c: −0.8 ± 1.1).
- Fitting geometry costs ~+7 points on its own.
- Fitting **both** adds another **~+8 points beyond the sum**: the two nuisance blocks co-adapt.
  A geometry error can be compensated by a dE/dx pattern, and together they mimic a
  drift-dependent charge change, which the calibration absorbs as lifetime (and long_diff).
- The A_b interaction (−0.26 ± 0.07, 3/3) is the charge-scale leg of the same valley.
- This is why every dE/dx-only and optimiser-only intervention was a null: the bias lives in the
  *joint* nuisance space.

**What to test next:** which geometry degrees of freedom carry the interaction. The prime suspect
is displacement along the drift direction, since moving a track in drift changes its attenuation
exactly like lifetime. Concretely, a production fit with the drift component of the spline frozen
(or strongly penalised) and transverse geometry free, on samples 3/7/8, compared against DSREP.
If the interaction disappears, that is the handle for real data as well: the drift coordinate
must come from t0/timing, not from the nuisance fit.

**Launched 27 Sep 13:49 — DRIFTFRZ3/7/8**:
- the exact DSREP production configuration plus `LARND_CHAIN_FREEZE_DRIFT=1`: a hard freeze of the
  e2 (drift-carrying) spline coefficients, so the geometry can move only within the pixel plane;
- the log shows the freeze active on every track.

**Readout rule, stated before the result** (paired against DSREP on the same samples):
- **The interaction goes through drift-direction geometry** if lifetime falls from DSREP's +13.4%
  to the no-interaction level: at or below DXTRU's +3.0%, i.e. a paired drop ≥ 8 points on all
  three samples. The long_diff excess (+21% in DSREP) should collapse the same way.
- **The drift direction is not the channel** if lifetime stays within ~3 points of DSREP. The
  pixel-plane geometry freedom would then carry it (for example via charge sharing between pixels
  and diffusion), and the next test freezes e1 instead.
- **Anything in between** means a partial channel.

**Result (28 Sep): outside both pre-registered outcomes — the bias got worse.**

![drift freeze](figs/fig15_driftfrz.png)

| samples 3/7/8, mean | lifetime | long_diff | position |
|---|---|---|---|
| DSREP (production) | +13.4% | +21% | 188 µm |
| **DRIFTFRZ** | **+22.1%** | **+45%** | **683 µm** |

Paired against DSREP:
- lifetime **+8.7 ± 6.4** (2/3 samples up);
- long_diff **+24 ± 5** (3/3);
- position **+495 µm**;
- loss **+4 700** (3/3).

Every arm settled to peak-to-peak ≤ 0.43%.

**Reading:** the test was flawed as designed. Freezing the e2 coefficients at their starting value
(0 = the straight line through the true endpoints) removes the fit's ability to *correct* the
starting line's drift-direction error; it does not remove the error. Every track's true curvature
along the drift direction stays unrepresented. The fit then absorbs that drift-direction mismatch
into long_diff (+24) and lifetime (+9).

So the arm confirms strong coupling between drift-direction geometry error and (long_diff,
lifetime). It cannot separate "freedom along drift" from "error along drift".

**The correct version:** fix the drift-direction coefficients at their **truth** projection, which
is what hit timing would give on data, and leave the pixel-plane (e1) geometry to fit from the
straight line.
- Lifetime back near GEOFRZ (≈ −1.5%) would show the whole co-adaptation runs through the drift
  direction. The handle for data would then be to take the drift geometry from hit timing rather
  than fitting it.
- Lifetime near DSREP (≈ +13%) would place it in the pixel plane.

**Launched 28 Sep 05:44 — DRIFTTRU3/7/8:**
- DSREP production configuration, plus `LARND_INIT_FROM_TRUTH=drift` (e2 = drift-carrying modes
  seeded at their truth projection, e1 at the straight line) and `LARND_CHAIN_FREEZE_DRIFT=1`;
- the logs show seeding and freeze on every track.

**Readout rule, stated before the result** (paired against DSREP):
- **The co-adaptation runs through the drift coordinate** if lifetime drops ≥ 8 points on 3/3
  samples, towards GEOFRZ's −1.5%.
- **It lives in the pixel-plane geometry** if lifetime stays within ~3 points of DSREP.
- **Anything in between** means both channels contribute.

**Result (28 Sep): the drift channel, decisively.**

![drift at truth](figs/fig16_drifttru.png)

| samples 3/7/8, mean | lifetime | long_diff | position |
|---|---|---|---|
| DSREP (production) | +13.4% | +21.1% | 188 µm |
| **DRIFTTRU** (drift at truth, rest fitted) | **−2.3%** | **−4.2%** | **104 µm** |
| GEOFRZ (whole geometry at truth, dE/dx fitted) | −1.5% | −5.5% | 172 µm |

Paired effects:
- **DRIFTTRU − DSREP:** lifetime **−15.6 ± 2.6** (3/3: −12.5, −13.7, −20.8), long_diff
  −25 ± 8 (3/3), position −84 µm, loss −820 (a better fit, 3/3).
- **DRIFTTRU − GEOFRZ:** lifetime −0.8 ± 0.7.
- Every arm settled to peak-to-peak ≤ 0.23%.

**Reading:** the whole geometry × dE/dx interaction, *and* the geometry-fit bias, runs through
the drift coordinate. Fix the drift coordinate correctly and let everything else fit freely, and
lifetime comes back to within ~2% of truth. That is as good as knowing the full geometry.

**Replication (29 Sep, fresh samples 1/2/4; n = 6): the shift replicates, the "back to truth" does not.**

![n=6](figs/fig17_drifttru_n6.png)

| | lifetime | long_diff |
|---|---|---|
| paired shift, replication 1/2/4 | **−13.3 ± 0.4** (−12.9, −12.9, −14.1) | −23.6 ± 0.7 |
| paired shift, all six | **−14.5 ± 1.3** (6/6) | −24.5 ± 3.7 (6/6) |
| level over six, production (DSREP) | +9.4 ± 2.0% | +15.6 ± 4.3% |
| level over six, drift at truth (DRIFTTRU) | **−5.0 ± 1.5%** | −8.8 ± 2.3% |

- **⚠ Correction:** "lifetime back to within ~2% of truth" held on samples 3/7/8 only. On 1/2/4,
  production sat lower (+5.5%), and the same ~−14-point shift overshoots to −6.3 / −7.7 / −9.3%.
  The drift coordinate carries a **constant ~−14-point component**. Fixing it cuts the mean
  |error| from 9.4% to 5.3%, but leaves lifetime ~5% low.
- On 1/2/4 the shift appears even where position and loss barely change (loss −48 ± 28,
  position +4 µm). So the fitted drift geometry differs *systematically* from the truth
  projection, in a way worth a constant 14 points of lifetime. It is not simply a worse fit.
- **Leading hypothesis for the −5% residual:** the "truth" drift coordinate used here is the
  3-mode spline projection of truth, not truth itself. BOTHTRU (everything at that projection)
  also sits low, at −3.7%.
- **Test launched 29 Sep 12:45:** DRIFTTRUK6 1/2/4, the same with 6 modes (`SCIKMIN=6`).
- **Readout rule, stated before the result:** lifetime moving ≥ 3 points towards zero on 3/3
  means the residual is the drift representation. On data, the drift coordinate would then have
  to come from per-hit timing, not a smooth fitted track shape. Moving < 1.5 means another
  source. (Confound: 6 modes also gives the pixel-plane geometry more freedom, previously shown
  harmless.)

**Consequence for real data:** the lifetime measurement stands or falls on the **drift
coordinate**, which is exactly the quantity cosmic data does not hand you without an event t0.
The fit must take drift positions from hit timing with a known t0, e.g. anode–cathode crossers or
the light system. It must not float them as nuisances. The tolerance on that drift information
(how many µm or ns) is the next thing to measure.

**6-mode result (30 Sep, samples 1/2/4): partial — the in-between outcome of the stated rule.**

| samples 1/2/4 | lifetime | long_diff | A_b | position |
|---|---|---|---|---|
| DSREP (production) | +5.5% | +10.2% | +0.15% | 167 µm |
| DRIFTTRU (3 modes) | −7.8% | −13.4% | +0.41% | 171 µm |
| **DRIFTTRUK6 (6 modes)** | **−5.0%** (spread 0.3) | **−2.0%** | +0.40% | **70 µm** |

- 6 vs 3 modes: lifetime **+2.8 ± 1.0** (+1.2, +2.6, +4.7). That is short of the ≥ 3 on 3/3
  "representation" criterion, but above 1.5.
- long_diff +11.4 ± 1.9 (3/3), now nearly unbiased; position −102 µm; loss −1 011 (3/3).
- **The finer drift representation fixes long_diff and the geometry, but only part of lifetime.**
- Lifetime now sits at a very reproducible **−5.0% (sample spread 0.3%)**, together with A_b
  +0.4%. Along the measured valley (dτ/τ ≈ −20 dA_b/A_b) that is one charge-scale offset.
- Offline diagnostic: in every fitted-dE/dx arm the dE/dx scale comes out 0.1–0.3% low and A_b
  0.3–0.5% high. With dE/dx pinned at truth (BOTHTRU3), A_b is still +0.2% and lifetime −3.8%.
  So the remainder is a mix of the dE/dx scale and the geometry-representation floor, not one
  clean cause.

### 3.3e Practical track (launched 29 Sep): a 3-parameter calibration with external lifetime

Resolving the lifetime mechanism may take several more rounds. So in parallel I launched the fit
a first data calibration would actually use: A_b, E field and tran_diff fitted (plus geometry and
dE/dx), with **lifetime and long_diff held fixed at external values**. On data, lifetime would come
from a standard charge-vs-drift measurement or a purity monitor.

The test is deliberately pessimistic: PINSTRESS3/7/8 hold them at **wrong** values (lifetime +5%,
long_diff +10%).

**Readout rule, stated before the result:** PASS if A_b stays within ±0.5%, E field within ±0.1%
and tran_diff within ±3% on 3/3 samples. The shifts also give the systematic coefficients to quote
for an external-lifetime error.

**Practical-track result (30 Sep): a narrow FAIL of the stated rule.**

| PINSTRESS 3/7/8 (lifetime +5%, long_diff +10% held wrong) | A_b | E field | tran_diff |
|---|---|---|---|
| per sample | −0.01, +0.12, +0.04 | −0.02, −0.02, −0.05 | +3.9, +0.7, +2.6 |
| verdict (±0.5 / ±0.1 / ±3) | PASS | PASS | **FAIL (1/3)** |
| shift vs DSREP | +0.12 ± 0.07 | +0.005 ± 0.005 | +1.7 ± 1.5 |

- A_b and E field are robust to realistic external-lifetime and long_diff errors.
- tran_diff mostly holds, but reached 3.9% on one sample; production fits already span −1.7 to
  +4.0% on it.

### 3.3f Why are lifetime and the drift geometry coupled? (30 Sep)

![drift residuals](figs/fig18_drift_residuals.png)

I rebuilt fitted-minus-true drift displacement (perpendicular to the track, signed as a change of
drift distance) for ~21 000 tracks across 21 fits (`scripts/geofrz/drift_residuals.py`).

- **No direct degeneracy.** The systematic drift displacement is ≤ 35 µm at every drift distance
  in every configuration. Production pooled over 6 samples: +30 µm, trend −0.8 ± 0.5 µm/cm within
  tracks. Faking the lifetime bias through attenuation alone would take ~3 mm at the far end,
  ~100× more.
- **The random drift error grows with drift distance** in the fitted configurations: production
  ~100 → ~150 µm; local drift-slope errors 300–450 µm/cm.
- **Lifetime moves with long_diff:** r = +0.92 over 21 fits (≈ 0.5 %/%), +0.59 within the 6
  production fits.
- **Drift-geometry quality sets long_diff only *between configurations*.** Fitted geometry
  gives positive long_diff; truth-projected drift gives ≤ 0. Within production, a sample's
  geometry error does not predict its long_diff (r ≈ 0).
- **Interpretation (partial):**
  - the coupling is **indirect**. A track's drift-direction shape sets the arrival-time profile
    of its charge on each pixel, which is also what long_diff controls. Lifetime moves with it.
  - The earlier long_diff-pinned test (−4.3 ± 2.9 of ~14 points, n = 3, other samples) suggests
    long_diff and lifetime share a *common cause* (the drift geometry) rather than forming a
    chain.
- **Test launched 30 Sep 08:35:** LDPIN3/7/8 — production with only long_diff pinned at truth.
- **Readout rule, stated before the result:** lifetime drop ≥ 10 points on 3/3 means long_diff
  is the channel, and an external long_diff constraint is a route to a working lifetime fit. A
  drop < 5 points means a common cause.

### 3.3g Results of 1 Oct: long_diff is a partial channel; the remaining −4% is a representation floor

**LDPIN3/7/8** (production, only long_diff pinned at truth):
- lifetime **+6.6%** vs +13.4% in production; paired **−6.8 ± 3.8** (−1.0, −5.6, −13.8);
- tran_diff **+3.1 ± 0.9** (3/3): the slack moves into transverse diffusion;
- stated rule (≥ 10 on 3/3 = channel; < 5 = common cause): **in between**. Pinning long_diff
  removes about half the lifetime bias, very unevenly. long_diff is a partial channel.

**DRIFTDXK6 1/2/4** (drift at truth with 6 modes, plus dE/dx at truth, frozen):
- lifetime **−4.1 ± 0.7%** vs −5.0% with dE/dx fitted; paired +0.9 ± 0.6 (n.s.);
- A_b +0.40 → +0.15 (−0.25 ± 0.05).
- **The dE/dx scale is not the cause of the residual.** With drift and dE/dx both correct,
  lifetime still sits at ~−4%, like BOTHTRU (−3.7%). That is a **floor of the production track
  representation** (spline geometry on the straight-line guess segmentation), not of any fitted
  nuisance. The harness on the *exact* true trajectories gave −0.5%.

**Consequence for a working fit:**
- In the drift-from-truth configuration lifetime is **stably** biased: −5.0% with a sample spread
  of 0.3%.
- A stable offset can be corrected with simulation, as is standard for estimator bias. That is
  only valid if it holds across true lifetimes.
- **Launched 1 Oct 14:50:**
  - **DRIFTK6T1200 D1/D3/D7**: same configuration at τ = 1 200 µs, paired against the production
    CURVE1200. A relative offset predicts −5%; a constant Δ(1/τ) predicts −2.6%.
  - **DRIFTNZ50 1/2/4**: 50 µm of noise on the truth drift shape (≈ hit-timing precision).
    Stated rule: lifetime within ±1.5 points of the noise-free run on 3/3 means 50 µm suffices.

### 3.4 The September "deliverables" do not hold

| claim | verified result | status |
|---|---|---|
| ×0.40 prior scale: mean \|err\| 7.26 → 2.95% | tuned seeds −4.31 ± 1.48 (p = 0.10 even in-sample); **fresh seeds −0.34 ± 0.65 (n.s.)** | **REFUTED** as a recipe (already self-corrected on 23 Sep) |
| calibration curve τ_true = (τ_fit + 204)/1.232, ±2.9% | in-sample fit reproduces; **leave-one-sample-out residual 101 µs = 5.4%**; bias depends significantly on the track sample (Friedman p = 0.03) | **overstated ~2×; not portable** — built on a simulation with truth-leaked inputs |
| Δ(1/τ) constant across τ | −4.47 ± 0.30 ×10⁻⁵/µs, χ²/dof 0.6 | consistent — but it describes the *fit*, not the forward model (§3.2) |
| long_diff pinning removes the lifetime bias | −4.3 ± 2.9 (1.5σ, n = 3) | not established |

![calibration curve](figs/fig6_calibration_curve.png)
![prior scale fresh seeds](figs/fig7_prior_scale_fresh_seeds.png)

### 3.5 Older headline results that are now unsupported
From the claims audit (`claims_audit.md`, 28-entry bug timeline):
- **`TS1` "production recipe, confirmed n = 6/6"** and **`BT2` scoreboard** (lifetime 1.65%, A_b
  0.27%): both ran 7–12 Aug, **before** the `data_seed` no-op fix (8 Sep). All their "seeds" share
  **one track sample**, and that sample sits on the favourable tail (the project's own addendum).
  The 1.65% lifetime is superseded by +8.85%. TS1's geometry gain may well be real, but it has
  never been re-run on independent samples.
- **"Lifetime is ×1.5–1.8 from its statistical floor"** (FINDINGS_SYNTHESIS): contradicted by the
  same document's addendum. The bias is ~10σ of the floor.
- **"The objective prefers truth"** (fig 66) and **"the objective is blind"**
  (IN_DEPTH_REVIEW): both are partially true, at different levels. Workstream E shows the *data
  likelihood at truth* does prefer (near-)truth. The *joint MAP objective with free nuisances*
  prefers a displaced lifetime.
- **The bug-discovery lag** is the systemic issue: the `data_seed` no-op was noticed on 30 Jul and
  fixed on 8 Sep. In between, the most-cited runs were produced. Five claims this month reversed
  between n ≤ 3 and a proper test.

---

## 4. The gap to real cosmic data (workstream A)

Severity is for readiness to fit real data.

| # | item | where | severity |
|---|---|---|---|
| L1 | **Geometry init = straight line through the furthest pair of TRUE segment endpoints**, oriented by true time; per-segment dE/dx init = true total dE / n_segments | `optimize/precompute_linear_guess.py:59-85,118-119` | high |
| L2 | **Track identity / hit association = MC (eventID, trackID)**; no pattern recognition | `precompute_linear_guess.py:52`, `dataio.py:243-244` | high |
| L3 | **Selection cuts on TRUE positions**, including true drift coordinate | `dataio.py:259,277-279` | high |
| L4 | **t0 = 0, known; no t0 parameter** — absolute drift distance is given | `dataio.py:217-218,563-564` | **critical** for lifetime |
| L5 | **dE/dx prior shape and mean target (1.887) are MLE fits to the true dE/dx of the fitted file** | `DEDX_PRIOR_DOSSIER.md` §1–2; `start_sci_case.sh:98-105` | high — and §3.3 shows lifetime is exquisitely sensitive to this block |
| L6 | kb, v_drift, gain, pedestal, W-value, shifts fixed at truth | `param_list_nd_noshift.yaml`, `module0.yaml` | medium-high |
| G1 | no per-pixel gain/threshold/pedestal spread, dead/noisy channels | global scalars in `consts_jax.py` | high |
| G2 | no space charge / E-field non-uniformity; single uniform eField | — | high for lifetime (drift-correlated by nature) |
| G3 | recombination form: Birks in both target and model | `consts_jax.py:501` | medium |
| G4 | secondaries filtered out (μ, p only); no multi-cosmic pile-up | `dataio.py:14-19` | medium |
| G5 | **no real-data loader.** `start_data_fit.sh` is stale (Jul 2025), uses truth-labelled "mockup" files, a different loss and different parameters | `optimize/scripts/start_data_fit.sh` | high |
| A1 | model threshold erf uses reset noise only (900 e⁻ vs √(900²+500²) = 1029 e⁻) | `fee_jax.py:95-96` | low: measured +0.4% hit excess, flat in drift (§3.2) |

**The precision requirement** (fig 1):
- Module-0 drift: 30.27 cm / 0.1587 cm/µs = 190.7 µs.
- At τ = 2.2 ms the whole lifetime signal is **8.3% of the charge** at full drift.
- A **1% drift-correlated charge error is an 11.5% lifetime error**; a 5% lifetime measurement
  needs ~0.4% drift-correlated charge accuracy.
- Space charge, E-field distortions, a t0 error of a few µs, or diffusion-dependent threshold losses
  each plausibly exceed 0.4% in real data.
- On ND-LAr-length drifts (50 cm) the lever arm improves by only 1.7×.

![lever arm](figs/fig1_lever_arm.png)

---

## 5. The path forward (ordered)

### Step 1 — Stop tuning the current objective; fix the estimator (simulation, ~1–2 weeks)
The optimiser, schedule, initialisation and prior-knob avenues are exhausted with measured nulls,
and I agree with that. The evidence now points at **which production ingredient** turns a
−2 ± 2% clean-setup result into +8.85%:
- **1a. ~~Ablation ladder on the dE/dx ingredients~~ — done (§3.3c): the full production dE/dx
  configuration is a null (−0.8 ± 1.1 points) when geometry is exact.** The cause is therefore on
  the geometry side. The deciding production run — geometry frozen at its best representation of
  truth, dE/dx fitted in production config (`GEOFRZ3/7/8`, the missing corner of the geometry ×
  dE/dx 2×2 next to DSREP / DXTRU / BOTHTRU) — was launched on 25 Sep. Readout rule, stated
  before the result: lands near BOTHTRU (≈ −4%) ⇒ dE/dx fitting is harmless whenever geometry is
  right and the bias is created by *fitting geometry* (with dE/dx free); lands near DSREP (≈ +9%)
  ⇒ an interaction inside the production code that the harness does not reproduce.
  ~~Marginalise dE/dx (Laplace)~~ — tested in §3.3b: no lifetime effect, halves the A_b shift.
- **1b. Measure the representation bias separately.** The −3.7% "ceiling" is the spline +
  resegmentation cost. Repeat BOTHTRU with finer resegmentation and more spline modes; if it moves,
  fix the representation before anything else.
- **1c. Always report the (A_b, lifetime) pair**, or lifetime *conditional on* A_b. Per-fit lifetime
  scatter is mostly A_b wobble. The 5×5 Hessian machinery (fig 71) should be used to quote the
  covariance for every fit.

### Step 2 — Establish a baseline the differentiable fit must beat (days)
Run a conventional **charge-scale-immune lifetime estimator** on the same simulated hits:
- per-pixel summed charge vs drift, truncated mean or shape fit on anode–cathode crossers
  (`analyze_lifetime_hits.py` has a start);
- **report its bias and resolution on DSREP1–16.**

If it beats +8.85 ± 4.7% (the project's own notes suggest a shape fit on segments reached ~−1%),
the pragmatic hybrid for data is: the **standard lifetime estimator + the differentiable fit for A_b,
E field and diffusion, with lifetime fixed or constrained.**

### Step 3 — Make the closure test honest (1–3 weeks)
Rebuild the pipeline so that nothing downstream of the target uses truth:
- geometry init and track association from a hit-level reconstruction (even a simple clustering +
  PCA line fit);
- selection on reconstructed quantities;
- **t0 treatment:**
  - either anode/cathode-crosser selection with a t0 from the track itself;
  - or a per-track t0 nuisance;
  - the lifetime lever arm makes this non-negotiable.
- a dE/dx prior derived from theory or a *different* simulated sample (Bethe/Landau-Vavilov
  for through-going muons of the sampled momentum spectrum), not from the fitted file.

Then re-measure DSREP-style ensembles. **Expect the numbers to get worse. That is the point.**

### Step 4 — Model-mismatch stress test (1–2 weeks)
Generate targets with effects the model does not have:
- ±1% global gain;
- 5% per-pixel gain spread;
- threshold spread;
- Box instead of Birks recombination;
- a mild E-field / space-charge distortion;
- a few-µs t0 error;
- 1–5% dead channels.

Measure the shift in every parameter. This gives the **systematic budget** that a real-data result
needs, and tells us which effects must be modelled (with nuisance parameters) before data.

### Step 5 — Real data (only after 1–4)
Start with the quantities that are robust in closure (A_b-as-charge-scale, E field, tran_diff), with
lifetime from Step 2's estimator. Write the real-data loader (flow/hit files → the fit's hit format)
with channel masks and per-channel pedestal/gain from calibration runs.

### Process rules (enforce, don't just write down)
- Launch production campaigns from a **clean, tagged commit**; refuse to start on `git_dirty`.
- **Pre-register** each test (hypothesis, arms, n ≥ 5, success criterion). Validate any tuned
  quantity on samples not used for tuning.
- Every derived "correction" (a calibration curve, a prior scale) must carry a leave-one-sample-out
  error.
- A bug found is a bug fixed *the same day*, with the list of affected runs marked stale in the
  docs.

---

## 6. Status table

| claim (latest docs) | verified status |
|---|---|
| A_b, E field recovered to ≲0.1% | ✅ solid (n = 16) — in closure, with the charge chain otherwise known |
| tran_diff ~1–2% | ✅ solid |
| lifetime bias +8.85% universal | ✅ solid (n = 16 + 4 truth points) |
| bias is a forward-model charge-loss defect | ❌ refuted (§3.2) |
| bias is not optimiser-limited | ✅ consistent with everything seen |
| (my claim) bias = generic Neyman–Scott from point-fitting dE/dx | ❌ refuted by the Laplace test: −2.1 ± 2.2 in a clean setup (§3.3b) |
| integrating dE/dx out (Laplace) fixes lifetime | ❌ no effect on lifetime (−0.2 ± 0.3); halves the A_b shift |
| joint dE/dx fit can be kept | ✅ unbiased within errors in a clean setup; costs 2.5× lifetime precision |
| production dE/dx configuration (prior + mean constraint + 1 cm grid) biases lifetime | ❌ null with exact geometry: −0.8 ± 1.1 points vs clean setup, n = 92 batches (§3.3c) |
| the +8.85% comes from the geometry side | ✅ refined (§3.3d): a geometry × dE/dx **interaction**, +8.1 ± 2.2 (3/3), on top of ~+7 from geometry fitting alone; dE/dx fitting alone is harmless |
| objective blind to lifetime / soft plane | ✅ quantified: 20–28 lifetime points per 1% A_b |
| ×0.40 prior recipe | ❌ refuted out of sample |
| calibration curve ±2.9% | ⚠️ ±5.4% out of sample; sample-dependent; not portable to data |
| TS1 production recipe / BT2 scoreboard | ⚠️ single shared track sample (pre-fix), never re-validated |
| lifetime near statistical floor | ❌ contradicted |
| long_diff measurable | ❌ +14% bias (verified); ±56% costing ~0 nats is the project’s own linked-diffusion result, not re-checked here |
| ready for cosmic data | ❌ truth leakage (L1–L6), no t0, no data path, no mismatch budget |

---

### Files
- `figs/` — all figures cited above.
- `ensemble_per_run.csv`, `curve_scan_per_run.csv` — per-run recomputed endpoints.
- `claims_audit.md` — full bug timeline, claim ledger and transcript-mined corrections.
- `scripts/` — every analysis and GPU-diagnostic script used here (`truthdiag*.py` + `run*.sbatch`
  for workstream E; raw outputs in `/sdf/scratch/neutrino/pgranger/review_2026_09/E_truthdiag/out/`).
