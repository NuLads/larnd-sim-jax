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
    one 400 m track sample (100 batches x 400 cm; log: 39 984 cm).
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

**Pre-paper runs (launched 7 Oct, at most 6 at a time).**
- **Round 1 (jobs 40145686–91):**
  - ENDSM10 1/2/4: R2 with the track endpoints smeared by 1 mm per coordinate (new knob
    `LARND_ENDPOINT_SMEAR_MM`). Every fit so far anchored both ends of each track at the *true*
    entry/exit points.
  - STRFRZ50 3/7/8: ensemble extension to n = 9.
  - **Rule (stated before results):** the endpoint test passes if |lifetime| ≤ 5% and
    |long_diff| ≤ 8% on 3/3 **and** the paired lifetime shift versus STRFRZ50 1/2/4 is within
    ±3 points on 3/3. Otherwise the endpoints must be fitted: per-track offset/tilt modes seeded
    from the hits, which the hit fit already estimates and currently discards.
  - New samples: |lifetime| ≤ 5% each.
- **Round 1 results (8 Oct 02:30; configs verified).**
  - **Endpoint test: FAIL, decisively.** With the track endpoints smeared by 1 mm per coordinate
    (ENDSM10 1/2/4), against the same fits with true endpoints:
    - lifetime +10.9 / +13.8 / +12.2% (paired shift **+12.8 ± 0.8 points, 3/3**);
    - long_diff +17 / +24 / +38%; A_b −0.3%; E field +0.2%;
    - position residual 850–900 µm (vs 130–145), coherent drift offset −75 to −115 µm;
    - objective +3 200 per batch.
    - Both ends of every track are anchored where they are given, so the fit cannot move them:
      the geometry stays ~0.9 mm off and the calibration absorbs it.
    - **R2 is therefore a closure result conditional on true endpoints.** A realistic version needs
      the endpoints fitted (per-track offset and tilt modes) and an absolute anchor for the drift
      coordinate, since free offsets would re-open the near-flat direction.
  - **New samples STRFRZ50 3/7/8: pass.** Lifetime +1.9 / +1.0 / −3.5% (production +9.1 / +14.6 /
    +16.4%).
  - **Pooled n = 9** (samples 1–9; production on the same samples in brackets):
    - lifetime **+0.5 ± 0.9%** (SD 2.6) [+9.0 ± 1.5]
    - long_diff **−1.6 ± 1.2%** [+15.0 ± 3.1]
    - A_b +0.18 ± 0.04% [+0.04]; E field +0.017 ± 0.002% [−0.035]; tran_diff +1.2 ± 0.4% [+1.0]
    - position 135 vs 178 µm
  - Dependency chains: each round-2 job started within 3 s of its predecessor finishing.
- **Paper-critical runs (8 Oct ~03:00).**
  - **Probabilistic vs stochastic** (fills "Why is it better?"; replaces the CL scan at the chain tails):
    - Design: calibration only on the TRUE segments (SCIMODE=ceiling, no dE/dx fit), so only the
      electronics model and loss differ. PROBCAL = probabilistic model + PPP likelihood;
      STOCHCAL = sampled electronics + the MMD/charge hit loss (`mse_adc`). Samples 1/2/4, 100 × 400 cm,
      10 000 iterations. One launcher copy for both arms (review scratch
      `PROBSTOCH/start_sci_case_ps.sh`, switches SCIPROBSIM / SCILOSS / SCIFITTYPE). Both arms
      smoke-tested.
    - Jobs: PROBCAL1/2/4 = 40213550/57/58, STOCHCAL1/2/4 = 40213535/36/37.
    - **Rule (stated before results):** the probabilistic likelihood is shown better if, paired by
      sample, its mean |error| over the five parameters is smaller on 3/3 samples, or if the
      stochastic arm does not settle (any parameter moving > 2% over the last 2 000 iterations).
      Otherwise the two are equivalent within noise.
    - **RESULT (9 Oct ~02:00, 6/6 at 10 000):** probabilistic BETTER on 3/3 (paired mean |err| 0.52 vs 27.8,
      0.57 vs 6.27, 1.11 vs 8.77). PROB: A_b +0.06, E +0.001, D_T +1.3, D_L +1.1, τ +1.2 (all ≤ 2.7% per sample).
      STOCH: D_L −54/−20/−30 (start −31: barely moves), D_T +36/+6/−13, τ +48/−5/+0.1, E −0.23/−0.08/−0.13,
      A_b −0.96/+0.07/−0.39. Both settled (ptp ≤ 0.26%). 2.5 vs 1.2 s/iteration. In paper 51f60fd (Table prob_vs_stoch).
  - **Short jobs, chained in one extra slot** (review scratch `PAPERJOBS/`): benchmark 40213577 →
    gradient check 40213578 → beam-width scan 40213579 → Hessian at truth 40213580 (ceiling mode,
    fit_type hess, sample 1).
  - **Results (8 Oct, written into the paper, commits d2350dc / ed52cdc / 9b1b5bc):**
    - **Benchmark (A100, eager):** probabilistic forward 0.5–0.7 s, likelihood + gradient 0.9–1.3 s
      per 3–4.5 m batch; peak 7.0 / 9.2 / 14.8 / 18.4 GB at 50 / 100 / 200 / 400 cm; stochastic
      gradient 0.24–0.33 s.
    - **Gradient check** (AD vs central FD, h = 1e-2…3e-4, two 400 cm batches, point displaced
      1–5 % from truth): A_b, long_diff and lifetime agree to < 0.3 % on both batches (best step;
      the small steps are float32-rounding limited). Batch 0: eField 0.5 %, tran_diff 0.05 %.
      **Batch 1: eField AD/FD = 1.168 / 1.172 / 1.173 at K = 50 / 100 / 200, the same at every
      step, so NOT the top-K stop_gradient** (my first guess, wrong). At K = 50, A_b FD sits 12–51 %
      above AD and scatters (rank exchanges visible); at K = 100/200 it agrees to 1 / 0.5 %.
      Source of the eField gap still unknown. Candidates are integer operations that move with
      timing, e.g. the ROI start (argmax over a threshold). The tick placement itself is
      differentiable: production `simulate_signals` splits each cluster linearly over two ticks.
    - **Beam width K** (500 throws, ~1900 pixels, K = 10…200): first-hit λ is K-independent but
      still shows the intermediate-probability deficit (−0.018 / −0.031 / −0.022 for sampled p in
      0.2–0.4 / 0.4–0.6 / 0.6–0.8). Second hit: truncation large at K = 10 (−0.44 at p > 0.95) and
      K = 25 (−0.08); below 0.01 from K = 50; K = 100 equals K = 200. ⇒ the deficit is **not**
      truncation, and the paper's explanation was removed.
    - **Deficit mechanism.** Analytic paths continue after every crossing (weight = crossing
      probability, without acceptance), and acceptance multiplies only the emitted hit. Two
      acceptance-tied candidates:
      - (a) hit-index bookkeeping. A failed first crossing promotes the next crossing to sampled hit 1.
        This moves probability between indices; the total is unchanged.
      - (b) re-arm noise. After a failed digitisation the stochastic sim re-arms with 500 e; the
        analytic model always uses 900 e.
      The 0.44 % total-hit deficit at truth (§ charge closure) rules out (a) alone. Test queued:
      `paper_deficit.py`, uncorrelated noise 0 in both models, job 40216462, runs after the Hessian.
    - **A_b offset +0.18 ± 0.04 % (R2): my first explanation was WRONG (corrected in the paper, 37c4669).**
      I had attributed it to the model's charge shortfall at truth (−0.34 % charge; Q ∝ A_b in Birks).
      The Hessian/score at truth says otherwise. With true inputs the likelihood optimum is
      A_b +0.02 ± 0.03 % with all five free, +0.007 % with (A_b, τ) free, and +0.07 % with A_b
      alone. The shortfall is in the NUMBER of hits; the per-hit charge, which pins A_b, is
      unbiased. So the offset arises in the nuisance estimation.
      - Supporting: across fits A_b follows the soft plane (lt = 6.7 − 22.6·Ab + 0.2·ld predicts
        +0.26 for R2 and +0.03 for production; observed +0.18 / +0.04).
      - Candidate: the fitted dE/dx scale (−0.1…−0.3 % in the fitted-dE/dx arms, 30 Sep).
    - **Hessian at truth:** the first attempt (40213580, jacfwd∘jacrev) **ran out of memory** (63 GB at 400 cm).
      Redone as 5 forward-over-reverse HVPs per batch, 200 cm batches, all events of data_seed 1
      (`paper_hessian.py`, job 40215173). The production sample is 398.1 m. Per-batch H is symmetric
      to 1e-8; it also stores per-batch scores for a sandwich estimate.
    - **Schedule invariance on the current recipe (user approved 8 Oct; queued at chain tails).**
      - Runs: SCHED3/7/8 = R2 config with the calibration LR held at peak for 1 200 iterations after
        warm-up, then decayed at 0.89768 per epoch.
      - Effect: final LR matched to R2 within 1.4%; total calibration travel 1.784× (July test: 1.78×).
      - Paired with STRFRZ50 3/7/8: same code, a5d7bdc plus the inactive endpoint knob. Samples 1/2/4
        predate the seed guards, so they are not used.
      - Launcher copy and submitter: review scratch `SCHED/`. Only the lr_kw line differs
        (`transition_begin`).
      - **Rule (stated before results):** a parameter is schedule-invariant if the mean paired shift
        (SCHED − STRFRZ50) is within ±1.5 points for lifetime and long_diff, ±1 for tran_diff, ±0.1
        for A_b and ±0.02 for E. Otherwise its shift is reported as dependence on the optimisation,
        and the parameter is not called converged.
    - **Paper-slot chain (8 Oct, after the Hessian 40215173):**
      - deficit test 40216462: uncorrelated noise 0 in both models.
      - ROI-window gradient test 40220530 (`paper_gradROI.py`): roi_split_length = 0 sends every pixel
        through the full-waveform path, so there is no integer window start.
        **Readout (stated before):** the window is the cause if eField AD/FD on batch 1 moves from 1.17 to
        within 2% of 1 in 'all_full'.
      - Ladder-as-profile 40220743 (`paper_ladder_profile.py`): score at truth for each quality-ladder
        rung on the Hessian batches. Linear-response profile δ = −H⁻¹g vs slice −g_j/H_jj.
        **Readout:** the paper's bias-vs-variance statement survives profiling if the dE/dx rungs move the
        profile optimum monotonically, and no position rung moves any parameter by more than 2 floor σ.
    - **Queue reordered (8 Oct ~04:30, scontrol):** the six ceiling-mode PROB/STOCH fits (≈1–5 h
      each) now run directly after round 2, ahead of the 14 h ensemble fits. "Why is it better?" gets
      its numbers on 8 Oct; n = 16 is delayed by a few hours. Chain map in `QUEUE/submitted.txt`.
    - **8 Oct ~05:10 results of the paper-slot chain:**
      - **Hessian at truth (final, 321 batches = 655 m, scaled to 398 m).**
        - Floor: A_b 0.029 %, E 0.0005 %, D_T 0.37 %, D_L 0.86 %, τ 0.72 %.
        - Others fixed: 0.016 / 0.0005 / 0.31 / 0.66 / 0.38 %.
        - Checks: the sandwich agrees within 0.85–1.3×; half-samples agree within 2 %; the A_b-only τ
          width is 0.68 % against 0.73 % from Laplace.
        - Observed R2 spreads (n = 9) are 3.5–4.2× the floor, and 12× for E.
        - Score-implied optimum at truth, all free: A_b +0.04 %, E −0.0005 %, D_T +0.84 ± 0.37 %,
          D_L +0.25 %, τ +1.58 ± 0.72 %.
        - In the paper as Table tab:floor (47b0085).
      - **Deficit test (40216462).** With uncorrelated noise 0 in both models, the first-hit deficit
        VANISHES (all bins |Δ| ≤ 0.002). Total hits sampled/analytic go from 1.00447 to 1.00178.
        ⇒ ~60 % of the total deficit is the digitisation treatment. The rest is in later hits (2nd hit
        −0.006…−0.008 at intermediate p). Next: re-arm stochastic with reset noise to separate
        bookkeeping from re-arm noise.
      - **ROI-window gradient test (40220530) was VACUOUS — my mistake.** The production probabilistic
        path (`_simulate_probabilistic_jit` → `get_adc_values_average_noise_chunked`) processes EVERY
        pixel over the full window in chunks of 128 and ignores the ROI counts. `select_split_roi` /
        `fee_sim_from_split` have had no caller since b020fcb, so roi_split_length = 0 changed nothing:
        identical loss and AD/FD = 1.1727.
        - The paper's "Region-of-interest extraction" paragraph described this superseded code and was
          removed (785a5e6); "four measures" became three.
        - The eField gap is still unexplained. Localisation job `paper_gradloc.py` (40223731) splits it
          by likelihood term and by pixel.
      - **Event display** (paper Fig. event_display, 8504d9b; script `scripts/eventdisplay/`): the fitted
        model is rebuilt from the checkpoint (contexts match exactly; positions 99–149 µm from truth).
      - **Overleaf branch merged into the paper's main (e352305).** Content was already in main except one
        obsolete comment.
    - **Ladder as first-order profiles (40220743, 321 batches, 655 m; `PAPERJOBS/ladder_profile_summary.json`).**
      The paper's bias-vs-variance statement for dE/dx survives profiling.
      - Uniform dE/dx (f = 0): D_L −15.3, τ −6.1, D_T −0.5, A_b +0.37 (%). The slice gives τ +6.1, so
        profiling REVERSES the lifetime sign through the A_b–τ charge-scale direction.
      - Straight trajectories (chord_c1.0): D_T −5.3, D_L −10.3, τ −7.2.
      - Random position offsets: linear response is not trusted beyond 50 µm (pos_400: D_L −8.7, τ −6.9),
        so those rungs remain slices.
      - dE/dx mean ±2 %: absorbed by A_b ∓1.75 % (= −0.87 × scale, Birks), as expected.
      - Partial (101-batch) numbers for chord were unstable (D_L −3.6 → −10.3 final); quote the final ones.
    - **Paper figures rebuilt (8 Oct), by sub-agents:**
      - Scripts are in `scripts/paperfigs/`: fig1_noise_effect, fig2_threshold, fig3_probabilistic_method,
        fig_performance, fig_mc_sampling; in preparation: fig_initialisation, fig_convergence.
      - Findings worth keeping:
        - **CODE BUG (cross-check path only):** `detsim_jax.apply_tran_diff` (the `mc_diff=False` branch of
          `simulate_drift`) needs `params.tran_diff_bin_edges`, which is never set. Its weights use
          `gaussian_1d_integral(bin_edges, sigma)` with NO sub-bin offset x0, unlike Eq. tran_diff_weights
          and the production `simulate_drift_new` (`gaussian_1d_integral_new`/`density_2d`).
          - Effect: disagrees with the MC mean by 6–33σ; an x0-aware fix agrees within 1.4σ.
          - Not used in production (`simulate_wfs` → `simulate_drift_new`).
          - The MC path also only works with number_pix_neighbors = 0 (shape mismatch at 4).
        - Production-fit step-time spikes (112 iterations of 10–26 s, 3.4 % of the time) are not
          checkpoint writes, a single slow batch, or recompilation (GPU allocations and peak memory
          unchanged), so they are host-side or shared-node contention.
        - Fig 2 (threshold scan): the original averaged over padding slots (~2 % dilution) and plotted
          hit 1 instead of hit 0.
    - **Naive calibration (8 Oct, user narrative: naive simplifications fail).** NAIVEC1/2/4:
      SCIMODE=floor (trajectories held at the straight lines between the true endpoints) on a guess file
      with dE/dx set to a CONSTANT 1.877 MeV/cm (the file's length-weighted mean;
      /sdf/data/neutrino/pgranger/lads-data/linear_guess_constdedx.h5). No dE/dx fit
      (DEDXFLAG=--non_deterministic); only the 5 calibration parameters are fitted. Otherwise R2
      settings (100 × 400 cm, 10 000 iterations, LR decay 0.91). Launcher copy in review scratch `NAIVE/`
      (LINEAR_GUESS overridable). Runs sequentially in the paper slot after gradloc.
      **Rule (stated before):** "naive inputs bias the calibration" is supported if |lifetime| or
      |long_diff| error > 5 % on 3/3 samples. The comparison is R2 on the same samples.
    - **Paper error fixed:** the paper said "≃ 40 m of track per sample" (and the Laplace widths were
      quoted for "≃ 38 m"). The production log says 39 984 cm ≈ **400 m** (Laplace: 95 batches ≈ 380 m).
      Exposure estimates were therefore 10× too small and are now 2.7 km (lifetime 1 %), 5.2 km
      (long_diff), 0.7 km (tran_diff).
- **Round 2 (prepared, launches when round 1 ends):** R2 at τ = 3 200 and 4 400 µs on samples 1/3/7,
  paired with the production CURVE3200/CURVE4400 fits.
  - **Rule (stated before results):** the per-sample spread grows ∝ τ, so the threshold scales with
    it: mean |lifetime| error over the 3 samples ≤ 5% at 3 200 µs and ≤ 7% at 4 400 µs.
  - **RESULT (8 Oct evening, all 6 at 10 000/10 000, `QUEUE/readout_round2.py`):**
    - 3 200 µs: lifetime −1.40 / −8.29 / +1.68 (mean −2.67 ± 2.95, mean |err| 3.79) → **PASS**.
      Straight start (CURVE3200): +21.1 / +13.4 / +21.2.
    - 4 400 µs: lifetime −8.93 / −10.72 / −4.54 (mean −8.06 ± 1.84) → **FAIL** (rule 7%).
      long_diff −4.03 ± 0.20; A_b +0.36 ± 0.02. Straight start (CURVE4400): +1.7 / +12.8 / +8.8.
    - A_b offset grows with τ: +0.19 (1 200), +0.21 (2 406), +0.30 (3 200), +0.36 (4 400).
    - Δ(1/τ) means [10⁻⁵/µs]: −0.83 (1 200), +0.10 (2 406), +0.92 (3 200), +2.01 (4 400).
    - **Every fit starts the lifetime at 2 200 µs** (lifetime_init; not the target). The final error is on
      the side of the start at 1 200 (+83% start → +1.1%), 3 200 (−31% → −2.7%) and 4 400 (−50% → −8.1%).
      At 4 400 the values at iteration 5 000 equal the final ones, and the mean lifetime gradient over the
      last 1 000 iterations has mixed sign (z −0.6, +2.2, −2.5): settled, not stopped short. Candidate:
      start-point (path) dependence via nuisance co-adaptation during the long travel. Not established.
  - **START-AT-TRUTH TEST (queued 8 Oct evening):** STRFRZ50T4400I{1,3,7}, same config + `SCIINIT=lifetime:4400`,
    jobs 40283659 (chain 4, before s14), 40283660 (chain 5, before s15), 40283661 (paper slot, after NAIVEC4).
    Same code as round 2 (only dirty change = endpoint smear, off by default, mtime 7 Oct 11:18).
    **Rule (stated before results):** path dependence if (init 4400 − init 2200) > +4 points on 3/3 AND
    |mean lifetime| ≤ 3%; genuine long-τ bias if mean lifetime ≤ −6%; otherwise inconclusive.
- **Queue (submitted 7 Oct 11:40 as 6 SLURM dependency chains, so slots refill the moment any job
  ends and at most 6 run at once; list in the review scratch dir `QUEUE/submitted.txt`).**
  - Round 2: jobs 40167559–64.
  - Ensemble samples 10–16: jobs 40167565–71.
  - CL scan, STRFRZ30 / STRFRZ70 on 1/2/4: jobs 40167572–77.
  - **Rules (stated before results):**
    - The ensemble confirms R2 at n = 16 (samples 1–16) if the pooled lifetime is within ±2%, the
      pooled long_diff within ±3%, and no sample exceeds |lifetime| = 8%.
    - CL 0.5 is not a knife-edge if CL 0.3 and CL 0.7 both give |mean lifetime| ≤ 3% and
      |mean long_diff| ≤ 5% on samples 1/2/4.

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
| L1 | **Geometry init = straight line through the furthest pair of TRUE segment endpoints**, oriented by true time; per-segment dE/dx init = true total dE / n_segments | `optimize/precompute_linear_guess.py:59-85,118-119` | high — measured 8 Oct (ENDSM10): 1 mm endpoint error held fixed → lifetime +12.8 ± 0.8 points; fix = ENDFIT (planned, see the 12 Oct entry at the end) |
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
- **track endpoints estimated from the hits and then FITTED (ENDFIT, requested by Pierre 8 Oct; spec in the
  12 Oct entry at the end)** — fixing them at a 1 mm-smeared value costs +12.8 lifetime points (ENDSM10);
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


## Reference larnd-sim comparison (9 Oct 2026)

- **Past work found:** a matched setup was started in Oct 2024 and never finished. It lives as uncommitted edits in
  ~/reference (TPC border hack, flat output writer, noise/diffusion off), with input_*.h5 equal to prepared_data and
  outputs output_*.h5. The repo's tests/test_simulation_consistency.py and optimize/comparison.py compare JAX vs a
  frozen JAX output, NOT the reference. cyifan ran a reference-only benchmark in Jun 2025 (1000 module-0 events in 21 394 s).
- **Pipeline:** /sdf/scratch/neutrino/pgranger/review_2026_09/LARNDSIM_CMP/
  - Reference in develop.sif with PYTHONPATH=reference_copy:pydeps and PYTHONNOUSERSITE=1. The python3.10 user site holds a
    copy of OUR larndsim, which would shadow the reference.
  - JAX in larndsim-jax_main.sif: the conda jax is CPU-only.
  - Matched input: 66 segments, 216.13 cm. The JAX side needs --max-batch-len 1000; its default of 50 cm drops all through-going muons.
- **Results (SLURM 40319804):**
  - Timing, warm, 216 cm: reference 4.2 s (19-20 ms/cm, end to end including output writing; compile 7 s);
    JAX stochastic 0.10 s (0.45-0.48 ms/cm), probabilistic 0.24-0.34 s (1.1-1.6 ms/cm).
    Memory: reference ~1.75 GB (nvidia-smi incl. context); JAX peak 6.7 GB. [CORRECTED 19 Oct (paper session, PAPERJOBS/bench2_run2): the 6.7 GB is the dataset/LUT loading transient; with inputs built on the CPU the JAX forward peak is 2.3-3.6 GB on 2-4.5 m batches, incl. 1.5 GB of tables]
  - Outputs: 506/520 JAX pixels match reference pixels (<0.01 cm, after the x -pitch/2, y +0.1247 cm transform).
    Stage-3 per-hit ADC spectra agree (means 88.3 vs 88.5, KS 0.085).
  - **CORRECTION (9 Oct):** the agent's empirical y transform (+0.1247) was off by one pixel pitch; it matched the track
    to the reference's NEIGHBOUR pixels. The correct one is y_ref = y_jax - 0.3187 (≈ the 0.3175 z-border difference), checked
    on hit pixels only: 411/505 JAX hit pixels then match a ref hit pixel, vs 247. With it (stage 1): 409 common hit pixels;
    hits per common pixel 1.78 (ref) vs 2.06 (JAX); ADC per hit 89.3 vs 89.7; first-hit time median diff -0.1 us (IQR ±0.5).
    The remaining excess of ~16% more hits per pixel and ~13% more hit pixels is under diagnosis.
  - With the correct mapping: integrated induced charge per collecting pixel JAX/ref = 0.995; top-pixel waveforms overlay
    (charge ratios 0.999-1.017); first-hit dt median -0.1 us. **Dead time ruled out** (REF_NO_DEADTIME=1 run SLURM 40321016:
    ref hits 770->776, 773->784 vs JAX 981/961). The ref threshold LUT is built from the overridden 5000 e-; the post-hit
    bookkeeping is equivalent tick for tick. Clue: the agent's FSM replay on the ref's OWN stored pixel_signals also gives
    +17% hits -> the ref may digitise a waveform different from the one it stores. Test: the exact kernel under CUDASIM.
  - (superseded) hits/pixel 0.87 (ref) vs 1.90 (JAX) on matched pixels; hit-time mean 97.4 vs 86.3 us;
    induced-current scale ratio inconsistent between stages (0.16, 0.65). Being diagnosed from the code.
  - Bug: simulate_probabilistic divides by zero with both noises exactly 0 (fee_jax _find_one_hit_step, z_scale=1/sigma).

- **ROOT CAUSE of the hit excess (9 Oct ~04:10): the reference never ran with our overrides.** run_simulation() calls
  importlib.reload(fee) (cli/simulate_pixels.py:424), which re-executes fee.py and resets every constant set by
  fee_overrides: threshold back to 7000 e- (verified by dumping the FEE inputs: pixel_thresholds = 7000 everywhere),
  ADC_BUSY_DELAY/RESET_CYCLES, V_CM/V_REF, noise (stage 3 "noise on" was noise-free: the Oct 2024 edit zeroes it in the file).
  The dumped FEE input waveforms equal the stored pixel_signals (1529/1529 rows), so the "stored != digitised" lead was a
  red herring (the CPU kernel test used 5000). The no-dead-time test was also invalid. The reload also re-creates the CUDA
  kernels, so the "warm" reference timings likely include recompilation. Fix: run_reference_stage.py now re-applies the
  overrides after every reload of fee. Rerun SLURM 40321613: stages 1-3, stage 1 without dead time, stage 3 on a 10x
  input (2161 cm, 30 events) for the marginal per-cm cost.
- **FINAL (9 Oct ~05:00, overrides applied: dump confirms threshold 5000):** hits ref/JAX 979/981 (no diffusion), 974/980
  (diffusion), 941/961 (noise); ADC/hit 89.60/89.54; summed induced charge on common hit pixels within 1.4%, median 0.99 on
  pixels > 2e4 e-; pixels < 1e4 e- get 2-3x more in JAX (neighbour treatment); first-hit dt constant offset, IQR 0.6 us;
  noise-on ADC/hit sd 6.6 (ref, includes 650 e- DISCRIMINATOR_NOISE) vs 5.5 (JAX has none). Dead time: +0.7% hits only.
  Speed: ref marginal 4.4 ms/cm (12.89 s for 2161 cm vs 4.40 s for 216 cm, incl. output writing), compile ~7 s;
  JAX 0.46 (stoch) / 1.1-1.6 (prob) ms/cm forward; memory ref cupy pool 1.3-2.5 GB vs JAX peak 6.7 GB. [CORRECTED 19 Oct (paper session, PAPERJOBS/bench2_run2): the 6.7 GB is the dataset/LUT loading transient; with inputs built on the CPU the JAX forward peak is 2.3-3.6 GB on 2-4.5 m batches, incl. 1.5 GB of tables]
  In paper ae91f2f (Fig. larndsim_comparison, Sec. II E). The earlier "40x faster" was an artefact of reload+recompile.

- **NAIVEC result (9 Oct 13:20, 3/3 at 10 000):** straight + uniform dE/dx, fixed, calibration only. D_L -31/-14/-16 (-20 +- 5),
  tau -12.8/-12.9/-4.5 (-10 +- 3), D_T -12.5/-14.3/+2.6 (-8 +- 5), A_b +0.86, E +0.02. R2 on the same samples (1/2/4):
  -1.3 / -0.6 / +1.3 / +0.22. Rule met via D_L (3/3 > 5%). Consistent with the summed first-order shifts. In paper 1cee063.
- **START-AT-TRUTH RESULT (9 Oct 22:57, 3/3 at 10 000):** STRFRZ50T4400I lifetime -4.54 / -4.42 / +9.50 (mean +0.18 +- 4.66)
  vs -8.93 / -10.72 / -4.54 from 2200 us; diffs +4.4 / +6.3 / +14.0. **Rule: PATH DEPENDENCE** (all diffs > +4, |mean| <= 3%).
  The spread at 4400 us is ~8% per sample (vs 2.6% at 2406). On data the lifetime start needs an independent estimate.
  In paper 9a66989 (sentences after "Lifetime sensitivity regimes" + query on the "geometry" sentence).
- **ENSEMBLE n=16 COMPLETE (10 Oct ~05:00; STRFRZ50 1-16 vs DSREP 1-16, all at 10 000):** pre-set rule MET.
  R2: A_b +0.219 +- 0.026 (SD 0.104), E +0.020 +- 0.002 (SD 0.008), D_T +1.34 +- 0.28 (SD 1.11),
  D_L -2.20 +- 0.99 (SD 3.97), tau -0.22 +- 0.57 (SD 2.26); pos (median last 1e3) 134 +- 2 um.
  Max |tau| 4.09% (s9), max |D_L| 10.17% (s13). Samples 10-16 alone: tau -1.17 +- 0.56, D_L -2.93 +- 1.74.
  Straight start (DSREP, same samples): A_b +0.03, E -0.033, D_T +1.29, D_L +13.86 +- 2.38, tau +8.85 +- 1.18, pos 172 +- 8.
  Paired R2-DSREP: D_L -16.1 +- 2.6, tau -9.1 +- 1.2, loss -1039 +- 39 per batch. D_T offset is the SAME in both arms
  (not from the initialisation); D_L sits 2.2 SE below truth. Exposure for 1%: tau 2.0 km, D_L 6.3 km, D_T 0.5 km.
  In paper 5ccd5c0 (nine -> sixteen everywhere; Figs. fit_convergence(_seeds) rebuilt, cache convergence16.npz).
  By-product rerun on 16 samples: figs/byproduct16/ (BYPRODUCT_OUT env).
  - Follow-ups (10 Oct ~06:00): Fig. initialisation rebuilt on 16 (13 817 drift-frame tracks; >500 um tail DSREP 4.91%,
    R2 0.82%; curved-track start error 724 -> 460 um drift, 220 um pixel plane), paper d664ed0. By-products on 16
    (figs/byproduct16: traj median 77 um, straight 432, track dE/dx 1.1%, segment 3.2%, frozen 12.4%), paper 5b47f13.
  - **Open (internal, not in paper):** across the 16 samples the A_b and tau errors correlate at r = -0.80 with slope
    -17.4 (Hessian soft axis: -0.75, -18); the line is tau = +3.6% - 17.4 dA_b. Taken literally, an unbiased A_b would put
    the lifetime at ~+3.6%, i.e. the -0.2% lifetime may partly be the +0.22% A_b offset projected along the soft axis.
    The sample scatter does not prove the MEAN offset follows the same line; untested. Relevant to "A_b offset from
    nuisance estimation" (37c4669).
- **SCHEDULE-INVARIANCE RESULT (10 Oct 07:31, SCHED3/7/8 all at 10 000; rule fixed 8 Oct above): FAILS for tau, D_L, A_b.**
  Config check: only lr_kw differs (decay 0.89768 + transition_begin 1200 vs 0.91); env identical; code a5d7bdc + the
  inactive endpoint knob (fit_params.py mtime 7 Oct 11:18, both arms after it).
  Paired SCHED - STRFRZ50 (n=3): tau +6.48 / +3.51 / +11.75 (+7.25 +- 2.41; limit 1.5) FAIL; D_L +1.58 / +4.99 / +8.54
  (+5.04 +- 2.01; limit 1.5) FAIL; A_b -0.22 / -0.12 / -0.37 (-0.236 +- 0.073; limit 0.1) FAIL; D_T -0.09 +- 0.67 PASS;
  E -0.001 +- 0.001 PASS. Position +1 +- 1 um. Loss: SCHED HIGHER by +22.8 / +3.2 / +24.4 per batch (+16.8 +- 6.8).
  Mechanism check: coherent drift offset (drift_residuals.py) R2 -10.9/-9.2/-13.9 um, SCHED -8.4/-8.6/-10.9 (shift
  +2.1 +- 0.7 um) vs DSREP +16.9/+15.5/+58.8 -> NOT the drift-coordinate mode. The shift runs along the A_b-tau
  charge-scale soft direction (A_b down, tau up), plus D_L up.
  Reading: the longer-travel schedule freezes at a different, HIGHER-loss point; the smallest loss excess (s7, +3) goes
  with the smallest tau shift (+3.5). R2 is nearer the minimum on all three, but nothing shows R2 is AT it.
  Per the rule: E and D_T converged; tau, D_L, A_b "stable but not converged" -> an optimisation dependence of several
  points not in the sample-to-sample spread. Paper: paragraph after "Convergence criteria" + query on framing.
  Candidate next tests (not launched): (1) cross-evaluate the loss with the calibration block swapped between the two
  endpoints (is the +17/batch in the calibration or the nuisance block?); (2) continue both endpoints at a common
  low-but-nonzero LR to see whether they converge to one point.
  - **Cross-evaluation test (launched 10 Oct ~08:00; rule fixed before results):** SCHED/cross_eval.py, one GPU job per
    sample 3/7/8. Per batch, data loss + calibration gradient at (th_R,nu_R), (th_S,nu_R), (th_R,nu_S), (th_S,nu_S)
    (theta = last-1000 mean, nu = final checkpoint, fit's own targets asserted identical); conditional Newton step
    -H^-1 g with the truth Hessian (hessian.json) scaled to the sample length.
    Rule: CALIBRATION-LIMITED if on >=2/3 samples L(th_R,nu_S) < L(th_S,nu_S) AND the Newton step from S moves tau
    toward R2 by >= half the SCHED-R2 difference. NUISANCE-COUPLED if on >=2/3 |Newton tau step| < 1.5 points at both
    end points AND each theta is preferred with its own nu. Otherwise MIXED.
  - **Cross-evaluation RESULT (10 Oct 08:05, 3/3 done, jobs 40426906-08; first submission 40426803-05 died on an over-strict
    target-equality assert: the two runs' stored targets agree to GPU float precision, ADC ~1e-5, s7 a few hits 1 tick
    / 0.6 ADC apart -> same data). Totals over 100 batches (data term, no priors):
    | s | L(thR,nuS)-L(thS,nuS) | L(thS,nuR)-L(thR,nuR) | L(S,S)-L(R,R) | Newton tau from R | from S | gap S-R |
    | 3 | -11.4 | +168.7 | +2001 | -3.03 | -4.47 | +6.48 |
    | 7 | -28.2 | +73.7 | +183 | -0.61 | -1.60 | +3.51 |
    | 8 | +5.0 | +177.1 | +2628 | -1.27 | -4.57 | +11.75 |
    Newton D_L from R -2.18/-3.33/-0.75, from S -2.61/-5.38/-4.51. **Verdict per rule: MIXED** (calibration-limited
    needs the S Newton step >= half the gap on 2/3: only s3; nuisance-coupled needs |tau step| < 1.5 at both ends: none).
    Reading: SCHED's loss excess is in the DATA term (+20/+1.8/+26 per batch vs history total +23/+3/+24) and sits in its
    NUISANCES (with either theta, nu_S is ~2000/150/2630 worse); swapping theta changes the loss by <= 1.8/batch.
    Conditional re-optimisation of theta closes 39-69% of the tau gap; the rest is carried by the nuisances. The
    likelihood clearly prefers R2's end point (Delta NLL ~ 2000 on s3/s8). R2 itself is not at its conditional theta
    optimum either (Newton tau -0.6 to -3.0; truth Hessian, approximate). Practical: lowest loss is a truth-free selector
    between end points. Next (proposed, not launched): continue both end points at a common constant calibration LR.
  - **Mobility check (10 Oct, from the histories):** SCHED was still descending when its LR ran out: lifetime net change
    SCHED3 -7.6 (5-6k) / -1.9 (6-8k) / -0.3 (8-10k); SCHED8 -4.4 / -3.0 / -0.7; R2 (STRFRZ50 3/8) flat after 5k
    (|net| <= 0.4). So SCHED froze mid-descent toward R2. ("Stable" by ptp, but trending.)
  - **CONTINUATION TEST (user-approved 10 Oct; rule fixed before results).** CONT/make_cmd.py + run_cont.sbatch: resume
    STRFRZ50{3,7,8} -> CONTR{s} and SCHED{3,7,8} -> CONTS{s} from iteration 10000 to 14000 with the run's own argv/env
    except: calibration LR 1e-3 constant 10000-12000 (production level near iter 5200), then x0.805/epoch to 1.6e-5;
    chain decay restarted (--chain_decay_start 5000: scale 0.22 -> 0.067); dE/dx unchanged (constant 1e-2,
    --dedx_freeze_iter 20000 = never frozen, as in production). Identical for both arms.
    Rule (lifetime, mean of iterations 13001-14000): MEET if |tau_CONTS - tau_CONTR| < 1.5 points on >= 2/3 samples;
    APART if the gap stays > half the original (3.2 / 1.75 / 5.9) on >= 2/3; otherwise PARTIAL. Secondary: production
    stability under re-heat, |tau_CONTR - tau_R2| > 1.5 on >= 2/3 -> the production value is not stable under re-heating
    either. D_L and A_b gaps and the losses reported alongside, not part of the verdict.
    - Launch problems (10 Oct, before any readout): (1) on resume --iterations is ADDITIONAL -> first submission would have run
      to 24000; cancelled. (2) the calibration LR is the per-parameter value of the param YAML (0.1), --lr is ignored ->
      the second submission re-heated the calibration to the full PEAK: D_L -1.2 -> -35% in 25 iterations, loss +585/batch
      at 10200; cancelled, checkpoints deleted. Also dropped the chain re-heat (restored geometry gets a fresh Adam state,
      so a higher chain LR kicks every coefficient by a full step). Final design: calibration peak 1e-3 via
      CONT/param_list_cont.yaml, chain and dE/dx schedules as in production. Rule unchanged.
  - **CONTINUATION RESULT (read 11 Oct 05:30; 6/6 done at 14000, jobs 40438752-60; start smooth: D_L moved <= 0.5 in the
    first 25 iterations).** Mean of iterations 13001-14000:
    | s | tau gap before -> after | tau CONT_R - R2 | tau CONT_S - SCHED | D_L CONT_R - R2 | loss CONT_R - R2 /batch | loss gap S-R before -> after |
    | 3 | +6.48 -> +4.01 | -2.85 | -5.32 | -1.79 | -12.2 | +22.8 -> +11.0 |
    | 7 | +3.51 -> +1.29 | +1.23 | -0.99 | -2.46 | -20.7 | +3.2 -> -1.1 |
    | 8 | +11.75 -> +7.18 | -1.05 | -5.62 | -1.08 | -22.6 | +24.4 -> +19.1 |
    **Verdict per rule: APART** (gap > half the original on s3 and s8; MEET only s7). Secondary: production stable under
    re-heat per rule (|shift| > 1.5 only on s3). CAVEAT: SCHED's continued lifetime was still descending at ~1-2 points per
    500 iterations when the re-anneal began at 12000 (s3: 8.4 -> 3.8, s8: 8.2 -> 3.5 over 10000-12000), so the test does
    not separate "distinct minima" from "too slow to meet in 2000 iterations".
    Also: the PRODUCTION end points are not at the minimum either: continuing lowers their loss by 12-23 per batch and
    moves D_L by -1.8 +- 0.4 points (3/3 negative, matching the cross-eval Newton step -2.1), lifetime -0.9 +- 1.2,
    A_b +0.07, D_T +0.1. More optimisation moves D_L AWAY from truth (reported -2.2% -> about -4% if it held for all 16).
  - **ENSEMBLE CONTINUATION (user: "Do option 2", 11 Oct; rule fixed before results).** CONTR{1..16}: every production fit
    STRFRZ50{s} resumed 10000 -> 14000 with the CONT protocol (calib peak 1e-3 const to 12000 then x0.805/epoch; chain and
    dE/dx as production). 3/7/8 already done; 13 new jobs, <= 7 concurrent (second wave afterany on the first).
    Code check: samples 1/2/4/5/6/9 ran at 9a45383 (dirty), the rest at a5d7bdc (dirty). The 9a45383..a5d7bdc diff includes
    the PPP padding-mask fix in strategies.py, but that file's mtime is 29 Jul: every run used the fixed loss (dirty tree).
    The other difference is the 7 Oct seeding coverage guard (one track in ~13k); on resume seeding is skipped, only the
    straight/free classification is recomputed -> at most ~1 track per sample can change status. Accepted.
    Readout (mean of 13001-14000 vs production 9001-10000), n=16:
      - paired shift per parameter, mean +- SE; "significant convergence shift" if |mean| > 2 SE;
      - D_L drift "real" if the mean shift is negative, > 2 SE, and >= 12/16 samples negative;
      - pre-set ensemble rule re-evaluated on the continued values (pooled tau within +-2%, pooled D_L within +-3%,
        max |tau| < 8%); if tau or D_L shift significantly, the paper reports the continued values as the result.
      - loss change reported (sanity: expected lower).
  - **ENSEMBLE CONTINUATION RESULT (read 11 Oct 19:00; 16/16 at 14000, all rc=0; CONT/ensemble_readout.json).**
    | param | production (n=16) | continued | paired shift | neg |
    | A_b | +0.219 +- 0.026 | +0.268 +- 0.028 | +0.049 +- 0.016 SIGNIFICANT | 3/16 |
    | E | +0.020 +- 0.002 | +0.022 +- 0.002 | +0.002 +- 0.001 | 6/16 |
    | D_T | +1.34 +- 0.28 | +1.30 +- 0.30 | -0.04 +- 0.14 | 6/16 |
    | D_L | -2.20 +- 0.99 | -4.74 +- 1.02 (SD 4.06) | -2.54 +- 0.41 SIGNIFICANT | 15/16 |
    | tau | -0.22 +- 0.57 | -1.75 +- 0.64 (SD 2.54) | -1.53 +- 0.39 SIGNIFICANT | 13/16 |
    | loss/batch | 8654.1 | 8635.6 | -18.4 +- 1.0 | 16/16 |  pos median 133.7 -> 130.8 um (-2.9 +- 0.5, 15/16) |
    Rule: D_L drift REAL (negative, 6 SE, 15/16). tau and D_L shift significantly -> per the rule the paper reports the
    continued values. Pre-set ensemble rule on continued values: tau -1.75 PASS (+-2), D_L -4.74 FAIL (+-3), max|tau| 4.94
    PASS (< 8); max|D_L| 12.5.
    Trend during the constant-LR phase (ensemble mean): D_L -1.50 +- 0.23 (10000-11000) then -0.49 +- 0.33 (11000-12000),
    then -0.5 more during the re-anneal; tau -0.81 +- 0.28 then -0.49 +- 0.15, then -0.25. Decelerating, not stopped:
    expect roughly another -0.3 to -0.5 (D_L) and -0.5 (tau) with more optimisation (rough extrapolation).
    Reading: better-converged fits (lower loss on 16/16) sit at D_L ~ -5% and tau ~ -2%: the production "unbiased" values were
    partly the optimiser stopping early on the way down. D_L has a real negative bias of the procedure (~ -5%, 4.7 sigma).
  - **Paper switched to the continued fits (12 Oct, user decision "best available fits in terms of converged values"),
    commit 5f06ca8.** Schedule query resolved; new query on the D_L -4.7% origin. Figs. fit_convergence(_seeds) extended
    to 1.4e4 (fig_convergence.py CONVCONT=1 default, cache convergence16_cont.npz). By-products rerun on CONTR at 14000
    (byproduct.py BYPRODUCT_ARM/ITER, figs/byproduct_cont) in progress. Not continued (stated in the paper): straight-start
    baseline, lifetime-regime fits, recipe_summary configurations, event display (STRFRZ508 at 1e4), Fig. initialisation.
- **Basis representation floor vs K (12 Oct, Pierre: "not enough position freedom? discrepancy mid-track").**
  paperfigs/basis_floor_vs_K.py, samples 1-4 (3512 tracks), output BASISK/basis_floor.json. Production geometry: K =
  clip(round(L/40 cm), 3, 12) = 3 for EVERY track (all < 140 cm) -> 6 coefficients per track (3 if drift held straight),
  ~4 930 geometry parameters per sample. Best LS projection of the true trajectory (perfect information), mean 3-D distance:
  K=3 median 46 / mean 70 / length-weighted 98 um; K=6 29 / 37 / 47; K=12 24 / 27 / 30; saturates ~22 um (along-track
  offset not modelled, median 22 um). Production FIT: 76 / 142 / 159 um.
  By length: < 60 cm fit 59-69 um vs floor 26-48 (fit limited by data/optimisation); 60-100 cm fit 110 vs floor 96;
  > 100 cm fit 213 vs floor 207 -> long tracks (29% of tracks, 49% of the length) sit ON the K=3 floor.
  Length-scaled rules: knot 15 cm 7.4 par/track, floor lw 59 um; knot 10 cm 9.8 par/track, lw 42 um (K=6 fixed: 12, 47).
  Mid-track: the production fit residual is flat ~75-80 um along alpha; the K=3 floor is 35-46 um mid-track (54 at ends).
  Link to note: DRIFTTRUK6 (truth drift, 6 modes) moved D_L -13.4 -> -2.0 and loss -1011 vs 3 modes: D_L is the parameter
  now biased (-4.7%); the basis truncation is a candidate cause. Fit test proposed, not launched.
- **KNOT10 FIT TEST (user: "Yes", 12 Oct; rule fixed before results).** KNOT/make_knot_cmd.py + run_knot.sbatch. Samples 3/7/8
  (production ran on the current code, a5d7bdc + fit_params of 7 Oct 11:18). KN10S{s}: the production run's own argv/env with
  --chain_spline_knot_cm 10 (K = clip(round(L/10), 3, 12), ~10 coefficients per track), 10^4 iterations; KN10C{s}: the CONT
  continuation of it (identical protocol to CONTR), 1.4e4. Paired against CONTR{s} (primary, = paper values) and STRFRZ50{s}
  at 1e4 (secondary).
  Rule (primary, KN10C - CONTR, mean of the last 1e3 iterations):
    - position (mean 3-D, batch median) lower on 3/3 -> the extra freedom is used;
    - D_L: "basis truncation drives the D_L bias" if the shift is >= +2 points on >= 2/3 samples and the mean >= +2;
      "no effect" if |mean| < 1; otherwise partial;
    - lifetime, A_b, D_T, loss reported; lifetime flagged if |mean shift| > 1.5. Frozen (straight) fraction and hit-seed
      counts recorded (more modes change the significance tests).
  - KN10S3 start (12 Oct 02:00): runs at production speed; first 6 batches: frozen counts as production (0/0/1/1/1/2 vs
    0/0/1/2/1/2); hit seeds noisier with more modes on some batches (median e2 seed error 760 vs 432, 884 vs 561 um on
    batches 1, 4; similar elsewhere). Expected risk, noted before the result.
- **Fig. 1 rebuilt (12 Oct, paper commit after 775eb51).** paperfigs/fig_mc_sampling_prod.py (production path; packets =
  each 10 um piece displaced by its own (sT, sT, sL) draw, then simulated with D_T = D_L = 1e-12). 48.5 cm track (input_49cm):
  packet charge spread median 2.1% (p10-p90 1.1-7.4%), 0.61x reset noise; timing spread 0.45 ticks; projection 10->100 um:
  0.023% median, max 29 e-, 9/93 first crossings +1 tick; readout-noise timing spread median 10.2 ticks (the text's
  "one tick" was wrong; came from the cross-check path's fast parametric current model).
  Not shown, noted: packet MEAN vs projection +0.4% charge (median pixel; total +0.8%) = the projection's
  cloud-centre-pixel limitation (Appendix B); packet mean first crossing +0.4 ticks later (median), not explained.
  The cross-check path (current_model vs current_model_diff) shows a -0.7/-0.9 tick packet/projection offset even
  with D_L = 0 (NOLONG run): an artefact of that path; fig_mc_sampling(_track).py superseded.

## 12 Oct 2026 — paper loss vs code audit; calibration gradient clipping is always active

- Paper Eqs. ppp_nll_full / time_window / full_loss + priors checked against the production path
  (a5d7bdc, run argv + LARND_* env). All terms match except description details, fixed in NuLads_draft
  323c011: Birks (not modified Box) recombination; split-t has one normalisation (no per-side log s);
  mean constraint is per batch; MCS node spacing L/floor(L/2cm); clipping now stated.
- **`--max_clip_norm_val 100` clips the calibration gradient (sigmoid variables) on 98-99% of
  iterations** (median pre-clip norm ~9 400 over the production fit; still ~3 400 in the last 2000).
  The calibration update is therefore effectively normalised-gradient Adam: each batch's gradient enters
  with weight 1/|g_b|, so the stationary point is that of sum_b g_b/|g_b|, not of the summed loss.
- Evidence at the end of the continued fits (CONTR1-16, last 1000 iterations, averaged per batch over
  10 epochs, n_eff = 100 per sample): the normalised gradient mean is consistent with zero (pooled z
  0.2-1.4), the RAW gradient mean is not: eField pooled z = -3.8 (negative on 16/16 samples),
  long_diff +2.6 (14/16 positive), lifetime +1.8, Ab +0.2, tran_diff +1.1. Per sample |z| < 2, i.e. the
  offset is below one sample's statistical noise at fixed nuisances, but systematic across samples.
- NOT measured: how far the end point would move without clipping (the profiled shift can exceed the
  fixed-nuisance estimate). Test = short continuation of a few CONTR fits with the clip off (or norm
  1e9) at the continuation LR, then compare D_L / tau / E.
- **Clip test launched 12 Oct (CLIP/, make_clip_cmd.py).** Samples 3, 7, 8; from CONTR<s> at 14 000, +2 000 iterations;
  CLIPON (norm 100) vs CLIPOFF (norm 1e12), both with LARND_CALIB_ADAM_RESET=1 (new opt-in knob: fresh calibration
  Adam moments, schedule count kept); calibration LR 1e-3 to 15 000 then x0.63/epoch to 1.6e-5 at 16 000.
  **Readout rule, stated before any result** (read only when all 6 jobs are done; values = mean of iterations
  15 500-16 000; primary = paired Δ = CLIPOFF - CLIPON per sample, so the common continuation drift cancels):
  - **Clip matters:** Δ long_diff or Δ lifetime has the same sign on 3/3 with |mean Δ| >= 1 point.
    Then rerun the 16 continuations without the clip before quoting the results.
  - **Clip irrelevant:** |Δ| < 0.5 point on 3/3 for long_diff and lifetime (and |Δ eField| < one sample spread,
    0.008%). Then the paper keeps its numbers; the clip is a methods detail.
  - Otherwise **partial**: quote Δ with its spread; decide with more samples.
  - Sanity: in CLIPOFF the raw end-gradient mean (last 500 iterations, per-batch averaged) should drop to |z| < 2;
    if it does not, the run did not reach the unclipped stationary point and the test is inconclusive.

## PLANNED — ENDFIT: endpoints estimated from the hits, then fitted (requested 8 Oct, not yet built)
Pierre, 8 Oct: "fitting the endpoints as well is necessary. Let's keep that as a note to ourselves." Recorded then
only as a conclusion of ENDSM10 (§0c) and never put on a to-do list; re-raised 12 Oct. **Nothing built or run yet.**
- **Why:** every fit anchors both ends of each track at the TRUE entry/exit points (the sine modes vanish there).
  ENDSM10 1/2/4 (endpoints smeared by 1 mm per coordinate, held fixed): lifetime +12.8 ± 0.8 points (3/3),
  long_diff +17/+24/+38%, position 850–900 µm, coherent drift offset −75 to −115 µm.
- **Initial estimate:** from the hits. The hit seed (_seed_drift_from_hits) already fits [1, alpha, sine modes] to the
  per-pixel hit offsets from the nominal line and DISCARDS the constant and linear terms; keep them as the start of
  per-track offset/tilt. Along-track extent from the first/last hit projections (held, not fitted, at first).
  Full version: nominal line itself from a hit PCA fit instead of true endpoints.
- **In the fit:** 4 extra coefficients per track (offset and tilt along e1 and e2, i.e. basis columns 1 and alpha,
  equivalently the two endpoint displacements), with a Gaussian prior of width = endpoint resolution. The MCS prior
  is blind to them (zero second difference), so the prior is needed.
- **Drift anchor (the hard part):** free drift offsets reopen the near-flat drift direction (ENDSM10's coherent
  offset). Candidates, to pre-register: keep the e2 offset of the straight-pinned tracks held; a soft zero-mean
  constraint on the batch's drift offsets; or a per-track t0 nuisance (Step 3).
- **Test:** samples 1/2/4 (as ENDSM10). Arms: smeared + fixed (= ENDSM10, exists); smeared + fitted; hit-estimated +
  fitted; reference = true endpoints (STRFRZ50/CONTR). Pass: paired lifetime shift vs true endpoints within ±3 points
  and long_diff within ±5 points on 3/3.
- **12 Oct — BUILT** (optimize/fit_params.py, opt-in `LARND_FIT_ENDPOINTS=1`, `LARND_ENDPOINT_PRIOR_MM`, default 1 mm).
  Per direction the coefficient block is [K sine modes, start, end]: the (1−alpha) and alpha columns move the two
  ends transversely (along-track position of the ends not fitted). Gaussian prior per endpoint coefficient, summed
  over tracks; MCS prior blind to them. Drift anchor = the existing straight-track drift freeze, which now also holds
  the drift endpoints of pinned tracks. The hit seed is unchanged (endpoint terms start at 0 = the given endpoints).
  Offline checks on STRFRZ508 batch 35 (scratchpad test_endfit.py): zero endpoint terms reproduce the production
  trajectory to 0.06 µm; endpoint terms give the exact linear end-to-end displacement (0.1 µm); MCS prior with
  endpoint terms only 6e-11; prior value exact; gradients reach the endpoint terms; the hit seed reproduces the
  production log line for this batch word for word; straight-track masks freeze the drift endpoints. Default-off
  path reduces to the previous expressions (pending CLIP8 / KN10C jobs unaffected).
  NB the earlier ENDSM10 runs used `LARND_ENDPOINT_SMEAR_MM=1.0` (1 mm per coordinate, smear seed 0).
- **ENDFIT test, queued 12 Oct** (ENDFIT/, make_endfit_cmd.py; argv + env of STRFRZ50<s>, 1e4 iterations, fresh):
  - EFT10 1/2/4: endpoints smeared by 1 mm per coordinate (`LARND_ENDPOINT_SMEAR_MM=1.0`, same realisation as
    ENDSM10) and FITTED (prior 1 mm).
  - EFT0 1/2/4: true endpoints, FITTED (the cost of the added freedom alone).
  - References (exist): STRFRZ50 1/2/4 (true, fixed), ENDSM10 1/2/4 (smeared, fixed).
  **Readout rule, stated before any result** (read only when all 6 jobs are done; values as compare_arms.read,
  mean of the last 1e3 of 1e4 iterations; paired per sample against STRFRZ50):
  - **ENDFIT works:** EFT10 − STRFRZ50 within ±3 lifetime points and ±5 long_diff points on 3/3 (vs +12.8 ± 0.8
    lifetime for ENDSM10).
  - **Partial:** EFT10 recovers ≥ half of the ENDSM10 lifetime shift on 3/3 but fails the ±3 band.
  - **Freedom cost:** if EFT0 − STRFRZ50 exceeds ±3 lifetime points on ≥ 2/3, the added freedom reopens the drift
    direction by itself and needs a stronger anchor (zero-mean drift-offset constraint or t0) before EFT10 can be
    interpreted.
  - Also read: position residual (STRFRZ50 130–145 µm, ENDSM10 850–900 µm) and the endpoint error before/after.
  - Next (not queued): endpoint starting values from the hits (the seed's discarded constant/linear terms; drift
    terms biased by the hit-time latency) and a nominal line from a hit fit.

## 13 Oct 2026 — clip test result: PARTIAL (read after all 6 jobs finished; configs verified from the job logs)
Values = mean of iterations 15 500–16 000; Δ = CLIPOFF − CLIPON, per sample 3 / 7 / 8:
- long_diff −0.07 / −1.96 / −0.19 (same sign 3/3, mean −0.74 points); lifetime −0.84 / −0.96 / +0.38 (mean −0.48);
  eField +0.003 / +0.002 / +0.003 % (below the 0.008% sample spread); A_b −0.015 mean; tran_diff −0.01 mean.
- Rule: neither "clip matters" (needs |mean| ≥ 1 point with a common sign) nor "irrelevant" (needs |Δ| < 0.5 on 3/3:
  long_diff on 7 and lifetime on 3 and 7 exceed it) → **partial**. The clip moves lifetime / long_diff by well under a
  point on average, against sample spreads of 2.5 / 4.1 points; eField and A_b unaffected.
- Sanity passed: in CLIPOFF the raw end-gradient mean is consistent with zero (|z| ≤ 0.6); in CLIPON eField still
  sits at z −0.6 / −0.6 / −0.9 (the clipped stationary point), as before.
- Not separated: run-to-run divergence of two identical continuations (GPU scatter-add nondeterminism over 2 000
  iterations); a duplicate CLIPON would give that noise floor.
- **TRAP found:** a resumed run's checkpoint carries the ORIGINAL run's provenance (argv) and env_overrides (resume
  copies the whole training history), so both arms' pickles claim clip 100 and no Adam reset. Identify resumed
  runs from their job logs (or test_name), never from the pickle provenance.

## 13 Oct 2026 — KNOT10 result: BASIS TRUNCATION DRIVES THE D_L BIAS (rule met; read after all 6 jobs finished)
Primary, KN10C − CONTR at 1.4e4 (mean of the last 1e3 iterations; configs verified from the job logs; ~9.8 vs 6
coefficients per track; frozen fraction 12.5–13.8%, as production):
- long_diff shift +3.92 / +7.61 / +4.27 (≥ +2 on 3/3, mean +5.27) → **"basis truncation drives the D_L bias"**.
  long_diff error: KN10C +0.9 / +5.6 / −5.9 (mean +0.2) vs CONTR −3.0 / −2.0 / −10.1 (mean −5.0).
- position (batch median of the per-batch residual): 90 vs 120–140 µm, lower on 3/3 → the extra freedom is used.
- loss −339 / −351 / −476 per batch (a far better fit; for scale, the schedule test separated end points by 2–26).
- lifetime shift −0.56 / +4.07 / +3.16 (mean +2.22) → **flagged** (> 1.5): lifetime error KN10C −1.5 / +6.3 / −1.4
  (mean +1.1) vs CONTR −0.9 / +2.2 / −4.5 (mean −1.1). A_b −0.11, tran_diff −0.27, eField +0.012 % (mean shifts).
- Secondary, KN10S − STRFRZ50 at 1e4: long_diff +4.5 / +10.9 / +6.4, lifetime −1.4 / +8.3 / +5.0, position −36 to −51
  µm, loss −316 to −465; same direction.
- Consequence: the paper's D_L −4.7 ± 1.0 % (K = 3 for every track) is a representation bias of the 3-mode basis,
  not an optimisation or physics effect. The paper's K choice ("sufficient precision") does not hold for D_L.
  Replacing the paper numbers needs the 16-sample ensemble at knot 10 cm (13 more samples, KN10S + KN10C each).
- **KN10 ENSEMBLE launched 13 Oct (Pierre: "Please do that").** Samples 1, 2, 4, 5, 6, 9–16 (13 more; with 3/7/8, n = 16),
  same protocol as KN10 3/7/8 (KN10S fresh fit at knot 10 cm, then KN10C = CONT continuation to 1.4e4). Seven lanes,
  <= 7 concurrent with ENDFIT, afterany chaining (KNOT/submitted.txt, jobs 40697334–59); outputs on scratch staging.
  **Readout, stated before any result** (read only when all 26 jobs are done): identical to CONT/ensemble_readout.json
  (mean of the last 1e3 iterations at 1.4e4; position = batch median, µm); per parameter mean ± SE and SD over n = 16,
  and the paired shift vs CONTR (n = 16) with its SE and sign count.
  - D_L: the K = 3 offset is a basis artefact on the ensemble if the paired shift is positive at > 3 SE.
  - lifetime: flagged if the paired shift exceeds 2 SE, or the SD grows by more than 30% (2.54 → > 3.3).
  - position and loss reported (expected lower). Which numbers the paper reports is Pierre's decision.

## 14 Oct 2026 — ENDFIT result: FREEDOM COST (rule's third branch) → EFT10 not interpretable yet
Read after all 6 jobs finished (configs verified from the job logs). Paired vs STRFRZ50 1/2/4 at 1e4 (mean of the last 1e3):
| arm | lifetime shift | long_diff shift | eField | position | loss/batch |
|---|---|---|---|---|---|
| ENDSM10 (1 mm smear, fixed) | +14.4 / +11.6 / +12.6 | +14 / +27 / +42 | +0.19% | 845–903 µm | +3 240 |
| EFT10 (1 mm smear, fitted) | +19.7 / +20.5 / +27.8 | +30 / +22 / +50 | +0.08% | 706–776 µm | +1 130 |
| EFT0 (true, fitted) | +5.2 / +11.4 / +8.5 | −4.0 / −1.1 / +2.2 | −0.08% | 139–157 µm (+11) | −93 |
- Rule: EFT0 exceeds ±3 lifetime points on 3/3 → **the added freedom reopens the drift direction by itself**; EFT10
  (worse than fixed smeared endpoints in lifetime, better in loss and position) cannot be interpreted until it is anchored.
- **Mechanism (measured):** mean signed drift offset of the fitted segments from the truth, away from the anode:
  STRFRZ50 −7 / −16 / −9 µm; **EFT0 +44 / +32 / +35 µm**, the endpoint terms alone +70 / +59 / +66 µm (rms ~165 µm per
  track). The fit uses the endpoint freedom for a COHERENT drift shift (lower loss, −93 per batch, while moving away
  from the truth) — the 10–60 µm coherent displacement of §IV that biased the lifetime by +9%. A per-track 1 mm prior
  cannot stop a common shift of tens of µm.
- Next (pre-registered options, not launched): (a) fit only the pixel-plane (e1) endpoint terms, drift endpoints held;
  (b) fit both with a strong batch-level zero-mean constraint on the drift-axis endpoint shifts (the coherent mode
  pinned, per-track shifts free; legitimate if the endpoint reconstruction is unbiased in drift, i.e. t0 known);
  (c) a per-track t0 nuisance. Each needs the EFT0-style control (true endpoints) and the smeared arm.
- **ENDFIT ON HOLD (Pierre, 14 Oct: "put this on the side and just wait for KN10, which is the biggest deliverable").**
  Proposed next step when resumed: option (b), batch zero-mean drift-shift constraint, with true-endpoint control.

## 15 Oct 2026 — KN10 ENSEMBLE RESULT (n = 16; read after all 26 jobs finished; all 32 configs verified in the logs)
Readout identical to CONT/ensemble_readout.json (mean of the last 1e3 iterations at 1.4e4); KNOT/ensemble_readout_kn10.json.
| param | KN10 (10 cm knot) | CONTR (paper, K = 3) | paired shift | positive |
|---|---|---|---|---|
| A_b | +0.185 ± 0.030 (SD 0.12) | +0.268 ± 0.028 | −0.083 ± 0.035 (−2.4 SE) | 4/16 |
| E | +0.032 ± 0.002 (SD 0.009) | +0.022 ± 0.002 | +0.010 ± 0.002 (+4.3 SE) | 13/16 |
| D_T | +0.61 ± 0.27 (SD 1.07) | +1.30 ± 0.30 | −0.69 ± 0.22 (−3.1 SE) | 2/16 |
| D_L | **+1.18 ± 1.11** (SD 4.45) | −4.74 ± 1.02 (SD 4.06) | **+5.92 ± 0.95 (+6.3 SE)** | 15/16 |
| tau | **+0.14 ± 0.79** (SD 3.16) | −1.75 ± 0.64 (SD 2.54) | +1.89 ± 0.88 (+2.2 SE) | 11/16 |
| position | 92.0 µm | 130.8 µm | −38.8 ± 1.5 | 0/16 |
| loss/batch | 8 278 | 8 636 | −357 ± 13 | 0/16 |
- Rule: D_L paired shift positive at 6.3 SE → **the K = 3 D_L offset is a basis artefact on the ensemble**; with ~10
  coefficients per track D_L is consistent with zero. Lifetime FLAGGED (shift 2.2 SE > 2) — it moves TOWARD zero
  (−1.75 → +0.14); its SD grows 2.54 → 3.16 (+24%, below the 30% flag). All five calibration parameters now have
  |mean| ≤ 1.2%; E's small offset grows (+0.022 → +0.032%), A_b's and D_T's shrink.
- Implied exposure for 1%: tau 3.16² ≈ 10 samples (≈ 4.0 km), D_L 4.45² ≈ 20 samples (≈ 7.9 km) (paper: 2.6 / 6.6 km).
- **Lifetime-regime reruns at 10 cm knot, queued 15 Oct (Pierre: "Rerun those lifetime jobs; wait before
  implementing in the paper").** K10T{1200,3200,4400}D{1,3,7}: the STRFRZ50T*D* runs (alternative true lifetimes,
  lifetime started at 2 200 µs) with --chain_spline_knot_cm 10, 1e4 iterations; each continued (K10T*C*, CONT
  protocol, 1.4e4). Jobs 41000239–56 (KNOT/submitted.txt), 7 lanes. Readout (read only when all 18 are done):
  per tau, the three samples' errors at 1e4 paired against STRFRZ50T*D* (the paper's numbers), and at 1.4e4.
  Paper NOT to be updated until Pierre says so.

## 18 Oct 2026 — lifetime-regime reruns at 10 cm knot: read (all 18 done, configs verified in the logs)
Errors (%) on samples 1/3/7; paper K=3 @1e4 | K10 @1e4 | K10 continued @1.4e4 (means):
- tau 1200: lifetime +1.1 | +0.9 | −0.5; long_diff +0.8 | +5.5 | +0.8.
- tau 3200: lifetime −2.7 | +0.7 | −0.3 (per sample −7.1/−3.0/+9.3); long_diff −1.1 | +5.9 | +3.0.
- tau 4400: lifetime −8.1 | −5.6 | −4.9 (−10.2/−8.7/+4.2); long_diff −4.0 | +3.0 | +2.1.
- A_b +0.25–0.31, E +0.02–0.03 %, D_T +0.7–1.2 (continued). The 4 400 µs lifetime deficit shrinks (−8.1 → −4.9) but
  stays on samples 1 and 3 (both start at 2 200 µs, −50%). Paper not updated (Pierre: wait).

## 18 Oct 2026 — paper-support jobs for the 10 cm-knot procedure (Pierre: "Yes"), rules stated before any result
Scripts: KNOT/make_knot_gen.py (fresh rerun of a run with another knot), KNOT/make_cont2_cmd.py (second continuation).
Read each block only when ALL its jobs are done; readout as CONT/ensemble_readout.json (mean of the last 1e3 iterations).
1. **Basis saturation, KN5{S,C}{3,7,8}** (5 cm knot, ~18 coefficients per track; fit + CONT continuation to 1.4e4).
   Paired KN5C − KN10C: **saturated** if |mean Δ| < 1.5 points for both long_diff and lifetime; **not saturated** if
   long_diff or lifetime shifts with the same sign on 3/3 and |mean| ≥ 2; otherwise partial.
2. **Convergence at 10 cm.** (a) K10SCHED{3,7,8}: the SCHED schedule (peak held 1 200 iterations, x0.898/epoch, 1.78x
   travel) at knot 10, 1e4 iterations; paired vs KN10S at 1e4. **Path-invariant** if |mean Δ| < 1.5 (lifetime) and
   < 2 (long_diff) points; K=3 reference: lifetime +7.2 ± 2.4. Loss difference reported (lowest-loss selection).
   (b) KN10D{3,7,8}: second continuation of KN10C (1e-3 over 14 000–16 000, annealed to ~1.3e-5 at 18 000).
   **Converged at 1.4e4** if |mean Δ(1.8e4 − 1.4e4)| < 0.5 points for lifetime and long_diff; else the drift is quoted.
3. **tau = 4400 µs start-at-truth, K10T4400I{1,3,7}** (lifetime started at 4 400; knot 10; fit + continuation
   K10T4400J). Same rule as the K = 3 test: **path dependence** if (start 4400 − start 2200) > +4 points on 3/3 AND
   |mean lifetime| ≤ 3%; **genuine long-tau bias** if mean lifetime ≤ −6%; otherwise inconclusive. Evaluated at 1e4
   (vs K10T4400D) and at 1.4e4 (K10T4400J vs K10T4400C).
- **Offline recomputations on the KN10 fits (18 Oct; NOT in the paper):**
  - figs/kn10/fit_convergence{,_seeds}.{png,pdf} (paperfigs scripts with CONV_PREFIX=KN10C, CONV_OUT_DIR).
  - Floor table, observed SD / floor (floors at truth unchanged): A_b 0.122 → 4.2; E 0.0094 → 19; D_T 1.07 → 2.9;
    D_L 4.45 → 5.2; tau 3.16 → 4.4 (paper K=3: 3.8 / 15 / 3.3 / 4.7 / 3.5).
  - Cost (KN10S 1–16): median 5.03 s per iteration (runs 4.5–5.7), 14.9 h per 1e4 (13.4–16.7), peak GPU 24.8 GB
    median (max 27.6); trajectory coefficients 8 520 per sample (9.9 per track, 864 tracks) vs ~5 000 at K = 3.
  - By-product figure (figs/kn10/byproduct.png; arrays figs/byproduct_kn10; same 13 817 tracks, straight-line and
    truth arrays identical to the K = 3 ones): fitted trajectory median 63 µm (K = 3: 76), p90 220 (249); by length
    < 60 cm 63 (65), 60–100 cm 62 (107), > 100 cm 70 (213) — the K = 3 representation floor of long tracks is gone;
    drift-carrying direction 47 (56). Hit-derived start noisier with more modes: 495 (435) µm median. Segment dE/dx
    3.1% (3.2), track-averaged 1.1% (1.1), mean −0.2%.
  - **Paper switched to KN10C at 1.4e4 (Pierre via the paper session, review R140 option b, 18 Oct).** KN10 versions of
    the three remaining K = 3 figures, in figs/kn10/ (the paper session copies them; NuLads_draft untouched):
    - initialisation.{pdf,png} (fig_initialisation.py, INIT_ARM=KN10C INIT_ITER=14000 INIT_OUT_DIR). Offline re-seed
      of sample 8 batch 35 reproduces the KN10S8 production log line exactly (2 frozen; e2 477 µm, before 294; e1 164).
      Worked track (63 cm, K = 6 sine modes per direction): straight 2.86 mm → hit start 0.69 mm (K = 3: 0.29) →
      fit 48 µm e2 shape, 78 µm mean 3-D distance (119). Inset track 0 (36 cm) still frozen. 16 samples: curved
      tracks e2 727 → 550 µm (724 → 460), e1 728 → 279 (725 → 220); hit start mean 416, median 329 µm (361, 286);
      frozen 1 681 / 13 817 = 12.2% (= the production logs; K = 3 12.4%), their straight-chord e2 error 151 µm mean
      vs raw truth (155); final e2 shape error > 500 µm 1.14% (STRFRZ50 at 1e4: 0.82%; vs raw truth, CONTR at 1.4e4:
      1.01%).
    - event_display.png (event_display.py --ckpt KN10C8 iter 14000, batch 35 event 2; numbers json alongside): data
      638 hits / 8 087 ke; fitted 637.4 / 8 038 (K = 3: 635.6 / 8 029); start 568.9 / 7 091. Longest track: 152
      timing pixels (153), residual +0.18 ± 7.15 ticks, predicted 8.12; start +11.4. True drift bend 4.37 mm; fitted
      e2 RMS vs truth 58 µm, mean 3-D distance 77 µm (119). Segment dE/dx RMS 2.7% (2.9).
    - fit_iteration_time.{pdf,png} (fig_performance.py, PERF_RUN=KN10S8 PERF_OUT_DIR; also rewrites an unchanged
      performance.{pdf,png} there): median 4.56 s (iter ≥ 20), 13.7 h, first iteration 54.5 s, peak 24.9 GB,
      88 stalls of 10–25 s (0.9% of iterations, 2.8% of the time); checkpoint writes at baseline (median 4.53 s).

## 19 Oct 2026 — paper-support blocks, read one by one (readout KNOT/readout_block.py = ensemble_readout protocol)
- **Block 1, basis saturation KN5 (read 19 Oct 10:10, all 6 jobs COMPLETED; configs verified in the logs: knot 5,
  CONT protocol resumed from KN5S at 1e4; 16.3 coefficients per track vs 9.8 at 10 cm). Verdict: PARTIAL.**
  Paired KN5C − KN10C at 1.4e4 (samples 3 / 7 / 8; KN10C readout reproduces ensemble_readout_kn10.json):
  - long_diff −5.69 / +2.19 / +8.65, mean +1.72 ± 4.14 (|mean| ≥ 1.5, signs mixed → neither saturated nor
    not-saturated); lifetime −1.80 / +2.12 / +1.09, mean +0.47 ± 1.17 (< 1.5).
  - A_b −0.09 ± 0.05, eField +0.001, tran_diff −0.95 ± 0.36 (0/3 positive).
  - The 5 cm fits are WORSE fits of the same objective: loss +150 / +168 / +239 per batch (3/3), position
    119 vs 90 µm (+25 to +34, 3/3). Excess in the data terms (no_match_penalty +92 / +140 / +171, tick
    log-likelihood +37 / +9 / +40, matched hits −5 / −8 / −9) and the MCS prior (+16 / +18 / +24; node-based, K-independent).
    KN5 still descending faster (1e4 → 1.4e4 loss −88 vs −38 for KN10): the finer basis is optimisation-limited within
    the 1.4e4 budget. At 1e4 (KN5S − KN10S): long_diff +0.56 ± 4.33, lifetime −0.56 ± 0.91, loss +236 (3/3).
  - Reading: going to 5 cm moves lifetime by less than the rule and long_diff by a mixed-sign amount of the size of
    the sample scatter, while the lowest-loss (truth-free) selector prefers 10 cm on 3/3.
- **Block 3, tau = 4400 µs start-at-truth at knot 10 (read 19 Oct 10:45, all 6 jobs COMPLETED; checkpoints verified:
  I/J start lifetime 4400, D/C start 2200, target 4400 in all; J resumed from I at 1e4 with the CONT protocol).
  Verdict: INCONCLUSIVE at both evaluation points** (path dependence needs > +4 on 3/3; genuine bias needs mean ≤ −6%).
  - At 1e4, I − D (samples 1 / 3 / 7): lifetime +0.59 / +17.35 / +8.72 (2/3 > +4); I mean +3.3 ± 6.9 (D −5.6 ± 4.5).
  - At 1.4e4, J − C: lifetime −0.06 / +14.02 / +7.00 (2/3 > +4); J mean +2.1 ± 6.4 (|mean| ≤ 3) vs C −4.9 ± 4.6.
    J per sample −10.30 / +5.38 / +11.21; C −10.24 / −8.65 / +4.21.
  - Sample 1 is start-independent (−10.3 from both starts, both checkpoints); samples 3 and 7 are strongly
    start-dependent (+14, +7). Loss J − C −22 / +13 / −0.1 per batch (mean −2.9 ± 10.3): end points 7–14 points apart in
    lifetime at equal loss → a flat lifetime direction at 4400 µs, as at K = 3. Other params: long_diff −0.8 ± 2.8 (mixed),
    A_b −0.16 ± 0.11, D_T +0.68 ± 0.61, eField +0.005; position +2.0 µm.
  - Compared with K = 3 (STRFRZ50T4400I at 1e4: diffs +4.4 / +6.3 / +14.0 → PATH DEPENDENCE, mean +0.2): same direction
    on samples 3 and 7, but sample 1 now lands at −10% from the truth start too, so the rule is one sample short.

## 19 Oct 2026 — pinning test at 10 cm knot (Pierre: "Go with run 2, you can send more seeds"), rule stated before any result
- **Block 4, K10HIT{1..6}**: the production 10 cm fit KN10S<s> re-run from scratch WITHOUT drift pinning (no
  LARND_CHAIN_FREEZE_DRIFT, no LARND_FREEZE_STRAIGHT_CL; hit-seeded trajectories all fitted freely). This is the 10 cm version
  of the three-mode HITSEED arm of Fig. recipe, whose provenance differs from STRFRZ50 by exactly these two env vars (checked).
  Builder KNOT/make_knot_hit_cmd.py (KN10S provenance argv + env, test name changed, the two vars unset); the inner scripts
  differ from inner_sci_full_KN10S<s>.sh only there. Code unchanged since 12 Oct 11:37 (before KN10S, 13 Oct): same tree.
  Jobs 41335841/2/3 (s1, s2, s4) now; s3 after KN10D8, s5 after K10SCHED7, s6 after K10SCHED8 (≤ 6 running, 1 slot for
  the paper session). Expected: s1/s2/s4 ~05:00 Tue, s3 ~07:00, s5 ~13:30, s6 ~15:00. Read only when all six are done.
- Readout: `readout_block.py K10HIT 10000 KN10S 10000 1,2,3,4,5,6` (paired, 1e4, no continuation, as Fig. recipe).
- Rule (paired mean shift K10HIT − KN10S, over six samples):
  - PINNING MATTERS at 10 cm if the lifetime or the long_diff shift exceeds both 2 points and 2 SE;
  - PINNING IMMATERIAL at 10 cm if |shift| < 1.5 points (lifetime) and < 2 points (long_diff) — the path-invariance limits;
  - otherwise INCONCLUSIVE. Always report the loss difference (lowest-loss selector), A_b, position and positive counts.
- K = 3 reference (HITSEED − STRFRZ50 at 1e4, samples 1/2/4): lifetime +4.7 ± 2.9 (+7.4 / +7.8 / −1.2), long_diff
  +4.4 ± 4.2, A_b −0.14 ± 0.07, position −9.6 µm, loss −83 ± 11 per batch (3/3 LOWER without pinning). Note: if the same
  holds at 10 cm, the lowest-loss criterion of the appendix would prefer the unpinned end point; flag this to the paper
  session with the verdict.
- **Block 2b, KN10D{3,7,8} second continuation 1.4e4 → 1.8e4 (read 19 Oct 16:25, all 3 COMPLETED rc=0; logs checked:
  resume from KN10C<s> history_iter14000, CONT/param_list_cont.yaml, --iterations 4000, transition_begin 15500,
  dedx_freeze 30000, knot 10). Verdict: NOT CONVERGED at 1.4e4 by the rule (|mean Δ| < 0.5 needed); the drift is quoted.**
  - KN10D@1.8e4 − KN10C@1.4e4: lifetime −1.68 / −3.00 / −0.43 (−1.70 ± 0.74, 0/3 positive); long_diff −2.85 / −2.90 /
    +0.03 (−1.91 ± 0.97); A_b +0.03 ± 0.02; D_T +0.09 ± 0.15; eField −0.002; position −0.6 ± 0.9 µm;
    loss −13.3 / −14.8 / −15.3 per batch (−14.5 ± 0.6, 3/3 lower).
  - Same samples, first continuation (1e4 → 1.4e4): lifetime −2.0 / −3.0 / −2.8 (−2.6), long_diff −2.4 / −5.7 / −3.2
    (−3.8), loss −35 / −46 / −32 (−38). The second step is smaller (lifetime ×0.65, long_diff ×0.5, loss ×0.38), same sign.
  - The movement happens in the re-heated constant-LR phase (14000–16000: lifetime −1.8 / −2.5 / −0.4); over the annealed
    16000–18000 it is flat (+0.1 / −0.5 / +0.0): a stopped fit is not a converged one, as the paper already says.
  - Values at 1.8e4: lifetime −3.2 / +3.3 / −1.8 (mean −0.6 ± 2.0), long_diff −1.9 / +2.7 / −5.8 (−1.7 ± 2.5). Sample 7
    (highest) moves toward zero, sample 3 away from it, sample 8 hardly moves: the step is coherent in sign, not a
    shrinkage toward truth.
  - Paper consequence: the appendix's "may still move by about a point with further optimization" is measured at one to
    two points per further 4000 iterations (lifetime −1.7 ± 0.7, long_diff −1.9 ± 1.0, three samples).
- **Block 2a, K10SCHED{3,7,8} path-invariance at 10 cm (read 20 Oct 01:00, all 3 COMPLETED rc=0; provenance vs KN10S<s>
  differs ONLY in lr_kw (decay 0.89768 + transition_begin 1200 vs 0.91) and test name; env identical; fresh 1e4).
  Verdict: NOT PATH-INVARIANT for the lifetime (+4.1 ± 2.9 > 1.5); long_diff passes (+1.9 ± 3.4 < 2, mixed sign).**
  - K10SCHED − KN10S at 1e4 (samples 3 / 7 / 8): lifetime +9.99 / +0.99 / +1.39 (+4.13 ± 2.94, 3/3 positive; one
    sample carries it); long_diff +8.27 / −3.53 / +1.06 (+1.93 ± 3.44, 2/3); A_b −0.31 / −0.17 / −0.12 (−0.20 ± 0.06,
    0/3; no A_b limit was set at 10 cm, K = 3 used ±0.1); D_T −0.46 ± 0.38; E −0.007 ± 0.001; position +5.9 ± 3.0 µm (3/3).
  - Loss: the longer schedule ends HIGHER on 3/3, +40.8 / +47.7 / +50.7 per batch (+46.4 ± 2.9; K = 3: +17 ± 7), so the
    lowest-loss selector picks the production end point on 3/3. Excess is in the data terms (last-1e3 aux means):
    tick log-likelihood +27.8 / +31.4 / +36.5, no-match +9.1 / +15.5 / +9.9; priors ≤ +4.3 (dedx), mcs ≤ +1.0.
  - Coherent drift offset (geofrz/drift_residuals.py, mean fitted−true drift displacement): SCHED −16.8 / −12.2 / −22.2,
    KN10S −21.3 / −15.9 / −24.4 µm → shift +3.5 ± 0.7 µm (K = 3: +2.1 ± 0.7). NOT the drift-coordinate mode: the shift
    runs along the charge-scale soft direction (lifetime up, A_b down), as at K = 3.
  - Mobility (lifetime error over 4–5k / 5–6k / 7–8k / 9–10k): SCHED3 +15.4 / +13.1 / +11.0 / +10.5, SCHED8 +8.1 / +6.6 /
    +3.6 / +2.9, SCHED7 +12.3 / +10.2 / +9.8 / +10.3; KN10S3 +4.0 / +1.1 / +0.5 / +0.5, KN10S7 flat +9.1–9.3, KN10S8 +3.5 /
    +3.1 / +1.9 / +1.5. The longer schedule freezes mid-descent toward the production end point (same as K = 3).
  - vs K = 3 (SCHED − STRFRZ50): lifetime +7.2 ± 2.4 → +4.1 ± 2.9; long_diff +5.0 ± 2.0 (fail) → +1.9 ± 3.4 (pass);
    A_b −0.24 → −0.20; loss excess +17 → +46 per batch (the likelihood separates the two end points more clearly).
  - Not done at 10 cm: the cross-evaluation / Newton step (SCHED/cross_eval.py) and the continuation of the SCHED end
    points (K = 3: lifetime gap 7.2 → 4.2). Offered to Pierre; not launched.

## 20 Oct 2026 — speed and calibration-polish tests (Pierre: "Please go ahead with those"), rules stated before any result
- **Host-bound check on a running 10 cm fit (K10HIT6, sdfampere020, 11:25):** GPU utilisation mean 15% over 60 s
  (non-zero in 32% of 0.25 s samples); the python process at 97% of one core; job has 2 CPUs, node load 12.
  py-spy 0.4.2 (installed in ~/.local, attached --nonblocking, 60 s, SPEED/prof_K10HIT6_idle.txt): main thread 82% inside
  value_and_grad of compute_loss, of which trace_to_jaxpr 54% and MLIR lowering 38% (+ persistent-cache lookups 7%) —
  the loss is re-traced and re-lowered EVERY iteration (eager path); _compute_pos_residual_batch (truth diagnostic,
  pos_residual_freq default 1) 13.5%; _chain_warp_spline 16%; predict 16%. An existing, unused option does the fix:
  LARND_JIT_LOSS=1 (3e775ae, July: cached jitted value_and_grad per batch; CPU A/B losses agree to 1.4e-6; off by default).
- **Speed bench (SPEED/, jobs 41413159-61):** KN10S1 config shortened to --max_nbatch 10 --iterations 300 (10 compiles,
  then steady): BENCHE2 (production, 2 CPUs), BENCHE8 (8 CPUs, 8g/CPU), BENCHJ2 (LARND_JIT_LOSS=1). All on one node.
  Readout: median step_time over iterations 100–300, first-epoch overhead (sum of steps 0–9 − 10 × steady median), GPU
  utilisation (in-job nvidia-smi), and agreement of the loss/calibration traces with BENCHE2.
  - JIT ADOPTABLE if steady median ≤ 0.7 × BENCHE2, iteration-0 loss agrees to < 1e-5 relative and the calibration at
    iteration 300 agrees with BENCHE2 within 0.1% (relative), and 100 compiles + 1e4 jitted steps < 1e4 eager steps.
  - 8 CPUs HELP if the steady median ≤ 0.9 × BENCHE2.
- **Newton polish (POLISH/newton_polish.py):** one conditional Newton step on the 5 calibration parameters of each of the
  16 KN10C end points at 1.4e4 (nuisances of the final checkpoint fixed; exact Hessian from 5 HVPs per batch; data loss
  re-evaluated at the new point). Smoke test 41413375 (KN10C8, 3 batches) first.
  - Step VALID if H is positive definite and the data loss decreases at theta_1 on ≥ 14/16 samples.
  - IMPROVES if the 16-sample SD of the lifetime and of the long_diff errors each drop by ≥ 10% and neither |mean|
    grows by more than 1 SE(theta_0); WORSENS if either SD rises by ≥ 10% or either |mean| grows by > 2 SE; else NEUTRAL.
    All five parameters reported.
  - Per-fit uncertainty: RMS pull (theta_0 − truth)/sigma over the 16 samples for sigma = conditional H^-1 and the two
    sandwiches; CALIBRATED if the RMS pull is within [0.7, 1.4]. Reported with the median sigma vs the observed SD
    (Table floor: lifetime 3.2%, long_diff 4.4%, A_b 0.12%, D_T 1.1%, E 0.009%).
- **Speed bench round 1 (read 20 Oct 12:30; all three on sdfampere011).**
  - BENCHE2 (production code, 10 batches): steady median 2.34 s (iterations 100–300), GPU util 46% (busy 74% of samples).
  - BENCHE8 (8 CPUs): 2.23 s (0.95×) → **8 CPUs do NOT help** (rule: ≤ 0.9×). Same iteration-0 loss (25508.2637).
  - BENCHJ2 (LARND_JIT_LOSS=1): **NOT ADOPTABLE as is** — ~170 s per compile, recompiles when the cached ROI padding
    changes (epoch 2: roi (512, 1024) vs (384, 768) in epoch 1), and the job died OUT_OF_MEMORY (64 GB host) after 15
    compiles, at iteration 15. No steady-state step reached.
  - **Main finding: per-iteration cost depends on the NUMBER OF DISTINCT BATCHES.** The same batches 0–9 take 4.7–6.8 s
    (median ~5.1) per iteration in KN10S1 (100 batches, iterations 100–3000) but 1.7–2.6 s (median 2.2) in BENCHE2
    (10 batches): 2.2× overhead at 100 batches, same code/config/batches. KN10S1/8 at iterations 100–300: 4.96 / 5.02 s.
    Hypothesis: JAX's in-memory trace/jvp/lowering caches (weakref_lru_cache, default maxsize 2048 in jax 0.6.2) are too
    small for 100 batch shapes in the eager value_and_grad path, so entries are evicted and re-lowered every iteration
    (profile: lowering 38%, persistent-cache lookups 7%).
- **Round 2 (SPEED/cachepatch/sitecustomize.py: import hook raising the weakref_lru_cache default, bench-only, nothing in
  site-packages modified; cache fill/hit/miss logged every 60 s):** KN10S1 config at the full 100 batches, 300 iterations.
  BENCHL32K (maxsize 32768) 41417695; BENCHL2K (2048 = default, diagnostics only) 41417696.
  - Rule: CACHE FIX CONFIRMED if BENCHL32K's median step over iterations 100–300 is ≤ 0.7 × the KN10S ensemble's median
    over the same iterations (16 runs) and its iteration-0 loss equals KN10S1's; BENCHL2K should show caches at capacity
    with misses growing. Also report host memory (MaxRSS).
  - (Polish method note, logged late: the smoke test 41413375 died OOM — forward-over-reverse HVP needs > 40 GB at 400 cm
    — so before the array the Hessian was switched to central finite differences of the AD gradient in ln theta,
    steps Ab 5e-4, eField 1.5e-5, tran_diff 1e-2, long_diff 2e-2, lifetime 1e-2; smoke2 41413784: asymmetry ~1e-5,
    actual data-loss change −0.97 = quadratic prediction −0.97. The rule above is unchanged.)
- **Round 2 result (read 20 Oct 14:30, both jobs done; step_time from the history pickles):**
  - BENCHL32K median step over iterations 100–300: 4.84 s vs the KN10S ensemble 5.29 s (16 runs, range 4.96–5.65) =
    **0.92× → CACHE FIX NOT CONFIRMED** (rule: ≤ 0.7×). Iteration-0 loss 25508.2637 in both arms and KN10S1 (rule met).
    BENCHL2K (default 2048): 5.17 s. MaxRSS **33.1 GB** (L32K) vs 8.8 GB (L2K) vs 7.4 GB (BENCHE2); 64 GB requested.
  - BENCHL2K as expected: 16 of 50 caches at capacity, ~30k misses/min in steady state, ~95% of them in full caches.
    BENCHL32K: no cache full, misses fall to ~1.3–1.8k/min in epochs 2–3 but never stop — entries keep growing
    (322k at iteration 300), i.e. a fraction of the keys is NEW every iteration (unbounded memory).
  - Not in the rule, so not a result: the gain appears only in epoch 3 — L32K median 5.73 s (100–200) then 3.31 s
    (200–300; 0.63× of KN10S's 5.29), while L2K stays at 5.11–5.24; and L32K's epoch 1 took 82.7 min vs 24.9 min
    (TotalCPU ≈ elapsed, so not CPU starvation; candidate: cyclic GC over the larger heap). KN10S's own step time
    stays ~5 s at all depths (5.29 / 5.22 / 5.26 / 5.17 / 5.03 / 4.73 s over 100–300 … 9000–10000).
- **Round 3 (rule stated before any result):** same config, **600 iterations** (epochs 4–6 steady), SPEED/cachepatch3
  (as cachepatch, plus every 120 s: RSS, cumulative cyclic-GC time per generation, and the 6 caches with the most new
  misses, named). BENCHR3A maxsize 32768; BENCHR3B maxsize 8192 (memory-bounded candidate).
  - Per arm: ADOPTABLE if (i) median step over iterations 300–600 ≤ 0.7 × the KN10S ensemble median over the same
    iterations (5.22 s → ≤ 3.65 s); (ii) RSS extrapolated linearly from its iteration-300→600 slope to 1e4 iterations
    stays < 60 GB; (iii) iteration-0 loss equals KN10S1's; (iv) the extra epoch-1 time vs KN10S1 is repaid within 1e4
    iterations at the measured saving. Otherwise report which condition failed.
  - Epoch-1 slowdown attributed to GC if gen-2 GC time is > 30% of epoch-1 wall time; else unexplained.
  - The cache names with persistent misses at 300–600 are reported as the code paths that create new keys each step.
- **Newton polish result (read 20 Oct 17:05, all 16 done, 100/100 batches each; POLISH/read_polish.py):**
  - **Step VALID**: H positive definite 16/16 (min eigenvalue 1.0–1.8e4); data loss decreases at theta_1 16/16, actual
    = quadratic prediction (e.g. −16.9/−18.4, −4.7/−4.6, −13.8/−13.8; total −1.7 to −17 per sample over 100 batches).
  - Step (mean ± SD over 16, %): lifetime −1.70 ± 0.74, long_diff −1.52 ± 0.90, A_b +0.040 ± 0.029, D_T +0.02 ± 0.26,
    E +0.001 — coherent, not noise: the reported end point lags its conditional optimum in lifetime and long_diff.
  - Errors before → after (mean, SD over 16, %): lifetime +0.14, 3.16 → −1.56, 2.87 (SD ×0.91; |mean| +1.79 SE);
    long_diff +1.18, 4.45 → −0.36, 3.97 (×0.89; |mean| −0.74 SE); A_b +0.185, 0.122 → +0.225, 0.114 (×0.93; +1.3 SE);
    D_T +0.61, 1.07 → +0.63, 0.91 (×0.85); E unchanged. → **NEUTRAL** (IMPROVES needs both SDs −10%: lifetime −9%;
    WORSENS needs |mean| +2 SE: lifetime +1.79). Not adopted as a recipe step.
  - Per-fit uncertainty: **NOT CALIBRATED**. RMS pull, conditional H^-1: A_b 7.4, E 80, D_T 3.3, D_L 5.5, τ 4.0;
    sandwich: 5.7, 31, 2.9, 2.9, 3.0 (rule [0.7, 1.4]). Median conditional sigma at the fits' own end points: A_b 0.029,
    D_T 0.38, D_L 0.82, τ 0.77% — the Table floor values (0.029, 0.37, 0.86, 0.72) evaluated at truth. So the paper's
    "curvature at fixed nuisance parameters underestimates the spreads by three to five (nineteen for E)" also holds at
    the fitted end points (pulls 3–5.5; A_b's 7.4 includes its +0.19% offset, bias-removed 4.1).
  - Not in the rule: on samples 3/7/8 the Newton step (lifetime −2.36/−1.82/−1.91, long_diff −2.41/−2.46/−1.69) points
    the way KN10D's 4000 further iterations moved (−1.68/−3.00/−0.43; −2.85/−2.90/+0.03): mean −2.0 vs −1.7 (τ),
    −2.2 vs −1.9 (D_L). The non-convergence of Block 2b is the calibration lagging its conditional optimum.
- **Block 4 result (read 20 Oct 17:15; all six K10HIT COMPLETED, 1e4 iterations each; KNOT/readout_k10hit_vs_kn10s.json):
  INCONCLUSIVE by the rule** (19 Oct section). Paired shift K10HIT − KN10S over samples 1–6:
  - lifetime **+3.17 ± 1.82** (−1.98 / +3.53 / +10.41 / +0.70 / +5.83 / +0.53; 5/6 positive) — above 2 points but
    1.74 SE (MATTERS needs > 2 SE); long_diff +2.28 ± 3.71 (−6.89 / −1.78 / +9.50 / −2.31 / +17.11 / −1.96; 2/6).
  - Values: lifetime unpinned +2.94 ± 2.07 (SD 5.07) vs pinned −0.23 ± 1.00 (SD 2.45); long_diff +5.82 ± 2.70 (SD 6.62)
    vs +3.54 ± 1.25 (SD 3.05). Without pinning the spread doubles for both.
  - A_b −0.090 ± 0.045 (1/6 positive), D_T +0.18 ± 0.61, E −0.008 ± 0.010; position −10.4 ± 3.7 µm (0/6 positive:
    better without pinning, as the 150 µm chords of the pinned tracks imply); **loss −65 ± 10 per batch, 6/6 LOWER
    without pinning**.
  - Same pattern as the three-mode reference (lifetime +4.7 ± 2.9, long_diff +4.4 ± 4.2, A_b −0.14, position −9.6 µm,
    loss −83 ± 11, 3/3 lower). The lower loss without pinning is the expected cost of removing freedom (the pinned fit
    is a constrained minimum of the same objective, so its loss cannot be lower at a common optimum); a loss comparison
    selects between end points of the SAME model (schedules), not between models with different constraints.
- **Steady-state profile of BENCHR3A (LRU 32768, iteration ~330, py-spy 90 s, SPEED/prof_R3A_steady.txt):** of the main
  thread, 75% in the eager value_and_grad (compute_loss:2428); within it, the dynamic-size unique / append / pad of
  simulate_wfs (sim_jax.py:797–799, pad_to_closest_multiple) take **27.6%** (lowering 12.8%, persistent-cache lookups
  16.9%): their output size is the exact active-pixel count, new every iteration; and **_compute_pos_residual_batch
  21.3%** (a Python loop over every point, fit_params.py:4471–4476). Also: the ROI padding takes a running GLOBAL max
  over batches, which grows through epoch 1 ((256,768) at batch 0 → (768,1280) at batch 93, vs (512,1024) for the
  10-batch bench) — so every epoch-1 entry is stale in epoch 2 (L32K epoch 2 5.73 s, epoch 3 3.31 s), and the 10-batch
  bench also computes with smaller ROI padding than 100 batches (round-1 "2.2×" is not all cache).
- **Two env-gated fixes (default OFF => identical behaviour), staged in SPEED/stage/ (symlink tree of the live code with
  only these two files replaced; the live tree is untouched until adopted):**
  - LARND_UNIQUE_BUCKET=1 (sim_jax.simulate_wfs): count the unique pixels n (jitted), then the existing static-size
    unique with size ((n + 1 + 127) // 128) · 128 — the dynamic path's own size — so the array is identical (−1 fill
    first after the sort, then the n ids) and the ops change shape only at 128 boundaries.
  - LARND_POSRES_VEC=1 (fit_params._compute_pos_residual_batch): one numpy expression per track; the first 3 calls also
    run the loop and raise if any point differs by > 1e-9 relative. Unit test (container numpy 2.2.6, 2000 random
    tracks): max relative difference 3e-16, mean identical.
  - Equivalence test of LARND_UNIQUE_BUCKET (SPEED/eqtest, job 41438948): eager value_and_grad w.r.t. (calibration,
    tracks) on slices of KN10C3 batches 0 and 5, flag 0 vs 1, deterministic GPU ops: must be bitwise identical in
    unique_pixels, wfs, loss and all gradients, or the round-4 benches are not run.
- **Round 4 (rule stated before any result):** KN10S1 config, 100 batches, 600 iterations, both fixes on, staged code,
  cachepatch3 logging. BENCHR4A default caches (2048); BENCHR4B maxsize 8192.
  - Per arm ADOPTABLE if (i) median step over 300–600 ≤ 0.7 × the KN10S ensemble (5.22 s → ≤ 3.65 s); (ii) RSS
    extrapolated to 1e4 iterations < 60 GB; (iii) iteration-0 loss and pos_residual_iter[0] equal KN10S1's bitwise;
    (iv) the extra epoch-1 time is repaid within 1e4 iterations. Attribution: R4A vs KN10S (both fixes alone),
    R4B vs R3B (fixes at equal cache size).
  - **Equivalence test PASSED** (41438948, 6 min): flag 0 vs 1, 18 arrays (unique_pixels, wfs, loss, the 5 calibration
    gradients and the 0.7 M-entry track gradient, batches 0 and 5 of KN10C3, 3 eager value_and_grad calls each) are
    bitwise identical. Round 4 submitted: BENCHR4A 41439809 (LRU 2048), BENCHR4B 41439810 (LRU 8192); outputs symlinked
    to fit_result_staging. BENCHR3A finished (02:02:00) and was moved to staging; R3 is read when R3B is done.
- **Round 4 result (read_r3.py + SPEED/r4_extra.py, r4_repro.py): NOT ADOPTABLE by the rule — both fixes are a real
  but smaller gain (0.80×), and a larger cache is HARMFUL.**
  - BENCHR4A (fixes, default caches, sdfampere013): median step 300–600 **4.16 s = 0.80×** the KN10S ensemble (5.22 s;
    the 16 KN10S samples span 4.89–5.82 s, SD 0.26, so the gain is ~4 SD and real), flat over 100–600 (4.05–4.21);
    (i) FAIL (≤ 3.65 needed). (ii) PASS: RSS 8.3 GB at 600, +0.04 GB/100 it → 12 GB at 1e4. (iii) losses bitwise equal
    to KN10S1 for the first 5 iterations (PASS); pos_residual_iter[0] differs by 7.6e-8 relative — but the UNPATCHED
    code does not reproduce it bitwise either (round-2 replicas BENCHL2K/L32K: 1.6e-9, 6e-10; all four replicas
    differ from KN10S1 at iterations 1–3 by 1e-9–2e-8): the run is non-deterministic on the GPU and iteration 0's
    value is computed after the first update by the original loop (the vectorised path takes over after 3 checks), so
    this check was mis-specified and cannot separate the fixes from run-to-run noise. [POSRES-VEC] checks 3/3 passed in
    both arms. (iv) PASS: epoch 1 23.1 min vs 24.0 (no extra time), saving ~2.9 h per 1e4-iteration fit.
  - BENCHR4B (fixes, LRU 8192, sdfampere032): **10.43 s = 2.00× SLOWER**, epoch 1 46 min, 2.3 h for 600 iterations.
    Not a cache effect in the counters: per iteration R4B has FEWER misses (2150 vs 2500) and more hits (1730 vs 1580)
    than R4A, the hit rate is ~40% in both, and gen-2 GC is 1% of epoch 1 in both; the nodes are identical hardware
    (EPYC 7542, A100-40GB, driver 535). [WITHDRAWN after round 3: "most misses are on jaxprs created fresh by each re-trace, which no capacity can
    hit" is WRONG — at 32768 the misses fall to a few per iteration (R3A below), so they are capacity misses.] Why a larger cache slows the step is not identified; R3B (also 8192, sdfampere020) is the
    second sample and is still running (2:36 for < 600 iterations, vs R3A 32768 2:02).
  - Next: steady-state py-spy of the patched code (BENCHR5P 41449503, R4A config, 300 iterations, 120 s profile at
    iteration ~150) to see what the remaining 4.2 s is.
- **Round 3 result (read 20:03, both done; read_r3.py): BENCHR3A (LRU 32768, unpatched code) ADOPTABLE by the rule;
  BENCHR3B (8192) fails.**
  - R3A (sdfampere011): median step 300–600 **2.60 s = 0.50×** KN10S, still falling (100–200 5.57, 200–300 3.12,
    300–400 2.82, 400–500 2.51, 500–600 2.46 s). (i) PASS. (ii) PASS but marginal: RSS 32.4 GB at 600, +0.28 GB/100 it
    → 58.9 GB at 1e4 (job limit 64 GB); the caches are still filling (25–28 k of 32768 entries at 600). (iii) PASS.
    (iv) PASS: epoch 1 81 min vs 24 (+57 min) against a 6.8 h saving per 1e4 iterations. GC is 0% of epoch 1 (not GC).
    At 300–600 the misses are ~5 per iteration per cache (the largest: convert_constvars_jaxpr 1505, _infer_params_cached
    1436, _prune_closed_jaxpr_outputs 1081 over 300 iterations): a small residue of new keys per step.
  - R3B (sdfampere020): **12.90 s = 2.47× SLOWER**, epoch 1 59 min; full caches thrash (~200 misses per iteration in each
    of the top caches). With R4B (8192 + fixes, 10.4 s) this is two nodes: 8192 is reproducibly worse than the default.
  - Reading of rounds 2–4: the working set of the eager path (100 batches) is between 8192 and ~30 k entries per cache.
    Below it the caches thrash, and a miss costs MORE the more entries are retained (epoch 1, all misses: 24 / 59 / 81 min
    at 2048 / 8192 / 32768; mechanism not identified, GC excluded) — hence 8192 is worse than 2048. At 32768 the working
    set fits and, once it is filled (~3 epochs), the step falls to ~2.5 s. Round 2's 0.92× was the 100–300 window,
    before the fill.
- **Round 5 (rule stated before any result):** staged code with both fixes + LRU 32768, KN10S1 config, 100 batches:
  BENCHR5A 600 iterations, BENCHR5B 1500 iterations (memory and long-run check).
  - ADOPTABLE (to propose for production) if (i) median step 300–600 ≤ 3.65 s in both arms; (ii) RSS extrapolated from
    R5B's iteration 600→1500 slope to 1e4 iterations < 50 GB (stricter than round 3: the production jobs have 64 GB);
    (iii) losses_iter[0:5] equal KN10S1's bitwise (CHANGED from round 4's pos_residual_iter[0], which the unpatched
    code does not reproduce run-to-run — the replicas agree with KN10S1 bitwise in the losses for 5 iterations);
    (iv) the extra epoch-1 time vs KN10S1 is repaid within 1e4 iterations. Attribution: R5A vs R3A (fixes at equal
    cache size). Also reported: R5B median over 1000–1500 and its cache fill at 1500.
- **Jit-ceiling micro-bench (SPEED/mbjit/mbjit.py, job 41449681, 23 min, read when done): a jitted data-as-args loss is
  NOT worth building — warm eager is already within 7% of it.** KN10C3 @ 14000, batches 0–3 (all 70174 rows; U 896–1024;
  ROI (256,896)), value_and_grad w.r.t. (calibration, tracks) of the PPP data loss:
  - EAGER, caches warm (4 batches ≪ 2048 entries): **1.13 s/call** (1.08–1.33).
  - JIT with tracks and target as ARGUMENTS, unique_size/ROI static: **1.055 s/call**; loss bitwise equal to eager on all
    4 batches; but **compile 132–149 s per batch shape** and **+2.9 GB host RSS per compiled program** (100 batches →
    ~4 h of compiles and ~290 GB); each persistent-cache entry is 1.34 GB EVEN WITH THE DATA AS ARGUMENTS, so the
    "entries embed the batch data" explanation in example_run.py's header is not what makes them large (some constant
    built at trace time; not pursued). GPU autotuning off (--xla_gpu_autotune_level=0): compile 142 s, 1.066 s/call —
    no effect.
  - The eager path's own executables are small (0.0–0.4 MB each in the persistent cache).
  - So the step floor is the GPU compute of the data loss (~1.1 s per 400 cm batch); the production step's excess over
    it (5.2 s at the default caches) is cache-capacity thrash + the two Python/host costs fixed above, which is what
    rounds 3–5 address. The data-as-args jit (the "next big lever" of the round-3 profile) is dropped.
- **BENCHR5P (2048 + both fixes, sdfampere032 = R4B's node; py-spy 120 s at iteration ~200, SPEED/prof_R5P_steady.txt):**
  epoch 1 24.2 min, median step 100–200 4.58 s, 200–300 4.67 s (R4A, same config on sdfampere013: 4.14 / 4.05) — the
  node is ~12% slower, NOT 2.5×: R4B's 10.4 s is the cache size, not the node. Profile (main thread; the logger thread's
  time.sleep is miscounted as active by --nonblocking py-spy and is excluded): compute_loss 88%, of which tracing the
  loss (loss_wrapper_combined) 56%, MLIR lowering + compile-cache lookups ~35% (mlir __init__/make_ir_context/
  lower_sharding_computation/lower_jaxpr_to_fun, compilation_cache), waiting on executions (_pjit_call_impl) 22%;
  _compute_pos_residual_batch 0.4% (was 21.3%), simulate_wfs 0.6% (unique/pad sites were 27.6%). What remains at
  the default caches is re-lowering of evicted entries — what LRU 32768 removes (round 5).
- **Round 5 result (read 22:05, both done; SPEED/read_r5.py, r5_repro.py): 0.32× — 5.2 → 1.7 s per iteration, flat to
  1500 iterations, RSS plateaued at 28 GB. (i), (ii), (iv) PASS in both arms; (iii) as written PASSES in R5B and FAILS
  in R5A (bitwise for 3 iterations, not 5).**
  - BENCHR5A (600 it, sdfampere020): median 300–600 **1.70 s = 0.33×** (100–200 3.85, 200–300 1.67, then 1.69–1.71).
    BENCHR5B (1500 it, sdfampere032, the ~12%-slower node): 300–600 **1.66 s**, 600–1000 1.67, **1000–1500 1.69 s =
    0.32×**. Attribution at equal cache size: R5A vs R3A 1.70 vs 2.60 s — the two fixes are worth 0.65× on top of the
    cache.
  - (ii) RSS: R5B 28.2 GB at 1462, slope 0.003 GB/100 it over 600–1500 → **28.5 GB at 1e4** (< 50: PASS); R5A 28.2 GB.
    The fixes stop the growth (R3A, no fixes: 32.4 GB and +0.28 GB/100 it). Caches: none at capacity, ~22–26 k entries
    of 32768; at 600–1500 only _infer_params_cached still misses (~2 per iteration).
  - (iv) epoch 1 77 / 69 min vs 24 (+53 / +45 min) against a 9.2–9.3 h saving per 1e4 iterations: PASS. A 1e4-iteration
    10 cm fit: ~14.5 h (KN10S) → ~1.2 h + 9900 × 1.7 s ≈ 5.9 h.
  - (iii) losses_iter vs KN10S1 over ALL nine bench replicas (r5_repro.py): every replica differs from KN10S1 by exactly
    1 ulp by iteration 5 (all at −6.0e-8 relative at iteration 5 — KN10S1 is the odd one there); R4B and R5A (both with
    the fixes) already differ by +1 ulp at iteration 3; the other seven (R4A, R5P, R5B with the fixes; L2K, L32K, R3A,
    R3B without) are bitwise for 5. Iteration 3 vs 5 is 2 of 5 fixes arms vs 0 of 4 unpatched (not significant, Fisher
    p ≈ 0.4) at the level of single-ulp GPU-atomics noise, but the rule as written fails for R5A, so it is settled
    directly by the next test rather than by argument.
- **Deterministic equivalence test of the full fit (rule stated before any result):** the KN10S1 config WITHOUT
  --non_deterministic (example_run then sets --xla_gpu_deterministic_ops=true), 10 iterations, two jobs on two nodes,
  each running both arms in swapped order: DET0 = live code, default caches; DET1 = staged code, both fixes, LRU 32768.
  The fixes + cache are NUMERICALLY NEUTRAL if (a) each arm reproduces itself bitwise across the two jobs (else the test
  is uninformative and says so) and (b) DET0 and DET1 agree bitwise in losses_iter, the five calibration *_iter traces
  and pos_residual_iter over all 10 iterations.
  - **Result (DETa 41457389 sdfampere014, DETb 41457390 sdfampere026; det/det_compare.py): UNINFORMATIVE — (a) fails.**
    Neither arm reproduces itself across the two jobs: DET0a vs DET0b differ in losses_iter at iterations 5, 8, 9
    (≤ 7.4e-8 relative = 1 ulp), long_diff at 1.1e-7, pos_residual 5e-8; DET1a vs DET1b at iterations 3, 5. The
    differences are SPORADIC single ulps that do not cascade (A_b, E, D_T and lifetime traces bitwise identical in
    every pair), i.e. non-deterministic reductions inside the loss, not a diverging trajectory; DET0 vs DET1 differ in
    the same way and at the same size. So with --non_deterministic off the production fit is still not bitwise
    reproducible — example_run sets XLA_FLAGS inside main(), possibly after the GPU backend has started.
  - **Rerun (same rule): XLA_FLAGS=--xla_gpu_deterministic_ops=true exported before python starts** (det/inner_DETX*,
    jobs DETXa / DETXb, same arm order). If (a) still fails, the bitwise criterion is not attainable for the full fit
    and the numerical-neutrality evidence is the unit-level equivalence test (bitwise loss/gradients, 41438948) plus
    the agreement of all replicas to 1 ulp.
  - **Rerun result (DETXa 41457872, DETXb 41457873): both FAILED at start-up — with the flag really in effect the fit
    cannot run.** With XLA_FLAGS=--xla_gpu_deterministic_ops=true in the environment, simulate_signals (target
    generation, precompute_static_pixels) asks for 40.7 GB in one allocation → RESOURCE_EXHAUSTED on the 40 GB A100
    (the deterministic scatter path). Two consequences:
    1. example_run's own `os.environ['XLA_FLAGS'] = ...deterministic_ops=true` (in main(), when --non_deterministic
       is absent) is IGNORED — the backend is already up — otherwise the DET runs above would have died the same way.
       A run without --non_deterministic is therefore not deterministic either (sporadic 1-ulp loss differences).
    2. Bitwise reproducibility of the full 400 cm fit is not attainable on these GPUs, so (a) cannot be met and the
       full-fit bitwise test is closed as UNINFORMATIVE, as pre-stated.
  - **Verdict on round 5:** (i), (ii), (iv) PASS in both arms; (iii) as written passes in R5B and fails in R5A, and
    could not be settled bitwise. The evidence on numerics, per the pre-stated fallback: the unit-level deterministic
    test of LARND_UNIQUE_BUCKET (bitwise loss, wfs and all gradients, 41438948); LARND_POSRES_VEC is a diagnostic
    (no effect on the fit; checked per point in every run); the cache size changes only what is cached, not the
    computation; and over 10 iterations the patched and unpatched fits differ from each other exactly as much as
    each differs from itself across runs (single ulps in the loss; A_b, E, D_T and lifetime traces bitwise identical).
    The bench trajectories diverge chaotically after that (1e-6 to 1e-2 in the loss by iteration 200 in ALL replicas,
    patched or not), so equality of END POINTS can only be tested statistically on full fits — not done.
- **Full-length validation of the adoption bundle (rule stated before any result):** KN10V1–3 = KN10S1–3 exactly (same
  argv, env, data seeds, 1e4 iterations; the inner scripts differ only in test_name) plus LARND_JAX_LRU=32768 (now set
  by the hook in example_run.py itself — SPEED/adopt/example_run.new.py, copied into SPEED/stage — not by
  cachepatch3), LARND_UNIQUE_BUCKET=1, LARND_POSRES_VEC=1, staged code. Readout: KNOT/readout_block.py KN10V 10000
  KN10S 10000 1,2,3 (mean of the last 1e3 / truth − 1; position median of the last 1e3; loss mean of the last 1e3).
  - SPEED: median step over 1000–1e4 and total wall vs the same KN10S sample; sacct MaxRSS < 50 GB.
  - RESULTS: per sample, Δ = V − S for the five calibration errors and position, against the KN10S 16-sample SD of the
    same quantity. NEUTRAL if every |Δ| < 0.5 SD (5 parameters + position × 3 samples); otherwise INCONCLUSIVE —
    there is no pair of unpatched replicas run to 1e4, so the run-to-run spread of an end point on the SAME data is
    unknown and a larger Δ cannot be attributed to the patch (it would then call for an unpatched replica).
  - **Result (21 Oct, all three done; SPEED/valid/read_valid.py, traj_valid.py):** SPEED holds over the full fit —
    median step 1000–1e4 1.65 / 1.77 / 1.71 s vs 5.06 / 5.21 / 5.40 s (0.32–0.34×); total wall 5.9 / 6.1 / 6.0 h vs
    15.1 / 15.5 / 16.2 h (0.37–0.40×, epoch 1 included: 67–72 vs 24–26 min); MaxRSS 28.4 / 28.4 / 28.7 GiB (< 50 GB);
    per-job disk cache ~100 MB, removed at exit. RESULTS by the rule: max |Δ|/SD = 1.85 → **INCONCLUSIVE**.
    Δ(V − S) per sample, % of truth (SD = KN10S 16-sample SD): Ab +0.25 / −0.18 / +0.09 (SD 0.13); eField +0.010 /
    −0.008 / −0.006 (0.010); tran_diff +1.57 / −0.23 / −0.09 (1.05); long_diff −8.8 / +6.0 / −3.5 (5.2); lifetime
    −5.9 / +7.1 / −3.3 (4.0); position +0.4 / −6.2 / +3.5 µm (8.2). Median |Δ|/SD over the 15 calibration numbers 0.97.
    No parameter has the same sign in all three samples. Trajectories: losses equal for 0–3 iterations, |Δloss|/loss
    1e-6 at 100 iterations, 0.04–0.27 at 1000 (macroscopic); window-mean Δ of the calibration parameters CHANGE SIGN
    during the fit (e.g. s2 lifetime −3.9 % at 0–1k → +8.8 % at 5–6k; s1 Ab −0.20 → +0.27) and freeze once the LR
    anneals (the last-1000 within-run SD is 0.001–0.05 %, ≪ Δ). Reading (not a verdict): consistent with chaotic
    divergence from 1-ulp seeds; if so, the same-data end-point scatter is of the order of the 16-sample SD itself.
    Code provenance: KN10S1/S2 ran the current live tree (a5d7bdc + the dirty fit_params.py of 12 Oct 11:37); KN10S3
    started 12 Oct 01:54, before that edit — its loss equals KN10V3's at iterations 0 and 10.
- **Replica test (rule stated before any result):** KN10V{1,2,3}{b,c} = KN10V{1,2,3} exactly (inner scripts differ
  only in test_name), 6 jobs. Per sample, 4 runs on the same data: V, Vb, Vc (patched) and S (unpatched).
  - Statistic, per quantity q (5 calibration errors + position): R_q = RMS of the 9 cross pairs (V−S, Vb−S, Vc−S over
    3 samples) / RMS of the 9 same-code pairs (V−Vb, V−Vc, Vb−Vc); T = mean over the 6 q of R_q. Null by
    exchangeability: recompute T for all 4³ = 64 choices of which run per sample plays "S"; p = fraction ≥ observed.
  - **NEUTRAL if p > 0.10; SHIFT if p ≤ 3/64 (observed among the top 3); INCONCLUSIVE between.** Per-q R_q and
    p reported as description. NEUTRAL → propose adoption. SHIFT → not adoptable; find which fix.
  - Descriptive, also pre-stated: same-data run-to-run SD per quantity = pooled within-sample SD of the 3 patched
    runs (6 dof), and its ratio to the KN10S 16-sample SD; lowest last-1e3 loss of the 4 runs per sample.
  - Departure from the earlier rule ("would call for an unpatched replica"): patched replicas instead, because they
    test the same exchangeability (a patch shift would make V, Vb, Vc agree with each other and differ from S) at
    6 h instead of 15 h per fit. Residual assumption: unpatched and patched runs have similar run-to-run scatter.
  - Readout script fixed before results: SPEED/valid/read_replica.py. Its calibration on synthetic Gaussian runs
    (200 trials each): no shift → NEUTRAL 91 %, SHIFT 4 %; S shifted on every quantity by 1 / 2 / 3 run-to-run SDs
    (random sign) → SHIFT 32 / 97 / 100 %. So it detects a patch shift ≥ 2 run-to-run SDs, not one of ~1.
  - Submitted 21 Oct 04:55: KN10V1b 41484832, KN10V1c 41484833, KN10V2b 41484834, KN10V2c 41484835, KN10V3b 41484836,
    KN10V3c 41484837.
- **Newton polish of the same-data runs (Pierre, 21 Oct: "start those whenever possible"; rule stated before any
  result):** POLISH/newton_polish.py (unchanged but for a --suffix option naming the replicas), --iter 10000, 100
  batches, on the 12 runs KN10V{s}, KN10V{s}b, KN10V{s}c, KN10S{s}, s = 1–3. Readout POLISH/replica/read_polish_replica.py.
  - Step VALID per run if H is positive definite and the data loss decreases at theta_1; reported as n/12.
  - Same-data scatter per parameter: pooled within-sample SD of the error over the 4 runs (9 dof), before (theta_0)
    and after (theta_1); r = after / before. If the replica test reads SHIFT, the 3 patched runs only (6 dof).
  - **COLLAPSES if r ≤ 0.5 for both lifetime and long_diff; HELPS if r ≤ 0.8 for both; else NO GAIN.** All five
    parameters reported, with the after-polish same-data SD against the KN10S 16-sample SD.
  - Also reported: the mean step over the 12 runs vs the KN10C polish (lifetime −1.70, long_diff −1.52 %).
- **Replica test result (read 21 Oct 11:25, all six COMPLETED, 5.9–6.5 h each, MaxRSS 28.4–28.7 GiB; SPEED/valid/
  read_replica.out): T = 0.895, permutation p = 0.75 → NEUTRAL.** The unpatched run is no farther from the patched runs
  than they are from each other: R_q Ab 0.73, E 0.85, D_T 0.75, D_L 0.61, τ 0.75, position 1.68 (descriptive, per-quantity
  permutation p 0.80 / 0.69 / 0.77 / 1.00 / 0.81 / 0.20). Per the rule, the adoption bundle is proposed to Pierre.
  - Descriptive, pre-stated: **same-data run-to-run SD ≈ the KN10S 16-sample SD for every calibration parameter** —
    A_b 0.137 vs 0.133 (ratio 1.03), E 0.009 vs 0.010 (0.86), D_T 0.87 vs 1.05 (0.83), D_L 5.6 vs 5.2 (1.09), τ 4.00 vs
    4.00 (1.00); 6 dof, so each ratio is ±~30 %. Position is the exception: 1.9 vs 8.2 µm (0.23). So the calibration
    spread of the ensemble is mostly the fit's own run-to-run scatter on fixed data, not sampling of the data; the
    position spread is mostly data. Consistent with the 20 Oct polish finding that H^-1 at fixed nuisances
    underestimates the spreads 3–5× (H^-1 measures the data part only).
  - Lowest last-1e3 loss among the 4 runs: S1, S2, V3c (no pattern). Same-data loss spread up to 55 per 100 batches (s2).
  - Not in the rule: means of the 4 same-data runs per sample — τ −1.04 / −1.89 / +2.03 (SD of 3 means 2.1, vs 4.0 for
    one run, ≈ 1/√4), D_L +5.4 / +3.0 / +4.5 (SD 1.3 vs 5.2), D_T +0.30 / +0.12 / +2.55 (SD 1.4: D_T's s3 offset is
    in all 4 runs, so a data effect), A_b SD 0.11 vs 0.13. 2 dof: indicative only.
- **ADOPTED (Pierre, 21 Oct 12:05: "Let's apply the speed improvements as you propose"):** SPEED/adopt/{example_run,
  sim_jax,fit_params}.patch applied to the live tree (not committed); the three files are now byte-identical to
  SPEED/adopt/*.new.py and to SPEED/stage, i.e. exactly the code the KN10V runs used. Code defaults stay OFF.
  - Switched on for production in the KN10 wrapper KNOT/run_knot.sbatch (fit and cont modes), as defaults inside the
    container: LARND_JAX_LRU=32768, LARND_UNIQUE_BUCKET=1, LARND_POSRES_VEC=1 (an inner script or the submitting shell
    that sets its own value wins; LARND_JAX_LRU=0 turns the cache change off). Checked in the container: wrapper line →
    inner sees 32768/1/1 and JAX's caches are 32768; with LARND_JAX_LRU=0 they stay 2048. Jobs need ~28.5 GB host RAM
    (the wrapper requests 64 GB).
  - Rollback: SPEED/adopt/pre_adopt_backup_21oct/ (the three live files as they were, incl. the uncommitted 12 Oct
    fit_params.py) and KNOT/run_knot.sbatch.pre_speed_21oct.
  - The polish jobs (PAPERJOBS/run_paper.sbatch) do not set the variables, so pol_KN10V3b/3c, which start after this,
    run the same code path as the other ten.
- **Polish of the same-data runs — result (read 21 Oct 13:15; all 12 COMPLETED, 100/100 batches, 49–54 min each;
  POLISH/replica/read_polish_replica.out). Replica test NEUTRAL → all 4 runs per sample used. Verdict: HELPS.**
  - Step VALID 12/12 (H positive definite; data loss falls by 1.5–34.7 per 100 batches, actual = quadratic prediction).
  - Same-data SD before → after (r): lifetime 3.54 → 2.06 (**0.58**), long_diff 4.67 → 3.04 (**0.65**), D_T 0.77 →
    0.49 (0.64), A_b 0.120 → 0.083 (0.70), E 0.008 → 0.008 (0.95). COLLAPSES needed r ≤ 0.5 for both; HELPS (≤ 0.8)
    met. After the polish the same-data SD is 0.47–0.79 of the 16-sample SD (lifetime 0.52, long_diff 0.59).
  - Mean step over the 12: lifetime −2.65 (negative 12/12), long_diff −2.46 (10/12), A_b +0.06, D_T −0.13, E +0.001 %
    (KN10C polish at 1.4e4: −1.70 / −1.52) — the same direction as further iterations (KN10D), larger at 1e4.
  - Not in the rule: the polish moves the MEAN. Over the 12 runs lifetime −0.30 → −2.95 % (3 sample means −3.64 /
    −3.84 / −1.35, SE 0.80), long_diff +4.30 → +1.72 %, A_b +0.16 → +0.22 %, D_T +0.99 → +0.85 %. So about 40 % of the
    same-data scatter is the calibration lagging its conditional optimum (removed by one Newton step); the rest sits
    in the nuisances, which a 5-parameter step cannot reach. The conditional optimum given the final nuisances lies
    ~2.5 % below the reported lifetime (16-sample KN10C: −1.7).
- **Joint calibration + dE/dx Newton step (Pierre, 21 Oct: "I'd like you to try option 2"; rules stated before any
  result):** POLISH/joint/joint_polish.py — one Newton step on (ln θ, per-segment log dE/dx) of the final checkpoint,
  positions fixed; objective = data loss + the fit's own dE/dx prior (split-t × 1.0088, barrier, mean constraint
  w 2500, as loss_wrapper_combined). dE/dx is batch-local → per-batch Schur complement, exact dense solve (233–584
  segments per batch). Derivatives by central differences of the AD gradient: 5 ln θ columns (newton_polish.py steps)
  + 5 structurally orthogonal probes of the data band |i − j| ≤ 2 (21 gradients per batch); the prior's Hessian
  (dense, rank-1 mean term) by AD. A non-positive-definite D_b is damped (counted); step halved up to 3× if the
  objective rises.
  - **Smoke (KN10S1, batches 48 [smallest, + dense N-column check], 0, 30 [largest]; eps 1e-2 and 3e-3):** method
    OK if band error ≤ 1e-2 and off-band energy ≤ 1e-3 of the dense block, S_b(band) vs S_b(dense) ≤ 1e-2,
    probe-vs-column H_ud symmetry ≤ 1e-2, S_b(eps 1e-2) vs S_b(eps 3e-3) ≤ 2e-2. Otherwise fix before the array.
  - **Array (12 same-data runs, as the 5-parameter polish), rule:** step VALID if S is positive definite, ≤ 10 % of
    batches damped and the objective decreases. Same-data SD of the errors, r = after / before (θ_0):
    **COLLAPSES if r ≤ 0.5 for lifetime and long_diff; HELPS if r ≤ 0.8 for both; else NO GAIN.** BETTER than the
    5-parameter polish if r_joint < r_5p (0.58 / 0.65) for both. The 5-parameter polish's mean lifetime shift
    (−2.65 %) is ATTRIBUTED to holding dE/dx fixed if the joint step's mean lifetime shift is < 1.33 % in magnitude;
    NOT if it is ≥ 2.0 %; in between unresolved. All five parameters, the batch mean dE/dx change and the marginal
    (dE/dx profiled) σ reported.
  - **Smoke 1 (41516212 eps 1e-2, 41516213 eps 3e-3; KN10S1 batches 48/0/30) — method NOT OK as specified, fixed
    before the array.** Passed: band structure (dense check on batch 48: band error 1.9e-3, off-band energy 5e-6, S_b
    band vs dense 6e-7) and the prior (|g_d| data 254–435 vs data + prior 47–92: the prior cancels ~80 % of the data
    gradient, as at a near-stationary MAP). Failed: probe-vs-column symmetry (batch 0: 3.9e-2 at eps 3e-3; batch 30
    0.26 at eps 1e-2) and eps stability (S_b 1e-2 vs 3e-3 differ 16–25 % on batches 0 and 30 → eps 1e-2 is
    nonlinear; 3e-3 kept). D_b indefinite on batch 0 at both eps (min eigenvalue −400). **Main finding: the pure
    Newton step is ill-posed.** With dE/dx profiled out, the softest direction of S is almost pure lifetime
    (eigenvector lifetime −0.98, long_diff −0.22) and 14× softer than at fixed dE/dx (eig 27 vs 376 on these 3
    batches; 190 at eps 1e-2 — a near-cancellation, so FD-sensitive); the step asks lifetime +973 % (eps 3e-3) /
    +64 % (1e-2) and the objective rises (+42508 for a predicted −1656; halvings to ×1/8 still +484). This is the known
    lifetime ↔ per-segment dE/dx degeneracy (drift-profile weight is 0 in KN10). Indicative only (3 batches): the
    dE/dx-marginal lifetime σ would be ~√14 ≈ 3.7× the conditional one — the size of the 3–5× under-coverage of the
    conditional H^-1 found on 20 Oct.
  - **Fix (stated before smoke 2):** saddle-free trust-region ladder — D_b → V(|Λ| + κ·median|Λ|)Vᵀ for κ ∈ {0, 0.03,
    0.1, 0.3, 1, 3, 10} and κ = ∞ (dE/dx fixed = the 5-parameter step, from the same derivatives); the objective is
    evaluated at every rung and the lowest kept (accepted only if it decreases). Reported per rung: objective change,
    quadratic model with the signed curvature, step, rms/max dE/dx change. eps 3e-3.
  - **Smoke 2 (KN10S1 batches 48/0/30, dense check on batch 0; eps 3e-3 and 1e-3):** OK if band error ≤ 1e-2 and
    off-band energy ≤ 1e-3 on batch 0, S_b(3e-3) vs S_b(1e-3) ≤ 5e-2 (relaxed from 2e-2 — S_b's soft direction is a
    near-cancellation), and at least one finite κ lowers the objective by more than κ = ∞ does. Array rule as above,
    with "step" = the best rung; additionally reported: κ chosen per run, and σ_marg (κ = 0) vs the same-data SD.
  - (Adapting the array's validity clause to the ladder, before any array result: VALID = the chosen rung lowers the
    objective; "S positive definite" holds by construction (|Λ|) and "≤ 10 % damped" no longer applies — the number of
    batches with negative D curvature is reported instead. Readout POLISH/joint/read_joint_replica.py.)
  - **Smoke 2 result (41517485 eps 3e-3 + dense on batch 0, 41517486 eps 1e-3):** band on batch 0 OK (error 5.9e-3,
    off-band energy 1.1e-5, S_b band vs dense 4.7e-3); a finite κ beats κ = ∞ (κ = 1: objective −110.8 vs −89.4,
    model −107.7 — the quadratic model holds for κ ≥ 1, overpredicts for κ ≤ 0.3). FAILED by the letter: S_b(3e-3) vs
    S_b(1e-3) on batch 0 = 0.157 (batches 48/30: 2e-5/4e-5), all of it in the eField entry: with |Λ| on batch 0's one
    negative-curvature direction (−400), S_b(eField) turns NEGATIVE (−2.5e7 vs H_uu 4.3e8). Fix: profile only the
    positive-curvature subspace of each D_b and hold the Λ ≤ 0 directions fixed (1 of 342 on batch 0, none elsewhere).
  - **CORRECTION of the smoke-1 reading:** the "14× softer, σ_marg ≈ 3.7× σ_cond" was an artifact of smoke 1's damping
    (D + 1.1|λ_min| I left batch 0's negative direction at ~+40, so its inverse blew up the subtraction). Smoke 2:
    softest eigenvalue of S 171 vs 376 at fixed dE/dx (2.2×; same at both eps), eigenvector still mostly lifetime;
    σ_marg / σ_cond on 3 batches: lifetime 7.59 / 4.86 % (1.56×), A_b 1.63×, D_T 1.12×, D_L 1.05×, E 1.5×. So
    profiling dE/dx does NOT explain the 3–5× under-coverage by itself (indicative, 3 batches).
  - **Smoke 3 (KN10S1 batches 48/0/30, eps 3e-3 and 1e-3, no dense check):** OK if S_b(3e-3) vs S_b(1e-3) ≤ 5e-2 on all
    three batches, every S_b positive semi-definite, and a finite κ beats κ = ∞. Then the array at eps 3e-3.
- **Joint step on the 16 KN10C end points at 1.4e4 (rule stated before any result; queued behind the 12-run set if smoke
  3 passes):** same script and eps, --run KN10C --iter 14000. Readout POLISH/joint/read_joint_kn10c.py. Same rules as
  the 20 Oct 5-parameter polish of these 16, for a direct comparison:
  - Step VALID if the chosen rung lowers the objective, on ≥ 14/16 samples.
  - **IMPROVES if the 16-sample SD of the lifetime and of the long_diff errors each drop by ≥ 10 % and neither |mean|
    grows by more than 1 SE(θ_0); WORSENS if either SD rises by ≥ 10 % or either |mean| grows by > 2 SE; else
    NEUTRAL.** All five parameters reported, beside the 5-parameter polish (lifetime SD ×0.91, long_diff ×0.89).
  - Per-fit uncertainty: RMS pull (θ_0 − truth)/σ_marg (dE/dx profiled, κ = 0) over the 16; CALIBRATED if within
    [0.7, 1.4] (20 Oct, conditional σ: 3.3–7.4, E 80).
  - **Smoke 3 result (41518500 eps 3e-3, 41518501 eps 1e-3):** S_b eps agreement 1.0e-2 (batch 0), 5e-5 / 2e-5 → OK;
    best finite κ = 0.3: objective −116.0 (model −117.0) vs −89.4 at κ = ∞ → OK; both eps give the same ladder to
    < 0.1 in the objective. "Every S_b positive semi-definite" FAILED as written: batch 0 min eigenvalue −15 (−18 at
    1e-3; other batches +40, +83). That clause was mis-specified — the block-arrow Hessian is positive definite iff
    every profiled D_b is (true by construction now) and the SUMMED S = Σ_b S_b is; single-batch S_b of a noisy loss
    need not be. Summed S min eigenvalue +176. Logged as a correction of my criterion, not of the result; the array
    reports the number of indefinite S_b as a diagnostic. σ_marg / σ_cond (3 batches): lifetime 1.54×, A_b 1.59×,
    D_T 1.12×, D_L 1.05×. **Array launched at eps 3e-3** (12 same-data runs, then the 16 KN10C, throttled ≤ 6 jobs).
- **Iterate averaging (CPU only, finished histories; rule stated before computing):** readout = mean of the last W
  iterations, W ∈ {1000 (current), 2000, 3000, 5000}. Sets: the 16 KN10S @ 1e4, the 16 KN10C @ 1.4e4 (the paper's),
  and the 12 same-data runs. Script SPEED/valid/iterate_avg.py.
  - A window is BETTER (on KN10S and on KN10C separately) if the 16-sample SD of the lifetime and of the long_diff
    errors each drop by ≥ 10 % vs W = 1000 and neither |mean| grows by more than 1 SE(W = 1000); WORSE if either SD
    rises ≥ 10 % or either |mean| grows > 2 SE; else NEUTRAL. Same-data SD (12 runs, pooled) reported as description.
  - **Result (SPEED/valid/iterate_avg.out): no window is BETTER.** KN10S: W = 2000 / 3000 / 5000 NEUTRAL (lifetime SD
    ×0.99 / 1.00 / 1.03, long_diff ×1.01 / 1.02 / 1.06). KN10C: 2000 / 3000 NEUTRAL, **5000 WORSE** (lifetime SD ×1.10,
    |mean| +1.24 SE: the window reaches back into the lifetime descent). Same-data SD grows slightly with W (lifetime
    3.54 → 3.75). The calibration is frozen over the last ~5000 iterations; the run-to-run scatter is set before
    that, so only independent replicas (or a step that moves the end point) average it out.
- **Joint step on the 12 same-data runs — result (read 21 Oct 17:25; 12/12 COMPLETED, 1 h 33–1 h 37 each;
  POLISH/joint/read_joint_replica.out). Verdict: scatter HELPS, BETTER than the 5-parameter polish; mean lifetime
  shift NOT attributed to holding dE/dx fixed (−5.82 %).**
  - VALID 12/12: the best rung lowers the objective by 216–370 (dE/dx fixed: 3.6–34.7). Best κ = 0.1 (6 runs) or 0.3
    (6 runs). κ = 0 raises the objective on 6/12 runs. Batches used: 100/100 for sample 1, 90–92/100 for samples 2 and 3
    (GPU out of memory on the largest batches, 4–8.7 GB allocations, mostly the same batches within a sample).
  - Same-data SD before → after (r): lifetime 3.54 → 0.91 (**0.26**), long_diff 4.67 → 2.98 (**0.64**), D_T 0.77 →
    0.45 (0.58), A_b 0.120 → 0.035 (0.29), E 0.008 → 0.008 (0.96). After the step, the lifetime same-data SD is 0.23 of
    the 16-sample SD (A_b 0.26, D_T 0.43, D_L 0.58).
  - Mean over the 12 runs: lifetime −0.30 → −5.99 % (step negative on 12/12), long_diff +4.30 → +1.04, A_b +0.16 →
    +0.34, D_T +0.99 → +0.84. End points −8.0 … −4.1 % from starts −6.3 … +5.4 %, so the step removes ~75 % of the start
    spread toward a common point near −6 … −8 % (positions fixed). The dE/dx change is small: rms 0.29–0.46 %, batch-mean
    change within ±0.05 %, i.e. a tilt, not a scale.
  - σ_marg / σ_cond (median, 100 batches): lifetime 1.31 / 0.79 % (1.66×), A_b 0.055 / 0.031, D_T 0.45 / 0.39,
    D_L 0.91 / 0.83.
  - Reading (not part of the rule): at the fitted positions, the objective (data + the fit's dE/dx prior) has its
    lifetime minimum near −6 % or lower. The Adam end points sit ~4–5 σ_marg from it in the soft valley, scattered by
    the optimizer. Same size as earlier floors: truth drift + truth dE/dx −4.1 ± 0.7 (DRIFTDXK6), MAP dE/dx at true
    geometry −2.4 ± 2.1 (L3). Open: (i) whether the one step is converged; (ii) whether freeing the positions moves the
    minimum (coherent drift offsets couple to lifetime, 5 Oct valley geometry).
- **Iterated joint step (rule stated before any result):** POLISH/joint/joint_polish2.py = the same step repeated from
  the accepted point (--nsteps 3; the state is saved after each step; adds a d-only rung, per-batch Schur gradients and
  the data/prior split). It also uses the speed bundle's LRU hook (LARND_JAX_LRU=32768, LARND_UNIQUE_BUCKET=1).
  - Smoke jsmoke4 (KN10S1, batches 48/0/30): OK if step 0 reproduces smoke 3 (κ 0.3, dJ −116.04, within 0.5) and the
    run completes. If OK, run KN10S1 and KN10S3 (100 batches, nsteps 3).
  - **CONVERGED if the step-2 lifetime move is < 1 % and its dJ is < 10 % of the step-0 dJ; NOT CONVERGED if the
    step-1 lifetime move is ≥ 2 % in the same direction as step 0.**
  - Queue: KN10C7–16 run through submit_loop2.sh, which keeps reserve.txt (= 2) slots free for these hand-submitted
    jobs; the total stays ≤ 6.
  - **Smoke 4 result (41540340, 6 min):** step 0 reproduces smoke 3 exactly (κ 0.3, dJ −116.04: data −102.17, prior
    −13.88; d-only rung −9.6, so the gain is almost all the calibration move) → OK. The LRU hook does not speed up the
    first pass (187 s for 3 batches vs 189 s), but step 1 at warm caches takes 80 s (27 s/batch, 2.4× faster). Bug found
    and fixed before the runs: in step 1, a NaN rung (κ ≤ 0.03: dE/dx step rms 9–25, overflow) won `min()`, so the step
    was rejected. Non-finite rungs are now ranked last; on these 3 batches step 1 would have taken κ = 1 (dJ −47.0,
    lifetime +1.12, long_diff +6.0 — a further move in the step-0 direction on this 3-batch objective). Launched
    jit_KN10S1 / jit_KN10S3 (100 batches, nsteps 3, 7 h limit, 96 GB host) into the reserved slots.
- **Does the fitted geometry cause the joint step's −6 %? (rule stated before any result):** joint_polish2.py
  --geom truth rebuilds every batch with the spline coefficients replaced by the least-squares projection of the true
  positions. This is fit_params._truth_spline_coeffs, the representation floor; dE/dx, calibration start and target
  are unchanged. CPU check (POLISH/joint/check_truth_geom.py, KN10S1 batches 0/1/48): mean distance to truth 40 / 53 /
  38 µm (fitted 110 / 149 / 66 µm, = rebuild's check); coherent drift (x) offset −0.6 / −5.1 / −2.0 µm (fitted −16.8 /
  −8.4 / −7.2); dE/dx columns identical.
  - Run TG_KN10S1 = KN10S1's end point (θ_0, d_0) with truth geometry, nsteps 3, compared with jit_KN10S1 (same start,
    fitted geometry). **POSITIONS if the truth-geometry lifetime end point is ≥ 3 points closer to 0 than the fitted-
    geometry one; NOT POSITIONS if the two are within ±1.5 points; else unresolved.** Valid only if both pass the
    CONVERGED clause above; otherwise indicative. long_diff, A_b and the data/prior split are reported alongside.
  - If a slot allows: TG_KN10V1 (same data, the other replica's start). The two truth-geometry end points should agree
    within 1 point (same objective → same minimum); if they don't, the iterated step has not converged.
  - **Iterated joint step — result (read 21 Oct 23:25; jit_KN10S1 41540835 / jit_KN10S3 41541011 COMPLETED, 4 h 29
    each; POLISH/joint/iter_KN10S{1,3}.json). Verdict: CONVERGED on both. At the fitted positions the objective's
    lifetime minimum is −6.5 %.**
    - Step 0 reproduces the one-step runs exactly (KN10S1 κ 0.1, dJ −284.01; KN10S3 κ 0.3, dJ −247.6). KN10S3 uses
      91/100 batches (the same 9 GPU out-of-memory batches in every step).
    - KN10S1: lifetime −0.41 → −6.49 → −6.46 → **−6.52**. Step-1 move +0.03 %, step-2 move −0.06 %; step-2 dJ −3.7 =
      1.3 % of step 0. KN10S3: +0.49 → −5.59 → −6.32 → **−6.49**. Step-1 move −0.77 % (same direction, < 2 %), step-2
      move −0.18 %; step-2 dJ −13.8 = 5.6 % of step 0. Both CONVERGED.
    - The two samples' different data land on the same lifetime (−6.52 / −6.49), 5.5 σ_marg (1.14 / 1.22 %) from truth.
      Other end points (S1 / S3): long_diff +3.51 / −0.30, A_b +0.44 / +0.35, D_T +0.18 / +1.93, E +0.03 / +0.04.
    - Where the step-0 gain comes from: the d-only rung (calibration held) gives −255 of −284 (S1) and −222 of −248
      (S3). The calibration move is worth only −32 / −24 (θ-profile gain −½ du·S·du), and −0.13 / −0.66 at step 1. The
      lifetime minimum is shallow but well defined (σ_marg ≈ 1.2 %).
    - The lifetime pull is broad, not a few batches: per-batch Schur shares are negative on 63/100 (S1) and 56/91 (S3)
      batches, and the largest single batch contributes −0.65 / −0.83 points of the −6.3 / −6.25.
    - Reading: the one-step −6 % is the real minimum at fixed (fitted) positions and fitted dE/dx, not a truncated
      step. The truth-geometry run (TG_KN10S1) decides whether the positions put it there.
  - **Truth geometry — result (read 21 Oct 23:27; jtg_KN10S1 41541218 COMPLETED, 4 h 29; POLISH/joint/tg_KN10S1.json).
    Verdict: TG_KN10S1 NOT CONVERGED (step-2 lifetime move +14.5 %, dJ 12 % of step 0), so the POSITIONS clause is
    indicative only. Indicative value: unresolved (2.0 points).**
    - Lifetime −0.41 → −5.27 (step 0, κ 1, dJ −3739) → −4.47 (step 1, κ 1, dJ −1541) → +10.49 (step 2, κ 0.3, dJ −459).
      Fitted geometry at the same steps: −6.49 / −6.46 / −6.52. After step 1 the truth-geometry point is 2.0 points
      closer to 0, inside the unresolved band (1.5–3).
    - Step 2 is a defect of the step, not physics. The reduced calibration Hessian S is indefinite on every rung
      (min eigenvalue −1.0e11 at κ 0.3; even the dE/dx-fixed block has −1.5e3). One batch (30) contributes +13.4 of the
      +14.5 lifetime points (−8.6 of long_diff); the other 99 batches sum to +1.1. Its d-only rung (calibration held)
      gives −577, better than the accepted joint rung (−459), but d-only is not a ladder candidate. In both fitted-
      geometry runs S stayed positive definite at every step.
    - dE/dx is far from relaxed in the truth geometry: 86/100 batches have negative-curvature D_b at step 0 (fitted: 26).
      κ ≤ 0.1 overflows, and the step gains (−3739, −1541, −459; d-only −3397, −1330, −577) shrink only ~2.5× per step.
    - Data loss at the same (θ, dE/dx): the truth-projected geometry is **23 740 lower** than the fitted geometry at the
      start (2.7 % of L; 237 per batch) and 28 100 lower after two steps. The fitted positions are far from the data
      optimum within the same spline family. The priors on position (MCS, drift penalty, endpoint) are not in this
      objective, so whether the fit's full objective prefers the truth geometry is not yet known.
    - Fix before any rerun (joint_polish2.py): (a) the d-only rung becomes a ladder candidate; (b) the calibration solve
      uses only the positive-eigenvalue subspace of S (a rung with no positive subspace is skipped).
- **Truth geometry, second attempt (rules stated before any result; 21 Oct 23:29).** joint_polish2.py fixed: the d-only
  rungs (κ finite, calibration held) are ladder candidates, and the calibration solve is restricted to the positive-
  eigenvalue subspace of S (and of H_uu for κ = inf). The old script is kept as joint_polish2_v1.py. Where S is
  positive definite, which held at every step of jit_KN10S1/S3, the step is unchanged.
  - Smoke 5 (= smoke 4 with the fixed script, fitted geometry, batches 48/0/30, nsteps 3). OK if step 0 reproduces
    κ 0.3, dJ −116.04 (within 0.5) or a d-only rung wins with a lower dJ, and the run completes.
  - TG2_KN10S1: --geom truth, nsteps 6, from KN10S1's end point. CONVERGED if the last step's lifetime move is < 1 % and
    its dJ is < 10 % of step 0's. **POSITIONS if its end point is ≥ 3 points closer to 0 than jit_KN10S1's −6.52
    (|lifetime| ≤ 3.52 %); NOT POSITIONS if within ±1.5 points of −6.52 (−8.02 … −5.02); else unresolved.** Valid only
    if CONVERGED.
  - TG2_KN10V1: the same from KN10V1's end point (same data, other replica). Its end point should agree with
    TG2_KN10S1's within 1 point if both converge; if not, the minimum is not identified.
- **Where is the truth-geometry data-loss gap? (rule stated before any result; 21 Oct ~23:45).** Context: KN10C1's
  position residual (pos_residual_iter) falls from 131 µm (mean, iterations 9000–9999) to 125 µm (13 000–13 999).
  The chain LR is decayed to 1.5 % of its start, against the ~40–50 µm truth-projection floor. Each batch's
  coefficients get ~100 Adam updates per 10 000 iterations. In addition, LARND_FREEZE_STRAIGHT_CL = 0.5 freezes the
  drift-direction (e2) coefficients at 0 on the tracks the hits call straight: 104/863 tracks in KN10C1's log.
  - POLISH/joint/check_geom_loss.py (GPU, KN10S1 end point, θ_0 and dE/dx_0) computes the per-batch data loss for
    four geometries: fit; truth (projection); hybrid (truth, but e2 kept at 0 on the tracks with fitted e2 exactly 0,
    i.e. the best geometry the fit's own masks allow); e1only (truth e1, fitted e2).
  - Validation: Σ fit = 875 485.6 and Σ truth = 851 745.0 (jit/jtg step-0 L0), each within 1.
  - **FREE if hybrid − fit ≤ 0.7 × (truth − fit)** (≥ 70 % of the gap is reachable with the fit's own masks, so
    positions are under-converged); **FROZEN if hybrid − fit ≥ 0.3 × (truth − fit)** (both are negative, so ≤ 30 %
    is reachable and the straight-track freeze costs the rest); else MIXED. e1only is descriptive.
  - MCS prior at the two geometries (CPU, check_geom_prior.py): first 2 batches 33.1 (fit) vs 15.2 (truth). The prior
    also prefers the truth geometry; the 100-batch sum is to follow.
  - **MCS prior, 100 batches (check_geom_prior.out):** fit 1494.6, truth 617.3 (−877; lower on 91/100 batches). The
    fit's full objective (data + MCS; drift penalty 0, endpoints off) prefers the truth-projected geometry by
    ≈ 24 600 at the KN10S1 end point. The fitted positions are not at the optimum of the fit's own objective.
  - **Smoke 5 (41565811, 8 min): OK.** Step 0 reproduces κ 0.3, dJ −116.04 exactly. Step 1 takes κ 1 (dJ −46.98,
    lifetime +1.15 points), as smoke 4's corrected ladder predicted. At step 2 the d-only rung κ 0.1 (−23.06) wins over
    the best joint rung (κ 0.3, −23.01), so the new candidate works.
- **Chain-LR re-heat (rules stated before any result; launched only if check_geom_loss reads FREE or MIXED).** Both
  runs continue KN10S1 @ 1e4 for 4000 iterations with today's tree (speed bundle on), using KN10C1's command built by
  CONT/make_cmd.py:
  - PRC1 is the control: the production chain schedule (1e-2 × 0.9997^it, i.e. 5e-4 → 1.5e-4 over 10 000–14 000).
  - PRA1 changes only `--chain_lr 2e-3 --chain_decay_start 10000`, i.e. 2e-3 → 6e-4. The restored geometry gets a
    fresh Adam state, so the first visit moves each free coefficient by ~20 µm.
  - Tree check: PRC1's mean pos_residual over iterations 13 000–13 999 is within 5 % of KN10C1's 0.01253 cm. If not,
    PRA1 is compared with PRC1 only.
  - **Positions: re-heat HELPS if PRA1's 13 000–13 999 mean pos_residual ≤ 0.85 × PRC1's; HURTS if ≥ 1.05 ×; else
    NEUTRAL.** Mean loss over the same window is descriptive.
  - Calibration: one Adam end point carries the 3.5 % same-data scatter, so the raw errors are descriptive only. The
    decisive readout is the iterated joint step (joint_polish2.py, fitted geometry, nsteps 3) at both 14 000 end
    points. **The joint-step minimum MOVES TOWARD TRUTH if |lifetime(PRA1)| ≤ |lifetime(PRC1)| − 1.5 points; NO MOVE if
    the two are within ±1 point; else unresolved.**
  - **Result (geomloss_S1 41566123, 19 min; POLISH/joint/check_geom_loss_KN10S1.json): MIXED (66 % reachable).**
    Validation passes: Σ fit 875 485.6, Σ truth 851 745.0 (both exact).
    - truth − fit −23 741; hybrid − fit −15 666 (0.66); e1only − fit −9 209 (0.39).
    - The straight-track freeze (104/863 tracks) costs the remaining −8 074; on the batches that have frozen tracks
      the gap is −16 551 truth vs −8 476 hybrid.
    - The truth projection is lower on 70/100 batches, not on all (hybrid 55/100). Per batch, the projection of the
      true positions is not the objective's optimum everywhere.
    - Reading: about two thirds of the geometry gap is in coefficients the fit moves but has not converged; one third
      is the frozen drift shape of tracks called straight. Per the rule, the chain-LR re-heat runs (PRA1 / PRC1).
- **Joint step on the 16 KN10C end points — result (read 22 Oct 01:00; 16/16 COMPLETED, 1 h 23–1 h 40;
  POLISH/joint/read_joint_kn10c.out). Verdict: VALID 16/16, WORSENS (lifetime |mean| +3.82 SE). Pulls NOT
  calibrated.**
  - Lifetime: mean, SD +0.14, 3.16 → **−3.16, 2.98** (SD ×0.94). The step is negative on 15/16 samples, −3.35 % on
    average. long_diff +1.18, 4.45 → −0.46, 4.31 (×0.97). A_b +0.19 → +0.29 (|mean| +3.6 SE), D_T SD ×0.75, E
    unchanged. Batches used: 78–100 per sample (GPU out of memory on the largest batches).
  - Pull RMS (θ_0 − truth)/σ_marg: lifetime 2.31, long_diff 4.53, D_T 2.66, A_b 3.94, E 58. The Hessian errors are
    2.3–4.5× smaller than the observed 16-sample scatter.
  - Reading (not part of the rule):
    - On the same data (12 runs) the step collapsed the lifetime scatter ×0.26. Across 16 different data samples it
      removes only ×0.94. After the step the spread is between samples (3.0 %) and not optimizer scatter, and it is
      2.3× σ_marg. That excess variance per sample is what position errors that differ between samples would produce
      (geometry gap above).
    - The fixed-position minimum averages −3.2 ± 0.7 % over 16 samples. Samples 1 and 3 (−7.0 / −7.0 here, −6.5
      iterated at 1e4) sit low in that distribution. The 5-parameter polish (dE/dx fixed) gave −1.6 %.
  - Added 22 Oct 01:16, before any re-heat result: PRA3 / PRC3, the same pair from KN10S3 @ 1e4 (idle slots). The
    rules apply per sample. The KN10C3 reference for the tree check is its own 13 000–13 999 pos_residual.
- **22 Oct: which tracks carry the position error? (descriptive, then rules for three tests).** Pierre (22 Oct)
  approved jobs on the chain-LR decay and the straight-track freeze. He suggested filtering out low-resolution tracks
  as an alternative.
  - Per-track residual (POLISH/joint/per_track_res.py; KN10C1 @ 1.4e4, 863 tracks; KN10S1 @ 1e4 the same within
    5 %): segment-weighted mean 126 µm, track median 61 µm, p90 246 µm, p99 1.4 mm.
    - The worst 10 % of tracks (|r| > 200 µm; 97 tracks, 10.4 % of segments) hold 53 % of the summed error; the rest
      average 66 µm. The worst are mm-scale misfits in the pixel plane (r_xy 1–3 mm, drift error 0.1–0.3 mm).
    - Called straight (frozen e2): median 134 µm vs 60 µm for the others; drift rms 163 vs 58 µm.
    - No existing selection cut isolates the tail. Through-going tracks reach the anode (63 % have guess
      max |z| > 29.5 cm; anode at |z| 30.6, cathode at 0), so `track_z_bound` 28 would cut 65 % of the tracks and 68 of
      the 97 outliers. Flat tracks mostly near the anode (|cos z| < 0.1, max |z| ≥ 28) are 72 tracks (mean 268 µm) but
      only 19 of the 97 outliers.
  - **G1, oracle split of the data-loss gap by track class (check_geom_loss2.py, KN10S1 end point; ~20 min GPU).**
    Only one class moves to the truth projection: out200, out500, in200 (= the rest) or frozen. **OUTLIERS DOMINATE if
    out200 − fit ≤ 0.6 × (truth − fit); SPREAD if out200 − fit ≥ 0.3 × (truth − fit); else MIXED.** Validation:
    Σ fit and Σ truth equal check_geom_loss's within 1.
  - **G2, oracle joint step (joint_polish3.py --geom outlier --thr_um 200, nsteps 6, KN10S1 end point).** The upper
    bound for any filter or repair of the tail. **TAIL CARRIES THE BIAS if CONVERGED and the lifetime end point is ≥ 3
    points closer to 0 than jit_KN10S1's −6.52 %; NOT THE TAIL if within ±1.5 points; else unresolved.** Runs whatever
    G1 says, because the loss share and the lifetime pull need not coincide.
  - **PRF1, freeze off + re-heat (= PRA1 with LARND_CHAIN_FREEZE_DRIFT=0, 4000 iterations from KN10S1).** The freeze is
    a deliberate lifetime anchor (fit_params comment: "frozen-drift fits pin lifetime tightly"), so lifetime is
    watched. Positions vs PRA1 at 1.4e4: **FREEZE-OFF HELPS if the mean |r| of the 104 formerly frozen tracks ≤ 0.7 ×
    PRA1's and the overall 13 000–13 999 pos_residual is ≤ 1.05 × PRA1's.** Calibration: iterated joint step at the
    end point, with the same thresholds as the re-heat rule (vs PRA1's).
  - **Truth-free misfit flag (POLISH/joint/hit_flag.py, KN10C1 @ 1.4e4; descriptive, no rule).** Each target hit is
    owned by the nearest fitted path of its event (within 1 cm). Per track, the flag is the largest charge-weighted
    in-plane offset of its own hits from the path, over 4 slices along the track.
    - Outliers have a median flag of 1073 µm vs 130 µm for the rest (Spearman with |r| +0.28).
    - Flagging the top 20 % (> 341 µm) catches 72/97 outliers with 101 false flags; the kept tracks average 74 µm vs
      130 for all. Top 10 % (> 1 mm): 53/97 caught, 34 false, kept 83 µm.
    - The rms in-plane distance of the hits carries no information (pixel quantisation dominates).
    - So a data-level cut on badly placed tracks is feasible. Whether it helps the calibration is what G2 (oracle)
      bounds.
  - **Repair by the same offset fails (repair_test.py).** Moving each path by the sine-basis fit of its hits'
    offset raises the mean |r| 126 → 184 µm (outliers 642 → 646, the rest 66 → 130). Repairing only the flag top
    10 %: 126 → 141 µm. The hit offset detects a misplaced track but does not measure the correction, so a repair
    would have to be a re-seed and refit inside the fit.
  - **G3, truth-free filter in the joint step (joint_polish3.py --drop flag --drop_q 0.8, fitted geometry, nsteps 3,
    KN10S1 end point; rule stated before any result).** The tracks above the run's 80 % flag quantile leave the data
    and the model: their dE → 0, the target hits they own are dropped, and their segments leave the dE/dx prior.
    **FILTER HELPS if CONVERGED and the lifetime end point is ≥ 3 points closer to 0 than jit_KN10S1's −6.52 %; NO
    EFFECT if within ±1.5; else unresolved.** G4 (--drop outlier --thr_um 200) is the oracle filter, the upper bound
    for any cut. Smoke 6 (--drop flag, batches 48/0/30, nsteps 1) must run and report the drop counts first.
    jtg2_KN10V1 (the replica consistency check of the truth geometry) was cancelled at 2 h 30 for the slot.
  - **Smoke 6 (41577013, 8 min): OK.** Flag threshold 356 µm (q 0.8, KN10S1 @ 1e4); 173/863 tracks dropped, 23 657 of
    169 589 target hits. J0 on batches 48/0/30 drops from 25 630 (all tracks) to 24 361. Step 0 is κ 0.3, dJ −117.4
    (the same size as with all tracks).
  - **G1 result (geomloss2_S1 41577614, 22 min; POLISH/joint/check_geom_loss2_KN10S1.json): OUTLIERS DOMINATE.**
    Validation passes (Σ fit 875 485.6, truth − fit −23 740.6, = check_geom_loss).
    - out200 (106 tracks at the 1e4 end point): **−24 342, 103 % of the gap**; out500 (48 tracks): −17 236 (73 %).
    - in200 (the other 757 tracks moved to truth): **+602**, i.e. no gain. frozen (104 tracks): −9 017 (38 %; it
      overlaps out200).
    - Reading: the whole geometry gap sits in the ~12 % of tracks that are misfit at the mm scale. The other 88 % are
      at or beyond the data optimum of the truth projection, so they are not under-converged. A slower or re-heated
      chain LR (20 µm Adam steps) is unlikely to move tracks out of mm-scale misfits; it is a local-minimum / seeding
      problem. The production levers are therefore a cut (G3 / G4) or a re-seed of flagged tracks, not the LR
      schedule. PRA/PRC/PRF still run to test that directly.
- **Chain-LR re-heat, sample 1 — result (read 22 Oct 03:15; PRC1 41566910 / PRA1 41566911 COMPLETED, 2 h 59 / 3 h 09;
  POLISH/joint/read_pr1.out). Tree check PASS (PRC1 / KN10C1 pos_residual 1.003). Positions: re-heat NEUTRAL
  (PRA1 / PRC1 = 0.952).**
  - Mean pos_residual over iterations 13 000–13 999: PRC1 125.7 µm, PRA1 119.6 µm. Mean loss 8254.1 → 8217.5 per
    iteration (−37 per batch, about −3 700 over 100 batches, ~15 % of the truth-geometry gap).
  - Raw end errors are almost the same (lifetime −2.12 / −2.10 %, long_diff +5.50 / +5.54, A_b +0.28 / +0.29).
  - The joint-step pair at both end points is DEFERRED, not dropped: with a 5 % position change it cannot resolve
    the 1.5-point threshold, and the slots go to the tail tests (G2–G4, PRF1). It will run if a slot is idle.
- **Chain-LR re-heat, sample 3 — result (read 22 Oct 04:25; PRA3 / PRC3 COMPLETED; POLISH/joint/read_pr3.out). Tree
  check PASS (PRC3 / KN10C3 = 0.998). Positions: re-heat NEUTRAL (PRA3 / PRC3 = 0.954), the same as sample 1.**
  - Mean pos_residual 13 000–13 999: PRC3 132.8 µm, PRA3 126.7 µm. Loss 8271.6 → 8251.3 per iteration (−20).
  - Raw end errors (PRC3 → PRA3): lifetime −1.58 → −1.37 %, long_diff +1.61 → +1.35, tran_diff +1.92 → +2.49,
    A_b +0.23 → +0.19.
  - Two samples agree: a chain-LR re-heat moves positions by ~5 % and does not move the calibration. The chain LR
    schedule is not a lever for KN10.
- **The outlier tracks are deterministic (per_track_res.py on the same-data replicas KN10S1, KN10V1, KN10V1b,
  KN10V1c @ 1e4).** Outliers (|r| > 200 µm): 106 / 106 / 109 / 107. 96 are outliers in all four runs, 126 in any;
  pairwise Jaccard 0.84–0.88; per-track |r| Spearman S1 vs V1 0.84.
  - Same data and same hit seeding give the same bad tracks; optimizer chaos does not choose them.
  - They are not multi-track contamination: the outlier fraction is 0.10–0.13 whatever the number of tracks in the
    event (1–4) or the distance to the nearest same-event track.
  - Their straight-line guess is no worse than the rest's (median 461 vs 426 µm), but the fit ends at 0.93 × that
    (rest 0.15 ×), and 44/97 end farther from truth than the straight line. The per-batch residual of outlier
    batches starts high at iteration 0 (after the hit seed; batch 13 1020 µm vs ~600 for clean batches) and walks
    back slowly, still improving when the chain LR has decayed.
  - Candidates: (a) a bad hit seed of the pixel-plane shape (LARND_HITS_SEED_E1=1 applies the e1 seed only when its
    rms > 500 µm); (b) a property of those tracks' data (a multimodal xy loss at sub-pixel offsets). Test: a full fit
    of sample 1 with LARND_HITS_SEED_E1=0 (SE0_1).
- **SE0_1: is the pixel-plane hit seed the cause of the outlier tracks? (rule stated before any result).** A full
  fit of sample 1 from scratch: KN10S1's own argv and env (KNOT/make_fresh.py), changing only LARND_HITS_SEED_E1=0,
  10 000 iterations, today's tree. Readout per_track_res.py at 1e4 vs the four same-data replicas (106–109
  outliers).
  - **SEED IS THE CAUSE if the count of outliers (|r| > 200 µm) is ≤ 55; NOT THE SEED if ≥ 95; else PARTIAL.**
  - Descriptive: the median and mean |r|, how many of the 96 always-outliers remain, the iteration-0 per-batch
    residual (seeded start), the raw calibration end errors.
  - **Shape of the outlier error (descriptive; KN10S1 @ 1e4, 106 outliers vs 757 others).**
    - Representation floor (rms of the truth projection's residual): 28 vs 24 µm. The spline represents these tracks
      as well as the rest, so there are no unrepresentable kinks.
    - True in-plane (e1) shape rms 364 vs 268 µm. The FITTED e1 shape rms is 698 vs 276 µm: the fit carries about
      twice the true bulge. The e1 coefficient error is 698 vs 38 µm (median norm) and 65 % of it is in the lowest
      sine mode (rest 38 %). e2 error 274 vs 48 µm.
    - Reading: a smooth, overshooting in-plane bulge on tracks with larger true curvature. This is the pattern an e1
      hit seed applied only above 500 µm rms would give if the seed overshoots (per-pixel means scatter ~1 mm). The
      data cannot pull it back because transverse diffusion (σ ≈ 0.6 mm at full drift) is small against the 4.43 mm
      pitch, leaving sub-pixel in-plane offsets weakly constrained. SE0_1 tests the seed directly.
- **The e1 hit seed creates most of the outliers (offline replay of GradientDescentFitter._seed_drift_from_hits with
  the KN10 env on KN10S1's kept targets; POLISH/joint/seed_replay_KN10S1.json, seed_rules_KN10S1.json).**
  - e1 seeded on 423/863 tracks. 74 of the 106 outliers were e1-seeded (P(outlier | seeded) 0.17 vs 0.07).
  - Median e1 shape error, straight → seed → final at 1e4:

    | Group | Tracks | Straight | Seed | Final (µm) |
    |---|---|---|---|---|
    | Outliers, seeded | 74 | 393 | **1310** | 588 |
    | Outliers, not seeded | 32 | 331 | — | 191 |
    | Others, seeded | 349 | 626 | 337 | 28 |
    | Others, not seeded | 408 | 160 | — | 27 |

  - The seed is bimodal: it helps 218 seeds by more than 200 µm and harms 159 by more than 200 µm. The fit does not
    recover from the harmful ones (weak in-plane constraint at sub-pixel offsets).
  - No simple acceptance rule separates them. A χ² significance test like e2's (T1 = b1ᵀ C⁻¹ b1 > χ²(c, Kf))
    keeps a larger share of the harmful seeds (median T1/Kf: harmful 1.7, helpful 1.3); seed amplitudes are alike
    (1041 vs 1011 µm). Start-error scores (mean, tracks > 1 mm): no seed 502 / 122; current 428 / 92; + χ² 0.999
    507 / 138; + amplitude ≤ 1 mm 480 / 117.
  - The harmful seeds are significant but systematically wrong. That is the signature of pixel-grid aliasing: the
    per-pixel means of a straight track crossing the 4.43 mm grid at a shallow angle form a staircase that low sine
    modes fit as a mm-scale bend. Candidate fixes: the seed off (SE0_1 tests it), or a seed accepted only if it lowers
    the data loss (one forward per track at the first visit).
- **Can the data loss tell a good e1 seed from a harmful one? (POLISH/joint/seed_loss_test.py, GPU; rule stated
  before any result).** For each e1-seeded track of KN10S1: dL = L(all seeds) − L(all seeds without this track's e1
  seed), at the calibration of history iteration 0 and the guess dE/dx (what the first visit knows).
  - **LOSS SEPARATES if the loss-accepted seeds (dL < 0) include ≤ 25 % of the harmful seeds and ≥ 75 % of the
    helpful ones; FAILS if ≥ 50 % of the harmful ones are accepted or ≤ 50 % of the helpful ones; else PARTIAL.**
  - If it separates, a "seed only if it lowers the loss" rule (one forward per seeded track at the first visit) is
    a production candidate next to "seed off" (SE0_1).
- **The re-heat moves only the tail, and does not empty it (per_track_res.py PRA 1 14000 vs KN10C1 @ 1.4e4).** Of the
  97 KN10C1 outliers, 91 are still outliers in PRA1 (PRA1 has 93). Their mean |r| drops 642 → 570 µm (median
  439 → 365); the other 766 tracks do not move (66.3 → 66.8 µm). The re-heat's 5 % is a partial pull on the stuck
  tracks, not an escape.
- **The harmful e1 seeds are pixel-grid aliasing (descriptive, POLISH/joint/alias_test.py, CPU; no rule was stated
  before this one, so it is a finding to be tested, not a test).** The seed fits sine modes to per-pixel points
  whose in-plane position is the pixel centre. Per track, the seed's own fit was run on the noise-free pixel centres
  of the nominal straight line (each pixel weighted by its path length): b_alias, the seed a perfectly straight
  track would get from quantisation alone. KN10S1, 423 e1-seeded tracks (159 harmful, 218 helpful):
  - Median alias amplitude: harmful 975 µm, helpful 104, unseeded 113. Spearman with the seed error +0.70.
  - The seed error points along b_alias: median cos(b_alias, seed − truth) +0.92 on the harmful seeds.
  - Median angle to the nearest grid axis: harmful 3.3°, helpful 24.5°. Pixel rows crossed in the minor direction:
    3.4 vs 43. A track at a shallow angle to a pixel row sees a long-period sawtooth of amplitude ~pitch / 2 in its
    pixel-centre offsets; the low sine modes read it as a mm-scale bend.
  - Truth-free gate "e1 seed only if the alias RMS < 500 µm": keeps 205 / 218 helpful and 46 / 159 harmful seeds.
    Mean e1 start error over the 423 tracks: 652 µm now, 384 with the gate, 803 with no seed; tracks starting
    > 1 mm off: 90 / 27 / 120. Subtracting b_alias from the seed instead does worse on the harmful ones (809).
  - This test assumed a pitch of 0.443 cm. The target pixel centres give 0.4434 cm with phase coherence 1.000 on
    every batch checked (one global lattice), so the numbers above are slightly smeared at the module edges.
- **Alias gate in production code, off by default (optimize/fit_params.py, _seed_drift_from_hits;
  LARND_HITS_SEED_E1_ALIAS_UM, 0 = off).** The batch's pixel lattice (pitch and origin per axis) is read from the
  target pixel centres; an e1 seed that passes the existing amplitude test is vetoed if the alias RMS of its nominal
  line is ≥ the threshold. The [GEOM-HITS] log line reports the vetoed count. With the variable unset, the code path
  is unchanged.
- **SA1, a full fit with the alias gate (rule stated 22 Oct 04:40, before the job ran).** It is SE0_1 with
  LARND_HITS_SEED_E1=1 and LARND_HITS_SEED_E1_ALIAS_UM=500 (fresh KN10 sample-1 fit, 1e4 iterations). Readout:
  per_track_res.py SA 1 10000 against the same-data replicas, whose outlier counts were 106–109.
  - **GATE FIXES THE TAIL if the outlier count (|r| > 200 µm) is ≤ 55; NO EFFECT if ≥ 95; else PARTIAL.**
  - The gate beats "seed off" if SA1 has at least 10 fewer outliers than SE0_1 with a median |r| no worse than
    SE0_1's by more than 5 %.
  - Also reported: median and mean |r|, how many of the 96 always-outliers remain, and the raw calibration end
    errors. One sample cannot resolve a calibration change (the same-data replica scatter is the 16-sample SD).
- **Seed loss check — result (read 22 Oct 04:45; seedloss_S1 41584033 COMPLETED, 23 min; PAPERJOBS/logs/
  seedloss_S1-41584033.out, POLISH/joint/seed_loss_KN10S1.json). PARTIAL.** Of 423 e1-seeded tracks, the loss
  (dL < 0 at the first-visit calibration and guess dE/dx) accepts 263: 42 / 159 harmful (26.4 %, just above the
  25 % bound) and 192 / 218 helpful (88 %).
  - Mean e1 start error over the 423 tracks: current 652 µm (> 1 mm: 90), loss-checked 409 (37), no seed 803
    (120).
  - It does about as well as the alias gate (384 / 27 in the rough-pitch test) and costs one forward per seeded
    track.
- **The production alias gate, replayed (POLISH/joint/seed_gate_test.py, CPU, KN10S1 kept targets, KN10 env; pitch
  and origin from the hits). With the gate off it reproduces the fit's seeding exactly (423 e1-seeded, 159 harmful,
  218 helpful).**

  | Gate (µm) | Seeded | Harmful kept | Helpful kept | 1e4 outliers still seeded | Start e1 error, all 863: mean / median (> 1 mm) |
  |---|---|---|---|---|---|
  | off | 423 | 159 | 218 | 74 / 74 | 428 / 253 (92) |
  | 300 | 239 | 25 | 189 | 13 | 292 / 206 (32) |
  | **500** | 279 | 39 | 207 | 16 | **288 / 213 (27)** |
  | 800 | 312 | 61 | 213 | 19 | 292 / 219 (23) |
  | no e1 seed | 0 | — | — | — | 502 / 280 (122) |

  - The loss check combined with the gate adds little (360 vs 367 µm mean over the 423 gate-off-seeded tracks;
    the loss check alone gives 409). The gate is the cheaper and better rule; 500 µm, as pre-set for SA1, is kept.
  - SA1 released 04:50 (queue_sa.sh, next free slot).
- **The 32 outliers the e1 seed did not touch (descriptive, KN10S1 @ 1e4; alias_KN10S1.json joined with per_track).**
  53 % are frozen (called straight, e2 = 0), against 12 % of the 408 other unseeded tracks. Their median drift-axis
  residual r_z is 221 µm (others 38) and the in-plane r_xy 282 (44); median grid angle 5.3° (22.0). These are the
  straight-call misses that PRF1 (freeze off) addresses, plus shallow-angle tracks. The alias gate cannot reach
  them; at best it brings the tail down to ~32 + the 16 still-seeded outliers.
- **What the straight call freezes (descriptive; POLISH/joint/freeze_cl_test.py, CPU replay of the call on KN10S1's
  targets; the replay at 0.50 reproduces the fit's 104 frozen tracks exactly).** A track is called straight if the
  χ² p-value of its e2 modes exceeds 1 − CL, so a lower CL freezes fewer tracks.

  | CL | Frozen | True e2 shape of the frozen, median (µm) | > 300 µm | > 500 µm | KN10S1 outliers among them | r_z > 200 µm |
  |---|---|---|---|---|---|---|
  | 0.05 | 11 | 130 | 2 | 1 | 5 | 3 |
  | 0.10 | 17 | 130 | 3 | 2 | 8 | 3 |
  | 0.20 | 32 | 154 | 6 | 3 | 14 | 7 |
  | 0.30 | 52 | 125 | 8 | 3 | 18 | 11 |
  | **0.50 (KN10)** | 104 | 125 | 15 | 4 | 33 | 24 |
  | 0.70 | 153 | 123 | 23 | 4 | 34 | 24 |

  - The median true e2 shape of all 863 tracks is 262 µm. The call does pick the straighter tracks, but at 0.50 it
    freezes 15 with a true bulge > 300 µm. A frozen track keeps that bulge as its drift-axis floor.
  - 33 of the 106 outliers are frozen (17 not e1-seeded, 16 also seeded). PRF1 (freeze off) tests whether
    unfreezing them helps without costing the calibration that the freeze protects. A lower CL is the milder
    variant if it does.
- **PRF1 (freeze off + chain re-heat from KN10S1 @ 1e4) — positions result (read 22 Oct 06:00; PRF1 41578930
  COMPLETED, 3 h 00; no "[CHAIN] ... FROZEN" line in the log, so the freeze was off; per_track_res.py PRF 1 14000 vs
  PRA 1 14000; POLISH/joint/per_track_PRF1.out, read_prf1.out). FREEZE-OFF HELPS (positions).**
  - The 104 tracks frozen in PRA1: segment-weighted mean |r| 224.3 → 128.2 µm (**0.572 ≤ 0.7**). Median 131 → 64
    (the rest: 61), r_z 163 → 63, r_xy 154 → 117; outliers among them 33 → 11.
  - Overall pos_residual 13 000–13 999: 119.6 → 113.1 µm (**0.946 ≤ 1.05**). The 759 never-frozen tracks do not
    move (110.5 → 111.0). Outliers 93 → 71.
  - Loss 8217.5 → 8137.4 per iteration (−80). Raw end errors PRA1 → PRF1: lifetime −2.10 → −2.10 %, long_diff
    +5.54 → +4.90, A_b +0.29 → +0.27, tran_diff −0.13 → −0.08. With the calibration LR annealed by 1e4, the
    unfrozen geometry cannot move the calibration much in 4 000 iterations.
  - The calibration half of the rule (iterated joint step at both end points, nsteps 3) is queued as jit_PRA1 /
    jit_PRF1 (queue_jf.sh).
  - Reading: the frozen drift shape is a pure position cost once the calibration has converged under the pin.
    Releasing the freeze AFTER the calibration has annealed (a staged freeze) keeps the pin where it matters (the
    K10HIT block: no pin doubles the calibration spread) and recovers the frozen tracks' positions. Whether the
    calibration minimum moves at the better geometry is what jit_PRF1 answers.
- **TG2_KN10S1 (truth geometry, fixed script, nsteps 6) — result (read 22 Oct 07:15; jtg2_KN10S1 41565812 COMPLETED,
  7 h 39; POLISH/joint/tg2_KN10S1.json). CONVERGED; NOT POSITIONS.**
  - Steps (κ, d-only, dJ, lifetime after): 0.3 d-only −4345 −0.41 %; 1.0 −1079 −4.80; 0.3 d-only −379 −4.80;
    0.3 −75 −5.04; 1.0 d-only −23 −5.04; 0.1 −9.0 −5.36. The last step moves lifetime −0.32 points (< 1) with dJ
    0.2 % of step 0's (< 10 %): CONVERGED. S had negative eigenvalues on 54–86 of 100 batches per step, handled by
    the positive-subspace solve; all 100 batches were usable at every step.
  - End point: lifetime **−5.36 %** (inside −8.02 … −5.02, 0.34 points from the edge), long_diff +3.40, A_b +0.50,
    tran_diff +0.98, eField +0.00.
  - Reading: at the representation floor of the true trajectories the joint minimum is still at lifetime ≈ −5.4 %,
    against −6.5 % at the fitted positions. Positions account for ~1.2 of the 6.5 points. The rest of the joint
    step's lifetime offset is in the objective at fixed true geometry, not in the position error. The tail tests
    G2–G4 move a subset of the same geometry, so a large calibration effect from them is not expected; they still
    run to their rules.
- **G3 jdf_KN10S1 (truth-free filter: flag top 20 % dropped, nsteps 3) — result (read 22 Oct 08:30; 41580661
  COMPLETED, 5 h 20; POLISH/joint/df_KN10S1.json). CONVERGED; NO EFFECT.**
  - Dropped 173 / 863 tracks (flag > 356 µm) and 23 657 / 169 589 target hits.
  - Steps: κ 0.1, dJ −205.6, lifetime −0.41 → −6.81; κ 0.3, −10.8, −6.82; κ 0.1, −1.5, −6.96. Last move −0.14
    points, dJ 0.7 % of step 0's: CONVERGED.
  - End point lifetime **−6.96 %** (within ±1.5 of −6.52), long_diff +3.47, A_b +0.52, tran_diff +0.08.
  - With TG2 (truth geometry: −5.36), removing the worst-placed fifth of the tracks does not move the joint
    minimum. Filtering low-resolution tracks is not a calibration lever; it would only cost statistics.
- **G4 jdo_KN10S1 (oracle filter: the 106 tracks with |r| > 200 µm dropped, nsteps 3) — result (read 22 Oct 09:17;
  41584287 COMPLETED, 4 h 51; POLISH/joint/do_KN10S1.json). CONVERGED; NO EFFECT.**
  - Dropped 106 / 863 tracks and 18 791 / 169 589 target hits. The log's "flag threshold 356 µm" line is a
    leftover print; the drop is by the truth residual.
  - Steps: κ 0.1, dJ −193.8, lifetime −0.41 → −6.40; κ 0.3, −7.4, −6.50; κ 0.3, −0.5, −6.63. Last move −0.14
    points, dJ 0.3 % of step 0's: CONVERGED.
  - End point lifetime **−6.63 %**, long_diff +6.14, A_b +0.58, tran_diff +0.05.
  - TG2, G3 and G4 agree: the position tail does not set the joint step's lifetime minimum. Fixing the tail is a
    position improvement, not a calibration one. G2 (oracle fix, nsteps 6) is still running.
- **PRF3, a replica of PRF1 on sample 3 (rule stated 22 Oct 09:16, before the job ran).** PRF3 = PRA3 with
  LARND_CHAIN_FREEZE_DRIFT=0, the same one-line delta as PRA1 → PRF1 (checked by diff); 4000 iterations from KN10S3 @
  1e4. Readout: per_track_res.py PRF 3 14000 vs PRA 3 14000.
  - **FREEZE-OFF REPLICATES if the mean |r| of the tracks frozen in PRA3 is ≤ 0.7 × PRA3's and the 13 000–13 999
    pos_residual is ≤ 1.05 × PRA3's; FAILS TO REPLICATE otherwise.** Raw end errors reported alongside.
- **G2 jog_KN10S1 (oracle fix: only the 106 tracks with |r| > 200 µm moved to their truth projection, nsteps 6) —
  result (read 22 Oct 10:45; 41579306 COMPLETED, 7 h 46; POLISH/joint/og_KN10S1.json). CONVERGED; NOT THE TAIL.**
  - Steps (κ, dJ, lifetime after): 0.3 −3226 −4.23; 0.3 −321 −4.84; 0.3 −52.6 −5.23; 0.1 −23.9 −5.46;
    0.3 d-only −0.7 −5.46; 1.0 d-only −0.1 −5.46. The last step moves 0.00 points with dJ ≈ 0: CONVERGED.
  - End point lifetime **−5.46 %** (1.06 points closer to 0 than −6.52, inside ±1.5), long_diff +6.11, A_b +0.51,
    tran_diff +0.18.
  - **Tail block summary (TG2, G2, G3, G4; all CONVERGED):** truth geometry −5.36, tail fixed −5.46, flag filter
    −6.96, oracle filter −6.63 (fitted reference −6.52). The tail carries the whole position effect on the joint
    minimum (tail fixed ≈ all truth), but that effect is ~1.1 points. The other ~5.4 points are in the objective at
    true geometry. The outlier tail is a position-quality issue; neither fixing nor filtering it is a calibration
    lever.
- **SE0_1 (full fit, e1 hit seed off) — result (read 22 Oct 10:55; 41584760 COMPLETED, 6 h 08;
  POLISH/joint/per_track_SE0_1.out, end10k_SE0_1.out). Outliers 108 → NOT THE SEED by the rule. Seed off is WORSE
  overall.**
  - Per class (truth-based seed classes of seed_replay; KN10S1 → SE0_1 outliers): helpful-seed tracks (218) 14 →
    **45**; harmful-seed tracks (159) 56 → **26**; unseeded (440) 32 → 33. Of the 96 always-outliers, 65 remain;
    37 outliers are new (not outliers in any of the four replicas).
  - The seed both creates and prevents outliers, in equal numbers, so the count rule cannot see it. Its net effect
    is positive: segment-weighted mean |r| 132.3 → 185.2 µm without the seed (median 62 → 59, p99 1.4 → 1.7 mm);
    pos_residual 9 000–9 999: 186.8 µm vs 131.2–136.0 for the four replicas; loss 8715 vs 8289–8304 per iteration
    (+420). Iteration-0 pos_residual 842 vs 661 µm (the seeded start).
  - Raw end errors (one sample; replica scatter is the 16-sample SD): lifetime −0.58 % (replicas −6.33 … +1.87),
    long_diff −5.24 (−1.89 … +9.63), A_b +0.09, tran_diff +1.88.
  - Reading: the 26 harmful-seed tracks that stay outliers without the seed are hard for the fit anyway (shallow
    grid angle). A selective gate keeps the 207 helpful seeds and so avoids most of the 31 new outliers; SA1 tests
    it.
- **SA1 (full fit, e1 seed with the alias gate at 500 µm) — result (read 22 Oct 11:45; 41589683 COMPLETED, 5 h 49;
  POLISH/joint/per_track_SA1.out, end10k_SA1.out). Outliers 73 → PARTIAL by the rule. The comparison clause
  against seed off FAILS on its median guard (62.9 vs 58.7 µm, +7.2 % > 5 %) although SA1 has 35 fewer outliers.**
  The live fit seeded 279 tracks and vetoed 144, as the offline replay predicted (279 at 500 µm).

  | Fit (sample 1, 1e4) | Outliers > 200 µm | > 500 µm | Weighted mean |r| | Median | p90 | p99 | Always-outliers left | Frozen outliers |
  |---|---|---|---|---|---|---|---|---|
  | KN10S1 / V1 / V1b / V1c | 106 / 106 / 109 / 107 | 48–53 | 132.3–137.3 | 59.6–67.0 | 261–294 | 1230–1362 | 96 | 32–36 |
  | SE0_1 (seed off) | 108 | 43 | 185.2 | 58.7 | 278 | 1742 | 65 | 29 |
  | **SA1 (alias gate)** | **73** | **23** | **107.8** | 62.9 | **177** | **719** | 59 | 28 |

  - The SE0_1 median sits below the replicas' median range (59.6–67.0) and SA1's inside it, so the median guard
    compares within the replica noise. It is reported as failed, as written.
  - By seed class (KN10S1 → SE0_1 → SA1 outliers): gate-vetoed 144 tracks 58 → 32 → 27; gate-kept 279 tracks
    16 → 43 → 14; unseeded 440: 32 → 33 → 32. The gate gets the seed-off gain on the vetoed tracks and keeps the
    seed's gain on the rest.
  - pos_residual 9 000–9 999: 106.0 µm (replicas 131.2–136.0; −20 %). Loss 8215.0 per iteration (replicas
    8288.8–8304.2; −80). The model and constraints are unchanged and only the start differs, so the lower loss is a
    better optimum of the same objective.
  - Raw end errors (one sample): lifetime −1.60 % (replicas −6.33 … +1.87), long_diff +9.55 (−1.89 … +9.63),
    A_b +0.27, tran_diff +0.10, eField +0.03. One sample cannot resolve a calibration change.
  - What remains: 32 unseeded outliers (about half frozen; PRF1 releases those) and 22 harmful-seed tracks that
    stay outliers with or without the seed.
- **SA block, samples 2–6 (rule stated 22 Oct 11:45, before any of these jobs ran).** Fresh 1e4 fits of KN10S<s>'s
  argv and env (make_fresh.py) plus LARND_HITS_SEED_E1_ALIAS_UM=500. The scripts differ from inner_sci_full_SA1.sh
  only in data_seed and test name (checked by diff; SA1 regenerates identically). Paired with KN10S<s> @ 1e4;
  per_track_res.py on both. The verdicts use samples 2–6 only (SA1 was read before this rule); SA1 is listed
  alongside. Read only when all five are done.
  - **Positions: GATE IMPROVES POSITIONS if the outlier count (|r| > 200 µm) is lower than KN10S<s>'s in ≥ 4 / 5
    samples and the mean paired change of the segment-weighted mean |r| is negative by more than 2 SE; NO POSITION
    GAIN if that mean change is within ±5 µm; else unresolved.**
  - **Calibration: CALIBRATION CHANGES if the paired mean shift (SA − KN10S, raw 1e4 end error) of lifetime or
    long_diff exceeds both 2 points and 2 SE; else NO GROSS CHANGE.** The same-data replica scatter (lifetime SD
    ~3.5 points) gives an SE of ~1.5–2 points on five pairs, so this block can rule out a gross shift, not prove
    neutrality at the 1-point level. A_b, tran_diff and eField are reported alongside.
- **SAF1: do the alias gate and the freeze release add? (rule stated 22 Oct 11:47, before the job ran).** SAF1 = the
  PRF1 continuation (freeze off, chain LR re-heat 2e-3, 4000 iterations) resumed from SA1 @ 1e4 instead of
  KN10S1 @ 1e4; env + LARND_HITS_SEED_E1_ALIAS_UM=500 (inert on resume: restored geometry is only classified).
  Readout per_track_res.py SAF 1 14000 against PRF1 (freeze release alone: 71 outliers, weighted mean 112.3 µm),
  SA1 @ 1e4 (gate alone: 73, 107.8) and PRA1 (neither, same continuation: 93, 119.3).
  - **COMBINED BEATS BOTH if SAF1's outlier count is at least 10 below both 71 and 73 (≤ 61) and its weighted mean
    |r| is ≤ 0.95 × 107.8 µm (≤ 102.4); NOT ADDITIVE if its outlier count is ≥ 66; else PARTIAL.**
  - Queued after the SA block's last job (queue_saf.sh).
- **PRF3 (freeze release + re-heat from KN10S3 @ 1e4) — result (read 22 Oct 12:35; 41601446 COMPLETED, 3 h 10; no
  "[CHAIN] ... FROZEN" line in the log; per_track_res.py PRF 3 14000 vs PRA 3 14000; POLISH/joint/read_prf3.out).
  FREEZE-OFF REPLICATES.**
  - The 115 tracks frozen in PRA3: weighted mean |r| 196.6 → 94.3 µm (**0.480 ≤ 0.7**); median 103 → 59; r_z 173
    → 75; outliers among them 29 → 5.
  - pos_residual 13 000–13 999: 126.7 → 118.9 µm (**0.938 ≤ 1.05**). The 774 never-frozen tracks do not move
    (131.4 → 130.6). Outliers 76 → 52. Loss 8251.3 → 8168.3 per iteration (−83; PRF1: −80).
  - Raw end errors PRA3 → PRF3: lifetime −1.37 → −0.65 %, long_diff +1.35 → +1.47, A_b +0.19 → +0.18, tran_diff
    +2.49 → +2.40, eField +0.03 → +0.02.
  - Two samples: releasing the straight-track freeze in the continuation halves the frozen tracks' error and removes
    80–85 % of their outliers, leaves every other track as it was, and moves no calibration parameter by more than
    0.72 points (PRF1 lifetime 0.00, PRF3 +0.72). The jit_PRF1 joint step (calibration half) is still running.
- **Steady-state profile of a production fit with the speed bundle (SA2, iteration ~306, py-spy 120 s at 50 Hz +
  nvidia-smi dmon; SPEED/prof_SA2_steady.txt, smi_SA2.txt).** 1.88 s/it on sdfampere010. GPU SM busy ~60 % on
  average (bursts at 99–100 %, gaps at 2–30 %); 31 GB device memory. Main thread self time: 51.5 % inside
  _pjit_call_impl (executions: the backward at compute_loss:2441 31.6 %, simulate_probabilistic 19.1 %), 14.2 %
  ArrayImpl._value, 9.9 % eager apply_primitive. Python tracing is no longer a large term.
  - **13.8 % is one host sync: strategies.py:101, int() on get_roi_counts.** It runs whenever the Npix level
    differs from the previous call (almost every iteration), blocks the host until simulate_wfs has run, and only
    updates a running max of the ROI padding that, after epoch 1, almost never changes.
  - The mbjit result (jitted loss 1.055 s vs eager 1.13 s per call) already showed that the loss itself is now
    execution-bound; the lever left on the host side is to stop serialising host and GPU.
- **LARND_ROI_LAZY (optimize/strategies.py + one line in fit_params.compute_loss; default off, uncommitted).** When the
  largest exact ROI counts seen for this (batch, Npix) leave ≥ 10 % headroom under the current padding, keep the
  padding (the value the exact path keeps unless the counts jumped) and check the device counts at the next
  predict call, by when they are complete. A count above the padding used is logged as [ROI-LAZY] overflow (that
  one call dropped ROIs) and raises the padding. With the variable unset the code path is unchanged; the
  compute_loss line only sets an attribute. CPU mock test (scratchpad test_roi_lazy.py): identical padding in
  every call, blocking syncs 40 → 16 over 20 calls; a forced count jump gives exactly one overflow call, then the
  exact path's padding.
  - SA6 and SAF1 start on this tree with the variable unset (inert).
- **Bench BENCHLZ (rule stated 22 Oct 13:16, before the job ran).** One job, one node: KN10S1 config + bundle, 600
  iterations, LARND_ROI_LAZY=1 arm first (BENCHLZ1), then the control (BENCHLZ0). Steady state = median s/it over
  iterations 300–600.
  - **LAZY ADOPTABLE if the lazy median is ≤ 0.90 × the control's and there are zero [ROI-LAZY] overflow lines;
    NOT WORTH IT if ≥ 0.95 ×; else marginal.** (Amended 13:16, before the job ran: a losses_iter clause was
    dropped. The lazy path only acts on revisits, i.e. from iteration 100, and identical-code replicas already
    differ by 1 ulp from iteration 3, so losses cannot test it. With zero overflows every call used the padding the
    exact path would have used, which is the equivalence that matters. A count of the [ROI-LAZY] lazy-call lines
    confirms the path ran.)
  - Queued after SAF1 (queue_lz.sh).
- **PRF1 calibration half — result (read 22 Oct 14:00; jit_PRA1 41594628 COMPLETED 4 h 55, jit_PRF1 41597904
  COMPLETED 5 h 28; POLISH/joint/iter_PRA1.json, iter_PRF1.json). Both CONVERGED; NO MOVE.**
  - jit_PRA1: κ 0.3 −521.2 → lifetime −6.14; κ 0.1 −38.3 → −6.88; κ 1.0 −1.9 → **−6.93** (last move −0.05, dJ
    0.4 % of step 0's). long_diff +3.52, A_b +0.50.
  - jit_PRF1: κ 0.3 −578.1 → −6.01; κ 0.3 −46.1 → −6.50; κ 0.3 −11.7 → **−6.66** (last move −0.16, dJ 2.0 %).
    long_diff +1.81, A_b +0.47, tran_diff +0.30.
  - |−6.66| vs |−6.93|: within ±1 point, so NO MOVE. Releasing the freeze leaves the joint minimum where it was
    (long_diff 1.7 points lower). With PRF1/PRF3's positions, the freeze release is a position gain that does not
    move the calibration.
- **SA block (alias gate, samples 2–6) — result (read 22 Oct 20:30; SA2–SA6 COMPLETED, 5 h 50 – 6 h 35;
  POLISH/joint/read_sa_block.out). Positions UNRESOLVED; calibration NO GROSS CHANGE.**

  | Sample | Outliers KN10S → SA | > 500 µm | Weighted mean |r| (µm) | Median | p99 | pos_residual 9–10k | Loss 9–10k | Δ lifetime | Δ long_diff |
  |---|---|---|---|---|---|---|---|---|---|
  | 1 (read before the rule) | 106 → 73 | 48 → 23 | 132.3 → 107.8 | 62 → 63 | 1362 → 719 | 131.5 → 106.0 | 8288.8 → 8215.0 | −1.13 | +2.60 |
  | 2 | 109 → 82 | 47 → 29 | 143.2 → 146.6 | 64 → 66 | 1249 → 958 | 150.7 → 163.4 | 8303.7 → 8329.5 | +5.41 | +1.44 |
  | 3 | 101 → 68 | 33 → 18 | 149.7 → 136.6 | 63 → 60 | 1399 → 793 | 139.8 → 129.5 | 8307.3 → 8315.7 | −2.19 | +2.24 |
  | 4 | 110 → 78 | 45 → 19 | 138.0 → 119.0 | 66 → 62 | 1287 → 826 | 136.0 → 118.5 | 8308.6 → 8269.1 | −2.12 | −0.32 |
  | 5 | 92 → 76 | 39 → 15 | 144.6 → 136.6 | 65 → 68 | 1575 → 989 | 151.8 → 145.1 | 8364.8 → 8328.1 | +2.39 | +6.69 |
  | 6 | 102 → 81 | 49 → 24 | 131.4 → 140.6 | 63 → 74 | 1356 → 925 | 132.4 → 143.7 | 8256.4 → 8274.0 | −4.16 | −10.84 |

  - Positions (s2–6): outliers lower in 5 / 5, but the paired mean change of the weighted mean |r| is −5.5 ± 5.2 µm
    (−1.1 SE, the rule needs 2 SE) → **unresolved**.
  - Calibration (s2–6): lifetime −0.13 ± 1.75, long_diff −0.16 ± 2.91 → **no gross change**. A_b +0.06 ± 0.06,
    tran_diff +0.19 ± 0.46, eField +0.00 ± 0.01.
  - Descriptive: the gate cuts the tail on every sample (outliers −16 … −33, > 500 µm roughly halved, p99 −25 …
    −50 %). On samples 2 and 6, though, the weighted mean, the pos_residual and the loss are higher, and sample 6's
    median rises 63 → 74 µm (the four same-data replicas of sample 1 span 59.6–67.0). So the gate removes the worst
    tracks but does not reliably improve the average fit. Loss lower on 3 / 6 samples (s1, s4, s5).
  - Reading: the alias gate is NOT ready for production as a default. It is a tail reducer whose effect on the bulk
    is mixed. The freeze release in the continuation (PRF1/PRF3: positions improved on both samples, calibration
    neutral) is the cleaner candidate. SAF1 (both combined) is still running.
- **BENCHLZ (LARND_ROI_LAZY) — result (read 22 Oct 21:55; 41642858 COMPLETED, 2 h 59). NOT WORTH IT.** Median s/it over
  iterations 300–600: lazy 1.662, control 1.657 (ratio 1.003); 0 [ROI-LAZY] overflow lines; epoch 1 67.8 vs 66.6 min.
  Removing that sync does not shorten the step, because the host stays blocked inside the FEE scans (opt team, host_loop
  report). In any case the opt team's LARND_ROI_SKIP removes the sync exactly (the counts are dead inputs), which
  supersedes LARND_ROI_LAZY.
- **SAF1 (alias gate + freeze release, continuation from SA1) — result (read 22 Oct 21:55; 41641829 COMPLETED, 3 h 09;
  per_track_res.py SAF 1 14000). COMBINED BEATS BOTH.** Outliers 37 (≤ 61) and weighted mean |r| 90.0 µm (≤ 102.4);
  > 500 µm: 11; median 61. For comparison: PRA1 (neither) 93 / 119.3, PRF1 (release) 71 / 112.3, SA1 @ 1e4 (gate) 73 /
  107.8. One sample; the gate's own block (SA2–6) left the bulk effect unresolved, so the release remains the
  production-ready half.

### 22 Oct (late): bitwise opt package ADOPTED in the live tree (Pierre: "Let's indeed adopt this package")
- Applied opt/adopt/package.patch (= opt/combo3: compile_mem + loss + host_loop_core + gpu_sim sim/FEE files) to
  optimize/{dedx_utils,example_run,fit_params,strategies}.py and src/larndsim/{fee_jax,sim_jax}.py; backups in
  /sdf/scratch/neutrino/pgranger/opt/adopt/live_before/. Defaults switched ON: LARND_SEG_PAD, LARND_ROI_SKIP,
  LARND_PRECOMP_LEAN, LARND_FEE_FAST (=0 restores each), LARND_MLIR_SHARED_CTX (ON when LARND_JAX_LRU > 2048, i.e. with
  run_knot.sbatch's bundle). Loss/host-loop changes were already default ON. Opt-in, not bitwise: SIG_REMAT,
  SIG_DROP_GARBAGE, SIG_COMPACT, PPP_JIT stay OFF. Uncommitted. Evidence: opt/reports/SUMMARY.md (CPU e2e 39/39 bitwise;
  GPU 100 batches x 300 it same node: epoch 1 0.18x, steady 0.67x, host RSS 28.3 -> 9.0 GiB).
- Validation KN10P1 (full 1e4 it, KN10V1 config, live tree): RULE before the result, readout
  /sdf/scratch/neutrino/pgranger/review_2026_09/SPEED/valid/read_pkg.py: z = (P1 - mean of the 4 same-data sample-1
  runs V1/V1b/V1c/S1) / (same-data SD pooled over samples 1-3 x sqrt(1.25)) for the 5 calibration errors + position.
  CONSISTENT if max |z| < 3; SHIFT if any |z| > 4; else INCONCLUSIVE. Also report wall time and MaxRSS vs V1-V3
  (5.9-6.3 h, 28.4-28.7 GiB).
- 23 Oct: **KN10P1 is INVALID as a package test** (not read with the rule): its script (copied from KN10V1c) put
  SPEED/stage first on the path, i.e. the frozen 20 Oct copies of example_run.py / fit_params.py / sim_jax.py, so it ran
  a mix (new strategies/dedx_utils/fee_jax only; no shared MLIR context: MaxRSS 25.3 GiB, 4.9 h). Only the SPEED test
  scripts use that overlay; production KNOT scripts run the live tree. Rerun KN10P2 (python3 -m optimize.example_run,
  same config), same rule and readout (read_pkg.py now reads KN10P2).
- 23 Oct **KN10P2 (adopted package, live tree, 1e4 it): CONSISTENT (max |z| = 1.54)**. z: Ab +0.11, eField +0.95,
  tran_diff -0.62, long_diff -0.61, lifetime -0.52, pos +1.54 (105.3 vs 100.8 um, same-data SD 2.6). Wall 3.34 h
  (vs 5.9-6.3 h for KN10V1-3), MaxRSS 9.2 GiB (vs 28.4-28.7 GiB), 1.13-1.18 s/it at the end. Package validated.

### 23 Oct — science programme (Pierre: "Please follow your program"); rules stated BEFORE any job ran
Context: the converged joint (calibration + per-segment dE/dx) minimum sits at lifetime -6.5 % at fitted positions and
-5.36 % at TRUE positions (TG2_KN10S1), while the calibration-only model at truth inputs is unbiased (+0.5 %, Sep review)
-> the offset comes from letting dE/dx float (a drift tilt, rms 0.4 %). Note: a reparametrisation of dE/dx (scale,
shape) cannot move the minimum of the SAME objective (it only changed where Adam stopped), so it is NOT tested here; the
test is an objective change: the existing truth-free drift-profile penalty (fit_params.py: DPW x (dx-weighted trend of
log dE/dx vs |z|, slope x sigma_z)^2 per batch; physics: Landau dE/dx does not depend on drift; production DPW = 0).
- **DP (joint minimum with the drift-profile penalty).** POLISH/joint/joint_polish_dp.py = joint_polish2.py + `--dpw`
  (prior term identical to fit_params.py; Hessian by jax.hessian as for the rest of the prior). All runs: --geom truth,
  from KN10S<s> @ 1e4, nsteps 6, eps 3e-3 (as TG2). Stage A, sample 1: DPW 1e5, 1e6, 1e7 (baseline TG2 = DPW 0:
  -5.36 %). Stage B: the SMALLEST stage-A weight whose converged lifetime is within +-1.2 % (one sigma_marg) of truth,
  on samples 2 and 3, each with its DPW 0 baseline (tg2-style). Converged = last accepted step moves lifetime < 0.3
  points; otherwise reported as NOT CONVERGED (indicative only).
  **Verdict: DRIFT PROFILE FIXES THE MINIMUM if the chosen weight puts the converged lifetime within +-1.2 % of truth on
  all 3 samples AND no other calibration error moves farther from 0 by > 2 points vs its DPW 0 baseline; DOES NOT FIX
  if no stage-A weight passes on sample 1; PARTIAL otherwise.** If it fixes: next = fitted geometry + 16-sample
  ensemble with that weight (Adam fit with DPW, then the joint step).
- **FB (freeze release in the continuation, 4 more samples).** PRA<s> / PRF<s> for s = 2, 4, 5, 6: copies of
  PRA3 / PRF3 (4000 it from KN10S<s> @ 1e4, chain LR re-heat 2e-3; PRF = LARND_CHAIN_FREEZE_DRIFT=0, the only delta,
  checked by diff), live tree with the adopted package. Per sample: **REPLICATES if the segment-weighted mean |r| of the
  tracks frozen in PRA<s> is <= 0.7 x PRA's in PRF<s> AND PRF's 13 000-13 999 pos_residual is <= 1.05 x PRA's.**
  Block: **CONFIRMED if it replicates on >= 3 of the 4 new samples**; calibration over all 6 samples (1, 3 included):
  paired raw end-error shift PRF - PRA for lifetime and long_diff, "no gross change" if |mean| < 2 SE. CONFIRMED + no
  gross change -> propose the continuation with freeze release as the production recipe.
- Slots: science <= 3 concurrent jobs until the opt agents' deadline (22:00), then <= 6 minus opt jobs.
- 23 Oct, opt round 2 — round-off flags (LARND_SIG_REMAT, SIG_DROP_GARBAGE, SIG_COMPACT, PPP_JIT) on top of the adopted
  package, same-node GPU test (opt/equiv/stack, job 41750762, 100 batches x 300 it): steady 1.096 -> 0.822 s/it (0.75x),
  epoch 1 0.82x, device peak 21.2 -> 8.6 GiB, host RSS 9.0 -> 8.0 GiB, first losses within 1e-7..6e-7. Validation
  **KN10R1** (live tree + these 4 flags, KN10P2 config, 1e4 it): SAME RULE as KN10P2 (read_pkg.py KN10R1; CONSISTENT if
  max |z| < 3 vs the 4 same-data sample-1 runs, SHIFT if > 4). Packing test (1/2/3 fits per A100, opt/equiv/pack,
  job 41753695) measures throughput per GPU.
- 23 Oct, packing result (job 41753695, sdfampere028, round-off flags, 100 batches x 300 it per fit, 4 cpus): solo
  steady 0.814 s/it; **2 fits on one A100: 1.56 s/it each = 1.04x throughput (no gain)**, so the steady state is
  GPU-bound. Epoch 1 (compile-dominated) parallelises: 11.6 min per fit vs 11.0 solo. 3 per GPU: one fit (p3b)
  died with GPU RESOURCE_EXHAUSTED (4.6 GiB transient on top of 3 x 8.7 GiB); the summary's "1.57x" for 3 is an
  artefact (only 2 fits ran after p3b died). **Verdict: do not pack full-length fits** (10k it: steady ~2.3 h vs ~11
  min compile); one fit per GPU stays the recipe.
- 23 Oct, **KN10R1 = CONSISTENT** (read_pkg.py KN10R1, rule above): max |z| 1.78 (Ab -1.78, eField -0.08, tran_diff
  -0.97, long_diff +0.67, lifetime +0.84, pos +0.72; end errors Ab +0.03, lifetime +2.29, long_diff +8.93 %, pos 102.9
  um). Wall 2.56 h (KN10P2 3.34 h: 0.77x; pre-package 5.9-6.3 h), MaxRSS 8.7 GiB (KN10P2 9.2). The 4 round-off flags
  are a candidate default; NOT flipped (Pierre to decide).
- 24 Oct, **DP stage A = DOES NOT FIX** (read_dp.py, rule above; sample 1, truth geometry, 6 accepted steps each):
  converged lifetime DPW 1e5 -5.02 %, 1e6 -4.83 %, 1e7 -4.88 % (DPW 0 = TG2: -5.36 %, last move -0.32, borderline
  converged). No weight within +-1.2 %; the effect saturates by 1e6 at ~+0.5 point. Other errors unchanged (Ab +0.48..
  +0.49, tran_diff +0.74..+0.89, long_diff +3.3..+3.9 vs TG2 +0.50 / +0.98 / +3.40). So the linear dE/dx drift trend
  carries only ~0.5 of the 5.4 points at the joint minimum; stage B not run (rule).
- **24 Oct ~05:35 PDT: round-off flags ADOPTED as defaults in the live tree (Pierre's OK; uncommitted).**
  LARND_SIG_REMAT, LARND_SIG_DROP_GARBAGE, LARND_SIG_COMPACT (src/larndsim/sim_jax.py), LARND_PPP_JIT
  (optimize/strategies.py) now default ON; LARND_RESP_CUM_CACHE follows SIG_REMAT (so ON too, as in KN10R1).
  =0 restores the old path for each. Backups: opt/adopt/live_before_rof/. Basis: KN10R1 CONSISTENT (above).
  **Reproducibility boundary:** every run started after this time differs from earlier runs at float32 round-off
  (loss 2-7 ulp), which the fit's chaos amplifies to the same-data replica scatter. Do not expect bitwise or
  replica-level agreement between runs on either side of this date; compare ensembles, not single runs.
- 24 Oct, **FB block = CONFIRMED, no gross calibration change** (read_fb_block.py, rule above; per_track_res.py at 14000).
  Replicates on 4/4 new samples (2, 4, 5, 6; also on 1 and 3 from the old tree): the tracks frozen in PRA<s> go from
  |r| 195-224 to 86-128 um when the freeze is released (0.43-0.59x), the overall 13-14k pos_residual drops 0.91-0.95x
  (e.g. s4 123.5 -> 112.1 um); segment-weighted mean |r| over all tracks 119-135 -> 112-126 um. Calibration, paired
  PRF - PRA over 6 samples (points): lifetime +0.08 +- 0.23, long_diff +0.46 +- 0.36, tran_diff -0.07 +- 0.25, Ab
  +0.01 +- 0.01, eField -0.01 +- 0.00. -> Proposed production recipe: after the 1e4-it fit, a 4000-it continuation with
  chain LR re-heat 2e-3 and LARND_CHAIN_FREEZE_DRIFT=0 (= PRF). Awaiting Pierre; not made default.
- 24 Oct, opt round 2 combined tree **opt/comb4** = ref2 + prof2 (fit_params.py, sim_jax.py) + fee2 (fee_jax.py,
  example_run.py). CPU (2 b x 50 cm, 4 it): default flags BITWISE vs ref2; all flags it-0 loss 8.5e-8, calib grads
  <= 9e-7. GPU same-node A/B (job 41804049, 100 b x 300 it) vs rof (= the new live defaults): **all flags steady
  0.798 -> 0.446 s/it (0.56x), epoch 1 0.75x, device 8.7 -> 10.3 GiB, host 8.0 -> 6.9 GiB**; without FEE_PIXEL_STOP
  0.564 s/it (0.71x), device 8.6 GiB. Validation **KN10R2** (comb4 tree, KN10R1 config + LARND_FITSIDE_JIT=1
  OPT_JIT=1 HITS_PAD=512 FEE_PIXEL_STOP=3 FEE_PEEL0=1 FEE_CMDBUF=1, 1e4 it): SAME RULE as KN10P2 / KN10R1
  (read_pkg.py KN10R2; CONSISTENT if max |z| < 3 vs the 4 same-data sample-1 runs, SHIFT if > 4). Written before
  the run.
- 24 Oct, **KN10R2 = CONSISTENT** (read_pkg.py KN10R2): max |z| 1.21 (Ab -0.49, eField +1.21, tran_diff +0.27,
  long_diff -0.34, lifetime -0.08, pos -0.53; end errors Ab +0.20, lifetime -1.37, long_diff +3.68 %, pos 99.3 um).
  **Wall 1.43 h** (KN10R1 2.56 h: 0.56x; KN10P2 3.34 h; pre-package 5.9-6.3 h), MaxRSS 7.5 GiB. Round-2 flags are a
  candidate default for the live tree; NOT ported (Pierre to decide).
- **24 Oct ~07:31 PDT: opt round 2 ADOPTED as defaults in the live tree (Pierre's OK; uncommitted).** Ported from
  opt/comb4: optimize/fit_params.py, optimize/example_run.py, src/larndsim/fee_jax.py copied (live == ref2 before);
  src/larndsim/sim_jax.py = prof2's 15-line diff (LARND_SEARCH_UNROLLED) applied on top of this morning's flips.
  Defaults now ON (=0 restores each): BITWISE LARND_FEW_SYNC, STRONG_INIT, SEARCH_UNROLLED; round-off LARND_FITSIDE_JIT,
  OPT_JIT, HITS_PAD=512, FEE_PEEL0, FEE_CMDBUF; behaviour-level-tiny LARND_FEE_PIXEL_STOP=3. Backups:
  opt/adopt/live_before_r2/. Checked in the container: every default loads ON, the =0 switches restore the old paths,
  XLA_FLAGS gets the command-buffer flags at import. Basis: KN10R2 CONSISTENT (above). **Second reproducibility
  boundary** (after this morning's): runs started after this time differ from earlier ones at round-off level.
- **24 Oct: PRODUCTION RECIPE = 1e4-it fit + freeze-release continuation (Pierre's OK).** Launch with
  `KNOT/submit_prod.sh <NAME> <inner_fit.sh>`: the fit (sci_full_<NAME>), then with afterok the 4000-it continuation
  sci_full_<NAME>FR built at job start by `run_knot.sbatch prodcont` = CONT/make_prod_cont.py (CONT protocol + --chain_lr
  2e-3 --chain_decay_start 10000 + LARND_CHAIN_FREEZE_DRIFT=0). The builder regenerates KNOT/inner_sci_full_PRF2.sh and
  PRF4.sh byte for byte. **The production end point is <NAME>FR @ 14000** (readout scripts still default to 1e4 of
  the fit; read the FR checkpoint for production numbers). run_knot.sbatch backup: run_knot.sbatch.pre_prodcont_24oct.
- 24 Oct, first production run on the new defaults + pipeline: **KN10N1** (KN10P2's command, live tree, no extra flags,
  sample 1) via `KNOT/submit_prod.sh KN10N1`, submitted only if the CPU port check (equiv/comb4/cpu_port.sh: live
  defaults vs comb4 + all flags) is BITWISE. Rules, written before the run: (1) the fit at 1e4: read_pkg.py KN10N1,
  CONSISTENT if max |z| < 3 (as KN10P2/R1/R2); (2) the pipeline: KN10N1FR reaches 14000 and its 13 000-13 999 mean
  pos_residual is within +-8 % (3 x the 2.6 % same-data replica SD) of PRF1's 113.1 um -> PIPELINE OK.
- 24 Oct 07:45: CPU port check (equiv/comb4/cpu_port.sh) **ALL BITWISE**: the live tree's new defaults = comb4 + every
  flag (the KN10R2 config), all history keys. KN10N1 submitted with submit_prod.sh: fit 41815294 -> KN10N1FR 41815295.
- **24 Oct, Pierre's OK: (a) production ensemble KN10N1-16 with the new recipe; (b) dE/dx mean-constraint diagnostic.**
  - (a) KN10N<s> = KNOT/inner_sci_full_KN10S<s>.sh with only --test_name changed, launched by submit_prod.sh (fit +
    freeze-release continuation KN10N<s>FR), samples 2-16 as 3 dependency chains whose heads wait for KN10N1FR
    (afterok); cancelled if KN10N1 fails its rules. Readout when ALL 16 FR runs are done: per-parameter mean +- SD/sqrt(16)
    and SD of the raw end errors at FR @ 14000, mean pos_residual 13-14k; descriptive comparison with the old KN10S
    ensemble @ 1e4, paired by sample.
  - (b) MW: joint_polish_dp.py --mw (new option, default = the run's own 2500), sample 1, --geom truth, from KN10S1 @ 1e4,
    nsteps 6, eps 3e-3, DPW 0 (= TG2's setup). mwA2500 = same-tree baseline L0 (tree check: within +-1 point of TG2's
    -5.36 %), mwA250 (x0.1), mwA25000 (x10). **Verdict: THE MEAN CONSTRAINT SETS THE OFFSET if |L(w) - L0| >= 1.5 points
    for at least one weight and L(250), L0, L(25000) are monotone; NOT THE MEAN CONSTRAINT if both |L(w) - L0| <= 1.0;
    PARTIAL otherwise.** Converged = last accepted step moves lifetime < 0.3 points, else indicative only.
  - Submitted 08:27: ensemble chains via KNOT/submit_prod_chains.sh (heads afterok:41815295 = KN10N1FR; next sample
    afterany:<prev FR>?afternotok:<prev fit>): A = N2 N5 N8 N11 N14, B = N3 N6 N9 N12 N15, C = N4 N7 N10 N13 N16
    (jobs 41818174-41818203; ids in KNOT/submitted.txt). MW jobs 41818149-51 (mwA2500 / mwA250 / mwA25000; the first
    submission 41818140-2 lacked the speed-bundle exports of the DP jobs and was cancelled within seconds). Readout
    POLISH/joint/read_mw.py. At most 6 running (3 MW + 3 chains).
- 24 Oct 09:56: **KN10N1 passes both rules.** Fit 41815294 COMPLETED in 1 h 26 min (MaxRSS 7.9 GB), KN10N1FR 41815295
  COMPLETED in 40 min. (1) read_pkg.py KN10N1 @ 1e4: max |z| = 1.87 (eField) -> CONSISTENT (Ab z +1.12, tran_diff +0.79,
  long_diff -0.63, lifetime -1.17, pos +0.76). (2) KN10N1FR reaches 14000; pos_residual 13-14k 110.0 um vs PRF1 113.1 um
  = 0.973 -> PIPELINE OK. The ensemble chains continue (heads N2/N3/N4 started 09:51). The automatic check printed
  nothing: the waiter ran the readouts through an unquoted `$C` command string, which zsh does not split into words; rerun
  by hand under bash.
  Ensemble readout script (written before any N2-N16 result): KNOT/read_ens.py. Error % = mean of the last 1000
  iterations / target - 1 (compare_arms.read convention; the raw last value is printed too), positions = mean
  pos_residual of the last 1000 iterations; refuses to print unless all 16 FR runs are at 14000.
- **24 Oct 11:07, MW block (dE/dx mean-constraint weight; rule above) = NOT THE MEAN CONSTRAINT.** read_mw.py; jobs
  41818149-51 COMPLETED (2 h 35 - 2 h 41 each; sample 1, truth geometry, 6 accepted steps each).
  | weight | lifetime | A_b | tran_diff | long_diff | last step |
  |---|---|---|---|---|---|
  | 250 (x0.1) | -5.37 | +0.52 | +0.98 | +3.41 | -0.34 (indicative) |
  | 2500 (L0) | -5.37 | +0.50 | +0.98 | +3.40 | -0.34 (indicative) |
  | 25000 (x10) | -5.19 | +0.39 | +0.74 | +3.99 | -0.06 (converged) |
  Tree check L0 -5.37 vs TG2 -5.36: PASS. L(250) - L0 = -0.00, L(25000) - L0 = +0.18, both within 1.0 point.
  The weight reaches the objective (prior term R0 -48 730 / -48 720 / -48 540 at step 0). The 250 and 2500 runs follow
  the same lifetime path step by step (-0.41, -4.79, -4.79, -5.03, -5.03, -5.37), so their 0.04-point miss of the
  convergence cut does not affect the comparison.
  - Reading (not part of the rule): with positions (TG2), the drift-profile trend (DP) and now the mean constraint
    excluded, the untested ingredients left are the representation (truth-projected spline + guess segmentation) and
    the split-t prior shape. The earlier floors point to the representation: truth drift + truth dE/dx -4.1 +- 0.7
    (DRIFTDXK6), MAP dE/dx at true geometry -2.4 +- 2.1 (L3), harness on exact tracks -0.5. The joint-step polish is already pruned as a production tool (KN10C 16-sample step WORSENS). This
    thread stops here: no further joint-minimum jobs.
- **24 Oct 20:43, PRODUCTION ENSEMBLE KN10N1-16 (new defaults + recipe) = COMPLETE, 30/30 jobs COMPLETED** (fits
  1 h 24 - 1 h 29, continuations 39 - 43 min; samples 2-16 ran 09:51 - 20:40 on 3 GPUs). KNOT/read_ens.py, end point
  FR @ 14000, error % = mean of the last 1000 iterations; n = 16, mean +- SE (SD):
  | | A_b | eField | tran_diff | long_diff | lifetime | pos (um) |
  |---|---|---|---|---|---|---|
  | KN10N FR @ 14000 (new) | +0.17 +- 0.03 (0.13) | +0.03 +- 0.00 (0.01) | +0.81 +- 0.26 (1.03) | +2.28 +- 0.84 (3.37) | +0.54 +- 0.79 (3.14) | 124.6 +- 3.9 (15.4) |
  | KN10S @ 1e4 (pre-stated comparison) | +0.11 +- 0.03 (0.13) | +0.03 +- 0.00 (0.01) | +1.00 +- 0.26 (1.05) | +4.21 +- 1.29 (5.17) | +2.26 +- 1.00 (4.00) | 145.3 +- 4.2 (16.7) |
  | KN10C @ 14000 (13 Oct continuation; added after the rule, equal length) | +0.18 +- 0.03 (0.12) | +0.03 +- 0.00 (0.01) | +0.61 +- 0.27 (1.07) | +1.18 +- 1.11 (4.45) | +0.14 +- 0.79 (3.16) | 138.7 +- 3.9 (15.6) |
  - Paired new - KN10S: pos -20.7 +- 1.1 um (ratio 0.857, range 0.82-0.90, 16/16 lower); long_diff -1.93 +- 1.02;
    lifetime -1.72 +- 0.85; A_b +0.06 +- 0.03; tran_diff -0.19 +- 0.22. This mixes the 4000 extra iterations with the
    recipe change.
  - Paired new - KN10C (same length; isolates the new continuation, plus round-off from the speed defaults): pos
    -14.0 +- 0.8 um (ratio 0.898, range 0.87-0.94, **16/16 lower**); calibration unchanged within errors: lifetime
    +0.40 +- 0.65, long_diff +1.10 +- 0.84, tran_diff +0.20 +- 0.19, A_b -0.01 +- 0.02. RMS errors new / KN10C: A_b 0.21 /
    0.22, tran_diff 1.28 / 1.20, long_diff 3.98 / 4.46, lifetime 3.09 / 3.06. = the FB block's finding (positions
    0.91-0.95x, calibration unchanged) now on 16/16 samples. Descriptive: the two ensembles straddle the 24 Oct
    round-off boundaries.
  - Paper NOT told (Pierre: "don't touch the paper yet"). Ready to hand over on request: KN10N FR @ 14000 as the
    production ensemble, cost ~2.1 h per sample end to end on one A100 (was ~6 h fit + continuation).
- 24 Oct 22:55: **speed work committed (Pierre's OK)** on dedx_soft_barrier, four commits, each a state the live tree
  ran: ab27ba1 (fit code before the package: 21 Oct speed fixes + study flags, all default off), e460ea5 (22 Oct bitwise
  package, default on; = KN10P2), 0e6ab67 (round-off flags; = KN10R1 / DP / FB runs), 19d2bb3 (round 2; = KN10R2 and the
  KN10N1-16 ensemble). REVIEW.md, make_ts2_fig.py and scripts/recipe/fig_recipe.py stay uncommitted. The opt worktrees
  and comb4 were removed (diffs kept in /sdf/scratch/neutrino/pgranger/opt/reports/worktree_diffs/).

## 24 Oct 2026 — ENDFIT resumed (Pierre: "yes"): option (b), batch zero-mean drift constraint on the fitted endpoints
- **Code** (optimize/fit_params.py, uncommitted): `LARND_ENDPOINT_DRIFT_ZERO_MEAN=1` (needs LARND_FIT_ENDPOINTS=1).
  After every chain Adam step the batch's e2 endpoint coefficients are projected onto sum_j w_j c_j = 0, w_j = change
  of endpoint j's drift distance |z - z_anode| per unit coefficient (e2 . z times sign(z - z_anode) of that end's TPC).
  The batch-mean drift shift of the endpoints is then exactly 0; per-track shifts stay free. Frozen e2 halves are left
  out. Minimum-norm projection of the coefficients; Adam moments untouched. Hooked in _process_chain_grads (all three
  update paths: per-track loop, flat eager, flat jitted). Equivalent to trusting the batch-average endpoint estimate in
  drift (t0 known) while fitting each end. Why: 14 Oct EFT0 used the endpoint freedom for a coherent +35..+45 um shift
  away from the anode (lifetime +5..+11 points).
- **CPU checks before the GPU block** (ENDFIT/cpu_check_b.sh): (1) HEAD vs live, flag off: bitwise; (2) endpoint path
  under the speed defaults vs every speed flag off: round-off level; (3) constraint on (with and without the drift
  freeze): batch sums ~0 on the final state (ENDFIT/check_zero_mean.py). The GPU block is submitted only if all pass.
- **Test block** (KNOT/inner_sci_full_<ARM><s>.sh = KN10N<s> with only the env lines and --test_name changed; production
  recipe via submit_prod.sh: 1e4 fit + freeze-release continuation, end point <ARM><s>FR @ 14000), samples 1, 2, 4:
  - EZT = true endpoints, FITTED + constraint (control: the cost of the constrained freedom);
  - EZS = endpoints smeared 1 mm per coordinate (LARND_ENDPOINT_SMEAR_MM=1.0, seed 0 = the ENDSM10 / EFT10
    realisation), FITTED + constraint (the test);
  - EXS = smeared 1 mm, HELD (the smear's cost in the current recipe; ENDSM10 gave +12.8 under the old one);
  - reference: KN10N<s>FR (true endpoints, held; same code, flag off = bitwise by check 1).
- **Readout rule, stated before any result** (ENDFIT/read_endc.py; read only when all 18 jobs are done; errors = mean
  of the last 1000 iterations at 14 000; shifts paired per sample vs KN10N<s>FR). A single paired lifetime difference
  carries ~2.6-3.5 points of run-to-run noise (KN10N vs KN10C SD 2.6; same-data replicas 3.5), so the rule uses the
  3-sample mean (noise ~1.7):
  1. Control EZT: **ANCHOR HOLDS** if the mean lifetime shift is within +-3 points and no sample exceeds +6 (EFT0 under
     the old recipe: +5.2 / +11.4 / +8.5); **ANCHOR FAILS** if the mean is >= +4.5; else UNRESOLVED.
  2. Reference EXS: a valid lever if its mean lifetime shift is >= +6 points; if smaller, the smear costs less in this
     recipe and (3) is judged against that smaller lever.
  3. Test EZS: **ENDFIT WORKS** if (1) holds, the mean lifetime shift is within +-3 points and the mean long_diff shift
     within +-5 points; **PARTIAL** if (1) holds and the mean lifetime shift is at most half of EXS's; else **FAILS**.
  - Also read (no rule): A_b, eField, tran_diff shifts, positions, the constraint residual on the final states, and
    the transverse endpoint error before -> after (EZS should move the ends toward the truth; EZT shows how far the
    fit pulls true ends away).
- **CPU checks, 24-25 Oct (ENDFIT/cpu/*-2307, all three PASS):** (1) head_def vs live_def: ALL BITWISE. (2) ep_def vs
  ep_off after 6 iterations: loss 5.9e-7, calibration gradients <= 5.7e-6, pos_residual 4.8e-8, dedx_cache 7.4e-5,
  chain_cache 2.2e-5 max rel (the 0.57 "worst series" is step_time, wall clock) -> round-off level. (3) |sum w.c| /
  (rms sqrt n) = 9.2e-1 / 3.3e-1 free (ep_def: batch-mean drift shift +101 / +9 um after 6 iterations) vs 7.6e-8 /
  1.6e-7 with the constraint (ep_zm and ep_zm_rel, identical since no track was frozen in this 6-track test); log
  "[CHAIN] batch b: endpoint drift zero-mean constraint on 2/2 (4/4) tracks". Untested on CPU: the frozen-track
  exclusion (index 2K-1 lies in the mask's zeroed e2 half m[K:], checked by reading); checked live in the GPU logs
  (constraint count + FROZEN count = batch size).
- **Submitted 25 Oct 00:31** (submit_prod_chains.sh none, 5 chains, <= 5 running): EZT1 41873499 -> EZT1FR 41873500,
  EXS1 41873501 -> 41873502; EZT2 41873503 -> 41873504, EXS2 41873505 -> 41873506; EZT4 41873508 -> 41873509, EXS4
  41873510 -> 41873511; EZS1 41873512 -> 41873513, EZS4 41873514 -> 41873515; EZS2 41873517 -> 41873518. Waiter
  ENDFIT/wait_endc.sh (all 18 done -> sacct + read_endc.py).
- **Live check (25 Oct 00:36, ENDFIT/live_check.sh, the 5 chain-head fit logs):** per batch, constraint tracks +
  FROZEN tracks = batch size on all 38 batches read (0 mismatches); the constraint covers 78-91% of the tracks in the
  frozen-drift fit stage (EZT1 71/81, EZT2 47/60, EZT4 39/44, EZS1 73/80, EZS2 56/67); no Traceback / Error / nan.
  The frozen-track exclusion is now checked on real batches.
- **RESULT 25 Oct 05:26 (all 18 jobs COMPLETED; ENDFIT/read_endc.py; paired shift vs KN10N<s>FR, points, s = 1 / 2 / 4; mean):**

  | arm | lifetime | long_diff | tran_diff | A_b | pos |
  |---|---|---|---|---|---|
  | EZT (true, fitted + constraint) | -0.3 / -1.6 / -2.1; **-1.3** | -3.6 / -7.6 / +0.2; -3.7 | +0.3 | -0.04 | -1.6 |
  | EZS (1 mm, fitted + constraint) | +23.8 / +15.7 / +18.8; **+19.4** | +23.3 / +27.4 / +44.5; **+31.8** | -2.8 | -0.6 | +627 |
  | EXS (1 mm, held) | +20.3 / +11.2 / +18.6; **+16.7** | +40.8 / +20.2 / +32.6; +31.2 | -3.6 | -0.6 | +725 |

  Rule: (1) **ANCHOR HOLDS** (EZT mean -1.3, max -0.3; 14 Oct EFT0 without the constraint +5 / +11 / +9): the
  constraint removes the freedom cost completely, positions unchanged. (2) EXS lever valid (+16.7 >= +6). (3) **EZS
  FAILS** (+19.4, long_diff +31.8): no better than holding the smeared ends. Constraint residual on the final states
  1-2e-7 (all EZT / EZS). Transverse endpoint error rms (um) e1 / e2: EZS 856-882 / 973-1000 -> 467-484 / 543-555; EZT
  0 -> 243-261 / 131-136.
- **Diagnosis (ENDFIT/shrink.py, per endpoint: fitted term a vs the smear b):** slope a~b = 0.55-0.56 (e1) / 0.42-0.45
  (e2) at the fit end (10 000), 0.66-0.67 / 0.55-0.58 after the continuation (14 000): the ends move about half way and
  are STILL moving (100 batches -> each track gets only ~100 + 40 Adam updates). Scatter about the line ~340-375 um.
  Not prior-limited: a 1 mm prior at a stationary slope ~0.6 would imply a data precision ~0.8 mm per end and EZT
  displacements ~0.5 mm, but EZT moves the true ends only 0.13-0.26 mm -> the data constrain the ends better than the
  prior; the fit has not reached them. Halving the endpoint error (EXS -> EZS) buys NOTHING in lifetime / long_diff:
  the calibration bias needs the residual drift error near the EZT level (~150 um, i.e. below the longitudinal
  diffusion width), not merely smaller than 1 mm.
- **Consequence for the plan:** the drift anchor, which (a) e1-only and (c) t0 nuisance were meant to provide, is now
  SOLVED by (b); the failure is endpoint recovery from a 1 mm start. (a) / (c) do not address it. Candidates: (d) seed
  the endpoint terms from the hits (the e1 hit seed already estimates per-track offset and tilt = the endpoint columns;
  e2 from the hit times, t0 known); (e) diagnostic: a long continuation of EZS<s>FR to see whether converged ends
  recover the calibration at all.

## 25 Oct 2026 — ENDFIT (e): long continuation (Pierre: "Launch (e)"), and (d) hit-based endpoint seed (build started)
- **(e) design** (ENDFIT/make_long_cont.py -> KNOT/inner_sci_full_<X>L.sh, diffed vs the <X>FR scripts: only the listed
  changes): resume <X>FR @ 14000 for 30000 more iterations (-> 44000; ~300 more Adam updates per track instead of 40);
  calibration LR 1e-3 held, then x0.805 per epoch from 42000 (same end annealing as the production continuation);
  chain LR re-heat 2e-3 -> 6e-4 (decay 0.99996 / iteration from 14000); dE/dx never frozen; freeze released;
  save_freq 2000. Runs: EZS1L, EZS2L, EZS4L and the paired references KN10N1L, KN10N2L, KN10N4L (the same extra
  iterations on true, held endpoints), 6 jobs at once, ~6 h. EZS checkpoints at 16000, 20000, ..., 40000 kept as hard
  links in ENDFIT/long_ckpt/ for the convergence trajectory.
- **Readout rule (stated before any result; ENDFIT/read_long.py, read only when all 6 are done; paired EZS<s>L vs
  KN10N<s>L at 44000, errors = mean of the last 1000 iterations):** converged = the endpoint slope a~b (shrink.py) >= 0.9
  on both e1 and e2 for all 3 samples.
  1. converged and the mean lifetime shift within +-3 points (long_diff within +-5): **CONVERGED ENDS RECOVER** the
     calibration -> (d) is the route (it must reach the same endpoint accuracy, faster).
  2. converged and the mean lifetime shift >= +8: **CONVERGED ENDS DO NOT RECOVER** -> the cost is not the endpoint
     error alone (e.g. the path from a wrong start, or the dE/dx / geometry state built on it); (d) still useful for
     data but not sufficient.
  3. not converged: **STILL CONVERGING** -> report the slope and the calibration shift vs iteration (16000 ... 44000);
     the question stays open.
  4. otherwise (converged, +3 < shift < +8): **PARTIAL RECOVERY**.
- **(e) submitted 25 Oct 10:44** (6 jobs at once, ~6 h): EZS1L 41912038, EZS2L 41912039, EZS4L 41912040, KN10N1L
  41912041, KN10N2L 41912042, KN10N4L 41912043. Watcher ENDFIT/wait_long.sh (hard links + read_long.py; readout
  functions checked on EZS1FR @ 14000: identical to read_ens / shrink.py).
- **FINDING 25 Oct ~10:50 (before any (e) result; ENDFIT/smear_coherence.py): the LARND_ENDPOINT_SMEAR_MM draw is not
  independent per track.** It is seeded by (eventID, trackID), which are batch-LOCAL ids: 39 / 43 / 39 distinct keys
  over the ~960 tracks of samples 1 / 2 / 4, so the same few dozen smear vectors repeat in every batch and every
  sample. Consequences: (i) the given (smeared) ends are coherently closer to the anode, -66 / -43 / -69 um on average
  (independent draws: +-24), with the same sign in all three samples; samples 1 / 2 / 4 are not independent replicates
  of the smear (affects ENDSM10, EFT10 of 14 Oct and EXS / EZS / (e)); (ii) the batch means of the given drift shift
  scatter +-220 um (natural for ~20 ends of 1 mm, 1000/sqrt(20)), and the per-batch zero-mean constraint PINS each
  batch to its own given mean: EZS can correct the within-batch errors but never the batch-mean drift error (+-220 um
  per batch), which EXS carries too. This may be why EZS = EXS in lifetime although EZS halves the per-track error;
  (iii) not the cause of the slow slope: the best e2 slope the constraint allows is 0.95-0.96.
  - Interpretation note for (e) (rule unchanged): EZS<s>L keeps the per-batch constraint, so a "CONVERGED ENDS DO NOT
    RECOVER" verdict would not separate "converged ends do not help" from "the batch-mean drift errors pinned by the
    constraint carry the cost". (e) still answers whether within-batch convergence alone recovers the calibration.
  - Code (uncommitted, default = legacy so the running (e) resumes rebuild the same smear):
    LARND_ENDPOINT_SMEAR_KEY=entry keys the draw on the true entry point (10 um grid): independent per physical
    track, identical across resumes. Smeared runs now also save the drawn offsets (training_history['endpoint_smear']).
  - Natural refinement of (b): pin the GLOBAL mean drift shift (all batches; +-24 um with independent draws) instead
    of each batch's.
- **(d) build, 25 Oct (uncommitted, default off):** LARND_HITS_SEED_ENDPOINTS=e1 (needs FIT_ENDPOINTS + DRIFT_FROM_HITS):
  the constant + linear terms of the hit fit in _seed_drift_from_hits (discarded so far) are the start / end
  displacements (start = c0, end = c0 + c1; same design matrix as the endpoint columns); e1 applied (5 mm cap), e2
  (hit times, carries the threshold-crossing latency) recorded only. LARND_SEED_DIAG_ONLY=1: seed every batch once, no
  step, save, stop. Estimates + standard errors saved as training_history['endpoint_seed']. CPU dry run on sample 1
  (as EZS1) + default-path bitwise check running (ENDFIT/seed_dry.sh); analysis ENDFIT/seed_quality.py.
- **(d) dry run RESULT (25 Oct 11:10, sample 1 as EZS1, 859 of 962 tracks seeded; ENDFIT/seed_quality.py):**
  - default path: d_head_def vs d_live_def ALL BITWISE (live run started 11:14, after every edit of the day).
  - e1 (pixel plane) hit estimate: unbiased (slope 0.99) but noisy: rms error 807 um vs 881 um before (smear); 593 um
    on the 88% of ends with se > 500 um, 1.5-2 mm on the rest. The fit's standard errors are unusable here (pull rms
    259): pixel-centre quantisation makes the residuals of grid-aligned tracks identical, so the MAD collapses.
    Reference: the EZS1 fit itself reaches 467 um (14000 iterations) -> the e1 seed would start the fit WORSE than
    where it already gets; at best a modest speed-up (combined with the given ends ~0.55 mm).
  - e2 (drift, hit times): raw rms 570-700 um vs ~1000 before, but BIASED by a drift-dependent latency: median
    drift-distance residual +137 um (d < 10 cm), -356 (10-20), -313 um (20-31); linear fit +163 - 24.9 um/cm x d.
    A drift-dependent bias of the starting ends is the lifetime-degenerate mode itself; removing only the median
    leaves ~690 um. Not usable as a drift seed without a latency model, which would itself need a drift anchor.
  - Verdict (d): the hits do not deliver endpoints near the ~150 um drift accuracy the calibration needs, and the
    free fit already does better than the seed. e2 seeding not pursued; e1 seeding kept as an option (default off),
    not a priority. Better levers: the endpoint fit's own convergence ((e)), the global-mean constraint, a per-track
    smear (LARND_ENDPOINT_SMEAR_KEY=entry).
- **(e) RESULT (25 Oct 15:15, all 6 COMPLETED, 4.0-4.5 h; ENDFIT/read_long.py): literal verdict STILL CONVERGING
  (the endpoint slope reaches 0.82-0.84, < 0.9), but the calibration criteria of branch 1 are met.**
  - Paired EZS<s>L - KN10N<s>L @ 44000 (s = 1 / 2 / 4; mean): lifetime +1.75 / +1.46 / -0.85, **+0.79** (14000:
    +19.4); long_diff +6.50 / +3.86 / +1.19, **+3.85** (14000: +31.8); tran_diff -1.24; Ab -0.13; eField +0.03; pos
    +601 um (the endpoint error itself). Absolute EZS<s>L lifetime +0.32 / +0.06 / -0.67, long_diff +9.2 / +6.8 / +5.6.
  - Trajectory of the paired mean lifetime shift: 19.4 (14000), 15.8 (16000), 10.9 (20000), 7.2 (24000), 5.7, 4.1,
    2.1 (36000), 0.9 (40000), 0.8 (44000); long_diff 31.8 -> 14.8 (20000) -> 6.7 (28000) -> 3.5-3.9 (40000-44000).
  - Endpoint slope a~b e1 | e2 (rms a-b um): 0.67 (467) | 0.58 (544) at 14000 -> 0.83 (381) | 0.83 (376) at 44000
    (s1; s2 / s4 alike), flat to +-0.01 over the last 8000 iterations.
  - POST-HOC (not in the rule; labelled as such): the 0.9 slope threshold was not reachable. The endpoint prior is
    1 mm per coefficient (default LARND_ENDPOINT_PRIOR_MM), matched to the smear, so the converged MAP is the
    posterior mean, shrunk by s = sigma_p^2 / (sigma_p^2 + sigma_d^2) < 1. Check: the reverse regression b~a is 1 for a
    calibrated posterior mean and 1/s for an unfinished fit a = s b. Measured (ENDFIT/rev_slope.py): e2 rev
    1.29-1.30 at 14000 -> 1.09-1.10 at 24000 -> 1.03 at 44000; e1 0.96-0.97 at 44000 (e1 realised smear 856-882 um <
    the 1 mm prior, expected < 1). s = 0.83 gives sigma_d ~ 450 um per endpoint and a predicted rms error
    sigma_p sqrt(1 - s) ~ 410 um; measured 369-411. Reading: at 44000 the fitted ends sit at (e2: within ~3 % of)
    their posterior mean; slope >= 0.9 would need sigma_d <= 330 um.
  - Reading: given ~300 more updates per track, the fitted ends converge and the calibration recovers to the
    true-endpoint reference within 1 lifetime point (long_diff +3.9). The per-track endpoint error that remains
    (~380 um) is far above the "~150 um" guessed on 25 Oct from the 14000 numbers; that guess is withdrawn. Open: (i)
    whether the long run alone removes the 14000 bias, endpoints aside (held-smear control) and whether the anchor
    holds over 30000 more iterations (true-end control) -> ENDFIT (f) below; (ii) samples 1 / 2 / 4 share the smear
    vectors (smear-key FINDING above), so they are not independent replicates of the smear; (iii) cost 44000
    iterations (~5.8 h) vs 14000.
- **ENDFIT (f), 25 Oct (before any result): the two long-continuation controls of (e)**, same tool and settings
  (ENDFIT/make_long_cont.py from the block-(b) <X>FR @ 14000 checkpoints; KNOT/inner_sci_full_<X>L.sh diffed vs
  <X>FR and vs EZS1L: only the intended fields differ), paired with the existing KN10N<s>L @ 44000:
  - EXS1L, EXS2L, EXS4L: 1 mm smear, ends HELD (not fitted).
  - EZT1L, EZT2L, EZT4L: true ends, fitted, per-batch zero-mean drift constraint.
  - **Rule (ENDFIT/read_ctrl.py; checked at 14000 on the <X>FR checkpoints: reproduces the block-(b) table exactly;
    read only when all 6 are done):**
    1. EXS, mean paired lifetime shift: >= +8 **HELD ENDS STAY BIASED** -> the (e) recovery is due to the fitted
       ends; within +-3 (long_diff within +-5) **LONG RUN ALONE RECOVERS** -> the 14000 bias was slow calibration
       convergence and (e) says nothing about endpoint fitting; otherwise **PARTIAL** (report the fraction of the
       (e) recovery that the held run also makes).
    2. EZT: **ANCHOR HOLDS LONG** if the mean lifetime shift is within +-3 and no sample exceeds +6 (the block-(b)
       anchor rule); **FREEDOM COST GROWS** if the mean >= +5; otherwise **MARGINAL**. Also reported: rms of the
       fitted endpoint coefficients and their mean drift shift at 14000 and 44000.
  - **(f) submitted 25 Oct 15:37** (6 jobs at once, ~4.5 h): EXS1L 41932229, EXS2L 41932230, EXS4L 41932231, EZT1L
    41932232, EZT2L 41932233, EZT4L 41932234. Watcher ENDFIT/wait_ctrl.sh (sacct + read_ctrl.py).
- **(f) RESULT (25 Oct 20:25, all 6 COMPLETED, 4.0-4.4 h; ENDFIT/read_ctrl.py): EXS = LONG RUN ALONE RECOVERS; EZT =
  ANCHOR HOLDS LONG.**
  | paired vs KN10N<s>L @ 44000 (s = 1 / 2 / 4; mean) | lifetime | long_diff | tran_diff | Ab | pos (um) |
  |---|---|---|---|---|---|
  | EXS<s>L (1 mm smear, ends held) | +4.77 / +1.73 / -0.83; **+1.89** | +7.87 / +5.01 / +0.41; **+4.43** | -1.48 | -0.32 | +702 |
  | EZS<s>L (1 mm smear, ends fitted; (e)) | +1.75 / +1.46 / -0.85; **+0.79** | +6.50 / +3.86 / +1.19; **+3.85** | -1.24 | -0.13 | +601 |
  | EZT<s>L (true ends, fitted) | -1.65 / +1.71 / -0.84; **-0.26** | +1.40 / +1.83 / +1.55; **+1.60** | -0.99 | -0.03 | +2 |
  - Trajectory of the paired mean lifetime shift, EXS (held) vs EZS (fitted): 16.7 vs 19.4 (14000), 10.7 vs 10.9
    (20000), 6.1 vs 7.2 (24000), 5.0 vs 4.1 (32000), 3.4 vs 0.9 (40000), 1.9 vs 0.8 (44000); long_diff 31.2 vs 31.8
    -> 10.3 vs 8.9 (24000) -> 4.4 vs 3.9 (44000). The same relaxation, held or fitted.
  - EZT fitted ends (truth = 0): rms e1 261 -> 183-207 um, e2 131-136 -> 91-97 um, mean drift shift 0.0 (pinned).
  - Reading (rule 1): the large 14000 bias of the 1 mm smear runs (EXS +16.7, EZS +19.4; also ENDSM10 +12.8 of 8 Oct
    and EFT10 of 14 Oct, all read at 1e4 or 14000) is mostly SLOW CALIBRATION CONVERGENCE, not the endpoint error:
    held at 1 mm off, the ends cost +1.9 lifetime / +4.4 long_diff once converged. Fitting the ends buys ~1 point of
    lifetime (mostly sample 1) and -100 um of position; with 3 samples sharing their smear vectors that is not
    resolved. The (e) recovery is NOT evidence for endpoint fitting.
  - The true-endpoint reference also moves over the long run: KN10N<s> lifetime -5.46 / -0.10 / +1.20 (14000) ->
    -1.43 / -1.41 / +0.18 (44000), long_diff +2.16 / +3.87 / -0.82 -> +2.73 / +2.97 / +4.37: the production end point
    (14000) is not fully converged either (sample 1: 4 points); n = 3, production recipe unchanged.
  - Caveats: smear vectors shared by samples 1 / 2 / 4 (smear-key FINDING); one smear level (1 mm); 44000 iterations.
- **ENDFIT (g), 25 Oct ~21:00 (before any result): the (f) / (e) test with an INDEPENDENT per-track smear**
  (LARND_ENDPOINT_SMEAR_KEY=entry, 1 mm; removes the shared-smear caveat). Arms, samples 1 / 2 / 4:
  - EXP<s>: ends held (= EXS<s> + SMEAR_KEY=entry); EZP<s>: ends fitted + per-batch zero-mean drift constraint (= EZS<s>
    + SMEAR_KEY=entry). Inner fit scripts KNOT/inner_sci_full_{EXP,EZP}<s>.sh diffed vs the EXS / EZS ones: only the
    export and the test name differ.
  - Each run is a chain of 3 jobs: the production pair (KNOT/submit_prod.sh: 1e4 fit -> <X>FR @ 14000), then
    <X>L @ 44000 (KNOT/inner_sci_full_<X>L.sh builds the long continuation at job start with ENDFIT/make_long_cont.py,
    the (e) / (f) tool). Paired with KN10N<s>FR (14000) and KN10N<s>L (44000). 6 chains, <= 6 running.
  - CPU pre-check (ENDFIT/smear_key_dry.sh + smear_key_check.py, samples 1 and 2, no step): distinct smear vectors =
    2 x tracks, rms ~1000 um per axis, coherent drift offset within ~2 SE of 0, no vectors shared across samples
    except physical tracks present in both. Submission only if it passes.
  - **Rule (ENDFIT/read_g.py; read only when all 18 jobs are done):**
    1. EXP @ 44000, mean paired lifetime shift within +-3 and long_diff within +-5: **CONFIRMED** (held 1 mm ends cost
       little once the calibration converges); lifetime >= +6 or long_diff >= +10: **NOT CONFIRMED** (the (f)
       recovery was specific to the shared smear); otherwise **WEAK**.
    2. EXP - EZP @ 44000: **FITTING HELPS** if EZP is lower than EXP in both lifetime and long_diff shift on all 3
       samples and by >= 1 lifetime point on average; otherwise **NO RESOLVED GAIN**.
    Also reported: both arms at 14000 (the production end point), the trajectory, the EZP endpoint slopes (forward
    and reverse, from the smear stored in the checkpoint).
- **CORRECTION 25 Oct ~21:30 (ENDFIT scratch real_only.py; smear_key_dry.sh): every batch carries one placeholder
  chain context, track_id (-1, -1), 0.185 cm, no hits (99 of 100 batches; fitted endpoint coefficients <= 11 um).** It
  was counted in the smear statistics and endpoint slopes above (~10 % of the ends). On REAL tracks only:
  - the legacy "coherent drift offset" vanishes: +9.8 / +32.6 / +5.5 um (SE ~25) instead of -66 / -43 / -69. FINDING
    (i) of the smear-key entry is WITHDRAWN; the shared-key part (samples share smear vectors) stands.
  - endpoint forward slope at 44000: e1 0.87 / 0.85 / 0.87, e2 0.85 / 0.82 / 0.86 (not 0.82-0.84); rms a-b 347-393
    um; reverse slopes unchanged (placeholder a ~ 0). The (e) / (f) calibration numbers are unaffected.
- **(g) CPU pre-check PASSED (25 Oct 20:48; ENDFIT/smear_key_dry.sh):** samples 1 / 2: all real tracks have distinct
  smear vectors (the only repeat is the placeholder, keyed on its fixed entry point); rms 950-1080 um per axis;
  coherent drift offset +34.9 +- 22.0 / -28.8 +- 21.3 um; 12 vectors shared across the samples (tracks present in
  both). (g) NOT submitted yet: Pierre asked why 44000 iterations are needed (below).
- **Why 44000 (25 Oct, Pierre: "Is it an LR issue?"; ENDFIT scratch steps.py / early.py):**
  - Sigmoid normalization: one Adam step at LR 1e-3 can move lifetime by <= 0.054 % (long_diff 0.039 %). Measured,
    14500-41500 (LR 1e-3 held): step rms 0.21-0.28 x that (the pure-noise Adam value ~0.23 for b1 = 0.9; lag-1 corr
    0.9), net drift 1-3 % of the LR-limited one (lifetime -0.4..-0.6 % / 1000 its). The per-iteration calibration
    gradient (one batch of ~9 tracks, stochastic simulation) is ~99 % noise; Adam drifts at ~LR x signal/noise.
    Mean dE/dx moves < 0.15 % over the long run: not a dE/dx co-adaptation stall.
  - In the 1e4 fit (LR 1e-1 x 0.91 per epoch) the calibration is effectively frozen from ~5000 (LR 1.4e-3 -> 1.3e-5
    at 10000): EXS1 lifetime +19.4 (5000) +18.3 (10000), long_diff +52.7 / +50.7; KN10N1 -5.0 / -5.7. The
    production continuation re-heats to 1e-3 only for 10000-11500 (budget ~1-2.5 % lifetime). The calibration keeps
    the offset it has at ~5000, when geometry and dE/dx are still converging; with 1 mm smears that offset is large.
  - Correction to the line above: the production continuation holds 1e-3 for 2000 iterations (10000-12000; the decay
    starts at warmup_steps + transition_begin = 12000), not 1500. Checked with optax on the restored count (14000).
- **ENDFIT (h), 26 Oct (Pierre: "Do what you think is best"; before any result): FAST continuation = the schedule
  test.** If the calibration is noise-limited (drift ~ LR x signal/noise), 10x the LR should do in ~2500 iterations
  what the long run did in 28000.
  - ENDFIT/make_fast_cont.py (= make_long_cont.py with): calibration peak LR 1e-2 (ENDFIT/param_list_cont_1e-2.yaml;
    with a YAML dict the per-parameter values are the peak LRs, --lr is ignored), held 14000-16500, then x0.805 per
    epoch -> 5.0e-6 at 20000 (optax-checked); integrated LR ~30 vs the long run's ~28.5; chain LR 2e-3 -> 6e-4 over
    14000-20000 (decay 0.9998; the long run's start and end values); dE/dx never frozen; freeze released.
  - Runs EXS1H, EXS2H, EXS4H (1 mm smear, ends held), KN10N1H, KN10N2H, KN10N4H (true ends), from <X>FR @ 14000;
    KNOT/inner_sci_full_<X>H.sh diffed vs <X>L: only the listed fields differ. ~1 h each, 6 at once.
  - **Rule (ENDFIT/read_fast.py, tested with the L runs as stand-ins; read only when all 6 are done):**
    1. EXS<s>H - KN10N<s>H @ 20000, mean lifetime within +-3 and long_diff within +-5: **FAST RECOVERS** (the
       44000 iterations were the LR schedule, not the problem); lifetime >= +8: **FAST DOES NOT RECOVER** (something
       besides the calibration LR is slow: the per-track nuisances, updated once per epoch); otherwise **PARTIAL**.
    2. <X>H @ 20000 vs <X>L @ 44000, all 6 runs within 3 (lifetime) / 5 (long_diff) points: **SAME END STATE**,
       otherwise **DIFFERENT END STATE**.
    Also reported: the 14000 / 20000 / 44000 tables, the fast trajectory (overshoot?) and the lifetime jitter.
  - If FAST RECOVERS: (g) (independent per-track smear) is run with the fast schedule; any change to the production
    recipe is Pierre's decision.
  - **(h) submitted 26 Oct 03:18:** EXS1H 41980603, EXS2H 41980604, EXS4H 41980605, KN10N1H 41980606, KN10N2H 41980607, KN10N4H 41980608; watcher ENDFIT/wait_fast.sh.
- **(h) RESULT (26 Oct 04:15, all 6 COMPLETED, 56 min each; ENDFIT/read_fast.py): rule 1 PARTIAL, rule 2 DIFFERENT
  END STATE.**
  | paired EXS<s> - KN10N<s> (s = 1 / 2 / 4; mean) | lifetime | long_diff | tran_diff | Ab |
  |---|---|---|---|---|
  | FR @ 14000 (production end point) | +20.33 / +11.19 / +18.59; **+16.70** | +40.84 / +20.23 / +32.56; **+31.21** | -3.59 | -0.59 |
  | H @ 20000 (fast, calib LR 1e-2) | +9.49 / +4.19 / +5.71; **+6.46** | +11.46 / +0.41 / +4.00; **+5.29** | -0.85 | -0.40 |
  | L @ 44000 (long, calib LR 1e-3) | +4.77 / +1.73 / -0.83; **+1.89** | +7.87 / +5.01 / +0.41; **+4.43** | -1.48 | -0.32 |
  - Fast trajectory (paired mean lifetime | long_diff): +12.2 | +22.0 (15000), +9.0 | +13.0 (16000), +7.6 | +7.7
    (17000), +6.8 | +5.8 (18000), +6.5 | +5.3 (19000, 20000); no overshoot; KN10N lifetime jitter 1.56 % (LR 1e-2) ->
    0.02 % (annealed). The long run had +10.7 at 20000 and +6.1 at 24000: the fast run reaches at 20000 what the long run
    reaches at ~24000.
  - H - L per run (lifetime, long_diff): KN10N +-1.6..2.3 / +1.3..3.0 (true ends: within rule 2); EXS +3.2 / +4.3 / +4.2
    lifetime (systematic, all 3 samples), long_diff +4.9 / -1.6 / +5.0.
  - Reading: the calibration LR WAS one limiter (long_diff fully recovered: 97 % of the long run's change in 6000
    iterations), but lifetime got only 69 % of it. Something else is slow for lifetime.
- **(h) follow-up, offline (ENDFIT scratch relax.py, 26 Oct 04:30; no jobs): the per-track nuisances of H are ~40-50 %
  relaxed.** Projection of the H change onto the long run's, c = <H - FR, L - FR> / |L - FR|^2 (1 = as far as L; real
  tracks only, placeholder excluded):
  - chain geometry: EXS 0.39 / 0.43 / 0.43, KN10N 0.36 / 0.35 / 0.35; log dE/dx: EXS 0.50 / 0.50 / 0.45, KN10N 0.48 /
    0.50 / 0.48. Mean dE/dx moves < 0.1 % (not a scale effect).
  - CONFOUND in the (h) design: the chain LR was 2e-3 -> 6e-4 over 6000 iterations, i.e. integrated chain LR ~7 vs the
    long run's ~35 (same start and end values over 30000). The geometry got 5x less LR budget; dE/dx (1e-2 constant
    in both) got 60 visits per batch instead of 300. Each track gets one Adam step per epoch (100 iterations).
- **ENDFIT (h2), 26 Oct 04:25 (before any result): (h) with the chain-geometry LR budget matched to the long run's
  (single factor vs (h)).** ENDFIT/make_fast2_cont.py = make_fast_cont.py with chain LR 1e-2 (the 1e4 fit's own
  initial chain LR) held 14000-17000, then x0.9990626 per iteration -> 6.0e-4 at 20000; integrated ~40 (H ~7, L ~35;
  the chain decay is indexed by the global iteration). Calibration (1e-2, x0.805 / epoch from 16500), dE/dx (1e-2
  constant) and everything else as (h). Runs EXS<s>H2 / KN10N<s>H2 from <X>FR @ 14000, s = 1, 2, 4;
  KNOT/inner_sci_full_<X>H2.sh diffed vs <X>H: only chain_lr, chain_decay_start, chain_decay_rate and the name differ.
  - **Rule (ENDFIT/read_h2.py, tested with the H runs as stand-ins; read only when all 6 are done):**
    1. EXS<s>H2 - KN10N<s>H2 @ 20000, mean lifetime within +-3 and long_diff within +-5: **FAST RECOVERS**;
       lifetime >= +5: **STILL SHORT**; otherwise **PARTIAL**.
    2. Paired lifetime shift H2 minus H, per sample: < 0 on all 3 and mean <= -2: **GEOMETRY LR HELPS**; otherwise
       **NO RESOLVED GAIN**.
    3. Geometry relaxation c of H2 (as relax.py): >= 0.7 on all 6 runs: **GEOMETRY LR-BUDGET-LIMITED**; < 0.5 on all
       6: **GEOMETRY VISIT-LIMITED** (more LR per step does not move it further); otherwise **MIXED**.
    Also reported: dE/dx c, H2 vs L per run, the KN10N position residual (does chain LR 1e-2 cost position?).
  - Reading planned: 1 FAST RECOVERS + 2 HELPS -> the 44000 iterations were two LR budgets (calibration and geometry)
    and a 6000-iteration continuation matches them; (g) then runs with the H2 schedule. Otherwise (g) runs with the
    long schedule, and the next suspect is dE/dx (60 visits per batch). Production recipe: Pierre's decision.
  - **(h2) submitted 26 Oct 04:27:** EXS1H2 41984914, EXS2H2 41984915, EXS4H2 41984916, KN10N1H2 41984917, KN10N2H2
    41984918, KN10N4H2 41984919; watcher ENDFIT/wait_h2.sh.
- **(h2) RESULT (26 Oct 05:28, all 6 COMPLETED, 55-57 min; ENDFIT/read_h2.py): 1 STILL SHORT, 2 NO RESOLVED GAIN,
  3 GEOMETRY LR-BUDGET-LIMITED (c 0.76-0.87, vs 0.35-0.43 in H).**
  | @ 20000 (s = 1 / 2 / 4; mean) | paired lifetime | paired long_diff | KN10N lifetime | KN10N tran_diff | KN10N pos (um) |
  |---|---|---|---|---|---|
  | H (chain 2e-3 -> 6e-4) | +9.49 / +4.19 / +5.71; **+6.46** | **+5.29** | -2.99 / +0.42 / -2.14 | -0.10 / +0.12 / +1.66 | 95 / 110 / 101 |
  | H2 (chain 1e-2 -> 6e-4) | +11.94 / +7.28 / +0.08; **+6.43** | **+8.71** | +2.69 / +3.17 / +11.24 | +2.32 / +2.36 / +8.21 | 152 / 166 / 185 |
  | L @ 44000 | **+1.89** | **+4.43** | -1.43 / -1.41 / +0.18 | +1.59 / -0.33 / +1.59 | 91 / 100 / 98 |
  - Reading: the geometry DID move further along the long run's direction (rule 3), but it brought noise with it: the
    true-end reference got WORSE in absolute terms (position +57..+84 um, lifetime +2.7..+11.2, tran_diff up to +8),
    and the EXS arm with it (lifetime +14.6 / +10.5 / +11.3). The paired shift is unchanged (+6.4) and noisier per
    sample. A 1e-2 chain LR annealed over 3000 iterations = 30 visits per track does not settle: each track gets one
    noisy Adam step per epoch, so the geometry needs many visits, not more LR. Rule-3 label is literal; it did not
    translate into calibration.
  - **Answer to "why 44000 iterations":** two slow things. (1) the calibration LR schedule (fixable: 10x LR fixes
    long_diff in 6000 iterations); (2) the per-track nuisances (one noisy step per epoch per track): raising their LR
    adds noise faster than it removes lag. With 1 mm endpoint errors, lifetime needs ~30000 continuation iterations
    with this optimizer. Untested remaining lever: dE/dx (60 visits per batch in H).
  - Per the (h2) plan: (g) runs with the LONG schedule (as designed 25 Oct; KNOT/inner_sci_full_<X>L.sh wrappers,
    ENDFIT/make_long_cont.py, checked to run in the container without the live PYTHONPATH).
- **(g) submitted 26 Oct 05:29 (long schedule, rule above unchanged):** chains fit -> FR -> L: EXP1 41986747 / 48 / 49,
  EXP2 41986750 / 51 / 52, EXP4 41986753 / 54 / 56, EZP1 41986757 / 58 / 59, EZP2 41986760 / 61 / 62, EZP4 41986763 /
  64 / 65; 6 running at most; watcher ENDFIT/wait_g.sh (read_g.py when all 18 are done; ~6 h).
- **(g) RESULT (26 Oct 12:48, all 18 COMPLETED; fits 1.4-1.6 h, FR 0.7 h, L 4.0-4.5 h; ENDFIT/read_g.py): 1 CONFIRMED,
  2 NO RESOLVED GAIN.** Independent per-track 1 mm smear (SMEAR_KEY=entry), paired with KN10N<s> (s = 1 / 2 / 4; mean):
  | | lifetime | long_diff | tran_diff | Ab | pos (um) |
  |---|---|---|---|---|---|
  | EXP @ 14000 (held) | +19.33 / +0.64 / +1.92; **+7.30** | +22.07 / +3.50 / +9.78; **+11.78** | +0.27 | -0.40 | +774 |
  | EZP @ 14000 (fitted) | +15.11 / +7.77 / +9.26; **+10.71** | +14.24 / +9.42 / +13.24; **+12.30** | -0.88 | -0.48 | +664 |
  | EXP @ 44000 (held) | +1.09 / -0.55 / -0.50; **+0.01** | -5.06 / +1.43 / +0.61; **-1.01** | +0.42 | -0.14 | +734 |
  | EZP @ 44000 (fitted) | -1.87 / +1.79 / -0.35; **-0.15** | +0.25 / +2.76 / +0.30; **+1.10** | -0.08 | -0.04 | +639 |
  - Trajectory (paired mean lifetime, EXP | EZP): +7.3 | +10.7 (14000), +3.9 | +5.2 (20000), +2.5 | +2.7 (24000),
    +1.5 | +0.9 (32000), +0.0 | -0.2 (44000). Same relaxation held or fitted, as in (f).
  - EZP endpoints (stored smear, real tracks): forward slope e1 0.54-0.55 -> 0.70-0.72, e2 0.57-0.59 -> 0.81-0.84;
    reverse 0.96-1.02 at 44000; rms a-b e2 511-525 -> 371-396 um (smear rms 941-952). Fitting the ends buys ~95 um of
    position, nothing resolved in calibration.
  - Reading: with independent per-track smears (the shared-smear caveat of (e) / (f) removed), 1 mm endpoint errors
    held fixed cost NOTHING resolvable once the calibration is converged (lifetime +0.0, long_diff -1.0; per-sample
    within +-1.1 / +-5.1). At the production end point (14000) they cost +7.3 / +11.8 on average, dominated by sample
    1 (+19.3): that is the convergence lag of (h) / (h2), not the endpoint error. Fitting the endpoints is not needed
    for the calibration; it improves position only.
  - Consequence for the recipe (Pierre's decision): with endpoints not from truth, the production end point (14000)
    is not converged; the long continuation (to 44000, +4 h per fit) is what makes the calibration endpoint-robust.
- **ENDFIT (h3), 26 Oct 22:55 (Pierre: "Go with 3"; before any result): (h) with a FAST dE/dx schedule (single factor
  vs (h)).** The last untested way to shorten the long continuation.
  - New option LARND_DEDX_LR_GLOBAL=<hold>:<end>:<frac> (optimize/fit_params.py, default off, uncommitted): multiplies
    the per-segment dE/dx Adam update by m(t) of the GLOBAL iteration (1 to <hold>, exponential to <frac> at <end>,
    constant after). Needed because LARND_DEDX_LR_FINAL_FRAC is indexed by the per-batch Adam count, which a resume
    restores (~140 at 14000): in a continuation it would start already decayed. Unset -> default path untouched
    (separate branch). CPU check (scratch test_dedx_global.py): m(t) as specified; update = m x default update to
    7e-7 relative (float32); Adam state identical.
  - Schedule: --dedx_lr 5e-2 (5x, the factor h2 used for the chain), LARND_DEDX_LR_GLOBAL=16500:18500:0.04: 5e-2 over
    the calibration's own 1e-2 window (14000-16500), then down to 2e-3 at 18500 (1e-2 at 17500), 2e-3 to 20000.
    Integrated LR x visits ~1.6 (H 0.6, L 3.0): deliberately below L's, because h2 showed that a high per-visit LR
    leaves noise the few remaining visits cannot remove. Everything else as (h).
  - ENDFIT/make_fast3_cont.py; KNOT/inner_sci_full_<X>H3.sh diffed vs <X>H: only the env line, --dedx_lr and the name.
  - **Rule (ENDFIT/read_h3.py, tested with the H runs as stand-ins; read only when all 6 are done):**
    1. EXS<s>H3 - KN10N<s>H3 @ 20000, mean lifetime within +-3 and long_diff within +-5: **FAST RECOVERS**;
       lifetime >= +5: **STILL SHORT**; otherwise **PARTIAL**.
    2. Paired lifetime shift H3 minus H per sample: < 0 on all 3 and mean <= -2: **DEDX LR HELPS**; otherwise
       **NO RESOLVED GAIN**.
    3. dE/dx relaxation c of H3: >= 0.7 on all 6: **DEDX LR-BUDGET-LIMITED**; < 0.6 on all 6 (H: 0.45-0.50):
       **DEDX VISIT-LIMITED**; otherwise **MIXED**.
    Also reported: KN10N lifetime and position vs H and L (the h2 failure mode: noise that worsens the true-end
    reference), the geometry c, the trajectory.
  - Reading planned: 1 FAST RECOVERS + 2 HELPS, with the KN10N reference not worse than H -> a 6000-iteration
    continuation (calibration and dE/dx fast) can replace the 30000-iteration one; proposal to Pierre. Otherwise the
    long continuation stays the endpoint-robust recipe.
  - **(h3) submitted 26 Oct 23:00:** EXS1H3 42080288, EXS2H3 42080289, EXS4H3 42080290, KN10N1H3 42080291, KN10N2H3
    42080292, KN10N4H3 42080293; watcher ENDFIT/wait_h3.sh.
- **(h3) RESULT (27 Oct 00:15, all 6 COMPLETED, 56-57 min; ENDFIT/read_h3.py): 1 STILL SHORT, 2 NO RESOLVED GAIN,
  3 DEDX VISIT-LIMITED.**
  | @ 20000 (s = 1 / 2 / 4; mean) | paired lifetime | paired long_diff | KN10N lifetime | KN10N pos (um) | dE/dx c |
  |---|---|---|---|---|---|
  | H (dE/dx 1e-2 constant) | +9.49 / +4.19 / +5.71; **+6.46** | **+5.29** | -2.99 / +0.42 / -2.14 | 95 / 110 / 101 | 0.45-0.50 |
  | H3 (dE/dx 5e-2 -> 2e-3) | +7.06 / +4.13 / +8.02; **+6.40** | **+5.40** | -5.76 / -2.93 / -2.19 | 97 / 110 / 100 | 0.41-0.56 |
  | L @ 44000 | **+1.89** | **+4.43** | -1.43 / -1.41 / +0.18 | 91 / 100 / 98 | 1 |
  - A 2.6x dE/dx LR budget leaves the dE/dx where it was along the long run's direction (c 0.41-0.56 vs 0.45-0.50) and
    the geometry too (0.33-0.43). No noise penalty in position (unlike h2); the true-end lifetime moves -3 points on
    samples 1 and 2 and the smeared arm with it (EXS1 +6.5 -> +1.3, EXS2 +4.6 -> +1.2), so the paired shift is
    unchanged.
  - Reading: the dE/dx is not lagging on its own. It sits near its optimum for the current geometry and calibration
    and follows them; its long-run change is driven by the geometry, which is the slow block (one low-SNR step per
    track per epoch; h2: more LR there adds noise, not progress).
  - **Closing the "why 44000" thread (h, h2, h3):** of the three LR levers only the calibration one helps (long_diff
    in 6000 iterations). The lifetime shift from 1 mm held endpoint errors needs ~300 geometry visits per track,
    i.e. ~30000 continuation iterations at 100 batches. The long continuation stays the endpoint-robust recipe
    (production recipe: Pierre's decision). LARND_DEDX_LR_GLOBAL stays in the tree, default off.
  - **CORRECTION (27 Oct, wording above about "noise"):** the production guess is the noise-marginalised probabilistic
    simulation (`compute_loss`: `probabilistic_sim and not probabilistic_sampling_sim` -> `rngkey = None`;
    `LUTProbabilisticSimulation.predict` takes no key), so each batch's loss is DETERMINISTIC in all parameters (up to
    `--non_deterministic` round-off). (i) The calibration gradient's "~99 % noise" (line ~3186, "stochastic
    simulation") is batch-to-batch heterogeneity of a fixed finite sum (100 batches), not random draws. (ii) The
    per-track geometry step is not low-SNR: a track's objective at fixed calibration is the same at every visit.
    The h2 position loss is then a conditioning effect (a larger step speeds the slow directions and overshoots or
    knocks tracks out of the stiff ones / into aliased minima), not noise. Conclusions on what was measured (only
    the calibration LR helps; ~30000 continuation iterations with 1 mm ends) are unchanged; the explanation is
    "visits needed ~ per-track condition number under a diagonal (Adam) preconditioner", and the remedies are
    curvature (per-track dense blocks) and finite-sum variance reduction (SAGA), not noise averaging. The July
    "GN fits the stochastic noise realisation" reason (spline GN verdict) also cannot hold: those fits used the
    same deterministic probabilistic guess.

## 27 Oct 2026 — ENDFIT (i): options 1 and 2 of the "why 44000" follow-up (Pierre: "try options 1 and 2 and present the results")

Written before any result (the smoke tests below are technical checks of the new code, not readouts).

- **Premise (CORRECTION above):** the loss is deterministic; the calibration gradient's spread is finite-sum batch
  heterogeneity, and the geometry's slow relaxation is conditioning (Adam's diagonal preconditioner vs dense per-track
  curvature), not gradient noise.
- **Code (live tree, uncommitted, both default off; CPU checks scratchpad test_saga_sweep.py all pass):**
  - option 2 `LARND_CALIB_SAGA=1`: the calibration optimizer gets g_i - G_i + mean_j G_j (G = last stored gradient
    of each batch; plain gradient during the fill epoch after a (re)start; table not checkpointed; history keeps raw g).
  - option 1 `LARND_GEOM_SWEEP=<t1,...>` (+ `_NEWTON`, `_EPS` 2e-3 cm, `_SEP` 4 cm, `_MAXSTEP` 0.1 cm, `_SAVE`): in
    the epoch starting at each t_k, each batch's visit replaces its Adam chain step by damped per-track Newton
    step(s) at the current calibration and dE/dx. Hessian blocks from central FD of the fit's own coefficient
    gradient, coefficient j perturbed for all tracks of a colour group at once (same-event tracks whose (x, y)
    projections on the same anode side come within 4 cm get different colours; KN10N1FR: 51 / 48 / 1 batches need
    1 / 2 / 3 colours, ~23 directions per batch, max 38). Positive-curvature solve, damping ladder kappa x
    median|eig| (0 ... 3), lowest batch objective wins, no step if none is below the start; Adam momentum reset.
  - generator ENDFIT/make_opt_cont.py (= make_fast_cont.py + flags; with no flag it reproduces the (h) scripts byte
    for byte, checked on KN10N1 and EXS1).
- **Smoke (technical):** SMGA / SMGB = EXS1FR, sweep at 14000 on the first 10 batches, 2 Newton steps, FD eps 2e-3 /
  5e-4 cm (FD validity: model vs actual dJ, block asymmetry, eps stability; cost per batch); SMSA = KN10N1FR +
  SAGA, 300 iterations (fill, no NaN, overhead). Jobs 42101363-65. The number of Newton steps per sweep for the main
  arm is set from the smoke: 2 if the second step gains >= 10 % of the first, else 1.
- **Main arms (12 jobs), each = the (h) fast continuation <X>FR @ 14000 -> 20000 plus one flag, X = EXS<s> (1 mm
  legacy smear, ends held) and KN10N<s> (true ends), s = 1, 2, 4:**
  - S (option 2): LARND_CALIB_SAGA=1.
  - G (option 1): LARND_GEOM_SWEEP=14000,16500,18500 (at FR's calibration / after the calibration's 1e-2 phase /
    after its decay), LARND_GEOM_SWEEP_SAVE=1 (checkpoint at 14100 = the geometry after one sweep at FR's
    calibration: the fixed-calibration diagnostic).
- **Readout ENDFIT/read_opt.py (tested on the (h3) runs as a stand-in arm). Verdicts, per arm:**
  - paired EXS - KN10N @ 20000: RECOVERS if |lifetime| <= 3 and |long_diff| <= 5 (the (h) thresholds); HELPS if
    lifetime is below H on all three samples and by >= 2 on average; else NO RESOLVED GAIN (H: +6.46 / +5.29; L @
    44000: +1.89 / +4.43).
  - G: RELAXES GEOMETRY if geometry c >= 0.7 on all 6 runs and position <= 1.1 x H on all 6; PARTIAL if mean c >= H's
    mean + 0.2 with position OK; POSITION PENALTY if any position > 1.1 x H; else NO.
  - S: REMOVES JITTER if the sd of the lifetime and long_diff iterates over 15500-16500 (LR held at 1e-2) is <= 0.5 x
    H's on >= 5 of 6 runs; CLOSER TO CONVERGED if mean |X - L @ 44000| (lifetime) <= 0.7 x H's.
  - also reported: sweep cost (hours), objective drop per sweep, c right after the first sweep.
- **Expectation (stated before the result):** S removes jitter but leaves the paired lifetime where H has it (H showed
  lifetime waits for the nuisances); G raises the geometry c and moves the paired lifetime toward L if the per-track
  problem is conditioning-limited. A combined arm (GS) only if G shows a gain.
- **Smoke RESULT (27 Oct 04:40, technical):** all 3 COMPLETED (12 min each).
  - sweep (EXS1FR, first 10 batches, 2 Newton steps): works; 10-45 s per Newton step per batch (12-38 directions).
    The geometry at FR is NOT at its conditional optimum: summed dJ over the 10 batches -998 (eps 2e-3) / -1485 (eps
    5e-4); the 2nd Newton step adds 42 % / 65 % of the 1st -> by the rule above, **2 Newton steps per sweep**.
    Steps up to 350-500 um on single batches (rms 2-90 um). FD block asymmetry 0.1-8 (max over a batch's tracks,
    relative to its largest diagonal), mostly lower at 5e-4; actual dJ often exceeds the quadratic model (non-
    quadratic objective), the ladder protects. **FD eps for the main arm: 5e-4 cm** (larger gain, lower asymmetry).
  - SAGA (KN10N1FR, 300 iterations): table complete at 14099, no NaN, no overhead (0.453 vs 0.456 s/iteration). BUT
    the lifetime swings MORE than in H over 14100-14300 (+-1.5 % in 50 iterations). Offline reconstruction of the
    SAGA direction from the logged raw batch gradients (d_t = g_t - g_{t-100} + mean(g_{t-100..t-1})):
    sd(g_i - g_i,previous visit) / sd(g) = 0.77-0.93 in H's 1e-2 phase (14200-16500), ~1 in the smoke, 0.08-0.36 in
    H's annealed tail (18000-20000) and 0.23-0.58 in the long run at 1e-3 (30000-44000). => **SAGA cannot reduce the
    variance at calibration LR 1e-2**: at that LR the calibration moves so much between two visits of a batch that
    the stored gradient is stale (premise of the proposal wrong for the fast schedule). At 1e-3 it cuts the sd 2-4x
    (lifetime 0.27, Ab 0.32, long_diff 0.44). Under Adam (step ~ m/sqrt(v), scale-free) the expected gain at a FIXED LR
    is a faster coherent drift (SNR 0.11 -> 0.43 for lifetime) and a smaller stationary jitter (~ sqrt of the sd
    ratio, 0.5-0.7), not a noise-free step.
- **AMENDMENT before any main result:** S kept as registered (expected: no gain, by the reconstruction above) and a
  third arm added, **SL = LARND_CALIB_SAGA=1 on the LONG run's schedule** (make_opt_cont.py --long: calibration 1e-3
  held, chain 2e-3 decaying 0.99996/it, 14000 -> 20000), compared with the long run's own 20000 window (L20: same
  schedule and iterations, no SAGA). SL rule: REMOVES JITTER if the sd of the lifetime and long_diff iterates over
  19000-20000 is <= 0.5 x L20's on >= 5 of 6 runs; CLOSER TO CONVERGED if mean |SL - L @ 44000| (lifetime) <= 0.7 x
  L20's; paired HELPS if the paired lifetime is below L20's on all three samples and by >= 2 on average. G uses NEWTON
  2, EPS 5e-4. All 18 scripts differ from their references (H, resp. L) by the intended flags only (checked).
- **Health check, first sweep of G (27 Oct, technical, no calibration read):** all 6 G runs alive at 1:29 with their
  14100 checkpoints; per run 194-200 Newton steps accepted (of 200; rejected = no κ rung lowers the batch objective),
  summed dJ -11198 / -12859 / -15218 (EXS 4/1/2) and -5555 / -7012 / -11511 (KN10N 1/2/4); median max|step| 78 µm
  (EXS1) and 40 µm (KN10N1), p90 315 / 171 µm; ~4470 s per sweep (22 s per batch, median).
- **(i) RESULT (27 Oct, all 18 COMPLETED; G 4.5-4.8 h, S / SL 0.93-0.95 h; ENDFIT/read_opt.py, RO_ARMS=S,G,SL):
  NEITHER OPTION CLOSES THE GAP; the per-track-conditioning diagnosis is NOT SUPPORTED.**
  - S (SAGA, LR 1e-2): NO RESOLVED GAIN / POSITION PENALTY / jitter NO / NOT CLOSER. Harmful, as the staleness
    reconstruction predicted: jitter 2.1-5.9x H, paired lifetime +19.75 (H +6.46, L +1.89), |S - L| lifetime 17.8 vs
    2.9, position up to 1.15x H. A stale table at high LR is lagged gradient feedback.
  - SL (SAGA, LR 1e-3, vs L20): NO RESOLVED GAIN / jitter NO / NOT CLOSER. The lifetime jitter falls to 0.49-0.92x L20
    on 6/6 (predicted ~0.5-0.7x), long_diff jitter mixed (0.49-3.6x); the drift is not faster: paired lifetime +13.06
    vs L20 +10.71, |SL - L| lifetime 6.0 vs 4.5.
  - G (curvature sweeps): NO RESOLVED GAIN / geometry rule NO (c mean 0.57 vs H 0.38: +0.19, just under the +0.2
    PARTIAL bar) / jitter NO / NOT CLOSER. Mechanically sound: 170-199 of 194-200 Newton steps per sweep lower the
    batch objective, summed dJ -2.6k to -39k per sweep (largest at 16500, after the 1e-2 phase moved the calibration),
    positions 0.92-0.99x H. But the calibration does not follow: paired lifetime +7.80 (-1.15 / +3.65 / +1.51 vs H),
    |G - L| lifetime 2.95 vs H 2.89; paired long_diff WORSE on all three samples (+19.0 / +8.5 / +11.5 vs H
    +11.5 / +0.4 / +4.0): G raises the held-end (EXS) long_diff and lowers the true-end (KN10N) one. 5x wall-clock.
  - Where the nuisances are at 20000 (scratchpad slow_mode.py; r = |Y - L| / |FR - L| in coefficient space):
    geometry r H 0.67-0.74, G 0.55-0.69, SL 0.57-0.65; log dE/dx r H 0.78-0.93, G 0.73-0.83, SL 0.72-0.83. One
    sweep at FR's calibration covers only c 0.13-0.19 of FR -> L. The coherent drift-axis geometry shift is small and
    inconsistent (-22..+24 um), the mean log dE/dx moves < 0.1%: no single collective mode stands out.
  - Reading: the 44000 iterations are a slow JOINT relaxation of calibration + dE/dx + geometry. Solving the geometry
    block per track at the current calibration (G) does not move along it, and the gradient variance (S, SL) is not
    what limits it. This corrects the 27 Oct "per-track conditioning" diagnosis (that block was tested and is not the
    bottleneck). Options left: wall-clock only (multi-GPU, option 3), or a joint calibration + nuisance step (the
    earlier joint calib + dE/dx step converged to its own minimum, lifetime -3.2 +- 0.7 % on KN10C). No new arm launched.

## 28 Oct 2026 — ENDFIT (j): is the fast continuation enough for REALISTIC endpoint errors? (Pierre: "Do we have solutions left in order to not have to run for so many steps?"; before any result)
- Why: every speed test (h, h2, h3, i) ran on EXS, the legacy smear (one smear set shared by samples 1/2/4, coherent
  drift offset). With the independent per-track smear (EXP, (g)) the 14000 paired lifetime is +7.3 (+19.3 / +0.6 /
  +1.9) instead of +16.7, and the LONG schedule is already at +3.9 at 20000. The fast schedule cut EXS's 20000 gap by
  ~4 points vs L20, so it may be enough on EXP. Also: the calibration-only extrapolation of L (exponential fit to
  the 500-iteration window means, scratchpad extrap.py) is unstable before ~28000 (asymptotes off by > 5 points
  or diverging on 8 of 12 run x parameter cases at T_stop 24000): no early-stopping shortcut from the trajectory.
- Arms (6 jobs, ENDFIT/make_opt_cont.py; no new code): EXP<s>H = (h) on EXP<s>FR @ 14000 -> 20000 (6000 its);
  EXP<s>HH = (h) with the calibration LR 1e-2 held to 20000 (new option --hold 20000, i.e. transition_begin 19500),
  then x0.805/epoch, -> 22000 (8000 its). s = 1, 2, 4. Scripts differ from EXS1H only by sample index, names,
  resume path and LARND_ENDPOINT_SMEAR_KEY=entry (checked); the generator without --hold reproduces EXS1H byte for byte.
- Readout ENDFIT/read_j.py. Metric: per run |X - EXP<s>L @ 44000| (lifetime, long_diff; last 1000 iterations, %),
  mean over s. Floor = the true-end fast runs' own distance, KN10N<s>H @ 20000 vs KN10N<s>L @ 44000 (lifetime
  1.90, long_diff 1.92). Rule per arm: SUFFICIENT if mean |X - L| <= 2.5 (lifetime) and <= 3 (long_diff); NOT
  SUFFICIENT if >= 4 or >= 6; else MARGINAL. Also reported: paired EXP H - KN10N H at 20000 (vs L's +0.01 / -1.01).
- Expected: H MARGINAL to SUFFICIENT (sample 1 carries the lag); HH closer than H if the remaining lag is the early
  anneal. If SUFFICIENT: realistic 1 mm endpoint errors need ~20000-22000 iterations, not 44000 (recipe change:
  Pierre's decision).
- **28 Oct, Pierre: "Please get ready to try 1 and 2. 3. Is not very relevant ... probe again the memory and speed for
  different batch sizes now that we optimized the code. Maybe that making bigger batches might help".** Built in
  optimize/fit_params.py (default off, uncommitted):
  - option 1, LARND_EPOCH_ACCEL_BETA (+ _START/_STOP/_BLOCKS/_CAP_*/_RESTART): epoch-level Nesterov extrapolation
    y_e = x_e + beta (x_e - x_{e-1}) of the geometry and dE/dx (optionally the calibration, along the difference of
    epoch means), clipped per element, adaptive restart on a rising epoch-mean loss, Adam states kept. CPU test
    (scratchpad test_epoch_accel.py): a steady drift is amplified by the geometric sum -> 1/(1-beta), caps and restart
    behave, Adam states untouched.
  - option 2, LARND_JOINT_NEWTON=<its> (+ _EPS_*/_CAP_*/_SEP/_SAVE): joint damped Newton step over calibration + every
    batch's dE/dx + geometry through the Schur complement of the block-arrow Hessian; 2 held epochs per step (A:
    central-FD blocks per batch -- 5 calibration columns, dE/dx band(2) + analytic mean-constraint rank-1 from 5
    probes as _dedx_gn_step, per-track geometry blocks + own-segment dE/dx cross blocks colour-compressed as
    LARND_GEOM_SWEEP; B: forward evaluation of 6 damped candidates; apply the best that lowers the total objective,
    zero all Adam first moments). CPU test (scratchpad test_joint_newton.py): on a synthetic block-arrow quadratic with
    exactly the assumed structure, one step at kappa 0 lands on the exact minimiser (dJ = model = exact to 1e-9)
    through the fitter's own assemble / solve / evaluate / apply code; padding tracks untouched; calibration cap holds.
  - GPU smokes (EXS1FR @ 14000, HH schedule): SMEA (beta 0.8, 500 its) 42190570, SMJN (one step at 14000, 300 its)
    42190571; ENDFIT/k_smoke_jobs.txt. make_opt_cont.py gained --hold N (calibration LR 1e-2 held to N).
  - Batch-size probe (BATCH/make_bs.py, read_bs.py): production 1e4 fit (KN10N1 config, live tree) at 400 / 800 / 1200
    / 1600 cm per batch, NB x LEN = 40000 cm, 5 epochs, XLA_PYTHON_CLIENT_MEM_FRACTION=0.95; per epoch time, steady
    s/it and s/epoch, throughput, GPU peak (training_history memory), MaxRSS. Jobs BATCH/bs_jobs.txt (42189607-11).
- **(j) RESULT (28 Oct, all 6 COMPLETED; H 0.98-1.03 h, HH 1.25-1.28 h; ENDFIT/read_j.py): EXP H SUFFICIENT, EXP HH
  MARGINAL.** Independent per-track 1 mm smear, ends held (s = 1 / 2 / 4):
  | | mean abs(X - own L @ 44000), lifetime | long_diff | paired EXP - KN10N lifetime / long_diff |
  |---|---|---|---|
  | floor: KN10N H @ 20000 vs KN10N L | 1.90 | 1.92 | |
  | EXP FR @ 14000 (production end point) | 6.71 (14.2 / 2.5 / 3.4) | 11.17 | |
  | EXP L20 (long run @ 20000) | 2.96 | 5.11 | |
  | **EXP H @ 20000 (fast, 6000 its)** | **2.17** (2.3 / 3.0 / 1.2) | **2.74** | -1.47 / -3.51 (L: +0.01 / -1.01) |
  | EXP HH @ 22000 (1e-2 held to 20000) | 1.95 | 5.56 (9.5 / 1.0 / 6.2) | |
  - Reading: with realistic (independent) 1 mm endpoint errors the fast continuation to 20000 reaches the long run's
    44000 state to within the true-end runs' own fast-vs-long distance: 24000 fewer iterations (~1 h instead of ~4 h
    after FR). Holding the calibration LR longer does not help (long_diff wanders further on 2 of 3 samples).
    Position at 20000 is 25-35 um (EXP, ~3 %) and 4-10 um (KN10N) above the 44000 value. The 44000 requirement found
    in (e)-(i) is specific to the legacy SHARED smear (EXS), where H stays at |X - L| 3.9 / 3.8. Recipe change
    (FR @ 14000 -> + (h) continuation to 20000): Pierre's decision.
- **Batch-size probe RESULT (28 Oct; BATCH/read_bs.py; production 1e4 fit, KN10N1 config, live tree, 40000 cm):**
  | cm / batch | batches | epoch 1 (min) | steady s/it | s / epoch | cm/s | GPU peak | MaxRSS |
  |---|---|---|---|---|---|---|---|
  | 400 | 100 | 8.1 | 0.447 | 44.7 | 895 | 10.3 GiB | 8.1 GB |
  | 800 | 50 | 6.3 | 0.810 | 40.5 | 988 | 15.2 GiB | 7.1 GB |
  | 1200 | 33 | 6.5 | 1.203 | 39.7 | 997 | 22.4 GiB | 7.1 GB |
  | 1600 | 25 | -- | -- | -- | -- | OOM (a 13.5 GiB request, BFC fragmentation, MEM_FRACTION 0.95) | |
  - Reading: after the speed package the fit is GPU-bound -- time per iteration grows ~linearly with cm per batch, so a
    pass over the data costs ~40-45 s whatever the batch size (800 cm: 0.91x, 1200 cm: 0.89x per epoch). Bigger
    batches buy <= 11 % per epoch and a shorter first-epoch compile (8.1 -> 6.3 min), at 1.5-2.2x GPU memory; 1200 cm
    is the ceiling on a 40 GB A100. Since the slow relaxation counts epochs (nuisance visits), bigger batches do not
    shorten it materially; they also give 2-3x fewer calibration steps per epoch (the calibration LR would need
    retuning). Not worth a recipe change on speed grounds.
- **Option 1 / 2 GPU smokes (28 Oct; both COMPLETED, no error; EXS1FR @ 14000):**
  - SMEA (epoch acceleration, beta 0.8 from 14000, 500 its, 14.4 min): pushes of geometry rms 9-11 um (max 27 um,
    none capped) and dE/dx rms 1.2-2.1e-3; adaptive restart fired once (it14200). Compared with EXS1H iteration by
    iteration (same schedule up to 14500): epoch-mean loss 10474 / 10573 after the pushes at 14300 / 14400 vs 10449 /
    10431 without (+0.2 % / +1.4 %); lifetime at 14500 +8.3 vs +10.4 % (vs truth), long_diff +30.8 vs +28.8 %. The
    pushes during the re-heated first epochs extrapolate Adam's epoch-to-epoch oscillation as much as the drift.
  - SMJN (one joint Newton step at 14000, 300 its, 71.6 min): assembly 3696 s for 100 batches (32.6 s/batch, ~45-100
    gradients each; one step = 2 held epochs + ~62 min, i.e. ~80 Adam epochs of wall-clock). Candidate ladder: the
    undamped quadratic model overstates the gain 5x at kappa 0 (model -4824, actual -959, 66 of 100 batches capped);
    the actual gain grows monotonically with damping up to the largest kappa 3 (dJ -5849, model -5367), which was
    applied: calibration Ab -0.13 %, lifetime -0.14 %, geometry rms 45 um (max 642 um), dE/dx rms 2.7e-3; the next
    epoch's mean loss 10372 vs 10433-10528 in EXS1H's first epochs. The kappa 0 calibration direction was Ab +1.95 %,
    lifetime -5 % (capped), long_diff +0.03 % (the long run moves long_diff -32 points from here).
  - Change after the smoke (fit_params.py, default unchanged): LARND_JOINT_NEWTON_LADDER='kc:kn,...' = candidate
    pairs of (damping on the 5x5 Schur complement, damping on the nuisance blocks); default = the old diagonal. The
    nuisance eigendecompositions are computed once per batch. CPU test re-passed (default and mixed ladders: exact
    minimiser at kc = kn = 0; cap holds); test_epoch_accel re-passed.
- **ENDFIT (k) -- options 1 and 2 on the legacy shared smear (pre-registered 28 Oct, before any result).** Question:
  does epoch acceleration or a joint Newton step close the EXS fast-vs-long gap (EXS H @ 20000: mean |X - EXS L @
  44000| lifetime 3.88, long_diff 3.82; floor 1.90 / 1.92)?
  - Arms (6 jobs, ENDFIT/make_opt_cont.py, (h) schedule as EXS<s>H, 14000 -> 20000; s = 1, 2, 4):
    EXS<s>EA = LARND_EPOCH_ACCEL_BETA=0.8, _START=14600, _STOP=19000 (geometry + dE/dx; start after the first
    re-heated epochs, per the smoke; stop 10 epochs before the end).
    EXS<s>JN = LARND_JOINT_NEWTON=14000,14400,14800 with LARND_JOINT_NEWTON_LADDER =
    0.3:0.3,1:1,3:3,10:10,30:30,0:1,0:3,0:10,0.1:3,0.3:10 (diagonal extended upward since kappa 3 won at the edge;
    mixed pairs = trusted nuisance damping + light calibration damping); 600 of the 6000 iterations are held.
  - Readout ENDFIT/read_k.py (read_j.py with RJ_BASE=EXS, arms H, EA, JN at 20000) + the push / step logs. Rule per
    arm on the mean over s of |X - EXS<s>L @ 44000| (lifetime, long_diff): CLOSES THE GAP if <= 2.5 and <= 3; HELPS if
    not that but both <= 0.8 x H's (3.10 / 3.06); HARMFUL if either >= 1.25 x H's (4.85 / 4.78); else NO GAIN.
    Practical value also needs the wall-clock to beat the long run (EXS L: ~4 h after FR); JN is expected at ~4.3 h.
  - Expected (from the smokes): EA NO GAIN to HELPS (pushes cost loss, restart fires often); JN NO GAIN unless the
    mixed candidates are accepted with a sizeable calibration move.
- **(k) RESULT (28 Oct; all 6 COMPLETED; EA 0.93-0.99 h, JN 3.63-3.71 h; ENDFIT/read_k.py, scratchpad k_traj.py):
  both options HELP by the pre-registered rule; neither formally closes the gap; only EA is practical.**
  | arm (EXS, @ 20000) | mean abs(X - own L @ 44000), lifetime (s = 1 / 2 / 4) | long_diff | position - L | step time |
  |---|---|---|---|---|
  | floor (KN10N H vs L) | 1.90 | 1.92 | | |
  | H (baseline fast) | 3.88 (3.2 / 4.3 / 4.2) | 3.82 (4.9 / 1.6 / 5.0) | +19.8 um | 0.90 h |
  | **EA (epoch accel)** | **2.01** (0.9 / 0.8 / 4.4) | **3.01** (5.9 / 0.5 / 2.7) | **+12.4 um** | **0.90 h** |
  | JN (joint Newton) | 2.52 (3.0 / 1.7 / 2.8) | 2.98 (2.1 / 4.2 / 2.7) | +19.9 um | 3.62 h |
  | long run L to 44000 | 0 | 0 | 0 | 3.92 h |
  - Trajectory (mean |X - L|, lifetime / long_diff): H 5.59/8.04 @16000, 4.79/4.20 @18000, 3.88/3.82 @20000; EA
    3.77/7.63, 2.33/3.78 @18000, 1.98/3.05 @19000, 2.01/3.01 @20000. EA at 18000 is already at H's 20000 level.
    Positions: EA ends closer to L on 3/3 samples (-2.5 / -9.3 / -10.6 um vs H; L is -15.7 / -27.6 / -16.2 um), its
    last-epoch loss is lower on 3/3 (-0.2 to -0.5 %). Lifetime gain on s = 1, 2 only; long_diff better on s = 2, 4,
    worse on s = 1 (+0.9). EA misses CLOSES THE GAP by 0.01 on long_diff (3.01 vs 3.00). Pushes: 29-32 per run, 13-16
    adaptive restarts over 44 epochs, median geometry push rms 4.4-4.9 um. Same step time as H: the gain is free.
  - JN: three steps per run, 53-68 min each. Steps 2 and 3 chose the most damped candidate (kappa 30, at the edge of
    the ladder) with calibration moves <= 0.35 %; step 1 took a mixed candidate on s = 1 (lifetime -2.13 %, winning by 2
    in 5852) and s = 2 (long_diff -5 %, capped), i.e. moves along nearly flat directions of the objective, and both
    overshot L (s = 1 lifetime +0.34 vs L +3.34; s = 2 long_diff +3.79 vs L +7.98). The undamped quadratic model keeps
    overstating the gain (up to 2x at the chosen candidate). Its 3.6 h cost is that of the long run itself: not practical.
  - Reading: epoch-level extrapolation of the nuisances (geometry + dE/dx) speeds up the slow joint relaxation at no
    cost; the joint Newton step, as built, is a nuisance polish whose calibration moves follow flat directions. Follow-up
    (l): does EA hold on the realistic smear (EXP) and true ends (KN10N), and how early does it reach H's 20000 level?
- **ENDFIT (l) -- epoch acceleration on realistic and true endpoints (pre-registered 28 Oct, before any result).**
  Question: is EA safe where the fast continuation already suffices, and does it get there earlier?
  - Arms (6 jobs): EXP<s>EA (independent per-track 1 mm smear) and KN10N<s>EA (true ends, production KN10N), s = 1, 2,
    4; the (h) continuation of <BASE><s>FR @ 14000 -> 20000 + the (k) EA settings (beta 0.8, 14600-19000, geometry +
    dE/dx); scripts from make_opt_cont.py, compared with <BASE><s>H.
  - Readout ENDFIT/read_l.py: mean over s of |X - own <BASE><s>L @ 44000| (lifetime, long_diff) for H and EA at
    17000-20000, position - L, last-epoch loss. Rules: SAFE if EA @ 20000 <= 1.25 x H @ 20000 in both (EXP H 2.17 /
    2.74 -> 2.71 / 3.43; KN10N H 1.90 / 1.92 -> 2.38 / 2.40), else HARMFUL. EXP EARLIER if EA meets the (j) SUFFICIENT
    rule (<= 2.5 / <= 3) at 18000. Recipe candidate if SAFE on both bases and positions closer to L on >= 5 of 6
    samples (any production change: Pierre's decision).
  - Expected: SAFE on both, EXP sufficient by ~18000, positions 5-10 um closer to L.
- **(l) RESULT (28 Oct; all 6 COMPLETED, 0.97-1.00 h; ENDFIT/read_l.py): EA is HARMFUL on both bases (long_diff) and
  NOT a recipe candidate (positions closer to L on 4 of 6).**
  | base, @ 20000 | H lifetime / long_diff (mean abs(X - own L)) | EA lifetime / long_diff | H / EA position - L |
  |---|---|---|---|
  | EXP (independent 1 mm smear) | 2.17 / 2.74 | 1.52 / **3.94** (long_diff 8.2 / 3.3 / 0.3 vs H 3.2 / 1.2 / 3.8) | +27.9 / +19.9 um |
  | KN10N (true ends) | 1.90 / 1.92 | 2.03 / **2.72** (3.55 vs 1.39 on s = 4) | +5.6 / +7.3 um |
  - EXP EARLIER: NO (EA @ 18000 1.43 / 4.18). For reference, H itself is already within the SUFFICIENT rule at 17000
    (1.78 / 2.20; 18000 2.34 / 3.22): on the realistic smear the fast continuation sits at the noise floor from ~17000.
  - Reading, over (k) and (l) (9 samples): the epoch pushes speed up lifetime (better on 6 of 9) and, where the
    nuisance drift is a large coherent mode (shared smear), positions; but they make long_diff wander (worse on 5 of 9,
    by up to +5 points) and, near convergence (true ends), cost position and loss. With realistic endpoints there is
    no gap left for them to close. Thread status: options 1 and 2 tested; neither is adopted. Answer to "fewer steps":
    for realistic 1 mm endpoint errors, FR @ 14000 + the plain (h) continuation to 20000 (~1 h) suffices; 44000 only
    for the legacy shared smear.
- **PRODUCTION CHANGE (28 Oct, Pierre: "Let's make this part of standard production now"): the production end point is
  now <NAME>H @ 20000.** KNOT/submit_prod.sh chains a third job after the fit and the freeze-release continuation:
  run_knot.sbatch prodfast = CONT/make_prod_fast.py, a frozen copy of the (h) fast continuation (FR @ 14000 -> 20000,
  calibration LR 1e-2 held to 16500 then x0.805 per epoch, chain LR re-heat 2e-3 decaying 0.9998 per iteration, freeze
  released; ~0.9 h); it regenerates KNOT/inner_sci_full_KN10N1H.sh byte for byte. submit_prod_chains.sh chains samples
  on the H job. Finished samples are brought forward with KNOT/backfill_fast.sh: KN10N1, 2, 4 H existed; KN10N3,
  5-16 H launched 28 Oct 08:43 (jobs 42247045-58, 6 chains). Readouts of the production ensemble move to H @ 20000.
- **ENDFIT (m) -- faster convergence, ideas 1 and 2 (pre-registered 28 Oct, before any result; Pierre: "Please try those
  as you proposed").** Bases: EXP (independent 1 mm smear) and KN10N (true ends), s = 1, 2, 4; reference each run's own
  <BASE><s>L @ 44000; metric as (j): mean over s of |X - L| (lifetime, long_diff, last 1000 its) and position - L.
  Production H @ 20000: EXP 2.17 / 2.74, +27.9 um; KN10N 1.90 / 1.92, +5.6 um.
  - Idea 1, one schedule instead of three stages (stage 1 anneals the calibration LR from 0.1 by x0.91 per epoch, below
    1e-3 by ~5000 and ~1e-4 by 7500; FR re-heats to 1e-3, H to 1e-2; the last stage does most of the converging).
    Builder ENDFIT/make_sched.py = the production fast continuation from any start S (calibration 1e-2 held H its, then
    x0.805 per epoch; chain re-heat 2e-3 x0.9998 per it from S; freeze released); at S = 14000 it reproduces
    CONT/make_prod_fast.py byte for byte.
    SK (skip FR): <BASE><s> fit @ 10000 -> make_sched 6000 its (hold 2500) -> end 16000 (production: 20000).
    FL (no early anneal): stage 1 as production but the calibration LR floored at 1e-2 (lr_kw end_value 0.01; same
    freeze, 10000 its), then FLH = make_sched 4000 its (hold 1500) -> end 14000.
    Verdict per arm and base: EQUIVALENT if lifetime <= max(2.5, 1.25 x H) and long_diff <= max(3, 1.25 x H) (EXP 2.71
    / 3.43, KN10N 2.5 / 3) and position - L <= H's + 5 um (EXP 32.9, KN10N 10.6 um); else WORSE. Recipe candidate if
    EQUIVALENT on both bases (production would end at 16000 or 14000 instead of 20000: -20 % / -30 % iterations).
  - Idea 2, low-mode push: LM = <BASE><s>FR @ 14000 + the production fast continuation + LARND_EPOCH_ACCEL_BETA=0.8
    (14600-19000) on the geometry only, sine modes 1-2 of the drift-carrying half of drift-frame tracks
    (LARND_EPOCH_ACCEL_BLOCKS=geom, _GEOM_MODES=2, _GEOM_HALVES=e2; new, default off; CPU test: only those
    coefficients move). Bases EXS, EXP, KN10N. Rule as (l): SAFE if <= 1.25 x H in both (EXP 2.71 / 3.43, KN10N 2.38 /
    2.40); ADOPTABLE if SAFE on EXP and KN10N and positions closer to L than H on >= 5 of 6. EXS diagnostic: HELPS if
    <= 0.8 x EXS H (3.10 / 3.06), as (k).
  - Expected: SK EQUIVALENT (FR at 1e-3 contributes little); FL uncertain (an unannealed calibration with frozen drift
    may oscillate); LM SAFE with a smaller lifetime gain than EA.
  - 27 jobs (SK 6, FL 6 + FLH 6, LM 9) appended to the 6 backfill chains (<= 6 running); readout ENDFIT/read_m.py.
- **ENDFIT (m) idea 1 RESULT (read 28 Oct after all 18 jobs COMPLETED; ENDFIT/read_m.py sched): NOT A RECIPE
  CANDIDATE -- both arms WORSE on both bases; the three stages each do a job.**

  | base, arm @ end | lifetime / long_diff | position - L | rule |
  |---|---|---|---|
  | EXP H @ 20000 (production) | 2.17 / 2.74 | +27.9 um | -- |
  | EXP SK @ 16000 | 2.88 / **9.54** (s: 14.2 / 9.1 / 5.4) | **+47.2 um** | WORSE (2.71 / 3.43, 32.9) |
  | EXP FLH @ 14000 | **5.63 / 15.83** | **+70.8 um** | WORSE |
  | KN10N H @ 20000 (production) | 1.90 / 1.92 | +5.6 um | -- |
  | KN10N SK @ 16000 | 1.40 / 1.47 | **+14.6 um** | WORSE on position only (10.6) |
  | KN10N FLH @ 14000 | 1.41 / **4.35** | **+38.1 um** | WORSE |

  - SK (skip FR): the expectation (FR contributes little) was wrong. With realistic endpoint errors (EXP) the
    freeze release at the high calibration LR locks long_diff away from L: flat at 9.9 / 9.9 / 9.5 at 12000 / 14000 /
    16000, against 2.7 for production, and positions +47 vs +28 um. On true ends (KN10N) SK converges the calibration
    faster than production (1.40 / 1.47 @ 16000, already 1.91 / 2.05 @ 14000) but leaves positions +9 um further from
    L. Reading: FR releases the drift freeze while the calibration is held near still (1e-3), so the geometry absorbs
    the endpoint errors before the calibration moves fast; doing both at once lets long_diff take up the geometry.
  - FL (no early anneal): worse on every quantity; with the drift frozen, a calibration LR of 1e-2 throughout stage 1
    leaves the calibration and the geometry far apart (EXP s = 4 lifetime 12.4, long_diff up to 26) and the 4000-it
    finish does not recover it. The stage-1 anneal is needed.
  - Answer for "fewer stages": no. Production (fit 10000 -> FR 4000 -> H 6000) stays. Side check: the 13 backfilled
    ensemble samples KN10N3, 5-16 H all reached 20000 (rc = 0, 28 Oct); with KN10N1, 2, 4 the ensemble is 16/16 at the
    production end point.
- **ENDFIT (m) idea 2 RESULT (read 28 Oct after all 9 LM jobs COMPLETED; ENDFIT/read_m.py lm): HARMFUL on EXP and
  KN10N, NO GAIN on EXS -> not adopted.** Restricting the push to sine modes 1-2 of the drift-carrying half keeps the
  lifetime gain but not the long_diff: the long_diff wander of the full push (k, l) is reproduced by these two modes alone.

  | base @ 20000 | H lifetime / long_diff | LM lifetime / long_diff | H / LM position - L | rule |
  |---|---|---|---|---|
  | EXS (shared smear) | 3.88 / 3.82 | **1.68** / 4.95 | +19.8 / +17.8 um | NO GAIN (needs <= 3.10 / 3.06) |
  | EXP (independent smear) | 2.17 / 2.74 | 3.52 / **7.74** (s: 10.3 / 1.8 / 11.1) | +27.9 / +24.2 um (closer 3/3) | HARMFUL (2.71 / 3.43) |
  | KN10N (true ends) | 1.90 / 1.92 | **0.93** / 2.90 (s = 1: 6.09) | +5.6 / +10.2 um (closer 0/3) | HARMFUL (2.38 / 2.40) |

  - 29-36 pushes and 9-16 restarts per run; median rms of the pushed coefficients 6-10 um (small).
  - Reading, over (k), (l), (m): every way of moving the large-scale drift-direction track shape faster (the full
    epoch push, the low-mode push, releasing the drift freeze at the high calibration LR) speeds lifetime but sends
    long_diff away from L, on 2 of 3 bases by more than the fit noise. The idea-2 premise (long_diff mostly sensitive
    to small-scale wiggles) is wrong: long_diff couples to the lowest drift-direction modes. On realistic endpoints the
    production schedule reaches the 44000 calibration within noise from ~17000; the last 3000 iterations mainly
    converge positions (EXP +40.6 -> +27.9 um, KN10N +25.3 -> +5.6 um from 17000 to 20000). Production unchanged.
- **ENDFIT (n) -- faster convergence, ideas 5 and 3 (pre-registered 28 Oct, before any result; Pierre: "Try 1. and
  2." = the shorter Adam window for the track parameters and coarse-to-fine track shape).** Both are new flags,
  default off (CPU test: off = the default path; on = the intended masks / update on the flat jitted and the per-track
  paths). Each arm is a full production pipeline (KNOT/submit_prod.sh: fit 10000 -> FR 4000 -> H 6000; the flag is
  set in the stage-1 inner script, and the continuations inherit it through the checkpoint's LARND_* env).
  - Idea 5, B2: LARND_CHAIN_ADAM_B2=0.95 (optax default 0.999). Code fact found while building it: the per-track
    geometry Adam is not checkpointed, so it restarts at every stage (~100 / 40 / 60 visits per track in fit / FR /
    H); with 0.999 the step scale is the stage's mean squared gradient and shrinks behind the gradient as the track
    converges. 0.95 is a ~20-visit window (beta1 0.9 < sqrt(b2)).
  - Idea 3, CF: LARND_CHAIN_COARSE_ITER=5000: before 5000 only the lowest ceil(n/2) (>= 2) sine modes per half are
    fitted (the 20 cm-knot basis), the upper ones held at 0 (the hit seed's upper modes dropped); free after. 5000 =
    the end of stage 1's high-calibration-LR half (1.4e-3 at 5000).
  - Bases EXP (independent 1 mm smear) and KN10N (true ends), s = 1, 2, 4: 12 pipelines, end sci_full_<BASE><s><ARM>H.
    Metric as (m): mean over s of |X - own <BASE><s>L @ 44000| (lifetime, long_diff) and position - L, at 16000-20000.
  - Rule, per arm and base, the (m) idea-1 test against production H @ 20000: EQUIVALENT at iteration t if lifetime
    <= max(2.5, 1.25 x H), long_diff <= max(3, 1.25 x H) (EXP 2.71 / 3.43, KN10N 2.5 / 3) and position - L <= H's +
    5 um (EXP 32.9, KN10N 10.6 um). SAFE = EQUIVALENT at 20000. FASTER = EQUIVALENT at 17000 or 18000 on both bases
    -> RECIPE CANDIDATE (production would end there: -15 % / -10 % iterations). BETTER = SAFE on both and positions
    closer to L than H at 20000 on >= 5 of 6. Control: H itself is not EQUIVALENT at 17000 or 18000 (positions).
  - Expected: B2 uncertain -- it targets the position convergence directly, but larger late geometry steps may make
    long_diff wander as (k)-(m); CF no gain (the lowest modes, which long_diff couples to, are free in both).
  - 36 jobs as 6 chains of two pipelines (<= 6 running, ~5-6 h); readout ENDFIT/read_n.py (dry-run on the H runs OK).
- **ENDFIT (n) RESULT (read 29 Oct after all 36 jobs COMPLETED; ENDFIT/read_n.py): neither arm is FASTER (no recipe
  candidate for fewer iterations); both SAFE; coarse-to-fine is BETTER by the rule (positions closer to L on 5 of 6).**

  | base @ 20000 | lifetime / long_diff | position - L | closer than H |
  |---|---|---|---|
  | EXP H (production) | 2.17 / 2.74 (s: 2.28 / 3.00 / 1.24 \| 3.23 / 1.22 / 3.76) | +27.9 (+23.7 / +26.4 / +33.5) | -- |
  | EXP B2 | 1.50 / 2.43 | +29.9 | 2/3 |
  | EXP CF | 1.68 / 3.22 (s: 3.49 / 0.10 / 1.45 \| 4.84 / 3.57 / 1.25) | **+21.3** (+19.0 / +28.5 / +16.3) | 2/3 |
  | KN10N H (production) | 1.90 / 1.92 (s: 1.56 / 1.82 / 2.33 \| 1.32 / 3.04 / 1.39) | +5.6 (+4.1 / +10.2 / +2.4) | -- |
  | KN10N B2 | 1.87 / 2.01 | +4.6 | 2/3 |
  | KN10N CF | **0.49 / 1.23** (s: 0.75 / 0.26 / 0.47 \| 0.34 / 1.70 / 1.66) | **+3.2** (+1.5 / +8.4 / -0.4) | 3/3 |

  - FASTER: no. At 18000 CF misses by a hair (EXP long_diff 3.65 vs 3.43; KN10N position +10.6 at the +10.6 um
    threshold); B2 misses on positions (EXP +37.0, KN10N +14.7). Not pre-registered, for the record: at 19000 every
    arm INCLUDING production H passes the rule (H 2.23 / 2.76 / +30.7 and 1.96 / 2.04 / +9.9), so 19000 says nothing
    about speed -- the +5 um position slack is what the rule allows.
  - B2 (Adam b2 0.95): neutral within noise (EXP calibration slightly closer, positions 4/6). Not adopted.
  - CF (coarse-to-fine until 5000): on true ends (KN10N) better on every sample in lifetime (3/3) and position (3/3,
    by 2-3 um), long_diff 2/3; on realistic ends (EXP) positions better on 2/3 (s = 4 by 17 um, s = 2 2 um worse) but
    lifetime 1/3 and long_diff 1/3 -- mixed. CF changes two things at once: the coarse phase AND dropping the hit
    seed's upper modes (zeroed at the first update), and the seed's upper modes are where the e1 seed outliers
    (aliasing, memory e1_hit_seed_outliers) come from; which one helps is not separated by this test. With n = 3 per
    base, CF is a candidate end-point QUALITY improvement at no cost, not a speed-up; adoption would need a
    confirmation (more samples against truth, plus a seed-upper-modes-dropped-only control) and Pierre's decision.
- **ENDFIT (o) -- coarse-to-fine CONFIRMATION against truth (pre-registered 29 Oct, before any result; Pierre: "Yes do
  the confirmation runs").** Samples KN10N3, 5, 6, 7, 8, 9 (true endpoints; not used in (n)). Three arms, each a full
  production pipeline (KNOT/submit_prod.sh) read at its H @ 20000:
  - P = production re-run on today's tree. Added while building this: the 24 Oct production runs of these samples
    (and of KN10N1, 2, 4 in (n)) ran at a5d7bdc + dirty tree, before the round-off and round-2 speed defaults
    (CONSISTENT, not bitwise); today's runs are at 19d2bb3 + flag-gated changes only (diff checked: every changed line
    on the default path is unchanged in effect). CF vs the 24 Oct runs would mix the method with that boundary and the
    replica scatter, so P is the paired reference, and P vs the 24 Oct H measures that noise.
  - CF = LARND_CHAIN_COARSE_ITER=5000 (as (n)).
  - SD = LARND_SEED_DROP_UPPER=1 (new, default off; CPU test on a synthetic curved track: lower modes = the full seed,
    upper 3 of 6 per half zeroed): the hit seed keeps only the modes the coarse phase fits, all modes free from the start.
  - Metrics vs truth (read_ens convention): error % per parameter (last 1000 iterations / target - 1) and position
    residual (um, last 1000 iterations).
  - Rules: POSITION CONFIRMED if CF's residual is below P's on >= 5 of 6 samples. CALIBRATION WORSE in a parameter if
    the paired mean of |err_CF| - |err_P| exceeds max(2 SE, 0.5 points). CF CONFIRMED = POSITION CONFIRMED and no
    parameter WORSE. Mechanism, only if CF is CONFIRMED: g_X = mean(pos_P - pos_X); SD/CF gain ratio >= 2/3 -> the
    SEED DROP explains it (the simpler change); <= 1/3 -> the COARSE PHASE; else both.
  - Expected: positions CONFIRMED (n: 3/3 on true ends, by 2-3 um against L), calibration NOT WORSE; mechanism
    uncertain, leaning seed drop (the e1 seed outliers are aliasing in the seed's upper modes).
  - 18 pipelines (54 jobs) as 6 chains of three (<= 6 running, ~9 h); readout ENDFIT/read_o.py (dry run OK).
- **29 Oct, while (o) runs: exploratory looks at the finished (n) runs (descriptive, no rule; the (o) rules and
  read_o.py are unchanged).**
  - Trajectories vs truth (100-iteration means, KN10N1 / 2 / 4): CF starts ahead (epoch 1: 620 / 622 / 604 vs H 661 /
    665 / 644 um), H catches up by the end of the coarse phase (iterations 4900-4999: 163.4 / 191.7 / 173.6 vs 163.5 /
    196.2 / 179.6), and at the release (5000) CF drops faster (5000 -> 6000: -23 / -25 / -23 vs H -13 / -17 / -18 um).
    CF's lead then shrinks to 2-3 um at 19000-20000. So the coarse phase does something at the release, not only at the
    start.
  - Seed level ((o) stage-1 [GEOM-HITS] lines = iteration-0 inputs, samples 5, 6, 8, 9): dropping the upper modes moves
    the per-batch median seed error little (e1 277 -> 271 um, e2 391 -> 376 um, batch averages).
  - Per track at 20000 (POLISH/joint/per_track_res.py; ENDFIT/read_n_tracks.py; KN10N1, 2, 4): CF's gain is in the
    tail. Of H's outliers (> 200 um) CF fixes 10 / 9 / 7 and creates 2 / 6 / 4; change of the weighted mean |r|
    -2.7 / -1.7 / -2.0 um = tail -4.0 / -2.0 / -1.4 + bulk +1.4 / +0.2 / -0.5; median unchanged. B2 neutral
    (+2.2 / -1.0 / -1.2). EXP not readable by the rebuild tool (endpoint columns), and EXP positions vs truth carry the
    1 mm smear anyway.
  - Calibration vs truth at 20000 (3 per base, H / B2 / CF mean |err|): KN10N lifetime 1.85 / 1.97 / 1.00, long_diff
    5.27 / 4.56 / 4.59; EXP lifetime 3.05 / 1.94 / 2.55, long_diff 1.77 / 2.97 / 4.73 (CF s2 +7.96). Within replica
    scatter; (o) decides.
  - Stage boundaries (production H runs): the first epoch after 10000 and after 14000 is 6-9 um worse in position
    (per-track chain Adam restarts + LR re-heats); at 10000 it recovers in one epoch, at 14000 positions stay ~3 um
    above the FR end while calibration is held at 1e-2 (loss +100), then the anneal recovers. Small; not pursued.
- **29 Oct: per-stage start-up cost and a per-PIPELINE compilation cache (engineering, no physics change).**
  - Measured on today's (o) pipelines (KN10N3P): every stage job spends ~9:50 in its first epoch (100 iterations) vs
    ~0:45 per epoch after, plus ~2:40 outside the iteration loop -> ~12 min per stage, ~36 of the 3:07 h pipeline. By
    the opt-team profile (opt/reports/prof2_r2.md section 11) ~6.5 min of each first epoch is XLA compilation (1356
    compiles). FR and H recompile everything stage 1 compiled for the same data, because the persistent cache is per
    JOB and deleted at exit. Cache size is no longer the problem it was (live per-job caches 6-41 MB, ~1900 entries:
    the data is no longer embedded since data-as-args).
  - CPU check (ENDFIT/cpu_pipecache/run.sh, 2 x 100 cm, 4 iterations, existing mode LARND_JAX_PERSISTENT_CACHE=shared
    with an explicit dir): A cold (197 entries) / B warm, same config: **BITWISE vs A, 0 new entries** / D default
    per-job cache: **BITWISE vs A** / E warm with FR-like learning rates (calib 1e-3, chain 2e-3): 2 new entries. So the
    compiled programs do not depend on the learning rates, and a warm cache returns the same executables. CPU timing
    says nothing (simulation compute dominates on CPU). F (warm, freeze released as in FR, FR-like rates): 2 new entries
    (201) -- weak, since no track of these 2 small batches was frozen in A either; the GPU test is the real check.
    G (warm, a DIFFERENT sample, --data_seed 2): 116 new entries vs 197 cold -> cross-sample reuse only partial (~40 %;
    the per-batch fit-side programs follow each batch's track count), so the natural unit is one cache per pipeline.
  - GPU test (pre-registered here, before it runs): KN10N3PC = the KN10N3P production pipeline (inner script identical
    apart from the name and the cache lines), all three stages sharing one cache dir
    (jax_job_cache/pipe_KN10N3PC, inside the sweeper's root -> removed after 2 idle days), chained after the (o) chain
    KN10N9 (afterany 42371120), so the running count stays 6. **CACHE HELPS if the first epochs of FR and H take
    <= 0.6 x KN10N3P's (9:50 / 9:50 -> <= 5:54) and the pipeline wall time is >= 10 min shorter than KN10N3P's
    (3:07:16)**; stage 1 is cold in both (control: its first epoch within +-15 % of P's 9:59). Physics: identical by
    construction (CPU bitwise); on GPU PC vs P is a same-data replica, read only as such (a free replica-noise point
    for (o)). Adoption into submit_prod.sh = Pierre's decision.
- **Per-pipeline cache RESULT (29 Oct 17:30; ENDFIT/wait_pc_h.sh, all 6 jobs COMPLETED, all on sdfampere020): CACHE
  HELPS** (all three pre-registered criteria met).
  - First epochs P / PC: fit 9:59 / 9:44 (control, -2.5 %, within +-15 %), **FR 9:50 / 3:20 (0.34 x), H 9:50 / 2:15
    (0.23 x)**; all three PC stages log `pipe_KN10N3PC (mode=shared`.
  - Walls P / PC: fit 1:28:04 / 1:27:33, FR 0:42:06 / 0:33:58 (-8:08), H 0:57:06 / 0:47:56 (-9:10); **pipeline 3:07:16
    / 2:49:27 = -17:49 (-9.5 %)**. The FR + H gain (17:18) is ~3.4 min more than the first-epoch gain (13:55): the
    start-up compiles outside the loop are reused too. Disk: 45 MB, 1984 files at the end; stage 1 wrote 1982 of them,
    so FR and H run almost entirely from the cache.
  - What is left: stage 1's cold ~9:45 first epoch. On CPU (G) a different sample reuses ~40 % of the entries, so one
    cache per ensemble would remove part of that; not tested on GPU.
  - Physics (descriptive, same-data replica): PCH vs PH @ 20000: pos 110.9 vs 111.1 um; Ab +0.23 / +0.21, eField +0.02
    / +0.01, tran_diff +1.92 / +1.96, long_diff +1.04 / -0.23, lifetime -5.30 / -1.68 (old 24 Oct H: -3.76). This is the
    known same-data replica scatter (GPU --non_deterministic): stage 1, cold in both, already differs (lifetime +0.86 /
    +0.31, long_diff +9.08 / +7.06), and on CPU the cache is bitwise.
  - Adoption (Pierre's decision; production unchanged): in KNOT/submit_prod.sh, export LARND_JAX_PERSISTENT_CACHE=shared
    and JAX_COMPILATION_CACHE_DIR=/sdf/scratch/neutrino/pgranger/jax_job_cache/pipe_<name> before the three sbatch
    calls (FR and H inherit both: the mode via env_overrides, the dir via --export=ALL). The sweeper removes each dir
    after 2 idle days. ~18 min per pipeline, ~10 %.
- **ENDFIT (o) RESULT (29 Oct 15:13; read_o.py, all 54 jobs COMPLETED, config check OK on all 18 end points).**
  Mean over KN10N3, 5, 6, 7, 8, 9 at 20000 vs truth (H old tree / P / CF / SD): position 111.6 / 111.8 / 104.1 / 109.2 um;
  mean |err| lifetime 2.64 / 2.45 / 2.47 / 3.93, long_diff 3.38 / 3.65 / 3.86 / 4.25, tran_diff 0.84 / 0.96 / 1.00 / 1.17,
  Ab ~0.2, eField ~0.01.
  - **CF CONFIRMED: positions below P on 6/6 (mean gain 7.7 +- 1.1 um, -6.9 %), calibration NOT WORSE** (paired
    d|err| CF - P: lifetime +0.02 +- 0.63, long_diff +0.21 +- 0.54, tran_diff +0.05 +- 0.32, Ab -0.01, eField 0.00).
  - **SD NOT CONFIRMED: positions below P on 4/6 (gain 2.6 +- 2.5 um), lifetime WORSE (+1.48 +- 0.47).**
  - Mechanism by the rule: SD/CF gain 0.34 -> "BOTH / mixed" (just above 1/3); SD's own gain is within 1 SE of zero and
    costs lifetime, so the coarse phase carries most of it -- consistent with the (n) trajectories (H catches up by the
    end of the coarse phase; CF pulls ahead at the release).
  - Per track (descriptive; per_track_res.py on all 18 end points, ENDFIT/read_o_tracks.py; outliers = |r| > 200 um
    in P): **CF's gain is all in the tail**: -7.7 +- 1.2 um = tail -8.4 +- 1.0 + bulk +0.7 +- 0.4; per sample it fixes
    11.2 of P's outliers and creates 4.3; median unchanged (+0.3 +- 0.3). SD: -2.8 +- 2.5 = tail -5.3 +- 1.5 + bulk
    +2.5 +- 1.1; fixes 7.7, creates 7.3. So dropping the seed's upper modes removes part of the aliasing tail but also
    real shape information the bulk tracks need; the coarse phase (upper modes re-learned from zero once the lower ones
    have settled) gets the tail without that cost.
  - Tree boundary: old 24 Oct H vs today's P = replica noise (positions -0.2 +- 0.8 um, |P - H| 0.5-3.7 um per sample;
    calibration within 0.3 +- 1.1). CF's gain is ~9 SE of that noise.
  - Trajectories (descriptive): CF is ahead in position by 8-11 um from 5000 on and stays ~8 um ahead to the end; CF at
    the FR end (14000) is only +2.6 um from P at 20000; CF @ 19000 is below P @ 20000 on 5/6. Lifetime is behind P by
    ~1.5 points through stage 1 (5k-10k) and catches up only at 19000-20000, so CF is a better end point, not a
    shorter run.
  - Production unchanged (adoption = Pierre's decision). Open before adopting: realistic ends, where (n) EXP showed
    long_diff vs truth worse with CF on 3/3 -> (p).
- **ENDFIT (p) -- CF on REALISTIC ends against truth (pre-registered 29 Oct 15:16, before any (p) result).**
  - Pairs at 20000: s = 1, 2, 4 from (n) (EXP<s>H, the 28 Oct production run, vs EXP<s>CFH; already read: long_diff
    d|err| +2.97 +- 0.94 = WORSE on those three alone) and s = 3, 5, 6 NEW (EXP<s>P = production re-run on today's
    tree vs EXP<s>CF). EXP = KN10N inner script + LARND_ENDPOINT_SMEAR_MM=1.0, _KEY=entry (as EXP1); CF adds
    LARND_CHAIN_COARSE_ITER=5000 (scripts = the (o) KN10N<s>{P,CF} ones + the two smear lines, diff checked). Mixing
    the old-tree reference for s = 1, 2, 4 is justified by (o): old H vs P = replica noise.
  - **Rule (n = 6): CALIBRATION WORSE in a parameter if the paired mean of |err_CF| - |err_ref| > max(2 SE, 0.5 points);
    CF SAFE ON REALISTIC ENDS if no parameter is WORSE, else NOT SAFE.** Because s = 1, 2, 4 are already known, the
    three new pairs are also reported alone (descriptive): if they show long_diff d|err| <= 0, the (n) signal was
    noise. Positions vs the unsmeared truth are smear-dominated (~850 um): descriptive only.
  - Expected: long_diff WORSE (the (n) signal is large), i.e. CF a candidate for true ends only, unless the new pairs
    contradict it.
  - 6 pipelines (18 jobs): 5 chains now + EXP6P after the cache test KN10N3PC (afterany 42401878), <= 6 running;
    readout ENDFIT/read_p.py (dry run on the (n) pairs OK).
- **Cache test 2 -- one cache per ENSEMBLE? (pre-registered 29 Oct 17:33, before it runs; engineering only).** With a
  per-pipeline cache only stage 1's cold first epoch is left (~7:20-10:00 over the (o) samples). KN10N5PE = stage 1 of
  KN10N5 (KN10N3PC's script with --data_seed 5 and its own name; nothing else differs) on a COPY of the warm
  pipe_KN10N3PC cache (jax_job_cache/pipe_ENS_test, swept after 2 idle days), cancelled after its first epoch. Cold
  reference: KN10N5's three (o) stage-1 jobs, 9:10 / 9:24 / 9:15. **ENSEMBLE CACHE WORTH IT if KN10N5PE's first epoch
  is <= 7:10 (>= 2 min below the cold ones)**; also reported: the entries it adds to the 1984 (CPU G: ~40 % reuse
  across samples). Not tested: many pipelines writing one dir at the same time. Chained after EXP5PH (afterany
  42422684), so the running count stays <= 6.
- **29 Oct 17:37, before any (p) result: what the (n) EXP long_diff signal looks like over time (exploratory; (n) set
  complete; scratchpad n_ld_traj.py / n_ld_L.py; does NOT change the (p) rule).** Signed long_diff error % vs truth,
  EXP1 / 2 / 4:
  - At the stage-1 end (10000) long_diff is far off on realistic ends in BOTH arms: H +28.4 / +10.8 / +11.3, CF -2.7 /
    +21.8 / +19.2 (CF - H mixed in sign: CF far better on EXP1, worse on EXP2, 4). FR barely moves it (14000: H +24.2 /
    +7.4 / +9.0, CF -2.2 / +15.6 / +16.3); the H stage does the work (20000: H +0.9 / +3.2 / +1.2, CF +2.5 / +8.0 / +3.7),
    and CF's EXP2 is still falling at 20000 (+10.2 @ 18000 -> +8.0). On true ends (KN10N1, 2, 4) the two arms track
    each other from ~6000 on and end equal (H +4.1 / +6.0 / +5.8, CF +3.1 / +4.7 / +6.0).
  - The reference is on the lucky side: the 44000-iteration long runs EXP<s>L end at -2.3 / +4.4 / +5.0 (mean |err| 3.9),
    the near-replica arm B2H (b2 0.95, neutral in (n)) at +1.8 / +1.7 / +5.4 (mean 3.0), H at 1.8, CF at 4.7. So on these
    three samples a single long_diff |err| at 20000 scatters by ~2-4 points between equivalent runs, and the (n) "WORSE
    +2.97 +- 0.94" is 3/3 in sign against one reference that sits below the converged value. The three NEW (p) pairs
    (both arms run now, neither selected) are the clean test; the pre-registered rule is applied unchanged.
- **29 Oct 17:39: per-pipeline cache adoption prepared, NOT applied (Pierre's decision): ENDFIT/cache_adoption/.**
  (1) submit_prod.sh exports both variables (pipe_<NAME>; overridable) = the tested path. (2) A trap found while
  preparing it: the mode is a LARND_* variable, so it is stored in every checkpoint and re-exported by all 10
  continuation builders; today mode=shared without a usable dir DISABLES the cache (~3x slower), so any later hand-made
  continuation of an adopted pipeline would run uncached. example_run.py.patch falls back to the per-job cache with a
  warning (logic test over all 7 env combinations: only the two broken cases change; nothing written under /sdf/group).
- **29 Oct 17:42, many writers on one cache dir (code read, JAX 0.6.2 in the container; for an ensemble-wide cache).**
  LRUCache.put writes the entry in place (no temp + rename) and takes no lock when eviction is off (our case: no
  filelock); it skips keys whose file already exists. A job reading an entry another job is still writing gets a
  truncated compressed payload -> decompression error -> compiler._cache_read warns and recompiles
  (raise_persistent_cache_errors defaults to False). So sharing fails safe (worst case a recompile and a warning).
  Within one pipeline the stages are sequential, so the per-pipeline cache never has two writers. One leak in both cases: a job
  killed mid-write leaves a truncated entry that put() never replaces (the file exists), i.e. that one program is
  recompiled in every later stage (cost only).
- **Cache test 2 RESULT (29 Oct 18:36; ENDFIT/wait_ens.sh): ENSEMBLE CACHE NOT WORTH IT.** KN10N5PE first epoch 9:10
  on the warm KN10N3 cache = the cold KN10N5 ones (9:10 / 9:24 / 9:15); the cache line confirms pipe_ENS_test was used.
  It added 1819 entries to the 1984 (83 MB): on GPU a different sample reuses almost nothing (CPU G: ~40 %), because the
  programs follow each sample's per-batch track counts and shapes. One cache per pipeline is the right unit (= the
  prepared submit_prod.sh change). Job cancelled after 11:30 as planned; its partial output sci_full_KN10N5PE and
  pipe_ENS_test are left for the 2-day sweep / not used anywhere.
- **ENDFIT (p) RESULT (29 Oct 20:42; read_p.py, all 18 jobs COMPLETED, config check OK on 6 pairs): CF NOT SAFE on
  realistic ends** (pre-registered rule applied unchanged).
  - Paired mean |err| reference / CF, d = CF - ref +- SE (n = 6): Ab 0.12 / 0.20 (+0.08 +- 0.04, below the 0.5 floor),
    eField 0.11 / 0.12 (+0.00), tran_diff 2.01 / 1.71 (-0.30 +- 0.31), **long_diff 1.71 / 3.44 (+1.73 +- 0.76 -> WORSE)**,
    lifetime 3.30 / 2.91 (-0.39 +- 0.71).
  - The three NEW pairs alone (descriptive): long_diff d +0.50 (s3 +1.30, s5 +1.13, s6 -0.92: worse on 2/3), so not
    "<= 0": the (n) signal is not just noise, but it is smaller on the clean pairs than on the (n) ones (+1.6 / +4.8 /
    +2.5, where the reference sat on the lucky side, see 17:37).
  - Positions (vs the unsmeared truth, smear-dominated, descriptive): CF lower on 5/6, 853.6 -> 843.1 um (-10.5 +- 3.4;
    new pairs -14.5): the coarse phase's position gain survives realistic ends.
  - So CF trades ~+0.5 to +3 points of long_diff |err| for ~10 um of position on realistic ends, and is a free gain only
    on true ends. Production unchanged; CF is NOT a default candidate as is (realistic ends are what data has).
- **ENDFIT (q) -- a SHORTER coarse phase on REALISTIC ends (pre-registered 29 Oct 21:42, before any (q) result;
  Pierre: "Please try to launch what you proposed").** Question: does a coarse phase of 2500 instead of 5000 iterations
  keep CF's position gain without its long_diff cost on realistic ends ((p): long_diff +1.73 +- 0.76 WORSE, positions
  -10.5 +- 3.4 um)?
  - C25 = EXP<s>CF's inner script with LARND_CHAIN_COARSE_ITER=2500 (and its own name; word-diff checked), s = 1..6,
    full production pipelines, same tree as (o)/(p) (code last changed 29 Oct 05:39). Compared per sample with the (p)
    reference (s = 1, 2, 4: EXP<s>H; 3, 5, 6: EXP<s>PH) and CF5000 = EXP<s>CFH, all at 20000. Readout ENDFIT/read_q.py
    (dry run with CF5000 in place of C25 reproduces (p) exactly).
  - **Rules (n = 6, vs the reference): C25 SAFE on realistic ends if no calibration parameter is WORSE (paired mean
    d|err| > max(2 SE, 0.5 points), as (p)); POSITION GAIN KEPT if C25 is below the reference on >= 5/6 AND its mean gain
    is >= half of CF5000's (>= 5.3 um). C25 = CANDIDATE if both; then (and only then) the true-end check on KN10N
    follows before any adoption.** C25 - CF5000 is reported descriptively.
  - Expected (honest prior): the long_diff cost and the position gain may both scale with the coarse window, i.e.
    "NOT A CANDIDATE" either way is plausible; a pass at n = 6 means "not shown worse", not "shown equal".
  - 6 chains (18 jobs), 6 running at once, no cache (kept identical to the arms it is compared with).
- **ENDFIT (q) RESULT (30 Oct 00:51; read_q.py, all 18 jobs COMPLETED, config check OK on 6 x 3 runs; logs confirm the
  2500 release on all 6): C25 NOT A CANDIDATE -- NOT SAFE (long_diff WORSE) and POSITION GAIN NOT KEPT.**
  - Paired vs the reference (mean |err| ref / CF5000 / C25; C25 - ref +- SE): long_diff 1.71 / 3.44 / **4.79 (+3.08 +-
    0.66 WORSE)**; Ab 0.12 / 0.20 / 0.18 (+0.06), eField +0.01, tran_diff -0.34 +- 0.51, lifetime -0.19 +- 1.28.
    Positions: C25 below ref on 4/6, gain +3.9 +- 2.6 um (needed >= 5.3; CF5000 +10.5).
  - C25 - CF5000 (descriptive): long_diff +1.35 +- 0.51 (C25 worse on 5/6), positions +6.7 +- 1.0 um (worse on 6/6). So
    a SHORTER window is worse on BOTH counts: the position gain grows with the coarse window, and the long_diff cost does
    not shrink with it -- it grows as the window shortens.
  - Hypothesis only (not tested): what costs long_diff is the RELEASE (upper modes switched on) happening while the
    calibration LR is still high -- at 2500 it is 0.1 x 0.91^20 ~ 1.5e-2, at 5000 ~ 1.4e-3 (optax decays after the 500
    warmup steps, one 0.91 step per 100 iterations) -- so the release's geometry
    shake is absorbed by long_diff. That predicts the opposite direction (a LONGER coarse phase, release at a lower
    calibration LR) would cost less long_diff and gain more position. Not launched (Pierre's call).
- **ENDFIT (r) -- a LONGER coarse phase on REALISTIC ends (pre-registered 30 Oct 01:15, before any (r) result; Pierre:
  "Please feel free to launch the jobs to test your hypothesis").** Hypothesis from (q): the long_diff cost comes from the
  release happening while the calibration LR is high (2500: ~1.5e-2, 5000: ~1.4e-3), so a later release costs less.
  - C75 = EXP<s>CF's inner script with LARND_CHAIN_COARSE_ITER=7500 (release at a calibration LR ~1.4e-4; nothing else
    happens at 7500, unlike 10000 where the FR stage re-heats the LRs and releases the freeze), s = 1..6, full production
    pipelines, same tree (code last changed 29 Oct 05:39), word-diff checked. Readout ENDFIT/read_r.py (dry run with
    CF5000 in place of C75 reproduces (p) and (q)).
  - **Rules (n = 6). HYPOTHESIS SUPPORTED if C75 - CF5000 long_diff paired d|err| < -max(2 SE, 0.5); TREND only if < 0 but
    not that; NOT SUPPORTED if >= 0. Candidacy as (q): C75 SAFE if no calibration parameter is WORSE vs the reference
    (d|err| > max(2 SE, 0.5)); POSITION GAIN KEPT if below the reference on >= 5/6 and gain >= 5.3 um; CANDIDATE if both
    (then the true-end KN10N check before any adoption).**
  - Expected under the hypothesis: long_diff cost below CF5000's +1.73 and position gain >= CF5000's 10.5 um. Against
    it: the H stage re-heats the calibration LR to 1e-2 with all modes free anyway, so the release timing in stage 1
    may not matter at 20000 (then C75 ~ CF5000 or worse).
  - 6 chains (18 jobs), 6 running, no cache.
- **ENDFIT (r) RESULT (30 Oct 04:55; read_r.py, all 18 jobs COMPLETED, config check OK on 6 x 4 runs; logs confirm the
  7500 release on all 6): by the rule C75 = CANDIDATE (SAFE + POSITION GAIN KEPT); HYPOTHESIS: TREND only.**
  - Paired vs the reference (mean |err| ref / C25 / CF5000 / C75; C75 - ref +- SE): long_diff 1.71 / 4.79 / 3.44 / 3.30
    (+1.59 +- 0.97, not WORSE: below 2 SE), lifetime 3.30 / 3.11 / 2.91 / 1.96 (-1.34 +- 1.29), tran_diff +0.13 +- 0.44,
    Ab -0.02, eField 0.00. Positions: below ref on 5/6, gain +11.0 +- 3.1 um (CF5000 +10.5, C25 +3.9).
  - Hypothesis: C75 - CF5000 long_diff -0.14 +- 0.98 -> TREND only. In substance C75's long_diff cost (+1.59) is the same
    as CF5000's (+1.73, which failed); it passes only because its scatter is larger (per sample d|err| -0.8 / +4.4 /
    -0.5 / +3.7 / -0.3 / +3.1: three samples ~3-4 points worse, three slightly better). So: 2500 is clearly worse, but
    5000 -> 7500 changes nothing measurable in long_diff; the release-timing hypothesis explains the short-window
    damage at most, not the ~+1.6 point cost of CF on realistic ends. Lifetime is better with C75 (n.s.), positions equal
    to CF5000 (C75 - CF5000 -0.5 +- 1.5 um).
  - Per the pre-registered program the next step is the true-end check of C75 on KN10N; even if it passes, the
    realistic-end long_diff question stays open ("not shown worse" at n = 6 with a mean cost of ~1.6 points).
- **ENDFIT (s) -- the TRUE-END check of C75 (pre-registered 30 Oct 04:56, before any (s) result; the next step of the
  pre-registered program after (r) "CANDIDATE").** KN10N<s>C75 = the (o) KN10N<s>CF inner script with
  LARND_CHAIN_COARSE_ITER=7500 (and its own name; word-diff checked), s = 3, 5, 6, 7, 8, 9, full production pipelines, same
  tree. Readout ENDFIT/read_s.py vs truth at 20000, arms P (the (o) reference) / CF5000 / C75 (dry run with CF5000 in
  place of C75 reproduces (o): 6/6, -7.7 +- 1.1 um).
  - **Rule = the (o) rule: C75 CONFIRMED on true ends if positions are below P on >= 5/6 AND no calibration parameter is
    WORSE vs P (paired d|err| > max(2 SE, 0.5)).** C75 - CF5000 reported descriptively.
  - Even if CONFIRMED, adoption remains Pierre's call on the realistic-end trade ((r): long_diff +1.59 +- 0.97, i.e. not
    shown worse, with the same mean cost as CF5000).
  - 6 chains (18 jobs), 6 running, no cache.
- **ENDFIT (s) RESULT (30 Oct 08:39; read_s.py, all 18 jobs COMPLETED, config check OK on 6 x 3 runs, 7500 release in all
  6 logs): C75 NOT CONFIRMED on true ends -- tran_diff WORSE.** Control: CF5000 reproduces (o) (CONFIRMED, 6/6, -7.7 um).
  - Paired vs P (mean |err| P / CF5000 / C75; C75 - P +- SE): **tran_diff 0.96 / 1.00 / 1.69 (+0.74 +- 0.24 WORSE; worse on
    5/6: s3 1.96 -> 2.97, s6 0.69 -> 1.91, s7 0.18 -> 1.00, s8 0.04 -> 1.33, s9 1.97 -> 2.27; s5 0.90 -> 0.68)**;
    long_diff 3.65 / 3.86 / 2.65 (-1.00 +- 0.79), lifetime 2.45 / 2.47 / 2.88 (+0.43 +- 0.64), Ab -0.01, eField 0.00.
    Positions below P on 6/6, gain +8.4 +- 1.5 um (CF5000 +7.7; C75 - CF5000 -0.8 +- 0.7).
  - Possible reading (not tested): with the release at a calibration LR of ~1.4e-4 the calibration that co-adapted to the
    coarse geometry has little room to re-adapt in stage 1; the transverse part is what is left at 20000 on true ends.
  - **Coarse-to-fine thread, overall: no variant is a clean default.** CF5000: true ends CONFIRMED, realistic ends
    long_diff WORSE (+1.73 +- 0.76). C25: worse on both. C75: realistic ends "not shown worse" (long_diff +1.59 +- 0.97,
    the same mean cost as CF5000), true ends tran_diff WORSE. The position gain (~8-11 um) is robust in every arm with a
    window >= 5000; every arm pays for it in one calibration parameter somewhere. Production unchanged.
- **30 Oct 21:50, the (s) "possible reading" checked on the completed (s) data (exploratory, post-hoc; Pierre: "Run the
  test you proposed").** The proposed test (re-heat the calibration LR at the 7500 release) only makes sense if C75's
  tran_diff offset is created in stage 1 and carried to 20000. The (s) histories say it is not, so that test was not
  launched:
  - tran_diff mean |err| P / CF5000 / C75 at 10000: 1.07 / 1.59 / 1.27 (C75 - P +0.20 +- 0.34); at 14000: 0.88 / 1.40 /
    1.12 (+0.24 +- 0.18); at 20000: 0.95 / 1.01 / 1.69 (+0.74 +- 0.24). CF5000 is the arm with the larger stage-1 offset
    (+0.53 at 10000), and it ends at +0.07.
  - Carry-over: per-sample signed (C75 - P) tran_diff at 10000 vs at 20000, Pearson r = -0.08 (at 14000: +0.26); at
    16500, the end of the H stage's calibration hold at 1e-2: r = +0.78. The 20000 end point is set during that hold, not
    left over from stage 1.
  - Not run-to-run noise: P (29 Oct) vs the production KN10N<s>H (same recipe) on the same six samples differ by a
    tran_diff signed SD of 0.30 (d|err| +0.11 +- 0.15; KN10N3PC vs P: 0.04), against a C75 - P signed SD of ~1.1. For
    comparison, replica SDs are long_diff 2.68 and lifetime 1.75.
  - So the C75 tran_diff cost on these six samples is real. Its mechanism is not the stage-1 calibration LR, and its sign
    is mixed per sample (signed C75 - P: +1.0 +0.2 +1.2 -1.2 -1.3 +0.3). The open question is whether the cost
    generalises to other samples: (t).
- **ENDFIT (t) -- does the C75 tran_diff cost REPLICATE on unseen samples? (pre-registered 30 Oct 21:50, before any (t)
  result).** KN10N<s>C75 for s = 1, 2, 4, 10, 11, 12 (true ends, never used in a coarse-to-fine test) = the production
  inner script KN10N<s> + LARND_CHAIN_COARSE_ITER=7500 and its own name. Diff-checked: identical to KN10N3C75 up to seed
  and name (KN10N1 also exports the three speed flags, at the run_knot defaults). Full production pipelines; no source
  change since the (s) runs. Reference = the production KN10N<s>H (24 Oct fit + FR, 28 Oct H; the control above shows P
  vs production consistent). Readout ENDFIT/read_t.py (dry run on the (s) samples with ref = P reproduces (s)).
  - **PRIMARY: REPLICATED if tran_diff C75 - ref paired d|err| > max(2 SE, 0.5) on the six new samples; NOT REPLICATED
    otherwise.** SECONDARY: the (o) rule on the new samples (positions below ref on >= 5/6 AND no calibration parameter
    WORSE). Pooled n = 12 (the (s) samples vs P plus the new samples vs production) reported descriptively.
  - Pre-stated reading: REPLICATED -> the (s) verdict stands and the CF thread stays closed. NOT REPLICATED -> the (s)
    tran_diff flag does not generalise; the pooled numbers are reported and C75's status (with the (r) realistic-end
    long_diff +1.59 +- 0.97) goes back to Pierre.
  - 6 chains (18 jobs), 6 running, no cache.
- **ENDFIT (t) RESULT (31 Oct 01:00; read_t.py, all 18 jobs COMPLETED, config check OK on 6 x 2 runs, 7500 release in all 6
  logs): tran_diff cost NOT REPLICATED; C75 CONFIRMED on the new samples by the (o) rule.**
  - New samples (mean |err| ref / C75; C75 - ref +- SE): tran_diff 1.30 / 1.24 (-0.06 +- 0.13, worse on 3/6), long_diff
    4.32 / 4.03 (-0.29 +- 0.98), lifetime 2.12 / 2.31 (+0.19 +- 0.44), Ab +0.04, eField 0.00. Positions below ref on 6/6,
    gain +9.1 +- 2.4 um.
  - Pooled n = 12 (descriptive): tran_diff +0.34 +- 0.18, long_diff -0.64 +- 0.61, lifetime +0.31 +- 0.37, Ab +0.02,
    eField 0.00, positions -8.8 +- 1.3 um (12/12 below).
  - Reading (pre-stated): the (s) tran_diff cost is real on those six samples (replica SD 0.30) but does not generalise.
    It is a sample-specific shift of the H-stage end point, not a systematic cost of C75. On true ends C75 is now
    CONFIRMED on 6 + 6 samples with no calibration parameter worse. C75's status goes back to Pierre. Its remaining open
    point is the realistic-end long_diff ((r): +1.59 +- 0.97, n = 6, not shown worse; long_diff replica SD ~2.7).
    Production unchanged.
- **ENDFIT (u) -- C75 on REALISTIC ends, six more samples (pre-registered 31 Oct 05:05, before any (u) result; Pierre: "Do that
  last test").** Settles the one open point after (t): (r)'s long_diff +1.59 +- 0.97 at n = 6.
  - New samples s = 7..12 (data seeds 7..12, never used with the endpoint smear), two full production pipelines each,
    same day and tree (no source change since 29 Oct 05:39). Ref = EXP<s> = the production KN10N<s> inner script + the 1 mm
    independent endpoint smear (LARND_ENDPOINT_SMEAR_MM=1.0, KEY=entry). C75 = EXP<s>C75 = EXP<s> +
    LARND_CHAIN_COARSE_ITER=7500. Diff-checked: EXP<s> = KN10N<s> + the two smear exports; EXP<s>C75 = EXP<s> + the
    coarse line (names aside). Readout ENDFIT/read_u.py (dry run on the (r) samples reproduces (r) for C75).
  - **PRIMARY = pooled n = 12 ((r) samples 1..6 with their (r) references + new samples 7..12), the (r) candidacy rule:
    SAFE if no calibration parameter is WORSE (paired d|err| > max(2 SE, 0.5)); POSITION GAIN KEPT if below the
    reference on >= 10/12 and gain >= 5.3 um. C75 passes on realistic ends if both hold.** The new six alone are
    reported with the same rule (>= 5/6).
  - Pre-stated reading: passes -> C75 has passed every pre-registered check (true ends n = 12, realistic ends n = 12), so
    it is a candidate production default; adoption is Pierre's call. long_diff (or any parameter) WORSE pooled -> C75 is not
    a clean default and the CF thread closes.
  - Power, for honesty: with the (r) scatter, a true long_diff cost of +1.6 would give a pooled SE of ~0.7, i.e. a
    roughly even chance of being flagged. A pass therefore means "no cost of about this size seen at n = 12", not
    "no cost".
  - 6 chains, each running EXP<s> then EXP<s>C75 (36 jobs, 6 running), no cache.
- **ENDFIT (u) RESULT (31 Oct 11:25; read_u.py, all 36 jobs COMPLETED, config check OK on 6 x 2 runs; logs: the 7500 release in all
  6 C75 runs, no coarse phase in the 6 references): C75 on realistic ends SAFE + POSITION GAIN KEPT (pooled n = 12) ->
  C75 passes every pre-registered check.**
  - New samples 7..12 (mean |err| ref / C75; C75 - ref +- SE): long_diff 4.10 / 4.36 (+0.26 +- 1.37; per sample mixed,
    e.g. s11 -1.95 -> +8.01, s12 -3.40 -> +0.06), lifetime -0.27 +- 1.29, tran_diff +0.13 +- 0.37, Ab -0.03, eField 0.00.
    Positions below ref on 6/6, gain +10.7 +- 1.6 um. The same rule alone: SAFE + GAIN KEPT.
  - PRIMARY, pooled n = 12: long_diff 2.90 / 3.83 (+0.93 +- 0.82, not WORSE), lifetime -0.80 +- 0.88, tran_diff +0.13 +-
    0.27, Ab -0.03, eField 0.00. Positions below ref on 11/12, gain +10.9 +- 1.6 um.
  - C75 overall: true ends n = 12 (positions -8.8 +- 1.3 um, 12/12; no parameter worse; tran_diff +0.34 +- 0.18) and
    realistic ends n = 12 (positions -10.9 +- 1.6 um, 11/12; no parameter worse; long_diff +0.93 +- 0.82). Same run time
    as production. As pre-stated: no long_diff cost of the (r) size is seen at n = 12; the data allow anything from about
    -0.7 to +2.6 points (2 SE). Candidate production default (one line: LARND_CHAIN_COARSE_ITER=7500 in the fit stage);
    adoption is Pierre's call. Production unchanged.
- **31 Oct: C75 ADOPTED as production (Pierre: "Ok, let's adopt this").** KNOT/submit_prod.sh passes
  LARND_CHAIN_COARSE_ITER=7500 to the fit job (sbatch --export=ALL,...). The FR and H stages inherit it from the checkpoint
  provenance, exactly as in (r)-(u). An inner script's own export wins; an empty value or 0 in the caller's environment
  turns it off. The submitted.txt line shows "[coarse 7500]". Previous launcher: submit_prod.sh.pre_c75_31oct. Checked:
  the variable reaches the container and the script parses. Existing ensembles (KN10N1-16 H) are pre-C75.
- **UQ -- uncertainty quantification and convergence (Pierre, 31 Oct: "investigate uncertainty quantification and
  robustness ... the hessian metric we use is not really well calibrated ... a technique that allows to estimate if the fit
  has converged or not and estimate the uncertainty"). Plan written 31 Oct 12:56, before any UQ result.**
  - Starting point: the paper's Hessian (at truth, true inputs, nuisances fixed: "Floor" / "Conditional" in tab:floor)
    is 3-5x below the observed sample-to-sample SD. It omits (i) the nuisance-profiling inflation and (ii) the
    optimisation scatter. Today's replica control: same recipe, same data, signed-difference SD tran_diff 0.30, long_diff
    2.68, lifetime 1.75, i.e. a per-run optimisation scatter of ~1.9 / 1.2 points in long_diff / lifetime, a large part of
    the ~3-4 / ~3 point sample SD. Also known: the full objective's minimum at fixed fitted positions is lifetime ~-6.5 %
    (joint step, 21 Oct), and the full-joint quadratic model is poor (28 Oct smoke: gain overstated 5x).
  - Candidate per-fit estimators, all from ONE held assembly epoch at the end point (new LARND_JOINT_NEWTON_DUMP; default
    off; CPU test: dumped Schur pieces = exact, -S^-1 r = exact step). With S = sum_b S_b and r = sum_b r_b:
    E1 conditional H_cc; E2 dE/dx-profiled Schur; E3 fully profiled Schur (dE/dx + geometry); E4 sandwich
    S^-1 J S^-1 with J the scatter of the per-batch profile scores r_b; convergence: the Newton offset d = -S^-1 r in
    units of sigma (MINUIT-EDM-like).
  - Builder UQ/make_jn_dump.py: the end point's own stage script (e.g. KNOT/inner_sci_full_<NAME>H.sh) changed only by
    --iterations 101, name, save_freq, --resume_from <checkpoint> and the two exports (diff-checked on KN10N3PH).
  - **Pilot (4 jobs): KN10N3PH, KN10N3H (its replica), KN10N3C75H, and KN10N3PH with all finite-difference steps x3.
    Method OK if: S (kn 0) is positive definite; at most 5 % of the nuisance eigenvalues are dropped; with the x3 steps every
    E3 sigma is within 10 % and every Newton offset within 0.3 sigma.** Otherwise fix before any ensemble run. The ensemble
    test (pull calibration against truth, and whether d predicts the replica differences and the FR -> H moves) gets
    its own rules before launch.
- **UQ pilot RESULT (31 Oct 14:05; read_pilot.py, 4/4 jobs COMPLETED, 100 batches each): method NOT OK, as pre-stated ->
  fix before any ensemble run.**
  - E3 (fully profiled) is not positive definite on any of the 4 dumps (S min eig ~-4e7 vs H_cc min eig ~2e3, normalised
    units). With the x3 steps the E3 sigma ratios are 0.96 2.35 3.50 3.18 0.96 and |dz| up to 5.4 -> NOT STABLE. Dropped
    nuisance eigenvalues 0.7-1.0 % (passes). E2 (dE/dx profiled only) is positive definite on 3 of 4 but unstable on
    KN10N3PH at x1 (sigma 1.4 / 2.1 / 1.6 % in tran / long / lifetime, ~0.45 / 0.93 / 1.2 % on the other two and at x3).
  - E1 (conditional, the paper's kind of Hessian at the fitted point) IS step-stable (sigma within 1-8 %, z within 0.1) and
    reproduces tab:floor "Floor" (0.030 0.0004 0.37 0.85 0.75 %). Its Newton offsets are ~1 sigma (EDM 2-4 for 5 parameters)
    on all three end points: at fixed nuisances every end point is converged in calibration.
  - Anatomy (diag_batches.py): the per-batch S_b are non-PD in 60-90 of 100 batches (not a few outliers). The curvature
    removed by profiling shrinks with the step (KN10N3PH, long_diff: 18x H_cc at x1, 3x at x3; tran_diff 11x -> 2x; Ab and
    lifetime, coupled to dE/dx, stable at 0.7 / 0.7) and is per-batch correlated 0.97-1.00 between x1 and x3. Not random
    round-off then (that would decorrelate): the joint gradient varies faster than quadratically on the scale of the
    steps (5 um geometry, 0.3 % dE/dx), deterministically per batch. Hypothesis: the curvature must be probed at the scale
    of the parameter uncertainty, not at the round-off-optimal scale.
- **UQ pilot 2: step-size plateau (rules written 31 Oct 14:10, before launch).** fit_params: LARND_JOINT_NEWTON_DUMP_FULL=1
  also dumps every batch's assembled blocks (compressed); LARND_JOINT_NEWTON_EPS_CALIB may be 5 per-parameter steps. Both
  default off / unchanged; CPU test passes (flattened blocks = assembled, Schur from them = exact, per-parameter steps).
  - 6 jobs, all FULL. Steps (calibration normalised, dE/dx log, geometry cm) x10 = (1e-2, 3e-2, 5e-3), x20 = (2e-2, 6e-2,
    1e-2), x30 = (3e-2, 9e-2, 1.5e-2). The x20 calibration step is 0.4-0.7 of the H @ 20000 sample SD in normalised units
    for Ab, tran_diff, long_diff, lifetime (eField: ~25 SD). P3e10, P3e20, P3e30 on KN10N3PH; O3e20 on KN10N3H; C3e20 on
    KN10N3C75H; P3e20f = P3e20 with the eField step kept at 1e-3.
  - **Method OK at x20 if: S (kn 0) is positive definite on P3e20, O3e20 and C3e20 with at most 5 % of nuisance eigenvalues
    dropped; AND both neighbours P3e10 and P3e30 agree with P3e20 (every E3 sigma within 10 %, every E3 z within 0.3).**
    P3e20f vs P3e20 with the same thresholds decides whether the eField step matters (descriptive).
  - If E3 fails but E2 passes the same rules, the ensemble runs on E2 (geometry conditional, stated as such). If both fail,
    the Schur route is closed: the uncertainty then comes from E1 (or its sandwich) plus a replica-measured optimisation
    scatter, and the convergence test from replicas only.
- **UQ exploratory (31 Oct 14:17, no rule; UQ/loss_vs_err.py): the end-point training loss is NOT a calibration-convergence
  signal.** Same-data fits at H @ 20000 (production O, replica P, C75; 24 pairs over samples 1-12) end with total losses
  (last epoch, 100 batches) that differ by a median 2.6e3 (range 3 to 6.5e3); the loss evaluation itself repeats to 0.24
  (P3 vs P3e3 dumps). Every fit still loses 30-250 per epoch at 20000. dL follows the position residual (corr +0.86; C75
  lowest on 12/12 samples) but not the calibration: the lower-loss fit has the smaller |error| in 0.33-0.67 of pairs
  (chance), corr(dL, d|err|) between -0.33 and +0.21. So the calibration differences between same-data fits cost no
  visible loss: they sit along directions the objective barely constrains once the nuisances are free, under a loss
  that keeps falling through the positions. A convergence test on the loss (or on its decrease) would therefore say
  nothing about the calibration.
- **UQ pilot 2 RESULT (31 Oct 15:25; read_pilot2.py, 6/6 jobs COMPLETED): E3 and E2 NOT OK -> as pre-stated, the Schur
  (profile-Hessian) route is CLOSED.**
  - E3 (dE/dx + geometry profiled): not positive definite on any x20 dump (min eig -1e8 to -1e9); x10 / x30 vs x20 sigma
    ratios 0.13-1.14, |dz| up to 16. Worse than pilot 1, not better.
  - E2 (dE/dx profiled): positive definite on all three x20 dumps; tran_diff and long_diff step-stable (sigma within 8 %,
    |dz| <= 0.2) but Ab and lifetime drift with the step (x10 / x20 / x30 = 0.056 / 0.062 / 0.078 and 1.31 / 1.41 / 1.71 %;
    E2 at x3 = 0.056 / 1.32, i.e. flat over x3-x10, then rising) and eField is noisy (+-25 %). Fails the x20 rule. (Post
    hoc: x3-x10 looks like a plateau for 4 of 5 parameters; not tested as a rule.)
  - E1 (conditional) again step-stable: 0.030 0.0004-0.0008 0.37 0.85-0.88 0.72-0.76 % on every dump, z ~1-2.
  - Anatomy (diag_full.py, neg_modes.py, track_neg.py on the FULL dumps): each batch's joint matrix [[H_cc, H_cn], [H_nc,
    H_nn]] is indefinite in 92-100 of 100 batches (Jacobi-scaled min eig median -0.3 to -0.5); the negative directions
    are geometry (eigenvector weight 0.5-0.8; calibration 0.04-0.07, dE/dx 0.07); the dE/dx block alone is PD in every
    batch. 9-11 % of the 889 per-track geometry blocks are indefinite, but WHICH tracks depends on the step: on the same
    end point the x10 / x20 sets overlap 0.28 (overlap / union), less than two different end points at the same step
    (0.54). No spectral cut (1e-6 ... 1e-3) restores a PD profile; a per-batch PSD projection gives PD but step-dependent
    sigmas (long_diff 1.5 / 1.9 / 2.2 % at x10 / x20 / x30) and meaningless offsets.
  - Reading: the joint objective is not smooth in the geometry on any probe scale from 5 um to 150 um (discrete pixel /
    tick selections, e.g. the stop_gradient'ed top-k ticks of fee_jax, make it piecewise), and the end point is not a
    stationary point in the geometry (the loss still falls ~100 / epoch, through the positions). A second-order profile
    over the geometry is therefore not defined at these end points; the 5 global calibration parameters, which sum over
    many pixels, keep a stable conditional curvature (E1).
- **UQ fallback, existing runs (31 Oct 15:40, descriptive; err_budget.py, rep_avg.py). H @ 20000 production, %:**

  | | Ab | eField | tran_diff | long_diff | lifetime |
  |---|---|---|---|---|---|
  | E1 sigma (fitted point, conditional) | 0.030 | 0.0006 | 0.37 | 0.86 | 0.75 |
  | E2 sigma (dE/dx profiled, x3-x10) | 0.056 | ~0.0007 | 0.46 | 1.0 | 1.31 |
  | sigma_opt (6 replica pairs) | 0.072 | 0.007 | 0.21 | 1.90 | 1.24 |
  | sample SD (n = 16) | 0.122 | 0.006 | 1.02 | 1.89 | 2.73 |
  | mean error (bias) | +0.27 | +0.010 | +0.66 | +3.77 | -0.78 |
  | SD / sqrt(E1^2 + sigma_opt^2) | 1.56 | 0.86 | 2.39 | 0.91 | 1.88 |

  - The fallback "E1 + replica scatter" covers long_diff and eField (both optimiser-limited: sigma_opt / SD = 1.0 / 1.1)
    but under-covers tran_diff (2.4x), lifetime (1.9x) and Ab (1.6x): the missing piece is the nuisance-profiled
    statistical part that E3 was meant to give (E2 closes part of it for Ab / lifetime). Biases of ~2 SD in Ab and
    long_diff are outside any sigma.
  - Averaging the two replicas of a pair (exploratory): spread long_diff 2.01 -> 1.50, lifetime 3.21 -> 3.09, tran_diff
    and Ab unchanged; RMS error vs truth long_diff 3.93 -> 3.74 (bias-dominated).
  - Dumps (FULL, 130 MB each, 6) in UQ/dumps/.
- **UQ TEST of the recommendations (Pierre, 3 Nov: "Test your recommendations as proposed"; rules written 3 Nov 06:58,
  before any job of this test was submitted; submitted 06:59).** Recommendations under test: (R1) per fit, E1 at the end point = the floor
  and a quick convergence check (|z| ~ 1: converged given the nuisances); (R2) convergence from replica pairs: a parameter is
  converged if sigma_opt < 0.5 x its total sigma; (R3) calibrated sigma = the SD of a simulated ensemble run with the same
  recipe, biases from the same ensemble; (R4) shortcut sigma = k x sigma_E1, k = SD / sigma_E1 measured on simulation.
  - Configuration under test: production C75 on REALISTIC ends (1 mm independent smear, key entry) = the EXP<s>C75 recipe,
    the closest to real data. TRAINING = frozen predictions, UQ/train_ref.py on EXP1-12C75H @ 20000 (n = 12, computed
    before any test job; error = last-1000 mean / truth - 1, %):

    | | Ab | eField | tran_diff | long_diff | lifetime |
    |---|---|---|---|---|---|
    | bias_train | +0.013 | +0.131 | +1.599 | +2.827 | +0.847 |
    | SD_train | 0.152 | 0.035 | 1.285 | 4.064 | 3.866 |

  - Jobs (UQ/submit_uqtest.sh; 6 dependency chains, <= 6 running; ~80 GPU-h, ~13-14 h):
    - Set T1: 18 E1 dumps of the training pipelines EXP1-6C75 at 10000 (stage 1), 14000 (FR) and 20000 (H), with the new
      LARND_JOINT_NEWTON_DUMP_CALIB=1 (calibration columns only, ~1/6 of a full dump; default off; CPU test: H_cc, g_c, H_nc
      exact, H_nn not assembled, all other modes unchanged). Command built at job start (UQ/run_dump.sbatch).
    - Set A: 12 FRESH samples, data_seed 17-28 (never used before), production pipeline EXP<s>C75 -> H @ 20000. Inner =
      inner_sci_full_EXP3C75.sh with only --data_seed and --test_name changed (diff checked).
    - Set R: full-pipeline replicas EXP<s>C75R (inner identical but for test_name; --non_deterministic makes GPU re-runs
      diverge: KN10N3PC vs KN10N3P, same tree, differ by 1.3 / 3.6 points in long_diff / lifetime at 20000) and
      H-stage-only replicas EXP<s>C75H2 (prodfast again from the same EXP<s>C75FR @ 14000), s = 17-24.
    - Set D: 12 E1 dumps (calibration only) of EXP17-28C75H @ 20000.
    - Code: since the training runs (30-31 Oct, no source change after 29 Oct 05:39 but these) fit_params.py changed only
      by the env-gated UQ dump flags (LARND_JOINT_NEWTON_DUMP / _FULL / _EPS_CALIB on 31 Oct, _CALIB today); the default
      path is unchanged (find: no other source file newer than 29 Oct 05:00).
  - **T1 (R1, quick check).** A checkpoint is FLAGGED if max_p |z_p| > 3 (E1, uq_lib). It MOVES if some parameter's
    final error (EXP<s>C75H last-1000 mean) differs from its error at the checkpoint (value at iteration t) by more than
    SD_train. The quick check HAS POWER if it flags >= 2/3 of the moving 10000 / 14000 checkpoints and <= 1/6 of the 20000
    end points; VACUOUS if it flags <= 1/3 of the moving checkpoints; PARTIAL otherwise; UNDETERMINED if < 3 move.
    Expected: VACUOUS (Adam tracks the conditional minimum; the end point moves because the nuisances move).
  - **T3 (R3, coverage on unseen data), Set A.** Pulls q = (e - bias_train) / (SD_train sqrt(1 + 1/12)), expected ~ t_11.
    Per parameter CALIBRATED if SD_test / SD_train is in [0.54, 1.86] (95 % of F(11, 11)); BIAS TRANSFERS if |mean_test -
    bias_train| < 0.82 SD_train (= 2 SD_train sqrt(1/12 + 1/12)). Pooled over the 60 pulls: |q| < 1 expected 0.66, |q| < 2
    0.93; PASS if the first is in [0.50, 0.82] and the second >= 0.83 (wider than binomial: the parameters are correlated).
    R3 CONFIRMED if every parameter is CALIBRATED and the pooled coverage PASSES. Descriptive: the same pulls with
    sigma_E1 alone (the current metric) and with E1 (+) sigma_opt.
  - **T2 (R2, replica convergence), Set R.** sigma_opt = SD(A - B) / sqrt(2) over the 8 full-pipeline pairs (ddof 1); rho
    = sigma_opt / SD_train; converged if rho < 0.5. Recorded against the pre-C75 true-end classes (tran_diff converged;
    long_diff, eField not; Ab, lifetime borderline), not a rule (recipe-specific). Single pair: the fraction of the 8
    single pairs whose |A - B| / sqrt(2) gives the 8-pair class; "one pair suffices" CONFIRMED if >= 7/8 for every
    parameter, else the number of pairs k at which a bootstrap gives >= 80 % agreement is reported (expected: NOT, one
    pair is a 1-dof estimate). Cheap replica: sigma_opt^H = SD(H - H2) / sqrt(2) over the 8 H-only pairs; for every
    parameter with rho >= 0.5 (all if none), H-ONLY SUFFICES if sigma_opt^H / sigma_opt >= 0.7, INSUFFICIENT if < 0.5 for
    any, else PARTIAL. Expected: INSUFFICIENT for long_diff (its scatter is seeded in stage 1, rep_stage.py).
  - **T4 (R4, k x E1), Sets T1 + D.** k_p = SD_train,p / median sigma_E1,p over the six training end points EXP1-6C75H;
    predicted per fit sigma_i = k sigma_E1,i on the 12 Set-D dumps; same CALIBRATED and coverage rules as T3. If the CV of
    sigma_E1 over the 12 test fits is < 10 % for every parameter, k x E1 equals the ensemble SD in this configuration
    (it could add information only across configurations, e.g. data size, which this does not test).
  - Reading order: T1 when all 18 T1 dumps are COMPLETED; T3 when all 36 Set-A jobs are; T2 when all 8 R pipelines and 8
    H2 jobs are; T4 when all 12 Set-D dumps are. Config check from the logs (coarse release at 7500, smear exports,
    data_seed) before each readout.
  - 07:05, before any result: the six 10000 dumps FAILED at start-up (the joint Newton refuses the stage-1 drift freeze,
    LARND_CHAIN_FREEZE_DRIFT=1). That flag only masks geometry updates (no effect on the point, the forward model or H_cc);
    the FR stage resumes from the same checkpoint with it at 0. Resubmitted with LARND_CHAIN_FREEZE_DRIFT=0 (make_jn_dump.py:
    an extra that the stage script exports now replaces it; without such an extra the built command is byte-identical),
    each chained after its chain's 20000 dump, with the chain's first A fit moved behind it (scontrol), <= 6 running.
    Readout scripts read_t1 / read_t2 / read_t3 / read_t4.py (+ uqt_common.py) written and dry-run on stand-in data.
  - **UQ TEST T1 RESULT (3 Nov 08:25; all 18 dumps COMPLETED 08:22, config check OK; UQ/T1_result.txt): the E1 quick
    check HAS POWER (by the rule; my expectation VACUOUS was wrong).** All 12 stage-1 / FR checkpoints move later by more
    than SD_train; 9 of 12 (0.75 >= 2/3) are flagged (max|z| 2.2-10.2); 1 of 6 end points at 20000 is flagged (EXP2, max|z|
    3.2; = the 1/6 bound). E1 sigma is the same at every checkpoint (median Ab 0.032, eField 0.00046, tran_diff 0.40,
    long_diff 0.83, lifetime 0.80 %), so the flags come from the offset, not from sigma. Descriptive: over 27 (checkpoint,
    parameter) moves > SD_train, the Newton step has the move's sign in 0.78 but only 0.29 of its size (median).
    Limits (post hoc, from the same table): the power is carried by tran_diff (10 moves: sign 10/10, own |z| > 3 5/10),
    Ab and lifetime; long_diff, the parameter whose scatter matters, is caught by its own z in 2 of 9 moves (sign 6/9;
    EXP2 @ 14000 moves +7.5 points later with z -1.4, unflagged). The 20000 part measures only the false-flag rate (no
    later move exists there); whether the end point is converged against the replica scatter is T2's question.

## 3 Nov 2026 — DATASZ: accuracy vs data volume and batch size (rules written 3 Nov 14:56, before any job of it was submitted)
Pierre, 3 Nov: "In traditional methods, the more data you add the better your fit gets. But here as we fit the tracks we
also add more nuisance parameters when we add tracks. So I would like to know how the fit accuracy evolves with the quantity
of data. As well, I would like to know if the batch size, independently of the number of tracks has an impact or not."
Work dir /sdf/scratch/neutrino/pgranger/review_2026_09/DATASZ/.
- **What production actually loads (probe_multi.py, CPU, data_seed 3):** --data_sz 600 selects 600 events = 1364 tracks,
  620 m, but --max_nbatch 100 x --max_batch_len 400 keeps the first 100 batches = **889 tracks, 399 m**: the batch cap
  binds, not data_sz. Pool after selection: 93 533 events, 216 091 tracks, 99.7 km (samples are 0.4 % of it ->
  independent). Longest track 146 cm (no track dropped for L >= 200). Nested: 25 / 50 batches = 214 / 425 tracks of the
  same sample. 50 x 800 cm = the same 889 tracks (39 914 cm); 200 x 200 cm = 924 tracks, 41 464 cm (the 889 + 35, +3.9 %:
  packing by floor(cumsum / L) reaches 200 splits later, multi-track events span several 200 cm units).
- **What depends on the batch partition (code read):** data NLL, split-t prior, soft barrier, MCS prior = sums over
  pixels / segments / tracks -> invariant. The dE/dx mean constraint is PER BATCH, w (m_b - 1.887)^2; its honest weight is
  1/(2 SD^2) of the true batch mean, SD ~ L^-1/2 (dedx_mean_constraint.md), so it is scaled as w = 2500 L/400 to keep the
  same prior. The calibration gradient clip (norm 100, active on 98-99 % of iterations, 12 Oct) gives each batch weight
  1/|g_b|: the stationary point itself depends on the partition. That is intrinsic to batch size and is part of what T6
  measures. Schedules: calibration LR decays per EPOCH (transition_steps = epoch size), geometry LR per ITERATION, dE/dx and
  geometry Adam step once per visit (lr_schedule_epoch_trap.md).
- **Design: every schedule element held fixed IN EPOCHS (make_ds.py).** r = N_b/100: all iteration indices x r (stage ends
  10000/14000/20000, coarse release 7500, warmups 500, transition_begin 11500/16000, dE/dx freeze, chain decay starts, save
  frequencies), per-iteration decays ^(1/r) (chain 0.9997, 0.9998). Each stage is built at job start (run_ds.sbatch). At r = 1,
  L = 400 the three builders reproduce the production commands byte-for-byte (fit vs KNOT/inner_sci_full_EXP3C75.sh; cont /
  fast vs CONT/make_prod_cont.py / make_prod_fast.py on EXP3C75 @ 10000 / EXP3C75FR @ 14000); scaled builds checked by diff
  on stand-in checkpoints; a checkpoint at the wrong iteration is refused. Recipe = the UQ test's (production C75 on
  REALISTIC ends, 1 mm independent smear, key entry), seeds 17-24 (as Set A), same tree as Set A (no source change).
- **Arms (n = 8 each; error = mean over the last 10 epochs = 1000 r iterations / truth - 1 at 20000 r; pos = mean
  pos_residual over the same window):**

  | arm | batches x cm | track | data_sz | r | mean-constraint w | est. GPU-h / pipeline |
  |---|---|---|---|---|---|---|
  | V025 | 25 x 400 | ~10 m, ~215 tracks (nested in Set A s) | 600 | 0.25 | 2500 | ~1.2 |
  | V050 | 50 x 400 | ~20 m, ~430 tracks (nested) | 600 | 0.5 | 2500 | ~2.0 |
  | V100 | 100 x 400 | = Set A of the UQ test, EXP17-28C75H (n = 12) | 600 | 1 | 2500 | (running) |
  | V200 | 200 x 400 | ~80 m, ~1780 tracks (fresh events) | 1200 | 2 | 2500 | ~6.2 |
  | B800 | 50 x 800 | the Set A tracks of sample s | 600 | 0.5 | 5000 | ~3.0 |
  | B200 | 200 x 200 | the Set A tracks + 3.9 % | 600 | 2 | 1250 | ~4.0 |

  Plus calibration-only E1 dumps (UQ/run_dump.sbatch, LARND_JOINT_NEWTON_DUMP_CALIB=1) of 3 end points per V arm (V025 /
  V050 s = 17, 19, 21; V200 s = 17, 18, 19); V100 uses Set D. Dumps built by DATASZ/make_ds_dump.py = UQ/make_jn_dump.py with
  --iterations N_b + 1 (the assembly takes one full epoch of the run: 101 would never finish one at 200 batches);
  identical output at N_b = 100 (checked on EXP1C75H). ~140 GPU-h, as 6 dependency chains that start only when the
  UQ test chains end (afterany on their last job), V arms first, B arms after: <= 6 running.
- **T5 (data volume) rules (read_ds.py t5; read only when all V-arm jobs and dumps are COMPLETED, config check first:
  data_seed, max_nbatch, max_batch_len, data_sz, w, stage-1 iterations, coarse release, smear from the pickle; batch count
  and track length from the stage-1 log; FR / H iteration counts from their inners):**
  - Spread exponent alpha_p: SD_p(N) ~ N^-alpha, weighted LS of ln SD on ln N_b over the 4 arms (var ln SD = 1/(2(n-1))),
    95 % CI = +- 1.96 SE (expected SE ~ 0.17). STATISTICAL if the CI contains 0.5 and not 0; FLOOR if it contains 0 and not
    0.5; UNRESOLVED if both; SLOWER THAN STATISTICS if inside (0, 0.5) excluding both; FASTER THAN STATISTICS if above 0.5;
    WORSE WITH DATA if below 0; suffix NOT A POWER LAW if the fit chi2 (2 dof) has p < 0.01. (V025 / V050 share data with
    V100 for s = 17-24; the shared part makes the arm SDs positively correlated, so the CI is conservative.)
  - Bias: BIAS CONSTANT IN N if the 4 arm means agree with their weighted mean (chi2, 3 dof, p > 0.05), else BIAS CHANGES
    WITH N (slope vs ln N reported).
  - Statistical floor: the median sigma_E1 per arm must fall as N^-1/2 (exponent within 0.50 +- 0.05: FLOOR SCALES AS
    STATISTICS); k(N) = SD / sigma_E1 reported per arm (k constant <=> the spread is statistics-like).
  - Descriptive: RMS error per arm; position per arm.
  - Predictions: sigma_E1 exponent 0.5 (sanity). Ab and tran_diff STATISTICAL (data-limited in the 31 Oct error budget).
    eField, long_diff, lifetime: no firm prediction, because their spread is mostly optimisation scatter (sigma_opt ~ SD at
    N = 100), which need not average down. Weak expectation: long_diff and lifetime between FLOOR and STATISTICAL. Biases
    CONSTANT IN N (an incidental-parameter bias is O(1) in the number of tracks: each track's nuisances are estimated from
    that track alone, Neyman-Scott). Position constant (same per-track information).
- **T6 (batch size at fixed tracks) rules (read_ds.py t6; read when all B-arm jobs are COMPLETED, same config check plus
  track length vs the paired reference):** reference = EXP17-24C75H (Set A), paired by sample. Per parameter and arm:
  shift if |mean paired delta| > 2 SE; spread change if SD_B / SD_ref is outside [0.45, 2.23] (95 % of F(7,7)). An arm has
  NO DETECTED IMPACT if no parameter shifts or changes spread; the batch size has NO DETECTED IMPACT if both arms pass.
  The smallest detectable shift (2 SE, from the paired SD) is printed. Expected 2 SE ~ 4 points on long_diff, ~3.5 on lifetime,
  i.e. this test resolves only large effects. Position paired delta: descriptive. Prediction: NO DETECTED IMPACT. A small
  eField shift is possible (clip weighting).
- Not in this test: replicas per arm (sigma_opt vs N), N above 200 batches (~13 h per pipeline).
- Submitted 3 Nov 14:56 (DATASZ/submit_dsall.py; 129 jobs, ids DATASZ/ds_jobs.txt; chain c after UQ chain c's last job;
  est. 21-24 GPU-h per chain). Probe sets (probe_sets.npz, seed 3), checked after submission, before any result: 25 / 50 x
  400 cm are strict subsets of the production 889 tracks; 50 x 800 cm is the identical set; 200 x 200 cm = all 889 + 35.
  For seed 17, production loads 872 tracks, 39 820 cm. 400 x 400 cm (data_sz 2400) would be 3446 tracks, 1.6 km (not run).
- **UQ TEST T3 RESULT (3 Nov 15:15; all 36 Set-A jobs COMPLETED 15:10, config check OK on 12; UQ/T3_result.txt): R3 NOT
  CONFIRMED -- every parameter passes its own band, but the pooled coverage FAILS at 1 sigma.** SD_test / SD_train: Ab 1.33,
  eField 1.73, tran_diff 1.04, long_diff 1.70 (6.90 vs 4.06), lifetime 1.33, all inside [0.54, 1.86] (CALIBRATED) but all
  > 1. Biases transfer on all 5 (mean_test - bias_train: Ab -0.11, eField -0.00, tran_diff -0.97, long_diff +1.79,
  lifetime +1.99; all < 0.82 SD_train). Pooled pulls: |q| < 1 0.47 (pass 0.50-0.82) FAIL, |q| < 2 0.85 (>= 0.83) pass.
  The current metric (sigma_E1 alone) gives pull SDs 6.5 / 136 / 3.3 / 8.3 / 6.5 (|q| < 1 in 0.15); E1 (+) the pre-C75
  sigma_opt still 2.6-8.6. Post hoc (posthoc_t3.py, descriptive): 5 of 12 test long_diff values are >= +9.3 (training max
  +8.0); eField s17 +0.266 (training max +0.204); corr(long_diff, lifetime) +0.57 test vs +0.08 train; first vs second
  pipeline of each chain long_diff +1.6 vs +7.6 (~1.5 sigma, nodes mixed, no shared state between jobs: chance assumed;
  T2's replicas of s = 17-24 will show how much is optimisation scatter). Pooled n = 24 (train + test): bias Ab -0.04,
  eField +0.129, tran_diff +1.12, long_diff +3.72, lifetime +1.84; SD 0.183, 0.049, 1.37, 5.61, 4.55. Reading: a 12-sample
  SD is itself uncertain by ~20 % and here underestimated the spread on every parameter; R3 as stated (n = 12) under-covers.
- **UQ TEST T2 RESULT (3 Nov 21:45; all 32 Set-R jobs COMPLETED 21:35, config check OK on 8 x 3; UQ/T2_result.txt): R2
  classes Ab NOT converged (rho 0.84), eField converged (0.38), tran_diff converged (0.48), long_diff NOT converged (0.70),
  lifetime converged (0.40); one pair does NOT suffice (as expected); H-ONLY replica SUFFICES (expected INSUFFICIENT: my
  expectation was wrong).** sigma_opt (8 full pairs) 0.127 / 0.013 / 0.616 / 2.84 / 1.54 vs SD_train 0.152 / 0.035 /
  1.285 / 4.064 / 3.866. Against the pre-C75 true-end classes: tran_diff, long_diff the same; eField now converged (was
  not), Ab now NOT (was borderline), lifetime converged (was borderline). Single-pair agreement 0.50 / 0.88 / 0.75 / 0.38 /
  0.88 (< 7/8 on four). sigma_opt^H (8 H-only pairs) / sigma_opt: Ab 0.94, long_diff 1.08 (both >= 0.7 -> SUFFICES); also
  eField 0.42, tran_diff 0.50, lifetime 2.10 (descriptive). Check (not in the rule): every H2 history equals its A history
  bit for bit through 14000 and first differs at 14001-14088; the full replicas differ from iteration 2-63.
  Deviation: the rule said "bootstrap" for k; read_t2.py enumerates all C(8, k) subsets (without replacement) and reports
  k = 8, which is trivial. Per k (T2_posthoc.txt): tran_diff stays at ~0.50 for every k because its rho (0.48) sits on the
  0.5 threshold; without it, k = 5 reaches >= 80 % (Ab 1.00, eField 1.00, long_diff 0.89, lifetime 1.00).
  Post hoc, descriptive (scratchpad t2_posthoc.py -> UQ/T2_posthoc.txt; 1000-iteration window means ending at t):
  - Full-replica spread long_diff 12.0 at 10000, 10.8 at 14000 (FR end), 7.3 at 16000, 3.4 at 18000, 2.8 at 20000; Ab
    0.15 -> 0.10 -> 0.13; lifetime 6.1 -> 5.6 -> 1.5. The H stage contracts the FR-end differences by ~4x and forgets them.
  - The H-only replicas (same state at 14000) separate by 3.0 in long_diff by 16000 and then stay there (3.3, 3.0, 3.05
    at 18000-20000); Ab grows to 0.12 by 19000. The scatter is created during the H stage, before and while the
    calibration LR anneals (transition_begin 16000), and the annealing freezes it.
  - The end points are still: within the last 1000 iterations the instantaneous value wanders by 0.02 (long_diff), and the
    window mean moves by 0.18 between 19000 and 20000, against a replica spread of 2.8-3.0. A flat trajectory at the end
    is no evidence of convergence here.
  - H2 - A mean: long_diff +1.57 (SE 1.53), lifetime +2.59 (1.62); RH - A +2.15 (1.42), +0.76 (0.77): no shift (< 2 SE).
  Reading: on this recipe the end point is roughly f(data) + a random offset drawn in the H stage, whose SD is sigma_opt.
  sigma_opt can be measured from H-only re-runs (6000 iterations from a shared FR checkpoint, about 1/3 of a pipeline),
  but not per fit from one pair. It is half or more of the spread for Ab (0.127 of 0.183, pooled n = 24 SD) and long_diff
  (2.84 of 5.61). If those offsets are independent between H-only replicas, as the H-only/full ratio of ~1 suggests,
  averaging m H-only replicas would divide sigma_opt by sqrt(m). That is a hypothesis, not tested here.
- **UQ TEST T4 RESULT (3 Nov 22:10; all 12 Set-D dumps COMPLETED 22:00, config check OK; UQ/T4_result.txt): R4 NOT
  CONFIRMED. Every parameter is CALIBRATED, but the pooled coverage FAILS.** k = SD_train / median sigma_E1 (6 training end
  points) = Ab 4.90, eField 78.8, tran_diff 3.18, long_diff 4.89, lifetime 4.93. SD(r): 1.34 / 1.71 / 1.11 / 1.71 / 1.28,
  all in [0.54, 1.86]. Pooled |q| < 1 0.43 (pass 0.50-0.82) FAIL, |q| < 2 0.82 (>= 0.83) FAIL. CV of sigma_E1 over the 12
  test fits: 2.3 / 3.6 / 20.2 / 7.1 / 6.0 %. So the "< 10 % everywhere" clause does not apply (tran_diff), and E1 varies per
  fit. Post hoc, descriptive (scratchpad t4_posthoc.py -> UQ/T4_posthoc.txt): that variation does not track the error.
  Spearman(sigma_E1, |e - bias|) is +0.18 / +0.18 / +0.10 / -0.27 / +0.37 (n = 12, SE ~0.3). k x sigma_E1 is therefore,
  in effect, SD_train, with T3's under-coverage. E1 at the fresh end points: max |z| > 3 in 5 of 12 (Ab, long_diff, Ab,
  lifetime, long_diff), against 1 of 6 on training. Spearman(|z|, |A - B| replica difference) is -0.05 to +0.31 (n = 8).
- **UQ TEST, overall reading (3 Nov 22:10; replaces the 31 Oct R1-R4 on this recipe; to Pierre):**
  - (R1) E1 sigma is a floor only, 3-136x below the spread. The z quick check is a useful ALARM in mid-fit (T1: 9 of 12
    moving checkpoints flagged), but through tran_diff, Ab and lifetime, not long_diff. At the end point it is no
    certificate: 6 of 18 end points flagged, and |z| does not track the replica distance.
  - (R2) Per-fit convergence cannot be read from one replica pair. With >= 5 pairs, the recipe is optimiser-limited on Ab
    and long_diff (rho 0.84, 0.70) and converged on eField and lifetime; tran_diff sits on the threshold. H-stage-only
    replicas measure sigma_opt as well as full ones, at ~1/3 of the cost.
  - (R3) The ensemble SD + bias is the calibrated uncertainty, but n = 12 was too few (it came out low on all 5). Use
    the pooled n = 24 with the prediction factor sqrt(1 + 1/24): sigma = 0.187 / 0.050 / 1.40 / 5.73 / 4.65 %, bias
    -0.04 / +0.13 / +1.12 / +3.72 / +1.84 %. Not yet tested on fresh samples; that would take 12 more fresh pipelines,
    ~45 GPU-h, not launched.
  - (R4) Drop k x sigma_E1 within one configuration: it equals R3 and adds no per-fit information. Its remaining possible
    use, carrying the SD across configurations, is measured by DATASZ (sigma_E1 exponent, k(N)).
  - Open, untested: averaging m H-only replicas to divide sigma_opt (half or more of Ab and long_diff spread) by sqrt(m).

## 4 Nov 2026 — FRESH TEST: the pooled-24 uncertainty on unseen samples, and averaging H-stage re-runs (rules written 4 Nov 03:04, before any job of it was submitted)

Pierre, 4 Nov: "Please do what you proposed" (proposals 1 + 2 of the UQ TEST overall reading).
- Jobs (UQ/submit_fresh.sh, ids UQ/fresh_jobs.txt): 12 FRESH samples, data_seed 29-40 (no script has ever used them;
  grep of every .sh / .sbatch / .py / .txt / .md under the repo and review_2026_09), production pipeline EXP<s>C75 -> A =
  EXP<s>C75H @ 20000. Inner = inner_sci_full_EXP3C75.sh with only --data_seed and --test_name changed (diff checked). Plus three
  H-stage-only re-runs per sample, EXP<s>C75H2 / H3 / H4 (run_knot.sbatch prodfast again from the same EXP<s>C75FR @
  14000, the T2 construction). 72 jobs, ~72 GPU-h (stage means from the UQ test: fit 1.52, FR 0.70, H 0.94 h), appended
  to the 6 DATASZ chains (afterany on each chain's tail; H re-runs afterok on their FR), so they start after DATASZ and
  <= 6 run at once. No source change since the Set A runs (find: no .py under src/ optimize/ newer than 3 Nov 06:55).
- Reference, frozen (UQ/ref24.py -> UQ/ref24.txt, computed now): the 24 single-fit end points EXP1-12C75H + EXP17-28C75H:

  | | Ab | eField | tran_diff | long_diff | lifetime |
  |---|---|---|---|---|---|
  | bias24 | -0.0403 | +0.1289 | +1.1161 | +3.7209 | +1.8440 |
  | sd24 | 0.1833 | 0.0485 | 1.3728 | 5.6139 | 4.5539 |

- **F1 (is the pooled-24 uncertainty calibrated on unseen single fits?).** On the 12 A end points, pulls q = (e - bias24)
  / (sd24 sqrt(1 + 1/24)) ~ t_23. Per parameter CALIBRATED if SD_fresh / sd24 is in [0.56, 1.62] (95 % of F(11, 23));
  BIAS TRANSFERS if |mean_fresh - bias24| < 0.707 sd24 (= 2 sd24 sqrt(1/12 + 1/24)). Pooled over the 60 pulls: |q| < 1
  expected 0.67, |q| < 2 expected 0.94; PASS if the first is in [0.50, 0.82] and the second >= 0.83 (the T3 bands).
  CONFIRMED if every parameter is CALIBRATED and the pooled coverage PASSES. Expected: CONFIRMED (a 24-sample SD is
  uncertain by ~15 %; failing again would mean the spread is not one stable number, e.g. heavy tails in long_diff).
- **F2 (does averaging the 4 H-stage end points of a sample shrink the spread?).** Per parameter the within-sample SD
  sigma_w over the 12 samples x 4 end points (36 dof; 95 % CI from chi2_36). The implied spread reduction of the 4-run
  average is g = 1 - sqrt(1 - (3/4) sigma_w^2 / sd24^2), exact in expectation because the 4 runs of a sample are
  exchangeable by construction. USEFUL if g >= 10 % at the lower CI bound of sigma_w, NO GAIN if g < 10 % at the upper
  bound, MARGINAL otherwise (10 %: the average costs ~+90 % GPU per fit, 3 more H stages; whether that is worth it is
  Pierre's call). Required check: every H2/H3/H4 history equals A's bit for bit through iteration 14000; otherwise no
  readout. Expected, from T2's sigma_opt^H (0.119 / 0.006 / 0.31 / 3.05 / 3.24): g Ab ~17 % (USEFUL or MARGINAL),
  long_diff ~11 % (MARGINAL), eField and tran_diff NO GAIN; lifetime no firm prediction (T2 H-only 3.24 implies ~20 %,
  T2 full-replica 1.54 only ~4 %). Descriptive only: SD(average) / SD(A) over the 12 samples with a paired bootstrap
  (n = 12 is too few to decide from it), and the coverage of the averages with sigma_avg = sqrt(sd24^2 - (3/4) sigma_w^2).
- Reading order: F1 when all 36 pipeline jobs are COMPLETED; F2 when those and all 36 H re-runs are. Config check (seed,
  coarse 7500, smear 1.0 / entry) in UQ/read_f.py; dry-run OK on stand-ins (Set A for F1; s = 17-24 with H, H2, RH, H2).
- Submitted 4 Nov 03:04:54 (UQ/fresh_jobs.txt): 12 pipelines + 36 H re-runs, 2 pipelines + 6 re-runs per DATASZ chain
  (afterany on its tail; re-runs afterok on their FR; dependencies checked with scontrol). 6 running, unchanged.
  Waiters: wait_set.sh F1 (setF1_ids.txt, 36) and F2 (+ setF2_ids.txt, 72); watch_health.sh on all 72. Expected: F1 ~5 Nov
  04:00, F2 ~5 Nov 10:00 (DATASZ ends ~4 Nov 21:00).
- **DATASZ T5 RESULT (4 Nov 09:30; all 81 V-arm jobs and 9 dumps COMPLETED 09:17, config check OK on 36 end points;
  DATASZ/T5_result.txt, figure DATASZ/fig_t5.png): the spread falls like statistics for Ab, eField, long_diff and lifetime;
  tran_diff UNRESOLVED; biases CONSTANT IN N except Ab; the E1 floor scales exactly as statistics.**

  | | Ab | eField | tran_diff | long_diff | lifetime |
  |---|---|---|---|---|---|
  | SD 100 / 199 / 400 / 799 m | 0.336 / 0.119 / 0.202 / 0.077 | 0.088 / 0.083 / 0.061 / 0.022 | 2.57 / 1.39 / 1.34 / 1.56 | 17.3 / 7.9 / 6.9 / 4.2 | 10.0 / 5.1 / 5.1 / 2.2 |
  | alpha [95 %] | 0.53 [0.19, 0.86] | 0.63 [0.30, 0.96] | 0.23 [-0.10, 0.56] | 0.63 [0.29, 0.96] | 0.63 [0.29, 0.96] |
  | class | STATISTICAL | STATISTICAL | UNRESOLVED | STATISTICAL | STATISTICAL |
  | bias 100 / 199 / 400 / 799 m | -0.08 / -0.07 / -0.09 / +0.09 | +0.14 / +0.12 / +0.13 / +0.13 | +1.56 / +0.67 / +0.63 / +1.19 | +6.0 / +3.6 / +4.6 / +5.5 | +5.7 / +4.3 / +2.8 / +0.1 |
  | bias test (chi2 3 dof) | CHANGES, p 0.001 | CONSTANT, p 0.98 | CONSTANT, p 0.70 | CONSTANT, p 0.94 | CONSTANT, p 0.053 |

  sigma_E1 exponent 0.495 / 0.483 / 0.495 / 0.498 / 0.500 -> FLOOR SCALES AS STATISTICS (the sanity prediction).
  k(N) = SD / sigma_E1 for 100 / 199 / 400 / 799 m: Ab 5.6 / 2.7 / 6.6 / 3.6; eField 103 / 135 / 137 / 69; tran_diff 3.3 /
  2.7 / 3.4 / 5.7; long_diff 10.8 / 7.1 / 8.5 / 7.5; lifetime 6.2 / 4.7 / 6.4 / 3.9. Constant within a factor ~2; an n = 8
  SD is itself uncertain by ~25 %. Position (descriptive, not paired: V200 is other events) 869 / 862 / 864 / 838 um.
  Against the predictions: E1 0.5 right. Ab STATISTICAL right. tran_diff predicted STATISTICAL: UNRESOLVED (its spread is
  flat from 199 m on). eField, long_diff, lifetime: weak expectation "between FLOOR and STATISTICAL" wrong; all three
  STATISTICAL, though each CI also allows anything from 0.3 to 0.96. So the optimisation scatter, half of the long_diff
  spread at 400 m (UQ T2), must shrink with data as well. Biases constant: right for eField, tran_diff, long_diff;
  lifetime constant only by the rule (p 0.053; the mean falls monotonically, V200 - V100 = -2.76 +- 1.68); Ab wrong (CHANGES:
  V200 +0.093 against -0.07 to -0.09 below, shift +0.19 +- 0.06, i.e. 0.2 % in absolute terms). V025 / V050 / V100 share
  tracks (nested), so the chi2 with independent errors is conservative.
  Reading for Pierre's question: adding tracks (and with them per-track nuisances) does NOT stop the statistical gain. Up
  to 1770 tracks / 800 m the spread of 4 of 5 parameters falls as ~1/sqrt(N). The biases of eField (+0.13), tran_diff
  (+0.8) and long_diff (+5.0) do not fall, so the error floors at them. At 800 m the bias already exceeds the spread on
  eField (0.13 vs 0.02) and long_diff (5.5 vs 4.2); for tran_diff they are comparable (1.2 vs 1.6). An O(1) bias is
  what a per-track-nuisance (Neyman-Scott) effect predicts, but any other recipe systematic would look the same; this test
  cannot tell them apart.

## 4 Nov 2026 — where the production bias sits (post hoc, descriptive, existing runs only; UQ/bias_diag.py)
Pierre, 4 Nov: "How to work on the bias? Do we have ways forward?"
- (a) E1 conditional Newton step d at the 20000 end points (d = step to the calibration minimum with geometry and dE/dx
  held where the fit left them), mean +- SEM: Set D (EXP17-28, n = 12) long_diff -1.32 +- 0.30, lifetime -1.41 +- 0.25,
  Ab +0.05 +- 0.01, eField -0.00, tran_diff -0.06 +- 0.08. T1 20k (EXP1-6, n = 6): long_diff -0.57 +- 0.52, lifetime -1.05
  +- 0.38. DATASZ dumps: long_diff -1.5 to -2.0, lifetime -1.5 to -2.5 at every N. The end points sit systematically
  above the conditional minimum in long_diff and lifetime by ~1.3-1.4 points. Pooled over the 18 fits, the conditional
  minimum is Ab 0.00, eField +0.13, tran_diff +0.96, long_diff +2.43, lifetime +0.30.
  Two candidate causes, not separated: the calibration clip (stationary point of sum g_b/|g_b|, REVIEW 12-13 Oct: CLIPOFF
  moved long_diff -0.74, lifetime -0.48 on the old recipe, PARTIAL), or an annealed optimiser lagging from one side (T2:
  the end point freezes).
- (b) True vs realistic ends, same 12 samples (KN10N<s>C75H vs EXP<s>C75H, data_seed checked): true-end bias Ab +0.28 +-
  0.05, eField +0.01 +- 0.00, tran_diff +0.88 +- 0.45, long_diff +3.23 +- 0.62, lifetime -0.23 +- 0.95; realistic - true
  paired: Ab -0.26 +- 0.05, eField +0.12 +- 0.01, tran_diff +0.72 +- 0.40, long_diff -0.40 +- 1.26, lifetime +1.08 +- 0.92.
  The eField bias is entirely the 1 mm end smearing; the long_diff bias is not (present on true ends).
- Reading: long_diff +3.7 ~ 1.3 (end point short of the conditional minimum) + ~2.4 (at the conditional minimum, i.e.
  carried by the nuisances or the objective; not end smearing). lifetime +1.8 ~ 1.4 + ~0.3. eField +0.13 = end smearing.
  tran_diff +1.1 ~ +0.9 already on true ends + ~0.7 from smearing, at its conditional minimum.

## 4 Nov 2026 — NOCLIP: is the end-point gap to the conditional minimum the calibration clip? (rules written 4 Nov 10:40, before any job of it was submitted)
Pierre, 4 Nov: "Please proceed as you proposed" (item 1 of the bias plan above).
- Arm: EXP<s>C75HNC, s = 17-28 = the production H stage (CONT/make_prod_fast.py, unchanged) from the same EXP<s>C75FR @
  14000 as A = EXP<s>C75H, with --max_clip_norm_val 100 -> 1e12 and LARND_CALIB_ADAM_RESET=1. Without the reset, the
  restored Adam nu holds clipped squared gradients and the first raw updates would be far larger than the LR. Inners
  pre-built by NOCLIP/make_noclip.py; diff against KNOT/inner_sci_full_EXP<s>C75H.sh = only test_name, the clip value
  and the reset export (checked on all 12). The clip acts only on the calibration optimiser (the dE/dx and geometry Adam
  have none). Plus one calibration-only E1 dump per end point, D_NC<s> (UQ/run_dump.sbatch, LARND_JOINT_NEWTON_DUMP_CALIB=1,
  LARND_CALIB_ADAM_RESET=0 replacing the export; Adam is held in a dump, so the clip value cannot enter it), i.e. the
  construction of the Set D dumps D_EXP<s> of the A end points. 24 jobs, ~16 GPU-h. Placed on the 6 chains after DATASZ
  and AHEAD of the FRESH TEST (whose chain heads are moved behind them; FRESH shifts by ~2.7 h).
- Hypotheses: (clip) the end point is the stationary point of sum_b g_b / |g_b| (clip active on 98-99 % of iterations),
  so with the clip off the Newton step d of the unclipped objective goes to ~0; (lag) the annealed optimiser stops short
  of the minimum from one side, so d stays at ~-1.3 / -1.4 whatever the clip.
- **Primary readout, per parameter p in {long_diff, lifetime}:** d_NC = mean over the 12 D_NC dumps (SEM s_NC); paired
  change Dd = d(D_NC<s>) - d(D_EXP<s>) (mean, SE).
  - CLIP IS THE CAUSE: |d_NC| < 2 s_NC and Dd toward zero by > 2 SE.
  - LAG: |Dd| < 2 SE and |d_NC| > 2 s_NC.
  - PARTIAL otherwise; report the fraction of the gap closed, Dd / (-d_A).
  Expected: CLIP or PARTIAL. October's no-clip continuation (2000 iterations, old recipe) moved long_diff -0.74 and
  lifetime -0.48, against today's gap of -1.3 / -1.4.
- Secondary, descriptive: end-point shift e(HNC) - e(A) and e(HNC) - mean(A, H2) (s = 17-24), with SE; RMS error vs truth;
  position; Ab, eField, tran_diff d and shifts. Sanity: every HNC history equals A's bit for bit through 14000; the job
  log shows the clip at 1e12 and the Adam-reset line (pickle provenance is stage 1 only); no parameter at a bound, no
  NaN. A failed or diverged HNC run is reported, not replaced.
- If CLIP: the fix is a production change (clip off at least in the H stage), so Pierre decides. If LAG: a longer or
  slower final anneal, tested separately.
- Reading order: when all 12 HNC runs and 12 dumps are COMPLETED. Readout NOCLIP/read_nc.py (dry run before submission).
- Submitted 4 Nov 10:42:30 (NOCLIP/submit_nc.sh, ids NOCLIP/nc_jobs.txt, setNC_ids.txt): 12 HNC + 12 dumps, 2 + 2 per chain
  after the DATASZ tails; the 6 FRESH chain heads moved behind them (scontrol; checked). Queue width still 6 (all 141 queued
  jobs). Waiter wait_set.sh NC, watch_health.sh on the 24. Readout NOCLIP/read_nc.py, dry-run OK (log check fixed
  before submission: the clip line prints 1000000000000.0).

## 4 Nov 2026 — bias item 3: is the 1 mm endpoint smear realistic? (descriptive, offline, TRUTHC/endpoint_res.py)
On the TARGET hits of the true-end fits KN10N1-4C75H: single-track events only (every hit belongs to the track), 415
tracks / 830 endpoints, median 41 cm. Estimate = charge x hit_prob weighted PCA line through (pixel_x, pixel_y, pixel_z),
endpoint at the extreme hit projection; error against the true first / last point (TRUTHC/endpoint_res_out.txt):
- one line through all hits: along-track offset -1.39 mm (inward) with SD 1.11; transverse RMS 1.29 mm (2.05 for tracks
  > 60 cm: MCS curvature); per coordinate RMS 1.45 / 1.17 / 1.16 mm (x, y, z).
- local line fit to the last 10 cm at each end: along-track offset -1.38 mm, SD 1.11; transverse RMS 0.77 mm (0.73-0.81 at
  every length); per coordinate RMS 1.31 / 1.00 / 1.02 mm including the offset. 5 cm: transverse RMS 0.93 mm.
- Reading: the spread of a simple hit-based estimate equals production's 1 mm per coordinate. The inward offset of 1.4
  mm (the last hit sits inside the true end) is systematic and calibratable on simulation. After that correction, the
  1 mm independent smear is a fair stand-in, and the end-smearing biases measured with it (eField +0.12, Ab -0.26,
  tran_diff +0.7, lifetime +1.1 n.s.) apply to a realistic estimate. Not covered: t0 uncertainty (a coherent z shift of
  both ends; t0 = 0 is known in simulation), multi-track events, noise and delta-ray hits on data. No production rerun
  needed for item 3. The fix for these biases stays endpoint fitting (LARND_FIT_ENDPOINTS, built 12 Oct) or a
  simulation correction.

## 4 Nov 2026 — TRUTHC: is the residual at the conditional minimum carried by the nuisances or by the objective? (rules written 4 Nov, before the 14:23:50 submission of the set)
Pierre, 4 Nov: "Please proceed as you proposed" (item 2 of the bias plan). After the Newton step, the true-end production
end points keep a residual at the conditional minimum m = e + d: long_diff ~+2.4, tran_diff ~+1.0 (bias anatomy above).
Question: does that residual vanish when the nuisances (dE/dx, geometry) are put at truth, or is it in the objective itself?
- Instrument: the LARND_TRUTHPOINT diagnostic (fresh run, evaluation only, no fitting step, one pass over the 100 batches)
  at the end point KN10N<s>C75H @ 20000, s = 1-8 (true ends). Points: P0 fit (calibration, dE/dx and geometry of the fit),
  P1 truth (all three at truth; geometry = truth projected on the 10 cm spline), P2 calibration at truth (nuisances of
  the fit), P3 dE/dx at truth, P4 geometry at truth. At every point and batch: the 5 x 5 calibration Hessian H (central FD
  of the autodiff gradient in the normalised variables, steps 1e-3 and 3e-3) and the gradient g, summed over batches;
  d = 100 jac (-H^-1 g) % with jac = d ln phys / d norm (sigmoid); m = e + d.
- Code: a review copy TRUTHC/tree (rsync of the live src + optimize; fit_result -> the live one), with 26 added lines in
  optimize/fit_params.py: the truth-point Hessian block also stores g (one compute_loss with_grad at the point) and jac
  (new method _tp_dlnphys_dnorm). Nothing removed; the live tree is untouched (fit_params.py md5 unchanged). Inners
  TRUTHC/inner/inner_TC_<RUN>.sh built by TRUTHC/make_tc.py from the stage inner KNOT/inner_sci_full_<RUN>C75H.sh: diff =
  cd to the copy, LARND_TRUTHPOINT, LARND_TP_HESS, LARND_KEEP_TARGETS='' and the argv (--resume_from removed,
  --iterations 100, --test_name, --save_freq) only (checked on all 9).
- Pilots: CPU, one batch, P0 + P1 at 1e-3 (TCSMOKE): g and jac stored, |H - H^T| / max|H| 3e-5 / 1e-4, H positive
  definite. GPU, 3 batches of KN10N1, all points, both steps (TCGPU, 43160543, 5 min): ~9 s per point per batch,
  so ~1.7 h per sample; asymmetry 1e-5 to 4e-3, positive definite everywhere. The pilot showed the raw step difference
  for P1 tran_diff would reach ~0.45 sigma of the 100-batch sigma, right at the edge of the V1 rule as first coded
  (|m(1e-3) - m(3e-3)| < 0.5 sigma). That difference is ~8 x the truncation error at 1e-3 (central FD is O(h^2);
  the asymmetry scales as h^2, so truncation, not round-off, dominates), so V1 is now stated on the Richardson
  estimate below. The pilot's 3-batch m values were printed by the check script but do not enter any verdict.
- Validation (all three must pass, else NO VERDICT and descriptive numbers only):
  - V1 step: |m(1e-3) - m(3e-3)| / 8 < 0.1 sigma (per-sample sigma = 100 jac sqrt(diag H^-1)) on >= 7/8 samples, for P0
    and P1, long_diff and tran_diff.
  - V2 instrument: TC_EXP17 (P0 only) against the Set D E1 dump D_EXP17 of the same end point: Hessian diagonal ratio
    within +-15 % on all 5 and |d_TC - d_E1| / sigma_E1 < 2.5 on all 5.
  - V3 quadratic: |m(P0) - m(P2)| < 2 sigma on >= 7/8 samples, long_diff and tran_diff.
- **Primary readout, per p in {long_diff, tran_diff}, step 1e-3:** m0 = mean m(P0), m1 = mean m(P1) over the 8 samples,
  paired difference D = m(P0) - m(P1) (mean, SEM).
  - CARRIED BY THE NUISANCES: |m1| < 2 SEM(m1) and D has the sign of m0 with |D| > 2 SE.
  - OBJECTIVE-LEVEL: |m1| >= 2 SEM(m1) with the sign of m0 and |D| < 2 SE.
  - MIXED otherwise.
  Expected: no firm prior. The September readiness review found the model at truth unbiased (+0.5 %, older setup),
  which points to NUISANCES; long_diff staying biased on true ends (+3.2) leaves OBJECTIVE-LEVEL or MIXED open.
- Secondary, descriptive: the share of D removed by geometry alone (P4) and by dE/dx alone (P3); Ab, eField, lifetime m;
  the loss differences truth - fit per batch (the existing SUMMARY line).
- If CARRIED BY THE NUISANCES: the bias is a nuisance-fit problem, so the levers are better geometry and dE/dx (endpoint
  fit, priors). If OBJECTIVE-LEVEL: the levers are the model or target (noise model, induction R089) or a simulation
  correction, not the optimiser.
- Cost and placement: 8 x ~1.7 h + EXP17 ~0.6 h ~ 15 GPU-h, 9 jobs, on the 6 chains after NOCLIP and AHEAD of the
  FRESH TEST (heads moved behind them; FRESH shifts by 1.7-3.4 h). Reading order: when all 9 are COMPLETED. Readout
  TRUTHC/read_tc.py (dry mode run on both pilots).
- Submitted 4 Nov 14:23:50 (TRUTHC/submit_tc.sh, ids TRUTHC/tc_jobs.txt, setTC_ids.txt): 9 jobs behind the 6 NOCLIP
  tails (chains 0-2 two each: KN10N1 + N7, N2 + N8, N3 + EXP17; chains 3-5 one each: N4, N5, N6); the 6 FRESH heads moved
  behind them (checked). Queue width still 6. Waiter wait_set.sh TC, watch_health.sh on the 9. The GPU pilot TCGPU had
  been slotted between B200s17H and B200s23 (afterany link) and finished in 5 min.

## 4 Nov 2026 — DATASZ T6 RESULT: batch size at fixed tracks (read 4 Nov ~21:05, all 48 B-arm jobs COMPLETED at 21:02; DATASZ/T6_result.txt)
read_ds.py t6, config check OK (seed, batch count / length, data_sz, w, iterations, coarse release, smear, FR / H counts).
Track length used: B800 39898 cm, B200 41644 cm (+4 %), reference EXP17-24C75H 39998 cm.

| arm | Ab | eField | tran_diff | long_diff | lifetime | position |
|---|---|---|---|---|---|---|
| B800 paired delta | -0.02 +- 0.07 | -0.01 +- 0.02 | +0.41 +- 0.56 | +1.84 +- 2.21 | +1.85 +- 1.81 | +3.8 +- 4.1 um |
| B800 SD ratio (F p) | 0.50 (0.09) | 0.56 (0.15) | 1.19 (0.66) | 1.16 (0.71) | 0.94 (0.88) | |
| B200 paired delta | +0.06 +- 0.05 | -0.01 +- 0.02 | +0.07 +- 0.37 | +0.13 +- 1.72 | +0.11 +- 1.21 | -9.8 +- 2.9 um |
| B200 SD ratio (F p) | **0.44 (0.047)** | 0.79 (0.55) | 0.69 (0.35) | 0.57 (0.17) | 0.45 (0.054) | |

- **Verdict by the rule: batch size HAS AN IMPACT, flag = B200 Ab spread (ratio 0.44, band [0.45, 2.23]).** B800: NO
  DETECTED IMPACT. No parameter shifts in either arm (smallest detectable shift: long_diff 3.4-4.4, lifetime 2.4-3.6).
  Prediction (NO DETECTED IMPACT) wrong by one flag, in the direction of a SMALLER spread.
- Reading (descriptive, post hoc): one flag at p = 0.047 among 10 spread tests is what chance gives, and B800 shows a
  similar Ab reduction (0.50), so the Ab flag alone is weak. But B200 is lower on 4 of 5 spreads and on every RMS error
  (long_diff 4.55 vs 7.17, lifetime 2.46 vs 4.80, Ab 0.089 vs 0.201), against the pooled n=24 SD too (Ab 0.51, tran_diff
  0.57, long_diff 0.75, lifetime 0.50), with the means unchanged and the position 9.8 +- 2.9 um better. B800 is not better
  (long_diff 1.16, means +1.8 n.s.). Biases do not move with the batch size in either direction.
- What differs at B200 with the schedule held in epochs: twice as many calibration Adam steps per epoch at the same per-
  epoch LR (dE/dx and geometry still one step per visit), half the data per step, twice as many clipped batch gradients.
  The T6 arms cannot separate these.
- GPU time per pipeline (median, sacct): reference 3.1 h, B200 3.9 h (+26 %), B800 2.8 h (-10 %).
- Hypothesis for a confirmation (not a result): 200 cm batches lower the spread of long_diff and lifetime at +26 % cost.
  Formed on these 8 samples, so it is tested on unseen ones (B200C below).

## 4 Nov 2026 — B200C: confirmation of the 200 cm batch hint on unseen samples (rules written 4 Nov, before the 21:12:04 submission)
From the T6 result above (hypothesis formed on s = 17-24, so not tested on them).
- Arm: B200s<s> (DATASZ/make_ds.py fit B200s<s> s 600 200 200 -> FR -> H, exactly the T6 B200 build: 200 x 200 cm, r = 2,
  w = 1250, every schedule element held in epochs), s = 29-40. Reference: the FRESH TEST production end points EXP<s>C75H
  (100 x 400 cm), paired by sample (same events up to the batch packing). Same live tree for both (fit_params.py md5
  24359dc42ba635b3458085bcc92b6e66 at submission; checked again at reading). 12 pipelines = 36 jobs, ~47 GPU-h, two per
  chain at the END of the 6 chains (after FRESH), so nothing already queued is delayed.
- **Primary, per p in {long_diff, lifetime}:** one-sided Pitman-Morgan test of var(B200) < var(ref) for paired samples
  (t = r sqrt((n - 2) / (1 - r^2)), r = corr(x - y, x + y), n - 2 = 10 dof); REDUCED if p < 0.05.
  - CONFIRMED: both REDUCED. PARTLY CONFIRMED: one. NOT CONFIRMED: neither.
  - SAFE: no parameter (all 5) has a paired shift > 2 SE that makes |mean error| larger. NOT SAFE otherwise.
  On the T6 samples (dry run of the readout, hypothesis-forming, not evidence) the same test gives long_diff p 0.033,
  lifetime p 0.006, with arm correlations 0.77 / 0.80.
- Secondary, descriptive: Ab, eField, tran_diff SD ratios with Pitman-Morgan and F p; RMS errors; position paired delta
  (one-sided p for < 0); GPU-h per pipeline; the 20-sample pool (T6 + B200C) flagged as containing the hypothesis data.
- Prediction: CONFIRMED and SAFE, position better.
- If CONFIRMED and SAFE: 200 cm batches with the epoch-matched schedule become a production candidate at +26 % GPU time
  (Pierre decides). The mechanism (twice the calibration steps per epoch vs half the data per step vs clip weighting) is a
  separate question; a 400 cm arm with two calibration steps per batch would separate the first.
- Reading order: when all 36 jobs are COMPLETED and FRESH F1 has been read. Readout DATASZ/read_b2c.py (dry run on the T6
  samples OK; it reproduces the T6 numbers).
- Submitted 4 Nov 21:12:04 (36 jobs; DATASZ/b2c_jobs.txt, setB2C_ids.txt; submit_ds.sh with DEP=afterany on the FRESH
  chain tails, chain c: s = 29 + c then 35 + c). Queue width still 6. Waiter wait_set.sh B2C, watch_health.sh on the 36.

## 4 Nov 2026 — NOCLIP RESULT (read 4 Nov ~23:45, all 24 jobs COMPLETED at 23:37; NOCLIP/NC_result.txt)
read_nc.py, config check OK (seed, coarse, smear / entry; every HNC history = A through 14000; no NaN; clip 1e12 and the
Adam reset line in every log).

| p | d_A | d_NC | Dd = d_NC - d_A | gap closed | class |
|---|---|---|---|---|---|
| long_diff | -1.32 +- 0.30 | -2.21 +- 0.20 | -0.89 +- 0.41 | -0.68 | PARTIAL (rule's "otherwise") |
| lifetime | -1.41 +- 0.25 | -2.00 +- 0.13 | -0.58 +- 0.30 | -0.41 | LAG |
| tran_diff / Ab / eField | -0.06 / +0.05 / -0.00 | -0.18 / +0.05 / -0.00 | | | descriptive |

- **Verdict by the rule: long_diff PARTIAL, lifetime LAG.** In substance the clip is NOT the cause: with the clip off the
  Newton step grew in magnitude on both (long_diff by 2.2 SE, lifetime by 1.9 SE); the "PARTIAL" label is the rule's
  catch-all branch, and the gap fraction closed is NEGATIVE (-0.68). Prediction (CLIP or PARTIAL, i.e. the gap shrinks)
  wrong.
- Descriptive: the clip-off end points sit higher (e(HNC) - e(A): long_diff +2.16 +- 0.94, lifetime +0.64 +- 0.76; mean
  error long_diff +4.61 -> +6.78, lifetime +2.84 -> +3.48), with smaller spread (SD long_diff 6.90 -> 4.23, lifetime 5.12
  -> 3.74; RMS 8.06 -> 7.89, 5.67 -> 4.99); position +1.6 +- 0.4 um.
- Confound: the arm needs the Adam reset at 14000 (else the first raw updates are far larger than the LR), so the arm is
  "clip off + Adam restart"; a restart kick with 6000 iterations to settle could itself push the end point. The test
  still answers its question: switching the clip off does not bring the end point to the conditional minimum.
- Consequence: production unchanged (the clip stays). The end point stays short of the conditional minimum by
  ~-1.3 / -1.4 (long_diff / lifetime) with or without the clip: an optimiser lag, not a property of the clipped objective.
- Post hoc, descriptive (NOCLIP + existing dumps; prompts the POLISH test below): taking the conditional minimum m = iterate
  20000 + calibration-only Newton step (the E1 dump) as the estimate, instead of the end point:

  | set (n) | long_diff RMS e -> m | lifetime RMS e -> m | tran_diff RMS e -> m | Ab RMS e -> m |
  |---|---|---|---|---|
  | A = EXP17-28C75H (12) | 8.06 -> 6.56 | 5.67 -> 4.54 | 1.43 -> 1.30 | 0.22 -> 0.17 |
  | HNC (12, same samples) | 7.89 -> 6.10 | 4.99 -> 3.81 | 1.42 -> 1.25 | 0.14 -> 0.12 |
  | T1 = EXP1-6C75H (6) | 4.16 -> 3.07 | 2.17 -> 2.81 | 2.38 -> 2.20 | 0.14 -> 0.14 |
  | DATASZ V025 / V050 / V200 (3 each) | 10.7 / 8.4 / 7.0 -> 9.0 / 6.2 / 5.8 | 6.8 / 6.7 / 2.4 -> 7.7 / 5.2 / 1.3 | 1.31 / 1.47 / 2.50 -> 1.01 / 0.74 / 2.08 | mixed |

  long_diff and tran_diff improve in every set; lifetime in 4 of 6 sets (7 of 15 runs outside A / HNC); eField unchanged.
  A held epoch dump costs ~23 min of GPU.

## 4 Nov 2026 — POLISH: is the conditional minimum a better final estimate than the end point? (rules written 4 Nov, before submission)
From the post hoc table in NOCLIP RESULT (hypothesis formed on A, HNC, T1, DATASZ V arms; tested on unseen samples).
- Estimator: m = calibration at the end-point iterate (20000) + the calibration-only Newton step of a held-epoch E1 dump
  (UQ/run_dump.sbatch <RUN> D_F<s> 20000 LARND_JOINT_NEWTON_DUMP_CALIB=1, exactly the Set D construction; nuisances held
  at the fit). Against the production estimate e (last-1000 mean). Samples: the 12 FRESH end points EXP<s>C75H, s = 29-40
  (unseen). 12 dumps, ~23 min each, ~4.6 GPU-h; on the 6 chains after the FRESH TEST and before B200C (whose heads move
  behind them; B200C shifts by ~46 min).
- **Primary: long_diff**, one-sided Wilcoxon signed-rank test of |e| - |m| > 0 over the 12 pairs; CONFIRMED if p < 0.05.
  SAFE if no parameter (all 5) gets worse (one-sided Wilcoxon of |e| - |m| < 0, p < 0.05 = worse). Secondary: tran_diff
  and lifetime (IMPROVES / WORSE / no detected change by the same tests); means, SDs, RMS of e and m; the per-run count
  |m| < |e|. Single primary because tran_diff reaches only p 0.065 even on the hypothesis-forming Set D at n = 12.
  Dry run on Set D (hypothesis-forming, not evidence): long_diff p 0.002, CONFIRMED, SAFE, tran_diff no detected change,
  lifetime IMPROVES.
- Checks in the readout: calibration-only dump; dumped point = the end-point iterate (|diff| < 1e-3 %); recipe config
  (seed, coarse 7500, smear 1.0 / entry).
- Prediction: CONFIRMED and SAFE; lifetime uncertain (worse on T1 and V025).
- If CONFIRMED and SAFE: a calibration Newton polish (one held epoch, ~23 min GPU) becomes a production candidate as a
  post-processing step (Pierre decides). It also needs an uncertainty: sigma_E1 of the same dump is available, but its
  coverage was poor in the UQ test (R4), so the pooled-SD route would be redone for m.
- Reading order: when all 12 dumps are COMPLETED and FRESH F1 has been read. Readout UQ/read_polish.py.
- Submitted 4 Nov 23:44 (UQ/submit_polish.sh; ids UQ/polish_jobs.txt, setPOL_ids.txt): 12 dumps, 2 per chain after the
  FRESH tails, afterok on each EXP<s>C75H; B200C heads moved behind them (checked). Chain 4: EXP33C75H had already
  COMPLETED (23:23) and was purged from slurmctld, so sbatch refused the afterok; checkpoint verified, that one
  dependency dropped (chains 4-5 finished by hand, same commands). Queue width still 6. Waiter wait_set.sh POL,
  watch_health.sh on the 12.

## 5 Nov 2026 — TRUTHC RESULT (read 5 Nov ~02:45, all 9 jobs COMPLETED at 02:39; TRUTHC/TC_result.txt)
read_tc.py. Every run logged "all 100 batches evaluated". Validation: **V1 PASS, V2 PASS, V3 PASS.** V2 is exact (Hessian
diagonal ratio 1.000, d difference 0.00 sigma): the truth-point block and the E1 dump compute the same central FD of the
same gradient at the same point, so V2 checks the plumbing (same objective, data, point), not the FD method; V1 does that.
P1 Hessians positive definite on all 8 (min eigenvalue 1800-2800), asymmetry <= 5e-2 (sample 2), else <= 3e-3.

Conditional minimum m = e + d (mean +- SEM over KN10N1-8 true ends, %, step 1e-3):

| point | Ab | eField | tran_diff | long_diff | lifetime |
|---|---|---|---|---|---|
| P0 fit | +0.28 +- 0.04 | +0.007 +- 0.002 | +0.43 +- 0.39 | +2.20 +- 0.63 | -1.40 +- 1.10 |
| P1 all at truth | +0.13 +- 0.01 | +0.000 +- 0.001 | +0.23 +- 0.15 | -1.33 +- 0.35 | -1.41 +- 0.23 |
| P2 calib at truth | +0.28 +- 0.03 | +0.008 +- 0.002 | +0.23 +- 0.41 | +2.49 +- 0.70 | -1.32 +- 0.93 |
| P3 dE/dx at truth | +0.15 +- 0.01 | +0.006 +- 0.002 | -0.19 +- 0.28 | -1.08 +- 0.68 | -2.80 +- 0.32 |
| P4 geometry at truth | +0.26 +- 0.03 | +0.001 +- 0.000 | -0.01 +- 0.43 | +3.10 +- 0.63 | -0.48 +- 0.96 |

Paired: m(P0) - m(P1) long_diff +3.53 +- 0.60; m(P0) - m(P3) long_diff **+3.28 +- 0.31 (positive on 8/8)**, tran_diff
+0.63 +- 0.22, Ab +0.14 +- 0.04; m(P0) - m(P4) long_diff -0.90 +- 0.27, lifetime -0.92 +- 0.24; m(P3) - m(P1) long_diff
+0.25 +- 0.55 (n.s.), lifetime -1.39 +- 0.31.
- **Verdict by the rule: long_diff MIXED, tran_diff MIXED.** long_diff: D = +3.53 has the sign of m0 and > 2 SE, but m1 =
  -1.33 is significant (3.8 SEM) with the OPPOSITE sign, so neither class applies. tran_diff: nothing significant on true
  ends (m0 +0.43 +- 0.39, D +0.21 +- 0.27): there is no tran_diff residual to decompose here (its ~+1 residual in the
  bias anatomy is a realistic-end effect, item 3).
- **Reading, long_diff (descriptive, from the pre-registered points):** the end-point error on these 8 true-end samples
  decomposes as e = +3.93 = (e - m0) lag +1.73 + (m0 - m3) **fitted dE/dx +3.28** + (m3 - m1) geometry +0.25 (n.s.) +
  m1 objective at truth -1.33. The fitted dE/dx carries the whole positive bias and more; geometry at truth on its own
  does not help (P4 is 0.9 HIGHER than P0). Prediction ("no firm prior"; the September "model at truth unbiased" pointed
  to NUISANCES): right in direction for long_diff (carried by a nuisance, specifically dE/dx), with an objective-level
  offset of the opposite sign that the rule did not foresee.
- Objective at truth (P1): Ab +0.13 +- 0.01, long_diff -1.33 +- 0.35, lifetime -1.41 +- 0.23, eField 0.000, tran_diff
  n.s. P1 keeps the 10 cm spline representation of the geometry (repr floor 35-53 um) and the model / target differences
  (noise, coarse induction R089), so this is the floor no nuisance fit removes; on a correction table it adds to the
  simulation correction.
- Lifetime: m0 -1.40 +- 1.10 is noisy; P1 and P3 are lower (-1.41, -2.80), geometry at truth raises it (+0.92 +- 0.24).
  No clean statement beyond the at-truth offset.
- Open question that decides the lever: is the dE/dx contribution (i) a frozen-in state (production freezes dE/dx at
  10200, while calibration still moves; a late dE/dx refit would remove it) or (ii) per-segment noise absorbed by the
  nuisance (incidental-parameter bias, constant in N as T5 found; levers = smoother / coarser dE/dx or a simulation
  correction)? Test designed below (TRUTHC2).

## 5 Nov 2026 — TRUTHC2: which part of the fitted dE/dx carries the long_diff bias, the per-track level or the shape? (rules written 5 Nov, before submission)
Follows TRUTHC RESULT (m(P0) - m(P3) long_diff +3.28 +- 0.31). An anatomy of that effect on the SAME 8 end points (not a
confirmation of it), so it uses the same samples.
- Points (review copy TRUTHC/tree, LARND_TP_DEDX_SPLIT=1, 26 more added lines, 52 in all, nothing removed; live tree
  untouched): with calibration and geometry at the fit, and per track (segments grouped by chain context via the parent
  segment ids of its steps), the log dE/dx difference truth - fit = (per-track level: mean log dE/dx over the track's
  segments) + (shape along the track). P5 = fit shape at the TRUE level; P6 = TRUE shape at the fit level. Calibration
  Hessian + gradient at P0, P3, P5, P6, step 1e-3 (V1 passed in TRUTHC). Inners TRUTHC/inner/inner_TC2_KN10N<s>.sh (make_tc.py
  + the point list, the split flag, step 1e-3; diff checked). 8 jobs, ~1 h each.
- CPU smoke (TC2SMOKE, KN10N1 batch 0, losses only): 342/342 segments covered (4 tracks); per-track level error (log)
  -0.0035, SD 0.0052; shape dispersion fit / truth 0.68; P0-P4 losses identical to TCSMOKE; P5 loss 7180.9 (P0 7171.0),
  P6 7318.7 (P3 7327.7).
- Checks: V0 reproducibility, m at P0 and P3 equal to TRUTHC's within 0.05 sigma on all 5 parameters for >= 7/8 samples
  (else NO VERDICT).
- **Primary, long_diff:** Delta_k = m(P0) - m(Pk), paired over the 8 (mean, SE).
  - LEVEL: Delta5 > 0 and > 2 SE, |Delta3 - Delta5| < 2 SE, |Delta6| < 2 SE.
  - SHAPE: Delta6 > 0 and > 2 SE, |Delta3 - Delta6| < 2 SE, |Delta5| < 2 SE.
  - BOTH otherwise (shares Delta5 / Delta3, Delta6 / Delta3 and the interaction Delta3 - Delta5 - Delta6 reported).
- Secondary, descriptive: the same for Ab, tran_diff, lifetime; per-track level error mean / SD; shape dispersion ratio.
- Prediction: SHAPE (the smoke's level error is 0.3 %, and the P6 loss sits next to P3).
- Lever map: LEVEL -> the frozen per-track level (dE/dx freeze at 10200 while calibration moves) -> test a late per-track
  level refit with calibration frozen. SHAPE -> the dE/dx prior's shrinkage of the profile -> test a production fit with
  a weaker / different dE/dx prior (Pierre decides any production change), or carry it in a simulation correction.
- Placement: on the 6 chains after the POLISH dumps and before the B200C heads (moved; B200C shifts by 1-2 h). Reading order:
  when all 8 are COMPLETED. Readout TRUTHC/read_tc2.py (dry mode run on the smoke).
- Submitted 5 Nov 03:15:47 (8 jobs; TRUTHC/tc2_jobs.txt, setTC2_ids.txt): chains 0-1 two each (N1 + N7, N2 + N8), chains 2-5
  one each; the 6 B200C heads moved behind them (checked). Queue width still 6. Waiter wait_set.sh TC2, watch_health.sh.

## 5 Nov 2026 — FRESH TEST F1 RESULT (rules: "4 Nov 2026 — FRESH TEST" above)
- **FRESH F1 RESULT (read 5 Nov 09:30; all 36 pipeline jobs COMPLETED 09:10, config check OK on 12 end points;
  UQ/F1_result.txt). By the rule: NOT CONFIRMED, only because eField's fresh spread is NARROWER than the reference
  (conservative side). Coverage PASSES and every bias transfers.**

  | | Ab | eField | tran_diff | long_diff | lifetime |
  |---|---|---|---|---|---|
  | bias24 | -0.040 | +0.129 | +1.116 | +3.721 | +1.844 |
  | fresh mean (n = 12) | -0.011 | +0.139 | +0.549 | +5.527 | +2.506 |
  | bias transfers (tol 0.707 sd24) | yes | yes | yes | yes | yes |
  | sd24 | 0.183 | 0.049 | 1.373 | 5.614 | 4.554 |
  | fresh SD | 0.152 | 0.024 | 1.265 | 6.028 | 3.309 |
  | ratio (band 0.56-1.62) | 0.83 | **0.50** | 0.92 | 1.07 | 0.73 |
  | \|q\| < 1 / < 2 | 0.75 / 1.00 | 1.00 / 1.00 | 0.67 / 1.00 | 0.75 / 0.83 | 0.75 / 1.00 |

  - Pooled over 60 pulls: |q| < 1 0.78 (band 0.50-0.82), |q| < 2 0.97 (>= 0.83) -> PASS.
  - Post hoc, descriptive (scratch f1_posthoc.py; not a rule): P(F(11, 23) <= 0.253) = 0.011 for eField. The reference's
    eField SD is lifted by its two extremes (0.024, 0.266; SD 0.049 -> 0.034 without them, MAD-SD 0.042; fresh MAD-SD
    0.028); Levene 24 vs 12 p = 0.10 (0.15-0.90 for the other four). Pooled n = 36: eField +0.132 +- SD 0.042.
  - long_diff: 2 of 12 fresh end points beyond the reference range (+15.8, +16.8; ref24 max +12.5); SD ratio 1.07 and
    Levene p = 0.77, so no detected change of spread, but the tail is the thing to watch.
  - Reading: the pooled-24 error bars are usable on unseen single fits: never too narrow on any parameter, too wide on
    eField (fresh SD half of sd24; 1.5x on the robust MAD-SDs), an over-statement in the safe direction. Pooled n = 36 (24 + 12 fresh) is the better reference from now.

## 5 Nov 2026 — Bias-correction table (item 4 of the 4 Nov bias plan; descriptive, no rule)
UQ/correction_table.py -> UQ/correction_table.txt. Production end points EXP<s>C75H @ 20000, realistic 1 mm ends: 24
reference (s = 1-12, 17-28) + 12 FRESH (s = 29-40), all COMPLETED. Correction c = pooled mean error (%); a corrected
single fit e - c has error SD sd sqrt(1 + 1/36). Out-of-sample: the 12 fresh fits corrected with the n = 24 bias only.

| | Ab | eField | tran_diff | long_diff | lifetime |
|---|---|---|---|---|---|
| bias c (n = 36) | -0.031 | +0.132 | +0.927 | +4.323 | +2.065 |
| SE of c | 0.029 | 0.007 | 0.225 | 0.956 | 0.691 |
| c / SE | -1.1 | 18.9 | 4.1 | 4.5 | 3.0 |
| corrected-fit SD | 0.174 | 0.042 | 1.366 | 5.813 | 4.201 |
| RMS raw / leave-one-out corrected (36) | 0.172 / 0.174 | 0.139 / 0.043 | 1.620 / 1.366 | 7.117 / 5.815 | 4.578 / 4.202 |
| fresh RMS raw / corrected with c24 (12) | 0.146 / 0.148 | 0.141 / 0.025 | 1.330 / 1.338 | 7.990 / 6.047 | 4.039 / 3.236 |

- Ab: no bias (c / SE -1.1); no correction. eField: the correction removes nearly all of the error (RMS 0.14 -> 0.04;
  out of sample 0.14 -> 0.03). long_diff and lifetime: RMS falls 18 % and 8 % (out of sample 24 % and 20 %). tran_diff:
  16 % in leave-one-out, none out of sample (the fresh mean +0.55 was already near zero).
- What the long_diff correction stands for (TRUTHC on true ends, not the same pipeline ends): lag to the conditional
  minimum ~+1.7, fitted dE/dx ~+3.3, objective at truth ~-1.3. POLISH (pending) tests removing the lag part per fit;
  TRUTHC2 (pending) splits the dE/dx part. After either, this table must be recomputed on the new end points.
- Valid for this configuration only: KN10, 400 cm per batch, production C75 pipeline, 1 mm smeared ends, simulation
  with the same physics model. A change of any of these needs a new table.

## 5 Nov 2026 — FRESH TEST F2 RESULT (rules: "4 Nov 2026 — FRESH TEST" above)
Read 5 Nov 15:30; all 72 jobs COMPLETED 15:20; config check OK on 12 x 4 end points; every H2/H3/H4 history equals A's
bit for bit through 14000 (UQ/F2_result.txt). **Rule verdict: averaging the 4 H-stage end points is MARGINAL on Ab,
tran_diff, long_diff and lifetime (gain 7-11 %), NO GAIN on eField.**

| | Ab | eField | tran_diff | long_diff | lifetime |
|---|---|---|---|---|---|
| sigma_w (36 dof) | 0.097 | 0.006 | 0.637 | 2.459 | 1.974 |
| 95 % CI | 0.079-0.127 | 0.005-0.008 | 0.518-0.827 | 2.000-3.194 | 1.606-2.565 |
| T2 sigma_opt^H (prediction input) | 0.119 | 0.006 | 0.310 | 3.053 | 3.236 |
| gain g (range) | 11.2 % (7.3-19.8) | 0.6 % | 8.4 % (5.5-14.7) | 7.5 % (4.9-13.0) | 7.3 % (4.8-12.7) |
| class | MARGINAL | NO GAIN | MARGINAL | MARGINAL | MARGINAL |

- Against the prediction: eField NO GAIN as expected; Ab and long_diff MARGINAL as expected (both at the low end of the
  predicted gain); tran_diff MARGINAL, not NO GAIN (sigma_w 0.64, twice T2's 0.31); lifetime 7 % (between the two T2 numbers).
- Descriptive: SD(average) / SD(A) over the 12: Ab 0.78, eField 1.07, tran_diff 0.91, long_diff 0.83, lifetime 0.91 (paired
  bootstrap CIs all reach ~1). The averages against bias24 with sigma_avg: pooled |q| < 1 0.80, < 2 0.98.
- Post hoc (scratch f2_posthoc.py; not a rule): A is not special. A - mean(H2-H4): long_diff +1.62 +- 0.91 (t p = 0.10,
  driven by s = 33: A +15.8, re-runs +7.1 to +10.7), all other parameters |t| < 1.6; Friedman over the 4 replicas
  p = 0.03 for tran_diff (H3, H4 +1.0 vs H, H2 +0.4-0.5; 1 of 5 tests, not significant after Bonferroni), >= 0.08 otherwise.
- Reading: the H stage carries only a small share of the single-fit error variance (sigma_w^2 / sd24^2: Ab 28 %, tran_diff 22 %,
  long_diff 19 %, lifetime 19 %, eField 2 %), so even an infinite average gains <= 10-15 %. The spread is the data
  (sample-to-sample), which only more data shrinks (DATASZ T5: ~1/sqrt(N)). My recommendation: do not adopt the 4-run
  average (+~90 % GPU per fit for 7-11 %); Pierre's call.

## 5 Nov 2026 — POLISH RESULT (read 5 Nov ~16:15; all 12 dumps COMPLETED 16:06, FRESH F1 already read; UQ/POLISH_result.txt)
UQ/read_polish.py; config checks OK (seed, coarse 7500, smear 1.0 / entry; calibration-only dumps at the end-point
iterate). **Verdict: CONFIRMED (long_diff p = 0.002, 11/12); SAFE; secondary: tran_diff no detected change, lifetime
IMPROVES (p = 0.032, 9/12).** As predicted (lifetime had no firm prediction).

| fresh, n = 12 | Ab | eField | tran_diff | long_diff | lifetime |
|---|---|---|---|---|---|
| e mean / SD / RMS | -0.01 / 0.15 / 0.15 | +0.14 / 0.02 / 0.14 | +0.55 / 1.27 / 1.33 | +5.53 / 6.03 / 7.99 | +2.51 / 3.31 / 4.04 |
| m mean / SD / RMS | +0.04 / 0.14 / 0.14 | +0.14 / 0.02 / 0.14 | +0.34 / 1.18 / 1.17 | +4.02 / 5.33 / 6.49 | +0.82 / 3.16 / 3.13 |
| \|m\| < \|e\| ; Wilcoxon p(better) | 6/12 ; 0.34 | 10/12 ; 0.010 | 6/12 ; 0.34 | 11/12 ; 0.002 | 9/12 ; 0.032 |

- eField: |m| < |e| on 10/12 (p = 0.010) but by ~0.001 points (RMS 0.14 both): no practical change.
- Follow-up the rules asked for (uncertainty of m; descriptive; UQ/polish_table.py -> polish_table.txt): every
  production end point with a calibration-only dump at 20000, n = 30 (s = 1-6 T1 dumps, 17-28 Set D, 29-40 fresh):

  | n = 30 | Ab | eField | tran_diff | long_diff | lifetime |
  |---|---|---|---|---|---|
  | e bias / SD | -0.036 / 0.174 | +0.130 / 0.043 | +0.838 / 1.419 | +4.318 / 6.131 | +1.955 / 4.131 |
  | m bias +- SE | +0.014 +- 0.028 | +0.129 +- 0.008 | +0.713 +- 0.242 | +3.063 +- 0.958 | +0.508 +- 0.682 |
  | m SD | 0.156 | 0.043 | 1.323 | 5.245 | 3.738 |
  | m RMS raw / leave-one-out corrected | 0.154 / 0.159 | 0.136 / 0.043 | 1.484 / 1.346 | 5.998 / 5.335 | 3.710 / 3.802 |
  | fresh m RMS raw / corrected with the other 18 | 0.141 / 0.142 | 0.141 / 0.028 | 1.175 / 1.289 | 6.494 / 5.344 | 3.134 / 3.070 |

  - The polish takes 1.26 points off the long_diff bias and 1.45 off lifetime, the lag measured in NOCLIP / TRUTHC
    (-1.3 / -1.4); the lifetime bias of m (+0.5 +- 0.7) is no longer significant. SDs shrink (long_diff -14 %, lifetime
    -10 %, Ab -10 %, tran_diff -7 %); eField unchanged.
  - With a correction on top, m needs only eField, tran_diff and long_diff corrected (Ab, lifetime biases n.s.).
- Reading: the calibration Newton polish (one held-epoch E1 dump, ~23 min GPU, nuisances held at the fit) is a
  production candidate as a post-processing step: Pierre decides. Production unchanged until then. If adopted, the
  uncertainty is the pooled SD of m (table above; n = 30) rather than the dump's own sigma_E1 (R4: poor coverage).

## 5 Nov 2026 — TRUTHC2 RESULT (read 5 Nov ~17:45; all 8 jobs COMPLETED 17:32, each logged "all 100 batches evaluated" and 100 dE/dx split lines; TRUTHC/TC2_result.txt)
read_tc2.py. **V0 PASS (7/8).** The miss is sample 3 at P0 (up to 0.70 sigma, on tran_diff; its P3 agrees to 0.002
sigma, so same data and code path); all others <= 0.024 sigma. Split: 100 % of segments covered; per-track level error
(log) -0.0018, SD 0.0109; shape dispersion fit / truth **0.787**. **Rule verdict (long_diff): BOTH.** In substance the
SHAPE carries the effect and the level pushes the other way:

| m (mean +- SEM, %) | Ab | eField | tran_diff | long_diff | lifetime |
|---|---|---|---|---|---|
| P0 fit | +0.28 +- 0.04 | +0.01 | +0.40 +- 0.37 | +2.21 +- 0.62 | -1.43 +- 1.11 |
| P3 dE/dx true | +0.15 +- 0.01 | +0.01 | -0.19 +- 0.28 | -1.08 +- 0.68 | -2.80 +- 0.32 |
| P5 true level, fit shape | +0.10 +- 0.03 | +0.01 | +0.45 +- 0.40 | +2.91 +- 0.63 | -1.32 +- 0.67 |
| P6 true shape, fit level | +0.31 +- 0.02 | +0.01 | -0.20 +- 0.27 | -1.72 +- 0.62 | -2.32 +- 0.49 |
| Delta3 (all dE/dx) | +0.14 +- 0.04 | 0 | +0.59 +- 0.21 | **+3.29 +- 0.31** | +1.37 +- 1.29 |
| Delta5 (level) | **+0.19 +- 0.02** | 0 | -0.04 +- 0.09 | -0.70 +- 0.10 | -0.12 +- 0.48 |
| Delta6 (shape) | -0.03 +- 0.03 | 0 | **+0.60 +- 0.21** | **+3.93 +- 0.29** | +0.88 +- 0.68 |
| interaction D3 - D5 - D6 | -0.02 +- 0.01 | 0 | +0.04 +- 0.05 | +0.06 +- 0.17 | +0.60 +- 0.18 |

- long_diff: Delta6 +3.93 +- 0.29 (8/8 positive, +2.90 to +5.35) = 119 % of Delta3; Delta5 -0.70 +- 0.10 (8/8 negative,
  -0.30 to -1.10) = -21 %. SHAPE fails its criteria only because the level term is significant and of the OPPOSITE sign
  (|Delta5| = 7 SE; |Delta3 - Delta6| = 0.64 +- 0.24 = 2.7 SE); the decomposition is additive (interaction +0.06 +- 0.17).
- tran_diff: all of its (smaller) dE/dx effect is shape (Delta6 +0.60 = 101 %). Ab: its dE/dx effect is the LEVEL
  (+0.19 +- 0.02; the per-track dE/dx level trades against the charge scale, as expected). lifetime: noisy; shape +0.88
  +- 0.68, interaction +0.60 +- 0.18.
- Reading: the long_diff bias from dE/dx is the profile ALONG the track: the fitted profile has 79 % of the true
  dispersion (under-dispersed), and long_diff absorbs it. The per-track level is right to 1 % (SD) and correcting it
  alone would make long_diff WORSE (+0.7). Lever map (pre-registered): SHAPE -> the dE/dx prior's shrinkage of the
  profile -> test a fit with a weaker / different dE/dx prior (any production change is Pierre's), or carry it in a
  simulation correction. The level refit is NOT a lever.
- **The pre-registered SHAPE lever has prior evidence against it (checked after the readout, 5 Nov):** the dE/dx prior
  weight / width was scanned both ways in September on the pre-KN10 pipeline: WIDE (x1.6) restored the width but filled
  it with noise (MAE 0.058 -> 0.088, lifetime worse; commit 1b82c91, "closes the prior-tuning programme"), and the
  narrower x0.40 optimum did not survive fresh seeds (commit 9a45383; table in §5 above: REFUTED). Production since then
  uses weight 1.0088 = 1 / dx (one prior count per segment). So no prior scan is launched. The September Laplace test
  (§3.3b) marginalised dE/dx for lifetime and Ab only, never long_diff. Remaining levers, none built: (a) coarser dE/dx
  segments (fewer incidental parameters per track; no knob exists: perseg 1 cm or scale_shape only); (b) dE/dx
  marginalisation for long_diff. Until then the bias is carried by the correction table (out of sample long_diff RMS
  7.99 -> 6.05) and the POLISH step (the lag part).

## 6 Nov 2026 — B200C RESULT (read 6 Nov ~01:05; all 36 jobs COMPLETED 00:58; fit_params.py md5 24359dc4... unchanged; FRESH F1 read; DATASZ/B2C_result.txt)
read_b2c.py, config checks OK, n = 12 (s = 29-40). **Verdict: NOT CONFIRMED; SAFE.** Against the prediction (CONFIRMED):
the T6 hint of a lower spread with 200 cm batches does not replicate on unseen samples.

| B200 vs reference (n = 12) | Ab | eField | tran_diff | long_diff | lifetime |
|---|---|---|---|---|---|
| SD | 0.142 vs 0.152 | 0.036 vs 0.024 | 1.475 vs 1.265 | 6.394 vs 6.028 | 3.465 vs 3.309 |
| SD ratio (Pitman-Morgan one-sided p) | 0.93 (0.42) | 1.47 (0.93) | 1.17 (0.70) | 1.06 (0.59) | 1.05 (0.56) |
| paired delta of mean error | -0.03 +- 0.06 | -0.00 +- 0.01 | -0.08 +- 0.43 | +1.12 +- 1.73 | +0.65 +- 1.43 |
| RMS error | 0.143 vs 0.146 | 0.141 vs 0.141 | 1.488 vs 1.330 | 9.033 vs 7.990 | 4.580 vs 4.039 |

- Arm correlations are lower than on T6 (long_diff +0.54 vs 0.77, lifetime -0.06 vs 0.80): the two batchings of the
  same sample land on different values, so the pairing gains little.
- Position paired delta -5.3 +- 4.6 um (one-sided p 0.13; prediction "better" not met). Track length used 41 330 vs
  39 906 cm. GPU time 3.87 vs 3.09 GPU-h per pipeline (+25 %).
- Pool of 20 (T6 + B200C; CONTAINS the hypothesis data, descriptive only): SD ratios Ab 0.73 (p 0.09), eField 0.94,
  tran_diff 1.05, long_diff 0.90 (p 0.28), lifetime 0.78 (p 0.14); no paired shift. A small spread gain cannot be
  excluded, but none is demonstrated.
- Consequence: production stays at 400 cm batches. DATASZ thread CLOSED: the spread falls like statistics with the data
  volume (T5), batch size at fixed tracks moves neither the biases (T6) nor, demonstrably, the spread (B200C).

## 6 Nov 2026 — Pierre's decisions on the recap, and the adoptions
Pierre, 6 Nov: "1. Lets adopt POLISH 2. This looks very much like cheating. I wouldn't do that [the bias correction]
3. Ensemble, yes. But it would be good to understand why the Huber sandwich and Hessian don't work while they are supposed
to in theory, right? ... 5. I think we are ok for now [precision / more data] ... 7. Let's adopt the cache and write targets
directly to data".
- **No bias correction** (the correction table above stays a diagnostic; it is not applied to any result).
- **Adoptions (production, 6 Nov; not committed):**
  - POLISH = 4th chained job of KNOT/submit_prod.sh: KNOT/run_polish.sbatch (the validated UQ/run_dump.sbatch construction;
    DRY build on EXP29C75 identical to the validated D_F29 inner except dump path and run name) -> KNOT/polish_estimate.py ->
    fit_result_staging/sci_full_<NAME>H/polish_estimate.json. The step is applied EXACTLY through the production sigmoid map
    (self-checked: scheme = sigmoid, one sigmoid scale implied by the dump's jac), not as e + d. Same tests on that exact form
    (UQ/polish_exact_check.py -> polish_exact_check.txt): fresh 12 long_diff RMS 7.99 -> 6.31, 11/12, p 0.002; lifetime
    4.04 -> 3.08, p 0.026; nothing worse; n = 30 m bias long_diff +2.95 +- 0.94, lifetime +0.47 +- 0.67, SD 0.156 / 0.043 /
    1.321 / 5.132 / 3.675 % (the quoted uncertainty, x sqrt(1 + 1/30)). KNOT/submit_prod_chains.sh: the next sample waits for
    POLISH (afterany:P, afternotok on fit / FR / H), so running jobs still = number of chains (fake-sbatch test OK).
  - Per-pipeline JAX cache: ENDFIT/cache_adoption example_run.py.new installed (live file was byte-identical to .orig);
    submit_prod.sh exports LARND_JAX_PERSISTENT_CACHE=shared, JAX_COMPILATION_CACHE_DIR=.../jax_job_cache/pipe_<NAME>.
  - Targets on /sdf/data: new knob LARND_TARGET_ROOT in fit_params.py (target dir under it; unset = unchanged), exported by
    submit_prod.sh as /sdf/data/neutrino/pgranger/larnd-sim-jax. fit_params.py md5 24359dc4... -> 627cf151...;
    example_run.py -> 61dd8b6a...; backups in ENDFIT/cache_adoption/*.before_*6nov.
  - Checks: CPU smoke pair (root unset vs set + shared cache; losses must be bitwise equal) running; then one full GPU
    acceptance pipeline before calling the adoption done.

## 6 Nov 2026 — Why the Hessian and the sandwich fail: the textbook case works (descriptive; TRUTHC/theory_check.py)
From the TRUTHC/TRUTHC2 truth points (KN10N1-8 C75H @ 20000, true ends; calibration moved to its exact conditional
minimum m, so no calibration optimiser scatter). SD of m over the 8 samples / median sandwich width (95 % band if the
width is right: 0.49-1.51):

| point | Ab | eField | tran_diff | long_diff | lifetime |
|---|---|---|---|---|---|
| P1 all nuisances true | 0.74 | 1.96 | 1.28 | 1.16 | 0.85 |
| P4 geometry true, dE/dx fitted | 2.29 | 1.10 | 2.45 | 2.39 | 2.62 |
| P3 dE/dx true, geometry fitted | 0.88 | 11.1 | 1.71 | 2.24 | 0.71 |
| P0 all fitted | 2.94 | 19.7 | 2.96 | 2.67 | 3.31 |

- With the nuisances known, the sandwich is right (4/5 inside the band; eField 1.96 at a width of 0.0009 %). Sandwich /
  Hessian is only 1.06-1.5 (mild misspecification). The theory fails only through the FITTED nuisances: the Hessian and
  the sandwich are conditional on dE/dx and geometry, so they miss the variance the nuisance estimates pass to the
  calibration. dE/dx carries it for Ab, tran_diff, long_diff, lifetime (x2.3-2.6); geometry for eField (x11), long_diff,
  tran_diff. On production (realistic ends) the gap is larger again (3-96x, UQ/hess_vs_sandwich.py), plus optimiser
  scatter (20-40 % of the variance, F2).
- The theoretical fix is the profiled (Schur) curvature and profile scores. Geometry: not computable (joint Hessian
  indefinite in 92-100 % of batches, FD-step dependent: the objective is not smooth in geometry; Pilot 2). dE/dx: computable
  exactly (tridiagonal + rank-1 block), so the dE/dx-profiled sandwich at P4 is the next test (proposed to Pierre).

## 6 Nov 2026 — Adoption checks (rules written before the acceptance pipeline is submitted)
- **CPU smoke pair: PASS** (read 6 Nov 02:20; scratchpad trtest). Production fit inner, 1 batch, 3 iterations: run A
  (LARND_TARGET_ROOT unset, per-job cache) vs run B (root set, LARND_JAX_PERSISTENT_CACHE=shared + pipe dir). Every numeric
  history entry bitwise equal except step_time (losses 27337.627 / 27139.725 / 26891.141 both; dE/dx equal); A's target dir
  in the repo root (old behaviour), B's under the root; B used the pipe dir (3.1 MB) and recorded LARND_TARGET_ROOT in
  env_overrides. Test leftovers removed.
- **Acceptance pipeline EXP41C75** (data_seed 41: no script has used it; inner = the FRESH inners with seed and name changed,
  diff checked), submitted with the real KNOT/submit_prod.sh, one job at a time, ~4.3 GPU-h. PASS if all of:
  - A1 the 4 jobs (fit, FR, H, P) COMPLETED;
  - A2 each of the 4 logs shows "Persistent JAX compilation cache ON at .../pipe_EXP41C75 (mode=shared";
  - A3 the 4 stages' target dirs are under /sdf/data/neutrino/pgranger/larnd-sim-jax and none is created in the repo root;
  - A4 sci_full_EXP41C75H/polish_estimate.json written (its self-checks: calibration-only dump at the 20000 iterate,
    sigmoid scheme, one implied scale);
  - A5 config check on the H end point (seed 41, coarse 7500, smear 1.0 / entry).
  Descriptive: FR / H first-epoch times (cache reuse; 29 Oct: 9:50 -> 3:20 / 2:15), pipeline wall time; polished errors
  against the quoted uncertainty (|q| < 3 expected; one sample, a sanity check, not a test).
- Submitted 6 Nov 02:38:55: fit 43486417 -> FR 43486418 -> H 43486419 -> P 43486421 (KNOT/accept_EXP41_ids.txt).

## 6 Nov 2026 — DXPROF: does the dE/dx-PROFILED sandwich predict the spread when dE/dx is fitted? (rules written 6 Nov, before the set is submitted)
Pierre, 6 Nov: "3. Go ahead with what you propose". Follows the theory check above (conditional sandwich x2.3-2.6 too
narrow at P4 = geometry true, dE/dx fitted).
- Code: review copy DXPROF/tree = TRUTHC/tree + DXPROF/patch_dxprof.py (live tree untouched): the dE/dx-exact Schur block of
  the truth-point evaluator, run at P4 (LARND_TP_SCHUR_POINT), all 100 batches, storing per batch H_cc,b, g_c,b and for
  each dE/dx probe step (3e-3, 1e-2) x ridge (0, 1e-3, 1e-2 x median |diag D|) Delta_b = H_cd D^-1 H_dc and the profiled
  score r_b = g_c - H_cd D^-1 g_d. D = dE/dx block (banded(2) from 5 FD probes + the analytic rank-1 of the mean
  constraint, Sherman-Morrison; no iterative solver). Inners DXPROF/inner/inner_DX_KN10N<s>.sh = the TRUTHC2 inners with
  only the point list and these variables changed (diff checked). 8 jobs (KN10N1-8 C75H @ 20000, true ends), ~1 h each.
- Estimator under test: the one-step profiled estimate m_prof = e + 100 jac (-S^-1 sum_b r_b), S = sum_b (H_cc,b - Delta_b)
  (Le Cam one-step from the P4 point; calibration and dE/dx jointly, geometry fixed at truth). Width: profiled sandwich
  S^-1 J S^-1, J = B/(B-1) sum_b (r_b - rbar)(r_b - rbar)^T.
- Checks: V1 dE/dx step stability (3e-3 vs 1e-2: profiled sandwich width within 10 % and m_prof within 0.1 sigma on the 4
  primary parameters, >= 7/8 samples); V2 the conditional pieces reproduce TRUTHC P4 (m within 0.05 sigma on all 5,
  >= 7/8); V3 banded solves at ridge 0 succeed on >= 95 % of batches of every sample (else the pre-registered ridge 1e-3
  is primary). V1 or V2 fails -> NO VERDICT.
- **Primary:** R = SD over the 8 samples of m_prof / median profiled sandwich width, on Ab, tran_diff, long_diff, lifetime
  (eField is geometry-driven, not tested here). 95 % band if the width is right: [0.49, 1.51] (chi2_7). CONFIRMED: all 4
  in the band; PARTLY CONFIRMED: 2-3; NOT CONFIRMED: 0-1.
- Secondary, descriptive: conditional vs profiled widths (inflation factor; the theory check measured x2.3-2.6 between
  the observed spread and the conditional sandwich); SD of the conditional m(P4) against the profiled sandwich; mean
  m_prof vs mean m(P4) (does profiling dE/dx at true geometry move the bias?).
- Prediction: PARTLY CONFIRMED. The 31 Oct pilot E2 (one sample, FD steps x10/x20, fitted geometry) gave a profiled sandwich
  close to the observed P4 spread for Ab and lifetime but about half of it for tran_diff and long_diff.
- Meaning: CONFIRMED -> the textbook machinery works once the nuisance is profiled; the production gap is then geometry
  (whose FD Hessian is unusable), optimiser scatter and the endpoint smear, and the route is an information matrix that
  does not need a smooth geometry Hessian (literature review running). NOT / PARTLY -> first-order profiling at P4 is not
  enough (dE/dx at P4 is not optimal for the true geometry; non-linearity): iterate the joint step or look elsewhere.
- Reading order: when all 8 are COMPLETED. Readout DXPROF/read_dx.py (dry mode on the GPU smoke DXSMOKE, 1 batch).
- Submitted 6 Nov 03:26:30 (DXPROF/submit_dx.sh; dx_jobs.txt, setDX_ids.txt): 4 chains of 2 (N1->N5, N2->N6, N3->N7, N4->N8); DXPROF/tree fit_params md5 85f4ab01...; with the acceptance pipeline <= 5 running. (A first attempt at 03:26:12 failed at sbatch before any job: zsh did not split the account flags.) GPU smoke DXSMOKE (1 batch): all 6 step x ridge solves ok, Schur H_cc = truth-point H_cc to 5e-6, readout dry run OK.

## 6 Nov 2026 — Literature: uncertainty with many fitted nuisances, off the optimum, non-smooth objective
Pierre, 6 Nov: "try to think or search in the literature if there is some strategy we could use to have an uncertainty
estimate with the hessian/derivatives information while not being exactly at the true point and with nuisance".
Full review (every reference confirmed by web search; two spot-checked again by me): UQ/LITERATURE_uncertainty_nuisances.md.
- Frame: batch-local nuisances make the fit an M-estimator over independent groups; the right variance is the PROFILE
  sandwich H_p^-1 (sum_b s_b s_b^T) H_p^-1 with the efficient (profiled) scores s_b and H_p = H_tt - H_te H_ee^-1 H_et.
  H_tt >= H_p, so the conditional sandwich must under-predict: our x2.3-3.3 is the expected direction.
- Not at the truth: fine (plug-in). Not at the optimum: fine to o(n^-1/2); a one-step (Le Cam) correction with the
  efficient score gives the efficient estimator, and the efficient score is Neyman-orthogonal (tolerates unconverged
  nuisances) -- the profiled version of POLISH, which DXPROF computes for dE/dx.
- Non-smooth geometry: asymptotics need a smooth EXPECTED criterion only (Pakes-Pollard 1989; Hong-Mahajan-Nekipelov 2015:
  FD steps must not shrink too fast on a discontinuous objective -- our step dependence). Bread from the expected
  information / Gauss-Newton (Poisson: J^T diag(1/mu) J): PSD by construction, first derivatives only, PSD Schur.
- Caveats: many-nuisance profile/Hessian intervals can under-cover even when correct (Sartori 2003 growth condition;
  HEP: Alexe, Bendavid, Bianchini, Bruschini, arXiv:2401.10542, Barlow-Beeston); SGD-trajectory covariance methods
  (Chen et al. 2020) do not apply (single pass, strongly convex); bias corrections (modified / adjusted profile = Laplace
  integration of the nuisances; split-panel jackknife) are truth-free but reduce, not remove, fixed-T bias.
- Ranked (review + mine): (1) profile sandwich with a Gauss-Newton Schur complement over dE/dx AND geometry, per track,
  meat from per-batch efficient scores, one-step correction included; (2) numerical profile bread as a cross-check
  (nuisance-only re-optimisation at theta +- h e_k); (3) calibrate both on the replica ensemble.

## 6 Nov 2026 — DXPROF RESULT (read 6 Nov ~05:00; all 8 jobs COMPLETED 04:58, 100 Schur records each; DXPROF/DXPROF_result.txt)
- **Rule verdict: NO VERDICT** -- V1 FAILED (0/8: the profiled sandwich and m_prof change by more than 10 % / 0.1 sigma
  between dE/dx probe steps 3e-3 and 1e-2). V2 PASS (7/8), V3 PASS (one failed solve in 800 batches).
- Primary numbers anyway (step 3e-3, ridge 0; median widths %, Ab eField tran long lifetime): conditional sandwich 0.040
  0.0009 0.50 0.74 1.03; profiled Hessian 0.051 0.0006 0.39 0.89 1.26; profiled sandwich 0.24 0.0053 1.02 2.21 5.29; SD of
  m_prof 0.74 0.021 1.44 6.82 14.25 -> ratio 3.09 3.86 1.41 3.09 2.70 (1/4 in the band). The one-step profiled estimate is
  ~4-8x NOISIER than the conditional minimum it was meant to improve (SD of m_cond 0.092 0.0009 1.22 1.80 2.71).
- POST HOC (DXPROF/diag_dx.py, diag_dx2.py; not pre-registered, n = 8):
  - The profiled BREAD S = sum (H_cc - Delta) is stable across steps and ridges (profiled Hessian widths agree to a few %
    in all 6 step x ridge combinations). The instability is entirely in the profiled SCORE r_b = g_c - H_cd D^-1 g_d: its
    correction H_cd D^-1 g_d agrees between steps on the median batch (|diff| 2-5 % of its SD) but is heavy-tailed (step
    correlation 0.2-0.7): a few batches with weakly constrained dE/dx directions amplify g_d (at P4 dE/dx is not at its
    optimum for the swapped-in true geometry). SD of m_prof explodes in some combinations (lifetime 276 % at step 1e-2).
  - The sandwich with the profiled bread and the RAW per-batch scores, S^-1 J_g S^-1 (the textbook form at a joint optimum,
    where every batch's nuisance gradient vanishes and the raw score IS the efficient score), matches the spread of the
    estimator we actually use (the conditional minimum m_cond at P4): SD(m_cond) / width = **0.78 0.95 1.79 1.54 1.02**
    (step 3e-3; 0.78 0.45 1.72 1.31 0.91 at step 1e-2), against 2.28 1.09 2.44 2.42 2.63 for the conditional sandwich.
    Ab, eField, lifetime within the band; long_diff at its edge; tran_diff 1.8x short.
- Reading: profiling dE/dx out of the CURVATURE is what closes most of the theory-check gap (x2.3-2.6 -> x0.8-1.8); the
  efficient-score one-step is not usable on this objective (weak nuisance directions, nuisance off its optimum). For the
  Gauss-Newton profile over dE/dx AND geometry (next), the primary width will therefore be profiled bread x raw-score
  meat, around the conditional minimum (POLISH = the production estimator); the efficient-score correction is a
  diagnostic only.
- Incident during this readout (no data lost): copying DXPROF/tree to GNPROF/tree, I ran rm -rf on GNPROF/tree/fit_result/*
  without checking that fit_result in the review trees is a SYMLINK to the live fit_result -> the live fit_result was
  emptied at 05:05 (2410 entries: 1381 links to /sdf/data, 979 links to fit_result_staging, 50 real directories, 8 GB).
  The link targets were never touched (rm does not follow links). Restored from the 05:00 /sdf/group snapshot
  (/sdf/group/.snapshots/@GMT-2026.11.06-13.00.00): all 2410 entries identical (name, type, link target), all 67 files of
  the real directories byte-identical (cmp). The running EXP41C75H job lost its output link for ~1 min with no save in that
  window (16000 checkpoint written to staging, no error in its log). New rule (memory): no rm of a glob or variable path in
  a copied tree.

## 6 Nov 2026 — GNPROF: does a Gauss-Newton profile over dE/dx AND geometry give the right width at the production fit point? (rules written 6 Nov, before the set is submitted)
Pierre, 6 Nov: "Please apply your proposals" (the ranked plan of the literature digest: (1) DXPROF, (2) Gauss-Newton profile
over dE/dx + geometry, (3) a cross-check that needs no smooth Hessian, (4) calibrate on the ensemble). Follows DXPROF
(profiled BREAD stable and closes most of the P4 gap; profiled SCORE unusable).
- Code: review copy GNPROF/tree = DXPROF/tree + GNPROF/patch_gnprof.py (live tree untouched; md5 8a7d4c2d...). Per batch at
  P0 (calibration, dE/dx, geometry of the fit = what production has), in addition to the DXPROF pieces (FD H_cc, g_c,
  observed dE/dx-exact Delta at step 3e-3): one jvp + one vjp through the simulation per column -- 5 calibration, 5 dE/dx
  band probes, every geometry coefficient (48-118 per batch) -- giving the curvature from FIRST derivatives of the
  predicted fields only (the non-smooth geometry never enters a second derivative). Two curvature models from the same
  tangents: A = expected Fisher of the marked Poisson process (PSD), C = exact generalised Gauss-Newton J^T (d2 loss /
  d fields2) J (the loss's own curvature, matching window included). Priors (dE/dx, MCS, endpoint) with their exact
  Hessian. Delta = F_cn A_nn^-1 F_nc (dense solve, nuisance dimension ~500 per batch) for ridges 0, 1e-3, 1e-2 x median
  |diag A_nn|; also dE/dx-only and geometry-only. At P1_truth: the calibration F_cc of both models and the FD H_cc.
  8 jobs (KN10N1-8 C75H @ 20000, true ends; inners = the DXPROF inners with only the point and these variables changed).
- Estimator under test: the conditional minimum at P0, m = e + 100 jac (-H^-1 sum_b g_c,b) (= POLISH, the production
  estimator). Width: B^-1 J B^-1 with the RAW per-batch scores (J from g_c,b; the joint-optimum form that DXPROF
  supported post hoc) and B = the profiled bread.
- Checks: V-b internal consistency, per model: median over batches of |jvp Gram - vjp rows| (calibration block), asym of
  the geometry block and of F_cg vs F_gc, all < 1e-2 (a model failing V-b is not used; both failing -> NO VERDICT).
  V-a curvature model AT P0, where it is used: sum_b F_cc / sum_b FD H_cc (the observed calibration curvature, smooth
  and step-stable), diagonal, median over samples; primary model = the V-b-passing model with the smaller max |ln ratio|
  on Ab, tran_diff, long_diff, lifetime; if its ratios are not all in [0.8, 1.25] the primary bread is the hybrid
  H - sum Delta. (First drafted at P1_truth; moved before submission because the 2-batch smoke showed the exact GGN
  strongly indefinite at P1 on a batch where the truth-projected geometry misfits -- loss +314 above the fit -- while it
  matches the FD Hessian at P0 on both batches. P1 ratios stay descriptive.) Descriptive cross-check: GN vs observed
  (DXPROF construction) dE/dx-only fractions Delta_d / F at P0.
  V-c solves at ridge 0 succeed on >= 95 % of batches of every sample (else ridge 1e-3 primary), and the primary widths at
  ridge 0 vs 1e-3 agree within 10 % on all 5 parameters on >= 7/8 samples; V-c fails -> NO VERDICT.
- **Primary:** R = SD over the 8 samples of m / median primary width, all 5 parameters (geometry is profiled now, so eField
  is in). Band [0.49, 1.51] (chi2_7). CONFIRMED: 5/5 in the band; PARTLY CONFIRMED: 3-4; NOT CONFIRMED: 0-2.
- Secondary, descriptive: R for the conditional, dE/dx-only (GN and observed) and geometry-only breads (which block
  carries which parameter); hybrid vs pure GN; the other curvature model; non-positive eigenvalues of A_nn; the efficient-
  score correction (diagnostic only); wall time per batch.
- Prediction: PARTLY CONFIRMED -- eField and lifetime close (geometry carries eField's x20, dE/dx carries lifetime),
  tran_diff and/or long_diff stay above the band (DXPROF post hoc: tran_diff 1.8x short with dE/dx profiled; the
  optimiser scatter of the nuisances, ~20-25 % of the variance of tran_diff and long_diff, is not a curvature effect).
- Meaning: CONFIRMED -> a per-fit, replica-free width from derivatives at the fit point exists for this configuration;
  next = the same at the production (realistic-end) end points against the n = 30 ensemble SD. PARTLY -> it is a
  partial tool: report which parameters it covers; the ensemble remains the reference for the rest.
- Not done (step (3) of the plan, re-optimise the nuisances at a nudged calibration and difference the gradient): it is
  the re-fit profile already measured 2105x too noisy in September (report fig 70, landing-point noise). Its role -- a
  curvature check that needs no smooth geometry Hessian -- is taken by V-a (model vs the FD Hessian at the truth) and by
  the GN vs observed dE/dx-only comparison at P0.
- GPU smoke GNSMOKE (2 batches of KN10N1, job 43501825, 5.6 min): GN block 100-106 s per batch (setup 9 s, 76-107 jvps 29-34 s,
  2 x 76-107 vjps 55-63 s, prior Hessian 6-7 s), peak 11.0 GiB, no non-finite values; V-b quantities 7e-7 - 2e-4; A_nn
  positive definite for A, 0 and 3 non-positive eigenvalues for C. F_cc(P0) / FD H_cc(P0): C 1.01 1.04 0.93 1.01 1.04,
  A 1.67 2.16 1.68 2.67 1.69 (A ignores the matching window: ~2x too much information). GN-C dE/dx-only fraction vs the
  observed dE/dx-exact one: 0.99 0.99 0.72 1.00 0.97. Expected cost ~2.2 min per batch -> ~3.7 h per sample.
- Reading order: when all 8 are COMPLETED. Readout GNPROF/read_gn.py (dry mode on the GPU smoke GNSMOKE, 2 batches).
- Submitted 6 Nov 05:20:56 (GNPROF/submit_gn.sh; gn_jobs.txt, setGN_ids.txt): 4 chains of 2 (N1->N5, N2->N6, N3->N7, N4->N8), jobs 43503348-55; GNPROF/tree fit_params md5 8a7d4c2d...; with the acceptance pipeline <= 5 running. Expected end ~13:00-14:00.

## 6 Nov 2026 — Acceptance pipeline EXP41C75 RESULT (read 6 Nov ~05:55; all 4 jobs COMPLETED 05:51): PASS -> the 6 Nov adoptions are done
- A1 fit 43486417 (1:28), FR 43486418 (0:35), H 43486419 (0:48), P 43486421 (0:15.5): COMPLETED. A2 every log: "Persistent
  JAX compilation cache ON at /sdf/scratch/neutrino/pgranger/jax_job_cache/pipe_EXP41C75 (mode=shared". A3 the four target
  dirs are /sdf/data/neutrino/pgranger/larnd-sim-jax/target_sci_full_EXP41C75{,FR,H,P}_..., none in the repo root. A4
  fit_result_staging/sci_full_EXP41C75H/polish_estimate.json written (self-checks passed; iteration 20000). A5 H end point
  provenance: data_seed 41, LARND_CHAIN_COARSE_ITER 7500, smear 1.0 / entry, LARND_TARGET_ROOT and the shared cache
  recorded in env_overrides.
- Descriptive: first epoch FR 3:23, H 2:06 (29 Oct cache test 3:20 / 2:15; per-job cache 9:50); pipeline 3.1 GPU-h.
  Polished error vs the quoted uncertainty (simulation; q = error / uncertainty, bias NOT subtracted): Ab -0.14 % (q -0.91),
  eField +0.108 (+2.48: its known +0.13 bias), tran_diff +1.32 (0.98), long_diff +4.55 (0.87), lifetime +4.92 (1.32); all
  |q| < 3. POLISH moved long_diff 5.79 -> 4.55 and lifetime 6.93 -> 4.92 on this sample (Newton step -1.14, -1.95).
- The H stage lost its fit_result link for ~1 min at 05:05 (the incident in "DXPROF RESULT"); no save fell in that window and
  the 20000 checkpoint and the POLISH outputs are in staging as designed.

## 6 Nov 2026 — GNPROF RESULT (read 6 Nov ~12:15; all 8 jobs COMPLETED 12:06): rule verdict NO VERDICT (V-c 4/8)
- Readout GNPROF/read_gn.py -> GNPROF/GNPROF_result.txt. 100 batches with a GN block on every sample; GN wall 89 s per batch.
  V-b PASS for both models (C: gram/vjp 5.8e-6, asym_gg 3.0e-4, asym_cg 1.7e-4). V-a at P0: C F_cc / FD H_cc = 1.000 1.039
  1.079 1.060 1.000 -> PASS, primary model C; A 1.64 2.61 1.86 2.63 1.59 (the expected Fisher ignores the +-2-tick matching
  window and counts ~2x the information). V-c: ridge-0 solves ok on every batch, but the per-sample widths at ridge 0 vs 1e-3
  agree within 10 % on 4/8 samples only -> FAIL -> NO VERDICT.
- What the rule would have read (descriptive only): estimator m (POLISH form at P0) SD over the 8 samples 0.105 0.0059 1.11
  1.77 3.10 %; SD(m) / width, primary GN-C full bread at ridge 0 (median width 0.103 0.0042 0.64 1.49 2.59 %): 1.02 1.42 1.73
  1.19 1.20 (4/5 in [0.49, 1.51]). The same ratio for the other breads: conditional H 2.94 19.7 2.96 2.67 3.31; observed
  dE/dx profile 1.05 16.8 2.13 2.29 1.30; GN-C dE/dx only 1.08 17.3 2.50 2.74 1.31; GN-C geometry only 2.75 1.87 2.49 1.75
  3.25; hybrid-C (H - Delta) 0.85 1.00 1.31 0.73 1.05; GN-A full 1.36 5.20 3.32 3.46 1.64 (ridge-stable to 0.2 %, but too
  narrow). The 6 Nov question ("why don't the Hessian / sandwich work although they should in theory") has its answer in this
  table: the conditional curvature ignores the freedom of the fitted nuisances (x3, eField x20); profiling it over dE/dx
  restores Ab and lifetime, profiling over the geometry restores eField; both together bring Ab, long_diff, lifetime to
  1.0-1.2 and leave eField 1.4 and tran_diff 1.7. Prediction (PARTLY: eField and lifetime close, tran_diff and/or long_diff
  above) was right on tran_diff, pessimistic on long_diff.
- POST HOC (GNPROF/diag_gn.py, diag_gn2.py; written after the readout, descriptive): the V-c failure is ONE batch per failing
  sample (s2 batch 81, s3 batch 13, s4 batch 87, s5 batch 6) where the exact-GGN nuisance block A_nn (indefinite: median 2,
  up to 12 non-positive eigenvalues per batch -- the -LSE matching term is concave in the fields) has an eigenvalue near 0 that
  the ridge moves across; Delta_b then jumps by 0.1-1.6 x the SAMPLE's F_cc diagonal. For any PSD joint curvature
  0 <= Delta_b <= F_cc,b (profiling removes at most the batch's own information). Clipping the generalised eigenvalues of
  Delta_b w.r.t. F_cc,b to [0, 1] (needed on 109 rho < 0 and 133 rho > 1 of 800 batches x 5) makes the per-sample widths
  ridge-stable (0 vs 1e-3: 8/8; 0 vs 1e-2: 7/8) and gives SD(m) / width 1.04 1.77 1.87 1.32 1.27 (3/5 in the band); the
  clipped hybrid 0.97 0.94 1.35 0.80 1.12 is NOT ridge-stable per sample (s7). Clipping does nothing to model A (1 of 4000).
- Reading: derivatives at the fit point DO give the width once the curvature is profiled over all fitted nuisances with
  first derivatives only (Gauss-Newton) and the loss's own curvature (C, not the expected Fisher A); the remaining factor on
  eField and tran_diff (1.4-1.9) is consistent with optimiser scatter of the nuisances (same-data replica spread; not a
  curvature effect, never visible to a derivative at one point) -- unmeasured on these true-end C75H fits. The bound rule is
  post hoc -> confirmation on fresh data, in the production configuration: GNREAL below.

## 6 Nov 2026 — GNREAL: does the bounded GN-C profile width, plus the measured optimiser scatter, cover the spread of the PRODUCTION estimate? (rules written 6 Nov 12:10-12:17, before submission)
Follows GNPROF RESULT (post hoc bound rule) and its stated "next = the same at the production (realistic-end) end points
against the n = 30 ensemble SD". Pierre 6 Nov: "Please apply your proposals" (plan item: check the GN profile against the
replica ensemble).
- Code: GNPROF/tree UNCHANGED (md5 8a7d4c2d...), shared with GNPROF (no copy). Runs sci_full_GR_EXP<s>, s = 29, 30, 31, 32
  (FRESH-set production end points EXP<s>C75H @ 20000, realistic 1 mm entry smear; members of the n = 30 POLISH set, each
  with 4 H-stage runs in F2). Inners = the GN_KN10N inners with only data_seed, the checkpoint, test_name and the two smear
  exports changed (= the production inner diff, KNOT/inner_sci_full_EXP29C75H.sh vs KN10N1C75H). 4 independent jobs, <= 4
  running, ~3.3 h each.
- Primary bread (fixed now): GN-C full, per-batch bound 0 <= Delta_b <= F_cc,b (generalised eigenvalues clipped to [0, 1]),
  ridge 0, raw-score meat; w = median over the 4 samples of 100 jac sqrt(diag(B^-1 J B^-1)).
- Reference (frozen before reading): REF30 = SD of the polished m over the n = 30 production end points = 0.156 0.043 1.323
  5.245 3.738 % (UQ/polish_table.txt; the readout recomputes it and must reproduce it); sigma_w = within-sample spread of 4
  H-stage runs = 0.097 0.006 0.637 2.459 1.974 % (UQ/F2_result.txt; H-only replicas = full replicas, T2).
- Checks: V-b model C as GNPROF. V-a at P0: C F_cc / FD H_cc diag median in [0.8, 1.25] on Ab, tran, long, lifetime. V-c:
  ridge-0 solves ok on >= 95 % of batches and the bounded widths at ridge 0 vs 1e-3 within 10 % on all 5 on >= 3/4 samples.
  V2: m from the run = the production POLISH m of that sample (D_F dump) within 0.05 conditional sigma on all 5, >= 3/4.
  Any check failing -> NO VERDICT.
- **Primary:** R_tot = REF30 / sqrt(w^2 + sigma_w^2), all 5 parameters, band [0.74, 1.26] (chi2_29). CONFIRMED 5/5; PARTLY
  3-4; NOT 0-2.
- Descriptive: R_stat = REF30 / w (no optimiser scatter; sigma_w is an end-point scatter, the scatter of m may be smaller,
  so the truth lies between R_tot and R_stat); w(realistic) / w(true ends, GNPROF clipped median 0.1004 0.0033 0.593 1.341
  2.436): does the derivative width see the smear? (the true-to-realistic SD of m grows eField x7, long_diff x3, Ab x1.5);
  unbounded GN-C, hybrid, model A, the one-block breads; robust scale of the n = 30 m; per-sample z.
- Prediction: NOT/PARTLY borderline, 2-3 in the band. Ab and lifetime in the band; tran_diff (~1.5) and long_diff (~1.9,
  n = 30 spread carries the 2/12 +16 tail) above; eField open: the per-track entry smear is independent between batches, so
  the between-batch score spread should carry it -- then w(eField) grows ~x10; if the smear acts coherently (its -1.4 mm
  inward offset) it is a bias and stays invisible (R_tot ~6). Stand-in check (readout run on the true-end GNPROF records):
  R_tot 1.11 6.26 1.50 1.87 1.24.
- Meaning: eField in the band -> derivative widths see input-geometry uncertainty that is independent between tracks;
  Ab/lifetime in the band -> for those two a per-fit width (GN profile + a few H-only replicas for sigma_opt) can stand in
  for the simulated ensemble. In every case the pooled ensemble stays the quoted uncertainty until a CONFIRMED.
- Reading order: when all 4 are COMPLETED. Readout GNREAL/read_gr.py (dry mode on GNSMOKE OK; full path exercised on the
  true-end records as stand-ins: REF30 reproduced, V2 correctly FAILs there).
- Submitted 6 Nov 12:17:56 (GNREAL/submit_gr.sh; gr_jobs.txt, setGR_ids.txt): jobs 43541099-102 (GR_EXP29-32), independent,
  GNPROF/tree fit_params md5 8a7d4c2d... checked by the script. Expected end ~15:30-16:30.

## 6 Nov 2026 — GNREAL RESULT (read 6 Nov 15:45; all 4 jobs COMPLETED 15:43): NOT CONFIRMED (2/5), all checks PASS
- Readout GNREAL/read_gr.py -> GNREAL/GNREAL_result.txt. REF30 reproduced from the dumps. V-b C PASS (gram/vjp 6.6e-6). V-a C at
  P0 1.01 1.08 1.13 1.06 1.01 PASS (A 1.7-2.9 again). V-c 4/4 PASS: the per-batch bound (GNPROF post hoc rule) is ridge-stable
  on fresh production fits (w(1e-3)/w(0) 0.991-1.013; rho < 0 on 9-20, rho > 1 on 11-29 of 100 batches) -> the stability fix
  is CONFIRMED out of sample. V2 4/4 PASS (|dm| / sigma_c <= 0.021: the run sits exactly at the production POLISH point).
- **Primary R_tot = REF30 / sqrt(w^2 + sigma_w^2) = 1.06 4.38 1.30 1.48 1.04** (band [0.74, 1.26]) -> Ab and lifetime in,
  tran_diff just above, long_diff and eField above -> 2/5 -> NOT CONFIRMED. Prediction (2-3/5, Ab and lifetime in, tran_diff
  and long_diff above, eField open) right; eField resolved to "not seen".
- w (bounded GN-C, median of 4) = 0.111 0.0078 0.79 2.54 3.00 %; R_stat (no optimiser scatter) 1.41 5.53 1.67 2.07 1.25.
  w realistic / w true ends 1.10 2.36 1.34 1.89 1.23: the derivative width DOES grow with the realistic smear, but eField
  only x2.4 of the x7 growth of its SD (long_diff x1.9 of x3). Per-sample z of eField +2.5 +1.0 +0.4 +4.0: its smear-driven
  sample-to-sample spread acts coherently within a sample (bias-like), so the between-batch score spread cannot carry it.
  Robust scale of the n = 30 m / sqrt(w^2 + sigma_w^2) = 1.08 4.00 1.64 1.78 1.10 (tran_diff, long_diff shortfall is not a
  tail effect). Model A and every one-block bread too narrow (R_tot 1.2-7.2).
- POST HOC (GNREAL/diag_gr.py): the UNBOUNDED GN-C full reads R_tot 0.78 2.21 1.15 0.90 0.72, but it is not usable: on s29
  and s32 its widths change x3-7 between ridges 0 / 1e-3 / 1e-2 (single blown-up batches, as in GNPROF); the median is
  carried by those samples.
- Reading: derivative-based widths at the fit point, profiled over all fitted nuisances (GN-C + bound) and combined with the
  replica-measured optimiser scatter, reproduce the production spread of Ab and lifetime; they miss part of tran_diff (x1.3)
  and long_diff (x1.5) and most of eField (x4.4: the endpoint smear acts per sample). As pre-registered, the pooled ensemble
  stays the quoted uncertainty. The derivative thread (DXPROF -> GNPROF -> GNREAL) is complete; nothing further launched.
