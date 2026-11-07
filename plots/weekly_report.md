# larnd-sim-jax — Joint Detector Calibration & Compute Performance
### Week summary (≈ 2026-06-30 → 2026-07-07)

This note consolidates a week of work on the differentiable `larnd-sim-jax` fitter along two
parallel threads: **(A)** getting the joint detector-calibration fit to converge, and **(B)** reducing
per-step compute. The headline is a much clearer physics picture — the joint calibration mostly
works, and the one remaining geometry problem has been traced to a specific, well-understood cause —
plus several committed, result-preserving speedups.

> **Methodology note.** All parameter numbers below are time-averaged over a window of iterations
> (≥300) with an explicit drift/slope check, *not* single end-points. Where a run is not yet converged
> (non-zero drift) this is stated explicitly and no cross-run conclusion is drawn.

---

## Part A — Physics: joint detector calibration

### A.1 Setup
The fit jointly optimises, from a differentiable simulation:
- **detector-calibration parameters**: `Ab`, `kb` (recombination), `eField`, `tran_diff`, `long_diff`,
  `lifetime`, and the global `shift_{x,y,z}`;
- **per-segment dEdx** nuisances;
- **per-track chain positions** (track geometry, via per-node deflection angles).

Input geometry is a straight-line **linear-guess reconstruction**; the target waveforms are simulated
at the **true** positions with known physics. The position mismatch between the two is the thing the
chain fit must correct.

### A.2 The two enabling fixes
| Fix | Effect |
|-----|--------|
| **Mean-dEdx constraint** (weight 1e5) — anchors charge normalisation | 1-D failures fixed: lifetime **+41% → +4%**, kb **−14% → −2%**, Ab **+2% → +0%** (see A.3) |
| **Fixing `kb`** (degenerate with `Ab` in our regime) | Essential for the N-D fit; without it the joint fit collapsed (lifetime +72%, kb −48%) |

### A.3 One-dimensional fits (single calibration parameter + dEdx + positions)
Fitting each parameter *individually* validates the method and isolates the mean-dEdx constraint.
Values are time-averaged over the last 300 iters (± = oscillation std; `~` = still drifting):

| parameter | mean-constraint OFF (+pos) | mean-constraint ON (+pos) | mean ON, positions fixed |
|-----------|----------------------------|----------------------------|---------------------------|
| Ab        | +2 ± 1% | **+0 ± 0%** | +0 ± 0% |
| kb        | −14 ± 4% | **−2 ± 1%** | −3 ± 1% |
| eField    | +0 ± 0% | +0 ± 0% | +0 ± 0% |
| tran_diff | −5 ± 3% | −4 ± 3% | −6 ± 2% |
| **long_diff** | −82 ± 0% | **−80 ± 0%** | −82 ± 0% |
| lifetime  | +41 ± 6% | **+4 ± 4%** | +4 ± 2% |
| shift_x   | −5 ± 52%~ | +2 ± 42% | +5 ± 36%~ |
| shift_y   | +12 ± 38% | +12 ± 42% | +14 ± 48%~ |
| shift_z   | +3 ± 64% | +9 ± 57%~ | +8 ± 63% |

**Findings:**
- **The mean-dEdx constraint is decisive** for the charge-coupled parameters: lifetime **+41 → +4%**,
  kb **−14 → −2%**, Ab **+2 → +0%**. This is the single most important fix.
- **eField and tran_diff** are recovered cleanly throughout (±0–6%).
- **Positions on vs off (last two columns) barely changes the 1-D result** — when only one calibration
  parameter is free, it is largely separable from the geometry. The position↔calibration entanglement
  is an *N-D* phenomenon (below).

![1D fits (config v7): per-parameter convergence vs truth (blue = trajectory, dashed = truth, orange band = last-300 mean ± std).](fit_1d_convergence.png)

> **Important caveat on `long_diff` — a `chain_update_freq` regression (now fixed).** The table above
> (config v7) shows long_diff at −80%, but this is **not** a fundamental 1-D limit: v7 used
> `chain_update_freq=5` (a ~2.5× speed optimisation that updates the chain gradient only every 5th step),
> which *starves the position fit* — and long_diff, the most position-sensitive parameter, is hit hardest.
> A clean run at **`chain_update_freq=1` + mean constraint (v8) recovers long_diff to −5.7 ± 1.1%** —
> essentially the ideal (−5% with known calibration), a ~14× improvement. Progression: freq=5 → −80%;
> freq=1 no-constraint → −24%; **freq=1 + constraint → −5.7%**. So the long_diff/shift rows in the v7
> table should be read as freq=5-limited, not intrinsic; all N-D fits use freq=1 (hence nd_v12's +8%).

**Effect of fitting positions** (mean-constraint on): the charge family (Ab, eField, lifetime) is
unchanged, while geometry-coupled parameters (tran_diff, shifts, long_diff) improve by ~1–3 pp — the
physically-expected direction. Shown with a 1-epoch (33-iter) rolling average:

![1D fits, positions fitted (blue) vs fixed (red), 1-epoch rolling average; raw faint. Charge params overlap; geometry params sit closer to truth with positions.](fit_1d_pos_vs_nopos_rolling.png)

### A.4 N-D joint-fit results (converged, iter 5000)
| parameter | nd_v11 (shifts fitted) | nd_v12 (shifts fixed) |
|-----------|------------------------|------------------------|
| Ab        | −1.6% | −1.0% |
| eField    | +0.0% | −0.0% |
| tran_diff | −0.9% | −4.2% |
| **long_diff** | **−50%** | **+8 ± 8%** |
| lifetime  | +15% | +27% |
| pos residual | 284 ± 200 µm | 223 ± 200 µm |

**Key finding — the shift ↔ long_diff coupling.** With the global shifts *fitted* (nd_v11), long_diff
is badly biased (−50%); with the shifts *held fixed* (nd_v12), long_diff recovers (**+8%**) at
essentially the same position residual. The global shifts and the chain geometry both move the track
in the drift direction, so when shifts float they inject exactly the noise long_diff is sensitive to.
There is a genuine trade-off: one can measure the shifts **or** long_diff cleanly, but the joint fit
struggles with both at once.

**Net:** with kb fixed + shifts fixed + mean-constraint, **four of five core parameters land within a
few percent** (Ab, eField, tran_diff, long_diff). `lifetime` (+27%) is the remaining straggler.

![Best converged N-D fit (nd_v12): per-parameter % bias and position residual vs iteration (33-iter rolling average). Ab/eField/tran_diff/long_diff settle within a few %; lifetime is the straggler.](fit_nd_v12_convergence.png)

#### Full N-D configuration sweep
All tested N-D configurations (time-averaged, last-300; `~` = still drifting). Three generations by
*fitted set*: **+kb+shifts** (kb degenerate), **kb fixed / shifts fitted**, **kb+shifts fixed (core-5)**.

| run | fitted set | geometry knobs | Ab | eField | tran_diff | long_diff | lifetime | resid µm |
|-----|-----------|----------------|----|--------|-----------|-----------|----------|----------|
| v7  | +kb +shifts | mcs0.5 clr1e-4 | −9% | +0% | −14% | −38%~ | +45% | 269 |
| v7nc | +kb +shifts, **no pos** | — | −8% | +0% | −10% | −45%~ | +51% | — |
| v8  | +kb +shifts | mcs0.1 clr5e-4 | −10% | −0% | −54% | +76%~ | +72% | 505 |
| v9  | shifts (kb fixed) | mcs0.1 clr5e-4 | −3% | −0% | −37%~ | +66% | +119%~ | 576 |
| v10 | shifts | mcs0.5 clr1e-4 | −4% | −0% | −21%~ | −58%~ | +73%~ | 262 |
| staged | shifts | mcs0.5 clr1e-4, chain@2000 | −3% | +0% | −12%~ | −44%~ | +105%~ | 282 |
| fastdecay | shifts | mcs0.5 clr1e-4, decay | −3% | −0% | −19%~ | −51%~ | +57%~ | 230 |
| v11 | shifts | mcs0.5 clr1e-4 decay0.999 | −2% | +0% | −2% | −54%~ | +17%~ | 282 |
| **v12** | **core-5** | mcs0.5 clr1e-4 decay0.999 | **−1%** | **−0%** | **−3%** | **+8%** | +27%~ | **223** |
| v13 | core-5 | mcs0.5 clr1e-4 decay0.995 | −2% | −0% | +8% | −69% | +53% | 302 |
| v14 | core-5 | mcs0.5 clr1e-4 chain@2000 | −2% | +0% | −0% | −1%~ | +47%~ | 245 |
| v15 | core-5 | mcs0.5 clr3e-5 decay0.999 | −1% | −0% | +3%~ | −63%~ | +34%~ | 276 |
| v16† | core-5 | mcs**0.02** clr5e-4 | −4% | −0% | −20%~ | +57%~ | +131%~ | 368 |
| v17† | core-5 | mcs**0.02** clr1e-3 | −2% | +0% | −1% | +68% | +130%~ | 586 |
| v18† | core-5, **bs400** | mcs0.02 clr5e-4 | −1% | +0% | −7% | −29%~ | +2%~ | 1007 |

† not yet converged (v16/v17 @ 3200, v18 @ 800) — listed for completeness, **not** used for conclusions.
Where `kb` is fitted (v7/v7nc/v8) it lands at **−49 to −51%**, confirming the Ab↔kb degeneracy that
motivated fixing it. Fitted shifts (v7–v11) are noisy (±10–40%).

**Reading the sweep:** the first generation fails on the kb degeneracy; fixing kb (v9+) recovers Ab/eField
but leaves long_diff/lifetime swinging; adding gentle geometry + LR decay + fixing shifts (**v12**) gives
the best converged result. The v13–v15 damping variants (faster decay, staged, lower LR) did **not**
improve on v12 — consistent with the reframing below (there was no temporal oscillation to damp).

### A.5 The "position oscillation" was a misread — it's outlier tracks
The per-step `pos_residual` trace looks like a wild oscillation (64 → 1154 µm). It is not temporal:
the metric logs the *just-processed batch's* residual each step, so plotting it in sequence cycles
through all 33 tracks.

![Position-residual "oscillation": trace, autocorrelation, parameter trajectories, and one track's geometry.](chain_oscillation.png)

Cross-checks confirm the reframing:
- **Per-track residuals are a static spread**, not a time signal: min 47, median 126, mean 198,
  max 1151 µm. **10/33 tracks < 100 µm**, **4/33 stuck > 400 µm**.
- **Node motion is small** (~20–30 µm RMS, slow monotonic settling — not fast swinging), and grows
  from the fixed anchor (≈0) toward the far end (a cantilever effect). So the residual is a *static*
  per-track error, not a time-oscillation.

![Reframed: per-track residual distribution; the worst track (b0, 1151 µm) vs a good track (b7, 80 µm).](track_residual_outliers.png)

The stability is best seen by contrasting a well-fit track with a stuck one over the full run. The
straight track (b28) converges and holds ~40 µm; the curved outlier (b0) is **stuck at ~1200 µm the
entire run** — in the chord-perpendicular frame (middle) its fitted chain sits on the straight chord
while the true track bows out ~2500 µm. It barely moves (~30 µm node RMS) but is ~1200 µm wrong: a
*stable* geometry that is simply the wrong (chord-cut) one.

![Curved outlier (b0, top) vs straight track (b28, bottom). Left: residual vs iteration (curved stuck at ~1200 µm; straight ~40 µm). Middle: chord-frame shape — curved track's fit sits on the chord, truth bows out. Right: per-node RMS motion vs final residual.](chain_dynamics_two.png)

So a handful of tracks are **stuck at wrong geometry**, dragging up the mean residual and injecting
noise into the shared calibration parameters (this noise *is* the apparent parameter jitter).

### A.6 Root cause: curved tracks, and the chain cuts chords
The residual correlates **+0.95** with the true-track curvature (sagitta = max deviation from the
straight chord):

| track | residual | true curvature (sagitta) | length | nodes |
|-------|----------|--------------------------|--------|-------|
| b0 (worst) | 1151 µm | **2491 µm** | 49 cm | 24 |
| b30 | 569 µm | 1343 µm | 44 cm | 21 |
| b8  | 503 µm | 1563 µm | 36 cm | 18 |
| b28 (best) | 47 µm | 0 µm (straight) | 12 cm | 6 |

Straight tracks (sagitta ≈ 0) fit to ~47 µm; genuinely curved tracks are the outliers. The fitted
chain **cuts a chord** through the curve, deviating most in the middle:

![The 3 worst tracks in the chord-perpendicular plane: true tracks (green) bow out 0.8–2.5 mm; the fitted chain (red) cuts across.](outlier_chordcut.png)

### A.7 Why — it's a convergence problem, not the prior
The chain *can* bend — so why doesn't it follow the curve? The evidence now points overwhelmingly to
**slow convergence, not the MCS prior clamping curvature**. Holding the prior **fixed at mcs 0.5** and
varying only the optimiser / learning rate (position-only, same tracks) collapses the curved-track
residual by ~5×:

| setup (mcs 0.5 fixed, position-only) | curved-track residual | lever |
|--------------------------------------|-----------------------|-------|
| Adam, clr 1e-4 (joint-fit setting) | **331 µm** @3000 (99 µm @5000) | — (non-convergence) |
| Adam, clr 1e-3 (higher LR) | **66 µm** | #2 |
| Adam, clr 1e-3, 400 cm batches | **61 µm** (by iter 1500) | #1 |
| L-BFGS (second-order) | **56 µm** in ~30 steps | #4 |

At the **same prior**, higher LR, bigger batches, and L-BFGS all reach ~55–66 µm. So the prior's true
floor on curved tracks is ~60 µm (only slightly above straight tracks' ~50 µm), and the dramatic
"chord-cutting" (331 µm) was simply **Adam failing to converge at clr 1e-4**, not the prior clamping
real curvature. (The prior *is* the Highland MCS NLL with σ fixed at 3 GeV; that over-clamping is real
but small — it accounts for only the ~10 µm curved-vs-straight gap at the floor, not the chord-cutting.)

**And the prior plays a different, essential role — breaking the geometry↔calibration
degeneracy** — now **confirmed on converged, drift-checked runs** (last-300 average, drift ≈ 0):

| joint fit (converged) | residual | long_diff | lifetime |
|-----------------------|----------|-----------|----------|
| **prior ON** (mcs 0.5) | 223 µm | **+8%** | **+27%** |
| prior off (mcs 0.02, clr 5e-4) | 388 µm | +58% | +134% |
| prior off (mcs 0.02, clr 1e-3) | 642 µm | +72% | +141% |

With the prior off, the flexible chain **absorbs calibration error** and settles into a much worse
equilibrium (long_diff +58–72% vs +8%, lifetime +130–140% vs +27%). So the prior is **not** a geometry
regulariser (position-only shows the data handles that) — it is the **degeneracy-breaker**, and it must
stay.

**Resolution:** keep the MCS prior (essential for degeneracy-breaking) and treat the chord-cutting as a
**convergence problem** — fixed cheaply by a higher chain LR (#2) or bigger batches (#1), and best by a
second-order optimiser (#4, below). A momentum-aware σ could close the small residual curved-vs-straight
gap (~60 vs 50 µm) but is now a *secondary refinement*, not the main fix.

### A.8 Fitting long/curved tracks faster — the optimizer (#4)
The curved tracks converge slowly because (i) each track is updated only once per epoch (~150 updates
over 5000 iters), (ii) the anchored-angle parametrization is ill-conditioned (far-node positions are a
cumulative sum of upstream angles), and (iii) Adam is first-order. A **second-order optimizer (L-BFGS)**
targets (ii)+(iii). Benchmarked head-to-head on a curved track (position-only, calibration frozen,
identical loss and start):

| reach residual | Adam | **L-BFGS** | L-BFGS advantage |
|----------------|------|------------|------------------|
| 200 µm | 181 steps, 96 s | **18 steps, 44 s** | 10× fewer iters, 2.2× faster |
| 100 µm | 227 steps, 121 s | **25 steps, 50 s** | 9× fewer iters, 2.4× faster |
| 80 µm  | 249 steps, 132 s | **29 steps, 54 s** | 8.5× fewer iters, 2.4× faster |

![Adam vs L-BFGS on a curved track: L-BFGS converges in ~25 steps vs Adam's ~230 (left), and is ~2.4× faster in wall-clock (middle) despite its costlier line search.](chain_opt_bench.png)

**L-BFGS converges ~9× faster per iteration and ~2.4× faster in wall-clock** — even though each L-BFGS
step costs ~6× more (eager Armijo line search: 3.1 s vs 0.53 s).

**End-to-end (whole dataset, position-only, mcs 0.5):** replacing Adam with L-BFGS + early stopping as
the actual geometry optimiser gives, on one pass over the batches:

| | **L-BFGS** | Adam (5000 iters) |
|---|-----------|-------------------|
| wall time | **59 min** | ~3.5 h |
| curved-track residual | **66 µm** | 99 µm |
| median | 42 µm | 42 µm |

**≈3.5× faster *and* better curved-track geometry** — L-BFGS reaches the prior's true ~60 µm floor that
Adam couldn't reach in 3.5 h. The wall-clock advantage is **conservative**: the line search is un-jitted
only because the sim uses a dynamic-size `jnp.unique`; making that jittable would widen it further.

### A.9 Physics status
- **Solved:** charge-calibration family (Ab, kb, eField) and, with shifts fixed, `long_diff` (+8%).
- **Confirmed:** the MCS prior is essential in the joint fit (degeneracy-breaker); the curved-track
  chord-cutting is a **convergence problem**, not the prior — L-BFGS fixes it (~3.5× faster end-to-end,
  better geometry).
- **Open:** `lifetime` bias (+27%); optional momentum-aware σ (secondary); fold L-BFGS +
  early stopping into the position fit end-to-end.

---

## Part B — Compute performance

### B.1 Diagnosis
Op-level profiling showed the GPU is **~4% utilised** — the fitter is **host-bound**. The dominant,
*variable* cost is XLA (re)compilation; the stable host cost is per-track Python post-processing.

### B.2 Committed, result-preserving optimisations
| Change | Effect | Notes |
|--------|--------|-------|
| Vectorised per-track chain warp + MCS | **1.8×** on position steps | validated bitwise-identical |
| Per-track chain-grad **de-sync** (jnp.where vs per-track `if isnan`) | **0.52 s/step (1.21×)** | removed 33 device→host syncs/step; bitwise-identical |
| Position-residual diagnostic **throttle** (`--pos_residual_freq`) | ~0.5 s/step | pure diagnostic, now sub-sampled |
| **Persistent cache** threshold `min_compile_time_secs` 1.0 → 0.0 | 439 → **7823** cached kernels | fast (~40 ms) sim kernels now persisted; cold-start recompiles once, then reused |

**Ruled out:** caching the `value_and_grad` closure (`cache_grad`) — **1.01×** (no speedup, warm *or*
cold); the cost is in the eager post-processing, not re-tracing.

### B.3 Batch size & memory (A100 **40 GB**, preallocation off → true usage)
| max_batch_len | peak memory | % of 40 GB | ~batches |
|---------------|-------------|-----------|----------|
| 50 cm (current) | **7.7 GB** | 19% | 33 |
| 100 cm | 13.8 GB | 34% | ~17 |
| 200 cm | 18.0 GB | 44% | ~9 |
| 400 cm | 24.1 GB | 59% | ~5 |
| 800 cm | 36.2 GB | 88% | ~2–3 |

![Real GPU memory and batch count vs max_batch_len. Left: memory scales sub-linearly, 40 GB ceiling ~800 cm. Right: batches/epoch drop from 33 to ~5 at the 400 cm target.](batchsize_memory.png)

We were **massively under-using the GPU** — the "30 GB" seen earlier was just JAX preallocating 75%.
Real usage at 50 cm is 7.7 GB, and memory scales sub-linearly. **~400 cm (24 GB, ~5 batches)** is a
comfortable aggressive target: ~5× fewer distinct shapes (less recompilation), near-full-batch
gradients (less noise), and more statistics. Caveat: `max_batch_len` also *filters out* tracks longer
than it and *packs events* into batches, so raising it pulls in more long/curved tracks — beneficial
only once the curvature issue (A.6) is resolved. A 400 cm run (v18) is currently running cleanly (no OOM).

---

## Part C — Running experiments (as of report time)
| Run | Purpose | Setting |
|-----|---------|---------|
| nd_v16 / nd_v17 | prior-off, 50 cm — isolate the MCS prior | mcs 0.02, clr 5e-4 / 1e-3, no decay |
| nd_v18 | prior-off, **400 cm** — batch-size effect | mcs 0.02, clr 5e-4, 400 cm |

All target the same converged comparison against nd_v12.

## Part D — Next steps
1. **Converged** prior-on vs prior-off comparison (v16/v17/v18 to iter 5000), on time-averaged metrics.
2. If the prior still helps calibration → implement **momentum-aware MCS σ** (per-track), the principled
   fix that keeps the degeneracy-breaking constraint without chord-cutting.
3. Address the residual **lifetime** bias (lifetime ↔ charge-normalisation coupling).
4. Adopt **~400 cm batches** once curved tracks are well-fit (speed + statistics + fewer recompiles).
