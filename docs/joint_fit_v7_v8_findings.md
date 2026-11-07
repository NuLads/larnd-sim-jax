# Joint Calibration — Findings (v7 → v9)

**Date:** 2026-07-05
**Scope:** everything since `joint_fit_v6_analysis.md`. Covers the dEdx mean-constraint
fix, the long_diff investigation (why it fails, what it needs), the position-resolution
study, the `chain_update_freq` discovery, the vectorized chain-warp performance fix, and
the N-D joint-fit failure + the kb-fixed follow-up.

---

## 0. Executive summary

1. **Re-enabling `dedx_mean_constraint_weight=100000` fixed the worst 1-D failures.** In the
   1-D fits (one detector parameter floated, the other 8 pinned at truth) it took lifetime
   from +40% → +4%, kb −14% → −1%, Ab +2% → +0.1%, with no cost elsewhere. This restored a
   setting that had been used in the older position-fixed dEdx fits and was accidentally
   left at 0 in the joint lineage.
2. **long_diff is recoverable, but only with near-exact geometry.** It is genuinely
   observable (LUT-encoded, differentiable) but weak (~1–3% effect on the self-triggered
   signal) and strongly degenerate with the track drift geometry. Exact positions → −10%;
   imperfect (260 µm) positions → −80%.
3. **The 260 µm position "floor" was self-inflicted.** It came from `chain_update_freq=5`
   (a speed throttle) starving the geometry fit. Updating every step (`freq=1`) reaches
   ~40 µm and brings long_diff within ~5%.
4. **The chain-warp code was un-vectorized.** Rewriting it (branch `perf/vectorized-chain-warp`)
   gives a validated 1.8× speedup on position-heavy fits, so `freq=1` now costs what the
   throttled `freq=5` used to.
5. **The full N-D joint fit (all params at once) fails** — calibration↔geometry coupling and
   residual degeneracies cause a coupled divergence (positions → 777 µm). Fixing kb (the
   Ab↔kb degeneracy) is being tested next (nd_v9).

---

## 1. The dEdx mean-constraint fix (1-D, v7)

`dedx_mean_constraint_weight` was `0.0` in all v5/v6 joint runs but `100000` in the older
position-fixed dEdx fits. It length-weights the mean dEdx to a physical MIP value (1.887
MeV/cm), anchoring the overall charge normalization. Without it, lifetime and dEdx jointly
rescale charge (runaway lifetime), and kb is absorbed by the dEdx profile.

**1-D results, constraint ON (v7) vs OFF (v6), each parameter floated with the other 8 at truth:**

| Param | v7 (ON) | v6 (OFF) |
|-------|---------|----------|
| Ab | +0.1% | +2.0% |
| kb | −1.1% | −14.2% |
| eField | +0.1% | +0.0% |
| tran_diff | −1.8% | −2.1% |
| lifetime | +3.9% | +39.9% |
| shift_z / x / y | −1.8 / −4.7 / +7.2% | −7.6 / −9.8 / +11.3% |
| long_diff | −79.3% | −82.1% |

Mechanism confirmed: mean dEdx stays pinned at 1.886 (v7) vs drifting to 1.94 (v6). The fix
works with or without chain fitting (v7nc no-chain lifetime = +3.6%), so it is a
charge-normalization effect, independent of geometry. Only long_diff remained broken.

---

## 2. Why long_diff fails — and what it actually needs

### 2a. It is faithfully modelled, just weak
Longitudinal diffusion is **built into the signal LUT**: at setup the base induced-current
response `(45,45,1891)` is convolved along the time axis with 100 Gaussians of widths
`long_diff_template = linspace(0.001,10,100)` → a `(100,45,45,1891)` template stack. During
simulation each electron's σ_L = √(2·long_diff·t_drift) selects and **quadratically
interpolates** between templates (fully differentiable). This is done identically for target
and reconstruction. (Note: `mc_diff=False` and `diffusion_in_current_sim=False` in these runs,
so neither the stochastic-displacement path nor the analytic current-model path is used — it
is the LUT template path.)

### 2b. The signal is intrinsically tiny (fidelity check)
The tracks probe σ_L ≈ 0–2.8 ticks (median 2.4), spanning ~27 grid points — the LUT grid is
**not** the bottleneck. But convolving an already-broad current pulse with a ~1–3 tick
Gaussian barely changes the self-triggered ADC:

| σ_L (drift) | change in induced current vs no diffusion |
|-------------|--------------------------------------------|
| 0.5 ticks (near anode) | 0.3% |
| 2.0 ticks | 2.4% |
| 2.8 ticks (max drift) | ~3.4% |

The gradient long_diff sees is ~1%, below the ±4–6% stochastic-sim noise. It is a genuine
physical limitation of self-triggered pixel readout, faithfully captured.

### 2c. But it *is* observable — the true-position test
Fitting long_diff with **true geometry** (zero position error) recovers it to **−10%** (still
climbing), versus **−79%** with the chain-fitted ~260 µm geometry. A 70 pp swing from geometry
alone (tran_diff swings only ~5 pp). So long_diff is not "blind" — its weak signal is swamped
by the temporal-width error that imperfect geometry injects.

### 2d. Why tran_diff survives imperfect geometry but long_diff doesn't
The distinction is **first-order vs charge-conserving second-order**, not spatial vs temporal:
- **tran_diff** redistributes charge *across pixels* → first-order per-pixel ADC change (grad ~4.4).
- **eField** changes charge (recombination) *and* shifts arrival time (v_drift) → first-order (grad ~7700). (This is why the time-domain term in the loss recovers eField beautifully.)
- **long_diff** only broadens the in-pixel time profile at fixed charge → second-order (grad ~0.3),
  and it shares the temporal-width observable with the track's drift-direction geometry, so it is
  the one parameter degenerate with position error.

---

## 3. How good must positions be? (basic study)

Since σ_L ∝ √(long_diff·t_drift), a per-node drift-z scatter creates a spurious temporal width:

| long_diff target precision | required per-node z-scatter |
|----------------------------|------------------------------|
| 10% | < 20 µm |
| 25% | < 49 µm |
| 50% | < 99 µm |
| factor-2 | < 198 µm |

The fundamental limit is *not* 260 µm: the linear-guess reconstruction already localizes the
track start to ~57 µm, and per-hit arrival-time resolution is ~30–125 µm depending on electron
count. So the readout carries the information; the chain fit simply wasn't using it.

---

## 4. The `chain_update_freq` discovery + position scan (pos_study)

The v7 long_diff run used `chain_update_freq=5` — the chain gradient updated every 5th step,
so positions got 1000 updates in 5000 steps. Switching to **`freq=1`** (every step) took the
**identical** configuration from **260 µm → 63 µm** and long_diff from **−79% → −11%**.

A scan of chain aggressiveness (all with `freq=1`, long_diff + dEdx + positions, other 8
detector params at truth):

| MCS prior | chain LR | pos residual | long_diff |
|-----------|----------|--------------|-----------|
| 0.5 | 1e-4 | 63 µm | −10.3% |
| 0.1 | 1e-4 | 44 µm | −6.4% |
| **0.5** | **5e-4** | **37 µm** | **−4.4%** |
| 0.1 | 5e-4 | 43 µm | −4.7% |
| 0.02 | 1e-3 | 41 µm | −6.2% |
| 0.5 | 1e-3 | 43 µm | −5.8% |
| *v7 baseline* | *freq=5* | *260 µm* | *−79%* |

**Sweet spot: `chain_lr=5e-4`** (37–43 µm, long_diff −4 to −5%, converges in budget). Maximally
aggressive settings (very low MCS) give no benefit and run slower.

![long_diff vs position accuracy](../plots/longdiff_vs_position.png)

**Answer to "can we fit positions well enough for long_diff?": yes** — `freq=1`, `chain_lr≈5e-4`
reaches ~40 µm and brings long_diff within ~5%, with no loss or detector-model changes.

---

## 5. Performance: vectorized chain warp

**Root cause of the chain slowdown** (vs shifts/dEdx, which are single vectorized ops): the
warp was a **Python `for` loop over ~33 tracks**, each doing a `lax.scan` + ~10 full-array
scatter writes + in-loop array conversions → ~330 scatters + 33 scans unrolled per batch, plus
recompilation because `n_chain`/track-count vary per batch.

**Fix** (branch `perf/vectorized-chain-warp`, committed): batched implementation — node
positions via one `cumsum` (absolute chain), all fields written with one scatter per column,
MCS prior via masked batched diffs.

- **Correctness:** validated vs the old loop — positions 1.9e-6, gradients 3.8e-5, MCS 7.6e-8
  relative; real-fit trajectory matches (residual 105 µm, long_diff 0.61 at iter 90).
- **Speed:** **1.8×** on the 1-param + chain path (10.5 → 5.8 s/step at `freq=1`). So `freq=1`
  now costs about what the throttled `freq=5` used to.
- **Caveat:** the N-D step gains less (~1.1×) because the full-sim backward dominates there;
  the warp is a smaller slice. The remaining chain cost (backward through geometry+LUT) is
  fundamental. A further win — padding to fully static shapes to kill per-batch recompilation —
  is a larger, separate change.

**Status:** on a branch, not merged. Needs review before merging to `dedx_soft_barrier`/main.

---

## 6. The N-D joint fit fails (nd_v7, nd_v8)

Fitting **all 9 detector parameters simultaneously** (realistic calibration) collapses, and
the optimized geometry settings make it *worse*:

| param | nd_v8 (freq1, clr5e-4, vec) | nd_v7 (freq5) | 1-D v7 (others pinned) |
|-------|------------------------------|---------------|------------------------|
| Ab | −9.9% | −8.6% | +0.1% |
| kb | −48.4% | −50.7% | −1.1% |
| eField | −0.5% | +0.2% | +0.1% |
| tran_diff | −47.7% | −12.2% | −1.8% |
| long_diff | +25.8% | −37.0% | −79.3% |
| lifetime | +71.8% | +44.2% | +3.9% |
| shift_x | −97.8% | +14.9% | −4.7% |
| **pos residual** | **777 µm** | 361 µm | ~40–260 µm |

**Why:** the 1-D successes relied on the other 8 params being pinned at truth. With everything
floating, (a) the calibration degeneracies return (kb↔dEdx, lifetime↔normalization,
**Ab↔kb**), and (b) the wildly-wrong calibration corrupts the geometry fit — the freed
positions absorb the calibration errors and wander to 777 µm. The aggressive geometry settings
that shine when calibration is known **backfire** when it isn't. It is a coupled divergence:
bad calibration → bad positions → bad calibration.

---

## 7. nd_v9 (kb fixed) — the Ab↔kb degeneracy was poisoning the N-D fit

kb is essentially degenerate with Ab in this setup, so kb is **held fixed at truth** and
removed from the floated parameters (`param_list_nd_v9.yaml`, 8 params). All other settings
identical to nd_v8 (job 30686153).

**Preliminary (iter 400 only — still running):** fixing kb transforms the joint fit. Already at
iter 400, most parameters are far closer to truth than nd_v8 reached at 5000:

| param | nd_v9 @400 (kb fixed) | nd_v8 final (kb floated) |
|-------|------------------------|--------------------------|
| Ab | −1.3% | −9.9% |
| eField | −0.0% | −0.5% |
| tran_diff | −14.1% (converging) | −47.7% |
| lifetime | **+0.8%** | +71.8% |
| shift_x | −7.3% | −97.8% |
| long_diff | −53.2% (lags positions) | +25.8% |
| pos residual | 530 µm (early) | 777 µm |

**Update (iter 1200) — the iter-400 snapshot was too optimistic.** Two distinct behaviours:
- **Charge/timing parameters are robustly fixed by removing kb:** Ab (−0.3%) and eField (−0.0%)
  are rock-solid throughout. Breaking Ab↔kb genuinely fixed the charge-degeneracy family.
- **The geometry fit is still unstable:** the position residual *oscillates* (1324 → 646 → 107 →
  476 → 218 → 394 µm over the run) rather than converging, and this drags the geometry-coupled
  parameters with it — at iter 1200 tran_diff −56%, long_diff +52%, shift_x +23%, and even
  lifetime swung back to −21%. The lucky iter-400 snapshot caught a moment mid-oscillation.

**Interpretation:** the aggressive geometry settings (`chain_lr=5e-4`, `MCS=0.1`) that were ideal
for the 1-D pos_study (calibration *known*) are **too hot for the N-D case**, where residual
calibration uncertainty couples to the freed positions and drives an oscillation. Fixing kb
removed one degeneracy but not the position↔calibration coupling.

**Recommended next run:** N-D with kb fixed **and gentler geometry** (lower `chain_lr` ≈ 1e-4,
higher `MCS` ≈ 0.5, still `freq=1`) so the positions converge smoothly instead of oscillating —
or a staged schedule (calibrate with stiff geometry, then relax). *(Final nd_v9 numbers to be
appended when it completes; the LR schedule decays, so it may partially stabilise by iter 5000.)*

---

## 8. Open questions / recommended next steps

- **Staged fitting for the N-D case.** nd_v8 argues against everything-at-once. Try: fit
  calibration first with geometry tightly regularized (high MCS, low chain LR) to break the
  charge degeneracies, then relax geometry — or alternate. nd_v9 (kb fixed) is the first step.
- **Fix the remaining degeneracies** beyond kb: lifetime↔normalization and residual
  Ab/dEdx coupling likely still need constraints or a wider observable (e.g. Bragg-peak tracks
  for recombination).
- **long_diff:** accept that it needs `freq=1` / ~40 µm geometry *and* known calibration; in the
  full joint fit it will only come in once the geometry stops diverging.
- **Merge decision** on `perf/vectorized-chain-warp`.

---

## Run reference

| Run | What | Key setting | Outcome |
|-----|------|-------------|---------|
| v6 / v6nc | 1-D, constraint OFF | mean-c=0 | lifetime +40%, kb −14% |
| v7 / v7nc | 1-D, constraint ON | mean-c=1e5 | lifetime +4%, kb −1% |
| truepos_test | 1-D long_diff, true geometry | no chain | long_diff −10% |
| pos_study | 1-D long_diff, chain scan | freq=1 | ~40 µm, long_diff −5% |
| bench_vec | perf benchmark | vectorized warp | 1.8× |
| nd_v7 | N-D all params | freq=5 | collapse |
| nd_v8 | N-D all params | freq=1, vec | collapse, 777 µm |
| nd_v9 | N-D minus kb | freq=1, vec | *running* |
