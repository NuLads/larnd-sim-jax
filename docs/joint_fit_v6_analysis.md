# Joint Calibration Fit — Analysis of the v6 Runs

**Date:** 2026-07-02
**Runs analysed:**
- `joint_1d_v6` — full joint fit (calibration parameter + per-segment dEdx nuisance + chain track positions), chain positions updated every 5th step (`--chain_update_freq 5`).
- `joint_1d_v6nc` — identical but **without** chain position fitting (geometry frozen at the linear-guess reconstruction).

Each run is a SLURM array of 9 tasks, one detector parameter fit per task (single-parameter fits, *not* a simultaneous multi-parameter fit). Both arrays completed 5000/5000 iterations cleanly.

- **Simulation input:** `linear_guess_segments.h5` (reconstructed / imperfect track geometry).
- **Target:** `true_through_muon_edep_10cm_vol1cm.h5` (waveforms at *true* geometry and true physics).
- **Sample:** 33 tracks (see the selection note in §5 — the sample is *unfiltered*).

---

## 1. Headline result: chain fitting reduces bias for 7 of 9 parameters

Final fitted value = mean of the last 50 iterations. Bias is relative to the injected target.

| Parameter | Target | v6 (chain) | v6nc (no chain) | v6 bias | v6nc bias | Chain helps? |
|-----------|--------|-----------|-----------------|---------|-----------|--------------|
| Ab        | 0.83488   | 0.85175   | 0.85334   | +2.02 % | +2.21 % | ✓ slightly |
| kb        | 0.056464  | 0.048439  | 0.047746  | −14.21 % | −15.44 % | ✓ |
| eField    | 0.50488   | 0.50501   | 0.50510   | +0.02 % | +0.04 % | ✓ (both excellent) |
| tran_diff | 9.488e-6  | 9.288e-6  | 8.839e-6  | −2.11 % | −6.84 % | ✓✓ (3× better) |
| long_diff | 5.842e-6  | 1.048e-6  | 1.017e-6  | −82.07 % | −82.59 % | — both fail |
| lifetime  | 2969.7    | 4154.3    | 4052.7    | +39.89 % | +36.47 % | — both fail |
| shift_z   | 0.021478  | 0.019845  | 0.019421  | −7.60 % | −9.58 % | ✓ |
| shift_x   | 0.021478  | 0.019364  | 0.018121  | −9.84 % | −15.63 % | ✓✓ |
| shift_y   | 0.021478  | 0.023910  | 0.024902  | +11.32 % | +15.94 % | ✓✓ |

The largest gains are exactly on the geometry-sensitive parameters (tran_diff, the shifts). `long_diff` and `lifetime` are badly biased **independent of chain fitting** — see §4.

![Failure-mode and position-fitting overview](../plots/v6_failure_and_position_analysis.png)

*Top-left:* failure-mode trajectories (normalised to target). *Top-right:* parameter sensitivity (gradient magnitude). *Bottom-left:* chain position residual convergence. *Bottom-right:* absolute-bias comparison.

---

## 2. Position fitting: ~81 % residual reduction

The chain fit drives the per-track geometry from the linear-guess error down toward the true positions. The reduction is remarkably uniform across all 9 runs (79–81 %), i.e. essentially decoupled from which detector parameter is being co-fit.

| | Position residual |
|---|---|
| Linear-guess start (= **no-chain**, frozen for the whole v6nc run) | 0.138 cm (1376 µm) |
| After chain fitting (v6, converged) | 0.026 cm (~260 µm) |
| **Reduction** | **~81 %** |

`v6nc` has no position residual to report — it never fits geometry, so it stays pinned at 1376 µm for the entire run. That frozen 1.4 mm geometry error is what propagates into the larger biases seen in the geometry-sensitive parameters.

---

## 3. The shift "bias" is misleading — the shifts are actually measured well

The shift parameters look biased at 7–11 %, but that is a percentage of a **deliberately tiny 215 µm injected shift**. In absolute terms the shifts are pinned to ~20 µm — about **10× below** the per-track geometry residual. This is the √N averaging expected for a global parameter shared across ~33 tracks.

| Parameter | Absolute error | Per-track residual | Ratio |
|-----------|----------------|--------------------|-------|
| shift_x | −21.1 µm | 266 µm | 0.08 |
| shift_y | +24.3 µm | 205 µm | 0.12 |
| shift_z | −16.3 µm | 306 µm | 0.05 |

**Why it averages down:** the per-track reconstruction error is essentially zero-mean and independent across tracks — the coherent (common) component is only ~5 µm (statistically consistent with zero), while the random scatter is ~57 µm/coordinate and averages to ~10 µm over 33 tracks.

| Anchor-error component (per coordinate) | x | y | z |
|-----------------------------------------|---|---|---|
| Coherent mean | +3.8 µm | +3.6 µm | +7.8 µm |
| Random scatter | 68.8 µm | 48.6 µm | 53.0 µm |
| Significance of the mean | 0.3σ | 0.4σ | 0.8σ |

Reference scales: σ/√N_tracks ≈ 46 µm, σ/√N_nodes ≈ 0.8 µm. The observed ~20 µm sits between these because nodes within a track are highly correlated (anchored, smooth trajectory), so the effective sample size is ~the number of tracks, not the number of nodes.

![Shift statistics](../plots/v6_shift_statistics.png)

**Practical implication:** the shift fitting works well. A realistic 1–2 mm injected shift would show this same ~20 µm error as a ~1–2 % bias. The demo just used a shift smaller than the method's ~20 µm systematic, making a good absolute result read as a bad relative one.

---

## 4. The failures (long_diff, lifetime, kb) are systematic, not statistical

Test: does the parameter **wander around** the truth (statistical → more data helps) or **sit at / drift toward** a wrong value (systematic → more data just converges tighter to the wrong answer)? Discriminator = bias magnitude vs. stochastic fluctuation amplitude.

| Parameter | Fluctuation (stochastic noise) | Distance to truth | Drifting? | Diagnosis |
|-----------|-------------------------------|-------------------|-----------|-----------|
| eField    | ±0.3 % | 0.0 % | plateaued | at truth ✓ |
| Ab        | ±0.8 % | 2.0 % (~2.5σ) | plateaued | mild systematic |
| **kb**    | ±4.2 % | 13.5 % (~3σ) | plateaued | systematic offset |
| **lifetime** | ±5.5 % | 37 % | **still climbing away** | systematic drift |
| **long_diff** | ±0.0 % | 82 % | pinned at floor | insensitive |

The bias is several times larger than the noise for all three failures, so **more identical tracks would not fix them.** The gradient-magnitude ranking (parameter sensitivity) lines up with the bias almost perfectly:

| Parameter | median \|grad\| | final bias |
|-----------|-----------------|-----------|
| eField | 7691 | +0.02 % |
| shift_z/y/x | ~1000 | 7–11 % |
| Ab | 27 | +2 % |
| kb | 10 | −14 % |
| tran_diff | 3.2 | −2 % |
| lifetime | 0.78 | +40 % |
| long_diff | 0.10 | −82 % |

**Root causes and what would actually help:**

- **kb** — degeneracy with the co-fit per-segment dEdx nuisance (both control collected charge through the same recombination term). It even passed *through* the true value (0.0564 vs 0.0565) at iteration 1538 before the dEdx freedom pulled it back. Fix: tighten the dEdx prior or freeze dEdx when fitting kb — not more tracks.
- **lifetime** — sits in a degenerate direction with charge normalization / dEdx and keeps drifting upward past truth (crossed 2970 exactly at iteration 652, kept climbing to 4154). Fix: a lifetime prior, or breaking the charge-normalisation degeneracy. *(Note: an earlier hypothesis blamed a narrow drift lever-arm from the track selection — that was wrong, see §5.)*
- **long_diff** — genuine insensitivity. Gradient ~75,000× weaker than eField; with no electronics noise and coarse time sampling the waveform barely encodes longitudinal charge spread, so the likelihood is nearly flat and the parameter rails to its floor. Fix: a more informative observable (finer time binning, include noise), not more tracks.

**One caveat in favour of more data:** the ±4–6 % fluctuation shows the probabilistic simulator injects real gradient noise; a larger batch would reduce that noise and could shave a few percent of stochastic bias off kb. It cannot touch the structural degeneracies or the flat likelihood.

---

## 5. Bug found: the track selection is never applied

The scripts passed `--track_len_sel 2`, `--max_abs_costheta_sel 0.966`, `--min_abs_segz_sel 15.` and `--track_z_bound 28`, but **none of these took effect.** The entire selection block in `optimize/dataio.py:249–273` is gated on `if live_selection:`, and `live_selection` defaults to `False` and was never set. The code falls through to the `else` branch and uses **all tracks**.

Evidence from the actual run:

| | Value |
|---|---|
| Tracks used | 33 (everything in the file) |
| Drift coord \|z\| range | 0 → 30.4 cm |
| If `min_abs_segz` (\|z\|<15) were applied | only 29 % of segments survive |
| If `track_z_bound` (\|z_end\|<28) were applied | only **7 of 33 tracks** survive |

Consequences:

1. The sample being fit is **not** the sample the settings describe.
2. It invalidated an earlier lifetime hypothesis: the data actually spans a *wide* drift range (|z| 0–30 cm, per-track z-span median ~11 cm), so lifetime has a good drift lever-arm and its bias is not a lever-arm problem.
3. **Naming inversion:** even if enabled, line 250 applies `min_abs_segz_sel` as `abs(z) < 15` — a *maximum* |z| cut, the opposite of what the "min" name implies.

**Action taken:** the four dead selection arguments were removed from `start_fit_1d_joint_v5.sh`, `start_fit_1d_joint_v6.sh` and `start_fit_1d_joint_v6nc.sh` so the scripts no longer advertise cuts that don't run. (The selection *code* itself was left untouched; enabling `--live_selection` would still trigger the `min`/`max` naming inversion and would shrink the sample to ~7 tracks.)

---

## 6. Summary

- The joint fit demonstration **works**: chain fitting corrects track geometry from ~1.4 mm to ~260 µm and reduces bias for 7 of 9 parameters.
- The shift parameters are measured to ~20 µm absolute (10× below the per-track residual) — the √N averaging behaves as expected; the large *percentage* is an artefact of a tiny injected signal.
- `long_diff`, `lifetime`, `kb` fail for **systematic** reasons (flat likelihood / degeneracies), not statistics. More tracks of the same kind will not fix them; the fixes are observable- and prior-side.
- The runtime track selection was inert (`live_selection=False`); the dead arguments have been removed from the scripts.

### Open follow-ups (not yet done)
- Test the kb↔dEdx degeneracy directly by refitting kb with the dEdx nuisance frozen.
- Decide whether to enable `--live_selection` (and fix the `min`/`max` naming) — note it shrinks the sample to ~7 tracks.
- Consider a larger injected shift (1–2 mm) for a cleaner shift-recovery demonstration.
