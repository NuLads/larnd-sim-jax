# Drift-coordinate (u = z grid) mode

**Opt-in, default off.** Enable with:

```python
from larndsim.consts_jax import enable_drift_coordinate_mode, u_ticks_to_time_ticks
params = enable_drift_coordinate_mode(params)          # freezes the u-grid unit at the current v_drift
...
tick_time = u_ticks_to_time_ticks(params, ticks, response.shape[-1])   # FEE tick outputs are u-ticks in this mode
```

Use it whenever derivatives with respect to **eField** are needed (Jacobians, Hessians, Taylor surrogates, fits).

## Why

In the default pipeline the electric field moves every pulse against the electronics sampling grid. The forward
response stays smooth at the percent scale (clock-phase spread ≲ 0.03 %), but its AD derivatives see the sub-tick
structure being swept (piecewise-linear placement, noise-width threshold sigmoids of the FEE): measured on the
current FEE, the eField Jacobian is off by up to 37× (and of the wrong sign), the Hessian by 10²–10⁴, and a
second-order Taylor expansion at ±10 % eField is off by up to 2000 %. Deposits are placed relative to the **cathode**
(`t0 = d_cathode / v(E)`), so a change of E shifts the whole event in time by ∝ L/v(E) and sweeps near-anode pulses (the
sharpest) at ≈ 0.5·d_cathode/(v·t_s) ≈ 900 ticks per unit ln E. That shift is unphysical, but removing it does not cure the
derivatives: with the anode anchor and the Ramo template (`enable_physical_time_grid`) AD/slope still spans −22 … +243
(8 of 24 entries within 5 %) and the second-order error reaches 1331 %, because the real pulse motion d_anode/v(E) sweeps
the clock too. The anchor only moves the worst case from near-anode to near-cathode tracks.

## What it does (defaults since `taylor/efield-u0`, Nov 2026)

Within a drift volume all charge shares one `v_drift(E)`, so the drift distance `u = z` is a common,
eField-independent coordinate. In this mode:

- **Placement** is indexed by drift distance (`get_vdrift_placement` returns the fixed u-grid unit): eField no longer
  moves anything against the grid. Lifetime and diffusion keep the full eField dependence (`drifting_jax`); the
  long-diffusion template row is selected in u-grid units.
- The **response template is used as is** on the u-grid. By Ramo's theorem the charge induced per unit drift distance
  does not depend on the velocity, so the u-grid template *is* the correct one: the pulse is automatically 1/r shorter
  in real time (`r = v(E)/v_ref`).
- The **FEE clocks are fixed in real time**: every FEE step scales with r. From the threshold crossing at tick `k`, the ADC
  reads at `k + 1/2 + r (interval + 1/2)`, the next window starts at `k + 1/2 + r (interval + 3/2)` and the reset ends at
  `k + 1/2 + r (2 interval + 3/2)`, in fractional u-ticks. These are read off the cumulative charge with a **C²
  prefiltered cubic B-spline** (`drift_mode_c2`; Keys is only C¹ and every one of its window-edge knots sits at r = 1, which
  made the AD Hessian one-sided at the nominal point).
- **Ticks** convert about the event time (`drift_mode_event_time`): `tick_t = c + (tick_u - c) / r`, i.e. the drift time
  `d_anode / v(E)` plus the in-pulse lag compressed by 1/r. (The template's index 0 is the event time: it starts with
  the charge at the cathode and each deposit is shifted by its distance to the cathode.) `c =
  params.drift_mode_tick_offset = -1.33` is **measured, not derived**: the hit-tick index carries a constant offset that
  must not be rescaled. It comes from the waveform and the FEE, varies by ± 0.09 ticks with the pulse shape, and only
  matters at the 0.03–0.06 tick level (3–6 ns) at ±10 % eField.

At the reference field (`r = 1`) the mode reproduces the default one to float32 round-off (`check_efield_u0.py`,
NERSC A100): waveforms max |Δ| 0.013 of 1.5e4 (~1e-6 relative, the frozen u-grid unit's rounding; the legacy mode
shows the same 0.012), first-hit tick ≤ 1.3e-4, ADC ≤ 2.3e-5, hit probability ≤ 1.8e-7. On identical waveforms the C²
windows reproduce the integer ones to 1.6e-5 in log-probability. The eField Jacobian agrees between the vmap and
chunked FEE paths (39.7594 vs 39.7593), and the second derivative is continuous through `r = 1` (−117.20, −117.18,
−117.18 at `d = −1e-5, 0, +1e-5`).

## Validation (Nov 2026, NERSC A100, current FEE, this branch at 421fa59)

`larndsim-taylor`: `run_efield_fine_scan.py` (6 cases), `run_efield_propagation.py` (7 cases, 500 draws),
`run_efield_factorial.py` (6 cases, 16 clock phases); see
`EFIELD_TAYLOR_FIX.md` there. Observables: λ-weighted first-hit and all-hit charge, hit-probability sum (N1),
charge-weighted first-hit tick.

| | default t-grid | drift mode (this) |
|---|---:|---:|
| forward wiggle rms over ±1 % eField (% of nominal) | up to 4e-2 | ≤ 1.3e-4 (4 cases), ≤ 1.2e-3 (2) |
| AD eField Jacobian / forward slope (24 entries) | −37 … +7 | 1.00 (19), 0.96–1.03 (4), 0.54 (1) |
| 2nd-order Taylor error at ±10 % eField (% of nominal) | up to 1955 | ≤ 0.35 |
| eField-only propagation, band / own forward MC, charges (σ_E = 1–3 %) | 0.8–155 | 0.97–1.04 |

The 0.54 is N1 of one event, a response of ~1e-3 of nominal. The two cases with the larger wiggle (events 18 and 23) were
first attributed here to a FEE window edge. That was wrong: they come from the hard edge of the FEE hit probability. The
smeared edge removes them, taking the eField Jacobian / slope on those cells from 0.545–1.033 to 0.995–1.001. Timing is 1.00
everywhere.

The table is for the FEE at 421fa59. Since 3ccfd99 the FEE default is the corrected noise cut (`LARND_FEE_SMEARED_EDGE`,
`LARND_FEE_CONDITIONAL_ADC`), the hybrid beam (`LARND_FEE_BEAM_MERGE`, `LARND_FEE_BEAM_SINGLE=0.5`) and the stochastic stop
(`LARND_FEE_STOCH_STOP`), each restored with `=0`. It changes nominal values, and it is faster: warm Hessian 0.73–0.80 of the
old time on this mode, with peak memory at or below the old one.

Against an independent **physical t-grid reference** (`enable_physical_time_grid`, `run_efield_propagation.py --truth tP`, single
clock phase): band / MC at σ_E = 1–3 % is 0.97–1.01 (first-hit total charge), 0.98–1.02 (all-hit charge), 0.98–1.02 (mean
hit charge), residual ≤ 0.045 %, where the 500-draw statistics are about 3 %. Averaged over 16 clock phases at ±10 % eField,
the drift mode is within 0.03 points of the reference for first-hit charge, 0.011 for all-hit charge, 0.016 for the
hit-probability sum, and 0.011 ticks for hit time, with the nominal point unchanged to 1e-6. Against stochastic hit data from
that reference (dress rehearsal, generated before 421fa59), the drift-mode timing has χ²/n = 1.25–1.37 at the true parameters.

Before 421fa59 the same comparison gave 0.95–0.98, 1.03–1.19 and 0.93–0.98, with forward gaps up to 0.15 points of first-hit
charge and 0.19 of all-hit charge. It was not the u-grid sampling the FEE in drift distance, as this note said earlier. Four
small offsets explain it (one more only at the −10 % edge for deep tracks):

| Cause | Mode | Correction | Effect at ±10 % eField |
|---|---|---|---|
| ADC window scaled `interval` but not its half-tick offsets, closing (r−1)/2 u-tick early | drift | `wpos = k + 1/2 + r (interval + 1/2)` | first-hit charge 0.09–0.11 points |
| Reset step stayed 1 u-tick, so dead time was 1 − 1/r too short | drift | `wpos1`, `fhe` above | all-hit charge 0.11–0.16 points |
| Tick conversion rescaled a constant offset | drift | `drift_mode_tick_offset` | hit time 0.03–0.06 ticks |
| Long diffusion row selected with v(E), then compressed by 1/r again | physical t-grid | row selected at `time_ref_vdrift` when `ramo_template` | first-hit charge up to 0.05 points |
| `t0` clipped at 0 for deposits beyond 28.84 cm at −10 % | physical t-grid | clip at −Nt when `anode_referenced_time`; cumulative lookups clip separately | 0.18 points of first-hit charge and 0.20 ticks, 28 cm at −10 % only |

**Time scale.** With physical timing, eField acts on hit times as a pure scale (`dt/t = −0.50 dE/E` at 0.5 kV/cm): a fit
that floats a time scale cannot separate the two. Measured in the dress rehearsal, `σ(dE/E)² = (0.021 %)² + (2 σ_s)²` for
an external time-scale anchor `σ_s` of 1e-3, 1e-4 and 1e-5. The default t-grid's tighter constraint came from the
cathode-referenced time axis.

## Related t-grid options

- `enable_anode_referenced_time(params)`: `t0 = L/v_ref − d_anode/v(E)` — no E-dependent event-time shift, hit times
  with the physical sign, a ≈ 30× smaller sweep for near-anode tracks. Identical at the current eField.
- `enable_physical_time_grid(params)`: anode-referenced time **and** the Ramo-scaled template (compressed by 1/r about
  the collection sample) — the clocked-detector forward reference and stochastic-data generator. The long-diffusion row is
  selected at `time_ref_vdrift`, so the pulse is compressed once. Its derivatives still see the physical sweep: differentiate
  the drift mode instead.

## Legacy behaviour (`enable_drift_coordinate_mode(params, legacy=True)`)

The July–Nov 2026 mode stretched the template by r (pulse fixed in *real* time — the t-grid's assumption, not Ramo),
anchored at the template end (~40 samples past the collection edge, so the pulse still moved ~22 u-ticks per unit
ln E), converted ticks about the template end (reproducing the cathode-referenced convention) and used Keys windows.
Its July validation (`|H/C_true|` 7300× → 1.4×) predates the garbage-row erratum of the Taylor study and was not
reproduced on the current FEE (measured: AD Jacobian/slope 0.21–2.2, 2nd-order error up to 3.4 %).

## Scope and caveats

- Implemented for the `simulate_wfs` → average-noise FEE path (the differentiable surrogate). The stochastic FEE
  (`get_adc_values`) has no drift-coordinate support: generate stochastic data with the t-grid
  (`enable_physical_time_grid`).
- Integer u-tick FEE windows would make their real-time duration scale as 1/r (a spurious charge trend); the
  fractional real-time windows are required.
- The u-grid unit must be the template's drift velocity (enable at the LUT field): a different unit mis-scales the
  template's built-in drift distance.

## Port onto `dedx_soft_barrier` (Nov 2026)

Branch `taylor/drift-coord-on-dedx` = `dedx_soft_barrier` @ `19d2bb3` + the three drift-coordinate commits;
`taylor/efield-u0` adds the defaults above. Interaction with the newer code:

- **FEE physics.** `_find_one_hit_step_frac` carries the `fcfdeb5` FEE fixes (`min_log_prob` floors,
  `jnp.where` in `log_diff_ndtr`, the unified Gaussian acceptance `log_ndtr((E - thr + excess_offset) /
  (sigma_uncorr * smooth_sigma_scale))`), term for term the `LARND_FEE_FAST=0` expressions of
  `_find_one_hit_step`.
- **Response cumsum cache.** `LARND_RESP_CUM_CACHE` passes the cumsum of the unresampled template; it is valid in the
  default drift mode and discarded (recomputed in-jit) for the legacy stretch and the physical t-grid.
- **FEE fast paths.** `LARND_FEE_PEEL0`, `LARND_FEE_PIXEL_STOP=2` and `=3` (`_fee_global_pixel_stop`, the
  default production path through `get_adc_values_average_noise_chunked`) hard-code integer windows;
  in drift-coordinate mode they are bypassed in favour of the generic scan (which keeps the
  `PIXEL_STOP` per-pixel-stop emulation, same semantics).
