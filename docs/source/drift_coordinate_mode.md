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
second-order Taylor expansion at ±10 % eField is off by up to 2000 %. The sweep is dominated by a convention: deposits
are placed relative to the **cathode** (`t0 = d_cathode / v(E)`), so a change of E shifts the whole event in time by
∝ L/v(E) and sweeps near-anode pulses (the sharpest) at ≈ 0.5·d_cathode/(v·t_s) ≈ 900 ticks per unit ln E.

## What it does (defaults since `taylor/efield-u0`, Nov 2026)

Within a drift volume all charge shares one `v_drift(E)`, so the drift distance `u = z` is a common,
eField-independent coordinate. In this mode:

- **Placement** is indexed by drift distance (`get_vdrift_placement` returns the fixed u-grid unit): eField no longer
  moves anything against the grid. Lifetime and diffusion keep the full eField dependence (`drifting_jax`); the
  long-diffusion template row is selected in u-grid units.
- The **response template is used as is** on the u-grid. By Ramo's theorem the charge induced per unit drift distance
  does not depend on the velocity, so the u-grid template *is* the correct one: the pulse is automatically 1/r shorter
  in real time (`r = v(E)/v_ref`).
- The **FEE clocks are fixed in real time**: the ADC/dead-time windows span `interval * r` u-ticks (fractional), read
  off the cumulative charge with a **C² prefiltered cubic B-spline** (`drift_mode_c2`; Keys is only C¹ and every one of
  its window-edge knots sits at r = 1, which made the AD Hessian one-sided at the nominal point).
- **Ticks** convert about the event time (`drift_mode_event_time`): `tick_t = tick_u / r`, i.e. the drift time
  `d_anode / v(E)` plus the in-pulse lag compressed by 1/r. (The template's index 0 is the event time: it starts with
  the charge at the cathode and each deposit is shifted by its distance to the cathode.)

At the reference field (`r = 1`) the mode reproduces the default one to float32 round-off (`check_efield_u0.py`,
NERSC A100): waveforms max |Δ| 0.02 of 1.5e4 (~1e-6 relative, the frozen u-grid unit's rounding; the legacy mode
shows the same 0.03), first-hit tick ≤ 6e-5, ADC ≤ 8e-6, hit probability ≤ 4e-7. On identical waveforms the C²
windows reproduce the integer ones to 1.6e-5 in log-probability. The eField Jacobian agrees between the vmap and
chunked FEE paths (40.7712 vs 40.7713), and the second derivative is continuous through `r = 1` (−123.00, −122.99,
−122.99 at `d = −1e-5, 0, +1e-5`).

## Validation (Nov 2026, NERSC A100, current FEE, this branch)

`larndsim-taylor`: `run_efield_fine_scan.py` (6 cases), `run_efield_propagation.py` (7 cases, 500 draws); see
`EFIELD_TAYLOR_FIX.md` there. Observables: λ-weighted first-hit and all-hit charge, hit-probability sum (N1),
charge-weighted first-hit tick.

| | default t-grid | drift mode (this) |
|---|---:|---:|
| forward wiggle rms over ±1 % eField (% of nominal) | up to 4e-2 | ≤ 1.3e-4 (4 cases), ≤ 1.1e-3 (2) |
| AD eField Jacobian / forward slope (24 entries) | −37 … +7 | 1.00 (20), 0.95–1.03 (3), 0.57 (1) |
| 2nd-order Taylor error at ±10 % eField (% of nominal) | up to 1955 | ≤ 0.34 |
| eField-only propagation, band / own forward MC, charges (σ_E = 1–3 %) | 0.8–155 | 0.96–1.04 |

The 0.57 is N1 of one event, a response of ~1e-3 of nominal; the two cases with the larger wiggle have a period of
0.67 / 1.0 % of eField, one u-tick of a 150 / 100 u-tick FEE window, which tilts the local derivative. Timing is 1.00
everywhere.

Against an independent **physical t-grid reference** (`enable_physical_time_grid`, single clock phase, `run_efield_propagation.py
--truth tP`): band / MC at σ_E = 1–3 % is 0.95–0.98 (first-hit total charge), 1.03–1.20 (all-hit charge), 0.93–0.98
(mean hit charge), residual ≤ 0.066 %. The offset is the u-grid sampling the FEE in drift distance rather than in real
time (largest for shallow, sharp pulses). Against stochastic hit data from that reference (dress rehearsal), the drift-mode
timing has χ²/n = 1.25–1.37 at the true parameters.

**Time scale.** With physical timing, eField acts on hit times as a pure scale (`dt/t = −0.50 dE/E` at 0.5 kV/cm): a fit
that floats a time scale cannot separate the two. Measured in the dress rehearsal, `σ(dE/E)² = (0.021 %)² + (2 σ_s)²` for
an external time-scale anchor `σ_s` of 1e-3, 1e-4 and 1e-5. The default t-grid's tighter constraint came from the
cathode-referenced time axis.

## Related t-grid options

- `enable_anode_referenced_time(params)`: `t0 = L/v_ref − d_anode/v(E)` — no E-dependent event-time shift, hit times
  with the physical sign, a ≈ 30× smaller sweep for near-anode tracks. Identical at the current eField.
- `enable_physical_time_grid(params)`: anode-referenced time **and** the Ramo-scaled template (compressed by 1/r about
  the collection sample) — the clocked-detector forward reference and stochastic-data generator. Its derivatives still
  see the physical sweep: differentiate the drift mode instead.

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
