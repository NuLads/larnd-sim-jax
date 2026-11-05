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

At the reference field (`r = 1`) the waveforms are bitwise those of the default mode and the FEE outputs agree to the
B-spline's ~1e-7.

## Validation (Nov 2026, 6 cases, NERSC A100, current FEE)

Prototype measurements (`larndsim-taylor`: `run_efield_fine_scan.py`, `run_efield_physref.py`,
`run_efield_propagation.py`; see `EFIELD_TAYLOR_FIX.md` there). Observables: λ-weighted first-hit and all-hit charge,
hit-probability sum, charge-weighted first-hit tick.

| | default t-grid | drift mode (this) |
|---|---:|---:|
| forward wiggle rms over ±1 % eField (% of nominal) | up to 4e-2 | ≤ 1.2e-3 |
| AD eField Jacobian / forward slope | −37 … +7 | 1.00 (19/24), 0.96–1.02 (4) |
| 2nd-order Taylor error at ±10 % eField (% of nominal) | up to 1955 | ≤ 0.33 |
| eField-only propagation, band / MC (σ_E = 1–3 %) | 0.8–155 | 0.96–1.04 |

It agrees with an independent **physical t-grid reference** (`enable_physical_time_grid`, 32-phase clock average):
timing ≤ 5.5 %, all-hit charge ≤ 0.65 points, first-hit charge ≤ 1.8 points of the response at ±10 % eField. The
residual is the u-grid sampling the FEE in drift distance rather than in real time (largest for shallow, sharp pulses).

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
