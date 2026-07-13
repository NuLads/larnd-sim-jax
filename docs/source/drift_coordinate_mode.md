# Drift-coordinate (u = z grid) mode

**Opt-in, default off.** Enable with:

```python
from larndsim.consts_jax import enable_drift_coordinate_mode, get_tick_time_scale
params = enable_drift_coordinate_mode(params)   # freezes the u-grid unit at the current v_drift
...
tick_time = ticks / get_tick_time_scale(params) # FEE tick outputs are u-ticks in this mode
```

## Why

In the default pipeline, the electric field moves every pulse against the electronics sampling grid
(`t0 = D / v_drift(E)`). For the differentiable (average-noise) surrogate this makes the **second
derivative w.r.t. eField pathological**: sub-tick-sharp structures (the piecewise-linear placement
interpolation and the noise-width threshold sigmoids) are swept past the observable with an
amplification `(dt0/dE)^2 ~ 1e6`, so the AD Hessian over-shoots the true response curvature by
factors of 10^3-10^4 while the true response is small and smooth. Ensemble dithering, derivative
temperatures and beam-width changes were all measured to be unable to fix this at fixed architecture.

## What it does

Within a drift volume all charge shares one `v_drift(E)`, so the drift distance `u = z` is a common,
eField-independent coordinate. In this mode:

- **Placement** is indexed by drift distance (`get_vdrift_placement` returns the fixed u-grid unit)
  — eField no longer moves anything against the grid. Lifetime and diffusion keep the full
  eField dependence (they use `v_drift(E)` as before, via `drifting_jax`).
- The time-domain **response template** appears stretched by `r = v(E)/v_ref` on the u-grid: it is
  resampled once per parameter set (Keys cubic, anchored at the arrival sample, exact at r = 1),
  with amplitude `1/r` for charge conservation.
- **FEE tick outputs are u-ticks**; convert to time ticks with `get_tick_time_scale(params)`
  (analytic and smooth in eField).

At the reference field (`r = 1`) the forward simulation is **bit-identical** to the default mode.

## Validation (6 events, eField ±10 %, average-noise surrogate)

| | default mode | drift-coordinate mode |
|---|---:|---:|
| `\|H_AD / C_true\|` (charge) | ~7300× | **1.4×** |
| `\|H_AD / C_true\|` (tick) | ~33× | **0.99×** |
| 2nd-order Taylor error @ ±10 % (charge) | ~12800 % | **1.5 %** |
| nominal forward vs default | — | identical (≤1e-4 %) |

## Scope and caveats

- Implemented for the `simulate_wfs` → average-noise FEE path (the differentiable surrogate).
- The FEE clock windows (ADC hold/dead time) are **fixed in REAL time** (fractional in u-ticks:
  `interval * r`, Keys-cubic interpolation of the cumulative charge at the window edges, hoisted out
  of the hit scan so the cost matches the integer-window step). This matters: keeping the windows
  integer in u-ticks would make their real-time duration scale as `1/r`, producing a spurious
  **wrong-sign** eField charge trend (~−2 % per +10 % eField instead of the physical +1–5 % from
  recombination + lifetime). The real-time windows restore the physical trend on every validated
  event while keeping the corrected Hessian.
- The eField dependence of observables flows exclusively through amplitudes (recombination,
  lifetime, diffusion), one smooth template resample, and the analytic tick conversion — which is
  what makes the eField Jacobian *and* Hessian correct.
- Derivative evaluations w.r.t. eField pay the template-resample cost (~×5 in the current form);
  other parameters are unaffected (r ≡ 1 in their tangents only through eField).
