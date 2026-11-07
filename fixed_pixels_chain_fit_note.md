# Note: fixed_pixels coverage in chain position fitting

**Date:** 2026-06-19  
**Status:** Known issue, left as-is for now. Worth revisiting if position fitting is extended to larger deflections or more tracks.

---

## What fixed_pixels is

`fixed_pixels` is a static sorted list of pixel IDs that the JAX simulation is compiled to run over. It exists because JAX needs a fixed-size array to JIT-compile the pixel routing step. All ionisation signal deposited outside this set is silently ignored.

It is built once per track as:

```
fixed_pixels = union(target_pixels, pixels_hit_by_initial_straight_chain)
```

padded to the next multiple of 128.

## Why this is fine in the main parameter fitting

In `fit_params.py`, the track geometry is fixed throughout the optimisation — only physics parameters (dEdx, recombination, gain…) are varied. The tracks never leave the pixels they occupied at initialisation, so `fixed_pixels` computed once from the simulation tracks is always valid.

## Why this is a problem in chain position fitting

In `run_chain_fit.py`, the chain geometry (angles θ₀, φ₀, δθᵢ, δφᵢ) is what is being optimised. As the chain curves away from the initial straight-line estimate, it can drift into pixels that are not in `fixed_pixels`. Signal the optimised chain deposits in those new pixels is not counted in the loss.

Concretely: the diagnostic run on 2026-06-19 showed that for ev=0 tr=4 (the only track with a real position gradient), the fitted chain endpoint displaced 0.25 cm (≈0.6 pixel pitches) from the initial straight-line exit. No target pixels were missed (target pixels are always in `fixed_pixels` by construction), but pixels at the far end of the curved chain that are not in the data are unaccounted for, creating a one-sided blind spot.

The arc-length shortfall due to curvature itself is negligible: at 3σ_MCS × √N total deflection (~23 mrad for a 49 cm, 30-segment track), the curved arc falls short of the chord by only 0.001 cm ≈ 0.003 pixel pitches.

## Proposed fixes (not yet implemented)

**A. Widen fixed_pixels at initialisation.** Simulate a few perturbed chains (e.g. deflected by ±3σ_MCS in θ₀/φ₀) and take the union of all hit pixels. One-time cost, no optimizer changes, slightly larger pixel set per step.

**B. Refresh fixed_pixels at stage boundaries.** After Stage 1 completes, rebuild `fixed_pixels` from the fitted chain before starting Stage 2. Cost is one extra forward pass (~0.5 s); re-JIT of the next stage's step function happens anyway at each stage transition. Fully correct for any deflection size.

**C. Use only target_pixels for routing.** Drop `guess_pix` from the union. The simulation only computes signal on pixels the data actually activated. Faster per step; loses the ability to penalise spurious off-track signal, but the MCS prior bounds excursions.

Option B is the cleanest and essentially free at stage boundaries.
