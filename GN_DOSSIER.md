# Dossier: Gauss-Newton / second-order attempts on this fit — measurements only

This document records what was run and what came out. It contains no interpretation or
recommendation; assessing what the record means is your job.

## 1. The fit being polished

`larnd-sim-jax`: differentiable LArTPC simulation. A marked-Poisson-point-process negative
log-likelihood compares simulated to target waveforms.

- **5 calibration parameters**: `Ab` (recombination), `eField`, `lifetime`, `tran_diff`, `long_diff`.
- **~4000 per-segment dE/dx nuisances** (student-t prior, soft barrier, mean constraint).
- **Per-track geometry**: transverse displacement in a sine basis, ~3-12 modes/track, 864 tracks.
- Parametrization: `phys = low + (high-low)*sigmoid(theta)`, bounds in `optimize/ranges.py`.
  (Note: before commit 51cbfe9, 2026-07-25 18:22, the prototype used unbounded log-space
  `phys = nom*exp(theta)` instead. Records below straddle that change.)
- Optimizer in production: Adam, LR schedule decayed once per 100-iteration epoch.

## 2. Current end state that a polisher would start from (`ANNEALLONG`, 3 seeds x 10000 iter)

Robust estimate (trailing-20% median), % error vs truth:

| param | seed0 | seed1 | seed2 | mean +- sd |
|---|---|---|---|---|
| Ab | +0.67 | +0.94 | +0.64 | +0.75 +- 0.13 |
| eField | -0.03 | -0.05 | -0.02 | -0.03 +- 0.01 |
| tran_diff | -1.26 | -0.25 | -1.35 | -0.96 +- 0.50 |
| long_diff | -8.72 | -3.44 | -4.05 | -5.40 +- 2.36 |
| lifetime | -4.27 | -3.79 | -0.05 | -2.70 +- 1.89 |
| position residual | 157 um | 162 um | 183 um | 168 +- 11 um |

Measured at that end state, from logged gradients, using a block bootstrap (block=200):

| run | param | seed | tail-mean grad | block SE | t |
|---|---|---|---|---|---|
| ANNEALLONG | lifetime | 2 | +4.750 | 0.104 | +45.87 |
| ANNEALLONG | Ab | 0 | -10.91 | 0.453 | -24.11 |
| ANNEALLONG | lifetime | 1 | -4.523 | 0.423 | -10.70 |
| ANNEALLONG | Ab | 1 | -4.503 | 0.428 | -10.51 |
| ANNEALLONG | lifetime | 0 | -1.957 | 0.230 | -8.50 |
| ANNEALLONG | Ab | 2 | +1.519 | 0.246 | +6.18 |

Gradient autocorrelation in the tail, across 693 run/param entries: median +0.649 at lag 100
(the epoch period), +0.664 at lag 200. Fraction of entries with |t|>2, by parameter:
Ab 50%, eField 72%, tran_diff 54%, long_diff 58%, lifetime 45%.

At readout the calibration LR is ~0.0085% of its peak value.

## 3. Curvature measurements

Exact 5x5 Hessian of the PPP loss, evaluated at truth, S2 config @400 cm, 20 batches:
```
NLL(truth) = 166735.7      |grad| = 45658.73
eig 3.287e+03  Ab:-0.02 eField:-0.00 tran_diff:-0.00 long_diff:+0.56 lifetime:+0.83
eig 6.134e+03  Ab:-0.03 eField:-0.00 tran_diff:+0.01 long_diff:-0.83 lifetime:+0.56
eig 3.880e+05  tran_diff:-1.00
eig 7.953e+06  Ab:-1.00
eig 1.285e+10  eField:-1.00
sigma from H^-1 (log-space): Ab 0.06% eField 0.00% tran_diff 0.16% long_diff 1.44% lifetime 1.61%
```
Condition number ~4e6. Assembly is float32 (no `jax_enable_x64` anywhere in the tree);
float32 reciprocal machine epsilon is ~8.4e6.

8x8 Hessian adding 3 *rigid* geometry modes (dz, drift-scale, dx), 12 batches:
```
sigma (geom FIXED)  : Ab=1.88% eField=0.01% tran_diff=0.58% long_diff=4.22% lifetime=3.75%
sigma (geom MARGIN.): Ab=2.02% eField=1.56% tran_diff=1.29% long_diff=4.48% lifetime=3.79%
amplification       : Ab x1.1  eField x119.4  tran_diff x2.2  long_diff x1.1  lifetime x1.0
geom-mode sigmas    : dz=1.279e-4 cm  zscale=6.97e-4  dx=4.971e-5 cm
```

A separate comparison recorded that the Gauss-Newton / Fisher approximation differs from the
exact Hessian by a relative Frobenius norm of 1.75-4.27x, and that this ratio increased with
the number of batches used (Ab diagonal ratio 1.74 at 1 batch -> 4.88 at 7 batches).

Loss values from an objective cross-check (different config, len200):
```
NLL(init) = 299825.9   NLL(truth) = 76627.1   NLL(adam-final) = 77570.5   NLL(gn-runaway) = 285203.4
```

A 13x13 NLL scan over (lifetime, tran_diff) in log-offset units, step 0.1, other three
parameters held at truth, placed its minimum at the (0,0) grid point, i.e. at truth.

## 4. The prototype

`optimize/prototype_gn_calib.py`. Flags:
`--history` (checkpoint providing config+targets), `--steps` (default 40), `--refresh` (5),
`--ridge` (0.02), `--mu` (0.3), `--step_max` (0.08), `--w_time` (1.0), `--eps` (1e-8),
`--polyak` (0), `--fd_check` (default True), `--max_batches`, `--warm_start` (fraction),
`--objective residual|ppp`, `--out`.

- `residual`: LUCiD-style sqrt-charge + time least-squares, uses JtJ.
- `ppp`: the exact PPP/llhd loss used by the Adam fits; 5x5 Hessian via `jacfwd(grad)`.
- Implemented over the course of the work: saddle-free modification (uses |eigenvalues|),
  Levenberg-Marquardt lambda adaptation with step accept/reject on accumulated loss,
  a norm trust region, the bounded sigmoid parametrization, a bound-saturation guard that
  stops when parameters pin at range bounds, and "data-as-args" jitting (one compiled program
  shared across batches).
- All runs below used 20 batches. Runs from 33123555 onward log
  `readout noise removed from target params`.

## 5. Recorded outcomes, in chronological order

Final % error vs truth. Blank cells = the run did not reach a FINAL line.

| job | outcome |
|---|---|
| 32883051 | OOM (tried to allocate 4.73 GB) |
| 32883052 | OOM (80.3 GB) |
| 32883893 | OOM (8.28 GB) |
| 32883894 | Ab +1.61, eField -3.48, tran_diff +149.67, long_diff -68.54, lifetime -86.02 |
| 32888021 | `ConcretizationTypeError` (traced array int32[63097]) |
| 32889672 | OOM (18.2 GB) |
| 32920136 | (no FINAL line) |
| 32969978 | `numpy.linalg.LinAlgError: Eigenvalues did not converge` |
| 32970922 | Ab -12.51, eField -2.00, tran_diff -49.29, long_diff -93.89, lifetime **+1002.95** |
| 32973490 | `ZeroDivisionError: float division by zero` |
| 32973708 | Ab -15.26, eField -3.07, tran_diff -47.82, long_diff -90.91, lifetime **+580.07** |
| 33091040 | Ab -16.23, eField -2.83, tran_diff -59.69, long_diff -96.61, lifetime **+1722.52** |
| 33093395 | Ab -17.41, eField -3.52, tran_diff -46.99, long_diff -98.56, lifetime **+464430.04** |
| 33123555 | Ab -10.16, eField -3.80, tran_diff -30.99, long_diff -82.17, lifetime +148.25 |
| **33130436** | **warm start from Adam@20%**: start `{Ab 0.8326, eField 0.5211, tran_diff 1.004e-5, long_diff 5.136e-6, lifetime 2362}` -> FINAL **Ab -0.09, eField -0.00, tran_diff +0.55, long_diff -12.24, lifetime -1.03** |

A separate attempt to apply a matrix-free GGN/CG solver to the **position/geometry** block
recorded: a shape mismatch (`63488` vs `63489`) in the GGN chain path; and, in a later run, a
compilation failure `xla.gpu.CompilationResultProto exceeded maximum protobuf size of 2GB:
3084031029` followed by a rate of 445 s/iteration (ETA ~604 h for 5000 iterations).

## 6. What is available to restart from

`fit_result/sci_full_ANNEALLONG/history_iter10000_*_seed{0,1,2}.pkl`, each containing:

- `norm_params_state` and `current_params_state` — the 5 calibration parameters (normalised
  and physical). Example seed0: `Ab 0.840513, eField 0.521342, tran_diff 1.0e-05,
  long_diff 5.0e-06, lifetime 2300.603`.
- `chain_cache` — 100 batches x per-track spline coefficients (`angles`), 864 tracks total;
  `chain_contexts` gives `track_id`, `idxs`, `n_chain`, `step_len`, `total_len`, `x0_fixed`,
  `theta0_i`, `phi0_i`, `alpha_mid`.
- `dedx_cache` — 100 batches of per-segment `log_dedx` **each with its own Adam opt_state**;
  plus `batch_parent_ids` and `true_dedx_cache`.
- `opt_state` — the calibration Adam state; `total_iter = 10000`.

Restart mechanics as they currently stand:
- `--resume_from <ckpt>` (fit_type=chain only) restores `norm_params`, `current_params`,
  `opt_state`, `dedx_cache`, `batch_parent_ids`, `true_dedx_cache`. It does **not** read
  `chain_cache`; `save_checkpoint` writes that key but `load_checkpoint` never restores it.
- `LARND_CHAIN_INIT_FROM=<ckpt>` separately initialises chain coefficients from a checkpoint's
  `chain_cache`, keyed by `track_id`. Requires matching batching (`data_seed`, `max_nbatch`,
  `max_batch_len`).

Cost scale: one production fit is ~7-14 h on one A100 for 3 seeds. The Hessian assembly used
for section 3 ran in well under an hour.

## 7. Your task

Given the above, evaluate the options for taking **a few second-order (Gauss-Newton /
Newton / quasi-Newton / other) steps starting from the recorded ANNEALLONG end state**, with
the goal of improving final fit precision.

Be concrete and specific. For each option you propose, state: what exactly would be computed,
over which parameter block(s), what it would cost, what result would show it worked, and what
result would show it failed. Rank your options. Say explicitly which parts of the record above
you consider decisive for your ranking, and flag anything in it you think is unreliable or
insufficient to draw on.
