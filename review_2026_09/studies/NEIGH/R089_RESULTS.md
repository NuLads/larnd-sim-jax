# R089: is the coarse neighbor-pixel induction OK? (21 Oct 2026)

**Short answer: no, not for the calibration.** For the readout it is small compared with the noise. But at fixed true track inputs, data that carry the exact induction shift the production-model calibration by +23% in D_T, −27% in D_L and −10% in the lifetime. A_b and E stay within 0.11%.

## What the approximation is
`src/larndsim/sim_jax.py`: `simulate_drift_new` and `simulate_signals`.

- **Collecting pixel:** each cloud is split into 5x5 transverse bins, each with the longitudinal-diffusion template.
- **Other 80 pixels of the 9x9 block:** the whole cloud acts as a point at its center bin, with the template without diffusion (`response_template[0]`).

## Variants (`neigh_exact.py`, same drift, same pixel set, same 9x9 footprint)
- **prod:** production.
- **P:** production rebuilt by the new code. It matches prod exactly (sanity check).
- **L:** center-point induction, but with the diffusion template.
- **T:** each bin induces at its own position; no diffusion template off its own pixel.
- **E:** exact, i.e. T plus the diffusion template everywhere.

All variants are accumulated with the production `simulate_signals`, in chunks of 512 clouds.

## 1. Waveforms and probabilistic readout (`readout.json`)
Fig. 2 batch (`load(200,1,1)` batch 0, 926 pixels), E − prod:

- **Accumulated charge Q(t):**
  - largest change over time: median 318 e⁻, 99th percentile 1650 e⁻, max 2180 e⁻ (reset noise: 900 e⁻);
  - change at the threshold crossing: median 40 e⁻.
- **First-hit time:** 68% half-width 0.57 ticks, median +0.02 ticks. For comparison, two noise seeds differ by 9 ticks and the probabilistic model differs from throws by 0.08 ticks.
- **First-hit charge:** 68% half-width 0.11 ADC counts (2.2 between seeds; 0.02 for the model).
- **Hit probability:** largest change 0.24 on the first hit, 0.16 on the second (the model's own maximum is 0.08).
- **Expected number of hits:** −0.09%.

On the 48.5 cm test track: first-hit time median **+0.47 ticks** (68% half-width 0.76), and the largest charge change is 950 e⁻.

Almost all of this comes from the transverse part: T ≈ E, while L alone gives only 0.009 ticks.

**Cost:** waveforms take 0.61 s against 0.14 s per batch for the Fig. 2 batch (4.4×), and 0.11 s against 0.03 s for the 48.5 cm track. The number of neighbor entries is 25 times larger (2025 against 81 per cloud).

## 2. larnd-sim comparison, aligned geometry (`compare_geo.json`)
Integrated charge per matched pixel, NuLads / reference:

| Reference charge (e⁻) | Pixels | Production | Exact (E) |
|---|---|---|---|
| 2000–5000 | 65 | 0.963 | 0.979 |
| 5000–10 000 | 105 | 0.967 | 0.978 |
| 10 000–20 000 | 216 | 0.989 | 0.992 |
| above 20 000 | 638 | 0.998 | 1.000 |

- E closes about half of the shortfall on weakly lit pixels.
- The 14 pixels below 2000 e⁻ are unchanged, so their mismatch has another cause (pixel set, R091).
- The "2–3× more charge on pixels < 1e4 e⁻" of REVIEW.md came from the earlier, mismatched geometry. It does not reproduce in the aligned comparison.

## 3. Calibration shift at fixed true track inputs
Targets are stochastic with seed batch+1. The same noise draws are used in every variant (checked: between prod and L, 7 of 155 hits move, each by one tick).

### First order (`bias.json`, 60 batches, 123.5 m)
Δθ = −H⁻¹ Σ_b [g_k − g_prod], using the per-batch Hessians of hessian.json.

| Variant | A_b | E | D_T | D_L | τ |
|---|---|---|---|---|---|
| E | +0.06 ± 0.03% | −0.005% | +18.0 ± 1.7% | −18.8 ± 3.2% | −6.5 ± 1.4% |
| T | +0.07% | −0.005% | +17.4 ± 1.9% | +2.0 ± 2.8% | −3.0 ± 1.3% |
| L | +0.01% | −0.001% | +2.4 ± 0.7% | −22.4 ± 3.2% | −4.5 ± 0.9% |

- The per-batch gradient difference is 2–4 times the score RMS. Its sign is the same in 100% of batches for A_b and τ, and in 88% for D_T.
- For comparison, the statistical floor at 398 m is A_b 0.03%, D_T 0.37%, D_L 0.87%, τ 0.72%.

### Direct fit (`fit.json`, 20 batches, 43.7 m)
Newton steps on the 5 parameters, with fixed H and step halving; end point of variant k minus end point of prod.

| Variant | A_b | E | D_T | D_L | τ |
|---|---|---|---|---|---|
| E | +0.11% | −0.005% | **+23.2%** | **−27.2%** | **−9.9%** |
| T | +0.10% | −0.006% | +22.0% | −5.4% | −6.0% |
| L | +0.05% | −0.002% | +4.5% | −28.5% | −6.2% |

- The fit on production targets ends at the noise level: τ +2.7%, D_L −0.4%.
- The loss at the end point is higher than on production targets: E +114 per batch, T +92, L +10.

### Likely mechanism
- The hit charge is read at the end of the hold window (Q(t+δ)), and the likelihood scores it with σ_Q = 500 e⁻.
- The neighbor induction changes Q(t+δ) by tens of e⁻ per hit, and the sign is the same over thousands of hits.
- That pulls the parameters that shape Q(t+δ) along soft directions of the likelihood. The loss changes by only about 10 along these large shifts.

## Caveats of this test
- Track inputs are held at truth and there are no nuisance parameters. In the joint fit, the energy losses and trajectories may absorb part of the effect, or move it elsewhere. This is untested; it needs joint fits on E targets.
- All of the paper's closure fits use targets with the same approximation, so their results are self-consistent and unaffected. The effect matters for detector data, where the induction is physical.
- The exact variant is exact only within the 1/10-pitch response table, the 5x5 transverse bins and the 9x9 footprint.

## Files
- `neigh_exact.py`, `run_neigh.py`, `run_geo_exact.py`, `compare_geo.py`, `run_fit.py`, `analyze_bias.py`
- Jobs 41496225 (readout, geo, bias) and 41496935 (fit). Logs in `logs/`.
