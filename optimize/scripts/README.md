# Grid Scan Scripts for Parameter Optimization

This directory contains scripts for running parameter scans on the SLURM cluster with different loss strategies.

## Available Scan Modes

### 1. Probabilistic NLL Loss (`scan_nll_lut.sh`)
**Uses:** `ProbabilisticLossStrategy` with `llhd_loss`

This mode uses the full probabilistic simulation output and computes a negative log-likelihood loss. The loss accounts for:
- Observed hits: -log P(tick|pixel) - log P(charge|tick,pixel)
- False positives: penalty for predicting hits where none observed

**Usage:**
```bash
sbatch optimize/scripts/scan_nll_lut.sh
```

**Key parameters:**
- `--probabilistic-sim`: Enables probabilistic simulation
- `--loss_fn llhd`: Uses log-likelihood loss
- Computes full probability distributions over (pixel, tick, charge)

### 2. Collapsed MSE Loss (`scan_collapsed_mse.sh`)
**Uses:** `CollapsedProbabilisticLossStrategy` with `mse_adc`

This mode collapses probabilistic distributions into expected hit values and applies MSE loss on ADC values. For each (pixel, hit) pair:
- λ = Σ_t P(tick|pixel,hit)
- E[tick] = Σ_t t·P(tick) / λ
- E[ADC] = Σ_t ADC(t)·P(tick) / λ

Only pseudo-hits with λ > threshold are kept. The MSE loss (via MMD) compares these pseudo-hits to target hits.

**Usage:**
```bash
sbatch optimize/scripts/scan_collapsed_mse.sh
```

**Key parameters:**
- `--probabilistic-sim`: Enables probabilistic simulation
- `--loss_fn mse_adc`: Uses MSE (via MMD) loss
- Automatically uses `CollapsedProbabilisticLossStrategy` when both flags are set

### 3. Collapsed Chamfer Loss (`scan_collapsed_chamfer.sh`)
**Uses:** `CollapsedProbabilisticLossStrategy` with `chamfer_3d`

Similar to Collapsed MSE, but uses Chamfer distance instead of MSE. Chamfer distance is an asymmetric metric that measures distances between point clouds in (x, y, z, Q) space.

**Usage:**
```bash
sbatch optimize/scripts/scan_collapsed_chamfer.sh
```

**Key parameters:**
- `--probabilistic-sim`: Enables probabilistic simulation
- `--loss_fn chamfer_3d`: Uses Chamfer distance loss
- `--chamfer_adc_norm`: ADC normalization factor (default: 1.0)
- `--chamfer_match_z`: Optional flag to match z-coordinates instead of ticks

## Customization

Each script scans over 9 parameters using SLURM array jobs:
- `Ab`: Box model amplitude
- `kb`: Box model width
- `eField`: Electric field
- `tran_diff`: Transverse diffusion
- `long_diff`: Longitudinal diffusion
- `lifetime`: Electron lifetime
- `shift_z`, `shift_x`, `shift_y`: Position shifts

To modify:
1. **Change scanned parameters:** Edit the `PARAMS` array
2. **Adjust scan range:** Parameters are scanned around nominal values (see `--scan_tgt_nom`)
3. **Batch size/iterations:** Modify `BATCH_SIZE` and `ITERATIONS` variables
4. **Resource requirements:** Adjust SBATCH directives at the top

## Output

Results are saved to `output/${test_name}/` with unique UUIDs. Each job outputs to `logs/scan/job_${mode}_${JOBID}.out`.

## Strategy Selection Logic

The loss strategy is automatically selected in `fit_params.py`:
- `loss_fn == 'llhd'` → `ProbabilisticLossStrategy`
- `probabilistic_sim == True` and `loss_fn != 'llhd'` → `CollapsedProbabilisticLossStrategy`
- Otherwise → `GenericLossStrategy`

## Bug Fix (Feb 2026)

The `CollapsedProbabilisticLossStrategy` was fixed to properly handle:
- Z-coordinates: Now computed from drift time using `get_hit_z()` or extracted from prediction
- Event IDs: Properly replicated per hit from prediction
- Hit probabilities: Uses 1.0 instead of λ to match stochastic behavior

This ensures collapsed strategies produce identical losses to stochastic when given exact hit information.
