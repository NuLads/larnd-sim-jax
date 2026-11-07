"""
Monkeypatch the reference larnd-sim's hardcoded Python constants (NOT
yaml-configurable) so that each comparison stage matches our TRUTH
parameters / staged plan.

Must be called AFTER `from larndsim import fee` / `from larndsim.consts import
physics` but BEFORE the first call into any numba.cuda-jitted function that
reads these globals (quenching.quench, fee's digitization kernels, etc.),
since numba bakes in global values at first-compile (lazy JIT). In practice:
call apply_stage() right after importing fee/physics and before calling
cli.simulate_pixels.run_simulation().

Stage semantics (see README at top of run_reference_stage.py):
  stage 1: diffusion off / lifetime off (via yaml) + noise OFF (here)
  stage 2: diffusion+lifetime at TRUTH (via yaml)   + noise OFF (here)
  stage 3: diffusion+lifetime at TRUTH (via yaml)   + noise ON  (here, defaults)

Ab (Birks recombination constant) is NOT yaml-configurable in the reference
(it's a plain module constant in larndsim/consts/physics.py, read directly
by quenching.quench via the hardcoded `physics.BIRKS` mode selected in
cli/simulate_pixels.py:702 -- recombination model is BIRKS on both sides by
default, consts_jax.py's RecombinationMode default is also BIRKS, so no
model mismatch). We override physics.BIRKS_Ab to our TRUTH Ab for every
stage; physics.BIRKS_kb is left at its default (0.0486, not in the TRUTH
param list, matches consts_jax default kb=0.0486 too).

V_CM/V_REF: our src/larndsim/detector_properties/module0.yaml carries
recalibrated ADC constants (284.27734375 / 1282.71484375 mV) that differ
~1.3% from the reference's generic fee.py defaults (288 / 1300 mV). These
are applied for every stage too, so the ADC<->charge conversion matches
exactly. If you want to test against the reference's own generic defaults
instead, pass align_adc_constants=False.
"""

TRUTH_Ab = 0.834881
TRUTH_V_CM_mV = 284.27734375
TRUTH_V_REF_mV = 1282.71484375
TRUTH_THRESHOLD_e = 5000


def apply_stage(fee, physics, units, stage, align_adc_constants=True):
    """
    fee: the larndsim.fee module (as imported in the running process)
    physics: the larndsim.consts.physics module
    units: the larndsim.consts.units module (for the `mV`, `e` unit constants)
    stage: 1, 2, or 3 (see module docstring)
    """
    assert stage in (1, 2, 3), f"unknown stage {stage}"

    e = units.e
    mV = units.mV

    # --- Recombination (all stages): BIRKS_Ab -> TRUTH ---
    physics.BIRKS_Ab = TRUTH_Ab
    # physics.BIRKS_kb left at default (0.0486)

    # --- Discrimination threshold (all stages): -> TRUTH (5000 e-) ---
    fee.DISCRIMINATION_THRESHOLD = TRUTH_THRESHOLD_e * e

    # --- ADC calibration constants (all stages, optional) ---
    if align_adc_constants:
        fee.V_CM = TRUTH_V_CM_mV * mV
        fee.V_REF = TRUTH_V_REF_mV * mV
        # V_PEDESTAL (580 mV) is not touched: our module0.yaml doesn't set
        # it either (it's absent from the JAX-side yaml), consts_jax falls
        # back to the same 580 default -- already aligned.

    # --- Noise (stage-dependent) ---
    if stage in (1, 2):
        fee.RESET_NOISE_CHARGE = 0
        fee.UNCORRELATED_NOISE_CHARGE = 0
        fee.DISCRIMINATOR_NOISE = 0
    else:  # stage 3
        fee.RESET_NOISE_CHARGE = 900 * e        # matches TRUTH reset-noise (900 e-)
        fee.UNCORRELATED_NOISE_CHARGE = 500 * e  # matches TRUTH uncorrelated-noise (500 e-)
        fee.DISCRIMINATOR_NOISE = 650 * e        # reference default; no JAX-side TRUTH equivalent given

    # --- Optional: remove the reference ADC dead time (test of the hit-count excess, 9 Oct) ---
    import os as _os
    if _os.environ.get('REF_NO_DEADTIME', '0') == '1':
        fee.ADC_BUSY_DELAY = 0
        fee.RESET_CYCLES = 0

    return dict(
        ADC_BUSY_DELAY=fee.ADC_BUSY_DELAY, RESET_CYCLES=fee.RESET_CYCLES,
        BIRKS_Ab=physics.BIRKS_Ab,
        DISCRIMINATION_THRESHOLD=fee.DISCRIMINATION_THRESHOLD,
        V_CM=fee.V_CM,
        V_REF=fee.V_REF,
        RESET_NOISE_CHARGE=fee.RESET_NOISE_CHARGE,
        UNCORRELATED_NOISE_CHARGE=fee.UNCORRELATED_NOISE_CHARGE,
        DISCRIMINATOR_NOISE=fee.DISCRIMINATOR_NOISE,
    )
