import os
import sys
import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

# Ensure src/ and current dir are in path
sys.path.insert(0, "/sdf/group/neutrino/pgranger/larnd-sim-jax")
sys.path.insert(0, "/sdf/group/neutrino/pgranger/larnd-sim-jax/src")

from optimize.strategies import ProbabilisticLossStrategy

@struct.dataclass
class MockParams:
    ADC_COUNTS: float = struct.field(default=256.0, pytree_node=False)
    V_REF: float = struct.field(default=1.3, pytree_node=False)
    V_CM: float = struct.field(default=0.3, pytree_node=False)
    V_PEDESTAL: float = struct.field(default=0.5, pytree_node=False)
    GAIN: float = struct.field(default=4.0, pytree_node=True)  # Make GAIN a pytree node to trace gradients w.r.t it

def test_loss_strategy():
    print("Initializing ProbabilisticLossStrategy...")
    # Initialize strategy with time_window > 0 (e.g., 2)
    strategy = ProbabilisticLossStrategy(
        sigma_charge=500.0,
        eps=1e-8,
        time_window=2,
        sigma_time=1.0,
        apply_deadtime=False,
        first_hit_only=False
    )

    params = MockParams()

    # Define mock prediction
    # Npix = 2, Nslots = 3, Nticks = 50
    Npix = 2
    Nslots = 3
    Nticks = 50
    
    unique_pixels = jnp.array([10, 20])
    
    # We will set high probability for specific ticks/slots
    # Slot 0 on pixel 10 at tick 10
    # Slot 1 on pixel 10 at tick 20
    # Slot 2 on pixel 10 at tick 30
    
    hit_prob_raw = np.full((Npix, Nslots, Nticks), -18.4) # background log prob (approx log(1e-8))
    # Pixel 10
    hit_prob_raw[0, 0, 10] = -0.5
    hit_prob_raw[0, 1, 20] = -0.5
    hit_prob_raw[0, 2, 30] = -0.5
    # Pixel 20
    hit_prob_raw[1, 0, 15] = -0.5
    hit_prob_raw[1, 1, 25] = -0.5
    
    hit_prob = jnp.array(hit_prob_raw)
    
    # Set expected ADCs
    adcs_distrib = jnp.ones((Npix, Nslots, Nticks)) * 100.0
    
    prediction = {
        'hit_prob': hit_prob,
        'adcs_distrib': adcs_distrib,
        'unique_pixels': unique_pixels
    }

    # Scenario A: Target hits perfectly matched to slots
    # target ADCs are set to 120.0 (different from simulation's 100.0) so GAIN has a non-zero derivative
    target_A = {
        'pixel_id': jnp.array([10, 10, 10, 20, 20]),
        'ticks': jnp.array([10, 20, 30, 15, 25]),
        'adcs': jnp.array([120.0, 120.0, 120.0, 120.0, 120.0])
    }

    # Scenario B: Target hits have one hit missing (stochastically omitted)
    target_B = {
        'pixel_id': jnp.array([10, 10, 20, 20]),
        'ticks': jnp.array([10, 30, 15, 25]),
        'adcs': jnp.array([120.0, 120.0, 120.0, 120.0])
    }

    print("\n--- Running Non-JIT Compute ---")
    loss_A, aux_A = strategy.compute(params, prediction, target_A)
    loss_B, aux_B = strategy.compute(params, prediction, target_B)
    
    print(f"Loss A (perfect match): {loss_A:.4f}")
    print(f"Loss B (missed hit): {loss_B:.4f}")
    print(f"Aux A: {aux_A}")
    print(f"Aux B: {aux_B}")

    # Verify JIT compilation
    print("\n--- Testing JIT Compilation ---")
    @jax.jit
    def jit_compute(p, pred, tgt):
        return strategy.compute(p, pred, tgt)

    loss_A_jit, aux_A_jit = jit_compute(params, prediction, target_A)
    print(f"JIT Loss A: {loss_A_jit:.4f}")
    assert np.isclose(loss_A, loss_A_jit)

    # Verify Gradient Flow
    print("\n--- Testing Gradient Flow ---")
    def loss_wrapper(gain_val):
        p = MockParams(GAIN=gain_val)
        l, _ = strategy.compute(p, prediction, target_A)
        return l

    grad_fn = jax.grad(loss_wrapper)
    grad_val = grad_fn(4.0)
    print(f"Gradient of Loss w.r.t. GAIN: {grad_val:.6f}")
    
    # Ensure gradient is non-zero (showing flow)
    assert not np.isnan(grad_val)
    assert grad_val != 0.0
    print("Gradient flow verified successfully!")

if __name__ == "__main__":
    test_loss_strategy()
