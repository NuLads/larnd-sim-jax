import os
import sys
import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

# Ensure the src/ directory is in the python path
PROJECT = '/sdf/group/neutrino/pgranger/larnd-sim-jax'
sys.path.insert(0, PROJECT)
sys.path.insert(0, f'{PROJECT}/src')
os.chdir(PROJECT)

from optimize.dataio import TracksDataset, TgtTracksDataset
from larndsim.consts_jax import build_params_class, load_detector_properties, load_lut
from larndsim.sim_jax import (
    simulate_wfs, simulate_probabilistic, pad_to_closest_multiple, get_roi_counts,
    simulate_drift_new, simulate_signals, id2pixel
)

# 1. Setup JAX Device
# Let JAX automatically detect the GPU backend, defaulting to CPU if none available.
print("JAX devices:", jax.devices())

# 2. Load Datasets (Batch 0)
print("Loading datasets...")
dataset_sim = TracksDataset(
    filename="/sdf/group/neutrino/pgranger/lads-data/linear_guess_segments.h5",
    nevents=-1,
    max_nbatch=1,
    random_nevents=True,
    data_seed=1,
    track_len_sel=2.0,
    max_abs_costheta_sel=0.966,
    min_abs_segz_sel=15.0,
    track_z_bound=28.0,
    max_batch_len=200.0,
    chopped=True,
    pad=True,
    electron_sampling_resolution=0.01
)

dataset_target = TgtTracksDataset(
    filename="/sdf/data/neutrino/cyifan/diffsim_input/true_through_muon_edep_10cm_vol1cm.h5",
    dataset_sim=dataset_sim,
    chopped=True,
    pad=True,
    electron_sampling_resolution=0.01
)

fields = tuple(dataset_sim.get_track_fields())
tracks_sim_bt = dataset_sim[0].reshape(-1, len(fields))
tracks_target_bt = dataset_target[0].reshape(-1, len(fields))

# Fix dEdx to uniform mean value of 1.887 MeV/cm
mean_dedx_val = 1.887
dedx_idx = fields.index('dEdx')
dE_idx = fields.index('dE')
dx_idx = fields.index('dx')

tracks_sim_bt = tracks_sim_bt.copy()
tracks_sim_bt[:, dedx_idx] = mean_dedx_val
tracks_sim_bt[:, dE_idx] = mean_dedx_val * tracks_sim_bt[:, dx_idx]

tracks_target_bt = tracks_target_bt.copy()
tracks_target_bt[:, dedx_idx] = mean_dedx_val
tracks_target_bt[:, dE_idx] = mean_dedx_val * tracks_target_bt[:, dx_idx]

tracks_sim = jnp.array(tracks_sim_bt)
tracks_tgt = jnp.array(tracks_target_bt)

# 3. Load Detector Props and Response LUT
Params = build_params_class(['Ab', 'kb'])
ref_params = load_detector_properties(Params, "src/larndsim/detector_properties/module0.yaml", "src/larndsim/pixel_layouts/multi_tile_layout-2.4.16_v4.yaml")
response, ref_params = load_lut("src/larndsim/detector_properties/response_44.npy", ref_params)

# Nominal parameters
ab_nominal = 0.8348813503927325
kb_nominal = 0.04860
params_case1 = ref_params.replace(Ab=jnp.array(ab_nominal), kb=jnp.array(kb_nominal))

# 4. Generate Target (Case 1) expected ADC
def run_simulation(params, tracks):
    wfs, unique_pixels = simulate_wfs(params, response, tracks, fields)
    unique_pixels_padded = pad_to_closest_multiple(unique_pixels, multiple=128, pad_value=-1, pad_front=True)
    wfs_padded = pad_to_closest_multiple(wfs, dims_to_pad=(0,), multiple=128, pad_value=0.0, pad_front=True)
    
    nb_small, nb_large = get_roi_counts(params, wfs_padded)
    padded_small_nb = int(((int(nb_small) + 127) // 128) * 128)
    padded_large_nb = int(((int(nb_large) + 127) // 128) * 128)
    
    adcs_distrib, pixel_x, pixel_y, ticks_prob, event = simulate_probabilistic(
        params, wfs_padded, unique_pixels_padded, padded_small_nb=padded_small_nb, padded_large_nb=padded_large_nb
    )
    
    expected_adc = jnp.sum(jnp.exp(ticks_prob) * adcs_distrib, axis=1) # (Npixels, Nticks)
    return unique_pixels_padded, expected_adc, padded_small_nb, padded_large_nb

print("Simulating Target (Case 1)...")
target_pixels_padded, target_expected_adc_padded, padded_small_nb, padded_large_nb = run_simulation(params_case1, tracks_tgt)

# Extract only valid active target pixels (expected_adc > 0.05 somewhere)
target_adc_sum = jnp.sum(target_expected_adc_padded, axis=-1)
valid_target_mask = (target_pixels_padded >= 0) & (target_adc_sum > 0.05)
target_pixels = target_pixels_padded[valid_target_mask]
target_expected_adc = target_expected_adc_padded[valid_target_mask]

print(f"Number of active target pixels: {len(target_pixels)}")

# Generate Initial Guess active pixels to define a static fixed pixel list
print("Simulating Initial Guess...")
guess_pixels_padded, _, _, _ = run_simulation(params_case1, tracks_sim)

# Combine target and initial guess pixels, sort them and pad to a multiple of 128
combined_pixels = np.unique(np.concatenate([np.array(target_pixels), np.array(guess_pixels_padded[guess_pixels_padded >= 0])]))
combined_pixels = np.append(combined_pixels, -1)
fixed_pixels_np = pad_to_closest_multiple(combined_pixels, multiple=128, pad_value=-1, pad_front=False)
fixed_pixels_np = np.sort(fixed_pixels_np)
fixed_pixels = jnp.array(fixed_pixels_np)
print(f"Fixed static pixel list size: {len(fixed_pixels)}")

# 5. Differentiable Rigid Transformation
def get_rotation_matrix(yaw, pitch, roll):
    cy = jnp.cos(yaw)
    sy = jnp.sin(yaw)
    cp = jnp.cos(pitch)
    sp = jnp.sin(pitch)
    cr = jnp.cos(roll)
    sr = jnp.sin(roll)
    
    Rx = jnp.array([[1.0, 0.0, 0.0],
                    [0.0, cr, -sr],
                    [0.0, sr, cr]])
    
    Ry = jnp.array([[cp, 0.0, sp],
                    [0.0, 1.0, 0.0],
                    [-sp, 0.0, cp]])
    
    Rz = jnp.array([[cy, -sy, 0.0],
                    [sy, cy, 0.0],
                    [0.0, 0.0, 1.0]])
    
    return jnp.dot(Rz, jnp.dot(Ry, Rx))

def apply_transformation(tracks, fields, translation, rotation_angles):
    yaw, pitch, roll = rotation_angles
    R = get_rotation_matrix(yaw, pitch, roll)
    
    x_idx = fields.index("x")
    y_idx = fields.index("y")
    z_idx = fields.index("z")
    
    xs = tracks[:, x_idx]
    ys = tracks[:, y_idx]
    zs = tracks[:, z_idx]
    
    cog_x = jnp.mean(xs)
    cog_y = jnp.mean(ys)
    cog_z = jnp.mean(zs)
    cog = jnp.array([cog_x, cog_y, cog_z])
    
    xs_idx = fields.index("x_start")
    ys_idx = fields.index("y_start")
    zs_idx = fields.index("z_start")
    
    xe_idx = fields.index("x_end")
    ye_idx = fields.index("y_end")
    ze_idx = fields.index("z_end")
    
    pts_start = jnp.stack([tracks[:, xs_idx], tracks[:, ys_idx], tracks[:, zs_idx]], axis=-1)
    pts_end = jnp.stack([tracks[:, xe_idx], tracks[:, ye_idx], tracks[:, ze_idx]], axis=-1)
    pts_mid = jnp.stack([xs, ys, zs], axis=-1)
    
    transformed_start = jnp.dot(pts_start - cog, R.T) + cog + translation
    transformed_end = jnp.dot(pts_end - cog, R.T) + cog + translation
    transformed_mid = jnp.dot(pts_mid - cog, R.T) + cog + translation
    
    t = tracks
    t = t.at[:, xs_idx].set(transformed_start[:, 0])
    t = t.at[:, ys_idx].set(transformed_start[:, 1])
    t = t.at[:, zs_idx].set(transformed_start[:, 2])
    
    t = t.at[:, xe_idx].set(transformed_end[:, 0])
    t = t.at[:, ye_idx].set(transformed_end[:, 1])
    t = t.at[:, ze_idx].set(transformed_end[:, 2])
    
    t = t.at[:, x_idx].set(transformed_mid[:, 0])
    t = t.at[:, y_idx].set(transformed_mid[:, 1])
    t = t.at[:, z_idx].set(transformed_mid[:, 2])
    
    # Recompute dx
    dx_idx = fields.index("dx")
    dx_new = jnp.sqrt(
        (transformed_start[:, 0] - transformed_end[:, 0])**2 +
        (transformed_start[:, 1] - transformed_end[:, 1])**2 +
        (transformed_start[:, 2] - transformed_end[:, 2])**2 + 1e-10
    )
    t = t.at[:, dx_idx].set(dx_new)
    
    return t

# Differentiable static simulation using pre-defined fixed_pixels
def simulate_wfs_static(params, response_template, tracks, fixed_pixels):
    main_pixels, pixels, nelectrons, t0_after_diff, long_diff, currents_idx, pIDs_neigh, currents_idx_neigh, nelectrons_neigh, t0_neigh = simulate_drift_new(params, tracks, fields)
    
    pix_renumbering_neigh = jnp.searchsorted(fixed_pixels, pIDs_neigh.ravel(), method='sort')
    mask = (pix_renumbering_neigh < fixed_pixels.size) & (fixed_pixels[pix_renumbering_neigh] == pIDs_neigh.ravel())
    pix_renumbering_neigh = jnp.where(mask, pix_renumbering_neigh, 0)
    
    wfs = simulate_signals(params, fixed_pixels, pixels, t0_after_diff, response_template, nelectrons, long_diff, currents_idx, nelectrons_neigh, pix_renumbering_neigh, t0_neigh, currents_idx_neigh)
    
    return wfs[:, 1:]

# 6. Loss Functions
@jax.jit
def loss_fn_fixed(opt_vars, tracks_sim, target_pixels, target_expected_adc):
    Ab_scale = opt_vars[0]
    translation = jnp.zeros(3)
    rotation_angles = jnp.zeros(3)
        
    Ab = ab_nominal * Ab_scale
    params = ref_params.replace(Ab=Ab, kb=jnp.array(kb_nominal))
    
    transformed_tracks = apply_transformation(tracks_sim, fields, translation, rotation_angles)
    
    # Run static simulation
    wfs = simulate_wfs_static(params, response, transformed_tracks, fixed_pixels)
    wfs_padded = pad_to_closest_multiple(wfs, dims_to_pad=(0,), multiple=128, pad_value=0.0, pad_front=True)
    
    adcs_distrib, pixel_x, pixel_y, ticks_prob, event = simulate_probabilistic(
        params, wfs_padded, fixed_pixels, 
        padded_small_nb=padded_small_nb, padded_large_nb=padded_large_nb
    )
    
    expected_adc_raw = jnp.sum(jnp.exp(ticks_prob) * adcs_distrib, axis=1)
    
    # Match simulated pixels to target pixels
    indices = jnp.searchsorted(fixed_pixels, target_pixels)
    indices_safe = jnp.clip(indices, 0, fixed_pixels.shape[0] - 1)
    matched_mask = (jnp.take(fixed_pixels, indices_safe) == target_pixels)[:, None]
    guess_expected_adc = jnp.where(matched_mask, jnp.take(expected_adc_raw, indices_safe, axis=0), 0.0)
    
    loss = jnp.mean((guess_expected_adc - target_expected_adc)**2)
    return loss

@jax.jit
def loss_fn_joint(opt_vars, tracks_sim, target_pixels, target_expected_adc):
    Ab_scale = opt_vars[0]
    translation = opt_vars[1:4]
    rotation_angles = opt_vars[4:7] * 0.1 # scale angles to ~0.1 rad
        
    Ab = ab_nominal * Ab_scale
    params = ref_params.replace(Ab=Ab, kb=jnp.array(kb_nominal))
    
    transformed_tracks = apply_transformation(tracks_sim, fields, translation, rotation_angles)
    
    # Run static simulation
    wfs = simulate_wfs_static(params, response, transformed_tracks, fixed_pixels)
    wfs_padded = pad_to_closest_multiple(wfs, dims_to_pad=(0,), multiple=128, pad_value=0.0, pad_front=True)
    
    adcs_distrib, pixel_x, pixel_y, ticks_prob, event = simulate_probabilistic(
        params, wfs_padded, fixed_pixels, 
        padded_small_nb=padded_small_nb, padded_large_nb=padded_large_nb
    )
    
    expected_adc_raw = jnp.sum(jnp.exp(ticks_prob) * adcs_distrib, axis=1)
    
    # Match simulated pixels to target pixels
    indices = jnp.searchsorted(fixed_pixels, target_pixels)
    indices_safe = jnp.clip(indices, 0, fixed_pixels.shape[0] - 1)
    matched_mask = (jnp.take(fixed_pixels, indices_safe) == target_pixels)[:, None]
    guess_expected_adc = jnp.where(matched_mask, jnp.take(expected_adc_raw, indices_safe, axis=0), 0.0)
    
    loss = jnp.mean((guess_expected_adc - target_expected_adc)**2)
    return loss

# Value and grad function for Scipy optimizer
val_and_grad_fn_fixed = jax.value_and_grad(lambda vars: loss_fn_fixed(vars, tracks_sim, target_pixels, target_expected_adc))
val_and_grad_fn_joint = jax.value_and_grad(lambda vars: loss_fn_joint(vars, tracks_sim, target_pixels, target_expected_adc))

def scipy_loss_fixed(x):
    loss, grads = val_and_grad_fn_fixed(x)
    return float(loss), np.array(grads, dtype=np.float64)

def scipy_loss_joint(x):
    loss, grads = val_and_grad_fn_joint(x)
    return float(loss), np.array(grads, dtype=np.float64)

# 7. Execute Fits
print("\n--- FIT 1: FIXED GEOMETRY (FIT Ab ONLY) ---")
x0_fixed = np.array([1.0])

res_fixed = minimize(
    scipy_loss_fixed,
    x0_fixed,
    jac=True,
    method='L-BFGS-B',
    bounds=[(0.5, 1.5)],
    options={'disp': True, 'maxiter': 50}
)

fitted_ab_fixed = ab_nominal * res_fixed.x[0]
print(f"Fitted Ab (Fixed Geometry): {fitted_ab_fixed:.6f} (True: {ab_nominal:.6f})")
print(f"Bias: {(fitted_ab_fixed - ab_nominal)/ab_nominal*100:+.4f}%")

print("\n--- FIT 2: JOINT FIT (FIT Ab + TRANSLATION + ROTATION) ---")
x0_joint = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

res_joint = minimize(
    scipy_loss_joint,
    x0_joint,
    jac=True,
    method='L-BFGS-B',
    bounds=[
        (0.5, 1.5),       # Ab_scale
        (-2.0, 2.0),      # Tx (cm)
        (-2.0, 2.0),      # Ty (cm)
        (-2.0, 2.0),      # Tz (cm)
        (-3.14, 3.14),    # yaw
        (-3.14, 3.14),    # pitch
        (-3.14, 3.14)     # roll
    ],
    options={'disp': True, 'maxiter': 100}
)

fitted_ab_joint = ab_nominal * res_joint.x[0]
tx, ty, tz = res_joint.x[1:4]
yaw, pitch, roll = res_joint.x[4:7] * 0.1

print("\n--- FIT RESULTS COMPARISON ---")
print(f"True Ab:                          {ab_nominal:.6f}")
print(f"Fitted Ab (Fixed Geometry):       {fitted_ab_fixed:.6f} (Bias: {(fitted_ab_fixed - ab_nominal)/ab_nominal*100:+.4f}%)")
print(f"Fitted Ab (Joint Fit):            {fitted_ab_joint:.6f} (Bias: {(fitted_ab_joint - ab_nominal)/ab_nominal*100:+.4f}%)")
print(f"Fitted Translation (Tx, Ty, Tz): ({tx:+.4f}, {ty:+.4f}, {tz:+.4f}) cm")
print(f"Fitted Rotation (yaw, pitch, roll): ({yaw:+.4f}, {pitch:+.4f}, {roll:+.4f}) rad")
print(f"Initial Loss:                     {float(loss_fn_joint(x0_joint, tracks_sim, target_pixels, target_expected_adc)):.6e}")
print(f"Fixed Geometry Final Loss:        {res_fixed.fun:.6e}")
print(f"Joint Fit Final Loss:             {res_joint.fun:.6e}")
