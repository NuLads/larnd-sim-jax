import os
import sys
import time
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Setup Python path
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
print("JAX devices:", jax.devices())

# 2. Load Datasets (5 batches)
print("Loading datasets...")
dataset_sim = TracksDataset(
    filename="/sdf/group/neutrino/pgranger/lads-data/linear_guess_segments.h5",
    nevents=-1,
    max_nbatch=5,
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

# Fix dEdx to uniform mean value of 1.887 MeV/cm
mean_dedx_val = 1.887
dedx_idx = fields.index('dEdx')
dE_idx = fields.index('dE')
dx_idx = fields.index('dx')

# 3. Load Detector Props and Response LUT
Params = build_params_class(['Ab', 'kb'])
ref_params = load_detector_properties(Params, "src/larndsim/detector_properties/module0.yaml", "src/larndsim/pixel_layouts/multi_tile_layout-2.4.16_v4.yaml")
response, ref_params = load_lut("src/larndsim/detector_properties/response_44.npy", ref_params)

# Nominal parameters
ab_nominal = 0.8348813503927325
kb_nominal = 0.04860
params_case1 = ref_params.replace(Ab=jnp.array(ab_nominal), kb=jnp.array(kb_nominal))

# 4. Rigid-Body Transformation helpers
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

# 5. Static simulator routing
def simulate_wfs_static(params, response_template, tracks, fixed_pixels):
    main_pixels, pixels, nelectrons, t0_after_diff, long_diff, currents_idx, pIDs_neigh, currents_idx_neigh, nelectrons_neigh, t0_neigh = simulate_drift_new(params, tracks, fields)
    
    pix_renumbering_neigh = jnp.searchsorted(fixed_pixels, pIDs_neigh.ravel(), method='sort')
    mask = (pix_renumbering_neigh < fixed_pixels.size) & (fixed_pixels[pix_renumbering_neigh] == pIDs_neigh.ravel())
    pix_renumbering_neigh = jnp.where(mask, pix_renumbering_neigh, 0)
    
    wfs = simulate_signals(params, fixed_pixels, pixels, t0_after_diff, response_template, nelectrons, long_diff, currents_idx, nelectrons_neigh, pix_renumbering_neigh, t0_neigh, currents_idx_neigh)
    
    return wfs[:, 1:]

results = []
os.makedirs("plots/large_fit", exist_ok=True)

for i in range(5):
    print(f"\n=========================================")
    print(f"BATCH {i}: RUNNING FIT STUDY")
    print(f"=========================================")
    
    # Load batch arrays
    tracks_sim_bt = dataset_sim[i].reshape(-1, len(fields))
    tracks_target_bt = dataset_target[i].reshape(-1, len(fields))
    
    # Apply uniform dEdx
    tracks_sim_bt = tracks_sim_bt.copy()
    tracks_sim_bt[:, dedx_idx] = mean_dedx_val
    tracks_sim_bt[:, dE_idx] = mean_dedx_val * tracks_sim_bt[:, dx_idx]
    
    tracks_target_bt = tracks_target_bt.copy()
    tracks_target_bt[:, dedx_idx] = mean_dedx_val
    tracks_target_bt[:, dE_idx] = mean_dedx_val * tracks_target_bt[:, dx_idx]
    
    tracks_sim = jnp.array(tracks_sim_bt)
    tracks_tgt = jnp.array(tracks_target_bt)
    
    # Compute active target pixels
    print(f"Batch {i}: Simulating Target expected ADC...")
    wfs_tgt, tgt_pixels = simulate_wfs(params_case1, response, tracks_tgt, fields)
    tgt_pixels_padded = pad_to_closest_multiple(tgt_pixels, multiple=128, pad_value=-1, pad_front=True)
    wfs_tgt_padded = pad_to_closest_multiple(wfs_tgt, dims_to_pad=(0,), multiple=128, pad_value=0.0, pad_front=True)
    
    nb_small, nb_large = get_roi_counts(params_case1, wfs_tgt_padded)
    padded_small_nb = int(((int(nb_small) + 127) // 128) * 128)
    padded_large_nb = int(((int(nb_large) + 127) // 128) * 128)
    
    adcs_distrib_tgt, pixel_x, pixel_y, ticks_prob_tgt, event = simulate_probabilistic(
        params_case1, wfs_tgt_padded, tgt_pixels_padded, padded_small_nb=padded_small_nb, padded_large_nb=padded_large_nb
    )
    expected_adc_tgt_all = jnp.sum(jnp.exp(ticks_prob_tgt) * adcs_distrib_tgt, axis=1)
    
    target_adc_sum = jnp.sum(expected_adc_tgt_all, axis=-1)
    valid_target_mask = (tgt_pixels_padded >= 0) & (target_adc_sum > 0.05)
    target_pixels = tgt_pixels_padded[valid_target_mask]
    target_expected_adc = expected_adc_tgt_all[valid_target_mask]
    
    print(f"Batch {i}: Active target pixels count: {len(target_pixels)}")
    
    # Guess active pixels
    print(f"Batch {i}: Simulating Initial Guess pixels...")
    _, guess_pixels = simulate_wfs(params_case1, response, tracks_sim, fields)
    guess_pixels_padded = pad_to_closest_multiple(guess_pixels, multiple=128, pad_value=-1, pad_front=True)
    
    # Combined static pixels list
    combined_pixels = np.unique(np.concatenate([np.array(target_pixels), np.array(guess_pixels_padded[guess_pixels_padded >= 0])]))
    combined_pixels = np.append(combined_pixels, -1)
    fixed_pixels_np = pad_to_closest_multiple(combined_pixels, multiple=128, pad_value=-1, pad_front=False)
    fixed_pixels_np = np.sort(fixed_pixels_np)
    fixed_pixels = jnp.array(fixed_pixels_np)
    print(f"Batch {i}: Fixed static pixel list size: {len(fixed_pixels)}")
    
    # Loss JIT functions
    @jax.jit
    def loss_fn_fixed(opt_vars):
        Ab_scale = opt_vars[0]
        translation = jnp.zeros(3)
        rotation_angles = jnp.zeros(3)
        
        Ab = ab_nominal * Ab_scale
        params = ref_params.replace(Ab=Ab, kb=jnp.array(kb_nominal))
        transformed_tracks = apply_transformation(tracks_sim, fields, translation, rotation_angles)
        
        wfs = simulate_wfs_static(params, response, transformed_tracks, fixed_pixels)
        wfs_padded = pad_to_closest_multiple(wfs, dims_to_pad=(0,), multiple=128, pad_value=0.0, pad_front=True)
        
        adcs_distrib, pixel_x, pixel_y, ticks_prob, event = simulate_probabilistic(
            params, wfs_padded, fixed_pixels, 
            padded_small_nb=padded_small_nb, padded_large_nb=padded_large_nb
        )
        expected_adc_raw = jnp.sum(jnp.exp(ticks_prob) * adcs_distrib, axis=1)
        
        indices = jnp.searchsorted(fixed_pixels, target_pixels)
        indices_safe = jnp.clip(indices, 0, fixed_pixels.shape[0] - 1)
        matched_mask = (jnp.take(fixed_pixels, indices_safe) == target_pixels)[:, None]
        guess_expected_adc = jnp.where(matched_mask, jnp.take(expected_adc_raw, indices_safe, axis=0), 0.0)
        
        return jnp.mean((guess_expected_adc - target_expected_adc)**2)

    @jax.jit
    def loss_fn_joint(opt_vars):
        Ab_scale = opt_vars[0]
        translation = opt_vars[1:4]
        rotation_angles = opt_vars[4:7] * 0.1 # scale angles to ~0.1 rad
        
        Ab = ab_nominal * Ab_scale
        params = ref_params.replace(Ab=Ab, kb=jnp.array(kb_nominal))
        transformed_tracks = apply_transformation(tracks_sim, fields, translation, rotation_angles)
        
        wfs = simulate_wfs_static(params, response, transformed_tracks, fixed_pixels)
        wfs_padded = pad_to_closest_multiple(wfs, dims_to_pad=(0,), multiple=128, pad_value=0.0, pad_front=True)
        
        adcs_distrib, pixel_x, pixel_y, ticks_prob, event = simulate_probabilistic(
            params, wfs_padded, fixed_pixels, 
            padded_small_nb=padded_small_nb, padded_large_nb=padded_large_nb
        )
        expected_adc_raw = jnp.sum(jnp.exp(ticks_prob) * adcs_distrib, axis=1)
        
        indices = jnp.searchsorted(fixed_pixels, target_pixels)
        indices_safe = jnp.clip(indices, 0, fixed_pixels.shape[0] - 1)
        matched_mask = (jnp.take(fixed_pixels, indices_safe) == target_pixels)[:, None]
        guess_expected_adc = jnp.where(matched_mask, jnp.take(expected_adc_raw, indices_safe, axis=0), 0.0)
        
        return jnp.mean((guess_expected_adc - target_expected_adc)**2)

    # Scipy wrappers
    loss_history_fixed = []
    loss_history_joint = []
    
    val_and_grad_fn_fixed = jax.value_and_grad(loss_fn_fixed)
    val_and_grad_fn_joint = jax.value_and_grad(loss_fn_joint)
    
    def scipy_loss_fixed(x):
        l, g = val_and_grad_fn_fixed(x)
        loss_history_fixed.append(float(l))
        return float(l), np.array(g, dtype=np.float64)
        
    def scipy_loss_joint(x):
        l, g = val_and_grad_fn_joint(x)
        loss_history_joint.append(float(l))
        return float(l), np.array(g, dtype=np.float64)

    # Run Fixed Geometry
    print(f"Batch {i}: Running Fixed Geometry Fit...")
    t0 = time.time()
    res_fixed = minimize(scipy_loss_fixed, np.array([1.0]), jac=True, method='L-BFGS-B', bounds=[(0.5, 1.5)], options={'maxiter': 50})
    time_fixed = time.time() - t0
    fitted_ab_fixed = ab_nominal * res_fixed.x[0]
    bias_fixed = (fitted_ab_fixed - ab_nominal) / ab_nominal * 100
    
    # Run Joint Geometry
    print(f"Batch {i}: Running Joint Geometry Fit...")
    t0 = time.time()
    x0_joint = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    res_joint = minimize(scipy_loss_joint, x0_joint, jac=True, method='L-BFGS-B', 
                         bounds=[(0.5, 1.5), (-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0), (-3.14, 3.14), (-3.14, 3.14), (-3.14, 3.14)], 
                         options={'maxiter': 100})
    time_joint = time.time() - t0
    fitted_ab_joint = ab_nominal * res_joint.x[0]
    bias_joint = (fitted_ab_joint - ab_nominal) / ab_nominal * 100
    
    tx, ty, tz = res_joint.x[1:4]
    yaw, pitch, roll = res_joint.x[4:7] * 0.1
    
    # Record statistics
    batch_results = {
        'batch': i,
        'ab_fixed': float(fitted_ab_fixed),
        'bias_fixed': float(bias_fixed),
        'time_fixed': float(time_fixed),
        'ab_joint': float(fitted_ab_joint),
        'bias_joint': float(bias_joint),
        'time_joint': float(time_joint),
        'tx': float(tx), 'ty': float(ty), 'tz': float(tz),
        'yaw': float(yaw), 'pitch': float(pitch), 'roll': float(roll),
        'init_loss': float(loss_history_joint[0]),
        'final_loss_fixed': float(res_fixed.fun),
        'final_loss_joint': float(res_joint.fun),
        'loss_history_joint': loss_history_joint
    }
    results.append(batch_results)
    
    # 6. Generate Plot
    print(f"Batch {i}: Plotting results...")
    tracks_fitted = apply_transformation(tracks_sim, fields, res_joint.x[1:4], res_joint.x[4:7]*0.1)
    
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    
    xs_idx = fields.index("x_start")
    xe_idx = fields.index("x_end")
    ys_idx = fields.index("y_start")
    ye_idx = fields.index("y_end")
    zs_idx = fields.index("z_start")
    ze_idx = fields.index("z_end")
    
    # Helper to plot segments
    def plot_segments(ax, tracks, color, style, label=None):
        x_coords = np.vstack([tracks[:, xs_idx], tracks[:, xe_idx]])
        y_coords = np.vstack([tracks[:, ys_idx], tracks[:, ye_idx]])
        ax.plot(x_coords, y_coords, style, color=color, alpha=0.7, label=label)
        
    def plot_segments_xz(ax, tracks, color, style, label=None):
        x_coords = np.vstack([tracks[:, xs_idx], tracks[:, xe_idx]])
        z_coords = np.vstack([tracks[:, zs_idx], tracks[:, ze_idx]])
        ax.plot(x_coords, z_coords, style, color=color, alpha=0.7, label=label)
        
    # X-Y Projection
    plot_segments(axs[0], tracks_tgt, 'black', '-', 'Target')
    plot_segments(axs[0], tracks_sim, 'red', '--', 'Initial Guess')
    plot_segments(axs[0], tracks_fitted, 'blue', '-', 'Fitted')
    # Filter unique labels in legend
    handles, labels = axs[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axs[0].legend(by_label.values(), by_label.keys())
    axs[0].set_xlabel('X [cm]')
    axs[0].set_ylabel('Y [cm]')
    axs[0].set_title('Track Geometry Projection (X-Y)')
    axs[0].grid(True)
    
    # X-Z Projection
    plot_segments_xz(axs[1], tracks_tgt, 'black', '-')
    plot_segments_xz(axs[1], tracks_sim, 'red', '--')
    plot_segments_xz(axs[1], tracks_fitted, 'blue', '-')
    axs[1].set_xlabel('X [cm]')
    axs[1].set_ylabel('Z [cm]')
    axs[1].set_title('Track Geometry Projection (X-Z)')
    axs[1].grid(True)
    
    # Convergence Curves
    axs[2].plot(loss_history_joint, 'b-o', markersize=3, label='Joint Fit')
    axs[2].set_yscale('log')
    axs[2].set_xlabel('Evaluations')
    axs[2].set_ylabel('MSE Loss')
    axs[2].set_title('L-BFGS-B Loss Trajectory')
    axs[2].grid(True, which="both", ls="-")
    axs[2].legend()
    
    plt.suptitle(f"Batch {i}: Joint Calibration & Geometry Fit\n"
                 f"Fitted Ab Bias: {bias_joint:+.3f}% | Tx={tx:+.2f}cm, Ty={ty:+.2f}cm, Tz={tz:+.2f}cm", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"plots/large_fit/batch_{i}_plots.png", dpi=150)
    plt.close()

# 7. Write aggregate report
print("\nSaving study results...")
csv_path = "plots/large_fit/results.csv"
with open(csv_path, "w") as f:
    f.write("batch,ab_fixed,bias_fixed,time_fixed,ab_joint,bias_joint,time_joint,tx,ty,tz,yaw,pitch,roll,init_loss,final_loss_fixed,final_loss_joint\n")
    for r in results:
        f.write(f"{r['batch']},{r['ab_fixed']},{r['bias_fixed']},{r['time_fixed']},{r['ab_joint']},{r['bias_joint']},{r['time_joint']},"
                f"{r['tx']},{r['ty']},{r['tz']},{r['yaw']},{r['pitch']},{r['roll']},{r['init_loss']},{r['final_loss_fixed']},{r['final_loss_joint']}\n")

# Compute averages
biases_fixed = [r['bias_fixed'] for r in results]
biases_joint = [r['bias_joint'] for r in results]
trans_errors = [np.sqrt(r['tx']**2 + r['ty']**2 + r['tz']**2) for r in results]
rot_errors = [np.sqrt(r['yaw']**2 + r['pitch']**2 + r['roll']**2) for r in results]

report_path = "/sdf/home/p/pgranger/.gemini/antigravity-cli/brain/aab645dc-9657-41f6-8cd7-406b6964777f/prototype_gpu_fit_results.md"
with open(report_path, "w") as f:
    f.write("# 5-Batch Track Fit Optimization Study\n\n")
    f.write("This study validates the static pixel routing method and joint track-calibration parameter fitting across 5 independent track batches.\n\n")
    
    f.write("## Summary Statistics\n\n")
    f.write("| Parameter | Fixed Geometry Fit | Joint (Geometry + Ab) Fit |\n")
    f.write("| --- | --- | --- |\n")
    f.write(f"| **Mean Ab Bias (%)** | {np.mean(biases_fixed):+.4f}% | {np.mean(biases_joint):+.4f}% |\n")
    f.write(f"| **Std Dev Ab Bias (%)** | {np.std(biases_fixed):.4f}% | {np.std(biases_joint):.4f}% |\n")
    f.write(f"| **Mean Translation Error (cm)** | - | {np.mean(trans_errors):.4f} cm |\n")
    f.write(f"| **Mean Rotation Error (rad)** | - | {np.mean(rot_errors):.4f} rad |\n\n")
    
    f.write("## Detailed Batch Results\n\n")
    f.write("| Batch | Ab Bias (Fixed) | Ab Bias (Joint) | Tx (cm) | Ty (cm) | Tz (cm) | Rot (rad) | Init Loss | Final Loss (Joint) |\n")
    f.write("| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n")
    for r in results:
        rot_err = np.sqrt(r['yaw']**2 + r['pitch']**2 + r['roll']**2)
        f.write(f"| {r['batch']} | {r['bias_fixed']:+.3f}% | {r['bias_joint']:+.3f}% | {r['tx']:+.3f} | {r['ty']:+.3f} | {r['tz']:+.3f} | {rot_err:.3f} | {r['init_loss']:.3e} | {r['final_loss_joint']:.3e} |\n")
        
    f.write("\n## Batch Performance Visualizations\n\n")
    for r in results:
        f.write(f"### Batch {r['batch']} Fit & Convergence\n")
        f.write(f"![Batch {r['batch']} Plot](file:///sdf/group/neutrino/pgranger/larnd-sim-jax/plots/large_fit/batch_{r['batch']}_plots.png)\n\n")

print("Study completed successfully!")
