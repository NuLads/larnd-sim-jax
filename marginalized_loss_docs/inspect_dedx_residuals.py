import pickle
import numpy as np
import os
import matplotlib.pyplot as plt

pkl_path = "/sdf/group/neutrino/pgranger/larnd-sim-jax/fit_result/fit_noise_fit_dedx_26976208/history_iter5000_true_throughmuons_1dfit_fit_dedx_priw0.5_dlr1e-2_dsi0_noise_tgtsim_n_neigh4_mode_lut_e_sampling_0.01cm_seed_stgy_random_grad_clip100_warmup_exponential_decay_schedule_bt200_tgtsd1_dtsd1_adam_llhd.pkl"

if not os.path.exists(pkl_path):
    print(f"File not found: {pkl_path}")
    # Let's search for files in that directory
    import glob
    files = glob.glob("/sdf/group/neutrino/pgranger/larnd-sim-jax/fit_result/fit_noise_fit_dedx_*/*.pkl")
    print(f"Found {len(files)} pkl files in fit_noise_fit_dedx_* directories:")
    for f in files[:10]:
        print("  ", f)
    if files:
        pkl_path = files[0]
        print(f"Using: {pkl_path}")
    else:
        exit(1)

with open(pkl_path, 'rb') as f:
    data = pickle.load(f)

print(f"Loaded: {os.path.basename(pkl_path)}")
print("Keys available:", list(data.keys()))

# Get history arrays
mae = data.get('dedx_mae_iter')
prior = data.get('dedx_prior_iter')
losses = data.get('losses_iter')

if mae is not None and len(mae) > 0:
    mae = np.array(mae)
    prior = np.array(prior)
    losses = np.array(losses)
    
    print(f"Length of MAE: {len(mae)}")
    print(f"Initial MAE: {mae[0]}")
    # Filter out NaNs (from warm-up phase)
    valid_mask = ~np.isnan(mae)
    valid_indices = np.where(valid_mask)[0]
    if len(valid_indices) > 0:
        valid_mae = mae[valid_mask]
        min_idx_valid = np.argmin(valid_mae)
        min_idx = valid_indices[min_idx_valid]
        print(f"dEdx MAE activates at iteration {valid_indices[0]}")
        print(f"Minimum MAE is {valid_mae[min_idx_valid]:.6f} at iteration {min_idx}")
        print(f"Final MAE is {valid_mae[-1]:.6f} at the end (iteration {valid_indices[-1]})")
        
        # Analyze first 50 iterations after activation
        activation_start = valid_indices[0]
        print(f"\nFirst 20 iterations of MAE after activation:")
        for idx in range(activation_start, min(activation_start + 20, len(mae))):
            print(f"  Iter {idx:4d}: MAE={mae[idx]:.5f}, Prior={prior[idx]:.5f}, Loss={losses[idx]:.3f}")
            
        print(f"\nAround minimum MAE (Iter {min_idx}):")
        for idx in range(max(activation_start, min_idx - 5), min(len(mae), min_idx + 10)):
            marker = "<- MIN" if idx == min_idx else ""
            print(f"  Iter {idx:4d}: MAE={mae[idx]:.5f}, Prior={prior[idx]:.5f}, Loss={losses[idx]:.3f} {marker}")

        print(f"\nLast 20 iterations of MAE:")
        for idx in range(max(activation_start, len(mae) - 20), len(mae)):
            print(f"  Iter {idx:4d}: MAE={mae[idx]:.5f}, Prior={prior[idx]:.5f}, Loss={losses[idx]:.3f}")
    else:
        print("All MAE values are NaN.")
else:
    print("No MAE data in this pickle file.")

# Check optimization of global parameters
config = data.get('config')
params = []
if hasattr(config, 'params'):
    params = config.params
elif isinstance(config, dict) and 'params' in config:
    params = config['params']
else:
    params = [k[:-7] for k in data.keys() if k.endswith('_target')]

print("\n--- Global Parameters Evolution ---")
for param in params:
    target_val = data.get(param + '_target')
    init_val = data.get(param + '_init')
    iter_vals = data.get(param + '_iter')
    if iter_vals is not None and len(iter_vals) > 0:
        iter_vals = np.array(iter_vals)
        print(f"Parameter: {param}")
        print(f"  Target: {target_val}")
        print(f"  Initial: {init_val}")
        print(f"  Value at activation ({valid_indices[0] if len(valid_indices) > 0 else 0}): {iter_vals[valid_indices[0]] if len(valid_indices) > 0 and valid_indices[0] < len(iter_vals) else 'N/A'}")
        if len(valid_indices) > 0:
            print(f"  Value at minimum MAE ({min_idx}): {iter_vals[min_idx] if min_idx < len(iter_vals) else 'N/A'}")
        print(f"  Final value: {iter_vals[-1]}")
