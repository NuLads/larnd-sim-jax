import pickle
import numpy as np
import os

pkl_path = "/sdf/group/neutrino/pgranger/larnd-sim-jax/fit_result/fit_noise_fit_dedx_26976208/history_iter5000_true_throughmuons_1dfit_fit_dedx_priw0.5_dlr1e-2_dsi0_noise_tgtsim_n_neigh4_mode_lut_e_sampling_0.01cm_seed_stgy_random_grad_clip100_warmup_exponential_decay_schedule_bt200_tgtsd1_dtsd1_adam_llhd.pkl"

with open(pkl_path, 'rb') as f:
    data = pickle.load(f)

# Config details
config = data.get('config')
c_dict = vars(config) if hasattr(config, '__dict__') else config
print("--- Selected Config Parameters ---")
for k in ['iterations', 'data_sz', 'max_nbatch', 'lr', 'dedx_lr', 'dedx_start_iter', 'dedx_freeze_iter', 'dedx_prior_weight']:
    print(f"  {k}: {c_dict.get(k)}")

mae = np.array(data.get('dedx_mae_iter'))
kb = np.array(data.get('kb_iter'))
losses = np.array(data.get('losses_iter'))

# Find minimum common length
# kb has size 5001 (includes init), losses has size 5000, mae has size 4999
n_min = min(len(mae), len(losses), len(kb) - 1)
print(f"Lengths: mae={len(mae)}, kb={len(kb)}, losses={len(losses)}")
print(f"Aligning to minimum length: {n_min}")

mae_aligned = mae[:n_min]
losses_aligned = losses[:n_min]
kb_aligned = kb[1:n_min + 1]  # kb[0] is init, so kb[1] is the value after 1st update

# Since batch is updated one-by-one, let's compute rolling mean of MAE
def rolling_mean(arr, window=100):
    if len(arr) == 0: return arr
    import pandas as pd
    return pd.Series(arr).rolling(window=window, min_periods=1).mean().values

mae_smooth = rolling_mean(mae_aligned, 100)
losses_smooth = rolling_mean(losses_aligned, 100)

print("\n--- Correlation Analysis ---")
valid_mask = ~np.isnan(mae_aligned)
if np.sum(valid_mask) > 10:
    corr_kb_mae = np.corrcoef(kb_aligned[valid_mask], mae_aligned[valid_mask])[0, 1]
    print(f"Correlation between kb and dEdx MAE: {corr_kb_mae:.4f}")
    
    # Check if there are other runs in the directory
    import glob
    run_dir = os.path.dirname(pkl_path)
    all_pkls = glob.glob(os.path.join(run_dir, "*.pkl"))
    print(f"\nOther pkl files in same directory: {len(all_pkls)}")
    for p in sorted(all_pkls):
        print("  ", os.path.basename(p))

    # Let's print out the min and max values of kb and mae in slices
    step = 500
    print("\nSlice-by-slice averages:")
    print(f"{'Iteration':<15} | {'Mean MAE':<10} | {'Mean kb':<10} | {'Mean Loss':<10}")
    print("-" * 60)
    for start in range(0, n_min, step):
        end = min(n_min, start + step)
        slice_mae = mae_aligned[start:end]
        slice_kb = kb_aligned[start:end]
        slice_loss = losses_aligned[start:end]
        
        # remove nan
        slice_mae = slice_mae[~np.isnan(slice_mae)]
        if len(slice_mae) > 0:
            m_mae = np.mean(slice_mae)
            m_kb = np.mean(slice_kb)
            m_loss = np.mean(slice_loss)
            print(f"{start:4d} - {end:4d} | {m_mae:8.5f} | {m_kb:8.5f} | {m_loss:8.1f}")
        else:
            m_kb = np.mean(slice_kb)
            m_loss = np.mean(slice_loss)
            print(f"{start:4d} - {end:4d} | {'NaN':>8} | {m_kb:8.5f} | {m_loss:8.1f}")

            
else:
    print("Not enough valid MAE values.")
