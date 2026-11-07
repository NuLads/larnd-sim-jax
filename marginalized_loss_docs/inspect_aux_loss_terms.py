import pickle
import numpy as np
import os

pkl_path = "/sdf/group/neutrino/pgranger/larnd-sim-jax/fit_result/fit_noise_fit_dedx_26976208/history_iter5000_true_throughmuons_1dfit_fit_dedx_priw0.5_dlr1e-2_dsi0_noise_tgtsim_n_neigh4_mode_lut_e_sampling_0.01cm_seed_stgy_random_grad_clip100_warmup_exponential_decay_schedule_bt200_tgtsd1_dtsd1_adam_llhd.pkl"

with open(pkl_path, 'rb') as f:
    data = pickle.load(f)

aux_iter = data.get('aux_iter')
mae = np.array(data.get('dedx_mae_iter'))

if aux_iter is not None and len(aux_iter) > 0:
    print(f"Number of aux entries: {len(aux_iter)}")
    # Print the keys of the first non-empty entry
    first_entry = None
    for entry in aux_iter:
        if entry:
            first_entry = entry
            break
    print("Keys in aux_iter:", list(first_entry.keys()) if first_entry else "None")
    
    # Collect history for each key
    history = {k: [] for k in first_entry.keys()} if first_entry else {}
    for entry in aux_iter:
        if entry:
            for k in history.keys():
                history[k].append(entry.get(k, np.nan))
        else:
            for k in history.keys():
                history[k].append(np.nan)
                
    # Align with MAE
    # Since mae is size 4999 and aux_iter is size 5000
    n_min = min(len(mae), len(aux_iter))
    
    print("\nSlice-by-slice detailed sub-loss averages:")
    headers = ["Iteration", "MAE", "Prior", "Loss"] + list(history.keys())
    header_str = " | ".join(f"{h:<10}" for h in headers)
    print(header_str)
    print("-" * len(header_str))
    
    step = 500
    for start in range(0, n_min, step):
        end = min(n_min, start + step)
        slice_mae = mae[start:end]
        slice_mae = slice_mae[~np.isnan(slice_mae)]
        
        m_mae = np.mean(slice_mae) if len(slice_mae) > 0 else np.nan
        m_prior = np.nanmean(np.array(data.get('dedx_prior_iter'))[start:end]) if 'dedx_prior_iter' in data else np.nan
        m_loss = np.nanmean(np.array(data.get('losses_iter'))[start:end])
        
        row_vals = [f"{start:4d}-{end:4d}", f"{m_mae:.5f}", f"{m_prior:.1f}", f"{m_loss:.1f}"]
        for k in history.keys():
            slice_val = history[k][start:end]
            m_val = np.nanmean(slice_val)
            row_vals.append(f"{m_val:.3f}" if abs(m_val) < 100 else f"{m_val:.1f}")
        print(" | ".join(f"{v:<10}" for v in row_vals))
else:
    print("No aux_iter data found.")
