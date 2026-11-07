import pickle
import numpy as np
import os

pkl_path = "/sdf/group/neutrino/pgranger/larnd-sim-jax/fit_result/fit_noise_fit_dedx_26976208/history_iter5000_true_throughmuons_1dfit_fit_dedx_priw0.5_dlr1e-2_dsi0_noise_tgtsim_n_neigh4_mode_lut_e_sampling_0.01cm_seed_stgy_random_grad_clip100_warmup_exponential_decay_schedule_bt200_tgtsd1_dtsd1_adam_llhd.pkl"

with open(pkl_path, 'rb') as f:
    data = pickle.load(f)

cache = data.get('dedx_cache')
true_cache = data.get('true_dedx_cache')
loc = float(data.get('dedx_student_loc', 1.864))

if cache is not None and true_cache is not None:
    print(f"Number of batches in cache: {len(cache)}")
    
    all_true = []
    all_fitted = []
    all_init = []
    
    for batch_idx in sorted(cache.keys()):
        entry = cache[batch_idx]
        log_dedx = entry.get('log_dedx')
        if log_dedx is None: continue
        fitted_vals = np.exp(np.asarray(log_dedx, dtype=float).ravel())
        
        true_vals = true_cache.get(batch_idx, true_cache.get(int(batch_idx)))
        if true_vals is not None:
            true_vals = np.asarray(true_vals, dtype=float).ravel()
            n = min(len(true_vals), len(fitted_vals))
            all_true.extend(true_vals[:n].tolist())
            all_fitted.extend(fitted_vals[:n].tolist())
            all_init.extend([loc]*n)
            
    all_true = np.array(all_true)
    all_fitted = np.array(all_fitted)
    all_init = np.array(all_init)
    
    residuals = all_fitted - all_true
    init_residuals = all_init - all_true
    
    print("\n--- Final Residuals (Fitted - True) ---")
    print(f"Mean residual: {np.mean(residuals):.5f} MeV/cm")
    print(f"Std of residuals: {np.std(residuals):.5f} MeV/cm")
    print(f"MAE: {np.mean(np.abs(residuals)):.5f} MeV/cm")
    
    print("\n--- Initial Residuals (Prior - True) ---")
    print(f"Mean residual: {np.mean(init_residuals):.5f} MeV/cm")
    print(f"Std of residuals: {np.std(init_residuals):.5f} MeV/cm")
    print(f"MAE: {np.mean(np.abs(init_residuals)):.5f} MeV/cm")
    
    # Let's count how many fitted values are extremely large or small
    print("\nFitted values summary:")
    print(f"  Min: {np.min(all_fitted):.5f}, Max: {np.max(all_fitted):.5f}")
    print(f"  True Min: {np.min(all_true):.5f}, True Max: {np.max(all_true):.5f}")
    print(f"  Fraction of fitted < 1e-3: {np.mean(all_fitted < 1.001e-3):.5%}")
    print(f"  Fraction of fitted > 10.0: {np.mean(all_fitted > 10.0):.5%}")
    
    # Print first few fitted vs true values
    print("\nFirst 15 segment comparisons (True vs Fitted):")
    for i in range(min(15, len(all_true))):
        print(f"  Seg {i:2d}: True={all_true[i]:.4f}, Fitted={all_fitted[i]:.4f}, Diff={all_fitted[i]-all_true[i]:+.4f}")
else:
    print("dedx_cache or true_dedx_cache not found.")
