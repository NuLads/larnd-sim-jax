import pickle
import glob
import os
import numpy as np

dir_path = "/sdf/group/neutrino/pgranger/larnd-sim-jax/fit_result/fit_noise_fit_dedx_26976208"
pkl_files = sorted(glob.glob(os.path.join(dir_path, "*.pkl")))

print(f"{'Filename':<30} | {'Init MAE':<10} | {'Min MAE':<10} | {'Min Iter':<8} | {'Final MAE':<10} | {'Final std':<10}")
print("-" * 95)

for fpath in pkl_files:
    fname = os.path.basename(fpath)
    # Extract target seed from name (e.g. tgtsd1)
    parts = fname.split('_')
    tgtsd = "unknown"
    for p in parts:
        if p.startswith("tgtsd"):
            tgtsd = p
            break
            
    try:
        with open(fpath, 'rb') as f:
            data = pickle.load(f)
            
        mae = data.get('dedx_mae_iter')
        cache = data.get('dedx_cache')
        true_cache = data.get('true_dedx_cache')
        
        if mae is not None and len(mae) > 0:
            mae = np.array(mae)
            valid_mask = ~np.isnan(mae)
            valid_indices = np.where(valid_mask)[0]
            
            if len(valid_indices) > 0:
                init_mae = mae[valid_indices[0]]
                min_idx_valid = np.argmin(mae[valid_mask])
                min_idx = valid_indices[min_idx_valid]
                min_mae = mae[min_idx]
                final_mae = mae[-1]
                
                # Compute final std
                if cache is not None and true_cache is not None:
                    residuals = []
                    for batch_idx in cache.keys():
                        log_dedx = cache[batch_idx].get('log_dedx')
                        if log_dedx is None: continue
                        fitted_vals = np.exp(np.asarray(log_dedx, dtype=float).ravel())
                        true_vals = true_cache.get(batch_idx, true_cache.get(int(batch_idx)))
                        if true_vals is not None:
                            true_vals = np.asarray(true_vals, dtype=float).ravel()
                            n = min(len(true_vals), len(fitted_vals))
                            residuals.extend((fitted_vals[:n] - true_vals[:n]).tolist())
                    final_std = np.std(residuals) if residuals else np.nan
                else:
                    final_std = np.nan
                    
                print(f"{tgtsd:<30} | {init_mae:.5f}  | {min_mae:.5f}  | {min_idx:<8d} | {final_mae:.5f}  | {final_std:.5f}")
            else:
                print(f"{tgtsd:<30} | All NaN")
        else:
            print(f"{tgtsd:<30} | No MAE")
    except Exception as e:
        print(f"{tgtsd:<30} | Error: {e}")
