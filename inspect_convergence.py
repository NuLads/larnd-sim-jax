
import pickle
import os
import glob
import sys
import numpy as np

# List of directories to check (most recent 10 with .pkl files)
dirs = [
    "fit_result/fit_noise_26794890",
    "fit_result/fit_noise_26550051",
    "fit_result/fit_noise_26550044",
    "fit_result/fit_noise_dedx_26022537",
    "fit_result/fit_noise_dedx_25974564",
    "fit_result/fit_noise_dedx_25975202",
    "fit_result/fit_noise_dedx_25920099",
    "fit_result/fit_noise_dedx_25925525",
    "fit_result/fit_noise_dedx_25918235",
    "fit_result/fit_noise_dedx_25866864"
]

results = []

for d in dirs:
    pkl_files = glob.glob(os.path.join(d, "*.pkl"))
    if not pkl_files:
        continue
    
    pkl_files.sort()
    fname = pkl_files[-1]
    try:
        with open(fname, 'rb') as f:
            data = pickle.load(f)
            
        # Basic config
        info = {
            "dir": d,
            "total_iter": data.get("total_iter", "N/A"),
        }
        
        # Check for 'config'
        if 'config' in data:
            c = data['config']
            c_dict = vars(c) if hasattr(c, '__dict__') else (c if isinstance(c, dict) else {})
            info.update({
                "fit_dedx": c_dict.get("fit_dedx"),
                "prob_sampling_target": c_dict.get("probabilistic_sampling_target"),
            })

        history = data
        
        # Loss convergence - using 'losses_iter'
        loss_key = "losses_iter" if "losses_iter" in history else ("loss" if "loss" in history else None)
        
        if loss_key and isinstance(history[loss_key], (list, np.ndarray)) and len(history[loss_key]) > 0:
            losses = history[loss_key]
            
            def get_val(v):
                if isinstance(v, dict): return 0.0 # Skip dicts like aux_iter
                if hasattr(v, 'tolist'): v = v.tolist()
                if isinstance(v, (list, np.ndarray)): return float(v[0])
                return float(v)

            initial_loss = get_val(losses[0])
            final_loss = get_val(losses[-1])

            info["final_loss"] = final_loss
            info["initial_loss"] = initial_loss
            info["loss_reduction"] = initial_loss / final_loss if final_loss != 0 else 0
        else:
            info["final_loss"] = "N/A"

        # Parameters
        param_info = {}
        for k in history.keys():
            # Only process params that are lists/arrays and NOT dictionaries (like aux_iter)
            if k.endswith("_iter") and isinstance(history[k], (list, np.ndarray)) and len(history[k]) > 0:
                if isinstance(history[k][0], dict):
                    continue # Skip aux_iter or similar
                
                param_name = k.replace("_iter", "")
                vals = history[k]
                
                try:
                    i_val = get_val(vals[0])
                    f_val = get_val(vals[-1])

                    param_info[param_name] = {
                        "final_val": f_val,
                        "initial_val": i_val,
                        "delta": f_val - i_val
                    }
                except (TypeError, ValueError):
                    continue
        
        info["params"] = param_info
            
        results.append(info)
    except Exception as e:
        results.append({"dir": d, "error": str(e), "type": str(type(e))})

import json
print(json.dumps(results, indent=2))
