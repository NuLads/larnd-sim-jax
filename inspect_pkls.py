
import pickle
import os
import glob
import sys

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
            
        # Extract fields from the history dict
        info = {
            "dir": d,
            "file": os.path.basename(fname),
            "epochs": data.get("epochs", "N/A"),
            "total_iter": data.get("total_iter", "N/A"),
            "fit_dedx": data.get("fit_dedx", "N/A"),
            "probabilistic_sim": data.get("probabilistic_sim", "N/A"),
            "probabilistic_sampling_target": data.get("probabilistic_sampling_target", "N/A"),
            "dedx_prior_weight": data.get("dedx_prior_weight", "N/A"),
            "normalization_scheme": data.get("normalization_scheme", "N/A"),
            "test_name": data.get("test_name", "N/A"),
            "out_label": data.get("out_label", "N/A"),
        }
        
        # If 'config' is present
        if 'config' in data:
            c = data['config']
            if hasattr(c, '__dict__'):
                c_dict = vars(c)
            elif isinstance(c, dict):
                c_dict = c
            else:
                c_dict = {}
            
            # Update with more detailed config if available
            info.update({
                "config_epochs": c_dict.get("epochs"),
                "config_fit_dedx": c_dict.get("fit_dedx"),
                "config_probabilistic_sim": c_dict.get("probabilistic_sim"),
                "config_prob_sampling_target": c_dict.get("probabilistic_sampling_target"),
                "config_batch_sz": c_dict.get("batch_sz"),
                "config_lr": c_dict.get("lr"),
                "config_max_clip_norm_val": c_dict.get("max_clip_norm_val"),
                "config_loss_fn": c_dict.get("loss_fn"),
            })
            
        results.append(info)
    except Exception as e:
        results.append({"dir": d, "error": str(e)})

import json
print(json.dumps(results, indent=2))
