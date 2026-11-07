
import pickle
import os
import glob

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
            
        history = data
        
        info = {
            "dir": d,
            "relevant_params": data.get("relevant_params_list", []),
            "fit_dedx": False,
            "true_dedx_cache_present": "true_dedx_cache" in data or "_batch_true_dedx" in data or "true_dedx_cache" in data.get("training_history", {}),
        }
        
        if 'config' in data:
            c = data['config']
            c_dict = vars(c) if hasattr(c, '__dict__') else (c if isinstance(c, dict) else {})
            info["fit_dedx"] = c_dict.get("fit_dedx")
            if "param_list" in c_dict:
                info["relevant_params"] = c_dict["param_list"]
        
        results.append(info)
    except Exception as e:
        results.append({"dir": d, "error": str(e)})

import json
print(json.dumps(results, indent=2))
