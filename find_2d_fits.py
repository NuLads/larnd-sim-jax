import os
import glob
import pickle

pkl_files = glob.glob("fit_result/**/*.pkl", recursive=True)
print(f"Found {len(pkl_files)} pickle files.")

found = 0
for f in pkl_files:
    try:
        with open(f, 'rb') as pf:
            hist = pickle.load(pf)
        if 'Ab_iter' in hist and 'kb_iter' in hist:
            print(f"File: {f}")
            print(f"  Final Ab: {hist['Ab_iter'][-1]:.6f} (Target: {hist.get('Ab_target', ['N/A'])[0] if isinstance(hist.get('Ab_target'), list) else hist.get('Ab_target', 'N/A')})")
            print(f"  Final kb: {hist['kb_iter'][-1]:.6f} (Target: {hist.get('kb_target', ['N/A'])[0] if isinstance(hist.get('kb_target'), list) else hist.get('kb_target', 'N/A')})")
            found += 1
            if found >= 10:
                break
    except Exception as e:
        pass
