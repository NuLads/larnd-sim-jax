import pickle
import sys
import os

def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def get_config_dict(data):
    if 'config' not in data:
        return {}
    c = data['config']
    if hasattr(c, '__dict__'):
        return vars(c)
    elif isinstance(c, dict):
        return c
    return {}

def main():
    if len(sys.argv) < 3:
        print("Usage: python compare_runs.py <path_to_pkl1> <path_to_pkl2>")
        sys.exit(1)
        
    pkl1 = sys.argv[1]
    pkl2 = sys.argv[2]
    
    data1 = load_pkl(pkl1)
    data2 = load_pkl(pkl2)
    
    config1 = get_config_dict(data1)
    config2 = get_config_dict(data2)
    
    # 1. Basic properties
    label1 = os.path.basename(pkl1)
    label2 = os.path.basename(pkl2)
    print(f"# Config Diff Comparison\n")
    print(f"*   **Run 1**: `{label1}`")
    print(f"*   **Run 2**: `{label2}`\n")
    
    # 2. Config diff
    all_keys = sorted(list(set(config1.keys()) | set(config2.keys())))
    diff_keys = []
    identical_keys = []
    
    for k in all_keys:
        v1 = config1.get(k, "<Not Present>")
        v2 = config2.get(k, "<Not Present>")
        if str(v1) != str(v2):
            diff_keys.append((k, v1, v2))
        else:
            identical_keys.append((k, v1))
            
    print("## Configuration Differences")
    if diff_keys:
        print("| Config Attribute | Run 1 | Run 2 |")
        print("| :--- | :--- | :--- |")
        for k, v1, v2 in diff_keys:
            print(f"| `{k}` | `{v1}` | `{v2}` |")
    else:
        print("No configuration differences found.")
    print()
    
    # 3. Top-level pickle fields diff (excluding history arrays/huge lists)
    print("## Top-Level Field Differences")
    top_keys1 = set(data1.keys())
    top_keys2 = set(data2.keys())
    
    print(f"*   Fields unique to Run 1: `{top_keys1 - top_keys2}`")
    print(f"*   Fields unique to Run 2: `{top_keys2 - top_keys1}`\n")
    
    common_top_keys = sorted(list(top_keys1 & top_keys2))
    diff_top = []
    for k in common_top_keys:
        # Skip huge histories to avoid log bloat
        if k in ['config', 'history', 'tracks', 'track_ids', 'aux_iter', 'dedx_cache', 'true_dedx_cache'] or k.endswith('_iter'):
            continue
        v1 = data1[k]
        v2 = data2[k]
        if str(v1) != str(v2):
            diff_top.append((k, v1, v2))
            
    if diff_top:
        print("| Field | Run 1 Value | Run 2 Value |")
        print("| :--- | :--- | :--- |")
        for k, v1, v2 in diff_top:
            # truncate output if too long
            v1_str = str(v1)[:80] + ("..." if len(str(v1)) > 80 else "")
            v2_str = str(v2)[:80] + ("..." if len(str(v2)) > 80 else "")
            print(f"| `{k}` | `{v1_str}` | `{v2_str}` |")
    else:
        print("No other scalar top-level field differences found.")

if __name__ == '__main__':
    main()
