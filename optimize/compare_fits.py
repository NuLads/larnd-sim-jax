import os
import glob
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set plotting style for clean, publication-quality figures
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['grid.linestyle'] = '--'

def load_run_results(run_id, fit_dir="fit_result"):
    """
    Loads all pickle files for a given run ID.
    Averages history curves and extracts configuration and final values.
    """
    dir_pattern = os.path.join(fit_dir, f"*{run_id}*")
    matching_dirs = glob.glob(dir_pattern)
    pkl_files = []
    if matching_dirs:
        for d in matching_dirs:
            if os.path.isdir(d):
                pkl_files.extend(glob.glob(os.path.join(d, "*.pkl")))
        
    if not pkl_files:
        raise ValueError(f"No pickle files found for run {run_id} in {fit_dir}")
        
    print(f"Loading {len(pkl_files)} seeds for run {run_id}...")
    
    seeds_data = []
    for fpath in pkl_files:
        try:
            with open(fpath, 'rb') as f:
                data = pickle.load(f)
            seeds_data.append(data)
        except Exception as e:
            print(f"Error loading {fpath}: {e}")
            
    if not seeds_data:
        raise ValueError(f"No data could be successfully loaded for run {run_id}")
        
    # Extract config from the first seed
    config = seeds_data[0].get("config", {})
    config_dict = vars(config) if hasattr(config, '__dict__') else (config if isinstance(config, dict) else {})
    
    # Clean config keys
    config_cleaned = {k: (str(v) if isinstance(v, (list, tuple, dict)) else v) for k, v in config_dict.items() if not k.startswith('_')}
    
    # We want to identify the parameters that were fitted in this run
    # This is stored in config or we can infer it from keys ending with '_iter'
    history_keys = seeds_data[0].keys()
    fit_params = []
    for k in history_keys:
        if k.endswith("_iter") and k.replace("_iter", "") != "losses" and k.replace("_iter", "") != "losses_iter":
            param_name = k.replace("_iter", "")
            # Check that it's a list/array of floats
            vals = seeds_data[0][k]
            if isinstance(vals, (list, np.ndarray)) and len(vals) > 0 and not isinstance(vals[0], dict):
                fit_params.append(param_name)
                
    # Average curves across seeds
    averaged_history = {}
    
    # 1. Average losses
    loss_key = "losses_iter" if "losses_iter" in seeds_data[0] else "losses"
    if loss_key in seeds_data[0]:
        all_losses_list = [seed[loss_key] for seed in seeds_data if loss_key in seed]
        min_len = min(len(v) for v in all_losses_list)
        all_losses = np.array([v[:min_len] for v in all_losses_list])
        averaged_history["loss_mean"] = np.mean(all_losses, axis=0)
        averaged_history["loss_std"] = np.std(all_losses, axis=0)
        averaged_history["loss_min"] = np.min(all_losses, axis=0)
        averaged_history["loss_max"] = np.max(all_losses, axis=0)
        
    # 2. Average parameters over iterations
    for p in fit_params:
        iter_key = f"{p}_iter"
        all_iter_vals = []
        for seed in seeds_data:
            if iter_key in seed:
                vals = seed[iter_key]
                # Sometimes lengths might slightly differ (e.g. if one terminated early)
                all_iter_vals.append(vals)
                
        # Find minimum length to truncate/align
        min_len = min(len(v) for v in all_iter_vals)
        all_iter_vals_aligned = np.array([v[:min_len] for v in all_iter_vals])
        
        averaged_history[f"{p}_mean"] = np.mean(all_iter_vals_aligned, axis=0)
        averaged_history[f"{p}_std"] = np.std(all_iter_vals_aligned, axis=0)
        
    # 2b. Average dEdx history metrics if present
    dedx_hist_keys = ["dedx_prior_iter", "dedx_barrier_iter", "dedx_mean_penalty_iter", "dedx_mae_iter", "mean_dedx_iter"]
    for k in dedx_hist_keys:
        all_vals = []
        for seed in seeds_data:
            if k in seed:
                vals = seed[k]
                all_vals.append(vals)
        if all_vals:
            min_len = min(len(v) for v in all_vals)
            all_vals_aligned = np.array([v[:min_len] for v in all_vals])
            averaged_history[f"{k}_mean"] = np.mean(all_vals_aligned, axis=0)
            averaged_history[f"{k}_std"] = np.std(all_vals_aligned, axis=0)
            
    # 3. Calculate final parameter errors and target values
    param_summaries = {}
    for p in fit_params:
        target_key = f"{p}_target"
        init_key = f"{p}_init"
        iter_key = f"{p}_iter"
        
        # Get target value
        targets = []
        inits = []
        finals = []
        
        for seed in seeds_data:
            # Target
            t_val = seed.get(target_key, [None])
            if isinstance(t_val, (list, np.ndarray)):
                t_val = t_val[0] if len(t_val) > 0 else None
            if t_val is not None:
                targets.append(t_val)
                
            # Init
            i_val = seed.get(init_key, [None])
            if isinstance(i_val, (list, np.ndarray)):
                i_val = i_val[0] if len(i_val) > 0 else None
            if i_val is not None:
                inits.append(i_val)
                
            # Final (last iteration value)
            if iter_key in seed and len(seed[iter_key]) > 0:
                finals.append(seed[iter_key][-1])
                
        target_val = np.mean(targets) if targets else None
        init_val = np.mean(inits) if inits else None
        
        finals = np.array(finals)
        final_mean = np.mean(finals)
        final_std = np.std(finals)
        
        bias = final_mean - target_val if target_val is not None else None
        rel_bias = bias / target_val if (target_val is not None and target_val != 0) else None
        
        param_summaries[p] = {
            "target": target_val,
            "init": init_val,
            "final_mean": final_mean,
            "final_std": final_std,
            "bias_mean": bias,
            "bias_std": final_std,
            "rel_bias": rel_bias
        }
        
    # 4. Calculate dEdx fit quality
    dedx_maes = []
    residuals = []
    init_residuals = []
    for seed in seeds_data:
        if "dedx_cache" in seed and "true_dedx_cache" in seed:
            seed_maes = []
            cache = seed["dedx_cache"]
            true_cache = seed["true_dedx_cache"]
            loc = float(seed.get('dedx_student_loc', 1.864))
            
            for track_id in cache.keys():
                entry = cache[track_id]
                log_dedx = entry.get('log_dedx')
                if log_dedx is None: continue
                fitted_val = np.exp(np.asarray(log_dedx, dtype=float).ravel())
                true_val = true_cache.get(track_id, true_cache.get(int(track_id)))
                if true_val is not None:
                    true_val = np.asarray(true_val, dtype=float).ravel()
                    n = min(len(true_val), len(fitted_val))
                    mae = np.mean(np.abs(fitted_val[:n] - true_val[:n]))
                    seed_maes.append(mae)
                    
                    residuals.extend((fitted_val[:n] - true_val[:n]).tolist())
                    init_residuals.extend((loc - true_val[:n]).tolist())
            if seed_maes:
                dedx_maes.append(np.mean(seed_maes))
                
    dedx_mae_mean = np.mean(dedx_maes) if dedx_maes else None
    dedx_mae_std = np.std(dedx_maes) if len(dedx_maes) > 1 else 0.0
    
    return {
        "run_id": run_id,
        "config": config_cleaned,
        "fit_params": fit_params,
        "averaged_history": averaged_history,
        "param_summaries": param_summaries,
        "dedx_mae_mean": dedx_mae_mean,
        "dedx_mae_std": dedx_mae_std,
        "residuals": np.array(residuals) if residuals else None,
        "init_residuals": np.array(init_residuals) if init_residuals else None,
        "num_seeds": len(seeds_data)
    }

def compare_runs(run_ids, fit_dir="fit_result"):
    """
    Compares multiple run IDs and returns a dictionary of compared results
    and a DataFrame summarizing configurations and final errors.
    """
    results = {}
    for r in run_ids:
        try:
            results[r] = load_run_results(r, fit_dir=fit_dir)
        except Exception as e:
            print(f"Skipping run {r} due to error: {e}")
            
    if not results:
        raise ValueError("No runs were successfully loaded for comparison.")
        
    # Build comparison tables
    # 1. Configs table
    all_configs = {}
    for r, res in results.items():
        all_configs[r] = res["config"]
    df_configs = pd.DataFrame(all_configs).T
    
    # 2. Parameters comparison table
    rows = []
    for r, res in results.items():
        for param, p_res in res["param_summaries"].items():
            rows.append({
                "run_id": r,
                "parameter": param,
                "target": p_res["target"],
                "init": p_res["init"],
                "final_val": f"{p_res['final_mean']:.4e} ± {p_res['final_std']:.4e}",
                "bias": p_res["bias_mean"],
                "bias_rel_percent": p_res["rel_bias"] * 100 if p_res["rel_bias"] is not None else None,
                "dEdx_MAE": res["dedx_mae_mean"],
                "seeds": res["num_seeds"]
            })
    df_params = pd.DataFrame(rows)
    
    return results, df_configs, df_params

def smooth_curve(data, window):
    """
    Applies rolling average (moving mean) over a given window.
    Uses pandas rolling with min_periods=1 to handle edge points.
    """
    if window <= 1 or len(data) == 0:
        return data
    return pd.Series(data).rolling(window=window, min_periods=1).mean().values

def plot_loss_convergence(results, smoothing_window=1, save_path=None):
    """
    Plots the loss convergence curves for all runs on a single plot.
    """
    plt.figure(figsize=(10, 6))
    
    for r, res in results.items():
        hist = res["averaged_history"]
        if "loss_mean" not in hist:
            continue
            
        iterations = np.arange(len(hist["loss_mean"]))
        
        # Color mapping or auto-cycle
        p_weight = res["config"].get("dedx_prior_weight", 0.5)
        prob_target = res["config"].get("probabilistic_sampling_target", True)
        
        label = f"Run {r} (prior_w={p_weight}, prob_tgt={prob_target})"
        
        # Apply smoothing
        loss_mean = smooth_curve(hist["loss_mean"], smoothing_window)
        
        # Plot mean curve
        line, = plt.plot(iterations, loss_mean, label=label, lw=2)
        color = line.get_color()
        
        # Shaded region for std deviation across seeds
        if "loss_std" in hist and res["num_seeds"] > 1:
            loss_std = smooth_curve(hist["loss_std"], smoothing_window)
            plt.fill_between(
                iterations,
                np.maximum(0, loss_mean - loss_std),
                loss_mean + loss_std,
                color=color,
                alpha=0.15
            )
            
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.title("Loss Convergence Comparison (Averaged Over Seeds)")
    plt.grid(True)
    plt.legend(loc="upper right")
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"Saved loss plot to {save_path}")
    return plt.gcf()

def plot_param_convergence(results, smoothing_window=1, save_path=None):
    """
    Plots a grid of parameter convergence curves for all runs.
    """
    # Find all unique parameters fitted across all runs
    all_params = set()
    for res in results.values():
        all_params.update(res["fit_params"])
    all_params = sorted(list(all_params))
    
    if not all_params:
        print("No fitted parameters found to plot.")
        return None
        
    # Determine grid dimensions
    num_params = len(all_params)
    ncols = min(3, num_params)
    nrows = (num_params + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), sharex=True)
    if num_params == 1:
        axes = np.array([axes])
    axes_flat = axes.flatten()
    
    for idx, p in enumerate(all_params):
        ax = axes_flat[idx]
        
        # Plot target value (assuming it's the same across runs)
        target_val = None
        for res in results.values():
            if p in res["param_summaries"]:
                target_val = res["param_summaries"][p]["target"]
                break
                
        if target_val is not None:
            ax.axhline(target_val, color="black", linestyle="--", alpha=0.8, label="True Target", lw=1.5)
            
        # Plot run trajectories
        for r, res in results.items():
            hist = res["averaged_history"]
            mean_key = f"{p}_mean"
            std_key = f"{p}_std"
            
            if mean_key in hist:
                iterations = np.arange(len(hist[mean_key]))
                label = f"Run {r}"
                
                # Apply smoothing
                p_mean = smooth_curve(hist[mean_key], smoothing_window)
                
                line, = ax.plot(iterations, p_mean, label=label, lw=1.8)
                
                if std_key in hist and res["num_seeds"] > 1:
                    p_std = smooth_curve(hist[std_key], smoothing_window)
                    ax.fill_between(
                        iterations,
                        p_mean - p_std,
                        p_mean + p_std,
                        color=line.get_color(),
                        alpha=0.15
                    )
                    
        ax.set_title(p)
        ax.set_ylabel("Value")
        ax.grid(True)
        if idx == 0:
            ax.legend(loc="best")
            
    # Set xlabel on the bottom row
    for col_idx in range(ncols):
        bottom_idx = (nrows - 1) * ncols + col_idx
        if bottom_idx < len(axes_flat):
            axes_flat[bottom_idx].set_xlabel("Iteration")
            
    # Hide unused subplots
    for idx in range(num_params, len(axes_flat)):
        axes_flat[idx].set_visible(False)
        
    plt.suptitle("Fitted Parameter Convergence over Iterations", y=1.0)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"Saved parameter convergence plot to {save_path}")
    return fig

def plot_parameter_biases(results, save_path=None):
    """
    Generates a bar chart comparing final parameter biases (errors) for all runs.
    Biases are normalized by target value (if non-zero) or shown as absolute.
    """
    all_params = set()
    for res in results.values():
        all_params.update(res["fit_params"])
    all_params = sorted(list(all_params))
    
    if not all_params:
        return None
        
    # We will plot relative error (percent error) for parameters where target != 0
    # and absolute error for others.
    # Group parameters by scale so the plot is legible, or plot relative errors in %
    plt.figure(figsize=(12, 6))
    
    run_ids = list(results.keys())
    x = np.arange(len(all_params))
    width = 0.8 / len(run_ids)
    
    for i, r in enumerate(run_ids):
        res = results[r]
        biases_pct = []
        errors_yerr = []
        
        for p in all_params:
            if p in res["param_summaries"]:
                p_sum = res["param_summaries"][p]
                tgt = p_sum["target"]
                bias = p_sum["bias_mean"]
                std = p_sum["final_std"]
                
                if tgt is not None and tgt != 0:
                    biases_pct.append((bias / tgt) * 100)
                    errors_yerr.append((std / abs(tgt)) * 100)
                else:
                    # absolute bias if no target
                    biases_pct.append(bias if bias is not None else 0.0)
                    errors_yerr.append(std if std is not None else 0.0)
            else:
                biases_pct.append(0.0)
                errors_yerr.append(0.0)
                
        offset = (i - (len(run_ids) - 1) / 2) * width
        p_weight = res["config"].get("dedx_prior_weight", 0.5)
        prob_target = res["config"].get("probabilistic_sampling_target", True)
        label = f"Run {r} (prior_w={p_weight}, prob_tgt={prob_target})"
        
        plt.bar(x + offset, biases_pct, width, yerr=errors_yerr, capsize=3, label=label)
        
    plt.axhline(0, color="black", linestyle="-", alpha=0.5, lw=1)
    plt.xticks(x, all_params, rotation=30, ha="right")
    plt.ylabel("Relative Error (Final - True) / True [%]")
    plt.title("Parameter Fit Relative Biases (With SD Across Seeds)")
    plt.grid(True, axis="y")
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"Saved parameter bias plot to {save_path}")
    return plt.gcf()

def plot_dedx_mae(results, save_path=None):
    """
    Plots a bar chart comparing final dEdx fit Mean Absolute Error (MAE) for all runs.
    """
    run_ids = list(results.keys())
    maes = []
    stds = []
    labels = []
    
    for r in run_ids:
        res = results[r]
        if res["dedx_mae_mean"] is not None:
            maes.append(res["dedx_mae_mean"])
            stds.append(res["dedx_mae_std"])
            p_weight = res["config"].get("dedx_prior_weight", 0.5)
            prob_target = res["config"].get("probabilistic_sampling_target", True)
            labels.append(f"Run {r}\n(prior_w={p_weight}, prob_tgt={prob_target})")
            
    if not maes:
        print("No dEdx MAE data found to plot.")
        return None
        
    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, maes, yerr=stds, capsize=5, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"][:len(maes)], alpha=0.85, edgecolor='black', linewidth=0.5)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                 f"{height:.4f}", ha='center', va='bottom', fontsize=10, fontweight='bold')
                 
    plt.ylabel("Average dEdx MAE [MeV/cm]")
    plt.title("dEdx Fit Quality Comparison (Mean Absolute Error, Lower is Better)")
    plt.grid(True, axis="y")
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"Saved dEdx MAE plot to {save_path}")
    return plt.gcf()

def plot_dedx_residuals_comparison(results, save_path=None):
    """
    Plots overlaid histograms of dEdx residuals (Fitted - True) for all runs.
    Overlays the Initial (Pre-fit) residuals as a reference.
    """
    plt.figure(figsize=(10, 6))
    
    # Plot initial residuals once (should be similar for all since loc is similar)
    initial_plotted = False
    
    for r, res in results.items():
        resids = res.get("residuals")
        init_resids = res.get("init_residuals")
        
        if resids is None or len(resids) == 0:
            continue
            
        p_weight = res["config"].get("dedx_prior_weight", 0.5)
        prob_target = res["config"].get("probabilistic_sampling_target", True)
        
        # Plot Initial reference (only once)
        if not initial_plotted and init_resids is not None and len(init_resids) > 0:
            plt.hist(init_resids, bins=80, range=(-1.5, 1.5), density=True, 
                     color="gray", alpha=0.3, label=f"Initial (Pre-fit) dEdx Residuals, std={np.std(init_resids):.3f}")
            initial_plotted = True
            
        label = f"Run {r} (prior_w={p_weight}, prob_tgt={prob_target}), std={np.std(resids):.3f}"
        plt.hist(resids, bins=80, range=(-1.5, 1.5), density=True, 
                 alpha=0.6, label=label, histtype='step', lw=2)
                 
    plt.axvline(0, color="black", linestyle=":", alpha=0.7)
    plt.xlabel("dEdx Residual (Fitted - True) [MeV/cm]")
    plt.ylabel("Probability Density")
    plt.title("dEdx Fit Residuals Distribution Comparison")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"Saved dEdx residuals comparison plot to {save_path}")
    return plt.gcf()

def plot_dedx_histories(results, smoothing_window=1, save_path=None):
    """
    Plots convergence of dEdx MAE from prior and fitted dEdx mean over iterations.
    """
    # Check if there is dEdx history
    has_dedx_hist = False
    for res in results.values():
        if "dedx_mae_iter_mean" in res["averaged_history"] or "mean_dedx_iter_mean" in res["averaged_history"]:
            has_dedx_hist = True
            break
            
    if not has_dedx_hist:
        print("No dEdx history metrics found to plot.")
        return None
        
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left plot: dEdx MAE from prior loc
    ax = axes[0]
    for r, res in results.items():
        hist = res["averaged_history"]
        if "dedx_mae_iter_mean" in hist:
            iters = np.arange(len(hist["dedx_mae_iter_mean"]))
            
            # Apply smoothing
            mae_mean = smooth_curve(hist["dedx_mae_iter_mean"], smoothing_window)
            
            line, = ax.plot(iters, mae_mean, label=f"Run {r}", lw=2)
            if "dedx_mae_iter_std" in hist and res["num_seeds"] > 1:
                mae_std = smooth_curve(hist["dedx_mae_iter_std"], smoothing_window)
                ax.fill_between(iters, 
                                mae_mean - mae_std,
                                mae_mean + mae_std,
                                color=line.get_color(), alpha=0.15)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("MAE from prior loc [MeV/cm]")
    ax.set_title("dEdx MAE from Prior Location vs Iterations")
    ax.grid(True)
    ax.legend()
    
    # Right plot: fitted dEdx mean
    ax = axes[1]
    for r, res in results.items():
        hist = res["averaged_history"]
        if "mean_dedx_iter_mean" in hist:
            iters = np.arange(len(hist["mean_dedx_iter_mean"]))
            
            # Apply smoothing
            mean_dedx = smooth_curve(hist["mean_dedx_iter_mean"], smoothing_window)
            
            line, = ax.plot(iters, mean_dedx, label=f"Run {r}", lw=2)
            if "mean_dedx_iter_std" in hist and res["num_seeds"] > 1:
                mean_dedx_std = smooth_curve(hist["mean_dedx_iter_std"], smoothing_window)
                ax.fill_between(iters,
                                mean_dedx - mean_dedx_std,
                                mean_dedx + mean_dedx_std,
                                color=line.get_color(), alpha=0.15)
                                
            # Add target line if available in config
            cfg = res.get("config", {})
            try:
                target_mean = float(cfg.get('dedx_mean_constraint_target', 1.887))
                ax.axhline(target_mean, color=line.get_color(), linestyle=":", alpha=0.5)
            except Exception:
                pass
               
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Fitted dEdx Mean (Weighted) [MeV/cm]")
    ax.set_title("Fitted dEdx Mean vs Iterations")
    ax.grid(True)
    ax.legend()
    
    plt.suptitle("dEdx Diagnostic Histories Convergence", y=1.0)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"Saved dEdx history plots to {save_path}")
    return fig

def df_to_markdown(df, index=True):
    """
    Manually formats a pandas DataFrame as a Markdown table (no 'tabulate' dependency).
    """
    if index:
        df = df.reset_index()
    headers = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in df.itertuples(index=False):
        row_str = [str(val).replace("\n", " ") for val in row]
        lines.append("| " + " | ".join(row_str) + " |")
    return "\n".join(lines)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Compare fit results across different larnd-sim-jax runs.")
    parser.add_argument("--runs", nargs="+", default=["27318914", "27339284", "27347468", "27354407"],
                        help="Run IDs to compare.")
    parser.add_argument("--fit_dir", default="fit_result",
                        help="Directory containing the fit result folders.")
    parser.add_argument("--out_dir", default="plots_comparison",
                        help="Output directory to save summary plots and table.")
    parser.add_argument("--smooth", type=int, default=100,
                        help="Smoothing window size for convergence plots (default: 100).")
    args = parser.parse_args()
    
    print(f"Comparing runs: {args.runs} with smoothing window: {args.smooth}")
    results, df_configs, df_params = compare_runs(args.runs, fit_dir=args.fit_dir)
    
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Print configurations comparison
    print("\n" + "="*50)
    print("RUN CONFIGURATION COMPARISON")
    print("="*50)
    # Find columns that have varying values
    varying_cols = []
    for col in df_configs.columns:
        if df_configs[col].nunique() > 1:
            varying_cols.append(col)
    if varying_cols:
        print(df_configs[varying_cols].to_string())
    else:
        print("All runs have identical configurations.")
        print(df_configs.to_string())
        
    # Print parameters summary
    print("\n" + "="*50)
    print("FITTED PARAMETERS COMPARISON")
    print("="*50)
    pd.set_option('display.max_columns', 10)
    pd.set_option('display.width', 1000)
    print(df_params.to_string(index=False))
    
    # Save parameter table as markdown
    table_md_path = os.path.join(args.out_dir, "comparison_table.md")
    with open(table_md_path, "w") as f:
        f.write("# Fit Results Comparison Table\n\n")
        f.write("### Run Configuration Differences:\n\n")
        if varying_cols:
            f.write(df_to_markdown(df_configs[varying_cols], index=True) + "\n\n")
        else:
            f.write("All configurations are identical:\n\n")
            f.write(df_to_markdown(df_configs, index=True) + "\n\n")
        f.write("### Parameter Fit Details:\n\n")
        f.write(df_to_markdown(df_params, index=False) + "\n")
    print(f"\nSaved summary markdown table to {table_md_path}")
    
    # Generate and save plots
    plot_loss_convergence(results, smoothing_window=args.smooth, save_path=os.path.join(args.out_dir, "loss_convergence.png"))
    plot_param_convergence(results, smoothing_window=args.smooth, save_path=os.path.join(args.out_dir, "parameter_convergence.png"))
    plot_parameter_biases(results, save_path=os.path.join(args.out_dir, "parameter_biases.png"))
    plot_dedx_mae(results, save_path=os.path.join(args.out_dir, "dedx_fit_mae.png"))
    plot_dedx_residuals_comparison(results, save_path=os.path.join(args.out_dir, "dedx_residuals_comparison.png"))
    plot_dedx_histories(results, smoothing_window=args.smooth, save_path=os.path.join(args.out_dir, "dedx_histories_convergence.png"))
    
    print("\nComparison successfully completed! All plots are saved in plots_comparison/")

if __name__ == "__main__":
    main()
