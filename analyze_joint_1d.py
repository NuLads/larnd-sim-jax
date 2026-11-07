import pickle, glob, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator

PARAMS = ["Ab", "kb", "eField", "tran_diff", "long_diff", "lifetime", "shift_z", "shift_x", "shift_y"]

PARAM_LABELS = {
    "Ab": r"$A_b$",
    "kb": r"$k_b$",
    "eField": r"$E_\mathrm{field}$ (kV/cm)",
    "tran_diff": r"$D_T$ (cm²/s)",
    "long_diff": r"$D_L$ (cm²/s)",
    "lifetime": r"$\tau_e$ (µs)",
    "shift_z": r"$\Delta z$ (cm)",
    "shift_x": r"$\Delta x$ (cm)",
    "shift_y": r"$\Delta y$ (cm)",
}

PKL_DIR = "fit_result/joint_1d"
OUT_DIR = "plots/joint_1d"
os.makedirs(OUT_DIR, exist_ok=True)

def smooth(x, w=50):
    if len(x) < w:
        return np.array(x)
    kernel = np.ones(w) / w
    return np.convolve(x, kernel, mode='valid')

def load_checkpoint(param):
    pattern = f"{PKL_DIR}/history_iter5000_joint_1d_{param}_*.pkl"
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No checkpoint found for {param}: {pattern}")
    with open(files[-1], 'rb') as f:
        return pickle.load(f)

# Load all
data = {}
for p in PARAMS:
    try:
        data[p] = load_checkpoint(p)
        print(f"Loaded {p}")
    except FileNotFoundError as e:
        print(f"MISSING: {e}")

# ── Figure 1: Summary — all 9 params, bias vs iteration ──────────────────────
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()
fig.suptitle("Joint 1D Fits: Parameter Convergence (5000 iters, 20 batches, 50cm)", fontsize=13)

for idx, param in enumerate(PARAMS):
    ax = axes[idx]
    if param not in data:
        ax.text(0.5, 0.5, "MISSING", ha='center', va='center', transform=ax.transAxes)
        continue
    h = data[param]
    key_iter = f"{param}_iter"
    vals = np.array(h[key_iter])
    true_val = h.get(f"{param}_target", [None])[0]
    init_val = h.get(f"{param}_init", [None])[0]
    iters = np.arange(len(vals))

    if true_val is not None and abs(true_val) > 1e-12:
        bias = (vals - true_val) / abs(true_val) * 100
        ax.axhline(0, color='green', lw=1.5, ls='--', label=f'True: {true_val:.4g}')
        ax.set_ylabel("Bias (%)")
        # Mark init
        init_bias = (init_val - true_val) / abs(true_val) * 100 if init_val is not None else None
        if init_bias is not None:
            ax.axhline(init_bias, color='gray', lw=1, ls=':', alpha=0.6, label=f'Init: {init_val:.4g}')
    else:
        bias = vals
        ax.axhline(0, color='green', lw=1.5, ls='--', label='True=0')
        ax.set_ylabel("Fitted value")

    ax.plot(iters, bias, lw=0.5, alpha=0.4, color='steelblue')
    if len(bias) > 50:
        sm = smooth(bias, 100)
        ax.plot(np.arange(len(sm)) + 50, sm, lw=1.5, color='steelblue', label='Smoothed')

    final_bias = bias[-1]
    label_str = f"Final: {vals[-1]:.4g}"
    if true_val is not None and abs(true_val) > 1e-12:
        label_str += f"\nBias: {final_bias:+.2f}%"
    ax.set_title(f"{PARAM_LABELS[param]}  —  {label_str}", fontsize=9)
    ax.set_xlabel("Iteration")
    ax.legend(fontsize=7, loc='best')
    ax.xaxis.set_major_locator(MaxNLocator(5))

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/summary_convergence.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved summary_convergence.png")

# ── Figure 2: Loss + aux quantities for all params ────────────────────────────
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()
fig.suptitle("Joint 1D Fits: Total Loss Convergence", fontsize=13)

for idx, param in enumerate(PARAMS):
    ax = axes[idx]
    if param not in data:
        continue
    h = data[param]
    loss = np.array(h['losses_iter'])
    iters = np.arange(len(loss))
    ax.plot(iters, loss, lw=0.5, alpha=0.3, color='orangered')
    sm = smooth(loss, 100)
    ax.plot(np.arange(len(sm)) + 50, sm, lw=1.5, color='orangered')
    ax.set_title(f"{PARAM_LABELS[param]}  (final={loss[-1]:.0f})", fontsize=9)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.xaxis.set_major_locator(MaxNLocator(5))

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/summary_loss.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved summary_loss.png")

# ── Figure 3: Position residual + mean dEdx for all params ───────────────────
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()
fig.suptitle("Joint 1D Fits: Position Residual (cm)", fontsize=13)

for idx, param in enumerate(PARAMS):
    ax = axes[idx]
    if param not in data:
        continue
    h = data[param]
    pos = np.array(h['pos_residual_iter'])
    iters = np.arange(len(pos))
    ax.plot(iters, pos, lw=0.5, alpha=0.4, color='purple')
    sm = smooth(pos, 100)
    ax.plot(np.arange(len(sm)) + 50, sm, lw=1.5, color='purple')
    ax.set_title(f"{PARAM_LABELS[param]}  (final={pos[-1]:.4f} cm)", fontsize=9)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Mean pos. residual (cm)")
    ax.xaxis.set_major_locator(MaxNLocator(5))

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/summary_pos_residual.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved summary_pos_residual.png")

# ── Figure 4: dEdx MAE for all params ─────────────────────────────────────────
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()
fig.suptitle("Joint 1D Fits: dE/dx MAE (MeV/cm)", fontsize=13)

for idx, param in enumerate(PARAMS):
    ax = axes[idx]
    if param not in data:
        continue
    h = data[param]
    mae = np.array(h['dedx_mae_iter'])
    mean_dedx = np.array(h['mean_dedx_iter'])
    iters = np.arange(len(mae))
    ax.plot(iters, mae, lw=0.5, alpha=0.4, color='teal', label='MAE')
    sm = smooth(mae, 100)
    ax.plot(np.arange(len(sm)) + 50, sm, lw=1.5, color='teal')
    ax.axhline(0, color='green', lw=1, ls='--')
    ax2 = ax.twinx()
    ax2.plot(iters, mean_dedx, lw=0.5, alpha=0.3, color='brown')
    ax2.axhline(1.887, color='brown', lw=1, ls='--', alpha=0.6)
    ax2.set_ylabel("Mean dE/dx", color='brown', fontsize=7)
    ax2.tick_params(labelsize=7)
    ax.set_title(f"{PARAM_LABELS[param]}  MAE_final={mae[-1]:.3f}", fontsize=9)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("dE/dx MAE (MeV/cm)")
    ax.xaxis.set_major_locator(MaxNLocator(5))

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/summary_dedx.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved summary_dedx.png")

# ── Figure 5: Final results summary bar chart ─────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
biases = []
labels = []
for param in PARAMS:
    if param not in data:
        biases.append(None); labels.append(param); continue
    h = data[param]
    vals = np.array(h[f"{param}_iter"])
    true_val = h.get(f"{param}_target", [None])[0]
    final = vals[-1]
    if true_val is not None and abs(true_val) > 1e-12:
        biases.append((final - true_val) / abs(true_val) * 100)
    else:
        biases.append(final)  # absolute value for zero-true params
    labels.append(PARAM_LABELS[param])

x = np.arange(len(PARAMS))
colors = ['red' if b is not None and abs(b) > 5 else 'steelblue' for b in biases]
bars = ax.bar(x, [b if b is not None else 0 for b in biases], color=colors, alpha=0.8, edgecolor='k', lw=0.5)
ax.axhline(0, color='green', lw=1.5, ls='--', label='True value')
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel("Bias (%) or final value for shift params")
ax.set_title("Joint 1D Fits: Final Parameter Bias after 5000 iterations")
ax.legend()

for bar, b in zip(bars, biases):
    if b is not None:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (0.1 if b >= 0 else -0.3),
                f"{b:.2f}%", ha='center', va='bottom', fontsize=8, rotation=45)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/summary_bias_bar.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved summary_bias_bar.png")

# ── Print text summary ──────────────────────────────────────────────────────
print("\n" + "="*70)
print(f"{'Param':<12} {'True':>12} {'Init':>12} {'Final':>12} {'Bias%':>9} {'Loss_i':>9} {'Loss_f':>9} {'PosRes_f':>10}")
print("="*70)
for param in PARAMS:
    if param not in data:
        print(f"{param:<12} {'MISSING':>12}")
        continue
    h = data[param]
    vals = np.array(h[f"{param}_iter"])
    true_val = h.get(f"{param}_target", [None])[0]
    init_val = h.get(f"{param}_init", [None])[0]
    final = vals[-1]
    loss_i = h['losses_iter'][0]
    loss_f = h['losses_iter'][-1]
    pos_f = h['pos_residual_iter'][-1]
    if true_val is not None and abs(true_val) > 1e-12:
        bias = (final - true_val) / abs(true_val) * 100
        print(f"{param:<12} {true_val:>12.5g} {init_val:>12.5g} {final:>12.5g} {bias:>+9.2f}% {loss_i:>9.0f} {loss_f:>9.0f} {pos_f:>10.4f}")
    else:
        print(f"{param:<12} {str(true_val):>12} {str(init_val):>12} {final:>12.5g} {'(abs)':>9} {loss_i:>9.0f} {loss_f:>9.0f} {pos_f:>10.4f}")
print("="*70)
