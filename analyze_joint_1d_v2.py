import pickle, glob, os, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

PARAMS = ["Ab", "kb", "eField", "tran_diff", "long_diff", "lifetime", "shift_z", "shift_x", "shift_y"]
PARAM_LABELS = {
    "Ab": r"$A_b$", "kb": r"$k_b$", "eField": r"$E_\mathrm{field}$",
    "tran_diff": r"$D_T$", "long_diff": r"$D_L$", "lifetime": r"$\tau_e$",
    "shift_z": r"$\Delta z$", "shift_x": r"$\Delta x$", "shift_y": r"$\Delta y$",
}
PKL_V1 = "fit_result/joint_1d"
PKL_V2 = "fit_result/joint_1d_v2"
OUT_DIR = "plots/joint_1d_v2"
os.makedirs(OUT_DIR, exist_ok=True)

def load_latest(param, directory):
    pattern = f"{directory}/history_iter*_*_{param}_*.pkl"
    files = sorted(glob.glob(pattern),
                   key=lambda f: int(re.search(r"history_iter(\d+)", f).group(1)))
    if not files:
        return None, None
    h = pickle.load(open(files[-1], "rb"))
    return h, files[-1]

def smooth(x, w=30):
    if len(x) < w:
        return np.array(x)
    return np.convolve(x, np.ones(w) / w, mode="valid")

data_v1, data_v2 = {}, {}
for p in PARAMS:
    h, f = load_latest(p, PKL_V1)
    if h:
        data_v1[p] = h
        print(f"v1 {p}: {os.path.basename(f)}")
    h, f = load_latest(p, PKL_V2)
    if h:
        data_v2[p] = h
        print(f"v2 {p}: {os.path.basename(f)}")

# ── Figure 1: Parameter convergence comparison ────────────────────────────────
fig, axes = plt.subplots(3, 3, figsize=(16, 13))
axes = axes.flatten()
fig.suptitle("Joint 1D: v1 (20 batches, dr=0.99) vs v2 (100 batches, dr=0.999)", fontsize=11)

for idx, param in enumerate(PARAMS):
    ax = axes[idx]
    true_val = None
    for h, color, lbl in [(data_v1.get(param), "steelblue", "v1 b=20"),
                           (data_v2.get(param), "orangered",  "v2 b=100")]:
        if h is None:
            continue
        vals = np.array(h[f"{param}_iter"])
        tv = h.get(f"{param}_target", [None])[0]
        if tv is not None:
            true_val = tv
        iters = np.arange(len(vals))
        if tv is not None and abs(tv) > 1e-10:
            y = (vals - tv) / abs(tv) * 100
        else:
            y = vals
        ax.plot(iters, y, lw=0.4, alpha=0.25, color=color)
        sm = smooth(y, 50)
        ax.plot(np.arange(len(sm)) + 25, sm, lw=1.8, color=color,
                label=f"{lbl}  [{len(vals)-1}it] {vals[-1]:.4g}")
    if true_val is not None and abs(true_val) > 1e-10:
        ax.axhline(0, color="green", lw=1.5, ls="--", label=f"True={true_val:.4g}")
        ax.set_ylabel("Bias (%)")
    else:
        ax.set_ylabel("Value")
    ax.set_title(PARAM_LABELS[param], fontsize=10)
    ax.set_xlabel("Iteration")
    ax.legend(fontsize=6.5)
    ax.xaxis.set_major_locator(MaxNLocator(5))

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/comparison_convergence.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved comparison_convergence.png")

# ── Figure 2: Step time comparison ────────────────────────────────────────────
fig, axes = plt.subplots(3, 3, figsize=(16, 13))
axes = axes.flatten()
fig.suptitle("Step time per gradient step (s)  —  v1 vs v2", fontsize=11)

for idx, param in enumerate(PARAMS):
    ax = axes[idx]
    for h, color, lbl in [(data_v1.get(param), "steelblue", "v1 b=20"),
                           (data_v2.get(param), "orangered",  "v2 b=100")]:
        if h is None:
            continue
        t = np.array(h["step_time"])[1:]  # drop first (JIT warmup)
        ax.plot(np.arange(1, len(t)+1), t, lw=0.4, alpha=0.25, color=color)
        sm = smooth(t, 30)
        med = np.median(t)
        ax.plot(np.arange(15, len(sm)+15), sm, lw=1.8, color=color,
                label=f"{lbl} med={med:.1f}s")
    ax.set_title(PARAM_LABELS[param], fontsize=10)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("s / gradient step")
    ax.set_ylim(0, 40)
    ax.legend(fontsize=7)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/comparison_step_time.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved comparison_step_time.png")

# ── Figure 3: Loss comparison ──────────────────────────────────────────────────
fig, axes = plt.subplots(3, 3, figsize=(16, 13))
axes = axes.flatten()
fig.suptitle("Loss convergence  —  v1 vs v2", fontsize=11)

for idx, param in enumerate(PARAMS):
    ax = axes[idx]
    for h, color, lbl in [(data_v1.get(param), "steelblue", "v1"),
                           (data_v2.get(param), "orangered",  "v2")]:
        if h is None:
            continue
        loss = np.array(h["losses_iter"])
        ax.plot(np.arange(len(loss)), loss, lw=0.4, alpha=0.25, color=color)
        sm = smooth(loss, 50)
        ax.plot(np.arange(25, len(sm)+25), sm, lw=1.8, color=color,
                label=f"{lbl} final={loss[-1]:.0f}")
    ax.set_title(PARAM_LABELS[param], fontsize=10)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.legend(fontsize=7)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/comparison_loss.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved comparison_loss.png")

# ── Text summary ───────────────────────────────────────────────────────────────
print()
print(f"{'Param':<12} {'True':>10}  {'v1 final':>10} {'v1 bias':>8}  {'v2 iters':>8} {'v2 final':>10} {'v2 bias':>8}  {'step_v1':>7} {'step_v2':>7}")
print("=" * 95)
for param in PARAMS:
    h1 = data_v1.get(param)
    h2 = data_v2.get(param)
    tv = (h1 or h2 or {}).get(f"{param}_target", [None])[0]

    def fmt(h, key):
        if h is None:
            return "N/A"
        v = np.array(h[f"{key}_iter"])[-1]
        return f"{v:.5g}"

    def biasfmt(h):
        if h is None or tv is None or abs(tv) < 1e-10:
            return "N/A"
        v = np.array(h[f"{param}_iter"])[-1]
        return f"{(v - tv) / abs(tv) * 100:+.2f}%"

    def stepmed(h):
        if h is None:
            return "N/A"
        return f"{np.median(np.array(h['step_time'])[1:]):.1f}s"

    v2_niters = len(data_v2[param]["losses_iter"]) if param in data_v2 else "N/A"
    print(f"{param:<12} {str(tv or '?'):>10}  {fmt(h1, param):>10} {biasfmt(h1):>8}  "
          f"{str(v2_niters):>8} {fmt(h2, param):>10} {biasfmt(h2):>8}  "
          f"{stepmed(h1):>7} {stepmed(h2):>7}")
print("=" * 95)
