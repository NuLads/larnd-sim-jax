import csv, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/lscratch/pgranger/tmp/claude-45184/-sdf-group-neutrino-pgranger-larnd-sim-jax/e77eab56-46d4-44e7-b115-2992a0b283ba/scratchpad/B_ensemble"

# Okabe-Ito colorblind-safe palette
C = dict(blue="#0072B2", orange="#E69F00", green="#009E73", vermillion="#D55E00",
         purple="#CC79A7", skyblue="#56B4E9", yellow="#F0E442", black="#000000",
         grey="#999999")

with open(f"{OUT}/per_run_summary.csv") as f:
    rows = list(csv.DictReader(f))
R = {r["tag"]: r for r in rows}

def v(tag, key):
    return float(R[tag][key])

plt.rcParams.update({
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

# ---------------- F1: strip plot ----------------
dsrep_tags = [f"DSREP{i}" for i in range(1, 17)]
dsrep_life = np.array([v(t, "lifetime_err_mean_pct") for t in dsrep_tags])

tru_groups = {
    "TRU1D\n(1160 us)": [f"TRU1D{i}" for i in (1, 3, 7, 8)],
    "TRU2D\n(2392 us)": [f"TRU2D{i}" for i in (1, 3, 7, 8)],
    "TRU3D\n(4518 us)": [f"TRU3D{i}" for i in (1, 3, 7, 8)],
}
other_groups = {
    "BOTHTRU\n(both @truth)": [f"BOTHTRU{i}" for i in (3, 7, 8)],
    "DXTRU\n(dEdx @truth)": [f"DXTRU{i}" for i in (3, 7, 8)],
    "DXFRZ\n(dEdx frozen\n@own fit)": [f"DXFRZ{i}" for i in (3, 7, 8)],
}

fig, ax = plt.subplots(figsize=(11, 5.5))
rng = np.random.default_rng(0)

groups_order = [("DSREP\n(production,\ntruth 2406 us)", dsrep_tags, C["blue"])]
for k, (name, tags) in enumerate(tru_groups.items()):
    groups_order.append((name, tags, C["skyblue"]))
for name, tags in other_groups.items():
    col = C["orange"] if "BOTHTRU" in name else (C["green"] if "DXTRU" in name else C["vermillion"])
    groups_order.append((name, tags, col))

xpos = np.arange(len(groups_order))
for xi, (name, tags, col) in zip(xpos, groups_order):
    vals = np.array([v(t, "lifetime_err_mean_pct") for t in tags])
    jitter = rng.uniform(-0.13, 0.13, size=len(vals))
    ax.scatter(xi + jitter, vals, color=col, edgecolor="black", linewidth=0.4,
               s=42, zorder=3, alpha=0.9)
    mean = vals.mean()
    se = vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0
    ax.errorbar([xi], [mean], yerr=[se], fmt="_", color="black", markersize=22,
                markeredgewidth=2.2, elinewidth=1.8, capsize=5, zorder=4)

ax.axhline(0, color="black", linewidth=0.8, linestyle="--", zorder=1)
ax.set_xticks(xpos)
ax.set_xticklabels([f"{g[0]}\n(n={len(g[1])})" for g in groups_order], fontsize=8.5)
ax.set_ylabel("Lifetime error, (fit/target − 1) × 100  [%]")
ax.set_title("Lifetime error by arm — DSREP ensemble (n=16), Arm B truth points (n=4 each),\n"
              "and nuisance-pinning ceiling tests (n=3 each). Black bars: mean ± SE.")
plt.tight_layout()
plt.savefig(f"{OUT}/F1_lifetime_error_by_arm.png", dpi=150)
plt.close()

# ---------------- F2: traces vs iteration, DSREP1-16 ----------------
data = np.load(f"{OUT}/dsrep_lifetime_traces.npz")
target = v("DSREP1", "lifetime_target")

fig, ax = plt.subplots(figsize=(8.5, 5.5))
cmap = plt.get_cmap("viridis")
final_errs = {t: v(t, "lifetime_err_mean_pct") for t in dsrep_tags}
order = sorted(dsrep_tags, key=lambda t: final_errs[t])
norm = plt.Normalize(vmin=min(final_errs.values()), vmax=max(final_errs.values()))

for t in order:
    a = data[t]
    it = np.arange(len(a))
    err_pct = 100.0 * (a / target - 1.0)
    ax.plot(it, err_pct, color=cmap(norm(final_errs[t])), linewidth=1.0, alpha=0.85)

ax.axhline(0, color="black", linewidth=1.0, linestyle="--", zorder=1, label="truth")
ax.set_xlabel("Iteration")
ax.set_ylabel("Lifetime relative error [%]")
ax.set_title("Lifetime convergence traces, 16 DSREP replicas\n(color = final tail-mean error)")
sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
cb = plt.colorbar(sm, ax=ax)
cb.set_label("final error [%]")
ax.legend(loc="upper right", fontsize=8)
plt.tight_layout()
plt.savefig(f"{OUT}/F2_dsrep_lifetime_traces.png", dpi=150)
plt.close()

# ---------------- F3: 5-panel per-parameter error distributions, DSREP ----------------
PARAMS = ["Ab", "eField", "tran_diff", "long_diff", "lifetime"]
PLABEL = {"Ab": "A$_b$", "eField": "E field", "tran_diff": "transverse diff.",
          "long_diff": "longitudinal diff.", "lifetime": "lifetime"}

fig, axes = plt.subplots(1, 5, figsize=(15, 4), sharey=False)
for ax, p in zip(axes, PARAMS):
    vals = np.array([v(t, f"{p}_err_mean_pct") for t in dsrep_tags])
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", zorder=1)
    jitter = rng.uniform(-0.12, 0.12, size=len(vals))
    ax.scatter(jitter, vals, color=C["blue"], edgecolor="black", linewidth=0.4, s=38, zorder=3)
    mean = vals.mean()
    se = vals.std(ddof=1) / np.sqrt(len(vals))
    ax.errorbar([0], [mean], yerr=[se], fmt="_", color=C["vermillion"], markersize=24,
                markeredgewidth=2.4, elinewidth=2, capsize=6, zorder=4)
    ax.set_xticks([])
    ax.set_title(f"{PLABEL[p]}\nmean={mean:+.2f}% SE={se:.2f}%", fontsize=9.5)
    ax.set_xlim(-0.4, 0.4)
axes[0].set_ylabel("Error, (fit/target − 1) × 100  [%]")
fig.suptitle("DSREP1-16: per-parameter endpoint error distribution (n=16 each)", y=1.03)
plt.tight_layout()
plt.savefig(f"{OUT}/F3_dsrep_param_distributions.png", dpi=150, bbox_inches="tight")
plt.close()

print("Figures written.")
