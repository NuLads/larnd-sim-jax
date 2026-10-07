import csv, numpy as np, json

OUT = "/lscratch/pgranger/tmp/claude-45184/-sdf-group-neutrino-pgranger-larnd-sim-jax/e77eab56-46d4-44e7-b115-2992a0b283ba/scratchpad/B_ensemble"

with open(f"{OUT}/per_run_summary.csv") as f:
    rows = list(csv.DictReader(f))
R = {r["tag"]: r for r in rows}

rng = np.random.default_rng(12345)


def bootstrap_ci(x, n_boot=20000, alpha=0.32):
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2:
        return (np.nan, np.nan)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = x[idx].mean(axis=1)
    lo = np.percentile(means, 100 * alpha / 2)
    hi = np.percentile(means, 100 * (1 - alpha / 2))
    return lo, hi


def summarize(name, x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    mean = np.mean(x)
    se = np.std(x, ddof=1) / np.sqrt(n) if n > 1 else np.nan
    med = np.median(x)
    lo, hi = bootstrap_ci(x)
    npos = int(np.sum(x > 0))
    t = mean / se if se and se == se and se != 0 else np.nan
    print(f"{name}: n={n} mean={mean:+.3f} SE={se:.3f} median={med:+.3f} "
          f"boot68%=[{lo:+.3f},{hi:+.3f}] n_pos={npos}/{n} t={t:.2f}")
    return dict(n=n, mean=mean, se=se, median=med, boot_lo=lo, boot_hi=hi, n_pos=npos, t=t)


PARAMS = ["Ab", "eField", "tran_diff", "long_diff", "lifetime"]

print("=" * 70)
print("C1: DSREP1-16 lifetime bias")
dsrep_tags = [f"DSREP{i}" for i in range(1, 17)]
life_err = [float(R[t]["lifetime_err_mean_pct"]) for t in dsrep_tags]
summarize("DSREP lifetime err% (endpoint=mean tail)", life_err)
life_err_med = [float(R[t]["lifetime_err_median_pct"]) for t in dsrep_tags]
summarize("DSREP lifetime err% (endpoint=median tail)", life_err_med)

# also relative 1/tau error
inv_err = [float(R[t]["lifetime_inv_tau_err_pct"]) for t in dsrep_tags]
summarize("DSREP 1/tau err%", inv_err)

print()
print("Per-parameter for DSREP:")
for p in PARAMS:
    vals = [float(R[t][f"{p}_err_mean_pct"]) for t in dsrep_tags]
    summarize(f"DSREP {p} err%", vals)

print()
print("Convergence (ptp last2000, %target) DSREP:")
for p in PARAMS:
    vals = [float(R[t][f"{p}_ptp_last2000_pct_of_target"]) for t in dsrep_tags]
    vals = np.array(vals)
    print(f"  {p}: max={vals.max():.3f}% mean={vals.mean():.3f}% (all <=0.5%? {(vals<=0.5).all()})")
    print("    per-run:", {t: round(float(R[t][f'{p}_ptp_last2000_pct_of_target']),3) for t in dsrep_tags})

print()
print("Tail drift (% over last 20%% window) DSREP lifetime:")
vals = [float(R[t]["lifetime_tail_drift_pct"]) for t in dsrep_tags]
print(" ", vals)

print("=" * 70)
print("C2: TRU arms per truth point")
for k, tp in [(1, "1160us"), (2, "2392us"), (3, "4518us")]:
    tags = [f"TRU{k}D{i}" for i in (1, 3, 7, 8)]
    vals = [float(R[t]["lifetime_err_mean_pct"]) for t in tags]
    summarize(f"TRU{k}D (truth {tp})", vals)

# grand mean over all TRU + DSREP-as-truth0, 15/16 style check requested by digest (4 truth pts x 4)
tp0_tags = [f"DSREP{i}" for i in (1, 3, 7, 8)]
tp0_vals = [float(R[t]["lifetime_err_mean_pct"]) for t in tp0_tags]
summarize("truth0 (DSREP subset 1/3/7/8, matched to TRU design)", tp0_vals)

all_arm_b = tp0_vals.copy()
for k in (1, 2, 3):
    tags = [f"TRU{k}D{i}" for i in (1, 3, 7, 8)]
    all_arm_b += [float(R[t]["lifetime_err_mean_pct"]) for t in tags]
summarize("Arm B grand (4 truth pts x 4 samples, n=16)", all_arm_b)
npos = sum(1 for v in all_arm_b if v > 0)
print(f"  positive count: {npos}/{len(all_arm_b)}")

print("=" * 70)
print("C3: BOTHTRU / DXTRU / DXFRZ")
for grp in ["BOTHTRU", "DXTRU", "DXFRZ"]:
    tags = [f"{grp}{i}" for i in (3, 7, 8)]
    print(f"-- {grp} --")
    for p in PARAMS:
        vals = [float(R[t][f"{p}_err_mean_pct"]) for t in tags]
        summarize(f"  {grp} {p} err%", vals)
    mean_abs_err = []
    for t in tags:
        mean_abs_err.append(np.mean([abs(float(R[t][f"{p}_err_mean_pct"])) for p in PARAMS]))
    print(f"  mean|err| per seed: {dict(zip(tags, [round(v,3) for v in mean_abs_err]))}, "
          f"avg={np.mean(mean_abs_err):.3f}")
    pos = [float(R[t].get("pos_residual_tail_median", np.nan)) for t in tags]
    print(f"  pos_residual tail median (data units): {dict(zip(tags, pos))}")

# paired comparison DXFRZ vs DXTRU position (same seeds 3/7/8), and vs DSREP-matched(own baseline)
print()
print("Paired DXTRU vs DXFRZ position (same seed):")
for i in (3, 7, 8):
    pt = float(R[f"DXTRU{i}"].get("pos_residual_tail_median", np.nan))
    pf = float(R[f"DXFRZ{i}"].get("pos_residual_tail_median", np.nan))
    pb = float(R[f"DSREP{i}"].get("pos_residual_tail_median", np.nan))
    print(f"  seed{i}: DSREP(own fit)={pb:.4f} DXFRZ(frozen@own)={pf:.4f} DXTRU(frozen@truth)={pt:.4f}")

print("=" * 70)
print("C4: A_b / eField solved? long_diff worst? (DSREP ensemble)")
for p in PARAMS:
    vals = [abs(float(R[t][f"{p}_err_mean_pct"])) for t in dsrep_tags]
    summarize(f"DSREP |{p}| err%", vals)

print("=" * 70)
print("C5: convergence check summary already printed above (ptp last2000)")

print("=" * 70)
print("Independence / provenance checks")
for t in dsrep_tags:
    r = R[t]
    print(t, "git=", r["git_sha"][:8], "dirty=", r["git_dirty"], "hash=", r["batch_parent_hash"],
          "seg_dx_len=", r["seg_dx_total_len"], "iters=", r["total_iter"])

print()
print("git shas used, by tag group:")
groups = {
    "DSREP1-8": [f"DSREP{i}" for i in range(1,9)],
    "DSREP9-16": [f"DSREP{i}" for i in range(9,17)],
    "TRU*D1": [f"TRU{k}D1" for k in (1,2,3)],
    "TRU*D3": [f"TRU{k}D3" for k in (1,2,3)],
    "TRU*D7": [f"TRU{k}D7" for k in (1,2,3)],
    "TRU*D8": [f"TRU{k}D8" for k in (1,2,3)],
    "BOTHTRU": [f"BOTHTRU{i}" for i in (3,7,8)],
    "DXTRU": [f"DXTRU{i}" for i in (3,7,8)],
    "DXFRZ": [f"DXFRZ{i}" for i in (3,7,8)],
    "TPREP": [f"TPREP{i}" for i in (1,7,8)],
}
for g, tags in groups.items():
    shas = set(R[t]["git_sha"][:10] for t in tags)
    print(g, "-> git shas:", shas)
