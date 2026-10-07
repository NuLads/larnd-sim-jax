import pickle, numpy as np, glob, os, gc, json, hashlib

BASE = "/sdf/group/neutrino/pgranger/larnd-sim-jax/fit_result"
OUT = "/lscratch/pgranger/tmp/claude-45184/-sdf-group-neutrino-pgranger-larnd-sim-jax/e77eab56-46d4-44e7-b115-2992a0b283ba/scratchpad/B_ensemble"

PARAMS = ["Ab", "eField", "tran_diff", "long_diff", "lifetime"]

TAGS = (
    [f"DSREP{i}" for i in range(1, 17)]
    + [f"TRU1D{i}" for i in (1, 3, 7, 8)]
    + [f"TRU2D{i}" for i in (1, 3, 7, 8)]
    + [f"TRU3D{i}" for i in (1, 3, 7, 8)]
    + [f"TPREP{i}" for i in (1, 7, 8)]
    + [f"BOTHTRU{i}" for i in (3, 7, 8)]
    + [f"DXTRU{i}" for i in (3, 7, 8)]
    + [f"DXFRZ{i}" for i in (3, 7, 8)]
)


def to_f(x):
    return float(np.asarray(x).ravel()[0])


def arr(lst):
    return np.array([to_f(x) for x in lst], dtype=float)


def tail_stats(a, tail_frac=0.10, ptp_window=2000, drift_frac=0.20):
    """a: 1-D array over iterations (index = iteration, possibly len iters+1)."""
    n = len(a)
    if n == 0:
        return dict(endpoint_mean=np.nan, endpoint_median=np.nan, ptp_last2000=np.nan,
                     tail_drift_pct=np.nan, n_tail=0)
    n_tail = max(1, int(round(n * tail_frac)))
    tail = a[-n_tail:]
    endpoint_mean = float(np.mean(tail))
    endpoint_median = float(np.median(tail))
    w = min(ptp_window, n)
    ptp = float(np.ptp(a[-w:]))
    n_drift = max(2, int(round(n * drift_frac)))
    drift_seg = a[-n_drift:]
    x = np.arange(len(drift_seg))
    if len(drift_seg) >= 2 and np.std(drift_seg) > 0:
        slope = np.polyfit(x, drift_seg, 1)[0]
    else:
        slope = 0.0
    span_val = slope * (len(drift_seg) - 1)
    ref = drift_seg[0] if drift_seg[0] != 0 else np.mean(np.abs(drift_seg)) + 1e-30
    drift_pct = 100.0 * span_val / ref
    return dict(
        endpoint_mean=endpoint_mean,
        endpoint_median=endpoint_median,
        ptp_last2000=ptp,
        tail_drift_pct=drift_pct,
        n_tail=n_tail,
    )


def hash_positions(tp):
    """true_positions: dict batch_idx -> array-like. Build a stable hash of concatenated values."""
    try:
        keys = sorted(tp.keys())
        h = hashlib.sha1()
        total_len = 0
        for k in keys[:5] + keys[-5:] if len(keys) > 10 else keys:
            v = np.asarray(tp[k])
            total_len += v.size
            h.update(v.tobytes() if v.dtype.kind in "fiu" else str(v).encode())
        # also full length sum over ALL keys (cheap, just .size)
        full_len = sum(np.asarray(tp[k]).size for k in tp.keys())
        return h.hexdigest()[:16], full_len
    except Exception as e:
        return f"ERR:{e}", -1


def seg_dx_total_len(sdc):
    try:
        return int(sum(np.asarray(v).size for v in sdc.values()))
    except Exception as e:
        return -1


def batch_parent_summary(bpi):
    try:
        keys = sorted(bpi.keys())
        allvals = []
        for k in keys:
            v = bpi[k]
            allvals.append(np.asarray(v).ravel())
        cat = np.concatenate(allvals) if allvals else np.array([])
        h = hashlib.sha1(cat.tobytes()).hexdigest()[:16] if cat.size else "EMPTY"
        return dict(n_batches=len(keys), total_entries=int(cat.size), hash=h)
    except Exception as e:
        return dict(n_batches=-1, total_entries=-1, hash=f"ERR:{e}")


def process(tag):
    d_dir = os.path.join(BASE, f"sci_full_{tag}")
    files = glob.glob(os.path.join(d_dir, "*.pkl"))
    if not files:
        return None, None
    fn = files[0]
    with open(fn, "rb") as f:
        d = pickle.load(f)

    c = d["config"]
    prov = d["provenance"]
    env = d.get("env_overrides", {})
    total_iter = d.get("total_iter", None)

    row = dict(
        tag=tag,
        file=fn,
        data_seed=getattr(c, "data_seed", None),
        seed=getattr(c, "seed", None),
        data_sz=getattr(c, "data_sz", None),
        random_nevents=getattr(c, "random_nevents", None),
        no_noise=getattr(c, "no_noise", None),
        no_noise_target=getattr(c, "no_noise_target", None),
        no_noise_guess=getattr(c, "no_noise_guess", None),
        iterations_cfg=getattr(c, "iterations", None),
        total_iter=total_iter,
        dedx_freeze_iter=getattr(c, "dedx_freeze_iter", None),
        dedx_prior_weight=getattr(c, "dedx_prior_weight", None),
        dedx_student_loc=getattr(c, "dedx_student_loc", None),
        chain_basis=getattr(c, "chain_basis", None),
        git_sha=prov.get("git_sha"),
        git_dirty=prov.get("git_dirty"),
        hostname=prov.get("hostname"),
        slurm_job_id=prov.get("slurm_job_id"),
        timestamp=prov.get("timestamp"),
        LARND_INIT_FROM_TRUTH=env.get("LARND_INIT_FROM_TRUTH", ""),
        LARND_CHAIN_FREEZE_ITER=env.get("LARND_CHAIN_FREEZE_ITER", ""),
        LARND_DEDX_INIT_FROM=env.get("LARND_DEDX_INIT_FROM", ""),
        LARND_CHAIN_DRIFT_W=env.get("LARND_CHAIN_DRIFT_W", ""),
    )

    # independence-check raw materials
    bpi = d.get("batch_parent_ids", {})
    row["batch_parent"] = batch_parent_summary(bpi)
    sdc = d.get("seg_dx_cache", {})
    row["seg_dx_total_len"] = seg_dx_total_len(sdc)
    tp = d.get("true_positions", {})
    h, full_len = hash_positions(tp)
    row["true_pos_hash"] = h
    row["true_pos_total_len"] = full_len

    lifetime_iter_arr = None

    for p in PARAMS:
        it = d.get(f"{p}_iter", None)
        tgt = d.get(f"{p}_target", None)
        init = d.get(f"{p}_init", None)
        if it is None:
            continue
        a = arr(it)
        target = to_f(tgt) if tgt is not None else None
        initv = to_f(init) if init is not None else None
        ts = tail_stats(a)
        err_mean = 100.0 * (ts["endpoint_mean"] / target - 1.0) if target else None
        err_median = 100.0 * (ts["endpoint_median"] / target - 1.0) if target else None
        row[f"{p}_target"] = target
        row[f"{p}_init"] = initv
        row[f"{p}_endpoint_mean"] = ts["endpoint_mean"]
        row[f"{p}_endpoint_median"] = ts["endpoint_median"]
        row[f"{p}_err_mean_pct"] = err_mean
        row[f"{p}_err_median_pct"] = err_median
        row[f"{p}_ptp_last2000_pct_of_target"] = (
            100.0 * ts["ptp_last2000"] / abs(target) if target else None
        )
        row[f"{p}_tail_drift_pct"] = ts["tail_drift_pct"]
        row[f"{p}_n_iters_in_series"] = len(a)
        if p == "lifetime":
            lifetime_iter_arr = a
            row["lifetime_inv_tau_target"] = 1.0 / target if target else None
            row["lifetime_inv_tau_fit"] = 1.0 / ts["endpoint_mean"] if ts["endpoint_mean"] else None
            if target:
                row["lifetime_inv_tau_err_pct"] = 100.0 * (row["lifetime_inv_tau_fit"] / row["lifetime_inv_tau_target"] - 1.0)

    li = d.get("losses_iter", None)
    if li is not None:
        la = arr(li)
        lts = tail_stats(la)
        row["loss_endpoint_mean"] = lts["endpoint_mean"]
        row["loss_endpoint_median"] = lts["endpoint_median"]
        row["loss_n_iters"] = len(la)

    pr = d.get("pos_residual_iter", None)
    if pr is not None:
        pa = arr(pr)
        n = len(pa)
        if n > 0:
            n_tail = max(1, int(round(n * 0.10)))
            row["pos_residual_tail_median"] = float(np.median(pa[-n_tail:]))
            row["pos_residual_tail_mean"] = float(np.mean(pa[-n_tail:]))
        row["pos_residual_n_iters"] = n

    dm = d.get("dedx_mae_iter", None)
    if dm is not None:
        da = arr(dm)
        n = len(da)
        if n > 0:
            n_tail = max(1, int(round(n * 0.10)))
            row["dedx_mae_tail_median"] = float(np.median(da[-n_tail:]))

    mdx = d.get("mean_dedx_iter", None)
    if mdx is not None:
        ma = arr(mdx)
        n = len(ma)
        if n > 0:
            n_tail = max(1, int(round(n * 0.10)))
            row["mean_dedx_tail_median"] = float(np.median(ma[-n_tail:]))

    del d
    gc.collect()
    return row, lifetime_iter_arr


def main():
    rows = []
    traces = {}
    for tag in TAGS:
        print("Processing", tag, flush=True)
        row, life = process(tag)
        if row is None:
            print("  MISSING:", tag)
            continue
        rows.append(row)
        if tag.startswith("DSREP") and life is not None:
            traces[tag] = life

    import csv

    # flatten batch_parent dict for csv
    flat_rows = []
    for r in rows:
        r2 = dict(r)
        bp = r2.pop("batch_parent")
        r2["batch_parent_n_batches"] = bp["n_batches"]
        r2["batch_parent_total_entries"] = bp["total_entries"]
        r2["batch_parent_hash"] = bp["hash"]
        flat_rows.append(r2)

    keys = []
    for r in flat_rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)

    with open(os.path.join(OUT, "per_run_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in flat_rows:
            w.writerow(r)

    with open(os.path.join(OUT, "per_run_summary.json"), "w") as f:
        json.dump(flat_rows, f, indent=2, default=str)

    np.savez(os.path.join(OUT, "dsrep_lifetime_traces.npz"), **traces)
    print("DONE. n_rows=", len(rows))


if __name__ == "__main__":
    main()
