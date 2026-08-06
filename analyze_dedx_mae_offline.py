"""Offline truth-referenced dE/dx MAE for checkpoints written BEFORE the in-fitter matcher existed.

WHY. `dedx_mae_iter` is computed against `--input_file_sim`. In every S3/S4 run that is the
straight-line guess, so the logged series is |fitted - starting values|, not an error against
truth. Since the guess file carries only ~40% of the true dE/dx spread, moving away from it is what
a working fit SHOULD do -- which is why "the dE/dx MAE ends ~30% worse than it starts" was never
evidence of a defect. `fit_params._build_true_dedx_matched` now logs the correct reference, but
only for runs started after it landed. This reconstructs the same quantity for everything already
on disk.

WHAT IS AVAILABLE OFFLINE. Old checkpoints lack `seg_dx_cache`, so per-parent lengths cannot be
read back. They do carry:
  * `chain_contexts[b][t]['idxs']`      -- the sub-step row indices belonging to track t
  * `chain_contexts[b][t]['track_id']`  -- (event_id, traj_id)
  * `batch_parent_ids[b]`               -- sub-step row -> parent segment index
  * `dedx_cache[b]['log_dedx']`         -- the fitted values

Sub-steps are produced at a FIXED `electron_sampling_resolution` (0.01 cm), so every sub-step has
the same length and a parent segment's length is simply its sub-step COUNT. Only *fractional*
arc-length spans within a track are needed, so the constant cancels entirely and no dx is required.

MATCHING. Identical in spirit to the in-fitter version: sim and target have different segmentations
(10.08M vs 3.03M), so there is no row correspondence -- only the trajectory is shared. For each
track, compare in normalised arc length and INTEGRATE dE over each parent's span:

    dEdx_true(p) = [E(a1) - E(a0)] / [(a1 - a0) * L_true]

Integrating rather than point-sampling matters: dE/dx is an average over a finite length and the
spans differ by ~3.3x between the files, so point sampling would alias true fluctuations and report
MAE driven by segmentation rather than by fit quality.
"""
import argparse, glob, os, pickle
import numpy as np
import h5py

TRUE = '/sdf/data/neutrino/cyifan/diffsim_input/true_through_muon_edep_10cm_vol1cm.h5'


def build_true_index(path=TRUE):
    """key = event_id*1e7 + traj_id  ->  (dx, dE) arrays in file order along the track."""
    with h5py.File(path, 'r') as f:
        s = f['segments'][:]
    key = s['event_id'].astype(np.int64) * 10 ** 7 + s['traj_id'].astype(np.int64)
    order = np.argsort(key, kind='stable')          # stable: preserves along-track order
    key_s = key[order]
    dx = s['dx'].astype(np.float64)[order]
    de = (s['dEdx'].astype(np.float64) * s['dx'].astype(np.float64))[order]
    uniq, start = np.unique(key_s, return_index=True)
    end = np.r_[start[1:], len(key_s)]
    return {int(k): (dx[a:b], de[a:b]) for k, a, b in zip(uniq, start, end)}


def matched_for_batch(h, b, tindex):
    """-> (fitted, matched) arrays over the parents that could be matched."""
    ctxs = h['chain_contexts'].get(b)
    pids = h['batch_parent_ids'].get(b)
    dc = h['dedx_cache'].get(b)
    if ctxs is None or pids is None or dc is None:
        return None, None
    pids = np.asarray(pids)
    fitted_all = np.exp(np.asarray(dc['log_dedx'] if isinstance(dc, dict) else dc, dtype=np.float64))
    fit_out, tru_out = [], []
    for c in ctxs:
        idxs = np.asarray(c['idxs']).ravel()
        idxs = idxs[(idxs >= 0) & (idxs < len(pids))]
        p = pids[idxs]
        p = p[p >= 0]
        if p.size == 0:
            continue
        parents, counts = np.unique(p, return_counts=True)      # sorted == along-track order
        if parents.max() >= len(fitted_all):
            continue
        # sub-steps are uniform in length, so counts ARE lengths up to a constant that cancels
        edges = np.concatenate([[0.0], np.cumsum(counts)]) / counts.sum()
        ev, tr = c['track_id']
        rec = tindex.get(int(round(float(ev))) * 10 ** 7 + int(round(float(tr))))
        if rec is None:
            continue
        tdx, tde = rec
        if len(tdx) < 2 or tdx.sum() <= 0:
            continue
        a_true = np.concatenate([[0.0], np.cumsum(tdx)]) / tdx.sum()
        E_true = np.concatenate([[0.0], np.cumsum(tde)])
        Ltot = tdx.sum()
        E0 = np.interp(edges[:-1], a_true, E_true)
        E1 = np.interp(edges[1:], a_true, E_true)
        span = (edges[1:] - edges[:-1]) * Ltot
        good = span > 1e-9
        if not good.any():
            continue
        fit_out.append(fitted_all[parents[good]])
        tru_out.append((E1 - E0)[good] / span[good])
    if not fit_out:
        return None, None
    return np.concatenate(fit_out), np.concatenate(tru_out)


def analyse(path, tindex):
    h = pickle.load(open(path, 'rb'))
    if 'chain_contexts' not in h or 'dedx_cache' not in h:
        return None
    F, T, S = [], [], []
    sim_ref = h.get('true_dedx_cache', {})
    for b in sorted(h['dedx_cache']):
        f, t = matched_for_batch(h, b, tindex)
        if f is None:
            continue
        F.append(f); T.append(t)
        s = sim_ref.get(b)
        if s is not None:
            S.append(np.asarray(s, dtype=np.float64))
    if not F:
        return None
    F, T = np.concatenate(F), np.concatenate(T)
    S = np.concatenate(S) if S else None
    out = dict(n=len(F), mae_truth=float(np.mean(np.abs(F - T))),
               fit_mean=float(F.mean()), fit_sd=float(F.std()),
               tru_mean=float(T.mean()), tru_sd=float(T.std()))
    if S is not None and len(S) == len(F):
        out['mae_sim'] = float(np.mean(np.abs(F - S)))
        out['sim_mean'] = float(S.mean()); out['sim_sd'] = float(S.std())
        # the reference the fit STARTED from: MAE of the initial values against truth
        out['mae_truth_at_init'] = float(np.mean(np.abs(S - T)))
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('dirs', nargs='+', help='fit_result/<run> directories')
    ap.add_argument('--seed', default='*')
    a = ap.parse_args()
    print('indexing the true file ...')
    tindex = build_true_index()
    print(f'  {len(tindex)} trajectories indexed\n')
    hdr = (f"{'run':26s}{'seed':>5s}{'n':>8s}{'MAE vs TRUTH':>14s}{'at init':>10s}"
           f"{'MAE vs SIM':>12s}{'fit sd':>9s}{'truth sd':>10s}")
    print(hdr); print('-' * len(hdr))
    for d in a.dirs:
        for f in sorted(glob.glob(os.path.join(d, f'history_iter*seed{a.seed}.pkl'))):
            r = analyse(f, tindex)
            if r is None:
                print(f'{os.path.basename(d):26s} {os.path.basename(f)}: not reconstructable')
                continue
            sd = f.split('seed')[1].split('.')[0]
            init = f"{r.get('mae_truth_at_init', float('nan')):.4f}"
            msim = f"{r.get('mae_sim', float('nan')):.4f}"
            print(f"{os.path.basename(d):26s}{sd:>5s}{r['n']:>8d}{r['mae_truth']:>14.4f}"
                  f"{init:>10s}{msim:>12s}{r['fit_sd']:>9.4f}{r['tru_sd']:>10.4f}")
