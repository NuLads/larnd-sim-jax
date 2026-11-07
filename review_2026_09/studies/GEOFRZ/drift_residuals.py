"""Fitted-minus-true DRIFT displacement vs drift distance, from the saved geometry of each fit.

For every track: fitted points = nominal line + spline displacement (e1, e2 frame, as in
fit_params._chain_warp_spline); true points = true_positions at the same arc-length fractions.
Delta is projected PERPENDICULAR to the track (so along-track slippage cannot leak in), and its
drift-axis (z) component is signed as a change of DRIFT DISTANCE: positive = charge placed farther
from the anode. Anodes at z = +-30.58975 cm (library tpc_borders), cathode ~0.
Errors are over TRACKS (points along one track are correlated): per-track bin means, then mean +- SE."""
import glob, pickle, json, sys
import numpy as np

R = '/sdf/group/neutrino/pgranger/larnd-sim-jax/fit_result'
OUT = '/sdf/scratch/neutrino/pgranger/review_2026_09/GEOFRZ'
Z_ANODE = 30.58975
DRIFT_AXIS = np.array([0., 0., 1.])
EDGES = np.arange(0, 32, 2.0)
f1 = lambda x: float(np.ravel(np.asarray(x))[0])

def dir_from_angles(th, ph):
    st = np.sin(th); return np.array([st * np.cos(ph), st * np.sin(ph), np.cos(th)])

def frame(u0):                                   # == fit_params._transverse_frame, 'drift' frame
    e1 = np.cross(DRIFT_AXIS, u0); n1 = np.linalg.norm(e1)
    if n1 > 1e-3:
        e1 = e1 / n1; e2 = np.cross(u0, e1); return e1, e2 / np.linalg.norm(e2)
    ref = np.array([0., 0., 1.]) if abs(u0[2]) <= 0.9 else np.array([1., 0., 0.])
    e1 = np.cross(u0, ref); e1 /= np.linalg.norm(e1); e2 = np.cross(u0, e1); return e1, e2 / np.linalg.norm(e2)

def run_residuals(tag, step=10):
    f = sorted(glob.glob(f'{R}/{tag}/history_iter*.pkl'), key=lambda x: int(x.split('iter')[-1].split('_')[0]))[-1]
    d = pickle.load(open(f, 'rb'))
    lt = np.array([f1(x) for x in d['lifetime_iter']])[:10001]
    ld = np.array([f1(x) for x in d['long_diff_iter']])[:10001]
    ab = np.array([f1(x) for x in d['Ab_iter']])[:10001]
    errs = {'lifetime': 100 * (lt[-1000:].mean() / f1(d['lifetime_target'][0]) - 1),
            'long_diff': 100 * (ld[-1000:].mean() / f1(d['long_diff_target'][0]) - 1),
            'Ab': 100 * (ab[-1000:].mean() / f1(d['Ab_target'][0]) - 1)}
    tracks = []                                   # per track: (drift distance array, delta-drift array [um], |delta| perp [um])
    for b, ctxs in d['chain_contexts'].items():
        states = d['chain_cache'][b]; tp = d['true_positions'][b]
        for ctx, st in zip(ctxs, states):
            key = ctx['track_id'] if ctx['track_id'] in tp else str(ctx['track_id'])
            if key not in tp: continue
            c = np.asarray(st['angles'], float); K = len(c) // 2
            u0 = dir_from_angles(ctx['theta0_i'], ctx['phi0_i']); e1, e2 = frame(u0)
            al = np.asarray(ctx['alpha_mid'], float); tr = np.asarray(tp[key], float)
            m = min(len(al), len(tr)); al, tr = al[:m:step], tr[:m:step]
            nom = np.asarray(ctx['x0_fixed'], float)[None, :] + al[:, None] * float(ctx['total_len']) * u0[None, :]
            B = np.sin(np.outer(al, np.pi * np.arange(1, K + 1)))
            fit = nom + (B @ c[:K])[:, None] * e1[None, :] + (B @ c[K:])[:, None] * e2[None, :]
            dl = fit - tr
            dperp = dl - (dl @ u0)[:, None] * u0[None, :]
            ddrift = -np.sign(tr[:, 2]) * dperp[:, 2] * 1e4            # um, + = farther from anode
            D = Z_ANODE - np.abs(tr[:, 2])
            ds = np.diff(al) * float(ctx['total_len'])
            slope_err = np.diff(ddrift) / np.maximum(ds, 1e-6) if len(ds) else np.zeros(0)   # um per cm of track
            tracks.append((D, ddrift, np.linalg.norm(dperp, axis=1) * 1e4, abs(u0[2]), slope_err))
    return tracks, errs

def summarise(tracks):
    nb = len(EDGES) - 1
    per = [[] for _ in range(nb)]; per_abs = [[] for _ in range(nb)]
    for D, dd, dp, _, _ in tracks:
        idx = np.digitize(D, EDGES) - 1
        for k in np.unique(idx):
            if 0 <= k < nb:
                sel = idx == k; per[k].append(dd[sel].mean()); per_abs[k].append(np.sqrt(np.mean(dd[sel] ** 2)))
    mean = np.array([np.mean(v) if len(v) > 2 else np.nan for v in per])
    se = np.array([np.std(v, ddof=1) / np.sqrt(len(v)) if len(v) > 2 else np.nan for v in per])
    rms = np.array([np.mean(v) if len(v) > 2 else np.nan for v in per_abs])
    # linear trend of per-track mean delta vs per-track mean drift distance, track-bootstrap error
    Dm = np.array([t[0].mean() for t in tracks]); dm = np.array([t[1].mean() for t in tracks])
    slope = np.polyfit(Dm, dm, 1)[0]
    rng = np.random.default_rng(0); bs = []
    for _ in range(400):
        i = rng.integers(0, len(Dm), len(Dm)); bs.append(np.polyfit(Dm[i], dm[i], 1)[0])
    # correlation of the delta-drift with drift distance WITHIN tracks (removes per-track offsets)
    within = []
    for D, dd, _, _, _ in tracks:
        if D.std() > 1.0: within.append(np.polyfit(D, dd, 1)[0])
    within = np.array(within)
    far = [np.sqrt(np.mean(t[1][t[0] > 20] ** 2)) for t in tracks if (t[0] > 20).sum() > 3]
    near = [np.sqrt(np.mean(t[1][t[0] < 10] ** 2)) for t in tracks if (t[0] < 10).sum() > 3]
    sl = np.concatenate([t[4] for t in tracks if len(t[4])])
    return dict(mean=mean.tolist(), se=se.tolist(), rms=rms.tolist(), n_tracks=len(tracks),
                rms_far=float(np.mean(far)), rms_near=float(np.mean(near)), slope_err_rms=float(np.sqrt(np.mean(sl ** 2))),
                slope_um_per_cm=float(slope), slope_se=float(np.std(bs)),
                within_slope=float(within.mean()), within_se=float(within.std(ddof=1) / np.sqrt(len(within))),
                overall_mean=float(np.mean([t[1].mean() for t in tracks])))

if __name__ == '__main__':
    ARMS = {'DSREP': [3, 7, 8, 1, 2, 4], 'DXTRU': [3, 7, 8], 'BOTHTRU': [3, 7, 8], 'DRIFTFRZ': [3, 7, 8],
            'GEOFRZ': [3, 7, 8], 'DRIFTTRU': [3, 7, 8, 1, 2, 4], 'DRIFTTRUK6': [1, 2, 4]}
    TAG = lambda a, s: f'sci_fullGEOFRZ{s}' if a == 'GEOFRZ' else f'sci_full_{a}{s}'
    out = {}
    for a, ss in ARMS.items():
        allt, per_sample = [], {}
        for s in ss:
            t, e = run_residuals(TAG(a, s)); allt += t
            sm = summarise(t); per_sample[s] = dict(errs=e, overall_mean=sm['overall_mean'], within_slope=sm['within_slope'], rms_far=sm['rms_far'], rms_near=sm['rms_near'], slope_err_rms=sm['slope_err_rms'],
                                                    within_se=sm['within_se'], slope=sm['slope_um_per_cm'])
            print(f'{a:10s} s{s}: tracks {len(t):4d}  lifetime {e["lifetime"]:+6.2f}  long_diff {e["long_diff"]:+6.1f}  '
                  f'mean ddrift {sm["overall_mean"]:+7.1f} um  within-track slope {sm["within_slope"]:+6.2f} +- {sm["within_se"]:.2f} um/cm  rms near/far {sm["rms_near"]:5.0f}/{sm["rms_far"]:5.0f} um  slope-err rms {sm["slope_err_rms"]:6.0f} um/cm', flush=True)
        out[a] = dict(pooled=summarise(allt), per_sample=per_sample)
        p = out[a]['pooled']
        print(f'  => {a:10s} POOLED n_tracks {p["n_tracks"]}: mean {p["overall_mean"]:+.1f} um; across-track slope {p["slope_um_per_cm"]:+.2f} +- {p["slope_se"]:.2f} um/cm; '
              f'within-track slope {p["within_slope"]:+.2f} +- {p["within_se"]:.2f} um/cm', flush=True)
    json.dump(dict(edges=EDGES.tolist(), arms=out), open(f'{OUT}/drift_residuals.json', 'w'), indent=1)
