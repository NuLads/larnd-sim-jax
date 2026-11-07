"""Item 3 (4 Nov, descriptive): how well can a track's endpoints be estimated from its own hits? Production stands in for
endpoint reconstruction with an independent 1 mm Gaussian smear per coordinate (LARND_ENDPOINT_SMEAR_MM=1.0). Here, on
the TARGET hits of true-end production fits (single-track events only, so every hit belongs to the track): a charge-
weighted PCA line through the hit positions (pixel_x, pixel_y, pixel_z), endpoints = the line at the extreme hit
projections. Error = estimate - true endpoint (true_positions first / last point), split into along-track and
transverse components and per coordinate; the mean along-track offset (inward bias, calibratable on simulation) is also
removed in a second version. usage (container): python3 endpoint_res.py KN10N1 KN10N2 ..."""
import sys, glob, pickle, numpy as np
ROOT = '/sdf/group/neutrino/pgranger/larnd-sim-jax'
import os
LOCAL = float(os.environ.get('EP_LOCAL_CM', '0') or 0)
rows = []
for run in sys.argv[1:]:
    ck = pickle.load(open(glob.glob(f'/sdf/scratch/neutrino/pgranger/fit_result_staging/sci_full_{run}C75H/history_iter20000_*.pkl')[0], 'rb'))
    tdir = glob.glob(f'{ROOT}/target_sci_full_{run}C75_sci_full_*')[0]
    for b, ctxs in ck['chain_contexts'].items():
        z = np.load(f'{tdir}/batch{b}_target.npz')
        ev_of = [int(np.ravel(c['track_id'])[0]) for c in ctxs]
        for c in ctxs:
            e = int(np.ravel(c['track_id'])[0])
            if ev_of.count(e) != 1:
                continue                                    # multi-track event: hits cannot be assigned
            m = z['event'] == e
            if m.sum() < 20:
                continue
            X = np.stack([z['pixel_x'][m], z['pixel_y'][m], z['pixel_z'][m]], 1).astype(float)
            w = np.clip(z['adcs'][m].astype(float), 0, None) * z['hit_prob'][m].astype(float)
            if w.sum() <= 0:
                continue
            mu = (w[:, None] * X).sum(0) / w.sum()
            C = ((X - mu) * w[:, None]).T @ (X - mu) / w.sum()
            u = np.linalg.eigh(C)[1][:, -1]
            tp = np.asarray(ck['true_positions'][b][str(tuple(float(v) for v in np.ravel(c['track_id'])))], float)
            a, bb = tp[0], tp[-1]
            if u @ (bb - a) < 0:
                u = -u
            s = (X - mu) @ u
            est = (mu + s.min() * u, mu + s.max() * u)
            if LOCAL:                                       # refit a line to the hits within LOCAL cm of each end
                est = []
                for sel, ext in ((s < s.min() + LOCAL, np.argmin), (s > s.max() - LOCAL, np.argmax)):
                    if sel.sum() < 6:
                        est.append(mu + (s.min() if ext is np.argmin else s.max()) * u); continue
                    Xl, wl = X[sel], w[sel]; ml = (wl[:, None] * Xl).sum(0) / wl.sum()
                    ul = np.linalg.eigh(((Xl - ml) * wl[:, None]).T @ (Xl - ml) / wl.sum())[1][:, -1]
                    ul = ul if ul @ u > 0 else -ul; sl = (Xl - ml) @ ul
                    est.append(ml + (sl.min() if ext is np.argmin else sl.max()) * ul)
            L = np.linalg.norm(bb - a); ut = (bb - a) / L
            for k, (ee, tt) in enumerate(zip(est, (a, bb))):
                err = ee - tt
                along = err @ ut * (1 if k == 1 else -1)    # > 0 = estimate beyond the true end (outward)
                rows.append([run, L, k] + list(err) + [along, np.linalg.norm(err - (err @ ut) * ut)] + list(np.abs(ut)))
R = np.array([r[1:] for r in rows], float)
L, err, along, trans, ua = R[:, 0], R[:, 2:5], R[:, 5], R[:, 6], R[:, 7:10]
print(f'[estimator: ' + (f'local line fit to the last {LOCAL:g} cm at each end' if LOCAL else 'one PCA line through all hits') + ']')
print(f'{len(R) // 2} single-track tracks ({len(R)} endpoints) from {", ".join(sys.argv[1:])}; track length median {np.median(L):.1f} cm')
mm = lambda x: 10 * x
print(f'  along-track offset (outward > 0): mean {mm(along.mean()):+.2f} mm, median {mm(np.median(along)):+.2f}, SD {mm(along.std()):.2f}, '
      f'p90 |.| {mm(np.percentile(np.abs(along), 90)):.2f}')
print(f'  transverse distance: median {mm(np.median(trans)):.2f} mm, RMS {mm(np.sqrt((trans ** 2).mean())):.2f}, p90 {mm(np.percentile(trans, 90)):.2f}')
print('  per coordinate RMS (x, y, z): ' + ' '.join(f'{mm(v):.2f}' for v in np.sqrt((err ** 2).mean(0))) + ' mm  (production smear: 1.00 per coordinate)')
print(f'  along-track spread after removing its mean offset: SD {mm(along.std()):.2f} mm; transverse RMS unchanged')
print('  per coordinate robust SD (1.4826 MAD): ' + ' '.join(f'{mm(1.4826 * np.median(np.abs(v - np.median(v)))):.2f}' for v in err.T) + ' mm')
for lo, hi in ((0, 30), (30, 60), (60, 200)):
    k = (L >= lo) & (L < hi)
    if k.sum():
        print(f'  tracks {lo}-{hi} cm (n = {k.sum() // 1}): along mean {mm(along[k].mean()):+.2f} SD {mm(along[k].std()):.2f} mm; transverse RMS {mm(np.sqrt((trans[k] ** 2).mean())):.2f} mm')
