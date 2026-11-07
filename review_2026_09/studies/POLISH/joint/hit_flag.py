"""CPU, truth-free per-track misfit flag: how far the track's own target hits sit from its fitted path. For every target
hit of the track's event, the nearest fitted segment of that event's tracks owns it (3-D distance, z from hit timing);
hits within 1 cm of their owner's path count. Per track: the charge-weighted mean transverse offset of its hits in the
pixel plane (signed along the in-plane normal to the track) in 4 equal slices along the track, and the overall
charge-weighted rms in-plane distance. A track displaced by ~1 mm shows a coherent offset that pixel quantisation
(pitch 4.43 mm, rms 1.3 mm per hit) averages away over many hits. The flag is compared with the truth residual of
per_track_res.py (same run / iteration).  usage: hit_flag.py RUN SAMPLE ITER"""
import sys, glob, json, numpy as np
sys.path.insert(0, '/sdf/group/neutrino/pgranger/larnd-sim-jax/review_2026_09/scripts/eventdisplay')
from rebuild import FitRebuild
run, s, it = sys.argv[1], sys.argv[2], sys.argv[3]
D = '/sdf/scratch/neutrino/pgranger/review_2026_09/POLISH/joint'
R = FitRebuild(sorted(glob.glob(f'/sdf/group/neutrino/pgranger/larnd-sim-jax/fit_result/sci_full_{run}{s}/history_iter{it}_*.pkl'))[0])
ck = R.ck; cols = [R.col['x'], R.col['y'], R.col['z']]; ev_c = R.sfields.index('eventID')
ref = {(o['batch'], round(o['L'], 3)): o for o in json.load(open(f'{D}/per_track_{run}{s}_{it}.json'))}
out = []
for i in range(len(ck['true_positions'])):
    bt = R.batch(i, check=False); f = np.asarray(bt['fitted']); tg = R.target(i)
    hx = np.stack([tg['pixel_x'], tg['pixel_y'], tg['pixel_z']], 1).astype(float); hq = np.asarray(tg['adcs'], float); hev = np.asarray(tg['event'])
    # target 'event' is the batch-local event index; map it to the eventIDs present in the batch (sorted, -1 = padding)
    evs = [e for e in np.unique(f[:, ev_c]) if e >= 0]
    trk = []
    for c in bt['ctxs']:
        idx = np.asarray(c.idxs)[::10]; P = f[idx][:, cols]; trk.append((c, P, float(f[idx[0], ev_c])))
    for le, e in enumerate(evs):
        H = hx[hev == le]; Q = hq[hev == le]
        T = [(c, P) for c, P, ee in trk if ee == e]
        if len(H) == 0 or not T: continue
        dist = np.stack([np.min(np.linalg.norm(H[:, None, :] - P[None, :, :], axis=2), axis=1) for _, P in T], 1)
        own = np.argmin(dist, 1); dmin = dist[np.arange(len(H)), own]
        for k, (c, P) in enumerate(T):
            m = (own == k) & (dmin < 1.0)
            if m.sum() < 8: continue
            h, q = H[m], Q[m]
            u = P[-1] - P[0]; L = np.linalg.norm(u); u = u / max(L, 1e-9)
            nxy = np.array([-u[1], u[0], 0.]); nxy /= max(np.linalg.norm(nxy), 1e-9)       # in-plane normal to the track
            j = np.argmin(np.linalg.norm(h[:, None, :] - P[None, :, :], axis=2), axis=1)     # nearest fitted segment
            off = (h - P[j]) @ nxy; al = ((h - P[0]) @ u) / max(L, 1e-9)
            sl = []
            for a0 in (0, .25, .5, .75):
                mm = (al >= a0) & (al < a0 + .25)
                if mm.sum() >= 4: sl.append(float(np.sum(q[mm] * off[mm]) / np.sum(q[mm])))
            dxy = np.linalg.norm((h - P[j])[:, :2], axis=1)
            key = (i, round(float(c.total_len), 3)); r = ref.get(key)
            out.append(dict(batch=i, L=float(c.total_len), nhit=int(m.sum()), max_slice_off_um=1e4 * float(np.max(np.abs(sl))) if sl else float('nan'),
                            rms_xy_um=1e4 * float(np.sqrt(np.sum(q * dxy ** 2) / np.sum(q))), r_mean=r['r_mean'] if r else float('nan'),
                            frozen=r['frozen'] if r else None))
json.dump(out, open(f'{D}/hit_flag_{run}{s}_{it}.json', 'w'), indent=1)
a = {k: np.array([o[k] for o in out], float) for k in ('max_slice_off_um', 'rms_xy_um', 'r_mean', 'nhit')}
ok = np.isfinite(a['r_mean']) & np.isfinite(a['max_slice_off_um']); bad = a['r_mean'] > 200
print(f'{run}{s}@{it}: {len(out)} tracks with >= 8 owned hits, {ok.sum()} matched to the truth residual; outliers (|r| > 200 um) {np.sum(bad & ok)}')
from scipy.stats import spearmanr
for k in ('max_slice_off_um', 'rms_xy_um'):
    x = a[k][ok]; y = a['r_mean'][ok]; b = bad[ok]
    print(f'  {k}: Spearman with |r| {spearmanr(x, y)[0]:+.2f}; median outliers {np.median(x[b]):.0f} vs rest {np.median(x[~b]):.0f}')
    for q in (0.8, 0.9, 0.95):
        t = np.quantile(x, q); f = x > t
        print(f'    flag top {100 * (1 - q):.0f}% (> {t:.0f}): catches {np.sum(f & b)}/{b.sum()} outliers, {np.sum(f & ~b)} false; '
              f'mean |r| kept {np.mean(y[~f]):.0f} um vs all {np.mean(y):.0f}')
