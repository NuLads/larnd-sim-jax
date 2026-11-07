"""CPU: per-track position residual (fitted vs true) at a KN10 end point, with the track properties available from
reconstruction (length, |cos| to the drift axis z, mean |z|, K, frozen e2 = called straight by the hits), to see
whether a 'low resolution' class of tracks carries the position error.  usage: per_track_res.py RUN SAMPLE ITER"""
import sys, glob, json, numpy as np
sys.path.insert(0, '/sdf/group/neutrino/pgranger/larnd-sim-jax/review_2026_09/scripts/eventdisplay')
from rebuild import FitRebuild
run, s, it = sys.argv[1], sys.argv[2], sys.argv[3]
ck_path = sorted(glob.glob(f'/sdf/group/neutrino/pgranger/larnd-sim-jax/fit_result/sci_full_{run}{s}/history_iter{it}_*.pkl'))[0]
R = FitRebuild(ck_path); fp, ck = R.fp, R.ck; cols = [R.col['x'], R.col['y'], R.col['z']]; out = []
for i in range(len(ck['true_positions'])):
    bt = R.batch(i, check=False); tp = ck['true_positions'][i]
    for c in bt['ctxs']:
        tr = tp.get(tuple(map(float, c.track_id)), tp.get(str(tuple(map(float, c.track_id)))))
        if tr is None: continue
        tr = np.asarray(tr); idx = np.asarray(c.idxs)[:len(tr)]; pf = bt['fitted'][idx][:, cols]; d = pf - tr[:len(idx)]
        K = fp._spline_K(c.total_len, R.knot_cm); fc = np.asarray(bt['coeffs'][tuple(c.track_id)])
        u0 = np.asarray(fp._dir_from_angles(c.theta0_i, c.phi0_i))
        out.append(dict(batch=i, L=float(c.total_len), K=int(K), cosz=float(abs(u0[2])), zabs=float(np.mean(np.abs(tr[:, 2]))),
                        frozen=bool(np.all(fc[K:] == 0)), r_mean=1e4 * float(np.linalg.norm(d, axis=1).mean()),
                        r_z=1e4 * float(np.sqrt(np.mean(d[:, 2] ** 2))), r_xy=1e4 * float(np.sqrt(np.mean(d[:, 0] ** 2 + d[:, 1] ** 2))),
                        r_max=1e4 * float(np.linalg.norm(d, axis=1).max()), n=len(idx)))
json.dump(out, open(f'/sdf/scratch/neutrino/pgranger/review_2026_09/POLISH/joint/per_track_{run}{s}_{it}.json', 'w'), indent=1)
a = {k: np.array([o[k] for o in out]) for k in out[0]}; w = a['n'] / a['n'].sum()
print(f'{run}{s}@{it}: {len(out)} tracks; segment-weighted mean |r| {np.sum(w * a["r_mean"]):.1f} um, track median {np.median(a["r_mean"]):.1f}')
def cls(name, m):
    if m.sum() == 0: return
    print(f'  {name:28s} n {m.sum():4d} ({100 * np.sum(w[m]):4.1f}% of segments)  |r| mean {np.average(a["r_mean"][m], weights=a["n"][m]):6.1f}  '
          f'median {np.median(a["r_mean"][m]):6.1f}  z-rms {np.average(a["r_z"][m], weights=a["n"][m]):6.1f}  xy-rms {np.average(a["r_xy"][m], weights=a["n"][m]):6.1f}')
cls('all', a['L'] > 0); cls('frozen (called straight)', a['frozen']); cls('not frozen', ~a['frozen'])
for lo, hi in [(0, 20), (20, 50), (50, 100), (100, 1e9)]: cls(f'L {lo}-{hi} cm', (a['L'] >= lo) & (a['L'] < hi))
for lo, hi in [(0, .3), (.3, .6), (.6, .8), (.8, .9), (.9, 1.01)]: cls(f'|cos z| {lo}-{hi}', (a['cosz'] >= lo) & (a['cosz'] < hi))
for lo, hi in [(0, 10), (10, 20), (20, 40)]: cls(f'mean |z| {lo}-{hi} cm', (a['zabs'] >= lo) & (a['zabs'] < hi))
q = np.quantile(a['r_mean'], [0.5, 0.9, 0.99]); top = a['r_mean'] >= q[1]
print(f'  track |r| quantiles 50/90/99: {q.round(1)}; worst 10% of tracks hold {100 * np.sum(w[top]):.1f}% of segments and '
      f'{100 * np.sum(w[top] * a["r_mean"][top]) / np.sum(w * a["r_mean"]):.1f}% of the segment-weighted |r|')
print(f'  worst 10%: median L {np.median(a["L"][top]):.0f} cm (all {np.median(a["L"]):.0f}), median |cos z| {np.median(a["cosz"][top]):.2f} '
      f'(all {np.median(a["cosz"]):.2f}), frozen {a["frozen"][top].mean():.2f} (all {a["frozen"].mean():.2f})')
