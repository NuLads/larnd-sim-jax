"""K6 drift-from-timing lifetime bias vs true tau: 1200 / 1800 / 2406 / 3200 us, paired with production."""
import glob, pickle, json, csv, numpy as np
ST = '/sdf/scratch/neutrino/pgranger/fit_result_staging'
P = ['Ab', 'eField', 'tran_diff', 'long_diff', 'lifetime']
f1 = lambda x: float(np.ravel(np.asarray(x))[0])
def read(d0):
    f = sorted(glob.glob(f'{ST}/{d0}/history_iter*.pkl') + glob.glob(f'/sdf/group/neutrino/pgranger/larnd-sim-jax/fit_result/{d0}/history_iter*.pkl'),
               key=lambda x: int(x.split('iter')[-1].split('_')[0]))[-1]
    d = pickle.load(open(f, 'rb')); assert d['total_iter'] >= 10000, d0
    r = {}
    for p in P:
        a = np.array([f1(x) for x in d[f'{p}_iter']])[:10001]; r[p] = 100 * (a[-1000:].mean() / f1(d[f'{p}_target'][0]) - 1)
        r[p + '_ptp'] = 100 * np.ptp(a[-2000:]) / f1(d[f'{p}_target'][0])
    return r
cur = {r['tag']: r for r in csv.DictReader(open('/sdf/group/neutrino/pgranger/larnd-sim-jax/review_2026_09/curve_scan_per_run.csv'))}
out = {}
for T in (1800, 3200, 1200):
    rows = []
    for s in (1, 3, 7):
        a = read(f'sci_full_DRIFTK6T{T}D{s}'); c = cur.get(f'CURVE{T}D{s}')
        prod = float(c['lifetime_err_pct']) if c else np.nan
        rows.append((s, a, prod))
        print(f"tau={T} s{s}: K6 " + ' '.join(f"{p} {a[p]:+.2f}" for p in P) + f"  ptp {max(a[p+'_ptp'] for p in P):.2f} | production lifetime {prod:+.2f}")
    lt = np.array([r[1]['lifetime'] for r in rows]); pr = np.array([r[2] for r in rows])
    out[T] = dict(k6=lt.tolist(), prod=pr.tolist())
    print(f"  tau={T}: K6 lifetime {lt.mean():+.2f} +- {lt.std(ddof=1)/np.sqrt(3):.2f}  | production {np.nanmean(pr):+.2f}")
out[2406] = dict(k6=[-5.18, -5.10, -4.61], prod=[6.53, 5.18, 4.83])
print('\ncurve (true tau -> K6 lifetime error, mean +- SE):')
for T in sorted(out):
    v = np.array(out[T]['k6']); print(f"  {T:5d} us: {v.mean():+6.2f} +- {v.std(ddof=1)/np.sqrt(len(v)):.2f}   (production {np.nanmean(out[T]['prod']):+.2f})")
json.dump(out, open('/sdf/scratch/neutrino/pgranger/review_2026_09/CURVEK6/curve.json', 'w'), indent=1)
