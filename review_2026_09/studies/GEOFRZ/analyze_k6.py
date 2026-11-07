"""DRIFTTRUK6 (drift at truth with 6 spline modes) vs DRIFTTRU (3 modes) vs DSREP, paired on samples 1/2/4."""
import glob, pickle, json, numpy as np
R = '/sdf/group/neutrino/pgranger/larnd-sim-jax/fit_result'
P = ['Ab', 'eField', 'tran_diff', 'long_diff', 'lifetime']
f1 = lambda x: float(np.ravel(np.asarray(x))[0])
def read(tag):
    f = sorted(glob.glob(f'{R}/{tag}/history_iter*.pkl'), key=lambda x: int(x.split('iter')[-1].split('_')[0]))[-1]
    d = pickle.load(open(f, 'rb')); assert d['total_iter'] >= 10000, tag
    r = {}
    for p in P:
        a = np.array([f1(x) for x in d[f'{p}_iter']])[:10001]; t = f1(d[f'{p}_target'][0])
        r[p] = 100 * (a[-1000:].mean() / t - 1); r[p + '_ptp'] = 100 * np.ptp(a[-2000:]) / t
    pr = np.array([f1(x) for x in d['pos_residual_iter']])[:10000]
    r['pos_um'] = float(np.nanmedian(pr[-2000:])) * (1e4 if np.nanmedian(pr) < 1 else 1)
    r['loss'] = float(np.mean([f1(x) for x in d['losses_iter']][9000:10000]))
    return r
S = [1, 2, 4]
res = {(a, s): read(f'sci_full_{a}{s}') for a in ('DSREP', 'DRIFTTRU', 'DRIFTTRUK6') for s in S}
print(f"{'s':>2s} {'arm':11s} " + ' '.join(f'{p:>10s}' for p in P) + '  pos_um  max_ptp')
for s in S:
    for a in ('DSREP', 'DRIFTTRU', 'DRIFTTRUK6'):
        r = res[(a, s)]; print(f"{s:2d} {a:11s} " + ' '.join(f'{r[p]:+10.2f}' for p in P) + f"  {r['pos_um']:6.0f}  {max(r[p+'_ptp'] for p in P):.2f}")
print('\npaired (mean +- SE, n=3):')
for a, b in [('DRIFTTRUK6', 'DRIFTTRU'), ('DRIFTTRUK6', 'DSREP')]:
    for p in ['lifetime', 'long_diff', 'Ab', 'tran_diff', 'pos_um', 'loss']:
        v = np.array([res[(a, s)][p] - res[(b, s)][p] for s in S])
        print(f"  {a}-{b:9s} {p:10s} {v.mean():+9.2f} +- {v.std(ddof=1)/np.sqrt(3):6.2f}  [{', '.join(f'{x:+.2f}' for x in v)}]")
json.dump({f'{a}{s}': r for (a, s), r in res.items()}, open('result_k6.json', 'w'), indent=1)
