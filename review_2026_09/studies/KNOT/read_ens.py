"""Production ensemble readout (rule: REVIEW.md 24 Oct, written before any result). KN10N<s>FR @ 14000 (1e4 fit + freeze-
release continuation, live tree) for s = 1..16, vs the old KN10S<s> @ 1e4 (18 Oct ensemble), paired by sample.
Per parameter: error % = mean of the last 1000 iterations / target - 1 (compare_arms.read convention), also the raw last
value; ensemble mean +- SD/sqrt(n) and SD. Positions: mean pos_residual over the last 1000 iterations (um).
The two ensembles straddle the 24 Oct reproducibility boundaries: the comparison is descriptive (ensemble level only).
usage (in the container): python3 read_ens.py"""
import glob, pickle, numpy as np
FR = '/sdf/group/neutrino/pgranger/larnd-sim-jax/fit_result'
P = ['Ab', 'eField', 'tran_diff', 'long_diff', 'lifetime']
f1 = lambda x: np.nan if x is None else float(np.mean(x))


def read(run, it):
    fs = sorted(glob.glob(f'{FR}/sci_full_{run}/history_iter{it}_*.pkl'))
    if not fs: raise SystemExit(f'{run}: no checkpoint at {it} -> ensemble not complete, no readout')
    d = pickle.load(open(fs[0], 'rb')); n = int(d['total_iter'])
    r = {}
    for p in P:
        a = np.array([f1(x) for x in d[f'{p}_iter']]); a = a[:len(a) - (n - it)] if n > it else a
        t = f1(d[f'{p}_target'][0]); r[p] = 100 * (a[-1000:].mean() / t - 1); r[p + '_last'] = 100 * (a[-1] / t - 1)
    pr = np.array([f1(x) for x in d['pos_residual_iter']]); pr = pr[:len(pr) - (n - it)] if n > it else pr
    r['pos'] = 1e4 * float(np.nanmean(pr[-1000:])); r['n'] = n
    return r


S = list(range(1, 17))
N = {s: read(f'KN10N{s}FR', 14000) for s in S}
O = {s: read(f'KN10S{s}', 10000) for s in S}
Q = P + ['pos']
print(f"{'s':>2s} " + ' '.join(f'{q:>17s}' for q in Q) + '    (new FR@14000 / old S@1e4)')
for s in S:
    print(f'{s:2d} ' + ' '.join(f'{N[s][q]:+8.2f}/{O[s][q]:+8.2f}' for q in Q))
ms = lambda v: (v.mean(), v.std(ddof=1) / np.sqrt(len(v)), v.std(ddof=1))
print('\nensemble (n = 16): mean +- SE (SD)')
for q in Q + [p + '_last' for p in P]:
    vn, vo = np.array([N[s][q] for s in S]), np.array([O[s][q] for s in S])
    (a, b, c), (d, e, f) = ms(vn), ms(vo); (g, h, _) = ms(vn - vo)
    print(f'{q:15s} new {a:+7.2f} +- {b:5.2f} (SD {c:5.2f})   old {d:+7.2f} +- {e:5.2f} (SD {f:5.2f})   new - old {g:+6.2f} +- {h:5.2f}')
r = np.array([N[s]['pos'] / O[s]['pos'] for s in S])
print(f'\npositions new / old per sample: ' + ' '.join(f'{x:.2f}' for x in r) + f'  -> mean {r.mean():.3f}, range {r.min():.2f}-{r.max():.2f}')
print('completeness: all 16 new runs at', sorted({N[s]['n'] for s in S}), '; old at', sorted({O[s]['n'] for s in S}))
