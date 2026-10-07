"""Generic paired comparison of two production-fit arms on the same samples, read at iteration 10000.
usage: python compare_arms.py ARM_A ARM_B s1,s2,s3     (ARM names as in fit_result/sci_full_<ARM><s>)
prints per-sample errors for both arms and the paired A-B difference (mean +- SE, per sample)."""
import sys, glob, pickle, numpy as np
R = '/sdf/group/neutrino/pgranger/larnd-sim-jax/fit_result'
P = ['Ab', 'eField', 'tran_diff', 'long_diff', 'lifetime']
f1 = lambda x: float(np.ravel(np.asarray(x))[0])

def tagdir(arm, s):
    for t in (f'{R}/sci_full_{arm}{s}', f'{R}/sci_full{arm}{s}'):
        if glob.glob(f'{t}/history_iter*.pkl'): return t
    raise FileNotFoundError(f'{arm}{s}')

def read(arm, s):
    f = sorted(glob.glob(f'{tagdir(arm, s)}/history_iter*.pkl'), key=lambda x: int(x.split('iter')[-1].split('_')[0]))[-1]
    d = pickle.load(open(f, 'rb'))
    if d['total_iter'] < 10000: raise RuntimeError(f'{arm}{s} only at {d["total_iter"]}')
    r = {}
    for p in P:
        a = np.array([f1(x) for x in d[f'{p}_iter']])[:10001]; t = f1(d[f'{p}_target'][0])
        r[p] = 100 * (a[-1000:].mean() / t - 1); r[p + '_ptp'] = 100 * np.ptp(a[-2000:]) / t
    pr = np.array([f1(x) for x in d['pos_residual_iter']])[:10000]
    r['pos_um'] = float(np.nanmedian(pr[-2000:])) * (1e4 if np.nanmedian(pr) < 1 else 1)
    r['loss'] = float(np.mean([f1(x) for x in d['losses_iter']][9000:10000]))
    return r

if __name__ == '__main__':
    A, B, S = sys.argv[1], sys.argv[2], [int(x) for x in sys.argv[3].split(',')]
    res = {(a, s): read(a, s) for a in (A, B) for s in S}
    print(f"{'s':>2s} {'arm':12s} " + ' '.join(f'{p:>10s}' for p in P) + '  pos_um  max_ptp')
    for s in S:
        for a in (A, B):
            r = res[(a, s)]
            print(f"{s:2d} {a:12s} " + ' '.join(f'{r[p]:+10.3f}' for p in P) + f"  {r['pos_um']:6.0f}  {max(r[p + '_ptp'] for p in P):.2f}")
    print(f'\npaired {A} - {B} (mean +- SE, n={len(S)}):')
    for p in P + ['pos_um', 'loss']:
        v = np.array([res[(A, s)][p] - res[(B, s)][p] for s in S])
        print(f"  {p:10s} {v.mean():+9.3f} +- {v.std(ddof=1) / np.sqrt(len(v)):7.3f}  [{', '.join(f'{x:+.2f}' for x in v)}]")
    for a in (A, B):
        for p in ('lifetime', 'long_diff', 'Ab', 'tran_diff'):
            v = np.array([res[(a, s)][p] for s in S])
            print(f"  level {a:12s} {p:10s} {v.mean():+7.2f} +- {v.std(ddof=1) / np.sqrt(len(v)):.2f}")
