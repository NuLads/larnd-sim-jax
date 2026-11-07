"""Paired readout of two arms, same protocol as CONT/ensemble_readout.json and KNOT/ensemble_readout_kn10.json:
parameter error (%) = mean of the last 1e3 iterations / truth - 1; position = batch median of the last 1e3
pos_residual_iter (um); loss = mean of the last 1e3 iterations.
usage: python readout_block.py ARM_A ITER_A ARM_B ITER_B 3,7,8 [out.json]   (prints A, B and A - B per sample)"""
import sys, glob, json, pickle, numpy as np
FR = '/sdf/scratch/neutrino/pgranger/fit_result_staging'
P = ['Ab', 'eField', 'tran_diff', 'long_diff', 'lifetime']


def read(arm, it, s):
    f = sorted(glob.glob(f'{FR}/sci_full_{arm}{s}/history_iter{it}_*.pkl'))[0]
    d = pickle.load(open(f, 'rb'))
    v = {p: 100 * (np.ravel(d[p + '_iter'])[-1000:].mean() / np.ravel(d[p + '_target'])[0] - 1) for p in P}
    v['pos'] = 1e4 * float(np.median(np.asarray(d['pos_residual_iter'], float).ravel()[-1000:]))
    v['loss'] = float(np.mean(np.ravel(d['losses_iter'])[-1000:]))
    v['n_iter'] = len(np.ravel(d['losses_iter']))
    return v


A, ia, B, ib, ss = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], [int(x) for x in sys.argv[5].split(',')]
R = {s: (read(A, ia, s), read(B, ib, s)) for s in ss}
for s in ss:
    a, b = R[s]
    print(f's{s}: n_iter {a["n_iter"]}/{b["n_iter"]} | ' + '  '.join(
        f'{p} {a[p]:+.2f} / {b[p]:+.2f} (d {a[p] - b[p]:+.2f})' for p in P + ['pos', 'loss']))
se = lambda x: x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else float('nan')
print(f'mean over {ss}: {A}@{ia} | {B}@{ib} | paired shift (A - B)')
for p in P + ['pos', 'loss']:
    a = np.array([R[s][0][p] for s in ss]); b = np.array([R[s][1][p] for s in ss]); d = a - b
    print(f'  {p:9s} {a.mean():+9.3f} +- {se(a):.3f} | {b.mean():+9.3f} +- {se(b):.3f} | {d.mean():+8.3f} +- {se(d):.3f}'
          f'  [{" ".join(f"{x:+.2f}" for x in d)}]  positive {int((d > 0).sum())}/{len(d)}')
if len(sys.argv) > 6:
    json.dump({str(s): {'A': R[s][0], 'B': R[s][1]} for s in ss}, open(sys.argv[6], 'w'), indent=1)
