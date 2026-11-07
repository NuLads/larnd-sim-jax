"""Readout of the Newton polish on the same-data runs (rule: REVIEW.md 21 Oct, stated before any result).
usage: python read_polish_replica.py [patched_only]"""
import sys, json, numpy as np
D = '/sdf/scratch/neutrino/pgranger/review_2026_09/POLISH/replica'
P = ['Ab', 'eField', 'tran_diff', 'long_diff', 'lifetime']
SD16 = dict(Ab=0.133, eField=0.010, tran_diff=1.051, long_diff=5.171, lifetime=3.999)   # KN10S 16-sample SD (read_valid.py)
runs = (lambda s: [f'KN10V{s}', f'KN10V{s}b', f'KN10V{s}c'] + ([] if 'patched_only' in sys.argv else [f'KN10S{s}']))
S = {s: [json.load(open(f'{D}/polish_{r}.json'))['summary'] for r in runs(s)] for s in (1, 2, 3)}
nv = 0
for s in S:
    for r, x in zip(runs(s), S[s]):
        ok = min(x['eig']) > 0 and x['dL_actual'] < 0; nv += ok
        print(f'{r:9s} {x["n_ok"]}/{x["n_batches"]} batches  valid {ok}  dL {x["dL_actual"]:+.2f} (pred {x["dL_pred"]:+.2f})  step % '
              + ' '.join(f'{k} {v:+.3f}' for k, v in zip(P, x['step_pct'])))
print(f'step VALID {nv}/{sum(len(v) for v in S.values())}')
print('mean step %: ' + '  '.join(f'{k} {np.mean([x["step_pct"][i] for s in S for x in S[s]]):+.3f}' for i, k in enumerate(P))
      + '   (KN10C polish: lifetime -1.70, long_diff -1.52)')
pool = lambda key, k: float(np.sqrt(np.mean([np.var([x[key][k] for x in S[s]], ddof=1) for s in S])))
r = {}
for k in P:
    b, a = pool('err0', k), pool('err1', k); r[k] = a / b
    print(f'  {k:9s} same-data SD before {b:.3f}  after {a:.3f}  r = {r[k]:.2f}   (after / 16-sample SD {a / SD16[k]:.2f})')
both = max(r['lifetime'], r['long_diff'])
print('verdict:', 'COLLAPSES' if both <= 0.5 else 'HELPS' if both <= 0.8 else 'NO GAIN')
for s in S:
    print(f's{s} errors before -> after: ' + '  '.join(f'{k} [' + ' '.join(f'{x["err0"][k]:+.2f}>{x["err1"][k]:+.2f}' for x in S[s]) + ']' for k in ('lifetime', 'long_diff')))
