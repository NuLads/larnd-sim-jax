"""Readout of the joint calibration + dE/dx Newton step on the 12 same-data runs (rule: REVIEW.md 21 Oct, stated before
any result), side by side with the 5-parameter polish of the same runs (POLISH/replica)."""
import json, numpy as np
D = '/sdf/scratch/neutrino/pgranger/review_2026_09/POLISH'
P = ['Ab', 'eField', 'tran_diff', 'long_diff', 'lifetime']
SD16 = dict(Ab=0.133, eField=0.010, tran_diff=1.051, long_diff=5.171, lifetime=3.999)   # KN10S 16-sample SD (read_valid.py)
runs = lambda s: [f'KN10V{s}', f'KN10V{s}b', f'KN10V{s}c', f'KN10S{s}']
J = {s: [json.load(open(f'{D}/joint/joint_{r}.json'))['summary'] for r in runs(s)] for s in (1, 2, 3)}
C = {s: [json.load(open(f'{D}/replica/polish_{r}.json'))['summary'] for r in runs(s)] for s in (1, 2, 3)}
nv = 0
for s in J:
    for r, x in zip(runs(s), J[s]):
        ok = x['accepted']; nv += ok
        print(f'{r:9s} {x["n_ok"]}/{x["n_batches"]}  valid {ok}  best kappa {x["best_kappa"]}  dJ {x["dJ"]:+.2f} (model {x["dJ_model"]:+.2f}; '
              f'd fixed {x["dJ_cond"]:+.2f})  neg-curvature batches {x["n_batches_neg"]}  joint step % ' + ' '.join(f'{k} {v:+.3f}' for k, v in zip(P, x['du_pct']))
              + f' | asym {x["max_band_asym"]:.1e} sym {x["max_sym_ud"]:.1e}')
        print('           ladder dJ: ' + '  '.join(f'{e["kappa"]}:{e["dJ"]:+.1f}' for e in x['ladder']))
print(f'step VALID {nv}/12')
pool = lambda T, key, k: float(np.sqrt(np.mean([np.var([x[key][k] for x in T[s]], ddof=1) for s in T])))
mstep = lambda T, key, i: float(np.mean([x[key][i] for s in T for x in T[s]]))
r = {}
for i, k in enumerate(P):
    b, aj, ac = pool(J, 'err0', k), pool(J, 'err1', k), pool(C, 'err1', k); r[k] = aj / b
    print(f'  {k:9s} same-data SD before {b:.3f}  joint {aj:.3f} (r {aj / b:.2f})  5-param {ac:.3f} (r {ac / b:.2f})  | '
          f'mean step joint {mstep(J, "du_pct", i):+.3f}  5-param {mstep(C, "step_pct", i):+.3f}  | '
          f'mean error before {np.mean([x["err0"][k] for s in J for x in J[s]]):+.2f}  joint {np.mean([x["err1"][k] for s in J for x in J[s]]):+.2f}  '
          f'5-param {np.mean([x["err1"][k] for s in C for x in C[s]]):+.2f}  | joint SD / 16-sample SD {aj / SD16[k]:.2f}')
w = max(r['lifetime'], r['long_diff'])
print('verdict (scatter):', 'COLLAPSES' if w <= 0.5 else 'HELPS' if w <= 0.8 else 'NO GAIN',
      '| BETTER than the 5-parameter polish' if r['lifetime'] < 0.58 and r['long_diff'] < 0.65 else '| not better than the 5-parameter polish')
lt = abs(mstep(J, 'du_pct', P.index('lifetime')))
print(f'mean lifetime shift {lt:.2f}% ->', 'ATTRIBUTED to holding dE/dx fixed' if lt < 1.33 else 'NOT attributed' if lt >= 2.0 else 'unresolved')
print('marginal / conditional sigma % (median over 12): ' + '  '.join(
    f'{k} {np.median([x["sigma_marg_pct"][i] for s in J for x in J[s]]):.3f}/{np.median([x["sigma_cond_pct"][i] for s in J for x in J[s]]):.3f}'
    for i, k in enumerate(P)))
for s in J:
    print(f's{s} lifetime before > joint (5-param): ' + '  '.join(f'{x["err0"]["lifetime"]:+.2f} > {x["err1"]["lifetime"]:+.2f} ({c["err1"]["lifetime"]:+.2f})'
                                                             for x, c in zip(J[s], C[s])))
