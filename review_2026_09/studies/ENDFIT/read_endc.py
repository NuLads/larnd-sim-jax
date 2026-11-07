"""ENDFIT option (b) readout (rule: REVIEW.md 24 Oct, written before any result). Production recipe (1e4 fit + freeze-
release continuation, end point <NAME>FR @ 14000), samples 1/2/4, paired with KN10N<s>FR (true endpoints, held):
  EZT = true endpoints, FITTED, batch zero-mean drift constraint (control: the cost of the constrained freedom)
  EZS = endpoints smeared 1 mm per coordinate, FITTED, constraint (the test)
  EXS = endpoints smeared 1 mm, HELD (the cost of the smear in the current recipe)
Errors % = mean of the last 1000 iterations / target - 1 (read_ens.read). Also: the constraint on the final state
(check_zero_mean), and the transverse endpoint error before / after the fit (smear re-drawn from its seed).
usage (in the container, PYTHONPATH = the live tree): python3 read_endc.py"""
import os, zlib, pickle, glob, numpy as np
from optimize.fit_params import _dir_from_angles, _transverse_frame
exec(open('/sdf/scratch/neutrino/pgranger/review_2026_09/KNOT/read_ens.py').read().split('\nS = list')[0])
ZB = np.array([[30.59, 0.159], [-30.59, -0.159]])        # module0 tpc_borders[:, 2]: z of [anode, cathode] (cm)
S, ARMS, Q = (1, 2, 4), ('EZT', 'EZS', 'EXS'), P + ['pos']


def ckpt(run):
    return pickle.load(open(sorted(glob.glob(f'{FR}/sci_full_{run}/history_iter14000_*.pkl'))[0], 'rb'))


def endpoints(run, smear_mm, fitted=True):
    """Constraint residual max |sum w.c| / (rms sqrt n) over batches; rms transverse endpoint error (um) along e1 / e2
    before (given endpoint) and after (given + fitted endpoint terms), vs the true endpoint. fitted=False (EXS: no
    endpoint columns): after = before, no constraint."""
    d = ckpt(run); cc, cx = d['chain_cache'], d['chain_contexts']; res, b1, b2, a1, a2 = [], [], [], [], []
    for bi in cc:
        terms = []
        for st, c in zip(cc[bi], cx[bi]):
            a = np.asarray(st['angles'], np.float64); K = len(a) // 2
            if not fitted:
                a, K = np.zeros(4), 2                                 # endpoint terms identically 0
            u0 = _dir_from_angles(c['theta0_i'], c['phi0_i']); e1, e2 = _transverse_frame(u0)
            x0 = np.asarray(c['x0_fixed'], np.float64); ends = (x0, x0 + c['total_len'] * u0)
            if smear_mm > 0:
                g = np.random.default_rng(zlib.crc32(f"{tuple(c['track_id'])}|0".encode()))
                sm = (g.normal(0.0, 0.1 * smear_mm, 3), g.normal(0.0, 0.1 * smear_mm, 3))
            else:
                sm = (np.zeros(3), np.zeros(3))
            for k, (j1, j2) in enumerate(((K - 2, 2 * K - 2), (K - 1, 2 * K - 1))):
                t = int(np.argmin(np.abs(ZB.mean(1) - ends[k][2])))
                terms.append(float(e2[2]) * np.sign(ends[k][2] - ZB[t, 0]) * a[j2])
                err = -sm[k]                                          # true - given
                b1.append(err @ e1); b2.append(err @ e2); a1.append(err @ e1 - a[j1]); a2.append(err @ e2 - a[j2])
        t = np.array(terms); rms = np.sqrt(np.mean(t ** 2))
        res.append(abs(t.sum()) / max(rms * np.sqrt(len(t)), 1e-30) if fitted else float('nan'))
    r = lambda v: 1e4 * float(np.sqrt(np.mean(np.square(v))))
    return max(res), r(b1), r(b2), r(a1), r(a2)


R = {(a, s): read(f'{a}{s}FR', 14000) for a in ARMS + ('KN10N',) for s in S}
print(f"{'run':8s} " + ' '.join(f'{q:>9s}' for q in Q))
for s in S:
    for a in ('KN10N',) + ARMS:
        print(f'{a}{s:<4d} ' + ' '.join(f'{R[a, s][q]:+9.2f}' for q in Q))
print('\npaired shift vs KN10N (per sample 1 / 2 / 4; mean)')
M = {}
for a in ARMS:
    for q in Q:
        v = np.array([R[a, s][q] - R['KN10N', s][q] for s in S]); M[a, q] = v
        print(f'  {a} {q:9s} ' + ' / '.join(f'{x:+7.2f}' for x in v) + f'   mean {v.mean():+7.2f}')
print('\nendpoints: constraint max|sum w.c|/(rms sqrt n); transverse endpoint error rms um, e1 / e2: before -> after')
for a, sm in (('EZT', 0.), ('EZS', 1.0), ('EXS', 1.0)):
    for s in S:
        c, b1, b2, a1, a2 = endpoints(f'{a}{s}FR', sm, a != 'EXS')
        print(f'  {a}{s}: constraint {c:.1e}   e1 {b1:6.1f} -> {a1:6.1f}   e2 {b2:6.1f} -> {a2:6.1f}')
lt = {a: M[a, 'lifetime'] for a in ARMS}; ld = M['EZS', 'long_diff']
anchor = ('HOLDS' if abs(lt['EZT'].mean()) <= 3 and lt['EZT'].max() <= 6 else
          'FAILS' if lt['EZT'].mean() >= 4.5 else 'UNRESOLVED')
print(f"\n(1) control EZT: mean lifetime shift {lt['EZT'].mean():+.2f}, max {lt['EZT'].max():+.2f} -> ANCHOR {anchor}")
print(f"(2) reference EXS: mean lifetime shift {lt['EXS'].mean():+.2f} -> " +
      ('valid lever (>= +6)' if lt['EXS'].mean() >= 6 else 'SMALL LEVER (< +6): the smear costs less in this recipe'))
m = lt['EZS'].mean()
v = ('ENDFIT WORKS' if anchor == 'HOLDS' and abs(m) <= 3 and abs(ld.mean()) <= 5 else
     'PARTIAL' if anchor == 'HOLDS' and m <= 0.5 * lt['EXS'].mean() else 'FAILS')
print(f"(3) test EZS: mean lifetime shift {m:+.2f}, long_diff {ld.mean():+.2f} -> {v}")
