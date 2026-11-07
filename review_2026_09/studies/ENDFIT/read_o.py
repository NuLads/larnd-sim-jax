"""ENDFIT (o) readout: coarse-to-fine confirmation against TRUTH (rules: REVIEW.md 29 Oct "ENDFIT (o)", written before
any result). Samples KN10N3, 5, 6, 7, 8, 9 (true endpoints); arms, each a full production pipeline read at its H @ 20000:
  P  = production re-run on today's tree (the paired reference), CF = LARND_CHAIN_COARSE_ITER=5000,
  SD = LARND_SEED_DROP_UPPER=1 (control: only the seed's upper modes dropped), and the 24 Oct production H (old tree).
Per parameter: error % = mean of the last 1000 iterations / target - 1 (read_ens convention); positions: mean
pos_residual over the last 1000 iterations (um, vs truth).
usage (container): python3 read_o.py   [RO_DRY=1: every arm read from the production H runs, to test the script]"""
import glob, os, pickle, numpy as np
FR = '/sdf/group/neutrino/pgranger/larnd-sim-jax/fit_result'
P = ['Ab', 'eField', 'tran_diff', 'long_diff', 'lifetime']
S = [3, 5, 6, 7, 8, 9]
f1 = lambda x: np.nan if x is None else float(np.mean(x))
DRY = os.environ.get('RO_DRY') == '1'


def read(run, it=20000):
    fs = sorted(glob.glob(f'{FR}/sci_full_{run}/history_iter{it}_*.pkl'))
    if not fs: raise SystemExit(f'{run}: no checkpoint at {it} -> set not complete, no readout')
    d = pickle.load(open(fs[0], 'rb')); n = int(d['total_iter']); assert n == it, (run, n)
    r = {}
    for p in P:
        a = np.array([f1(x) for x in d[f'{p}_iter']])
        t = f1(d[f'{p}_target'][0]); r[p] = 100 * (a[-1000:].mean() / t - 1)
    pr = np.array([f1(x) for x in d['pos_residual_iter']]); r['pos'] = 1e4 * float(np.nanmean(pr[-1000:]))
    r['env'] = {k: v for k, v in d.get('env_overrides', {}).items() if k in ('LARND_CHAIN_COARSE_ITER', 'LARND_SEED_DROP_UPPER')}
    return r


ARMS = {'H': 'H', 'P': 'PH', 'CF': 'CFH', 'SD': 'SDH'}
R = {a: {s: read(f'KN10N{s}' + ('H' if DRY else suf)) for s in S} for a, suf in ARMS.items()}
for a, want in (('P', {}), ('CF', {'LARND_CHAIN_COARSE_ITER': '5000'}), ('SD', {'LARND_SEED_DROP_UPPER': '1'})):
    if not DRY:
        assert all(R[a][s]['env'] == want for s in S), (a, [R[a][s]['env'] for s in S])
print('config check: P no flag, CF COARSE_ITER=5000, SD SEED_DROP_UPPER=1 on all 6 (from each checkpoint env)' if not DRY else 'DRY RUN')
Q = P + ['pos']
print(f"\n{'':4s} " + ' '.join(f'{q:>27s}' for q in Q) + '   (H / P / CF / SD; error % vs truth, pos um)')
for s in S:
    print(f'{s:4d} ' + ' '.join('/'.join(f'{R[a][s][q]:+6.2f}' if q != 'pos' else f'{R[a][s][q]:6.1f}' for a in ARMS) for q in Q))
ms = lambda v: (v.mean(), v.std(ddof=1) / np.sqrt(len(v)))
print('\nmean |error| over the 6 samples (pos: mean um), and paired vs P: mean d|err| +- SE')
worse = {}
for q in Q:
    v = {a: np.array([abs(R[a][s][q]) for s in S]) for a in ARMS}
    line = f'{q:10s} ' + '  '.join(f'{a} {v[a].mean():6.2f}' for a in ARMS)
    for a in ('H', 'CF', 'SD'):
        m, se = ms(v[a] - v['P']); line += f'   {a}-P {m:+6.2f} +- {se:5.2f}'
        if q != 'pos':
            worse[a, q] = m > max(2 * se, 0.5)
    print(line)
pp = {a: np.array([R[a][s]['pos'] for s in S]) for a in ARMS}
cl = {a: int((pp[a] < pp['P']).sum()) for a in ('H', 'CF', 'SD')}
g = {a: float((pp['P'] - pp[a]).mean()) for a in ('H', 'CF', 'SD')}
print(f"\npositions below P: H {cl['H']}/6 (old tree, the noise reference), CF {cl['CF']}/6, SD {cl['SD']}/6; "
      f"mean gain P - X: H {g['H']:+.1f}, CF {g['CF']:+.1f}, SD {g['SD']:+.1f} um; replica |P - H| per sample: "
      + ' '.join(f'{x:.1f}' for x in np.abs(pp['P'] - pp['H'])))
for a in ('CF', 'SD'):
    bad = [q for q in P if worse[a, q]]
    print(f"{a}: POSITION {'CONFIRMED' if cl[a] >= 5 else 'not confirmed'} ({cl[a]}/6); CALIBRATION "
          f"{'WORSE in ' + ', '.join(bad) if bad else 'NOT WORSE'} -> {'CONFIRMED' if cl[a] >= 5 and not bad else 'NOT CONFIRMED'}")
if cl['CF'] >= 5 and not [q for q in P if worse['CF', q]]:
    f = g['SD'] / g['CF'] if g['CF'] > 0 else float('nan')
    print(f"mechanism: SD gain / CF gain = {f:.2f} -> "
          + ('SEED DROP explains it' if f >= 2 / 3 else 'COARSE PHASE explains it' if f <= 1 / 3 else 'BOTH / mixed'))
