"""ENDFIT (p) readout: coarse-to-fine (CF, LARND_CHAIN_COARSE_ITER=5000) on REALISTIC ends (EXP = KN10N + independent
1 mm endpoint smear) against TRUTH (rules: REVIEW.md 29 Oct "ENDFIT (p)", written before any result).
Pairs: s = 1, 2, 4 from (n): EXP<s>H (production, 28 Oct tree) vs EXP<s>CFH; s = 3, 5, 6 new: EXP<s>PH (production
re-run, today's tree) vs EXP<s>CFH. All read at 20000. Error % = last-1000 mean / target - 1; pos = mean pos_residual
(um, vs the unsmeared truth: dominated by the smear, descriptive only).
usage (container): python3 read_p.py   [RP_DRY=1: only the (n) pairs, to test the script]"""
import glob, os, pickle, numpy as np
FR = '/sdf/group/neutrino/pgranger/larnd-sim-jax/fit_result'
P = ['Ab', 'eField', 'tran_diff', 'long_diff', 'lifetime']
f1 = lambda x: np.nan if x is None else float(np.mean(x))
DRY = os.environ.get('RP_DRY') == '1'
PAIRS = {1: ('EXP1H', 'EXP1CFH'), 2: ('EXP2H', 'EXP2CFH'), 4: ('EXP4H', 'EXP4CFH')}
if not DRY: PAIRS.update({s: (f'EXP{s}PH', f'EXP{s}CFH') for s in (3, 5, 6)})


def read(run, it=20000):
    fs = sorted(glob.glob(f'{FR}/sci_full_{run}/history_iter{it}_*.pkl'))
    if not fs: raise SystemExit(f'{run}: no checkpoint at {it} -> set not complete, no readout')
    d = pickle.load(open(fs[0], 'rb')); assert int(d['total_iter']) == it, run
    r = {p: 100 * (np.array([f1(x) for x in d[f'{p}_iter']])[-1000:].mean() / f1(d[f'{p}_target'][0]) - 1) for p in P}
    r['pos'] = 1e4 * float(np.nanmean(np.array([f1(x) for x in d['pos_residual_iter']])[-1000:]))
    r['env'] = {k: d.get('env_overrides', {}).get(k) for k in ('LARND_CHAIN_COARSE_ITER', 'LARND_ENDPOINT_SMEAR_MM', 'LARND_ENDPOINT_SMEAR_KEY')}
    return r


R = {s: (read(a), read(b)) for s, (a, b) in PAIRS.items()}
for s, (r, c) in R.items():
    assert r['env']['LARND_ENDPOINT_SMEAR_MM'] == c['env']['LARND_ENDPOINT_SMEAR_MM'] == '1.0', (s, r['env'], c['env'])
    assert r['env']['LARND_ENDPOINT_SMEAR_KEY'] == c['env']['LARND_ENDPOINT_SMEAR_KEY'] == 'entry', (s, r['env'], c['env'])
    assert r['env']['LARND_CHAIN_COARSE_ITER'] in (None, '', '0') and c['env']['LARND_CHAIN_COARSE_ITER'] == '5000', (s, r['env'], c['env'])
print(f"config check OK on {len(R)} pairs (smear 1.0 mm / entry on both; reference no coarse phase; CF 5000)" + (' -- DRY RUN' if DRY else ''))
Q = P + ['pos']
print('\n   ' + ' '.join(f'{q:>15s}' for q in Q) + '   (reference / CF; error % vs truth, pos um)')
for s, (r, c) in R.items():
    print(f'{s:2d} ' + ' '.join(f'{r[q]:+7.2f}/{c[q]:+7.2f}' if q != 'pos' else f'{r[q]:7.1f}/{c[q]:7.1f}' for q in Q))
S = list(R); worse = []
print(f'\npaired over {len(S)} samples: mean |err| reference / CF, mean d|err| (CF - ref) +- SE')
for q in Q:
    a = np.array([abs(R[s][0][q]) for s in S]); b = np.array([abs(R[s][1][q]) for s in S]); d = b - a
    m, se = d.mean(), d.std(ddof=1) / np.sqrt(len(d)); w = q != 'pos' and m > max(2 * se, 0.5)
    if w: worse.append(q)
    new = [s for s in S if s in (3, 5, 6)]
    extra = f'   new samples only: {np.mean([abs(R[s][1][q]) - abs(R[s][0][q]) for s in new]):+.2f}' if new else ''
    print(f'  {q:10s} {a.mean():6.2f} / {b.mean():6.2f}   d {m:+6.2f} +- {se:5.2f}' + ('   WORSE' if w else '') + extra)
pb = sum(R[s][1]['pos'] < R[s][0]['pos'] for s in S)
print(f'\npositions (vs unsmeared truth, descriptive): CF below the reference on {pb}/{len(S)}')
print(f"VERDICT: CF on realistic ends: CALIBRATION {'WORSE in ' + ', '.join(worse) if worse else 'NOT WORSE'} -> "
      f"{'NOT SAFE on realistic ends' if worse else 'SAFE on realistic ends'}")
