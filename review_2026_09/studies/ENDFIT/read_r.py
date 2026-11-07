"""ENDFIT (r) readout: LONGER coarse phase (C75, LARND_CHAIN_COARSE_ITER=7500) on REALISTIC ends (EXP = KN10N + independent
1 mm endpoint smear) against TRUTH (rules: REVIEW.md 30 Oct "ENDFIT (r)", written before any result).
Per sample s = 1..6: reference (s = 1, 2, 4: EXP<s>H; s = 3, 5, 6: EXP<s>PH), C25 = EXP<s>C25H ((q)), CF5000 = EXP<s>CFH
((p)), C75 = EXP<s>C75H, all read at 20000. Error % = last-1000 mean / target - 1; pos = mean pos_residual (um, vs the
unsmeared truth; the smear sets the level, the paired differences are what is read).
usage (container): python3 read_r.py   [RR_DRY=1: the CF5000 arm stands in for C75, to test the script]"""
import glob, os, pickle, numpy as np
FR = '/sdf/group/neutrino/pgranger/larnd-sim-jax/fit_result'
P = ['Ab', 'eField', 'tran_diff', 'long_diff', 'lifetime']
f1 = lambda x: np.nan if x is None else float(np.mean(x))
DRY = os.environ.get('RR_DRY') == '1'
S = [1, 2, 3, 4, 5, 6]
REF = {s: f'EXP{s}H' if s in (1, 2, 4) else f'EXP{s}PH' for s in S}


def read(run, it=20000):
    fs = sorted(glob.glob(f'{FR}/sci_full_{run}/history_iter{it}_*.pkl'))
    if not fs: raise SystemExit(f'{run}: no checkpoint at {it} -> set not complete, no readout')
    d = pickle.load(open(fs[0], 'rb')); assert int(d['total_iter']) == it, run
    r = {p: 100 * (np.array([f1(x) for x in d[f'{p}_iter']])[-1000:].mean() / f1(d[f'{p}_target'][0]) - 1) for p in P}
    r['pos'] = 1e4 * float(np.nanmean(np.array([f1(x) for x in d['pos_residual_iter']])[-1000:]))
    r['env'] = {k: d.get('env_overrides', {}).get(k) for k in ('LARND_CHAIN_COARSE_ITER', 'LARND_ENDPOINT_SMEAR_MM', 'LARND_ENDPOINT_SMEAR_KEY')}
    return r


A = ('ref', 'C25', 'CF5000', 'C75')
RUN = {'ref': lambda s: REF[s], 'C25': lambda s: f'EXP{s}C25H', 'CF5000': lambda s: f'EXP{s}CFH',
       'C75': lambda s: f'EXP{s}CFH' if DRY else f'EXP{s}C75H'}
R = {s: {a: read(RUN[a](s)) for a in A} for s in S}
want = {'ref': (None, '', '0'), 'C25': ('2500',), 'CF5000': ('5000',), 'C75': ('5000',) if DRY else ('7500',)}
for s in S:
    for a in A:
        e = R[s][a]['env']
        assert e['LARND_ENDPOINT_SMEAR_MM'] == '1.0' and e['LARND_ENDPOINT_SMEAR_KEY'] == 'entry', (s, a, e)
        assert e['LARND_CHAIN_COARSE_ITER'] in want[a], (s, a, e)
print('config check OK on 6 x 4 runs (smear 1.0 mm / entry; coarse iter ref none, 2500, 5000, 7500)' + (' -- DRY RUN' if DRY else ''))
Q = P + ['pos']
print('\n   ' + ' '.join(f'{q:>31s}' for q in Q) + '   (ref / C25 / CF5000 / C75; error % vs truth, pos um)')
for s in S:
    print(f'{s:2d} ' + ' '.join('/'.join(f'{R[s][a][q]:+6.2f}' if q != 'pos' else f'{R[s][a][q]:6.1f}' for a in A) for q in Q))
ms = lambda v: (v.mean(), v.std(ddof=1) / np.sqrt(len(v)))
V = {q: {a: np.array([abs(R[s][a][q]) if q != 'pos' else R[s][a][q] for s in S]) for a in A} for q in Q}
worse = []
print('\nmean |err| ref / C25 / CF5000 / C75 (pos: mean um); paired d vs ref +- SE; C75 - CF5000')
for q in Q:
    v = V[q]; line = f'  {q:10s} ' + ' / '.join(f'{v[a].mean():6.2f}' for a in A)
    for a in ('C25', 'CF5000', 'C75'):
        m, se = ms(v[a] - v['ref']); w = q != 'pos' and m > max(2 * se, 0.5)
        if w and a == 'C75': worse.append(q)
        line += f'   {a}-ref {m:+6.2f} +- {se:4.2f}' + (' WORSE' if w else '      ')
    m, se = ms(v['C75'] - v['CF5000']); line += f'   C75-CF5000 {m:+6.2f} +- {se:4.2f}'
    print(line)
pos = {a: V['pos'][a] for a in A}
below = int((pos['C75'] < pos['ref']).sum()); gain = float((pos['ref'] - pos['C75']).mean())
gcf = float((pos['ref'] - pos['CF5000']).mean())
safe = not worse; keeps = below >= 5 and gain >= 0.5 * gcf
print(f"\npositions: C75 below ref on {below}/6, mean gain {gain:+.1f} um (C25 {float((pos['ref'] - pos['C25']).mean()):+.1f}, "
      f"CF5000 {gcf:+.1f}); C75 below CF5000 on {int((pos['C75'] < pos['CF5000']).sum())}/6")
m, se = ms(V['long_diff']['C75'] - V['long_diff']['CF5000'])
hyp = 'SUPPORTED' if m < -max(2 * se, 0.5) else ('TREND only (right sign, not significant)' if m < 0 else 'NOT SUPPORTED')
print(f"HYPOTHESIS (a later release costs less long_diff): C75 - CF5000 long_diff d|err| {m:+.2f} +- {se:.2f} -> {hyp}")
print(f"C75: CALIBRATION {'WORSE in ' + ', '.join(worse) if worse else 'NOT WORSE'} -> {'SAFE' if safe else 'NOT SAFE'} on "
      f"realistic ends; POSITION GAIN {'KEPT' if keeps else 'NOT KEPT'} ({below}/6 below ref, gain {gain:+.1f} vs >= {0.5 * gcf:.1f} um)")
print(f"VERDICT: C75 {'CANDIDATE (next: the true-end check on KN10N)' if safe and keeps else 'NOT A CANDIDATE'}")
