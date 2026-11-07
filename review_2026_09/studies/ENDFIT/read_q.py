"""ENDFIT (q) readout: SHORTER coarse phase (C25, LARND_CHAIN_COARSE_ITER=2500) on REALISTIC ends (EXP = KN10N + independent
1 mm endpoint smear) against TRUTH (rules: REVIEW.md 29 Oct "ENDFIT (q)", written before any result).
Per sample s = 1..6: reference (s = 1, 2, 4: EXP<s>H; s = 3, 5, 6: EXP<s>PH), CF5000 = EXP<s>CFH (the (p) arm), C25 =
EXP<s>C25H, all read at 20000. Error % = last-1000 mean / target - 1; pos = mean pos_residual (um, vs the unsmeared truth;
the smear sets the level, the paired differences are what is read).
usage (container): python3 read_q.py   [RQ_DRY=1: the CF5000 arm stands in for C25, to test the script]"""
import glob, os, pickle, numpy as np
FR = '/sdf/group/neutrino/pgranger/larnd-sim-jax/fit_result'
P = ['Ab', 'eField', 'tran_diff', 'long_diff', 'lifetime']
f1 = lambda x: np.nan if x is None else float(np.mean(x))
DRY = os.environ.get('RQ_DRY') == '1'
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


A = ('ref', 'CF5000', 'C25')
R = {s: dict(zip(A, (read(REF[s]), read(f'EXP{s}CFH'), read(f'EXP{s}CFH' if DRY else f'EXP{s}C25H')))) for s in S}
want = {'ref': (None, '', '0'), 'CF5000': ('5000',), 'C25': ('5000',) if DRY else ('2500',)}
for s in S:
    for a in A:
        e = R[s][a]['env']
        assert e['LARND_ENDPOINT_SMEAR_MM'] == '1.0' and e['LARND_ENDPOINT_SMEAR_KEY'] == 'entry', (s, a, e)
        assert e['LARND_CHAIN_COARSE_ITER'] in want[a], (s, a, e)
print('config check OK on 6 x 3 runs (smear 1.0 mm / entry; coarse iter ref none, CF 5000, C25 2500)' + (' -- DRY RUN' if DRY else ''))
Q = P + ['pos']
print('\n   ' + ' '.join(f'{q:>23s}' for q in Q) + '   (ref / CF5000 / C25; error % vs truth, pos um)')
for s in S:
    print(f'{s:2d} ' + ' '.join('/'.join(f'{R[s][a][q]:+6.2f}' if q != 'pos' else f'{R[s][a][q]:6.1f}' for a in A) for q in Q))
ms = lambda v: (v.mean(), v.std(ddof=1) / np.sqrt(len(v)))
worse = {'CF5000': [], 'C25': []}
print('\nmean |err| ref / CF5000 / C25 (pos: mean um), paired d|err| vs ref +- SE, and C25 - CF5000')
for q in Q:
    v = {a: np.array([abs(R[s][a][q]) if q != 'pos' else R[s][a][q] for s in S]) for a in A}
    line = f'  {q:10s} ' + ' / '.join(f'{v[a].mean():6.2f}' for a in A)
    for a in ('CF5000', 'C25'):
        m, se = ms(v[a] - v['ref']); w = q != 'pos' and m > max(2 * se, 0.5)
        if w: worse[a].append(q)
        line += f'   {a}-ref {m:+6.2f} +- {se:5.2f}' + (' WORSE' if w else '      ')
    m, se = ms(v['C25'] - v['CF5000']); line += f'   C25-CF5000 {m:+6.2f} +- {se:5.2f}'
    print(line)
pos = {a: np.array([R[s][a]['pos'] for s in S]) for a in A}
below = int((pos['C25'] < pos['ref']).sum()); gain = float((pos['ref'] - pos['C25']).mean())
gcf = float((pos['ref'] - pos['CF5000']).mean())
print(f"\npositions: C25 below ref on {below}/6, mean gain {gain:+.1f} um (CF5000: "
      f"{int((pos['CF5000'] < pos['ref']).sum())}/6, {gcf:+.1f} um)")
safe = not worse['C25']; keeps = below >= 5 and gain >= 0.5 * gcf
print(f"C25: CALIBRATION {'WORSE in ' + ', '.join(worse['C25']) if worse['C25'] else 'NOT WORSE'} -> "
      f"{'SAFE' if safe else 'NOT SAFE'} on realistic ends; POSITION GAIN {'KEPT' if keeps else 'NOT KEPT'} "
      f"({below}/6 below ref, gain {gain:+.1f} vs >= {0.5 * gcf:.1f} um needed)")
print(f"VERDICT: C25 {'CANDIDATE (next: the true-end check on KN10N)' if safe and keeps else 'NOT A CANDIDATE'}"
      f"  [control: CF5000 WORSE in {', '.join(worse['CF5000']) or 'nothing'} -- (p) said long_diff]")
