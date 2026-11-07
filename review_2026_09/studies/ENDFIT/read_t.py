"""ENDFIT (t) readout: does the (s) C75 tran_diff cost REPLICATE on six samples it has not seen? (rules: REVIEW.md 30 Oct
"ENDFIT (t)", written before any result). New samples KN10N1, 2, 4, 10, 11, 12 (true endpoints); arms at H @ 20000:
ref = the production KN10N<s>H (24 Oct fit + FR, 28 Oct H; same recipe as the (s) arm P, which re-ran it on 29 Oct: P vs
production differ by tran_diff signed SD 0.30, d|err| +0.11 +- 0.15, all parameters d|err| ~0), C75 = KN10N<s>C75H.
Error % = last-1000 mean / target - 1; pos = mean pos_residual over the last 1000 iterations (um, vs truth).
usage (container): python3 read_t.py   [RT_DRY=1: runs on the (s) samples 3, 5-9 with ref = P, to test the script; it must
reproduce (s): tran_diff C75 - P +0.74 +- 0.24 WORSE]"""
import glob, os, pickle, numpy as np
FR = '/sdf/group/neutrino/pgranger/larnd-sim-jax/fit_result'
P = ['Ab', 'eField', 'tran_diff', 'long_diff', 'lifetime']
f1 = lambda x: np.nan if x is None else float(np.mean(x))
DRY = os.environ.get('RT_DRY') == '1'
OLD, NEW = [3, 5, 6, 7, 8, 9], [1, 2, 4, 10, 11, 12]
S = OLD if DRY else NEW


def read(run, it=20000):
    fs = sorted(glob.glob(f'{FR}/sci_full_{run}/history_iter{it}_*.pkl'))
    if not fs: raise SystemExit(f'{run}: no checkpoint at {it} -> set not complete, no readout')
    d = pickle.load(open(fs[0], 'rb')); assert int(d['total_iter']) == it, run
    r = {p: 100 * (np.array([f1(x) for x in d[f'{p}_iter']])[-1000:].mean() / f1(d[f'{p}_target'][0]) - 1) for p in P}
    r['pos'] = 1e4 * float(np.nanmean(np.array([f1(x) for x in d['pos_residual_iter']])[-1000:]))
    r['env'] = {k: d.get('env_overrides', {}).get(k) for k in ('LARND_CHAIN_COARSE_ITER', 'LARND_ENDPOINT_SMEAR_MM')}
    return r


A = ('ref', 'C75')
REF = (lambda s: f'KN10N{s}PH') if DRY else (lambda s: f'KN10N{s}H')
R = {s: {'ref': read(REF(s)), 'C75': read(f'KN10N{s}C75H')} for s in S}
for s in S:
    for a, want in (('ref', (None, '', '0')), ('C75', ('7500',))):
        e = R[s][a]['env']
        assert e['LARND_ENDPOINT_SMEAR_MM'] in (None, '', '0', '0.0'), (s, a, e)
        assert e['LARND_CHAIN_COARSE_ITER'] in want, (s, a, e)
print(f'config check OK on 6 x 2 runs (true ends; coarse iter ref none, C75 7500), samples {S}' + (' -- DRY RUN (ref = P)' if DRY else ''))
Q = P + ['pos']
print('\n   ' + ' '.join(f'{q:>15s}' for q in Q) + '   (ref / C75; error % vs truth, pos um)')
for s in S:
    print(f'{s:2d} ' + ' '.join('/'.join(f'{R[s][a][q]:+6.2f}' if q != 'pos' else f'{R[s][a][q]:6.1f}' for a in A) for q in Q))
ms = lambda v: (v.mean(), v.std(ddof=1) / np.sqrt(len(v)))
worse = []; D = {}
print('\nmean |err| ref / C75 (pos: mean um), paired C75 - ref +- SE')
for q in Q:
    v = {a: np.array([abs(R[s][a][q]) if q != 'pos' else R[s][a][q] for s in S]) for a in A}
    D[q] = v['C75'] - v['ref']; m, se = ms(D[q]); w = q != 'pos' and m > max(2 * se, 0.5)
    if w: worse.append(q)
    print(f'  {q:10s} {v["ref"].mean():6.2f} / {v["C75"].mean():6.2f}   C75-ref {m:+6.2f} +- {se:4.2f}' + (' WORSE' if w else ''))
m, se = ms(D['tran_diff']); rep = m > max(2 * se, 0.5)
print(f"\nPRIMARY: tran_diff C75 - ref d|err| {m:+.2f} +- {se:.2f} (worse on {int((D['tran_diff'] > 0).sum())}/6) -> "
      f"{'REPLICATED (the (s) cost is real)' if rep else 'NOT REPLICATED'}")
below = int((D['pos'] < 0).sum())
print(f"SECONDARY ((o) rule): positions below ref {below}/6 (gain {-D['pos'].mean():+.1f} um); CALIBRATION "
      f"{'WORSE in ' + ', '.join(worse) if worse else 'NOT WORSE'} -> C75 on these samples "
      f"{'CONFIRMED' if below >= 5 and not worse else 'NOT CONFIRMED'}")
if not DRY:
    Ro = {s: {'ref': read(f'KN10N{s}PH'), 'C75': read(f'KN10N{s}C75H')} for s in OLD}
    print('\nPOOLED n = 12 (descriptive; (s) samples vs P + new samples vs production): C75 - ref d|err| +- SE')
    for q in Q:
        do = np.array([(abs(Ro[s]['C75'][q]) - abs(Ro[s]['ref'][q])) if q != 'pos' else Ro[s]['C75'][q] - Ro[s]['ref'][q] for s in OLD])
        m, se = ms(np.concatenate([do, D[q]])); print(f'  {q:10s} {m:+6.2f} +- {se:4.2f}')
