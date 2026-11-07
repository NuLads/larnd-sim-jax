"""ENDFIT (u) readout: C75 (LARND_CHAIN_COARSE_ITER=7500) on REALISTIC ends, six more samples, to settle the (r) long_diff
question (rules: REVIEW.md 31 Oct "ENDFIT (u)", written before any result). New samples s = 7..12: ref = EXP<s>H (fresh
production pipeline with the 1 mm independent endpoint smear), C75 = EXP<s>C75H, same day, same tree. Pooled with the (r)
samples 1..6 (ref EXP<s>H for s = 1, 2, 4, EXP<s>PH for 3, 5, 6; C75 = EXP<s>C75H). All read at 20000 vs truth.
Error % = last-1000 mean / target - 1; pos = mean pos_residual (um, vs the unsmeared truth; paired differences are read).
usage (container): python3 read_u.py   [RU_DRY=1: runs the new-sample block on the (r) samples, to test the script; it must
reproduce (r) for C75: long_diff +1.59 +- 0.97, positions 5/6, gain +11.0 um]"""
import glob, os, pickle, numpy as np
FR = '/sdf/group/neutrino/pgranger/larnd-sim-jax/fit_result'
P = ['Ab', 'eField', 'tran_diff', 'long_diff', 'lifetime']
f1 = lambda x: np.nan if x is None else float(np.mean(x))
DRY = os.environ.get('RU_DRY') == '1'
OLD, NEW = [1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]
REF_OLD = {s: f'EXP{s}H' if s in (1, 2, 4) else f'EXP{s}PH' for s in OLD}


def read(run, it=20000):
    fs = sorted(glob.glob(f'{FR}/sci_full_{run}/history_iter{it}_*.pkl'))
    if not fs: raise SystemExit(f'{run}: no checkpoint at {it} -> set not complete, no readout')
    d = pickle.load(open(fs[0], 'rb')); assert int(d['total_iter']) == it, run
    r = {p: 100 * (np.array([f1(x) for x in d[f'{p}_iter']])[-1000:].mean() / f1(d[f'{p}_target'][0]) - 1) for p in P}
    r['pos'] = 1e4 * float(np.nanmean(np.array([f1(x) for x in d['pos_residual_iter']])[-1000:]))
    r['env'] = {k: d.get('env_overrides', {}).get(k) for k in ('LARND_CHAIN_COARSE_ITER', 'LARND_ENDPOINT_SMEAR_MM', 'LARND_ENDPOINT_SMEAR_KEY')}
    return r


def load(S, ref):
    R = {s: {'ref': read(ref(s)), 'C75': read(f'EXP{s}C75H')} for s in S}
    for s in S:
        for a, want in (('ref', (None, '', '0')), ('C75', ('7500',))):
            e = R[s][a]['env']
            assert e['LARND_ENDPOINT_SMEAR_MM'] == '1.0' and e['LARND_ENDPOINT_SMEAR_KEY'] == 'entry', (s, a, e)
            assert e['LARND_CHAIN_COARSE_ITER'] in want, (s, a, e)
    return R


ms = lambda v: (v.mean(), v.std(ddof=1) / np.sqrt(len(v)))
Q = P + ['pos']


def block(R, S, title):
    n = len(S); print(f'\n{title} (n = {n}): mean |err| ref / C75 (pos: mean um), paired C75 - ref +- SE')
    worse = []; D = {}
    for q in Q:
        v = {a: np.array([abs(R[s][a][q]) if q != 'pos' else R[s][a][q] for s in S]) for a in ('ref', 'C75')}
        D[q] = v['C75'] - v['ref']; m, se = ms(D[q]); w = q != 'pos' and m > max(2 * se, 0.5)
        if w: worse.append(q)
        print(f'  {q:10s} {v["ref"].mean():6.2f} / {v["C75"].mean():6.2f}   C75-ref {m:+6.2f} +- {se:4.2f}' + (' WORSE' if w else '')
              + (f'   (worse on {int((D[q] > 0).sum())}/{n})' if q != 'pos' else ''))
    below = int((D['pos'] < 0).sum()); gain = -float(D['pos'].mean())
    keeps = below >= (5 if n == 6 else 10) and gain >= 5.3
    print(f"  -> CALIBRATION {'WORSE in ' + ', '.join(worse) if worse else 'NOT WORSE'} ({'NOT SAFE' if worse else 'SAFE'}); "
          f"POSITION GAIN {'KEPT' if keeps else 'NOT KEPT'} ({below}/{n} below ref, gain {gain:+.1f} um vs >= 5.3)")
    return not worse and keeps


S = OLD if DRY else NEW
R = load(S, (lambda s: REF_OLD[s]) if DRY else (lambda s: f'EXP{s}H'))
print(f'config check OK on {len(S)} x 2 runs (smear 1.0 mm / entry; coarse iter ref none, C75 7500), samples {S}' + (' -- DRY RUN' if DRY else ''))
print('\n   ' + ' '.join(f'{q:>15s}' for q in Q) + '   (ref / C75; error % vs truth, pos um)')
for s in S:
    print(f'{s:2d} ' + ' '.join('/'.join(f'{R[s][a][q]:+6.2f}' if q != 'pos' else f'{R[s][a][q]:6.1f}' for a in ('ref', 'C75')) for q in Q))
ok_new = block(R, S, 'NEW SAMPLES' if not DRY else '(r) SAMPLES [dry]')
if not DRY:
    Ro = load(OLD, lambda s: REF_OLD[s]); Rp = {**Ro, **R}
    ok = block(Rp, OLD + NEW, 'PRIMARY: POOLED (r) + new')
    print(f"\nnew samples alone: {'SAFE + GAIN KEPT' if ok_new else 'not both'}")
    print(f"VERDICT (pooled n = 12): C75 on realistic ends {'SAFE + POSITION GAIN KEPT -> passes every pre-registered check' if ok else 'NOT CLEAN -> not a clean default'}")
