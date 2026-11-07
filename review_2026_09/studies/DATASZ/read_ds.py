"""DATASZ readout (rules: REVIEW.md "DATASZ: accuracy vs data volume and batch size", written before any job).
T5 (data volume, L = 400 cm): arms V025 / V050 / V100 / V200 = 25 / 50 / 100 / 200 batches (~10 / 20 / 40 / 80 m of track);
  V100 = Set A of the UQ test (EXP17-28C75H). T6 (batch size at fixed tracks): B800 / B200 (50 x 800 cm, 200 x 200 cm, mean
  constraint w = 2500 L/400) vs EXP<s>C75H, s = 17-24, paired. Error % = mean over the last 10 epochs (1000 r iterations) /
  truth - 1 at the end point 20000 r; pos = mean pos_residual over the same window (um).
usage (container): python3 read_ds.py [t5|t6|both]     DS_DRY=1: stand-in runs (script test only, no physics)."""
import glob, os, pickle, re, sys, numpy as np
from scipy import stats
sys.path.insert(0, '/sdf/scratch/neutrino/pgranger/review_2026_09/UQ')
import uq_lib as U
FR = '/sdf/group/neutrino/pgranger/larnd-sim-jax/fit_result'; K = '/sdf/scratch/neutrino/pgranger/review_2026_09/KNOT'
DUMPS = '/sdf/scratch/neutrino/pgranger/review_2026_09/UQ/dumps'
P = ['Ab', 'eField', 'tran_diff', 'long_diff', 'lifetime']
f1 = lambda x: np.nan if x is None else float(np.mean(x))
DRY = os.environ.get('DS_DRY') == '1'
SEEDS = list(range(17, 25))
# arm -> (N_b, L, data_sz, runs); r = N_b / 100 for the V arms, (40000 / L) / 100 for the B arms
ARMS = {'V025': (25, 400, 600, [f'V025s{s}H' for s in SEEDS]), 'V050': (50, 400, 600, [f'V050s{s}H' for s in SEEDS]),
        'V100': (100, 400, 600, [f'EXP{s}C75H' for s in range(17, 29)]),
        'V200': (200, 400, 1200, [f'V200s{s}H' for s in SEEDS]),
        'B800': (50, 800, 600, [f'B800s{s}H' for s in SEEDS]), 'B200': (200, 200, 600, [f'B200s{s}H' for s in SEEDS])}
DUMP = {'V025': [f'D_V025s{s}' for s in (17, 19, 21)], 'V050': [f'D_V050s{s}' for s in (17, 19, 21)],
        'V100': [f'D_EXP{s}' for s in range(17, 29)], 'V200': [f'D_V200s{s}' for s in (17, 18, 19)]}
if DRY:   # stand-ins: production runs relabelled; checks off (they are not these configurations)
    tr = [f'EXP{s}C75H' for s in range(1, 13)]
    ARMS = {'V025': (100, 400, 600, tr[:8]), 'V050': (100, 400, 600, tr[4:]), 'V100': (100, 400, 600, tr),
            'V200': (100, 400, 600, tr[2:10]), 'B800': (100, 400, 600, tr[:8]), 'B200': (100, 400, 600, tr[1:9])}
    DUMP = {a: [f'T1_EXP{s}_20k' for s in (1, 2, 3, 4, 5, 6)] for a in ('V025', 'V050', 'V100', 'V200')}
    REF6 = tr[:8]
else:
    REF6 = [f'EXP{s}C75H' for s in SEEDS]
NBN = {'V025': 25, 'V050': 50, 'V100': 100, 'V200': 200}


def read(run, nb, L, dsz):
    r = nb / 100
    it = int(round(20000 * r))
    fs = sorted(glob.glob(f'{FR}/sci_full_{run}/history_iter{it}_*.pkl'))
    if not fs:
        raise SystemExit(f'{run}: no checkpoint at {it} -> set not complete, no readout')
    d = pickle.load(open(fs[0], 'rb')); assert int(d['total_iter']) == it, (run, d['total_iter'])
    w = int(round(1000 * r))
    out = {}
    for p in P:
        h = np.array([f1(x) for x in d[f'{p}_iter']]); assert len(h) == it + 1, (run, p, len(h))   # initial value + one per iteration
        out[p] = 100 * (h[-w:].mean() / f1(d[f'{p}_target'][0]) - 1)
    out['pos'] = 1e4 * float(np.nanmean(np.array([f1(x) for x in d['pos_residual_iter']])[-w:]))
    if not DRY:   # config check: stage-1 provenance (a resumed run keeps it), env, the stage-1 log, the FR / H inners
        a = d['provenance']['argv']; g = lambda k: a[a.index(k) + 1]; env = d.get('env_overrides', {})
        base = run[:-1]
        seed = int(re.search(r'(?:s|EXP)(\d+)', base).group(1))
        want = dict(seed=str(seed), nb=str(nb), L=('400' if L == 400 else repr(float(L)).rstrip('0').rstrip('.')),
                    dsz=str(dsz), w=('2500.0' if L == 400 else repr(2500.0 * L / 400)), its=str(int(10000 * r)),
                    coarse=str(int(7500 * r)), smear='1.0', key='entry')
        got = dict(seed=g('--data_seed'), nb=g('--max_nbatch'), L=g('--max_batch_len'), dsz=g('--data_sz'),
                   w=g('--dedx_mean_constraint_weight'), its=g('--iterations'), coarse=env.get('LARND_CHAIN_COARSE_ITER'),
                   smear=env.get('LARND_ENDPOINT_SMEAR_MM'), key=env.get('LARND_ENDPOINT_SMEAR_KEY'))
        assert got == want, (run, got, want)
        logs = [f for f in glob.glob(f'{K}/logs/{base}-*.out') if 'rc=0' in open(f).read()[-300:]]
        assert len(logs) >= 1, (run, 'no successful stage-1 log')
        txt = open(sorted(logs)[-1]).read()
        assert f'The number of simulation batches is {nb}.' in txt, (run, 'batch count')
        out['cm'] = float(re.search(r'simulation data includes a total track length of ([0-9.]+) cm', txt).group(1))
        for st, n_it in (('FR', 4000), ('H', 6000)):
            inner = open(f'{K}/inner_sci_full_{base}{st}.sh').read()
            assert f'--iterations {int(n_it * r)} ' in inner, (run, st)
    return out


def arm(name):
    nb, L, dsz, runs = ARMS[name]
    R = [read(x, nb, L, dsz) for x in runs]
    E = np.array([[x[p] for p in P] for x in R]); pos = np.array([x['pos'] for x in R])
    return E, pos, R


def e1sig(names):
    s = []
    for n in names:
        d = U.load(f'{DUMPS}/{n}.npz'); s.append(100 * U.estimators(d)['E1']['sig'])
    return np.array(s)


def t5():
    A = {a: arm(a) for a in ('V025', 'V050', 'V100', 'V200')}
    N = np.array([NBN[a] for a in A]); n = np.array([len(A[a][0]) for a in A])
    print(f"T5 config check OK{' -- DRY RUN (stand-ins)' if DRY else ''}: arms " +
          ', '.join(f"{a} n={len(A[a][0])}" + (f" ({np.mean([x['cm'] for x in A[a][2]]) / 100:.0f} m)" if not DRY else '') for a in A))
    print('\n arm      ' + ''.join(f'{p:>24s}' for p in P) + '      pos um    (mean +- SEM (SD), error %)')
    M = np.array([A[a][0].mean(0) for a in A]); SDv = np.array([A[a][0].std(0, ddof=1) for a in A])
    for i, a in enumerate(A):
        print(f' {a} n={n[i]:2d} ' + ''.join(f'{M[i, j]:+.3f}+-{SDv[i, j] / np.sqrt(n[i]):.3f} ({SDv[i, j]:.3f})'.rjust(24) for j in range(5))
              + f'   {A[a][1].mean():6.1f}+-{A[a][1].std(ddof=1) / np.sqrt(n[i]):3.1f}')
    x = np.log(N); wts = 2 * (n - 1)
    print('\n spread exponent alpha (SD ~ N^-alpha; WLS of ln SD on ln N, var ln SD = 1/(2(n-1)); 95 % CI) and bias:')
    verdict = {}
    for j, p in enumerate(P):
        y = np.log(SDv[:, j]); W = np.diag(wts); X = np.c_[np.ones(4), x]
        C = np.linalg.inv(X.T @ W @ X); b = C @ X.T @ W @ y; al, se = -b[1], np.sqrt(C[1, 1])
        chi2 = float(((y - X @ b) ** 2 * wts).sum()); pfit = 1 - stats.chi2.cdf(chi2, 2)
        lo, hi = al - 1.96 * se, al + 1.96 * se
        c0, c5 = lo <= 0 <= hi, lo <= 0.5 <= hi
        cls = ('UNRESOLVED' if c0 and c5 else 'STATISTICAL' if c5 else 'FLOOR' if c0 else
               'FASTER THAN STATISTICS' if lo > 0.5 else 'SLOWER THAN STATISTICS' if lo > 0 else 'WORSE WITH DATA')
        if pfit < 0.01:
            cls += ' (NOT A POWER LAW, chi2 p %.3f)' % pfit
        sem = SDv[:, j] / np.sqrt(n); wb = 1 / sem ** 2; mb = (M[:, j] * wb).sum() / wb.sum()
        chib = float((((M[:, j] - mb) / sem) ** 2).sum()); pb = 1 - stats.chi2.cdf(chib, 3)
        bcls = 'BIAS CONSTANT IN N' if pb > 0.05 else 'BIAS CHANGES WITH N'
        sl = np.polyfit(x, M[:, j], 1, w=1 / sem)[0]
        verdict[p] = (cls, bcls)
        print(f'  {p:10s} alpha {al:+.2f} [{lo:+.2f}, {hi:+.2f}] fit p {pfit:.2f} -> {cls:24s} | bias {bcls} (chi2 p {pb:.2f}; '
              f'weighted mean {mb:+.3f}; d bias / d ln N {sl:+.3f})')
    rms = np.sqrt(M ** 2 + SDv ** 2)
    print('\n descriptive: RMS error (sqrt(bias^2 + SD^2)) per arm: ' + ' | '.join(f'{a} ' + ' '.join(f'{v:.3f}' for v in rms[i]) for i, a in enumerate(A)))
    S1 = {a: e1sig(DUMP[a]) for a in DUMP}
    med = np.array([np.median(S1[a], 0) for a in A])
    ae = [-np.polyfit(x, np.log(med[:, j]), 1)[0] for j in range(5)]
    ok = all(abs(v - 0.5) <= 0.05 for v in ae)
    print(' sigma_E1 (median over the dumps; %): ' + ' | '.join(f'{a} ' + ' '.join(f'{v:.4g}' for v in med[i]) for i, a in enumerate(A)))
    print(f" sigma_E1 exponent: {' '.join(f'{v:.3f}' for v in ae)} -> {'FLOOR SCALES AS STATISTICS' if ok else 'FLOOR DOES NOT SCALE AS 1/sqrt(N)'}")
    print(' k(N) = SD / sigma_E1: ' + ' | '.join(f'{a} ' + ' '.join(f'{v:.2f}' for v in SDv[i] / med[i]) for i, a in enumerate(A)))
    pm = np.array([A[a][1].mean() for a in A]); pse = np.array([A[a][1].std(ddof=1) / np.sqrt(len(A[a][1])) for a in A])
    print(f" position (descriptive): {' '.join(f'{a} {m:.1f}+-{s:.1f}' for a, m, s in zip(A, pm, pse))} um")
    return verdict


def t6():
    nb, L, dsz, runs = ARMS['V100']
    ref = {x: read(x, nb, L, dsz) for x in REF6}
    Er = np.array([[ref[x][p] for p in P] for x in REF6]); posr = np.array([ref[x]['pos'] for x in REF6])
    lo, hi = np.sqrt(stats.f.ppf(0.025, 7, 7)), np.sqrt(stats.f.ppf(0.975, 7, 7))
    print(f"\nT6 config check OK{' -- DRY RUN (stand-ins)' if DRY else ''}; reference EXP17-24C75H (400 cm x 100), paired by sample; "
          f'SD ratio band [{lo:.2f}, {hi:.2f}] (F(7,7) 95 %)')
    allok = True
    for b in ('B800', 'B200'):
        E, pos, R = arm(b)
        if not DRY:
            cm = np.array([x['cm'] for x in R]); cmr = np.array([ref[x]['cm'] for x in REF6])
            print(f'  {b}: track length used {cm.mean():.0f} cm vs ref {cmr.mean():.0f} cm (max |diff| {np.abs(cm - cmr).max():.0f} cm)')
        D = E - Er; m = D.mean(0); se = D.std(0, ddof=1) / np.sqrt(len(D)); ratio = E.std(0, ddof=1) / Er.std(0, ddof=1)
        flags = []
        for j, p in enumerate(P):
            fb = abs(m[j]) > 2 * se[j]; fs = not (lo <= ratio[j] <= hi)
            if fb or fs:
                flags.append(p + (' (shift)' if fb else '') + (' (spread)' if fs else ''))
            print(f'    {b} {p:10s} paired delta {m[j]:+7.3f} +- {se[j]:6.3f} (detectable >= {2 * se[j]:.3f}) | SD {E[:, j].std(ddof=1):.3f} '
                  f'vs ref {Er[:, j].std(ddof=1):.3f} (ratio {ratio[j]:.2f})' + ('  <- IMPACT' if (abs(m[j]) > 2 * se[j] or not (lo <= ratio[j] <= hi)) else ''))
        dp = pos - posr
        print(f'    {b} position paired delta {dp.mean():+.1f} +- {dp.std(ddof=1) / np.sqrt(len(dp)):.1f} um (descriptive)')
        print(f"  -> {b}: {'NO DETECTED IMPACT' if not flags else 'IMPACT: ' + ', '.join(flags)}")
        allok &= not flags
    print(f"T6 VERDICT: batch size {'has NO DETECTED IMPACT (both arms, all parameters)' if allok else 'HAS AN IMPACT (see flags)'}")


if __name__ == '__main__':
    w = sys.argv[1] if len(sys.argv) > 1 else 'both'
    if w in ('t5', 'both'):
        t5()
    if w in ('t6', 'both'):
        t6()
