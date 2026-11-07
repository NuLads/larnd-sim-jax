"""TRUTHC2 readout (5 Nov; rules: REVIEW.md "TRUTHC2"). Truth-point runs sci_full_TC2_KN10N<s> (s = 1-8, the TRUTHC end
points) with LARND_TP_DEDX_SPLIT=1: calibration Hessian + gradient (step 1e-3) at P0 fit, P3 dE/dx at truth, P5 = fitted
dE/dx shape at the TRUE per-track level, P6 = TRUE shape at the fitted level. Delta_k = m(P0) - m(Pk), paired over the 8.
usage (container): python3 read_tc2.py [--dry <TAG>]"""
import sys, numpy as np
sys.path.insert(0, '/sdf/scratch/neutrino/pgranger/review_2026_09/TRUTHC')
from read_tc import P, point, load
NAMES = ('P0_fit', 'P3_dedx_true', 'P5_dedx_level_true', 'P6_dedx_shape_true')


def run(tag, names=NAMES):
    recs, th = load(tag)
    truth = np.array([float(np.ravel(th[f'{p}_target'])[0]) for p in P])
    return {n: point(recs, n, truth, '1e-3') for n in names}, recs


def split_stats(recs):
    s = [r['dedx_split'] for r in recs if 'dedx_split' in r]
    w = np.array([x['n_tracks'] for x in s], float)
    return dict(cov=sum(x['n_seg_covered'] for x in s) / sum(x['n_seg'] for x in s),
                lev_mean=np.average([x['level_err_mean'] for x in s], weights=w),
                lev_sd=np.sqrt(np.average(np.array([x['level_err_sd'] for x in s]) ** 2, weights=w)),
                disp=np.average([x['disp_ratio'] for x in s], weights=w))


if __name__ == '__main__':
    if '--dry' in sys.argv:
        tag = sys.argv[sys.argv.index('--dry') + 1]
        R, recs = run(tag, [n for n in NAMES if n != 'P3_dedx_true'] if 'SMOKE' in tag else NAMES)
        print(f'DRY RUN on {tag}: {len(recs)} batch record(s); split: {split_stats(recs)}')
        for n, x in R.items():
            print(f'  {n:20s}', 'absent (no Hessian)' if x is None else 'e ' + ' '.join(f'{a:+.3f}' for a in x['e']) + ' | m ' + ' '.join(f'{a:+.3f}' for a in x['m']))
        sys.exit(0)
    S = range(1, 9)
    R2 = {s: run(f'TC2_KN10N{s}') for s in S}      # read_tc.load: sci_full_TC_<tag> absent -> sci_full_TC2_KN10N<s>
    from read_tc import run as run1
    R1 = {s: run1(f'KN10N{s}', ('P0_fit', 'P3_dedx_true'))[0] for s in S}
    # V0 reproducibility: TC2 P0 and P3 against TRUTHC (same code path, data, point, step)
    ok = [all(np.all(np.abs(R2[s][0][n]['m'] - R1[s][n]['1e-3']['m']) < 0.05 * R1[s][n]['1e-3']['sig']) for n in ('P0_fit', 'P3_dedx_true')) for s in S]
    v0 = sum(ok) >= 7
    print(f'V0 reproducibility (TC2 P0, P3 vs TRUTHC within 0.05 sigma on all 5, >= 7/8 samples): {sum(ok)}/8 -> {"PASS" if v0 else "FAIL"}')
    M = {n: np.array([R2[s][0][n]['m'] for s in S]) for n in NAMES}
    st = [split_stats(R2[s][1]) for s in S]
    print('  dE/dx split (mean over samples): segments covered %.3f; per-track level error (log) mean %+.4f, SD %.4f; shape dispersion fit / truth %.3f'
          % tuple(np.mean([x[k] for x in st]) for k in ('cov', 'lev_mean', 'lev_sd', 'disp')))
    sem = lambda X: X.std(0, ddof=1) / np.sqrt(len(X))
    print('  m (mean +- SEM, %):   ' + ' '.join(f'{p:>15s}' for p in P))
    for n in NAMES:
        print(f'  {n:20s} ' + ' '.join(f'{a:+7.2f}+-{b:5.2f}' for a, b in zip(M[n].mean(0), sem(M[n]))))
    D = {k: M['P0_fit'] - M[n] for k, n in ((3, 'P3_dedx_true'), (5, 'P5_dedx_level_true'), (6, 'P6_dedx_shape_true'))}
    D['3-5'] = D[3] - D[5]; D['3-6'] = D[3] - D[6]; D['int'] = D[3] - D[5] - D[6]
    for k, lab in ((3, 'Delta3 = m0 - m3 (all dE/dx)'), (5, 'Delta5 = m0 - m5 (level)'), (6, 'Delta6 = m0 - m6 (shape)'),
                   ('int', 'interaction D3 - D5 - D6')):
        print(f'  {lab:30s} ' + ' '.join(f'{a:+7.2f}+-{b:5.2f}' for a, b in zip(D[k].mean(0), sem(D[k]))))
    print('  shares of Delta3: level ' + ' '.join(f'{p} {a:.2f}' for p, a in zip(P, D[5].mean(0) / D[3].mean(0)))
          + ' | shape ' + ' '.join(f'{p} {a:.2f}' for p, a in zip(P, D[6].mean(0) / D[3].mean(0))))
    j = P.index('long_diff'); sig = lambda k: abs(D[k].mean(0)[j]) > 2 * sem(D[k])[j]
    if sig(5) and D[5].mean(0)[j] > 0 and not sig('3-5') and not sig(6): c = 'LEVEL'
    elif sig(6) and D[6].mean(0)[j] > 0 and not sig('3-6') and not sig(5): c = 'SHAPE'
    else: c = 'BOTH'
    print(f'TRUTHC2 VERDICT (long_diff): {c if v0 else "NO VERDICT (V0 failed)"}')
