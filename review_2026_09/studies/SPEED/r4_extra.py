import glob, pickle, numpy as np
FR = '/sdf/scratch/neutrino/pgranger/fit_result_staging'
ld = lambda arm, it: pickle.load(open(glob.glob(f'{FR}/sci_full_{arm}/history_iter{it}_*.pkl')[0], 'rb'))
K1 = ld('KN10S1', 10000)
med = [np.median(np.ravel(ld(f'KN10S{s}', 10000)['step_time']).astype(float)[300:600]) for s in range(1, 17)]
print('KN10S per-sample median 300-600: ' + ' '.join(f'{m:.2f}' for m in med), f'| min {min(med):.2f} max {max(med):.2f} SD {np.std(med, ddof=1):.2f}')
for arm in ['BENCHR4A', 'BENCHR4B']:
    H = ld(arm, 600)
    p, q = H['pos_residual_iter'][0], K1['pos_residual_iter'][0]
    p, q = np.asarray(p, dtype=object), np.asarray(q, dtype=object)
    try:
        pa, qa = np.asarray(H['pos_residual_iter'][0], float), np.asarray(K1['pos_residual_iter'][0], float)
        print(arm, 'pos_residual_iter[0]', pa.shape, qa.shape, 'bitwise equal' if pa.shape == qa.shape and np.array_equal(pa, qa) else f'DIFFER max rel {np.max(np.abs(pa / qa - 1)):.2e}', pa.ravel()[:3], qa.ravel()[:3])
    except Exception as e:
        print(arm, 'pos_residual_iter[0] type', type(H['pos_residual_iter'][0]), repr(e))
    L, LK = np.ravel(H['losses_iter']), np.ravel(K1['losses_iter'])
    n_eq = next((i for i in range(len(L)) if L[i] != LK[i]), len(L))
    print(f'  losses equal to KN10S1 for the first {n_eq} iterations; |rel diff| at 100/300/599: ' + ' '.join(f'{abs(L[i] / LK[i] - 1):.1e}' for i in (100, 300, 599)))
    for k in ['lifetime', 'long_diff', 'Ab']:
        a, b = np.ravel(H.get(k + '_iter', [np.nan]))[:600], np.ravel(K1.get(k + '_iter', [np.nan]))[:600]
        if len(a) == 600: print(f'  {k}_iter at 599: bench {a[599]:.6g} vs KN10S1 {b[599]:.6g} (rel {a[599] / b[599] - 1:+.2e})')
