# Is the unpatched code bitwise-reproducible against KN10S1? (round-2 benches: unpatched code, same config, done)
import glob, pickle, numpy as np
FR = '/sdf/scratch/neutrino/pgranger/fit_result_staging'
ld = lambda arm, it: pickle.load(open(glob.glob(f'{FR}/sci_full_{arm}/history_iter{it}_*.pkl')[0], 'rb'))
K1 = ld('KN10S1', 10000)
for arm, it in [('BENCHL2K', 300), ('BENCHL32K', 300), ('BENCHR4A', 600), ('BENCHR4B', 600)]:
    H = ld(arm, it); P, Q = np.ravel(np.asarray(H['pos_residual_iter'], float)), np.ravel(np.asarray(K1['pos_residual_iter'], float))
    L, LK = np.ravel(H['losses_iter']), np.ravel(K1['losses_iter'])
    nl = next((i for i in range(len(L)) if L[i] != LK[i]), len(L)); npr = next((i for i in range(len(P)) if P[i] != Q[i]), len(P))
    print(f'{arm}: losses bitwise = KN10S1 for first {nl} it; pos_residual for first {npr} it; pos_residual[0] rel diff {P[0] / Q[0] - 1:+.2e}; '
          f'pos_residual[1..3] rel ' + ' '.join(f'{P[i] / Q[i] - 1:+.1e}' for i in (1, 2, 3)))
