# Bitwise agreement of every bench replica's losses_iter with KN10S1 (same config, same data; GPU non-deterministic).
import glob, pickle, numpy as np
FR = '/sdf/scratch/neutrino/pgranger/fit_result_staging'
ld = lambda arm, it: pickle.load(open(glob.glob(f'{FR}/sci_full_{arm}/history_iter{it}_*.pkl')[0], 'rb'))
LK = np.ravel(ld('KN10S1', 10000)['losses_iter']).astype(np.float32)
for arm, it, tag in [('BENCHL2K', 300, 'unpatched 2048'), ('BENCHL32K', 300, 'unpatched 32768'), ('BENCHR3A', 600, 'unpatched 32768'),
                     ('BENCHR3B', 600, 'unpatched 8192'), ('BENCHR4A', 600, 'fixes 2048'), ('BENCHR4B', 600, 'fixes 8192'),
                     ('BENCHR5P', 300, 'fixes 2048'), ('BENCHR5A', 600, 'fixes 32768'), ('BENCHR5B', 1500, 'fixes 32768')]:
    L = np.ravel(ld(arm, it)['losses_iter']).astype(np.float32)
    n = next((i for i in range(len(L)) if L[i] != LK[i]), len(L))
    ulp = lambda i: abs(float(L[i]) - float(LK[i])) / float(np.spacing(LK[i]))
    print(f'{arm:10s} ({tag:15s}): bitwise for {n} it; first diff at it {n}: rel {L[n] / LK[n] - 1:+.1e} = {ulp(n):.0f} ulp; '
          f'it 5..9 max {max(ulp(i) for i in range(5, 10)):.0f} ulp; |rel| at 50/100/200: ' + ' '.join(f'{abs(L[i] / LK[i] - 1):.0e}' for i in (50, 100, 200)))
