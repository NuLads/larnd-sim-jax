"""KN10V vs KN10S on the same data: first iteration where the loss differs; per 1000-iteration window the paired loss
difference (same batch each iteration) and the calibration-error differences (%), to tell a systematic shift (same sign
from early on) from chaotic divergence (difference grows, then wanders in sign)."""
import glob, pickle, numpy as np
FR = '/sdf/scratch/neutrino/pgranger/fit_result_staging'
P = ['Ab', 'eField', 'tran_diff', 'long_diff', 'lifetime']
ld = lambda arm: pickle.load(open(sorted(glob.glob(f'{FR}/sci_full_{arm}/history_iter10000_*.pkl'))[0], 'rb'))
for s in (1, 2, 3):
    V, S = ld(f'KN10V{s}'), ld(f'KN10S{s}')
    lv, ls = np.ravel(V['losses_iter']).astype(float), np.ravel(S['losses_iter']).astype(float)
    ne = np.nonzero(lv != ls)[0]
    print(f'\ns{s}: losses equal for the first {ne[0] if len(ne) else len(lv)} iterations; |dloss|/loss at it 10/100/1000: '
          + ' '.join(f'{abs(lv[i] - ls[i]) / ls[i]:.1e}' for i in (10, 100, 1000)))
    tr = {p: (100 * (np.ravel(V[p + '_iter']) / np.ravel(V[p + '_target'])[0] - 1),
              100 * (np.ravel(S[p + '_iter']) / np.ravel(S[p + '_target'])[0] - 1)) for p in P}
    print('  window   dloss(V-S)  sd_within(S)  ' + '  '.join(f'{p:>10s}' for p in P) + '   (V-S, % of truth; window means)')
    for w in range(10):
        a, b = w * 1000, (w + 1) * 1000
        dl = (lv[a:b] - ls[a:b]); sdw = ls[a:b].std()
        print(f'  {a:5d}-{b:5d} {dl.mean():+9.2f}+-{dl.std() / np.sqrt(b - a):5.2f} {sdw:8.1f}     '
              + '  '.join(f'{tr[p][0][a:b].mean() - tr[p][1][a:b].mean():+10.3f}' for p in P))
    print('  within-run SD of the last-1000 trace (S): ' + '  '.join(f'{p} {tr[p][1][-1000:].std():.3f}' for p in P))
