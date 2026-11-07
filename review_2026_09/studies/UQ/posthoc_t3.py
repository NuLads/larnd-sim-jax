"""UQ TEST T3, POST HOC (3 Nov, after the T3 verdict): training vs test values side by side, the pooled n = 24 estimate,
and a check for structure (chain position, nodes) in the test long_diff values. Descriptive only."""
import numpy as np
from uqt_common import P, end
tr = np.array([end(f'EXP{s}C75H', check=s)['e'] for s in range(1, 13)])
te = np.array([end(f'EXP{s}C75H', check=s)['e'] for s in range(17, 29)])
for name, E in (('train 1-12', tr), ('test 17-28', te)):
    print(f'{name}: ' + ' | '.join(f'{p} ' + ' '.join(f'{v:+.1f}' if p not in ('Ab', 'eField') else f'{v:+.3f}' for v in E[:, j]) for j, p in enumerate(P) if p in ('long_diff', 'lifetime', 'eField')))
A = np.vstack([tr, te])
print('pooled n=24: mean ' + ' '.join(f'{v:+.3f}' for v in A.mean(0)) + ' | SD ' + ' '.join(f'{v:.3f}' for v in A.std(0, ddof=1)))
print('test first-in-chain (17-22) vs second (23-28) long_diff mean: %+.2f vs %+.2f' % (te[:6, 3].mean(), te[6:, 3].mean()))
print('corr(long_diff, lifetime): train %+.2f test %+.2f' % (np.corrcoef(tr[:, 3], tr[:, 4])[0, 1], np.corrcoef(te[:, 3], te[:, 4])[0, 1]))
