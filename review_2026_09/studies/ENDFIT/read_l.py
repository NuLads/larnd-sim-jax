"""ENDFIT (l) readout (rule: REVIEW.md 28 Oct "ENDFIT (l)", written before any result): epoch acceleration (EA, beta 0.8,
14600-19000) on the realistic smear (EXP) and true ends (KN10N): mean over s of |X - own <BASE><s>L @ 44000| for the
fast continuation H and EA at 17000-20000 (window = last 1000 iterations), position - L, last-epoch loss, verdicts.
usage (container, PYTHONPATH = live tree): python3 read_l.py"""
import os
os.environ['RJ_BASE'] = 'EXP'; os.environ['RJ_ARMS'] = 'H:20000'
exec(open('/sdf/scratch/neutrino/pgranger/review_2026_09/ENDFIT/read_j.py').read().split('\nR = {}')[0])
LIM = {'EXP': (2.17, 2.74), 'KN10N': (1.90, 1.92)}
safe, closer = {}, []
for B in ('EXP', 'KN10N'):
    Lr = {s: win(ck(f'{B}{s}L', 44000), 44000, 44000) for s in S}
    D = {(s, a): ck(f'{B}{s}{a}', 20000) for s in S for a in ('H', 'EA')}
    print(f'\n{B}: mean over s of |X(it) - {B} L @ 44000|, lifetime | long_diff | position - L (um); per sample at 20000 in brackets')
    M = {}
    for it in (17000, 18000, 19000, 20000):
        row = f'  {it}: '
        for a in ('H', 'EA'):
            r = [win(D[s, a], it, 20000) for s in S]
            l = np.array([abs(r[k]['lifetime'] - Lr[s]['lifetime']) for k, s in enumerate(S)])
            d = np.array([abs(r[k]['long_diff'] - Lr[s]['long_diff']) for k, s in enumerate(S)])
            p = np.array([r[k]['pos'] - Lr[s]['pos'] for k, s in enumerate(S)])
            M[it, a] = (l, d, p)
            row += f'{a:2s} {l.mean():5.2f} | {d.mean():5.2f} | {p.mean():+6.1f}    '
        print(row)
    for a in ('H', 'EA'):
        l, d, p = M[20000, a]
        lo = [np.array(D[s, a]['losses_iter'], float)[-100:].mean() for s in S]
        print(f'  {a:2s} @20000 lifetime [' + ' / '.join(f'{x:.2f}' for x in l) + '] long_diff [' + ' / '.join(f'{x:.2f}' for x in d)
              + '] pos - L [' + ' / '.join(f'{x:+.1f}' for x in p) + '] last-epoch loss [' + ' / '.join(f'{x:.1f}' for x in lo) + ']')
    hl, hd = LIM[B]; l, d, _ = M[20000, 'EA']
    safe[B] = l.mean() <= 1.25 * hl and d.mean() <= 1.25 * hd
    closer += list(M[20000, 'EA'][2] < M[20000, 'H'][2])
    print(f'  VERDICT {B}: EA @ 20000 {l.mean():.2f} / {d.mean():.2f} vs 1.25 x H ({1.25 * hl:.2f} / {1.25 * hd:.2f}) -> {"SAFE" if safe[B] else "HARMFUL"}')
    if B == 'EXP':
        l8, d8, _ = M[18000, 'EA']
        print(f'  EXP EARLIER: EA @ 18000 {l8.mean():.2f} / {d8.mean():.2f} vs SUFFICIENT 2.5 / 3 -> {"YES" if l8.mean() <= 2.5 and d8.mean() <= 3 else "NO"}')
n = int(np.sum(closer))
print(f'\npositions closer to L than H: {n} of 6 -> recipe candidate: {"YES" if all(safe.values()) and n >= 5 else "NO"}')
