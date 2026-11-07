"""ENDFIT (m) readout (rules: REVIEW.md 28 Oct "ENDFIT (m)", written before any result).
  python3 read_m.py sched : idea 1 -- SK (fit @ 10000 -> fast continuation, end 16000) and FL -> FLH (stage 1 with the
                            calibration LR floored at 1e-2, then a 4000-it finish, end 14000) vs production H @ 20000
  python3 read_m.py lm    : idea 2 -- LM (low-mode epoch push, end 20000) vs H @ 20000 on EXS, EXP, KN10N
Metric: mean over s of |X - own <BASE><s>L @ 44000| (lifetime, long_diff; last 1000 iterations) and position - L."""
import os, sys
os.environ['RJ_BASE'] = 'EXP'; os.environ['RJ_ARMS'] = 'H:20000'
exec(open('/sdf/scratch/neutrino/pgranger/review_2026_09/ENDFIT/read_j.py').read().split('\nR = {}')[0])
WHAT = sys.argv[1]
HREF = {'EXS': (3.88, 3.82), 'EXP': (2.17, 2.74), 'KN10N': (1.90, 1.92)}


def table(B, arms):
    """arms: [(tag, end, [checkpoints to report])] -> {(tag, it): (lifetime[], long_diff[], pos[])}"""
    Lr = {s: win(ck(f'{B}{s}L', 44000), 44000, 44000) for s in S}
    M = {}
    print(f'\n{B}: |X - {B} L @ 44000| mean over s (lifetime | long_diff | position - L um) [per sample at the end point]')
    for tag, end, its in arms:
        D = {s: ck(f'{B}{s}{tag}', end) for s in S}
        for it in its:
            r = [win(D[s], it, end) for s in S]
            l = np.array([abs(r[k]['lifetime'] - Lr[s]['lifetime']) for k, s in enumerate(S)])
            d = np.array([abs(r[k]['long_diff'] - Lr[s]['long_diff']) for k, s in enumerate(S)])
            p = np.array([r[k]['pos'] - Lr[s]['pos'] for k, s in enumerate(S)])
            M[tag, it] = (l, d, p)
            extra = ''
            if it == end:
                extra = ('   [' + ' / '.join(f'{x:.2f}' for x in l) + ' | ' + ' / '.join(f'{x:.2f}' for x in d) + ' | '
                         + ' / '.join(f'{x:+.1f}' for x in p) + ']')
            print(f'  {tag:4s} @{it}: {l.mean():5.2f} | {d.mean():5.2f} | {p.mean():+6.1f}' + extra)
    return M


if WHAT == 'sched':
    ok = {}
    for B in ('EXP', 'KN10N'):
        M = table(B, [('H', 20000, (16000, 18000, 20000)), ('SK', 16000, (12000, 14000, 16000)), ('FLH', 14000, (12000, 14000))])
        hl, hd = HREF[B]; hp = M['H', 20000][2].mean()
        for tag, end in (('SK', 16000), ('FLH', 14000)):
            l, d, p = (x.mean() for x in M[tag, end])
            v = l <= max(2.5, 1.25 * hl) and d <= max(3, 1.25 * hd) and p <= hp + 5
            ok[B, tag] = v
            print(f'  VERDICT {B} {tag} @ {end}: {l:.2f} / {d:.2f} vs {max(2.5, 1.25 * hl):.2f} / {max(3, 1.25 * hd):.2f}, '
                  f'position - L {p:+.1f} vs {hp + 5:+.1f} -> {"EQUIVALENT" if v else "WORSE"}')
    for tag, end in (('SK', 16000), ('FLH', 14000)):
        print(f'{tag}: recipe candidate (end {end} instead of 20000): {"YES" if ok["EXP", tag] and ok["KN10N", tag] else "NO"}')
elif WHAT == 'lm':
    safe, closer = {}, []
    for B in ('EXS', 'EXP', 'KN10N'):
        M = table(B, [('H', 20000, (17000, 18000, 19000, 20000)), ('LM', 20000, (17000, 18000, 19000, 20000))])
        hl, hd = HREF[B]; l, d, p = M['LM', 20000]
        if B == 'EXS':
            print(f'  EXS diagnostic: LM {l.mean():.2f} / {d.mean():.2f} vs 0.8 x H {0.8 * hl:.2f} / {0.8 * hd:.2f} -> '
                  f'{"HELPS" if l.mean() <= 0.8 * hl and d.mean() <= 0.8 * hd else "NO GAIN"}')
            continue
        safe[B] = l.mean() <= 1.25 * hl and d.mean() <= 1.25 * hd
        closer += list(p < M['H', 20000][2])
        print(f'  VERDICT {B}: LM {l.mean():.2f} / {d.mean():.2f} vs 1.25 x H {1.25 * hl:.2f} / {1.25 * hd:.2f} -> {"SAFE" if safe[B] else "HARMFUL"}')
    for B in ('EXS', 'EXP', 'KN10N'):
        for s in S:
            ea = ck(f'{B}{s}LM', 20000).get('epoch_accel', [])
            pu = [r for r in ea if 'geom' in r]; g = [r['geom']['rms_um'] for r in pu]
            print(f'  {B}{s}LM pushes {len(pu)}, restarts {sum(1 for r in ea if r.get("restart"))}, pushed-mode rms median {np.median(g) if g else float("nan"):.1f} um')
    n = int(np.sum(closer))
    print(f'\npositions closer to L than H: {n} of 6 -> ADOPTABLE: {"YES" if all(safe.values()) and n >= 5 else "NO"}')
