"""ENDFIT (n) readout (rules: REVIEW.md 28 Oct "ENDFIT (n)", written before any result).
  python3 read_n.py : ideas 5 (B2 = per-track geometry Adam b2 0.95) and 3 (CF = coarse-to-fine geometry, lowest half of
                      the sine modes until 5000), each a full production pipeline <BASE><s><ARM> -> FR -> H (end
                      sci_full_<BASE><s><ARM>H @ 20000), vs production <BASE><s>H, on EXP and KN10N, s = 1, 2, 4.
Metric: mean over s of |X - own <BASE><s>L @ 44000| (lifetime, long_diff; last 1000 iterations) and position - L."""
import os, sys
os.environ['RJ_BASE'] = 'EXP'; os.environ['RJ_ARMS'] = 'H:20000'
exec(open('/sdf/scratch/neutrino/pgranger/review_2026_09/ENDFIT/read_j.py').read().split('\nR = {}')[0])
exec(open('/sdf/scratch/neutrino/pgranger/review_2026_09/ENDFIT/read_m.py').read().split('\nif WHAT ==')[0].split("WHAT = sys.argv[1]")[1])
ITS = (16000, 17000, 18000, 19000, 20000)


def equiv(M, tag, it, hl, hd, hp):
    l, d, p = (x.mean() for x in M[tag, it])
    return l <= max(2.5, 1.25 * hl) and d <= max(3, 1.25 * hd) and p <= hp + 5, (l, d, p)


res, closer = {}, {}
for B in ('EXP', 'KN10N'):
    M = table(B, [('H', 20000, ITS), ('B2H', 20000, ITS), ('CFH', 20000, ITS)])
    hl, hd = HREF[B]; hp = M['H', 20000][2].mean()
    print(f'  thresholds vs H @ 20000: lifetime <= {max(2.5, 1.25 * hl):.2f}, long_diff <= {max(3, 1.25 * hd):.2f}, '
          f'position - L <= {hp + 5:+.1f} um')
    for tag in ('H', 'B2H', 'CFH'):
        row = []
        for it in (17000, 18000, 20000):
            v, (l, d, p) = equiv(M, tag, it, hl, hd, hp); res[B, tag, it] = v
            row.append(f'@{it} {"EQUIV" if v else "worse"} ({l:.2f}/{d:.2f}/{p:+.1f})')
        if tag != 'H':
            closer[B, tag] = M[tag, 20000][2] < M['H', 20000][2]
        print(f'  {tag:4s} ' + '  '.join(row) + ('' if tag == 'H' else f'  closer to L than H @ 20000: {int(closer[B, tag].sum())}/3'))
print()
for tag, name in (('B2H', 'idea 5 (b2 0.95)'), ('CFH', 'idea 3 (coarse-to-fine)')):
    end = next((it for it in (17000, 18000) if res['EXP', tag, it] and res['KN10N', tag, it]), None)
    safe = res['EXP', tag, 20000] and res['KN10N', tag, 20000]
    n = int(closer['EXP', tag].sum() + closer['KN10N', tag].sum())
    print(f'{name}: SAFE @ 20000 {"YES" if safe else "NO"}; FASTER (EQUIVALENT on both bases at): {end or "no"}; '
          f'positions closer than H on {n}/6 -> BETTER {"YES" if safe and n >= 5 else "NO"}; '
          f'RECIPE CANDIDATE {"YES (end " + str(end) + ")" if end else "NO"}')
print(f'control: H itself EQUIVALENT at 17000 / 18000 on both bases: {res["EXP", "H", 17000] and res["KN10N", "H", 17000]} / '
      f'{res["EXP", "H", 18000] and res["KN10N", "H", 18000]} (expected False: positions still converging)')
