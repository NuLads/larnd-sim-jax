"""KN10N1 pipeline rule (REVIEW.md 24 Oct): KN10N1FR reaches 14000 and its 13 000-13 999 mean pos_residual is within
+-8 % of PRF1's -> PIPELINE OK. usage (in the container): python3 read_n1.py"""
import sys
sys.argv = [sys.argv[0]]; exec(open('/sdf/scratch/neutrino/pgranger/review_2026_09/POLISH/joint/read_pr.py').read().split('\nfor s in sys.argv')[0])
n, f = load('KN10N1FR'), load('PRF1')
for k, x in (('KN10N1FR', n), ('PRF1', f)):
    print(f'{k}: n {x["n"]}  pos_residual 9-10k {1e4 * x["pr_9k"]:.1f} um, 13-14k {1e4 * x["pr_end"]:.1f} | end error % ' + ' '.join(f'{p} {x["err"][p]:+.2f}' for p in P))
q = n['pr_end'] / f['pr_end']
print(f'KN10N1FR / PRF1 pos_residual 13-14k = {q:.3f} -> ' + ('PIPELINE OK' if n['n'] >= 14000 and abs(q - 1) <= 0.08 else 'PIPELINE CHECK FAILED'))
