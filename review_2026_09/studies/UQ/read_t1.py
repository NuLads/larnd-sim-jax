"""UQ TEST T1 (R1: is E1's Newton offset a quick convergence check?). Rules: REVIEW.md "UQ TEST of the recommendations".
18 calib-only E1 dumps T1_EXP<s>_{10k,14k,20k} of the training pipelines EXP<s>C75 (s = 1..6) at 10000 / 14000 / 20000.
FLAGGED: max_p |z_p| > 3. MOVES: some |final_p - e_t,p| > SD_train,p (final = EXP<s>C75H last-1000 mean, e_t = the dumped
point's error). usage (container): python3 read_t1.py   [T1_DRY=1: every dump replaced by pilot dump C3, to test the script]"""
import os, numpy as np
from uqt_common import P, SD, end, e1
DRY = os.environ.get('T1_DRY') == '1'
rows = []
for s in range(1, 7):
    fin = end(f'EXP{s}C75H', check=s)
    for t in ('10k', '14k', '20k'):
        x = e1('C3' if DRY else f'T1_EXP{s}_{t}', fin['truth'])
        if not DRY:
            assert x['calib_only'] and x['it'] == {'10k': 10100, '14k': 14100, '20k': 20100}[t], (s, t, x['it'])
        M = fin['e'] - x['e']; mv = bool(np.any(np.abs(M) > SD)); fl = bool(np.max(np.abs(x['z'])) > 3)
        rows.append(dict(s=s, t=t, x=x, M=M, mv=mv, fl=fl))
print('T1' + (' -- DRY RUN (C3 stands in for every dump)' if DRY else '') + ': config check OK (seed s, coarse 7500, smear 1.0 / entry)')
print(' s   t    | error at t (%)                         | later move M = final - e_t (%)          | E1 z                            | max|z| flag move')
for r in rows:
    x = r['x']
    print(f" {r['s']} {r['t']:4s} | " + ' '.join(f'{v:+7.2f}' for v in x['e']) + ' | ' + ' '.join(f'{v:+7.2f}' for v in r['M'])
          + ' | ' + ' '.join(f'{v:+5.1f}' for v in x['z']) + f" | {np.max(np.abs(x['z'])):5.1f} {'F' if r['fl'] else '.'} {'M' if r['mv'] else '.'}")
sig = np.array([r['x']['sig'] for r in rows])
print('E1 sigma (%) median over the 18 dumps: ' + ' '.join(f'{p} {v:.3g}' for p, v in zip(P, np.median(sig, 0)))
      + ' | at 20000: ' + ' '.join(f'{v:.3g}' for v in np.median([r['x']['sig'] for r in rows if r['t'] == '20k'], 0)))
mov = [r for r in rows if r['t'] != '20k' and r['mv']]; ends = [r for r in rows if r['t'] == '20k']
fm = np.mean([r['fl'] for r in mov]) if mov else np.nan; fe = np.mean([r['fl'] for r in ends])
print(f'moving 10000/14000 checkpoints: {len(mov)} of 12, flagged {sum(r["fl"] for r in mov)} ({fm:.2f}); '
      f'20000 end points flagged {sum(r["fl"] for r in ends)} of 6 ({fe:.2f})')
# descriptive: does the Newton step point the way the fit later moves (parameters that move > SD_train)?
ag = [(np.sign(r['x']['d'][j]) == np.sign(r['M'][j]), abs(r['x']['d'][j]) / abs(r['M'][j])) for r in rows if r['t'] != '20k'
      for j in range(5) if abs(r['M'][j]) > SD[j]]
if ag:
    print(f'descriptive: over {len(ag)} (checkpoint, parameter) moves > SD_train, the E1 Newton step has the move\'s sign in '
          f'{np.mean([a for a, _ in ag]):.2f}; median |step| / |move| {np.median([b for _, b in ag]):.3f}')
if len(mov) < 3:
    v = 'UNDETERMINED (< 3 moving checkpoints)'
elif fm >= 2 / 3 and fe <= 1 / 6:
    v = 'HAS POWER'
elif fm <= 1 / 3:
    v = 'VACUOUS'
else:
    v = 'PARTIAL'
print(f'T1 VERDICT: the E1 quick check {v}' + ('  -- DRY RUN' if DRY else ''))
