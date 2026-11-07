"""MW readout (rule: REVIEW.md 24 Oct, stated before any result). Converged joint minimum (truth geometry, sample 1) vs the
dE/dx mean-constraint weight: mwA2500 (same-tree baseline L0), mwA250 (x0.1), mwA25000 (x10). usage: python3 read_mw.py"""
import json, os
D = os.path.dirname(os.path.abspath(__file__)); P = ['Ab', 'eField', 'tran_diff', 'long_diff', 'lifetime']
def summ(f):
    r = json.load(open(f)); st = [s for s in r['steps'] if s.get('accepted')]
    mv = (st[-1]['err_after']['lifetime'] - st[-1]['err_before']['lifetime']) if st else float('nan')
    return dict(mw=r.get('mw'), err=r['err_final'], nst=len(r['steps']), nacc=len(st), last_move=mv, conv=abs(mv) < 0.3)
R = {w: summ(f'{D}/mwA{w}_KN10S1.json') for w in (250, 2500, 25000)}
for w, x in R.items():
    e = x['err']
    print(f'mwA{w:<5d} (mw {x["mw"]:g})  lifetime {e["lifetime"]:+6.2f}  ' + ' '.join(f'{p} {e[p]:+6.2f}' for p in P[:4]) +
          f'  | steps {x["nst"]} acc {x["nacc"]} last move {x["last_move"]:+.2f} {"CONVERGED" if x["conv"] else "NOT CONVERGED (indicative)"}')
L = {w: R[w]['err']['lifetime'] for w in R}; L0 = L[2500]
print(f'tree check: L0 {L0:+.2f} vs TG2 -5.36 -> {"PASS" if abs(L0 + 5.36) <= 1.0 else "FAIL"}')
d = {w: L[w] - L0 for w in (250, 25000)}; mono = (L[250] - L0) * (L0 - L[25000]) >= 0
v = ('THE MEAN CONSTRAINT SETS THE OFFSET' if max(abs(t) for t in d.values()) >= 1.5 and mono else
     'NOT THE MEAN CONSTRAINT' if max(abs(t) for t in d.values()) <= 1.0 else 'PARTIAL')
print(f'L(250) - L0 {d[250]:+.2f}, L(25000) - L0 {d[25000]:+.2f}, monotone {mono} -> {v}')
