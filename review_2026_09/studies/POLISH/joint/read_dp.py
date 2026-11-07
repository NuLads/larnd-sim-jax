"""DP readout (rule: REVIEW.md 23 Oct, stated before any result). Converged joint minimum (truth geometry) with the
drift-profile penalty. usage: python3 read_dp.py   (reads every dp*_KN10S<s>.json present + the DPW 0 baselines)"""
import json, glob, os
D = os.path.dirname(os.path.abspath(__file__)); P = ['Ab', 'eField', 'tran_diff', 'long_diff', 'lifetime']
def summ(f):
    r = json.load(open(f)); st = [s for s in r['steps'] if s.get('accepted')]
    mv = (st[-1]['err_after']['lifetime'] - st[-1]['err_before']['lifetime']) if st else float('nan')
    return dict(dpw=r.get('dpw', 0.0), err=r['err_final'], nst=len(r['steps']), nacc=len(st), last_move=mv, conv=abs(mv) < 0.3)
base = {1: f'{D}/tg2_KN10S1.json', 2: f'{D}/dp0_KN10S2.json', 3: f'{D}/dp0_KN10S3.json'}
rows = {}
for f in sorted(glob.glob(f'{D}/dp[AB]*_KN10S[0-9].json')) + [v for v in base.values() if os.path.exists(v)]:
    if 'smoke' in f: continue
    s = int(f.split('KN10S')[1][0]); x = summ(f); rows.setdefault(s, []).append((os.path.basename(f), x))
for s in sorted(rows):
    b = summ(base[s]) if os.path.exists(base[s]) else None
    for n, x in sorted(rows[s], key=lambda t: t[1]['dpw']):
        e = x['err']; worse = [p for p in P if p != 'lifetime' and b and abs(e[p]) - abs(b['err'][p]) > 2]
        print(f's{s} {n:22s} DPW {x["dpw"]:8.0e}  lifetime {e["lifetime"]:+6.2f}  ' + ' '.join(f'{p} {e[p]:+6.2f}' for p in P[:4]) +
              f'  | steps {x["nst"]} acc {x["nacc"]} last move {x["last_move"]:+.2f} {"CONVERGED" if x["conv"] else "NOT CONVERGED"}'
              f'  | within 1.2: {abs(e["lifetime"]) <= 1.2}  others >2 pts worse vs DPW0: {worse or "none"}')
