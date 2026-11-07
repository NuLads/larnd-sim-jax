"""Readout of the STRFRZ50 replication against the rule stated before results (6 Oct ~11:30)."""
import sys, numpy as np
sys.path.insert(0, '/sdf/scratch/neutrino/pgranger/review_2026_09/GEOFRZ')
import compare_arms as ca, drift_residuals as dr
P = ['Ab', 'eField', 'tran_diff', 'long_diff', 'lifetime']
def show(arm, ss, ref=None):
    rows = {s: ca.read(arm, s) for s in ss}
    refs = {s: ca.read(ref, s) for s in ss} if ref else {}
    for s in ss:
        r = rows[s]; tr, _ = dr.run_residuals(ca.tagdir(arm, s).split('/')[-1]); off = dr.summarise(tr)['overall_mean']
        line = f'  {arm}{s}: ' + ' '.join(f'{p} {r[p]:+6.2f}' for p in P) + f'  pos {r["pos_um"]:4.0f}um  offset {off:+5.1f}um'
        if ref: line += f'   | {ref}: lifetime {refs[s]["lifetime"]:+6.2f} long_diff {refs[s]["long_diff"]:+6.2f}'
        print(line)
    return rows
print('=== (a) fresh samples 5/6/9, tau = 2406 us (ref: production DSREP)')
A = show('STRFRZ50', [5, 6, 9], 'DSREP')
print('=== earlier samples 1/2/4 (ref: production DSREP)')
B = show('STRFRZ50', [1, 2, 4], 'DSREP')
lt_new = np.array([A[s]['lifetime'] for s in (5, 6, 9)]); ld_new = np.array([A[s]['long_diff'] for s in (5, 6, 9)])
allr = {**A, **B}
print('\n=== pooled n=6 (samples 1,2,4,5,6,9), mean +- SE (SD):')
for p in P:
    v = np.array([allr[s][p] for s in (1, 2, 4, 5, 6, 9)])
    print(f'  {p:10s} {v.mean():+7.3f} +- {v.std(ddof=1)/np.sqrt(6):.3f}  (SD {v.std(ddof=1):.3f})  [{", ".join(f"{x:+.2f}" for x in v)}]')
lt6 = np.array([allr[s]['lifetime'] for s in (1, 2, 4, 5, 6, 9)])
print('\n=== (b) tau = 1200 us, samples 1/3/7 (ref: production CURVE1200, frozen-truth K6)')
C = show('STRFRZ50T1200D', [1, 3, 7], 'CURVE1200D')
K = {s: ca.read('DRIFTK6T1200D', s) for s in (1, 3, 7)}
print('  frozen-truth K6 @1200 lifetime: ' + ', '.join(f'{K[s]["lifetime"]:+.2f}' for s in (1, 3, 7)))
lt1200 = np.array([C[s]['lifetime'] for s in (1, 3, 7)])
print('\n=== RULE (stated before results)')
c1 = np.all(np.abs(lt_new) <= 5); c2 = np.all(np.abs(ld_new) <= 8); c3 = abs(lt6.mean()) <= 3
print(f'  5/6/9 |lifetime| <= 5% on 3/3: {c1}  ({", ".join(f"{x:+.2f}" for x in lt_new)})')
print(f'  5/6/9 |long_diff| <= 8% on 3/3: {c2}  ({", ".join(f"{x:+.2f}" for x in ld_new)})')
print(f'  pooled n=6 lifetime mean within +-3%: {c3}  ({lt6.mean():+.2f} +- {lt6.std(ddof=1)/np.sqrt(6):.2f})')
print(f'  ==> REPLICATES: {c1 and c2 and c3}')
print(f'  robustness tau=1200 |lifetime| <= 5% on 3/3: {np.all(np.abs(lt1200) <= 5)}  ({", ".join(f"{x:+.2f}" for x in lt1200)})')
