"""POST HOC (6 Nov, after the DXPROF readout: V1 FAILED -> no verdict). Where does the step / ridge dependence sit?
Per (step, ridge): median widths, SD of m_prof, and the size of the dE/dx Newton step D^-1 g_d."""
import sys, numpy as np
sys.path.insert(0, '/sdf/scratch/neutrino/pgranger/review_2026_09/DXPROF')
from read_dx import load, profile, P, fmt
S = range(1, 9); R = {s: load(f'DX_KN10N{s}') for s in S}
for st in ('0.003', '0.01'):
    for lm in ('0', '0.001', '0.01'):
        fr = {s: profile(*R[s], st, lm) for s in S}
        M = np.array([fr[s]['m'] for s in S])
        sw = np.median([fr[s]['sig_sw'] for s in S], 0); sh = np.median([fr[s]['sig'] for s in S], 0)
        dmax = [np.median([x['schur']['res'][f'{st}_{lm}']['dinv_gd_max'] for x in R[s][0]
                           if x.get('schur') and x['schur']['res'].get(f'{st}_{lm}', {}).get('ok')]) for s in S]
        print(f'step {st:5s} ridge {lm:5s} | prof Hess {fmt(sh, "{:7.3f}")} | prof sandw {fmt(sw, "{:7.3f}")} | SD m_prof {fmt(M.std(0, ddof=1), "{:7.3f}")}'
              f' | median max|D^-1 g_d| {np.median(dmax):.3g}')
# per-batch: the profiled-score correction vs the raw score, step 3e-3 vs 1e-2 at ridge 0, sample 1
recs = R[1][0]
a = []; b = []; g = []
for x in recs:
    s = x.get('schur')
    if not s: continue
    r0 = s['res'].get('0.003_0'); r1 = s['res'].get('0.01_0')
    if not (r0 and r1 and r0['ok'] and r1['ok']): continue
    a.append(np.array(s['gc']) - np.array(r0['r_prof'])); b.append(np.array(s['gc']) - np.array(r1['r_prof'])); g.append(np.array(s['gc']))
a, b, g = map(np.array, (a, b, g))
print('sample 1, per batch: correction c = g_c - r_prof = H_cd D^-1 g_d')
print('  SD over batches of g_c        ', fmt(g.std(0), '{:10.3g}'))
print('  SD over batches of c (3e-3)   ', fmt(a.std(0), '{:10.3g}'))
print('  SD over batches of c (1e-2)   ', fmt(b.std(0), '{:10.3g}'))
print('  corr(c 3e-3, c 1e-2) per param', fmt([np.corrcoef(a[:, j], b[:, j])[0, 1] for j in range(5)], '{:10.3f}'))
print('  median |c3 - c1| / SD(c3)     ', fmt(np.median(np.abs(a - b), 0) / a.std(0), '{:10.3f}'))
