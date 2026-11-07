"""POST HOC (6 Nov, after the DXPROF readout). Sandwich variants at P4, median over the 8 samples (%), vs the SD over the
8 samples of the conditional minimum m_cond (= TRUTHC P4, the estimator actually in hand):
  cond   = H_cc^-1 J_g H_cc^-1 (conditional bread, raw-score meat)          [theory check: SD/this = 2.3-2.6]
  pbread = S^-1 J_g S^-1        (dE/dx-profiled bread, raw-score meat)      [the joint-optimum form: g_d,b = 0 there]
  prof   = S^-1 J_r S^-1        (profiled bread, profiled-score meat)        [DXPROF primary]"""
import sys, numpy as np
sys.path.insert(0, '/sdf/scratch/neutrino/pgranger/review_2026_09/DXPROF')
from read_dx import load, P, fmt, PT
S_ = range(1, 9); R = {s: load(f'DX_KN10N{s}') for s in S_}
def J(G):
    G = np.array(G); c = G - G.mean(0); return len(G) / (len(G) - 1) * c.T @ c
for st, lm in (('0.003', '0'), ('0.01', '0'), ('0.003', '0.01')):
    W = {k: [] for k in ('cond', 'pbread', 'prof')}; MC = []
    for s in S_:
        recs, truth = R[s]; H = np.zeros((5, 5)); D = np.zeros((5, 5)); gc = []; rp = []; jac = cal = None
        for r in recs:
            sc = r.get('schur'); q = r['points'].get(PT); x = sc and sc['res'].get(f'{st}_{lm}')
            if not (sc and q and x and x['ok']): continue
            H += np.array(sc['hcc']); D += np.array(x['delta']); gc.append(sc['gc']); rp.append(x['r_prof'])
            jac = np.array([q['jac'][p] for p in P]); cal = np.array([q['calib'][p] for p in P])
        Sm = H - D; Sm = 0.5 * (Sm + Sm.T); Vi = np.linalg.inv(Sm); Hi = np.linalg.inv(0.5 * (H + H.T))
        for k, (B, M) in (('cond', (Hi, J(gc))), ('pbread', (Vi, J(gc))), ('prof', (Vi, J(rp)))):
            W[k].append(100 * jac * np.sqrt(np.diag(B @ M @ B)))
        MC.append(100 * (cal / truth - 1) + 100 * jac * (-Hi @ np.sum(gc, 0)))
    sd = np.array(MC).std(0, ddof=1)
    print(f'step {st} ridge {lm}:   SD(m_cond) = {fmt(sd, "{:8.4f}")}')
    for k in W:
        w = np.median(W[k], 0); print(f'   {k:7s} width {fmt(w, "{:8.4f}")}   SD(m_cond)/width {fmt(sd / w, "{:6.2f}")}')
