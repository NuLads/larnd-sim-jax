import sys, numpy as np
sys.path.insert(0, '/sdf/scratch/neutrino/pgranger/review_2026_09/GNPROF'); sys.path.insert(0, '/sdf/scratch/neutrino/pgranger/review_2026_09/TRUTHC')
from read_gn import load, P
recs, truth = load('GNSMOKE')
for r in recs:
    gn = r['schur']['gn']; p1 = r['points']['P1_truth']['hessian_fd_0.001']
    H1 = np.array([[p1[b][a] for b in P] for a in P])
    print('batch', r['batch'])
    print('  FD H_cc(P1) diag ', np.diag(H1).round(3), ' eig', np.linalg.eigvalsh(0.5 * (H1 + H1.T)).round(3))
    for v in ('A', 'C'):
        F1 = np.array(gn['p1'][v]); print(f'  {v} F_cc(P1) diag  ', np.diag(F1).round(3), ' eig', np.linalg.eigvalsh(F1).round(3))
    print('  C F_cc(P0) diag  ', np.diag(np.array(gn['C']['fcc'])).round(3), '| FD H_cc(P0)', np.diag(np.array(r['schur']['hcc'])).round(3))
    print('  loss P0 / P1', r['points']['P0_fit']['loss'], r['points']['P1_truth']['loss'])
