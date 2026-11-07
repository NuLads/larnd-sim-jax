"""Theory check (6 Nov; Pierre: why do the Hessian and the Huber sandwich not work, while they should in theory?).
Descriptive, from the TRUTHC / TRUTHC2 truth-point runs (KN10N1-8 C75H @ 20000, true ends). At each point the calibration
is moved to its exact conditional minimum m (Newton step on the summed FD Hessian, step 1e-3), so optimiser scatter in the
calibration is gone. Per point: SD of m over the 8 samples vs the Hessian width and the sandwich width
(J = B/(B-1) sum_b (g_b - gbar)(g_b - gbar)^T over the 100 per-batch gradients). P1 = every nuisance at truth: the
textbook case, where the sandwich should match. usage (container): python3 theory_check.py"""
import sys, numpy as np
sys.path.insert(0, '/sdf/scratch/neutrino/pgranger/review_2026_09/TRUTHC'); sys.path.insert(0, '/sdf/scratch/neutrino/pgranger/review_2026_09/UQ')
from read_tc import P, load
from scipy import stats


def widths(recs, name, truth, step='1e-3'):
    H = np.zeros((5, 5)); G = []; jac = cal = None
    for r in recs:
        q = r['points'].get(name)
        if q is None or f'hessian_fd_{float(step):g}' not in q:
            continue
        cols = q[f'hessian_fd_{float(step):g}']
        H += np.array([[cols[P[b]][P[a]] for b in range(5)] for a in range(5)])
        G.append([q['grad'][p] for p in P])
        jac = np.array([q['jac'][p] for p in P]) if jac is None else jac
        cal = np.array([q['calib'][p] for p in P]) if cal is None else cal
    G = np.array(G); B = len(G); Hs = 0.5 * (H + H.T); V = np.linalg.inv(Hs)
    gc = G - G.mean(0); J = B / (B - 1) * gc.T @ gc; Vs = V @ J @ V
    m = 100 * (cal / truth - 1) + 100 * jac * (-V @ G.sum(0))
    return m, 100 * jac * np.sqrt(np.diag(V)), 100 * jac * np.sqrt(np.diag(Vs)), B


PTS = [('P1_truth', 'all nuisances true', 'KN10N'), ('P4_geom_true', 'geometry true, dE/dx fitted', 'KN10N'),
       ('P3_dedx_true', 'dE/dx true, geometry fitted', 'KN10N'), ('P0_fit', 'all fitted (true ends)', 'KN10N')]
S = range(1, 9)
lo, hi = np.sqrt(stats.chi2.ppf(0.025, 7) / 7), np.sqrt(stats.chi2.ppf(0.975, 7) / 7)
print(f'n = 8 samples; SD ratio 95 % band if the width were right: [{lo:.2f}, {hi:.2f}] x width')
for name, lab, pre in PTS:
    R = []
    for s in S:
        recs, th = load(f'{pre}{s}'); truth = np.array([float(np.ravel(th[f'{p}_target'])[0]) for p in P])
        R.append(widths(recs, name, truth))
    M = np.array([r[0] for r in R]); SH = np.median([r[1] for r in R], 0); SW = np.median([r[2] for r in R], 0); sd = M.std(0, ddof=1)
    print(f'{name} ({lab}; {R[0][3]} batches):')
    print('  ' + ' '.join(f'{p:>10s}' for p in P))
    print('  Hessian  ' + ' '.join(f'{v:10.4f}' for v in SH)); print('  sandwich ' + ' '.join(f'{v:10.4f}' for v in SW))
    print('  SD of m  ' + ' '.join(f'{v:10.4f}' for v in sd)); print('  SD/sandw ' + ' '.join(f'{v:10.2f}' for v in sd / SW))
    print('  mean m   ' + ' '.join(f'{v:+10.3f}' for v in M.mean(0)))
