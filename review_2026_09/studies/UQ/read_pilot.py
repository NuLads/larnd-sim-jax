"""UQ pilot readout (rules: REVIEW.md 31 Oct "UQ ... Pilot"): method sanity on KN10N3 (P3, O3, C3) and FD steps x3 (P3e3)."""
import glob, numpy as np, sys
sys.path.insert(0, '/sdf/scratch/neutrino/pgranger/review_2026_09/UQ')
import uq_lib as U
D = '/sdf/scratch/neutrino/pgranger/review_2026_09/UQ/dumps'; S = '/sdf/scratch/neutrino/pgranger/fit_result_staging'
RUN = {'P3': 'KN10N3PH', 'O3': 'KN10N3H', 'C3': 'KN10N3C75H', 'P3e3': 'KN10N3PH'}
P = ['Ab', 'eField', 'tran_diff', 'long_diff', 'lifetime']
R = {}
for k, run in RUN.items():
    d = U.load(f'{D}/{k}.npz'); assert list(d['params']) == P, d['params']
    R[k] = U.estimators(d)
    ck = sorted(glob.glob(f'{S}/sci_full_{run}/history_iter20000_*.pkl'))[0]
    R[k]['err'], R[k]['err1k'] = U.truth_errors(ck, P)
    R[k]['it'] = int(d['iteration']); R[k]['n'] = int(d['n'].sum()); R[k]['eps'] = d['eps']
print('paper tab:floor Conditional (%, at truth, 398 m): 0.016 0.0005 0.31 0.66 0.38; observed SD 0.11 0.008 1.2 4.1 2.5')
for k in RUN:
    r = R[k]
    print(f"\n== {k} ({RUN[k]} @ {r['it']}, eps {r['eps']}, {r['B']} batches, {r['n']} nuisances; dropped eigenvalues "
          f"{100 * r['dropped_frac']:.2f} % (dE/dx block alone {100 * r['dropped_dedx_frac']:.2f} %))")
    print('   error vs truth (%), current / last-1000: ' + ' '.join(f'{p} {100 * a:+.2f}/{100 * b:+.2f}' for p, a, b in zip(P, r['err'], r['err1k'])))
    for e in ('E1', 'E2', 'E3'):
        x = r[e]
        print(f"   {e}: S min eig {x['S_mineig']:.3g} cond {x['S_cond']:.3g} | sigma(%) " + ' '.join(f'{100 * v:.4g}' for v in x['sig'])
              + ' | sandwich(%) ' + ' '.join(f'{100 * v:.4g}' for v in x['sig_sw']))
        print(f"       Newton offset d(%) " + ' '.join(f'{100 * v:+.3g}' for v in x['d']) + ' | z ' + ' '.join(f'{v:+.2f}' for v in x['z'])
              + ' | z_sw ' + ' '.join(f'{v:+.2f}' for v in x['z_sw']) + f" | EDM {x['edm']:.3g}")
    c = r['E3']['corr']; print('   E3 correlations: ' + ' '.join(f'{P[i]}-{P[j]} {c[i, j]:+.2f}' for i in range(5) for j in range(i + 1, 5)))
ok = True
for k in RUN:
    pd = R[k]['E3']['S_mineig'] > 0; dr = R[k]['dropped_frac'] <= 0.05; ok &= pd and dr
    print(f"{k}: S positive definite {pd}; dropped <= 5 % {dr}")
ra = R['P3e3']['E3']['sig'] / R['P3']['E3']['sig']; dz = np.abs(R['P3e3']['E3']['z'] - R['P3']['E3']['z'])
st = bool(np.all(np.abs(ra - 1) <= 0.10) and np.all(dz <= 0.3)); ok &= st
print('FD steps x3: E3 sigma ratio ' + ' '.join(f'{v:.3f}' for v in ra) + ' | |dz| ' + ' '.join(f'{v:.2f}' for v in dz) + f" -> {'STABLE' if st else 'NOT STABLE'}")
print(f"PILOT VERDICT: method {'OK' if ok else 'NOT OK (fix before the ensemble)'}")
