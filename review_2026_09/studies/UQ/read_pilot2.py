"""UQ pilot 2 readout (rules: REVIEW.md 31 Oct "UQ pilot 2: step-size plateau", written before launch).
Operating step x20; plateau neighbours x10 and x30 on KN10N3PH; positivity on the x20 dumps of KN10N3PH, KN10N3H,
KN10N3C75H; P3e20f (eField step 1e-3) descriptive. E3 is the primary estimator, E2 the stated fallback.
usage (container): python3 read_pilot2.py   [RP2_DRY=1: pilot-1 dumps stand in, to test the script]"""
import glob, os, sys, numpy as np
sys.path.insert(0, '/sdf/scratch/neutrino/pgranger/review_2026_09/UQ')
import uq_lib as U
D = '/sdf/scratch/neutrino/pgranger/review_2026_09/UQ/dumps'; S = '/sdf/scratch/neutrino/pgranger/fit_result_staging'
DRY = os.environ.get('RP2_DRY') == '1'
RUN = {'P3e10': 'KN10N3PH', 'P3e20': 'KN10N3PH', 'P3e30': 'KN10N3PH', 'O3e20': 'KN10N3H', 'C3e20': 'KN10N3C75H', 'P3e20f': 'KN10N3PH'}
STAND = {'P3e10': 'P3', 'P3e20': 'P3e3', 'P3e30': 'P3e3', 'O3e20': 'O3', 'C3e20': 'C3', 'P3e20f': 'P3e3'}
P = ['Ab', 'eField', 'tran_diff', 'long_diff', 'lifetime']
R = {}
for k, run in RUN.items():
    d = U.load(f"{D}/{STAND[k] if DRY else k}.npz"); assert list(d['params']) == P, d['params']
    if not DRY:
        assert 'Hnn_flat' in d, f'{k}: not a FULL dump'
    R[k] = U.estimators(d)
    ck = sorted(glob.glob(f'{S}/sci_full_{run}/history_iter20000_*.pkl'))[0]
    R[k]['err'], R[k]['err1k'] = U.truth_errors(ck, P)
    R[k]['it'] = int(d['iteration']); R[k]['eps'] = d['eps']; R[k]['epsc'] = d['eps_calib'] if 'eps_calib' in d else None
print('DRY RUN (pilot-1 dumps stand in)' if DRY else 'pilot 2 dumps')
for k in RUN:
    r = R[k]
    print(f"\n== {k} ({RUN[k]} @ {r['it']}, eps {r['eps']}, calib eps {r['epsc']}; dropped {100 * r['dropped_frac']:.2f} %)")
    print('   error vs truth (%), last-1000: ' + ' '.join(f'{p} {100 * b:+.2f}' for p, b in zip(P, r['err1k'])))
    for e in ('E1', 'E2', 'E3'):
        x = r[e]
        print(f"   {e}: S min eig {x['S_mineig']:.3g} | sigma(%) " + ' '.join(f'{100 * v:.4g}' for v in x['sig'])
              + ' | sandwich(%) ' + ' '.join(f'{100 * v:.4g}' for v in x['sig_sw'])
              + ' | d(%) ' + ' '.join(f'{100 * v:+.3g}' for v in x['d']) + ' | z ' + ' '.join(f'{v:+.2f}' for v in x['z'])
              + f" | EDM {x['edm']:.3g}")


def verdict(e):
    ok = True
    for k in ('P3e20', 'O3e20', 'C3e20'):
        pd = R[k][e]['S_mineig'] > 0; dr = R[k]['dropped_frac'] <= 0.05; ok &= pd and dr
        print(f"  {e} {k}: S positive definite {pd}; dropped <= 5 % {dr}")
    for nb in ('P3e10', 'P3e30', 'P3e20f'):
        ra = R[nb][e]['sig'] / R['P3e20'][e]['sig']; dz = np.abs(R[nb][e]['z'] - R['P3e20'][e]['z'])
        st = bool(np.all(np.abs(ra - 1) <= 0.10) and np.all(dz <= 0.3))
        if nb != 'P3e20f':
            ok &= st
        print(f"  {e} {nb} vs P3e20{' (descriptive)' if nb == 'P3e20f' else ''}: sigma ratio " + ' '.join(f'{v:.3f}' for v in ra)
              + ' | |dz| ' + ' '.join(f'{v:.2f}' for v in dz) + f" -> {'AGREES' if st else 'DIFFERS'}")
    return ok


print('\nRULES')
ok3 = verdict('E3'); ok2 = verdict('E2')
if ok3:
    v = 'E3 OK at x20 -> the ensemble runs on E3'
elif ok2:
    v = 'E3 NOT OK, E2 OK at x20 -> the ensemble runs on E2 (geometry conditional)'
else:
    v = 'E3 and E2 NOT OK -> the Schur route is closed (E1 / sandwich + replica-measured optimisation scatter)'
print(f"PILOT 2 VERDICT: {v}" + ('  -- DRY RUN' if DRY else ''))
