"""TRUTHC readout (4 Nov; rules: REVIEW.md "TRUTHC"). Truth-point runs sci_full_TC_<RUN> of production end points (true ends,
KN10N<s>C75H @ 20000, s = 1-8) and the instrument check TC_EXP17 (P0 only) against the E1 dump D_EXP17.
At each point the 5 x 5 calibration Hessian H (central FD of the gradient, steps 1e-3 and 3e-3 in norm variables) and the
gradient g are summed over the 100 batches; d = 100 jac (-H^-1 g) (%) is the conditional Newton step; the conditional
minimum m = e + d, e = error of the point's calibration. Points: P0 fit (calib, dE/dx, geometry of the fit), P1 truth (all
three at truth; geometry = truth projected on the 10 cm spline), P2 calib true (nuisances of the fit), P3 dE/dx true,
P4 geometry true. usage (container): python3 read_tc.py [--dry <TCSMOKE>]"""
import sys, glob, pickle, numpy as np
sys.path.insert(0, '/sdf/scratch/neutrino/pgranger/review_2026_09/UQ')
from uqt_common import P, e1
FR = '/sdf/group/neutrino/pgranger/larnd-sim-jax/fit_result'
ST = ('1e-3', '3e-3')
PTS = ('P0_fit', 'P1_truth', 'P2_calib_true', 'P3_dedx_true', 'P4_geom_true')


def load(tag):
    fs = sorted(glob.glob(f'{FR}/sci_full_TC_{tag}/history_iter*_*.pkl')) or sorted(glob.glob(f'{FR}/sci_full_{tag}/history_iter*_*.pkl'))
    if not fs:
        raise SystemExit(f'{tag}: no truth-point output -> set not complete, no readout')
    th = pickle.load(open(fs[-1], 'rb'))
    return th['truthpoint'], th


def point(recs, name, truth, step):
    H = np.zeros((5, 5)); g = np.zeros(5); jac = None; cal = None; nb = 0
    for r in recs:
        q = r['points'].get(name)
        if q is None or f'hessian_fd_{float(step):g}' not in q:
            continue
        cols = q[f'hessian_fd_{float(step):g}']
        H += np.array([[cols[P[b]][P[a]] for b in range(5)] for a in range(5)])
        g += np.array([q['grad'][p] for p in P]); nb += 1
        jac = np.array([q['jac'][p] for p in P]) if jac is None else jac
        cal = np.array([q['calib'][p] for p in P]) if cal is None else cal
    if nb == 0:
        return None
    Hs = 0.5 * (H + H.T); V = np.linalg.inv(Hs)
    d = 100 * jac * (-V @ g); sig = 100 * jac * np.sqrt(np.abs(np.diag(V)))
    e = 100 * (cal / truth - 1)
    return dict(e=e, d=d, m=e + d, sig=sig, Hdiag=np.diag(Hs), asym=np.abs(H - H.T).max() / np.abs(H).max(),
                mineig=np.linalg.eigvalsh(Hs).min(), nb=nb)


def run(tag, names=PTS):
    recs, th = load(tag)
    truth = np.array([float(np.ravel(th[f'{p}_target'])[0]) for p in P])
    return {n: {s: point(recs, n, truth, s) for s in ST} for n in names}, len(recs)


if __name__ == '__main__':
    if '--dry' in sys.argv:
        tag = sys.argv[sys.argv.index('--dry') + 1]
        R, nb = run(tag, ('P0_fit', 'P1_truth'))
        print(f'DRY RUN on {tag}: {nb} batch record(s)')
        for n, v in R.items():
            for s, x in v.items():
                if x is None:
                    print(f'  {n} step {s}: absent'); continue
                print(f'  {n:13s} {s}: e ' + ' '.join(f'{a:+.3f}' for a in x['e']) + ' | d ' + ' '.join(f'{a:+.3f}' for a in x['d'])
                      + ' | sig ' + ' '.join(f'{a:.3f}' for a in x['sig']) + f' | asym {x["asym"]:.2e} mineig {x["mineig"]:.3g}')
        sys.exit(0)
    S = list(range(1, 9))
    R = {s: run(f'KN10N{s}')[0] for s in S}
    # V2 instrument check: TC_EXP17 P0 against the E1 dump D_EXP17 of the same end point
    v, _ = run('EXP17', ('P0_fit',)); x = v['P0_fit']['1e-3']
    import uq_lib as U
    dd = U.load('/sdf/scratch/neutrino/pgranger/review_2026_09/UQ/dumps/D_EXP17.npz'); Hd = np.diag(dd['Hcc'].sum(0))
    y = e1('D_EXP17')
    rH = x['Hdiag'] / Hd; zd = (x['d'] - y['d']) / y['sig']
    v2 = bool(np.all(np.abs(rH - 1) < 0.15) and np.all(np.abs(zd) < 2.5))
    print('V2 instrument check (TC_EXP17 P0 vs D_EXP17): Hessian diag ratio ' + ' '.join(f'{a:.3f}' for a in rH)
          + ' | (d_TC - d_E1) / sigma_E1 ' + ' '.join(f'{a:+.2f}' for a in zd) + f' -> {"PASS" if v2 else "FAIL"}')
    # V1 step stability and V3 quadratic consistency, on long_diff and tran_diff
    J = [P.index('long_diff'), P.index('tran_diff')]
    # central FD is O(h^2): m(1e-3) - m(3e-3) = -8 x the truncation error at 1e-3 (Richardson); V1 bounds that error
    v1 = all(sum(abs(R[s][n]['1e-3']['m'][j] - R[s][n]['3e-3']['m'][j]) / 8 < 0.1 * R[s][n]['1e-3']['sig'][j] for s in S) >= 7
             for n in ('P0_fit', 'P1_truth') for j in J)
    v3 = all(sum(abs(R[s]['P0_fit']['1e-3']['m'][j] - R[s]['P2_calib_true']['1e-3']['m'][j]) < 2 * R[s]['P0_fit']['1e-3']['sig'][j] for s in S) >= 7
             for j in J)
    print(f'V1 step stability (Richardson |m(1e-3) - m(3e-3)| / 8 < 0.1 sigma on >= 7/8, P0 and P1): {"PASS" if v1 else "FAIL"}')
    print(f'V3 quadratic consistency (|m(P0) - m(P2)| < 2 sigma on >= 7/8): {"PASS" if v3 else "FAIL"}')
    M = {n: np.array([R[s][n]['1e-3']['m'] for s in S]) for n in PTS}
    print('\n  conditional minimum m = e + d (mean +- SEM over 8 samples, %, step 1e-3):')
    print('  ' + ' ' * 14 + ' '.join(f'{p:>15s}' for p in P))
    for n in PTS:
        print(f'  {n:14s}' + ' '.join(f'{a:+8.3f}+-{b:5.3f}' for a, b in zip(M[n].mean(0), M[n].std(0, ddof=1) / np.sqrt(8))))
    D = M['P0_fit'] - M['P1_truth']; mD, sD = D.mean(0), D.std(0, ddof=1) / np.sqrt(8)
    m1, s1 = M['P1_truth'].mean(0), M['P1_truth'].std(0, ddof=1) / np.sqrt(8); m0 = M['P0_fit'].mean(0)
    print('  paired m(P0) - m(P1): ' + ' '.join(f'{p} {a:+.3f}+-{b:.3f}' for p, a, b in zip(P, mD, sD)))
    for lab, n in (('geometry', 'P4_geom_true'), ('dE/dx', 'P3_dedx_true')):
        fr = (M['P0_fit'] - M[n]).mean(0) / mD
        print(f'  descriptive, share of m(P0) - m(P1) removed by {lab} at truth alone: ' + ' '.join(f'{p} {a:.2f}' for p, a in zip(P, fr)))
    ok = v1 and v2 and v3
    cls = {}
    for j in J:
        if abs(m1[j]) < 2 * s1[j] and np.sign(mD[j]) == np.sign(m0[j]) and abs(mD[j]) > 2 * sD[j]: c = 'CARRIED BY THE NUISANCES'
        elif abs(m1[j]) >= 2 * s1[j] and np.sign(m1[j]) == np.sign(m0[j]) and abs(mD[j]) < 2 * sD[j]: c = 'OBJECTIVE-LEVEL'
        else: c = 'MIXED'
        cls[P[j]] = c
    print('TRUTHC VERDICT: ' + (', '.join(f'{p} {c}' for p, c in cls.items()) if ok else 'NO VERDICT (a validation check failed)'))
