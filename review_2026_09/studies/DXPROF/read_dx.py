"""DXPROF readout (6 Nov; rules: REVIEW.md "DXPROF"). Truth-point runs sci_full_DX_KN10N<s> (s = 1-8, the TRUTHC end points
KN10N<s>C75H @ 20000, true ends) at P4 = calibration and dE/dx of the fit, geometry at truth. Per batch b (DXPROF/tree,
dE/dx-exact Schur block): H_cc,b, g_c,b, Delta_b = H_cd D^-1 H_dc and the profiled score r_b = g_c - H_cd D^-1 g_d, for dE/dx
probe steps 3e-3, 1e-2 and ridges 0, 1e-3, 1e-2 (x median |diag D|). Profiled curvature S = sum_b (H_cc,b - Delta_b);
sigma_prof = sqrt(diag S^-1); sandwich S^-1 J S^-1 with J = B/(B-1) sum_b (r_b - rbar)(r_b - rbar)^T; one-step profiled
estimate m_prof = e + 100 jac (-S^-1 sum_b r_b). The theory says SD over samples of m_prof = the profiled sandwich width.
usage (container): python3 read_dx.py [--dry <TAG>]"""
import sys, glob, pickle, numpy as np
from scipy import stats
sys.path.insert(0, '/sdf/scratch/neutrino/pgranger/review_2026_09/TRUTHC')
from read_tc import P, point
FR = '/sdf/group/neutrino/pgranger/larnd-sim-jax/fit_result'
PT = 'P4_geom_true'; STEP0, STEP1 = '0.003', '0.01'; LAM0, LAM1 = '0', '0.001'
PRIM = ['Ab', 'tran_diff', 'long_diff', 'lifetime']


def load(tag):
    fs = sorted(glob.glob(f'{FR}/sci_full_{tag}/history_iter*_*.pkl'))
    if not fs:
        raise SystemExit(f'{tag}: no output -> set not complete, no readout')
    th = pickle.load(open(fs[-1], 'rb'))
    return th['truthpoint'], np.array([float(np.ravel(th[f'{p}_target'])[0]) for p in P])


def profile(recs, truth, step, lam):
    Hcc = np.zeros((5, 5)); D = np.zeros((5, 5)); R = []; gc = []; nb = nfail = 0; jac = cal = None; ndiag = 0
    for r in recs:
        s = r.get('schur'); q = r['points'].get(PT)
        if s is None or q is None:
            continue
        assert s['point'] == PT and list(s['params']) == P
        x = s['res'].get(f'{step}_{lam}')
        if x is None or not x['ok']:
            nfail += 1; continue
        Hcc += np.array(s['hcc']); D += np.array(x['delta']); R.append(x['r_prof']); gc.append(s['gc']); nb += 1
        ndiag += x['n_diag_nonpos']
        jac = np.array([q['jac'][p] for p in P]) if jac is None else jac
        cal = np.array([q['calib'][p] for p in P]) if cal is None else cal
    R = np.array(R); S = Hcc - D; S = 0.5 * (S + S.T); V = np.linalg.inv(S); Vc = np.linalg.inv(0.5 * (Hcc + Hcc.T))
    rc = R - R.mean(0); J = nb / (nb - 1) * rc.T @ rc if nb > 1 else np.full((5, 5), np.nan)   # 1 batch (smoke): no sandwich
    G = np.array(gc); gcc = G - G.mean(0); Jc = nb / (nb - 1) * gcc.T @ gcc if nb > 1 else np.full((5, 5), np.nan)
    e = 100 * (cal / truth - 1)
    return dict(nb=nb, nfail=nfail, ndiag=ndiag, mineig=float(np.linalg.eigvalsh(S).min()),
                sig=100 * jac * np.sqrt(np.abs(np.diag(V))), sig_sw=100 * jac * np.sqrt(np.abs(np.diag(V @ J @ V))),
                sig_c=100 * jac * np.sqrt(np.diag(Vc)), sig_c_sw=100 * jac * np.sqrt(np.diag(Vc @ Jc @ Vc)),
                m=e + 100 * jac * (-V @ R.sum(0)), m_c=e + 100 * jac * (-Vc @ G.sum(0)), e=e)


def fmt(v, f='{:9.4f}'):
    return ' '.join(f.format(a) for a in v)


if __name__ == '__main__':
    if '--dry' in sys.argv:
        tag = sys.argv[sys.argv.index('--dry') + 1]; recs, truth = load(tag)
        print(f'DRY RUN on {tag}: {len(recs)} batch record(s)')
        for st in (STEP0, STEP1):
            for lm in ('0', '0.001', '0.01'):
                x = profile(recs, truth, st, lm)
                print(f'  step {st} lam {lm}: nb {x["nb"]} fail {x["nfail"]} diag<=0 {x["ndiag"]} mineig(S) {x["mineig"]:.3g} | '
                      f'sig_c {fmt(x["sig_c"])} | sig_prof {fmt(x["sig"])} | m_prof {fmt(x["m"], "{:+8.3f}")}')
        sys.exit(0)
    S = range(1, 9)
    RECS = {s: load(f'DX_KN10N{s}') for s in S}
    # V3 solves: at ridge 0 the banded solves succeed on >= 95 % of batches of every sample; else the pre-registered
    # fallback ridge 1e-3 is primary
    lam = LAM0
    fr = {s: profile(*RECS[s], STEP0, LAM0) for s in S}
    v3 = all(fr[s]['nfail'] <= 0.05 * (fr[s]['nb'] + fr[s]['nfail']) for s in S)
    if not v3:
        lam = LAM1; fr = {s: profile(*RECS[s], STEP0, LAM1) for s in S}
    print(f'V3 banded solves at ridge 0 on >= 95 % of batches: {"PASS" if v3 else "FAIL -> primary ridge 1e-3"} '
          f'(fails per sample: {[profile(*RECS[s], STEP0, LAM0)["nfail"] for s in S]})')
    # V2: conditional pieces reproduce TRUTHC P4 (same point, step 1e-3): m_c within 0.05 sigma on all 5, >= 7/8
    from read_tc import load as load_tc
    ok2 = []
    for s in S:
        rt, th = load_tc(f'KN10N{s}'); truth = np.array([float(np.ravel(th[f'{p}_target'])[0]) for p in P])
        y = point(rt, PT, truth, '1e-3'); x = point(RECS[s][0], PT, truth, '1e-3')
        ok2.append(bool(np.all(np.abs(x['m'] - y['m']) < 0.05 * y['sig'])))
    v2 = sum(ok2) >= 7
    print(f'V2 reproduction of TRUTHC P4 (conditional m within 0.05 sigma on all 5, >= 7/8): {sum(ok2)}/8 -> {"PASS" if v2 else "FAIL"}')
    # V1 dE/dx step stability: 3e-3 vs 1e-2, sandwich width within 10 % and m_prof within 0.1 sigma, primary params, >= 7/8
    alt = {s: profile(*RECS[s], STEP1, lam) for s in S}
    jP = [P.index(p) for p in PRIM]
    ok1 = [bool(np.all(np.abs(alt[s]['sig_sw'][jP] / fr[s]['sig_sw'][jP] - 1) < 0.10)
                and np.all(np.abs(alt[s]['m'][jP] - fr[s]['m'][jP]) < 0.1 * fr[s]['sig_sw'][jP])) for s in S]
    v1 = sum(ok1) >= 7
    print(f'V1 dE/dx step stability (3e-3 vs 1e-2): {sum(ok1)}/8 -> {"PASS" if v1 else "FAIL"}')
    M = np.array([fr[s]['m'] for s in S]); MC = np.array([fr[s]['m_c'] for s in S])
    W = {k: np.median([fr[s][k] for s in S], 0) for k in ('sig_c', 'sig_c_sw', 'sig', 'sig_sw')}
    lo, hi = np.sqrt(stats.chi2.ppf(0.025, 7) / 7), np.sqrt(stats.chi2.ppf(0.975, 7) / 7)
    print(f'ridge {lam}, dE/dx step {STEP0}; widths = median over the 8 samples (%); band [{lo:.2f}, {hi:.2f}]')
    print('  ' + ' ' * 28 + ' '.join(f'{p:>9s}' for p in P))
    for lab, v in (('conditional Hessian', W['sig_c']), ('conditional sandwich', W['sig_c_sw']), ('profiled Hessian', W['sig']),
                   ('profiled sandwich', W['sig_sw']), ('SD of m_prof (one-step)', M.std(0, ddof=1)),
                   ('SD of m_cond (TRUTHC P4)', MC.std(0, ddof=1))):
        print(f'  {lab:28s}' + fmt(v))
    rat = M.std(0, ddof=1) / W['sig_sw']; ratc = MC.std(0, ddof=1) / W['sig_sw']
    print(f'  {"SD(m_prof) / prof. sandwich":28s}' + fmt(rat, '{:9.2f}'))
    print(f'  {"SD(m_cond) / prof. sandwich":28s}' + fmt(ratc, '{:9.2f}'))
    print(f'  {"inflation prof/cond sandw.":28s}' + fmt(W['sig_sw'] / W['sig_c_sw'], '{:9.2f}'))
    print(f'  {"mean m_prof":28s}' + fmt(M.mean(0), '{:+9.3f}') + '   | mean m_cond ' + fmt(MC.mean(0), '{:+8.3f}'))
    n_in = sum(lo <= rat[P.index(p)] <= hi for p in PRIM)
    v = 'CONFIRMED' if n_in == 4 else ('PARTLY CONFIRMED' if n_in >= 2 else 'NOT CONFIRMED')
    ok = v1 and v2
    print(f'DXPROF VERDICT (SD of m_prof within the band of the profiled sandwich on {n_in}/4 of Ab, tran_diff, long_diff, '
          f'lifetime): {v if ok else "NO VERDICT (V1 or V2 failed)"}')
