"""GNPROF readout (6 Nov; rules: REVIEW.md "GNPROF"). Truth-point runs sci_full_GN_KN10N<s> (s = 1-8, the TRUTHC end points
KN10N<s>C75H @ 20000, true ends) at P0 = the production fit point (calibration, dE/dx and geometry of the fit). Per batch b:
the FD calibration Hessian H_cc,b and gradient g_c,b at P0 (the POLISH ingredients), the dE/dx-exact observed profile
(DXPROF construction, step 3e-3), and the Gauss-Newton profile over dE/dx AND geometry in two curvature models (A =
expected Fisher, C = exact generalised Gauss-Newton): F_cc,b and Delta_b = F_cn A_nn^-1 F_nc per ridge.
Estimator: the conditional minimum at P0, m = e + 100 jac (-H^-1 sum_b g_c,b), H = sum_b H_cc,b (= POLISH, production).
Widths: 100 jac sqrt(diag(B^-1 J B^-1)), J = B/(B-1) sum_b (g_b - gbar)(g_b - gbar)^T (raw scores: the joint-optimum form),
for the breads B = H (conditional), sum (F - Delta) (GN), H - sum Delta (hybrid), and the one-block variants.
usage (container): python3 read_gn.py [--dry <TAG>]"""
import sys, glob, pickle, numpy as np
from scipy import stats
sys.path.insert(0, '/sdf/scratch/neutrino/pgranger/review_2026_09/TRUTHC')
from read_tc import P
FR = '/sdf/group/neutrino/pgranger/larnd-sim-jax/fit_result'
PT = 'P0_fit'; LAMS = ('0', '0.001', '0.01'); V = ('A', 'C')


def load(tag):
    fs = sorted(glob.glob(f'{FR}/sci_full_{tag}/history_iter*_*.pkl'))
    if not fs:
        raise SystemExit(f'{tag}: no output -> set not complete, no readout')
    th = pickle.load(open(fs[-1], 'rb'))
    return th['truthpoint'], np.array([float(np.ravel(th[f'{p}_target'])[0]) for p in P])


def M(x):
    return np.array(x, dtype=np.float64)


def sample(recs, truth):
    """Sums over the batches of one sample; None entries where a piece is missing."""
    o = dict(H=np.zeros((5, 5)), G=[], nb=0, Dobs={l: np.zeros((5, 5)) for l in ('0', '0.01')}, H1=np.zeros((5, 5)),
             F1={v: np.zeros((5, 5)) for v in V}, F={v: np.zeros((5, 5)) for v in V}, chk={v: [] for v in V},
             D={(v, l, k): np.zeros((5, 5)) for v in V for l in LAMS for k in ('delta', 'delta_d', 'delta_g')},
             ok={(v, l): 0 for v in V for l in LAMS}, ceff={(v, l): [] for v in V for l in LAMS}, t=[], nneg={v: [] for v in V})
    for r in recs:
        s = r.get('schur'); q = r['points'].get(PT); gn = s and s.get('gn')
        if not (s and q and gn):
            continue
        assert s['point'] == PT and list(s['params']) == P
        o['nb'] += 1; o['H'] += M(s['hcc']); o['G'].append(s['gc'])
        o['jac'] = np.array([q['jac'][p] for p in P]) if 'jac' in q else o.get('jac')
        o['cal'] = np.array([q['calib'][p] for p in P])
        for l in ('0', '0.01'):
            x = s['res'].get(f'0.003_{l}')
            if x and x['ok']:
                o['Dobs'][l] += M(x['delta'])
        p1 = r['points'].get('P1_truth', {}).get('hessian_fd_0.001')
        if p1 is not None and 'p1' in gn:
            o['H1'] += M([[p1[b][a] for b in P] for a in P])
            for v in V:
                o['F1'][v] += M(gn['p1'][v])
        o['t'].append(sum(gn['t'].values()))
        for v in V:
            g = gn[v]; o['F'][v] += M(g['fcc']); o['chk'][v].append((g['gram_vs_vjp'], g['asym_gg'], g['asym_cg']))
            o['nneg'][v].append(g['ann_nneg'])
            for l in LAMS:
                x = g['res'].get(l)
                if x and x['ok']:
                    o['ok'][(v, l)] += 1; o['ceff'][(v, l)].append(x['ceff'])
                    for k in ('delta', 'delta_d', 'delta_g'):
                        o['D'][(v, l, k)] += M(x[k])
    G = M(o['G']); o['G'] = G
    if o.get('jac') is None:                          # P0 record without jac: TRUTHC stores it only at LARND_TP_HESS points
        o['jac'] = None
    o['e'] = 100 * (o['cal'] / truth - 1)
    return o


def J_of(G):
    c = G - G.mean(0); return len(G) / (len(G) - 1) * c.T @ c


def width(B, Jm, jac):
    Bi = np.linalg.inv(0.5 * (B + B.T)); return 100 * jac * np.sqrt(np.abs(np.diag(Bi @ Jm @ Bi)))


def breads(o, v, l):
    H = o['H']; F = o['F'][v]; D = o['D'][(v, l, 'delta')]
    return {'cond (H)': H, 'obs dE/dx (H - Dobs)': H - o['Dobs']['0'], f'GN-{v} full (F - D)': F - D,
            f'hybrid-{v} (H - D)': H - D, f'GN-{v} dE/dx only': F - o['D'][(v, l, 'delta_d')],
            f'GN-{v} geometry only': F - o['D'][(v, l, 'delta_g')], f'GN-{v} unprofiled (F)': F}


def fmt(v, f='{:9.4f}'):
    return ' '.join(f.format(a) for a in v)


if __name__ == '__main__':
    lo, hi = np.sqrt(stats.chi2.ppf(0.025, 7) / 7), np.sqrt(stats.chi2.ppf(0.975, 7) / 7)
    if '--dry' in sys.argv:
        tag = sys.argv[sys.argv.index('--dry') + 1]; recs, truth = load(tag); o = sample(recs, truth)
        print(f'DRY RUN on {tag}: {o["nb"]} batch record(s) with a GN block; GN wall per batch {o["t"]} s')
        for v in V:
            print(f'  {v}: (gram/vjp, asym_gg, asym_cg) per batch {o["chk"][v]}; A_nn non-positive eigenvalues {o["nneg"][v]}')
            print(f'     sum F_cc diag {fmt(np.diag(o["F"][v]), "{:10.4g}")} | sum H_cc diag {fmt(np.diag(o["H"]), "{:10.4g}")}')
            for l in LAMS:
                D = o['D'][(v, l, 'delta')]
                print(f'     ridge {l}: ok {o["ok"][(v, l)]}  Delta/F_cc diag {fmt(np.diag(D) / np.diag(o["F"][v]), "{:7.3f}")}'
                      f'  (dE/dx {fmt(np.diag(o["D"][(v, l, "delta_d")]) / np.diag(o["F"][v]), "{:7.3f}")}, geometry '
                      f'{fmt(np.diag(o["D"][(v, l, "delta_g")]) / np.diag(o["F"][v]), "{:7.3f}")})')
            if np.any(o['H1']):
                print(f'     P1: sum F_cc / sum FD H_cc diag {fmt(np.diag(o["F1"][v]) / np.diag(o["H1"]), "{:7.3f}")}')
        print(f'  observed dE/dx Delta/H_cc diag (DXPROF construction) {fmt(np.diag(o["Dobs"]["0"]) / np.diag(o["H"]), "{:7.3f}")}')
        sys.exit(0)

    S = range(1, 9); O = {}
    for s in S:
        recs, truth = load(f'GN_KN10N{s}'); O[s] = sample(recs, truth)
        assert O[s]['jac'] is not None, 'P0 has no jac: the runs must have LARND_TP_HESS=P0_fit:P1_truth'
    print(f'batches with a GN block per sample: {[O[s]["nb"] for s in S]}; GN wall per batch median '
          f'{np.median(np.concatenate([O[s]["t"] for s in S])):.0f} s')
    # V-b internal consistency (median over all batches): jvp-Gram vs vjp rows, and the symmetry of the geometry block
    chk = {v: np.median(np.concatenate([np.array(O[s]['chk'][v]) for s in S]), 0) for v in V}
    vb = {v: bool(chk[v][0] < 1e-2 and chk[v][1] < 1e-2 and chk[v][2] < 1e-2) for v in V}
    for v in V:
        print(f'V-b {v}: median gram/vjp {chk[v][0]:.1e}, asym_gg {chk[v][1]:.1e}, asym_cg {chk[v][2]:.1e} -> '
              f'{"PASS" if vb[v] else "FAIL"}')
    # V-a curvature model AT P0 (where it is used; moved from P1 after the smoke, see REVIEW.md GNPROF): sum_b F_cc / sum_b
    # FD H_cc, diagonal, median over samples. P1 (truth point) is descriptive only: the exact GGN is indefinite where the
    # truth-projected geometry misfits (smoke batch 1).
    jP = [0, 2, 3, 4]
    va = {}
    for v in V:
        rr = np.median([np.diag(O[s]['F'][v]) / np.diag(O[s]['H']) for s in S], 0)
        r1 = np.median([np.diag(O[s]['F1'][v]) / np.diag(O[s]['H1']) for s in S], 0)
        dd = np.median([(np.diag(O[s]['D'][(v, '0', 'delta_d')]) / np.diag(O[s]['F'][v]))
                        / (np.diag(O[s]['Dobs']['0']) / np.diag(O[s]['H'])) for s in S], 0)
        va[v] = (rr, float(np.max(np.abs(np.log(rr[jP])))))
        print(f'V-a {v}: F_cc(P0) / FD H_cc(P0) diag {fmt(rr, "{:7.3f}")}  (max |ln| on Ab, tran, long, lifetime {va[v][1]:.3f})')
        print(f'      descriptive: F_cc(P1) / FD H_cc(P1) {fmt(r1, "{:7.3f}")}; GN / observed dE/dx-only fraction Delta_d/F '
              f'{fmt(dd, "{:7.3f}")}')
    vok = [v for v in V if vb[v]]
    if not vok:
        print('V-b failed for both variants -> NO VERDICT'); sys.exit(0)
    vp = min(vok, key=lambda v: va[v][1])
    va_pass = bool(np.all((va[vp][0][jP] > 0.8) & (va[vp][0][jP] < 1.25)))
    print(f'primary curvature model: {vp} (V-a {"PASS" if va_pass else "FAIL -> primary bread = hybrid H - Delta"})')
    # V-c ridge stability of the primary bread: ridge 0 vs 1e-3, widths within 10 % on all 5, >= 7/8 samples
    def prim(o, l):
        b = breads(o, vp, l); return b[f'GN-{vp} full (F - D)'] if va_pass else b[f'hybrid-{vp} (H - D)']
    lam = '0'
    okr = all(O[s]['ok'][(vp, '0')] >= 0.95 * O[s]['nb'] for s in S)
    if not okr:
        lam = '0.001'
    st = [bool(np.all(np.abs(width(prim(O[s], '0.001'), J_of(O[s]['G']), O[s]['jac'])
                             / width(prim(O[s], lam), J_of(O[s]['G']), O[s]['jac']) - 1) < 0.10)) for s in S]
    vc = sum(st) >= 7
    print(f'V-c ridge: solves at ridge 0 ok on >= 95 % of batches: {okr} -> primary ridge {lam}; ridge 0 vs 1e-3 widths '
          f'within 10 %: {sum(st)}/8 -> {"PASS" if vc else "FAIL"}')
    # the estimator and the widths
    m = np.array([O[s]['e'] + 100 * O[s]['jac'] * (-np.linalg.solve(0.5 * (O[s]['H'] + O[s]['H'].T), O[s]['G'].sum(0)))
                  for s in S])
    sd = m.std(0, ddof=1)
    print(f'\nestimator m = conditional minimum at P0 (POLISH form); SD over the 8 samples (%): {fmt(sd)}')
    print(f'mean m (%): {fmt(m.mean(0), "{:+9.3f}")};  band for SD/width [{lo:.2f}, {hi:.2f}]')
    print('  ' + ' ' * 34 + ' '.join(f'{p:>9s}' for p in P) + '   | SD(m) / width')
    keys = list(breads(O[1], vp, lam).keys())
    for k in keys:
        w = np.median([width(breads(O[s], vp, lam)[k], J_of(O[s]['G']), O[s]['jac']) for s in S], 0)
        print(f'  {k:34s}' + fmt(w) + '   | ' + fmt(sd / w, '{:6.2f}'))
    other = [v for v in V if v != vp][0]
    for k in (f'GN-{other} full (F - D)', f'hybrid-{other} (H - D)'):
        w = np.median([width(breads(O[s], other, lam)[k], J_of(O[s]['G']), O[s]['jac']) for s in S], 0)
        print(f'  {k:34s}' + fmt(w) + '   | ' + fmt(sd / w, '{:6.2f}'))
    wp = np.median([width(prim(O[s], lam), J_of(O[s]['G']), O[s]['jac']) for s in S], 0)
    rat = sd / wp; n_in = int(sum(lo <= x <= hi for x in rat))
    verdict = 'CONFIRMED' if n_in == 5 else ('PARTLY CONFIRMED' if n_in >= 3 else 'NOT CONFIRMED')
    ok = vb[vp] and vc
    print(f'\nGNPROF VERDICT (primary bread {"GN-" + vp + " full" if va_pass else "hybrid-" + vp}, ridge {lam}: SD(m) / width '
          f'{fmt(rat, "{:5.2f}")} -> {n_in}/5 in the band): {verdict if ok else "NO VERDICT (V-b or V-c failed)"}')
    print(f'A_nn non-positive eigenvalues (median per batch): ' + ', '.join(
        f'{v} {np.median(np.concatenate([O[s]["nneg"][v] for s in S])):.0f}' for v in V))
