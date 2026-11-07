"""GNREAL readout (6 Nov; rules: REVIEW.md "GNREAL"). Runs sci_full_GR_EXP<s> (s = 29-32) = the GNPROF tree (unchanged, md5
8a7d4c2d) at P0 of the PRODUCTION end points EXP<s>C75H @ 20000 (realistic 1 mm entry smear). Per sample: the GN-C (exact
GGN) profile over dE/dx + geometry with the per-batch bound 0 <= Delta_b <= F_cc,b (generalised eigenvalues of Delta_b
w.r.t. F_cc,b clipped to [0, 1]; the GNPROF post hoc rule, fixed here before reading), raw-score meat, ridge 0.
Reference: SD of the polished estimate m over the n = 30 production end points (UQ/polish_table.py; frozen REF30) and the
H-stage optimiser scatter sigma_w (UQ F2, frozen SIGW). Primary R_tot = SD30 / sqrt(w^2 + sigma_w^2), band chi2_29.
usage (container): python3 read_gr.py [--dry <TAG>]"""
import sys, numpy as np
from scipy import stats
sys.path.insert(0, '/sdf/scratch/neutrino/pgranger/review_2026_09/GNPROF')
sys.path.insert(0, '/sdf/scratch/neutrino/pgranger/review_2026_09/UQ')
from read_gn import load, sample, breads, width, J_of, fmt, P, LAMS, M, V
SAMPLES = (29, 30, 31, 32)
REF30 = np.array([0.156, 0.043, 1.323, 5.245, 3.738])      # UQ/polish_table.txt, SD of m, n = 30 (frozen 6 Nov)
SIGW = np.array([0.097, 0.006, 0.637, 2.459, 1.974])       # UQ/F2_result.txt, sigma_w of the 4 H-stage runs (frozen)
WTRUE = np.array([0.1004, 0.0033, 0.5927, 1.3408, 2.4356])  # GNPROF post hoc, clipped GN-C median width, true ends
LO, HI = np.sqrt(stats.chi2.ppf(0.025, 29) / 29), np.sqrt(stats.chi2.ppf(0.975, 29) / 29)


def clip_delta(F, D):
    """0 <= Delta <= F: generalised eigenvalues of D w.r.t. F clipped to [0, 1] (F first made PD)."""
    F = 0.5 * (F + F.T); D = 0.5 * (D + D.T)
    w, U = np.linalg.eigh(F); nF = int(np.sum(w <= 0))
    F = (U * np.maximum(w, 1e-9 * np.abs(w).sum())) @ U.T
    Lc = np.linalg.cholesky(F); Li = np.linalg.inv(Lc)
    r, Q = np.linalg.eigh(Li @ D @ Li.T)
    return F, Lc @ (Q * np.clip(r, 0, 1)) @ Q.T @ Lc.T, nF, int(np.sum(r < 0)), int(np.sum(r > 1))


def clipped(recs, v, l):
    Fs = np.zeros((5, 5)); Ds = np.zeros((5, 5)); Hs = np.zeros((5, 5)); c = np.zeros(4, int)
    for r in recs:
        sc = r.get('schur'); gn = sc and sc.get('gn')
        if not gn:
            continue
        x = gn[v]['res'].get(l)
        if not (x and x['ok']):
            continue
        F2, D2, nF, nlo, nhi = clip_delta(M(gn[v]['fcc']), M(x['delta']))
        Fs += F2; Ds += D2; Hs += M(sc['hcc']); c += [nF, nlo, nhi, 1]
    return Fs, Ds, Hs, c


def m_of(o):
    return o['e'] + 100 * o['jac'] * (-np.linalg.solve(0.5 * (o['H'] + o['H'].T), o['G'].sum(0)))


if __name__ == '__main__':
    if '--dry' in sys.argv:
        tag = sys.argv[sys.argv.index('--dry') + 1]; recs, truth = load(tag); o = sample(recs, truth)
        print(f'DRY RUN on {tag}: {o["nb"]} batch record(s) with a GN block')
        for v in V:
            for l in LAMS:
                Fs, Ds, Hs, c = clipped(recs, v, l)
                jac = o['jac'] if o['jac'] is not None else np.ones(5)
                print(f'  {v} ridge {l}: counts (F non-PD, rho<0, rho>1, batches) {c}; clipped Delta/F diag '
                      f'{fmt(np.diag(Ds) / np.diag(Fs), "{:7.3f}")}; mineig(F-D) {np.linalg.eigvalsh(Fs - Ds).min():.3g}')
        sys.exit(0)
    O, RC = {}, {}
    for s in SAMPLES:
        recs, truth = load(f'GR_EXP{s}'); O[s] = sample(recs, truth); RC[s] = recs
        assert O[s]['jac'] is not None, 'P0 has no jac'
    print(f'batches with a GN block per sample: {[O[s]["nb"] for s in SAMPLES]}; GN wall per batch median '
          f'{np.median(np.concatenate([O[s]["t"] for s in SAMPLES])):.0f} s')
    # REF30 check: recompute the n = 30 SD of m from the dumps, must equal the frozen values
    from uqt_common import end, e1
    ROWS = [(s, f'T1_EXP{s}_20k') for s in range(1, 7)] + [(s, f'D_EXP{s}') for s in range(17, 29)] + [(s, f'D_F{s}') for s in range(29, 41)]
    MP = {}
    for s, dn in ROWS:
        x = end(f'EXP{s}C75H'); y = e1(dn, x['truth'])
        if y['calib_only'] and y['it'] in (20000, 20100) and np.abs(y['e'] - x['last']).max() < 1e-3:
            MP[s] = y['e'] + y['d']
    sd30 = np.array(list(MP.values())).std(0, ddof=1)
    print(f'REF30 recomputed (n = {len(MP)}): {fmt(sd30)} | frozen {fmt(REF30)} -> {"OK" if np.allclose(sd30, REF30, atol=6e-4) else "MISMATCH"}')
    # V-b
    chk = {v: np.median(np.concatenate([np.array(O[s]['chk'][v]) for s in SAMPLES]), 0) for v in V}
    vb = {v: bool(np.all(chk[v] < 1e-2)) for v in V}
    for v in V:
        print(f'V-b {v}: median gram/vjp {chk[v][0]:.1e}, asym_gg {chk[v][1]:.1e}, asym_cg {chk[v][2]:.1e} -> {"PASS" if vb[v] else "FAIL"}')
    # V-a at P0 for model C (the pre-registered model): sum F_cc / sum FD H_cc diag, median over samples, Ab tran long lifetime
    jP = [0, 2, 3, 4]
    rr = {v: np.median([np.diag(O[s]['F'][v]) / np.diag(O[s]['H']) for s in SAMPLES], 0) for v in V}
    for v in V:
        print(f'V-a {v}: F_cc(P0) / FD H_cc(P0) diag {fmt(rr[v], "{:7.3f}")}')
    va = bool(np.all((rr['C'][jP] > 0.8) & (rr['C'][jP] < 1.25)))
    # V2: the run sits at the production point: m from this run = the POLISH m from the D_F dump within 0.05 conditional sigma
    m = {s: m_of(O[s]) for s in SAMPLES}
    sc = {s: width(O[s]['H'], J_of(O[s]['G']), O[s]['jac']) for s in SAMPLES}
    sigc = {s: 100 * O[s]['jac'] * np.sqrt(np.diag(np.linalg.inv(0.5 * (O[s]['H'] + O[s]['H'].T)))) for s in SAMPLES}
    ok2 = [bool(np.all(np.abs(m[s] - MP[s]) < 0.05 * sigc[s])) for s in SAMPLES]
    for s in SAMPLES:
        print(f'  V2 s{s}: m here {fmt(m[s], "{:+8.3f}")} | POLISH m {fmt(MP[s], "{:+8.3f}")} | |dm|/sigma_c {fmt(np.abs(m[s] - MP[s]) / sigc[s], "{:6.3f}")}')
    v2 = sum(ok2) >= 3
    print(f'V2 reproduction of the production POLISH m (within 0.05 sigma_c on all 5, >= 3/4): {sum(ok2)}/4 -> {"PASS" if v2 else "FAIL"}')
    # V-c: clipped GN-C widths ridge 0 vs 1e-3 within 10 % on all 5, >= 3/4; ridge-0 solves ok on >= 95 % of batches
    W = {l: {} for l in LAMS}; CNT = {}
    for s in SAMPLES:
        for l in LAMS:
            Fs, Ds, Hs, c = clipped(RC[s], 'C', l); W[l][s] = width(Fs - Ds, J_of(O[s]['G']), O[s]['jac']); CNT[(s, l)] = c
    okr = all(O[s]['ok'][('C', '0')] >= 0.95 * O[s]['nb'] for s in SAMPLES)
    st = [bool(np.all(np.abs(W['0.001'][s] / W['0'][s] - 1) < 0.10)) for s in SAMPLES]
    vc = okr and sum(st) >= 3
    for s in SAMPLES:
        print(f'  V-c s{s}: w(0) {fmt(W["0"][s])} | w(1e-3)/w(0) {fmt(W["0.001"][s] / W["0"][s], "{:6.3f}")} | counts {CNT[(s, "0")]}')
    print(f'V-c ridge-0 solves ok: {okr}; clipped widths ridge 0 vs 1e-3 within 10 %: {sum(st)}/4 -> {"PASS" if vc else "FAIL"}')
    # primary
    w = np.median([W['0'][s] for s in SAMPLES], 0); tot = np.sqrt(w ** 2 + SIGW ** 2)
    Rt = REF30 / tot; Rs = REF30 / w; n_in = int(sum(LO <= x <= HI for x in Rt))
    print(f'\nband [{LO:.2f}, {HI:.2f}] (chi2_29); widths = median over the 4 samples (%)')
    print('  ' + ' ' * 34 + ' '.join(f'{p:>9s}' for p in P))
    print(f'  {"REF30 SD of m (n = 30)":34s}' + fmt(REF30))
    print(f'  {"sigma_w (F2, H-stage scatter)":34s}' + fmt(SIGW))
    print(f'  {"w = clipped GN-C full, ridge 0":34s}' + fmt(w))
    print(f'  {"sqrt(w^2 + sigma_w^2)":34s}' + fmt(tot))
    print(f'  {"R_tot = REF30 / sqrt(w^2+sw^2)":34s}' + fmt(Rt, '{:9.2f}'))
    print(f'  {"R_stat = REF30 / w (descr.)":34s}' + fmt(Rs, '{:9.2f}'))
    print(f'  {"w realistic / w true ends (descr.)":34s}' + fmt(w / WTRUE, '{:9.2f}'))
    # descriptive: other breads and models, robust scale
    for v in V:
        for k in (f'GN-{v} full (F - D)', f'hybrid-{v} (H - D)', f'GN-{v} dE/dx only', f'GN-{v} geometry only'):
            wk = np.median([width(breads(O[s], v, '0')[k], J_of(O[s]['G']), O[s]['jac']) for s in SAMPLES], 0)
            print(f'  {k + " (unclipped)":34s}' + fmt(wk) + '  R_tot ' + fmt(REF30 / np.sqrt(wk ** 2 + SIGW ** 2), '{:5.2f}'))
    wc = np.median([sc[s] for s in SAMPLES], 0)
    print(f'  {"conditional sandwich (H)":34s}' + fmt(wc) + '  R_tot ' + fmt(REF30 / np.sqrt(wc ** 2 + SIGW ** 2), '{:5.2f}'))
    MPa = np.array(list(MP.values())); rob = 1.4826 * np.median(np.abs(MPa - np.median(MPa, 0)), 0)
    print(f'  {"robust scale of m, n = 30 (descr.)":34s}' + fmt(rob) + '  / sqrt(w^2+sw^2) ' + fmt(rob / tot, '{:5.2f}'))
    zs = np.array([(MP[s] - np.array(list(MP.values())).mean(0)) / np.sqrt(W['0'][s] ** 2 + SIGW ** 2) for s in SAMPLES])
    print(f'  per-sample z of m (mean of n = 30 removed) / sqrt(w_s^2 + sw^2):\n' + '\n'.join(f'     s{s} ' + fmt(z, '{:+7.2f}') for s, z in zip(SAMPLES, zs)))
    verdict = 'CONFIRMED' if n_in == 5 else ('PARTLY CONFIRMED' if n_in >= 3 else 'NOT CONFIRMED')
    ok = vb['C'] and va and vc and v2
    print(f'\nGNREAL VERDICT (R_tot in the band on {n_in}/5): {verdict if ok else "NO VERDICT (V-b, V-a, V-c or V2 failed)"}'
          f'  [V-b C {vb["C"]}, V-a C {va}, V-c {vc}, V2 {v2}]')
