"""GNPROF post hoc #2 (AFTER the readout; descriptive, no verdict). For a PSD per-batch joint curvature the profile term obeys
0 <= Delta_b <= F_cc,b (profiling can only remove information, never more than the batch has). Model C is indefinite, so
single batches break the bound and blow up. Rule tested here: per batch, generalised eigenvalues rho of Delta_b w.r.t.
F_cc,b clipped to [0, 1] (needs F_cc,b PD; if not, F_cc,b is first clipped to PSD + 1e-9 x trace). Same score meat."""
import sys, numpy as np
sys.path.insert(0, '/sdf/scratch/neutrino/pgranger/review_2026_09/GNPROF')
from read_gn import load, sample, width, J_of, fmt, P, LAMS, PT, M
from scipy import stats
lo, hi = np.sqrt(stats.chi2.ppf(0.025, 7) / 7), np.sqrt(stats.chi2.ppf(0.975, 7) / 7)


def clip_delta(F, D):
    F = 0.5 * (F + F.T); D = 0.5 * (D + D.T)
    w, U = np.linalg.eigh(F); nF = int(np.sum(w <= 0))
    w = np.maximum(w, 1e-9 * np.abs(w).sum()); F = (U * w) @ U.T
    Lc = np.linalg.cholesky(F); Li = np.linalg.inv(Lc)
    r, Q = np.linalg.eigh(Li @ D @ Li.T)
    nlo, nhi = int(np.sum(r < 0)), int(np.sum(r > 1)); rc = np.clip(r, 0, 1)
    return F, Lc @ (Q * rc) @ Q.T @ Lc.T, nF, nlo, nhi, r


S = range(1, 9); O = {}; R = {}
for s in S:
    recs, truth = load(f'GN_KN10N{s}'); O[s] = sample(recs, truth); R[s] = recs
m = np.array([O[s]['e'] + 100 * O[s]['jac'] * (-np.linalg.solve(0.5 * (O[s]['H'] + O[s]['H'].T), O[s]['G'].sum(0))) for s in S])
sd = m.std(0, ddof=1)
for v in ('C', 'A'):
    W = {}; cnt = {}
    for l in LAMS:
        W[l] = []; cnt[l] = np.zeros(4, int)
        for s in S:
            Fs = np.zeros((5, 5)); Ds = np.zeros((5, 5)); Hs = np.zeros((5, 5))
            for r in R[s]:
                sc = r.get('schur'); gn = sc and sc.get('gn')
                if not gn:
                    continue
                g = gn[v]; x = g['res'].get(l)
                if not (x and x['ok']):
                    continue
                F2, D2, nF, nlo, nhi, _ = clip_delta(M(g['fcc']), M(x['delta']))
                Fs += F2; Ds += D2; Hs += M(sc['hcc']); cnt[l] += [nF, nlo, nhi, 1]
            W[l].append((width(Fs - Ds, J_of(O[s]['G']), O[s]['jac']), width(Hs - Ds, J_of(O[s]['G']), O[s]['jac'])))
        W[l] = np.array(W[l])
    print(f'=== model {v}, clipped Delta_b (rho in [0,1]); counts over all batches x samples: F_cc,b non-PD eigs, rho<0, rho>1, batches')
    for l in LAMS:
        print(f'  ridge {l:6s} counts {cnt[l]}')
    for k, lab in ((0, 'GN full'), (1, 'hybrid')):
        st = [bool(np.all(np.abs(W['0.001'][i, k] / W['0'][i, k] - 1) < 0.10)) for i in range(8)]
        st2 = [bool(np.all(np.abs(W['0.01'][i, k] / W['0'][i, k] - 1) < 0.10)) for i in range(8)]
        print(f'  {lab}: per-sample ridge stability 0 vs 1e-3 within 10 % on all 5: {sum(st)}/8; 0 vs 1e-2: {sum(st2)}/8')
        for i, s in enumerate(S):
            print(f'     s{s} w0 {fmt(W["0"][i, k])} | w1e-3/w0 {fmt(W["0.001"][i, k] / W["0"][i, k], "{:6.3f}")} | w1e-2/w0 {fmt(W["0.01"][i, k] / W["0"][i, k], "{:6.3f}")}')
        for l in LAMS:
            w = np.median(W[l][:, k], 0); rat = sd / w
            print(f'   ridge {l:6s} median width {fmt(w)} | SD(m)/width {fmt(rat, "{:6.2f}")}  in band {int(sum(lo <= x <= hi for x in rat))}/5')
