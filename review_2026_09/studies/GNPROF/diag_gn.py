"""GNPROF post hoc diagnostics (written AFTER the readout; descriptive only, no verdict). Why V-c failed: per-sample widths
of the GN-C full bread at ridges 0 / 1e-3 / 1e-2, per parameter; SD(m)/width at each ridge; per-batch contributions of
Delta at ridge 0 vs 1e-3 (outlier batches); non-positive eigenvalues of A_nn per batch; same for model A."""
import sys, numpy as np
sys.path.insert(0, '/sdf/scratch/neutrino/pgranger/review_2026_09/GNPROF')
from read_gn import load, sample, breads, width, J_of, fmt, P, LAMS, PT, M
S = range(1, 9); O = {}; R = {}
for s in S:
    recs, truth = load(f'GN_KN10N{s}'); O[s] = sample(recs, truth); R[s] = recs
m = np.array([O[s]['e'] + 100 * O[s]['jac'] * (-np.linalg.solve(0.5 * (O[s]['H'] + O[s]['H'].T), O[s]['G'].sum(0))) for s in S])
sd = m.std(0, ddof=1)
for v in ('C', 'A'):
    print(f'=== model {v}: GN full (F - D) widths per sample (%), ridges 0 / 1e-3 / 1e-2; ratio w(1e-3)/w(0)')
    W = {}
    for l in LAMS:
        W[l] = np.array([width(breads(O[s], v, l)[f'GN-{v} full (F - D)'], J_of(O[s]['G']), O[s]['jac']) for s in S])
    for i, s in enumerate(S):
        print(f'  s{s} ok {[O[s]["ok"][(v, l)] for l in LAMS]} | w0 {fmt(W["0"][i])} | w1e-3/w0 {fmt(W["0.001"][i] / W["0"][i], "{:6.3f}")}'
              f' | w1e-2/w0 {fmt(W["0.01"][i] / W["0"][i], "{:6.3f}")}')
    for l in LAMS:
        w = np.median(W[l], 0)
        print(f'  ridge {l:6s} median width {fmt(w)} | SD(m)/width {fmt(sd / w, "{:6.2f}")}')
    for l in LAMS:
        b = breads(O[1], v, l)
        print(f'  ridge {l:6s} hybrid  ' + fmt(sd / np.median([width(breads(O[s], v, l)[f"hybrid-{v} (H - D)"], J_of(O[s]["G"]), O[s]["jac"]) for s in S], 0), '{:6.2f}'))
    # the profiled fraction per ridge: diag Delta / diag F
    for l in LAMS:
        fr = np.median([np.diag(O[s]['D'][(v, l, 'delta')]) / np.diag(O[s]['F'][v]) for s in S], 0)
        print(f'  ridge {l:6s} Delta/F diag (median over samples) {fmt(fr, "{:7.3f}")}')
# per-batch: which batches move Delta between ridge 0 and 1e-3 (model C)
print('=== model C per batch: |Delta(0) - Delta(1e-3)| diag / F diag of the SAMPLE; top 5 batches per sample, and A_nn nneg')
for s in S:
    rows = []
    Fs = np.diag(O[s]['F']['C'])
    for j, r in enumerate(R[s]):
        sc = r.get('schur'); gn = sc and sc.get('gn')
        if not gn:
            continue
        g = gn['C']; x0 = g['res'].get('0'); x1 = g['res'].get('0.001')
        if not (x0 and x1 and x0['ok'] and x1['ok']):
            continue
        dd = (np.diag(M(x0['delta'])) - np.diag(M(x1['delta']))) / Fs
        rows.append((np.max(np.abs(dd)), j, dd, g['ann_nneg'], x0.get('minS'), x1.get('minS')))
    rows.sort(key=lambda t: -t[0])
    tot = np.sum([t[2] for t in rows], 0)
    print(f'  s{s} total shift {fmt(tot, "{:+7.3f}")} | nneg per batch: median {np.median([t[3] for t in rows]):.0f}, max {max(t[3] for t in rows)}')
    for t in rows[:5]:
        print(f'     batch {t[1]:3d} nneg {t[3]:2d} shift {fmt(t[2], "{:+7.3f}")} minS {t[4]} {t[5]}')
