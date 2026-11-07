"""Replica test readout (rule: REVIEW.md 21 Oct, stated before any result). Per sample s=1..3, runs V, Vb, Vc (patched)
and S (unpatched) on the same data. R_q = RMS(cross pairs X-S) / RMS(same-code pairs among V, Vb, Vc) over the 3 samples;
T = mean of R_q over the 5 calibration errors + position. Null: all 4^3 choices of which run per sample plays S.
NEUTRAL if p > 0.10; SHIFT if p <= 3/64; INCONCLUSIVE between."""
import glob, itertools, pickle, numpy as np
FR = '/sdf/scratch/neutrino/pgranger/fit_result_staging'
P = ['Ab', 'eField', 'tran_diff', 'long_diff', 'lifetime']; Q = P + ['pos']


def read(arm):
    d = pickle.load(open(sorted(glob.glob(f'{FR}/sci_full_{arm}/history_iter10000_*.pkl'))[0], 'rb'))
    v = {p: 100 * (np.ravel(d[p + '_iter'])[-1000:].mean() / np.ravel(d[p + '_target'])[0] - 1) for p in P}
    v['pos'] = 1e4 * float(np.median(np.asarray(d['pos_residual_iter'], float).ravel()[-1000:]))
    v['loss'] = float(np.mean(np.ravel(d['losses_iter'])[-1000:])); v['n'] = len(np.ravel(d['losses_iter']))
    return v


ARMS = lambda s: [f'KN10V{s}', f'KN10V{s}b', f'KN10V{s}c', f'KN10S{s}']   # index 3 = the unpatched run
R = {s: [read(a) for a in ARMS(s)] for s in (1, 2, 3)}
SD16 = {q: float(np.std([read(f'KN10S{k}')[q] for k in range(1, 17)], ddof=1)) for q in Q}


def stat(which):   # which[s] = index of the run playing "S" in sample s
    Rq = {}
    for q in Q:
        cross, same = [], []
        for i, s in enumerate((1, 2, 3)):
            x = [r[q] for r in R[s]]; o = which[i]; rest = [j for j in range(4) if j != o]
            cross += [x[j] - x[o] for j in rest]; same += [x[a] - x[b] for a, b in itertools.combinations(rest, 2)]
        Rq[q] = np.sqrt(np.mean(np.square(cross))) / np.sqrt(np.mean(np.square(same)))
    return float(np.mean(list(Rq.values()))), Rq


T, Rq = stat((3, 3, 3))
null = [stat(w)[0] for w in itertools.product(range(4), repeat=3)]
p = float(np.mean([t >= T - 1e-12 for t in null]))
for s in (1, 2, 3):
    print(f's{s} n_iter {[r["n"] for r in R[s]]}')
    for q in Q + ['loss']:
        print(f'   {q:9s} ' + '  '.join(f'{a[4:]:>6s} {r[q]:+9.3f}' for a, r in zip(ARMS(s), R[s])))
print('R_q: ' + '  '.join(f'{q} {Rq[q]:.2f}' for q in Q))
print(f'T = {T:.3f}; null 64 relabelings: median {np.median(null):.3f}, max {max(null):.3f}; p = {p:.3f} -> '
      + ('NEUTRAL' if p > 0.10 else 'SHIFT' if p <= 3 / 64 + 1e-12 else 'INCONCLUSIVE'))
print('same-data run-to-run SD (pooled within-sample, 3 patched runs, 6 dof) vs KN10S 16-sample SD:')
for q in Q:
    w = np.sqrt(np.mean([np.var([r[q] for r in R[s][:3]], ddof=1) for s in (1, 2, 3)]))
    print(f'   {q:9s} {w:8.3f} vs {SD16[q]:8.3f}  ratio {w / SD16[q]:.2f}')
for s in (1, 2, 3):
    L = [r['loss'] for r in R[s]]; print(f's{s} lowest last-1e3 loss: {ARMS(s)[int(np.argmin(L))]}  ' + ' '.join(f'{x:.2f}' for x in L))
