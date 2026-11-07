"""Adopted-package validation readout (rule: REVIEW.md 22 Oct, stated before the result). KN10P2 = live tree (KN10P1 was invalid: stale SPEED/stage overlay) with the
bitwise opt package (defaults ON), same data/config as KN10V1/V1b/V1c/S1. Same-data SD pooled over samples 1-3 (4 runs
each, 9 dof). z_q = (P1 - mean of the 4 sample-1 runs) / (SD_pool * sqrt(1 + 1/4)) for the 5 calibration errors + pos.
CONSISTENT if max |z| < 3; SHIFT if any |z| > 4; INCONCLUSIVE between."""
import glob, itertools, pickle, numpy as np
FR = '/sdf/scratch/neutrino/pgranger/fit_result_staging'
P = ['Ab', 'eField', 'tran_diff', 'long_diff', 'lifetime']; Q = P + ['pos']


def read(arm):
    d = pickle.load(open(sorted(glob.glob(f'{FR}/sci_full_{arm}/history_iter10000_*.pkl'))[0], 'rb'))
    v = {p: 100 * (np.ravel(d[p + '_iter'])[-1000:].mean() / np.ravel(d[p + '_target'])[0] - 1) for p in P}
    v['pos'] = 1e4 * float(np.median(np.asarray(d['pos_residual_iter'], float).ravel()[-1000:]))
    v['loss'] = float(np.mean(np.ravel(d['losses_iter'])[-1000:])); v['n'] = len(np.ravel(d['losses_iter']))
    return v


ARMS = lambda s: [f"KN10V{s}", f"KN10V{s}b", f"KN10V{s}c", f"KN10S{s}"]
Q = P + ["pos"]
R = {s: [read(a) for a in ARMS(s)] for s in (1, 2, 3)}
sd = {q: np.sqrt(np.mean([np.var([r[q] for r in R[s]], ddof=1) for s in (1, 2, 3)])) for q in Q}
import sys
ARM = sys.argv[1] if len(sys.argv) > 1 else "KN10P2"
p1 = read(ARM); print(ARM, "n_iter", p1["n"])
z = {}
for q in Q:
    m = np.mean([r[q] for r in R[1]]); z[q] = (p1[q] - m) / (sd[q] * np.sqrt(1.25))
    print(f"{q:9s} P1 {p1[q]:+9.3f}  mean4 {m:+9.3f}  SD_same {sd[q]:7.3f}  z {z[q]:+5.2f}")
mz = max(abs(v) for v in z.values())
print("max |z| = %.2f ->" % mz, "CONSISTENT" if mz < 3 else ("SHIFT" if mz > 4 else "INCONCLUSIVE"))
