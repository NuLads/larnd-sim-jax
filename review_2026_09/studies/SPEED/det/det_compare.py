# Deterministic full-fit equivalence (rule: REVIEW.md 20 Oct): (a) each arm reproduces itself across jobs; (b) DET0 == DET1.
import glob, pickle, itertools, numpy as np
FR = '/sdf/scratch/neutrino/pgranger/fit_result_staging'
import sys; P = sys.argv[1] if len(sys.argv) > 1 else 'DET'
KEYS = ['losses_iter', 'Ab_iter', 'eField_iter', 'tran_diff_iter', 'long_diff_iter', 'lifetime_iter', 'pos_residual_iter']
H = {n: pickle.load(open(glob.glob(f'{FR}/sci_full_{n}/history_iter10_*.pkl')[0], 'rb')) for n in [P + s for s in ('0a', '0b', '1a', '1b')]}
def cmp(x, y):
    out = []
    for k in KEYS:
        a, b = np.ravel(np.asarray(H[x][k], dtype=np.float64)), np.ravel(np.asarray(H[y][k], dtype=np.float64))
        same = a.shape == b.shape and np.array_equal(a, b)
        out.append((k, len(a), same, 0.0 if same or a.shape != b.shape else float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300)))))
    return out
for x, y in [(P + x, P + y) for x, y in [('0a', '0b'), ('1a', '1b'), ('0a', '1a'), ('0b', '1b'), ('0a', '1b'), ('0b', '1a')]]:
    r = cmp(x, y); ok = all(s for _, _, s, _ in r)
    print(f'{x} vs {y}: {"BITWISE IDENTICAL" if ok else "DIFFER"}  ' + '  '.join(f'{k.replace("_iter", "")}[{n}] ' + ('=' if s else f'{d:.1e}') for k, n, s, d in r))
print('losses DET0a', np.ravel(H[P + '0a']['losses_iter'])[:10])
