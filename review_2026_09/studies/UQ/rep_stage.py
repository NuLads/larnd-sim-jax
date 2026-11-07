"""UQ (descriptive): where do same-recipe replicas (production KN10N<s> vs the 29 Oct re-run KN10N<s>P, s = 3, 5..9)
separate? Signed-difference SD of the error (%) at the end of each stage (mean of the 200 iterations before t), and the
sample-to-sample SD of the production runs at the same t."""
import glob, pickle, numpy as np
FR = '/sdf/group/neutrino/pgranger/larnd-sim-jax/fit_result'
P = ['Ab', 'eField', 'tran_diff', 'long_diff', 'lifetime']
f1 = lambda x: float(np.mean(x))
def hist(run):
    d = pickle.load(open(sorted(glob.glob(f'{FR}/sci_full_{run}/history_iter20000_*.pkl'))[0], 'rb'))
    return {p: 100 * (np.array([f1(x) for x in d[f'{p}_iter']]) / f1(d[f'{p}_target'][0]) - 1) for p in P}
S = [3, 5, 6, 7, 8, 9]
O = {s: hist(f'KN10N{s}H') for s in S}; R = {s: hist(f'KN10N{s}PH') for s in S}
A = {s: hist(f'KN10N{s}H') for s in range(1, 17)}
T = [2000, 5000, 10000, 12000, 14000, 16500, 18000, 20000]
w = lambda v, t: v[t - 200:t].mean()
print('replica signed-diff SD / sqrt(2) (= per-run optimisation scatter)  ||  16-sample SD (production)')
print('     t ' + ' '.join(f'{p:>10s}' for p in P) + '   || ' + ' '.join(f'{p:>10s}' for p in P))
for t in T:
    so = [np.std([w(R[s][p], t) - w(O[s][p], t) for s in S], ddof=1) / np.sqrt(2) for p in P]
    sd = [np.std([w(A[s][p], t) for s in A], ddof=1) for p in P]
    print(f'{t:6d} ' + ' '.join(f'{v:10.3f}' for v in so) + '   || ' + ' '.join(f'{v:10.3f}' for v in sd))
