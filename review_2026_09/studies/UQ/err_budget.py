"""UQ: empirical error budget at the production end point H @ 20000 (descriptive, existing runs). Errors = last-1000 mean
vs truth (%). Ensembles: pre-C75 production KN10N1-16 H; C75 KN10N{1..12}C75H; replica pairs: production KN10N<s>H vs the
29 Oct re-run KN10N<s>PH (s = 3, 5..9; same recipe) -> optimisation scatter sigma_opt = SD(diff) / sqrt(2)."""
import glob, pickle, numpy as np
FR = '/sdf/group/neutrino/pgranger/larnd-sim-jax/fit_result'
P = ['Ab', 'eField', 'tran_diff', 'long_diff', 'lifetime']
f1 = lambda x: float(np.mean(x))
def rd(run, it=20000):
    d = pickle.load(open(sorted(glob.glob(f'{FR}/sci_full_{run}/history_iter{it}_*.pkl'))[0], 'rb'))
    return np.array([100 * (np.mean([f1(x) for x in d[f'{p}_iter'][-1000:]]) / f1(d[f'{p}_target'][0]) - 1) for p in P])
prod = np.array([rd(f'KN10N{s}H') for s in range(1, 17)])
c75 = np.array([rd(f'KN10N{s}C75H') for s in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12) if s != 0 and glob.glob(f'{FR}/sci_full_KN10N{s}C75H/history_iter20000_*.pkl')])
reps = [3, 5, 6, 7, 8, 9]
dif = np.array([rd(f'KN10N{s}PH') - rd(f'KN10N{s}H') for s in reps])
so = dif.std(0, ddof=1) / np.sqrt(2)
print(f'n: production {len(prod)}, C75 {len(c75)}, replica pairs {len(dif)}')
print(f"{'':10s} {'prod mean':>10s} {'prod SD':>8s} {'C75 mean':>9s} {'C75 SD':>7s} {'sig_opt':>8s} {'opt/SD':>7s} {'sig_stat(prod)':>14s}")
for j, p in enumerate(P):
    sd = prod[:, j].std(ddof=1); st = np.sqrt(max(sd ** 2 - so[j] ** 2, 0))
    print(f'{p:10s} {prod[:, j].mean():+10.3f} {sd:8.3f} {c75[:, j].mean():+9.3f} {c75[:, j].std(ddof=1):7.3f} {so[j]:8.3f} {so[j] / sd:7.2f} {st:14.3f}')
