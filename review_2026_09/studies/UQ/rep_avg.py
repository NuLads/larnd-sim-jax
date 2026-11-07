"""UQ (31 Oct, exploratory, no rule): does averaging two same-recipe replicas (production O and the 29 Oct re-run P, same
data) beat a single fit against truth? 6 pairs (samples 3, 5..9), H @ 20000, last-1000 mean errors (%)."""
import glob, pickle, numpy as np
FR = '/sdf/group/neutrino/pgranger/larnd-sim-jax/fit_result'
P = ['Ab', 'eField', 'tran_diff', 'long_diff', 'lifetime']
f1 = lambda x: float(np.mean(x))
def rd(run, it=20000):
    d = pickle.load(open(sorted(glob.glob(f'{FR}/sci_full_{run}/history_iter{it}_*.pkl'))[0], 'rb'))
    return np.array([100 * (np.mean([f1(x) for x in d[f'{p}_iter'][-1000:]]) / f1(d[f'{p}_target'][0]) - 1) for p in P])
S = [3, 5, 6, 7, 8, 9]
O = np.array([rd(f'KN10N{s}H') for s in S]); Pr = np.array([rd(f'KN10N{s}PH') for s in S]); A = 0.5 * (O + Pr)
allO = np.array([rd(f'KN10N{s}H') for s in range(1, 17)])
so = (Pr - O).std(0, ddof=1) / np.sqrt(2)
print(f"{'':10s} {'SD single (O,P pooled)':>22s} {'SD mean(O,P)':>13s} {'pred sqrt(SD^2-so^2/2)':>23s} {'RMS single':>11s} {'RMS mean':>9s} {'mean':>7s}")
for j, p in enumerate(P):
    sd1 = np.sqrt(0.5 * (O[:, j].var(ddof=1) + Pr[:, j].var(ddof=1))); sdA = A[:, j].std(ddof=1)
    pred = np.sqrt(max(sd1 ** 2 - so[j] ** 2 / 2, 0))
    rms1 = np.sqrt(0.5 * (np.mean(O[:, j] ** 2) + np.mean(Pr[:, j] ** 2))); rmsA = np.sqrt(np.mean(A[:, j] ** 2))
    print(f"{p:10s} {sd1:22.3f} {sdA:13.3f} {pred:23.3f} {rms1:11.3f} {rmsA:9.3f} {A[:, j].mean():+7.3f}")
print('(SD of the 16-sample production ensemble: ' + ' '.join(f'{p} {v:.3f}' for p, v in zip(P, allO.std(0, ddof=1))) + ')')
