"""PROBCAL vs STOCHCAL readout (calibration only, true segments, samples 1/2/4; 10 000 iterations).
Rule (REVIEW, stated before results): the probabilistic likelihood is shown better if, paired by sample, its mean |error|
over the five parameters is smaller on 3/3 samples, or if the stochastic arm does not settle (any parameter moving > 2%
over the last 2000 iterations). Otherwise the two are equivalent within noise.
Endpoint = mean of the last 1000 iterations; ptp = peak-to-peak over the last 2000; also the iteration after which each
parameter stays within a band of its endpoint (THR, percentage points), as a convergence-speed measure."""
import glob, pickle, json, numpy as np
ST = '/sdf/scratch/neutrino/pgranger/fit_result_staging'; FR = '/sdf/group/neutrino/pgranger/larnd-sim-jax/fit_result'
P = ['Ab', 'eField', 'tran_diff', 'long_diff', 'lifetime']
THR = dict(Ab=0.05, eField=0.01, tran_diff=0.5, long_diff=1.0, lifetime=1.0)   # settle band, percentage points
f1 = lambda x: float(np.ravel(np.asarray(x))[0])
def read(d0):
    fs = glob.glob(f'{ST}/{d0}/history_iter*.pkl') + glob.glob(f'{FR}/{d0}/history_iter*.pkl')
    f = sorted(fs, key=lambda x: int(x.split('iter')[-1].split('_')[0]))[-1]
    d = pickle.load(open(f, 'rb')); n = d['total_iter']
    r = dict(n_iter=n)
    for p in P:
        a = np.array([f1(x) for x in d[f'{p}_iter']])[:10001]; t = f1(d[f'{p}_target'][0])
        e = 100 * (a / t - 1); end = e[-1000:].mean()
        r[p] = end; r[p + '_ptp'] = float(np.ptp(e[-2000:])); r[p + '_sd_tail'] = float(e[-1000:].std())
        inside = np.abs(e - end) <= THR[p]; last_out = np.where(~inside)[0]
        r[p + '_settle_iter'] = int(last_out[-1] + 1) if len(last_out) else 0
    return r
res = {}
for s in (1, 2, 4):
    pr, st = read(f'sci_ceiling_PROBCAL{s}'), read(f'sci_ceiling_STOCHCAL{s}'); res[s] = dict(prob=pr, stoch=st)
    for name, r in (('prob ', pr), ('stoch', st)):
        print(f"s{s} {name} (n={r['n_iter']}): " + ' '.join(f"{p} {r[p]:+.3f}" for p in P)
              + f" | mean|err| {np.mean([abs(r[p]) for p in P]):.3f} | max ptp {max(r[p+'_ptp'] for p in P):.2f}"
              + f" | settle(1%) " + ' '.join(f"{r[p+'_settle_iter']}" for p in P))
better = [np.mean([abs(res[s]['prob'][p]) for p in P]) < np.mean([abs(res[s]['stoch'][p]) for p in P]) for s in (1, 2, 4)]
unsettled = any(res[s]['stoch'][p + '_ptp'] > 2 for s in (1, 2, 4) for p in P)
print(f"\nprob mean|err| smaller on {sum(better)}/3 samples; stochastic unsettled (ptp>2%): {unsettled}")
print('VERDICT:', 'probabilistic better' if (all(better) or unsettled) else 'equivalent within noise')
for p in P:
    a = np.array([res[s]['prob'][p] for s in (1, 2, 4)]); b = np.array([res[s]['stoch'][p] for s in (1, 2, 4)])
    print(f"  {p:10s} prob {a.mean():+.3f} +- {a.std(ddof=1)/np.sqrt(3):.3f} (SD {a.std(ddof=1):.3f})   stoch {b.mean():+.3f} +- {b.std(ddof=1)/np.sqrt(3):.3f} (SD {b.std(ddof=1):.3f})")
json.dump({str(k): v for k, v in res.items()}, open('/sdf/scratch/neutrino/pgranger/review_2026_09/PROBSTOCH/probstoch.json', 'w'), indent=1)
