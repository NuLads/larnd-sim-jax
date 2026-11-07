"""Round 2 readout: recipe R2 (STRFRZ50) at true lifetime 3200 and 4400 us, samples 1/3/7, paired with the production
(straight-start) CURVE fits on the same samples; plus the R2 curve at 1200 and 2406 us for the lifetime-dependence.
Rule (REVIEW section 0c, stated before results): mean |lifetime error| over the 3 samples <= 5% at 3200 and <= 7% at 4400.
Endpoint = mean of the last 1000 iterations (of 10 000); ptp = peak-to-peak over the last 2000 (settling check)."""
import glob, pickle, csv, json, numpy as np
ST = '/sdf/scratch/neutrino/pgranger/fit_result_staging'; FR = '/sdf/group/neutrino/pgranger/larnd-sim-jax/fit_result'
P = ['Ab', 'eField', 'tran_diff', 'long_diff', 'lifetime']
f1 = lambda x: float(np.ravel(np.asarray(x))[0])
def read(d0):
    fs = glob.glob(f'{ST}/{d0}/history_iter*.pkl') + glob.glob(f'{FR}/{d0}/history_iter*.pkl')
    f = sorted(fs, key=lambda x: int(x.split('iter')[-1].split('_')[0]))[-1]
    d = pickle.load(open(f, 'rb')); assert d['total_iter'] >= 10000, (d0, d['total_iter'])
    r = {}
    for p in P:
        a = np.array([f1(x) for x in d[f'{p}_iter']])[:10001]; t = f1(d[f'{p}_target'][0])
        r[p] = 100 * (a[-1000:].mean() / t - 1); r[p + '_ptp'] = 100 * np.ptp(a[-2000:]) / t; r[p + '_true'] = t
    return r
cur = {r['tag']: r for r in csv.DictReader(open('/sdf/group/neutrino/pgranger/larnd-sim-jax/review_2026_09/curve_scan_per_run.csv'))}
inv = lambda err, tau: 1e5 * (1 / (tau * (1 + err / 100)) - 1 / tau)          # Delta(1/tau) in 1e-5 / us
out = {}
for T, rule in ((3200, 5.0), (4400, 7.0)):
    rows = []
    for s in (1, 3, 7):
        a = read(f'sci_full_STRFRZ50T{T}D{s}'); c = cur[f'CURVE{T}D{s}']
        prod = {p: float(c[f'{p}_err_pct']) for p in P}
        rows.append((s, a, prod))
        print(f"tau={T} s{s}: R2 " + ' '.join(f"{p} {a[p]:+.2f}" for p in P) + f"  max ptp {max(a[p+'_ptp'] for p in P):.2f}"
              f" | production lifetime {prod['lifetime']:+.2f} long_diff {prod['long_diff']:+.2f}")
    lt = np.array([r[1]['lifetime'] for r in rows]); ld = np.array([r[1]['long_diff'] for r in rows]); pr = np.array([r[2]['lifetime'] for r in rows])
    m_abs = np.abs(lt).mean(); verdict = 'PASS' if m_abs <= rule else 'FAIL'
    print(f"  tau={T}: R2 lifetime {lt.mean():+.2f} +- {lt.std(ddof=1)/np.sqrt(3):.2f} (mean |err| {m_abs:.2f} vs rule {rule}: {verdict}); "
          f"long_diff {ld.mean():+.2f} +- {ld.std(ddof=1)/np.sqrt(3):.2f}; production lifetime {pr.mean():+.2f}; "
          f"paired R2-production {np.mean(lt-pr):+.2f} +- {np.std(lt-pr, ddof=1)/np.sqrt(3):.2f}")
    print(f"  Delta(1/tau) [1e-5/us]: R2 " + ' '.join(f'{inv(x, T):+.2f}' for x in lt) + "  production " + ' '.join(f'{inv(x, T):+.2f}' for x in pr))
    out[T] = dict(r2={p: [r[1][p] for r in rows] for p in P}, prod={p: [r[2][p] for r in rows] for p in P}, verdict=verdict, mean_abs=m_abs)
# R2 lifetime curve over all true lifetimes (samples 1/3/7 where available)
print('\nR2 lifetime error vs true tau (samples 1/3/7):')
for T, tag in ((1200, 'STRFRZ50T1200D{}'), (2406, 'STRFRZ50{}'), (3200, 'STRFRZ50T3200D{}'), (4400, 'STRFRZ50T4400D{}')):
    v = []
    for s in (1, 3, 7):
        try: v.append(read('sci_full_' + tag.format(s))['lifetime'])
        except Exception as e: print('   missing', tag.format(s), repr(e)[:80])
    v = np.array(v)
    if len(v): print(f"  {T:5d} us: {' '.join(f'{x:+.2f}' for x in v)}  mean {v.mean():+.2f} +- {v.std(ddof=1)/np.sqrt(len(v)):.2f}   "
                     f"Delta(1/tau) {np.mean([inv(x, T) for x in v]):+.2f} e-5/us")
json.dump(out, open('/sdf/scratch/neutrino/pgranger/review_2026_09/QUEUE/round2.json', 'w'), indent=1)
