"""Robust convergence numbers for all recent fits.

ESTIMATOR (chosen given this campaign's failure modes):
  * central value  = MEDIAN over the last 20% of iterations, per seed.
      - the endpoint is a single sample of a noisy (and sometimes still-moving) trajectory;
        reading endpoints produced two false positives in this campaign.
      - median (not mean) because trajectories show occasional large spikes.
  * within-run scatter = 1.4826 * MAD over the same window (robust sigma).
      - tells you how noisy the trajectory is at read-out time.
  * residual DRIFT = median(last 10%) - median(the 80-90% window), in % points.
      - the convergence diagnostic: a settled fit has drift ~ 0. A biased walk does not.
      - this is what distinguishes "converged at X%" from "passing through X%".
  * across-seed spread = std of the per-seed central values (the usual error bar).

VERDICT: |drift| is compared with 3.0 points, the measured run-to-run reproducibility at
FIXED target (ANNEAL vs FRAMEONLY2 paired replicates). Drift below that is unresolvable.

Only len400 files are used, and exactly one file per seed -- the S1/S3 directories also
contain len50 runs and an unfiltered glob silently mixes them.
"""
import pickle, glob, re, os
import numpy as np

P = ['Ab', 'eField', 'tran_diff', 'long_diff', 'lifetime']
NOISE_PT = 3.0          # measured run-to-run reproducibility (percentage points)


def seed_files(d, require_len400=True):
    out = {}
    for f in sorted(glob.glob(d + '/history_iter*.pkl')):
        if require_len400 and 'len400' not in f:
            continue
        m = re.search(r'seed(\d+)', f)
        if not m:
            continue
        it = int(re.search(r'iter(\d+)', f).group(1))
        s = int(m.group(1))
        if s not in out or it > out[s][0]:
            out[s] = (it, f)
    return {s: f for s, (it, f) in out.items()}


def robust(v, t):
    """-> (central %, within-run robust sigma %, drift points)"""
    e = (np.asarray(v, float) / t - 1.0) * 100.0
    n = len(e)
    tail = e[int(n * 0.80):]
    central = float(np.median(tail))
    mad = float(np.median(np.abs(tail - central))) * 1.4826
    late = float(np.median(e[int(n * 0.90):]))
    early = float(np.median(e[int(n * 0.80):int(n * 0.90)]))
    return central, mad, late - early


def analyse(tag, d, require_len400=True):
    fs = seed_files(d, require_len400)
    if not fs:
        return None
    rows = {p: [] for p in P}
    drifts = {p: [] for p in P}
    scat = {p: [] for p in P}
    pos = []
    for s, f in sorted(fs.items()):
        h = pickle.load(open(f, 'rb'))
        for p in P:
            if p + '_iter' not in h:
                continue
            c, m, dr = robust(np.ravel(np.array(h[p + '_iter'])), np.ravel(h[p + '_target'])[0])
            rows[p].append(c); scat[p].append(m); drifts[p].append(dr)
        if 'pos_residual_iter' in h:
            pr = np.ravel(h['pos_residual_iter']) * 1e4
            pos.append(float(np.median(pr[int(len(pr) * .8):])))
    return dict(tag=tag, n=len(fs), rows=rows, drifts=drifts, scat=scat, pos=pos)


RUNS = [
    ('S1  true dEdx+pos',        'fit_result/sci_ceiling_noise_s1',      True),
    ('S2  +fitted dEdx',         'fit_result/sci_ceiling_noise_s2',      True),
    ('S3  +wrong frozen geom',   'fit_result/sci_floor_noise_s3',        True),
    ('S4  +fitted geom',         'fit_result/sci_full_noise_s4',         True),
    ('S3ANNEAL  frozen+anneal',  'fit_result/sci_floor_S3ANNEAL',        True),
    ('ANNEAL    seeds 0-2',      'fit_result/sci_full_noise_s4_anneal',  True),
    ('ANNEALMORE seeds 3-5',     'fit_result/sci_full_ANNEALMORE',       True),
    ('FRAMEONLY2 drift frame',   'fit_result/sci_full_FRAMEONLY2',       True),
    ('DRIFTW6b   w=1e6',         'fit_result/sci_full_DRIFTW6b',         True),
    ('DRIFTW7b   w=1e7',         'fit_result/sci_full_DRIFTW7b',         True),
    ('WIN (old best) 8k',        'fit_result/sci_full_noise_s4_WIN',     True),
]

res = [r for r in (analyse(*x) for x in RUNS) if r]

print('ROBUST CONVERGENCE TABLE  —  central = median of last 20% of iterations')
print('value ± spread   (spread = s.d. across seeds);   [drift] = residual movement in the tail, % points')
print('=' * 132)
hdr = f"{'run':26s} {'n':>2s} " + ' '.join(f'{p:>19s}' for p in P) + f" {'pos(µm)':>9s}"
print(hdr); print('-' * 132)
for r in res:
    cells = []
    for p in P:
        if not r['rows'][p]:
            cells.append(f"{'-':>19s}"); continue
        mu = np.mean(r['rows'][p]); sd = np.std(r['rows'][p]); dr = np.mean(np.abs(r['drifts'][p]))
        cells.append(f'{mu:+7.2f}±{sd:5.2f}[{dr:4.1f}]')
    ps = f"{np.mean(r['pos']):9.0f}" if r['pos'] else f"{'frozen':>9s}"
    print(f"{r['tag']:26s} {r['n']:2d} " + ' '.join(cells) + ps)

print()
print('CONVERGENCE VERDICT  (|tail drift| vs the %.1f-point run-to-run noise floor)' % NOISE_PT)
print('-' * 96)
print(f"{'run':26s} " + ' '.join(f'{p:>13s}' for p in P))
for r in res:
    cells = []
    for p in P:
        if not r['drifts'][p]:
            cells.append(f"{'-':>13s}"); continue
        d = np.mean(np.abs(r['drifts'][p]))
        cells.append(f"{'settled' if d < NOISE_PT else 'MOVING':>9s}{d:4.1f}")
    print(f"{r['tag']:26s} " + ' '.join(cells))

print()
print('ENDPOINT vs ROBUST (how much the choice of estimator matters, lifetime only)')
print('-' * 78)
for r in res:
    fs = seed_files(dict(RUNS_MAP := {t: d for t, d, _ in RUNS})[r['tag']])
    ends, cents = [], []
    for s, f in sorted(fs.items()):
        h = pickle.load(open(f, 'rb'))
        v = np.ravel(np.array(h['lifetime_iter'])); t = np.ravel(h['lifetime_target'])[0]
        ends.append((v[-1] / t - 1) * 100)
        cents.append(robust(v, t)[0])
    print(f"  {r['tag']:26s} endpoint {np.mean(ends):+7.2f}   robust {np.mean(cents):+7.2f}   "
          f"shift {np.mean(cents)-np.mean(ends):+6.2f} pts")
