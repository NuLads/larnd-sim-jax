"""Readout of the SA block (REVIEW.md 22 Oct 11:45; written before any SA2-6 result): alias gate 500 um (SA<s>) vs
production KN10S<s>, fresh 1e4 fits, paired per sample. Verdicts on samples 2-6; sample 1 listed alongside.
Needs per_track_{KN10S,SA}<s>_10000.json (per_track_res.py) and the 1e4 checkpoints.
usage (in the container): python3 read_sa_block.py"""
import glob, json, pickle, numpy as np
D = '/sdf/scratch/neutrino/pgranger/review_2026_09/POLISH/joint'
FR = '/sdf/group/neutrino/pgranger/larnd-sim-jax/fit_result'
P = ['Ab', 'eField', 'tran_diff', 'long_diff', 'lifetime']
TRUTH = dict(Ab=0.83488135, eField=0.52151894, tran_diff=1.00276338e-05, long_diff=5.81418228e-06, lifetime=2406.44659703)


def tracks(run):
    a = json.load(open(f'{D}/per_track_{run}_10000.json'))
    r = np.array([x['r_mean'] for x in a]); n = np.array([x['n'] for x in a], float)
    return dict(out=int(np.sum(r > 200)), out500=int(np.sum(r > 500)), wmean=float(np.average(r, weights=n)), median=float(np.median(r)),
                p99=float(np.quantile(r, 0.99)))


def end(run):
    ck = pickle.load(open(sorted(glob.glob(f'{FR}/sci_full_{run}/history_iter10000_*.pkl'))[0], 'rb'))
    pr = np.array([np.nan if x is None else float(np.mean(x)) for x in ck['pos_residual_iter']])
    L = np.array([float(np.ravel(x)[0]) for x in ck['losses_iter']])
    return dict(err={k: 100 * (float(np.ravel(ck[k + '_iter'][-1])[0]) / TRUTH[k] - 1) for k in P},
                pr=1e4 * float(np.nanmean(pr[9000:10000])), L=float(np.mean(L[9000:10000])))


rows = {}
for s in range(1, 7):
    b, g = tracks(f'KN10S{s}'), tracks(f'SA{s}'); eb, eg = end(f'KN10S{s}'), end(f'SA{s}')
    rows[s] = (b, g, eb, eg)
    print(f's{s}: outliers {b["out"]:3d} -> {g["out"]:3d} (> 500 um {b["out500"]:2d} -> {g["out500"]:2d}); weighted mean |r| {b["wmean"]:6.1f} -> {g["wmean"]:6.1f} '
          f'(median {b["median"]:.0f} -> {g["median"]:.0f}, p99 {b["p99"]:.0f} -> {g["p99"]:.0f}); pos_residual 9-10k {eb["pr"]:.1f} -> {eg["pr"]:.1f}; '
          f'loss 9-10k {eb["L"]:.1f} -> {eg["L"]:.1f} | shift ' + ' '.join(f'{k} {eg["err"][k] - eb["err"][k]:+.2f}' for k in P))
V = [2, 3, 4, 5, 6]
lower = sum(rows[s][1]['out'] < rows[s][0]['out'] for s in V)
dm = np.array([rows[s][1]['wmean'] - rows[s][0]['wmean'] for s in V]); se = dm.std(ddof=1) / np.sqrt(len(V))
print(f'\nPOSITIONS (s2-6): outliers lower in {lower}/5; paired mean change of weighted mean |r| {dm.mean():+.1f} +- {se:.1f} um ({dm.mean() / se:+.1f} SE)')
print('  -> ' + ('GATE IMPROVES POSITIONS' if lower >= 4 and dm.mean() < -2 * se else 'NO POSITION GAIN' if abs(dm.mean()) <= 5 else 'unresolved'))
for k in ('lifetime', 'long_diff', 'Ab', 'tran_diff', 'eField'):
    d = np.array([rows[s][3]['err'][k] - rows[s][2]['err'][k] for s in V]); se = d.std(ddof=1) / np.sqrt(len(V))
    flag = (abs(d.mean()) > 2 and abs(d.mean()) > 2 * se) if k in ('lifetime', 'long_diff') else None
    print(f'CALIBRATION {k:9s}: paired shift {d.mean():+.2f} +- {se:.2f} (values {" ".join(f"{x:+.2f}" for x in d)})'
          + ('' if flag is None else f' -> {"CHANGES" if flag else "no gross change"}'))
