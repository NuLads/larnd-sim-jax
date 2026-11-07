"""Iterate-averaging readout (rule: REVIEW.md 21 Oct, stated before computing): calibration error from the mean of the
last W iterations, W in 1000 (current) / 2000 / 3000 / 5000, on KN10S (16 @ 1e4), KN10C (16 @ 1.4e4) and the 12 same-data runs."""
import glob, pickle, numpy as np
FR = '/sdf/scratch/neutrino/pgranger/fit_result_staging'
P = ['Ab', 'eField', 'tran_diff', 'long_diff', 'lifetime']; WS = [1000, 2000, 3000, 5000]


def errs(arm, it):
    d = pickle.load(open(sorted(glob.glob(f'{FR}/sci_full_{arm}/history_iter{it}_*.pkl'))[0], 'rb'))
    return {W: {p: 100 * (np.ravel(d[p + '_iter'])[-W:].mean() / np.ravel(d[p + '_target'])[0] - 1) for p in P} for W in WS}


se = lambda v: np.std(v, ddof=1) / np.sqrt(len(v))
for name, arms, it in (('KN10S', [f'KN10S{s}' for s in range(1, 17)], 10000), ('KN10C', [f'KN10C{s}' for s in range(1, 17)], 14000)):
    E = [errs(a, it) for a in arms]
    print(f'\n{name} (16 @ {it}): mean, SD per window')
    base = {p: np.array([e[1000][p] for e in E]) for p in P}
    for W in WS:
        v = {p: np.array([e[W][p] for e in E]) for p in P}
        print(f'  W {W:5d}: ' + '  '.join(f'{p} {v[p].mean():+.3f},{np.std(v[p], ddof=1):.3f}' for p in P))
        if W > 1000:
            r = {p: np.std(v[p], ddof=1) / np.std(base[p], ddof=1) for p in ('lifetime', 'long_diff')}
            g = {p: (abs(v[p].mean()) - abs(base[p].mean())) / se(base[p]) for p in ('lifetime', 'long_diff')}
            vd = ('BETTER' if all(r[p] <= 0.9 and g[p] <= 1 for p in r) else 'WORSE' if any(r[p] >= 1.1 or g[p] > 2 for p in r) else 'NEUTRAL')
            print(f'           SD ratio lifetime {r["lifetime"]:.2f} long_diff {r["long_diff"]:.2f}; |mean| change (SE) lifetime {g["lifetime"]:+.2f} '
                  f'long_diff {g["long_diff"]:+.2f} -> {vd}')
runs = lambda s: [f'KN10V{s}', f'KN10V{s}b', f'KN10V{s}c', f'KN10S{s}']
R = {s: [errs(a, 10000) for a in runs(s)] for s in (1, 2, 3)}
print('\nsame-data SD (12 runs, pooled within sample) per window:')
for W in WS:
    print(f'  W {W:5d}: ' + '  '.join(f'{p} {np.sqrt(np.mean([np.var([e[W][p] for e in R[s]], ddof=1) for s in R])):.3f}' for p in P))
