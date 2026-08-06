"""Read out the likelihood-scan quality ladder.

Each run is a `--fit_type scan --scan_tgt_nom` job: the 5 calibration parameters are scanned one
at a time, on every batch, around the nominal point (which IS the target here). The history stores
one row per (batch, param, scan-step) in that nesting order, so

    block k  ->  batch = k // 5,  param = ORDER[k % 5],  rows = k*21 .. k*21+20

and the scanned value for those rows is `<param>_iter[1:][rows]` (index 0 is the init).
Do NOT identify blocks by masking on "all other params at nominal" -- that leaks rows from
neighbouring blocks whenever a scan grid passes through nominal.

TWO FILE-LAYOUT TRAPS:
 * `history_<param>_batch<N>_<label>.pkl` is a SNAPSHOT of the single growing history, dumped when
   the scan reaches (batch N, param). The files are therefore NOT interchangeable: the `Ab` file
   stops 4 blocks short of the `lifetime` file. Always read the LAST param in ORDER.
 * `--max_nbatch 30` was silently clamped to `iterations`=21 by example_run.py.

The total NLL is the SUM over batches, so the minimum is located on the summed curve. Uncertainty
is a bootstrap over batches -- the 3 job "seeds" are NOT independent noise draws: with
`--probabilistic_sim` the forward model is analytic, and the three seeds agree to <1 part in 1e4
of the loss, i.e. they are one measurement repeated, not three.
"""
import os, pickle
import numpy as np

ORDER = ['Ab', 'eField', 'tran_diff', 'long_diff', 'lifetime']
NSTEP = 21
NOM = {'Ab': 0.8, 'eField': 0.5, 'tran_diff': 8.8e-6, 'long_diff': 4.0e-6, 'lifetime': 2200.0}
RES = 'fit_result/loss_profile'
TAGS = ['true', 'pos50', 'pos170', 'pos400', 'pos880',
        'dedx0.75', 'dedx0.5', 'dedx0.25', 'dedx0.0', 'guess']
SEEDS = [0, 1, 2]


def load_run(tag, seed, nstep=None):
    """Return {param: (x_grid, per_batch_loss[nbatch, nstep])}. Reads the most complete snapshot.

    The batch index in the filename varies (fine scans run 41 batches, a killed run stops early),
    so glob and take the highest -- that is also the most complete history, since each file is a
    snapshot of the single growing history rather than a separate record.
    """
    import glob as _glob
    cand = _glob.glob(f'{RES}/history_{ORDER[-1]}_batch*_prof_{tag}_seed{seed}.pkl')
    if not cand:
        return None
    f = max(cand, key=lambda p: int(p.split('_batch')[1].split('_')[0]))
    d = pickle.load(open(f, 'rb'))
    loss = np.asarray(d['losses_iter'], float)
    # step count = iterations, recoverable from the run's own argv rather than assumed
    if nstep is None:
        pv = d['provenance']; pv = pv.item() if hasattr(pv, 'item') else pv
        av = pv['argv']; nstep = int(av[av.index('--iterations') + 1])
    global NSTEP
    NSTEP = nstep
    nblock = len(loss) // NSTEP
    out = {}
    for p in ORDER:
        v = np.asarray(d[p + '_iter'], float)[1:]
        grids, curves = [], []
        for k in range(nblock):
            if ORDER[k % 5] != p:
                continue
            sl = slice(k * NSTEP, (k + 1) * NSTEP)
            grids.append(v[sl]); curves.append(loss[sl])
        g = np.array(grids)
        assert np.allclose(g, g[0]), f'{p}: scan grid differs between batches'
        out[p] = (g[0], np.array(curves))
    return out


def vertex5(x, y):
    """5-point parabola vertex -> x_min (nan at the edges). Used ONLY to size the grid systematic.

    The scan grid is very coarse next to the likelihood width (the lifetime step is 10.2% of truth
    while the Fisher sigma is 0.88%, so the argmin's NEIGHBOURS already sit ~6-10 sigma out). The
    vertex is therefore an extrapolation and depends on how many points you fit. Comparing the 3-
    and 5-point estimates measures that dependence: it is ~+2.4 points on lifetime and <~1 point
    on long_diff, and on lifetime it is almost perfectly COMMON-MODE across conditions, so it
    cancels in the condition-to-condition differences the ladder is built from. Absolute minimum
    positions carry it as a systematic; ladder *differences* do not.
    """
    i = int(np.argmin(y))
    if i < 2 or i > len(y) - 3:
        return np.nan
    sl = slice(i - 2, i + 3)
    a, b, _ = np.polyfit(x[sl], y[sl], 2)
    return -b / (2 * a) if a > 0 else np.nan


def vertex(x, y):
    """3-point parabola vertex around the argmin -> (x_min, curvature, at_edge)."""
    i = int(np.argmin(y))
    if i == 0 or i == len(y) - 1:
        return x[i], np.nan, True
    x0, x1, x2 = x[i - 1], x[i], x[i + 1]
    y0, y1, y2 = y[i - 1], y[i], y[i + 1]
    den = (x0 - x1) * (x0 - x2) * (x1 - x2)
    a = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / den
    b = (x2 * x2 * (y0 - y1) + x1 * x1 * (y2 - y0) + x0 * x0 * (y1 - y2)) / den
    if a <= 0:
        return x[i], np.nan, True
    return -b / (2 * a), 2 * a, False


def analyse(tag, seed=0, nboot=400, rng=None):
    r = load_run(tag, seed)
    if r is None:
        return None
    rng = rng or np.random.default_rng(0)
    res = {}
    for p in ORDER:
        x, C = r[p]                                   # C: [nbatch, NSTEP]
        nb = len(C)
        xm, curv, edge = vertex(x, C.sum(0))
        # bootstrap over batches
        boot = []
        for _ in range(nboot):
            idx = rng.integers(0, nb, nb)
            bxm, _, be = vertex(x, C[idx].sum(0))
            if not be:
                boot.append(bxm)
        x5 = vertex5(x, C.sum(0))
        grid_sys = 100.0 * abs(x5 - xm) / NOM[p] if np.isfinite(x5) else np.nan
        pct = 100.0 * (xm - NOM[p]) / NOM[p]
        err = 100.0 * np.std(boot, ddof=1) / NOM[p] if len(boot) > 5 else np.nan
        # Fisher 1-sigma from the NLL curvature (loss is a PPP negative log-likelihood):
        # d2(NLL)/dx2 = 1/sigma^2
        sig = 100.0 * (1.0 / np.sqrt(curv)) / NOM[p] if np.isfinite(curv) and curv > 0 else np.nan
        res[p] = dict(pct=pct, boot=err, fisher=sig, grid_sys=grid_sys, edge=edge, nbatch=nb,
                      step_pct=100.0 * (x[1] - x[0]) / NOM[p],
                      lo_pct=100.0 * (x[0] - NOM[p]) / NOM[p],
                      hi_pct=100.0 * (x[-1] - NOM[p]) / NOM[p])
    return res


if __name__ == '__main__':
    import json
    rng = np.random.default_rng(0)
    table = {t: analyse(t, 0, rng=rng) for t in TAGS}

    g = table['true']
    print('SCAN GRID (identical for every condition)')
    print(f"{'param':11s}{'range vs nominal':>22s}{'step':>9s}{'batches':>9s}")
    for p in ORDER:
        print(f"{p:11s}{g[p]['lo_pct']:>+9.2f}% .. {g[p]['hi_pct']:>+7.2f}%"
              f"{g[p]['step_pct']:>8.3f}%{g[p]['nbatch']:>9d}")

    print('\nMINIMUM LOCATION, % offset from truth  (+/- = bootstrap over batches)')
    hdr = f"{'condition':10s}" + ''.join(f'{p:>19s}' for p in ORDER)
    print(hdr); print('-' * len(hdr))
    for t in TAGS:
        row = f'{t:10s}'
        for p in ORDER:
            d = table[t][p]
            row += f"{d['pct']:>+11.2f}±{d['boot']:5.2f}" + ('*' if d['edge'] else ' ')
        print(row)
    print('* minimum sat at a grid edge -> value is the edge, a lower bound on the offset')

    print('\nFISHER 1-sigma of the likelihood itself (statistical resolution of this dataset, %)')
    row = f"{'true':10s}"
    for p in ORDER:
        row += f"{table['true'][p]['fisher']:>19.3f}"
    print(row)

    print('\nSEED CHECK (probabilistic_sim -> analytic forward model, seeds are not independent)')
    for t in ['true', 'guess']:
        v = [analyse(t, s, nboot=0, rng=rng)['lifetime']['pct'] for s in SEEDS]
        print(f'  {t:6s} lifetime min over seeds 0/1/2: ' + ', '.join(f'{x:+.3f}%' for x in v))

    json.dump({t: {p: {k: (None if isinstance(x, float) and not np.isfinite(x) else
                          (bool(x) if isinstance(x, (bool, np.bool_)) else float(x)))
                       for k, x in d.items()} for p, d in table[t].items()} for t in TAGS},
              open('quality_ladder_summary.json', 'w'), indent=1)
