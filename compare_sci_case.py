#!/usr/bin/env python3
"""Science-case analysis: does fitting track position improve Ab (and other) recovery?

Compares three matched modes (same v12 config, differ only in position handling):
  floor   - linear_guess positions, FROZEN (no position fit)
  full    - linear_guess positions, FITTED via Adam chain
  ceiling - TRUE positions, frozen (best case)

The scientific claim is proven iff, across seeds:  |Ab_floor| > |Ab_full| -> |Ab_ceiling|

Run inside the container:
  JAX_PLATFORMS=cpu apptainer exec -B /sdf,/fs,/lscratch larndsim-jax_main.sif \
    bash -c 'PYTHONPATH=$PWD/src:$PWD python3 compare_sci_case.py'
"""
import pickle, glob, os
import numpy as np

PARS = ['Ab', 'eField', 'tran_diff', 'long_diff', 'lifetime']
MODES = ['floor', 'full', 'ceiling']
TAIL = 300  # time-average window (methodology: never single end-points)


def load(p):
    with open(p, 'rb') as f:
        return pickle.load(f)


def bias_row(h, n=TAIL):
    out = {}
    for k in PARS:
        it = np.asarray(h[k + '_iter'], dtype=float)
        tgt = float(np.asarray(h[k + '_target']).ravel()[0])
        val = float(np.mean(it[-n:]))
        drift = float(np.mean(it[-n:]) - np.mean(it[-2 * n:-n]))  # tail drift check
        out[k] = (100 * (val - tgt) / tgt, 100 * drift / tgt)
    if 'pos_residual_iter' in h:
        out['pos_resid_cm'] = (float(np.mean(np.asarray(h['pos_residual_iter'])[-n:])), 0.0)
    it = np.asarray(h['Ab_iter'], dtype=float)
    out['_niter'] = len(it)
    return out


def collect(mode):
    paths = sorted(glob.glob(f'fit_result/sci_{mode}/history_iter*_sci_{mode}_*seed*.pkl'))
    rows = []
    for p in paths:
        try:
            rows.append((os.path.basename(p), bias_row(load(p))))
        except Exception as e:
            print(f'  [skip] {os.path.basename(p)}: {e}')
    return rows


def main():
    summary = {}
    for mode in MODES:
        rows = collect(mode)
        print(f'\n=== {mode.upper()}  (n={len(rows)} seeds) ===')
        if not rows:
            print('  (no results yet)')
            continue
        agg = {}
        for k in PARS + ['pos_resid_cm']:
            vals = [r[k][0] for _, r in rows if k in r]
            drifts = [r[k][1] for _, r in rows if k in r]
            if not vals:
                continue
            m, s = np.mean(vals), np.std(vals)
            dr = np.mean(np.abs(drifts))
            agg[k] = (m, s)
            unit = 'cm' if k == 'pos_resid_cm' else '%'
            flag = '  <-- DRIFTING' if (unit == '%' and dr > 0.5) else ''
            print(f'   {k:12s} {m:+7.2f} ± {s:5.2f} {unit}   (|tail drift| {dr:.2f}){flag}')
        summary[mode] = agg

    # Headline verdict on Ab
    print('\n' + '=' * 60)
    print('SCIENCE-CASE VERDICT (Ab recombination bias):')
    for mode in MODES:
        if mode in summary and 'Ab' in summary[mode]:
            m, s = summary[mode]['Ab']
            print(f'   {mode:8s}  |Ab bias| = {abs(m):5.2f}%   (Ab bias {m:+.2f} ± {s:.2f}%)')
    if all(m in summary and 'Ab' in summary[m] for m in MODES):
        fa = abs(summary['floor']['Ab'][0])
        fu = abs(summary['full']['Ab'][0])
        ce = abs(summary['ceiling']['Ab'][0])
        ok = fa > fu >= ce - 1e-9
        print(f"\n   Ordering |floor|>|full|->|ceiling|: {fa:.2f} > {fu:.2f} -> {ce:.2f}  "
              f"=> {'CONFIRMED' if ok else 'NOT confirmed'}")
        print(f"   Position fitting recovers {fa - fu:+.2f}% of Ab bias "
              f"({100 * (fa - fu) / max(fa - ce, 1e-9):.0f}% of the floor->ceiling gap)")

    _plot(summary)


def _plot(summary):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f'\n[plot skipped: {e}]')
        return
    metrics = ['Ab', 'long_diff', 'lifetime']
    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 4))
    colors = {'floor': '#d62728', 'full': '#1f77b4', 'ceiling': '#2ca02c'}
    for ax, k in zip(axes, metrics):
        xs, means, errs, cs = [], [], [], []
        for i, mode in enumerate(MODES):
            if mode in summary and k in summary[mode]:
                m, s = summary[mode][k]
                xs.append(i); means.append(m); errs.append(s); cs.append(colors[mode])
        ax.bar(xs, means, yerr=errs, color=cs, capsize=4)
        ax.axhline(0, color='k', lw=0.8)
        ax.set_xticks(range(len(MODES)))
        ax.set_xticklabels(MODES, rotation=15)
        ax.set_title(f'{k} bias [%]')
        ax.grid(axis='y', alpha=0.3)
    fig.suptitle('Science case: effect of position handling on calibration recovery '
                 f'(time-avg last {TAIL} iters)')
    fig.tight_layout()
    out = 'plots/sci_case_comparison.png'
    os.makedirs('plots', exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f'\n[saved {out}]')


if __name__ == '__main__':
    main()
