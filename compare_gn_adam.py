#!/usr/bin/env python3
"""Compare Gauss-Newton/LM (fit_type=gn_calib) vs Adam (fit_type=chain) convergence on the
calibration-only, ground-truth-track problem.

Both fits share config/init/batches; only the optimizer differs. GN takes O(10) full-batch
iterations; Adam takes O(1000) per-batch iterations — so we report BOTH the trajectory (per
its own iteration axis) and the final bias, plus iterations-to-tolerance.

Run in the container:
  JAX_PLATFORMS=cpu apptainer exec -B /sdf,/fs,/lscratch larndsim-jax_main.sif \
    bash -c 'PYTHONPATH=$PWD/src:$PWD python3 compare_gn_adam.py'
"""
import pickle, glob, os
import numpy as np

PARS = ['Ab', 'eField', 'tran_diff', 'long_diff', 'lifetime']


def load_latest(test_name, label_sub):
    paths = glob.glob(f'fit_result/{test_name}/history_iter*_{label_sub}*.pkl')
    if not paths:
        return None
    # highest iter number
    p = max(paths, key=lambda s: int(s.split('history_iter')[1].split('_')[0]))
    with open(p, 'rb') as f:
        return pickle.load(f)


def bias_traj(h, k):
    it = np.asarray(h[k + '_iter'], dtype=float)
    tgt = float(np.asarray(h[k + '_target']).ravel()[0])
    return 100 * (it - tgt) / tgt


def iters_to_tol(h, tol_pct=1.0):
    """First iteration index where ALL params are within tol_pct of target (abs %)."""
    n = len(h['Ab_iter'])
    for i in range(n):
        if all(abs(bias_traj(h, k)[i]) < tol_pct for k in PARS):
            return i
    return None


def summarize(name, h):
    if h is None:
        print(f'\n=== {name}: (no results yet) ===')
        return
    n = len(h['Ab_iter'])
    print(f'\n=== {name}  ({n} logged iters) ===')
    for k in PARS:
        b = bias_traj(h, k)
        print(f'   {k:12s} final {b[-1]:+7.2f}%   (start {b[0]:+7.2f}%)')
    it1 = iters_to_tol(h, 1.0)
    print(f'   iters to all-within-1%: {it1}')
    if 'losses_iter' in h and len(h['losses_iter']):
        L = np.asarray(h['losses_iter'], dtype=float)
        print(f'   loss: {L[0]:.3f} -> {L[-1]:.3f}')


def main():
    tn = os.environ.get('GN_TEST_NAME', 'gn_compare')
    gn = load_latest(tn, 'ggn_b')          # Fisher/GGN curvature run
    if gn is None:
        gn = load_latest(tn, 'gn_b')       # exact-Hessian run
    adam = load_latest(tn, 'adam_b')
    if gn is None:
        gn = load_latest('gn_cpu_smoke', 'gn_cpu_smoke')  # fallback to CPU smoke

    summarize('GAUSS-NEWTON / LM', gn)
    summarize('ADAM', adam)

    if gn is not None and adam is not None:
        print('\n' + '=' * 60)
        print('HEADLINE:')
        for k in PARS:
            print(f'   {k:12s}  GN final {bias_traj(gn,k)[-1]:+6.2f}%   |   '
                  f'Adam final {bias_traj(adam,k)[-1]:+6.2f}%')
        print(f'   iters-to-1%:  GN {iters_to_tol(gn)}   vs   Adam {iters_to_tol(adam)}')

    _plot(gn, adam)


def _plot(gn, adam):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f'\n[plot skipped: {e}]')
        return
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.ravel()
    # loss panel
    ax = axes[0]
    for h, lab, c in [(gn, 'GN/LM', 'C0'), (adam, 'Adam', 'C1')]:
        if h is not None and len(h.get('losses_iter', [])):
            L = np.asarray(h['losses_iter'], dtype=float)
            ax.plot(np.arange(len(L)), L, label=lab, color=c)
    ax.set_yscale('log'); ax.set_xlabel('iteration'); ax.set_title('total loss'); ax.legend()
    ax.grid(alpha=0.3)
    # per-param bias panels
    for ax, k in zip(axes[1:], PARS):
        for h, lab, c in [(gn, 'GN/LM', 'C0'), (adam, 'Adam', 'C1')]:
            if h is not None:
                b = bias_traj(h, k)
                ax.plot(np.arange(len(b)), b, label=lab, color=c)
        ax.axhline(0, color='k', lw=0.8)
        ax.axhline(1, color='gray', ls=':', lw=0.6); ax.axhline(-1, color='gray', ls=':', lw=0.6)
        ax.set_title(f'{k} bias [%]'); ax.set_xlabel('iteration'); ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle('Gauss-Newton/LM vs Adam — calibration-only, ground-truth tracks')
    fig.tight_layout()
    os.makedirs('plots', exist_ok=True)
    out = 'plots/gn_vs_adam.png'
    fig.savefig(out, dpi=130)
    print(f'\n[saved {out}]')


if __name__ == '__main__':
    main()
