"""Paper figure: convergence of the joint calibration fit (NuLads).

Eight panels (2 rows x 4 columns, full page width):
  (a)-(e) relative error (%) vs. iteration for the five calibration parameters
          A_b, E field, transverse diffusion D_T, longitudinal diffusion D_L,
          electron lifetime tau -- one thin line per track sample (9, recipe),
          the sample mean (thick), zero/truth (black dashed), and the last-1e3
          averaging window (shaded). The naive straight-start fit (grey, same
          9 track samples) is overlaid in (d) D_L and (e) tau, the two
          parameters Sec. "Calibration results" identifies as biased by it;
          A_b, E and D_T are omitted there because the naive fit recovers them
          too (checked below -- gap at the last-1e3-iter mean is <0.2 points).
  (f) per-epoch mean loss per batch vs. epoch, recipe vs. naive (log-y: the
      loss falls almost two decades in the first few epochs, then the
      recipe/naive gap -- order 1e3 per batch -- sits at the bottom).
  (g) per-epoch mean trajectory error (um) vs. epoch, recipe vs. naive.
  (h) per-epoch mean |fitted-true| per-segment dE/dx vs. epoch (% of the mean
      dE/dx, 1.887 MeV/cm), recipe vs. naive.

Data: one pickle per fit, 'history_iter10000_*.pkl', under
  /sdf/group/neutrino/pgranger/larnd-sim-jax/fit_result/<run>/
(symlinks to /sdf/data or to the /sdf/scratch staging area -- resolved
transparently by open()). Runs:
  recipe (R2): sci_full_STRFRZ501 .. STRFRZ509   (9 track samples)
  naive straight-start: sci_full_DSREP1 .. DSREP9 (SAME 9 track samples --
    Sec. "Calibration results" explicitly restricts its own naive-vs-recipe
    comparison to "the same nine samples"; DSREP has samples up to 16 but
    only 1-9 pair with a recipe fit).

Field semantics, verified by reading optimize/fit_params.py (grep'd, not
assumed):
  '<p>_iter'   (p in Ab, eField, tran_diff, long_diff, lifetime): current_params
               value, appended once per Adam step BEFORE total_iter is
               incremented, plus one initial entry at construction -> length
               10001, index 0 = the starting guess, index k (k=1..10000) = the
               state after update k. We drop index 0 (common to every sample:
               flat at the fixed starting guess -- not interesting, and lets
               the panels use a clean log-x from iteration 1).
  '<p>_target': the truth value used to generate that run's target simulation
               (length-1 list). Checked identical across all 18 runs to the
               values quoted in Sec. "Calibration results" (A_b=0.835 etc).
  'losses_iter': loss_val.item(), i.e. the loss of the single batch processed
               at that iteration (batch_sz=1, sz_mini_bt=1 in every run here,
               so this is already a true per-batch loss, not a sum over a
               mini-batch); length 10000, index k = iteration k+1. Batches
               cycle sequentially 0..99 (shuffle_bt=False, confirmed from
               'pos_residual_batch'), so a per-epoch mean over 100 consecutive
               entries is the mean loss per batch for that epoch, matching how
               Sec. "Calibration results" quotes "final objective ... per
               batch".
  'pos_residual_iter' / 'pos_residual_batch': '_compute_pos_residual_batch'
               docstring says explicitly "Mean 3D distance (cm) between
               current fitted chain positions and true target positions" for
               the batch named by the paired 'pos_residual_batch' entry;
               length 10000, same indexing as losses_iter. Verified in cm, not
               um: STRFRZ501's values are 0.0066-0.091 cm over the first/last
               three entries = 66-910 um, matching the in-code comment "881 um
               median ... spanning 62-3051 um" for the full run. Converted to
               um here (x1e4) because the task and the paper both quote
               trajectory errors in um.
  'dedx_mae_truth_iter': mean |fitted - true| per-segment dE/dx of the batch,
               arc-length matched against the truth file; units MeV/cm (same
               units as 'mean_dedx_iter', which sits at the 1.887 MeV/cm
               constraint target). Length 9999, ONE SHORT of losses_iter in
               every one of the 18 checkpoints -- an existing +/-1 instrument
               quirk in fit_params.py's dEdx-diagnostics block (not something
               this script can fix without editing tracked code, which is out
               of scope). Padded with one NaN at the front (nanmean used in
               all aggregation), which can shift which single iteration falls
               in epoch 1 by one slot out of 100 and has no visible effect on
               any epoch mean.
  'mean_dedx_iter': the CURRENT batch's mean dE/dx (aux['mean_dedx']), not a
               running average; same length/indexing quirk as above. Used
               here only to confirm the 1.887 MeV/cm reference, not plotted.

Regenerate with:
    python fig_convergence.py [--rebuild]
Caches the extracted per-iteration arrays (9 recipe + 9 naive runs x 5 calib
params + loss + pos residual + dEdx MAE) to avoid re-reading the ~100 MB
pickles on every run:
    /sdf/scratch/neutrino/pgranger/review_2026_09/paperfig_cache/convergence.npz
Writes:
    /sdf/group/neutrino/pgranger/NuLads_draft/figures/fit_convergence.pdf
    /sdf/group/neutrino/pgranger/NuLads_draft/figures/fit_convergence.png
and prints the summary numbers (final relative errors, recipe vs. naive loss/
geometry/dEdx gap) to stdout.

This script does no GPU work and never writes to a tracked file: it only
reads existing checkpoint pickles and writes its own cache + the two figure
files above.
"""
import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')  # defensive: unpickling must not try to grab a GPU
import sys

LARND_DIR = '/sdf/group/neutrino/pgranger/larnd-sim-jax'
SRC_DIR = os.path.join(LARND_DIR, 'src')
for _p in (SRC_DIR, LARND_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import glob
import pickle
import argparse

import numpy as np
import matplotlib
import matplotlib.ticker
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib import patheffects as pe

FIT_RESULT_DIR = os.path.join(LARND_DIR, 'fit_result')
CACHE_DIR = '/sdf/scratch/neutrino/pgranger/review_2026_09/paperfig_cache'
N_SAMPLES = int(os.environ.get('NCONV', '16'))  # track samples 1..N (9 for the 8 Oct version of the figure)
# CONVCONT=1 (default since 12 Oct): the procedure = the production fits CONTINUED to 1.4e4 iterations (CONTR<i>: production
# history 0-10000 + continuation 10000-14000, same pickle); the straight-start fit stays at 1e4 (not continued).
CONT = os.environ.get('CONVCONT', '1') == '1'
# CONV_NAIVE=1 overlays the straight-start fits; off by default since 12 Oct (Pierre: the straight start is mentioned
# in the text only, not compared in the figures)
SHOW_NAIVE = os.environ.get('CONV_NAIVE', '0') == '1'
N_ITER_R = 14000 if CONT else 10000
# CONV_PREFIX (default CONTR / STRFRZ50): run-name prefix of the procedure fits, e.g. KN10C (10 cm knot, continued, 15 Oct).
# CONV_OUT_DIR (default: the paper's figures directory): where the figure is written.
RUN_PREFIX = os.environ.get('CONV_PREFIX', 'CONTR' if CONT else 'STRFRZ50')
CACHE_FILE = os.path.join(CACHE_DIR, ('convergence.npz' if N_SAMPLES == 9 else f'convergence{N_SAMPLES}.npz').replace('.npz', '_cont.npz' if CONT else '.npz'))
if RUN_PREFIX not in ('CONTR', 'STRFRZ50'):
    CACHE_FILE = CACHE_FILE.replace('.npz', f'_{RUN_PREFIX}.npz')
OUT_DIR = os.environ.get('CONV_OUT_DIR', '/sdf/group/neutrino/pgranger/NuLads_draft/figures')

RECIPE_RUNS = [f'sci_full_{RUN_PREFIX}{i}' for i in range(1, N_SAMPLES + 1)]
NAIVE_RUNS = [f'sci_full_DSREP{i}' for i in range(1, N_SAMPLES + 1)]

PARAMS = ['Ab', 'eField', 'tran_diff', 'long_diff', 'lifetime']
PARAM_TITLE = {
    'Ab': r'$A_b$', 'eField': r'$E$ field', 'tran_diff': r'$D_T$',
    'long_diff': r'$D_L$', 'lifetime': r'$\tau$ (lifetime)',
}
PARAM_TRUTH_UNIT = {
    'Ab': '', 'eField': 'kV/cm', 'tran_diff': r'cm$^2/\mu$s',
    'long_diff': r'cm$^2/\mu$s', 'lifetime': r'$\mu$s',
}

N_ITER = 10000   # Adam iterations (per run)
N_EPOCH = 100    # iterations per epoch == max_nbatch
N_LAST = 1000    # averaging window quoted in the paper

# Okabe-Ito palette (shared paper style)
VERM = '#D55E00'   # recipe
GREY = '#999999'   # naive
BLACK = '#000000'  # truth / zero
BLUE = '#0072B2'
GREEN = '#009E73'

plt.rcParams.update({
    'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8,
    'legend.fontsize': 7, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'axes.linewidth': 0.6, 'lines.linewidth': 1.2,
    'font.family': 'sans-serif',
})


def find_checkpoint(run):
    d = os.path.join(FIT_RESULT_DIR, run)
    n = 14000 if (CONT and run in RECIPE_RUNS) else 10000
    matches = sorted(glob.glob(os.path.join(d, f'history_iter{n}_*.pkl')))
    if not matches:
        raise FileNotFoundError(f'no history_iter{n}_*.pkl under {d}')
    if len(matches) > 1:
        raise RuntimeError(f'ambiguous checkpoint for {run}: {matches}')
    return matches[0]


def load_run(run):
    path = find_checkpoint(run)
    with open(path, 'rb') as f:
        h = pickle.load(f)

    out = {'run': run, 'path': path}
    N_ITER = 14000 if (CONT and run in RECIPE_RUNS) else 10000   # per-run length (shadows the module constant)
    for p in PARAMS:
        it = np.asarray(h[p + '_iter'], dtype=float)
        if len(it) != N_ITER + 1:
            raise ValueError(f'{run}: {p}_iter has length {len(it)}, expected {N_ITER + 1}')
        tgt = float(np.asarray(h[p + '_target'], dtype=float).ravel()[0])
        out[p + '_relerr'] = (it[1:] - tgt) / tgt * 100.0  # drop iter-0; length N_ITER
        out[p + '_target'] = tgt
        out[p + '_final'] = float(it[-1])

    loss = np.asarray(h['losses_iter'], dtype=float)
    if len(loss) != N_ITER:
        raise ValueError(f'{run}: losses_iter has length {len(loss)}, expected {N_ITER}')
    out['loss'] = loss

    pr = np.asarray(h['pos_residual_iter'], dtype=float)  # cm, per '_compute_pos_residual_batch' docstring
    if len(pr) != N_ITER:
        raise ValueError(f'{run}: pos_residual_iter has length {len(pr)}, expected {N_ITER}')
    out['pos_um'] = pr * 1.0e4  # cm -> um

    dmae = np.asarray(h['dedx_mae_truth_iter'], dtype=float)  # MeV/cm
    if N_ITER - 2 <= len(dmae) < N_ITER:      # known +/-1 quirk; a resumed run (CONTR) drops one more entry at the boundary
        dmae = np.concatenate([np.full(N_ITER - len(dmae), np.nan), dmae])
    if len(dmae) != N_ITER:
        raise ValueError(f'{run}: dedx_mae_truth_iter has length {len(dmae)}, expected {N_ITER} (+/-1)')
    out['dedx_mae'] = dmae

    cfg = h['config']
    out['dedx_mean_ref'] = float(cfg.dedx_mean_constraint_target)
    out['freeze_drift'] = h['env_overrides'].get('LARND_CHAIN_FREEZE_DRIFT')
    out['drift_from_hits'] = h['env_overrides'].get('LARND_DRIFT_FROM_HITS')
    out['max_nbatch'] = int(cfg.max_nbatch)
    return out


def build_cache():
    os.makedirs(CACHE_DIR, exist_ok=True)
    data = {}
    for group, runs in (('recipe', RECIPE_RUNS), ('naive', NAIVE_RUNS)):
        recs = [load_run(r) for r in runs]
        print(f'[{group}] loaded {len(recs)} runs: {[r["run"] for r in recs]}')
        for r in recs:
            tag = 'FREEZE' if r['freeze_drift'] == '1' else ('free' if r['freeze_drift'] in (None, '0', '') else r['freeze_drift'])
            print(f"    {r['run']}: freeze_drift={tag} drift_from_hits={r['drift_from_hits']} "
                  f"dedx_mean_ref={r['dedx_mean_ref']} max_nbatch={r['max_nbatch']}")
        for p in PARAMS:
            data[f'{group}_{p}_relerr'] = np.stack([r[p + '_relerr'] for r in recs])
            data[f'{group}_{p}_target'] = np.array([r[p + '_target'] for r in recs])
        data[f'{group}_loss'] = np.stack([r['loss'] for r in recs])
        data[f'{group}_pos_um'] = np.stack([r['pos_um'] for r in recs])
        data[f'{group}_dedx_mae'] = np.stack([r['dedx_mae'] for r in recs])
        data[f'{group}_dedx_mean_ref'] = np.array([r['dedx_mean_ref'] for r in recs])
    np.savez_compressed(CACHE_FILE, **data)
    print(f'wrote cache: {CACHE_FILE}')
    return data


def load_cache(rebuild=False):
    if (not rebuild) and os.path.exists(CACHE_FILE):
        print(f'reading cache: {CACHE_FILE}')
        with np.load(CACHE_FILE) as z:
            return {k: z[k] for k in z.files}
    return build_cache()


def epoch_mean(arr2d, n_epoch=N_EPOCH):
    """arr2d: (n_samples, n_iter) -> (n_samples, n_iter // n_epoch) nanmean per epoch."""
    n_samples, n_iter = arr2d.shape
    n_full = n_iter // n_epoch
    trimmed = arr2d[:, :n_full * n_epoch].reshape(n_samples, n_full, n_epoch)
    return np.nanmean(trimmed, axis=2)


def sem(x, axis=0):
    n = x.shape[axis]
    return np.std(x, axis=axis, ddof=1) / np.sqrt(n)


def panel_label(ax, text):
    ax.text(0.05, 0.94, text, transform=ax.transAxes, fontsize=7, va='top', ha='left', zorder=6)


def mean_line(ax, x, y, color, zorder, linewidth=1.5):
    """Thick mean line with a white halo so it reads clearly against a dense cloud of thin
    per-sample lines of the SAME colour (e.g. panel (b), where the per-iteration noise band is
    wide enough to otherwise bury the mean)."""
    line, = ax.plot(x, y, color=color, linewidth=linewidth, zorder=zorder, clip_on=True,
                     solid_capstyle='round')
    line.set_path_effects([pe.Stroke(linewidth=linewidth + 1.1, foreground='white'), pe.Normal()])
    return line


def style_axis(ax):
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.set_xscale('log')


def plot_param_panel(ax, label, it_axis, recipe_re, naive_re, show_naive, last_lo, last_hi, ylim=None,
                      sample_alpha=0.22):
    it_r = np.arange(1, recipe_re.shape[1] + 1); it_n = np.arange(1, naive_re.shape[1] + 1)   # groups may differ in length
    for s in range(recipe_re.shape[0]):
        ax.plot(it_r, recipe_re[s], color=VERM, linewidth=0.5, alpha=sample_alpha, zorder=2, clip_on=True)
    if show_naive:
        for s in range(naive_re.shape[0]):
            ax.plot(it_n, naive_re[s], color=GREY, linewidth=0.5, alpha=0.30, zorder=1, clip_on=True)
        mean_line(ax, it_n, naive_re.mean(axis=0), GREY, zorder=3)
    mean_line(ax, it_r, recipe_re.mean(axis=0), VERM, zorder=4)
    ax.axhline(0.0, color=BLACK, linestyle='--', linewidth=0.7, zorder=5)
    ax.axvspan(last_lo, last_hi, color='0.4', alpha=0.35, linewidth=0, zorder=0.5)
    if CONT:
        ax.axvline(10000, color='0.25', linestyle=':', linewidth=0.8, zorder=5)
    style_axis(ax)
    ax.set_xlim(1, it_r[-1])
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel('iteration')
    ax.set_ylabel(f'{label} rel. error (%)')


def plot_epoch_panel(ax, label, ep_axis, recipe_e, naive_e, logy, last_lo_ep, last_hi_ep):
    ep_r = np.arange(1, recipe_e.shape[1] + 1); ep_n = np.arange(1, naive_e.shape[1] + 1)
    for s in range(naive_e.shape[0] if SHOW_NAIVE else 0):
        ax.plot(ep_n, naive_e[s], color=GREY, linewidth=0.5, alpha=0.30, zorder=1)
    for s in range(recipe_e.shape[0]):
        ax.plot(ep_r, recipe_e[s], color=VERM, linewidth=0.5, alpha=0.25, zorder=2)
    if SHOW_NAIVE:
        mean_line(ax, ep_n, naive_e.mean(axis=0), GREY, zorder=3)
    mean_line(ax, ep_r, recipe_e.mean(axis=0), VERM, zorder=4)
    ax.axvspan(last_lo_ep, last_hi_ep, color='0.4', alpha=0.35, linewidth=0, zorder=0.5)
    if CONT:
        ax.axvline(100, color='0.25', linestyle=':', linewidth=0.8, zorder=5)
    style_axis(ax)
    if logy:
        ax.set_yscale('log')
    ax.set_xlim(1, ep_r[-1])
    ax.set_xlabel('epoch (100 iter.)')
    ax.set_ylabel(label)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rebuild', action='store_true', help='force re-read of the pickles, ignore cache')
    args = ap.parse_args()

    d = load_cache(rebuild=args.rebuild)

    it_axis = np.arange(1, N_ITER_R + 1)
    N_EPOCH_R = N_ITER_R // (N_ITER // N_EPOCH)
    ep_axis = np.arange(1, N_EPOCH_R + 1)
    last_lo_it, last_hi_it = N_ITER_R - N_LAST + 1, N_ITER_R
    last_lo_ep, last_hi_ep = N_EPOCH_R - N_LAST // (N_ITER // N_EPOCH) + 1, N_EPOCH_R

    dedx_ref = float(np.mean(d['recipe_dedx_mean_ref']))
    assert np.allclose(d['recipe_dedx_mean_ref'], dedx_ref) and np.allclose(d['naive_dedx_mean_ref'], dedx_ref), \
        'dedx_mean_constraint_target differs across runs -- check configs'

    # ---------------- summary numbers (printed; also drives the "verify" asks) ----------------
    print('\n# Final (mean over last %d iterations) relative error, mean +/- SEM over %d samples' % (N_LAST, N_SAMPLES))
    print('# (truth value used to generate the targets shown for reference; checked identical across runs)')
    print('# param        truth                  recipe                naive(straight-start)')
    for p in PARAMS:
        rv = d[f'recipe_{p}_relerr'][:, -N_LAST:].mean(axis=1)
        nv = d[f'naive_{p}_relerr'][:, -N_LAST:].mean(axis=1)
        tgt = float(np.mean(d[f'recipe_{p}_target']))
        unit = PARAM_TRUTH_UNIT[p]
        print(f'  {p:10s}  {tgt:.6g} {unit:10s}  {rv.mean():+7.3f} +/- {sem(rv):.3f} %      '
              f'{nv.mean():+7.3f} +/- {sem(nv):.3f} %')

    recipe_loss_last = d['recipe_loss'][:, -N_LAST:].mean(axis=1)
    naive_loss_last = d['naive_loss'][:, -N_LAST:].mean(axis=1)
    print('\n# Loss per batch, mean over last %d iterations (mean +/- SEM over %d samples)' % (N_LAST, N_SAMPLES))
    print(f'  recipe: {recipe_loss_last.mean():.1f} +/- {sem(recipe_loss_last):.1f}')
    print(f'  naive : {naive_loss_last.mean():.1f} +/- {sem(naive_loss_last):.1f}')
    print(f'  naive - recipe = {naive_loss_last.mean() - recipe_loss_last.mean():.1f} per batch')

    recipe_pos_last = d['recipe_pos_um'][:, -N_LAST:].mean(axis=1)
    naive_pos_last = d['naive_pos_um'][:, -N_LAST:].mean(axis=1)
    print('\n# Trajectory error (um), mean over last %d iterations' % N_LAST)
    print(f'  recipe: {recipe_pos_last.mean():.1f} +/- {sem(recipe_pos_last):.1f} um')
    print(f'  naive : {naive_pos_last.mean():.1f} +/- {sem(naive_pos_last):.1f} um')

    recipe_dedx_last = np.nanmean(d['recipe_dedx_mae'][:, -N_LAST:], axis=1)
    naive_dedx_last = np.nanmean(d['naive_dedx_mae'][:, -N_LAST:], axis=1)
    print('\n# dE/dx MAE vs truth (MeV/cm and %% of %.3f MeV/cm), mean over last %d iterations' % (dedx_ref, N_LAST))
    print(f'  recipe: {recipe_dedx_last.mean():.4f} MeV/cm = {100*recipe_dedx_last.mean()/dedx_ref:.2f} %')
    print(f'  naive : {naive_dedx_last.mean():.4f} MeV/cm = {100*naive_dedx_last.mean()/dedx_ref:.2f} %')

    # epoch-mean dynamic ranges, to pick log/linear y for panels (f)-(h)
    loss_ep_r, loss_ep_n = epoch_mean(d['recipe_loss']), epoch_mean(d['naive_loss'])
    pos_ep_r, pos_ep_n = epoch_mean(d['recipe_pos_um']), epoch_mean(d['naive_pos_um'])
    dedx_ep_r, dedx_ep_n = epoch_mean(d['recipe_dedx_mae']), epoch_mean(d['naive_dedx_mae'])
    for name, arr in [('loss', np.concatenate([loss_ep_r.ravel(), loss_ep_n.ravel()])),
                       ('pos_um', np.concatenate([pos_ep_r.ravel(), pos_ep_n.ravel()])),
                       ('dedx_mae', np.concatenate([dedx_ep_r.ravel(), dedx_ep_n.ravel()]))]:
        mn, mx = np.nanmin(arr), np.nanmax(arr)
        print(f'  epoch-mean range {name}: {mn:.4g} .. {mx:.4g}  (ratio {mx/mn:.2f})')

    # ================= figure =================
    fig, axes = plt.subplots(2, 4, figsize=(7.0, 3.9))

    show_naive_param = {'Ab': False, 'eField': False, 'tran_diff': False, 'long_diff': SHOW_NAIVE, 'lifetime': SHOW_NAIVE}
    # A_b and E are shown on a finer y-scale (per the figure spec): the first ~100 iterations
    # (Adam warmup, lr_kw.warmup_steps=500) swing to -4..-5%, common to every parameter and
    # already visible in (c)-(e); clipping it here instead resolves the annealed, narrowing
    # late-time noise band that the asymptotic +0.18%/+0.017% figures (Table tab:recipe) come
    # from -- confirmed by the per-iteration std falling from 1.1%/0.26% (iter 101-500) to
    # 0.11%/0.02% (iter 6000-9000) for A_b/E respectively.
    # per-panel scales (12 Oct: a common linear and then a common symlog scale were tried and reverted at Pierre's request)
    ylim_param = {'Ab': (-1.5, 2.0), 'eField': (-0.5, 0.8), 'tran_diff': None, 'long_diff': None, 'lifetime': None}
    sample_alpha_param = {'Ab': 0.20, 'eField': 0.10, 'tran_diff': 0.22, 'long_diff': 0.22, 'lifetime': 0.22}
    order = ['Ab', 'eField', 'tran_diff', 'long_diff', 'lifetime']
    labels = ['a', 'b', 'c', 'd', 'e']
    for ax, p, lab in zip(axes.flat[:5], order, labels):
        plot_param_panel(ax, PARAM_TITLE[p], it_axis, d[f'recipe_{p}_relerr'], d[f'naive_{p}_relerr'],
                          show_naive_param[p], last_lo_it, last_hi_it, ylim=ylim_param[p],
                          sample_alpha=sample_alpha_param[p])
        panel_label(ax, f'({lab})')
        if ylim_param[p] is not None:
            ax.text(0.20, 0.5, 'early\ntransient\nclipped', transform=ax.transAxes, linespacing=1.3,
                     fontsize=4.8, ha='center', va='center', color='0.4', style='italic')

    # one explicit callout for the averaging window, which the log x-axis compresses to a
    # hairline at the right edge of every panel (true in all 8; annotated once here)
    ax_e = axes.flat[4]
    ax_e.annotate(f'last {N_LAST} iter.\n(reported mean)', xy=(0.95 * N_ITER_R, ax_e.get_ylim()[0] * 0.35),
                  xytext=(55, ax_e.get_ylim()[1] * 0.78),
                  fontsize=5.3, ha='left', va='center', color='0.25',
                  arrowprops=dict(arrowstyle='->', color='0.3', linewidth=0.6, shrinkA=1, shrinkB=2))

    ax = axes.flat[5]
    plot_epoch_panel(ax, 'mean loss / batch', ep_axis, loss_ep_r, loss_ep_n, True, last_lo_ep, last_hi_ep)
    panel_label(ax, '(f)')

    ax = axes.flat[6]
    plot_epoch_panel(ax, r'mean traj. error ($\mu$m)', ep_axis, pos_ep_r, pos_ep_n, True, last_lo_ep, last_hi_ep)
    panel_label(ax, '(g)')

    ax = axes.flat[7]
    dedx_pct_r = 100.0 * dedx_ep_r / dedx_ref
    dedx_pct_n = 100.0 * dedx_ep_n / dedx_ref
    plot_epoch_panel(ax, r'dE/dx MAE (% of $\overline{\mathrm{d}E/\mathrm{d}x}$)', ep_axis,
                      dedx_pct_r, dedx_pct_n, False, last_lo_ep, last_hi_ep)
    panel_label(ax, '(h)')

    handles = [
        Line2D([0], [0], color=VERM, linewidth=1.4, label=f'mean ({N_SAMPLES} track samples)'),
        Line2D([0], [0], color=VERM, linewidth=0.7, alpha=0.5, label='per track sample'),
    ] + ([Line2D([0], [0], color=GREY, linewidth=1.4, label=f'naive straight-start, mean ({N_SAMPLES} samples)'),
          Line2D([0], [0], color=GREY, linewidth=0.7, alpha=0.5, label='naive straight-start, per sample')] if SHOW_NAIVE else []) + [
        Line2D([0], [0], color=BLACK, linestyle='--', linewidth=0.7, label='truth / zero'),
    ] + ([Line2D([0], [0], color='0.25', linestyle=':', linewidth=0.8, label='start of continuation')] if CONT else []) + [
        Patch(facecolor='0.5', alpha=0.25, linewidth=0, label=f'last {N_LAST} iter. (reported mean)'),
    ]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1.025), ncol=3,
               frameon=False, handletextpad=0.5, columnspacing=1.3, labelspacing=0.3,
               borderaxespad=0.0)

    fig.tight_layout(rect=(0, 0, 1, 0.935), w_pad=1.1, h_pad=1.3)
    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(os.path.join(OUT_DIR, 'fit_convergence.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(OUT_DIR, 'fit_convergence.png'), dpi=300, bbox_inches='tight')
    print(f'\nsaved {OUT_DIR}/fit_convergence.pdf and .png')


if __name__ == '__main__':
    main()
