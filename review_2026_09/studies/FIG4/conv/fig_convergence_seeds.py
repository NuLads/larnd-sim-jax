"""Paper figure: ALTERNATIVE rendering of the joint-fit convergence (NuLads), emphasising
individual track samples ("seeds") rather than the sample mean.

This is NOT a replacement for figures/fit_convergence.pdf (Fig. 12, fig:convergence, made by
fig_convergence.py in this directory): it is a second, additional figure the paper's authors
asked for, built on the exact same cached data, that (a) uses a linear iteration axis
throughout -- no log axes anywhere -- and (b) gives each of the 9 track samples its own
colour, held consistent across every panel, instead of a dense cloud of same-colour thin
lines behind a thick mean. The idea is borrowed from the convergence figure of the original
PyTorch differentiable-LArTPC paper:

    S. Gasiorowski et al., "Differentiable Simulation of a Liquid Argon Time Projection
    Chamber", arXiv:2309.04639, Fig. 15 (p. 16) [and the companion Fig. 17, same layout on a
    mixed-particle sample].

Fig. 15 of that paper is a 3x2 grid, one panel per detector parameter (A_B, k_B, E, tau, D_T,
D_L), linear "training iterations" on the x axis (0-5000), the RAW fitted parameter value on
the y axis, one solid line per fit/seed in its own colour (10 seeds, each sampled from a
different random target point), and a dashed horizontal line per seed marking that seed's own
target. We emulate the visual idea -- linear axis, one colour per seed held fixed across
panels, dashed target line, compact multi-panel layout -- but adapt the content to ours:
*our* 9 track samples all share the SAME truth (one calibration is being recovered 9 times
over independent track samples, not 9 different sampled targets), so the natural analogue of
"fitted value with a dashed per-seed target line" is "relative error with a single dashed
truth/zero line" -- this also lets the panel reuse the cache's already-computed '*_relerr'
arrays unchanged, and reads directly as "how far from converged is each sample, in %", which
is the quantity Table tab:recipe and the running text actually quote. We write relative error
rather than the fitted/truth ratio for the same reason the original fig_convergence.py does
(see its docstring): it is the quantity quoted throughout the paper, and it is centred on a
meaningful zero instead of 1, which reads better on a LINEAR y axis (the ratio would squeeze
undershoots into [0,1) and let overshoots run to arbitrarily large numbers).

Panels (2 rows x 3 cols, full page width, linear x = iteration 0-10000 throughout):
  (a)-(c), top row:   Ab, E field, D_T          -- recipe only (9 coloured lines + truth).
  (d)-(e), bottom row: D_L, tau                 -- recipe (9 coloured lines) + the naive
                       straight-start fit on the SAME 9 track samples, shown as plain grey
                       lines (no per-sample colour: the naive fit is background context here,
                       not the subject). D_L and tau are the two parameters the paper's running
                       text singles out as biased by a naive start ("A naive initialisation
                       fails", Sec. "Calibration results"); the original fig_convergence.py
                       makes and verifies the identical choice ("gap at the last-1e3-iter mean
                       is <0.2 points" for Ab/E/D_T) -- reused here so the two figures tell a
                       consistent story. Showing naive in all 5 panels would add 9 more lines
                       to 3 panels that are already making their point cleanly with recipe
                       alone; omitting it everywhere would drop the one comparison the text
                       actually leans on. This is the explicit "decide what is clearest" call
                       the figure spec asks for.
  (f), bottom row:    per-epoch mean loss/batch, recipe only, 9 coloured lines, LINEAR y (the
                       per-epoch-mean range is only a factor ~2.7 top to bottom -- unlike the
                       per-ITERATION loss, which really does span ~2 decades at the very start
                       -- so no clipping or log scale is needed to resolve it). Plotted at
                       x = epoch * 100 so it sits on the same 0-10000 iteration axis as (a)-(e).

Per-panel y-ranges are chosen to resolve the POST-transient convergence (the actual claim of
the paper), which means the first ~1-2x10^3 iterations are partly clipped in every one of
(a)-(e); this is stated once here and in the caption rather than re-annotated in every panel.
Concretely (see the comment above YLIM below): Ab and E field reuse the exact y-limits of
fig_convergence.py panels (a)-(b) (same data, kept consistent across the two figures); D_T,
D_L and tau use new limits chosen from the actual post-iter-2000/3000 data range of this cache
(computed, not guessed -- see the range-finding commands in the session this script was
written in) so that in (d)-(e) the limits also comfortably contain the NAIVE fit's converged
(biased) band, not just the recipe's.

The last 10^3 iterations (the window Table tab:recipe averages over) are marked with a light
grey vertical band in every panel, per the figure spec ("mark ... lightly").

Data: reuses fig_convergence.py's own cache-loading code UNCHANGED (imported, not copy-pasted)
from
    /sdf/scratch/neutrino/pgranger/review_2026_09/paperfig_cache/convergence.npz
which that script builds from the 'history_iter10000_*.pkl' checkpoints of the recipe runs
sci_full_STRFRZ501..509 and the naive runs sci_full_DSREP1..9. See fig_convergence.py's own
module docstring for the full field-by-field provenance of that cache; this script only reads
it (via fig_convergence.load_cache(), rebuild=False by default) and never touches the
checkpoints or that script.

This script does no GPU work (JAX_PLATFORMS=cpu is set, defensively, before any import that
could pull in jax -- fig_convergence.py does the same when unpickling the checkpoints), makes
no git or SLURM calls, and only ever CREATES new files: this script itself, and
    /sdf/group/neutrino/pgranger/NuLads_draft/figures/fit_convergence_seeds.{pdf,png}
It never edits main.tex, fig_convergence.py, or any other tracked file; the new figure is not
wired into main.tex by this script (that is an editorial decision for the paper's authors).

Run with:
    python fig_convergence_seeds.py [--rebuild]
"""
import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')  # defensive: unpickling (inside the imported
                                                # fig_convergence module) must not grab a GPU
import argparse
import importlib.util

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ---------------------------------------------------------------------------
# Reuse fig_convergence.py's loading/cache code verbatim, by importing the module (not
# copy-pasting it): gives us load_cache(), epoch_mean(), sem(), and the shared constants
# (N_ITER, N_EPOCH, N_LAST, PARAMS, PARAM_TITLE, PARAM_TRUTH_UNIT, CACHE_FILE). Importing runs
# only the module's top-level def/constant statements (its `if __name__ == '__main__'` guard
# keeps it from doing any work on import).
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location('fig_convergence', os.path.join(_THIS_DIR, 'fig_convergence.py'))
fc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fc)

OUT_DIR = fc.OUT_DIR  # /sdf/group/neutrino/pgranger/NuLads_draft/figures
N_ITER, N_EPOCH, N_LAST = fc.N_ITER, fc.N_EPOCH, fc.N_LAST
PARAMS, PARAM_TITLE = fc.PARAMS, fc.PARAM_TITLE

# ---------------------------------------------------------------------------
# Style guide (shared across the paper's alternative figures)
# ---------------------------------------------------------------------------
plt.rcParams.update({
    'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8,
    'legend.fontsize': 7, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'axes.linewidth': 0.6, 'lines.linewidth': 1.0,
    'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'],
})

# One colour per track sample (1-9), consistent across every panel. Adapted from Paul Tol's
# 9-colour "muted" qualitative scheme (the standard choice for >6 simultaneous categorical
# lines in physics figures, where no ordering can make every pairwise contrast CVD-safe --
# confirmed with the dataviz skill's validate_palette.js: this adapted set passes the
# adjacent-pair colour-blind-separation and normal-vision-separation checks, which the
# original 9-colour Tol "muted" set does not quite, on a white page background). Grey is
# reserved for the naive fit and is never used for a sample.
SEED_COLORS = [
    '#CC6677',  # 1 rose
    '#332288',  # 2 indigo
    '#B8A426',  # 3 gold/ochre
    '#117733',  # 4 green
    '#3E8FBF',  # 5 sky blue
    '#882255',  # 6 wine
    '#44AA99',  # 7 teal
    '#9C5A00',  # 8 burnt orange (moved off dark-olive, which sat a validator-failing
                #   Delta E=13 from gold/ochre above; burnt orange clears the adjacent-pair
                #   colour-blind and normal-vision floors against every other slot)
    '#AA4499',  # 9 purple
    # samples 10-16 (added 10 Oct when the ensemble grew to 16): Tol "vibrant"/"light" picks,
    # chosen away from the first nine and from the grey of the naive fit
    '#88CCEE',  # 10 light cyan
    '#DDCC77',  # 11 sand
    '#EE7733',  # 12 orange
    '#EE3377',  # 13 magenta
    '#661100',  # 14 dark brown
    '#BBCC33',  # 15 pear
    '#004488',  # 16 dark blue
]
GREY = '#999999'    # naive straight-start fit (plain, not per-sample)
BLACK = '#000000'   # truth / zero
N_SAMPLES = fc.N_SAMPLES

N_ITER_R = fc.N_ITER_R                          # 14000 with the continued fits (fc.CONT), else 10000
LAST_LO, LAST_HI = N_ITER_R - N_LAST + 1, N_ITER_R  # the averaging window (last 1000 iterations of the reported fits)

# Post-transient y-ranges (%, relative error), chosen to resolve the convergence that the
# paper's text and Table tab:recipe actually discuss. Ab and eField reuse the exact limits of
# fig_convergence.py (a)-(b) (identical underlying data -- kept consistent across the two
# figures). tran_diff/long_diff/lifetime are new: chosen from this cache's own post-iter-2000/
# iter-3000 range (recipe), widened for long_diff/lifetime so the naive fit's converged band
# (shown in those two panels) sits inside the frame too, not just skimming its edge. In every
# case the clipped region is iterations before roughly 1-3x10^3 -- the same warm-up the paper's
# text describes each parameter as needing before it settles -- not the true endpoint.
YLIM = {
    'Ab': (-1.5, 2.0),
    'eField': (-0.5, 0.8),
    'tran_diff': (-6.0, 10.0),
    'long_diff': (-22.0, 50.0),
    'lifetime': (-12.0, 28.0),
}
SHOW_NAIVE = {k: (os.environ.get('CONV_NAIVE', '0') == '1' and k in ('long_diff', 'lifetime')) for k in ('Ab', 'eField', 'tran_diff', 'long_diff', 'lifetime')}
LOSS_YLIM = (8000.0, 24000.0)


def panel_label(ax, text):
    ax.text(0.04, 0.93, text, transform=ax.transAxes, fontsize=7, va='top', ha='left', zorder=6)


def mark_avg_window(ax):
    ax.axvspan(LAST_LO, LAST_HI, color='0.4', alpha=0.18, linewidth=0, zorder=0.5)


def style_axis(ax, ylim):
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.set_xlim(0, N_ITER_R)
    ax.set_xticks(list(range(0, N_ITER_R + 1, 4000)))
    ax.set_ylim(*ylim)
    mark_avg_window(ax)
    if fc.CONT:
        ax.axvline(N_ITER, color='0.25', linestyle=':', linewidth=0.8, zorder=4)


def plot_param_panel(ax, p, it_axis, d):
    recipe = d[f'recipe_{p}_relerr']  # (9, N_ITER)
    for s in range(N_SAMPLES):
        ax.plot(np.arange(1, recipe.shape[1] + 1), recipe[s], color=SEED_COLORS[s], linewidth=0.7, alpha=0.85,
                 zorder=3, clip_on=True, solid_capstyle='round')
    if SHOW_NAIVE[p]:
        naive = d[f'naive_{p}_relerr']
        for s in range(N_SAMPLES):
            ax.plot(np.arange(1, naive.shape[1] + 1), naive[s], color=GREY, linewidth=0.6, alpha=0.5,
                     zorder=2, clip_on=True)
    ax.axhline(0.0, color=BLACK, linestyle='--', linewidth=0.8, zorder=4)
    style_axis(ax, YLIM[p])
    ax.set_xlabel('iteration')
    ax.set_ylabel(f'{PARAM_TITLE[p]} rel. error (%)')


def plot_loss_panel(ax, ep_x, loss_ep):
    for s in range(N_SAMPLES):
        ax.plot(ep_x, loss_ep[s], color=SEED_COLORS[s], linewidth=0.7, alpha=0.9,
                 zorder=3, clip_on=True, solid_capstyle='round')
    style_axis(ax, LOSS_YLIM)
    ax.set_xlabel('iteration')
    ax.set_ylabel('mean loss / batch')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rebuild', action='store_true', help='force re-read of the pickles, ignore cache')
    args = ap.parse_args()

    d = fc.load_cache(rebuild=args.rebuild)
    it_axis = np.arange(1, N_ITER + 1)

    # per-epoch mean loss (recipe only), plotted at the iteration index ending each epoch so it
    # shares the (0, N_ITER) axis with the parameter panels
    loss_ep = fc.epoch_mean(d['recipe_loss'])  # (9, N_EPOCH)
    ep_x = np.arange(1, loss_ep.shape[1] + 1) * (N_ITER // N_EPOCH)

    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.35))

    order = ['Ab', 'eField', 'tran_diff', 'long_diff', 'lifetime']
    labels = ['a', 'b', 'c', 'd', 'e']
    for ax, p, lab in zip(axes.flat[:5], order, labels):
        plot_param_panel(ax, p, it_axis, d)
        panel_label(ax, f'({lab})')

    # one explicit note that the early warm-up runs off-scale (stated once, in (a), rather than
    # repeated in every panel -- it applies to all of (a)-(e), see the YLIM comment above)
    ax_a = axes.flat[0]

    ax = axes.flat[5]
    plot_loss_panel(ax, ep_x, loss_ep)
    panel_label(ax, '(f)')

    # one explicit callout for the averaging window (light shading would otherwise go
    # unexplained -- same approach as fig_convergence.py's single callout). Placed in the loss
    # panel, where all 9 samples have already collapsed onto one flat band by iteration ~4000,
    # leaving the upper-middle of the panel empty and free of any line to overlap.
    ax_f = axes.flat[5]
    lo, hi = LOSS_YLIM
    ax_f.annotate(f'last {N_LAST} iter.\n(avg. window)', xy=(N_ITER_R - 500, lo + 0.08 * (hi - lo)),
                  xytext=(0.53 * N_ITER_R, lo + 0.55 * (hi - lo)),
                  fontsize=5.3, ha='left', va='center', color='0.25',
                  arrowprops=dict(arrowstyle='->', color='0.3', linewidth=0.6, shrinkA=1, shrinkB=2))

    handles = [Line2D([0], [0], color=SEED_COLORS[s], linewidth=1.1, label=str(s + 1)) for s in range(N_SAMPLES)]
    handles += [
        Line2D([0], [0], color=BLACK, linestyle='--', linewidth=0.8, label='truth (0% error)'),
        Patch(facecolor='0.4', alpha=0.18, linewidth=0, label=f'last {N_LAST} iter. (avg. window)'),
    ] + ([Line2D([0], [0], color='0.25', linestyle=':', linewidth=0.8, label='start of continuation')] if fc.CONT else [])
    leg = fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1.045), ncol=10,
                      frameon=False, handletextpad=0.5, columnspacing=1.2, labelspacing=0.35,
                      borderaxespad=0.0, title='track sample', title_fontsize=7)
    leg.get_title().set_fontstyle('normal')

    fig.tight_layout(rect=(0, 0, 1, 0.90), w_pad=1.1, h_pad=1.3)
    os.makedirs(OUT_DIR, exist_ok=True)
    out_pdf = os.path.join(OUT_DIR, 'fit_convergence_seeds.pdf')
    out_png = os.path.join(OUT_DIR, 'fit_convergence_seeds.png')
    fig.savefig(out_pdf, bbox_inches='tight')
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    print(f'saved {out_pdf}\nsaved {out_png}')

    # quick sanity numbers, mirroring fig_convergence.py's summary so the two can be cross-checked
    print('\n# Final (mean over last %d iterations) relative error, mean +/- range over %d samples' % (N_LAST, N_SAMPLES))
    for p in PARAMS:
        rv = d[f'recipe_{p}_relerr'][:, -N_LAST:].mean(axis=1)
        print(f'  {p:10s}  recipe {rv.mean():+7.3f} %  (sample range {rv.min():+6.2f} .. {rv.max():+6.2f})')
    loss_last = d['recipe_loss'][:, -N_LAST:].mean(axis=1)
    print(f'  loss/batch  recipe {loss_last.mean():8.1f}  (sample range {loss_last.min():7.1f} .. {loss_last.max():7.1f})')

    # per-seed MEDIAN over the last N_LAST iterations (one number per sample per panel, as
    # plotted): the per-iteration median within each sample's own trailing window, i.e. what
    # that sample's line in (a)-(e) settles to, robust to single-iteration Adam noise.
    print('\n# Per-sample MEDIAN over the trailing %d iterations (one value per line in (a)-(e))' % N_LAST)
    header = '  sample  ' + '  '.join(f'{PARAM_TITLE[p]:>14s}' for p in PARAMS) + '      loss/batch'
    print(header)
    for s in range(N_SAMPLES):
        row = f'    {s + 1:2d}    '
        for p in PARAMS:
            med = np.median(d[f'recipe_{p}_relerr'][s, -N_LAST:])
            row += f'  {med:+13.3f}%'
        loss_med = np.median(d['recipe_loss'][s, -N_LAST:])
        row += f'   {loss_med:10.1f}'
        print(row)


if __name__ == '__main__':
    main()
