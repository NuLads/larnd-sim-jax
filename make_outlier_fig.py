"""Fig 88 — the full anatomy of the degenerate outlier (CINITA8).

!! WHY A_b IS AT +7.8%, AND IT IS MY LAUNCH ERROR, NOT A DISCOVERY: I set the all-shifted init to
A_b = 0.92, which is ABOVE the parameter's allowed maximum of 0.90 (ranges.py: min 0.75, max 0.90).
The sigmoid normalisation clamps it, so A_b saturates at the bound 0.8999 by iteration ~50 and is
frozen there for all 10 000 iterations -- it is not fitted at all in this run.

That accident turned the arm into a BETTER experiment than the one I intended: a controlled test of
"pin A_b 7.8% wrong and let everything else adapt". The result is the campaign's cleanest single
counter-example to 'lower loss = better fit': lifetime slides to -59.3% to compensate, and the
FINAL LOSS IS 9026 -- better than the nominal-init baseline's 9167 (A_b +0.01%, lifetime +16.4%).
The objective actively prefers the ruined calibration.

Every panel is the same run (red) against its nominal-init baseline (black), same data, same
target, same schedule. What to look for:
  * A_b and lifetime move together in the FIRST 200 ITERATIONS and never recover -- the -0.83
    anticorrelation, executed;
  * mean dE/dx compensates: the fit rescales its charge field to absorb the A_b error, which is
    exactly the degeneracy direction the Hessian predicts;
  * position, dE/dx MAE and loss are all NORMAL or BETTER than baseline -- every diagnostic we
    would actually monitor looks healthy while the calibration is destroyed.
"""
import glob, pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view

OUT = 'plots/noise_report'
RED, INK, GREEN = '#D55E00', '#222222', '#009E73'
plt.rcParams.update({'font.size': 9.5, 'axes.grid': True, 'grid.alpha': .25,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.dpi': 130, 'savefig.bbox': 'tight'})
W = 150


def sm(v, w=W):
    v = np.asarray(v, float)
    return np.median(sliding_window_view(v, w), axis=1) if len(v) > w else v


def load(tag):
    h = pickle.load(open(glob.glob(f'fit_result/sci_full_{tag}/history_iter10000_*.pkl')[0], 'rb'))
    d = {}
    for p in ['Ab', 'eField', 'tran_diff', 'long_diff', 'lifetime']:
        t = float(np.ravel(h[p + '_target'])[0])
        d[p] = sm(100 * (np.ravel(h[p + '_iter'])[1:] / t - 1))
    d['loss'] = sm(np.asarray(h['losses_iter'], float))
    d['pos'] = sm(np.ravel(h['pos_residual_iter']) * 1e4)
    d['dedx'] = sm(np.ravel(h['dedx_mae_truth_iter']))
    d['mean_dedx'] = sm(np.ravel(h['mean_dedx_iter']))
    return d


A = load('CINITA8')   # the outlier
B = load('DSREP8')    # nominal-init baseline, same replica

PAN = [('Ab', 'A$_b$ error (%)', True, None),
       ('lifetime', 'lifetime error (%)', True, None),
       ('eField', 'E field error (%)', True, (-0.4, 0.4)),
       ('tran_diff', 'transverse diffusion error (%)', True, (-15, 20)),
       ('long_diff', 'longitudinal diffusion error (%)', True, (-30, 60)),
       ('mean_dedx', 'mean dE/dx (MeV/cm)', False, None),
       ('dedx', 'dE/dx MAE (vs matched truth)', False, (0.048, 0.085)),
       ('pos', 'position residual (µm)', False, (80, 700)),
       ('loss', 'loss', False, (8800, 12500))]

fig, axes = plt.subplots(3, 3, figsize=(13.5, 9.6))
for a, (k, lab, zero, yl) in zip(axes.ravel(), PAN):
    for d, col, nm in ((B, INK, 'nominal init (baseline)'), (A, RED, 'all-shifted init (outlier)')):
        x = np.linspace(0, 10, len(d[k]))
        a.plot(x, d[k], color=col, lw=1.3, label=nm)
        a.plot(x[-1], d[k][-1], 'o', color=col, ms=5)
    if zero:
        a.axhline(0, color=GREEN, lw=1.5)
    if k == 'Ab':
        a.axhline(100 * (0.90 / 0.834881 - 1), color=RED, lw=1.2, ls=':')
        a.annotate('upper BOUND (0.90) — init 0.92 was out of range,\n'
                   'so A_b is clamped here from iter ~50 and never fitted',
                   (.30, .62), xycoords='axes fraction', fontsize=8, color=RED)
    if k == 'mean_dedx':
        a.axhline(1.887, color=GREEN, lw=1.5, ls='--')
        a.annotate('prior target 1.887', (.03, .06), xycoords='axes fraction',
                   fontsize=8, color=GREEN)
    if yl:
        a.set_ylim(*yl)
    a.set_ylabel(lab, fontsize=9)
    a.set_xlabel('iteration (×1000)', fontsize=8.5)
    a.annotate(f'end: {A[k][-1]:+.3g}  vs  {B[k][-1]:+.3g}', (.02, .93),
               xycoords='axes fraction', fontsize=8, color='#444444')
axes[0][0].legend(frameon=False, fontsize=8.5, loc='center right')

fig.suptitle('Fig 88 — the degenerate outlier in full (CINITA8, replica 8). A_b was initialised OUT OF RANGE (0.92 > max 0.90) and is\n'
             'clamped at the bound from iteration ~50 — an accidental but clean experiment: pin A_b 7.8% wrong, let everything else adapt.\n'
             'Lifetime slides to −59.3% to compensate and the loss ends at 9026 — BETTER than the baseline\'s 9167 (A_b +0.01%, lifetime +16.4%).\n'
             'Position, dE/dx MAE and loss all look normal-or-better while the calibration is ruined.',
             fontsize=10.5, y=1.02)
fig.tight_layout()
fig.savefig(f'{OUT}/fig88_outlier_anatomy.png')
print('wrote fig88')
for k, _, _, _ in PAN:
    print(f'  {k:10s} outlier {A[k][-1]:+10.4g}   baseline {B[k][-1]:+10.4g}')
