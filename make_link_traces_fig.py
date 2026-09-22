"""Fig 100 — the linked-diffusion fits, iteration by iteration.

LINKED (solid) fits 4 parameters and derives tran_diff = long_diff / ratio through the mobility
relation; the CONTROL (dashed) fits all 5 independently. Both see the SAME link-consistent target
(long 5.81418e-06 -> tran 1.18080e-05), verified identical in both arms.

WHAT THE TRACES SHOW
  * lifetime: the linked arm sits BELOW the control for essentially the whole run on all three
    seeds -- the -5.70 +- 1.73 point gain (3.3 sigma, 3/3) is not an endpoint accident, it is
    established early and held.
  * the two diffusions in the linked arm move in LOCKSTEP by construction (their traces are the
    same curve up to the fixed ratio), and both run away to ~+56%.
  * long_diff (the ANCHOR) climbs to +58% and flattens there. That is BELOW its own bound
    (+72% equivalent), so the anchor is not bound-limited -- it simply converges high.
  * tran_diff (DERIVED) crosses its own allowed maximum 1.5e-05 and keeps going, because only the
    anchor passes through the sigmoid normalisation. The red band marks the forbidden region.
  * A_b and E field are untouched by any of it, as everywhere else in this campaign.
"""
import glob, pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view

OUT = 'plots/noise_report'
S = '/sdf/scratch/neutrino/pgranger/fit_result'
BLUE, ORANGE, GREEN, RED, GREY, INK = ('#0072B2', '#E69F00', '#009E73', '#D55E00',
                                       '#888888', '#222222')
plt.rcParams.update({'font.size': 9.5, 'axes.grid': True, 'grid.alpha': .25,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.dpi': 130, 'savefig.bbox': 'tight'})
W = 150
SEEDCOL = {3: ORANGE, 7: BLUE, 8: RED}
RANGE_MAX = {'long_diff': 1.0e-05, 'tran_diff': 1.5e-05}


def sm(v, w=W):
    v = np.asarray(v, float)
    return np.median(sliding_window_view(v, w), axis=1) if len(v) > w else v


def load(tag):
    fs = glob.glob(f'{S}/sci_full_{tag}/history_iter10000_*.pkl')
    if not fs:
        return None
    h = pickle.load(open(fs[0], 'rb'))
    d = {}
    for p in ['Ab', 'eField', 'tran_diff', 'long_diff', 'lifetime']:
        if p + '_iter' not in h:
            continue
        v = np.ravel(h[p + '_iter'])
        t = float(np.ravel(h[p + '_target'])[0]) if p + '_target' in h else None
        d[p] = sm(100 * (v[1:] / t - 1)) if t else None
        d[p + '_abs'] = sm(v[1:])
    d['pos'] = sm(np.ravel(h['pos_residual_iter']) * 1e4)
    d['loss'] = sm(np.asarray(h['losses_iter'], float))
    return d


D = {}
for ds in (3, 7, 8):
    D[('link', ds)] = load(f'LINKON{ds}')
    D[('ctrl', ds)] = load(f'LINKOFF{ds}')

PAN = [('lifetime', 'lifetime error (%)', (-10, 45), True),
       ('long_diff', 'long. diffusion error (%)', (-30, 90), True),
       ('tran_diff', 'tran. diffusion error (%)', (-30, 90), True),
       ('Ab', 'A$_b$ error (%)', (-3, 3), True),
       ('pos', 'position residual (µm)', (100, 700), False),
       ('loss', 'loss', None, False)]

fig, axes = plt.subplots(2, 3, figsize=(14.5, 8))
for a, (k, lab, yl, zero) in zip(axes.ravel(), PAN):
    for ds in (3, 7, 8):
        for arm, ls, lw in (('ctrl', '--', 1.1), ('link', '-', 1.7)):
            d = D[(arm, ds)]
            if d is None or d.get(k) is None:
                continue
            y = d[k]
            a.plot(np.linspace(0, 10, len(y)), y, ls, color=SEEDCOL[ds], lw=lw, alpha=.9)
    if zero:
        a.axhline(0, color=GREEN, lw=1.6)
    if yl:
        a.set_ylim(*yl)
    a.set_xlabel('iteration (×1000)')
    a.set_ylabel(lab)
    a.set_title({'Ab': 'A$_b$ — untouched by the tie',
                 'pos': 'position residual — no cost from the tie',
                 'loss': 'loss — both arms converge alike'}.get(k, k), fontsize=9.5)

# mark the bound violations on the two diffusion panels
for a, k in ((axes[0][1], 'long_diff'), (axes[0][2], 'tran_diff')):
    tgt = {'long_diff': 5.81418e-06, 'tran_diff': 1.18080e-05}[k]
    lim = 100 * (RANGE_MAX[k] / tgt - 1)
    a.axhline(lim, color=RED, lw=1.5, ls=':')
    a.axhspan(lim, a.get_ylim()[1], color=RED, alpha=.10, lw=0)
    a.annotate(f'allowed max ({RANGE_MAX[k]:.1e})', (.03, lim + 3), xycoords=('axes fraction', 'data'),
               fontsize=8, color=RED)
axes[0][1].set_title('long_diff — the ANCHOR (settles at +58%, below its bound)', fontsize=9.5)
axes[0][2].set_title('tran_diff — DERIVED (escapes its bound: unnormalised)', fontsize=9.5)
axes[0][0].set_title('lifetime — the linked arm is lower throughout', fontsize=9.5)

h = [plt.Line2D([], [], color=INK, ls='-', lw=1.8, label='LINKED (4 fitted, tran derived)'),
     plt.Line2D([], [], color=INK, ls='--', lw=1.2, label='control (5 fitted independently)')]
h += [plt.Line2D([], [], color=SEEDCOL[s], lw=2, label=f'seed {s}') for s in (3, 7, 8)]
fig.legend(handles=h, ncol=5, frameon=False, loc='lower center', bbox_to_anchor=(.5, -0.03),
           fontsize=9)

fig.suptitle('Fig 100 — linked vs independent diffusion, iteration by iteration (same link-consistent target in both arms).\n'
             'Lifetime (top left): the linked arm runs below the control on all three seeds for the whole fit — the −5.70 ± 1.73 point gain (3.3σ, 3/3)\n'
             'is established early, not an endpoint artefact. But the tied diffusions run away together to ~+56%: the anchor converges high (below its own\n'
             'bound), dragging the DERIVED parameter clean through its allowed maximum (red band) — only the anchor is sigmoid-normalised.',
             fontsize=10.3, y=1.05)
fig.tight_layout()
fig.savefig(f'{OUT}/fig100_link_traces.png')
print('wrote fig100')
for ds in (3, 7, 8):
    l, c = D[('link', ds)], D[('ctrl', ds)]
    print(f'  seed {ds}: lifetime end  linked {l["lifetime"][-1]:+6.2f}  vs control {c["lifetime"][-1]:+6.2f}')
