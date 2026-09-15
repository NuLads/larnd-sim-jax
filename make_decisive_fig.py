"""Figs 82/83 — the decisive versions of the fig 80/81 finding.

FIG 82 'the loss knows the fit, not the physics': three aligned panels, same x (distance from
the seed's best loss). Position and dE/dx lie on a line (r = +0.8); lifetime is a flat cloud —
and the runs within 150 nats of the best loss span the whole lifetime range.

FIG 83 'selection on loss fails': per seed, every truth-free arm as (loss, lifetime). The
best-loss arm (what any loss-based procedure returns) is circled red; the best-lifetime arm is
starred green. On 3 of 4 seeds the loss picks a worse lifetime than what was available — on
seed 7 it picks the WORST (+24.2% vs +13.5% available).
"""
import glob, pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = 'plots/noise_report'
plt.rcParams.update({'font.size': 10, 'axes.grid': True, 'grid.alpha': .25,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.dpi': 130, 'savefig.bbox': 'tight'})
BLUE, ORANGE, GREEN, RED, GREY = '#0072B2', '#E69F00', '#009E73', '#D55E00', '#999999'


def endpoint(tag):
    fs = glob.glob(f'fit_result/sci_full_{tag}/history_iter10000_*.pkl')
    if not fs:
        return None
    h = pickle.load(open(fs[0], 'rb'))
    e = {'loss': float(np.median(np.asarray(h['losses_iter'], float)[-500:])),
         'pos': float(np.median(np.ravel(h['pos_residual_iter'])[-500:] * 1e4)),
         'dedx': float(np.median(np.ravel(h['dedx_mae_truth_iter'])[-500:]))}
    t = float(np.ravel(h['lifetime_target'])[0])
    e['lt'] = float(np.median(100 * (np.ravel(h['lifetime_iter'])[-500:] / t - 1)))
    return e


ARMS = {1: ['DSREP1', 'PASS2R1', 'PASS3R1'],
        3: ['DSREP3', 'PASS2R3', 'PASS3R3', 'RLXR3'],
        7: ['DSREP7', 'PASS2R7', 'PASS3R7', 'RLXR7'],
        8: ['DSREP8', 'PASS2R8', 'PASS3R8', 'PASS4R8', 'RLXR8', 'RLXP38', 'HOT8', 'WU3000'],
        2: ['DSREP2'], 4: ['DSREP4'], 5: ['DSREP5'], 6: ['DSREP6']}
SHORT = {'DSREP': 'pass 1', 'PASS2R': 'pass 2', 'PASS3R': 'pass 3', 'PASS4R': 'pass 4',
         'RLXR': 'relax', 'RLXP38': 'relax@p3', 'HOT8': 'slow-anneal', 'WU3000': 'warmup'}


def short(tag):
    for k in ['RLXP38', 'HOT8', 'WU3000', 'PASS2R', 'PASS3R', 'PASS4R', 'RLXR', 'DSREP']:
        if tag.startswith(k):
            return SHORT[k]
    return tag


data = {s: [(t, endpoint(t)) for t in ts] for s, ts in ARMS.items()}
data = {s: [(t, e) for t, e in v if e] for s, v in data.items()}

# ---------- FIG 82 ----------
pts = []
for s, v in data.items():
    if len(v) < 2:      # single-run seeds: their only fit IS their "best" — not comparable
        continue
    best = min(e['loss'] for _, e in v)
    for t, e in v:
        pts.append(dict(seed=s, dl=e['loss'] - best, **e))

fig, ax = plt.subplots(1, 3, figsize=(13, 4.1))
for a, key, ylab, col in [(ax[0], 'pos', 'position error (µm)', BLUE),
                          (ax[1], 'dedx', 'dE/dx error (MAE)', BLUE),
                          (ax[2], 'lt', 'lifetime error (%)', RED)]:
    x = np.array([p['dl'] for p in pts])
    y = np.array([p[key] for p in pts])
    a.scatter(x, y, s=48, c=col, alpha=.75, edgecolors='white', lw=.6, zorder=3)
    r = np.corrcoef(x, y)[0, 1]
    if key != 'lt':
        b, c0 = np.polyfit(x, y, 1)
        xx = np.linspace(0, x.max(), 10)
        a.plot(xx, b * xx + c0, '--', color='#333333', lw=1.2, zorder=2)
    else:
        a.axhline(0, color=GREEN, lw=1.6)
        a.annotate('truth', (.985, 0), xycoords=('axes fraction', 'data'),
                   textcoords='offset points', xytext=(0, 5), ha='right',
                   color=GREEN, fontweight='bold')
        m = x < 150
        a.axvspan(-20, 150, color=GREY, alpha=.18, lw=0)
        a.annotate('runs within 150 nats of the best loss:\nlifetime anywhere in '
                   f'{y[m].min():+.1f}…{y[m].max():+.1f}%',
                   (.35, .86), xycoords='axes fraction', fontsize=8.5, color='#444444')
    a.set_title(f'r = {r:+.2f}', color=(RED if key == 'lt' else '#333333'), fontsize=11)
    a.set_xlabel('final loss − seed best (nats)')
    a.set_ylabel(ylab)
fig.suptitle('Fig 82 — the loss knows the fit, not the physics. Every arm of the multi-arm seeds (n = %d), same x-axis:\n'
             'position and dE/dx errors are read straight off the loss; lifetime is not — the best fits (grey band) span the entire range.'
             % len(pts), fontsize=11.5, y=1.05)
fig.tight_layout()
fig.savefig(f'{OUT}/fig82_loss_two_axes.png')
print('wrote fig82')

# ---------- FIG 83 ----------
fig, axes = plt.subplots(1, 4, figsize=(13.6, 3.9))
for a, s in zip(axes, (1, 3, 7, 8)):
    v = data[s]
    L = [e['loss'] for _, e in v]
    T = [e['lt'] for _, e in v]
    a.scatter(L, T, s=46, c=BLUE, alpha=.8, edgecolors='white', lw=.5, zorder=3)
    for (t, e) in v:
        a.annotate(short(t), (e['loss'], e['lt']), textcoords='offset points',
                   xytext=(6, 4), fontsize=7.5, color='#555555')
    ib = int(np.argmin(L)); ig = int(np.argmin(np.abs(T)))
    a.scatter(L[ib], T[ib], s=210, facecolors='none', edgecolors=RED, lw=2, zorder=4)
    a.scatter(L[ig], T[ig], marker='*', s=170, c=GREEN, edgecolors='white', lw=.5, zorder=5)
    a.axhline(0, color=GREEN, lw=1.2)
    a.set_title(f'seed {s}:  loss picks {T[ib]:+.1f}%   (best available {T[ig]:+.1f}%)',
                fontsize=9, color=(RED if abs(T[ib]) > abs(T[ig]) + .5 else '#333333'))
    a.set_xlabel('final loss')
    if s == 1:
        a.set_ylabel('lifetime error (%)')
    a.set_ylim(-3, 27)
h = [plt.Line2D([], [], marker='o', ls='', mfc='none', mec=RED, mew=2, ms=12,
                label='best loss — what any loss-based selection returns'),
     plt.Line2D([], [], marker='*', ls='', color=GREEN, ms=13, label='best lifetime available')]
fig.legend(handles=h, ncol=2, frameon=False, loc='lower center', bbox_to_anchor=(.5, -0.06), fontsize=9)
fig.suptitle('Fig 83 — selection on the loss FAILS for lifetime: on 3 of 4 seeds the best-loss arm has a worse lifetime than\n'
             'another available arm — on seed 7 it picks the worst of the whole ladder. (Truth-free arms only.)',
             fontsize=11.5, y=1.08)
fig.tight_layout()
fig.savefig(f'{OUT}/fig83_loss_selection.png')
print('wrote fig83')
for s in (1, 3, 7, 8):
    v = data[s]
    ib = min(v, key=lambda te: te[1]['loss'])
    ig = min(v, key=lambda te: abs(te[1]['lt']))
    print(s, 'loss picks', ib[0], '%+.1f' % ib[1]['lt'], '| best', ig[0], '%+.1f' % ig[1]['lt'])
